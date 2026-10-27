"""
================================================================================
TRAIN: U-Net heatmap baseline, patient-level k-fold
================================================================================
DESCRIPTION:
    Config-driven training loop for the pure-PyTorch U-Net heatmap baseline
    (code/models/unet.py). One run trains ONE fold - the SLURM launch script
    (slurm/train_unet.sbatch) runs this as a job array, one array task per
    fold, so all folds train in parallel and each fold's results can be
    inspected/compared independently (see the project's k-fold CV
    requirement for small-n evaluation rigor).

    Every checkpoint embeds the full resolved config (including the landmark
    vocabulary actually used), so evaluate.py never needs to be told the
    model architecture or data settings separately - point it at a
    checkpoint and it reconstructs everything from that.

USAGE:
    python train.py --config ../../configs/unet_baseline.yaml --fold 0
    python train.py --config ../../configs/unet_baseline.yaml --fold 0 --set training.lr=0.0005

CONFIG SCHEMA: see configs/unet_baseline.yaml.
================================================================================
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'datasets')))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')))

from config import load_config, apply_overrides
from seeding import set_seed
from cephalometric_dataset import CephalometricDataset, TransformedView
from heatmap_transform import HeatmapTargetTransform, resolve_landmark_vocabulary
from kfold import make_patient_kfold_splits
from fold_manifest import resolve_kfold_split, resolve_relative
from label_efficiency import load_patient_subset, restrict_indices_to_patients
from unet import UNetHeatmapModel
from visualize import save_validation_overlays
from eval_core import evaluate_and_write
from optim_factory import build_optimizer, warmup_scale, apply_lr_scale, format_group_summary
from output_paths import run_outputs_path, run_output_dir
from provenance import write_run_provenance


def masked_heatmap_mse(pred, target, mask):
    """pred, target: (N, C, H, W); mask: (N, C) - 1 where that landmark is
    actually present in this sample's tracing. Landmarks absent from a
    (possibly partial) tracing don't contribute to the loss for that sample,
    rather than being penalized as an all-zero target."""
    per_channel_mse = ((pred - target) ** 2).mean(dim=(2, 3))  # (N, C)
    denom = mask.sum().clamp(min=1.0)
    return (per_channel_mse * mask).sum() / denom


def build_base_dataset(config):
    return CephalometricDataset(
        images_dir=config['data']['images_dir'],
        fcd_dir=config['data']['fcd_dir'],
        review_csv_path=config['data'].get('review_csv_path'),
        combine_method=config['data'].get('combine_method', 'median'),
        exclude_images_path=config['data'].get('exclude_images_path'),
    )


def run_epoch(model, loader, device, optimizer=None, max_grad_norm=None,
              global_step=0, on_step=None, grad_accum_steps=1):
    """One pass over `loader`. If `optimizer` is given, trains; else evaluates.
    Returns (mean_loss, n_optimizer_steps) - 0 steps for an eval pass.

    max_grad_norm: if set, clips the gradient norm before the optimizer step
    (a stability aid for the pretrained arms - see train_backbone_variant.py;
    left None by the from-scratch baseline, so its behaviour is unchanged).
    global_step/on_step: `on_step(step)` is called before each optimizer step
    with the run-global step counter, for STEP-based LR schedules (warmup - see
    optim_factory). Counting in steps rather than epochs matters for the
    label-efficiency curve, where a 25% epoch is 4x shorter.
    grad_accum_steps: micro-batches per optimizer step (effective batch size =
    batch_size * grad_accum_steps). Only needed if a model doesn't fit at the
    configured batch size - e.g. the fine-tuned DINOv2 arm; 1 (the default)
    is exactly the original loop. Reported losses are unscaled."""
    is_train = optimizer is not None
    model.train(is_train)
    losses = []
    steps = 0
    micro_batches = 0
    grad_accum_steps = max(1, int(grad_accum_steps))
    if is_train:
        optimizer.zero_grad()
    with torch.set_grad_enabled(is_train):
        for batch in loader:
            images = batch['image'].to(device)
            heatmaps = batch['heatmaps'].to(device)
            mask = batch['mask'].to(device)

            pred = model(images)
            loss = masked_heatmap_mse(pred, heatmaps, mask)

            if is_train:
                (loss / grad_accum_steps).backward()
                micro_batches += 1
                if micro_batches % grad_accum_steps == 0:
                    if on_step is not None:
                        on_step(global_step + steps)
                    if max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    steps += 1

            losses.append(loss.item())
    return (float(np.mean(losses)) if losses else float('nan')), steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--fold', type=int, default=None,
                         help="overrides kfold.fold from the config (SLURM job array passes this)")
    parser.add_argument('--all-data', action='store_true',
                         help="train the FINAL deployable model on ALL data (no k-fold). A small "
                              "patient-level holdout (training.holdout_fraction, default 0.1) is used "
                              "ONLY for early stopping / best-checkpoint selection; set it to 0 to "
                              "train the full epoch budget on 100%% of the data. Writes to "
                              "<checkpoint_dir>/final/best.pt.")
    parser.add_argument('--set', action='append', default=[], dest='overrides',
                         help="dot-path override, e.g. --set training.lr=0.0005 (repeatable)")
    args = parser.parse_args()

    config = load_config(args.config)
    config = apply_overrides(config, args.overrides)
    if args.fold is not None:
        config.setdefault('kfold', {})['fold'] = args.fold

    # A relative landmark_config_path is resolved against this config file's
    # own directory (e.g. 'landmarks.yaml' when both live in configs/),
    # so the same config works regardless of the repo's absolute location.
    config_dir = os.path.dirname(os.path.abspath(args.config))
    for rel_key in ('landmark_config_path', 'exclude_images_path'):
        rel = config['data'].get(rel_key)
        if rel and not os.path.isabs(rel):
            config['data'][rel_key] = os.path.join(config_dir, rel)

    set_seed(config['seed'])

    base_dataset = build_base_dataset(config)
    if len(base_dataset) == 0:
        raise RuntimeError(
            "No usable images found - check data.images_dir/fcd_dir/review_csv_path in the config. "
            f"{len(base_dataset.skipped)} images were skipped; see dataset.skipped for reasons."
        )

    landmark_names = resolve_landmark_vocabulary(base_dataset, config['data'])

    # Train split gets data augmentation (config-driven), val split never does.
    # Both views share the single parsed base_dataset (see TransformedView).
    aug_cfg = config['data'].get('augmentation') or {}
    tf_common = dict(
        landmark_names=landmark_names,
        image_size=tuple(config['data']['image_size']),
        heatmap_size=tuple(config['data']['heatmap_size']),
        sigma=config['data'].get('sigma', 2.0),
    )
    train_transform = HeatmapTargetTransform(**tf_common, augment=bool(aug_cfg.get('enabled', False)),
                                             aug_params=aug_cfg.get('params'))
    val_transform = HeatmapTargetTransform(**tf_common, augment=False)
    if train_transform.augment:
        print(f"Data augmentation ENABLED (train only): {train_transform.aug}")

    train_view = TransformedView(base_dataset, train_transform)
    val_view = TransformedView(base_dataset, val_transform)

    if args.all_data:
        # Final deployable model: train on ALL data. A small patient-level holdout is
        # carved (via one split of a k=round(1/frac) partition) ONLY for early stopping
        # and best-checkpoint selection, so the final model still trains on the large
        # majority of the data - more than any single CV fold. holdout_fraction=0 turns
        # this off and trains the full epoch budget on 100% (best.pt = the last epoch).
        holdout_fraction = config['training'].get('holdout_fraction', 0.1)
        if holdout_fraction and holdout_fraction > 0:
            k_holdout = max(2, round(1.0 / holdout_fraction))
            train_idx, val_idx = make_patient_kfold_splits(base_dataset, k=k_holdout, seed=config['seed'])[0]
        else:
            train_idx, val_idx = np.arange(len(base_dataset)), np.array([], dtype=int)
        run_name, fold, k = 'final', 'final', None
    else:
        k = config['kfold']['k']
        fold = config['kfold']['fold']
        train_idx, val_idx = resolve_kfold_split(base_dataset, config, fold=fold, config_dir=config_dir)
        run_name = f"fold{fold}"

    # Label-efficiency runs train on a SUBSET of this fold's training patients
    # (code/datasets/label_efficiency.py). The validation split is deliberately
    # untouched: same held-out patients as every other point on the curve and
    # as Run 6, so only the training-set size varies.
    subset_path = resolve_relative(config['data'].get('train_patient_subset_path'), config_dir)
    subset_patients = None
    if subset_path:
        val_before = set(val_idx.tolist())
        subset_patients = load_patient_subset(subset_path)
        train_idx = restrict_indices_to_patients(base_dataset, train_idx, subset_patients)
        assert set(val_idx.tolist()) == val_before, "validation split changed while subsetting training patients"
        assert not (set(train_idx.tolist()) & val_before), "training subset overlaps the validation split"
        print(f"[label-efficiency] training restricted to {len(subset_patients)} patients from {subset_path}; "
              f"validation split unchanged ({len(val_idx)} images)")

    n_train_patients = len({base_dataset.samples[i]['serial'] for i in train_idx})
    n_val_patients = len({base_dataset.samples[i]['serial'] for i in val_idx})
    label = "ALL-DATA (final)" if args.all_data else f"Fold {fold}/{k}"
    print(f"{label}: {len(train_idx)} train images ({n_train_patients} patients), "
          f"{len(val_idx)} val/holdout images ({n_val_patients} patients), "
          f"{len(landmark_names)} landmark channels: {landmark_names}")

    train_loader = DataLoader(
        Subset(train_view, train_idx),
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training'].get('num_workers', 0),
        drop_last=len(train_idx) > config['training']['batch_size'],
    )
    val_loader = DataLoader(
        Subset(val_view, val_idx),
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 0),
    )

    requested_device = config['training'].get('device', 'cuda')
    device = torch.device(requested_device if (requested_device == 'cpu' or torch.cuda.is_available()) else 'cpu')
    if requested_device != 'cpu' and device.type == 'cpu':
        print(f"[WARN] device '{requested_device}' requested but CUDA is not available - falling back to CPU")

    model = UNetHeatmapModel(
        num_landmarks=len(landmark_names),
        base_channels=config['model'].get('base_channels', 32),
    ).to(device)
    # With no training.optimizer section this is exactly the single-group Adam
    # of Runs 1-6; the baseline is from-scratch, so it has no encoder/decoder
    # split to make (see optim_factory).
    optimizer, group_summary = build_optimizer(model, config['training'])
    warmup_steps = int((config['training'].get('optimizer') or {}).get('warmup_steps', 0) or 0)
    grad_accum_steps = int(config['training'].get('grad_accum_steps', 1) or 1)
    if grad_accum_steps > 1:
        print(f"Gradient accumulation: {grad_accum_steps} micro-batches per step "
              f"(effective batch size {config['training']['batch_size'] * grad_accum_steps})")
    print(f"Optimizer: {format_group_summary(group_summary)}"
          + (f" | linear warmup over {warmup_steps} steps" if warmup_steps else ""))

    checkpoint_dir = os.path.join(config['training']['checkpoint_dir'], run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Provenance is written up front so a crashed/preempted run still records
    # what it was attempting (rewritten with the outcome at the end).
    output_dir = run_output_dir(config, run_name)
    provenance = dict(
        arm=os.path.basename(os.path.normpath(config['training']['checkpoint_dir'])),
        script='train.py', run_name=run_name, fold=fold, k=k, all_data=bool(args.all_data),
        train_patient_subset_path=subset_path,
        n_train_patients=n_train_patients, n_train_images=len(train_idx),
        n_val_patients=n_val_patients, n_val_images=len(val_idx),
        landmark_names=landmark_names, optimizer_groups=group_summary,
        cli_overrides=args.overrides, status='started',
    )
    write_run_provenance(output_dir, config, **provenance)

    def save_checkpoint(path, epoch, val_loss):
        torch.save({
            'model_state_dict': model.state_dict(),
            'landmark_names': landmark_names,
            'config': config,
            'epoch': epoch,
            'val_loss': val_loss,
        }, path)

    has_val = len(val_idx) > 0   # False only for --all-data with holdout_fraction=0
    if not has_val:
        print("No validation/holdout split - training the full epoch budget on 100% of data; "
              "best.pt tracks the last epoch (no early stopping).")

    best_val_loss = float('inf')
    patience = config['training'].get('early_stopping_patience')
    epochs_without_improvement = 0
    history = []
    total_steps = 0
    best_epoch = None

    def on_step(step):
        apply_lr_scale(optimizer, warmup_scale(step, warmup_steps))

    for epoch in range(config['training']['epochs']):
        train_loss, steps = run_epoch(model, train_loader, device, optimizer,
                                      global_step=total_steps,
                                      on_step=on_step if warmup_steps else None,
                                      grad_accum_steps=grad_accum_steps)
        total_steps += steps
        val_loss, _ = run_epoch(model, val_loader, device, optimizer=None) if has_val else (float('nan'), 0)
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
                        'total_steps': total_steps,
                        'lr': [g['lr'] for g in optimizer.param_groups]})
        print(f"[{run_name}] epoch {epoch}: train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
              f"(steps={total_steps})")

        if not has_val:
            # No held-out signal: keep the latest epoch as best.pt (trains full budget).
            save_checkpoint(os.path.join(checkpoint_dir, 'best.pt'), epoch, val_loss)
        elif val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(os.path.join(checkpoint_dir, 'best.pt'), epoch, val_loss)
        else:
            epochs_without_improvement += 1

        save_checkpoint(os.path.join(checkpoint_dir, 'last.pt'), epoch, val_loss)

        with open(os.path.join(checkpoint_dir, 'history.json'), 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)

        if has_val and patience is not None and epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch} (no val improvement for {patience} epochs)")
            break

    print(f"Done. Best val_loss={best_val_loss:.6f} after {total_steps} optimizer steps "
          f"({len(history)} epochs). Checkpoints in {checkpoint_dir}")
    write_run_provenance(output_dir, config, **{
        **provenance, 'status': 'finished', 'epochs_run': len(history), 'total_steps': total_steps,
        'best_epoch': best_epoch, 'best_val_loss': best_val_loss if has_val else None,
        'early_stopped': bool(has_val and patience is not None
                              and epochs_without_improvement >= patience),
    })

    best_checkpoint = torch.load(os.path.join(checkpoint_dir, 'best.pt'), map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    # Generated overlays go under the gitignored outputs/ (mirroring the
    # checkpoint's model/fold), not into checkpoints/ - see output_paths.py.
    if len(val_idx) > 0:
        overlay_dir = run_outputs_path(config, run_name, 'val_overlays', make_parent=False)
        written = save_validation_overlays(
            model=model,
            base_dataset=base_dataset,
            transform=val_transform,
            indices=val_idx,
            device=device,
            output_dir=overlay_dir,
            max_images=config['training'].get('visualize_max_images'),
        )
        print(f"Wrote {len(written)} validation overlay image(s) (best model, epoch {best_checkpoint['epoch']}) to {overlay_dir}")

    # Detailed per-landmark MRE/SDR of the best model on BOTH splits (see
    # eval_core / evaluate.py), always via the PLAIN (non-augmenting) view.
    # eval_val.json is what compare_backbones.py aggregates; eval_train.json
    # exposes per-landmark train-vs-val gaps (overfitting). For --all-data the
    # 'val' report is the small holdout (skipped entirely if holdout_fraction=0).
    report_meta = {'all_data': True} if args.all_data else {'fold': fold, 'k': k}
    # The training cohort actually used - so eval_val.json alone is enough to
    # place a run on the label-efficiency curve (see aggregate_label_efficiency.py).
    report_meta.update({'n_train_patients': n_train_patients, 'n_train_images': len(train_idx),
                        'train_patient_subset': os.path.basename(subset_path) if subset_path else None,
                        'total_steps': total_steps})
    for split_name, split_idx in (('val', val_idx), ('train', train_idx)):
        if len(split_idx) == 0:
            continue
        evaluate_and_write(
            model, val_view, split_idx,
            config=config, landmark_names=landmark_names,
            output_prefix=run_outputs_path(config, run_name, f"eval_{split_name}"),
            split=split_name, report_meta=report_meta,
            batch_size=config['training']['batch_size'], device=device,
        )


if __name__ == "__main__":
    main()
