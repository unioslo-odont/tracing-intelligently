"""
================================================================================
TRAIN: pretrained-backbone variants, patient-level k-fold
================================================================================
DESCRIPTION:
    Same training mechanics as train.py (imports its masked_heatmap_mse,
    run_epoch, and build_base_dataset directly, unchanged) but builds the
    model via code/models/model_factory.py instead of hardcoding
    UNetHeatmapModel - so it can train any of the pretrained-backbone arms
    (see configs/resnet_unet.yaml, configs/hrnet_pretrained.yaml,
    configs/dinov2_frozen.yaml) with the exact same loss, optimizer,
    checkpointing, and overlay-generation code path as the from-scratch
    baseline, for a fair comparison.

    Kept as a separate script (rather than adding model.type support to
    train.py directly) to avoid touching train.py while it has an active
    training run depending on it. Once that settles, these two scripts'
    main() are candidates to merge into one, both going through
    model_factory.

USAGE:
    python train_backbone_variant.py --config ../../configs/resnet_unet.yaml --fold 0
================================================================================
"""

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'datasets')))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')))

from config import load_config, apply_overrides
from seeding import set_seed
from cephalometric_dataset import TransformedView
from heatmap_transform import HeatmapTargetTransform, resolve_landmark_vocabulary
from fold_manifest import resolve_kfold_split, resolve_relative
from label_efficiency import load_patient_subset, restrict_indices_to_patients
from model_factory import build_model
from visualize import save_validation_overlays
from eval_core import evaluate_and_write
from optim_factory import build_optimizer, warmup_scale, apply_lr_scale, format_group_summary
from output_paths import run_outputs_path, run_output_dir
from provenance import write_run_provenance

from train import masked_heatmap_mse, build_base_dataset, run_epoch  # noqa: F401 (masked_heatmap_mse used via run_epoch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--fold', type=int, default=None,
                         help="overrides kfold.fold from the config (SLURM job array passes this)")
    parser.add_argument('--set', action='append', default=[], dest='overrides',
                         help="dot-path override, e.g. --set training.lr=0.0005 (repeatable)")
    args = parser.parse_args()

    config = load_config(args.config)
    config = apply_overrides(config, args.overrides)
    if args.fold is not None:
        config.setdefault('kfold', {})['fold'] = args.fold

    # Relative landmark_config_path / exclude_images_path are resolved against
    # this config file's own directory (e.g. 'landmarks.yaml' when both live in
    # configs/), so the same config works regardless of the repo's absolute location.
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

    # Train split augments (config-driven), val split never does (see train.py / TransformedView).
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

    k = config['kfold']['k']
    fold = config['kfold']['fold']
    train_idx, val_idx = resolve_kfold_split(base_dataset, config, fold=fold, config_dir=config_dir)

    # Label-efficiency runs subsample this fold's TRAINING patients only; the
    # validation split stays exactly as Run 6 had it (see train.py for the
    # same block and code/datasets/label_efficiency.py for why).
    subset_path = resolve_relative(config['data'].get('train_patient_subset_path'), config_dir)
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
    model_type = config['model'].get('type', 'unet')
    print(f"Model: {model_type} ({config['model']})")
    print(f"Fold {fold}/{k}: {len(train_idx)} train images ({n_train_patients} patients), "
          f"{len(val_idx)} val images ({n_val_patients} patients), "
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

    checkpoint_dir = os.path.join(config['training']['checkpoint_dir'], f"fold{fold}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Provenance up front (rewritten with the outcome at the end) - see provenance.py.
    output_dir = run_output_dir(config, f"fold{fold}")
    provenance = dict(
        arm=os.path.basename(os.path.normpath(config['training']['checkpoint_dir'])),
        script='train_backbone_variant.py', run_name=f"fold{fold}", fold=fold, k=k,
        model_type=model_type, train_patient_subset_path=subset_path,
        n_train_patients=n_train_patients, n_train_images=len(train_idx),
        n_val_patients=n_val_patients, n_val_images=len(val_idx),
        landmark_names=landmark_names, cli_overrides=args.overrides, status='started',
    )
    write_run_provenance(output_dir, config, **provenance)

    def save_checkpoint(m, path, epoch, val_loss):
        torch.save({
            'model_state_dict': m.state_dict(),
            'landmark_names': landmark_names,
            'config': config,
            'epoch': epoch,
            'val_loss': val_loss,
        }, path)

    # Training-stability controls (all default OFF, so resnet/hrnet configs that
    # don't set them behave exactly as before). The frozen-DINOv2 arm sometimes
    # fails to converge on a fold (decoder diverges early -> heatmaps collapse,
    # training loss never came down), so
    # dinov2_frozen.yaml enables:
    #   warmup_epochs   - linear LR ramp 0->lr over the first N epochs (gentle start)
    #   max_grad_norm   - gradient-norm clip (stops an early blow-up)
    #   max_restarts    - if a run doesn't converge, retry with a fresh init (new seed)
    #
    # training.optimizer (see optim_factory) adds the Run 7 confound fix on top:
    # separate encoder/decoder LRs, optional ViT layer-wise decay, and a
    # STEP-based warmup (optimizer.warmup_steps) which takes precedence over the
    # epoch-based warmup_epochs above when both are set.
    epochs = config['training']['epochs']
    patience = config['training'].get('early_stopping_patience')
    warmup_epochs = config['training'].get('warmup_epochs', 0)
    warmup_steps = int((config['training'].get('optimizer') or {}).get('warmup_steps', 0) or 0)
    max_grad_norm = config['training'].get('max_grad_norm', None)
    grad_accum_steps = int(config['training'].get('grad_accum_steps', 1) or 1)
    if grad_accum_steps > 1:
        print(f"Gradient accumulation: {grad_accum_steps} micro-batches per step "
              f"(effective batch size {config['training']['batch_size'] * grad_accum_steps})")
    max_restarts = config['training'].get('max_restarts', 0)
    convergence_ratio = config['training'].get('convergence_ratio', 0.7)  # converged if best_val < ratio * initial_val

    best_overall_val = float('inf')
    history = []
    converged = False
    total_steps = 0
    group_summary = None

    for attempt in range(max_restarts + 1):
        if attempt > 0:
            set_seed(config['seed'] + attempt * 10007)  # fresh decoder init on a restart
            print(f"[{model_type} fold {fold}] restart {attempt}/{max_restarts}: previous attempt did not "
                  f"converge (best_val >= {convergence_ratio:.2f} x initial) - retrying with a new init")

        model = build_model(config['model'], num_landmarks=len(landmark_names)).to(device)
        optimizer, group_summary = build_optimizer(model, config['training'])
        if attempt == 0:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Trainable params: {trainable:,} | "
                  f"Frozen params: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")
            # Sanity check for the Run 7 confound fix: the LR each group STARTS
            # at (step 0, i.e. after warmup scaling) and the LR it reaches once
            # warmup is over. If these two lines don't say what the config says,
            # nothing downstream is a fair comparison.
            print(f"Optimizer groups @ step 0:    "
                  f"{format_group_summary(group_summary, warmup_scale(0, warmup_steps))}")
            print(f"Optimizer groups after warmup: {format_group_summary(group_summary)}"
                  + (f"  (warmup {warmup_steps} steps)" if warmup_steps
                     else f"  (warmup {warmup_epochs} epochs)" if warmup_epochs else "  (no warmup)"))

        attempt_initial_val = None
        attempt_best_val = float('inf')
        epochs_without_improvement = 0
        attempt_steps = 0   # warmup restarts with each attempt (it is a fresh run)

        def on_step(step):
            apply_lr_scale(optimizer, warmup_scale(step, warmup_steps))

        for epoch in range(epochs):
            if not warmup_steps:
                # Legacy epoch-based ramp, unchanged for configs that use it
                # (dinov2_frozen); scales every group's base LR by the same factor.
                epoch_scale = ((epoch + 1) / warmup_epochs
                               if (warmup_epochs and epoch < warmup_epochs) else 1.0)
                apply_lr_scale(optimizer, epoch_scale)

            train_loss, steps = run_epoch(model, train_loader, device, optimizer,
                                          max_grad_norm=max_grad_norm,
                                          global_step=attempt_steps,
                                          on_step=on_step if warmup_steps else None,
                                          grad_accum_steps=grad_accum_steps)
            attempt_steps += steps
            total_steps += steps
            val_loss, _ = run_epoch(model, val_loader, device, optimizer=None)
            lrs = [g['lr'] for g in optimizer.param_groups]
            history.append({'attempt': attempt, 'epoch': epoch, 'lr': lrs[0], 'lrs': lrs,
                            'train_loss': train_loss, 'val_loss': val_loss,
                            'attempt_steps': attempt_steps, 'total_steps': total_steps})
            print(f"[{model_type} fold {fold} a{attempt}] epoch {epoch}: "
                  f"lr={'/'.join(f'{v:.2e}' for v in lrs[:3])}{'...' if len(lrs) > 3 else ''} "
                  f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} (steps={attempt_steps})")

            if attempt_initial_val is None:
                attempt_initial_val = val_loss
            if val_loss < attempt_best_val:
                attempt_best_val = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # best.pt always holds the best model across ALL attempts
            if val_loss < best_overall_val:
                best_overall_val = val_loss
                save_checkpoint(model, os.path.join(checkpoint_dir, 'best.pt'), epoch, val_loss)
            save_checkpoint(model, os.path.join(checkpoint_dir, 'last.pt'), epoch, val_loss)
            with open(os.path.join(checkpoint_dir, 'history.json'), 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2)

            if patience is not None and epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch} (no val improvement for {patience} epochs)")
                break

        converged = bool(attempt_initial_val) and attempt_best_val < convergence_ratio * attempt_initial_val
        ratio = attempt_best_val / attempt_initial_val if attempt_initial_val else float('nan')
        print(f"[{model_type} fold {fold}] attempt {attempt}: best_val={attempt_best_val:.6f} "
              f"(initial={attempt_initial_val:.6f}, ratio={ratio:.2f}) -> {'converged' if converged else 'NOT converged'}")
        if converged:
            break

    if not converged and max_restarts > 0:
        print(f"[WARN] {model_type} fold {fold} did not converge after {max_restarts + 1} attempt(s); "
              f"keeping the best (val_loss={best_overall_val:.6f}).")

    print(f"Done. Best val_loss={best_overall_val:.6f} after {total_steps} optimizer steps. "
          f"Checkpoints in {checkpoint_dir}")
    write_run_provenance(output_dir, config, **{
        **provenance, 'status': 'finished', 'epochs_run': len(history), 'total_steps': total_steps,
        'best_val_loss': best_overall_val, 'converged': converged,
        'attempts': history[-1]['attempt'] + 1 if history else 0,
        'optimizer_groups': group_summary,
    })

    best_checkpoint = torch.load(os.path.join(checkpoint_dir, 'best.pt'), map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    # Generated overlays go under the gitignored outputs/ (mirroring the
    # checkpoint's model/fold), not into checkpoints/ - see output_paths.py.
    overlay_dir = run_outputs_path(config, f"fold{fold}", 'val_overlays', make_parent=False)
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
    # eval_core / evaluate_backbone_variant.py), always via the PLAIN
    # (non-augmenting) view. eval_val.json is what compare_backbones.py
    # aggregates; eval_train.json exposes per-landmark train-vs-val gaps.
    for split_name, split_idx in (('val', val_idx), ('train', train_idx)):
        evaluate_and_write(
            model, val_view, split_idx,
            config=config, landmark_names=landmark_names,
            output_prefix=run_outputs_path(config, f"fold{fold}", f"eval_{split_name}"),
            split=split_name, report_meta={'fold': fold, 'k': k, 'model_type': model_type,
                                           'n_train_patients': n_train_patients,
                                           'n_train_images': len(train_idx),
                                           'train_patient_subset': (os.path.basename(subset_path)
                                                                    if subset_path else None),
                                           'total_steps': total_steps},
            batch_size=config['training']['batch_size'], device=device,
        )


if __name__ == "__main__":
    main()
