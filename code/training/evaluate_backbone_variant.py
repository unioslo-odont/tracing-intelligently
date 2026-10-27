"""
================================================================================
EVALUATE: pretrained-backbone variant checkpoints
================================================================================
DESCRIPTION:
    Same as evaluate.py, but builds the model via model_factory instead of
    hardcoding UNetHeatmapModel, so it can load checkpoints produced by
    train_backbone_variant.py. See evaluate.py's docstring for the metrics
    themselves (MRE/SDR, mm conversion, per-image CSV, optional overlays) -
    identical here, this file only differs in how the model is constructed.

USAGE:
    python evaluate_backbone_variant.py --checkpoint /path/to/fold0/best.pt --split val
================================================================================
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'datasets')))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')))

from heatmap_transform import HeatmapTargetTransform
from fold_manifest import resolve_kfold_split
from label_efficiency import load_patient_subset, restrict_indices_to_patients
from model_factory import build_model
from eval_core import evaluate_and_write
from output_paths import outputs_path, model_fold_from_checkpoint

from evaluate import build_base_dataset  # noqa: F401 (reused unchanged from evaluate.py)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--split', choices=['val', 'train', 'all'], default='val')
    parser.add_argument('--output', default=None,
                         help="output path prefix (default: outputs/<model>/<fold>/eval_<split>, "
                              "derived from the checkpoint path - see code/training/output_paths.py)")
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--sdr-thresholds-mm', type=float, nargs='+', default=[2.0, 2.5, 3.0, 4.0])
    parser.add_argument('--landmarks', nargs='+', default=None,
                         help="restrict 'overall' and 'per_landmark' stats to just these landmarks "
                              "(e.g. --landmarks A B S N); default is all landmarks")
    parser.add_argument('--save-overlays', action='store_true',
                         help="also write one ground-truth-vs-predicted overlay PNG per image")
    parser.add_argument('--max-overlay-images', type=int, default=None)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    landmark_names = checkpoint['landmark_names']

    transform = HeatmapTargetTransform(
        landmark_names=landmark_names,
        image_size=tuple(config['data']['image_size']),
        heatmap_size=tuple(config['data']['heatmap_size']),
        sigma=config['data'].get('sigma', 2.0),
    )
    base_dataset = build_base_dataset(config)
    dataset = build_base_dataset(config)
    dataset.transform = transform

    k = config['kfold']['k']
    fold = config['kfold']['fold']
    # Same split resolution as training: verified against kfold.manifest_path when
    # the checkpoint's config has one (paths in a checkpoint are already absolute).
    train_idx, val_idx = resolve_kfold_split(base_dataset, config, fold=fold)

    # If this checkpoint was trained on a label-efficiency subset, 'train' means
    # the subset it actually saw - not the whole fold, which it never trained on.
    subset_path = config['data'].get('train_patient_subset_path')
    if subset_path:
        train_idx = restrict_indices_to_patients(base_dataset, train_idx, load_patient_subset(subset_path))
        print(f"[label-efficiency] --split train = the {len(train_idx)} images this checkpoint "
              f"was trained on ({os.path.basename(subset_path)})")

    indices = {'val': val_idx, 'train': train_idx, 'all': np.arange(len(base_dataset))}[args.split]

    if len(indices) == 0:
        raise RuntimeError(f"No images in the '{args.split}' split for fold {fold}/{k}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(config['model'], num_landmarks=len(landmark_names)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    default_suffix = f"eval_{args.split}" + (f"_{'-'.join(args.landmarks)}" if args.landmarks else "")
    if args.output:
        output_prefix = args.output
    else:
        model_name, fold_dir = model_fold_from_checkpoint(args.checkpoint)
        # training.output_group (if set) keeps an experiment family together under
        # outputs/<group>/ - mirrors what training itself writes (output_paths.py).
        group = (config.get('training') or {}).get('output_group')
        parts = ([group] if group else []) + [model_name, fold_dir, default_suffix]
        output_prefix = outputs_path(*parts)

    evaluate_and_write(
        model, dataset, indices,
        config=config, landmark_names=landmark_names, output_prefix=output_prefix,
        split=args.split, report_meta={'fold': fold, 'k': k, 'model_type': config['model'].get('type', 'unet')},
        sdr_thresholds_mm=args.sdr_thresholds_mm, select_landmarks=args.landmarks,
        batch_size=args.batch_size, device=device,
        save_overlays=args.save_overlays, base_dataset=base_dataset, transform=transform,
        max_overlay_images=args.max_overlay_images,
    )


if __name__ == "__main__":
    main()
