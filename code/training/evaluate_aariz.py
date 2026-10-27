"""
================================================================================
EVALUATE: a trained checkpoint against the Aariz external test set
================================================================================
DESCRIPTION:
    Loads a checkpoint already trained on the TSD/Facad cohort (see
    train.py / train_backbone_variant.py) and evaluates it, WITHOUT any
    further training, against the Aariz / CEPHA29 benchmark read in its
    ORIGINAL published directory layout (see code/datasets/aariz_dataset.py:
    what this dataset is, why it is evaluation-only, and how its landmark
    symbols map onto this project's vocabulary). Uses the SENIOR orthodontist
    tracings by default. Reuses the exact same per-landmark evaluation loop as
    the internal validation scripts (code/training/eval_core.py) - only the
    dataset source and the landmark-name mapping differ - so the report format
    is byte-identical (overall + per-landmark MRE/SDR, per-image CSV, and the
    fine-grained per-(image, landmark) CSV).

    Only landmarks with a confirmed name mapping
    (aariz_dataset.AARIZ_TO_REFERENCE_MAPPING) are checkable. The rest of a
    checkpoint's vocabulary (e.g. Ba, GNs, MS, Pt, iGo/pGo - no Aariz
    counterpart) show 0 points / null stats in the report: that is correct,
    not a bug - there is no ground truth for them in this dataset.

    Calibration: per-image mm/pixel comes from
    <root>/cephalogram_machine_mappings.csv (7 scanners, 0.089-1.139 mm/px).
    aariz_dataset.py auto-detects the id and mm-per-pixel columns; override with
    --id-column / --mm-per-pixel-column if needed. Without a usable value an
    image's mm error is null (the per-(image, landmark) CSV still records raw
    pixel error), and a pooled pixel-space number across scanners of such
    different resolution is not meaningful, so calibration matters here.

USAGE:
    python evaluate_aariz.py \
        --checkpoint checkpoints/unet_baseline/final/best.pt \
        --root /path/to/project/data/Aariz \
        --split test --annotator senior
================================================================================
"""

import argparse
import copy
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'datasets')))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')))

from heatmap_transform import HeatmapTargetTransform
from aariz_dataset import AarizDataset, AARIZ_TO_REFERENCE_MAPPING
from model_factory import build_model
from eval_core import evaluate_and_write
from output_paths import outputs_path, model_fold_from_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--root', required=True,
                        help="Aariz dataset root (contains train/ valid/ test/ and "
                             "cephalogram_machine_mappings.csv)")
    parser.add_argument('--split', default='test', choices=['train', 'valid', 'test', 'all'],
                        help="Aariz split to evaluate on (default: test - the held-out benchmark split)")
    parser.add_argument('--annotator', default='senior', choices=['senior', 'junior', 'both'],
                        help="which orthodontist tracing to score against (default: senior)")
    parser.add_argument('--machine-mappings-csv', default=None,
                        help="per-image mm/px calibration CSV (default: "
                             "<root>/cephalogram_machine_mappings.csv if present)")
    parser.add_argument('--id-column', default=None,
                        help="override the auto-detected id column in the machine-mappings CSV")
    parser.add_argument('--mm-per-pixel-column', default=None,
                        help="override the auto-detected mm-per-pixel column in the machine-mappings CSV")
    parser.add_argument('--no-pns-pm', action='store_true',
                        help="drop the Aariz PNS -> our Pm mapping (ON by default; confirmed equivalent "
                             "2026-08-03). Use only for an 18-landmark sensitivity check.")
    parser.add_argument('--output', default=None,
                        help="output path prefix (default: outputs/<model>/<fold>/eval_aariz_<split>, "
                             "derived from the checkpoint path - see code/training/output_paths.py)")
    parser.add_argument('--unmap', nargs='+', default=None, metavar='OUR_NAME',
                        help="mark these OUR-landmark names UNCHECKABLE on Aariz by dropping their "
                             "mapping entries, e.g. --unmap Gn (used for the 15-point set, where our "
                             "Gn=Menton-candidate's identity is pending confirmation).")
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--sdr-thresholds-mm', type=float, nargs='+', default=[2.0, 2.5, 3.0, 4.0])
    parser.add_argument('--save-overlays', action='store_true')
    parser.add_argument('--max-overlay-images', type=int, default=None)
    parser.add_argument('--overlay-landmarks', nargs='+', default=None,
                        help="restrict BOTH ground-truth and predicted points drawn on overlays to "
                             "this subset of trained channels; default draws every trained channel")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    landmark_names = checkpoint['landmark_names']

    name_mapping = copy.deepcopy(AARIZ_TO_REFERENCE_MAPPING)
    if args.no_pns_pm:
        name_mapping.pop('PNS', None)
    if args.unmap:
        # name_mapping is {aariz_symbol: our_name}; drop every entry pointing at a requested our_name
        for our_name in args.unmap:
            for a_sym in [k for k, v in name_mapping.items() if v == our_name]:
                name_mapping.pop(a_sym, None)
        print(f"[unmap] marked uncheckable on Aariz: {sorted(set(args.unmap))}")

    transform = HeatmapTargetTransform(
        landmark_names=landmark_names,
        image_size=tuple(config['data']['image_size']),
        heatmap_size=tuple(config['data']['heatmap_size']),
        sigma=config['data'].get('sigma', 2.0),
    )

    common = dict(root=args.root, split=args.split, annotator=args.annotator,
                  name_mapping=name_mapping, machine_mappings_csv=args.machine_mappings_csv,
                  id_column=args.id_column, mm_per_pixel_column=args.mm_per_pixel_column)

    base_dataset = AarizDataset(**common)
    if len(base_dataset) == 0:
        raise RuntimeError(
            f"No usable Aariz images found for split={args.split!r} "
            f"({len(base_dataset.skipped)} skipped). First few reasons: "
            f"{base_dataset.skipped[:3]} - check --root/--split/--annotator.")

    mappable = set()
    for entry in base_dataset.samples:
        mappable.update(entry['landmarks'].keys())
    checkable = sorted(set(landmark_names) & mappable)
    uncheckable = sorted(set(landmark_names) - mappable)
    n_cal = sum(1 for e in base_dataset.samples if e['pixel_spacing_mm'] is not None)
    print(f"Aariz {args.split} ({args.annotator}): {len(base_dataset)} images "
          f"({len(base_dataset.skipped)} skipped), {n_cal} calibrated (mm/px known)")
    print(f"Checkable landmarks (confirmed Aariz correspondence): {checkable}")
    if uncheckable:
        print(f"NOT checkable (no correspondence in aariz_dataset.py's mapping): {uncheckable}")

    dataset = AarizDataset(transform=transform, **common)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(config['model'], num_landmarks=len(landmark_names)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if args.output:
        output_prefix = args.output
    else:
        model_name, fold_dir = model_fold_from_checkpoint(args.checkpoint)
        output_prefix = outputs_path(model_name, fold_dir, f'eval_aariz_{args.split}')

    report = evaluate_and_write(
        model, dataset, list(range(len(dataset))),
        config=config, landmark_names=landmark_names,
        output_prefix=output_prefix, split=f'aariz_{args.split}',
        report_meta={
            'source': 'aariz_external_test',
            'checkpoint': args.checkpoint,
            'annotator': args.annotator,
            'checkable_landmarks': checkable,
            'uncheckable_landmarks': uncheckable,
        },
        sdr_thresholds_mm=tuple(args.sdr_thresholds_mm),
        batch_size=args.batch_size, device=device,
        save_overlays=args.save_overlays, base_dataset=base_dataset,
        transform=transform, max_overlay_images=args.max_overlay_images,
        overlay_landmarks=args.overlay_landmarks,
    )

    if report['overall']['mre_mm'] is None:
        print("\n[WARN] No image had a usable mm/pixel calibration - every mm-based MRE/SDR is null. "
              "Raw per-landmark pixel error is still in <output>_per_landmark_image.csv. "
              "Aariz spans 7 scanners (0.089-1.139 mm/px), so a pooled pixel-space number across the "
              "whole set is not meaningful either - supply/repair the machine-mappings calibration "
              "(see --machine-mappings-csv / --id-column / --mm-per-pixel-column).")


if __name__ == "__main__":
    main()
