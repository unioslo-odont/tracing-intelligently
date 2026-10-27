"""
================================================================================
LANDMARK OVERLAYS FOR A NAMED SET OF IMAGES (manual inspection)
================================================================================
DESCRIPTION:
    Render the ground-truth .fcd landmarks on a specific list of images and
    save one PNG each, for eyeballing. Unlike evaluate.py --save-overlays, this
    DELIBERATELY bypasses the dataset's exclusion list and the in-bounds guard
    (min_landmarks_in_bounds=0), so you can see the very images those filters
    removed - e.g. the frame-mismatched tracings named in the exclusion list.

    Ground truth is drawn as green circles + labels. On a frame-mismatched
    image the whole set of green points sits OFF the anatomy (shifted into the
    margin / off the skull) - that's the visual confirmation.

    Optional --checkpoint additionally overlays the model's predictions (red
    crosshairs), so you can compare: green (stored tracing, displaced) vs red
    (model, on the anatomy) - the same picture used to diagnose these images.

USAGE:
    # ground truth only (no model needed - runs anywhere, e.g. a login node):
    python visualize_images.py --config configs/unet_baseline.yaml \
        --images-file configs/excluded_images.csv --output-dir outputs/excluded_overlays

    # also overlay a model's predictions:
    python visualize_images.py --config configs/unet_baseline.yaml \
        --images-file configs/excluded_images.csv \
        --checkpoint checkpoints/unet_baseline/fold0/best.pt

    # a handful of explicit images instead of a file:
    python visualize_images.py --config configs/unet_baseline.yaml \
        --images <case>_1.tif <othercase>_2.tif
================================================================================
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch
from PIL import ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'datasets')))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')))

from config import load_config
from cephalometric_dataset import CephalometricDataset
from heatmap_transform import HeatmapTargetTransform
from model_factory import build_model
from visualize import _to_display_rgb, predict_landmarks
from output_paths import outputs_path


def read_image_list(args):
    if args.images:
        return list(args.images)
    with open(args.images_file, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
    if lines and 'imagefile' in lines[0].lower():
        return [r['imagefile'].strip() for r in csv.DictReader(lines) if (r.get('imagefile') or '').strip()]
    return [ln.split(',')[0].strip() for ln in lines if ln.strip() and not ln.startswith('#')]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="a model config YAML - used only for its data.{images_dir,fcd_dir,review_csv_path}")
    parser.add_argument('--checkpoint', default=None, help="optional: also overlay this model's predictions (red)")
    parser.add_argument('--images', nargs='+', default=None, help="explicit image filenames")
    parser.add_argument('--images-file', default=None, help="CSV with an 'imagefile' column, or one filename per line")
    parser.add_argument('--output-dir', default=None, help="default: outputs/landmark_overlays")
    args = parser.parse_args()

    if not args.images and not args.images_file:
        parser.error("give --images or --images-file")

    # data location comes from the checkpoint's own config if available, else --config
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        data_cfg = checkpoint['config']['data']
    elif args.config:
        data_cfg = load_config(args.config)['data']
    else:
        parser.error("give --config (for data paths) and/or --checkpoint")

    # Build the dataset with the exclusion list AND the in-bounds guard OFF, so
    # the flagged/excluded images load and can be inspected.
    dataset = CephalometricDataset(
        images_dir=data_cfg['images_dir'],
        fcd_dir=data_cfg['fcd_dir'],
        review_csv_path=data_cfg.get('review_csv_path'),
        combine_method=data_cfg.get('combine_method', 'median'),
        min_landmarks_in_bounds=0.0,
        exclude_images_path=None,
    )
    by_name = {s['imagefile']: i for i, s in enumerate(dataset.samples)}
    skipped_reason = dict(dataset.skipped)

    model = transform = device = None
    if args.checkpoint:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        landmark_names = checkpoint['landmark_names']
        transform = HeatmapTargetTransform(
            landmark_names=landmark_names,
            image_size=tuple(checkpoint['config']['data']['image_size']),
            heatmap_size=tuple(checkpoint['config']['data']['heatmap_size']),
            sigma=checkpoint['config']['data'].get('sigma', 2.0),
        )
        model = build_model(checkpoint['config']['model'], num_landmarks=len(landmark_names)).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

    output_dir = args.output_dir or outputs_path('landmark_overlays', make_parent=False)
    os.makedirs(output_dir, exist_ok=True)

    for name in read_image_list(args):
        idx = by_name.get(name)
        if idx is None:
            reason = skipped_reason.get(name, "not found in images_dir / no matching .fcd")
            print(f"[skip] {name}: {reason}")
            continue

        sample = dataset[idx]
        pil = _to_display_rgb(sample['image'])
        draw = ImageDraw.Draw(pil)

        for lname, (x, y) in zip(sample['landmark_names'], sample['landmarks']):
            r = 5
            draw.ellipse((x - r, y - r, x + r, y + r), outline='lime', width=2)
            draw.text((x + 7, y - 7), lname, fill='lime')

        if model is not None:
            coords_px, _ = predict_landmarks(model, transform, sample['image'], device)
            for lname, (x, y) in zip(transform.landmark_names, coords_px):
                r = 6
                draw.line((x - r, y, x + r, y), fill='red', width=2)
                draw.line((x, y - r, x, y + r), fill='red', width=2)

        out_path = os.path.join(output_dir, os.path.splitext(name)[0] + '_overlay.png')
        pil.save(out_path)
        n_out = sum(1 for (x, y) in sample['landmarks']
                    if not (0 <= x <= sample['image'].shape[1] and 0 <= y <= sample['image'].shape[0]))
        flag = f"  [{n_out}/{len(sample['landmarks'])} GT points OUTSIDE the image]" if n_out else ""
        print(f"Wrote {out_path}  (image {sample['image'].shape[1]}x{sample['image'].shape[0]}){flag}")

    legend = "green = ground-truth .fcd landmarks" + (" ; red = model prediction" if model is not None else "")
    print(f"\nDone. {legend}. Output: {output_dir}")


if __name__ == "__main__":
    main()
