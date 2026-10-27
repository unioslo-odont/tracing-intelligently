"""
================================================================================
EVAL CORE — one shared per-landmark evaluation loop
================================================================================
DESCRIPTION:
    The single implementation of "run a trained model over a set of sample
    indices, convert pixel error to mm via each image's own Facad
    calibration, and write the detailed report". Used by:
      - evaluate.py / evaluate_backbone_variant.py (the standalone
        validation scripts), and
      - train.py / train_backbone_variant.py, which call it at the END of
        training on the BEST checkpoint for BOTH the train and val splits,
        so every run automatically emits detailed per-landmark accuracy on
        each split (train-vs-val per landmark is the direct signal for
        per-landmark overfitting).

    Keeping it here means all four scripts emit byte-identical report
    formats and the metric/decoding logic exists in exactly one place.

    Writes, for a given output_prefix:
      - <output_prefix>.json                    overall + per-landmark MRE/SDR (see metrics.py)
      - <output_prefix>_per_image.csv           one row per image (error summarized over landmarks)
      - <output_prefix>_per_landmark_image.csv  finest grain: one row per (image, landmark) -
                                                predicted point, ground-truth point, error (mm/px)
                                                and peak heatmap confidence. One such file per
                                                checkpoint, so together they give per-model x
                                                per-image x per-landmark data (e.g. to reconstruct
                                                the displacement vector for a mis-annotated image).
      - <output_prefix>_overlays/               optional GT-vs-pred PNGs (save_overlays=True)
================================================================================
"""

import csv
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from heatmap_codec import decode_heatmaps
from metrics import compute_radial_errors_px, summarize_errors
from visualize import save_validation_overlays


def evaluate_and_write(model, dataset, indices, *, config, landmark_names, output_prefix, split,
                       report_meta=None, sdr_thresholds_mm=(2.0, 2.5, 3.0, 4.0), select_landmarks=None,
                       batch_size=8, device=None, save_overlays=False, base_dataset=None,
                       transform=None, max_overlay_images=None, overlay_landmarks=None, quiet=False):
    """
    model: trained model (moved to `device`); switched to eval() here.
    dataset: a CephalometricDataset WITH the HeatmapTargetTransform set (yields tensors).
    indices: sample indices of the split to evaluate (e.g. a fold's val_idx / train_idx).
    config: the resolved run config (only config['data']['heatmap_size'] is read here).
    landmark_names: the model's ordered channel vocabulary.
    output_prefix: path prefix; '.json' and '_per_image.csv' are appended.
    split: 'val' / 'train' / 'all' - recorded in the report and printed.
    report_meta: extra top-level fields to merge into the report (e.g. fold, k, model_type).
    select_landmarks: optional subset to restrict the report to (see metrics.summarize_errors).
    save_overlays: also render GT-vs-pred overlays (needs base_dataset + transform).
    overlay_landmarks: optional trained-landmark subset to draw in both the
        ground-truth and prediction layers. The reported metrics are unaffected.

    Returns the report dict.
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    indices = list(indices)

    landmark_col_idx = None
    if select_landmarks:
        unknown = [name for name in select_landmarks if name not in landmark_names]
        if unknown:
            raise ValueError(f"Unknown landmark(s) {unknown}; available: {landmark_names}")
        landmark_col_idx = [landmark_names.index(name) for name in select_landmarks]

    hm_w, hm_h = config['data']['heatmap_size']
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False)

    all_errors_mm = []
    per_image_rows = []
    per_landmark_image_rows = []   # long/tidy: one row per (image, landmark)
    n_uncalibrated = 0

    def _num(x):
        """float(x) unless it's NaN, in which case '' (blank cell)."""
        return float(x) if not np.isnan(x) else ''

    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            pred_heatmaps = model(images)

            pred_coords_hm, pred_conf = decode_heatmaps(pred_heatmaps)
            gt_coords_hm = batch['landmarks_hm'].numpy()
            mask = batch['mask'].numpy()

            orig_w = batch['orig_size'][0].numpy().astype(np.float32)
            orig_h = batch['orig_size'][1].numpy().astype(np.float32)
            scale_x = (orig_w / hm_w)[:, None]
            scale_y = (orig_h / hm_h)[:, None]

            pred_coords_px = pred_coords_hm.copy()
            pred_coords_px[..., 0] *= scale_x
            pred_coords_px[..., 1] *= scale_y
            gt_coords_px = gt_coords_hm.copy()
            gt_coords_px[..., 0] *= scale_x
            gt_coords_px[..., 1] *= scale_y

            errors_px = compute_radial_errors_px(pred_coords_px, gt_coords_px, mask)

            pixel_spacing = batch['pixel_spacing_mm'].numpy().astype(np.float32)
            n_uncalibrated += int(np.isnan(pixel_spacing).sum())
            errors_mm = errors_px * pixel_spacing[:, None]
            all_errors_mm.append(errors_mm)

            for i, imagefile in enumerate(batch['imagefile']):
                row_errors = errors_mm[i]
                if landmark_col_idx is not None:
                    row_errors = row_errors[landmark_col_idx]
                valid = row_errors[~np.isnan(row_errors)]
                per_image_rows.append({
                    'imagefile': imagefile,
                    'serial': batch['serial'][i],
                    'n_landmarks': int(valid.size),
                    'mean_error_mm': float(valid.mean()) if valid.size else '',
                    'max_error_mm': float(valid.max()) if valid.size else '',
                    'pixel_spacing_mm': float(pixel_spacing[i]) if not np.isnan(pixel_spacing[i]) else '',
                })

                # one row per landmark of this image: predicted point (always,
                # in native px), ground-truth point + error (only where the
                # landmark is present in this image's tracing), and the peak
                # heatmap confidence. This is the finest-grained export -
                # per model (one file per checkpoint) x per image x per landmark.
                present_i = mask[i] > 0
                for j, lname in enumerate(landmark_names):
                    p = bool(present_i[j])
                    per_landmark_image_rows.append({
                        'imagefile': imagefile,
                        'serial': batch['serial'][i],
                        'landmark': lname,
                        'present': int(p),
                        'error_mm': _num(errors_mm[i, j]) if p else '',
                        'error_px': _num(errors_px[i, j]) if p else '',
                        'pred_x_px': float(pred_coords_px[i, j, 0]),
                        'pred_y_px': float(pred_coords_px[i, j, 1]),
                        'gt_x_px': float(gt_coords_px[i, j, 0]) if p else '',
                        'gt_y_px': float(gt_coords_px[i, j, 1]) if p else '',
                        'peak_conf': float(pred_conf[i, j]),
                        'pixel_spacing_mm': float(pixel_spacing[i]) if not np.isnan(pixel_spacing[i]) else '',
                    })

    all_errors_mm = np.concatenate(all_errors_mm, axis=0)
    report = summarize_errors(all_errors_mm, landmark_names, sdr_thresholds_mm=tuple(sdr_thresholds_mm),
                               select_landmarks=select_landmarks)
    report['split'] = split
    report['n_images'] = len(indices)
    report['n_uncalibrated_images'] = n_uncalibrated
    if select_landmarks:
        report['landmarks'] = list(select_landmarks)
    if report_meta:
        report.update(report_meta)

    if n_uncalibrated and not quiet:
        print(f"[WARN] {n_uncalibrated} image(s) had no valid Facad <scale> calibration - "
              "excluded from mm-based metrics (still detected/decoded, just not measurable in mm)")

    os.makedirs(os.path.dirname(output_prefix) or '.', exist_ok=True)
    with open(output_prefix + '.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    with open(output_prefix + '_per_image.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['imagefile', 'serial', 'n_landmarks',
                                               'mean_error_mm', 'max_error_mm', 'pixel_spacing_mm'])
        writer.writeheader()
        writer.writerows(per_image_rows)
    with open(output_prefix + '_per_landmark_image.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['imagefile', 'serial', 'landmark', 'present',
                                               'error_mm', 'error_px', 'pred_x_px', 'pred_y_px',
                                               'gt_x_px', 'gt_y_px', 'peak_conf', 'pixel_spacing_mm'])
        writer.writeheader()
        writer.writerows(per_landmark_image_rows)

    if not quiet:
        overall = report['overall']
        if overall['mre_mm'] is not None:
            print(f"[{split}] {report['n_images']} images, MRE={overall['mre_mm']:.3f} mm")
            for threshold, fraction in overall['sdr'].items():
                if fraction is not None:
                    print(f"  SDR @ {threshold} mm: {fraction * 100:.1f}%")
        else:
            print(f"[{split}] no valid points")
        print(f"Wrote {output_prefix}.json, {output_prefix}_per_image.csv and "
              f"{output_prefix}_per_landmark_image.csv")

    if save_overlays:
        overlay_dir = output_prefix + '_overlays'
        written = save_validation_overlays(
            model=model, base_dataset=base_dataset, transform=transform,
            indices=indices, device=device, output_dir=overlay_dir, max_images=max_overlay_images,
            display_landmarks=overlay_landmarks,
        )
        if not quiet:
            print(f"Wrote {len(written)} overlay image(s) to {overlay_dir}")

    return report
