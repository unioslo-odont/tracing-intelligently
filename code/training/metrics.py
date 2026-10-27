"""
================================================================================
LANDMARK ERROR METRICS
================================================================================
DESCRIPTION:
    MRE (mean radial error) and SDR (successful detection rate) at one or
    more mm thresholds, both overall and per-landmark, following standard
    cephalometric landmark detection evaluation practice (e.g. ISBI 2015/
    CL-Detection). Reports per-landmark breakdowns, not just an aggregate
    number, and excludes landmarks that weren't actually marked in a given
    tracing (via `mask`) rather than silently counting them as zero error.

    pixel_spacing_mm comes from each image's own Facad calibration (see
    facad_io.get_image_pixel_spacing_mm) - images with unknown/invalid
    calibration produce NaN mm errors and are excluded from mm-based
    metrics; track how many separately (see evaluate.py) since a large
    uncalibrated count would make the mm metrics unreliable.
================================================================================
"""

import numpy as np


def compute_radial_errors_px(pred_coords, gt_coords, mask):
    """pred_coords, gt_coords: (N, C, 2) arrays in the SAME pixel space.
    mask: (N, C) array, 1 where that landmark is present in the ground truth.
    Returns (N, C) radial error in pixels, NaN where mask == 0."""
    diff = pred_coords - gt_coords
    errors = np.sqrt((diff ** 2).sum(axis=-1))
    return np.where(mask > 0, errors, np.nan)


def summarize_errors(errors_mm, landmark_names, sdr_thresholds_mm=(2.0, 2.5, 3.0, 4.0), select_landmarks=None):
    """errors_mm: (N, C) array of per-sample, per-landmark radial error in mm
    (NaN for missing/uncalibrated entries). Returns:
        {
          'overall': {mre_mm, n_points, sdr: {threshold: fraction}},
          'per_landmark': {name: {mre_mm, n_points, sdr: {...}}, ...},
        }

    select_landmarks: optional subset of landmark_names. When given, both
    'overall' and 'per_landmark' are restricted to just these landmarks (e.g.
    the four landmarks feeding SNA/SNB/ANB), rather than all of them.
    """
    def _summary(values):
        values = values[np.isfinite(values)]
        if values.size == 0:
            return {'mre_mm': None, 'median_mm': None, 'n_points': 0,
                    'sdr': {t: None for t in sdr_thresholds_mm}}
        return {
            'mre_mm': float(values.mean()),
            'median_mm': float(np.median(values)),
            'n_points': int(values.size),
            'sdr': {t: float((values <= t).mean()) for t in sdr_thresholds_mm},
        }

    if select_landmarks is not None:
        unknown = [name for name in select_landmarks if name not in landmark_names]
        if unknown:
            raise ValueError(f"Unknown landmark(s) {unknown}; available: {landmark_names}")
        col_idx = [landmark_names.index(name) for name in select_landmarks]
        errors_mm = errors_mm[:, col_idx]
        landmark_names = list(select_landmarks)

    overall = _summary(errors_mm.reshape(-1))
    per_landmark = {name: _summary(errors_mm[:, i]) for i, name in enumerate(landmark_names)}
    return {'overall': overall, 'per_landmark': per_landmark}
