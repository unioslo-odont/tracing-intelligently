"""Explicit landmark selection for saved evaluation reports.

Means and SDR can be pooled using point counts. Medians cannot be reconstructed
from per-landmark medians; use the saved per-point CSV for a subset median.
"""

import math


def select_report(report, exclude=()):
    excluded = set(exclude)
    per_landmark = report['per_landmark']
    unknown = excluded - set(per_landmark)
    if unknown:
        raise ValueError(f"Excluded landmarks absent from report: {sorted(unknown)}")
    if not excluded:
        return report
    selected = {name: stats for name, stats in per_landmark.items()
                if name not in excluded}
    valid = [s for s in selected.values() if s.get('n_points', 0) > 0
             and s.get('mre_mm') is not None and math.isfinite(s['mre_mm'])]
    if not valid:
        raise ValueError('No calibrated points remain after landmark selection')
    total = sum(s['n_points'] for s in valid)
    thresholds = set.intersection(*(set(s['sdr']) for s in valid))
    overall = {
        'mre_mm': sum(s['n_points'] * s['mre_mm'] for s in valid) / total,
        'n_points': total,
        'median_mm': None,
        'sdr': {},
    }
    for threshold in sorted(thresholds, key=float):
        values = [s for s in valid if s['sdr'][threshold] is not None]
        count = sum(s['n_points'] for s in values)
        overall['sdr'][threshold] = (
            sum(s['n_points'] * s['sdr'][threshold] for s in values) / count
            if count else None)
    return {**report, 'overall': overall, 'per_landmark': selected,
            'landmarks': sorted(selected), 'excluded_landmarks': sorted(excluded)}
