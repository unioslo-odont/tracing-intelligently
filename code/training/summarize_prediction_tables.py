"""Pool saved per-point errors without inference. Run inside the secure environment."""

import argparse
import csv
import json
import math
import os
import statistics


def summarize_tables(paths, landmarks=None, exclude=()):
    selected = set(landmarks) if landmarks else None
    excluded = set(exclude)
    errors, by_landmark, observed, seen = [], {}, set(), set()
    for path in paths:
        with open(path, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                name = row['landmark']
                observed.add(name)
                if name in excluded or (selected is not None and name not in selected):
                    continue
                raw = row['error_mm']
                if not raw:
                    continue
                value = float(raw)
                if not math.isfinite(value):
                    raise ValueError('Non-finite recorded error; inspect input tables privately')
                key = (row.get('serial', ''), row['imagefile'], name)
                if key in seen:
                    raise ValueError('Duplicate image/landmark: do not pool overlapping evaluation tables')
                seen.add(key)
                errors.append(value)
                by_landmark.setdefault(name, []).append(value)
    unknown = ((selected or set()) | excluded) - observed
    if unknown:
        raise ValueError(f'Unknown requested landmarks: {sorted(unknown)}')
    if not errors:
        raise ValueError('No scored points')

    def summary(values):
        return {'n_points': len(values), 'mre_mm': statistics.mean(values),
                'median_mm': statistics.median(values),
                'sdr': {str(t): sum(v <= t for v in values) / len(values)
                        for t in (2.0, 2.5, 3.0, 4.0)}}

    return {'aggregation': 'pooled image-landmark errors; not patient-weighted',
            'landmarks': sorted(by_landmark), 'excluded_landmarks': sorted(excluded),
            'overall': summary(errors),
            'per_landmark': {name: summary(values) for name, values in sorted(by_landmark.items())}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs='+', required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--landmarks', nargs='+')
    group.add_argument('--exclude-landmarks', nargs='+', default=[])
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = summarize_tables(args.inputs, args.landmarks, args.exclude_landmarks)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'x', encoding='utf-8') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps(report['overall'], indent=2))


if __name__ == '__main__':
    main()
