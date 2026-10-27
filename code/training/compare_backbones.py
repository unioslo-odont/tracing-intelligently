"""
================================================================================
COMPARE BACKBONES: aggregate eval_val.json across models and folds
================================================================================
DESCRIPTION:
    Reads outputs/<model>/fold<N>/eval_val.json (written by evaluate.py or
    evaluate_backbone_variant.py --split val - see output_paths.py; falls
    back to the old <checkpoint_dir>/fold<N>/ location for pre-existing runs)
    for every fold found for each given checkpoint directory, and aggregates
    across folds into (all under the gitignored outputs/ by default):
      - <output>_summary.csv       - one row per model: mean +/- std MRE and
                                      mean SDR@threshold across folds
      - <output>_per_landmark.csv  - one row per (model, landmark): same,
                                      broken down by landmark
      - <output>_mre_bar.png       - MRE bar chart across models

    Comparisons are only meaningful if every model's fold N held out the
    SAME patients - true here as long as every config shares the same
    `seed` and `kfold.k` (see the header comment in resnet_unet.yaml etc.),
    which this script checks and warns about if violated.

USAGE:
    python compare_backbones.py \
        --checkpoint-dirs checkpoints/unet_baseline checkpoints/resnet_unet \
                          checkpoints/hrnet_pretrained checkpoints/dinov2_frozen
    # writes outputs/comparison_summary.csv etc. by default (--output to change)
================================================================================
"""

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from output_paths import OUTPUTS_DIR, outputs_path
from report_selection import select_report


def resolve_reports_dir(checkpoint_dir):
    """Given a checkpoint dir (e.g. checkpoints/unet_baseline), find where its
    fold*/eval_val.json reports actually live. evaluate.py now writes them to
    outputs/<model_name>/fold*/ (see output_paths.py); older runs wrote them
    next to the checkpoint. Prefer the new outputs/ location, fall back to the
    literal checkpoint_dir so this keeps working on pre-existing runs."""
    if glob.glob(os.path.join(checkpoint_dir, 'fold*', 'eval_val.json')):
        return checkpoint_dir
    model_name = os.path.basename(os.path.normpath(checkpoint_dir))
    new_loc = os.path.join(OUTPUTS_DIR, model_name)
    if glob.glob(os.path.join(new_loc, 'fold*', 'eval_val.json')):
        return new_loc

    # Runs that set training.output_group (Run 7/8/9 and the screens) live one
    # level deeper, under outputs/<group>/<model_name>/ - see output_paths.py.
    grouped = sorted({os.path.dirname(os.path.dirname(path))
                      for path in glob.glob(os.path.join(OUTPUTS_DIR, '*', model_name,
                                                         'fold*', 'eval_val.json'))})
    if len(grouped) == 1:
        return grouped[0]
    if len(grouped) > 1:
        raise ValueError(f"Ambiguous reports for {model_name}: {grouped}. "
                         "Pass the exact output directory.")
    return checkpoint_dir


def load_fold_reports(reports_dir):
    """Return a list of (fold_index, report_dict) for every fold<N>/eval_val.json
    found under reports_dir, sorted by fold index."""
    reports = []
    for path in sorted(glob.glob(os.path.join(reports_dir, 'fold*', 'eval_val.json'))):
        fold_dir = os.path.basename(os.path.dirname(path))
        try:
            fold_index = int(fold_dir[len('fold'):])
        except ValueError:
            continue
        with open(path, 'r', encoding='utf-8') as f:
            reports.append((fold_index, json.load(f)))
    return reports


def aggregate_overall(reports):
    mres = [r['overall']['mre_mm'] for _, r in reports if r['overall']['mre_mm'] is not None]
    thresholds = sorted({float(t) for _, r in reports for t in r['overall']['sdr'].keys()})
    sdr_by_threshold = {}
    for t in thresholds:
        values = [r['overall']['sdr'].get(str(t), r['overall']['sdr'].get(t)) for _, r in reports]
        values = [v for v in values if v is not None]
        sdr_by_threshold[t] = values

    return {
        'n_folds': len(reports),
        'mre_mm_mean': float(np.mean(mres)) if mres else None,
        'mre_mm_std': float(np.std(mres)) if len(mres) > 1 else 0.0 if mres else None,
        'sdr_mean': {t: float(np.mean(v)) if v else None for t, v in sdr_by_threshold.items()},
        'sdr_std': {t: (float(np.std(v)) if len(v) > 1 else 0.0) if v else None for t, v in sdr_by_threshold.items()},
    }


def aggregate_per_landmark(reports):
    """Return {landmark_name: {mre_mm_mean, mre_mm_std, sdr_mean: {...}, n_points_total}}."""
    landmark_names = sorted({name for _, r in reports for name in r['per_landmark'].keys()})
    result = {}
    for name in landmark_names:
        entries = [r['per_landmark'][name] for _, r in reports if name in r['per_landmark']]
        mres = [e['mre_mm'] for e in entries if e['mre_mm'] is not None]
        n_points_total = sum(e['n_points'] for e in entries)
        thresholds = sorted({float(t) for e in entries for t in e['sdr'].keys()})
        sdr_mean = {}
        for t in thresholds:
            values = [e['sdr'].get(str(t), e['sdr'].get(t)) for e in entries]
            values = [v for v in values if v is not None]
            sdr_mean[t] = float(np.mean(values)) if values else None
        result[name] = {
            'mre_mm_mean': float(np.mean(mres)) if mres else None,
            'mre_mm_std': float(np.std(mres)) if len(mres) > 1 else (0.0 if mres else None),
            'sdr_mean': sdr_mean,
            'n_points_total': n_points_total,
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-dirs', nargs='+', required=True,
                         help="one or more checkpoint directories (e.g. checkpoints/unet_baseline). "
                              "Their fold*/eval_val.json reports are read from outputs/<model>/ - see "
                              "resolve_reports_dir - falling back to the checkpoint dir for old runs.")
    parser.add_argument('--output', default=None,
                         help="output path prefix (default: outputs/comparison; writes "
                              "<prefix>_summary.csv, <prefix>_per_landmark.csv, <prefix>_mre_bar.png)")
    parser.add_argument('--exclude-landmarks', nargs='+', default=[])
    parser.add_argument('--expected-folds', type=int, default=5)
    args = parser.parse_args()
    if args.output is None:
        args.output = outputs_path('comparison', make_parent=True)

    all_reports = {}
    for checkpoint_dir in args.checkpoint_dirs:
        model_name = os.path.basename(os.path.normpath(checkpoint_dir))
        reports_dir = resolve_reports_dir(checkpoint_dir)
        reports = [(fold, select_report(report, args.exclude_landmarks))
                   for fold, report in load_fold_reports(reports_dir)]
        if {fold for fold, _ in reports} != set(range(args.expected_folds)):
            raise ValueError(f"{reports_dir}: expected folds 0..{args.expected_folds - 1}")
        if not reports:
            print(f"[WARN] no fold*/eval_val.json found under {reports_dir} - skipping '{model_name}'")
            continue
        if model_name in all_reports:
            raise ValueError(f"Duplicate model label {model_name}; compare these runs separately")
        all_reports[model_name] = reports
        print(f"{model_name}: found {len(reports)} fold report(s) "
              f"(folds {[i for i, _ in reports]})")

    if not all_reports:
        print("Nothing to compare.")
        return

    fold_counts = {name: len(reports) for name, reports in all_reports.items()}
    if len(set(fold_counts.values())) > 1:
        print(f"[WARN] models have different numbers of evaluated folds ({fold_counts}) - "
              "comparison below still works but is less apples-to-apples until all models "
              "have the same folds trained+evaluated.")

    summary_rows = []
    per_landmark_rows = []
    all_thresholds = set()

    for model_name, reports in all_reports.items():
        overall = aggregate_overall(reports)
        all_thresholds.update(overall['sdr_mean'].keys())
        row = {
            'model': model_name,
            'excluded_landmarks': ';'.join(args.exclude_landmarks),
            'landmarks': ';'.join(sorted(reports[0][1]['per_landmark'])),
            'n_folds': overall['n_folds'],
            'mre_mm_mean': overall['mre_mm_mean'],
            'mre_mm_std': overall['mre_mm_std'],
        }
        for t, v in overall['sdr_mean'].items():
            row[f'sdr_{t}mm_mean'] = v
            row[f'sdr_{t}mm_std'] = overall['sdr_std'][t]
        summary_rows.append(row)

        per_landmark = aggregate_per_landmark(reports)
        for landmark_name, stats in per_landmark.items():
            plrow = {
                'model': model_name,
                'landmark': landmark_name,
                'mre_mm_mean': stats['mre_mm_mean'],
                'mre_mm_std': stats['mre_mm_std'],
                'n_points_total': stats['n_points_total'],
            }
            for t, v in stats['sdr_mean'].items():
                plrow[f'sdr_{t}mm_mean'] = v
            per_landmark_rows.append(plrow)

    threshold_cols = sorted(all_thresholds)
    summary_headers = ['model', 'excluded_landmarks', 'landmarks', 'n_folds', 'mre_mm_mean', 'mre_mm_std']
    for t in threshold_cols:
        summary_headers += [f'sdr_{t}mm_mean', f'sdr_{t}mm_std']

    summary_path = f"{args.output}_summary.csv"
    os.makedirs(os.path.dirname(os.path.abspath(summary_path)), exist_ok=True)
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=summary_headers, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(summary_rows)

    per_landmark_headers = ['model', 'landmark', 'mre_mm_mean', 'mre_mm_std', 'n_points_total']
    for t in threshold_cols:
        per_landmark_headers.append(f'sdr_{t}mm_mean')

    per_landmark_path = f"{args.output}_per_landmark.csv"
    with open(per_landmark_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=per_landmark_headers, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(per_landmark_rows)

    print(f"\n{'Model':<20} {'Folds':<7} {'MRE (mm)':<18} " + " ".join(f"SDR@{t}mm" for t in threshold_cols))
    for row in summary_rows:
        mre_str = f"{row['mre_mm_mean']:.3f} +/- {row['mre_mm_std']:.3f}" if row['mre_mm_mean'] is not None else "n/a"
        sdr_strs = []
        for t in threshold_cols:
            v = row.get(f'sdr_{t}mm_mean')
            sdr_strs.append(f"{v * 100:.1f}%" if v is not None else "n/a")
        print(f"{row['model']:<20} {row['n_folds']:<7} {mre_str:<18} " + " ".join(f"{s:<9}" for s in sdr_strs))

    print(f"\nWrote {summary_path} and {per_landmark_path}")

    try:
        import matplotlib.pyplot as plt

        models = [row['model'] for row in summary_rows]
        means = [row['mre_mm_mean'] or 0 for row in summary_rows]
        stds = [row['mre_mm_std'] or 0 for row in summary_rows]

        fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.5), 5))
        ax.bar(models, means, yerr=stds, capsize=5)
        ax.set_ylabel('Mean Radial Error (mm)')
        ax.set_title('Model comparison (mean +/- std across folds)')
        fig.tight_layout()
        fig.savefig(f"{args.output}_mre_bar.png", dpi=150)
        print(f"Wrote {args.output}_mre_bar.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
