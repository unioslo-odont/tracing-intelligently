"""
================================================================================
AGGREGATE THE LABEL-EFFICIENCY CURVE (Run 8)
================================================================================
DESCRIPTION:
    One command turns the per-(arm, fraction, fold) eval_val.json reports into
    the curve's table and its figure:

      outputs/label_efficiency/summary.csv              per-fold + pooled rows
      outputs/label_efficiency/label_efficiency_curve.png

    Fraction 1.0 is Run 6 itself: its reports are READ from outputs/<arm>/fold*/,
    never regenerated, which is the whole reason the fold manifest is asserted
    (a re-dealt split would silently make the 100% point incomparable with the
    rest of the curve - see code/datasets/fold_manifest.py).

    The figure has two panels because the curve alone cannot answer the actual
    question. Panel A is the level (how accuracy degrades as the cohort shrinks);
    panel B is the PAIRED difference between the arms at each size, with the five
    per-fold points shown, because the folds are paired (identical held-out
    patients) and a 0.1 mm mean difference with fold-to-fold scatter of 0.3 mm is
    a tie no matter how far apart the two lines look in panel A.

USAGE:
    python code/training/aggregate_label_efficiency.py                     # 1.0, 0.5, 0.25
    python code/training/aggregate_label_efficiency.py --fractions 0.5 0.25 0.125
================================================================================
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'datasets')))

from fold_manifest import read_fold_manifest, train_patients_for_fold
from label_efficiency import arm_run_name, format_fraction
from output_paths import OUTPUTS_DIR
from report_selection import select_report

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
DEFAULT_MANIFEST = os.path.join(REPO_ROOT, 'configs', 'folds', 'oslo26_k5_seed0.csv')

# Categorical slots 1 and 2 of the reference palette, in fixed order - one hue
# per arm, assigned by identity and never by rank, so the colours mean the same
# thing in every figure of this project.
ARM_COLOURS = {'unet_baseline': '#2a78d6', 'hrnet_pretrained': '#eb6834'}
FALLBACK_COLOURS = ['#1baf7a', '#eda100', '#e87ba4']
ARM_LABELS = {'unet_baseline': 'U-Net (from scratch)', 'hrnet_pretrained': 'HRNet (ImageNet-pretrained)'}
# Short forms for the direct labels at the line ends: the legend carries the full
# name, so the in-plot label only has to disambiguate two lines that may sit a
# tenth of a millimetre apart.
ARM_SHORT = {'unet_baseline': 'U-Net', 'hrnet_pretrained': 'HRNet'}
INK, INK_MUTED, GRID = '#0b0b0b', '#52514e', '#d8d7d2'


def sdr_at(report_overall, threshold):
    """SDR at a threshold, tolerating '2.0' / 2.0 / '2' key spellings."""
    sdr = report_overall.get('sdr') or {}
    for key in (str(threshold), str(float(threshold)), threshold, float(threshold)):
        if key in sdr:
            return sdr[key]
    return None


def subset_overall(report, exclude):
    """Recompute an `overall` block from the per-landmark stats, EXCLUDING `exclude`.

    Lets the curve be reported over the SAME landmark subset as the headline metric.
    Concretely: channels that are trained but not reported as clinical points (the two
    gonion tangent channels, whose clinical Gonion is constructed post hoc) would
    otherwise inflate every cell and make the figure disagree with the reported number.
    Weighted by each landmark's n_points, so it equals a pooled per-point mean.
    """
    per_lm = report.get('per_landmark') or {}
    items = [v for name, v in per_lm.items()
             if name not in exclude and v.get('mre_mm') is not None and v.get('n_points')]
    if not items:
        return report.get('overall', {})
    total = sum(v['n_points'] for v in items)
    out = {'mre_mm': sum(v['n_points'] * v['mre_mm'] for v in items) / total,
           'n_points': total, 'sdr': {}}
    for key in (items[0].get('sdr') or {}):
        vals = [(v['n_points'], (v.get('sdr') or {}).get(key)) for v in items]
        vals = [(n, s) for n, s in vals if s is not None]
        if vals:
            out['sdr'][key] = sum(n * s for n, s in vals) / sum(n for n, _ in vals)
    return out


def report_path(arm, fraction, fold, group, baseline_root=None):
    """Where this cell's val report lives. Fraction 1.0 = the Run 6 arm itself."""
    if float(fraction) >= 1.0:
        return os.path.join(baseline_root or OUTPUTS_DIR, arm, f"fold{fold}", 'eval_val.json')
    return os.path.join(OUTPUTS_DIR, group, arm_run_name(arm, fraction), f"fold{fold}", 'eval_val.json')


def load_cell(arm, fraction, fold, group, manifest_map, exclude_landmarks=(), baseline_root=None):
    """One (arm, fraction, fold) row, or None if that run hasn't finished."""
    path = report_path(arm, fraction, fold, group, baseline_root)
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        report = json.load(f)
    overall = select_report(report, exclude_landmarks)['overall']

    n_patients = report.get('n_train_patients')
    n_images = report.get('n_train_images')
    if n_patients is None and manifest_map is not None:
        # Run 6 predates the training-cohort fields in the report: recover the
        # training-patient count from the manifest (this fold's train side)...
        n_patients = len(train_patients_for_fold(manifest_map, fold))
    if n_images is None:
        # ...and the image count from that run's own eval_train.json, which is
        # exactly the set it trained on.
        train_report = path.replace('eval_val.json', 'eval_train.json')
        if os.path.exists(train_report):
            with open(train_report, 'r', encoding='utf-8') as f:
                n_images = json.load(f).get('n_images')

    return {
        'row_type': 'fold',
        'arm': arm,
        'fraction': format_fraction(fraction),
        'fold': fold,
        'n_patients': n_patients,
        'n_images': n_images,
        'mre_mm': overall.get('mre_mm'),
        'sdr2': sdr_at(overall, 2.0),
        'sdr4': sdr_at(overall, 4.0),
        'n_val_images': report.get('n_images'),
        'total_steps': report.get('total_steps'),
        'source': os.path.relpath(path, REPO_ROOT).replace('\\', '/'),
    }


def pooled_row(arm, fraction, rows):
    """mean +/- SD across folds - the only form any poster/paper number may take."""
    mres = [r['mre_mm'] for r in rows if r['mre_mm'] is not None]
    patients = [r['n_patients'] for r in rows if r['n_patients'] is not None]
    images = [r['n_images'] for r in rows if r['n_images'] is not None]
    return {
        'row_type': 'pooled',
        'arm': arm,
        'fraction': format_fraction(fraction),
        'fold': f"mean+/-sd (n={len(mres)})",
        'n_patients': float(np.mean(patients)) if patients else None,
        'n_images': float(np.mean(images)) if images else None,
        'mre_mm': float(np.mean(mres)) if mres else None,
        'mre_mm_std': float(np.std(mres, ddof=0)) if len(mres) > 1 else (0.0 if mres else None),
        'sdr2': float(np.mean([r['sdr2'] for r in rows if r['sdr2'] is not None])) if rows else None,
        'sdr4': float(np.mean([r['sdr4'] for r in rows if r['sdr4'] is not None])) if rows else None,
        'n_folds': len(mres),
    }


def paired_deltas(rows_by_key, fractions, folds, arm_a, arm_b):
    """Per-fold MRE(arm_b) - MRE(arm_a), only where BOTH arms ran that fold.
    Paired because fold k holds out the same patients for every arm."""
    deltas = {}
    for fraction in fractions:
        per_fold = []
        for fold in folds:
            a = rows_by_key.get((arm_a, format_fraction(fraction), fold))
            b = rows_by_key.get((arm_b, format_fraction(fraction), fold))
            if a and b and a['mre_mm'] is not None and b['mre_mm'] is not None:
                per_fold.append((fold, b['mre_mm'] - a['mre_mm']))
        if per_fold:
            deltas[format_fraction(fraction)] = per_fold
    return deltas


def make_figure(rows, fractions, arms, deltas, output_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    pooled = {(r['arm'], r['fraction']): r for r in rows if r['row_type'] == 'pooled'}
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={'width_ratios': [1.35, 1]})
    fig.patch.set_facecolor('#fcfcfb')

    # ---- Panel A: the curve itself -------------------------------------------
    ends = []
    for i, arm in enumerate(arms):
        colour = ARM_COLOURS.get(arm, FALLBACK_COLOURS[i % len(FALLBACK_COLOURS)])
        xs, ys, errs = [], [], []
        for fraction in fractions:
            row = pooled.get((arm, format_fraction(fraction)))
            if row and row['mre_mm'] is not None and row['n_patients']:
                xs.append(row['n_patients'])
                ys.append(row['mre_mm'])
                errs.append(row.get('mre_mm_std') or 0.0)
        if not xs:
            continue
        order = np.argsort(xs)
        xs, ys, errs = np.array(xs)[order], np.array(ys)[order], np.array(errs)[order]
        ax.errorbar(xs, ys, yerr=errs, color=colour, linewidth=2, marker='o', markersize=7,
                    markeredgecolor='#fcfcfb', markeredgewidth=1.5, capsize=4, elinewidth=1.5,
                    label=ARM_LABELS.get(arm, arm), zorder=3 - i)
        ends.append((ys[-1], xs[-1], arm))

    # Direct labels at the largest-cohort end, so identity survives greyscale and
    # a legend-free crop. The two arms can land within 0.01 mm of each other, so
    # push the upper one up and the lower one down instead of overlapping them.
    ends.sort()
    for j, (y, x, arm) in enumerate(ends):
        offset = -9 if (len(ends) > 1 and j == 0) else (9 if len(ends) > 1 else 0)
        ax.annotate(ARM_SHORT.get(arm, arm), xy=(x, y), xytext=(8, offset),
                    textcoords='offset points', va='center', ha='left',
                    fontsize=9, color=INK_MUTED)

    ax.set_xscale('log', base=2)
    all_x = sorted({r['n_patients'] for r in rows
                    if r['row_type'] == 'pooled' and r['n_patients']})
    if all_x:
        ax.set_xticks(all_x)
        ax.set_xticklabels([f"{x:.0f}" for x in all_x])
        ax.set_xlim(min(all_x) * 0.78, max(all_x) * 1.45)
    ax.minorticks_off()
    ax.set_xlabel('Training patients per fold (log$_2$ scale)', fontsize=9.5, color=INK_MUTED)
    ax.set_ylabel('MRE (mm)', fontsize=9.5, color=INK_MUTED)
    ax.set_title('A. Accuracy vs annotated cohort size', fontsize=11, color=INK, loc='left', pad=10)
    ax.legend(frameon=False, fontsize=9, loc='upper right', handlelength=1.6, labelspacing=0.4)

    # ---- Panel B: the paired difference --------------------------------------
    labels, means = [], []
    for i, (fraction_tag, per_fold) in enumerate(sorted(deltas.items(), key=lambda kv: float(kv[0]))):
        values = [d for _, d in per_fold]
        jitter = np.linspace(-0.13, 0.13, len(values))
        ax2.scatter([i + j for j in jitter], values, s=34, color=INK_MUTED, alpha=0.55,
                    edgecolor='#fcfcfb', linewidth=1, zorder=3)
        mean = float(np.mean(values))
        ax2.plot([i - 0.26, i + 0.26], [mean, mean], color=INK, linewidth=2.5, zorder=4)
        means.append(mean)
        pooled_row_a = pooled.get((arms[0], fraction_tag))
        n_patients = pooled_row_a['n_patients'] if pooled_row_a and pooled_row_a['n_patients'] else None
        labels.append(f"{fraction_tag}\n({n_patients:.0f} pts)" if n_patients else fraction_tag)

    ax2.axhline(0, color=GRID, linewidth=1.5, zorder=1)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_xlim(-0.6, len(labels) - 0.4)
    ax2.set_xlabel('Fraction of training patients', fontsize=9.5, color=INK_MUTED)
    ax2.set_ylabel('$\\Delta$ MRE (mm),  HRNet $-$ U-Net', fontsize=9.5, color=INK_MUTED)
    ax2.set_title('B. Paired per-fold difference', fontsize=11, color=INK, loc='left', pad=10)
    if means:
        span = max(0.25, max(abs(v) for per_fold in deltas.values() for _, v in per_fold) * 1.35)
        ax2.set_ylim(-span, span)
        ax2.annotate('HRNet worse', xy=(0.985, 0.965), xycoords='axes fraction', ha='right',
                     va='top', fontsize=8.5, color=INK_MUTED)
        ax2.annotate('HRNet better', xy=(0.985, 0.035), xycoords='axes fraction', ha='right',
                     va='bottom', fontsize=8.5, color=INK_MUTED)

    for axis in (ax, ax2):
        axis.set_facecolor('#fcfcfb')
        axis.grid(axis='y', color=GRID, linewidth=0.8, alpha=0.9)
        axis.set_axisbelow(True)
        for side in ('top', 'right'):
            axis.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            axis.spines[side].set_color(GRID)
        axis.tick_params(colors=INK_MUTED, labelsize=9, length=0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, facecolor=fig.get_facecolor())
    print(f"Wrote {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fractions', type=float, nargs='+', default=[1.0, 0.5, 0.25],
                        help="fractions to include; 1.0 (Run 6) is always added if missing")
    parser.add_argument('--arms', nargs='+', default=['unet_baseline', 'hrnet_pretrained'],
                        help="arm names = the Run 6 checkpoint-dir basenames")
    parser.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--group', default='label_efficiency',
                        help="training.output_group the runs used (outputs/<group>/...)")
    parser.add_argument('--manifest', default=DEFAULT_MANIFEST)
    parser.add_argument('--exclude-landmarks', nargs='*', default=[], metavar='NAME',
                        help="landmark channels to EXCLUDE when computing each cell's MRE/SDR, so the "
                             "curve is reported over the same subset as the headline metric "
                             "(e.g. --exclude-landmarks iGo pGo for the 15-point clinical set, whose "
                             "Gonion is a derived point rather than a trained channel)")
    parser.add_argument('--output-dir', default=None,
                        help="default: outputs/<group>/")
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--baseline-root', default=OUTPUTS_DIR,
                        help='Directory containing the full-data arm/fold reports')
    parser.add_argument('--allow-incomplete', action='store_true',
                        help='Allow partial exploratory summaries; never use for publication')
    args = parser.parse_args()

    fractions = sorted(set(args.fractions) | {1.0}, reverse=True)
    output_dir = args.output_dir or os.path.join(OUTPUTS_DIR, args.group)
    os.makedirs(output_dir, exist_ok=True)

    manifest_map = None
    if os.path.exists(args.manifest):
        manifest_map, _ = read_fold_manifest(args.manifest)
    else:
        print(f"[WARN] {args.manifest} not found - patient counts fall back to whatever the "
              "reports themselves recorded")

    rows, missing = [], []
    for arm in args.arms:
        for fraction in fractions:
            fold_rows = []
            for fold in args.folds:
                row = load_cell(arm, fraction, fold, args.group, manifest_map,
                                args.exclude_landmarks, args.baseline_root)
                if row is None:
                    missing.append((arm, format_fraction(fraction), fold))
                    continue
                fold_rows.append(row)
            rows.extend(fold_rows)
            if fold_rows:
                rows.append(pooled_row(arm, fraction, fold_rows))

    if not rows:
        raise SystemExit('No eval_val.json reports found')
    if missing:
        if not args.allow_incomplete:
            raise SystemExit(f'Missing reports: {missing}; refusing an incomplete summary')
        print(f"[WARN] {len(missing)} (arm, fraction, fold) cell(s) have no eval_val.json yet: "
              f"{missing[:8]}{' ...' if len(missing) > 8 else ''}")

    for row in rows:
        row['excluded_landmarks'] = ';'.join(args.exclude_landmarks)
    fields = ['row_type', 'arm', 'excluded_landmarks', 'fraction', 'fold', 'n_patients', 'n_images',
              'mre_mm', 'mre_mm_std', 'sdr2', 'sdr4', 'n_folds', 'n_val_images',
              'total_steps', 'source']
    summary_path = os.path.join(output_dir, 'summary.csv')
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {summary_path} ({sum(r['row_type'] == 'fold' for r in rows)} fold rows, "
          f"{sum(r['row_type'] == 'pooled' for r in rows)} pooled rows)")

    print(f"\n{'arm':<20}{'frac':<8}{'patients':<10}{'images':<9}{'MRE (mm)':<18}{'SDR@2':<9}{'SDR@4':<9}")
    for row in rows:
        if row['row_type'] != 'pooled':
            continue
        mre = 'n/a' if row['mre_mm'] is None else f"{row['mre_mm']:.3f} +/- {row['mre_mm_std']:.3f}"
        patients = '?' if not row['n_patients'] else format(row['n_patients'], '.0f')
        images = '?' if not row['n_images'] else format(row['n_images'], '.0f')
        sdr2 = 'n/a' if row['sdr2'] is None else format(row['sdr2'] * 100, '.1f') + '%'
        sdr4 = 'n/a' if row['sdr4'] is None else format(row['sdr4'] * 100, '.1f') + '%'
        print(f"{row['arm']:<20}{row['fraction']:<8}{patients:<10}{images:<9}"
              f"{mre:<18}{sdr2:<9}{sdr4:<9}")

    rows_by_key = {(r['arm'], r['fraction'], r['fold']): r for r in rows if r['row_type'] == 'fold'}
    deltas = {}
    if len(args.arms) >= 2:
        deltas = paired_deltas(rows_by_key, fractions, args.folds, args.arms[0], args.arms[1])
        print(f"\nPaired per-fold delta ({args.arms[1]} - {args.arms[0]}), mm - "
              "positive means the pretrained arm is WORSE:")
        for fraction_tag, per_fold in sorted(deltas.items(), key=lambda kv: -float(kv[0])):
            values = [d for _, d in per_fold]
            print(f"  frac {fraction_tag:<6} mean {np.mean(values):+.3f}  sd {np.std(values):.3f}  "
                  f"per-fold {[f'{v:+.2f}' for v in values]}")

    if not args.no_plot:
        make_figure(rows, fractions, args.arms, deltas,
                    os.path.join(output_dir, 'label_efficiency_curve.png'))


if __name__ == "__main__":
    main()
