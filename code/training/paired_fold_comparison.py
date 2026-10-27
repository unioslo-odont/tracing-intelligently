"""
================================================================================
PAIRED FOLD COMPARISON - is a difference between two runs real?
================================================================================
DESCRIPTION:
    compare_backbones.py answers "what did each arm score" (mean +/- SD across
    folds). It does NOT answer "is arm A better than arm B", because those two
    numbers are not independent samples: fold k holds out exactly the same
    patients in every arm, so the arms are PAIRED, and the right statistic is the
    per-fold difference, not the difference of the means. With n=5 folds, a
    0.1 mm gap between two means whose per-fold differences scatter by 0.3 mm is
    inconclusive; the unpaired table cannot show that and has repeatedly made ties look
    like rankings in this project.

    This reports, for each run against a reference run:
      - per-fold MRE for both,
      - the per-fold difference, its mean and SD,
      - a 95% confidence interval for the mean difference (paired t, n-1 df),
      - an exploratory summary, without treating an inconclusive result as equivalence.

    Use it for Run 7 vs Run 6 (did fixing the optimizer confound change the
    conclusion?) and for any arm vs unet_baseline within a run.

USAGE:
    # Run 7 vs Run 6, per arm:
    python code/training/paired_fold_comparison.py \
        --reference hrnet_pretrained --runs run7/hrnet_pretrained
    # every arm against the from-scratch baseline, within Run 7:
    python code/training/paired_fold_comparison.py --reference unet_baseline \
        --runs run7/resnet_unet run7/hrnet_pretrained run9/dinov2_finetune
================================================================================
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_backbones import load_fold_reports
from output_paths import OUTPUTS_DIR, outputs_path
from report_selection import select_report

# Two-sided t critical values at 95%, indexed by degrees of freedom (n_folds - 1).
# Hardcoded so this has no scipy dependency; k=5 folds -> df=4 -> 2.776.
T_CRIT_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
             8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179}


def resolve_run_dir(name):
    """Accept 'unet_baseline', 'run7/hrnet_pretrained', or an explicit path."""
    for candidate in (name, os.path.join(OUTPUTS_DIR, name)):
        if os.path.isdir(candidate):
            return candidate
    raise SystemExit(f"[ERROR] No results directory for '{name}' "
                     f"(looked for {name} and {os.path.join(OUTPUTS_DIR, name)})")


def fold_mres(name, exclude=(), expected_folds=5):
    """{fold: mre_mm} for one run."""
    reports = load_fold_reports(resolve_run_dir(name))
    if not reports:
        raise SystemExit(f"[ERROR] No fold*/eval_val.json under {resolve_run_dir(name)}")
    if {fold for fold, _ in reports} != set(range(expected_folds)):
        raise ValueError(f"{name}: expected folds 0..{expected_folds - 1}")
    selected = {fold: select_report(report, exclude)['overall']['mre_mm']
                for fold, report in reports}
    if any(value is None or not np.isfinite(value) for value in selected.values()):
        raise ValueError(f"{name}: missing or non-finite fold MRE")
    return selected


def paired_stats(reference, candidate):
    """Per-fold differences (candidate - reference) and their paired summary.
    The CLI requires complete folds. This helper intersects keys; an unpaired fold is
    incomparable, not a free data point."""
    folds = sorted(set(reference) & set(candidate))
    deltas = np.array([candidate[f] - reference[f] for f in folds])
    n = len(deltas)
    mean = float(deltas.mean()) if n else float('nan')
    # ddof=1: we are estimating the SD of the difference from the sample.
    sd = float(deltas.std(ddof=1)) if n > 1 else float('nan')
    half_width = (T_CRIT_95.get(n - 1, 2.0) * sd / np.sqrt(n)) if n > 1 else float('nan')
    return folds, deltas, mean, sd, half_width


def verdict(mean, half_width, tolerance=0.0):
    if not np.isfinite(half_width):
        return "n/a (needs >= 2 paired folds)"
    if abs(mean) <= half_width:
        return "No clear difference detected (exploratory CI includes 0; not equivalence)"
    return ("Lower MRE (exploratory)" if mean < -tolerance else "Higher MRE (exploratory)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', required=True,
                        help="the run to compare against, e.g. unet_baseline or hrnet_pretrained")
    parser.add_argument('--runs', nargs='+', required=True,
                        help="one or more runs, e.g. run7/hrnet_pretrained")
    parser.add_argument('--output', default=None,
                        help="CSV path (default outputs/paired_comparison.csv)")
    parser.add_argument('--exclude-landmarks', nargs='+', default=[])
    parser.add_argument('--expected-folds', type=int, default=5)
    args = parser.parse_args()
    print('Exploratory paired-fold intervals: overlapping training folds are dependent; '
          'these intervals do not establish equivalence or confirmatory significance.')

    reference = fold_mres(args.reference, args.exclude_landmarks, args.expected_folds)
    print(f"Reference: {args.reference}   per-fold MRE "
          f"{[round(reference[f], 3) for f in sorted(reference)]}   "
          f"mean {np.mean(list(reference.values())):.3f} "
          f"+/- {np.std(list(reference.values())):.3f}\n")

    rows = []
    for name in args.runs:
        candidate = fold_mres(name, args.exclude_landmarks, args.expected_folds)
        folds, deltas, mean, sd, half_width = paired_stats(reference, candidate)
        print(f"{name}")
        print(f"  per-fold MRE   {[round(candidate[f], 3) for f in sorted(candidate)]}   "
              f"mean {np.mean(list(candidate.values())):.3f} "
              f"+/- {np.std(list(candidate.values())):.3f}")
        print(f"  paired delta   {[round(float(d), 3) for d in deltas]}   (folds {folds})")
        print(f"  mean delta     {mean:+.3f} mm  (SD {sd:.3f}, 95% CI "
              f"{mean - half_width:+.3f} .. {mean + half_width:+.3f})")
        print(f"  VERDICT        {verdict(mean, half_width)}\n")
        rows.append({
            'reference': args.reference, 'run': name, 'n_paired_folds': len(folds),
            'excluded_landmarks': ';'.join(args.exclude_landmarks),
            'interval_method': 'exploratory paired t; fold dependence not corrected',
            'reference_mre_mean': float(np.mean([reference[f] for f in folds])),
            'run_mre_mean': float(np.mean([candidate[f] for f in folds])),
            'mean_delta_mm': mean, 'sd_delta_mm': sd,
            'ci95_low': mean - half_width, 'ci95_high': mean + half_width,
            'verdict': verdict(mean, half_width),
            'per_fold_delta_mm': ';'.join(f"{d:.4f}" for d in deltas),
        })

    output = args.output or outputs_path('paired_comparison.csv')
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
