"""
================================================================================
MAKE LABEL-EFFICIENCY SUBSETS - the sampled patient lists for Run 8
================================================================================
DESCRIPTION:
    Writes one CSV per (fold, fraction) listing the TRAINING patients that
    fraction of the curve is allowed to use. Reads only the fold manifest, so
    it needs no images and can be run, reviewed and committed off-cluster.

    Guarantees checked here rather than trusted (see
    code/datasets/label_efficiency.py for the reasoning):
      - the validation patients of each fold appear in NO subset for that fold,
      - subsets are nested across fractions within a fold,
      - re-running reproduces the same files exactly, or refuses to overwrite.

USAGE:
    python code/training/make_label_efficiency_subsets.py                    # 0.5, 0.25
    python code/training/make_label_efficiency_subsets.py --fractions 0.5 0.25 0.125
================================================================================
"""

import argparse
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'datasets')))

from fold_manifest import read_fold_manifest, train_patients_for_fold, val_patients_for_fold
from label_efficiency import (nested_patient_subset, write_patient_subset, load_patient_subset,
                              subset_filename, format_fraction)

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
DEFAULT_MANIFEST = os.path.join(REPO_ROOT, 'configs', 'folds', 'oslo26_k5_seed0.csv')
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, 'configs', 'label_efficiency')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default=DEFAULT_MANIFEST,
                        help=f"fold manifest (default: {os.path.relpath(DEFAULT_MANIFEST, REPO_ROOT)})")
    parser.add_argument('--fractions', type=float, nargs='+', default=[0.5, 0.25],
                        help="fractions of each fold's TRAINING patients (default: 0.5 0.25; "
                             "add 0.125 for the optional smallest point). 1.0 is Run 6 itself - "
                             "never sampled, never re-run.")
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--seed', type=int, default=0,
                        help="sampling seed; the per-fold shuffle uses (seed, fold) and is "
                             "independent of the fraction, which is what makes subsets nested")
    parser.add_argument('--force', action='store_true',
                        help="overwrite existing subset files that differ (invalidates any run "
                             "already trained on them)")
    args = parser.parse_args()

    fold_map, meta = read_fold_manifest(args.manifest)
    k = int(meta.get('k', max(fold_map.values()) + 1))
    fractions = sorted((f for f in args.fractions if f < 1.0), reverse=True)
    if not fractions:
        print("Nothing to do: fraction 1.0 is Run 6 and is reused, not regenerated.")
        return

    print(f"Manifest: {args.manifest} ({len(fold_map)} patients, k={k})")
    print(f"Fractions: {[format_fraction(f) for f in fractions]}  seed={args.seed}\n")
    print(f"{'fold':<6}{'fraction':<10}{'n_patients':<12}{'of pool':<10}{'status':<10}file")

    statuses = []
    for fold in range(k):
        pool = train_patients_for_fold(fold_map, fold)
        held_out = set(val_patients_for_fold(fold_map, fold))
        previous = None
        for fraction in fractions:
            serials = nested_patient_subset(pool, fraction, fold=fold, seed=args.seed)

            leaked = sorted(held_out.intersection(serials))
            if leaked:
                raise RuntimeError(f"fold {fold} frac {fraction}: {len(leaked)} validation patient(s) "
                                   f"in the training subset, e.g. {leaked[:5]}")
            if previous is not None and not set(serials).issubset(previous):
                raise RuntimeError(f"fold {fold} frac {fraction}: subset is not nested inside the "
                                   "next-larger fraction - the sampler is not prefix-stable")
            previous = set(serials)

            path = os.path.join(args.output_dir, subset_filename(fold, fraction))
            status = write_patient_subset(path, serials, force=args.force, meta={
                'generated': date.today().isoformat(),
                'manifest': os.path.relpath(os.path.abspath(args.manifest), REPO_ROOT).replace('\\', '/'),
                'fold': fold,
                'fraction': format_fraction(fraction),
                'seed': args.seed,
                'n_patients': len(serials),
                'pool_patients': len(pool),
                'nested': 'prefix of one seeded shuffle per fold - subsets nest across fractions',
            })
            statuses.append(status)
            print(f"{fold:<6}{format_fraction(fraction):<10}{len(serials):<12}{len(pool):<10}{status:<10}"
                  f"{os.path.relpath(path, REPO_ROOT)}")

    # Re-read from disk and re-check nesting, so what was WRITTEN is verified,
    # not just what was computed in memory.
    for fold in range(k):
        loaded = {f: set(load_patient_subset(os.path.join(args.output_dir, subset_filename(fold, f))))
                  for f in fractions}
        for smaller, larger in zip(fractions[1:], fractions[:-1]):
            if not loaded[smaller].issubset(loaded[larger]):
                raise RuntimeError(f"fold {fold}: {smaller} is not a subset of {larger} on disk")
    print(f"\nVerified on disk: validation patients excluded, subsets nested across fractions. "
          f"({statuses.count('created')} written, {statuses.count('unchanged')} already correct)")


if __name__ == "__main__":
    main()
