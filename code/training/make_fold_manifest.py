"""
================================================================================
MAKE FOLD MANIFEST - persist the patient->fold mapping once, then assert on it
================================================================================
DESCRIPTION:
    Builds the dataset exactly as training does (same images/fcd/review CSV and
    the same exclusion list), computes the canonical patient-level k-fold
    assignment from (k, seed), and writes it to a committed CSV that every
    later run verifies itself against - see code/datasets/fold_manifest.py for
    why regeneration-on-every-run is a silent-comparability hazard.

    Run this ONCE against the Run 6 data configuration, commit the result, then
    point every config's `kfold.manifest_path` at it.

    Re-running is safe and non-destructive: with the file already present it
    VERIFIES instead of overwriting, and exits non-zero if the mapping has
    drifted. --force is the deliberate "yes, re-deal every fold and invalidate
    the comparisons" escape hatch.

USAGE:
    python code/training/make_fold_manifest.py --config configs/unet_baseline.yaml
    python code/training/make_fold_manifest.py --config configs/unet_baseline.yaml --check
================================================================================
"""

import argparse
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'datasets')))

from config import load_config, apply_overrides
from fold_manifest import (write_fold_manifest, read_fold_manifest, verify_fold_map,
                           val_patients_for_fold)
from kfold import assign_patient_folds
from provenance import git_commit_info
from train import build_base_dataset

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
DEFAULT_MANIFEST = os.path.join(REPO_ROOT, 'configs', 'folds', 'oslo26_k5_seed0.csv')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True,
                        help="config whose data section + seed/kfold.k define the split "
                             "(configs/unet_baseline.yaml - all four Run 6 arms share these)")
    parser.add_argument('--output', default=DEFAULT_MANIFEST,
                        help=f"manifest path (default: {os.path.relpath(DEFAULT_MANIFEST, REPO_ROOT)})")
    parser.add_argument('--check', action='store_true',
                        help="verify an existing manifest and exit non-zero on drift; never writes")
    parser.add_argument('--force', action='store_true',
                        help="overwrite an existing, DIFFERENT manifest - re-deals every fold and "
                             "invalidates comparisons with runs made under the old one")
    parser.add_argument('--set', action='append', default=[], dest='overrides',
                        help="dot-path override, e.g. --set data.images_dir=/some/path (repeatable)")
    args = parser.parse_args()

    config = apply_overrides(load_config(args.config), args.overrides)
    config_dir = os.path.dirname(os.path.abspath(args.config))
    for rel_key in ('landmark_config_path', 'exclude_images_path'):
        rel = config['data'].get(rel_key)
        if rel and not os.path.isabs(rel):
            config['data'][rel_key] = os.path.join(config_dir, rel)

    k, seed = config['kfold']['k'], config['seed']
    dataset = build_base_dataset(config)
    if len(dataset) == 0:
        raise RuntimeError("No usable images found - check the data paths in the config")

    serials = {entry['serial'] for entry in dataset.samples}
    fold_map = assign_patient_folds(serials, k=k, seed=seed)
    print(f"{len(dataset)} images / {len(serials)} patients, k={k}, seed={seed} "
          f"({len(dataset.skipped)} images skipped by the dataset's own filters)")
    for fold in range(k):
        val_patients = val_patients_for_fold(fold_map, fold)
        val_images = sum(1 for e in dataset.samples if e['serial'] in set(val_patients))
        print(f"  fold {fold}: {len(serials) - len(val_patients):>4} train / {len(val_patients):>3} val patients"
              f"   {len(dataset) - val_images:>4} train / {val_images:>3} val images")

    if os.path.exists(args.output) and not args.force:
        existing, meta = read_fold_manifest(args.output)
        verify_fold_map(serials, existing, k=k, seed=seed, manifest_path=args.output)
        print(f"\nOK: existing manifest {args.output} still matches this dataset "
              f"({len(existing)} patients){' [generated ' + meta['generated'] + ']' if 'generated' in meta else ''}.")
        return

    if args.check:
        print(f"\n[ERROR] --check requested but {args.output} does not exist yet.")
        sys.exit(1)

    git = git_commit_info(REPO_ROOT)
    write_fold_manifest(args.output, fold_map, meta={
        'generated': date.today().isoformat(),
        'config': os.path.relpath(os.path.abspath(args.config), REPO_ROOT).replace('\\', '/'),
        'k': k,
        'seed': seed,
        'n_patients': len(serials),
        'n_images': len(dataset),
        'images_dir': config['data']['images_dir'],
        'exclude_images_path': config['data'].get('exclude_images_path'),
        'review_csv_path': config['data'].get('review_csv_path'),
        'git_commit': git.get('commit'),
        'git_dirty': git.get('dirty'),
    })
    print(f"\nWrote {args.output}. Commit it, then set kfold.manifest_path in every config that "
          "must share this split.")


if __name__ == "__main__":
    main()
