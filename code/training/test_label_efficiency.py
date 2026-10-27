"""
================================================================================
TESTS - fold manifest, label-efficiency subsets, curve aggregation
================================================================================
DESCRIPTION:
    Everything the Run 8 curve's VALIDITY rests on, checked against synthetic
    fixtures so it runs anywhere - no GPU, no torch, no patient data. In
    particular the acceptance criterion "fold membership identical to Run 6 and
    validation sets untouched at every fraction" is a test here, not a hope.

    What is covered:
      1. the split refactor is bit-identical to the pre-Run-7 implementation,
      2. manifest round-trip; drift in the patient set or in (k, seed) RAISES,
      3. subsets: nested across fractions, never contain a validation patient,
         idempotent on rewrite, and refuse to change under the same filename,
      4. restricting training indices to a subset leaves validation untouched,
      5. the aggregator reads a Run-6-shaped tree + label-efficiency tree,
         computes pooled mean/SD and paired per-fold deltas, and draws the figure.

    NOT covered here (needs torch, so it runs on the cluster):
      code/training/test_optim_groups.py - parameter groups, warmup, layer-wise
      decay, and the "old configs are unchanged" guarantee.

USAGE:
    python code/training/test_label_efficiency.py
================================================================================
"""

import json
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'datasets')))

from kfold import assign_patient_folds, make_patient_kfold_splits, splits_from_fold_map
from fold_manifest import (write_fold_manifest, read_fold_manifest, verify_fold_map,
                           train_patients_for_fold, val_patients_for_fold)
from label_efficiency import (nested_patient_subset, write_patient_subset, load_patient_subset,
                              restrict_indices_to_patients, subset_filename, arm_run_name)

PASSED, FAILED = [], []


def check(name, condition, detail=''):
    (PASSED if condition else FAILED).append(name)
    print(f"  {'PASS' if condition else 'FAIL'}  {name}{(' - ' + detail) if detail and not condition else ''}")


def raises(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
        return False
    except Exception:
        return True


class FakeDataset:
    """Minimal stand-in for CephalometricDataset: only .samples is used by the
    split/subset code, and only its 'serial' key."""

    def __init__(self, serial_to_n_images):
        self.samples = [{'serial': serial, 'imagefile': f"{serial}_{i}.tif"}
                        for serial, n in sorted(serial_to_n_images.items())
                        for i in range(1, n + 1)]

    def __len__(self):
        return len(self.samples)


def cohort(n_patients=60, seed=7):
    """A cohort with the real one's awkward property: patients contribute
    different numbers of images, so image counts never scale exactly with
    patient counts."""
    rng = np.random.default_rng(seed)
    return {f"g{i:04d}": int(rng.integers(1, 4)) for i in range(1, n_patients + 1)}


# ---------------------------------------------------------------- 1. the split
def test_split_is_unchanged():
    print("\n[1] patient-level split")
    dataset = FakeDataset(cohort())

    def previous_implementation(dataset, k, seed=0):
        """The pre-Run-7 make_patient_kfold_splits, verbatim."""
        serial_to_indices = {}
        for idx, entry in enumerate(dataset.samples):
            serial_to_indices.setdefault(entry['serial'], []).append(idx)
        serials = sorted(serial_to_indices.keys())
        rng = np.random.default_rng(seed)
        rng.shuffle(serials)
        fold_of_serial = {serial: i % k for i, serial in enumerate(serials)}
        splits = []
        for fold in range(k):
            val_indices, train_indices = [], []
            for serial, indices in serial_to_indices.items():
                (val_indices if fold_of_serial[serial] == fold else train_indices).extend(indices)
            splits.append((np.array(sorted(train_indices)), np.array(sorted(val_indices))))
        return splits

    now, before = make_patient_kfold_splits(dataset, 5, 0), previous_implementation(dataset, 5, 0)
    check("refactored split is bit-identical to the pre-Run-7 implementation",
          all((a == c).all() and (b == d).all() for (a, b), (c, d) in zip(now, before)))

    fold_map = assign_patient_folds({e['serial'] for e in dataset.samples}, 5, 0)
    from_map = splits_from_fold_map(dataset, fold_map, 5)
    check("splitting on a fold map reproduces the same splits",
          all((a == c).all() and (b == d).all() for (a, b), (c, d) in zip(now, from_map)))

    all_val = [set(v.tolist()) for _, v in now]
    check("validation folds are disjoint and cover the dataset",
          set().union(*all_val) == set(range(len(dataset)))
          and sum(len(v) for v in all_val) == len(dataset))
    patients_per_fold = [{dataset.samples[i]['serial'] for i in v} for _, v in now]
    check("no patient appears in two folds",
          all(not (patients_per_fold[i] & patients_per_fold[j])
              for i in range(5) for j in range(i + 1, 5)))


# ------------------------------------------------------------- 2. the manifest
def test_manifest(tmp):
    print("\n[2] fold manifest")
    serials = sorted(cohort().keys())
    fold_map = assign_patient_folds(serials, 5, 0)
    path = os.path.join(tmp, 'manifest.csv')
    write_fold_manifest(path, fold_map, {'generated': '2026-08-31', 'k': 5, 'seed': 0})

    loaded, meta = read_fold_manifest(path)
    check("manifest round-trips exactly", loaded == fold_map)
    check("manifest keeps its metadata", meta.get('k') == '5' and meta.get('seed') == '0')
    check("verify passes for the unchanged cohort",
          verify_fold_map(serials, loaded, 5, 0, path) == loaded)
    check("an ADDED patient raises", raises(verify_fold_map, serials + ['g9999'], loaded, 5, 0, path))
    check("a REMOVED patient raises", raises(verify_fold_map, serials[:-1], loaded, 5, 0, path))
    check("a changed seed raises", raises(verify_fold_map, serials, loaded, 5, 1, path))
    check("a changed k raises", raises(verify_fold_map, serials, loaded, 4, 0, path))

    tampered = dict(loaded)
    tampered[serials[0]] = (tampered[serials[0]] + 1) % 5
    check("a hand-edited fold assignment raises",
          raises(verify_fold_map, serials, tampered, 5, 0, path))


# -------------------------------------------------------------- 3. the subsets
def test_subsets(tmp):
    print("\n[3] label-efficiency subsets")
    dataset = FakeDataset(cohort())
    serials = sorted({e['serial'] for e in dataset.samples})
    fold_map = assign_patient_folds(serials, 5, 0)
    fractions = [0.5, 0.25, 0.125]

    nested_ok, disjoint_ok, size_ok = True, True, True
    for fold in range(5):
        pool = train_patients_for_fold(fold_map, fold)
        held_out = set(val_patients_for_fold(fold_map, fold))
        subsets = {f: nested_patient_subset(pool, f, fold=fold, seed=0) for f in fractions}
        for smaller, larger in zip(fractions[1:], fractions[:-1]):
            nested_ok &= set(subsets[smaller]).issubset(set(subsets[larger]))
        for f in fractions:
            disjoint_ok &= not (held_out & set(subsets[f]))
            size_ok &= len(subsets[f]) == max(1, round(f * len(pool)))
    check("subsets are nested across fractions, in every fold", nested_ok)
    check("no validation patient ever enters a training subset", disjoint_ok)
    check("subset size is round(fraction x pool)", size_ok)

    pool0 = train_patients_for_fold(fold_map, 0)
    check("sampling is deterministic for a given (seed, fold)",
          nested_patient_subset(pool0, 0.25, 0, 0) == nested_patient_subset(pool0, 0.25, 0, 0))
    check("a different fold gets a different draw",
          nested_patient_subset(pool0, 0.25, 0, 0)
          != nested_patient_subset(train_patients_for_fold(fold_map, 1), 0.25, 1, 0))
    check("the whole pool comes back at fraction 1.0",
          nested_patient_subset(pool0, 1.0, 0, 0) == sorted(pool0))

    path = os.path.join(tmp, subset_filename(0, 0.25))
    first = nested_patient_subset(pool0, 0.25, 0, 0)
    check("first write creates", write_patient_subset(path, first, meta={'fold': 0}) == 'created')
    check("identical rewrite is a no-op", write_patient_subset(path, first) == 'unchanged')
    check("a DIFFERENT list under the same name raises",
          raises(write_patient_subset, path, first[:-1]))
    check("force overwrites deliberately",
          write_patient_subset(path, first[:-1], force=True) == 'created')
    check("subset CSV round-trips", load_patient_subset(path) == sorted(first[:-1]))

    # The invariant that makes the curve valid: subsetting touches training only.
    train_idx, val_idx = splits_from_fold_map(dataset, fold_map, 5)[0]
    val_before = set(val_idx.tolist())
    restricted = restrict_indices_to_patients(dataset, train_idx, first)
    check("restricting training indices leaves the validation split untouched",
          set(val_idx.tolist()) == val_before)
    check("restricted training never overlaps validation",
          not (set(restricted.tolist()) & val_before))
    check("restricted training is a subset of the fold's training pool",
          set(restricted.tolist()).issubset(set(train_idx.tolist())))
    check("restricted training contains exactly the selected patients",
          {dataset.samples[i]['serial'] for i in restricted} == set(first))
    check("images per patient are kept whole (n_images != fraction x pool images)",
          len(restricted) == sum(1 for e in dataset.samples if e['serial'] in set(first)))
    check("a subset naming an unknown patient raises",
          raises(restrict_indices_to_patients, dataset, train_idx, first + ['g9999']))


# ------------------------------------------------------------ 4. the aggregator
def write_report(path, mre, n_val_images=76, **extra):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    report = {
        'overall': {'mre_mm': mre, 'sdr': {'2.0': 0.62, '2.5': 0.70, '3.0': 0.77, '4.0': 0.87}},
        'per_landmark': {}, 'split': 'val', 'n_images': n_val_images, **extra,
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f)


def test_aggregator(tmp):
    print("\n[4] curve aggregation")
    import aggregate_label_efficiency as agg

    outputs = os.path.join(tmp, 'outputs')
    agg.OUTPUTS_DIR = outputs   # redirect the module's output root at the fixture

    serials = sorted(cohort().keys())
    fold_map = assign_patient_folds(serials, 5, 0)
    manifest = os.path.join(tmp, 'folds.csv')
    write_fold_manifest(manifest, fold_map, {'k': 5, 'seed': 0})

    arms = ['unet_baseline', 'hrnet_pretrained']
    fractions = [1.0, 0.5, 0.25]
    truth = {}
    for arm_index, arm in enumerate(arms):
        for fraction in fractions:
            for fold in range(5):
                # A plausible shape: error grows as the cohort shrinks, the
                # pretrained arm degrades slightly less, plus per-fold noise.
                mre = 2.2 + 0.9 * (1.0 / fraction - 1) * (1.0 - 0.25 * arm_index) + 0.03 * fold
                truth[(arm, fraction, fold)] = mre
                if fraction >= 1.0:
                    base = os.path.join(outputs, arm, f"fold{fold}")
                    write_report(os.path.join(base, 'eval_val.json'), mre)
                    # Run 6 has no training-cohort fields; the counts must come
                    # from the manifest + this file, which is the fallback path.
                    write_report(os.path.join(base, 'eval_train.json'), 1.9, n_val_images=300)
                else:
                    base = os.path.join(outputs, 'label_efficiency',
                                        arm_run_name(arm, fraction), f"fold{fold}")
                    write_report(os.path.join(base, 'eval_val.json'), mre,
                                 n_train_patients=int(round(48 * fraction)),
                                 n_train_images=int(round(300 * fraction)), total_steps=1234)

    sys.argv = ['aggregate_label_efficiency.py', '--fractions', '1.0', '0.5', '0.25',
                '--manifest', manifest, '--output-dir', os.path.join(tmp, 'curve')]
    agg.main()

    import csv as _csv
    with open(os.path.join(tmp, 'curve', 'summary.csv'), encoding='utf-8') as f:
        rows = list(_csv.DictReader(f))
    fold_rows = [r for r in rows if r['row_type'] == 'fold']
    pooled_rows = [r for r in rows if r['row_type'] == 'pooled']
    check("one row per (arm, fraction, fold)", len(fold_rows) == 2 * 3 * 5, str(len(fold_rows)))
    check("one pooled row per (arm, fraction)", len(pooled_rows) == 6, str(len(pooled_rows)))

    unet_25 = [r for r in pooled_rows if r['arm'] == 'unet_baseline' and r['fraction'] == '0.25'][0]
    expected = np.mean([truth[('unet_baseline', 0.25, fold)] for fold in range(5)])
    check("pooled MRE equals the mean over folds",
          abs(float(unet_25['mre_mm']) - expected) < 1e-9,
          f"{unet_25['mre_mm']} vs {expected}")
    check("pooled SD is the across-fold SD", abs(float(unet_25['mre_mm_std']) - 0.03 * np.std(range(5))) < 1e-9)

    run6 = [r for r in pooled_rows if r['arm'] == 'unet_baseline' and r['fraction'] == '1'][0]
    check("fraction 1.0 patient count recovered from the manifest",
          abs(float(run6['n_patients']) - np.mean([len(train_patients_for_fold(fold_map, f))
                                                   for f in range(5)])) < 1e-9)
    check("fraction 1.0 image count recovered from eval_train.json",
          float(run6['n_images']) == 300)

    rows_by_key = {(r['arm'], r['fraction'], int(r['fold'])): {'mre_mm': float(r['mre_mm'])}
                   for r in fold_rows}
    deltas = agg.paired_deltas(rows_by_key, fractions, list(range(5)), arms[0], arms[1])
    check("paired deltas exist for every fraction", set(deltas) == {'1', '0.5', '0.25'})
    check("paired deltas are per-fold (5 points each)",
          all(len(v) == 5 for v in deltas.values()))
    expected_delta = (truth[(arms[1], 0.25, 0)] - truth[(arms[0], 0.25, 0)])
    check("delta is arm_b - arm_a on the same fold",
          abs(deltas['0.25'][0][1] - expected_delta) < 1e-9)

    figure = os.path.join(tmp, 'curve', 'label_efficiency_curve.png')
    check("figure was written", os.path.exists(figure) and os.path.getsize(figure) > 10_000,
          f"{os.path.getsize(figure) if os.path.exists(figure) else 'missing'} bytes")

    # Publication summaries must reject incomplete experiments by default.
    shutil.rmtree(os.path.join(outputs, 'label_efficiency', arm_run_name('hrnet_pretrained', 0.25)))
    sys.argv = ['aggregate_label_efficiency.py', '--fractions', '1.0', '0.5', '0.25',
                '--manifest', manifest, '--output-dir', os.path.join(tmp, 'curve_partial')]
    try:
        agg.main()
    except SystemExit as exc:
        check("incomplete aggregation fails with an actionable error",
              'Missing reports:' in str(exc) and 'refusing an incomplete summary' in str(exc))
    else:
        check("incomplete aggregation fails with an actionable error", False)
    partial_summary = os.path.join(tmp, 'curve_partial', 'summary.csv')
    check("incomplete default run writes no summary", not os.path.exists(partial_summary))

    # Partial summaries remain available only through the explicit exploratory flag.
    sys.argv.extend(['--allow-incomplete', '--no-plot'])
    agg.main()
    with open(partial_summary, encoding='utf-8') as f:
        partial_rows = list(_csv.DictReader(f))
    check("explicit partial aggregation includes only available folds",
          sum(r['row_type'] == 'fold' for r in partial_rows) == 25)
    check("explicit partial aggregation does not invent the missing cell",
          not any(r['arm'] == 'hrnet_pretrained' and r['fraction'] == '0.25'
                  for r in partial_rows))


def main():
    tmp = tempfile.mkdtemp(prefix='openceph_test_')
    try:
        test_split_is_unchanged()
        test_manifest(tmp)
        test_subsets(tmp)
        test_aggregator(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print("FAILED: " + ", ".join(FAILED))
        sys.exit(1)


if __name__ == "__main__":
    main()
