"""
================================================================================
PATIENT-LEVEL K-FOLD SPLITTING
================================================================================
DESCRIPTION:
    Splits a CephalometricDataset into K folds at the PATIENT level (grouped
    by `serial`), not the image level. A patient can contribute more than one
    image to the dataset (e.g. two exposures); splitting by image would let
    the same patient appear in both train and validation, leaking
    patient-specific anatomy/positioning into the "held-out" set and
    inflating validation performance.

    The fold assignment is a pure function of (patient serials, k, seed) -
    it does NOT depend on the model, image_size, or anything else - which is
    why every backbone-comparison config can share one seed/k and get
    identical folds. `assign_patient_folds` is that function, factored out so
    fold_manifest.py can persist and re-verify the mapping WITHOUT rebuilding
    the dataset (see that module: the mapping is an experiment invariant, and
    from Run 7/8/9 on it is asserted against a persisted private manifest rather than
    silently regenerated).
================================================================================
"""

import numpy as np


def assign_patient_folds(serials, k, seed=0):
    """Return {serial: fold_index} for the given patient serials.

    The canonical fold assignment: sort the serials (so input order can't
    matter), shuffle once with a seeded RNG, then deal them round-robin into
    k folds. Changing anything here changes every fold in the project -
    don't, without regenerating the fold manifest and re-running everything
    that was compared under the old one."""
    if k < 2:
        raise ValueError("k must be >= 2")
    serials = sorted(serials)
    if len(serials) < k:
        raise ValueError(f"Only {len(serials)} unique patients available, cannot make {k} folds")

    rng = np.random.default_rng(seed)
    rng.shuffle(serials)
    return {serial: i % k for i, serial in enumerate(serials)}


def splits_from_fold_map(dataset, fold_of_serial, k):
    """Turn a {serial: fold} mapping into the usual list of k
    (train_indices, val_indices) tuples indexing into `dataset`.

    Every patient in the dataset must appear in the mapping; a missing one is
    a programming error here (fold_manifest.verify_fold_map is the place that
    turns that into an actionable message)."""
    serial_to_indices = {}
    for idx, entry in enumerate(dataset.samples):
        serial_to_indices.setdefault(entry['serial'], []).append(idx)

    missing = sorted(s for s in serial_to_indices if s not in fold_of_serial)
    if missing:
        raise KeyError(f"{len(missing)} patient(s) have no fold assignment, e.g. {missing[:5]}")

    splits = []
    for fold in range(k):
        val_indices, train_indices = [], []
        for serial, indices in serial_to_indices.items():
            (val_indices if fold_of_serial[serial] == fold else train_indices).extend(indices)
        splits.append((np.array(sorted(train_indices)), np.array(sorted(val_indices))))
    return splits


def make_patient_kfold_splits(dataset, k, seed=0):
    """Return a list of k (train_indices, val_indices) tuples (sorted numpy
    int arrays indexing into `dataset`), split by unique patient serial."""
    serials = {entry['serial'] for entry in dataset.samples}
    fold_of_serial = assign_patient_folds(serials, k, seed)
    return splits_from_fold_map(dataset, fold_of_serial, k)
