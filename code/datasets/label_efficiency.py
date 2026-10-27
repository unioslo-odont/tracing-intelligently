"""
================================================================================
LABEL-EFFICIENCY SUBSETS - nested, patient-level training-set subsampling
================================================================================
DESCRIPTION:
    The label-efficiency curve (Run 8) asks: how does accuracy fall off as the
    annotated cohort shrinks, and does ImageNet pretraining buy anything when
    labels are scarce? To answer that honestly, only the TRAINING side may
    shrink:

      - The VALIDATION fold is never touched. Every point on the curve, at
        every fraction, is scored on exactly the Run 6 validation set of that
        fold - otherwise a smaller training set would also mean a different
        (easier or harder) test set, and the curve would measure both at once.
      - The unit of sampling is the PATIENT, not the image. Sampling images
        would put two exposures of the same patient in "different" training
        examples and overstate how much independent labelling was done. All
        images of a selected patient are used, so n_images per fraction is
        NOT exactly fraction*n_images - both counts are logged.
      - Subsets are NESTED: one seeded shuffle per fold, and each fraction is
        a PREFIX of it, so P(0.125) c P(0.25) c P(0.50) c P(1.0). The 25%
        cohort is literally the 50% cohort with half the patients removed, so
        successive points on the curve differ by *which patients were added*,
        not by an unrelated redraw - much less variance than sampling each
        fraction independently.

    The sampler needs only the fold manifest (patient -> fold), never the
    images, so subsets can be generated and reviewed off-cluster and committed
    to git alongside the configs that consume them.

    Idempotence is enforced rather than assumed: re-running the generator over
    an existing subset file verifies it byte-for-byte and refuses to silently
    rewrite it (a changed subset would invalidate every run already trained on
    it while leaving the filename identical).

FORMAT (configs/label_efficiency/fold<k>_frac<f>.csv):
    '#'-prefixed metadata lines, then a single 'patient_serial' column, sorted.
================================================================================
"""

import csv
import os

import numpy as np


def format_fraction(fraction):
    """Canonical short form used in filenames and directory names: 0.5, 0.25,
    0.125 (never '0.50' / '0.2500', so one fraction never gets two names)."""
    return f"{float(fraction):g}"


def subset_filename(fold, fraction):
    return f"fold{fold}_frac{format_fraction(fraction)}.csv"


def arm_run_name(arm, fraction):
    """Directory name for one (arm, fraction) cell of the curve, e.g.
    'unet_baseline_frac0.25'. Fraction 1.0 is the Run 6 arm itself, which keeps
    its own plain name - those runs are reused, never repeated."""
    if float(fraction) >= 1.0:
        return arm
    return f"{arm}_frac{format_fraction(fraction)}"


def nested_patient_subset(train_serials, fraction, fold, seed=0):
    """Return the sorted patient subset for one (fold, fraction) cell.

    One shuffle per (seed, fold) - deliberately independent of `fraction` - and
    each fraction takes a prefix of it, which is what makes the subsets nested.
    At least one patient is always returned."""
    fraction = float(fraction)
    if not 0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    ordered = sorted(train_serials)
    if fraction >= 1.0:
        return ordered

    # A per-fold stream (seed + fold) so folds don't share an ordering, but a
    # fixed one across fractions and arms so the same patients are used by
    # every arm at a given (fold, fraction) - the arms stay paired.
    rng = np.random.default_rng([seed, fold])
    rng.shuffle(ordered)
    n = max(1, int(round(fraction * len(ordered))))
    return sorted(ordered[:n])


def write_patient_subset(path, serials, meta=None, force=False):
    """Write a patient-subset CSV, idempotently.

    Returns 'created' | 'unchanged'. Raises if the file exists with DIFFERENT
    contents and force is False - that file may already have runs trained
    against it, and rewriting it in place would make those runs unreproducible
    while leaving every path and table label unchanged."""
    serials = sorted(serials)
    if os.path.exists(path):
        existing = load_patient_subset(path)
        if existing == serials:
            return 'unchanged'
        if not force:
            raise RuntimeError(
                f"{path} already exists with a DIFFERENT patient list "
                f"({len(existing)} patients on disk vs {len(serials)} now, "
                f"{len(set(existing) ^ set(serials))} differing). Runs may already have been "
                "trained on the version on disk. Delete it explicitly (and re-run every job that "
                "used it) if the change is intended, or pass force=True."
            )

    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        for key, value in (meta or {}).items():
            f.write(f"# {key}: {value}\n")
        f.write("# Generated by code/training/make_label_efficiency_subsets.py - DO NOT hand-edit.\n")
        f.write("# Training reads this as data.train_patient_subset_path; the validation fold is\n")
        f.write("# unaffected (see code/datasets/label_efficiency.py).\n")
        writer = csv.writer(f)
        writer.writerow(['patient_serial'])
        for serial in serials:
            writer.writerow([serial])
    return 'created'


def load_patient_subset(path):
    """Return the sorted patient serials listed in a subset CSV."""
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.startswith('#'):
                rows.append(line)
    reader = csv.DictReader(rows)
    if 'patient_serial' not in (reader.fieldnames or []):
        raise ValueError(f"{path} has no 'patient_serial' column (found {reader.fieldnames})")
    serials = sorted({row['patient_serial'].strip() for row in reader if row['patient_serial'].strip()})
    if not serials:
        raise ValueError(f"{path} lists no patients")
    return serials


def restrict_indices_to_patients(dataset, indices, serials, what='training'):
    """Keep only the `indices` whose sample belongs to one of `serials`.

    Every listed patient must actually be present in `indices`; a subset naming
    patients that aren't in this fold's training pool means the subset and the
    fold manifest have drifted apart, which would silently shrink the cohort."""
    wanted = set(serials)
    available = {dataset.samples[i]['serial'] for i in indices}
    unknown = sorted(wanted - available)
    if unknown:
        raise RuntimeError(
            f"{len(unknown)} patient(s) in the {what} subset are not in this fold's {what} pool "
            f"(e.g. {unknown[:5]}). The subset was generated from a different fold manifest, fold "
            "index, or exclusion list - regenerate it with "
            "code/training/make_label_efficiency_subsets.py against the manifest this run uses."
        )
    kept = [i for i in indices if dataset.samples[i]['serial'] in wanted]
    if not kept:
        raise RuntimeError(f"The {what} subset selected 0 images - nothing to train on")
    return np.array(sorted(kept))
