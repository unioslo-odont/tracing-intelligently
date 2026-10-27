# Contributing

Thanks for your interest. This is an academic research project (an open, honestly-evaluated baseline
for hard-tissue cephalometric landmark detection), and outside contributions — bug reports, fixes,
evaluations on new cohorts, or method improvements — are welcome.

## Before you start

- **There is no patient data in this repository, and there never will be.** The internal training
  cohort is GDPR/REK/Sikt-restricted and cannot be shared. To train, you need your own cephalometric
  data (Facad `.fcd` tracings, or adapt the dataset loader in `code/datasets/`). For external
  evaluation you can use the public Aariz / CEPHA29 dataset (evaluation only — check its licence).
- Please **never** include radiographs, tracings, patient identifiers, or model outputs derived from
  private data in an issue, PR, notebook, or commit.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m compileall code/            # quick sanity
python code/training/test_optim_groups.py       # unit tests for the optimizer/param-group logic
python code/training/test_label_efficiency.py    # unit tests for patient subsetting
python code/training/test_reporting.py           # metrics, subsets and report resolution
```

## Conventions that keep results comparable

For this project, execution and testing are performed manually by the maintainer
on the closed HPC environment. Local review may read and edit source and inspect
saved outputs, but must not run project code or access patient images.

- **Config-driven and seeded.** Every result must be traceable to a config file + a git commit. Prefer
  `--set a.b.c=value` overrides to editing configs for quick experiments; checkpoints embed the resolved
  config.
- **Patient-level splits only** (`code/datasets/kfold.py`), asserted against a persisted fold manifest
  (`configs/folds/*.csv`) rather than regenerated.
- **New experiments set `training.output_group`** so they land in their own `outputs/<group>/` and never
  overwrite an existing run's results.
- **Cross-model claims use `code/training/paired_fold_comparison.py`** (folds are paired), not two
  mean±SD columns.
- Keep new code in the style of the surrounding module; keep the datasets I/O-only (encoding and
  augmentation live in the transform).

## Submitting changes

1. Open an issue describing the bug or proposal first for anything non-trivial.
2. Branch from `main`, keep the change focused, and run the tests above.
3. Open a pull request with a clear description of what changed and why, and note whether it affects any
   reported result.

## Reporting issues

Use the issue templates (bug report / question). For anything security- or privacy-sensitive, contact
the maintainer directly rather than opening a public issue.
