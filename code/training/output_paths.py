"""
================================================================================
OUTPUT PATHS — one gitignored home for everything the pipeline GENERATES
================================================================================
DESCRIPTION:
    Every artifact a script produces (evaluation JSON/CSV reports, per-image
    error CSVs, ground-truth-vs-predicted overlays, raw-heatmap diagnostics,
    cross-model comparison tables, Aariz external-test results) is written
    under a single repo-root `outputs/` directory.

    Why centralize here rather than next to each checkpoint:
      - `outputs/` is gitignored (see .gitignore). So is `checkpoints/`, but
        two earlier destinations - `heatmap_diagnostics/` and `aariz_eval/` -
        were NOT under any ignore rule, which risked committing patient-
        derived overlays/CSVs. Routing everything through here removes that
        whole class of accident: if it's a generated artifact, it lands
        somewhere gitignored by construction.
      - One predictable place to collect results from (no more hunting
        through per-model checkpoint folders, or hand-copying CSVs).

    This module is ONLY about generated outputs. Trained model WEIGHTS stay
    under `checkpoints/` - they're an input to these scripts, not a result,
    and are already gitignored via *.pt / checkpoints/.
================================================================================
"""

import os

# This file is code/training/output_paths.py, so the repo root is two
# directories up from its own directory (code/training -> code -> repo root).
REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
OUTPUTS_DIR = os.path.join(REPO_ROOT, 'outputs')


def outputs_path(*parts, make_parent=True):
    """Return a path under the repo-root outputs/ directory, creating its
    parent directory by default. E.g.
        outputs_path('unet_baseline', 'fold0', 'eval_val')
        -> <repo>/outputs/unet_baseline/fold0/eval_val  (dirs created)
    Pass make_parent=False to just build the path string without touching
    the filesystem."""
    path = os.path.join(OUTPUTS_DIR, *parts)
    if make_parent:
        os.makedirs(os.path.dirname(path) or OUTPUTS_DIR, exist_ok=True)
    return path


def run_outputs_path(config, run_name, *parts, make_parent=True):
    """Where one training/eval RUN's artifacts go:
        outputs/[<training.output_group>/]<model_name>/<run_name>/<parts...>
    `model_name` is the checkpoint dir's basename (as it always has been), and
    the optional `training.output_group` adds one grouping level so a whole
    experiment family lands together - e.g. output_group 'label_efficiency'
    with checkpoint_dir './checkpoints/label_efficiency/unet_baseline_frac0.25'
    gives outputs/label_efficiency/unet_baseline_frac0.25/fold0/. Unset (the
    default, and every Run 1-6 config) reproduces the old flat layout."""
    training = config.get('training') or {}
    model_name = os.path.basename(os.path.normpath(training['checkpoint_dir']))
    group = training.get('output_group')
    prefix = ([group] if group else []) + [model_name, run_name]
    return outputs_path(*prefix, *parts, make_parent=make_parent)


def run_output_dir(config, run_name):
    """The directory run_outputs_path writes into (created)."""
    path = run_outputs_path(config, run_name, '_', make_parent=True)
    return os.path.dirname(path)


def model_fold_from_checkpoint(checkpoint_path):
    """Derive ('<model_name>', '<fold_dir>') from a checkpoint laid out as
    .../<model_name>/<fold_dir>/<file>.pt - the structure train.py writes,
    e.g. checkpoints/unet_baseline/fold0/best.pt -> ('unet_baseline', 'fold0').
    Used so a checkpoint's generated results mirror its own name/fold under
    outputs/ instead of colliding. Falls back to safe directory-name-able
    defaults if the path doesn't follow that layout."""
    abspath = os.path.abspath(checkpoint_path)
    fold_dir = os.path.basename(os.path.dirname(abspath)) or 'fold'
    model_name = os.path.basename(os.path.dirname(os.path.dirname(abspath))) or 'unknown_model'
    return model_name, fold_dir
