"""
================================================================================
PROVENANCE - git commit + resolved config next to every result
================================================================================
DESCRIPTION:
    "Every result traceable to a config + commit hash" is a project rule, and
    until now it was satisfied indirectly: the resolved config is embedded in
    each checkpoint (.pt), but the commit hash was not recorded anywhere and
    the config was only readable by unpickling a torch checkpoint.

    This module writes a plain `run_meta.json` into the run's OUTPUT directory
    (next to eval_val.json, where results are actually read from), holding:
      - git commit, branch, and whether the tree was dirty at launch,
      - the fully resolved config (after --set overrides and path resolution),
      - the cohort actually trained on (n patients / n images per split, the
        subset file if any, the fold manifest if any),
      - the realised training budget (epochs run, total optimizer steps),
      - SLURM job/array ids, so a result can be traced back to its log file.

    Written twice: once at the START of training (so a crashed or preempted run
    still leaves a record of what it was trying to do) and once at the END with
    the outcome fields filled in.
================================================================================
"""

import json
import os
import subprocess
import hashlib
import platform
from importlib.metadata import version, PackageNotFoundError
from datetime import datetime, timezone


def _git(repo_root, *args):
    try:
        out = subprocess.run(['git', '-C', repo_root, *args], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def git_commit_info(repo_root):
    """Best-effort {commit, branch, dirty}. Never raises: a missing git (or a
    tarball copy of the repo) must not take down a training run."""
    commit = _git(repo_root, 'rev-parse', 'HEAD')
    status = _git(repo_root, 'status', '--porcelain')
    return {
        'commit': commit,
        'branch': _git(repo_root, 'rev-parse', '--abbrev-ref', 'HEAD'),
        'dirty': None if status is None else bool(status.strip()),
    }


def slurm_info():
    """The SLURM identifiers that tie a result to its log file, if any."""
    keys = ('SLURM_JOB_ID', 'SLURM_ARRAY_JOB_ID', 'SLURM_ARRAY_TASK_ID', 'SLURM_JOB_NAME', 'SLURM_NODELIST')
    info = {key.lower(): os.environ[key] for key in keys if key in os.environ}
    return info or None


def write_run_provenance(output_dir, config, repo_root=None, **extra):
    """Write <output_dir>/run_meta.json. Returns the path.

    `extra` is merged in at the top level - pass whatever describes this run
    (arm, fold, fraction, n_train_patients, n_train_images, total_steps, ...)."""
    repo_root = repo_root or os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
    os.makedirs(output_dir, exist_ok=True)
    # Retain uncommitted tracked changes privately with the run, so a dirty
    # commit reference can be reconstructed. Never include this file in a release.
    patch = _git(repo_root, 'diff', '--binary', 'HEAD', '--', 'code', 'configs', 'slurm')
    if patch:
        with open(os.path.join(output_dir, 'source_changes.patch'), 'w', encoding='utf-8') as f:
            f.write(patch + '\n')
    source_hashes = {}
    for folder in ('code', 'configs', 'slurm'):
        for directory, _, filenames in os.walk(os.path.join(repo_root, folder)):
            for filename in sorted(filenames):
                if not filename.endswith(('.py', '.yaml', '.sh', '.sbatch')):
                    continue
                source = os.path.join(directory, filename)
                with open(source, 'rb') as f:
                    source_hashes[os.path.relpath(source, repo_root)] = hashlib.sha256(f.read()).hexdigest()
    packages = {}
    for package in ('torch', 'numpy', 'Pillow', 'PyYAML', 'timm', 'transformers'):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    meta = {
        'written_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'git': git_commit_info(repo_root),
        'slurm': slurm_info(),
        'python_version': platform.python_version(),
        'package_versions': packages,
        'source_sha256': source_hashes,
        **extra,
        'config': config,
    }
    path = os.path.join(output_dir, 'run_meta.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, default=str)
    return path
