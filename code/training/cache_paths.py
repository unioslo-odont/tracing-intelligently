"""
================================================================================
CACHE PATHS - the Python-side twin of slurm/cache_env.sh
================================================================================
DESCRIPTION:
    The SLURM scripts and `source slurm/env.sh` redirect every cache off the
    home directory onto /path/to/project. A script run interactively in a shell
    where that has not been done would silently download pretrained weights into
    ~/.cache/huggingface and eat the (small) home quota - which has already
    happened once in this project.

    So the two scripts that can trigger a download call `ensure_project_caches()`
    first. It only fills in variables that are UNSET, so an explicitly exported
    HF_HOME (or a machine that is not Fox) always wins, and it never creates
    directories outside a project root that actually exists.
================================================================================
"""

import os

PROJECT_ROOT_CANDIDATES = ()  # No institution-specific defaults in public code.
HF_HOME_SUBDIR = 'hf_cache'
CACHE_SUBDIR = 'cache'


def project_root():
    """The first Fox project directory that exists, or None off-cluster."""
    for root in PROJECT_ROOT_CANDIDATES:
        if os.path.isdir(root):
            return root
    return None


def ensure_project_caches(verbose=True):
    """Point the download/build caches at the project area, unless already set.
    Returns the dict of variables this call established (empty off-cluster)."""
    root = os.environ.get('OPENCEPH_CACHE_ROOT') or project_root()
    if root is None:
        return {}

    cache_root = os.environ.get('OPENCEPH_CACHE_ROOT') or os.path.join(root, CACHE_SUBDIR)
    defaults = {
        'OPENCEPH_CACHE_ROOT': cache_root,
        'HF_HOME': os.path.join(cache_root, 'huggingface'),
        'TORCH_HOME': os.path.join(cache_root, 'torch'),
        'XDG_CACHE_HOME': os.path.join(cache_root, 'xdg'),
        'TRITON_CACHE_DIR': os.path.join(cache_root, 'triton'),
        'MPLCONFIGDIR': os.path.join(cache_root, 'matplotlib'),
    }

    applied = {}
    for key, value in defaults.items():
        if os.environ.get(key):
            continue
        os.environ[key] = value
        applied[key] = value
        if key != 'OPENCEPH_CACHE_ROOT':
            try:
                os.makedirs(value, exist_ok=True)
            except OSError:
                pass

    if applied and verbose:
        print(f"[cache] redirected {', '.join(sorted(applied))} onto {root} "
              "(home-directory quota is too small; see slurm/cache_env.sh)")
    return applied
