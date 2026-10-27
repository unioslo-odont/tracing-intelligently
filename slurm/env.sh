# ==============================================================================
# INTERACTIVE ENVIRONMENT - one command to get a working login-node shell
# ==============================================================================
# SOURCE this file (never execute it - it has to modify your shell):
#     cd /path/to/project/GitHub/Tracing-intelligently
#     source slurm/env.sh
#
# It does exactly what the SLURM scripts do to their own environment, in the
# same order, so an interactive session and a batch job behave identically:
#   1. module purge, then the EXACT Python module the venv was built against,
#   2. CUDA,
#   3. the project venv,
#   4. cache redirection off the home directory (slurm/cache_env.sh).
#
# Step 1 is not optional and not cosmetic. A venv has no standard library of its
# own - it uses the one from the interpreter it was built against. Loading a
# different PATCH version (3.11.3 instead of 3.11.5) puts the wrong
# libpython3.11.so on LD_LIBRARY_PATH, and the first stdlib C extension to load
# fails with an undefined-symbol error that looks nothing like a module problem:
#     ImportError: .../math.cpython-311-x86_64-linux-gnu.so: undefined symbol: _PyModule_Add
# If that appears, you have the wrong Python module loaded - check
# `head -3 $VENV_PATH/pyvenv.cfg` for the version the venv actually needs.
# ==============================================================================

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    OPENCEPH_REPO="$SLURM_SUBMIT_DIR"
elif [ -n "${BASH_SOURCE[0]:-}" ]; then
    OPENCEPH_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
else
    OPENCEPH_REPO="$(pwd)"
fi
export OPENCEPH_REPO

# Private site settings (venv, caches, SLURM account, module versions, data
# roots): git-ignored local/site.env, template slurm/site.example.env. Sourced
# first so it can set everything below. Exported SBATCH_ACCOUNT also reaches
# sbatch, where it overrides the scripts' #SBATCH --account placeholder.
OPENCEPH_SITE_ENV="${OPENCEPH_SITE_ENV:-$OPENCEPH_REPO/local/site.env}"
if [ -f "$OPENCEPH_SITE_ENV" ]; then
    # shellcheck disable=SC1090
    source "$OPENCEPH_SITE_ENV"
fi

OPENCEPH_PYTHON_MODULE="${OPENCEPH_PYTHON_MODULE:-Python/3.11.5-GCCcore-13.2.0}"
OPENCEPH_CUDA_MODULE="${OPENCEPH_CUDA_MODULE:-CUDA/12.6.0}"
if [ -z "${OPENCEPH_VENV:-}" ] || [ -z "${OPENCEPH_CACHE_ROOT:-}" ]; then
    echo "[ERROR] Set OPENCEPH_VENV and OPENCEPH_CACHE_ROOT for your HPC environment"
    echo "        (cp slurm/site.example.env local/site.env, then edit it)."
    return 1
fi

# --- Ensure the `module` command exists -------------------------------------
# `module` is a shell FUNCTION defined by the site's Lmod/env-modules init
# script at login. A bare non-login shell (e.g. you typed `bash`, or an `srun`
# shell) inherits Lmod's env VARIABLES (LMOD_PKG, MODULESHOME, ...) but NOT the
# function, so `module load` fails with "module: command not found" - and then
# the Python module never loads, which surfaces downstream as the venv's
# interpreter failing with "libpython3.11.so.1.0: cannot open shared object
# file". Re-source the init script (found via the inherited vars) to define it.
if ! command -v module >/dev/null 2>&1; then
    for _mod_init in \
        "${LMOD_PKG:+$LMOD_PKG/init/bash}" \
        "${LMOD_DIR:+$LMOD_DIR/../init/bash}" \
        "${MODULESHOME:+$MODULESHOME/init/bash}" \
        /etc/profile.d/lmod.sh /etc/profile.d/z00_lmod.sh /etc/profile.d/modules.sh \
        /usr/share/lmod/lmod/init/bash ; do
        [ -n "$_mod_init" ] && [ -f "$_mod_init" ] || continue
        # shellcheck disable=SC1090
        source "$_mod_init" 2>/dev/null
        command -v module >/dev/null 2>&1 && break
    done
    unset _mod_init
fi
if ! command -v module >/dev/null 2>&1; then
    echo "[ERROR] 'module' command not available and could not be auto-initialised."
    echo "        You are almost certainly in a non-login shell. Easiest fix:"
    echo "            exec bash -l          # then: cd \$OPENCEPH_REPO && source slurm/env.sh"
    echo "        Or source your site's Lmod init directly (check: echo \$MODULESHOME \$LMOD_PKG),"
    echo "        e.g.:   source \$MODULESHOME/init/bash"
    echo "        Not loading modules - the venv Python would fail without libpython."
    return 1 2>/dev/null || exit 1
fi

module purge
module load "$OPENCEPH_PYTHON_MODULE"
module load "$OPENCEPH_CUDA_MODULE"

# A PYTHONPATH/PYTHONHOME inherited from another module or an old .bashrc will
# happily point the interpreter at a foreign standard library - clear both.
unset PYTHONHOME PYTHONPATH

if [ ! -f "$OPENCEPH_VENV/bin/activate" ]; then
    echo "[ERROR] No venv at $OPENCEPH_VENV - create it with: bash slurm/setup_env.sh"
else
    # shellcheck disable=SC1091
    source "$OPENCEPH_VENV/bin/activate"
fi

# shellcheck disable=SC1091
source "$OPENCEPH_REPO/slurm/cache_env.sh"

echo "OpenCeph environment:"
echo "  repo    $OPENCEPH_REPO"
echo "  python  $(command -v python)  ($(python -V 2>&1))"
echo "  venv    $OPENCEPH_VENV"
echo "  caches  HF_HOME=$HF_HOME  OPENCEPH_CACHE_ROOT=$OPENCEPH_CACHE_ROOT  (home directory untouched)"
python - <<'PY' 2>/dev/null || echo "  [WARN] the interpreter cannot import its own standard library - see the header of this file"
import math, sys
print(f"  stdlib  OK ({sys.version.split()[0]}, base_prefix {sys.base_prefix})")
try:
    import torch
    print(f"  torch   {torch.__version__} (cuda build {torch.version.cuda}; cuda available: {torch.cuda.is_available()} - False is expected on a login node)")
except ImportError as e:
    print(f"  torch   NOT importable: {e}")
PY
