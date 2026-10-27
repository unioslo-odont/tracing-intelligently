# ==============================================================================
# CACHE REDIRECTION - keep every cache off the home directory
# ==============================================================================
# SOURCE this file (never execute it); it only exports variables:
#     source /path/to/repo/slurm/cache_env.sh
#
# Fox home directories have a small quota; /path/to/project does not. Anything
# that downloads or generates cache - pretrained weights, torch/triton compiled
# kernels, matplotlib's font cache, pip wheels - defaults to ~/.cache and will
# silently fill that quota. The failure mode is nasty: a quota overrun surfaces
# as an unrelated-looking crash somewhere else entirely (this project has
# already hit "RuntimeError: Disk quota exceeded" from a HuggingFace download),
# often hours into a job array, on every fold at once.
#
# An exported HF_HOME (e.g. from local/site.env) always wins, so weights already
# downloaded to an existing cache are still found - do not "tidy" it into the
# cache dir below without moving the contents.
#
# Sourced by every SLURM script in this directory and by slurm/env.sh (the
# interactive setup). Keep it exports-only: no `set -e`, no module loads, no
# venv activation - so that sourcing it can never take down a job.
# ==============================================================================

if [ -z "${OPENCEPH_CACHE_ROOT:-}" ]; then
    echo "[ERROR] Set OPENCEPH_CACHE_ROOT to a writable project cache directory."
    return 1
fi
export OPENCEPH_CACHE_ROOT

export HF_HOME="${HF_HOME:-$OPENCEPH_CACHE_ROOT/huggingface}"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$OPENCEPH_CACHE_ROOT/torch"          # torch.hub checkpoints
export XDG_CACHE_HOME="$OPENCEPH_CACHE_ROOT/xdg"        # the generic ~/.cache fallback
export TRITON_CACHE_DIR="$OPENCEPH_CACHE_ROOT/triton"   # compiled GPU kernels
export MPLCONFIGDIR="$OPENCEPH_CACHE_ROOT/matplotlib"   # font cache (writes a warning if unwritable)
export PIP_CACHE_DIR="$OPENCEPH_CACHE_ROOT/pip"

mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" \
         "$MPLCONFIGDIR" "$PIP_CACHE_DIR" 2>/dev/null || true
