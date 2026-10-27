#!/bin/bash
#SBATCH --job-name=openceph_compare
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=0-00:15:00
#SBATCH --output=logs/compare_%A.out
#SBATCH --error=logs/compare_%A.err

# ==============================================================================
# Aggregate the four models' per-fold eval_val.json into the cross-model
# comparison tables (outputs/comparison_*). Reads the reports that each
# training run writes AUTOMATICALLY (train.py / train_backbone_variant.py ->
# outputs/<model>/fold<N>/eval_val.json), so no separate evaluate step is
# needed - just run this after training finishes. This is the final,
# dependency-gated step of slurm/run_all.sh.
#
# NOTE: this job does not actually need a GPU (it only reads JSON and draws a
# bar chart). It requests accel + gpu:1 only because that is the known-good
# partition config for this account, so the automated chain is guaranteed to
# run. To avoid tying up a GPU, switch to a CPU partition your allocation has
# (e.g. change to `--partition=normal` and delete the `--gres=gpu:1` line).
#
# Submit from the REPO ROOT:
#   cd /path/to/Tracing-intelligently && mkdir -p logs
#   sbatch slurm/compare_backbones.sbatch
# ==============================================================================

echo "=================================================="
echo "OpenCeph cross-model comparison"
echo "Job Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=================================================="

module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.6.0

export PYTHONUNBUFFERED=1

# All caches - pretrained weights, torch/triton kernels, matplotlib - are
# redirected off the small home-directory quota onto /path/to/project.
# Single source of truth, shared with the interactive setup (slurm/env.sh):
source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/cache_env.sh"

VENV_PATH="${OPENCEPH_VENV:?Set OPENCEPH_VENV to the existing HPC environment}"
echo "Activating virtual environment: $VENV_PATH"
source "$VENV_PATH/bin/activate"

REPO_ROOT="$SLURM_SUBMIT_DIR"
cd "$REPO_ROOT"

if [ ! -f "$REPO_ROOT/code/training/compare_backbones.py" ]; then
    echo "[ERROR] $REPO_ROOT/code/training/compare_backbones.py not found."
    echo "        SLURM_SUBMIT_DIR='$SLURM_SUBMIT_DIR' - submit this job from the repo root."
    exit 1
fi

# Optional args: a list of checkpoint dirs to compare, then --output <prefix>.
# With no args, compares the four standard arms and writes outputs/comparison_*.
# The HP sweep (slurm/sweep_unet.sh) passes its sweep dirs + a sweep output prefix.
if [ "$#" -eq 0 ]; then
    DIRS=(checkpoints/unet_baseline checkpoints/resnet_unet \
          checkpoints/hrnet_pretrained checkpoints/dinov2_frozen)
    OUT_ARGS=()
else
    DIRS=()
    OUT_ARGS=()
    seen_option=0
    for a in "$@"; do
        [[ "$a" == --* ]] && seen_option=1
        if [ "$seen_option" -eq 1 ]; then OUT_ARGS+=("$a"); else DIRS+=("$a"); fi
    done
fi

echo "Comparing: ${DIRS[*]}"
python "$REPO_ROOT/code/training/compare_backbones.py" --checkpoint-dirs "${DIRS[@]}" "${OUT_ARGS[@]}"

EXIT_CODE=$?
echo ""
echo "=================================================="
echo "Job Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo "Comparison tables in $REPO_ROOT/outputs/"
echo "=================================================="
exit $EXIT_CODE
