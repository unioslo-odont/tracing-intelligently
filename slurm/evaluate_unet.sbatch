#!/bin/bash
#SBATCH --job-name=openceph_unet_eval
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-01:00:00
#SBATCH --array=0-4                            # TODO: must match the folds you trained
#SBATCH --output=logs/evaluate_unet_%A_fold%a.out
#SBATCH --error=logs/evaluate_unet_%A_fold%a.err

# ==============================================================================
# Evaluate the U-Net heatmap baseline's checkpoints: one array task per fold,
# evaluating that fold's best.pt on its own held-out validation split. Run
# after slurm/train_unet.sbatch has produced
# checkpoints/unet_baseline/fold<N>/best.pt for each fold.
#
# Submit from the REPO ROOT (SLURM_SUBMIT_DIR is used below to locate the
# code/config - this must be the repo root, not the slurm/ folder):
#   cd /path/to/Tracing-intelligently
#   mkdir -p logs
#   sbatch slurm/evaluate_unet.sbatch
# (array range must match the folds you trained)
# ==============================================================================

echo "=================================================="
echo "OpenCeph U-Net evaluation - fold ${SLURM_ARRAY_TASK_ID}"
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

# NOTE: sbatch copies this script into a spool directory before running it,
# so deriving the repo root from this script's own path (e.g. BASH_SOURCE)
# does not work here. SLURM_SUBMIT_DIR is the directory `sbatch` was invoked
# from - submit with `cd` to the repo root first (as documented above).
REPO_ROOT="$SLURM_SUBMIT_DIR"
cd "$REPO_ROOT"

if [ ! -f "$REPO_ROOT/code/training/evaluate.py" ]; then
    echo "[ERROR] $REPO_ROOT/code/training/evaluate.py not found."
    echo "        SLURM_SUBMIT_DIR='$SLURM_SUBMIT_DIR' - submit this job with 'cd' to the repo root first."
    exit 1
fi

CHECKPOINT="$REPO_ROOT/checkpoints/unet_baseline/fold${SLURM_ARRAY_TASK_ID}/best.pt"

python "$REPO_ROOT/code/training/evaluate.py" \
    --checkpoint "$CHECKPOINT" \
    --split val \
    --save-overlays

EXIT_CODE=$?
echo ""
echo "=================================================="
echo "Job Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=================================================="
exit $EXIT_CODE
