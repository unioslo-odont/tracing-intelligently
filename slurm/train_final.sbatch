#!/bin/bash
#SBATCH --job-name=openceph_final
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/train_final_%A.out
#SBATCH --error=logs/train_final_%A.err

# ==============================================================================
# Train the FINAL deployable model on ALL data (train.py --all-data). A single
# job, NOT a k-fold array. A small patient-level holdout (config
# training.holdout_fraction, default 0.1) is used only for early stopping; the
# model trains on the rest (~90%, more than any CV fold). Writes the deployable
# checkpoint to <checkpoint_dir>/final/best.pt.
#
# The config should already be the frozen final recipe (e.g. unet_baseline.yaml
# with augmentation on + the swept lr). Reported accuracy is the CV estimate
# (compare_backbones) and the external Aariz number - NOT this run's own splits.
#
# Submit from the REPO ROOT:
#   sbatch slurm/train_final.sbatch [config.yaml]     # default: configs/unet_baseline.yaml
# ==============================================================================

CONFIG="${1:-configs/unet_baseline.yaml}"

echo "=================================================="
echo "OpenCeph FINAL all-data training"
echo "Config: $CONFIG"
echo "Job Started: $(date) | Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST"
echo "=================================================="

module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.6.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

# All caches - pretrained weights, torch/triton kernels, matplotlib - are
# redirected off the small home-directory quota onto /path/to/project.
# Single source of truth, shared with the interactive setup (slurm/env.sh):
source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/cache_env.sh"

VENV_PATH="${OPENCEPH_VENV:?Set OPENCEPH_VENV to the existing HPC environment}"
source "$VENV_PATH/bin/activate"

REPO_ROOT="$SLURM_SUBMIT_DIR"
cd "$REPO_ROOT"

if [ ! -f "$REPO_ROOT/code/training/train.py" ]; then
    echo "[ERROR] $REPO_ROOT/code/training/train.py not found - submit from the repo root."
    exit 1
fi

python "$REPO_ROOT/code/training/train.py" --config "$CONFIG" --all-data

EXIT_CODE=$?
echo ""
echo "=================================================="
echo "Job Finished: $(date) | Exit code: $EXIT_CODE"
echo "Deployable checkpoint: <checkpoint_dir>/final/best.pt"
echo "=================================================="
exit $EXIT_CODE
