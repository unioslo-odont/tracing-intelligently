#!/bin/bash
#SBATCH --job-name=openceph_variant_eval
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-01:00:00
#SBATCH --array=0-4                            # TODO: must match the folds you trained
#SBATCH --output=logs/evaluate_variant_%A_fold%a.out
#SBATCH --error=logs/evaluate_variant_%A_fold%a.err

# ==============================================================================
# Evaluate one of the pretrained-backbone variants' checkpoints (see
# slurm/train_variant.sbatch). One array task per fold, evaluating that
# fold's best.pt on its own held-out validation split.
#
# Submit from the REPO ROOT, passing the checkpoint base dir (matching the
# config's training.checkpoint_dir):
#   cd /path/to/Tracing-intelligently
#   mkdir -p logs
#   sbatch slurm/evaluate_variant.sbatch checkpoints/resnet_unet
#   sbatch slurm/evaluate_variant.sbatch checkpoints/hrnet_pretrained
#   sbatch slurm/evaluate_variant.sbatch checkpoints/dinov2_frozen
# (array range must match the folds you trained)
# ==============================================================================

CHECKPOINT_BASE_DIR="$1"
if [ -z "$CHECKPOINT_BASE_DIR" ]; then
    echo "[ERROR] Usage: sbatch slurm/evaluate_variant.sbatch <checkpoint_base_dir>"
    echo "        e.g.:  sbatch slurm/evaluate_variant.sbatch checkpoints/resnet_unet"
    exit 1
fi

echo "=================================================="
echo "OpenCeph backbone-variant evaluation - fold ${SLURM_ARRAY_TASK_ID}"
echo "Checkpoint base dir: $CHECKPOINT_BASE_DIR"
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

# See slurm/train_variant.sbatch: redirect the timm/transformers pretrained-
# weight cache off $HOME (small quota, has been observed to fill up
# entirely) onto /path/to/project instead.

VENV_PATH="${OPENCEPH_VENV:?Set OPENCEPH_VENV to the existing HPC environment}"
echo "Activating virtual environment: $VENV_PATH"
source "$VENV_PATH/bin/activate"

REPO_ROOT="$SLURM_SUBMIT_DIR"
cd "$REPO_ROOT"

if [ ! -f "$REPO_ROOT/code/training/evaluate_backbone_variant.py" ]; then
    echo "[ERROR] $REPO_ROOT/code/training/evaluate_backbone_variant.py not found."
    echo "        SLURM_SUBMIT_DIR='$SLURM_SUBMIT_DIR' - submit this job with 'cd' to the repo root first."
    exit 1
fi

case "$CHECKPOINT_BASE_DIR" in
    /*) FULL_CHECKPOINT_BASE_DIR="$CHECKPOINT_BASE_DIR" ;;
    *)  FULL_CHECKPOINT_BASE_DIR="$REPO_ROOT/$CHECKPOINT_BASE_DIR" ;;
esac

CHECKPOINT="$FULL_CHECKPOINT_BASE_DIR/fold${SLURM_ARRAY_TASK_ID}/best.pt"

python "$REPO_ROOT/code/training/evaluate_backbone_variant.py" \
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
