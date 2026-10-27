#!/bin/bash
#SBATCH --job-name=openceph_sweep_train
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --array=0-4                            # one k-fold fold per array task (0..k-1)
#SBATCH --output=logs/sweep_%x_%A_fold%a.out
#SBATCH --error=logs/sweep_%x_%A_fold%a.err

# ==============================================================================
# Generic k-fold trainer for the hyperparameter sweep: trains train.py with an
# arbitrary config plus arbitrary `--set key=value` overrides (one fold per
# array task). Every HP variant is just a different set of overrides + its own
# training.checkpoint_dir. Used by slurm/sweep_unet.sh.
#
# Submit from the REPO ROOT:
#   sbatch slurm/train_generic.sbatch <config.yaml> [--set k=v ...]
# e.g.:
#   sbatch slurm/train_generic.sbatch configs/unet_baseline.yaml \
#       --set training.checkpoint_dir=checkpoints/sweep/aug \
#       --set data.augmentation.enabled=true
# ==============================================================================

CONFIG="$1"
if [ -z "$CONFIG" ]; then
    echo "[ERROR] Usage: sbatch slurm/train_generic.sbatch <config.yaml> [--set k=v ...]"
    exit 1
fi
shift
EXTRA=("$@")   # the --set overrides

echo "=================================================="
echo "OpenCeph sweep training - fold ${SLURM_ARRAY_TASK_ID}"
echo "Config: $CONFIG"
echo "Overrides: ${EXTRA[*]}"
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

python "$REPO_ROOT/code/training/train.py" \
    --config "$CONFIG" \
    --fold "${SLURM_ARRAY_TASK_ID}" \
    "${EXTRA[@]}"

EXIT_CODE=$?
echo ""
echo "=================================================="
echo "Job Finished: $(date) | Exit code: $EXIT_CODE"
echo "=================================================="
exit $EXIT_CODE
