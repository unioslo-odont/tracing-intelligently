#!/bin/bash
#SBATCH --job-name=openceph_variant_train
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --array=0-4                            # TODO: must be 0..(kfold.k - 1) in the config used
#SBATCH --output=logs/train_variant_%A_fold%a.out
#SBATCH --error=logs/train_variant_%A_fold%a.err

# ==============================================================================
# Train one of the pretrained-backbone variants (resnet_unet, hrnet_pretrained,
# dinov2_frozen - see configs/), one k-fold fold per job array task. Generic
# over which config you pass, so one script covers all three arms instead of
# tripling the SLURM boilerplate.
#
# Submit from the REPO ROOT, passing the config as an argument:
#   cd /path/to/Tracing-intelligently
#   mkdir -p logs
#   sbatch slurm/train_variant.sbatch configs/resnet_unet.yaml
#   sbatch slurm/train_variant.sbatch configs/hrnet_pretrained.yaml
#   sbatch slurm/train_variant.sbatch configs/dinov2_frozen.yaml
# (array range must match that config's kfold.k - 1, i.e. 0..k-1)
#
# Any further arguments are passed straight through to the training script, so
# the same launcher covers screening runs (one fold, overridden hyper-parameters):
#   sbatch --array=0 slurm/train_variant.sbatch configs/hrnet_pretrained_run7.yaml
#          --set training.optimizer.encoder_lr=1.0e-5
#          --set training.checkpoint_dir=./checkpoints/screen/hrnet_enc1e-5
#   (all on one command line - the wrapping here is only for readability)
# ==============================================================================

CONFIG_PATH="$1"
if [ -z "$CONFIG_PATH" ]; then
    echo "[ERROR] Usage: sbatch slurm/train_variant.sbatch <config_path> [--set k=v ...]"
    echo "        e.g.:  sbatch slurm/train_variant.sbatch configs/resnet_unet.yaml"
    exit 1
fi
shift
EXTRA=("$@")   # optional --set overrides, forwarded verbatim

echo "=================================================="
echo "OpenCeph backbone-variant training - fold ${SLURM_ARRAY_TASK_ID}"
echo "Config: $CONFIG_PATH"
echo "Overrides: ${EXTRA[*]}"
echo "Job Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_JOB_GPUS"
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
echo "Activating virtual environment: $VENV_PATH"
source "$VENV_PATH/bin/activate"

# NOTE: sbatch copies this script into a spool directory before running it,
# so deriving the repo root from this script's own path (e.g. BASH_SOURCE)
# does not work here. SLURM_SUBMIT_DIR is the directory `sbatch` was invoked
# from - submit with `cd` to the repo root first (as documented above).
REPO_ROOT="$SLURM_SUBMIT_DIR"
cd "$REPO_ROOT"

if [ ! -f "$REPO_ROOT/code/training/train_backbone_variant.py" ]; then
    echo "[ERROR] $REPO_ROOT/code/training/train_backbone_variant.py not found."
    echo "        SLURM_SUBMIT_DIR='$SLURM_SUBMIT_DIR' - submit this job with 'cd' to the repo root first."
    exit 1
fi

case "$CONFIG_PATH" in
    /*) FULL_CONFIG_PATH="$CONFIG_PATH" ;;
    *)  FULL_CONFIG_PATH="$REPO_ROOT/$CONFIG_PATH" ;;
esac

python "$REPO_ROOT/code/training/train_backbone_variant.py" \
    --config "$FULL_CONFIG_PATH" \
    --fold "${SLURM_ARRAY_TASK_ID}" \
    "${EXTRA[@]}"

EXIT_CODE=$?
echo ""
echo "=================================================="
echo "Job Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=================================================="
exit $EXIT_CODE
