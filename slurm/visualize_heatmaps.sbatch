#!/bin/bash
#SBATCH --job-name=openceph_heatmap_viz
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/visualize_heatmaps_%A.out
#SBATCH --error=logs/visualize_heatmaps_%A.err

# ==============================================================================
# Heatmap overlay diagnostic (see code/training/visualize_heatmaps.py):
# renders raw predicted heatmaps for all four model arms, side by side, for
# the same images/landmarks. Single job (not an array) - the script itself
# takes all four checkpoints in one invocation so it can lay them out as a
# [checkpoints x landmarks] grid per image.
#
# Requires all four best.pt already trained:
#   checkpoints/unet_baseline/fold<FOLD>/best.pt
#   checkpoints/resnet_unet/fold<FOLD>/best.pt
#   checkpoints/hrnet_pretrained/fold<FOLD>/best.pt
#   checkpoints/dinov2_frozen/fold<FOLD>/best.pt
#
# The val split is NOT looked up separately - visualize_heatmaps.py
# reconstructs it deterministically per checkpoint from the seed/k/fold baked
# into that checkpoint's own saved config (make_patient_kfold_splits), then
# intersects across all four. As long as all four were trained with the same
# seed/k/fold, that intersection is the full shared validation set.
#
# MAX_IMAGES=0 means no cap - render the whole (intersected) validation split.
#
# Submit from the REPO ROOT:
#   cd /path/to/Tracing-intelligently
#   mkdir -p logs
#   sbatch slurm/visualize_heatmaps.sbatch [FOLD] [MAX_IMAGES] [LANDMARKS...|ALL]
#
# LANDMARKS may be a list of names, or the single word ALL to use every
# enabled landmark from configs/landmarks.yaml instead of typing them out
# (avoids the list going stale if landmarks.yaml changes; the underlying
# script pages large landmark sets across multiple output images per
# source image automatically - see --landmarks-per-page in
# code/training/visualize_heatmaps.py).
#
# Examples:
#   sbatch slurm/visualize_heatmaps.sbatch                  # fold0, whole val set, landmark T
#   sbatch slurm/visualize_heatmaps.sbatch 0 0 T             # fold0, whole val set, landmark T
#   sbatch slurm/visualize_heatmaps.sbatch 0 0 S Pm Pt T OLp # whole val set, DINOv2's 5 failing channels
#   sbatch slurm/visualize_heatmaps.sbatch 0 8 T             # capped at 8 images
#   sbatch slurm/visualize_heatmaps.sbatch 0 0 ALL           # every enabled landmark, whole val set
# ==============================================================================

FOLD="${1:-0}"
MAX_IMAGES="${2:-0}"
shift 2 2>/dev/null
LANDMARKS=("$@")
if [ ${#LANDMARKS[@]} -eq 0 ]; then
    LANDMARKS=(T)
fi
USE_ALL_LANDMARKS=false
if [ "${#LANDMARKS[@]}" -eq 1 ] && [ "${LANDMARKS[0]}" = "ALL" ]; then
    USE_ALL_LANDMARKS=true
fi

echo "=================================================="
echo "OpenCeph heatmap visualization - fold ${FOLD}"
if [ "$USE_ALL_LANDMARKS" = true ]; then
    echo "Landmarks: ALL enabled landmarks in configs/landmarks.yaml"
else
    echo "Landmarks: ${LANDMARKS[*]}"
fi
echo "Max images: ${MAX_IMAGES}"
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

if [ ! -f "$REPO_ROOT/code/training/visualize_heatmaps.py" ]; then
    echo "[ERROR] $REPO_ROOT/code/training/visualize_heatmaps.py not found."
    echo "        SLURM_SUBMIT_DIR='$SLURM_SUBMIT_DIR' - submit this job with 'cd' to the repo root first."
    exit 1
fi

CHECKPOINTS=(
    "$REPO_ROOT/checkpoints/unet_baseline/fold${FOLD}/best.pt"
    "$REPO_ROOT/checkpoints/resnet_unet/fold${FOLD}/best.pt"
    "$REPO_ROOT/checkpoints/hrnet_pretrained/fold${FOLD}/best.pt"
    "$REPO_ROOT/checkpoints/dinov2_frozen/fold${FOLD}/best.pt"
)

for ckpt in "${CHECKPOINTS[@]}"; do
    if [ ! -f "$ckpt" ]; then
        echo "[ERROR] Missing checkpoint: $ckpt"
        exit 1
    fi
done

# All generated artifacts live under the gitignored outputs/ (see
# code/training/output_paths.py) - not a bare heatmap_diagnostics/ at the
# repo root, which was NOT covered by any .gitignore rule.
OUTPUT_DIR="$REPO_ROOT/outputs/heatmap_diagnostics/fold${FOLD}"

if [ "$USE_ALL_LANDMARKS" = true ]; then
    python "$REPO_ROOT/code/training/visualize_heatmaps.py" \
        --checkpoints "${CHECKPOINTS[@]}" \
        --landmarks-config "$REPO_ROOT/configs/landmarks.yaml" \
        --max-images "$MAX_IMAGES" \
        --output-dir "$OUTPUT_DIR"
else
    python "$REPO_ROOT/code/training/visualize_heatmaps.py" \
        --checkpoints "${CHECKPOINTS[@]}" \
        --landmarks "${LANDMARKS[@]}" \
        --max-images "$MAX_IMAGES" \
        --output-dir "$OUTPUT_DIR"
fi

EXIT_CODE=$?
echo ""
echo "=================================================="
echo "Job Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo "Output dir: $OUTPUT_DIR"
echo "=================================================="
exit $EXIT_CODE
