#!/bin/bash
#SBATCH --job-name=openceph_aariz_eval
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/evaluate_aariz_%A.out
#SBATCH --error=logs/evaluate_aariz_%A.err

# ==============================================================================
# Evaluate trained model checkpoint(s) against the Aariz / CEPHA29 dataset as an
# EXTERNAL TEST SET ONLY - Aariz is never trained on (see
# code/datasets/aariz_dataset.py for why: CC-BY-NC-ND 4.0 licensing makes
# training-and-releasing-weights a real question mark, while pure evaluation of
# an already-trained checkpoint is not). Reads the ORIGINAL Aariz directory
# layout and scores against the SENIOR orthodontist tracings by default.
#
# Reads each model's deployable checkpoint at:
#   checkpoints/<model>/<MODEL_SUBDIR>/best.pt     (MODEL_SUBDIR default: final)
# Missing checkpoints are skipped with a [WARN] (so this also works when only
# the U-Net final model exists). Pass MODEL_SUBDIR=fold0 to score the CV
# fold-0 checkpoints of all four variants instead.
#
# Only landmarks with a confirmed Aariz correspondence (see
# aariz_dataset.AARIZ_TO_REFERENCE_MAPPING - 18 of our 26) are checkable; the
# rest of each checkpoint's vocabulary reports null/0-points, which is correct
# (no Aariz ground truth exists for them), not a bug.
#
# CALIBRATION: per-image mm/pixel is read from
# <AARIZ_ROOT>/cephalogram_machine_mappings.csv (columns auto-detected). If that
# file/columns can't be resolved, mm-based MRE/SDR are null but raw per-landmark
# pixel error is still written to each *_per_landmark_image.csv.
#
# Submit from the REPO ROOT:
#   cd /path/to/Tracing-intelligently
#   mkdir -p logs
#   sbatch slurm/evaluate_aariz.sbatch [AARIZ_ROOT] [SPLIT] [ANNOTATOR] [MODEL_SUBDIR]
#
# AARIZ_ROOT defaults to the exported AARIZ_ROOT (local/site.env):
#   sbatch slurm/evaluate_aariz.sbatch
#   sbatch slurm/evaluate_aariz.sbatch /path/to/project/data/Aariz test senior final
# ==============================================================================

AARIZ_ROOT="${1:-${AARIZ_ROOT:-/path/to/project/data/Aariz}}"
SPLIT="${2:-test}"
ANNOTATOR="${3:-senior}"
MODEL_SUBDIR="${4:-final}"

# Optional overrides for the machine-mappings calibration CSV (leave empty to
# use <AARIZ_ROOT>/cephalogram_machine_mappings.csv with auto-detected columns).
# All four accept an environment override, e.g.:
#   sbatch --export=ALL,DISABLE_PNS_PM=1 slurm/evaluate_aariz.sbatch ...
MACHINE_MAPPINGS_CSV="${MACHINE_MAPPINGS_CSV:-}"
ID_COLUMN="${ID_COLUMN:-}"
MM_PER_PIXEL_COLUMN="${MM_PER_PIXEL_COLUMN:-}"
# PNS -> Pm is ON by default (confirmed equivalent 2026-08-03: 19 checkable landmarks).
# Set DISABLE_PNS_PM=1 for an 18-landmark sensitivity check.
DISABLE_PNS_PM="${DISABLE_PNS_PM:-0}"

echo "=================================================="
echo "OpenCeph Aariz external test evaluation"
echo "Aariz root: ${AARIZ_ROOT} | split=${SPLIT} | annotator=${ANNOTATOR} | model subdir=${MODEL_SUBDIR}"
echo "Job Started: $(date) | Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST"
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

if [ ! -f "$REPO_ROOT/code/training/evaluate_aariz.py" ]; then
    echo "[ERROR] $REPO_ROOT/code/training/evaluate_aariz.py not found."
    echo "        SLURM_SUBMIT_DIR='$SLURM_SUBMIT_DIR' - submit this job with 'cd' to the repo root first."
    exit 1
fi

if [ ! -d "$AARIZ_ROOT/$SPLIT" ]; then
    echo "[ERROR] Aariz split dir not found: $AARIZ_ROOT/$SPLIT"
    exit 1
fi

CHECKPOINTS=(
    "unet_baseline_15pt:$REPO_ROOT/checkpoints/unet_baseline_15pt/${MODEL_SUBDIR}/best.pt"
)

# All generated artifacts live under the gitignored outputs/ (see
# code/training/output_paths.py). A distinct suffix for the 18-landmark (no-Pm)
# sensitivity run so it never overwrites the default 19-landmark output.
SUFFIX=""
[ "$DISABLE_PNS_PM" = "1" ] && SUFFIX="_noPm"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/release_review/aariz/${MODEL_SUBDIR}_${SPLIT}_${ANNOTATOR}${SUFFIX}}"
if [ -e "$OUTPUT_ROOT/unet_baseline_15pt_eval_aariz.json" ]; then
    echo "[ERROR] Output exists; set OUTPUT_ROOT to a new directory."
    exit 1
fi
mkdir -p "$OUTPUT_ROOT"

# Shared optional args
EXTRA_ARGS=(--unmap Gn)
[ -n "$MACHINE_MAPPINGS_CSV" ] && EXTRA_ARGS+=(--machine-mappings-csv "$MACHINE_MAPPINGS_CSV")
[ -n "$ID_COLUMN" ] && EXTRA_ARGS+=(--id-column "$ID_COLUMN")
[ -n "$MM_PER_PIXEL_COLUMN" ] && EXTRA_ARGS+=(--mm-per-pixel-column "$MM_PER_PIXEL_COLUMN")
[ "$DISABLE_PNS_PM" = "1" ] && EXTRA_ARGS+=(--no-pns-pm)

OVERALL_EXIT=0
for entry in "${CHECKPOINTS[@]}"; do
    MODEL_NAME="${entry%%:*}"
    CKPT="${entry#*:}"

    if [ ! -f "$CKPT" ]; then
        echo "[ERROR] Missing checkpoint $CKPT"
        exit 1
    fi

    echo ""
    echo "--- Evaluating ${MODEL_NAME} (${MODEL_SUBDIR}) against Aariz ${SPLIT}/${ANNOTATOR} ---"

    python "$REPO_ROOT/code/training/evaluate_aariz.py" \
        --checkpoint "$CKPT" \
        --root "$AARIZ_ROOT" \
        --split "$SPLIT" \
        --annotator "$ANNOTATOR" \
        --output "$OUTPUT_ROOT/${MODEL_NAME}_eval_aariz" \
        --save-overlays \
        --max-overlay-images 20 \
        "${EXTRA_ARGS[@]}"

    if [ $? -ne 0 ]; then
        echo "[ERROR] Evaluation failed for ${MODEL_NAME}"
        OVERALL_EXIT=1
    fi
done

echo ""
echo "=================================================="
echo "Job Finished: $(date)"
echo "Exit code: $OVERALL_EXIT"
echo "Output dir: $OUTPUT_ROOT"
echo "=================================================="
exit $OVERALL_EXIT
