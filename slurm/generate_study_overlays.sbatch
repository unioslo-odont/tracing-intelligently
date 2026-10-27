#!/bin/bash
#SBATCH --job-name=openceph_overlays
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-03:00:00
#SBATCH --output=logs/generate_study_overlays_%A.out
#SBATCH --error=logs/generate_study_overlays_%A.err

# Generate poster/review overlays from the selected final U-Net checkpoint.
# This performs inference only; it does not train or modify a checkpoint.
#
# The two displayed sets match the reported study endpoints:
#   internal: 34 final-run holdout images, all 16 trained channels
#   external: 150 Aariz test images, 12 comparable landmarks
#
# Submit from the repository root after exporting OPENCEPH_VENV and
# OPENCEPH_CACHE_ROOT:
#   mkdir -p logs
#   sbatch --account=<account> slurm/generate_study_overlays.sbatch /path/to/Aariz
#
# Optional second argument: a new output directory. If omitted, a timestamped
# directory is created under outputs/release_review/.

set -euo pipefail

AARIZ_ROOT="${1:?Usage: sbatch slurm/generate_study_overlays.sbatch AARIZ_ROOT [OUTPUT_ROOT]}"
RUN_STAMP="$(date +%Y%m%d-%H%M%S)"
OUTPUT_ROOT="${2:-outputs/release_review/study_overlays_${RUN_STAMP}}"
REPO_ROOT="${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"
CHECKPOINT="$REPO_ROOT/checkpoints/unet_baseline_15pt/final/best.pt"

if [ ! -f "$CHECKPOINT" ]; then
    echo "[ERROR] Final checkpoint not found: $CHECKPOINT"
    exit 1
fi
if [ ! -d "$AARIZ_ROOT/test" ]; then
    echo "[ERROR] Aariz test directory not found: $AARIZ_ROOT/test"
    exit 1
fi
if [ -e "$OUTPUT_ROOT" ]; then
    echo "[ERROR] Output already exists: $OUTPUT_ROOT"
    echo "        Pass a new OUTPUT_ROOT as the second argument."
    exit 1
fi

cd "$REPO_ROOT"
source slurm/env.sh
mkdir -p "$OUTPUT_ROOT"

INTERNAL_LANDMARKS=(A B Gn Ii Iia Is Isa N Or Pg Po Pt S Sp iGo pGo)
EXTERNAL_LANDMARKS=(A B Ii Iia Is Isa N Or Pg Po S Sp)

python code/training/evaluate.py \
    --checkpoint "$CHECKPOINT" \
    --all-data \
    --split val \
    --landmarks "${INTERNAL_LANDMARKS[@]}" \
    --save-overlays \
    --overlay-landmarks "${INTERNAL_LANDMARKS[@]}" \
    --output "$OUTPUT_ROOT/internal_final16"

python code/training/evaluate_aariz.py \
    --checkpoint "$CHECKPOINT" \
    --root "$AARIZ_ROOT" \
    --split test \
    --annotator senior \
    --unmap Gn \
    --save-overlays \
    --overlay-landmarks "${EXTERNAL_LANDMARKS[@]}" \
    --output "$OUTPUT_ROOT/aariz_external12"

INTERNAL_COUNT="$(find "$OUTPUT_ROOT/internal_final16_overlays" -maxdepth 1 -type f -name '*_overlay.png' | wc -l)"
EXTERNAL_COUNT="$(find "$OUTPUT_ROOT/aariz_external12_overlays" -maxdepth 1 -type f -name '*_overlay.png' | wc -l)"

echo "Internal final-holdout overlays: $INTERNAL_COUNT (expected 34)"
echo "Aariz test overlays:             $EXTERNAL_COUNT (expected 150)"
echo "Output directory: $OUTPUT_ROOT"

if [ "$INTERNAL_COUNT" -ne 34 ] || [ "$EXTERNAL_COUNT" -ne 150 ]; then
    echo "[ERROR] Overlay counts do not match the study sets. Review the job log."
    exit 1
fi
