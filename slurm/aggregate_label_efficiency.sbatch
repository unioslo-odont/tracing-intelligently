#!/bin/bash
#SBATCH --job-name=openceph_labeleff_agg
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=0-00:15:00
#SBATCH --output=logs/labeleff_agg_%A.out
#SBATCH --error=logs/labeleff_agg_%A.err

# ==============================================================================
# Aggregate the label-efficiency runs into outputs/label_efficiency/summary.csv
# and the two-panel poster figure. Reads the eval_val.json each training fold
# already writes, so no separate evaluation step is needed.
#
# NOTE: like slurm/compare_backbones.sbatch this needs no GPU (it reads JSON and
# draws a figure); it asks for accel + gpu:1 only because that is the known-good
# partition for this account. Switch to a CPU partition if your allocation has
# one.
#
# Submit from the REPO ROOT:
#   sbatch slurm/aggregate_label_efficiency.sbatch [fractions...]
# (slurm/run_label_efficiency.sh submits it dependency-gated after training.)
# ==============================================================================

FRACTIONS=("$@")
if [ ${#FRACTIONS[@]} -eq 0 ]; then
    FRACTIONS=(0.5 0.25)
fi

echo "=================================================="
echo "OpenCeph label-efficiency aggregation"
echo "Fractions: ${FRACTIONS[*]}"
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

source "${OPENCEPH_VENV:?Set OPENCEPH_VENV}/bin/activate"

REPO_ROOT="$SLURM_SUBMIT_DIR"
cd "$REPO_ROOT"

python "$REPO_ROOT/code/training/aggregate_label_efficiency.py" \
    --arms unet_baseline_15pt hrnet_pretrained_15pt \
    --exclude-landmarks iGo pGo \
    --baseline-root "${BASELINE_ROOT:-outputs/run10_15pt}" \
    --fractions "${FRACTIONS[@]}"

EXIT_CODE=$?
echo ""
echo "=================================================="
echo "Job Finished: $(date) | Exit code: $EXIT_CODE"
echo "=================================================="
exit $EXIT_CODE
