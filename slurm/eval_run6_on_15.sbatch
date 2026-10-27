#!/bin/bash
#SBATCH --job-name=openceph_run6_on_15
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-01:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/run6_on_15_%A_fold%a.out
#SBATCH --error=logs/run6_on_15_%A_fold%a.err

# ==============================================================================
# TASK 2 - restricted re-evaluation. Score the EXISTING Run 6 (26-channel)
# checkpoints on their validation folds but over ONLY the 15-point subset of
# channels - NO retraining, NO refitting. One array task per fold, all four arms.
#
# This isolates the EVALUATION effect of the reduced set (dropping the hard
# landmarks mechanically lowers the mean) from the TRAINING effect (Run 10,
# fewer output channels). Task 3 (Run 10) minus this = the training effect.
#
# Requires Run 6 checkpoints at checkpoints/<model>/fold<N>/best.pt.
# Writes outputs/run10_15pt/task2_run6_scored_on_15/<model>/fold<N>/eval_val_<subset>.*
#
# Submit from the REPO ROOT:
#   sbatch slurm/eval_run6_on_15.sbatch
# ==============================================================================

# The 16 trained channels of the 15-point set (15 clinical points; Go is derived
# from iGo+pGo). Keep in sync with configs/landmarks_15.yaml.
LM15="S N A B Po Or Gn Pg Sp Pt Is Isa Ii Iia iGo pGo"

FOLD="${SLURM_ARRAY_TASK_ID}"
echo "=================================================="
echo "Task 2: Run 6 checkpoints scored on the 15-point subset - fold ${FOLD}"
echo "Landmarks: ${LM15}"
echo "Job Started: $(date) | Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST"
echo "=================================================="

module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.6.0
export PYTHONUNBUFFERED=1
source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/cache_env.sh"

VENV_PATH="${OPENCEPH_VENV:?Set OPENCEPH_VENV to the existing HPC environment}"
source "$VENV_PATH/bin/activate"
REPO_ROOT="$SLURM_SUBMIT_DIR"; cd "$REPO_ROOT"

# model_name : eval script (unet uses evaluate.py; backbone arms use evaluate_backbone_variant.py)
declare -A EVAL_SCRIPT=(
    [unet_baseline]=evaluate.py
    [resnet_unet]=evaluate_backbone_variant.py
    [hrnet_pretrained]=evaluate_backbone_variant.py
    [dinov2_frozen]=evaluate_backbone_variant.py
)

OVERALL_EXIT=0
for MODEL in unet_baseline resnet_unet hrnet_pretrained dinov2_frozen; do
    CKPT="$REPO_ROOT/checkpoints/${MODEL}/fold${FOLD}/best.pt"
    if [ ! -f "$CKPT" ]; then
        echo "[WARN] skipping ${MODEL}: missing $CKPT"
        continue
    fi
    OUT="$REPO_ROOT/outputs/run10_15pt/task2_run6_scored_on_15/${MODEL}/fold${FOLD}/eval_val_15pt"
    mkdir -p "$(dirname "$OUT")"
    echo ""
    echo "--- ${MODEL} fold ${FOLD} (Run 6 ckpt, scored on 15) ---"
    python "$REPO_ROOT/code/training/${EVAL_SCRIPT[$MODEL]}" \
        --checkpoint "$CKPT" --split val --landmarks $LM15 --output "$OUT"
    [ $? -ne 0 ] && OVERALL_EXIT=1
done

echo ""
echo "=================================================="
echo "Job Finished: $(date) | Exit code: $OVERALL_EXIT"
echo "=================================================="
exit $OVERALL_EXIT
