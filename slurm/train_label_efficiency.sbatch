#!/bin/bash
#SBATCH --job-name=openceph_labeleff
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --array=0-4                            # one k-fold fold per array task (0..k-1)
#SBATCH --output=logs/labeleff_%x_%A_fold%a.out
#SBATCH --error=logs/labeleff_%x_%A_fold%a.err

# ==============================================================================
# ONE POINT OF THE LABEL-EFFICIENCY CURVE (Run 8): train `arm` on `fraction` of
# each fold's TRAINING patients, evaluated on that fold's UNCHANGED validation
# set. One fold per array task.
#
#   sbatch slurm/train_label_efficiency.sbatch <base_config> <fraction>
#   e.g. sbatch slurm/train_label_efficiency.sbatch configs/unet_baseline.yaml 0.25
#        sbatch slurm/train_label_efficiency.sbatch configs/hrnet_pretrained.yaml 0.5
#
# The BASE CONFIG is the Run 6 config of that arm, used unchanged - the curve
# varies data quantity, not tuning, so the hyper-parameters must not drift. Only
# these things are overridden, all derived here rather than duplicated in a new
# config file (which could silently diverge from Run 6):
#   data.train_patient_subset_path  configs/label_efficiency/fold<k>_frac<f>.csv
#   training.checkpoint_dir         checkpoints/label_efficiency/<arm>_frac<f>
#   training.output_group           label_efficiency  -> outputs/label_efficiency/...
#   training.epochs                 base / fraction   (see EPOCH BUDGET below)
#   training.early_stopping_patience
# The full resolved config + git commit still land in run_meta.json next to the
# results, so provenance is unchanged.
#
# EPOCH BUDGET: at fraction f an epoch is f x as many optimizer steps, so the
# base epoch budget would give a 25% run a quarter of the training it needs, and
# a patience of 30 epochs would be a quarter of the patience. Both are therefore
# scaled by 1/f, which keeps the run terminating on EARLY STOPPING (the actual
# stopping rule) rather than on the epoch cap, and keeps "patience" comparable in
# steps. Total optimizer steps are logged and recorded in run_meta.json.
#
# SMOKE=1 runs the acceptance smoke test instead: 2 epochs, its own output group,
# so nothing lands in the real curve:
#   SMOKE=1 sbatch --array=0 slurm/train_label_efficiency.sbatch configs/unet_baseline.yaml 0.125
# ==============================================================================

CONFIG="$1"
FRACTION="$2"
if [ -z "$CONFIG" ] || [ -z "$FRACTION" ]; then
    echo "[ERROR] Usage: sbatch slurm/train_label_efficiency.sbatch <base_config> <fraction>"
    echo "        e.g.:  sbatch slurm/train_label_efficiency.sbatch configs/unet_baseline.yaml 0.25"
    exit 1
fi
shift 2
EXTRA=("$@")

FOLD="${SLURM_ARRAY_TASK_ID:-0}"

echo "=================================================="
echo "OpenCeph label-efficiency training - fold ${FOLD}, fraction ${FRACTION}"
echo "Base config: $CONFIG"
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

case "$CONFIG" in
    /*) FULL_CONFIG="$CONFIG" ;;
    *)  FULL_CONFIG="$REPO_ROOT/$CONFIG" ;;
esac

# Derive the arm name, the training script (from model.type) and the scaled
# epoch budget from the base config itself, so this launcher stays generic.
read -r ARM SCRIPT EPOCHS PATIENCE FRAC_TAG <<EOF
$(python - "$FULL_CONFIG" "$FRACTION" <<'PY'
import math, os, sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding='utf-8'))
fraction = float(sys.argv[2])
training = cfg['training']
arm = os.path.basename(os.path.normpath(training['checkpoint_dir']))
script = 'train.py' if cfg.get('model', {}).get('type', 'unet') == 'unet' else 'train_backbone_variant.py'
epochs = int(math.ceil(training['epochs'] / fraction))
patience = training.get('early_stopping_patience')
patience = 'null' if patience is None else int(math.ceil(patience / fraction))
print(arm, script, epochs, patience, f"{fraction:g}")
PY
)
EOF
if [ -z "$ARM" ]; then
    echo "[ERROR] Could not read the base config - is PyYAML available in $VENV_PATH?"
    exit 1
fi

SUBSET="$REPO_ROOT/configs/label_efficiency/fold${FOLD}_frac${FRAC_TAG}.csv"
if [ ! -f "$SUBSET" ]; then
    echo "[ERROR] Missing patient subset $SUBSET"
    echo "        Generate it first (login node, no GPU needed):"
    echo "        python code/training/make_label_efficiency_subsets.py --fractions ${FRAC_TAG}"
    exit 1
fi

GROUP="label_efficiency"
if [ "${SMOKE:-0}" = "1" ]; then
    GROUP="smoke_test"
    EPOCHS=2
    PATIENCE="null"
    echo "[SMOKE TEST] 2 epochs, results under outputs/smoke_test/ - not part of the curve"
fi

echo "Arm=$ARM  script=$SCRIPT  fraction=$FRAC_TAG  epochs=$EPOCHS  patience=$PATIENCE"
echo "Subset=$SUBSET"

python "$REPO_ROOT/code/training/$SCRIPT" \
    --config "$FULL_CONFIG" \
    --fold "${FOLD}" \
    --set data.train_patient_subset_path="$SUBSET" \
    --set training.checkpoint_dir="./checkpoints/${GROUP}/${ARM}_frac${FRAC_TAG}" \
    --set training.output_group="${GROUP}" \
    --set training.epochs="${EPOCHS}" \
    --set training.early_stopping_patience="${PATIENCE}" \
    "${EXTRA[@]}"

EXIT_CODE=$?
echo ""
echo "=================================================="
echo "Job Finished: $(date) | Exit code: $EXIT_CODE"
echo "Results: outputs/${GROUP}/${ARM}_frac${FRAC_TAG}/fold${FOLD}/"
echo "=================================================="
exit $EXIT_CODE
