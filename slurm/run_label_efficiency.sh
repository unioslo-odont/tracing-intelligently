#!/bin/bash
# ==============================================================================
# RUN 8 - the whole label-efficiency curve in one command (Fox LOGIN node, not
# itself a SLURM job).
#
# Submits, for each arm x each fraction, a 5-fold job array, then one
# dependency-gated aggregation job that writes the summary table and the poster
# figure. Fraction 1.0 is Run 6 and is NOT re-run - the aggregator reads its
# existing outputs/<arm>/fold*/eval_val.json.
#
# Prerequisites (both cheap, both fail loudly if skipped):
#   1. the fold manifest exists and matches the data:
#        python code/training/make_fold_manifest.py --config configs/unet_baseline.yaml
#   2. the patient subsets exist:
#        python code/training/make_label_efficiency_subsets.py --fractions 0.5 0.25
#
# Usage:
#   cd /path/to/Tracing-intelligently
#   bash slurm/run_label_efficiency.sh                 # fractions 0.5 0.25
#   FRACTIONS="0.5 0.25 0.125" bash slurm/run_label_efficiency.sh
#
# Monitor: squeue -u $USER
# Results: outputs/label_efficiency/summary.csv + label_efficiency_curve.png
# ==============================================================================
set -o errexit

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

# The two arms of the curve: from-scratch vs the strongest pretrained arm, on the
# 15-point set. 100% (fraction 1.0) is Run 10 and is NOT re-run - the aggregator
# reads the existing baseline reports. Set BASELINE_ROOT=outputs for older
# ungrouped runs; new runs use outputs/run10_15pt. Headline scoring excludes iGo/pGo.
ARMS=("configs/unet_baseline_15pt.yaml" "configs/hrnet_pretrained_15pt.yaml")
FRACTIONS="${FRACTIONS:-0.5 0.25}"

for fraction in $FRACTIONS; do
    for fold in 0 1 2 3 4; do
        subset="configs/label_efficiency/fold${fold}_frac${fraction}.csv"
        if [ ! -f "$subset" ]; then
            echo "[ERROR] Missing $subset - run:"
            echo "        python code/training/make_label_efficiency_subsets.py --fractions $FRACTIONS"
            exit 1
        fi
    done
done

JOBS=()
for config in "${ARMS[@]}"; do
    arm=$(basename "$config" .yaml)
    for fraction in $FRACTIONS; do
        jid=$(sbatch --parsable --job-name="le_${arm}_${fraction}" \
                slurm/train_label_efficiency.sbatch "$config" "$fraction")
        echo "  ${arm}  frac ${fraction}  -> array job ${jid}"
        JOBS+=("$jid")
    done
done

DEP="afterok:$(IFS=:; echo "${JOBS[*]}")"
AGG=$(sbatch --parsable --dependency="$DEP" slurm/aggregate_label_efficiency.sbatch $FRACTIONS)
echo ""
echo "Submitted ${#JOBS[@]} arrays (${#ARMS[@]} arms x $(echo $FRACTIONS | wc -w) fractions x 5 folds)."
echo "Aggregation job ${AGG} runs after ${DEP}"
echo "Results: outputs/label_efficiency/summary.csv, label_efficiency_curve.png"
echo "Watch:   squeue -u \$USER"
