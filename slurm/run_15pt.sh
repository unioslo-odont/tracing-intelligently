#!/bin/bash
# ==============================================================================
# RUN 10 - 15-point hard-tissue architecture comparison (4 arms x 5 folds).
# Login-node launcher (NOT itself a SLURM job). Mirrors slurm/run_all.sh but uses
# the *_15pt.yaml configs, which write to checkpoints/<model>_15pt/ and
# outputs/run10_15pt/ - Run 6 (26-point) is never touched.
#
# Same fold split as Run 6 (identical patients/seed/exclusions), asserted against
# the SAME manifest configs/folds/oslo26_k5_seed0.csv. Generate that manifest once
# first (see below) or this will fail loudly rather than train a different split.
#
# Usage:
#   cd /path/to/Tracing-intelligently
#   bash slurm/run_15pt.sh
#
# Monitor:  squeue -u $USER
# Results:  outputs/run10_15pt/<model>_15pt/fold<N>/eval_{train,val}.json
#           outputs/run10_15pt/comparison_15pt_*.csv
# ==============================================================================
set -o errexit

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
if [ ! -f code/training/train.py ]; then
    echo "[ERROR] Run from the repo root - code/training/train.py not found."
    exit 1
fi
mkdir -p logs

MANIFEST="configs/folds/oslo26_k5_seed0.csv"
if [ ! -f "$MANIFEST" ]; then
    echo "[ERROR] Missing $MANIFEST - the patient->fold mapping is asserted, not regenerated."
    echo "        Generate and retain it privately ONCE (it captures the current = Run 6 split):"
    echo "          python code/training/make_fold_manifest.py --config configs/unet_baseline.yaml"
    exit 1
fi

echo "Submitting Run 10 (15-point) training arrays from $REPO_ROOT ..."
# U-Net arm goes through train.py (train_generic.sbatch = train.py + arbitrary config);
# the three backbone arms go through train_backbone_variant.py (train_variant.sbatch).
J_UNET=$(sbatch --parsable slurm/train_generic.sbatch configs/unet_baseline_15pt.yaml)
echo "  unet_baseline_15pt     -> array job $J_UNET"
J_RESNET=$(sbatch --parsable slurm/train_variant.sbatch configs/resnet_unet_15pt.yaml)
echo "  resnet_unet_15pt       -> array job $J_RESNET"
J_HRNET=$(sbatch --parsable slurm/train_variant.sbatch configs/hrnet_pretrained_15pt.yaml)
echo "  hrnet_pretrained_15pt  -> array job $J_HRNET"
J_DINO=$(sbatch --parsable slurm/train_variant.sbatch configs/dinov2_frozen_15pt.yaml)
echo "  dinov2_frozen_15pt     -> array job $J_DINO"

# Gated cross-arm comparison over the 15-point checkpoints, once all four arrays finish.
DEP="afterok:${J_UNET}:${J_RESNET}:${J_HRNET}:${J_DINO}"
J_CMP=$(sbatch --parsable --dependency="$DEP" slurm/compare_backbones.sbatch \
    outputs/run10_15pt/unet_baseline_15pt outputs/run10_15pt/resnet_unet_15pt \
    outputs/run10_15pt/hrnet_pretrained_15pt outputs/run10_15pt/dinov2_frozen_15pt \
    --exclude-landmarks iGo pGo \
    --output outputs/run10_15pt/comparison_15pt)
echo "  comparison (15pt)      -> job $J_CMP  (runs after $DEP)"

echo ""
echo "All Run 10 jobs submitted. Watch:  squeue -u \$USER"
echo "Headline tables score 14 direct landmarks; 16 channels are trained. Derived Gonion is not evaluated."
