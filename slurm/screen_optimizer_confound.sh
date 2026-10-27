#!/bin/bash
# ==============================================================================
# TASK 2 FOLD-0 SCREEN - encoder learning rate for the pretrained CNN arms
# (run on a Fox LOGIN node, not itself a SLURM job).
#
# Runs 1-6 trained the pretrained encoder and the random decoder at ONE flat LR
# with no warmup, which is why Run 6's table is a practical finding rather than a
# ranking. This screen picks the encoder LR per arm; the winner then goes into
# configs/<arm>_run7.yaml for the full 5-fold Run 7.
#
# Grid: decoder_lr fixed at 3e-4, warmup 500 steps, encoder_lr in
#       {1e-5, 3e-5, 1e-4}  x  {resnet_unet, hrnet_pretrained}  = 6 runs,
# FOLD 0 ONLY - screening never touches the other folds, and no screening number
# is ever quoted: the poster/paper number comes from the full 5-fold Run 7.
#
# Usage:
#   cd /path/to/Tracing-intelligently
#   bash slurm/screen_optimizer_confound.sh
#
# Then read the six val MREs and set training.optimizer.encoder_lr in
# configs/resnet_unet_run7.yaml and configs/hrnet_pretrained_run7.yaml:
#   python code/training/compare_backbones.py --checkpoint-dirs checkpoints/screen/*
#   (or just: grep -h '"mre_mm"' outputs/screen/*/fold0/eval_val.json)
# ==============================================================================
set -o errexit

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

ENCODER_LRS=("1.0e-5" "3.0e-5" "1.0e-4")
CONFIGS=("configs/resnet_unet_optim_15pt.yaml" "configs/hrnet_pretrained_optim_15pt.yaml")

for config in "${CONFIGS[@]}"; do
    arm=$(basename "$config" _optim_15pt.yaml)
    for lr in "${ENCODER_LRS[@]}"; do
        tag="${arm}_enc${lr}"
        jid=$(sbatch --parsable --array=0 --job-name="scr_${tag}" \
                slurm/train_variant.sbatch "$config" \
                --set training.optimizer.encoder_lr="$lr" \
                --set training.checkpoint_dir="./checkpoints/screen_15pt/${tag}" \
                --set training.output_group=screen_15pt)
        echo "  ${tag}  -> job ${jid}  (fold 0 only)"
    done
done

echo ""
echo "6 fold-0 screening runs submitted. When they finish:"
echo "  for f in outputs/screen_15pt/*/fold0/eval_val.json; do"
echo "      printf '%-40s %s\\n' \"\$f\" \"\$(python -c \"import json,sys;print(json.load(open(sys.argv[1]))['overall']['mre_mm'])\" \"\$f\")\";"
echo "  done"
echo "Pick the best encoder_lr PER ARM, write it into configs/<arm>_optim_15pt.yaml, then run the full 5-fold:"
echo "  sbatch slurm/train_variant.sbatch configs/resnet_unet_optim_15pt.yaml"
echo "  sbatch slurm/train_variant.sbatch configs/hrnet_pretrained_optim_15pt.yaml"
