#!/bin/bash
# ==============================================================================
# ONE-TIME environment setup on a Fox login node (interactive, not a SLURM job).
#
# Creates a project-dedicated venv under /path/to/project/python_envs/,
# pip-installing torch directly from requirements.txt - mirroring the
# already-proven-working setup at /path/to/example/venv
# (torch 2.11.0+cu130, confirmed working on this cluster's GPU nodes),
# rather than layering on the older EasyBuild PyTorch module
# (PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 is the newest available there, from
# 2023 - fine functionally for this U-Net baseline, but there's no reason to
# prefer it over a recent pip build that's already known to work here).
#
# Usage:
#   module purge
#   module load Python/3.11.5-GCCcore-13.2.0
#   module load CUDA/12.6.0
#   bash slurm/setup_env.sh
# ==============================================================================
set -o errexit

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${OPENCEPH_VENV:?Set OPENCEPH_VENV to the new environment path}"

# pip's wheel cache is one of the biggest home-directory writers (torch alone is
# ~2.5 GB of wheels) - redirect it, and every other cache, onto /path/to/project.
source "$REPO_ROOT/slurm/cache_env.sh"

mkdir -p "$(dirname "$ENV_DIR")"
python3 -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"
pip install --upgrade pip
pip install -r "$REPO_ROOT/requirements.txt"

echo ""
echo "Installed torch:"
python -c "import torch; print(' ', torch.__version__, '- built for CUDA', torch.version.cuda)"

echo ""
echo "Environment ready at $ENV_DIR"
echo "IMPORTANT: this was set up on a login node, so torch.cuda.is_available()"
echo "will report False here regardless - that's expected. Confirm it reports"
echo "True on an actual GPU node (e.g. via a short test job, or:"
echo "  srun --account=YOUR_ACCOUNT --partition=accel --gres=gpu:1 --time=00:05:00 \\"
echo "       --pty bash -c 'source $ENV_DIR/bin/activate && python -c \"import torch; print(torch.cuda.is_available())\"'"
echo ") before relying on it for a real training run."
echo ""
echo "Once confirmed, freeze it for reproducibility:"
echo "  pip freeze > requirements.lock.txt"
