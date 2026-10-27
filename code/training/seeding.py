"""
================================================================================
SEEDING
================================================================================
DESCRIPTION:
    Single place to seed Python, NumPy, and PyTorch (CPU + all CUDA devices)
    for reproducibility. Call this before building the dataset/model so a
    given config + seed + git commit hash reproduces the same run.
================================================================================
"""

import os
import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
