"""
================================================================================
PREFETCH PRETRAINED WEIGHTS - run on a LOGIN node before submitting jobs
================================================================================
DESCRIPTION:
    Fox compute nodes have no internet access, so a job that needs to download a
    pretrained checkpoint fails minutes into the queue - after the wait, and
    usually on all five folds at once. This downloads into the SHARED cache
    (HF_HOME=/path/to/project/hf_cache, which is also where the sbatch scripts
    point) so every later job finds it locally.

    Run it once per new backbone: any new HuggingFace model_name (the DINOv2
    register variant, dinov2-base, ...) or any new timm encoder_name.

USAGE:
    python code/training/prefetch_weights.py --config configs/dinov2_finetune.yaml
    python code/training/prefetch_weights.py --model facebook/dinov2-with-registers-small
    python code/training/prefetch_weights.py --timm hrnet_w32
================================================================================
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache_paths import ensure_project_caches
from config import load_config


def prefetch_hf(model_name):
    from transformers import AutoModel
    print(f"  downloading HuggingFace model '{model_name}' ...")
    model = AutoModel.from_pretrained(model_name)
    config = model.config
    print(f"  OK: {sum(p.numel() for p in model.parameters()):,} params, "
          f"patch_size={getattr(config, 'patch_size', '?')}, "
          f"hidden={getattr(config, 'hidden_size', '?')}, "
          f"blocks={getattr(config, 'num_hidden_layers', '?')}")


def prefetch_timm(encoder_name):
    import timm
    print(f"  downloading timm encoder '{encoder_name}' ...")
    model = timm.create_model(encoder_name, pretrained=True, in_chans=1, features_only=True)
    print(f"  OK: feature strides {model.feature_info.reduction()}, "
          f"channels {model.feature_info.channels()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', action='append', default=[],
                        help="config(s) whose model.model_name / model.encoder_name to fetch "
                             "(repeatable)")
    parser.add_argument('--model', action='append', default=[],
                        help="explicit HuggingFace model id (repeatable)")
    parser.add_argument('--timm', action='append', default=[],
                        help="explicit timm encoder name (repeatable)")
    args = parser.parse_args()

    # Never let a download land in the home directory (see cache_paths.py).
    ensure_project_caches()
    print(f"HF_HOME = {os.environ.get('HF_HOME', '(default ~/.cache - check your home quota!)')}")

    hf_models, timm_models = list(args.model), list(args.timm)
    for config_path in args.config:
        model_cfg = (load_config(config_path) or {}).get('model', {})
        model_type = model_cfg.get('type', 'unet')
        if model_type == 'dinov2':
            hf_models.append(model_cfg.get('model_name', 'facebook/dinov2-small'))
        elif model_type == 'pretrained_encoder' and model_cfg.get('pretrained', True):
            timm_models.append(model_cfg.get('encoder_name', 'resnet34'))
        else:
            print(f"{config_path}: model.type '{model_type}' needs no pretrained weights")

    if not hf_models and not timm_models:
        print("Nothing to fetch - pass --config, --model or --timm.")
        return

    failures = []
    for name in dict.fromkeys(hf_models):
        try:
            prefetch_hf(name)
        except Exception as e:
            failures.append((name, e))
            print(f"  [FAIL] {name}: {e}")
    for name in dict.fromkeys(timm_models):
        try:
            prefetch_timm(name)
        except Exception as e:
            failures.append((name, e))
            print(f"  [FAIL] {name}: {e}")

    if failures:
        print(f"\n{len(failures)} download(s) FAILED - fix these before submitting jobs; a compute "
              "node cannot download them.")
        sys.exit(1)
    print("\nAll weights are in the shared cache. Compute nodes will load them offline.")


if __name__ == "__main__":
    main()
