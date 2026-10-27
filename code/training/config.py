"""
================================================================================
CONFIG LOADING
================================================================================
DESCRIPTION:
    Minimal YAML config loader. A flat, plain dict-of-dicts config (no Hydra
    composition/inheritance) - see configs/unet_baseline.yaml for the
    expected schema. `--set a.b.c=value` overrides on the train.py/
    evaluate.py CLI let you sweep individual values without editing the YAML
    or losing the config-as-source-of-truth property (the resolved config is
    always saved into the checkpoint - see train.py).

    SITE SETTINGS: tracked configs carry placeholder data paths. If
    local/site.yaml exists (git-ignored; template configs/site.example.yaml),
    or OPENCEPH_SITE_CONFIG names another file, it is deep-merged over every
    config load_config reads, so one private file holds this machine's data
    locations for every arm. Keep it to site-specific keys. Precedence:
    config YAML < site file < --set. The merged values are what the checkpoint
    and run_meta.json record.
================================================================================
"""

import os

import yaml

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
DEFAULT_SITE_CONFIG = os.path.join(REPO_ROOT, 'local', 'site.yaml')


def _deep_merge(base, overlay, prefix=''):
    """Merge `overlay` into `base` in place; returns the dotted keys it set."""
    changed = []
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            changed += _deep_merge(base[key], value, f'{prefix}{key}.')
        else:
            base[key] = value
            changed.append(f'{prefix}{key}')
    return changed


def apply_site_config(config):
    """Overlay the site file onto `config`. A missing default file is fine (a
    public clone runs on its own configs); a missing OPENCEPH_SITE_CONFIG is not."""
    site_path = os.environ.get('OPENCEPH_SITE_CONFIG')
    if site_path:
        if not os.path.isfile(site_path):
            raise FileNotFoundError(f"OPENCEPH_SITE_CONFIG={site_path} does not exist")
    elif os.path.isfile(DEFAULT_SITE_CONFIG):
        site_path = DEFAULT_SITE_CONFIG
    else:
        return config
    with open(site_path, 'r', encoding='utf-8') as f:
        site = yaml.safe_load(f) or {}
    changed = _deep_merge(config, site)
    print(f"[config] site settings from {site_path}: {', '.join(changed) or 'none'}")
    return config


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}
    return apply_site_config(config)


def apply_overrides(config, overrides):
    """overrides: list of 'a.b.c=value' strings (from --set, repeatable).
    Values are parsed with yaml.safe_load so '0.001' -> float, 'true' -> bool,
    'null' -> None, etc., matching normal YAML scalar semantics."""
    for item in overrides:
        key_path, sep, raw_value = item.partition('=')
        if not sep:
            raise ValueError(f"--set override '{item}' must be of the form a.b.c=value")
        value = yaml.safe_load(raw_value)
        keys = key_path.split('.')
        node = config
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = value
    return config
