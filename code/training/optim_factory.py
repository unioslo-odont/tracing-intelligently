"""
================================================================================
OPTIMIZER FACTORY - parameter groups, warmup, layer-wise LR decay
================================================================================
DESCRIPTION:
    Runs 1-6 gave every parameter of every arm one flat learning rate and no
    warmup. For the from-scratch U-Net that is the right thing. For the
    PRETRAINED arms it is a methodological confound: a pretrained encoder and a
    randomly-initialised decoder want different learning rates (and the encoder
    wants a gentle start, or the first large gradients from the random decoder
    destroy the features that were the entire reason for pretraining). A flat
    LR therefore systematically disadvantages the pretrained arms, which is why
    Run 6's architecture table was labelled a practical finding rather than a
    ranking.

    This module makes that treatment config-driven and explicit:
      - {encoder, decoder} parameter groups with separate LRs and weight decay,
      - linear LR warmup counted in optimizer STEPS (not epochs, so it means
        the same thing at every training-set size - which matters for the
        label-efficiency curve, where an epoch can be 4x shorter),
      - layer-wise LR decay across ViT blocks, for the fine-tuned DINOv2 arm
        (deep blocks hold general features and want smaller updates than the
        task-specific top).

    STRICTLY ADDITIVE: with no `training.optimizer` section, this returns
    exactly what Runs 1-6 built - one Adam over all trainable parameters at
    training.lr / training.weight_decay - so old configs reproduce bit for bit.

CONFIG (all keys optional):
    training:
      optimizer:
        type: adam | adamw        # default adam (Runs 1-6 behaviour)
        decoder_lr: 3.0e-4        # default: training.lr
        encoder_lr: 3.0e-5        # pretrained encoder/backbone; default: decoder_lr
        decoder_weight_decay:     # default: training.weight_decay
        encoder_weight_decay:     # default: training.weight_decay
        warmup_steps: 500         # linear 0 -> group LR over the first N steps
        layerwise_decay: 0.75     # ViT blocks only: block i gets encoder_lr * decay^(top - i)
================================================================================
"""

import re

import torch

# Attribute names that mark the PRETRAINED half of a model: `encoder` on
# pretrained_encoder_unet.PretrainedEncoderUNetHeatmapModel, `backbone` on
# dinov2_head.DINOv2HeatmapModel. Everything else (decoder blocks, head) is
# randomly initialised and belongs in the decoder group.
ENCODER_PREFIXES = ('encoder.', 'backbone.')

# transformers' Dinov2Model block naming, e.g. backbone.encoder.layer.7.attention...
VIT_BLOCK_RE = re.compile(r'(?:^|\.)(?:encoder\.)?layer\.(\d+)\.')


def _is_encoder_param(name):
    return name.startswith(ENCODER_PREFIXES)


def _vit_block_index(name):
    match = VIT_BLOCK_RE.search(name)
    return int(match.group(1)) if match else None


def build_param_groups(model, optimizer_cfg, default_lr, default_weight_decay):
    """Return a list of {name, params, lr, weight_decay} dicts.

    One group when no encoder/decoder split is configured; two when encoder_lr
    differs; one per ViT block (plus embeddings) when layerwise_decay is set."""
    decoder_lr = float(optimizer_cfg.get('decoder_lr', default_lr))
    encoder_lr = float(optimizer_cfg.get('encoder_lr', decoder_lr))
    decoder_wd = float(optimizer_cfg.get('decoder_weight_decay', default_weight_decay))
    encoder_wd = float(optimizer_cfg.get('encoder_weight_decay', default_weight_decay))
    layerwise_decay = optimizer_cfg.get('layerwise_decay')

    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    # No differentiated treatment requested -> ONE group over everything, exactly
    # as Runs 1-6 built it. Splitting into two groups that happen to carry the
    # same LR would behave identically, but this keeps the legacy path
    # structurally identical too (one group in the checkpoint, in the logs, and
    # in history.json), so "old configs are unchanged" is literally true.
    if not any(key in optimizer_cfg for key in ('encoder_lr', 'encoder_weight_decay', 'layerwise_decay')):
        return [{'name': 'all', 'params': [p for _, p in named],
                 'lr': decoder_lr, 'weight_decay': decoder_wd}]

    encoder_named = [(n, p) for n, p in named if _is_encoder_param(n)]
    decoder_named = [(n, p) for n, p in named if not _is_encoder_param(n)]

    groups = []
    if decoder_named:
        groups.append({'name': 'decoder', 'params': [p for _, p in decoder_named],
                       'lr': decoder_lr, 'weight_decay': decoder_wd})

    if not encoder_named:
        return groups

    if not layerwise_decay:
        groups.append({'name': 'encoder', 'params': [p for _, p in encoder_named],
                       'lr': encoder_lr, 'weight_decay': encoder_wd})
        return groups

    # Layer-wise decay: the TOP block keeps encoder_lr and each block below it
    # is scaled by decay once more; anything before the first block (patch
    # embeddings, cls/register tokens) sits one step below the lowest block.
    layerwise_decay = float(layerwise_decay)
    block_indices = [i for i in (_vit_block_index(n) for n, _ in encoder_named) if i is not None]
    if not block_indices:
        raise ValueError(
            "training.optimizer.layerwise_decay is set but no transformer blocks were found in the "
            "encoder parameter names - layer-wise decay only applies to a ViT backbone (see "
            "code/models/dinov2_head.py). Remove the key for CNN encoders."
        )
    top_block = max(block_indices)

    by_depth = {}
    for name, param in encoder_named:
        index = _vit_block_index(name)
        if index is not None:
            depth = index
        elif 'embeddings' in name:
            depth = -1                      # patch embedding / cls + register tokens
        else:
            depth = top_block               # post-block norms (backbone.layernorm.*)
        by_depth.setdefault(depth, []).append(param)

    for depth in sorted(by_depth, reverse=True):
        scale = layerwise_decay ** (top_block - depth)
        label = f"encoder.block{depth}" if depth >= 0 else "encoder.embeddings"
        groups.append({'name': label, 'params': by_depth[depth],
                       'lr': encoder_lr * scale, 'weight_decay': encoder_wd})
    return groups


def build_optimizer(model, training_cfg):
    """Build the optimizer for a run. Returns (optimizer, groups_summary).

    groups_summary is a list of {name, n_params, n_tensors, lr, weight_decay}
    for logging/provenance - the LR each group actually starts at, which is the
    sanity check that the confound fix is doing what the config says."""
    optimizer_cfg = training_cfg.get('optimizer') or {}
    default_lr = float(training_cfg['lr'])
    default_wd = float(training_cfg.get('weight_decay', 0.0))

    groups = build_param_groups(model, optimizer_cfg, default_lr, default_wd)
    if not groups:
        raise ValueError("Model has no trainable parameters")

    optimizer_type = str(optimizer_cfg.get('type', 'adam')).lower()
    optimizer_class = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW}.get(optimizer_type)
    if optimizer_class is None:
        raise ValueError(f"Unknown training.optimizer.type '{optimizer_type}' (expected adam or adamw)")

    optimizer = optimizer_class(
        [{'params': g['params'], 'lr': g['lr'], 'weight_decay': g['weight_decay']} for g in groups])
    # base_lr is the value warmup/scheduling scales; group['lr'] is the CURRENT value.
    for group, spec in zip(optimizer.param_groups, groups):
        group['base_lr'] = spec['lr']
        group['name'] = spec['name']

    summary = [{'name': g['name'], 'n_tensors': len(g['params']),
                'n_params': int(sum(p.numel() for p in g['params'])),
                'lr': g['lr'], 'weight_decay': g['weight_decay']} for g in groups]
    return optimizer, summary


def warmup_scale(step, warmup_steps):
    """Linear ramp over the first `warmup_steps` optimizer steps; 1.0 after.
    `step` is 0-based, so the first step already gets a non-zero LR."""
    if not warmup_steps or warmup_steps <= 0:
        return 1.0
    return min(1.0, (step + 1) / float(warmup_steps))


def apply_lr_scale(optimizer, scale):
    """Set every group's LR to its base_lr * scale (groups keep their ratios)."""
    for group in optimizer.param_groups:
        group['lr'] = group.get('base_lr', group['lr']) * scale


def format_group_summary(summary, scale=1.0):
    return " | ".join(f"{g['name']}: lr={g['lr'] * scale:.3g} wd={g['weight_decay']:.3g} "
                      f"({g['n_params']:,} params)" for g in summary)
