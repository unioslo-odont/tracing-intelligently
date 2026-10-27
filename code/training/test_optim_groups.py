"""
================================================================================
TESTS (need torch - run on Fox) - optimizer groups, warmup, layer-wise decay
================================================================================
DESCRIPTION:
    The Run 7 confound fix and the Run 9 fine-tuned DINOv2 arm both hinge on
    parameters actually landing in the right group at the right learning rate.
    That is invisible in a loss curve, so it is asserted here.

    The most important test is the FIRST one: a config with no
    `training.optimizer` section must produce exactly what Runs 1-6 produced -
    one Adam group at training.lr. If that ever fails, the new code has silently
    changed the baseline every result in the project is measured against.

    Run it in the project venv on a compute or login node:
        python code/training/test_optim_groups.py
    Tests that need pretrained weights are SKIPPED (not failed) when the weights
    aren't cached - see prefetch_weights.py.
================================================================================
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# BEFORE torch/transformers are imported: test [5] can trigger a DINOv2
# download, which must not land in the home directory (see cache_paths.py).
from cache_paths import ensure_project_caches
ensure_project_caches()

import torch
import torch.nn as nn

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'datasets')))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')))

from optim_factory import build_optimizer, warmup_scale, apply_lr_scale, build_param_groups

PASSED, FAILED, SKIPPED = [], [], []


def check(name, condition, detail=''):
    (PASSED if condition else FAILED).append(name)
    print(f"  {'PASS' if condition else 'FAIL'}  {name}{(' - ' + detail) if detail and not condition else ''}")


def skip(name, why):
    SKIPPED.append(name)
    print(f"  SKIP  {name} - {why}")


class FakeCNNModel(nn.Module):
    """Shaped like pretrained_encoder_unet: an `encoder` plus decoder + head."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.Conv2d(4, 4, 3, padding=1))
        self.dec2 = nn.Conv2d(4, 4, 3, padding=1)
        self.head = nn.Conv2d(4, 2, 1)

    def forward(self, x):
        return self.head(self.dec2(self.encoder(x)))


class FakeViTModel(nn.Module):
    """Shaped like dinov2_head: `backbone.embeddings` + `backbone.encoder.layer.N`."""

    def __init__(self, n_blocks=12):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.embeddings = nn.Linear(4, 4)
        self.backbone.encoder = nn.Module()
        self.backbone.encoder.layer = nn.ModuleList([nn.Linear(4, 4) for _ in range(n_blocks)])
        self.backbone.layernorm = nn.LayerNorm(4)
        self.proj = nn.Linear(4, 4)
        self.head = nn.Linear(4, 2)


def group_by_name(summary):
    return {g['name']: g for g in summary}


# --------------------------------------------------- 1. the untouched baseline
def test_backward_compatibility():
    print("\n[1] configs without training.optimizer are UNCHANGED")
    model = FakeCNNModel()
    training = {'lr': 0.0001, 'weight_decay': 0.0001}
    optimizer, summary = build_optimizer(model, training)

    check("one parameter group (as in Runs 1-6)", len(optimizer.param_groups) == 1,
          str(len(optimizer.param_groups)))
    check("that group covers everything, undivided", summary[0]['name'] == 'all',
          summary[0]['name'])
    check("group LR is training.lr", optimizer.param_groups[0]['lr'] == 0.0001)
    check("group weight decay is training.weight_decay",
          optimizer.param_groups[0]['weight_decay'] == 0.0001)
    check("optimizer is Adam by default", isinstance(optimizer, torch.optim.Adam)
          and not isinstance(optimizer, torch.optim.AdamW))
    check("every trainable parameter is in the optimizer",
          sum(p.numel() for g in optimizer.param_groups for p in g['params'])
          == sum(p.numel() for p in model.parameters() if p.requires_grad))
    check("no warmup means no LR scaling", warmup_scale(0, 0) == 1.0 and warmup_scale(99, None) == 1.0)


# ------------------------------------------------ 2. the encoder/decoder split
def test_encoder_decoder_split():
    print("\n[2] encoder/decoder parameter groups (Run 7)")
    model = FakeCNNModel()
    training = {'lr': 0.0001, 'weight_decay': 0.0001,
                'optimizer': {'decoder_lr': 3.0e-4, 'encoder_lr': 3.0e-5, 'warmup_steps': 500}}
    optimizer, summary = build_optimizer(model, training)
    groups = group_by_name(summary)

    check("two groups: encoder and decoder", set(groups) == {'encoder', 'decoder'}, str(set(groups)))
    check("decoder gets decoder_lr", groups['decoder']['lr'] == 3.0e-4)
    check("encoder gets encoder_lr", groups['encoder']['lr'] == 3.0e-5)

    encoder_params = sum(p.numel() for n, p in model.named_parameters() if n.startswith('encoder.'))
    check("encoder group holds exactly the encoder parameters",
          groups['encoder']['n_params'] == encoder_params)
    check("decoder group holds everything else",
          groups['decoder']['n_params'] == sum(p.numel() for p in model.parameters()) - encoder_params)

    # Warmup must scale BOTH groups and preserve their ratio.
    apply_lr_scale(optimizer, warmup_scale(0, 500))
    lrs_first = {g['name']: g['lr'] for g in optimizer.param_groups}
    check("step 0 LR is base/warmup_steps", abs(lrs_first['encoder'] - 3.0e-5 / 500) < 1e-12)
    check("warmup preserves the encoder:decoder ratio",
          abs(lrs_first['decoder'] / lrs_first['encoder'] - 10.0) < 1e-9)
    apply_lr_scale(optimizer, warmup_scale(499, 500))
    check("LR reaches the configured value at the end of warmup",
          abs(optimizer.param_groups[0]['lr'] - 3.0e-4) < 1e-12)
    apply_lr_scale(optimizer, warmup_scale(5000, 500))
    check("LR stays at the configured value after warmup",
          abs(optimizer.param_groups[0]['lr'] - 3.0e-4) < 1e-12)

    check("adamw is selectable",
          isinstance(build_optimizer(model, {'lr': 1e-4, 'optimizer': {'type': 'adamw'}})[0],
                     torch.optim.AdamW))
    check("an unknown optimizer type raises",
          _raises(build_optimizer, model, {'lr': 1e-4, 'optimizer': {'type': 'lbfgs'}}))


def _raises(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
        return False
    except Exception:
        return True


# -------------------------------------------------- 3. ViT layer-wise LR decay
def test_layerwise_decay():
    print("\n[3] ViT layer-wise LR decay (Run 9)")
    model = FakeViTModel(n_blocks=12)
    training = {'lr': 3.0e-4, 'weight_decay': 0.0001,
                'optimizer': {'type': 'adamw', 'decoder_lr': 3.0e-4, 'encoder_lr': 5.0e-5,
                              'layerwise_decay': 0.75, 'encoder_weight_decay': 0.05}}
    optimizer, summary = build_optimizer(model, training)
    groups = group_by_name(summary)

    check("one group per block plus embeddings plus the decoder",
          len(groups) == 12 + 1 + 1, str(len(groups)))
    check("decoder keeps decoder_lr", abs(groups['decoder']['lr'] - 3.0e-4) < 1e-12)
    check("top block gets encoder_lr", abs(groups['encoder.block11']['lr'] - 5.0e-5) < 1e-12)
    check("each block below is scaled by the decay once more",
          abs(groups['encoder.block10']['lr'] - 5.0e-5 * 0.75) < 1e-15
          and abs(groups['encoder.block0']['lr'] - 5.0e-5 * 0.75 ** 11) < 1e-15)
    check("embeddings get the smallest LR of all",
          groups['encoder.embeddings']['lr'] < groups['encoder.block0']['lr'])
    check("LR is monotonically increasing with depth",
          all(groups[f'encoder.block{i}']['lr'] < groups[f'encoder.block{i + 1}']['lr']
              for i in range(11)))
    check("backbone weight decay is the encoder one",
          groups['encoder.block5']['weight_decay'] == 0.05
          and groups['decoder']['weight_decay'] == 0.0001)
    check("the post-block layernorm rides with the top block",
          groups['encoder.block11']['n_params']
          > sum(p.numel() for p in model.backbone.encoder.layer[11].parameters()))

    check("layerwise_decay on a CNN encoder raises rather than silently doing nothing",
          _raises(build_param_groups, FakeCNNModel(),
                  {'encoder_lr': 1e-5, 'layerwise_decay': 0.75}, 3e-4, 0.0))

    frozen = FakeViTModel(n_blocks=12)
    for p in frozen.backbone.parameters():
        p.requires_grad_(False)
    _, frozen_summary = build_optimizer(frozen, {'lr': 5e-4, 'optimizer': {'encoder_lr': 1e-5}})
    check("a frozen backbone contributes no optimizer group",
          {g['name'] for g in frozen_summary} == {'decoder'},
          str({g['name'] for g in frozen_summary}))


# --------------------------------------------- 4. gradient accumulation is a no-op at 1
def test_grad_accumulation():
    print("\n[4] gradient accumulation")
    try:
        from train import run_epoch
    except Exception as e:
        skip("gradient accumulation", f"could not import train.py ({e})")
        return

    def make_batches(n, batch_size):
        torch.manual_seed(0)
        images = torch.randn(n, 1, 8, 8)
        heatmaps = torch.randn(n, 2, 8, 8)
        return [{'image': images[i:i + batch_size], 'heatmaps': heatmaps[i:i + batch_size],
                 'mask': torch.ones(len(images[i:i + batch_size]), 2)}
                for i in range(0, n, batch_size)]

    def train_once(batch_size, accum):
        torch.manual_seed(1234)
        model = nn.Conv2d(1, 2, 3, padding=1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss, steps = run_epoch(model, make_batches(8, batch_size), torch.device('cpu'),
                                optimizer, grad_accum_steps=accum)
        return model.weight.detach().clone(), steps, loss

    big, steps_big, loss_big = train_once(8, 1)
    accumulated, steps_accum, loss_accum = train_once(2, 4)
    check("accumulating 4 micro-batches == one big batch (weights)",
          torch.allclose(big, accumulated, atol=1e-6),
          f"max diff {float((big - accumulated).abs().max()):.2e}")
    check("both count one optimizer step", steps_big == 1 and steps_accum == 1,
          f"{steps_big} vs {steps_accum}")
    check("reported loss is unscaled in both", abs(loss_big - loss_accum) < 1e-6)

    _, steps_plain, _ = train_once(2, 1)
    check("without accumulation every micro-batch is a step", steps_plain == 4, str(steps_plain))


# ------------------------------------------------------- 5. the real DINOv2 arm
def test_real_dinov2():
    print("\n[5] real DINOv2 model (needs cached weights)")
    try:
        from dinov2_head import DINOv2HeatmapModel
        frozen = DINOv2HeatmapModel(num_landmarks=3, freeze_backbone=True)
    except Exception as e:
        skip("real DINOv2 arm", f"weights unavailable ({type(e).__name__}) - run prefetch_weights.py")
        return

    trainable_frozen = sum(p.numel() for p in frozen.parameters() if p.requires_grad)
    finetuned = DINOv2HeatmapModel(num_landmarks=3, freeze_backbone=False)
    trainable_ft = sum(p.numel() for p in finetuned.parameters() if p.requires_grad)
    check("freeze_backbone=True trains only the decoder (Run 6 behaviour)",
          trainable_frozen < trainable_ft)
    check("freeze_backbone=False trains the whole backbone too",
          trainable_ft == sum(p.numel() for p in finetuned.parameters()))

    frozen.train()
    check("a frozen backbone stays in eval mode while the model trains",
          not frozen.backbone.training and frozen.head.training)
    finetuned.train()
    check("an unfrozen backbone follows the model into train mode",
          finetuned.backbone.training)

    image = torch.randn(1, 1, 518, 518)
    with torch.no_grad():
        out_last = frozen.eval()(image)
    check("output is (N, landmarks, grid*4, grid*4)",
          tuple(out_last.shape) == (1, 3, 37 * 4, 37 * 4), str(tuple(out_last.shape)))

    fused = DINOv2HeatmapModel(num_landmarks=3, freeze_backbone=True, feature_layers=[3, 6, 9, 12])
    with torch.no_grad():
        out_fused = fused.eval()(image)
    check("multi-depth fusion keeps the output shape", out_fused.shape == out_last.shape)
    check("fusion widens the projection input 4x",
          fused.proj.in_channels == 4 * frozen.proj.in_channels)
    check("an out-of-range feature layer raises",
          _raises(DINOv2HeatmapModel, num_landmarks=3, feature_layers=[3, 99]))

    deeper = DINOv2HeatmapModel(num_landmarks=3, decoder_extra_blocks=2)
    check("decoder_extra_blocks adds capacity without changing the stride",
          len(list(deeper.refine.children())) == 2
          and tuple(deeper.eval()(image).shape) == tuple(out_last.shape))

    optimizer, summary = build_optimizer(finetuned, {
        'lr': 3e-4, 'weight_decay': 1e-4,
        'optimizer': {'type': 'adamw', 'decoder_lr': 3e-4, 'encoder_lr': 5e-5,
                      'layerwise_decay': 0.75, 'encoder_weight_decay': 0.05}})
    names = {g['name'] for g in summary}
    check("real ViT parameter names are grouped per block",
          'encoder.block11' in names and 'encoder.embeddings' in names and 'decoder' in names,
          str(sorted(names)))
    check("every trainable parameter is covered exactly once",
          sum(g['n_params'] for g in summary)
          == sum(p.numel() for p in finetuned.parameters() if p.requires_grad))


def main():
    test_backward_compatibility()
    test_encoder_decoder_split()
    test_layerwise_decay()
    test_grad_accumulation()
    test_real_dinov2()
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed, {len(SKIPPED)} skipped")
    if FAILED:
        print("FAILED: " + ", ".join(FAILED))
        sys.exit(1)


if __name__ == "__main__":
    main()
