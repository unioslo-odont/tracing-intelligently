"""
================================================================================
DINOv2 BACKBONE + TRAINABLE DECODER HEAD
================================================================================
DESCRIPTION:
    The self-supervised-pretraining arm: a DINOv2 ViT backbone
    (facebookresearch, via the `transformers` library) with a convolutional
    decoder on top. Two modes, chosen by `freeze_backbone`:

      - FROZEN (default, Run 6's dinov2_frozen): only the small decoder is
        trained. Almost nothing is fitted to this project's small dataset,
        which is the point of the label-efficiency question.
      - FINE-TUNED (Run 9's dinov2_finetune): the whole ViT trains too, with
        layer-wise LR decay (see code/training/optim_factory.py). This exists
        because Run 6 compared a FROZEN DINOv2 against three FULLY FINE-TUNED
        CNNs, which conflates "self-supervised features" with "no adaptation" -
        the frozen arm was never a like-for-like fourth arm.

    DINOv2 was pretrained on natural RGB images, so a single-channel
    grayscale radiograph is replicated to 3 channels before the backbone
    (the standard workaround for reusing RGB-pretrained ViTs on grayscale
    medical images - transformers' Dinov2Model has no equivalent of timm's
    `in_chans=1` weight-averaging adaptation for patch embeddings).

    ViT patch tokens are coarse (stride = patch_size, 14 for dinov2-small/
    dinov2-base): the backbone's output is a (grid_h x grid_w) token map,
    not a pixel-dense feature map. The decoder upsamples this by 4x (two
    stride-2 transposed convs) to recover some spatial precision, but the
    input resolution needs to be a multiple of 14 for the patch grid to be
    well-defined - see configs/dinov2_frozen.yaml (image_size: [518, 518],
    NOT the 512 used by the other arms).

    `feature_layers` optionally fuses tokens from several depths (e.g.
    [3, 6, 9, 12]) instead of the last block only: for dense localization the
    last block of a ViT is the most semantic but not the most spatially
    precise, and mid-depth tokens carry finer positional detail. Fusion is
    concatenate-then-1x1-project, so the decoder shape is unchanged.

    The 4x decoder stride is deliberate and fixed: data.heatmap_size in the
    config, the mm conversion in eval_core, and the sub-pixel decoding in
    heatmap_codec all assume it. Extra decoder CAPACITY is therefore added as
    conv blocks at the final resolution (`decoder_extra_blocks`), not as more
    upsampling stages.
================================================================================
"""

import torch
import torch.nn as nn
from transformers import AutoModel


def decoder_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class DINOv2HeatmapModel(nn.Module):
    def __init__(self, num_landmarks, model_name='facebook/dinov2-small', decoder_channels=256,
                 freeze_backbone=True, feature_layers=None, decoder_extra_blocks=0):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.patch_size = self.backbone.config.patch_size
        hidden_size = self.backbone.config.hidden_size
        num_layers = self.backbone.config.num_hidden_layers

        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            self.backbone.eval()

        # 1-indexed transformer blocks; hidden_states[0] is the embedding output,
        # so block i is hidden_states[i]. None = last block only (Run 6 behaviour).
        if feature_layers:
            feature_layers = [int(i) for i in feature_layers]
            bad = [i for i in feature_layers if not 1 <= i <= num_layers]
            if bad:
                raise ValueError(f"feature_layers {bad} out of range for {model_name} "
                                 f"(1..{num_layers}, 1-indexed)")
        self.feature_layers = feature_layers or None
        fused_channels = hidden_size * (len(self.feature_layers) if self.feature_layers else 1)

        self.proj = nn.Conv2d(fused_channels, decoder_channels, kernel_size=1)
        self.up1 = nn.ConvTranspose2d(decoder_channels, decoder_channels // 2, kernel_size=2, stride=2)
        self.dec1 = decoder_block(decoder_channels // 2, decoder_channels // 2)
        self.up2 = nn.ConvTranspose2d(decoder_channels // 2, decoder_channels // 4, kernel_size=2, stride=2)
        self.dec2 = decoder_block(decoder_channels // 4, decoder_channels // 4)
        self.refine = nn.Sequential(*[decoder_block(decoder_channels // 4, decoder_channels // 4)
                                      for _ in range(int(decoder_extra_blocks))])
        self.head = nn.Conv2d(decoder_channels // 4, num_landmarks, kernel_size=1)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()  # frozen backbone stays in eval mode (no BN/dropout stat updates)
        return self

    def _tokens_to_map(self, tokens, grid_h, grid_w):
        """(N, 1 + [registers] + grid_h*grid_w, C) -> (N, C, grid_h, grid_w).
        Patch tokens are last, so taking the final grid_h*grid_w drops the CLS
        token and any register tokens without needing to know how many there are."""
        patch_tokens = tokens[:, -(grid_h * grid_w):, :]
        return patch_tokens.permute(0, 2, 1).reshape(-1, patch_tokens.shape[-1], grid_h, grid_w)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        h, w = x.shape[-2:]
        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise ValueError(
                f"Input H/W ({h}, {w}) must be divisible by the DINOv2 patch size "
                f"({self.patch_size}) - see configs/dinov2_frozen.yaml: data.image_size."
            )
        grid_h, grid_w = h // self.patch_size, w // self.patch_size

        with torch.set_grad_enabled(self.training and not self.freeze_backbone):
            outputs = self.backbone(x, output_hidden_states=bool(self.feature_layers))
            if self.feature_layers:
                features = torch.cat([self._tokens_to_map(outputs.hidden_states[i], grid_h, grid_w)
                                      for i in self.feature_layers], dim=1)
            else:
                features = self._tokens_to_map(outputs.last_hidden_state, grid_h, grid_w)

        d = self.proj(features)
        d = self.dec1(self.up1(d))
        d = self.dec2(self.up2(d))
        d = self.refine(d)
        return self.head(d)  # (N, num_landmarks, grid_h * 4, grid_w * 4)
