"""
================================================================================
PRETRAINED-ENCODER U-NET HEATMAP MODEL
================================================================================
DESCRIPTION:
    Same overall shape as unet.py's from-scratch baseline - a U-Net decoder
    producing one heatmap channel per landmark at 1/4 input resolution - but
    the encoder is a pretrained backbone loaded via `timm` instead of conv
    blocks trained from scratch.

    This single class covers BOTH the "ImageNet-pretrained CNN encoder" arm
    and the "pretrained HRNet" arm: any timm model that supports
    `features_only=True` works here, selected purely via `encoder_name` in
    the config (e.g. 'resnet34', 'hrnet_w32', 'convnext_tiny', ...) - they
    all expose the same 5-level feature pyramid interface
    (strides [2, 4, 8, 16, 32]), including timm's HRNet, which - contrary to
    HRNet's usual "native" multi-resolution fusion output - exposes this
    same standard pyramid in features_only mode. So there is no HRNet-
    specific code path here; it is just another encoder_name.

    `in_chans=1` lets timm adapt the pretrained (3-channel, ImageNet) first
    conv layer to our single-channel grayscale radiographs by averaging the
    pretrained RGB filters across the channel dimension, rather than
    repeating the image to 3 channels - the standard, slightly more
    parameter-faithful way to reuse RGB-pretrained weights on grayscale
    input (see timm's PatchEmbed/stem adaptation).

    NOTE ON PRETRAINING TASK: these timm checkpoints are ImageNet
    CLASSIFICATION-pretrained, not pretrained for keypoint/pose detection.
    A COCO-keypoint-pretrained HRNet would be a closer task match but
    requires the MMPose/mmcv stack, which failed to even build when tried
    for this project (a known, recurring mmcv/setuptools compatibility
    break) - not pursued for now.
================================================================================
"""

import timm
import torch
import torch.nn as nn


def decoder_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class PretrainedEncoderUNetHeatmapModel(nn.Module):
    def __init__(self, num_landmarks, encoder_name='resnet34', pretrained=True,
                 in_channels=1, freeze_encoder=False):
        super().__init__()
        self.encoder = timm.create_model(
            encoder_name, pretrained=pretrained, in_chans=in_channels, features_only=True,
        )
        strides = self.encoder.feature_info.reduction()
        channels = self.encoder.feature_info.channels()
        if strides[-4:] != [4, 8, 16, 32]:
            raise ValueError(
                f"encoder_name='{encoder_name}' has feature strides {strides}, "
                "expected the last four levels to be [4, 8, 16, 32] (standard "
                "timm features_only convention) - this encoder isn't a drop-in fit."
            )
        c1, c2, c3, c4 = channels[-4:]  # stride 4, 8, 16, 32 channel counts

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        self.freeze_encoder = freeze_encoder

        self.up4 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec4 = decoder_block(c3 + c3, c3)

        self.up3 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec3 = decoder_block(c2 + c2, c2)

        self.up2 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec2 = decoder_block(c1 + c1, c1)

        self.head = nn.Conv2d(c1, num_landmarks, kernel_size=1)

    def forward(self, x):
        if self.freeze_encoder:
            with torch.no_grad():
                features = self.encoder(x)
        else:
            features = self.encoder(x)
        f1, f2, f3, f4 = features[-4:]  # stride 4, 8, 16, 32

        d4 = self.up4(f4)                      # stride 16
        d4 = self.dec4(torch.cat([d4, f3], dim=1))

        d3 = self.up3(d4)                      # stride 8
        d3 = self.dec3(torch.cat([d3, f2], dim=1))

        d2 = self.up2(d3)                      # stride 4
        d2 = self.dec2(torch.cat([d2, f1], dim=1))

        return self.head(d2)                   # (N, num_landmarks, H/4, W/4)
