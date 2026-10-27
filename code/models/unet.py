"""
================================================================================
U-NET HEATMAP MODEL (baseline "backbone + head")
================================================================================
DESCRIPTION:
    A plain-PyTorch U-Net for heatmap-regression landmark detection - the
    project's "full control" baseline arm (as opposed to the MMPose/HRNet
    stack). Single-channel grayscale input; output is one heatmap channel
    per landmark in the configured vocabulary, at 1/4 of the input
    resolution (matches the `heatmap_size = image_size / 4` convention used
    by code/datasets/heatmap_transform.py and common heatmap-regression
    practice, e.g. HRNet/Simple Baselines).

    Architecture: 4 encoder downsampling stages (x16 total), 2 decoder
    upsampling stages (x4 total) with skip connections from the matching
    encoder resolutions, then a 1x1 conv to `num_landmarks` output channels.
    Input height/width must be divisible by 16.

    Deliberately a single self-contained "backbone + head" module - swap it
    for an MMPose/HRNet wrapper with the same
    (image_tensor) -> (N, num_landmarks, H/4, W/4) interface to compare
    architectures without touching the dataset, codec, training loop, or
    evaluation code.
================================================================================
"""

import torch
import torch.nn as nn


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class UNetHeatmapModel(nn.Module):
    def __init__(self, num_landmarks, in_channels=1, base_channels=32):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        self.enc1 = conv_block(in_channels, c1)
        self.enc2 = conv_block(c1, c2)
        self.enc3 = conv_block(c2, c3)
        self.enc4 = conv_block(c3, c4)
        self.bottleneck = conv_block(c4, c5)

        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(c5, c4, kernel_size=2, stride=2)
        self.dec4 = conv_block(c4 + c4, c4)

        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = conv_block(c3 + c3, c3)

        self.head = nn.Conv2d(c3, num_landmarks, kernel_size=1)

    def forward(self, x):
        if x.shape[-1] % 16 != 0 or x.shape[-2] % 16 != 0:
            raise ValueError(f"Input H/W must be divisible by 16, got {tuple(x.shape[-2:])}")

        e1 = self.enc1(x)                    # H
        e2 = self.enc2(self.pool(e1))        # H/2
        e3 = self.enc3(self.pool(e2))        # H/4
        e4 = self.enc4(self.pool(e3))        # H/8
        b = self.bottleneck(self.pool(e4))   # H/16

        d4 = self.up4(b)                     # H/8
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)                    # H/4
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        return self.head(d3)                 # (N, num_landmarks, H/4, W/4)
