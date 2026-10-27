"""
================================================================================
HEATMAP ENCODE / DECODE (the "decoder" piece)
================================================================================
DESCRIPTION:
    Converts between (x, y) landmark coordinates and 2D Gaussian heatmaps, in
    both directions:
      - encode_landmark_heatmap: splat a single (x, y) point onto a heatmap
        as a truncated Gaussian (used when building training targets).
      - decode_heatmaps: recover sub-pixel (x, y) + confidence from a batch
        of predicted (or target) heatmaps, via argmax + a quarter-pixel
        offset towards the higher of the two neighbouring pixels (the
        "Simple Baselines" offset refinement, Xiao et al. 2018) - cheap and
        good enough for a first baseline.

    Kept separate from both the model (backbone/head) and the dataset
    transform so any of the three can be swapped independently: a different
    codec (e.g. DARK pose Taylor-expansion decoding, or soft-argmax) can
    replace this module without touching the model or the data pipeline.
================================================================================
"""

import numpy as np
import torch


def encode_landmark_heatmap(heatmap, cx, cy, sigma):
    """In-place max-splat a truncated 2D Gaussian centred at (cx, cy) (heatmap
    pixel coordinates, can be fractional) onto a single-channel `heatmap`
    (H, W) numpy array. No-ops if the Gaussian falls entirely outside the
    heatmap (e.g. a landmark just off the edge of the resized image)."""
    h, w = heatmap.shape
    tmp_size = int(sigma * 3)

    x0, y0 = int(np.floor(cx - tmp_size)), int(np.floor(cy - tmp_size))
    x1, y1 = int(np.floor(cx + tmp_size)) + 1, int(np.floor(cy + tmp_size)) + 1
    if x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h:
        return

    size = 2 * tmp_size + 1
    grid_x = np.arange(0, size, dtype=np.float32)
    grid_y = grid_x[:, None]
    center = tmp_size + (cx - int(np.floor(cx)))  # sub-pixel-aware center within the patch
    center_y = tmp_size + (cy - int(np.floor(cy)))
    gaussian = np.exp(-((grid_x - center) ** 2 + (grid_y - center_y) ** 2) / (2 * sigma ** 2))

    src_x_min, src_y_min = max(0, -x0), max(0, -y0)
    dst_x_min, dst_x_max = max(0, x0), min(x1, w)
    dst_y_min, dst_y_max = max(0, y0), min(y1, h)

    # Slice both patches, then clip to their COMMON shape. The destination extent
    # (y1 - y0) is not guaranteed to equal the fixed Gaussian size: with a
    # floating-point coordinate, floor(c + tmp) - floor(c - tmp) can be 2*tmp +/- 1
    # (e.g. a float32 value just below an integer where c + tmp rounds up), which
    # otherwise mismatches the (2*tmp+1)-sized Gaussian by one row/column and raises
    # a broadcast error. Taking the min shape is robust to that and to edge clipping.
    dst = heatmap[dst_y_min:dst_y_max, dst_x_min:dst_x_max]
    src = gaussian[src_y_min:src_y_min + dst.shape[0], src_x_min:src_x_min + dst.shape[1]]
    hh, ww = min(dst.shape[0], src.shape[0]), min(dst.shape[1], src.shape[1])
    if hh <= 0 or ww <= 0:
        return
    heatmap[dst_y_min:dst_y_min + hh, dst_x_min:dst_x_min + ww] = np.maximum(
        dst[:hh, :ww], src[:hh, :ww],
    )


def decode_heatmaps(heatmaps):
    """heatmaps: (N, C, H, W) torch tensor or numpy array (predicted or
    target). Returns (coords, confidence): coords is (N, C, 2) float32
    [x, y] in heatmap pixel coordinates, confidence is (N, C) float32 peak
    value (useful as a rough per-landmark reliability signal)."""
    if isinstance(heatmaps, torch.Tensor):
        heatmaps = heatmaps.detach().cpu().numpy()

    n, c, h, w = heatmaps.shape
    flat = heatmaps.reshape(n, c, -1)
    flat_idx = np.argmax(flat, axis=-1)
    confidence = np.take_along_axis(flat, flat_idx[..., None], axis=-1).squeeze(-1)

    py = (flat_idx // w).astype(np.float32)
    px = (flat_idx % w).astype(np.float32)

    for i in range(n):
        for j in range(c):
            x_i, y_i = int(px[i, j]), int(py[i, j])
            if 1 <= x_i <= w - 2 and 1 <= y_i <= h - 2:
                diff_x = heatmaps[i, j, y_i, x_i + 1] - heatmaps[i, j, y_i, x_i - 1]
                diff_y = heatmaps[i, j, y_i + 1, x_i] - heatmaps[i, j, y_i - 1, x_i]
                px[i, j] += 0.25 * np.sign(diff_x)
                py[i, j] += 0.25 * np.sign(diff_y)

    coords = np.stack([px, py], axis=-1).astype(np.float32)
    return coords, confidence.astype(np.float32)
