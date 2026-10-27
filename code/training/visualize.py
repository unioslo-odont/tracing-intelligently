"""
================================================================================
VALIDATION OVERLAY VISUALIZATION
================================================================================
DESCRIPTION:
    Renders ground-truth vs. predicted landmark points on top of each
    validation-set image, drawn at that image's own NATIVE resolution.
    Images in this dataset are not a fixed size (it varies per .fcd/image);
    the network always resizes to a common input resolution for training,
    but predictions are rescaled back using that specific image's own
    original size (read from the image file itself, not assumed from
    config) so points land in the right place regardless of how large or
    small the source radiograph was.

    Called automatically by train.py at the end of training, using the best
    checkpoint's weights. Can also be re-run standalone against any
    checkpoint (see evaluate.py --save-overlays).

    Ground truth is drawn as green circles and predictions as crosshairs. By
    default, both layers are restricted to the model's trained landmark
    vocabulary: a source tracing may contain additional landmarks that the
    model never predicted, and drawing those would make the overlay misleading.
    An explicit display_landmarks subset can further restrict both layers.
    Predictions are red where ground truth is present and orange where it is
    absent (not necessarily wrong, just not checkable on that image).
================================================================================
"""

import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')))
from heatmap_codec import decode_heatmaps


def _to_display_rgb(image_array):
    """1st/99th-percentile contrast stretch to 8-bit RGB, for readability only."""
    vmin, vmax = np.percentile(image_array, [1, 99])
    normalized = np.clip((image_array.astype(np.float32) - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    rgb = (np.stack([normalized] * 3, axis=-1) * 255).astype(np.uint8)
    return Image.fromarray(rgb)


def resolve_overlay_landmarks(trained_landmarks, display_landmarks=None):
    """Return the trained channels to draw, validating an optional subset."""
    trained_landmarks = list(trained_landmarks)
    if display_landmarks is None:
        return trained_landmarks
    display_landmarks = list(display_landmarks)
    unknown = [name for name in display_landmarks if name not in trained_landmarks]
    if unknown:
        raise ValueError(
            f"Overlay landmark(s) were not trained: {unknown}; "
            f"trained landmarks: {trained_landmarks}"
        )
    duplicates = sorted({name for name in display_landmarks if display_landmarks.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate overlay landmark(s): {duplicates}")
    return display_landmarks


def save_validation_overlays(model, base_dataset, transform, indices, device, output_dir,
                             max_images=None, display_landmarks=None):
    """
    model: trained model, will be switched to eval mode.
    base_dataset: a CephalometricDataset built WITHOUT a transform (so
        __getitem__ returns the raw, native-resolution image + landmarks).
    transform: the HeatmapTargetTransform used for training (its
        `landmark_names` defines the model's output channel order).
    indices: sample indices into base_dataset to visualize (e.g. a fold's val_idx).
    output_dir: folder to write one PNG per image into.
    max_images: cap the number of images written (None = all).
    display_landmarks: optional ordered subset of the model's trained channels.
        Both the ground-truth and prediction layers are restricted to this
        subset. Default: every trained channel and no untrained tracing points.

    Returns the list of written file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    hm_w, hm_h = transform.heatmap_size
    display_landmarks = resolve_overlay_landmarks(transform.landmark_names, display_landmarks)
    display_set = set(display_landmarks)
    trained_index = {name: i for i, name in enumerate(transform.landmark_names)}
    indices = list(indices)
    if max_images is not None:
        indices = indices[:max_images]

    written_paths = []
    with torch.no_grad():
        for idx in indices:
            raw_sample = base_dataset[idx]
            transformed = transform(raw_sample)

            image_tensor = transformed['image'].unsqueeze(0).to(device)
            pred_heatmaps = model(image_tensor)
            pred_coords_hm, _ = decode_heatmaps(pred_heatmaps)
            pred_coords_hm = pred_coords_hm[0]  # (C, 2)

            orig_h, orig_w = raw_sample['image'].shape[:2]
            scale_x = orig_w / hm_w
            scale_y = orig_h / hm_h
            pred_coords_px = pred_coords_hm * np.array([scale_x, scale_y], dtype=np.float32)

            pil_image = _to_display_rgb(raw_sample['image'])
            draw = ImageDraw.Draw(pil_image)

            mask = transformed['mask'].numpy()

            for name, (x, y) in zip(raw_sample['landmark_names'], raw_sample['landmarks']):
                if name not in display_set:
                    continue
                r = 4
                draw.ellipse((x - r, y - r, x + r, y + r), outline='lime', width=2)
                draw.text((x + 6, y - 6), name, fill='lime')

            for name in display_landmarks:
                i = trained_index[name]
                x, y = pred_coords_px[i]
                r = 5
                color = 'red' if mask[i] > 0 else 'orange'
                draw.line((x - r, y, x + r, y), fill=color, width=2)
                draw.line((x, y - r, x, y + r), fill=color, width=2)
                draw.text((x + 6, y + 6), name, fill=color)

            out_name = os.path.splitext(raw_sample['imagefile'])[0] + '_overlay.png'
            out_path = os.path.join(output_dir, out_name)
            pil_image.save(out_path)
            written_paths.append(out_path)

    return written_paths


def load_image_array(path):
    """Load an arbitrary image file (any of the formats PIL supports, any
    bit depth) as a single-channel numpy array - grayscale images are kept
    at their native dtype (e.g. 16-bit Facad TIFFs stay uint16, which
    matters for HeatmapTargetTransform's normalization); colour images are
    converted to 8-bit grayscale. For use with images that have no
    associated .fcd tracing (see predict_landmarks below) - CephalometricDataset
    is not involved."""
    with Image.open(path) as img:
        array = np.array(img)
    if array.ndim == 3:
        array = np.array(Image.fromarray(array).convert('L'))
    return array


def predict_landmarks(model, transform, image_array, device):
    """Run `model` on a single raw image (arbitrary size/dtype), using
    `transform`'s resize/normalize/heatmap settings, with no ground truth
    required - for inference on new images that were never traced. Returns
    (coords_px, confidence): predicted landmark coordinates already rescaled
    back into this image's OWN native pixel space, and each channel's peak
    heatmap value, in `transform.landmark_names` order."""
    orig_h, orig_w = image_array.shape[:2]
    dummy_sample = {
        'image': image_array,
        'landmarks': np.zeros((0, 2), dtype=np.float32),
        'landmark_names': [],
        'imagefile': '', 'serial': '', 'pixel_spacing_mm': None,
    }
    transformed = transform(dummy_sample)
    image_tensor = transformed['image'].unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        pred_heatmaps = model(image_tensor)
    pred_coords_hm, pred_conf = decode_heatmaps(pred_heatmaps)
    pred_coords_hm, pred_conf = pred_coords_hm[0], pred_conf[0]

    hm_w, hm_h = transform.heatmap_size
    scale_x, scale_y = orig_w / hm_w, orig_h / hm_h
    coords_px = pred_coords_hm * np.array([scale_x, scale_y], dtype=np.float32)
    return coords_px, pred_conf


def render_prediction_only_overlay(image_array, landmark_names, coords_px, color='deepskyblue'):
    """Like the prediction side of save_validation_overlays, but for images
    with no ground truth to compare against - draws every predicted point as
    a single-color labelled crosshair (no red/orange split, which depends on
    a ground-truth mask that doesn't exist for these images)."""
    pil_image = _to_display_rgb(image_array)
    draw = ImageDraw.Draw(pil_image)
    for name, (x, y) in zip(landmark_names, coords_px):
        r = 5
        draw.line((x - r, y, x + r, y), fill=color, width=2)
        draw.line((x, y - r, x, y + r), fill=color, width=2)
        draw.text((x + 6, y + 6), name, fill=color)
    return pil_image
