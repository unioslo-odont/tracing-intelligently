"""
================================================================================
HEATMAP TARGET TRANSFORM
================================================================================
DESCRIPTION:
    Callable to pass as `transform=` to CephalometricDataset. Converts a raw
    sample (arbitrary-size image + a per-image subset of named landmarks)
    into fixed-shape training tensors:
      - resizes the image to `image_size` and normalizes it to [0, 1]
      - encodes each landmark onto its own channel of a `heatmap_size`
        Gaussian heatmap target, using a FIXED landmark vocabulary
        (`landmark_names`, given at construction) so every sample produces
        the same number of channels in the same channel order regardless of
        which landmarks that particular tracing actually has
      - returns a `mask` (num_landmarks,) marking which channels are real
        for this sample (tracings are sometimes partial - see
        select_reviewed_tracings.py) so the training loss can ignore the
        rest instead of penalizing the model for "missing" a landmark that
        was never marked in this image's tracing.

    The landmark vocabulary should be built once from the training set (see
    `collect_landmark_vocabulary`) and reused for validation/test so channel
    order is consistent across a run.
================================================================================
"""

import os
import sys
from collections import Counter

import numpy as np
import torch
import yaml
from PIL import Image

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')))
from heatmap_codec import encode_landmark_heatmap


def collect_landmark_vocabulary(dataset, min_fraction=0.5):
    """Return a sorted list of landmark names present in at least
    `min_fraction` of `dataset.samples` (default: at least half the images).
    Deterministic order (alphabetical) so channel indices are reproducible
    across runs given the same data.

    min_fraction=0.5 is a real-data-informed default, not arbitrary: an
    analysis of this project's actual TSD cohort (see
    data_preparation_TSD/analyze_landmark_names.py) found a sharp cliff -
    the genuine ~32-landmark protocol appears in >=50% of images, and
    everything else (typos, one-off freehand annotations, an older/rarer
    protocol variant) appears in under 25% of images, usually just a
    handful. min_fraction=0.0 recovers the old "collect literally every
    name ever seen" behavior if you need it for debugging."""
    counts = Counter()
    for entry in dataset.samples:
        counts.update(entry['landmarks'].keys())
    n_samples = len(dataset.samples)
    threshold = min_fraction * n_samples
    return sorted(name for name, count in counts.items() if count >= threshold)


def load_landmark_config(yaml_path):
    """Read a landmark-set YAML (see configs/landmarks.yaml) and return the
    sorted list of landmark names with enabled: true. This is an explicit,
    curated alternative to collect_landmark_vocabulary's automatic
    frequency threshold - use it when specific landmarks are clinically
    required regardless of how common they are in the raw data (e.g. rare
    but clinically-needed points), or when specific landmarks should be
    excluded regardless of how common they are (e.g. one under
    investigation for an annotation-quality problem)."""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        spec = yaml.safe_load(f)

    non_string_keys = [name for name in spec['landmarks'].keys() if not isinstance(name, str)]
    if non_string_keys:
        raise ValueError(
            f"{yaml_path}: landmark key(s) {non_string_keys!r} parsed as non-string values, "
            "not landmark names. This happens when a landmark code is a YAML boolean-like "
            "token left unquoted (e.g. bare 'N', 'T', 'Y', 'no', 'true', 'on', ...) - YAML's "
            "implicit typing turns those into True/False rather than strings (the classic "
            "'Norway problem'; some YAML parsers, e.g. R's, apply this even to single letters "
            "like bare N/T). Fix: quote that landmark's key in the YAML, e.g. \"N\": not N:."
        )

    return sorted(name for name, info in spec['landmarks'].items() if info.get('enabled', False))


def resolve_landmark_vocabulary(dataset, data_config):
    """Decide the trained landmark vocabulary from a config's `data` block:
    if `landmark_config_path` is set, use that explicit enable/disable list
    (see configs/landmarks.yaml and load_landmark_config) - with a warning
    for any enabled landmark never actually seen in this dataset (a likely
    spelling mismatch). Otherwise, fall back to the automatic frequency
    threshold (`min_landmark_fraction`, default 0.5 - see
    collect_landmark_vocabulary)."""
    landmark_config_path = data_config.get('landmark_config_path')
    if landmark_config_path:
        landmark_names = load_landmark_config(landmark_config_path)
        available = collect_landmark_vocabulary(dataset, min_fraction=0.0)
        missing = [name for name in landmark_names if name not in available]
        if missing:
            print(f"[WARN] enabled landmark(s) not found in any image - check spelling in "
                  f"'{landmark_config_path}': {missing}")
        return landmark_names
    return collect_landmark_vocabulary(dataset, min_fraction=data_config.get('min_landmark_fraction', 0.5))


DEFAULT_AUG = {
    'rotation_deg': 15.0,   # random rotation in +/- this many degrees
    'scale': 0.10,          # random scale in [1-scale, 1+scale]
    'translate': 0.05,      # random translation in +/- this fraction of width/height
    'brightness': 0.10,     # additive brightness jitter in +/- this (on [0,1] intensities)
    'contrast': 0.10,       # random contrast factor in [1-contrast, 1+contrast]
}


class HeatmapTargetTransform:
    def __init__(self, landmark_names, image_size=(512, 512), heatmap_size=(128, 128), sigma=2.0,
                 augment=False, aug_params=None):
        """
        landmark_names: fixed, ordered vocabulary - channel i is always this landmark.
        image_size: (width, height) the network input is resized to.
        heatmap_size: (width, height) of the target/predicted heatmaps.
        sigma: Gaussian std-dev in heatmap pixels.
        augment: if True, apply random affine (rotation/scale/translation) + brightness/contrast
            jitter. The affine is applied CONSISTENTLY to the image and the landmark coordinates, so
            targets stay aligned. Deliberately NO horizontal flip (lateral cephalograms have a fixed
            left/right orientation). Use ONLY for the training split - build a separate transform
            with augment=False for validation/test/overlays.
        aug_params: overrides for DEFAULT_AUG (any subset).
        """
        self.landmark_names = list(landmark_names)
        self.name_to_index = {name: i for i, name in enumerate(self.landmark_names)}
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.augment = augment
        self.aug = {**DEFAULT_AUG, **(aug_params or {})}

    def _sample_affine(self, w, h):
        """Random forward affine (2x3): p' = R*scale*(p - centre) + centre + translate."""
        rot = np.deg2rad(np.random.uniform(-self.aug['rotation_deg'], self.aug['rotation_deg']))
        scale = np.random.uniform(1 - self.aug['scale'], 1 + self.aug['scale'])
        tx = np.random.uniform(-self.aug['translate'], self.aug['translate']) * w
        ty = np.random.uniform(-self.aug['translate'], self.aug['translate']) * h
        cx, cy = w / 2.0, h / 2.0
        cos, sin = np.cos(rot) * scale, np.sin(rot) * scale
        return np.array([
            [cos, -sin, cx + tx - cos * cx + sin * cy],
            [sin,  cos, cy + ty - sin * cx - cos * cy],
        ], dtype=np.float32)

    @staticmethod
    def _affine_image(img_norm, M):
        """Apply the forward affine M to a normalized float32 image (H, W) in [0, 1].
        Done on a PIL 'F' (float) image - PIL's AFFINE transform silently returns black
        on 16-bit ('I;16') images, so the image must be float here. PIL AFFINE maps
        OUTPUT->INPUT, so it takes the inverse of the forward matrix. Out-of-frame fills with 0."""
        h, w = img_norm.shape
        inv = np.linalg.inv(np.vstack([M, [0, 0, 1]]).astype(np.float64))
        coeffs = (inv[0, 0], inv[0, 1], inv[0, 2], inv[1, 0], inv[1, 1], inv[1, 2])
        out = Image.fromarray(img_norm, mode='F').transform((w, h), Image.AFFINE, coeffs, resample=Image.BILINEAR)
        return np.asarray(out, dtype=np.float32)

    def _photometric(self, img):
        if self.aug['brightness']:
            img = img + np.random.uniform(-self.aug['brightness'], self.aug['brightness'])
        if self.aug['contrast']:
            img = (img - 0.5) * np.random.uniform(1 - self.aug['contrast'], 1 + self.aug['contrast']) + 0.5
        return np.clip(img, 0.0, 1.0).astype(np.float32)

    def __call__(self, sample):
        image = sample['image']
        orig_h, orig_w = image.shape[:2]
        target_w, target_h = self.image_size
        hm_w, hm_h = self.heatmap_size

        image_resized = np.array(
            Image.fromarray(image).resize((target_w, target_h), resample=Image.BILINEAR)
        ).astype(np.float32)
        max_val = float(np.iinfo(image.dtype).max) if np.issubdtype(image.dtype, np.integer) else float(image_resized.max() or 1.0)
        image_norm = image_resized / max_val  # (H, W) float in [0, 1]

        # landmark coords in network-input (image_size) space
        names = list(sample['landmark_names'])
        pts = np.array(sample['landmarks'], dtype=np.float32).reshape(-1, 2)
        if len(pts):
            pts = pts * np.array([target_w / orig_w, target_h / orig_h], dtype=np.float32)

        if self.augment:
            M = self._sample_affine(target_w, target_h)
            image_norm = self._affine_image(image_norm, M)  # same affine on the float image
            if len(pts):
                pts = (np.hstack([pts, np.ones((len(pts), 1), dtype=np.float32)]) @ M.T)
            image_norm = self._photometric(image_norm)

        image_tensor = torch.from_numpy(np.ascontiguousarray(image_norm)).unsqueeze(0).float()  # (1, H, W)

        num_landmarks = len(self.landmark_names)
        heatmaps = np.zeros((num_landmarks, hm_h, hm_w), dtype=np.float32)
        mask = np.zeros((num_landmarks,), dtype=np.float32)
        landmarks_hm = np.zeros((num_landmarks, 2), dtype=np.float32)

        hx_scale, hy_scale = hm_w / target_w, hm_h / target_h

        for name, (lx, ly) in zip(names, pts):
            idx = self.name_to_index.get(name)
            if idx is None:
                continue  # not in the configured vocabulary (e.g. a rare marker) - ignore
            hx, hy = lx * hx_scale, ly * hy_scale
            if self.augment and not (0 <= hx <= hm_w - 1 and 0 <= hy <= hm_h - 1):
                continue  # augmented off-frame - don't supervise this channel for this sample
            encode_landmark_heatmap(heatmaps[idx], hx, hy, self.sigma)
            mask[idx] = 1.0
            landmarks_hm[idx] = (hx, hy)

        return {
            'image': image_tensor,
            'heatmaps': torch.from_numpy(heatmaps),
            'mask': torch.from_numpy(mask),
            'landmarks_hm': torch.from_numpy(landmarks_hm),
            'orig_size': (orig_w, orig_h),
            'imagefile': sample['imagefile'],
            'serial': sample['serial'],
            'pixel_spacing_mm': sample['pixel_spacing_mm'] if sample['pixel_spacing_mm'] is not None else float('nan'),
        }
