"""
================================================================================
AARIZ EXTERNAL TEST SET LOADER  (original CEPHA29 directory format)
================================================================================
DESCRIPTION:
    torch.utils.data.Dataset over the Aariz / CEPHA29 cephalometric benchmark
    (1000 lateral cephalograms, Islamic International Dental College / Riphah
    International University, Islamabad), read from the ORIGINAL published
    directory layout - NOT the flat `landmark_coordinates_anon.csv` an earlier
    version of this file expected. Layout (see the Aariz GitHub, manwaarkhd/aariz):

        <root>/<split>/Cephalograms/<ceph_id>.png
        <root>/<split>/Annotations/Cephalometric Landmarks/Senior Orthodontists/<ceph_id>.json
        <root>/<split>/Annotations/Cephalometric Landmarks/Junior Orthodontists/<ceph_id>.json
        <root>/cephalogram_machine_mappings.csv        # per-image mm/px calibration

    <split> is one of train / valid / test (Aariz's own 700/150/150 split). Each
    annotation JSON is {"landmarks": [{"symbol": "<code>", "value": {"x": .., "y": ..}}, ...]}
    with coordinates in the ORIGINAL cephalogram's pixel space (same space as the
    .png), so no re-scaling is needed before the HeatmapTargetTransform resizes.

    We use the SENIOR orthodontist tracings by default (annotator='senior'); the
    Aariz reference code averages junior+senior, which is deliberately NOT the
    default here.

    USE: evaluation only, never training. Aariz is CC-BY-NC-ND 4.0
    (NonCommercial-NoDerivatives) - evaluating an already-trained checkpoint is
    fine; training on it and releasing weights sits in unresolved "derivative
    work" territory. There is deliberately no training entry point - see
    code/training/evaluate_aariz.py.

    Landmark names are remapped at load time via `name_mapping` (Aariz's own
    symbol -> the name THIS project's checkpoints were trained with), NEVER
    auto-matched by case-folding or fuzzy similarity - see
    AARIZ_TO_REFERENCE_MAPPING below for why (the Pg/PG lesson from this
    project's own data: same-looking codes are not always the same landmark, so
    every mapping here is an explicit, individually-reasoned entry).

    mm calibration: Aariz spans 7 scanners with different pixel spacings, so mm
    error is only meaningful with per-image calibration. That lives in
    cephalogram_machine_mappings.csv. Its real columns (confirmed 2026-08-03 on
    the Fox copy) are:
        cephalogram_id, machine, pixel_size, image_format, mode
    where cephalogram_id == the .png/.json basename and pixel_size is mm/pixel
    (e.g. 0.1 for "ART Plus"). Columns are auto-detected (id-like column + an
    mm-per-pixel / pixel-size / spacing column) - the two above are picked with
    no help needed; pass id_column / mm_per_pixel_column only if a future copy
    renames them. Without a usable value, pixel_spacing_mm is NaN for that image
    and the existing "uncalibrated image" handling in metrics.py/eval_core.py
    excludes it from mm-based MRE/SDR rather than treating the spacing as zero.
================================================================================
"""

import csv
import json
import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# Aariz landmark SYMBOL (as it appears in the annotation JSON / the repo's
# config.py) -> the name used by this project's trained checkpoints (the
# 26-point Oslo protocol, configs/landmarks.yaml). Only landmarks with a real,
# confirmed anatomical correspondence are listed - explicit and auditable,
# NEVER case-folded or fuzzy-matched, because a matching-looking code is not a
# guarantee of the same point.
#
# Confirmed correspondences (18):
#   A<-A, Ar<-Ar, B<-B, Gn<-Gn, N<-N, Or<-Or, Po<-Po, S<-S,
#   Is<-UIT (upper incisor tip),   Isa<-UIA (upper incisor apex),
#   Ii<-LIT (lower incisor tip),   Iia<-LIA (lower incisor apex),
#   Sp<-ANS (anterior nasal spine == spina nasalis anterior),
#   Pg<-Pog (hard-tissue pogonion),
#   PGs<-Pog` (SOFT-tissue pogonion; backtick in the Aariz symbol),
#   PLs<-Ls (labrale superius),  PLi<-Li (labrale inferius),  PRN<-Pn (pronasale).
#
# DELIBERATELY NOT MAPPED - no Aariz equivalent / structural mismatch:
#   Our Ba, GNs, MS (Steiner S-point), Pgn, Pt, iGo, pGo. Aariz has Me (Menton,
#   a different chin point from our Gn/Pgn), a single 'Go' (structurally not our
#   split inferior/posterior gonion), and no pterygoid/basion/S-point/GNs. These
#   get mask=0 on every Aariz image; the masked metrics report them as
#   "uncheckable", not wrong.
#
# CONFIRMED equivalent (2026-08-03): Aariz 'PNS' (posterior nasal spine) -> our
#   'Pm' (pterygomaxillare). Anatomically PNS and pterygomaxillare are adjacent
#   posterior-maxilla points, and the raw TSD data carried explicit 'PNS (Pm)'
#   markers, so this protocol treats them as one. This was additionally checked
#   empirically: when mapped, the point behaves like a matching landmark (a low,
#   unbiased error distribution) rather than like a definition mismatch (which
#   shows up as a large, low-scatter systematic offset). So it is ON by default;
#   drop it (evaluate_aariz.py --no-pns-pm) for a sensitivity check.
#
# => 19 of the 26 Oslo landmarks are externally checkable on Aariz.
AARIZ_TO_REFERENCE_MAPPING = {
    'A': 'A', 'Ar': 'Ar', 'B': 'B', 'Gn': 'Gn',
    'N': 'N', 'Or': 'Or', 'Po': 'Po', 'S': 'S',
    'UIT': 'Is', 'UIA': 'Isa', 'LIT': 'Ii', 'LIA': 'Iia',
    'ANS': 'Sp',
    'Pog': 'Pg',
    'Pog`': 'PGs',   # SOFT-tissue pogonion - note the backtick, distinct from 'Pog'
    'Ls': 'PLs', 'Li': 'PLi', 'Pn': 'PRN',
    'PNS': 'Pm',     # posterior nasal spine == pterygomaxillare in this protocol (confirmed, see note)
}

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

CEPHALOGRAM_SUBDIR = 'Cephalograms'
ANNOTATION_SUBDIR = os.path.join('Annotations', 'Cephalometric Landmarks')
ANNOTATOR_DIRS = {
    'senior': ['Senior Orthodontists'],
    'junior': ['Junior Orthodontists'],
    'both': ['Senior Orthodontists', 'Junior Orthodontists'],
}


def _parse_float(text):
    """Pull the first float out of a cell that might carry units ('0.1 mm')."""
    if text is None:
        return None
    text = str(text).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        num = ''
        for ch in text:
            if ch.isdigit() or ch in '.-+eE':
                num += ch
            elif num:
                break
        try:
            return float(num)
        except ValueError:
            return None


class AarizDataset(Dataset):
    def __init__(self, root, split='test', annotator='senior', name_mapping=None,
                 machine_mappings_csv=None, id_column=None, mm_per_pixel_column=None,
                 transform=None):
        """
        root: Aariz dataset root (contains train/ valid/ test/ and, at the root,
            cephalogram_machine_mappings.csv).
        split: 'train' | 'valid' | 'test' | 'all' (concatenate all three).
        annotator: 'senior' (default) | 'junior' | 'both' (per-landmark average
            of whichever of the two tracings are present).
        name_mapping: {aariz_symbol: target_landmark_name}; defaults to
            AARIZ_TO_REFERENCE_MAPPING.
        machine_mappings_csv: per-image mm/px calibration CSV. Defaults to
            <root>/cephalogram_machine_mappings.csv if that file exists.
        id_column / mm_per_pixel_column: override the auto-detected column names
            in that CSV.
        transform: optional callable (e.g. HeatmapTargetTransform) applied to
            each sample dict before it's returned.
        """
        self.root = root
        self.annotator = annotator
        if annotator not in ANNOTATOR_DIRS:
            raise ValueError(f"annotator must be one of {sorted(ANNOTATOR_DIRS)}, got {annotator!r}")
        self.name_mapping = name_mapping if name_mapping is not None else AARIZ_TO_REFERENCE_MAPPING
        self.transform = transform

        if machine_mappings_csv is None:
            default_csv = os.path.join(root, 'cephalogram_machine_mappings.csv')
            machine_mappings_csv = default_csv if os.path.exists(default_csv) else None
        self.pixel_spacing_by_id = {}
        if machine_mappings_csv is not None:
            self.pixel_spacing_by_id = self._load_machine_mappings(
                machine_mappings_csv, id_column, mm_per_pixel_column)

        splits = ['train', 'valid', 'test'] if split == 'all' else [split]
        self.samples = []
        self.skipped = []  # list of (ceph_id, reason)
        for sp in splits:
            self._build_index(sp)

    # ---- calibration -------------------------------------------------------
    @staticmethod
    def _load_machine_mappings(csv_path, id_column=None, mm_per_pixel_column=None):
        mapping = {}
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
            if id_column is None:
                id_column = next(
                    (c for c in cols if c and c.strip().lower() in (
                        'ceph_id', 'cephalogram', 'cephalogram_id', 'id', 'image',
                        'imagefile', 'image_file', 'filename', 'file_name', 'name', 'file')),
                    None)
            if mm_per_pixel_column is None:
                # Prefer an explicit mm-per-pixel / pixel-size / spacing column.
                mm_per_pixel_column = next(
                    (c for c in cols if c and any(t in c.strip().lower() for t in (
                        'mm_per_pixel', 'mm per pixel', 'mm/pixel', 'mm/px',
                        'pixel_size', 'pixel size', 'pixelsize',
                        'pixel_spacing', 'pixel spacing', 'spacing', 'mm_per_px'))),
                    None)
                if mm_per_pixel_column is None:
                    mm_per_pixel_column = next(
                        (c for c in cols if c and ('resolution' in c.strip().lower()
                                                   or ('mm' in c.strip().lower()
                                                       and 'pixel' in c.strip().lower()))),
                        None)
            if not id_column or not mm_per_pixel_column:
                print(f"[WARN] Aariz calibration: could not auto-detect id / mm-per-pixel "
                      f"columns in {os.path.basename(csv_path)} (columns={cols}). "
                      f"No mm calibration applied - pass id_column/mm_per_pixel_column. "
                      f"mm-based MRE/SDR will be skipped.")
                return {}
            for row in reader:
                key = (row.get(id_column) or '').strip()
                key = os.path.splitext(key)[0]  # tolerate '<id>.png' as well as '<id>'
                val = _parse_float(row.get(mm_per_pixel_column))
                if key and val:
                    mapping[key] = val
        print(f"[INFO] Aariz calibration: loaded {len(mapping)} per-image mm/px values "
              f"from {os.path.basename(csv_path)} "
              f"(id column={id_column!r}, mm/px column={mm_per_pixel_column!r}).")
        return mapping

    # ---- indexing ----------------------------------------------------------
    def _annotation_dir(self, split, annotator_dirname):
        return os.path.join(self.root, split, ANNOTATION_SUBDIR, annotator_dirname)

    def _resolve_image_path(self, split, ceph_id):
        base = os.path.join(self.root, split, CEPHALOGRAM_SUBDIR)
        for ext in IMAGE_EXTENSIONS:
            candidate = os.path.join(base, ceph_id + ext)
            if os.path.exists(candidate):
                return candidate
        # last resort: the id might already include an extension
        direct = os.path.join(base, ceph_id)
        return direct if os.path.exists(direct) else None

    @staticmethod
    def _read_landmarks(json_path):
        """Return {aariz_symbol: (x, y)} from one annotation JSON."""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        out = {}
        for lm in data.get('landmarks', []):
            symbol = lm.get('symbol')
            value = lm.get('value') or {}
            x, y = value.get('x'), value.get('y')
            if symbol is None or x is None or y is None:
                continue
            try:
                out[symbol] = (float(x), float(y))
            except (TypeError, ValueError):
                continue
        return out

    def _build_index(self, split):
        annotator_dirnames = ANNOTATOR_DIRS[self.annotator]
        primary_dir = self._annotation_dir(split, annotator_dirnames[0])
        if not os.path.isdir(primary_dir):
            self.skipped.append((f"<split {split}>", f"annotation dir not found: {primary_dir}"))
            return

        for fname in sorted(os.listdir(primary_dir)):
            if not fname.lower().endswith('.json'):
                continue
            ceph_id = os.path.splitext(fname)[0]

            # gather raw Aariz-symbol coords from each requested annotator, averaging
            per_annotator = []
            for dirname in annotator_dirnames:
                jp = os.path.join(self._annotation_dir(split, dirname), fname)
                if os.path.exists(jp):
                    per_annotator.append(self._read_landmarks(jp))
            if not per_annotator:
                self.skipped.append((ceph_id, "no annotation JSON for requested annotator"))
                continue

            image_path = self._resolve_image_path(split, ceph_id)
            if image_path is None:
                self.skipped.append((ceph_id, "cephalogram image not found"))
                continue

            # map symbols -> our names, averaging annotators where a symbol appears in >1
            landmarks = {}
            for aariz_symbol, target_name in self.name_mapping.items():
                pts = [a[aariz_symbol] for a in per_annotator if aariz_symbol in a]
                if not pts:
                    continue
                xs = sum(p[0] for p in pts) / len(pts)
                ys = sum(p[1] for p in pts) / len(pts)
                landmarks[target_name] = (xs, ys)

            if not landmarks:
                self.skipped.append((ceph_id, "no mappable landmarks on this image"))
                continue

            self.samples.append({
                'ceph_id': ceph_id,
                'split': split,
                'imagefile': os.path.basename(image_path),
                'image_path': image_path,
                'serial': ceph_id,       # patient/image identity; used by grouped reporting
                'image_index': 1,
                'landmarks': landmarks,
                'tracing_source': f'aariz:{self.annotator}',
                'pixel_spacing_mm': self.pixel_spacing_by_id.get(ceph_id),
            })

    # ---- Dataset API -------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]

        with Image.open(entry['image_path']) as img:
            image = np.array(img.convert('L'))  # single-channel grayscale, matching the training pipeline

        names = sorted(entry['landmarks'].keys())
        points = np.array([entry['landmarks'][name] for name in names], dtype=np.float32)

        sample = {
            'image': image,
            'landmarks': points,
            'landmark_names': names,
            'imagefile': entry['imagefile'],
            'serial': entry['serial'],
            'image_index': entry['image_index'],
            'tracing_source': entry['tracing_source'],
            'pixel_spacing_mm': entry['pixel_spacing_mm'],
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
