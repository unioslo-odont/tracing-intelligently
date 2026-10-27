"""
================================================================================
CEPHALOMETRIC LANDMARK DATASET
================================================================================
DESCRIPTION:
    torch.utils.data.Dataset over a gold/silver-style TSD export (as produced
    by data_preparation_TSD/create_anonyzed_base_2.py): an images/ folder of
    "<serial>_<image_index>.<ext>" files and a data/ folder of "<serial>.fcd"
    tracing files.

    Landmark resolution per image (in priority order):
      1. If a review-selection CSV is given (the format produced/consumed by
         select_reviewed_tracings.py) and the image is listed there, use that
         row's tracing selection - 'x' excludes the image, a comma-separated
         list selects and combines tracings (median/geomean).
      2. Otherwise, if the .fcd has exactly one tracing for that image, use
         it directly - unambiguous, no review decision needed.
      3. Otherwise (multiple tracings and no review decision on record), the
         image is EXCLUDED. Picking a tracing arbitrarily would silently bake
         an unreviewed choice into training data. Excluded images are listed
         in `dataset.skipped` with the reason, so they can be reviewed and
         added to the review CSV.

    This dataset only does I/O: it returns the raw image array plus the
    landmark points and their names. Heatmap target encoding, normalization,
    resizing and augmentation are deliberately left to an optional
    `transform` callable so they can be swapped independently (per-landmark
    heatmap sigma, CLAHE, resolution, etc. are training-config concerns, not
    data-loading concerns).
================================================================================
"""

import os
import re
import sys

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data_preparation_TSD')))
from facad_io import (
    parse_fcd,
    get_cases_for_image,
    get_case_landmarks,
    get_image_pixel_spacing_mm,
    load_review_rows,
    resolve_serial_and_image_index,
    build_canonical_case_map,
    fraction_landmarks_in_bounds,
    load_excluded_images,
)

IMAGE_FILENAME_RE = re.compile(r'^(?P<serial>[A-Za-z]\d+)_(?P<image_index>\d+)\.(tif|tiff|png)$', re.IGNORECASE)


class TransformedView(Dataset):
    """Apply a transform to a base dataset that returns RAW sample dicts (i.e. a
    CephalometricDataset built without a transform). Lets the train split use an
    AUGMENTING transform and the val split a plain one while sharing a single
    parsed base dataset - build the base once, wrap it twice."""

    def __init__(self, base_dataset, transform):
        self.base = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        return self.transform(self.base[i])


class CephalometricDataset(Dataset):
    def __init__(self, images_dir, fcd_dir, review_csv_path=None, combine_method='median', transform=None,
                 min_landmarks_in_bounds=0.9, exclude_images_path=None):
        """
        images_dir: folder of "<serial>_<image_index>.<ext>" images (e.g. gold/images).
        fcd_dir: folder of "<serial>.fcd" tracing files (e.g. gold/data).
        review_csv_path: optional expert review-selection CSV (see facad_io.load_review_rows).
        combine_method: 'median' or 'geomean', used when a review selection lists more than one tracing.
        transform: optional callable applied to each sample dict before it is returned.
        min_landmarks_in_bounds: an image is EXCLUDED if fewer than this fraction of its landmarks
            fall within the ACTUAL image dimensions (see facad_io.fraction_landmarks_in_bounds).
            This catches the multi-image frame-mismatch bug (coordinates paired to a different-sized
            image of the same patient, displacing every landmark by the image-size difference - see
            a rigid offset equal to the image-size difference). Set to 0.0 to disable the check.
        exclude_images_path: optional path to an explicit exclusion list (CSV with an 'imagefile'
            column, or one filename per line - see facad_io.load_excluded_images). Any listed image
            is dropped. Used for the deterministic frame-mismatch exclusion in
            configs/excluded_images.csv - catches even the moderate mismatches the geometric
            min_landmarks_in_bounds check can miss (offset small enough that landmarks stay in bounds).
        """
        self.images_dir = images_dir
        self.fcd_dir = fcd_dir
        self.combine_method = combine_method
        self.transform = transform
        self.min_landmarks_in_bounds = min_landmarks_in_bounds
        self.exclude_images = load_excluded_images(exclude_images_path) if exclude_images_path else set()

        self.review_by_key = {}
        if review_csv_path is not None:
            for image_filename, raw_selection, tracing_numbers in load_review_rows(review_csv_path):
                serial, image_index = resolve_serial_and_image_index(image_filename)
                if serial is not None:
                    self.review_by_key[(serial, image_index)] = (raw_selection, tracing_numbers)

        # Built once, up front: folds marker names that differ only by case
        # (e.g. 'gN'/'gn' -> 'Gn') so a tracing's inconsistent capitalization
        # doesn't create a separate landmark channel - see
        # facad_io.build_canonical_case_map and analyze_landmark_names.py.
        self.name_map = build_canonical_case_map(fcd_dir)

        self.samples = []
        self.skipped = []  # list of (image_filename, reason)
        self._build_index()

    def _build_index(self):
        image_filenames = sorted(f for f in os.listdir(self.images_dir) if IMAGE_FILENAME_RE.match(f))

        for image_filename in image_filenames:
            if image_filename in self.exclude_images:
                self.skipped.append((image_filename, "on the explicit exclusion list (exclude_images_path)"))
                continue

            match = IMAGE_FILENAME_RE.match(image_filename)
            serial = match.group('serial')
            image_index = int(match.group('image_index'))

            fcd_path = os.path.join(self.fcd_dir, f"{serial}.fcd")
            if not os.path.exists(fcd_path):
                self.skipped.append((image_filename, f"no matching .fcd file ({serial}.fcd not found)"))
                continue

            try:
                root = parse_fcd(fcd_path)
            except Exception as e:
                self.skipped.append((image_filename, f"could not parse {fcd_path}: {e}"))
                continue

            cases = get_cases_for_image(root, image_index)
            if not cases:
                self.skipped.append((image_filename, "no tracings found for this image in its .fcd file"))
                continue

            review_entry = self.review_by_key.get((serial, image_index))
            if review_entry is not None:
                raw_selection, tracing_numbers = review_entry
                if tracing_numbers is None:
                    self.skipped.append((image_filename, f"excluded by review CSV ('{raw_selection}')"))
                    continue
                selected_cases = [cases[n - 1] for n in tracing_numbers if 1 <= n <= len(cases)]
                if not selected_cases:
                    self.skipped.append((image_filename, f"review selection '{raw_selection}' had no valid tracing numbers"))
                    continue
                source = f"reviewed:{raw_selection}"
            elif len(cases) == 1:
                selected_cases = cases
                source = "single"
            else:
                self.skipped.append((image_filename, f"{len(cases)} tracings found, none selected by a review CSV - ambiguous"))
                continue

            landmarks = get_case_landmarks(selected_cases, self.combine_method, name_map=self.name_map)
            if not landmarks:
                self.skipped.append((image_filename, "no markers found in the selected tracing(s)"))
                continue

            image_path = os.path.join(self.images_dir, image_filename)

            # Frame-mismatch guard: verify the landmarks actually fall within THIS image. Many
            # patients have the same cephalogram at several pixel sizes, and a tracing can be paired
            # with a different-sized image of the same patient, displacing every landmark by the
            # image-size difference (see
            # facad_io.fraction_landmarks_in_bounds). Checked against the real image size (read from
            # the file header), not the unreliable .fcd <imsize>.
            if self.min_landmarks_in_bounds > 0.0:
                try:
                    with Image.open(image_path) as im:
                        img_w, img_h = im.size
                except Exception as e:
                    self.skipped.append((image_filename, f"could not read image size: {e}"))
                    continue
                in_bounds = fraction_landmarks_in_bounds(landmarks, img_w, img_h)
                if in_bounds < self.min_landmarks_in_bounds:
                    self.skipped.append((image_filename,
                        f"{(1 - in_bounds) * 100:.0f}% of landmarks fall outside the {img_w}x{img_h} image - "
                        f"coordinates likely belong to a different-sized image of this patient "
                        f"(frame mismatch)"))
                    continue

            self.samples.append({
                'imagefile': image_filename,
                'image_path': image_path,
                'serial': serial,
                'image_index': image_index,
                'landmarks': landmarks,
                'tracing_source': source,
                'pixel_spacing_mm': get_image_pixel_spacing_mm(root, image_index),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]

        with Image.open(entry['image_path']) as img:
            image = np.array(img)

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
