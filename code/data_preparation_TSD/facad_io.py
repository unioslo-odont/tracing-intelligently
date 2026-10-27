"""
================================================================================
FACAD (.fcd) PARSING PRIMITIVES
================================================================================
DESCRIPTION:
    Shared low-level helpers for reading Facad .fcd tracing files: tolerant
    XML parsing, locating which <case> (tracing) elements belong to which
    <imagefile>, reading <marker> points from a case, combining points from
    more than one tracing of the same landmark, and reading the expert
    review-selection CSV format.

    Used by:
      - select_reviewed_tracings.py (expert-reviewed multi-tracing selection)
      - ../datasets/cephalometric_dataset.py (training data loading)

    Kept dependency-free (standard library only) so it can be imported from
    either a plain script or a torch Dataset without pulling in heavy
    dependencies transitively.
================================================================================
"""

import os
import csv
import re
import statistics
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict


def parse_fcd(fcd_path):
    """Load a .fcd file into an ElementTree root, tolerating the legacy
    Latin-1 encoding and malformed XML declarations seen in older exports."""
    with open(fcd_path, 'rb') as f:
        raw_bytes = f.read()
    xml_string = raw_bytes.decode('latin-1', errors='ignore')
    xml_string = re.sub(r'<\?xml[^>]+\?>', '', xml_string)
    return ET.fromstring(xml_string)


def get_cases_for_image(root, image_index):
    """Return the list of <case> elements that reference the Nth <imagefile>
    (1-based, document order), in document order. This defines the local
    'tracing number' used in the review CSV: tracing N for a given image is
    cases_for_image[N - 1]."""
    patient = root.find('patient')
    if patient is None:
        return None

    img_nodes = patient.findall('imagefile')
    if image_index < 1 or image_index > len(img_nodes):
        return None

    idnr = img_nodes[image_index - 1].findtext('idnr', '').strip()
    if not idnr:
        return None

    return [case for case in patient.findall('case') if case.findtext('xray_ref', '').strip() == idnr]


def get_image_pixel_spacing_mm(root, image_index):
    """Return the Facad-calibrated mm-per-pixel value for the Nth <imagefile>
    (1-based), read directly from its <scale> tag (<scaledef>calib_xray</scaledef>
    in the sample data - Facad's own calibration, not a raw DICOM PixelSpacing).

    NOTE: some .fcd files also carry <xrayfactorknown>/<xrayfactor> (percent
    magnification correction for x-ray beam divergence). This function does
    NOT apply that correction - it returns <scale> as-is. Verify with a known
    physical distance (e.g. a ruler/ball-bearing calibration object) before
    trusting this for mm-level clinical measurements; flagging rather than
    guessing since an ANB/SNA-style angle is scale-invariant but any Dist3p-
    style linear measurement or SDR-in-mm metric is not."""
    patient = root.find('patient')
    if patient is None:
        return None

    img_nodes = patient.findall('imagefile')
    if image_index < 1 or image_index > len(img_nodes):
        return None

    scale_text = img_nodes[image_index - 1].findtext('scale', '').strip()
    try:
        return float(scale_text)
    except ValueError:
        return None


def get_markers(case, name_map=None):
    """Return {marker_name: (x, y)} for every <marker> anywhere under a case,
    including ones nested inside <tooth> elements.

    name_map: optional {exact_spelling: canonical_spelling} dict (see
    build_canonical_case_map) applied to each name before it becomes a dict
    key - e.g. folds 'gN'/'gn' into 'Gn'. Without it (the default), names are
    used exactly as found in the .fcd, which is what every caller did before
    this parameter existed."""
    markers = {}
    for marker in case.findall('.//marker'):
        name = marker.findtext('name', '').strip()
        coord = marker.findtext('coord', '').strip()
        if not name or not coord:
            continue
        if name_map is not None:
            name = name_map.get(name, name)
        parts = coord.split()
        if len(parts) < 2:
            continue
        try:
            markers[name] = (float(parts[0]), float(parts[1]))
        except ValueError:
            continue
    return markers


# Names that must NEVER be case-folded together even though they collide
# case-insensitively, because they are genuinely distinct landmarks - a
# hard-tissue point and its soft-tissue analog, sharing a letter by
# convention rather than by typo. Confirmed against this project's clinical
# reference document: 'Pg' = hard-tissue
# Pogonion, 'PG' = soft-tissue Pogonion (Bloetvevspogonion) - a real,
# separate landmark, not an inconsistently-capitalized 'Pg'. Without this,
# build_canonical_case_map would silently fold 'PG' into 'Pg', contaminating
# the hard-tissue channel with soft-tissue coordinates for every image that
# used 'PG'.
PROTECTED_FROM_CASE_FOLDING = {'Pg', 'PG'}


def build_canonical_case_map(fcd_dir):
    """Scan every .fcd in fcd_dir once and build a case-insensitive alias
    map: {any_exact_spelling: canonical_spelling}, where the canonical
    spelling is whichever exact casing occurs most often within each
    case-insensitive group (e.g. 'Gn' beats 'gN'/'gn' by a wide margin, so
    all three map to 'Gn'). Names with only one casing map to themselves
    (a harmless no-op).

    Deliberately narrow: only names that are IDENTICAL except for case are
    grouped, and even then, names in PROTECTED_FROM_CASE_FOLDING are never
    merged with their case-variants (they map to themselves). This does NOT
    do fuzzy/typo matching either - e.g. 'iGo' and 'pGo' stay distinct,
    since collapsing anything looser than an exact case-insensitive match
    risks silently merging genuinely different cephalometric landmarks (see
    analyze_landmark_names.py, which found real short-name pairs like
    Is/Isa, Ii/Iia, Pg/Pgn that must NOT be merged despite looking similar)."""
    exact_counts = Counter()
    for fname in os.listdir(fcd_dir):
        if not fname.lower().endswith('.fcd'):
            continue
        try:
            root = parse_fcd(os.path.join(fcd_dir, fname))
        except Exception:
            continue
        patient = root.find('patient')
        if patient is None:
            continue
        for case in patient.findall('case'):
            for name in get_markers(case).keys():
                exact_counts[name] += 1

    by_lower = defaultdict(list)
    for name, count in exact_counts.items():
        by_lower[name.lower()].append((name, count))

    canonical_map = {}
    for variants in by_lower.values():
        protected = [nc for nc in variants if nc[0] in PROTECTED_FROM_CASE_FOLDING]
        foldable = [nc for nc in variants if nc[0] not in PROTECTED_FROM_CASE_FOLDING]

        for name, _ in protected:
            canonical_map[name] = name

        if foldable:
            canonical_name = max(foldable, key=lambda name_count: name_count[1])[0]
            for name, _ in foldable:
                canonical_map[name] = canonical_name
    return canonical_map


def combine_points(points, method='median'):
    """Combine a list of (x, y) points from different tracings of the same
    landmark into a single point. A single point is returned as-is."""
    if len(points) == 1:
        return points[0]

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    if method == 'median':
        return statistics.median(xs), statistics.median(ys)
    if method == 'geomean':
        return statistics.geometric_mean(xs), statistics.geometric_mean(ys)
    raise ValueError(f"Unknown combine method: {method}")


def fraction_landmarks_in_bounds(landmarks, width, height, margin=0.02):
    """Fraction of {name: (x, y)} landmark points that fall within the image
    rectangle [-m*W, (1+m)*W] x [-m*H, (1+m)*H]. Returns 1.0 (i.e. "cannot
    judge, don't flag") if there are no landmarks or the dimensions are
    unknown/zero.

    This is the guard against the multi-image frame-mismatch documented in
    observed in this cohort: many patients have the same cephalogram
    stored at several pixel sizes, and a tracing can end up paired with a
    DIFFERENT-sized image of the same patient - which displaces every landmark
    by exactly the image-size difference, so a large fraction land outside the
    actual image. Crucially this must be checked against the ACTUAL image
    dimensions (e.g. PIL Image.size), not the .fcd <imsize> field, since the
    coordinates typically still fit their own (wrong) <imsize> - and <imsize>
    is 0x0 for a chunk of the cohort anyway."""
    if not landmarks or not width or not height:
        return 1.0
    mx, my = margin * width, margin * height
    inside = sum(1 for (x, y) in landmarks.values()
                 if -mx <= x <= width + mx and -my <= y <= height + my)
    return inside / len(landmarks)


def load_excluded_images(path):
    """Read a list of image filenames to drop from a dataset. Accepts either a
    CSV with an 'imagefile' column (any other columns, e.g. 'reason', are
    ignored) or a plain-text file with one filename per line. Blank lines and
    '#' comments are ignored. Returns a set of filenames.

    Lets known-bad images be excluded WITHOUT editing the expert review CSV -
    e.g. tracings found to be mis-paired with a differently-sized copy of their
    image, pending a fixed re-extraction.

    A MISSING file is not an error: an exclusion list names images of one
    specific cohort, so it is local to that dataset (and git-ignored). A config
    that points at one you don't have simply excludes nothing, with a warning."""
    excluded = set()
    if not os.path.exists(path):
        print(f"[WARN] exclusion list not found: {path} - no images will be excluded. "
              "This list is dataset-specific; create your own or set "
              "data.exclude_images_path to null.")
        return excluded
    with open(path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
    header_is_csv = lines and 'imagefile' in lines[0].lower()
    if header_is_csv:
        for row in csv.DictReader(lines):
            name = (row.get('imagefile') or '').strip()
            if name and not name.startswith('#'):
                excluded.add(name)
    else:
        for line in lines:
            name = line.split(',')[0].strip()
            if name and not name.startswith('#'):
                excluded.add(name)
    return excluded


def get_case_landmarks(cases, method='median', name_map=None):
    """Given a list of already-selected <case> elements for one image,
    return {marker_name: (x, y)} combined across them with `combine_points`.
    A landmark present in only one of the cases is used as-is.

    name_map: optional case-canonicalization map, see get_markers/
    build_canonical_case_map."""
    points_by_landmark = {}
    for case in cases:
        for name, coord in get_markers(case, name_map=name_map).items():
            points_by_landmark.setdefault(name, []).append(coord)
    return {name: combine_points(pts, method) for name, pts in points_by_landmark.items()}


EXCLUDE_SELECTION_VALUES = {'x', 'slettet'}  # 'slettet' = Norwegian for "deleted" - seen in real review CSVs


def parse_selection(raw_selection):
    """Parse a review CSV selection field into a list of 1-based tracing
    numbers, or None if it means 'exclude this image' ('x', empty, or a
    known deletion marker - see EXCLUDE_SELECTION_VALUES).

    Getting this wrong is not just noise: an unrecognized exclusion marker
    falls through to load_review_rows' parse-error path, which SKIPS the
    row entirely rather than excluding the image - and if that image
    happens to have exactly one tracing in its .fcd, CephalometricDataset's
    fallback ('no review decision on record + single tracing => use it')
    would then silently include an image the reviewer explicitly marked
    unusable."""
    selection = raw_selection.strip().lower()
    if selection in EXCLUDE_SELECTION_VALUES or selection == '':
        return None
    numbers = []
    for part in selection.split(','):
        part = part.strip()
        if not part:
            continue
        numbers.append(int(part))
    return numbers


def _decode_text_file(path):
    """Decode a text file as UTF-8 (with or without BOM) if possible,
    falling back to Latin-1/cp1252 (same byte range) for CSVs saved by
    Windows tools in a legacy encoding - e.g. Excel/Notepad on a Norwegian
    locale, which is where a stray å/æ/ø usually comes from."""
    with open(path, 'rb') as f:
        raw_bytes = f.read()
    try:
        return raw_bytes.decode('utf-8-sig')
    except UnicodeDecodeError:
        return raw_bytes.decode('latin-1')


def load_review_rows(csv_path):
    """Read a review-selection CSV (semicolon-delimited, no header:
    '<serial>_<image_index>_review.png;<selection>;'). Returns a list of
    (image_filename, raw_selection, tracing_numbers_or_None)."""
    rows = []
    text = _decode_text_file(csv_path)
    reader = csv.reader(text.splitlines(), delimiter=';')
    for line_num, row in enumerate(reader, 1):
        if not row or not row[0].strip():
            continue
        image_filename = row[0].strip()
        raw_selection = row[1].strip() if len(row) > 1 else 'x'
        try:
            numbers = parse_selection(raw_selection)
        except ValueError:
            print(f"[WARN] line {line_num}: could not parse selection '{raw_selection}' for {image_filename}, skipping")
            continue
        rows.append((image_filename, raw_selection, numbers))
    return rows


def resolve_serial_and_image_index(image_filename):
    """'g9002_2_review.png' -> ('g9002', 2). Also handles filenames without
    the '_review' suffix, e.g. 'g9002_2.tif' -> ('g9002', 2)."""
    stem = os.path.splitext(image_filename)[0]
    if stem.endswith('_review'):
        stem = stem[:-len('_review')]
    serial, _, idx_str = stem.rpartition('_')
    if not serial or not idx_str.isdigit():
        return None, None
    return serial, int(idx_str)
