"""
================================================================================
REVIEWED TRACING SELECTOR & COMBINER
================================================================================
DESCRIPTION:
    Some Facad images were traced more than once (e.g. by different students,
    or re-traced after a correction). An expert reviewed each image and, in a
    CSV, recorded which of the tracings (<case> elements in the .fcd) are
    trustworthy for that image - by number, as a comma-separated list if more
    than one tracing should be used, or 'x' if the image should be excluded
    entirely.

    For a given image, "tracing number N" means the Nth <case> element (in
    document order) among the cases whose <xray_ref> matches that image's
    <idnr> - i.e. cases are numbered per-image, starting at 1.

    Review CSV format (semicolon-delimited, no header, trailing ';' allowed):
        <serial>_<image_index>_review.png;<selection>;
    where <selection> is 'x' (skip) or a comma-separated list of 1-based
    tracing numbers, e.g.:
        g9001_1_review.png;x;
        g9001_2_review.png;3;
        g9002_1_review.png;1,2,3,4;

    For each accepted image:
      - If only one tracing is selected, its marker coordinates are used as-is.
      - If several are selected, coordinates are combined per landmark: any
        landmark present in only one of the selected tracings is used as-is
        (this is expected - tracings are sometimes partial); a landmark
        present in more than one selected tracing is combined across those
        occurrences using COMBINE_METHOD ('median' or 'geomean').

    Landmark names are not restricted to a fixed whitelist: every <marker>
    found under a selected <case> (including markers nested inside <tooth>
    elements, e.g. Is/Isa/Ii/Iia) is picked up by name. This avoids silently
    dropping points if a tracing uses a different marker set than expected.

OUTPUT:
    A single CSV (comma-delimited) with one row per accepted image:
        imagefile, tracing_selection, <landmark>_x, <landmark>_y, ...

OPTIONS:
    - DEBUG (bool): if True, only logs what would happen, writes no CSV.
    - COMBINE_METHOD: 'median' (default, robust to a single bad click) or
      'geomean'.
================================================================================
"""

import os
import csv
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from facad_io import parse_fcd, get_cases_for_image, get_markers, combine_points, load_review_rows, resolve_serial_and_image_index

# ==================== CONFIGURATION ====================
REVIEW_CSV_PATH = r'N:\durable\Tracing-intelligently\data\kandidater\to_process\output\review_selection.csv'
FCD_DIR = r'N:\durable\Tracing-intelligently\data\kandidater\to_process\output\gold\data'
OUTPUT_CSV = r'N:\durable\Tracing-intelligently\data\kandidater\to_process\output\selected_landmarks.csv'

COMBINE_METHOD = 'median'  # 'median' or 'geomean'
DEBUG = False
# ========================================================


def process_review_csv():
    if not os.path.exists(REVIEW_CSV_PATH):
        print(f"[ERROR] Review CSV not found: {REVIEW_CSV_PATH}")
        return

    rows = load_review_rows(REVIEW_CSV_PATH)
    output_rows = []
    all_landmark_names = set()

    for image_filename, raw_selection, tracing_numbers in rows:
        if tracing_numbers is None:
            print(f"[SKIP] {image_filename}: excluded ('{raw_selection}')")
            continue

        serial, image_index = resolve_serial_and_image_index(image_filename)
        if serial is None:
            print(f"[WARN] {image_filename}: could not parse serial/image index, skipping")
            continue

        fcd_path = os.path.join(FCD_DIR, f"{serial}.fcd")
        if not os.path.exists(fcd_path):
            print(f"[MISSING] {fcd_path} (for {image_filename}), skipping")
            continue

        try:
            root = parse_fcd(fcd_path)
        except ET.ParseError as e:
            print(f"[ERROR] Could not parse {fcd_path}: {e}")
            continue

        cases = get_cases_for_image(root, image_index)
        if not cases:
            print(f"[WARN] {image_filename}: no tracings found for image index {image_index}, skipping")
            continue

        selected_cases = []
        for number in tracing_numbers:
            if number < 1 or number > len(cases):
                print(f"[WARN] {image_filename}: tracing number {number} out of range (found {len(cases)}), skipping that number")
                continue
            selected_cases.append(cases[number - 1])

        if not selected_cases:
            print(f"[WARN] {image_filename}: none of the requested tracing numbers were valid, skipping image")
            continue

        # Gather every occurrence of each landmark across the selected tracings.
        points_by_landmark = {}
        for case in selected_cases:
            for name, coord in get_markers(case).items():
                points_by_landmark.setdefault(name, []).append(coord)

        combined = {name: combine_points(pts, COMBINE_METHOD) for name, pts in points_by_landmark.items()}
        all_landmark_names.update(combined.keys())

        row = {'imagefile': image_filename, 'tracing_selection': raw_selection}
        for name, (x, y) in combined.items():
            row[f"{name}_x"] = round(x, 2)
            row[f"{name}_y"] = round(y, 2)
        output_rows.append(row)

        n_combined = sum(1 for pts in points_by_landmark.values() if len(pts) > 1)
        print(f"[OK] {image_filename}: tracings {[n for n in tracing_numbers]} -> "
              f"{len(combined)} landmarks ({n_combined} combined via {COMBINE_METHOD})")

    if DEBUG:
        print(f"\n[DRY-RUN] Would write {len(output_rows)} rows to {OUTPUT_CSV}")
        return

    if not output_rows:
        print("\nNo rows to write.")
        return

    landmark_headers = []
    for name in sorted(all_landmark_names):
        landmark_headers.extend([f"{name}_x", f"{name}_y"])
    headers = ['imagefile', 'tracing_selection'] + landmark_headers

    os.makedirs(os.path.dirname(OUTPUT_CSV) or '.', exist_ok=True)
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers, delimiter=',', extrasaction='ignore')
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"\nDone. {len(output_rows)} images written to {OUTPUT_CSV}")


if __name__ == "__main__":
    process_review_csv()
