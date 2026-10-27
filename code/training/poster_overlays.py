"""
================================================================================
POSTER OVERLAYS - reference vs. predicted landmarks on a few selected images
================================================================================
DESCRIPTION:
    Print-ready overlays for hand-picked example images, drawn from the
    coordinates an evaluation already saved (eval_*_per_landmark_image.csv).
    No checkpoint, GPU or torch is needed, and the overlay shows exactly the
    predictions the reported metrics were computed from.

    Differences from the validation overlays (visualize.py):
      - colour-blind-safe Okabe-Ito pair with distinct shapes: reference =
        blue open circle, prediction = orange cross. The circle radius is a
        physical distance (default 2 mm), so a cross inside its circle reads
        directly as "within 2 mm".
      - only the requested landmark subset (default: the 12 scored in both the
        internal and the Aariz evaluation), cropped to those landmarks plus a
        margin in mm, contrast-stretched on the crop, with a mm scale bar.
      - sized in print units (--width-cm, points) and written as PNG and as
        PDF/SVG with vector markers and labels (each label its own group, so it
        can be nudged in Inkscape/Illustrator; the halo turns its glyphs into
        outlines), plus a separate legend with editable text for a key shared
        between panels.
      - nothing identifying is drawn into the image. manifest.csv beside the
        figures records each image's source CSV, crop box and the errors of
        the points shown, and run_meta.json the commit.

    Each requested image must appear in exactly one predictions CSV: for
    internal cross-validation pass every fold's eval_val CSV, and the image's
    single out-of-fold prediction is used. Before drawing it also checks that
    every shown landmark has a reference point, that the stored error_mm
    matches the coordinates and pixel spacing, and that all points fall
    inside the actual image (the frame-mismatch guard).

USAGE:
    # internal: out-of-fold predictions of the fold models
    python code/training/poster_overlays.py \
        --predictions "outputs/run10_15pt/unet_baseline_15pt/fold*/eval_val_per_landmark_image.csv" \
        --images-dir /path/to/data/images --images-file <candidates>.csv \
        --output-dir outputs/poster/example_overlays/internal

    # external: the Aariz test split
    python code/training/poster_overlays.py \
        --predictions outputs/run10_15pt/aariz/unet_baseline_15pt_eval_aariz_per_landmark_image.csv \
        --images-dir /path/to/Aariz/test/Cephalograms --images <ceph_id>.png \
        --output-dir outputs/poster/example_overlays/external

    Use the same --width-cm/--margin-mm/--aspect for both sets so the panels
    share one physical scale, and set --width-cm to the printed panel width so
    --font-pt and the marker sizes are true print sizes.
================================================================================
"""

import argparse
import csv
import glob
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from provenance import write_run_provenance

REFERENCE_COLOR = '#0072B2'   # Okabe-Ito blue
PREDICTION_COLOR = '#E69F00'  # Okabe-Ito orange

# Scored in both the internal and the Aariz evaluation (Gn is unmapped on Aariz;
# Pt, iGo and pGo are not annotated there).
DEFAULT_LANDMARKS = ['S', 'N', 'Or', 'Po', 'Sp', 'A', 'B', 'Pg', 'Is', 'Isa', 'Ii', 'Iia']

# Label direction per landmark for a profile facing RIGHT: (+x anterior, +y superior).
# Mirrored automatically for a left-facing profile; override with --label-dir.
# Chosen so the crowded incisor/chin neighbours (Sp/A, Is/Ii, B/Pg) separate.
LABEL_DIRECTIONS = {
    'S': (-0.7, 0.7), 'N': (0.7, 0.7), 'Or': (0.0, 1.0), 'Po': (-0.7, 0.7),
    'Sp': (0.9, 0.45), 'A': (0.9, -0.5), 'Is': (0.8, -0.6), 'Isa': (-1.0, 0.0),
    'Ii': (-0.9, 0.45), 'Iia': (-1.0, 0.0), 'B': (1.0, 0.0), 'Pg': (0.8, -0.6),
    'Gn': (0.0, -1.0), 'Pt': (-0.7, 0.7), 'iGo': (-0.7, -0.7), 'pGo': (-1.0, 0.0),
}


def read_image_list(images, images_file):
    """Same input forms as visualize_images.py (not imported: it pulls in torch)."""
    if images:
        return list(images)
    with open(images_file, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
    if lines and 'imagefile' in lines[0].lower():
        return [r['imagefile'].strip() for r in csv.DictReader(lines) if (r.get('imagefile') or '').strip()]
    return [ln.split(',')[0].strip() for ln in lines if ln.strip() and not ln.startswith('#')]


def expand_paths(patterns):
    """Expand globs here, so quoted patterns work from PowerShell too."""
    paths = []
    for pattern in patterns:
        matched = sorted(glob.glob(pattern))
        if not matched:
            raise FileNotFoundError(f"No predictions CSV matches {pattern!r}")
        paths.extend(matched)
    return paths


def load_predictions(csv_paths, imagefiles, landmarks):
    """Per requested image: its source CSV, pixel spacing, profile direction and
    {landmark: (pred_xy, gt_xy, error_mm)} for `landmarks`, after the checks
    described in the module docstring (except bounds, which needs the image)."""
    wanted = set(imagefiles)
    rows_by_image = {}
    for path in csv_paths:
        with open(path, 'r', encoding='utf-8', newline='') as f:
            for row in csv.DictReader(f):
                if row['imagefile'] in wanted:
                    rows_by_image.setdefault(row['imagefile'], {}).setdefault(path, []).append(row)

    records = {}
    for imagefile in imagefiles:
        by_source = rows_by_image.get(imagefile, {})
        if not by_source:
            raise ValueError(f"{imagefile}: not in any predictions CSV")
        if len(by_source) > 1:
            raise ValueError(
                f"{imagefile}: in {len(by_source)} predictions CSVs {sorted(by_source)} - pass only "
                "held-out (eval_val) CSVs so each image has exactly one out-of-fold prediction")
        (source, rows), = by_source.items()
        by_landmark = {row['landmark']: row for row in rows}
        if len(by_landmark) != len(rows):
            raise ValueError(f"{imagefile}: duplicate landmark rows in {source}")

        spacings = {row['pixel_spacing_mm'] for row in rows}
        if len(spacings) != 1 or not next(iter(spacings)):
            raise ValueError(f"{imagefile}: needs one calibrated pixel spacing, got {sorted(spacings)}")
        spacing = float(next(iter(spacings)))

        points = {}
        for name in landmarks:
            row = by_landmark.get(name)
            if row is None or row['present'] != '1' or not row['gt_x_px']:
                raise ValueError(f"{imagefile}: {name} has no reference point in {source}")
            pred = np.array([float(row['pred_x_px']), float(row['pred_y_px'])])
            gt = np.array([float(row['gt_x_px']), float(row['gt_y_px'])])
            error_mm = float(row['error_mm'])
            recomputed = float(np.linalg.norm(pred - gt)) * spacing
            if not np.isclose(recomputed, error_mm, rtol=1e-4, atol=1e-3):
                raise ValueError(f"{imagefile}: {name} stored error {error_mm:.4f} mm but coordinates "
                                 f"and spacing give {recomputed:.4f} mm")
            points[name] = (pred, gt, error_mm)

        # Profile direction from the predicted S and N, which every model outputs.
        if 'S' in by_landmark and 'N' in by_landmark:
            facing = 1 if float(by_landmark['N']['pred_x_px']) > float(by_landmark['S']['pred_x_px']) else -1
        else:
            facing = 1
        records[imagefile] = {'source': source, 'spacing': spacing, 'facing': facing, 'points': points}
    return records


def load_image(path):
    """Grayscale array at native bit depth, as the datasets load it (16-bit TIFFs stay 16-bit)."""
    with Image.open(path) as img:
        if img.mode not in ('L', 'F') and not img.mode.startswith('I'):
            img = img.convert('L')
        return np.array(img)


def crop_box(xy, spacing, margin_mm, aspect, width, height, crop):
    """Integer (x0, y0, x1, y1): the points' bounding box plus margin, widened or
    heightened to `aspect` (width/height), shifted into and clamped to the image."""
    if not crop:
        return 0, 0, width, height
    x0, y0 = xy.min(axis=0) - margin_mm / spacing
    x1, y1 = xy.max(axis=0) + margin_mm / spacing
    if aspect:
        w, h = x1 - x0, y1 - y0
        if w / h < aspect:
            x0, x1 = x0 - (h * aspect - w) / 2, x1 + (h * aspect - w) / 2
        else:
            y0, y1 = y0 - (w / aspect - h) / 2, y1 + (w / aspect - h) / 2

    def fit(lo, hi, size):
        if lo < 0:
            lo, hi = 0, hi - lo
        if hi > size:
            lo, hi = lo - (hi - size), size
        return max(0, int(np.floor(lo))), min(size, int(np.ceil(hi)))

    x0, x1 = fit(x0, x1, width)
    y0, y1 = fit(y0, y1, height)
    return x0, y0, x1, y1


def halo(color, linewidth):
    return [pe.Stroke(linewidth=linewidth, foreground=color), pe.Normal()]


def text_alignment(dx, dy):
    ha = 'left' if dx > 0.3 else 'right' if dx < -0.3 else 'center'
    va = 'bottom' if dy > 0.3 else 'top' if dy < -0.3 else 'center'
    return ha, va


def render(image, record, landmarks, box, args, label_dirs):
    """One figure, axes filling it, in image pixel coordinates (pixel centres on integers,
    as in visualize.py). Returns (figure, image artist, reference radius in points)."""
    x0, y0, x1, y1 = box
    spacing, facing, points = record['spacing'], record['facing'], record['points']
    crop = image[y0:y1, x0:x1]
    vmin, vmax = np.percentile(crop, [1, 99])

    width_in = args.width_cm / 2.54
    fig = plt.figure(figsize=(width_in, width_in * (y1 - y0) / (x1 - x0)))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    image_artist = ax.imshow(crop, cmap='gray', vmin=vmin, vmax=vmax,
                             extent=(x0 - 0.5, x1 - 0.5, y1 - 0.5, y0 - 0.5))
    ax.set_xlim(x0 - 0.5, x1 - 0.5)
    ax.set_ylim(y1 - 0.5, y0 - 0.5)

    points_per_px = width_in * 72 / (x1 - x0)
    radius_px = args.ref_radius_mm / spacing
    radius_pt = radius_px * points_per_px
    label_offset_pt = max(radius_pt, args.cross_pt / 2) + 2.5

    for name in landmarks:
        pred, gt, _ = points[name]
        if args.error_lines:
            ax.plot([gt[0], pred[0]], [gt[1], pred[1]], color=PREDICTION_COLOR,
                    linewidth=0.75 * args.line_pt, zorder=2)
        ax.add_patch(Circle(gt, radius_px, fill=False, edgecolor=REFERENCE_COLOR, linewidth=args.line_pt,
                            path_effects=halo('white', args.line_pt + 1.5), zorder=3))
        ax.plot(pred[0], pred[1], linestyle='none', marker='+', markersize=args.cross_pt,
                markeredgewidth=args.line_pt + 0.3, color=PREDICTION_COLOR,
                path_effects=halo('black', args.line_pt + 1.8), zorder=4)

        dx, dy = label_dirs.get(name, (1.0, 0.0))
        dx *= facing
        norm = np.hypot(dx, dy) or 1.0
        ha, va = text_alignment(dx / norm, dy / norm)
        ax.annotate(name, xy=gt, xytext=(dx / norm * label_offset_pt, dy / norm * label_offset_pt),
                    textcoords='offset points', ha=ha, va=va, fontsize=args.font_pt, color='white',
                    path_effects=halo('black', 2.5), zorder=5)

    if args.scale_bar_mm:
        # Bottom corner in front of the chin, where the radiograph is usually soft tissue or air.
        length_px = args.scale_bar_mm / spacing
        inset_px = 4 / spacing
        bar_y = y1 - 0.5 - inset_px
        bar_x = x1 - 0.5 - inset_px - length_px if facing > 0 else x0 - 0.5 + inset_px
        ax.plot([bar_x, bar_x + length_px], [bar_y, bar_y], color='white', linewidth=2.5,
                solid_capstyle='butt', path_effects=halo('black', 4.5), zorder=5)
        ax.annotate(f'{args.scale_bar_mm:g} mm', xy=(bar_x + length_px / 2, bar_y), xytext=(0, 3),
                    textcoords='offset points', ha='center', va='bottom', fontsize=args.font_pt,
                    color='white', path_effects=halo('black', 2.5), zorder=5)
    return fig, image_artist, radius_pt


def save_figure(fig, image_artist, stem, args):
    written = []
    for fmt in args.formats:
        # Vector formats embed the crop unresampled; PNG is resampled to --dpi.
        image_artist.set_interpolation('none' if fmt in ('pdf', 'svg') else 'antialiased')
        path = os.path.join(args.output_dir, f'{stem}.{fmt}')
        fig.savefig(path, dpi=args.dpi)
        written.append(path)
    plt.close(fig)
    return written


def save_legend(radius_pt, args):
    reference_label = args.reference_label or f'Reference (circle radius {args.ref_radius_mm:g} mm)'
    handles = [
        Line2D([], [], linestyle='none', marker='o', markersize=2 * radius_pt, markerfacecolor='none',
               markeredgecolor=REFERENCE_COLOR, markeredgewidth=args.line_pt, label=reference_label),
        Line2D([], [], linestyle='none', marker='+', markersize=args.cross_pt,
               markeredgewidth=args.line_pt + 0.3, color=PREDICTION_COLOR, label=args.prediction_label),
    ]
    fig = plt.figure(figsize=(args.width_cm / 2.54, 0.5))
    fig.legend(handles=handles, loc='center', ncol=2, frameon=False, fontsize=args.font_pt,
               handletextpad=0.4, columnspacing=1.5)
    for fmt in args.formats:
        fig.savefig(os.path.join(args.output_dir, f'legend.{fmt}'), dpi=args.dpi,
                    bbox_inches='tight', pad_inches=0.02, transparent=True)
    plt.close(fig)


def parse_label_dirs(overrides):
    label_dirs = dict(LABEL_DIRECTIONS)
    for item in overrides or []:
        name, _, value = item.partition('=')
        try:
            dx, dy = (float(v) for v in value.split(','))
        except ValueError:
            raise SystemExit(f"--label-dir expects NAME=dx,dy (e.g. Is=1,-1), got {item!r}")
        label_dirs[name] = (dx, dy)
    return label_dirs


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('USAGE:')[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--predictions', nargs='+', required=True,
                        help="eval_*_per_landmark_image.csv path(s) or quoted glob(s)")
    parser.add_argument('--images-dir', required=True, help="folder holding the image files named in the CSVs")
    parser.add_argument('--images', nargs='+', default=None, help="explicit image filenames")
    parser.add_argument('--images-file', default=None, help="CSV with an 'imagefile' column, or one filename per line")
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--landmarks', nargs='+', default=DEFAULT_LANDMARKS,
                        help="landmarks to draw (default: the 12 scored in both datasets)")
    parser.add_argument('--formats', nargs='+', default=['png', 'pdf', 'svg'], choices=['png', 'pdf', 'svg'])
    parser.add_argument('--width-cm', type=float, default=15.0, help="printed panel width")
    parser.add_argument('--dpi', type=int, default=300, help="PNG resolution")
    parser.add_argument('--margin-mm', type=float, default=15.0, help="crop margin around the landmarks")
    parser.add_argument('--aspect', type=float, default=None, help="crop width/height, e.g. 0.85 (default: as cropped)")
    parser.add_argument('--no-crop', action='store_true', help="draw the full radiograph")
    parser.add_argument('--ref-radius-mm', type=float, default=2.0, help="reference circle radius")
    parser.add_argument('--cross-pt', type=float, default=9.0, help="prediction cross size")
    parser.add_argument('--line-pt', type=float, default=1.5, help="marker line width")
    parser.add_argument('--font-pt', type=float, default=11.0)
    parser.add_argument('--font-family', default=None, help="e.g. Arial, to match the poster")
    parser.add_argument('--scale-bar-mm', type=float, default=10.0, help="0 = no scale bar")
    parser.add_argument('--error-lines', action='store_true', help="join each reference to its prediction")
    parser.add_argument('--label-dir', nargs='+', default=None, metavar='NAME=dx,dy',
                        help="label direction override, right-facing frame (+x anterior, +y up)")
    parser.add_argument('--reference-label', default=None)
    parser.add_argument('--prediction-label', default='Model prediction')
    args = parser.parse_args()

    if not args.images and not args.images_file:
        parser.error("give --images or --images-file")
    plt.rcParams.update({'svg.fonttype': 'none', 'pdf.fonttype': 42})  # un-haloed text (legend) stays text
    if args.font_family:
        plt.rcParams['font.family'] = [args.font_family]

    imagefiles = read_image_list(args.images, args.images_file)
    csv_paths = expand_paths(args.predictions)
    records = load_predictions(csv_paths, imagefiles, args.landmarks)
    label_dirs = parse_label_dirs(args.label_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    manifest, radii_pt = [], []
    for imagefile in imagefiles:
        record = records[imagefile]
        image_path = os.path.join(args.images_dir, imagefile)
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"{imagefile}: no such file in {args.images_dir}")
        image = load_image(image_path)
        height, width = image.shape[:2]

        all_xy = np.array([xy for name in args.landmarks for xy in record['points'][name][:2]])
        if (all_xy < 0).any() or (all_xy[:, 0] >= width).any() or (all_xy[:, 1] >= height).any():
            raise ValueError(f"{imagefile}: points fall outside the {width}x{height} image - tracing "
                             "and image frames disagree (see configs/excluded_images.csv)")

        box = crop_box(all_xy, record['spacing'], args.margin_mm, args.aspect, width, height, not args.no_crop)
        fig, image_artist, radius_pt = render(image, record, args.landmarks, box, args, label_dirs)
        stem = os.path.splitext(imagefile)[0]
        save_figure(fig, image_artist, stem, args)
        radii_pt.append(radius_pt)

        errors = np.array([record['points'][name][2] for name in args.landmarks])
        x0, y0, x1, y1 = box
        manifest.append({
            'imagefile': imagefile, 'predictions_csv': record['source'], 'n_landmarks': len(errors),
            'mre_mm': errors.mean(), 'median_mm': np.median(errors), 'max_mm': errors.max(),
            'worst_landmark': args.landmarks[int(errors.argmax())],
            'n_within_ref_radius': int((errors <= args.ref_radius_mm).sum()),
            'pixel_spacing_mm': record['spacing'], 'facing': 'right' if record['facing'] > 0 else 'left',
            'crop_x0': x0, 'crop_y0': y0, 'crop_x1': x1, 'crop_y1': y1,
            'crop_w_mm': (x1 - x0) * record['spacing'], 'crop_h_mm': (y1 - y0) * record['spacing'],
        })
        print(f"{stem}: MRE {errors.mean():.2f} mm over {len(errors)} landmarks -> {args.output_dir}")

    save_legend(float(np.mean(radii_pt)), args)
    with open(os.path.join(args.output_dir, 'manifest.csv'), 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    write_run_provenance(args.output_dir, vars(args), script='poster_overlays.py',
                         predictions_csvs=csv_paths, landmarks=args.landmarks)
    print(f"Wrote {len(manifest)} overlays, legend and manifest.csv to {args.output_dir}")


if __name__ == '__main__':
    main()
