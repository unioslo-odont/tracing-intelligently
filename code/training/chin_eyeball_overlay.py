"""
================================================================================
CHIN EYEBALL TEST - is our `Gn` Menton or Gnathion?
================================================================================
DESCRIPTION:
    Renders the chin-region landmarks on a sample of cephalograms from EITHER
    the internal Facad cohort OR the Aariz dataset, cropped to the chin and
    labelled, so an orthodontist can judge by eye where each point sits.

    The question this answers: our archive has a point labelled `Gn` whose
    stored definition ("lowest point of the mandibular symphysis") is actually
    the definition of MENTON. Is it Menton, or the antero-inferior Gnathion?

    Shared colour scheme across BOTH datasets, so the two montages can be
    compared side by side:
        RED    = the point labelled "Gn"   (our Gn  /  Aariz Gn)   <- the point in question
        YELLOW = Menton                    (our Me where traced /  Aariz Me)
        GREEN  = Pogonion                  (our Pg  /  Aariz Pog)
        CYAN   = our Pgn (facial-axis chin point; internal only)

    Aariz annotates BOTH Gn and Me on every image, so the Aariz montage is the
    reference: it shows what "Gn" (red, mid-chin, constructed) vs "Me" (yellow,
    bottom) look like. Then on OUR montage the expert asks: does our RED (Gn)
    sit at the bottom (like Aariz's YELLOW/Menton) or mid-chin (like Aariz's
    RED/Gnathion)? On the ~34 of our images that also carry a Me marker, red
    (Gn) vs yellow (Me) can be compared directly.

    Pure I/O + drawing - no model, no GPU. Runs on a login node.

USAGE (on Fox):
    # our data (prioritises images that also have a Me marker):
    python chin_eyeball_overlay.py --dataset internal \
        --config configs/unet_baseline.yaml --n 20 --out outputs/chin_eyeball/internal

    # Aariz reference:
    python chin_eyeball_overlay.py --dataset aariz \
        --aariz-root /path/to/project/data/Aariz --split all --n 20 \
        --out outputs/chin_eyeball/aariz
================================================================================
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'datasets')))

# colour per concept - shared across datasets
COLORS = {
    'Gn':  (255, 45, 45),    # RED   - the point in question
    'Me':  (255, 225, 0),    # YELLOW- Menton
    'Pg':  (40, 220, 70),    # GREEN - Pogonion (our code)
    'Pog': (40, 220, 70),    # GREEN - Pogonion (Aariz code)
    'Pgn': (0, 200, 255),    # CYAN  - facial-axis chin point (ours)
}
INTERNAL_CHIN = ['Gn', 'Me', 'Pg', 'Pgn']
AARIZ_CHIN = ['Gn', 'Me', 'Pog']


def to_rgb(arr):
    """2-D image array (8- or 16-bit) -> contrast-stretched RGB PIL image."""
    a = arr.astype(np.float32)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    if hi <= lo:
        hi = a.max() or 1.0
        lo = a.min()
    a = np.clip((a - lo) / (hi - lo), 0, 1) * 255
    return Image.fromarray(a.astype(np.uint8)).convert('RGB')


def _font(size):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def draw_and_crop(image_rgb, landmarks, draw_order, image_id):
    """Draw the chin landmarks present in `landmarks` and return a cropped,
    labelled RGB PIL image centred on the chin. `landmarks` is {name:(x,y)}."""
    pil = image_rgb.copy()
    draw = ImageDraw.Draw(pil)
    present = [(n, landmarks[n]) for n in draw_order if n in landmarks]
    if not present:
        return None

    # scale marker/label size to the image so it reads on 900px and 2500px images alike
    S = max(pil.width, pil.height)
    r = max(5, int(S * 0.006))
    fs = max(12, int(S * 0.014))
    font = _font(fs)

    for name, (x, y) in present:
        col = COLORS.get(name, (255, 255, 255))
        draw.ellipse((x - r, y - r, x + r, y + r), outline=(0, 0, 0), width=max(2, r // 3))
        draw.ellipse((x - r + 1, y - r + 1, x + r - 1, y + r - 1), outline=col, width=max(2, r // 3))
        draw.line((x - r, y, x + r, y), fill=col, width=2)
        draw.line((x, y - r, x, y + r), fill=col, width=2)
        label = name
        lx, ly = x + r + 3, y - fs // 2
        for dx in (-1, 1):
            for dy in (-1, 1):
                draw.text((lx + dx, ly + dy), label, fill=(0, 0, 0), font=font)
        draw.text((lx, ly), label, fill=col, font=font)

    xs = [p[0] for _, p in present]
    ys = [p[1] for _, p in present]
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    span = max(max(xs) - min(xs), max(ys) - min(ys), S * 0.06)
    pad = span * 1.9  # generous context so the whole chin outline is visible
    L = int(max(0, cx - pad)); R = int(min(pil.width, cx + pad))
    T = int(max(0, cy - pad)); B = int(min(pil.height, cy + pad))
    crop = pil.crop((L, T, R, B))

    # id caption bar
    cap_h = max(16, int(crop.width * 0.05))
    canvas = Image.new('RGB', (crop.width, crop.height + cap_h), (20, 20, 20))
    canvas.paste(crop, (0, 0))
    d = ImageDraw.Draw(canvas)
    d.text((4, crop.height + 1), image_id, fill=(235, 235, 235), font=_font(max(11, cap_h - 5)))
    return canvas


def montage(tiles, cols=5, tile_w=360):
    """Grid-tile the labelled crops into one contact sheet."""
    if not tiles:
        return None
    resized = []
    for t in tiles:
        h = int(t.height * (tile_w / t.width))
        resized.append(t.resize((tile_w, h)))
    tile_h = max(t.height for t in resized)
    rows = (len(resized) + cols - 1) // cols
    sheet = Image.new('RGB', (cols * tile_w, rows * tile_h), (10, 10, 10))
    for i, t in enumerate(resized):
        rr, cc = divmod(i, cols)
        sheet.paste(t, (cc * tile_w, rr * tile_h))
    return sheet


def samples_internal(config_path, n, seed):
    from config import load_config
    from cephalometric_dataset import CephalometricDataset
    data = load_config(config_path)['data']
    # Relative exclude_images_path is resolved against the config's own dir, exactly
    # as train.py/evaluate do (train.py:146-150) - otherwise it's looked up in the CWD.
    config_dir = os.path.dirname(os.path.abspath(config_path))
    rel = data.get('exclude_images_path')
    if rel and not os.path.isabs(rel):
        data['exclude_images_path'] = os.path.join(config_dir, rel)
    ds = CephalometricDataset(
        images_dir=data['images_dir'], fcd_dir=data['fcd_dir'],
        review_csv_path=data.get('review_csv_path'),
        combine_method=data.get('combine_method', 'median'),
        min_landmarks_in_bounds=0.9,
        exclude_images_path=data.get('exclude_images_path'),
    )
    # prioritise images that carry a Me marker (direct Gn-vs-Me view), then fill
    idx_with_me = [i for i, s in enumerate(ds.samples) if 'Me' in s['landmarks']]
    idx_rest = [i for i, s in enumerate(ds.samples) if 'Me' not in s['landmarks']]
    rng = np.random.default_rng(seed)
    rng.shuffle(idx_with_me); rng.shuffle(idx_rest)
    chosen = (idx_with_me + idx_rest)[:n]
    print(f"internal: {len(ds.samples)} images, {len(idx_with_me)} carry a Me marker; "
          f"showing {len(chosen)} ({min(len(idx_with_me), n)} of them with Me)")
    out = []
    for i in chosen:
        s = ds[i]
        lm = {nm: (float(x), float(y)) for nm, (x, y) in zip(s['landmark_names'], s['landmarks'])}
        out.append((os.path.splitext(s['imagefile'])[0], to_rgb(s['image']), lm))
    return out


def samples_aariz(root, split, n, seed):
    from aariz_dataset import AarizDataset
    mapping = {'Gn': 'Gn', 'Me': 'Me', 'Pog': 'Pog'}  # identity: keep Aariz's own labels
    ds = AarizDataset(root=root, split=split, annotator='senior',
                      name_mapping=mapping, machine_mappings_csv=None)
    rng = np.random.default_rng(seed)
    idx = list(range(len(ds.samples)))
    rng.shuffle(idx)
    chosen = idx[:n]
    print(f"aariz ({split}): {len(ds.samples)} images; showing {len(chosen)}")
    out = []
    for i in chosen:
        s = ds[i]
        lm = {nm: (float(x), float(y)) for nm, (x, y) in zip(s['landmark_names'], s['landmarks'])}
        out.append((s['imagefile'].rsplit('.', 1)[0], to_rgb(s['image']), lm))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['internal', 'aariz'], required=True)
    ap.add_argument('--config', help="model config YAML for data paths (internal)")
    ap.add_argument('--aariz-root', default=os.environ.get('AARIZ_ROOT', '/path/to/project/data/Aariz'))
    ap.add_argument('--split', default='all', choices=['train', 'valid', 'test', 'all'])
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    if args.dataset == 'internal':
        if not args.config:
            ap.error("--config is required for --dataset internal")
        samples = samples_internal(args.config, args.n, args.seed)
        draw_order = INTERNAL_CHIN
        out_dir = args.out or 'outputs/chin_eyeball/internal'
    else:
        samples = samples_aariz(args.aariz_root, args.split, args.n, args.seed)
        draw_order = AARIZ_CHIN
        out_dir = args.out or 'outputs/chin_eyeball/aariz'

    os.makedirs(out_dir, exist_ok=True)
    tiles = []
    for image_id, rgb, lm in samples:
        tile = draw_and_crop(rgb, lm, draw_order, image_id)
        if tile is None:
            print(f"[skip] {image_id}: no chin landmarks present")
            continue
        tile.save(os.path.join(out_dir, f"{image_id}_chin.png"))
        tiles.append(tile)

    sheet = montage(tiles)
    if sheet is not None:
        sheet_path = os.path.join(out_dir, f"_montage_{args.dataset}.png")
        sheet.save(sheet_path)
        print(f"\nWrote {len(tiles)} chin crops + montage -> {sheet_path}")
    legend = ("RED=Gn (point in question)  YELLOW=Me/Menton  GREEN=Pogonion"
              + ("  CYAN=Pgn" if args.dataset == 'internal' else ""))
    print("LEGEND:", legend)


if __name__ == "__main__":
    main()
