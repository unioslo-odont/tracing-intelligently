"""
================================================================================
HEATMAP VISUALIZATION (diagnostic)
================================================================================
DESCRIPTION:
    Renders the model's RAW predicted heatmap for one or more landmark
    channels, overlaid on the source image - not just the decoded point.
    A decoded point alone can't distinguish between:
      - a confident, well-localized peak that's simply in the wrong place
        (model learned the wrong association)
      - a diffuse, low-confidence blob spread over a large area (channel
        hasn't converged / is unsure)
      - a near-flat, near-zero heatmap (a "dead" channel - the model isn't
        really predicting anything for this landmark; argmax on near-noise
        picks an effectively arbitrary point)
      - multiple competing peaks (confusion between plausible locations)
    All four decode to "some point" via argmax and look identical in a
    plain overlay of just the predicted point - this tool is for telling
    them apart.

    Supports comparing MULTIPLE checkpoints (e.g. all four model arms) side
    by side for the SAME image and landmark(s), which is the direct way to
    investigate an architecture-independent failure (e.g. is landmark T's
    heatmap diffuse/confused across every model for a given patient - which
    would point at an annotation problem - or does it look fine for most
    models and only fail for one - which would point at that model
    specifically).

    Layout: one PNG per image, grid of [checkpoints x landmarks]. More than
    `--landmarks-per-page` landmarks (default 6) are split across multiple
    PNGs per image (`..._heatmaps_page1.png`, `_page2.png`, ...) rather than
    one increasingly wide figure - e.g. requesting all ~22 landmarks from
    landmarks.yaml at once still produces readable, page-sized grids instead
    of a single ~90-inch-wide image.

USAGE:
    # one model, several suspect channels:
    python visualize_heatmaps.py --checkpoints checkpoints/dinov2_frozen/fold0/best.pt \
        --landmarks S Pm Pt T OLp --max-images 6

    # compare all four models on the one universally-bad landmark:
    python visualize_heatmaps.py \
        --checkpoints checkpoints/unet_baseline/fold0/best.pt \
                      checkpoints/resnet_unet/fold0/best.pt \
                      checkpoints/hrnet_pretrained/fold0/best.pt \
                      checkpoints/dinov2_frozen/fold0/best.pt \
        --landmarks T --max-images 8

    # every enabled landmark, whole validation set, paginated automatically:
    python visualize_heatmaps.py \
        --checkpoints checkpoints/unet_baseline/fold0/best.pt \
                      checkpoints/resnet_unet/fold0/best.pt \
                      checkpoints/hrnet_pretrained/fold0/best.pt \
                      checkpoints/dinov2_frozen/fold0/best.pt \
        --landmarks A Ar B Ba GL GNs Gn Ii Is MS N Or PGs PLi PLs PRN Pg Pm Po RN S Sp iGo \
        --max-images 0
================================================================================
"""

import argparse
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'datasets')))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')))

from heatmap_transform import HeatmapTargetTransform, load_landmark_config
from fold_manifest import resolve_kfold_split
from model_factory import build_model
from heatmap_codec import decode_heatmaps
from visualize import _to_display_rgb

from evaluate import build_base_dataset  # noqa: F401 (reused unchanged from evaluate.py)
from output_paths import outputs_path


def load_checkpoint_bundle(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    landmark_names = checkpoint['landmark_names']

    transform = HeatmapTargetTransform(
        landmark_names=landmark_names,
        image_size=tuple(config['data']['image_size']),
        heatmap_size=tuple(config['data']['heatmap_size']),
        sigma=config['data'].get('sigma', 2.0),
    )
    base_dataset = build_base_dataset(config)

    model = build_model(config['model'], num_landmarks=len(landmark_names)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    k, fold = config['kfold']['k'], config['kfold']['fold']
    train_idx, val_idx = resolve_kfold_split(base_dataset, config, fold=fold)

    filename_to_index = {entry['imagefile']: i for i, entry in enumerate(base_dataset.samples)}

    label = os.path.basename(os.path.dirname(os.path.dirname(checkpoint_path))) or checkpoint_path
    return {
        'label': label,
        'model': model,
        'transform': transform,
        'base_dataset': base_dataset,
        'landmark_names': landmark_names,
        'val_filenames': [base_dataset.samples[i]['imagefile'] for i in val_idx],
        'filename_to_index': filename_to_index,
        'device': device,
    }


def render_cell(ax, bundle, imagefile, landmark_name):
    index = bundle['filename_to_index'].get(imagefile)
    if index is None or landmark_name not in bundle['landmark_names']:
        ax.axis('off')
        ax.set_title(f"{landmark_name}: n/a", fontsize=9, color='gray')
        return

    raw_sample = bundle['base_dataset'][index]
    transformed = bundle['transform'](raw_sample)
    image_tensor = transformed['image'].unsqueeze(0).to(bundle['device'])

    with torch.no_grad():
        pred_heatmaps = bundle['model'](image_tensor)
    pred_coords_hm, pred_conf = decode_heatmaps(pred_heatmaps)
    pred_coords_hm, pred_conf = pred_coords_hm[0], pred_conf[0]

    ch = bundle['landmark_names'].index(landmark_name)
    orig_h, orig_w = raw_sample['image'].shape[:2]
    hm_h, hm_w = pred_heatmaps.shape[-2:]

    heatmap = pred_heatmaps[0, ch].detach().cpu().numpy()
    peak = float(heatmap.max())
    heatmap_norm = heatmap / max(peak, 1e-8)
    heatmap_img = Image.fromarray((heatmap_norm * 255).astype(np.uint8))
    heatmap_resized = np.array(
        heatmap_img.resize((orig_w, orig_h), resample=Image.BILINEAR), dtype=np.float32
    ) / 255.0

    base_gray = np.array(_to_display_rgb(raw_sample['image']).convert('L'), dtype=np.float32) / 255.0
    ax.imshow(base_gray, cmap='gray')
    ax.imshow(heatmap_resized, cmap='inferno', alpha=0.55)

    gt_by_name = dict(zip(raw_sample['landmark_names'], raw_sample['landmarks']))
    if landmark_name in gt_by_name:
        gx, gy = gt_by_name[landmark_name]
        ax.plot(gx, gy, marker='o', markeredgecolor='lime', markerfacecolor='none', markersize=10, markeredgewidth=2)

    scale_x, scale_y = orig_w / hm_w, orig_h / hm_h
    px, py = pred_coords_hm[ch]
    ax.plot(px * scale_x, py * scale_y, marker='+', color='cyan', markersize=14, markeredgewidth=2)

    ax.axis('off')
    ax.set_title(f"{bundle['label']} / {landmark_name}\npeak={peak:.2f}", fontsize=9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--landmarks', nargs='+', default=None,
                         help="landmark names to visualize (space-separated). Exactly one of "
                              "--landmarks / --landmarks-config is required.")
    parser.add_argument('--landmarks-config', default=None,
                         help="path to a landmarks.yaml-style file - use every landmark with "
                              "enabled: true from there, instead of typing them out via "
                              "--landmarks (avoids the list going stale if landmarks.yaml changes).")
    parser.add_argument('--max-images', type=int, default=6,
                         help="Cap on number of images to render. 0 = no cap (whole val split).")
    parser.add_argument('--landmarks-per-page', type=int, default=6,
                         help="split many landmarks across multiple output images per source "
                              "image (one 'page' each), so a single figure doesn't become "
                              "unreasonably wide when requesting more than a handful of "
                              "landmarks (e.g. all of landmarks.yaml at once).")
    parser.add_argument('--output-dir', default=None,
                         help="where to write the heatmap grids (default: outputs/heatmap_diagnostics "
                              "- gitignored; see code/training/output_paths.py)")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = outputs_path('heatmap_diagnostics', make_parent=False)

    if bool(args.landmarks) == bool(args.landmarks_config):
        parser.error("specify exactly one of --landmarks or --landmarks-config")
    landmarks = args.landmarks if args.landmarks else load_landmark_config(args.landmarks_config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bundles = [load_checkpoint_bundle(path, device) for path in args.checkpoints]
    os.makedirs(args.output_dir, exist_ok=True)

    # Reference image list: the first checkpoint's val split, restricted to
    # images every checkpoint actually has (same source data, but a review
    # CSV exclusion could differ in principle).
    common_filenames = set(bundles[0]['val_filenames'])
    for b in bundles[1:]:
        common_filenames &= set(b['filename_to_index'].keys())
    filenames = [f for f in bundles[0]['val_filenames'] if f in common_filenames]
    if args.max_images > 0:
        filenames = filenames[:args.max_images]

    if not filenames:
        raise RuntimeError("No common images found across the given checkpoints' val splits.")
    print(f"Rendering {len(filenames)} image(s) across {len(bundles)} checkpoint(s).")

    page_size = max(1, args.landmarks_per_page)
    landmark_pages = [landmarks[i:i + page_size] for i in range(0, len(landmarks), page_size)]
    multi_page = len(landmark_pages) > 1
    if multi_page:
        print(f"{len(landmarks)} landmarks requested - splitting into {len(landmark_pages)} "
              f"page(s) of up to {page_size} each per image.")

    n_rows = len(bundles)
    for imagefile in filenames:
        for page_idx, landmark_page in enumerate(landmark_pages):
            n_cols = len(landmark_page)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 4.2 * n_rows), squeeze=False)
            for row, bundle in enumerate(bundles):
                for col, landmark_name in enumerate(landmark_page):
                    render_cell(axes[row][col], bundle, imagefile, landmark_name)

            fig.suptitle(imagefile, fontsize=12)
            fig.tight_layout()
            suffix = f"_page{page_idx + 1}" if multi_page else ""
            out_path = os.path.join(args.output_dir, os.path.splitext(imagefile)[0] + f'_heatmaps{suffix}.png')
            fig.savefig(out_path, dpi=130)
            plt.close(fig)
            print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
