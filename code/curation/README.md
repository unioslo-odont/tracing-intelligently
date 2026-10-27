# OpenCeph tracing review workbench

A local browser application for selecting the best Facad tracing of each
cephalogram, refining landmark coordinates, and preparing the next training
round. It uses the repository's `facad_io` parser and preserves exact landmark
names, including distinct `Pg` and `PG` points and markers nested inside teeth.
No hosted service, CDN, account, or internet connection is needed.

## Start

Run this in the environment permitted to access your data. Python 3.8 or newer
is required (tested on 3.8–3.10 and 3.13). An existing training environment already has the dependencies;
otherwise install only the small review app requirements:

```powershell
python -m pip install -r code/curation/requirements.txt
```

Try the procedural **synthetic diagrams** first:

```powershell
python code/curation/server.py --demo --output outputs/curation-demo --landmarks configs/landmarks_15.yaml --open
```

Start with your anonymized Facad export. On Windows, double-click
`code\curation\start_workbench.cmd`. It also works from a network share; the
"UNC paths are not supported" notice it prints there is harmless. Keep its
console window open while you review; closing it stops the workbench. Or run:

```powershell
python code/curation/server.py --landmarks configs/landmarks_15.yaml --open
```

The page only works when it is opened through this server, at the address it
prints. Opening `static/index.html` directly from disk shows a warning banner,
and its buttons do nothing: a page loaded from disk cannot read folders or
decode TIFFs.

When no folders are given, the **Open folders…** dialog opens automatically. Paste a path or use **Browse…**
to pick the images folder, the Facad (`.fcd`) folder and the review workspace
(where decisions are saved; defaults to `--output`). Picking a folder that holds
both images and `.fcd` files fills in both fields. The same dialog, from the top
bar, switches to another dataset later; each pair of folders keeps its own saved
reviews in the workspace. Browsers cannot reveal real folder paths to a page, so
the folder list comes from the local server. Nothing is uploaded or copied.

Or pass them on the command line:

```powershell
python code/curation/server.py --images "PATH/TO/gold/images" --facad "PATH/TO/gold/data" --output "PATH/TO/review-workspace" --landmarks configs/landmarks_15.yaml --open
```

If `--landmarks` is omitted, folders chosen in the page use
`configs/landmarks_15.yaml` when it exists.

Open **http://127.0.0.1:8765** in your browser if it does not open automatically.
Use `--port 8766` if that port is in use. Stop the server with Ctrl+C. The app
binds only to loopback; run the server and browser inside the approved data
environment. It is a local workbench, not a network service with user accounts.

Input folders follow `CephalometricDataset`: images named
`<letter><digits>_<image_index>.png`, `.tif`, or `.tiff`, and matching
`<letter><digits>.fcd` files. The image index is 1-based in document order.
Cases are numbered independently per image using their `xray_ref`.
Use the anonymized export, rather than the original Facad patient directory
with arbitrary image names. The queue shows parse errors, missing files, and
duplicate image-index warnings. Duplicate files for one serial/index are not
silently treated as separate training samples.

Optionally pass `--review-csv "PATH/TO/review_selection.csv"`. Old decisions
are shown as suggestions, including multi-tracing selections and exclusions;
they do not automatically accept images into a new round. Select one source
tracing and save a new decision. `--landmarks` adds descriptions and configured
missing points; every source marker remains available even if not configured.

## Review

1. Enter a reviewer name. Search or filter the queue.
2. Click a source tracing to use it as the reference. **Compare all** overlays
   all original tracings in distinct colors. **Source** adds crosshairs for
   the currently selected original tracing.
3. Drag a landmark, type its X/Y coordinates, or select it and nudge with
   arrow keys (1 native pixel; Shift + arrow = 0.1 pixel). Edited points turn
   amber. **Locate** zooms into the selected point; **Restore source** restores
   that point. Undo/redo operate on landmark edits.
4. Select a missing landmark and click the image to place it. **Mark missing**
   removes an unidentifiable point; it remains absent and masked in training.
   There is no inferred anatomical mapping or automatic filling of points.
5. Save a **draft** to resume later, **exclude** an unusable image, or
   **accept & next** to save the cleaned reference and advance to the next
   pending/draft image in the current queue. Add notes as appropriate.
6. Export a training round when ready. Only accepted records become training
   images. Later drafts or exclusions remove previously accepted images from
   subsequent exports. Existing exports are immutable snapshots.

Scroll to zoom, drag the background to pan, and press F to fit. `[` and `]`
navigate images. Ctrl/Cmd+Z undoes; Ctrl/Cmd+Shift+Z redoes. Switching images
or tracings warns before discarding unsaved edits.

Image display retains native dimensions and orientation. High-bit-depth TIFF
previews use a 0.5–99.5 percentile intensity window; brightness/contrast only
affect the display. The export copies the **original image bytes**, not preview
PNGs. Coordinates always use native image pixels with origin at the top left.
Facad scale is carried through without a magnification correction.

Points outside the actual image are flagged and block acceptance. Correct or
mark them missing, or exclude the image. A plausible in-bounds point does not
prove correct alignment: inspect the anatomy, especially when one patient has
multiple image sizes. The tool deliberately applies no rescaling or offset
correction to the original tracings.

## Saved data and training exports

`--output` contains `reviews.sqlite3` (plus SQLite WAL sidecars while running).
On a Windows network share or mapped network drive the database uses SQLite's
rollback journal instead, because WAL is unreliable over network filesystems.
Run one workbench at a time against a workspace on a share.
Review decisions survive server restarts. Each save records reviewer, UTC time,
selected original tracing, original and cleaned coordinates, notes, source
hashes, and an incrementing revision. The history is append-only. Optimistic
revision checks reject saves from stale browser tabs. Reload an image to resolve
a conflict. Source changes invalidate an old acceptance: re-review it; restart
the server if Facad content or image dimensions changed.

The browser downloads `curated-training-round.zip` containing:

- `images/`: only accepted images, unchanged from input.
- `data/`: minimal `.fcd` training references with one corrected case per
  accepted image. Image-index positions and pixel scale are retained.
- `review_selection.csv`: `1` for the exported corrected case and `x` for
  other indexed images, in the existing semicolon-delimited format.
- `cleaned_landmarks.csv`: wide coordinate table plus source tracing and
  reviewer information, for inspection or other consumers.
- `audit.json`: current decisions, source hashes, and complete revision history
  up to the snapshot, including rejected/draft images.

The generated `.fcd` is an **OpenCeph training interchange format**. It is not
intended for re-import into Facad and contains no original patient demographics
or clinical analyses. The original input `.fcd` files are never changed.

Extract the ZIP to a new private training-round directory. Point your config at:

```yaml
data:
  images_dir: /private/training-round/images
  fcd_dir: /private/training-round/data
  review_csv_path: /private/training-round/review_selection.csv
  landmark_config_path: landmarks_15.yaml
```

Keep the existing patient-level split rules. Adding patients changes the cohort:
create an explicitly new fold manifest for the new round rather than bypassing
the assertion on an existing manifest. Give the new experiment its own
`training.output_group` and checkpoint directory.

Store workspaces, ZIPs, and extracted rounds under private locations. Inside
this repository, use the already git-ignored `outputs/` or `data/` directories.
The audit and database contain image identifiers, review notes, and reviewer
names. Back up the workspace within the approved environment. Stop the server
before copying the SQLite database alone, or use SQLite's backup facility.
The server reads only the folders you chose (on the command line or in the
**Open folders…** dialog). Image and review requests name an indexed file, never
a path. The folder browser is the one exception: it lists sub-folder names and
file counts for a path typed or clicked in the page, and nothing else. It, like
opening a dataset and every save or export, rejects cross-origin requests and
requires the session token, and the server listens on loopback only. If you run it
on a shared machine, other local users can reach the port; prefer
`--images`/`--facad` there and avoid leaving the server running.

## Verification in the approved environment

The repository reserves execution and patient-image access for the maintainer
in the closed environment. Synthetic regression tests are provided:

```powershell
python -m unittest discover -s code/curation -p "test_*.py" -v
```

They use generated fixtures only: nested markers and exact names, image-index
preservation, source change checks, review persistence, conflicts, accepted-only
exports, 16-bit image previews, and HTTP write/origin protection. They require
the small review dependencies and never read repository patient folders.

Browser acceptance check with `--demo`: toggle/overlay tracings, drag and nudge
at different zoom levels, restore/undo, place a missing point, save a draft,
restart/resume, accept and exclude images, export and inspect the ZIP. Verify
the corrected references with `CephalometricDataset` in the training environment
before starting a new run.
