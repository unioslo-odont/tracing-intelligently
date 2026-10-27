# OpenCeph — cephalometric landmark detection toolkit

Open-source research code for **automatic landmark detection on 2-D lateral cephalograms** using
heatmap regression, with a reproducible pipeline for training, cross-validation, and evaluation —
including evaluation on an external dataset without fine-tuning.

> **Research code, not a medical device.** Nothing here is validated for clinical decision-making;
> predictions require review by a qualified clinician.

Code is licensed **Apache-2.0**. Trained weights are a separate release artifact:
their license and availability must be confirmed by the University of Oslo before
distribution. `NOTICE` records the project attribution; `CITATION.cff` provides the
machine-readable citation metadata. A license preserves attribution notices, but
academic users should also cite the accompanying publication.
Quantitative results are reported in the accompanying poster/publication (see *Citation*), not here.

---

## What this code does

**Landmark detection by heatmap regression.** Rather than regressing coordinates directly, each model
predicts one spatial confidence map per landmark — trained against a 2-D Gaussian centred on the
annotated point — and the coordinate is decoded as that map's sub-pixel peak, then rescaled to the
image's native resolution so error can be expressed in millimetres. The peak height is recorded
as a per-point score. Failure detection from that score requires validation; it is not a calibrated probability.

**Four interchangeable model arms** behind one interface, `image → (N landmarks, H/4, W/4)`, selected
by `model.type` in the config:

| `model.type` | Arm | File |
|---|---|---|
| `unet` | U-Net trained from scratch | `code/models/unet.py` |
| `pretrained_encoder` | ImageNet-pretrained encoder (e.g. ResNet, HRNet) + U-Net decoder | `code/models/pretrained_encoder_unet.py` |
| `dinov2` | Frozen DINOv2 backbone + trainable decoder | `code/models/dinov2_head.py` |

Gaussian target encoding and sub-pixel decoding live in `code/models/heatmap_codec.py`. Datasets are
I/O-only — heatmap encoding and augmentation live in the transform, so they can be swapped
independently of the data source.

**Evaluation.** Mean and median radial error and successful detection rate (SDR) at configurable
thresholds, reported **per landmark** as well as overall, with millimetre conversion from each image's
own pixel spacing. Training automatically evaluates the best checkpoint on both the training and
validation splits and writes a per-(image, landmark) table with predicted point, reference point,
error, and peak confidence.

**Experiment tooling.** Patient-level k-fold cross-validation against a persisted fold manifest;
exploratory paired per-fold model comparison; a label-efficiency harness (retrain on nested subsets of the
training patients); optional per-parameter-group optimiser settings (separate encoder/decoder learning
rates with warm-up); and external-dataset evaluation with an explicit, hand-verified landmark-name
mapping.

---

## Configuring the landmark set (and adding your own)

**The landmark vocabulary is data, not code.** It lives in a YAML file referenced by
`data.landmark_config_path`, e.g.:

```yaml
landmarks:
  "S":
    enabled: true
    category: reference
    description: "Sella - center of the sella turcica"
  "N":
    enabled: true
    ...
```

The trained set is exactly the entries with `enabled: true`; the number of network output channels
follows automatically, and every report is keyed by landmark name. To **extend or change the set** —
add soft-tissue points, adopt a different analysis protocol, or work with another institution's
convention — add or enable entries and point a config at that file. Two provided examples:

- `configs/landmarks.yaml` — a larger reference protocol
- `configs/landmarks_15.yaml` — a reduced hard-tissue subset

The reduced protocol trains **16 channels**. Its internal headline scores **14 direct
landmarks**, excluding `iGo` and `pGo`. A 15th clinical point, Gonion, is a design target:
its construction and evaluation are not implemented. `Gn` remains a provisional
Menton candidate and is excluded from external evaluation pending confirmation.

Notes when defining a set:
- A landmark that is sparsely annotated in your data will train poorly; check coverage first with
  `code/data_preparation_TSD/analyze_landmark_names.py`, which reports per-name annotation frequency
  and flags likely spelling/case duplicates.
- Missing annotations are handled by a per-sample mask (masked MSE), so images lacking a landmark do
  not need to be dropped.
- **Never match landmark names across protocols by case-folding or string similarity.** Identical
  codes can denote different anatomical points in different conventions. Cross-dataset mappings go
  through an explicit, individually verified table (see `AARIZ_TO_REFERENCE_MAPPING` in
  `code/datasets/aariz_dataset.py`) — landmarks with no verified counterpart are reported as
  uncheckable rather than silently scored.

---

## Repository layout

```
code/
  datasets/              dataset loaders, heatmap transform, patient-level k-fold + fold manifest
  models/                unet, pretrained_encoder_unet, dinov2_head, heatmap_codec, model_factory
  training/              train / evaluate entry points, shared eval core, metrics, comparison and
                         paired-fold statistics, provenance, config and output-path handling
  data_preparation_TSD/  Facad (.fcd) parsing, landmark-name analysis, tracing selection
  analysis/              R Markdown cross-model reporting
configs/                 landmark vocabularies + one YAML per model arm
                         (folds/ holds the generated fold manifest; git-ignored)
slurm/                   environment setup and job scripts for a SLURM cluster
```

---

## Data

**No patient data is included in this repository, and none should ever be added to it.** The
development data are access-restricted and cannot be redistributed; `data/`, `outputs/`,
`checkpoints/` and working notes are git-ignored.

To use this code you supply your own cephalograms and reference landmarks. The reference loader reads
the Facad `.fcd` tracing format (`code/data_preparation_TSD/facad_io.py`); for any other source,
implement a loader that yields the same sample dictionary (image array, landmark coordinates, names,
and pixel spacing) — see `code/datasets/cephalometric_dataset.py` as the reference implementation and
`code/datasets/aariz_dataset.py` for a second, differently-formatted example.

Millimetre conversion is explicit and per image: images without a known pixel spacing report pixel
error and are excluded from millimetre statistics rather than silently assumed.

**External datasets:** `aariz_dataset.py` reads the public Aariz/CEPHA29 benchmark for *evaluation
only*. Verify that dataset's current licence terms before use — its terms restrict derivative use.

---

## Usage

### Curating Facad tracings in the browser

The [local tracing review workbench](code/curation/README.md) lets reviewers
compare per-image Facad tracings, select a reference, adjust landmark points,
save drafts or exclusions, and export accepted images with corrected `.fcd`
references in the existing training-loader layout. Reviews and revision history
are persisted locally; original images and tracing files are preserved.

```bash
python code/curation/server.py --demo --output outputs/curation-demo --open
```

Run on real data inside the approved environment with `--images`, `--facad`,
and `--output`; see the workbench guide for input conventions and training-round
exports. The demo generates synthetic diagrams, not patient radiographs.

### Training and evaluation

Environment (adapt paths for your cluster):

Tracked configs and scripts contain placeholder paths. Your own settings go in two
git-ignored files under `local/`, so tracked files never need site-specific edits:

- `local/site.env` (template `slurm/site.example.env`): venv, cache root, optional
  `HF_HOME` weight cache, module versions, SLURM account and the Aariz root.
  `slurm/env.sh` sources it; submit jobs from that shell so `sbatch` inherits it.
- `local/site.yaml` (template `configs/site.example.yaml`): data locations, merged over
  every config the code loads (config < site file < `--set`). Checkpoints and
  `run_meta.json` record the merged values.

The partition in the `#SBATCH` headers is set for the cluster this was developed on;
adjust it for yours. Do not recreate an existing experiment environment simply to
re-score saved results.

```bash
mkdir -p local
cp slurm/site.example.env local/site.env      # then edit
cp configs/site.example.yaml local/site.yaml  # then edit
bash slurm/setup_env.sh          # one-time: create the virtual environment
source slurm/env.sh              # per session: site settings, modules, venv, cache redirection
```

Pipeline:

```bash
# 1. Freeze the patient->fold split as a local artifact (once, for your dataset).
#    Every later run asserts against it instead of re-deriving it, so results stay comparable.
#    It lists your own cohort's patient IDs, so it is git-ignored - generate your own.
python code/training/make_fold_manifest.py --config configs/<your_config>.yaml

# 2. Train one arm, one fold (SLURM array over folds)
sbatch slurm/train_variant.sbatch configs/<your_config>.yaml
#    or all arms + a gated cross-model comparison:
bash slurm/run_15pt.sh

# 3. Refit with a patient holdout for checkpoint selection (default 10%)
sbatch slurm/train_final.sbatch configs/<your_config>.yaml

# 4. Evaluate a checkpoint (internal split, or an external dataset)
python code/training/evaluate.py --checkpoint <ckpt> --split val
python code/training/evaluate_aariz.py --checkpoint <ckpt> --root /path/to/aariz --split test --unmap Gn
```

To regenerate the final model's internal holdout overlays, pass `--all-data` so
`evaluate.py` reconstructs the final run's patient-level checkpoint-selection
holdout instead of the config's cross-validation fold. `--overlay-landmarks`
restricts both the ground-truth circles and prediction crosshairs to trained
channels named explicitly. The same option is supported by `evaluate_aariz.py`.
For the final model's 16 trained internal channels and 12 externally comparable
channels, use
`slurm/generate_study_overlays.sbatch`.

Optional experiments:

```bash
# label-efficiency: retrain on nested subsets of each fold's training patients
python code/training/make_label_efficiency_subsets.py --fractions 0.5 0.25
bash slurm/run_label_efficiency.sh
python code/training/aggregate_label_efficiency.py --arms <armA> <armB> \
       --exclude-landmarks <channels reported as derived points>

# separate encoder/decoder learning rates for pretrained arms (fold-0 screen, then full CV)
bash slurm/screen_optimizer_confound.sh
```

Cross-model claims should use `code/training/paired_fold_comparison.py`: folds hold out the same
patients in every arm, so the meaningful statistic is the per-fold difference, not two independent
mean ± SD columns.

For the reduced protocol, pass `--exclude-landmarks iGo pGo` to both comparison
scripts and label the headline as 14 directly scored landmarks. Paired-fold intervals
are exploratory: training folds overlap, and an interval including zero does not
establish equivalence. Compare internal/external metrics on an identical landmark set.

---

## Reproducibility

Runs record a **resolved config and git commit**, together with whether the source was dirty. Checkpoints embed their fully
resolved config and landmark vocabulary, so evaluation never has to be told the architecture; each run
writes a provenance record (commit, resolved config, realised cohort) beside its outputs; splits are
patient-level and asserted against a persisted fold manifest; and quick variations use
`--set key=value` overrides rather than edited configs.

New provenance records also include source hashes, package versions and a private
patch of tracked source changes when present. A dirty historical commit alone does
not identify the exact experiment code; retain its actual source snapshot privately.
Subset medians must be calculated from per-point errors, never by averaging landmark medians.

---

## Further reading

- `MODEL_CARD.md` — intended use, limitations, and ethical considerations
- `CONTRIBUTING.md` — development setup and conventions
- `LICENSE` — Apache-2.0
- `NOTICE` — attribution and institutional acknowledgement
- `CITATION.cff` — machine-readable citation metadata
- `RELEASING.md` — manual HPC validation and a public source export without history

## Citation

If you use this code, please cite the accompanying publication (in preparation),
acknowledge the University of Oslo and the authors listed in `CITATION.cff`, and
preserve the `NOTICE` file in redistributions. If you evaluate on
the Aariz/CEPHA29 benchmark, cite Khalid M.A. et al., *A Benchmark Dataset for Automatic Cephalometric
Landmark Detection and CVM Stage Classification*, Scientific Data 12:1336 (2025).

Contributions, independent validation on other cohorts, and collaboration are welcome — see
`CONTRIBUTING.md`.
