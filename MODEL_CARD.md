# Model Card — OpenCeph hard-tissue cephalometric landmark detector

A model card for the deployable model in this repository, following common model-card conventions.
**This is a research proof-of-concept, not a medical device.**

## Model details

- **Developed by:** Gerald Torgersen, University of Oslo, and collaborators.
- **Model type:** Fully-convolutional **U-Net** trained from scratch for **heatmap regression** of
  cephalometric landmarks on 2-D lateral cephalograms. Landmarks are localised *indirectly* — the
  network predicts a per-landmark spatial confidence map (Gaussian-supervised) and the point is decoded
  as that map's sub-pixel peak. Input 512×512; heatmaps 128×128; per-point confidence available.
- **Landmark set:** **16 trained channels**, targeting a 15-point clinical protocol — see `configs/landmarks_15.yaml`.
  Internal headline evaluation scores 14 direct landmarks, excluding `iGo` and `pGo`.
  Derived Gonion is not implemented or evaluated. Soft tissue is out of scope.
- **Status:** research prototype. **Trained weights are not yet released**; any future
  weight license and release must be authorized separately by the University of Oslo.
- **Licence:** Apache-2.0 for the code. Model-weight licensing and release are
  separate decisions requiring University of Oslo authorization; do not infer a
  weight license from this model card until that review is complete.
- **Repository:** this repo. Each checkpoint embeds its full resolved config + landmark vocabulary, so
  evaluation never needs to be told the architecture.

## Intended use

- **Primary use:** an **open, reusable baseline** that clinics and researchers can **fine-tune on their
  own small cohorts**, and a **first-pass tracing aid** whose output a qualified orthodontist reviews
  and corrects.
- **Intended users:** orthodontic and medical-imaging researchers.

## Out-of-scope and prohibited uses

- **Not a medical device.** Not for autonomous clinical diagnosis, treatment planning, or any
  unsupervised clinical decision. A qualified clinician must review every prediction.
- Not validated for soft-tissue landmarks, non-lateral projections, dentition or pathology outside the
  training distribution, or imaging hardware markedly different from the validation scanners.

## Training data

- A single-centre **Norwegian** clinical cohort of lateral cephalograms with expert-reviewed reference
  tracings. The data are access-restricted (GDPR / REK / Sikt), **not distributable**, and not included
  in this repository. Splitting is **patient-level** (no patient appears in both training and
  validation). Cohort composition and size are described in the accompanying publication.

## Evaluation

- **Protocol:** patient-level 5-fold cross-validation against a persisted fold manifest, plus
  evaluation on an **independent external dataset without fine-tuning**.
- **External benchmark:** Aariz / CEPHA29 (Khalid et al., *Scientific Data* 2025), a public dataset from
  a different country and imaging hardware; used **for evaluation only**, never for training. Verify
  that dataset's current licence terms before use.
- **Metrics:** mean and median radial error (MRE) and successful detection rate (SDR) at 2/2.5/3/4 mm,
  reported per landmark. Architecture comparisons use exploratory paired per-fold summaries.
  Training folds overlap; these intervals do not establish equivalence.
- **Quantitative results are reported in the accompanying poster/publication**, not in this repository.
  Accuracy is specific to the landmark set scored. Removing higher-error landmarks can
  lower aggregate error without improving predictions; state the set alongside every number.

## Limitations

- **Hard tissue only** — 16 channels, 14 directly scored headline landmarks; soft-tissue analysis is out of scope.
- **Single-centre training population** → possible population/scanner bias; external generalisation has
  been assessed on only one further cohort. The error distribution has a **gross-outlier tail** (the mean
  exceeds the median), concentrated on imaging hardware least similar to the training data; the
  per-point score requires validation before use for failure detection.
- **Cross-dataset landmark *definition* mismatches** materially affect reported error (e.g. gnathion is
  a constructed point defined differently across datasets; porion may be anatomic vs. machine/ear-rod).
  Landmark definitions must match before comparing numbers across studies.
- One landmark (`Gn`) is used provisionally as Menton, pending orthodontist confirmation; it is marked
  uncheckable in the external comparison until confirmed.
- The model outputs **landmark positions**, not the derived clinical measurements (SNA, SNB, ANB, FMA,
  …) — a measurement-aware evaluation layer is planned, not yet built.

## Ethical and safety considerations

- **Privacy:** training radiographs and tracings never leave the secure environment; only code and
  (on release) derived weights are shared.
- **Clinical safety:** research prototype; a human clinician must remain in the loop.
- **Bias / fairness:** trained on a single Norwegian centre — validate on your own population and
  hardware before any use.

## How to use (once weights are released)

Load a released checkpoint and run the evaluation entry points (`code/training/evaluate.py`,
`evaluate_aariz.py`), or fine-tune with `train.py` on your own cohort. See the README for the pipeline.

## Citation

Please cite the accompanying work (in preparation) and the external benchmark: Khalid M.A. et al.,
*A Benchmark Dataset for Automatic Cephalometric Landmark Detection and CVM Stage Classification*,
Scientific Data 12:1336 (2025).

For software attribution, see `NOTICE` and `CITATION.cff` in the repository.
