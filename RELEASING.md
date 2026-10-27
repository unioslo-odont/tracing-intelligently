# Preparing a public source release

The public repository starts with new history. Keep this research checkout and its
history private. Do not upload the working directory: ignored folders contain
private research material and patient-level artifacts.

## Validate on the HPC

Run the checks in CONTRIBUTING.md manually in the existing HPC environment.
Do not install or upgrade dependencies in the environment used for the reported
experiments. Capture `python --version`, `pip freeze`, and GPU/CUDA information
privately first; review the package listing for private URLs or local paths before
publishing a lockfile. Validate any new environment separately.

The supplied configs use `/path/to/data/...` placeholders and SLURM headers use
`YOUR_ACCOUNT`. Real paths, account, module versions and `HF_HOME` live only in the
git-ignored `local/site.env` and `local/site.yaml` (templates `slurm/site.example.env`
and `configs/site.example.yaml`). Preserve your original HPC configuration privately;
do not replace the experiment's recorded config.

For comparisons, provide exact report directories, require all folds, and use the
same explicit landmark subset across arms. The reduced protocol trains 16 channels;
the internal headline scores 14 directly annotated landmarks by excluding `iGo`
and `pGo`. Derived Gonion is not implemented or evaluated. External scoring uses
only verified correspondences and currently excludes provisional `Gn`.

## Review the release contents

- Confirm documentation, model card, actual weight availability, and citation agree.
- Remove internal locations and unpublished results from tracked material.
- Keep fold manifests, exclusions, patient subset lists, run metadata, source patches,
  prediction tables, images, notebooks, and draft manuscripts private.
- A trained checkpoint embeds its config, which may contain private paths. Prepare
  a deliberately reviewed inference artifact before releasing weights; do not upload
  an original research checkpoint or its entire output directory automatically.
- Confirm the source and any weight release are authorised by the project owners.
- Archive the exact experiment source privately. Existing dirty-run records cannot
  be retroactively repaired by assigning the current commit to old results.

## Export without history, when ready

These are manual commands. First review and commit the intended release changes in
the private research repository, adding new source/docs/tests explicitly. Check
`git status --short` and `git ls-files`; do not force-add ignored files.

```bash
# From the private repository, after the release source has been committed:
mkdir -p release-exports
git archive --format=zip --output=release-exports/openceph-source.zip HEAD
```

`git archive` contains only the selected committed tree, with no `.git` history.
The archive excludes CLAUDE.md through `.gitattributes`. Inspect the ZIP contents,
extract into a NEW empty directory, and inspect again before starting new history:

```bash
# Run only inside that new extracted directory:
git init -b main
git add .
git diff --cached --stat
git diff --cached
git commit -m "Initial public source release"
```

Create the empty destination repository and add its URL only after this review.
Push this new repository, never a mirror of the research repository. Record the
new public commit and its corresponding private source commit in private release
notes. Add the public URL and publication citation to the README once available.
