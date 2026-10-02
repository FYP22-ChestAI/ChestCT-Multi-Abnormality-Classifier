# ChestCT-Multi-Abnormality-Classifier

Part of the ChestAI project (Resource-Efficient Multi-Abnormality Chest CT
Classification and Local Adaptation). This repo currently holds **M1: CT
data loading and preprocessing** -- the pipeline that turns a raw CT-RATE
scan into clean, standardised slices ready for slice selection (M3) and
classification (M4), which will be added to this same repo later. Report-to-label
extraction (M2) lives in a sibling repo in this GitHub organization
(`ChestCT-Report2Label`).

See [`docs/preprocessing/data_contract.md`](docs/preprocessing/data_contract.md) for exactly what this
pipeline produces, the format decisions behind it, and why. That file is the
actual agreement between this code and M3/M4 -- read it before writing code
that consumes the cache.

**Scope note:** the same pipeline handles the public **CT-RATE** dataset
(NIfTI, downloaded) and local hospital data (DICOM, from any mounted folder --
Drive, SSD or server). No labels are read or produced anywhere in this
pipeline, for either source -- see "Architecture" in `docs/preprocessing/data_contract.md`.
You choose the source when you run it; see the Colab notebooks
`ChestAI_M1_training_pipeline.ipynb` (builds the manifest, preprocesses,
runs QC) and `ChestAI_M1_inference.ipynb` (one scan or a folder of several,
no manifest, no persistence), and "Local DICOM data" in
`docs/preprocessing/data_contract.md`. Series selection (choosing a chest series among
several) is not built (the data is lung series only).

## Project layout

```
configs/preprocessing.yaml          all pipeline settings (spacing, size, windows, QC thresholds, sources, device)
docs/preprocessing/data_contract.md      what M1 hands to M3/M4, and why -- read this first
src/ct_preprocessing/
  types.py                 the one shared Volume type every stage uses
  loader.py                NIfTI -> Volume, real HU, explicit calibration check (Step 4)
  dicom_loader.py          DICOM series -> Volume; finds scans under any root folder
  loaders.py               picks the loader by format (nifti / dicom)
  staging.py               copy scans from a slow mount (Drive) to local disk
  spacing.py                resample to a common mm/voxel; optional GPU path (Step 5)
  crop.py                  cut away background around the body, memory-safe (Step 7)
  resize.py                force every scan to a fixed pixel size; optional GPU path (Step 7)
  windows.py                HU windowing -- runs in the Dataset, not here (Step 6)
  pipeline.py                the ONE shared core (load->calibrate->resample->crop->resize->QC)
  preprocess.py              training-side wrapper around pipeline.py: persists to disk + cache fingerprint
  preprocess_config.py        the PreprocessConfig settings object
  device.py                   GPU/CPU resolution shared by spacing.py/resize.py
  quality.py                automatic QC checks + montage images (Step 8)
  manifest.py                build the one-row-per-scan manifest + patient splits (Step 3)
  config.py                  load configs/preprocessing.yaml, incl. multiple named data `sources`
  dataset.py                  the M1 -> M3/M4 hand-off (label_cols optional, for training vs. inference; M1 itself never has labels)
src/ct_preprocessing/inference.py     the inference front door -- one scan or a batch, shares pipeline.py's core
scripts/
  download_subset.py         one combined CT-RATE download from Hugging Face (needs your own login); runs with no arguments (configs/preprocessing.yaml defaults), or override --n-train/--n-val/--n-test/--seed
  build_manifest.py          Step 3 CLI, one named source at a time (--source-name, --raw-dir, --append); runs with no --n-train/--n-val/--n-test/--seed (config defaults), auto-detects patient_id_source for DICOM; no labels, ever
  preprocess_all.py          Steps 5-9 CLI (parallel, resumable via cache fingerprint, --device)
  qc_report.py               Step 8 CLI
  stage_folder.py            copy scans from a Drive mount to local disk (resumable)
tests/                       pytest, all using synthetic data -- no CT-RATE download needed to run them
```

## Setup

```bash
python -m pip install -e ".[dev]"
```

Torch is only needed for `ct_preprocessing.dataset.ChestCTDataset.__getitem__`
(the final hand-off to M3/M4's training code); everything else in this repo
runs without it. Install it yourself when you need it:

```bash
pip install torch
```

## Quickstart: pilot run on a small subset

1. **Get CT-RATE access.** Create a Hugging Face account, accept the
   [CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE) terms,
   and log in locally (`pip install huggingface_hub && huggingface-cli login`).
   Do this yourself -- don't share your token with anyone or commit it.

2. **Download a small pilot subset.** With no arguments at all, this uses
   `configs/preprocessing.yaml`'s `sources.ctrate` defaults (4 train + 1 val scans
   from CT-RATE's official TRAIN pool, 1 test scan from its official VALID
   pool, screened to skip anything needing over 1.5 GB of resample memory --
   from the corrected `_fixed` folder by default, fast, never lists the
   whole repository, and picked at random rather than "the first N
   alphabetically"):
   ```bash
   python scripts/download_subset.py
   ```
   Override any of the defaults on the command line, e.g.
   `--n-train 40 --n-val 10 --n-test 10 --seed 0` for a bigger, reproducible
   run. Scans land flat in `data/raw/`; metadata (never labels -- this
   pipeline doesn't read them) lands flat in `data/metadata/`.

3. **Build the manifest** (Step 3 -- ids, patient-level splits; no labels).
   Run once per named source in `configs/preprocessing.yaml`'s `sources:`, with
   `--append` after the first, to accumulate several sources (e.g. CT-RATE
   and later local NHRD data) into one manifest. With no `--n-train`/`--n-val`/
   `--seed`, this uses the SAME `configs/preprocessing.yaml` defaults the download
   step used, so the split is reproducible without repeating them:
   ```bash
   python scripts/build_manifest.py --source-name ctrate --builder ctrate \
     --raw-dir data/raw --metadata data/metadata/train_metadata.csv
   ```

4. **Preprocess** (Steps 5-9 -- HU calibration, spacing, crop, resize, save):
   ```bash
   python scripts/preprocess_all.py --workers 4
   ```
   Resumable via a config fingerprint -- re-running skips a volume only if
   its cache was produced by the exact settings active now; changing any
   setting (or switching data sources) correctly triggers a reprocess
   instead of silently reusing stale output. Add `--device auto` to use a
   GPU for resampling when one's available (e.g. on Colab).

5. **Run quality checks** (Step 8):
   ```bash
   python scripts/qc_report.py
   ```
   Look at a few files in `data/qc_montages/` -- they should show upright,
   correctly oriented chest slices.

6. **Run the tests** (no download needed for this part):
   ```bash
   pytest
   ```

## Notes for whoever builds M3 / M4 next

- Read `docs/preprocessing/data_contract.md` first -- it defines the exact tensor shape,
  dtype, and value range you'll get, and the two `ChestCTDataset` modes
  (`all_lowres` for M3's scoring pass, `selected` for M4's full-resolution
  encoding).
- Nothing here is windowed until you ask for it (`ct_preprocessing.dataset.slices_to_tensor`)
  -- the cache stores raw HU so windows can still change without re-running
  preprocessing.
- `configs/preprocessing.yaml` is the single source of truth for every tunable
  number. Don't hard-code spacing/size/window values elsewhere.
