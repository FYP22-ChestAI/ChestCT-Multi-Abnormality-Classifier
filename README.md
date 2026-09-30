# ChestCT-Multi-Abnormality-Classifier

Part of the ChestAI project (Resource-Efficient Multi-Abnormality Chest CT
Classification and Local Adaptation). This repo currently holds **M1: CT
data loading and preprocessing** -- the pipeline that turns a raw CT-RATE
scan into clean, standardised slices ready for slice selection (M3) and
classification (M4), which will be added to this same repo later. Report-to-label
extraction (M2) lives in a sibling repo in this GitHub organization
(`ChestCT-Report2Label`).

See [`docs/data_contract.md`](docs/data_contract.md) for exactly what this
pipeline produces, the format decisions behind it, and why. That file is the
actual agreement between this code and M3/M4 -- read it before writing code
that consumes the cache.

**Scope note:** the same pipeline handles the public **CT-RATE** dataset
(NIfTI, downloaded) and local hospital data (DICOM, from any mounted folder --
Drive, SSD or server). No labels are read or produced anywhere in this
pipeline, for either source -- see "Architecture" in `docs/data_contract.md`.
You choose the source when you run it; see the Colab notebooks
`ChestAI_M1_training_pipeline.ipynb` (builds the manifest, preprocesses,
runs QC) and `ChestAI_M1_inference.ipynb` (one scan or a folder of several,
no manifest, no persistence), and "Local DICOM data" in
`docs/data_contract.md`. Series selection (choosing a chest series among
several) is not built (the data is lung series only).

## Project layout

```
configs/data.yaml          all pipeline settings (spacing, size, windows, QC thresholds, sources, device)
docs/data_contract.md      what M1 hands to M3/M4, and why -- read this first
src/chestct/preprocessing/
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
  config.py                  load configs/data.yaml, incl. multiple named data `sources`
  dataset.py                  the M1 -> M3/M4 hand-off (label_cols optional, for training vs. inference; M1 itself never has labels)
src/chestct/inference.py     the inference front door -- one scan or a batch, shares pipeline.py's core
scripts/
  download_subset.py         one combined CT-RATE download from Hugging Face (--n-train/--n-val/--n-test/--seed; needs your own login)
  build_manifest.py          Step 3 CLI, one named source at a time (--source-name, --raw-dir, --n-train/--n-val/--n-test/--seed, --append); no labels, ever
  preprocess_all.py          Steps 5-9 CLI (parallel, resumable via cache fingerprint, --device)
  qc_report.py               Step 8 CLI
  stage_folder.py            copy scans from a Drive mount to local disk (resumable)
  dicom_tags.py              print what the DICOM tags of a scan contain (header-only)
tests/                       pytest, all using synthetic data -- no CT-RATE download needed to run them
```

## Setup

```bash
python -m pip install -e ".[dev]"
```

Torch is only needed for `chestct.preprocessing.dataset.ChestCTDataset.__getitem__`
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

2. **Download a small pilot subset** (e.g. 4 train + 1 val scans from
   CT-RATE's official TRAIN pool, 1 test scan from its official VALID pool,
   from the corrected `_fixed` folder by default -- fast, never lists the
   whole repository, and picked at random rather than "the first N
   alphabetically"):
   ```bash
   python scripts/download_subset.py --n-train 4 --n-val 1 --n-test 1 --seed 0 --out-dir data/raw
   ```
   Scans land flat in `data/raw/`; metadata (never labels -- this pipeline
   doesn't read them) lands flat in `data/metadata/`. Add
   `--max-combined-gb 1.5` to skip volumes whose estimated resample memory
   footprint would be too large for a small machine (uses the metadata
   already fetched -- no extra download needed just to screen candidates).

3. **Build the manifest** (Step 3 -- ids, patient-level splits; no labels).
   Run once per named source in `configs/data.yaml`'s `sources:`, with
   `--append` after the first, to accumulate several sources (e.g. CT-RATE
   and later local NHRD data) into one manifest. Use the SAME
   `--n-train`/`--n-val`/`--seed` as the download step so the split is
   reproducible:
   ```bash
   python scripts/build_manifest.py --source-name ctrate --builder ctrate \
     --raw-dir data/raw --metadata data/metadata/train_metadata.csv \
     --n-train 4 --n-val 1 --seed 0
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

- Read `docs/data_contract.md` first -- it defines the exact tensor shape,
  dtype, and value range you'll get, and the two `ChestCTDataset` modes
  (`all_lowres` for M3's scoring pass, `selected` for M4's full-resolution
  encoding).
- Nothing here is windowed until you ask for it (`chestct.preprocessing.dataset.slices_to_tensor`)
  -- the cache stores raw HU so windows can still change without re-running
  preprocessing.
- `configs/data.yaml` is the single source of truth for every tunable
  number. Don't hard-code spacing/size/window values elsewhere.
