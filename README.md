# ChestCT-Multi-Abnormality-Classifier

Part of the ChestAI project (Resource-Efficient Multi-Abnormality Chest CT
Classification and Local Adaptation). This repo currently holds the **CT
preprocessing pipeline**: it turns raw chest CT -- the public **CT-RATE** dataset and
the local **NHRD** hospital DICOM data -- into a compact, model-ready cache of
Hounsfield-unit volumes plus a manifest with frozen patient-level train / val / test
splits. Slice selection and classification (to be added to this repo) consume that
output. Report-to-label extraction lives in a sibling repo in this GitHub
organization (`ChestCT-Report2Label`).

The pipeline processes data in chunks (fetch → preprocess → delete the raw data), so
datasets far larger than the server's disk can be handled unattended, and it can be
stopped and resumed at any time.

| Read this | For |
|---|---|
| **[docs/preprocessing/README.md](docs/preprocessing/README.md)** | **How to run the pipeline**: setup, the scripts and their arguments, CT-RATE and NHRD walkthroughs, running unattended, config, troubleshooting |
| [docs/preprocessing/data_contract.md](docs/preprocessing/data_contract.md) | What the pipeline produces (cache and manifest format) and why: the agreement with the model code |

## Quickstart

```bash
python -m pip install -e ".[dev]"
```
```bash
python scripts/preprocessing/make_worklist.py        # CT-RATE: decide what to ingest
```
```bash
python scripts/preprocessing/ingest.py --source ctrate --max-chunks 1   # calibration run
```

Then the full run, merge, QC and split: see the
[pipeline README](docs/preprocessing/README.md). Every script runs with defaults from
`configs/preprocessing.yaml`; any default can be overridden with an argument.

## Project layout

```
configs/preprocessing.yaml        all defaults (paths, sources, ingest, split, preprocess, QC)
docs/preprocessing/               README.md (how to run) and data_contract.md (what comes out)
src/ct_preprocessing/             the package
  pipeline.py                     the ONE shared core: load -> calibrate -> resample -> crop -> resize -> QC
  loader.py, dicom_loader.py,     NIfTI / DICOM loaders -> one standardised Volume (RAS+, real HU)
  loaders.py
  spacing.py, crop.py, resize.py  the transformation steps (optional GPU path)
  quality.py                      automatic QC checks + montage images
  preprocess.py                   preprocess_one: saves the cache + fingerprint sidecar
  manifest.py                     manifest rows (CT-RATE and folder builders); no labels, no split
  config.py                       typed config loading, with validation
  inference.py                    run_inference: the same core for one scan or a batch, nothing saved
  ingest/                         the chunked ingest engine
    worklist.py, ctrate.py          CT-RATE: what to fetch, and fetching it from Hugging Face
    archives.py                     NHRD: .zip/.tar archives from Drive (rclone) or a folder
    engine.py                       the resumable ingest loop, disk guard, preprocessing batches
    merge.py, splits.py, state.py   merge, frozen patient-level splits, resume markers
scripts/preprocessing/            make_worklist.py, ingest.py, merge_manifests.py, qc_report.py, assign_splits.py
tests/preprocessing/              pytest, synthetic data and fake backends: no download or network needed
notebooks/                        Colab notebooks for trying the pipeline on a small sample
```

## Setup notes

`pip install -e ".[dev]"` installs everything needed to run and test. Optional extras:
`".[torch]"` for GPU resampling, `".[dicom-codecs]"` if the DICOM files are compressed.
CT-RATE needs your own Hugging Face login and NHRD needs an rclone remote for Drive;
both are explained in the [pipeline README](docs/preprocessing/README.md#setup).

## Notes for whoever builds the model side

- Read [docs/preprocessing/data_contract.md](docs/preprocessing/data_contract.md)
  first: it defines the cache and manifest you consume.
- **The cache is raw int16 HU, one channel.** Windowing (e.g. the 3-channel lung / soft
  tissue / all-tissue convention) is the model code's job; the data contract has a
  ready snippet and the agreed window ranges.
- Use only manifest rows with `split` in `train` / `val` / `test`.
- `configs/preprocessing.yaml` is the single source of truth for every tunable number.
  Do not hard-code spacing or size values elsewhere.
