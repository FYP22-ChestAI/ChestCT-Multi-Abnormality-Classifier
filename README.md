# ChestCT-Multi-Abnormality-Classifier

Part of the ChestAI project (Resource-Efficient Multi-Abnormality Chest CT
Classification and Local Adaptation). This repo currently holds the **CT
preprocessing pipeline**: it turns raw chest CT -- the public **CT-RATE** dataset and
the local **NHRD** hospital DICOM data -- into a compact, model-ready cache of
Hounsfield-unit volumes plus a manifest with frozen patient-level train / val / test
splits. It also holds the **model side** that consumes that output (`ct_model`):
stage 2, a swappable 2D slice encoder (DALE-CT-2S first) that writes per-slice
embeddings; stage 3, MIL aggregation of those embeddings (gated ABMIL now, query-based /
QGMIL later); stage 4, the 18-abnormality classifier; plus experiment records and optional
per-label evidence (which slices, and where in them). Report-to-label extraction lives in a sibling repo in this GitHub
organization (`ChestCT-Report2Label`).

The pipeline processes data in chunks (fetch → preprocess → delete the raw data), so
datasets far larger than the server's disk can be handled unattended, and it can be
stopped and resumed at any time.

| Read this | For |
|---|---|
| **[docs/preprocessing/README.md](docs/preprocessing/README.md)** | **How to run the pipeline**: setup, the scripts and their arguments, CT-RATE and NHRD walkthroughs, running unattended, config, troubleshooting |
| [docs/preprocessing/data_contract.md](docs/preprocessing/data_contract.md) | What the pipeline produces (cache and manifest format) and why: the agreement with the model code |
| **[docs/model/README.md](docs/model/README.md)** | **How to run the model side**: stage 2 (encode), stages 3-4 (train, evaluate, compare), evidence |
| [docs/model/architecture.md](docs/model/architecture.md) | Stages 2-4 and why they are built this way: store format, ABMIL, imbalance, evaluation, evidence, roadmap |

## Quickstart

```bash
python -m pip install -e ".[dev]"
```
```bash
python scripts/preprocessing/make_worklist.py        # CT-RATE: plan the run "train-sharp"
```
```bash
python scripts/preprocessing/ingest.py --source ctrate --run train-sharp --max-chunks 1   # calibration run
```

Then the full run, merge, QC and split: see the
[pipeline README](docs/preprocessing/README.md). Every script runs with defaults from
`configs/preprocessing.yaml`; any default can be overridden with an argument.

Then stage 2, per-slice embeddings for a run (see the [model README](docs/model/README.md)):

```bash
python -m pip install -e ".[dev,model]"
```
```bash
python scripts/model/check_encoder.py --run train-sharp
```
```bash
python scripts/model/encode_volumes.py --run train-sharp
```

## Control panel (Streamlit)

Every script can also be run from a browser UI. It shows each step in pipeline order, with every argument
pre-filled with the value the script would use anyway, read from the same YAML configs. Hover over ⓘ on a
field to see what it means and which config the default comes from. Change only what you need: the
page shows the exact command (only changed arguments are added). **Run** starts it as a background job
that keeps going if the browser is closed. **Jobs** shows live logs and a Stop button, and **Results**
compares finished experiments.

```bash
python -m pip install -e ".[ui]"
```
```bash
streamlit run app/main.py
```

**Run whole pipeline** does every step in one click: choose the run, the experiments and the seeds. Each
step is checked against what is already saved (worklist, ingested chunks, manifest, QC report, splits,
embeddings, trained runs by config hash, evaluations), and only the missing steps run, in order, as one
background job that stops at the first failure. Settings you changed on a step's own page are used too.

On the server, open it from your laptop through an SSH tunnel (`ssh -L 8501:localhost:8501 <server>`,
then http://localhost:8501). Job logs go to `outputs/ui_jobs/`.

To add a script: give it a module-level `build_parser()` (like `scripts/model/train_mil.py`), then add one
`ScriptSpec` to [app/registry.py](app/registry.py). Its fields are read from the parser. Optional extras:
plain-language labels and tooltips in [app/arg_docs.yaml](app/arg_docs.yaml), and a defaults resolver in
`app/defaults.py`. New experiment YAMLs, runs and encoders appear in the dropdowns automatically. To make
a new script part of the one-click pipeline, also add a `Stage` to `app/pipeline.py` and a status check
("is it done?") to `app/status.py`.

## Project layout

```
configs/preprocessing.yaml        all defaults (paths, sources, ingest, split, preprocess, QC)
configs/model/                    encode.yaml (stage-2 defaults), encoders/<name>.yaml (one backbone each),
                                  experiments/<name>.yaml (stages 3+4: ABMIL, per-label ABMIL, mean-pool baseline)
docs/preprocessing/               README.md (how to run) and data_contract.md (what comes out)
docs/model/                       README.md (how to run stage 2) and architecture.md (stages, store format, roadmap)
src/ct_preprocessing/             the package
  pipeline.py                     the ONE shared core: load -> calibrate -> resample -> crop -> resize -> QC
  loader.py, dicom_loader.py,     NIfTI / DICOM loaders -> one standardised Volume (RAS+, real HU)
  loaders.py
  spacing.py, crop.py, resize.py  the transformation steps (optional GPU path)
  quality.py                      automatic QC checks + montage images
  preprocess.py                   preprocess_one: saves the cache + fingerprint sidecar
  manifest.py                     manifest rows (CT-RATE and folder builders); no labels, no split
  config.py                       typed config loading, with validation
  runs.py                         runs: one folder per worklist, sharing one cache
  kernels.py                      sharp / soft kernel classes (configs/kernel_classes.csv)
  cache_record.py                 the cache remembers its preprocessing settings
  inference.py                    run_inference: the same core for one scan or a batch, nothing saved
  ingest/                         the chunked ingest engine
    worklist.py, ctrate.py          CT-RATE: what to fetch, and fetching it from Hugging Face
    archives.py                     NHRD: .zip/.tar archives from Drive (rclone) or a folder
    engine.py                       the resumable ingest loop, disk guard, preprocessing batches
    merge.py, splits.py, state.py   merge, frozen patient-level splits, resume markers
    labels.py                       CT-RATE / NHRD labels, joined into the manifest at merge time
src/ct_model/                     the model side (stages 2-4)
  encoders/                       stage 2: SliceEncoder interface, HU input transforms, timm ViT backbones, LoRA (TODO, phase 2)
  embeddings/                     the embedding store (derived, fingerprinted) and the encode loop
  data/                           manifest -> volume records, slice samplers, datasets (HU slices, embedding bags)
  aggregators/ heads/ models/     stage 3 (ABMIL, mean pool; query MIL TODO) and 4 (linear head), composed
  training/                       experiment config, losses, metrics, trainer, evaluation, experiment records
  explain/                        optional evidence: exact slice contributions, in-slice Grad-CAM
scripts/preprocessing/            make_worklist.py, ingest.py, merge_manifests.py, qc_report.py, assign_splits.py,
                                  make_kernel_table.py, kernel_survey.py
scripts/model/                    encode_volumes.py, check_encoder.py, train_mil.py, evaluate_mil.py,
                                  summarize_experiments.py, explain_volume.py
app/                              Streamlit control panel: registry.py (scripts by phase), one generic page per script,
                                  background jobs (outputs/ui_jobs/), results viewer
tests/preprocessing/, tests/model/, tests/ui/  pytest, synthetic data and fake backends: no download or network needed
notebooks/                        Colab notebooks for trying the pipeline on a small sample; encoder_embeddings_report,
                                  experiments_report, evidence_viewer
```

## Setup notes

`pip install -e ".[dev]"` installs everything needed to run and test. Optional extras:
`".[torch]"` for GPU resampling, `".[dicom-codecs]"` if the DICOM files are compressed,
`".[model]"` (torch, timm, safetensors) for the model side.
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
