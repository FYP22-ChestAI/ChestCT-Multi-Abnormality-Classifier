# Preprocessing data contract

This is the agreement between the preprocessing pipeline (this repo) and the
model code that consumes its output (slice selection, classification): what it
produces, in what format, and why. If any of this needs to change, change it
here first and tell the team, since the model code depends on it. To *run* the
pipeline, see [README.md](README.md).

Scope: **CT-RATE (NIfTI) and local hospital data (DICOM)**, through the same
pipeline. Series selection (choosing a chest series among many) is deliberately
not built: the current data is lung series only.

## What comes out

Everything the model code needs is two things, plus a record:

1. **`data/cache/{volume_id}.npy`** -- one file per volume.
   - Shape `(N, 224, 224)`; N is the number of axial slices (~200-400).
   - `int16`, **real Hounsfield Units** (air ≈ -1000, water ≈ 0, bone up to
     +1000/2000+).
   - **Not windowed, not scaled, one channel.** Windowing and any other
     model-input shaping belong to the model code (see "Windowing" below).
   - Uncompressed, so slices can be read without loading the whole file:
     `np.load(path, mmap_mode="r")[indices]`.
   - Next to it, `{volume_id}.meta.json`: the settings fingerprint, the raw
     input's signature, and the per-volume stats. Bookkeeping, not an interface.

2. **`data/manifest.csv`** -- one row per volume:
   - always: `volume_id`, `patient_id`, `scan_path`, `format`, `source_name`,
     `ingest_chunk`, `split`, `qc_passed`, `n_slices`, `spacing_z_mm` /
     `spacing_y_mm` / `spacing_x_mm`, `crop_shape`, `npy_path`;
   - CT-RATE rows also: `scan_id`, `reconstruction_id`, `source_split` (which
     official pool the volume came from) and CT-RATE's own metadata columns;
   - DICOM rows also: `series_uid` and acquisition fields read from the tags
     (manufacturer, model, kernel, slice thickness, series description, contrast
     flag, transfer syntax). Dates and identifying tags are never recorded.
   - `split` is one of `train` / `val` / `test`, `excluded` (the volume failed
     QC or was not preprocessed) or `unassigned` (its patient has no frozen
     assignment yet). **Use only `train` / `val` / `test` rows.**
   - **No labels, ever.** The model code joins labels on its own side, by
     `volume_id`.

3. **`data/preprocessing_manifest.json`** -- the exact settings that produced the
   cache, a fingerprint of them, and every volume that failed. A reproducibility
   record, and a fingerprint the model code can compare to know a derived cache
   (e.g. precomputed embeddings) is stale.

`data/splits/<source>.csv` holds the frozen patient -> split assignment the
manifest's `split` column is joined from.

## Windowing is the model code's job

The cache is raw HU on purpose. Windowing turns it into 3 float32 channels,
which is **6x** the bytes (3 channels x 4 bytes vs 1 channel x 2 bytes); storing
that would make the cache ~3 TB instead of ~560 GB. It is also model-input
shaping, so the windows stay changeable without re-running preprocessing.

The windows the project settled on (AnyMC3D / CT-RATE convention; also the
3-channel shape an ImageNet-pretrained 2D encoder expects) are:

| channel | window (HU) |
|---|---|
| lung | `[-1000, 400]` |
| soft tissue | `[-150, 250]` |
| all tissue | `[-1000, 1000]` |

```python
import numpy as np

WINDOWS = {"lung": (-1000, 400), "soft_tissue": (-150, 250), "all_tissue": (-1000, 1000)}

def to_channels(hu_slices: np.ndarray) -> np.ndarray:
    """(K, H, W) int16 HU -> (K, 3, H, W) float32 in [0, 1]."""
    chans = [(np.clip(hu_slices, lo, hi).astype(np.float32) - lo) / (hi - lo) for lo, hi in WINDOWS.values()]
    return np.stack(chans, axis=1)

slices = np.load("data/cache/train_1_a_1.npy", mmap_mode="r")[[40, 41, 88]]  # reads only those slices
x = to_channels(slices)   # then ImageNet mean/std normalisation, if the encoder wants it
```

`ct_preprocessing.inference.run_inference` returns the same preprocessed volume
(`hu`, int16) as the cache holds, so training (from the cache) and inference
(from `run_inference`) go through identical windowing in the model code.

## Architecture: one shared core, two front doors

There is exactly one place the image transformation happens:
`ct_preprocessing.pipeline.process_scan()` -- load -> calibrate HU -> resample ->
crop -> resize -> QC. Everything else calls it:

1. **Ingest** (`scripts/preprocessing/ingest.py` ->
   `ct_preprocessing.ingest.engine`): fetches a chunk of raw scans, runs
   `process_scan`, saves the cache, deletes the raw data, repeats. Always saves
   regardless of QC outcome; exclusion happens later, after a human has looked.
2. **Inference** (`ct_preprocessing.inference.run_inference`): one scan, or a
   folder of several; nothing is written to disk. A QC failure or hard error is
   recorded on that scan's row (`passed=False`, no volume) rather than raised,
   so one bad scan never stops the rest.

This is deliberate: if training and clinical use ever processed a scan even
slightly differently, the model would see different data in the clinic than it
was trained on ("training-serving skew"). Sharing the core is a correctness
requirement, not a style choice.

## Data flow, and why it is chunked

Raw data is far bigger than the server's disk (CT-RATE ~21 TB, NHRD ~0.9 TB, a
server with ~850 GB), but the cache is ~20x smaller (a raw scan of 350-450 MB
becomes ~20 MB), so **raw data only has to pass through, once**:

```
fetch a chunk -> preprocess into data/cache -> delete the raw chunk -> next chunk
```

* **CT-RATE** chunks come from `data/worklists/ctrate.csv` (`make_worklist.py`);
  each volume is downloaded by its exact path from Hugging Face.
* **NHRD** chunks are the `.zip` / `.tar` archives (whole patient folders,
  ~10 GB each) found in a Drive folder; one archive is one chunk.
* A chunk's progress is a `.done` marker, so the run can be stopped and resumed
  at any point, and a half-finished chunk does not re-fetch volumes already cached.
* Free disk is checked before every chunk; below `min_free_gb` the run stops
  cleanly instead of crashing the server.

**Consequence worth planning around:** because raw data is deleted, changing the
`preprocess:` settings after a large ingest means fetching everything again.
Settle them with a small pilot run first.

### The worklist is not a split

`make_worklist.py` only decides what is worth ingesting when not everything fits:
the whole `valid_fixed` pool (the test set, kept whole so numbers stay comparable
with published CT-RATE results) and, from `train_fixed`, one reconstruction per
scan, in a seeded random order of patients, optionally capped. The order is a
seeded patient shuffle and a cap truncates it, so the worklist is
*prefix-stable*: raising or lowering `max_train_patients` never changes a chunk
that was already ingested.

## Splitting: by patient, once, after QC, frozen

`assign_splits.py` runs after ingest, merge and QC:

* **CT-RATE:** test = patients from the official `valid_fixed` pool (a rule, not
  a draw); val = `n_val_patients` drawn from train-pool patients (seeded); the
  rest = train. CT-CLIP's authors use that valid pool as their held-out
  benchmark, not for tuning, which is why it maps to *test* here.
* **NHRD:** no official split, so `n_val_patients` and `n_test_patients` are
  drawn (seeded); the rest = train. Patients are grouped by top folder.
* Only **QC-passed** volumes count, so a patient whose scans all failed is not
  assigned, and val/test counts mean usable patients.
* The assignment is written to `data/splits/<source>.csv` and **never rewritten**.
  Running again only assigns patients not in the file yet (a top-up), leaving
  every earlier assignment untouched. Freeze it before the first training run.

Why this is leak-free: nothing in this pipeline is fitted to the data (fixed
spacing, fixed crop threshold, fixed size, no dataset-wide mean or std), so
preprocessing before the split cannot leak anything into the test set. Splitting
after QC also means failed scans never leave holes in val/test. Splitting is by
**patient**, so the reconstructions and scans of one person never land in two
splits; CT-RATE patient ids are pool-qualified (`train_1`, `valid_1`) so a
patient number is never confused across its two official pools.

Stratified (label-aware) selection was considered and deliberately not built:
M1 never reads labels, and CT-RATE's labels are per-abnormality with heavy
co-occurrence and no single natural stratum. Imbalance is better handled in the
loss (e.g. asymmetric / class-weighted) by the model code.

## Decisions

| Decision | Value | Why |
|---|---|---|
| Slice size | **224x224** | A multiple of 14 (DINOv2 ViT-*/14 patch size). Every volume must end up the same size because a batch needs identical tensor shapes. |
| Cache dtype | int16, uncompressed, unwindowed | Fast partial reads; 6x smaller than windowed float32 channels; windows stay changeable. |
| Slice axis / order | Axial, head-to-foot (RAS+ z-axis) | Standard; verified visually via QC montages. |
| Target spacing | **1.5 / 0.75 / 0.75 mm** (z, y, x) | Reference value from the CT-CLIP paper that built CT-RATE. |
| Resize mode | **"stretch"** | Simple; matches CT-CLIP/AnyMC3D. "pad" (aspect-preserving) is available via config. |
| Reconstructions | one per scan for the train pool; all for the test pool | The two reconstructions of a scan are near-duplicate rebuilds; keeping both halves the number of distinct patients for the same disk. The test pool stays whole for comparability. |
| Labels | not M1's concern | The model code joins them by `volume_id`. |

## Real bugs fixed along the way, worth knowing about

**A high HU max is not a defect -- `hu_max_reasonable` was removed.** The QC check
used to flag any volume whose max HU exceeded 4000. Cross-checking against CT-RATE's
own labels showed "Medical material" (metal implants, contrast, dense hardware)
positive on 12.3% of real volumes, and a high max is exactly what that finding
looks like -- a normal clinical finding, not something to flag.

**HU calibration is explicit, not assumed.** CT-RATE scanners legitimately use
different `RescaleIntercept` values (-1024 *and* -8192 both occur on correctly
calibrated data). `ct_preprocessing.loader.ensure_calibrated_hu` checks whether a
scan looks calibrated (a low percentile comfortably below -500) and, only if not,
applies that scan's own `RescaleSlope`/`RescaleIntercept` (from the metadata CSV
for NIfTI; DICOM applies each slice's own). QC flags a minimum *suspiciously near
0* (the real uncalibrated signature), not "not exactly -1024".

**A stale cache can no longer be reused silently.** Every cached `.npy` has a
sidecar with a fingerprint of the exact `PreprocessConfig` that produced it, and a
signature of the raw input. A volume is skipped only if both still match. The
compute device is deliberately not part of the fingerprint (CPU and GPU produce the
same cache, and raw data is gone after ingest, so a false "stale" could not even be
repaired); once the raw file has been deleted, the fingerprint alone decides.

**Downloads must not be stored twice.** `huggingface_hub` keeps every download in a
cache even after our copy is deleted, which would fill the disk. The CT-RATE
fetcher points that cache at a scratch folder on the same filesystem and *moves*
(renames) the file out of it.

## GPU-accelerated resampling (optional)

`preprocess.device` (`cpu` / `auto` / `cuda`) lets spacing and resize use `torch`
instead of `scipy` when a GPU exists. Beyond speed, offloading the large
intermediate array to VRAM is what avoided a real out-of-memory crash on a big
CT-RATE volume. `cpu` (the default) always works with no GPU and no torch. On a GPU
use `--workers 1` so several processes do not fight over one device.

## Q&A worth keeping

**Do we need series selection for CT-RATE?** No: it contains only non-contrast
chest CT. What it has is several *reconstructions* of the same scan
(`train_1_a_1`, `train_1_a_2`): the same raw data rebuilt with different
slice thickness / filter, not different body parts. Series selection is a
local-hospital problem only.

**`no_chest_train.txt` / `no_chest_valid.txt`:** CT-RATE's `metadata/` folder has
these two files, presumably volumes that do not contain the chest. Their contents
were never confirmed, so nothing here uses them. Check before relying on them.

**Does spacing give the same pixel count or the same real-world pixel size?** The
same real-world size (mm/voxel). A bigger patient still ends up with more voxels
than a smaller one -- correct, not a bug. The step that forces a fixed pixel
*count* (224x224) is resize, after spacing and crop.

**What about air in the middle of the body when cropping?** It does not break
anything. The crop box is set by the *outer* edge of tissue (chest wall, ribs,
spine, skin), which always surrounds the lungs on the same slice, so the lungs sit
inside the box keeping their real HU values.

**CT-RATE has no single native size.** In-plane 512x512 for 65.4% of scans,
768x768 for 4.2%, 1024x1024 for 30.4%, slice counts 100-600 (CT-CLIP paper).
That is why spacing and resize are required for every scan.

## Local DICOM data

**Any folder, any layout.** `ct_preprocessing.dicom_loader.discover_scans(root)`
walks the root it is given and groups the DICOM files in each folder by their
`SeriesInstanceUID` tag -- how SimpleITK and `dcm2niix` identify a scan -- not by
folder location. A folder holding one series is one scan; a folder holding several
mixed series becomes several scans. Paths in the manifest are relative to the
scratch folder, never absolute.

**Metadata lives in the files.** The loader applies each slice's own
`RescaleSlope`/`RescaleIntercept`, orders slices by their real position along the
scan axis (not by file name), measures slice spacing from those positions (not the
`SliceThickness` tag), then hands the geometry to the same nibabel-based
standardisation the NIfTI loader uses, so both formats end up in an identical
(Z, Y, X) RAS+ convention. A test writes one volume as NIfTI and as DICOM and
checks the pipeline output is identical.

**What is checked** (a failing folder is reported, never guessed at): missing tags,
mixed image sizes, duplicate positions -> error with the reason; non-uniform slice
spacing (a missing slice) -> fails QC, because resampling would distort the anatomy;
a tilted slice axis -> flagged; compressed pixel data with no decoder -> an error
naming the transfer syntax and what to install
(`pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg`).

**Patients.** `P00001` repeats across the top folders, so a patient is identified by
`patient_path_depth: 1` (the top folder) or by a hash of the DICOM `PatientID`
tag. `patient_id_source: auto` (the default) uses the tag only when it is present
AND distinct on every DICOM scan, and otherwise the folder; real hospital data is
routinely anonymised, which strips the tag. The decision is made once, on the first
archive, and remembered, so every later chunk groups patients the same way.

**Archives.** Each `.zip` / `.tar` holds *whole* patient folders (never half a
patient: a half series fails QC) and no patient appears in two archives (the merge
drops duplicates by `volume_id`, but do not rely on it). Unpacking verifies
integrity as a side effect -- zip CRCs are checked while reading, a truncated
archive fails outright -- so a half-uploaded archive is rejected and retried on the
next run instead of being silently used.

**Not built:** series selection among several non-chest series, stratified
(label-aware) subset selection, and any per-hospital protocol logic beyond the
per-source `qc:` overrides in `configs/preprocessing.yaml`.
