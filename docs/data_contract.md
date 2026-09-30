# M1 data contract

This is the agreement between M1 (this pipeline) and M3 (slice selection) /
M4 (classification model): what M1 produces, in what format, and why. If any
of this needs to change, change it here first and tell the team, since M3/M4
code depends on it.

Scope: **CT-RATE (NIfTI) and local hospital data (DICOM)**, through the same
pipeline. See "Local DICOM data" below. Series selection (choosing a chest
series among many) is deliberately not built: the current data is lung series
only.

## Architecture: one shared core, three front doors

There is exactly one place the actual image transformation happens:
`chestct.preprocessing.pipeline.process_scan()` (load -> calibrate -> resample ->
crop -> resize -> QC). Everything else is a thin "front door" calling it:

1. **Training batch** (`scripts/preprocess_all.py` -> `chestct.preprocessing.preprocess.preprocess_one`):
   many scans, parallel, always saves to disk regardless of QC outcome
   (QC exclusion happens later, reviewed by a human via `qc_report.py`).
2. **Local batch import** (built): NHRD hospital data, already on disk (a
   Drive mount, an SSD, a server folder -- no download step), through the
   DICOM loader, which returns the same `Volume` shape as the NIfTI loader.
3. **Inference** (`chestct.inference.run_inference`): one scan, or a folder
   holding several, no labels, no manifest, no persistent cache. A QC
   failure or a hard error is recorded on that scan's own row
   (`passed=False`, no tensor) rather than raised -- one bad scan in a batch
   of many never stops the rest, the same log-and-continue shape
   `preprocess_all.py` uses for training. The caller (a real clinical UI, or
   M4) is expected to check `passed` per row before using a prediction.

This is deliberate, not just convenient: if training and clinical use ever
processed a scan even slightly differently, the model would see different
data in the clinic than it was trained on ("training-serving skew") --
sharing the core is a correctness requirement, not a style choice.

**Labels are not part of this pipeline (M1) at all, for either data
source, and never will be.** M1 produces images, patient ids and splits
only -- no `build_manifest*` function reads or joins a labels CSV. M4 joins
labels on its own side, by `volume_id`, when it needs them for supervision.
`ChestCTDataset(label_cols=None)` (the default) is the same class used by
both training and inference -- `label_cols` just names existing columns to
read off the manifest DataFrame handed to it, so a training caller merges
its own label source into its own copy of the manifest first, then passes
`label_cols=[...]`; inference omits it entirely, since a real scan has no
label -- producing one is the model's job.

## Multiple data sources (CT-RATE + local data, via config)

`configs/data.yaml`'s `sources:` section lists named data sources, each
just a format + a folder. A plain filesystem path works identically for a
CT-RATE download that landed locally, a folder on a laptop, or a mounted
university server -- the only thing ever special about CT-RATE is the
one-time download that populates it (`scripts/download_subset.py`); after
that, it's "just a folder of files" like any local source. Run
`scripts/build_manifest.py --source-name <name> --append` once per source
to accumulate all of them into one `manifest.csv`, distinguished by a
`source_name` column -- this is what lets training on CT-RATE and later
local adaptation on NHRD data share the exact same cache and Dataset class.

**Selection is a fixed amount, random and seeded, for both sources.**
`scripts/download_subset.py --n-train N --n-val N --n-test N --seed S` is
one combined run: it pulls `n_train`+`n_val` scans from CT-RATE's official
TRAIN pool and `n_test` scans from its official VALID pool (the held-out
benchmark set), replacing the old two-separate-runs flow. Selection is by
`random.Random(seed).sample(...)` over the candidate ids, not "the first N
alphabetically" -- a pilot subset should not be systematically biased
toward one corner of the patient id range. `scripts/build_manifest.py`
takes the same `--n-train`/`--n-val`/`--n-test`/`--seed` for local (folder)
sources too, via `chestct.preprocessing.manifest.assign_splits_by_amount`, which
picks exactly that many PATIENTS (never scans) at random; for CT-RATE, only
`--n-train`/`--n-val` are given here (the official VALID-pool patients
become `test` directly, with no further splitting). Stratified (label-aware)
selection was considered and deliberately not built for M1 -- see "Q&A"
below.

## GPU-accelerated resampling (optional)

`configs/data.yaml`'s `preprocess.device` (`cpu`/`auto`/`cuda`) lets
`spacing.py`/`resize.py` use `torch`'s GPU-based trilinear/bilinear
interpolation instead of `scipy` on CPU, when a GPU is actually available.
This isn't just a speed optimisation -- offloading the large intermediate
array to GPU VRAM instead of system RAM is what avoided a real OOM crash we
hit on a large CT-RATE volume in a CPU-only Colab session. `device: cpu`
(the default) always works with no GPU and no torch required; `auto`
upgrades only when it can.

## Three other real bugs fixed here, worth knowing about

**A high HU max is not a defect -- `hu_max_reasonable` was removed.** The QC
check used to flag any volume whose max HU exceeded 4000 as a possible
artifact. Real CT-RATE data proved this wrong: cross-checking against
CT-RATE's own labels showed "Medical material" (metal implants, contrast,
dense hardware) positive on 12.3% of real volumes, and a high max is exactly
what that finding looks like in the data -- a normal, wanted clinical
finding, not something to flag. See `configs/data.yaml`'s `qc:` section for
the current (metal-tolerant) thresholds.

**HU calibration is now explicit, not assumed.** CT-RATE scanners
legitimately use different `RescaleIntercept` values (confirmed: -1024 *and*
-8192 both occur on real, correctly-calibrated data, depending on the
scanner). The loader used to just trust whatever `nibabel` auto-applied
from each file's own header, with no check. `chestct.preprocessing.loader.ensure_calibrated_hu`
now explicitly checks whether a scan looks calibrated (a low-percentile
value already comfortably below -500) and, if not, applies that scan's own
`RescaleSlope`/`RescaleIntercept` (from the metadata CSV, threaded through
by `build_manifest.py` -> `preprocess_all.py`). `quality.py`'s QC check was
also fixed to match: it now flags a minimum *suspiciously near 0* (the real
uncalibrated signature), not "not exactly -1024" (which incorrectly flagged
legitimate scanner variation as broken).

**Stale cache reuse across a settings change is now impossible.** Every
cached `.npy` gets a sidecar recording a fingerprint of the exact
`PreprocessConfig` that produced it (`chestct.preprocessing.preprocess.config_fingerprint`).
`scripts/preprocess_all.py` only skips a volume if that fingerprint matches
the *current* config -- switching CT-RATE download folders, or changing any
preprocessing setting, correctly triggers a reprocess instead of silently
reusing old, differently-processed output (this is exactly what happened
before this fix: 4 of 5 test volumes kept stale, wrong-calibration data
after switching to the corrected download folder).

## What comes out of this pipeline

Three things, produced once per scan by `scripts/preprocess_all.py`:

1. **`data/cache/{volume_id}.npy`** -- one file per scan.
   - Shape `(N, 224, 224)`, where N is the number of axial slices (~200-400).
   - `int16`, **real Hounsfield Units**, e.g. air ≈ -1000, water ≈ 0, bone up
     to +1000/2000+.
   - **Not windowed, not scaled to [0, 1].** Windowing happens later, every
     time a scan is loaded for training (see "Why unwindowed" below).
   - Uncompressed, so slices can be read from disk without loading the whole
     file (`np.load(path, mmap_mode="r")[indices]`).

2. **`data/manifest.csv`** -- one row per scan: `volume_id`, `patient_id`,
   `scan_path`, `format`, `source_name`, `split` (train/val/test); for
   CT-RATE also `scan_id`, `reconstruction_id`, `source_split`; for DICOM
   also `series_uid` and the acquisition fields read from tags; plus (after
   preprocessing) `n_slices`, `spacing_z_mm` / `spacing_y_mm` /
   `spacing_x_mm`, `crop_shape`, `npy_path`. **No labels, ever** -- see
   "Architecture" above.

3. **`data/preprocessing_manifest.json`** -- the exact settings used
   (spacing, size, thresholds, a `version` string) plus a list of which
   volumes failed and why. Reproducibility record; also a fingerprint M4 can
   use to know when a cached-feature cache is stale.

Nobody downstream opens the `.npy` directly. `chestct.preprocessing.dataset.ChestCTDataset`
reads the `.npy` + manifest row together and hands out a ready tensor:

```
.npy (int16 HU, N slices)
  -> pick slices -> windowing -> (K, 3, 224, 224) float32 in [0, 1]
  -> optional ImageNet mean/std normalisation
  -> torch tensor, fed to the frozen 2D encoder
```

- **M3 (Stage A, scoring all slices):** `ChestCTDataset(..., mode="all_lowres")`
  returns every slice, downsampled (default 112x112) on the fly.
- **M4 (Stage C, encoding selected slices):** `mode="selected"` reads only
  the K chosen slice indices at full 224x224 resolution. M3 hasn't been built
  yet, so for now this mode expects the chosen indices as a JSON list in a
  `selected_indices` manifest column; until M3 writes that column, call
  `chestct.preprocessing.dataset.slices_to_tensor()` directly with your own indices.

## Decisions (Step 0)

| Decision | Value | Why |
|---|---|---|
| Slice size | **224x224** | Must be a multiple of 14 (DINOv2 ViT-*/14 patch size) -- 224 = 16x14 patches is the common default. Every scan must end up exactly this size because a PyTorch batch requires identical tensor shapes. |
| Windows / channels | **3**: lung `[-1000, 400]`, soft tissue `[-150, 250]`, all tissue `[-1000, 1000]` | Matches AnyMC3D / CT-RATE convention; also fits the 3-channel input a standard 2D encoder expects. |
| Where windowing runs | **In the Dataset, not in preprocessing** | The cached `.npy` holds raw HU. Windows can be changed during development without re-running the whole (expensive) preprocessing step, and get fixed only before final evaluation, per the proposal. |
| Slice axis / order | Axial, head-to-foot (RAS+ z-axis) | Standard; verified visually via QC montages. |
| Target spacing | **1.5 / 0.75 / 0.75 mm** (z, y, x) | Reference value from the CT-CLIP paper that built CT-RATE. |
| Resize mode | **"stretch"** (independent W, H resize) | Simple; matches CT-CLIP/AnyMC3D. Alternative "pad" (aspect-preserving + black padding) is implemented and available via config if the team wants less distortion later. |
| `.npy` format | int16, uncompressed, unwindowed HU | Fast partial reads for M4; half the size of float32; windows stay changeable. |
| Label mask | Not M1's concern | M1 never reads or stores labels for either data source (see "Architecture" above); a future label source (e.g. M2's report pipeline, or CT-RATE's own labels CSV) is joined by M4, by `volume_id`, on its own side. `ChestCTDataset(mask_cols=...)` exists for whoever does that join to expose an uncertain/not-mentioned state, if their label source has one. |
| ImageNet normalisation | Applied in the Dataset (`normalize_imagenet=True` by default) | DINOv2 (like most ImageNet-pretrained ViTs) expects this. |
| Splitting | By **patient**, a fixed amount (not a ratio), random and seeded | So reconstructions/scans of the same person never land in two splits, and a subset is reproducible from its `--seed`. CT-RATE's own `valid`-pool patients are always the held-out **test** set; `train`-pool patients are split into train/val by `--n-train`/`--n-val`. Local data draws all three amounts from one discovered pool. See `assign_splits_by_amount` in `chestct.preprocessing.manifest`. |

## Q&A worth keeping (came up while designing this)

**Do we need series selection for CT-RATE?** No. CT-RATE only contains
non-contrast chest CT -- the chest is already chosen. What it *does* have is
several **reconstructions** of the same scan (`train_1_a_1`, `train_1_a_2`,
...): the same raw scanner data rebuilt several ways (thin/thick slices,
sharp/smooth filter). These are not different body parts and not different
moments -- just several versions of the same chest. We keep them all, and
only enforce patient-level splitting so they never leak across train/test.
Series selection (finding the *chest* series among many) is a local-hospital
problem only (see Part B).

**Why random selection, not stratified (label-aware)?** M1 never reads
labels at all (see "Architecture" above), and CT-RATE's labels are
per-abnormality with heavy co-occurrence and no single natural stratum to
balance on -- a "representative" subset would need a policy decision (which
label(s) to balance, at what target rate) that belongs with M4/the modeling
plan, not with data acquisition. Kept simple for now: uniform random,
seeded, for both CT-RATE and local data. Revisit if a specific class turns
out under-represented once labels are joined downstream.

**`no_chest_train.txt` / `no_chest_valid.txt`**: CT-RATE's `metadata/` folder
has these two files, presumably listing volumes that don't actually contain
the chest. We could not confirm their exact contents without a Hugging Face
login. `scripts/build_manifest.py --no-chest-list ...` drops any volume
listed there -- **check what the file actually contains before relying on
this**, and update this note once confirmed.

**Does spacing give the same pixel count or the same real-world pixel size?**
Same real-world size (mm/voxel), not pixel count. A bigger patient's body
still ends up with more voxels after resampling to a common spacing than a
smaller patient's -- that's correct, not a bug (see `tests/test_spacing.py`
for a worked example). The step that forces a fixed pixel *count* (224x224)
is resize, which runs after spacing and after crop.

**What about air in the middle of the body (lungs, trachea, bowel gas) when
cropping?** It doesn't break anything. The crop's bounding box is set by the
*outer* edge of tissue (chest wall, ribs, spine, skin), which always
surrounds the lungs at the same slice. So even though lung air reads
"background" under the crop's tissue threshold, the box still spans the full
thorax, and the lungs simply sit inside it, keeping their real HU values
(nothing is zeroed or masked -- see `tests/test_crop.py`).

**CT-RATE has no single native size.** Confirmed from the CT-CLIP paper: raw
in-plane matrix is 512x512 for 65.4% of scans, 768x768 for 4.2%, 1024x1024
for 30.4%, with slice counts from 100-600. This is exactly why the spacing
and resize steps are required for every scan, not just the unusual ones.

## Local DICOM data

**Any folder, any layout.** `chestct.preprocessing.dicom_loader.discover_scans(root)`
walks whatever root it is given and groups the DICOM files found in each
folder by their `SeriesInstanceUID` tag -- matching how SimpleITK's
`GetGDCMSeriesIDs` and `dcm2niix` identify a scan, not by folder location. A
folder holding exactly one series is one scan, as before; a folder holding
several series (mixed together) now correctly becomes several scans, one per
series, instead of being flagged as a problem. NIfTI files anywhere under the
root are scans too. Nothing depends on the Drive layout --
`4214-26/P00001/S0001` today, something else on an SSD tomorrow. The root is
given at run time (`build_manifest.py --raw-dir ...`,
`preprocess_all.py --source-root NAME=PATH`), and the manifest stores only
paths **relative to that root**, never absolute ones.

**Metadata lives in the files.** DICOM has no separate CSV: rescale values,
pixel spacing, slice positions and orientation are read from tags in every
file. The loader always applies each slice's own `RescaleSlope`/
`RescaleIntercept` (DICOM has no automatic scaling), orders slices by their
real position along the scan axis (not by file name), measures slice spacing
from those positions (not the `SliceThickness` tag), and then hands the
geometry to the *same* nibabel-based standardisation the NIfTI loader uses --
so both formats end up in an identical (Z, Y, X) RAS+ convention. A test writes
one volume as NIfTI and as DICOM and checks the full pipeline output is
identical.

**What is checked** (a folder that fails is still reported, never guessed at):
`chestct.preprocessing.pipeline.process_scan()` on a folder given *without* a
`series_uid` still hard-errors if it holds more than one series (a caller
that hasn't disambiguated shouldn't get a silently-wrong scan); missing
tags, mixed image sizes, duplicate positions -> error with the reason;
non-uniform slice spacing (a missing slice) -> fails QC, because resampling
would distort the anatomy; a tilted slice axis -> flagged; compressed pixel
data with no decoder installed -> an error naming the transfer syntax and
what to install (`pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg`).

**Manifest for local data** (`manifest_builder: folder`): no labels are ever
read, and no split is assigned by `build_manifest_folder()` itself -- both
are the caller's job (`scripts/build_manifest.py --n-train/--n-val/--n-test/--seed`,
see "Multiple data sources" above), matching how the CT-RATE builder never
sets `split` either. Columns: `volume_id` (from the relative scan_path, plus
a short suffix of the series' own tag), `patient_id`, `scan_path`, `format`,
`n_files`, `series_uid`, and acquisition fields read from the tags
(manufacturer, model, kernel, slice thickness, series description, contrast
flag, transfer syntax). Acquisition dates and identifying tags are
deliberately never recorded. `P00001` repeats across the top folders, so a
patient is identified by `patient_path_depth: 1` (the top folder, kept as
the default/fallback) or, with `patient_id_source: dicom_tag`, by a hash of
the DICOM PatientID tag (falling back to the path if the tag is empty).
`scripts/dicom_tags.py` shows which of these the real files support.

**Getting files from Drive:** Colab's Drive mount only shows My Drive, so a
folder shared with you needs a shortcut added to My Drive first. Reading many
small DICOM files straight from a mount is slow, so
`scripts/stage_folder.py` copies the whole folder to local disk first
(resumable, retried); the notebook stages everything rather than letting you
pick folders by name, since random `--n-train`/`--n-val`/`--n-test` selection
happens afterwards, at the manifest step, over whatever was staged.
`stage_folder.py --only-folders` still exists as a lower-level escape hatch
for a manual partial copy, it just isn't surfaced as a notebook input.

**Stale cache, now also for the raw data.** The cache sidecar records the
raw input's file count and total size next to the config fingerprint. Earlier
the guard only compared the config, which would NOT have caught re-downloading
the same volume ids from a different CT-RATE folder; it now does.

**Not built:** series selection choosing a chest series among several
non-chest series (the current data is lung series only), stratified
(label-aware) subset selection (both sources currently select uniformly at
random -- see "Q&A" below), and any per-hospital protocol logic beyond the
per-source `qc:` overrides in `configs/data.yaml`.
