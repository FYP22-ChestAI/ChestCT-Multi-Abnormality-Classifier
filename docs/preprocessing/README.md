# Preprocessing pipeline

Turns raw chest CT -- the public **CT-RATE** dataset and the local **NHRD** hospital
DICOM data -- into a compact, model-ready cache, on a server that is far too small
to hold the raw data. It runs unattended, in chunks, and can be stopped and resumed
at any point.

```
 raw chunk  ──►  preprocess  ──►  data/cache/*.npy   (permanent, ~20 MB per volume)
 (fetched)       (shared core)    data/manifests/    (one small manifest per chunk)
     │                                  │
     └── deleted after every chunk      └──► merge ► QC ► assign splits ► runs/<source>/<run>/manifest.csv
```

What it hands to the model code: **`data/cache/`** (raw int16 HU, one `.npy` per
volume, shared by every run) and, per run, **`data/runs/<source>/<run>/manifest.csv`**
(one row per volume, with its kernel, labels and a frozen patient-level `split`). See
[data_contract.md](data_contract.md) for the exact format and the reasoning behind it.

## Contents

1. [Setup](#setup)
2. [The scripts](#the-scripts)
3. [Runs: several plans, one cache](#runs-several-plans-one-cache)
4. [Kernels: sharp and soft](#kernels-sharp-and-soft)
5. [CT-RATE, start to end](#ct-rate-start-to-end)
6. [NHRD, start to end](#nhrd-start-to-end)
7. [Labels](#labels)
8. [Running it unattended](#running-it-unattended)
9. [Configuration and overrides](#configuration-and-overrides)
10. [Disk budget](#disk-budget)
11. [Safety rules](#safety-rules)
12. [Troubleshooting](#troubleshooting)
13. [Development](#development)

## Setup

Run these on the server, from the repository root.

```bash
python -m pip install -e ".[dev]"
```

Optional extras: `".[torch]"` for GPU resampling (`--device cuda`),
`".[dicom-codecs]"` if the DICOM files are compressed.

**CT-RATE** needs your own Hugging Face account. Accept the terms on the
[CT-RATE dataset page](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE),
then log in once. Never share or commit the token.

```bash
huggingface-cli login
```

**NHRD** is read from Google Drive through [rclone](https://rclone.org). On a server
with no browser, set the remote up in two steps:

```bash
rclone config
```

Choose `n` (new remote), name it `gdrive`, storage `drive`, and answer **`n`** to
"Use auto config?". rclone prints a command like `rclone authorize "drive" "eyJ..."`.
Run exactly that command on a computer **with a browser** (install rclone there too),
log in, and paste the JSON token it prints back into the server prompt. Check it:

```bash
rclone lsd gdrive:
```

The token is a key to the Drive account: run `chmod 600 ~/.config/rclone/rclone.conf`,
and revoke it from your Google account's third-party access page when finished.
Check that the data-handling approval for the hospital data covers using Drive.

**tmux** (to survive logging out): `tmux new -s ingest`, detach with `Ctrl+b d`,
reattach with `tmux a -t ingest`.

## The scripts

All live in `scripts/preprocessing/`. Run them from the repository root. **Every
one runs with no arguments beyond the ones marked required**: defaults come from
[`configs/preprocessing.yaml`](../../configs/preprocessing.yaml), any command-line
argument overrides its default for that one run, and the first line of output
echoes the settings actually used.

| Script | Needed for | What it does | Key arguments (all optional unless marked) |
|---|---|---|---|
| `make_kernel_table.py` | CT-RATE | Drafts `configs/kernel_classes.csv`: which (manufacturer, kernel) pairs are sharp or soft. A starting point for a person to check. | `--metadata`, `--out`, `--add`, `--force` |
| `make_worklist.py` | CT-RATE | Downloads the metadata (~16 MB) and label (~3 MB) CSVs, no images, and creates a new **run** with its `worklist.csv`: exactly which volumes to ingest. A work list, **not a split**. Never overwrites a run. | `--name`, `--train-kernel`, `--chunk-size`, `--max-train-patients`, `--max-test-patients`, `--max-combined-gb`, `--seed`, `--from-run` + `--train-patients` + `--val-patients`, `--survey` |
| `ingest.py` | both | The continuous script: for each pending chunk of a run, fetch → preprocess into the shared cache → record manifest rows → delete the raw data. Resumable. | `--source` **(required)**, `--run`, `--max-chunks`, `--min-free-gb`, `--workers`, `--device`, `--drive-remote`, `--retry-failed`, `--allow-new-settings`, `--dry-run`, `--status` |
| `merge_manifests.py` | both | Merges the run's per-chunk manifests into its `manifest.csv` (kernel columns, labels, frozen split) and writes `preprocessing_manifest.json`. | `--source` **(required)**, `--run`, `--labels-file`, `--patients-file` |
| `qc_report.py` | both | Quality-checks every cached volume of the run: `qc_report.csv` and montage images to look at. | `--source` **(required)**, `--run`, `--montage-every` |
| `assign_splits.py` | both | Assigns train / val / test **by patient, once**, and freezes it (one split file per source, shared by every run). | `--source` **(required)**, `--run`, `--n-val-patients`, `--n-test-patients`, `--seed`, `--skip-qc`, `--dry-run` |
| `kernel_survey.py` | CT-RATE | After a `make_worklist.py --survey N` run is ingested: measures which reconstruction of each scan is the sharper one, to check the kernel table against data. | `--source`, `--run` |

Every script also takes `--config PATH` (default `configs/preprocessing.yaml`) and
`--help`. `--run NAME` picks the run; with only one run on disk it is chosen for you, with
several the script lists them and asks.

`ingest.py` exit codes: `0` finished (or stopped by a limit you set), `1` some chunk
failed, `2` stopped early (low disk, or repeated failures).

## Runs: several plans, one cache

A **run** is a named plan -- "the sharp reconstruction of these patients" -- together with everything
derived from it. Each run has its own folder, and runs **coexist: nothing is ever overwritten or deleted**.

```
data/
  cache/                        the preprocessed volumes ({volume_id}.npy + .meta.json), shared, only grows
  splits/<source>.csv           frozen patient -> split, one per source, shared by every run
  runs/<source>/<run name>/
      run.json                  the settings that made the run
      worklist.csv              what to download (CT-RATE)
      chunk_manifests/  state/  per-chunk rows, done / failed markers
      manifest.csv  qc_report.csv  qc_montages/  preprocessing_manifest.json  splits.csv
```

Why it is built this way:

- **The cache is shared**, so a second run only downloads what no earlier run cached. A soft run
  after a sharp run does not fetch the test pool again.
- **The split is shared**, because it is by patient: a patient is in the same split in every run, so
  runs can never leak into each other, and a sharp-vs-soft comparison uses the same validation patients.
  Each run folder keeps a copy (`splits.csv`) limited to its patients.
- **The cache remembers its preprocessing settings** (`data/cache/.cache_fingerprint.json`). If the
  `preprocess:` block changed, `ingest.py` refuses to write into it, because that would overwrite volumes
  an earlier run's manifest points to. `--allow-new-settings` replaces them on purpose (old volumes go stale).
- NHRD has one run, called `main`, created the first time you ingest it.

A later run is a new `make_worklist.py` call with another name or another setting; it fails if a run of that
name exists. To build the second run from patients the first has already split, see below.

**A run built from a finished run** takes a subset of the patients its parent already split, with the number
you ask for from each split, and each patient keeps the split it has:

```bash
python scripts/preprocessing/make_worklist.py --train-kernel soft --from-run train-sharp --train-patients 800 --val-patients 200
```

This picks 800 patients from the parent's train split and 200 from its val split (seeded and repeatable), so
there is no leakage and the train / val mix is the one you chose. The parent needs its split first
(`assign_splits.py --run train-sharp`).

## Kernels: sharp and soft

One scan is usually reconstructed twice from the same raw data, with different **kernels**: a *sharp* one
(fine detail, more noise; lung findings) and a *soft* one (smooth; mediastinum). CT-RATE stores both, as
`train_1_a_1` and `train_1_a_2`. Which one a run trains on is a setting:

- `train_kernel: sharp` (default) takes the sharp reconstruction of each train scan; `soft` takes the soft one.
  A scan that has no reconstruction of that kernel is skipped. A kernel that is neither (bone, ultra-high-resolution,
  or simply not in the table) is never chosen.
- The **test pool** (`valid_fixed`) is always taken whole, with **every reconstruction**, so a model can be
  tested on sharp and on soft, and results stay comparable with published CT-RATE numbers.
- Which kernels are sharp or soft comes from `configs/kernel_classes.csv`, a table a person reviews.
  `make_kernel_table.py` drafts it from the metadata; `make_worklist.py --survey 5` plus `kernel_survey.py` check
  it against the images (which reconstruction of each pair really is the sharper one).
- Every manifest has `kernel`, `kernel_class` (`sharp` / `soft` / `other`) and `manufacturer`, so training and
  evaluation just filter the table:

```python
m = pd.read_csv("data/runs/ctrate/train-sharp/manifest.csv")
train = m[(m.split == "train") & (m.kernel_class == "sharp") & m.qc_passed]
test_sharp = m[(m.split == "test") & (m.kernel_class == "sharp")]
test_soft = m[(m.split == "test") & (m.kernel_class == "soft")]
```

`notebooks/kernel_label_distribution.ipynb` (run it right after `make_worklist.py`, no images needed) shows the
abnormality distribution by kernel and scanner, and how many scans a run skips.

## CT-RATE, start to end

CT-RATE has two official pools: `train_fixed/` (feeds train + val) and
`valid_fixed/` (the **test** set). Volumes are named like `train_1_a_1` (patient 1,
scan `a`, reconstruction 1; the two reconstructions of a scan are near-duplicate
rebuilds of the same raw data, usually a sharp and a soft kernel).

**0. (Once) check the kernel table.** `configs/kernel_classes.csv` says which kernels are sharp and which
are soft. Have it reviewed; to check it against the images, build and ingest a small survey run (about 150
volumes), then measure:

```bash
python scripts/preprocessing/make_worklist.py --survey 5
```
```bash
python scripts/preprocessing/ingest.py --source ctrate --run kernel-survey
```
```bash
python scripts/preprocessing/kernel_survey.py --source ctrate --run kernel-survey
```

**1. Build the worklist** (minutes; downloads only the metadata and label CSVs):

```bash
python scripts/preprocessing/make_worklist.py
```

This creates the run `train-sharp`: the whole valid pool (every reconstruction) and, from the train pool, the
sharp reconstruction of each scan in a seeded random order of patients. It prints the totals, how many scans
are skipped for lack of a sharp reconstruction, how many volumes are already cached and how many must be
downloaded, and the projected cache size. It refuses to overwrite a run that exists.

**2. Calibration run: one chunk.** This measures the real cache size per volume and
the download speed, and proves the server can reach Hugging Face:

```bash
python scripts/preprocessing/ingest.py --source ctrate --run train-sharp --max-chunks 1
```

Then `du -sh data/cache` and compare with the projection. If the full cache would
not fit (see [Disk budget](#disk-budget)), make a smaller run, with a cap on the train pool. The cache
is shared, so nothing already ingested is lost or downloaded again:

```bash
python scripts/preprocessing/make_worklist.py --name train-sharp-12k --max-train-patients 12000
```

**3. Ingest everything**, inside tmux:

```bash
python scripts/preprocessing/ingest.py --source ctrate --run train-sharp
```

Check on it any time (from any shell) with:

```bash
python scripts/preprocessing/ingest.py --source ctrate --run train-sharp --status
```

**4. Merge, QC, split:**

```bash
python scripts/preprocessing/merge_manifests.py --source ctrate --run train-sharp
```
```bash
python scripts/preprocessing/qc_report.py --source ctrate --run train-sharp
```
```bash
python scripts/preprocessing/assign_splits.py --source ctrate --run train-sharp
```

Look at the run's `qc_report.csv` and `qc_montages/` before splitting: volumes that
failed QC are excluded from the split.

**5. Later, a soft run** (for example to adapt a model to soft-kernel scans), from patients the first run
already split, so nothing leaks:

```bash
python scripts/preprocessing/make_worklist.py --train-kernel soft --from-run train-sharp --train-patients 800 --val-patients 200
```
```bash
python scripts/preprocessing/ingest.py --source ctrate --run train-soft
```

and then merge, QC and assign_splits again with `--run train-soft`. Only the missing soft volumes are downloaded.

## NHRD, start to end

NHRD has no official split and arrives as DICOM folders on a portable disk. The
server has no physical access and limited disk, so the data goes **disk → Google
Drive → server, one archive at a time**.

### For whoever uploads (no scripts needed)

1. Make a Drive folder for the archives and share it with the people uploading.
2. Put the patient folders of the disk into **`.zip` archives of about 10 GB** with
   7-Zip (zip format, compression level "Store"). Name each one uniquely, including
   the uploader, e.g. `nhrd_A_001.zip`. The archive must contain the patient folders
   **directly** (`4203-26/P00001/S0001/...`), not the full path of the disk.
3. Rules that keep the data correct:
   - **Whole patient folders only.** Never split a patient across two archives: a
     half-series fails QC.
   - **No patient in two archives.** The merge drops duplicates, but do not rely on it.
   - Wait for an upload to finish before relying on it. A cut-off upload is rejected
     by the server and retried on the next run, never used.
4. Upload each archive to the Drive folder (browser or the Drive desktop app). Share
   out the work by giving each person a contiguous range of the sorted patient list.
5. Once, make a checklist of every patient folder on the disk (`dir /b /ad` in cmd)
   and keep it, e.g. as `patients_on_disk.txt`. It lets the server report any patient
   that never arrived.

About 10 GB per archive means ~90 archives for ~940 GB. `.tar` and `.tar.gz` work too.

### On the server

Point the config at the Drive folder, once:
`sources.nhrd_local.ingest.drive_remote: "gdrive:nhrd_raw"` (or pass `--drive-remote`).

**1. Ingest.** Every archive found in the folder is one chunk. Run it inside tmux,
and re-run it whenever more archives have been uploaded:

```bash
python scripts/preprocessing/ingest.py --source nhrd_local
```

Each archive is downloaded, unpacked, preprocessed and deleted before the next one is
touched, so the server never holds more than about two archives of raw data.

**2. Merge, with the checklist:**

```bash
python scripts/preprocessing/merge_manifests.py --source nhrd_local --patients-file patients_on_disk.txt
```

It lists any patient folder that is on the disk but missing from the manifest (exit
code 1 if there are any), so you know exactly what to upload next.

**3. QC, then split:**

```bash
python scripts/preprocessing/qc_report.py --source nhrd_local
```
```bash
python scripts/preprocessing/assign_splits.py --source nhrd_local
```

By default 150 val and 150 test patients are drawn and the rest are train. The script
prints the patient counts first; adjust with `--n-val-patients` / `--n-test-patients`
the first time if your cohort is smaller or larger. NHRD has one run, `main`, created by the first
`ingest.py` call, so `--run` is not needed.

## Labels

Labels are joined into a run's manifest as `label_<name>` columns by `merge_manifests.py`. They are **only**
joined there: preprocessing, worklists and splits never read them, so which scans are kept or which patient goes
where cannot depend on a label.

- **CT-RATE:** `make_worklist.py` downloads the two label CSVs (about 3 MB) next to the metadata. They are
  *predicted* labels (parsed from the radiology reports by a model), not human-checked ground truth. If the
  download fails the worklist still works (a warning says so); copy `train_predicted_labels.csv` and
  `valid_predicted_labels.csv` into `data/metadata/` by hand to get them in the manifest.
- **NHRD:** put a labels CSV in the **same Drive folder as the archives** (it is not mistaken for an archive).
  The first column identifies the scan -- by default its path, e.g. `4203-26/P00001/S0001`; the other columns are
  the labels:

  ```
  scan_path,Lung nodule,Pleural effusion
  4203-26/P00001/S0001,1,0
  ```

  Name it in the config and say what the first column holds (`scan_path`, `patient_id` or `volume_id`):

  ```yaml
  sources:
    nhrd_local:
      labels: {file: labels.csv, key: scan_path}
  ```

  `merge_manifests.py` reads it from Drive (or a local copy, `--labels-file`) and reports how many scans
  were labelled and how many label rows match no scan. The labels file holds patient identifiers: it is covered by
  the same data-handling approval as the images.

## Running it unattended

```bash
tmux new -s ingest
```
```bash
python scripts/preprocessing/ingest.py --source ctrate --run train-sharp 2>&1 | tee -a ingest_ctrate.log
```

Detach with `Ctrl+b d`, log out, and come back later; `tmux a -t ingest` reattaches.
If the session or the server dies, run the same command again: finished chunks are
skipped, and a half-finished chunk resumes without re-fetching volumes that are
already cached.

What keeps an unattended run safe:

- **It never fills the disk.** Before every chunk it checks free space and stops
  cleanly below `min_free_gb` (default 100), with exit code 2.
- **It stops on a persistent problem.** After 3 chunks in a row fail (a network or
  login problem) it stops instead of grinding through every chunk.
- **One bad scan does not stop it.** A volume that fails to process is recorded in
  the chunk's marker and in the run's `preprocessing_manifest.json`; the chunk still
  finishes. Re-run just those with `--retry-failed`.
- **Raw data cannot be deleted by mistake.** Ingest empties its scratch folder
  (`sources.<name>.raw_dir`) after every chunk, so it refuses to start unless that
  folder is new, empty, or already its own.

On a cluster with SLURM, wrap the same command in a batch job instead of tmux
(`#SBATCH --time=...`, then `python scripts/preprocessing/ingest.py --source ctrate --run train-sharp`);
a job killed by the time limit simply resumes when resubmitted. Check first that
compute nodes can reach `huggingface.co` and Google Drive; if they cannot, the fetch
has to run on a node that can.

## Configuration and overrides

`configs/preprocessing.yaml` holds every default. A script's argument beats the
config for that run only. The settings you are most likely to touch:

| Setting (`sources.<name>.ingest`) | Default | Meaning |
|---|---|---|
| `chunk_size` | 40 (ctrate) | Volumes fetched, preprocessed and cleaned up together. |
| `train_kernel` | `sharp` | Which reconstruction of each train scan: `sharp` (lung) or `soft`. The test pool always takes every reconstruction. |
| `max_train_patients` | none | Cap on the train pool. |
| `max_test_patients` | none | Cap on the test pool; only for small pilot runs (the default keeps it whole). |
| `max_combined_gb` | none | Skip *train* volumes needing more resample memory than this (from the metadata, before downloading). |
| `min_free_gb` | 100 | Stop cleanly when free disk drops below this. |
| `workers` | 4 | Parallel preprocessing processes (use 1 on a GPU). |
| `fetch_workers` | 4 | Parallel downloads inside one chunk. |
| `drive_remote` | `gdrive:nhrd_raw` | Where NHRD's archives are (rclone remote or a folder path). |
| `est_mb_per_volume` | 22 | Only for the projected-size estimate; use the measured value. |
| `est_raw_gb_per_volume` | 0.43 | Only for the download-size estimate `make_worklist.py` prints. |
| `labels` (NHRD) | none | `{file: labels.csv, key: scan_path}`: the labels CSV in the Drive folder. |

| Setting (`sources.<name>.split`) | ctrate | nhrd_local |
|---|---|---|
| `n_val_patients` | 1000 | 150 |
| `n_test_patients` | (test = the valid pool) | 150 |

The `paths:` block names the shared `cache_dir`, the `runs_dir`, the per-source `splits_dir` and the
`kernel_table`. The `preprocess:` block (spacing, size, crop, HU floor) defines the cache. **Settle it with a
small pilot before the big ingest**: raw data is deleted as it goes, so changing it
afterwards means fetching everything again. A typo or an invalid value in the YAML
is an error at startup, never a silent fallback.

## Disk budget

Raw volumes are 90-450 MB; each becomes a ~20 MB int16 `.npy`. For an 850 GB server:

| Item | Volumes | Cache at ~22 MB |
|---|---|---|
| CT-RATE test (whole `valid_fixed`) | ~3,000 | ~67 GB |
| CT-RATE train + val (one reconstruction per scan) | ~24,000 | ~530 GB |
| NHRD | ~2,500 | ~55 GB |
| **Total** | | **~650 GB** |

A second run adds only the volumes the cache does not hold yet: a soft run of 1,000 patients adds about
1,000 volumes (~22 GB), and `make_worklist.py` prints the numbers before you start. Using both kernels for every scan
would roughly double the train cache (~1,060 GB), which does not fit; a subset does.

That leaves ~200 GB for scratch (one chunk of raw data at a time), checkpoints and the
OS. These are estimates: the **calibration run** gives the real MB per volume. If the
projected total exceeds ~700 GB, lower `max_train_patients`. Keep at least 100 GB free;
`min_free_gb` enforces it.

## Safety rules

- Split by **patient**, after QC, **once**, and freeze it before the first training
  run. Never re-split after seeing results. To start over, delete
  `data/splits/<source>.csv` by hand, on purpose. The split is **shared by every run** of a source: a patient
  has the same split in a sharp run and a soft run. Build a second run from the first one's patients
  (`--from-run`) so it never picks a validation patient for training.
- **Never train on a validation or test patient's other kernel.** The sharp and the soft volume of one scan are
  the same patient: keep them in the same split (the shared split does this).
- CT-RATE and NHRD have **separate** splits. Keep NHRD's test patients untouched
  during local adaptation.
- Preprocessing fits no dataset statistics, so processing before the split is
  leak-free.
- Use only `train` / `val` / `test` rows of the manifest (`excluded` failed QC,
  `unassigned` has no frozen split yet).

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `... not found -- run scripts/preprocessing/make_worklist.py first` / `no runs for source ...` | Run step 1 (it also downloads the metadata CSVs). |
| `run 'X' ... already exists -- runs are never overwritten` | A run of that name exists. Use it (`--run X`) or pick another name with `--name`. |
| `source ... has several runs (...) -- choose one with --run NAME` | More than one run on disk: say which. |
| `the cache ... was built with other preprocessing settings` | The `preprocess:` block changed since the cache was built. Restore it, use another `cache_dir`, or pass `--allow-new-settings` to replace the cache's settings (old volumes become stale and are fetched again). |
| `kernel table ... not found` | Draft one with `make_kernel_table.py` and have it reviewed. |
| `run ... has no frozen split yet` | `--from-run` needs the parent's split: run `merge_manifests.py`, `qc_report.py` and `assign_splits.py` with `--run <parent>`. |
| `asked for N train patients but only K ...` | The parent's split has fewer patients with that kernel than you asked for; lower `--train-patients` / `--val-patients`. |
| `warning: labels not read` / `could not download ..._predicted_labels.csv` | Labels are optional. Copy the CSVs into `data/metadata/` by hand, or fix the Drive file name (`labels.file`). |
| `every download in chunk ... failed (network or Hugging Face login?)` | Run `huggingface-cli login`, accept the CT-RATE terms, check the server can reach `huggingface.co`. |
| `rclone is not installed or not on PATH` / `rclone lsf failed` | Install rclone and complete `rclone config`; test with `rclone lsd gdrive:`. |
| `corrupt or incomplete archive` | A cut-off upload. Re-upload it; the next `ingest.py` run retries it. |
| `... is not empty and was not created by ingest` | `sources.<name>.raw_dir` points at a folder with other files. Point it at a new or empty folder; ingest deletes everything inside it. |
| `another ingest for source ... is already running` | Only one `ingest.py` per source at a time (two would delete each other's raw data). Attach to the running tmux session. A crashed run releases its lock automatically. |
| `stopped: free disk ... below min_free_gb` | Free space (or lower `--min-free-gb` knowingly) and re-run; it resumes. |
| `WARNING ... patient '...' has N scans, far more than the rest` | An archive was zipped one folder too high (the patient folders must sit directly inside it). Fix the archive and re-ingest it. |
| The process is `Killed` / out of memory | Lower `--workers` (each worker holds a whole volume in memory), or set `max_combined_gb` to skip the biggest CT-RATE volumes. |
| `no chunk manifests found for run ...` | Run `ingest.py --run <name>` before `merge_manifests.py`. |
| `qc_report.csv not found` | Run `qc_report.py --run <name>` before `assign_splits.py` (or `--skip-qc`). |
| `need N test + M val patients but only K ... are available` | Lower `--n-val-patients` / `--n-test-patients`, or ingest more patients. |
| Volumes listed as `raw file missing and no fresh cache entry` | The cache is stale (the `preprocess:` settings changed) and the raw file is gone: that chunk has to be re-ingested. |

## Development

```bash
python -m pytest
```

The tests use synthetic scans and fake Hugging Face / rclone backends, so they need no
download and no network. They cover the ingest loop (resume, disk guard, failures,
scratch safety), both sources, the frozen split, the merge, and both sources end to end
through the real scripts. Real Hugging Face downloads, real Drive transfers and tmux
can only be verified on the server, which is what the calibration run is for.

Code layout: `src/ct_preprocessing/` (the shared core: `pipeline.py`, loaders,
`spacing.py`, `crop.py`, `resize.py`, `quality.py`; `runs.py`, `kernels.py` and `cache_record.py`; the ingest
engine, worklist, labels and merge in `ingest/`), `scripts/preprocessing/` (thin wrappers),
`tests/preprocessing/`, `configs/` (the settings and the kernel table).
