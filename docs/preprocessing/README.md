# Preprocessing pipeline

Turns raw chest CT -- the public **CT-RATE** dataset and the local **NHRD** hospital
DICOM data -- into a compact, model-ready cache, on a server that is far too small
to hold the raw data. It runs unattended, in chunks, and can be stopped and resumed
at any point.

```
 raw chunk  ──►  preprocess  ──►  data/cache/*.npy   (permanent, ~20 MB per volume)
 (fetched)       (shared core)    data/manifests/    (one small manifest per chunk)
     │                                  │
     └── deleted after every chunk      └──► merge ► QC ► assign splits ► data/manifest.csv
```

What it hands to the model code: **`data/cache/`** (raw int16 HU, one `.npy` per
volume) and **`data/manifest.csv`** (one row per volume, with a frozen patient-level
`split`). See [data_contract.md](data_contract.md) for the exact format and the
reasoning behind it.

## Contents

1. [Setup](#setup)
2. [The scripts](#the-scripts)
3. [CT-RATE, start to end](#ct-rate-start-to-end)
4. [NHRD, start to end](#nhrd-start-to-end)
5. [Running it unattended](#running-it-unattended)
6. [Configuration and overrides](#configuration-and-overrides)
7. [Disk budget](#disk-budget)
8. [Safety rules](#safety-rules)
9. [Troubleshooting](#troubleshooting)
10. [Development](#development)

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
| `make_worklist.py` | CT-RATE | Downloads the two metadata CSVs (~16 MB, no images) and writes `data/worklists/ctrate.csv`: exactly which volumes to ingest, in chunks. A work list, **not a split**. | `--chunk-size`, `--train-pool`, `--test-pool`, `--max-train-patients`, `--max-test-patients`, `--max-combined-gb`, `--seed`, `--force` |
| `ingest.py` | both | The continuous script: for each pending chunk, fetch → preprocess into the cache → record manifest rows → delete the raw data. Resumable. | `--source` **(required)**, `--max-chunks`, `--min-free-gb`, `--workers`, `--device`, `--drive-remote`, `--retry-failed`, `--dry-run`, `--status` |
| `merge_manifests.py` | both | Merges the per-chunk manifests into `data/manifest.csv` and writes `data/preprocessing_manifest.json`. | `--source` (repeatable), `--patients-file` |
| `qc_report.py` | both | Quality-checks every cached volume: `data/qc_report.csv` and montage images to look at. | `--montage-every`, `--source` |
| `assign_splits.py` | both | Assigns train / val / test **by patient, once**, and freezes it. | `--source` (repeatable), `--n-val-patients`, `--n-test-patients`, `--seed`, `--skip-qc`, `--dry-run` |

Every script also takes `--config PATH` (default `configs/preprocessing.yaml`) and
`--help`.

`ingest.py` exit codes: `0` finished (or stopped by a limit you set), `1` some chunk
failed, `2` stopped early (low disk, or repeated failures).

## CT-RATE, start to end

CT-RATE has two official pools: `train_fixed/` (feeds train + val) and
`valid_fixed/` (the **test** set). Volumes are named like `train_1_a_1` (patient 1,
scan `a`, reconstruction 1; the two reconstructions of a scan are near-duplicate
rebuilds of the same raw data).

**1. Build the worklist** (minutes; downloads only the two metadata CSVs):

```bash
python scripts/preprocessing/make_worklist.py
```

By default it takes the whole valid pool and, from the train pool, one
reconstruction per scan in a seeded random order of patients. It prints the totals
and the projected cache size, and refuses to overwrite an existing worklist.

**2. Calibration run: one chunk.** This measures the real cache size per volume and
the download speed, and proves the server can reach Hugging Face:

```bash
python scripts/preprocessing/ingest.py --source ctrate --max-chunks 1
```

Then `du -sh data/cache` and compare with the projection. If the full cache would
not fit (see [Disk budget](#disk-budget)), cap the train pool and rebuild; already
ingested chunks stay valid, because the worklist is prefix-stable:

```bash
python scripts/preprocessing/make_worklist.py --force --max-train-patients 12000
```

**3. Ingest everything**, inside tmux:

```bash
python scripts/preprocessing/ingest.py --source ctrate
```

Check on it any time (from any shell) with:

```bash
python scripts/preprocessing/ingest.py --source ctrate --status
```

**4. Merge, QC, split:**

```bash
python scripts/preprocessing/merge_manifests.py
```
```bash
python scripts/preprocessing/qc_report.py
```
```bash
python scripts/preprocessing/assign_splits.py --source ctrate
```

Look at `data/qc_report.csv` and `data/qc_montages/` before splitting: volumes that
failed QC are excluded from the split.

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
python scripts/preprocessing/merge_manifests.py --patients-file patients_on_disk.txt
```

It lists any patient folder that is on the disk but missing from the manifest (exit
code 1 if there are any), so you know exactly what to upload next.

**3. QC, then split:**

```bash
python scripts/preprocessing/qc_report.py
```
```bash
python scripts/preprocessing/assign_splits.py --source nhrd_local
```

By default 150 val and 150 test patients are drawn and the rest are train. The script
prints the patient counts first; adjust with `--n-val-patients` / `--n-test-patients`
the first time if your cohort is smaller or larger.

## Running it unattended

```bash
tmux new -s ingest
```
```bash
python scripts/preprocessing/ingest.py --source ctrate 2>&1 | tee -a ingest_ctrate.log
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
  the chunk's marker and in `data/preprocessing_manifest.json`; the chunk still
  finishes. Re-run just those with `--retry-failed`.
- **Raw data cannot be deleted by mistake.** Ingest empties its scratch folder
  (`sources.<name>.raw_dir`) after every chunk, so it refuses to start unless that
  folder is new, empty, or already its own.

On a cluster with SLURM, wrap the same command in a batch job instead of tmux
(`#SBATCH --time=...`, then `python scripts/preprocessing/ingest.py --source ctrate`);
a job killed by the time limit simply resumes when resubmitted. Check first that
compute nodes can reach `huggingface.co` and Google Drive; if they cannot, the fetch
has to run on a node that can.

## Configuration and overrides

`configs/preprocessing.yaml` holds every default. A script's argument beats the
config for that run only. The settings you are most likely to touch:

| Setting (`sources.<name>.ingest`) | Default | Meaning |
|---|---|---|
| `chunk_size` | 40 (ctrate) | Volumes fetched, preprocessed and cleaned up together. |
| `train_pool` / `test_pool` | `one_per_scan` / `all` | Which reconstructions to keep. |
| `max_train_patients` | none | Cap on the train pool. |
| `max_test_patients` | none | Cap on the test pool; only for small pilot runs (the default keeps it whole). |
| `max_combined_gb` | none | Skip *train* volumes needing more resample memory than this (from the metadata, before downloading). |
| `min_free_gb` | 100 | Stop cleanly when free disk drops below this. |
| `workers` | 4 | Parallel preprocessing processes (use 1 on a GPU). |
| `fetch_workers` | 4 | Parallel downloads inside one chunk. |
| `drive_remote` | `gdrive:nhrd_raw` | Where NHRD's archives are (rclone remote or a folder path). |
| `est_mb_per_volume` | 22 | Only for the projected-size estimate; use the measured value. |

| Setting (`sources.<name>.split`) | ctrate | nhrd_local |
|---|---|---|
| `n_val_patients` | 1000 | 150 |
| `n_test_patients` | (test = the valid pool) | 150 |

The `preprocess:` block (spacing, size, crop) defines the cache. **Settle it with a
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

That leaves ~200 GB for scratch (one chunk of raw data at a time), checkpoints and the
OS. These are estimates: the **calibration run** gives the real MB per volume. If the
projected total exceeds ~700 GB, lower `max_train_patients`. Keep at least 100 GB free;
`min_free_gb` enforces it.

## Safety rules

- Split by **patient**, after QC, **once**, and freeze it before the first training
  run. Never re-split after seeing results. To start over, delete
  `data/splits/<source>.csv` by hand, on purpose.
- CT-RATE and NHRD have **separate** splits. Keep NHRD's test patients untouched
  during local adaptation.
- Preprocessing fits no dataset statistics, so processing before the split is
  leak-free.
- Use only `train` / `val` / `test` rows of the manifest (`excluded` failed QC,
  `unassigned` has no frozen split yet).

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `... not found -- run scripts/preprocessing/make_worklist.py first` | Run step 1 (it also downloads the metadata CSVs). |
| `every download in chunk ... failed (network or Hugging Face login?)` | Run `huggingface-cli login`, accept the CT-RATE terms, check the server can reach `huggingface.co`. |
| `rclone is not installed or not on PATH` / `rclone lsf failed` | Install rclone and complete `rclone config`; test with `rclone lsd gdrive:`. |
| `corrupt or incomplete archive` | A cut-off upload. Re-upload it; the next `ingest.py` run retries it. |
| `... is not empty and was not created by ingest` | `sources.<name>.raw_dir` points at a folder with other files. Point it at a new or empty folder; ingest deletes everything inside it. |
| `another ingest for source ... is already running` | Only one `ingest.py` per source at a time (two would delete each other's raw data). Attach to the running tmux session. A crashed run releases its lock automatically. |
| `stopped: free disk ... below min_free_gb` | Free space (or lower `--min-free-gb` knowingly) and re-run; it resumes. |
| `WARNING ... patient '...' has N scans, far more than the rest` | An archive was zipped one folder too high (the patient folders must sit directly inside it). Fix the archive and re-ingest it. |
| The process is `Killed` / out of memory | Lower `--workers` (each worker holds a whole volume in memory), or set `max_combined_gb` to skip the biggest CT-RATE volumes. |
| `no manifest rows for source ...` | Run `merge_manifests.py` after ingest. |
| `qc_report.csv not found` | Run `qc_report.py` before `assign_splits.py` (or `--skip-qc`). |
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
`spacing.py`, `crop.py`, `resize.py`, `quality.py`; the ingest engine in `ingest/`),
`scripts/preprocessing/` (the five scripts, thin wrappers), `tests/preprocessing/`.
