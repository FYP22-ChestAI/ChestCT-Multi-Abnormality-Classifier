# Server runbook: CT-RATE and NHRD, start to end

Copy-paste commands for running the preprocessing on the server, with what each one
does. Run everything from the repository root. Every script reads its defaults from
`configs/preprocessing.yaml` (chunk size 40, `min_free_gb` 100, sharp kernel, ...); you only
add arguments to override them. The first line of every run prints the settings actually used.

The full explanation is in [README.md](README.md); this page is just the order of commands.

**Order:** 0 Setup → 1 CT-RATE → 2 NHRD → 3 Later: a soft run.
The two sources are independent: do either one first.

**Two ideas to know first**
- A **run** is one named plan with its own folder (`data/runs/<source>/<run>/`: worklist, manifest,
  QC report). Runs coexist, nothing is overwritten, and they all share one cache (`data/cache/`) and one
  split per source. Scripts take `--run NAME`; with a single run it is chosen for you.
- The kernel setting `train_kernel: sharp` (default) takes the sharp (lung) reconstruction of each
  train scan. The test pool (`valid_fixed`) always takes every reconstruction, both kernels.

---

## 0. Setup (once)

```bash
python -m pip install -e ".[dev]"
```
Installs the `ct_preprocessing` package and its dependencies.

```bash
huggingface-cli login
```
Logs in to Hugging Face with your own token (needed for CT-RATE). First accept the terms
on the dataset page: https://huggingface.co/datasets/ibrahimhamamci/CT-RATE.
Never share or commit the token.

```bash
df -h .
nproc
free -g
```
Check free disk (the plan assumes about 850 GB), CPU count and RAM.

Ask the server admin before starting:
- Do compute nodes have outbound internet (Hugging Face, Google)?
- Is there a disk quota or a walltime limit? Is SLURM used?
- How much RAM does a job get? (Use `--workers 1` for the first calibration run and watch peak memory.)

---

## 1. CT-RATE

**1.0 Check the sharp / soft kernel table (once).** `configs/kernel_classes.csv` says which kernels are
sharp and which are soft. Have it reviewed by a person; to check it against the images, run the small survey:
```bash
python scripts/preprocessing/make_worklist.py --survey 5
python scripts/preprocessing/ingest.py --source ctrate --run kernel-survey
python scripts/preprocessing/kernel_survey.py --source ctrate --run kernel-survey
```
`--survey 5` makes a run of about 150 volumes: 5 scans per kernel pair, both reconstructions of each. After
ingest, `kernel_survey.py` measures which reconstruction of each pair is the sharper one and reports, per kernel,
whether it agrees with the table (`ok` / `DISAGREES` / `check`). The table is never changed for you: edit the CSV.

**1.1 Plan the run** (minutes, downloads only the metadata and label CSVs, about 19 MB):
```bash
python scripts/preprocessing/make_worklist.py
```
Creates the run `train-sharp` in `data/runs/ctrate/train-sharp/`: the whole test pool (3,039 volumes, both kernels)
and the sharp reconstruction of each train scan, in a seeded random order of patients, in chunks of 40. It prints
how many scans are skipped for lack of a sharp reconstruction, how many volumes are already in the cache, how many
must be downloaded, and the projected cache size. Check that it fits the disk; if not, add
`--max-train-patients N` (for example 10000). It refuses to overwrite a run that exists.
Open `notebooks/kernel_label_distribution.ipynb` now (no images needed) to see how the abnormalities
are distributed by kernel and scanner.

**1.2 Calibration run, one chunk only:**
```bash
python scripts/preprocessing/ingest.py --source ctrate --run train-sharp --max-chunks 1
```
Fetches chunk 0 (40 test volumes), preprocesses them into `data/cache/*.npy`, writes the chunk manifest, deletes
the raw files, then stops. It shows the real MB per volume and the seconds per chunk, and proves the server can
reach Hugging Face.
```bash
python scripts/preprocessing/ingest.py --source ctrate --run train-sharp --status
```
Prints chunks done, volumes ok/failed, cache size, free disk, and the measured size per volume.

**1.3 Full run, in tmux so it survives logging out:**
```bash
tmux new -s ctrate
python scripts/preprocessing/ingest.py --source ctrate --run train-sharp
```
Loops over every pending chunk: fetch → preprocess → write manifest → delete raw → mark done. It stops cleanly if
free disk drops below 100 GB, or after 3 failed chunks in a row. Detach with `Ctrl+b` then `d`; come back with
`tmux attach -t ctrate`. If it stops for any reason, run the same command again: finished chunks and cached volumes
are skipped, nothing is downloaded twice.
```bash
python scripts/preprocessing/ingest.py --source ctrate --run train-sharp --dry-run        # list pending chunks, change nothing
python scripts/preprocessing/ingest.py --source ctrate --run train-sharp --retry-failed   # re-run chunks that had failed volumes
```
The cache remembers the preprocessing settings that built it. If the `preprocess:` block of the config changed,
ingest refuses to write into it; fix the config, or pass `--allow-new-settings` on purpose.

**1.4 Merge, QC, split:**
```bash
python scripts/preprocessing/merge_manifests.py --source ctrate --run train-sharp
python scripts/preprocessing/qc_report.py --source ctrate --run train-sharp
python scripts/preprocessing/assign_splits.py --source ctrate --run train-sharp --dry-run
python scripts/preprocessing/assign_splits.py --source ctrate --run train-sharp
```
- `merge_manifests.py` writes the run's `manifest.csv`: one row per volume with `kernel`, `kernel_class`,
  `manufacturer`, the `label_*` columns (CT-RATE's predicted labels) and the frozen split (`unassigned` until the
  split is made). It also prints the kernel class counts and how many rows have labels.
- `qc_report.py` re-checks every cached volume and writes `qc_report.csv` plus montage pictures
  (one per 25 volumes, plus every failure). Look at the failures and a few montages.
- `assign_splits.py` splits **by patient**, after QC, and freezes it in `data/splits/ctrate.csv`: test = the
  official valid pool, validation = 1,000 patients drawn from the train pool, the rest = train. The `--dry-run`
  line previews it. The split is shared by every run, and a copy for this run is written into its folder.

Final output for the run: `data/cache/*.npy` and `data/runs/ctrate/train-sharp/manifest.csv`.
Training and evaluation filter that table:
```python
m = pd.read_csv("data/runs/ctrate/train-sharp/manifest.csv")
train = m[(m.split == "train") & (m.kernel_class == "sharp") & m.qc_passed]
test_sharp = m[(m.split == "test") & (m.kernel_class == "sharp")]
test_soft = m[(m.split == "test") & (m.kernel_class == "soft")]
```

---

## 2. NHRD (local hospital DICOM data, via Google Drive)

### 2.1 Prepare the archives (done by the data owner, on the Windows laptop)

One archive holds whole patient folders, about 10 GB per archive.

**The rule that matters:** open the archive and the first thing you see must be the
**patient folders** (for example `4203-26/`). Not a folder that contains them.
If a zip starts with a folder like `nhrd_A_001/` or `NHRD - CT Scan/`, it was zipped one
level too high, and the patient ids will be wrong.

PowerShell, with the patient folders inside a staging folder `<dir>`:
```powershell
$dir = "C:\path\to\folder_with_patient_folders"
$names = Get-ChildItem $dir -Directory | ForEach-Object Name
tar -a -c -f "C:\path\to\nhrd_A_001.zip" -C $dir @names
```
Uses Windows' own `tar`, which writes proper forward-slash paths. (Do not use
`Compress-Archive` in PowerShell 5.1: it writes backslashes and breaks on Linux.)

Check the layout before uploading; it must print patient names:
```powershell
python -c "import zipfile; print(sorted({n.split('/')[0] for n in zipfile.ZipFile(r'C:\path\to\nhrd_A_001.zip').namelist()}))"
```

Make the checklist of every patient folder on the original disk (one name per line):
```powershell
Get-ChildItem "D:\NHRD" -Directory | ForEach-Object Name | Set-Content patients_on_disk.txt
```
This file is **not** uploaded to Drive. It only goes to the server for step 2.4.

Upload each zip by hand in the browser to a folder named `nhrd_raw` in **My Drive**
(not "Shared with me"). Name the archives anything ending in `.zip` or `.tar`
(for example `nhrd_A_001.zip`, `nhrd_A_002.zip`). Make sure the hospital-data approval
allows storing it on Google Drive.

**Labels (optional).** Put a CSV in the **same Drive folder** (for example `labels.csv`): the first column
identifies the scan (by default its path, `4203-26/P00001/S0001`), the other columns are the abnormality labels:
```
scan_path,Lung nodule,Pleural effusion
4203-26/P00001/S0001,1,0
```
Then set `sources.nhrd_local.labels: {file: labels.csv, key: scan_path}` in `configs/preprocessing.yaml`
(`key` can also be `patient_id` or `volume_id`). It holds patient identifiers: the same approval applies.

### 2.2 Connect the server to Drive (once)

```bash
rclone config
```
Answers: `n` (new remote), name `gdrive`, storage `drive`, client_id and client_secret blank,
scope `2` (`drive.readonly`: the pipeline only lists and downloads, it can never change
the Drive), service account blank, advanced `n`, **"Use auto config?" `n`**.
rclone then prints a command like `rclone authorize "drive" "eyJ..."`. Run exactly that
command on a computer **with a browser** (rclone installed there too), log in, and paste
the JSON token it prints back into the server prompt. Then:
```bash
chmod 600 ~/.config/rclone/rclone.conf
rclone lsd gdrive:
rclone lsf gdrive:nhrd_raw
```
The first command must list `nhrd_raw`; the second must list the zip files (and `labels.csv`). If either
fails, stop and fix it. Revoke the token from the Google account's third-party access page when finished.

### 2.3 Ingest

Preview first:
```bash
python scripts/preprocessing/ingest.py --source nhrd_local --dry-run
```
Lists the pending archives (one chunk per archive). Nothing is downloaded. NHRD has a single run, `main`, which is
created the first time you ingest, so `--run` is not needed.

Calibration, one archive:
```bash
python scripts/preprocessing/ingest.py --source nhrd_local --max-chunks 1
```
Per archive: rclone copies it from Drive → it is extracted safely (a corrupt or unsafe
archive is rejected and retried next run) → the archive file is deleted → scans are
preprocessed into `data/cache/` → the extracted files are deleted → the chunk is marked done.
The first archive also writes `data/ingest_state/nhrd_local/patient_id_source.txt`
(contains `path` or `dicom_tag`): how patients are identified. Leave it alone, it keeps
every later archive grouped the same way.

**Right after the first archive, run the checks in 2.4 before continuing.**

Full run, in tmux:
```bash
tmux new -s nhrd
python scripts/preprocessing/ingest.py --source nhrd_local
```
Processes every archive on Drive. To add more later, upload more archives and run the same
command again: only the new ones are processed.

### 2.4 Check that patients were identified correctly

```bash
python scripts/preprocessing/merge_manifests.py --source nhrd_local --patients-file patients_on_disk.txt
```
Writes `data/runs/nhrd_local/main/manifest.csv`. With the checklist it also reports patient folders that never
arrived and folders that were not on the checklist. If labels are configured it reads them from Drive and says
how many scans were labelled.
**Expected:** `0 patient folder(s) missing`, `0 not on the checklist`, and no
"far more scans than the rest" warning.
If it says every patient is missing and shows archive names as "not on the checklist",
the archives were zipped one level too high (see 2.1). Fix the zips, re-upload, then delete
the NHRD run's state and its results and re-ingest:
```bash
rm -rf data/ingest_state/nhrd_local data/runs/nhrd_local
```
(and remove the NHRD `.npy` and `.meta.json` files from `data/cache/`).

### 2.5 QC and split

```bash
python scripts/preprocessing/qc_report.py --source nhrd_local
python scripts/preprocessing/assign_splits.py --source nhrd_local --dry-run
python scripts/preprocessing/assign_splits.py --source nhrd_local
```
150 validation and 150 test patients are drawn (the NHRD defaults) and the rest are train; the split is frozen in
`data/splits/nhrd_local.csv`. Then merge once more so the manifest carries the frozen split:
```bash
python scripts/preprocessing/merge_manifests.py --source nhrd_local
```

---

## 3. Later: a soft run (to adapt to soft-kernel scans)

Build it from patients the sharp run has **already split**, so nothing leaks: here 800 patients from its train
split and 200 from its validation split, each keeping the split it has.
```bash
python scripts/preprocessing/make_worklist.py --train-kernel soft --from-run train-sharp --train-patients 800 --val-patients 200
python scripts/preprocessing/ingest.py --source ctrate --run train-soft
python scripts/preprocessing/merge_manifests.py --source ctrate --run train-soft
python scripts/preprocessing/qc_report.py --source ctrate --run train-soft
python scripts/preprocessing/assign_splits.py --source ctrate --run train-soft
```
Only the soft volumes missing from the cache are downloaded (the test pool is already there). The first run's
files are not touched. With two runs on disk, every script needs `--run`.

---

## Rules to remember

- Do not delete or edit anything under `data/ingest_state/`, `data/runs/` or `data/splits/`
  between runs. They are what make runs resumable and the splits permanent. A run is never overwritten:
  a new plan is a new run.
- Freeze the preprocessing settings (`preprocess:` in the config) before the real run. The
  raw data is deleted after each chunk, so changing them later means re-downloading everything.
  The cache refuses new settings unless you pass `--allow-new-settings`.
- One `ingest.py` per source at a time (all of a source's runs share one scratch folder). A second one
  stops with a lock message.
- Re-running any command after an interruption is always safe.
- Labels are only joined into the manifest. They never decide which scans are kept or which patient goes where.

## If something goes wrong

| Message | Fix |
|---|---|
| `stopped: free disk ... is below min_free_gb` | Free space, or lower it for a test: `--min-free-gb 10`. |
| `rclone is not installed or not on PATH` / `rclone lsf failed` | Install rclone and finish `rclone config`; test with `rclone lsd gdrive:`. |
| 401 / 403 from Hugging Face | Accept the CT-RATE terms on the dataset page, then `huggingface-cli login`. |
| `run ... already exists -- runs are never overwritten` | Use it (`--run NAME`) or choose another name with `--name`. |
| `source ... has several runs (...)` | Add `--run NAME`. |
| `the cache ... was built with other preprocessing settings` | Restore the `preprocess:` block, or pass `--allow-new-settings` knowingly. |
| `kernel table ... not found` | `python scripts/preprocessing/make_kernel_table.py` drafts one; have it reviewed. |
| `run ... has no frozen split yet` | Run merge, QC and `assign_splits.py` on the parent run first. |
| `warning: labels not read` | Labels are optional; check the Drive file name, or use `--labels-file`. |
| Chunk finished with failed volumes | `ingest.py --source <name> --run <run> --retry-failed`. |
| Run died or SSH dropped | Run the same `ingest.py` command again. |
