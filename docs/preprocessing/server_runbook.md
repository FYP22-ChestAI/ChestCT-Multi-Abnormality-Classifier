# Server runbook: CT-RATE and NHRD, start to end

Copy-paste commands for running the preprocessing on the server, with what each one
does. Run everything from the repository root. Every script reads its defaults from
`configs/preprocessing.yaml` (chunk size 40, `min_free_gb` 100, ...); you only add
arguments to override them. The first line of every run prints the settings actually used.

The full explanation is in [README.md](README.md); this page is just the order of commands.

**Order:** 0 Setup → 1 CT-RATE → 2 NHRD → 3 Finish (merge, QC, splits).
The two sources are independent: do either one first.

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
```
Check free disk (the plan assumes about 850 GB) and CPU count.

Ask the server admin before starting:
- Do compute nodes have outbound internet (Hugging Face, Google)?
- Is there a disk quota or a walltime limit? Is SLURM used?

---

## 1. CT-RATE

**1.1 Build the worklist** (minutes, downloads only two small CSVs, about 16 MB):
```bash
python scripts/preprocessing/make_worklist.py
```
Writes `data/worklists/ctrate.csv`: exactly which volumes to fetch, grouped into chunks
of 40. Test pool (the official `valid_fixed`, 3,039 volumes) comes first, then the train
pool, one reconstruction per scan, in a seeded random order of patients. It prints the
totals and the projected cache size, so check that it fits the disk. It refuses to
overwrite an existing worklist.
If the projected size does not fit, add `--max-train-patients N` (for example 10000).

**1.2 Calibration run, one chunk only:**
```bash
python scripts/preprocessing/ingest.py --source ctrate --max-chunks 1
```
Fetches chunk 0 (40 test volumes), preprocesses them into `data/cache/*.npy`, writes the
chunk manifest, deletes the raw files, then stops. It shows the real MB per volume and the
seconds per chunk, and proves the server can reach Hugging Face.
Check the result:
```bash
python scripts/preprocessing/ingest.py --source ctrate --status
```
Prints chunks done, volumes ok/failed, cache size, free disk, and the measured size per volume.

**1.3 Full run, in tmux so it survives logging out:**
```bash
tmux new -s ctrate
python scripts/preprocessing/ingest.py --source ctrate
```
Loops over every pending chunk: fetch → preprocess → write manifest → delete raw → mark done.
It stops cleanly if free disk drops below 100 GB, or after 3 failed chunks in a row.
Detach with `Ctrl+b` then `d`. Come back with `tmux attach -t ctrate`.
If it stops for any reason, run the same command again: finished chunks and cached volumes
are skipped, nothing is downloaded twice.

Useful extras:
```bash
python scripts/preprocessing/ingest.py --source ctrate --dry-run        # list pending chunks, change nothing
python scripts/preprocessing/ingest.py --source ctrate --retry-failed   # re-run chunks that had failed volumes
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
The first command must list `nhrd_raw`; the second must list the zip files. If either fails,
stop and fix it. Revoke the token from the Google account's third-party access page when finished.

### 2.3 Ingest

Preview first:
```bash
python scripts/preprocessing/ingest.py --source nhrd_local --dry-run
```
Lists the pending archives (one chunk per archive). Nothing is downloaded.

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
Writes `data/manifest.csv`. With the checklist it also reports patient folders that never
arrived and folders that were not on the checklist.
**Expected:** `0 patient folder(s) missing`, `0 not on the checklist`, and no
"far more scans than the rest" warning.
If it says every patient is missing and shows archive names as "not on the checklist",
the archives were zipped one level too high (see 2.1). Fix the zips, re-upload, then delete
the NHRD results and re-ingest:
```bash
rm -rf data/ingest_state/nhrd_local data/manifests/nhrd_local data/raw/nhrd_local
rm -f data/manifest.csv data/preprocessing_manifest.json data/qc_report.csv
```
(and remove the NHRD `.npy` and `.meta.json` files from `data/cache/`).

---

## 3. Finish (both sources)

**3.1 Merge:**
```bash
python scripts/preprocessing/merge_manifests.py
```
Combines the per-chunk manifests of every source into `data/manifest.csv` (one row per
volume) and writes `data/preprocessing_manifest.json` (the settings and fingerprint that
produced the cache, plus what failed). Every `split` shows `unassigned` for now: that is
expected, splits are decided last.

**3.2 Quality check:**
```bash
python scripts/preprocessing/qc_report.py
```
Re-checks every cached volume (slice count, HU range, flat images) and writes
`data/qc_report.csv` plus montage pictures in `data/qc_montages/` (one picture per 25
volumes, plus every failure). Look at the failures and a few montages. It can be re-run any
time without touching raw data.

**3.3 Freeze the splits (once per source):**
```bash
python scripts/preprocessing/assign_splits.py --source ctrate --dry-run
python scripts/preprocessing/assign_splits.py --source ctrate
python scripts/preprocessing/assign_splits.py --source nhrd_local --dry-run
python scripts/preprocessing/assign_splits.py --source nhrd_local
```
Splits are assigned **by patient**, after QC, and then frozen in `data/splits/<source>.csv`.
- CT-RATE: test = the official valid pool; validation = 1,000 patients drawn from the train
  pool; the rest = train.
- NHRD: 150 validation and 150 test patients are drawn; the rest = train.
- Volumes that failed QC are marked `excluded`. `--dry-run` previews without freezing.

**3.4 Merge once more so the frozen splits appear in the manifest:**
```bash
python scripts/preprocessing/merge_manifests.py
```
Final output: `data/cache/*.npy` and `data/manifest.csv` with a `train`/`val`/`test`
value for every usable volume.

---

## Rules to remember

- Do not delete or edit anything under `data/ingest_state/`, `data/worklists/` or `data/splits/`
  between runs. They are what make runs resumable and the splits permanent.
- Freeze the preprocessing settings (`preprocess:` in the config) before the real run. The
  raw data is deleted after each chunk, so changing them later means re-downloading.
- One `ingest.py` per source at a time. A second one stops with a lock message.
- Re-running any command after an interruption is always safe.

## If something goes wrong

| Message | Fix |
|---|---|
| `stopped: free disk ... is below min_free_gb` | Free space, or lower it for a test: `--min-free-gb 10`. |
| `rclone is not installed or not on PATH` / `rclone lsf failed` | Install rclone and finish `rclone config`; test with `rclone lsd gdrive:`. |
| 401 / 403 from Hugging Face | Accept the CT-RATE terms on the dataset page, then `huggingface-cli login`. |
| Worklist already exists | It is protected. Use `--force` only if no finished chunk would change. |
| Chunk finished with failed volumes | `ingest.py --source <name> --retry-failed`. |
| Run died or SSH dropped | Run the same `ingest.py` command again. |
