"""Both sources, start to end, through the real scripts:
(worklist ->) ingest -> merge -> qc -> assign_splits, with runs that coexist and share one cache."""
import hashlib
import importlib.util
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts" / "preprocessing"


def run_script(name: str, *args):
    """Run a script as a user would, from the repo root."""
    proc = subprocess.run([sys.executable, str(SCRIPTS / name), *map(str, args)], cwd=REPO, capture_output=True, text=True)
    return proc


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def call(monkeypatch, name: str, *args) -> int:
    """Run a script's main() in-process (so fakes can be injected)."""
    monkeypatch.setattr(sys, "argv", [f"{name}.py", *map(str, args)])
    return load_script(name).main()


def digest(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


# ================================================================== NHRD (archives)
def nhrd_config(tmp_path, drive: Path, labels: bool = False) -> Path:
    w = tmp_path / "work"
    cfg = w / "preprocessing.yaml"
    w.mkdir(exist_ok=True)
    cfg.write_text(f"""
paths:
  raw_dir: {w / 'data/raw'}
  cache_dir: {w / 'data/cache'}
  runs_dir: {w / 'data/runs'}
  metadata_dir: {w / 'data/metadata'}
  state_dir: {w / 'data/ingest_state'}
  splits_dir: {w / 'data/splits'}
  kernel_table: {REPO / 'configs/kernel_classes.csv'}
sources:
  nhrd_local:
    format: dicom
    raw_dir: {w / 'data/raw/nhrd_local'}
    manifest_builder: folder
    ingest: {{drive_remote: "{drive.as_posix()}", min_free_gb: 0, workers: 1}}
    split: {{n_val_patients: 1, n_test_patients: 1}}
    labels: {{file: {"labels.csv" if labels else "null"}, key: scan_path}}
preprocess:
  target_size_hw: [32, 32]
qc:
  min_slices: 3
""", encoding="utf-8")
    return cfg


def build_drive(tmp_path, dicom_writer, synthetic_hu, n_patients=6):
    """A portable disk with `n_patients` patient folders, uploaded as two zip archives."""
    disk, drive = tmp_path / "disk", tmp_path / "drive"
    drive.mkdir()
    patients = [f"42{i:02d}-26" for i in range(n_patients)]
    for p in patients:
        dicom_writer(disk / p / "P00001" / "S0001", synthetic_hu[:6])  # every PatientID is the same "ANON01"
    half = n_patients // 2
    for name, group in (("nhrd_A_001", patients[:half]), ("nhrd_B_001", patients[half:])):
        with zipfile.ZipFile(drive / f"{name}.zip", "w") as zf:
            for p in group:
                for f in sorted((disk / p).rglob("*")):
                    if f.is_file():
                        zf.write(f, f.relative_to(disk).as_posix())
    (tmp_path / "patients_on_disk.txt").write_text("\n".join(patients) + "\n")
    return drive, patients


def test_nhrd_from_uploaded_archives_to_frozen_splits(tmp_path, dicom_writer, synthetic_hu):
    drive, patients = build_drive(tmp_path, dicom_writer, synthetic_hu)
    cfg = nhrd_config(tmp_path, drive)
    data = tmp_path / "work" / "data"
    run_dir = data / "runs" / "nhrd_local" / "main"

    # 1. ingest: both archives, no arguments beyond the source (its run is created and called "main")
    proc = run_script("ingest.py", "--config", cfg, "--source", "nhrd_local")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.splitlines()[0].startswith("using: source=nhrd_local, run=main")  # settings are shown first
    assert "2 chunk(s) pending" in proc.stdout
    assert len(list((data / "cache").glob("*.npy"))) == 6
    assert [p.name for p in (data / "raw" / "nhrd_local").iterdir()] == [".ingest_scratch"]  # raw is gone
    assert (data / "cache" / ".cache_fingerprint.json").exists()  # the cache remembers its settings

    # re-running is a no-op
    again = run_script("ingest.py", "--config", cfg, "--source", "nhrd_local")
    assert again.returncode == 0 and "0 chunk(s) pending" in again.stdout
    status = run_script("ingest.py", "--config", cfg, "--source", "nhrd_local", "--status")
    assert "chunks done: 2 / 2" in status.stdout

    # 2. merge, with the checklist of patients on the disk
    merged = run_script("merge_manifests.py", "--config", cfg, "--source", "nhrd_local", "--patients-file", tmp_path / "patients_on_disk.txt")
    assert merged.returncode == 0, merged.stdout + merged.stderr
    assert "0 patient folder(s) missing" in merged.stdout and "kernel classes:" in merged.stdout
    manifest = pd.read_csv(run_dir / "manifest.csv")
    assert len(manifest) == 6 and set(manifest.split) == {"unassigned"}
    assert sorted(manifest.patient_id) == patients  # no PatientID tag survives -> grouped by top folder
    assert {"kernel", "kernel_class", "manufacturer"} <= set(manifest.columns)
    assert (run_dir / "preprocessing_manifest.json").exists()

    # 3. QC
    qc = run_script("qc_report.py", "--config", cfg, "--source", "nhrd_local")
    assert qc.returncode == 0, qc.stdout + qc.stderr
    assert pd.read_csv(run_dir / "qc_report.csv").passed.all()

    # 4. split once: 1 val + 1 test patient, the rest train
    split = run_script("assign_splits.py", "--config", cfg, "--source", "nhrd_local")
    assert split.returncode == 0, split.stdout + split.stderr
    final = pd.read_csv(run_dir / "manifest.csv")
    assert final.split.value_counts().to_dict() == {"train": 4, "val": 1, "test": 1}
    assert final.groupby("patient_id").split.nunique().max() == 1  # a patient is in exactly one split
    frozen = (data / "splits" / "nhrd_local.csv").read_text()
    assert len(pd.read_csv(run_dir / "splits.csv")) == 6  # a copy of the split is kept in the run's folder

    # 5. frozen: running it again changes nothing, even with a different seed
    run_script("assign_splits.py", "--config", cfg, "--source", "nhrd_local", "--seed", 999)
    assert (data / "splits" / "nhrd_local.csv").read_text() == frozen
    assert pd.read_csv(run_dir / "manifest.csv").split.tolist() == final.split.tolist()


def test_nhrd_labels_are_read_from_the_drive_folder_and_joined_into_the_manifest(tmp_path, dicom_writer, synthetic_hu):
    drive, patients = build_drive(tmp_path, dicom_writer, synthetic_hu)
    (drive / "labels.csv").write_text(
        "scan_path,Lung nodule,Pleural effusion\n"
        + "".join(f"{p}/P00001/S0001,{i % 2},{(i + 1) % 2}\n" for i, p in enumerate(patients[:4]))  # only 4 of 6 scans labelled
    )
    cfg = nhrd_config(tmp_path, drive, labels=True)
    assert run_script("ingest.py", "--config", cfg, "--source", "nhrd_local").returncode == 0  # the csv is not an archive

    merged = run_script("merge_manifests.py", "--config", cfg, "--source", "nhrd_local")
    assert merged.returncode == 0, merged.stdout + merged.stderr
    assert "labels: 2 column(s), 4 of 6 manifest rows labelled" in merged.stdout
    manifest = pd.read_csv(tmp_path / "work" / "data" / "runs" / "nhrd_local" / "main" / "manifest.csv")
    assert {"label_lung_nodule", "label_pleural_effusion"} <= set(manifest.columns)
    assert manifest.label_lung_nodule.notna().sum() == 4
    assert (tmp_path / "work" / "data" / "metadata" / "nhrd_local_labels.csv").exists()


def test_labels_that_cannot_be_read_only_warn(tmp_path, dicom_writer, synthetic_hu):
    drive, _ = build_drive(tmp_path, dicom_writer, synthetic_hu)  # labels.csv is configured but not in the folder
    cfg = nhrd_config(tmp_path, drive, labels=True)
    run_script("ingest.py", "--config", cfg, "--source", "nhrd_local")
    merged = run_script("merge_manifests.py", "--config", cfg, "--source", "nhrd_local")
    assert merged.returncode == 0 and "warning: labels not read" in merged.stdout


def test_a_missing_patient_is_reported_by_the_checklist(tmp_path, dicom_writer, synthetic_hu):
    drive, patients = build_drive(tmp_path, dicom_writer, synthetic_hu)
    cfg = nhrd_config(tmp_path, drive)
    run_script("ingest.py", "--config", cfg, "--source", "nhrd_local")
    (tmp_path / "patients_on_disk.txt").write_text("\n".join([*patients, "9999-99"]) + "\n")
    merged = run_script("merge_manifests.py", "--config", cfg, "--source", "nhrd_local", "--patients-file", tmp_path / "patients_on_disk.txt")
    assert merged.returncode == 1 and "MISSING 9999-99" in merged.stdout


def test_a_half_uploaded_archive_fails_its_chunk_and_the_rest_still_ingest(tmp_path, dicom_writer, synthetic_hu):
    drive, _ = build_drive(tmp_path, dicom_writer, synthetic_hu)
    broken = drive / "nhrd_B_001.zip"
    broken.write_bytes(broken.read_bytes()[: broken.stat().st_size // 2])  # upload was cut off
    cfg = nhrd_config(tmp_path, drive)

    proc = run_script("ingest.py", "--config", cfg, "--source", "nhrd_local")

    assert proc.returncode == 1  # a chunk failed, and we are told
    assert "nhrd_B_001" in proc.stdout and "FAILED" in proc.stdout
    assert len(list((tmp_path / "work" / "data" / "cache").glob("*.npy"))) == 3  # archive A is fine


def test_low_disk_stops_the_script_with_a_distinct_exit_code(tmp_path, dicom_writer, synthetic_hu):
    drive, _ = build_drive(tmp_path, dicom_writer, synthetic_hu)
    cfg = nhrd_config(tmp_path, drive)
    proc = run_script("ingest.py", "--config", cfg, "--source", "nhrd_local", "--min-free-gb", 1000000000)
    assert proc.returncode == 2 and "min_free_gb" in proc.stdout


def test_the_script_arguments_override_the_config_and_are_echoed(tmp_path, dicom_writer, synthetic_hu):
    drive, _ = build_drive(tmp_path, dicom_writer, synthetic_hu)
    cfg = nhrd_config(tmp_path, drive)
    proc = run_script("ingest.py", "--config", cfg, "--source", "nhrd_local", "--max-chunks", 1, "--workers", 1, "--dry-run")
    first = proc.stdout.splitlines()[0]
    assert "workers=1" in first and "max_chunks=1" in first
    assert not (tmp_path / "work" / "data" / "cache").exists()  # --dry-run touched nothing


def test_assign_splits_needs_the_qc_report_unless_told_to_skip_it(tmp_path, dicom_writer, synthetic_hu):
    drive, _ = build_drive(tmp_path, dicom_writer, synthetic_hu)
    cfg = nhrd_config(tmp_path, drive)
    run_script("ingest.py", "--config", cfg, "--source", "nhrd_local")
    run_script("merge_manifests.py", "--config", cfg, "--source", "nhrd_local")
    proc = run_script("assign_splits.py", "--config", cfg, "--source", "nhrd_local")
    assert proc.returncode != 0 and "qc_report.py" in (proc.stdout + proc.stderr)
    assert run_script("assign_splits.py", "--config", cfg, "--source", "nhrd_local", "--skip-qc", "--dry-run").returncode == 0


def test_changed_preprocessing_settings_cannot_overwrite_the_cache_unless_allowed(tmp_path, dicom_writer, synthetic_hu):
    drive, _ = build_drive(tmp_path, dicom_writer, synthetic_hu)
    cfg = nhrd_config(tmp_path, drive)
    assert run_script("ingest.py", "--config", cfg, "--source", "nhrd_local", "--max-chunks", 1).returncode == 0
    cfg.write_text(cfg.read_text().replace("target_size_hw: [32, 32]", "target_size_hw: [48, 48]"), encoding="utf-8")
    refused = run_script("ingest.py", "--config", cfg, "--source", "nhrd_local")
    assert refused.returncode == 1 and "other preprocessing settings" in refused.stderr and "--allow-new-settings" in refused.stderr
    allowed = run_script("ingest.py", "--config", cfg, "--source", "nhrd_local", "--allow-new-settings")
    assert allowed.returncode == 0


# ================================================================== CT-RATE (Hugging Face)
def make_fake_hf(nifti: Path, train_meta: str, valid_meta: str, downloads: list):
    def download(repo_id, filename, repo_type=None, cache_dir=None, **kw):
        base = Path(cache_dir) if cache_dir else Path(nifti).parent / "hf_default"
        target = base / "snapshots" / "x" / Path(filename).name
        target.parent.mkdir(parents=True, exist_ok=True)
        if filename.endswith("train_metadata.csv"):
            target.write_text(train_meta)
        elif filename.endswith("validation_metadata.csv"):
            target.write_text(valid_meta)
        elif filename.endswith("_predicted_labels.csv"):
            pool = "train" if "train" in filename else "valid"
            patients = range(1, 9) if pool == "train" else (1, 2)
            target.write_text("VolumeName,Lung nodule,Medical material\n" + "".join(
                f"{pool}_{p}_a_{r}.nii.gz,{p % 2},{(p + r) % 2}\n" for p in patients for r in (1, 2)))
        else:
            downloads.append(Path(filename).name)
            shutil.copyfile(nifti, target)
        return str(target)
    return download


KERNEL_TABLE = "manufacturer,kernel,class\nPhilips,YA,sharp\nPhilips,B,soft\n"


def ctrate_config(tmp_path) -> Path:
    w = tmp_path / "work"
    w.mkdir(exist_ok=True)
    (w / "kernels.csv").write_text(KERNEL_TABLE)
    cfg = w / "preprocessing.yaml"
    cfg.write_text(f"""
paths:
  raw_dir: {w / 'data/raw'}
  cache_dir: {w / 'data/cache'}
  runs_dir: {w / 'data/runs'}
  metadata_dir: {w / 'data/metadata'}
  state_dir: {w / 'data/ingest_state'}
  splits_dir: {w / 'data/splits'}
  kernel_table: {w / 'kernels.csv'}
sources:
  ctrate:
    format: nifti
    raw_dir: {w / 'data/raw/ctrate'}
    manifest_builder: ctrate
    ingest: {{chunk_size: 4, min_free_gb: 0, workers: 1, fetch_workers: 2}}
    split: {{n_val_patients: 2}}
preprocess:
  target_size_hw: [32, 32]
qc:
  min_slices: 10
""", encoding="utf-8")
    return cfg


def test_ctrate_two_runs_share_one_cache_and_one_split(tmp_path, monkeypatch, synthetic_nifti):
    from ct_preprocessing.ingest import sources

    nifti = synthetic_nifti[0]
    meta = lambda pool, patients: f"VolumeName,Rows,Manufacturer,ConvolutionKernel\n" + "".join(  # noqa: E731
        f"{pool}_{p}_a_{r}.nii.gz,512,Philips,{'YA' if r == 1 else 'B'}\n" for p in patients for r in (1, 2))
    downloads: list[str] = []
    fake = make_fake_hf(nifti, meta("train", range(1, 9)), meta("valid", (1, 2)), downloads)
    monkeypatch.setattr(sources, "get_hf_download", lambda: fake)
    cfg = ctrate_config(tmp_path)
    data = tmp_path / "work" / "data"
    sharp_dir, soft_dir = data / "runs" / "ctrate" / "train-sharp", data / "runs" / "ctrate" / "train-soft"

    # 1. the first run: the whole valid pool (4 volumes, both kernels) + the sharp reconstruction of 8 train scans
    mod = load_script("make_worklist")
    monkeypatch.setattr(mod, "get_hf_download", lambda: fake)
    monkeypatch.setattr(sys, "argv", ["make_worklist.py", "--config", str(cfg)])
    assert mod.main() == 0
    wl = pd.read_csv(sharp_dir / "worklist.csv")
    assert len(wl) == 12 and wl.chunk.nunique() == 3 and (wl[wl.chunk == 0].source_split == "valid").all()
    assert set(wl[wl.source_split == "train"].kernel_class) == {"sharp"} and set(wl[wl.source_split == "valid"].kernel_class) == {"sharp", "soft"}
    assert (data / "metadata" / "train_predicted_labels.csv").exists()  # the labels came with the metadata
    monkeypatch.setattr(sys, "argv", ["make_worklist.py", "--config", str(cfg)])
    with pytest.raises(SystemExit, match="already exists"):  # a run is never silently rebuilt or overwritten
        mod.main()

    # 2. ingest every chunk of that run
    assert call(monkeypatch, "ingest", "--config", cfg, "--source", "ctrate", "--run", "train-sharp") == 0
    assert len(list((data / "cache").glob("*.npy"))) == 12 and len(downloads) == 12
    assert [p.name for p in (data / "raw" / "ctrate").iterdir()] == [".ingest_scratch"]

    # 3-4. merge, QC
    assert call(monkeypatch, "merge_manifests", "--config", cfg, "--source", "ctrate", "--run", "train-sharp") == 0
    assert call(monkeypatch, "qc_report", "--config", cfg, "--source", "ctrate", "--run", "train-sharp") == 0

    # a child run cannot be built before the parent has a frozen split
    monkeypatch.setattr(sys, "argv", ["make_worklist.py", "--config", str(cfg), "--train-kernel", "soft", "--from-run", "train-sharp",
                                      "--train-patients", "3", "--val-patients", "1"])
    with pytest.raises(SystemExit, match="no frozen split yet"):
        mod.main()

    # 5. split: test = official valid pool; val = 2 train patients; the other 6 train patients = train
    assert call(monkeypatch, "assign_splits", "--config", cfg, "--source", "ctrate", "--run", "train-sharp") == 0
    m = pd.read_csv(sharp_dir / "manifest.csv")
    assert m[m.source_split == "valid"].split.eq("test").all() and m[m.source_split == "train"].split.isin(["train", "val"]).all()
    per_patient = m.groupby("patient_id").split.first()
    assert per_patient[per_patient.index.str.startswith("train_")].value_counts().to_dict() == {"train": 6, "val": 2}
    assert (m.split == "test").sum() == 4  # both reconstructions of both valid patients
    assert {"label_lung_nodule", "label_medical_material", "kernel", "kernel_class", "manufacturer"} <= set(m.columns)
    assert m.label_lung_nodule.notna().all()  # CT-RATE labels are in the manifest
    split_file = data / "splits" / "ctrate.csv"
    before = {f: digest(f) for f in (sharp_dir / "worklist.csv", sharp_dir / "manifest.csv", sharp_dir / "qc_report.csv", split_file)}

    # 6. a SOFT run built from the parent's already-split patients: 3 from its train split, 1 from its val split
    n_downloaded = len(downloads)
    monkeypatch.setattr(sys, "argv", ["make_worklist.py", "--config", str(cfg), "--train-kernel", "soft", "--from-run", "train-sharp",
                                      "--train-patients", "3", "--val-patients", "1"])
    assert mod.main() == 0
    soft = pd.read_csv(soft_dir / "worklist.csv")
    assert set(soft[soft.source_split == "train"].kernel_class) == {"soft"} and soft[soft.source_split == "train"].patient_id.nunique() == 4
    assert len(soft) == 8  # 4 soft train volumes + the same 4 test volumes
    chosen = {p: per_patient[p] for p in soft[soft.source_split == "train"].patient_id.unique()}
    assert sorted(chosen.values()) == ["train", "train", "train", "val"]  # exactly the mix that was asked for

    assert call(monkeypatch, "ingest", "--config", cfg, "--source", "ctrate", "--run", "train-soft") == 0
    assert len(downloads) - n_downloaded == 4  # only the 4 new soft volumes: the test pool was already in the shared cache
    assert len(list((data / "cache").glob("*.npy"))) == 16
    assert call(monkeypatch, "merge_manifests", "--config", cfg, "--source", "ctrate", "--run", "train-soft") == 0
    assert call(monkeypatch, "qc_report", "--config", cfg, "--source", "ctrate", "--run", "train-soft") == 0
    assert call(monkeypatch, "assign_splits", "--config", cfg, "--source", "ctrate", "--run", "train-soft") == 0

    ms = pd.read_csv(soft_dir / "manifest.csv")
    soft_split = ms[ms.source_split == "train"].groupby("patient_id").split.first().to_dict()
    assert soft_split == chosen  # every patient kept the split the first run gave it: no leakage between runs
    assert (ms[ms.source_split == "valid"].kernel_class.isin(["sharp", "soft"])).all()

    # the first run and the shared split were not touched by any of it
    assert {f: digest(f) for f in before} == before
    both = run_script("ingest.py", "--config", cfg, "--source", "ctrate", "--status")  # two runs: a name is required
    assert both.returncode == 1 and "several runs" in both.stderr and "train-sharp" in both.stderr


def test_the_kernel_survey_run_is_a_small_run_of_scan_pairs(tmp_path, monkeypatch, synthetic_nifti):
    from ct_preprocessing.ingest import sources

    nifti = synthetic_nifti[0]
    meta = lambda pool, patients: "VolumeName,Rows,Manufacturer,ConvolutionKernel\n" + "".join(  # noqa: E731
        f"{pool}_{p}_a_{r}.nii.gz,512,Philips,{'YA' if r == 1 else 'B'}\n" for p in patients for r in (1, 2))
    fake = make_fake_hf(nifti, meta("train", range(1, 9)), meta("valid", (1, 2)), [])
    monkeypatch.setattr(sources, "get_hf_download", lambda: fake)
    cfg = ctrate_config(tmp_path)
    data = tmp_path / "work" / "data"

    mod = load_script("make_worklist")
    monkeypatch.setattr(mod, "get_hf_download", lambda: fake)
    monkeypatch.setattr(sys, "argv", ["make_worklist.py", "--config", str(cfg), "--survey", "2"])
    assert mod.main() == 0
    wl = pd.read_csv(data / "runs" / "ctrate" / "kernel-survey" / "worklist.csv")
    assert len(wl) == 4 and (wl.groupby(["patient_id", "scan_id"]).size() == 2).all() and set(wl.source_split) == {"train"}

    assert call(monkeypatch, "ingest", "--config", cfg, "--source", "ctrate", "--run", "kernel-survey") == 0
    assert call(monkeypatch, "kernel_survey", "--config", cfg, "--run", "kernel-survey") == 0
    summary = pd.read_csv(data / "runs" / "ctrate" / "kernel-survey" / "kernel_survey_summary.csv")
    assert set(summary.kernel) == {"YA", "B"} and (summary.pairs == 2).all()  # the same synthetic scan twice: a tie, nothing sharper
    assert (data / "runs" / "ctrate" / "kernel-survey" / "kernel_survey.csv").exists()


def test_a_user_error_prints_one_clean_line_not_a_traceback(tmp_path):
    cfg = ctrate_config(tmp_path)  # a CT-RATE source, but make_worklist.py was never run
    proc = run_script("ingest.py", "--config", cfg, "--source", "ctrate")
    assert proc.returncode == 1
    assert proc.stderr.startswith("error:") and "make_worklist.py" in proc.stderr and "Traceback" not in proc.stderr

    unknown = run_script("ingest.py", "--config", cfg, "--source", "nope")
    assert unknown.returncode == 1 and "unknown source" in unknown.stderr and "Traceback" not in unknown.stderr

    merge = run_script("merge_manifests.py", "--config", cfg, "--source", "ctrate")
    assert merge.returncode == 1 and "make_worklist.py" in merge.stderr


def test_make_worklist_flags_that_do_not_belong_together_are_refused(tmp_path):
    cfg = ctrate_config(tmp_path)
    for args, message in (
        (["--from-run", "x"], "needs both --train-patients and --val-patients"),
        (["--train-patients", "3", "--val-patients", "1"], "only make sense with --from-run"),
        (["--survey", "2", "--max-train-patients", "5"], "--survey cannot be combined"),
    ):
        proc = run_script("make_worklist.py", "--config", cfg, *args)
        assert proc.returncode != 0 and message in (proc.stdout + proc.stderr)
