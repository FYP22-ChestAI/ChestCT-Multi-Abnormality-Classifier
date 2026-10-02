"""Both sources, start to end, through the real scripts:
(worklist ->) ingest -> merge -> qc -> assign_splits."""
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


# ================================================================== NHRD (archives)
def nhrd_config(tmp_path, drive: Path) -> Path:
    w = tmp_path / "work"
    cfg = w / "preprocessing.yaml"
    w.mkdir(exist_ok=True)
    cfg.write_text(f"""
paths:
  raw_dir: {w / 'data/raw'}
  cache_dir: {w / 'data/cache'}
  manifest_path: {w / 'data/manifest.csv'}
  qc_report_path: {w / 'data/qc_report.csv'}
  montage_dir: {w / 'data/qc_montages'}
  preprocessing_manifest_path: {w / 'data/preprocessing_manifest.json'}
  chunk_manifest_dir: {w / 'data/manifests'}
  state_dir: {w / 'data/ingest_state'}
  splits_dir: {w / 'data/splits'}
sources:
  nhrd_local:
    format: dicom
    raw_dir: {w / 'data/raw/nhrd_local'}
    manifest_builder: folder
    ingest: {{drive_remote: "{drive.as_posix()}", min_free_gb: 0, workers: 1}}
    split: {{n_val_patients: 1, n_test_patients: 1}}
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

    # 1. ingest: both archives, no arguments beyond the source
    proc = run_script("ingest.py", "--config", cfg, "--source", "nhrd_local")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.splitlines()[0].startswith("using: source=nhrd_local")  # settings are shown first
    assert "2 chunk(s) pending" in proc.stdout
    assert len(list((data / "cache").glob("*.npy"))) == 6
    assert [p.name for p in (data / "raw" / "nhrd_local").iterdir()] == [".ingest_scratch"]  # raw is gone

    # re-running is a no-op
    again = run_script("ingest.py", "--config", cfg, "--source", "nhrd_local")
    assert again.returncode == 0 and "0 chunk(s) pending" in again.stdout
    status = run_script("ingest.py", "--config", cfg, "--source", "nhrd_local", "--status")
    assert "chunks done: 2 / 2" in status.stdout

    # 2. merge, with the checklist of patients on the disk
    merged = run_script("merge_manifests.py", "--config", cfg, "--patients-file", tmp_path / "patients_on_disk.txt")
    assert merged.returncode == 0, merged.stdout + merged.stderr
    assert "0 patient folder(s) missing" in merged.stdout
    manifest = pd.read_csv(data / "manifest.csv")
    assert len(manifest) == 6 and set(manifest.split) == {"unassigned"}
    assert sorted(manifest.patient_id) == patients  # no PatientID tag survives -> grouped by top folder
    assert (data / "preprocessing_manifest.json").exists()

    # 3. QC
    qc = run_script("qc_report.py", "--config", cfg)
    assert qc.returncode == 0, qc.stdout + qc.stderr
    assert pd.read_csv(data / "qc_report.csv").passed.all()

    # 4. split once: 1 val + 1 test patient, the rest train
    split = run_script("assign_splits.py", "--config", cfg)
    assert split.returncode == 0, split.stdout + split.stderr
    final = pd.read_csv(data / "manifest.csv")
    assert final.split.value_counts().to_dict() == {"train": 4, "val": 1, "test": 1}
    assert final.groupby("patient_id").split.nunique().max() == 1  # a patient is in exactly one split
    frozen = (data / "splits" / "nhrd_local.csv").read_text()

    # 5. frozen: running it again changes nothing, even with a different seed
    run_script("assign_splits.py", "--config", cfg, "--seed", 999)
    assert (data / "splits" / "nhrd_local.csv").read_text() == frozen
    assert pd.read_csv(data / "manifest.csv").split.tolist() == final.split.tolist()


def test_a_missing_patient_is_reported_by_the_checklist(tmp_path, dicom_writer, synthetic_hu):
    drive, patients = build_drive(tmp_path, dicom_writer, synthetic_hu)
    cfg = nhrd_config(tmp_path, drive)
    run_script("ingest.py", "--config", cfg, "--source", "nhrd_local")
    (tmp_path / "patients_on_disk.txt").write_text("\n".join([*patients, "9999-99"]) + "\n")
    merged = run_script("merge_manifests.py", "--config", cfg, "--patients-file", tmp_path / "patients_on_disk.txt")
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
    run_script("merge_manifests.py", "--config", cfg)
    proc = run_script("assign_splits.py", "--config", cfg)
    assert proc.returncode != 0 and "qc_report.py" in (proc.stdout + proc.stderr)
    assert run_script("assign_splits.py", "--config", cfg, "--skip-qc", "--dry-run").returncode == 0


# ================================================================== CT-RATE (Hugging Face)
def make_fake_hf(nifti: Path, train_meta: str, valid_meta: str):
    def download(repo_id, filename, repo_type=None, cache_dir=None, **kw):
        base = Path(cache_dir) if cache_dir else Path(nifti).parent / "hf_default"
        target = base / "snapshots" / "x" / Path(filename).name
        target.parent.mkdir(parents=True, exist_ok=True)
        if filename.endswith("train_metadata.csv"):
            target.write_text(train_meta)
        elif filename.endswith("validation_metadata.csv"):
            target.write_text(valid_meta)
        else:
            shutil.copyfile(nifti, target)
        return str(target)
    return download


def ctrate_config(tmp_path) -> Path:
    w = tmp_path / "work"
    w.mkdir(exist_ok=True)
    cfg = w / "preprocessing.yaml"
    cfg.write_text(f"""
paths:
  raw_dir: {w / 'data/raw'}
  cache_dir: {w / 'data/cache'}
  manifest_path: {w / 'data/manifest.csv'}
  qc_report_path: {w / 'data/qc_report.csv'}
  montage_dir: {w / 'data/qc_montages'}
  preprocessing_manifest_path: {w / 'data/preprocessing_manifest.json'}
  metadata_dir: {w / 'data/metadata'}
  worklist_dir: {w / 'data/worklists'}
  chunk_manifest_dir: {w / 'data/manifests'}
  state_dir: {w / 'data/ingest_state'}
  splits_dir: {w / 'data/splits'}
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


def test_ctrate_from_metadata_to_frozen_splits(tmp_path, monkeypatch, synthetic_nifti):
    from ct_preprocessing.ingest import sources

    nifti = synthetic_nifti[0]
    train = "VolumeName,Rows\n" + "".join(f"train_{p}_a_{r}.nii.gz,512\n" for p in range(1, 9) for r in (1, 2))
    valid = "VolumeName,Rows\n" + "".join(f"valid_{p}_a_{r}.nii.gz,512\n" for p in (1, 2) for r in (1, 2))
    fake = make_fake_hf(nifti, train, valid)
    monkeypatch.setattr(sources, "get_hf_download", lambda: fake)
    cfg = ctrate_config(tmp_path)
    data = tmp_path / "work" / "data"

    # 1. worklist: the whole valid pool (4 volumes = 1 chunk) + 8 train volumes (one reconstruction per scan) = 2 chunks
    mod = load_script("make_worklist")
    monkeypatch.setattr(mod, "get_hf_download", lambda: fake)
    monkeypatch.setattr(sys, "argv", ["make_worklist.py", "--config", str(cfg)])
    assert mod.main() == 0
    wl = pd.read_csv(data / "worklists" / "ctrate.csv")
    assert len(wl) == 12 and wl.chunk.nunique() == 3
    assert (wl[wl.chunk == 0].source_split == "valid").all()  # test pool first
    monkeypatch.setattr(sys, "argv", ["make_worklist.py", "--config", str(cfg)])
    with pytest.raises(SystemExit, match="already exists"):  # never silently rebuilt
        mod.main()

    # 2. ingest every chunk
    assert call(monkeypatch, "ingest", "--config", cfg, "--source", "ctrate") == 0
    assert len(list((data / "cache").glob("*.npy"))) == 12
    assert [p.name for p in (data / "raw" / "ctrate").iterdir()] == [".ingest_scratch"]

    # 3-4. merge, QC
    assert call(monkeypatch, "merge_manifests", "--config", cfg) == 0
    assert call(monkeypatch, "qc_report", "--config", cfg) == 0

    # 5. split: test = official valid pool; val = 2 train patients; the other 6 train patients = train
    assert call(monkeypatch, "assign_splits", "--config", cfg) == 0
    m = pd.read_csv(data / "manifest.csv")
    assert m[m.source_split == "valid"].split.eq("test").all() and m[m.source_split == "train"].split.isin(["train", "val"]).all()
    per_patient = m.groupby("patient_id").split.first()
    assert per_patient[per_patient.index.str.startswith("train_")].value_counts().to_dict() == {"train": 6, "val": 2}
    assert (m.split == "test").sum() == 4  # both reconstructions of both valid patients



def test_a_user_error_prints_one_clean_line_not_a_traceback(tmp_path):
    cfg = ctrate_config(tmp_path)  # a CT-RATE source, but make_worklist.py was never run
    proc = run_script("ingest.py", "--config", cfg, "--source", "ctrate")
    assert proc.returncode == 1
    assert proc.stderr.startswith("error:") and "make_worklist.py" in proc.stderr and "Traceback" not in proc.stderr

    unknown = run_script("ingest.py", "--config", cfg, "--source", "nope")
    assert unknown.returncode == 1 and "unknown source" in unknown.stderr and "Traceback" not in unknown.stderr

    merge = run_script("merge_manifests.py", "--config", cfg)
    assert merge.returncode == 1 and "run ingest.py" in merge.stderr
