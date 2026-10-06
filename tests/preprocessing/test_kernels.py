"""Reconstruction kernels: names, the reviewed table, the draft guess, and the survey's sharpness measure."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ct_preprocessing.kernels import (
    add_kernel_columns, classify, draft_class, load_kernel_table, normalize_kernel, sharpness_score,
)

REPO = Path(__file__).resolve().parents[2]


def test_kernel_names_are_normalised_from_every_form_the_metadata_uses():
    assert normalize_kernel("['Br40f', '3']") == "Br40f"
    assert normalize_kernel("YA") == "YA" and normalize_kernel("  FC81 ") == "FC81"
    assert normalize_kernel("['B70s']") == "B70s"
    assert normalize_kernel("") == "" and normalize_kernel(None) == "" and normalize_kernel(float("nan")) == ""


def test_the_table_is_read_case_insensitively_and_unknown_pairs_are_other(tmp_path):
    f = tmp_path / "k.csv"
    f.write_text("manufacturer,kernel,class\nPhilips,YA,sharp\nSIEMENS,Br36d,soft\n")
    table = load_kernel_table(f)
    assert classify("philips", "ya", table) == "sharp" and classify("PHILIPS", "['YA', '3']", table) == "sharp"
    assert classify("Siemens", "Br36d", table) == "soft"
    assert classify("Philips", "ZZ", table) == "other" and classify("Nobody", "YA", table) == "other"


def test_a_bad_class_or_missing_table_is_a_clear_error(tmp_path):
    f = tmp_path / "k.csv"
    f.write_text("manufacturer,kernel,class\nPhilips,YA,medium\n")
    with pytest.raises(ValueError, match="class must be one of"):
        load_kernel_table(f)
    with pytest.raises(FileNotFoundError, match="make_kernel_table.py"):
        load_kernel_table(tmp_path / "missing.csv")


def test_kernel_columns_are_added_to_a_frame():
    df = pd.DataFrame({"m": ["Philips", "Philips"], "k": ["YA", "['B', '3']"]})
    out = add_kernel_columns(df, {("philips", "ya"): "sharp", ("philips", "b"): "soft"}, manufacturer_col="m", kernel_col="k")
    assert out.kernel.tolist() == ["YA", "B"] and out.kernel_class.tolist() == ["sharp", "soft"]
    assert list(df.columns) == ["m", "k"]  # the input is not modified


def test_the_draft_uses_the_description_first_then_the_kernel_name():
    assert draft_class("Philips", "YA", "HRCT") == "sharp"
    assert draft_class("Philips", "B", "MEDIASTEN, iDose (4)") == "soft"
    assert draft_class("Philips", "L", "PARANKIM, iDose (4)") == "sharp"
    assert draft_class("Siemens Healthineers", "Br40f", "Thorax 1,50 Br40 S3") == "soft"
    assert draft_class("Siemens Healthineers", "Br60f", "") == "sharp"
    assert draft_class("Siemens Healthineers", "Bl56f", "") == "sharp"
    assert draft_class("SIEMENS", "B70s", "") == "sharp" and draft_class("SIEMENS", "B31s", "") == "soft"
    assert draft_class("Philips", "UB", "240") == "other" and draft_class("Philips", "D", "KEMIK") == "other"


def test_the_shipped_table_loads_and_covers_the_main_ctrate_and_nhrd_kernels():
    table = load_kernel_table(REPO / "configs" / "kernel_classes.csv")
    assert classify("Philips", "YA", table) == "sharp" and classify("Philips", "B", table) == "soft"
    assert classify("Siemens Healthineers", "Br60f", table) == "sharp" and classify("Siemens Healthineers", "Br40f", table) == "soft"
    assert classify("TOSHIBA", "FC81", table) == "sharp"  # NHRD's scanner
    assert len(table) >= 30


# --------------------------------------------------------------------- the survey measure
def _scan(noise, seed=0, n=40):
    rng = np.random.default_rng(seed)
    base = np.full((n, 48, 48), -800.0, dtype=np.float32)
    base[:, 10:38, 10:38] = 40.0
    return base + rng.normal(0, noise, base.shape).astype(np.float32)


def test_a_noisier_sharper_reconstruction_scores_higher_than_a_smooth_one():
    assert sharpness_score(_scan(60)) > sharpness_score(_scan(5))


def test_the_sharpness_score_is_stable_and_ignores_air_outside_the_body():
    a = _scan(20)
    assert sharpness_score(a) == pytest.approx(sharpness_score(a.copy()))
    only_air = np.full((10, 20, 20), -1000.0, dtype=np.float32)
    assert sharpness_score(only_air) == 0.0
