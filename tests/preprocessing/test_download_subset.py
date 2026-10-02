"""Tests for download_subset.py's pure logic (path construction, memory
screening) -- none of these touch the network."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "preprocessing"))
from download_subset import _num, _passes_screen, ctrate_rel_path  # noqa: E402


def test_ctrate_rel_path_matches_the_confirmed_fixed_layout():
    assert ctrate_rel_path("train_10_a_1", "train", fixed=True) == (
        "dataset/train_fixed/train_10/train_10_a/train_10_a_1.nii.gz"
    )
    assert ctrate_rel_path("valid_1002_a_2", "valid", fixed=True) == (
        "dataset/valid_fixed/valid_1002/valid_1002_a/valid_1002_a_2.nii.gz"
    )


def test_ctrate_rel_path_rejects_the_wrong_split():
    with pytest.raises(ValueError):
        ctrate_rel_path("valid_1_a_1", "train", fixed=True)


def test_num_extracts_the_first_number_from_a_list_like_string():
    assert _num("[0.91796875, 0.91796875]") == pytest.approx(0.91796875)
    assert _num("1.5") == 1.5


def test_passes_screen_lets_small_volumes_through():
    metadata = pd.DataFrame(
        {"VolumeName": ["v1"], "Rows": [512], "Columns": [512], "NumberofSlices": [200], "XYSpacing": ["[1.0, 1.0]"], "ZSpacing": [1.5]}
    ).set_index("VolumeName")
    assert _passes_screen("v1", metadata, max_combined_gb=10.0, target_spacing_zyx=(1.5, 0.75, 0.75))


def test_passes_screen_rejects_the_confirmed_heavy_real_volume():
    # The exact real CT-RATE case that OOM'd on Colab's free tier.
    metadata = pd.DataFrame(
        {
            "VolumeName": ["train_10_a_1"],
            "Rows": [1024],
            "Columns": [1024],
            "NumberofSlices": [237],
            "XYSpacing": ["[1.0, 1.0]"],
            "ZSpacing": [1.0],
        }
    ).set_index("VolumeName")
    assert not _passes_screen("train_10_a_1", metadata, max_combined_gb=1.5, target_spacing_zyx=(1.5, 0.75, 0.75))


def test_passes_screen_lets_unknown_volumes_through():
    metadata = pd.DataFrame({"VolumeName": [], "Rows": [], "Columns": []}).set_index("VolumeName")
    assert _passes_screen("not_in_metadata", metadata, max_combined_gb=1.0, target_spacing_zyx=(1.5, 0.75, 0.75))
