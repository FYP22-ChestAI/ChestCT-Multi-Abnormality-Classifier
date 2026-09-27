"""Tests for build_manifest.py's pure logic (SOURCE_NAME validation)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_manifest import validate_source_name  # noqa: E402


def test_validate_source_name_trims_incidental_leading_and_trailing_space():
    # A stray space at either end is an easy accidental keystroke in a form
    # field (Colab's @param boxes included) -- it should be silently
    # trimmed, not rejected, unlike a space genuinely inside the name.
    assert validate_source_name("ct-rate ") == "ct-rate"
    assert validate_source_name(" ct-rate") == "ct-rate"
    assert validate_source_name("  ct-rate  ") == "ct-rate"


def test_validate_source_name_rejects_an_internal_space():
    with pytest.raises(SystemExit):
        validate_source_name("ct rate")


def test_validate_source_name_rejects_quote_characters():
    with pytest.raises(SystemExit):
        validate_source_name('ct"rate')
    with pytest.raises(SystemExit):
        validate_source_name("ct'rate")


def test_validate_source_name_rejects_empty_or_all_whitespace():
    with pytest.raises(SystemExit):
        validate_source_name("")
    with pytest.raises(SystemExit):
        validate_source_name("   ")
