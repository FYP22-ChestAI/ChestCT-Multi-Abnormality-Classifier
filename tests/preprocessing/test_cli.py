"""The small shared command-line helpers."""
import pytest

from ct_preprocessing.cli import friendly_errors, show_settings


def test_expected_errors_become_one_clean_line_and_exit_code_1(capsys):
    @friendly_errors
    def main():
        raise FileNotFoundError("data/worklists/ctrate.csv not found -- run make_worklist.py first")

    assert main() == 1
    out = capsys.readouterr()
    assert out.err == "error: data/worklists/ctrate.csv not found -- run make_worklist.py first\n" and out.out == ""


def test_a_key_error_shows_its_message_without_python_quoting(capsys):
    @friendly_errors
    def main():
        raise KeyError("unknown source 'beta'; configured sources: ['alpha']")

    assert main() == 1
    assert capsys.readouterr().err == "error: unknown source 'beta'; configured sources: ['alpha']\n"


def test_a_real_bug_is_not_swallowed_and_debug_mode_shows_every_traceback(monkeypatch):
    @friendly_errors
    def bug():
        return {}["x"]  # a KeyError from a bug: still reported as a clean line normally...

    @friendly_errors
    def typeerror():
        raise TypeError("a genuine bug")  # ...but anything unexpected is never hidden

    with pytest.raises(TypeError):
        typeerror()
    monkeypatch.setenv("CT_DEBUG", "1")
    with pytest.raises(KeyError):
        bug()


def test_a_normal_return_value_is_passed_through():
    @friendly_errors
    def main():
        return 2

    assert main() == 2


def test_show_settings_prints_one_line_with_every_value(capsys):
    show_settings("using", {"source": "ctrate", "workers": 4, "max_chunks": None})
    assert capsys.readouterr().out == "using: source=ctrate, workers=4, max_chunks=None\n"
