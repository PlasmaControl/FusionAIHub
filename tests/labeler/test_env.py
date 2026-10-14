"""Environment settings are read under their LABELER_* names."""

import subprocess
import sys
from pathlib import Path

import pytest

from labeler.config import Paths


@pytest.mark.parametrize("suffix,field", [
    ("ROOT", "root"),
    ("CORPUS", "corpus"),
    ("TEXT_ROOT", "text_root"),
    ("LOGS_JSONL", "logs_jsonl"),
    ("LABEL_TABLES", "label_tables"),
])
@pytest.mark.parametrize("mode", ["set", "unset"])
def test_paths_read_the_labeler_names(suffix, field, mode, monkeypatch, tmp_path):
    name = f"LABELER_{suffix}"
    monkeypatch.delenv(name, raising=False)
    default = getattr(Paths.from_env(), field)
    if mode == "set":
        monkeypatch.setenv(name, str(tmp_path / "value"))
    expected = tmp_path / "value" if mode == "set" else default
    assert getattr(Paths.from_env(), field) == expected


def test_empty_root_is_an_empty_path_not_the_default(monkeypatch):
    monkeypatch.setenv("LABELER_ROOT", "")
    assert Paths.from_env().root == Path("")


@pytest.mark.parametrize("suffix", [
    "ROOT", "CORPUS", "TEXT_ROOT", "LOGS_JSONL", "LABEL_TABLES", "FDP", "GIT_SHA",
    "DEMO_OUT", "DEMO_CONFIG", "DEMO_FORCE",
])
@pytest.mark.parametrize("mode", ["set", "empty", "unset"])
def test_environment_lookup(suffix, mode, monkeypatch):
    from labeler.env import getenv

    name = "LABELER_" + suffix
    monkeypatch.delenv(name, raising=False)
    if mode != "unset":
        monkeypatch.setenv(name, "" if mode == "empty" else "value")
    want = {"set": "value", "empty": "", "unset": "default"}[mode]
    assert getenv(name, "default") == want


def test_shell_entrypoint_resolves_settings_without_importing_the_package(monkeypatch):
    from labeler import env

    monkeypatch.setenv("LABELER_ROOT", "a path with spaces")
    result = subprocess.run(
        [sys.executable, env.__file__, "LABELER_ROOT", "default"],
        capture_output=True, text=True, check=True,
    )
    assert result.stdout == "a path with spaces\n"
    assert result.stderr == ""
