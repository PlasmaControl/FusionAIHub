"""Environment migration preserves old deployments and gives new names priority."""

import logging
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
@pytest.mark.parametrize("mode", ["new", "old", "both"])
def test_paths_use_new_names_first_and_warn_only_for_legacy(
    suffix, field, mode, monkeypatch, caplog, tmp_path,
):
    new, old = f"LABELER_{suffix}", f"LABELMAKER_{suffix}"
    monkeypatch.delenv(new, raising=False)
    monkeypatch.delenv(old, raising=False)
    if mode in ("new", "both"):
        monkeypatch.setenv(new, str(tmp_path / "new"))
    if mode in ("old", "both"):
        monkeypatch.setenv(old, str(tmp_path / "old"))
    with caplog.at_level(logging.WARNING):
        paths = Paths.from_env()
    assert getattr(paths, field) == tmp_path / ("old" if mode == "old" else "new")
    notices = [r.getMessage() for r in caplog.records if old in r.getMessage()]
    assert len(notices) == (1 if mode == "old" else 0)
    if notices:
        assert new in notices[0] and "\n" not in notices[0]


def test_empty_new_root_does_not_fall_back(monkeypatch, caplog):
    monkeypatch.setenv("LABELER_ROOT", "")
    monkeypatch.setenv("LABELMAKER_ROOT", "/unused")
    assert Paths.from_env().root == Path("")
    assert not any("LABELMAKER_ROOT" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("suffix", ['ROOT', 'CORPUS', 'TEXT_ROOT', 'LOGS_JSONL', 'LABEL_TABLES', 'FDP', 'GIT_SHA', 'DEMO_OUT', 'DEMO_CONFIG', 'DEMO_FORCE'])
@pytest.mark.parametrize("query_legacy", [False, True])
@pytest.mark.parametrize("mode", ["new", "old", "both", "empty", "unset"])
def test_environment_lookup_supports_either_spelling(
    suffix, query_legacy, mode, monkeypatch, caplog,
):
    from labeler.env import getenv

    new, old = "LABELER_" + suffix, "LABELMAKER_" + suffix
    monkeypatch.delenv(new, raising=False)
    monkeypatch.delenv(old, raising=False)
    if mode in ("new", "both", "empty"):
        monkeypatch.setenv(new, "" if mode == "empty" else "new-value")
    if mode in ("old", "both", "empty"):
        monkeypatch.setenv(old, "old-value")
    want = {"new": "new-value", "old": "old-value", "both": "new-value",
            "empty": "", "unset": "default"}[mode]
    assert getenv(old if query_legacy else new, "default") == want
    assert len(caplog.records) == (1 if mode == "old" else 0)
    if caplog.records:
        message = caplog.records[0].getMessage()
        assert old in message and new in message
        assert "old-value" not in message and "\n" not in message


@pytest.mark.parametrize("legacy", [False, True])
def test_shell_entrypoint_resolves_settings_without_importing_the_package(
    legacy, monkeypatch,
):
    import subprocess
    import sys

    from labeler import env

    monkeypatch.setenv("LABELMAKER_ROOT" if legacy else "LABELER_ROOT", "a path with spaces")
    result = subprocess.run(
        [sys.executable, env.__file__, "LABELER_ROOT", "default"],
        capture_output=True, text=True, check=True,
    )
    assert result.stdout == "a path with spaces\n"
    assert bool(result.stderr) == legacy
    assert len(result.stderr.splitlines()) == int(legacy)
