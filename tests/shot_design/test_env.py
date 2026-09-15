"""Path resolution remains compatible with legacy environment names."""

import logging

import pytest

from shot_design import config


@pytest.mark.parametrize("mode", ["new", "old", "both"])
def test_data_root_uses_new_name_first_and_warns_only_for_legacy(
    mode, monkeypatch, caplog, tmp_path,
):
    monkeypatch.delenv("SHOT_DESIGN_DATA_ROOT", raising=False)
    monkeypatch.delenv("IDEATE_DATA_ROOT", raising=False)
    monkeypatch.delenv("SHOT_DESIGN_PATHS", raising=False)
    monkeypatch.delenv("IDEATE_PATHS", raising=False)
    if mode in ("new", "both"):
        monkeypatch.setenv("SHOT_DESIGN_DATA_ROOT", str(tmp_path / "new"))
    if mode in ("old", "both"):
        monkeypatch.setenv("IDEATE_DATA_ROOT", str(tmp_path / "old"))
    with caplog.at_level(logging.WARNING):
        paths = config.load_paths()
    assert paths.db_dir == tmp_path / ("old" if mode == "old" else "new") / "db"
    notices = [r.getMessage() for r in caplog.records if "IDEATE_DATA_ROOT" in r.getMessage()]
    assert len(notices) == (1 if mode == "old" else 0)
    if notices:
        assert "SHOT_DESIGN_DATA_ROOT" in notices[0] and "\n" not in notices[0]


def test_empty_new_data_root_is_rejected_instead_of_falling_back(monkeypatch):
    monkeypatch.setenv("SHOT_DESIGN_DATA_ROOT", "")
    monkeypatch.setenv("IDEATE_DATA_ROOT", "/unused")
    with pytest.raises(ValueError, match="SHOT_DESIGN_DATA_ROOT is empty"):
        config.load_paths()


@pytest.mark.parametrize("prefix", ["SHOT_DESIGN_", "IDEATE_", "LABELER_", "LABELMAKER_"])
def test_interpolation_accepts_either_namespace_and_mapping_keys_win(prefix, monkeypatch, caplog):
    new = "LABELER_ROOT" if prefix.startswith("LABEL") else "SHOT_DESIGN_DATA_ROOT"
    old = "LABELMAKER_ROOT" if prefix.startswith("LABEL") else "IDEATE_DATA_ROOT"
    query = prefix + ("ROOT" if prefix.startswith("LABEL") else "DATA_ROOT")
    monkeypatch.setenv(old, "/legacy")
    assert config._interpolate({"path": "${" + query + "}/db"})["path"] == "/legacy/db"
    monkeypatch.setenv(new, "/new")
    caplog.clear()
    assert config._interpolate({"path": "${" + query + "}/db"})["path"] == "/new/db"
    monkeypatch.delenv(new)
    assert config._interpolate({query: "/mapped", "path": "${" + query + "}/db"})["path"] == "/mapped/db"
    assert not caplog.records


@pytest.mark.parametrize("legacy", [False, True])
def test_paths_file_fallback_and_origin(legacy, monkeypatch, tmp_path):
    import yaml

    paths_file = tmp_path / "paths.yaml"
    raw = config.load_yaml("paths.yaml")
    raw["data_root"] = str(tmp_path / "database")
    paths_file.write_text(yaml.safe_dump(raw))
    name = "IDEATE_PATHS" if legacy else "SHOT_DESIGN_PATHS"
    monkeypatch.setenv(name, str(paths_file))
    assert config.load_paths().db_dir == tmp_path / "database" / "db"
    assert config.data_root_origin() == f"{name}={paths_file}"


@pytest.mark.parametrize("legacy", [False, True])
def test_config_directory_override_is_resolved_before_import(legacy, monkeypatch, tmp_path):
    import subprocess
    import sys

    (tmp_path / "probe.yaml").write_text("selected: custom\n")
    name = "IDEATE_CONFIG_DIR" if legacy else "SHOT_DESIGN_CONFIG_DIR"
    monkeypatch.setenv(name, str(tmp_path))
    result = subprocess.run(
        [sys.executable, "-c", "from shot_design.config import load_yaml; print(load_yaml('probe.yaml')['selected'])"],
        capture_output=True, text=True, check=True,
    )
    assert result.stdout == "custom\n"
    assert ("IDEATE_CONFIG_DIR is deprecated" in result.stderr) == legacy


@pytest.mark.parametrize("suffix", ['DATA_ROOT', 'CORPUS', 'CONFIG_DIR', 'PATHS', 'HF_ONLINE', 'TEXT_ROOT', 'IGNITE_CKPT', 'PY', 'ENV'])
@pytest.mark.parametrize("query_legacy", [False, True])
@pytest.mark.parametrize("mode", ["new", "old", "both", "empty", "unset"])
def test_environment_lookup_supports_either_spelling(
    suffix, query_legacy, mode, monkeypatch, caplog,
):
    from shot_design.env import getenv

    new, old = "SHOT_DESIGN_" + suffix, "IDEATE_" + suffix
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

    from shot_design import env

    monkeypatch.setenv("IDEATE_DATA_ROOT" if legacy else "SHOT_DESIGN_DATA_ROOT", "a path with spaces")
    result = subprocess.run(
        [sys.executable, env.__file__, "SHOT_DESIGN_DATA_ROOT", "default"],
        capture_output=True, text=True, check=True,
    )
    assert result.stdout == "a path with spaces\n"
    assert bool(result.stderr) == legacy
    assert len(result.stderr.splitlines()) == int(legacy)
