"""Path resolution reads the SHOT_DESIGN_* and LABELER_* environment names."""

import pytest

from shot_design import config


def test_data_root_comes_from_the_environment(monkeypatch, tmp_path):
    monkeypatch.delenv("SHOT_DESIGN_PATHS", raising=False)
    monkeypatch.setenv("SHOT_DESIGN_DATA_ROOT", str(tmp_path / "root"))
    assert config.load_paths().db_dir == tmp_path / "root" / "db"
    assert config.data_root_origin() == "SHOT_DESIGN_DATA_ROOT env"


def test_empty_data_root_is_rejected_instead_of_falling_back(monkeypatch):
    monkeypatch.setenv("SHOT_DESIGN_DATA_ROOT", "")
    with pytest.raises(ValueError, match="SHOT_DESIGN_DATA_ROOT is empty"):
        config.load_paths()


@pytest.mark.parametrize("prefix,suffix", [
    ("SHOT_DESIGN_", "DATA_ROOT"), ("LABELER_", "ROOT"),
])
def test_interpolation_reads_the_environment_and_mapping_keys_win(
    prefix, suffix, monkeypatch,
):
    name = prefix + suffix
    monkeypatch.delenv(name, raising=False)
    value = {"path": "${" + name + "}/db"}
    assert config._interpolate(value)["path"] == "${" + name + "}/db"
    monkeypatch.setenv(name, "/new")
    assert config._interpolate(value)["path"] == "/new/db"
    monkeypatch.delenv(name)
    assert config._interpolate({name: "/mapped", **value})["path"] == "/mapped/db"


def test_paths_file_selection_and_origin(monkeypatch, tmp_path):
    import yaml

    paths_file = tmp_path / "paths.yaml"
    raw = config.load_yaml("paths.yaml")
    raw["data_root"] = str(tmp_path / "database")
    paths_file.write_text(yaml.safe_dump(raw))
    monkeypatch.delenv("SHOT_DESIGN_DATA_ROOT", raising=False)
    monkeypatch.setenv("SHOT_DESIGN_PATHS", str(paths_file))
    assert config.load_paths().db_dir == tmp_path / "database" / "db"
    assert config.data_root_origin() == f"SHOT_DESIGN_PATHS={paths_file}"


def test_config_directory_override_is_resolved_before_import(monkeypatch, tmp_path):
    import subprocess
    import sys

    (tmp_path / "probe.yaml").write_text("selected: custom\n")
    monkeypatch.setenv("SHOT_DESIGN_CONFIG_DIR", str(tmp_path))
    result = subprocess.run(
        [sys.executable, "-c", "from shot_design.config import load_yaml; print(load_yaml('probe.yaml')['selected'])"],
        capture_output=True, text=True, check=True,
    )
    assert result.stdout == "custom\n"
    assert result.stderr == ""


@pytest.mark.parametrize("suffix", [
    "DATA_ROOT", "CORPUS", "CONFIG_DIR", "PATHS", "HF_ONLINE", "TEXT_ROOT",
    "IGNITE_CKPT", "PY", "ENV",
])
@pytest.mark.parametrize("mode", ["set", "empty", "unset"])
def test_environment_lookup(suffix, mode, monkeypatch):
    from shot_design.env import getenv

    name = "SHOT_DESIGN_" + suffix
    monkeypatch.delenv(name, raising=False)
    if mode != "unset":
        monkeypatch.setenv(name, "" if mode == "empty" else "value")
    want = {"set": "value", "empty": "", "unset": "default"}[mode]
    assert getenv(name, "default") == want


def test_shell_entrypoint_resolves_settings_without_importing_the_package(monkeypatch):
    import subprocess
    import sys

    from shot_design import env

    monkeypatch.setenv("SHOT_DESIGN_DATA_ROOT", "a path with spaces")
    result = subprocess.run(
        [sys.executable, env.__file__, "SHOT_DESIGN_DATA_ROOT", "default"],
        capture_output=True, text=True, check=True,
    )
    assert result.stdout == "a path with spaces\n"
    assert result.stderr == ""


# ------------------------------------------------------------------ cluster defaults


def test_data_root_is_the_clusters_when_nothing_else_names_one(monkeypatch):
    from labeler import env

    monkeypatch.delenv("SHOT_DESIGN_DATA_ROOT", raising=False)
    monkeypatch.delenv("SHOT_DESIGN_PATHS", raising=False)
    for name, row in env.CLUSTERS.items():
        monkeypatch.setenv("LABELER_CLUSTER", name)
        root = row["env"]["SHOT_DESIGN_DATA_ROOT"]
        assert str(config.load_paths().data_root) == root
        assert config.data_root_origin() == f"cluster {name} default"


def test_an_export_or_a_paths_file_outranks_the_clusters_root(monkeypatch, tmp_path):
    import yaml

    monkeypatch.setenv("LABELER_CLUSTER", "stellar")
    paths_file = tmp_path / "paths.yaml"
    raw = config.load_yaml("paths.yaml")
    raw["data_root"] = str(tmp_path / "database")
    paths_file.write_text(yaml.safe_dump(raw))
    monkeypatch.setenv("SHOT_DESIGN_PATHS", str(paths_file))
    assert config.load_paths().data_root == tmp_path / "database"
    assert config.data_root_origin() == f"SHOT_DESIGN_PATHS={paths_file}"
    monkeypatch.setenv("SHOT_DESIGN_DATA_ROOT", str(tmp_path / "root"))
    assert config.load_paths().data_root == tmp_path / "root"
    assert config.data_root_origin() == "SHOT_DESIGN_DATA_ROOT env"


@pytest.mark.parametrize("name", ["stellar", "stellarai"])
def test_shell_entrypoint_answers_with_the_clusters_root(name, monkeypatch):
    """Launchers ask this file as well as labeler's; both read the one table, so neither
    can send a Stellar job to the StellarAI root (or the reverse)."""
    import subprocess
    import sys

    from labeler import env as labeler_env
    from shot_design import env

    monkeypatch.delenv("SHOT_DESIGN_DATA_ROOT", raising=False)
    monkeypatch.setenv("LABELER_CLUSTER", name)
    result = subprocess.run(
        [sys.executable, env.__file__, "SHOT_DESIGN_DATA_ROOT", "default"],
        capture_output=True, text=True, check=True,
    )
    assert result.stdout == labeler_env.CLUSTERS[name]["env"]["SHOT_DESIGN_DATA_ROOT"] + "\n"


def test_labeler_addressed_readers_use_the_export_then_the_clusters_root(monkeypatch):
    """The shot-design pixi environment no longer exports LABELER_ROOT, so the readers that
    used to see it there resolve it the way `labeler.config.Paths.from_env` does."""
    import argparse
    from pathlib import Path

    from labeler import env
    from shot_design import cli
    from shot_design.shotdb import corpus_signals

    args = argparse.Namespace(pending_out=None, name="shots")
    monkeypatch.delenv("LABELER_ROOT", raising=False)
    monkeypatch.setenv("LABELER_CLUSTER", "")
    assert corpus_signals.default_features_dir() is None
    assert cli._pending_path(args) is None
    monkeypatch.setenv("LABELER_CLUSTER", "stellar")
    root = Path(env.CLUSTERS["stellar"]["env"]["LABELER_ROOT"])
    assert corpus_signals.default_features_dir() == root / "features"
    assert cli._pending_path(args) == root / "shots_pending_features.txt"
    monkeypatch.setenv("LABELER_ROOT", "/exported")
    assert corpus_signals.default_features_dir() == Path("/exported/features")
    assert cli._pending_path(args) == Path("/exported/shots_pending_features.txt")
