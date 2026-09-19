"""Ported from shot-recommender-system (shotrec) @565d548."""

# tests/test_config.py
from pathlib import Path

import pytest

from shot_design import config

# A minimal but complete Paths YAML: every field the model requires, all but data_root
# fixed to an arbitrary literal so tests only need to vary the one field they care about.
_PATHS_YAML_TEMPLATE = """\
data_root: {data_root}
raw_dir: /x/raw
db_dir: /x/db
text_cache_dir: /x/text_cache
ignite_inputs_dir: /x/ignite_inputs
models_dir: /x/models
eval_dir: /x/eval
sessions_dir: /x/sessions
actuations_dir: /x/actuations
llm_cache_dir: /x/llm_cache
staged_raw_dir: /x/staged_raw
text_root: /x/text_root
logs_jsonl: /x/text_root/sql/logs.jsonl
shot_index_json: /x/text_root/sql/index.json
shotsummary_raw_dir: /x/text_root/shotsummary/raw
per_shot_txt_dir: /x/text_root/shotsummary/processed/per_shot_txt
qh_database_csv: /x/qh.csv
foundation_model_processed_dir: /x/foundation_model
fdp_project_dir: /x/fdp
"""


def test_interpolate_resolves_keys_and_env(monkeypatch):
    monkeypatch.setenv("SHOT_DESIGN_TEST_HOME", "/h")
    out = config._interpolate(
        {"root": "/r", "a": "${root}/a", "b": "${a}/b", "c": "${SHOT_DESIGN_TEST_HOME}/c"}
    )
    assert out == {"root": "/r", "a": "/r/a", "b": "/r/a/b", "c": "/h/c"}


def test_interpolate_leaves_unknown_variable_literal(monkeypatch):
    monkeypatch.delenv("SHOT_DESIGN_NOPE_UNSET", raising=False)
    out = config._interpolate({"a": "${SHOT_DESIGN_NOPE_UNSET}/x"})
    # Designed fallback (see config._interpolate docstring): a ${var} matching neither
    # another key nor an environment variable is left as literal text, not raised.
    assert out == {"a": "${SHOT_DESIGN_NOPE_UNSET}/x"}


def test_load_paths_data_root_override(monkeypatch):
    monkeypatch.setenv("SHOT_DESIGN_DATA_ROOT", "/tmp/shot-design-test")
    p = config.load_paths()
    assert p.raw_dir == Path("/tmp/shot-design-test/raw")
    assert p.staged_raw_dir == Path("/scratch/gpfs/EKOLEMEN/d3d_fusion_data")
    # Covers the text_root -> logs_jsonl interpolation chain, not just that Pydantic can
    # coerce a string into a Path (it always can, even unresolved).
    assert str(p.logs_jsonl).startswith(str(p.text_root))
    assert str(p.logs_jsonl).endswith("/sql/logs.jsonl")


def test_shot_design_paths_env_selects_alternate_file(tmp_path, monkeypatch):
    monkeypatch.delenv("SHOT_DESIGN_DATA_ROOT", raising=False)
    alt = tmp_path / "alt_paths.yaml"
    alt.write_text(_PATHS_YAML_TEMPLATE.format(data_root="/alt/root"))
    monkeypatch.setenv("SHOT_DESIGN_PATHS", str(alt))
    p = config.load_paths()
    assert p.data_root == Path("/alt/root")


def test_load_paths_explicit_path_overrides_shot_design_paths_env(tmp_path, monkeypatch):
    monkeypatch.delenv("SHOT_DESIGN_DATA_ROOT", raising=False)
    env_file = tmp_path / "env_paths.yaml"
    env_file.write_text(_PATHS_YAML_TEMPLATE.format(data_root="/env/root"))
    explicit_file = tmp_path / "explicit_paths.yaml"
    explicit_file.write_text(_PATHS_YAML_TEMPLATE.format(data_root="/explicit/root"))
    monkeypatch.setenv("SHOT_DESIGN_PATHS", str(env_file))
    p = config.load_paths(explicit_file)
    assert p.data_root == Path("/explicit/root")


def test_load_paths_missing_file_raises(tmp_path):
    missing = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError):
        config.load_paths(missing)


def test_load_yaml_reads_configs_dir():
    assert "data_root" in config.load_yaml("paths.yaml")


def test_config_discovery_is_independent_of_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert config.CONFIG_DIR == Path(config.__file__).resolve().parents[2] / "configs" / "shot_design"
    assert "signals" in config.load_yaml("signals.yaml")


def test_config_dir_environment_override(tmp_path, monkeypatch):
    import os
    import subprocess
    import sys

    (tmp_path / "custom.yaml").write_text("port: shot_design\n")
    monkeypatch.setenv("SHOT_DESIGN_CONFIG_DIR", str(tmp_path))
    result = subprocess.run(
        [sys.executable, "-c", "from shot_design.config import load_yaml; print(load_yaml('custom.yaml')['port'])"],
        cwd=tmp_path, env=os.environ.copy(), capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "shot_design\n"


def test_default_models_and_corpus_paths_remain_read_only(monkeypatch):
    monkeypatch.delenv("SHOT_DESIGN_PATHS", raising=False)
    monkeypatch.delenv("SHOT_DESIGN_DATA_ROOT", raising=False)
    paths = config.load_paths()
    assert paths.data_root == Path("/scratch/gpfs/EKOLEMEN/nc1514/ideate")
    assert paths.models_dir / "IGNITE" == Path("/scratch/gpfs/EKOLEMEN/nc1514/shot-recommender/models/IGNITE")
    assert paths.shotsummary_raw_dir == Path("/scratch/gpfs/EKOLEMEN/big_d3d_data/foundation_model_text/shotsummary/raw")
    assert paths.sentence_transformers_model == "sentence-transformers/all-MiniLM-L6-v2"


def test_load_yaml_is_cached_until_the_file_changes_and_hands_out_copies(tmp_path, monkeypatch):
    """signals.yaml parses in 24 ms and one query read it dozens of times. The cache key is the
    file's size and mtime, so an edit is seen on the next call; the result is a deep copy, so a
    caller that mutates it cannot change what the next caller sees."""
    import os

    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    p = tmp_path / "x.yaml"
    p.write_text("a: {b: 1}\n")
    assert config.load_yaml("x.yaml") == {"a": {"b": 1}}
    parsed = []
    real = config._load_yaml_file
    monkeypatch.setattr(config, "_load_yaml_file", lambda name: parsed.append(name) or real(name))
    doc = config.load_yaml("x.yaml")
    assert parsed == []  # served from the cache
    doc["a"]["b"] = 99
    assert config.load_yaml("x.yaml") == {"a": {"b": 1}}  # the mutation did not leak
    p.write_text("a: {b: 2}\n")
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))  # a same-second rewrite
    assert config.load_yaml("x.yaml") == {"a": {"b": 2}} and parsed == ["x.yaml"]
