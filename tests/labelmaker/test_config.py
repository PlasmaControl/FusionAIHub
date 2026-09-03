"""Paths resolve from the environment and nothing else hard-codes a root."""
from pathlib import Path

from labelmaker.config import Paths, git_sha


def test_default_root_is_group_storage():
    p = Paths()
    assert p.root == Path("/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")
    assert p.corpus == Path("/scratch/gpfs/EKOLEMEN/foundation_model")


def test_from_env_overrides_both_roots(monkeypatch, tmp_path):
    monkeypatch.setenv("LABELMAKER_ROOT", str(tmp_path / "out"))
    monkeypatch.setenv("LABELMAKER_CORPUS", str(tmp_path / "corpus"))
    p = Paths.from_env()
    assert p.root == tmp_path / "out"
    assert p.corpus == tmp_path / "corpus"


def test_per_shot_paths_and_mkdirs(tmp_path):
    p = Paths(root=tmp_path, corpus=tmp_path / "corpus")
    assert p.features_file(190000) == tmp_path / "features" / "190000_features.h5"
    assert p.labels_file(190000) == tmp_path / "labels" / "190000_labels.h5"
    assert p.corpus_file(190000) == tmp_path / "corpus" / "190000_processed.h5"
    assert p.labels_index == tmp_path / "labels_index.parquet"
    p.mkdirs()
    for sub in ("features", "labels", "models", "runs", "validation"):
        assert (tmp_path / sub).is_dir()
    p.mkdirs()  # idempotent


def test_git_sha_is_a_string():
    sha = git_sha()
    assert isinstance(sha, str) and sha
