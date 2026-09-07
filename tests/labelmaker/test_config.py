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


def test_event_paths_and_mkdirs(tmp_path):
    p = Paths(root=tmp_path, corpus=tmp_path / "corpus")
    assert p.events == tmp_path / "events"
    assert p.masks == tmp_path / "masks"
    assert p.annotate == tmp_path / "annotate"
    assert p.events_file(190000) == tmp_path / "events" / "190000_events.parquet"
    assert p.masks_file(190000) == tmp_path / "masks" / "190000_masks.npz"
    assert p.events_index == tmp_path / "events_index.parquet"
    p.mkdirs()
    for sub in ("events", "masks", "annotate"):
        assert (tmp_path / sub).is_dir()


def test_the_text_root_is_a_third_input_root(monkeypatch, tmp_path):
    # The operator-text bundles: read-only, a different group's directory,
    # and named `shot_<N>.txt` rather than `<N>_something`.
    assert Paths().text_root == Path(
        "/scratch/gpfs/EKOLEMEN/big_d3d_data/foundation_model_text"
        "/shotsummary/processed/per_shot_txt"
    )
    monkeypatch.setenv("LABELMAKER_TEXT_ROOT", str(tmp_path / "text"))
    assert Paths.from_env().text_root == tmp_path / "text"
    p = Paths(root=tmp_path, corpus=tmp_path, text_root=tmp_path / "text")
    assert p.text_file(198658) == tmp_path / "text" / "shot_198658.txt"
    # It is an input, so `mkdirs` does not create it - as with the corpus.
    p.mkdirs()
    assert not (tmp_path / "text").exists()


def test_git_sha_is_a_string():
    sha = git_sha()
    assert isinstance(sha, str) and sha
