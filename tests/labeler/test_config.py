"""Paths resolve from the environment and nothing else hard-codes a root."""

from pathlib import Path

import pandas as pd
import pytest

from labeler import config
from labeler.config import Paths, git_sha
from labeler.events.catalog.points import validate_csv_fields


def test_default_root_is_group_storage():
    p = Paths()
    assert p.root == Path("/scratch/gpfs/EKOLEMEN/nc1514/labeler")
    assert p.corpus == Path("/scratch/gpfs/EKOLEMEN/foundation_model")


def test_from_env_overrides_both_roots(monkeypatch, tmp_path):
    monkeypatch.setenv("LABELER_ROOT", str(tmp_path / "out"))
    monkeypatch.setenv("LABELER_CORPUS", str(tmp_path / "corpus"))
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
    # Not `tmp_path / "text"`: that is `text_cache`, which labeler owns
    # and `mkdirs` does create.
    bundles = tmp_path / "bundles"
    monkeypatch.setenv("LABELER_TEXT_ROOT", str(bundles))
    assert Paths.from_env().text_root == bundles
    p = Paths(root=tmp_path, corpus=tmp_path, text_root=bundles)
    assert p.text_file(198658) == bundles / "shot_198658.txt"
    # It is an input, so `mkdirs` does not create it - as with the corpus.
    p.mkdirs()
    assert not bundles.exists()


def test_git_sha_is_a_string():
    sha = git_sha()
    assert isinstance(sha, str) and sha


def test_the_logbook_jsonl_is_a_read_only_file_and_the_cache_is_ours(
    monkeypatch,
    tmp_path,
):
    # The shot-scope text source: one 616 MB file with a JSON record per
    # line, read-only like the corpus. What labeler writes is the SUBSET
    # of it for the shots in hand, under our own root, so the cache is a
    # thing we own and can delete and the source is a thing we never touch.
    assert Paths().logs_jsonl == Path(
        "/scratch/gpfs/EKOLEMEN/big_d3d_data/foundation_model_text/sql/logs.jsonl"
    )
    monkeypatch.setenv("LABELER_LOGS_JSONL", str(tmp_path / "logs.jsonl"))
    assert Paths.from_env().logs_jsonl == tmp_path / "logs.jsonl"
    p = Paths(root=tmp_path)
    assert p.text_cache == tmp_path / "text"
    assert p.logs_subset == tmp_path / "text" / "logs_subset.jsonl"
    # Beside the subset, the shots the source was searched for and did not
    # have. Without it a record-less shot is searched again on every call,
    # which is a full pass over the 616 MB source for nothing.
    assert p.logs_subset_missing == tmp_path / "text" / "logs_subset.missing"
    p.mkdirs()
    assert p.text_cache.is_dir()


def test_the_label_tables_root_is_the_repo_and_is_overridable(
    monkeypatch,
    tmp_path,
):
    # Curated label tables are DATA and live under `data/events/`, never
    # under `src/`, and git ignores them (a clone restores them from the
    # Hugging Face release). The default resolves relative to the package
    # rather than to the caller's cwd, so a run from anywhere finds them; a
    # table too large or too restricted to keep here lives on /scratch and
    # the root moves with it.
    default = Paths().label_tables
    assert default.name == "events" and default.parent.name == "data"
    assert default.parent.parent == Paths().config_dir.parents[1]
    tables = tmp_path / "elsewhere"
    monkeypatch.setenv("LABELER_LABEL_TABLES", str(tables))
    assert Paths.from_env().label_tables == tables
    # An input, like the corpus and the bundles: `mkdirs` does not make it.
    Paths(root=tmp_path, label_tables=tables).mkdirs()
    assert not tables.exists()


def test_the_manifests_are_tracked_config_and_overridable(monkeypatch, tmp_path):
    # events.yaml and best_models.yaml are source: configs/labeler/, beside the
    # package, never in the git-ignored label tables.
    paths = Paths()
    assert paths.config_dir.parts[-2:] == ("configs", "labeler")
    assert paths.events_manifest == paths.config_dir / "events.yaml"
    assert paths.best_models == paths.config_dir / "best_models.yaml"
    assert paths.events_manifest.is_file() and paths.best_models.is_file()
    monkeypatch.setenv("LABELER_CONFIG_DIR", str(tmp_path))
    assert Paths.from_env().events_manifest == tmp_path / "events.yaml"
    assert Paths.from_env().label_tables == paths.label_tables


def test_raw_cache_follows_the_root(tmp_path):
    # Beside the 181 prefetched AE shots, on group storage, not in the repository.
    assert Paths().raw_cache == Paths().root / "raw"
    assert Paths(root=tmp_path).raw_cache == tmp_path / "raw"


def test_spectrograms_live_under_the_root(tmp_path):
    paths = Paths(root=tmp_path)
    assert paths.spectrograms == tmp_path / "spectrograms"
    assert paths.spectrogram_file("alfven_eigenmode", 170790) == (
        tmp_path / "spectrograms" / "alfven_eigenmode" / "170790.h5"
    )


def test_confinement_reuses_the_h_mode_stores(tmp_path):
    paths = Paths(root=tmp_path)
    assert paths.spectrogram_file("confinement", 186636) == (
        paths.spectrogram_file("high_confinement_mode", 186636)
    )


def test_raw_cache_honours_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("LABELER_RAW_CACHE", str(tmp_path / "elsewhere"))
    assert Paths.from_env().raw_cache == tmp_path / "elsewhere"


def test_named_bytes_reads_snapshot_and_names_source(tmp_path):
    path = tmp_path / "snapshot.csv"
    data = b"snapshot\x00\xff\n"
    source = config.NamedBytes(data, path)
    assert source.read() == data
    assert source.name == str(path)
    assert str(source) == str(path)


def test_named_bytes_reader_refusal_names_source(tmp_path):
    path = tmp_path / "ragged.csv"
    source = config.NamedBytes(b"shot,time\n185601,1,extra\n", path)
    with pytest.raises(pd.errors.ParserError) as exc:
        validate_csv_fields(source)
    error = str(exc.value)
    assert "row 2: expected 2 fields, got 3" in error
    assert str(path) in error
    assert "BytesIO" not in error
