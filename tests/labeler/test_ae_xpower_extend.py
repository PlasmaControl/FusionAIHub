"""The AE extension: population shots, suggestion shards, one merged table."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from labeler.ae.xpower import extend, gallery
from labeler.config import Paths, sha256_of
from labeler.events.catalog.check import CatalogError
from labeler.events.catalog.states import ABSENT, NOT_OBSERVABLE, PRESENT

from . import ae_tree
from .test_catalog_cohort import _tables


def _cohort(paths):
    frame, _ = _tables()
    paths.catalog.mkdir(parents=True, exist_ok=True)
    frame.to_csv(paths.catalog / "cohort.csv", index=False)
    return frame


def _approved_model(paths, split, version="v1"):
    models = ae_tree.chosen(paths, split, version)
    (models / "evaluation.json").write_text(
        json.dumps(
            {
                "bar": {"A1": True, "A2": True, "A3": False, "all": False},
                "meta": {
                    "candidate": "band80-mhd3",
                    "limit": 0,
                    "model_sha256": sha256_of(extend.chosen_model(models)),
                    "chosen_sha256": sha256_of(models / "chosen.json"),
                },
            }
        )
    )
    return models


@pytest.mark.parametrize("change", ["choice", "weights", "pilot", "legacy"])
@pytest.mark.parametrize("merging", [False, True])
def test_evaluation_must_name_the_current_full_model(
    tmp_path, monkeypatch, capsys, change, merging
):
    paths = ae_tree.build(tmp_path, {101: "train"})
    models = _approved_model(paths, {101: "train"})
    evaluation = models / "evaluation.json"
    if change == "choice":
        (models / "chosen.json").write_text('{"candidate": "band0-mhd3"}')
    elif change == "weights":
        extend.chosen_model(models).write_bytes(b"changed weights")
    else:
        record = json.loads(evaluation.read_text())
        if change == "pilot":
            record["meta"]["limit"] = 1
        else:
            del record["meta"]["model_sha256"]
        evaluation.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="evaluation.json"):
        if merging:
            extend.merge(paths, models=models, of=1)
        else:
            extend.run_shard(paths, models=models, k=0, of=1)
    ae_tree.env(monkeypatch, paths)
    with pytest.raises(SystemExit) as error:
        extend.main(["--models", str(models), *(["--merge"] if merging else [])])
    assert error.value.code != 0
    stderr = capsys.readouterr().err
    assert str(evaluation) in stderr and "Traceback" not in stderr


def test_frames_the_rows_miss_are_not_observable():
    states = extend.frame_states([0.9, 0.1, 0.9], [True, True, False], 0.5)
    assert states.tolist() == [PRESENT, ABSENT, NOT_OBSERVABLE]


def test_the_extension_takes_population_shots_with_two_seconds_of_co2(tmp_path):
    path = tmp_path / "population.csv"
    pd.DataFrame(
        {
            "shot": [3, 1, 2, 4],
            "year": [2024] * 4,
            "span_co2_s": [2.5, 5.0, 1.0, None],
            "window_start_ms": [0, 100, 0, 0],
            "window_end_ms": [2000, 3000, 2000, 2000],
        }
    ).to_csv(path, index=False)
    shots = extend.population_shots(path)
    assert shots.shot.tolist() == [1, 3]
    assert shots.window_start_ms.tolist() == [100, 0]


def test_the_extension_runs_in_shards_and_merges_into_one_table(tmp_path):
    paths = ae_tree.build(tmp_path, {101: "train"})
    models = _approved_model(paths, {101: "train"})
    ae_tree.corpus(tmp_path, (201, 202, 203))
    _cohort(paths)
    pd.DataFrame(
        {
            "shot": [201, 202, 203],
            "year": [2024, 2025, 2025],
            "span_co2_s": [2.5] * 3,
            "window_start_ms": [0, 0, 0],
            "window_end_ms": [600, 600, 600],
        }
    ).to_csv(paths.catalog / "population.csv", index=False)
    for k in (0, 1):
        result = extend.run_shard(
            paths, models=models, k=k, of=2, workers=1, pictures=k == 0
        )
        assert result["failed"] == 0
    merged = extend.merge(paths, models=models, of=2)
    assert merged["shots"] == 3 and merged["failed"] == 0 and merged["pictures"] == 2
    table = pd.read_csv(merged["table"], keep_default_na=False)
    assert sorted(table.shot.unique()) == [201, 202, 203]
    for _, rows in table.groupby("shot"):
        assert rows.t_start.min() == 0 and rows.t_end.max() == 600
        assert (rows.t_start.to_numpy()[1:] == rows.t_end.to_numpy()[:-1]).all()
    meta_path = Path(merged["table"]).with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text())
    assert meta["tier"] == "suggestions" and meta["candidate"] == "band80-mhd3"
    assert meta["bar"] == {"A1": True, "A2": True, "A3": False, "all": False}
    summary = pd.read_csv(extend.suggestions_dir(paths) / "summary.csv")
    assert summary.frames.tolist() == [60, 60, 60]
    with np.load(extend.suggestions_dir(paths) / "shards" / "0.npz") as z:
        assert z["p201"].shape == (60,) and int(z["f201"]) == 0
    index = pd.read_csv(gallery.gallery_dir(paths) / "index.csv")
    assert index[index.group == "extension"].shot.tolist() == [201, 203]


def test_blind_shots_are_excluded_before_sharding(tmp_path):
    paths = ae_tree.build(tmp_path, {101: "train"})
    models = _approved_model(paths, {101: "train"})
    cohort = _cohort(paths)
    blind = int(cohort.loc[cohort.blind, "shot"].min())
    eligible = sorted(cohort.loc[~cohort.blind & (cohort.shot > blind), "shot"])[:2]
    eligible.append(int(cohort.shot.max()) + 1)  # Outside the cohort is eligible.
    shots = [blind, *eligible]
    ae_tree.corpus(tmp_path, shots)
    pd.DataFrame(
        {
            "shot": shots,
            "year": [2024] * len(shots),
            "span_co2_s": [2.5] * len(shots),
            "window_start_ms": [0] * len(shots),
            "window_end_ms": [600] * len(shots),
        }
    ).to_csv(paths.catalog / "population.csv", index=False)
    results = [
        extend.run_shard(paths, models=models, k=k, of=2, workers=1) for k in (0, 1)
    ]
    merged = extend.merge(paths, models=models, of=2)
    table = pd.read_csv(merged["table"])
    assert sorted(table.shot.unique()) == eligible
    root = gallery.gallery_dir(paths)
    assert {int(p.stem) for p in (root / "extension").glob("*.jpg")} == set(eligible)
    assert pd.read_csv(root / "index.csv").shot.tolist() == eligible
    assert merged["shots"] == merged["pictures"] == 3 and merged["failed"] == 0
    shards = extend.suggestions_dir(paths) / "shards"
    for k, result in enumerate(results):
        assert result["shots"] == result["done"] == len(eligible[k::2])
        assert result["failed"] == 0
        assert pd.read_csv(shards / f"{k}.summary.csv").shot.tolist() == eligible[k::2]
        with np.load(shards / f"{k}.npz") as probabilities:
            assert f"p{blind}" not in probabilities and f"f{blind}" not in probabilities


@pytest.mark.parametrize(
    "problem", ["missing", "directory", "schema", "ragged", "window", "encoding"]
)
def test_an_unreadable_cohort_refuses_with_its_path(
    tmp_path, monkeypatch, capsys, problem
):
    paths = ae_tree.build(tmp_path, {101: "train"})
    models = _approved_model(paths, {101: "train"})
    cohort = _cohort(paths)
    pd.DataFrame(
        {
            "shot": [201],
            "year": [2024],
            "span_co2_s": [2.5],
            "window_start_ms": [0],
            "window_end_ms": [600],
        }
    ).to_csv(paths.catalog / "population.csv", index=False)
    path = paths.catalog / "cohort.csv"
    if problem in ("missing", "directory"):
        path.unlink()
        if problem == "directory":
            path.mkdir()
    elif problem == "schema":
        cohort.drop(columns="blind").to_csv(path, index=False)
    elif problem == "ragged":
        with path.open("a") as stream:
            stream.write("201,False\n")
    elif problem == "window":
        cohort["window_start_ms"] = "invalid"
        cohort.to_csv(path, index=False)
    else:
        path.write_bytes(b"\xff")

    def refuse_work(*args, **kwargs):
        pytest.fail("workers started before the cohort was read")

    monkeypatch.setattr(extend, "run_all", refuse_work)
    with pytest.raises((CatalogError, FileNotFoundError)) as error:
        extend.run_shard(paths, models=models, k=0, of=1)
    assert str(path) in str(error.value)
    ae_tree.env(monkeypatch, paths)
    with pytest.raises(SystemExit) as exit_code:
        extend.main(["--workers", "1"])
    assert exit_code.value.code != 0
    stderr = capsys.readouterr().err
    assert str(path) in stderr and "Traceback" not in stderr
    assert not extend.suggestions_dir(paths).exists()
    assert not gallery.gallery_dir(paths).exists()


def test_a_merge_refuses_a_missing_shard(tmp_path):
    paths = ae_tree.build(tmp_path, {101: "train"})
    models = _approved_model(paths, {101: "train"})
    try:
        extend.merge(paths, models=models, of=2)
    except FileNotFoundError as error:
        assert "shards [0, 1] of 2" in str(error)
    else:
        raise AssertionError("merge ran without its shards")


@pytest.mark.parametrize("merge", [False, True])
@pytest.mark.parametrize(
    "evaluation",
    [
        None,
        {"bar": {"A1": False, "A2": True}},
        {"bar": {"A1": True, "A2": False}},
        {"bar": {"A1": True}},
        {"bar": {"A1": "true", "A2": True}},
        {"bar": {"A1": True, "A2": 1}},
        {"bar": None},
        {},
        [],
        "malformed JSON",
    ],
)
def test_extension_requires_a_passing_evaluation(
    tmp_path, monkeypatch, capsys, merge, evaluation
):
    paths = Paths(
        root=tmp_path / "root",
        label_tables=tmp_path / "events",
        corpus=tmp_path / "corpus",
    )
    models = tmp_path / "models"
    models.mkdir()
    path = models / "evaluation.json"
    if evaluation is not None:
        path.write_text(
            evaluation if isinstance(evaluation, str) else json.dumps(evaluation)
        )
    # There are no cohort, population or model inputs: refuse before reading them.
    with pytest.raises(ValueError, match="evaluation.json"):
        if merge:
            extend.merge(paths, models=models, of=1)
        else:
            extend.run_shard(paths, models=models, k=0, of=1)
    ae_tree.env(monkeypatch, paths)
    with pytest.raises(SystemExit) as exit_code:
        extend.main(["--models", str(models), *(["--merge"] if merge else [])])
    assert exit_code.value.code != 0
    stderr = capsys.readouterr().err
    assert str(path) in stderr and "Traceback" not in stderr
    assert not extend.suggestions_dir(paths).exists()
    assert not gallery.gallery_dir(paths).exists()


def _three_shots(tmp_path, monkeypatch, version="v1"):
    paths = ae_tree.build(tmp_path, {101: "train"})
    models = _approved_model(paths, {101: "train"}, version)
    _cohort(paths)
    pd.DataFrame(
        {
            "shot": [201, 202, 203],
            "year": [2024] * 3,
            "span_co2_s": [2.5] * 3,
            "window_start_ms": [0] * 3,
            "window_end_ms": [600] * 3,
        }
    ).to_csv(paths.catalog / "population.csv", index=False)

    def work(fn, jobs, workers, init, initargs):
        for shot, year, lo, hi in jobs:
            yield (
                (shot, year, lo, hi),
                {
                    "rows": [[shot, 1, lo, hi, 0.9]],
                    "prob": np.full(60, 0.9, dtype=np.float16),
                    "first": 0,
                    "summary": dict(
                        zip(
                            extend.SUMMARY_COLUMNS,
                            [shot, year, lo, hi, 60, 60, 0, 1, 0.9],
                        )
                    ),
                },
            )

    monkeypatch.setattr(extend, "run_all", work)
    return paths, models


def test_a_pilot_cannot_stand_in_for_a_full_shard(tmp_path, monkeypatch):
    paths, models = _three_shots(tmp_path, monkeypatch)
    extend.run_shard(paths, models=models, k=0, of=2, limit=1, pictures=False)
    extend.run_shard(paths, models=models, k=1, of=2, pictures=False)
    with pytest.raises((ValueError, FileNotFoundError), match="shard"):
        extend.merge(paths, models=models, of=2)
    shards = extend.suggestions_dir(paths) / "shards"
    assert not (shards / "0.csv").exists()
    assert (shards / "pilot/0.csv").is_file()


@pytest.mark.parametrize(
    "change",
    ["model", "population", "cohort", "of", "missing", "duplicate", "given"],
)
def test_merge_checks_identity_and_exact_shot_accounting(
    tmp_path, monkeypatch, capsys, change
):
    paths, models = _three_shots(tmp_path, monkeypatch)
    for k in (0, 1):
        extend.run_shard(paths, models=models, k=k, of=2, pictures=False)
    shards = extend.suggestions_dir(paths) / "shards"
    manifest = shards / "0.json"
    assert manifest.is_file()
    record = json.loads(manifest.read_text())
    assert record["shots"] == [201, 203]
    assert record["model_sha256"] == sha256_of(extend.chosen_model(models))
    assert record["evaluation_sha256"] == sha256_of(models / "evaluation.json")
    assert record["git_sha"] and record["made_at"]
    if change in ("population", "cohort"):
        path = paths.catalog / f"{change}.csv"
        path.write_text(path.read_text() + "\n")
    elif change in ("missing", "duplicate"):
        summary = shards / "0.summary.csv"
        frame = pd.read_csv(summary)
        frame = frame.iloc[:1] if change == "missing" else pd.concat([frame, frame])
        frame.to_csv(summary, index=False)
    else:
        key = {"model": "model_sha256", "of": "of", "given": "shots"}[change]
        record[key] = {"model": "other model", "of": 3, "given": [201]}[change]
        manifest.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="0.json"):
        extend.merge(paths, models=models, of=2)
    ae_tree.env(monkeypatch, paths)
    with pytest.raises(SystemExit) as error:
        extend.main(["--merge", "--of", "2"])
    assert error.value.code != 0
    stderr = capsys.readouterr().err
    assert str(manifest) in stderr and "Traceback" not in stderr


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_the_extension_names_the_labels_it_learned_from(tmp_path, monkeypatch, version):
    """The reviewer's Minor 11: the table's meta names the labels the model was
    trained on and the version's label snapshot (v1 has none), and so does each
    of its gallery index rows."""
    paths, models = _three_shots(tmp_path, monkeypatch, version)
    snapshot = ae_tree.snapshot(paths, monkeypatch) if version == "v2" else None
    monkeypatch.setattr(extend, "run_all", gallery.run_all)
    ae_tree.corpus(tmp_path, (201, 202, 203))
    result = extend.run_shard(paths, models=models, k=0, of=1, version=version)
    assert result["failed"] == 0
    merged = extend.merge(paths, models=models, of=1, version=version)
    meta_path = Path(merged["table"]).with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text())
    _, blob = extend.load(extend.chosen_model(models))
    assert meta["labels_sha256"] == blob["labels_sha256"]
    assert meta["snapshot_sha256"] == snapshot
    index = pd.read_csv(
        gallery.gallery_dir(paths, version) / "index.csv", keep_default_na=False
    )
    assert index.snapshot_sha256.tolist() == [snapshot or ""] * 3
