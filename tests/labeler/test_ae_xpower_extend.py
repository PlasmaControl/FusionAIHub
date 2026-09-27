"""The AE extension: population shots, suggestion shards, one merged table."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from labeler.ae.xpower import extend, gallery
from labeler.events.catalog.states import ABSENT, NOT_OBSERVABLE, PRESENT

from . import ae_tree


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
    models = ae_tree.chosen(paths, {101: "train"})
    ae_tree.corpus(tmp_path, (201, 202, 203))
    paths.catalog.mkdir(parents=True)
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
    summary = pd.read_csv(extend.suggestions_dir(paths) / "summary.csv")
    assert summary.frames.tolist() == [60, 60, 60]
    with np.load(extend.suggestions_dir(paths) / "shards" / "0.npz") as z:
        assert z["p201"].shape == (60,) and int(z["f201"]) == 0
    index = pd.read_csv(gallery.gallery_dir(paths) / "index.csv")
    assert index[index.group == "extension"].shot.tolist() == [201, 203]


def test_a_merge_refuses_a_missing_shard(tmp_path):
    paths = ae_tree.build(tmp_path, {101: "train"})
    models = ae_tree.chosen(paths, {101: "train"})
    try:
        extend.merge(paths, models=models, of=2)
    except FileNotFoundError as error:
        assert "shards [0, 1] of 2" in str(error)
    else:
        raise AssertionError("merge ran without its shards")
