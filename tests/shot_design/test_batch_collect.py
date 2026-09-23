"""A batch summary: the recorded frame origin and the scores from metrics.json."""

import json
import runpy
from pathlib import Path

import h5py
import pytest

_design = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "scripts/shot_design/batch_collect.py")
)["_design"]


def test_collect_records_origin_without_requiring_generation(tmp_path):
    sim = tmp_path / "outputs" / "abc" / "simulation"
    sim.mkdir(parents=True)
    with h5py.File(sim / "simulation.h5", "w") as f:
        f.attrs["frame_origin_s"] = 1.0
    row = _design("abc", tmp_path)
    assert row["frame_origin_s"] == 1.0
    assert row["codec_generation"] is None


def test_collect_requires_recorded_origin(tmp_path):
    sim = tmp_path / "outputs" / "abc" / "simulation"
    sim.mkdir(parents=True)
    with h5py.File(sim / "simulation.h5", "w") as f:
        f.attrs["codec_generation"] = "v4"
    with pytest.raises(KeyError, match="frame_origin_s"):
        _design("abc", tmp_path)


def test_collect_reads_the_scores_from_metrics_json(tmp_path):
    sim = tmp_path / "outputs" / "abc" / "simulation"
    sim.mkdir(parents=True)
    mse = {
        "family": "slowts", "feature": "intra-frame mean",
        "nrmse": {"real": 0.8, "persistence": 1.1, "seed_mean": 1.3},
        "crps": {"real": 0.2, "persistence": 0.3}, "skill": 0.33, "spread_error": None,
        "effect": 0.5, "noise": 0.1, "effect_to_noise": 5.0, "resolved": True,
    }  # fmt: skip
    doc = {"modalities": {"mse": mse, "co2": {"held": True}}}
    (sim / "metrics.json").write_text(json.dumps(doc))
    row = _design("abc", tmp_path)
    assert row["mse.skill"] == 0.33 and row["mse.nrmse_persistence"] == 1.1
    assert row["mse.spread_error"] is None and row["mse.resolved"] is True
    assert not any(k.startswith("co2.") for k in row)
