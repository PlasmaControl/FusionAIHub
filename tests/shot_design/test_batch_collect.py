"""Simulation summaries require the recorded frame origin."""

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
