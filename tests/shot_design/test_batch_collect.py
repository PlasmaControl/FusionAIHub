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


@pytest.mark.parametrize("with_real", [False, True])
def test_collect_staticness_from_report_columns(tmp_path, with_real):
    sim = tmp_path / "outputs" / "abc" / "simulation"
    sim.mkdir(parents=True)
    header = "| modality | frac_static | token_acc | persistence_acc | skill | divergence_vs_real |"
    row = "| mse | 0.25 | 0.8 | 0.5 | 0.3 | 0.1 |"
    if with_real:
        header += " frac_static_real |"
        row += " 1.0 |"
    (sim / "report.md").write_text(header + "\n" + row + "\n")
    collected = _design("abc", tmp_path)
    assert collected["mse.frac_static"] == 0.25
    assert collected["mse.token_acc"] == 0.8
    if with_real:
        assert collected["mse.frac_static_real"] == 1.0
