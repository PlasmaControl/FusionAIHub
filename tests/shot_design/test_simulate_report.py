"""``shot_design.simulate.report``: simulation.h5, metrics.json, report.md and panels."""

import json

import h5py
import numpy as np
import pytest
import torch

from shot_design.simulate import core, report

F, K0, M = 6, 2, 4
FAMILIES = {"mse": "slowts", "co2": "spectro"}
META = {
    "design_id": "abc",
    "codec_generation": "v4",
    "dynamics_step": 3200,
    "window_s": [2.0, 6.0],
    "frame_origin_s": 1.0,
    "frame_s": 0.05,
    "t0_s": 2.0,
    "k0": K0,
    "n_predict": F - K0,
    "members": M,
    "decode_steps": 10,
    "temperature": 1.0,
    "precision": "fp32",
}


def _ensemble(held=("co2",)):
    g = torch.Generator().manual_seed(0)

    def tokens(*shape):
        return torch.randint(0, 16, (*shape, F, 2), generator=g)

    arms = ("real", "proposed", "null")
    return core.Ensemble(
        k0=K0,
        gt={"mse": tokens(), "co2": tokens()},
        arms={arm: {"mse": tokens(M), "co2": tokens(M)} for arm in arms},
        seeds={"real": 0, "proposed": 0, "null": M},
        held=held,
        batch=2,
    )


def _feats(edit=0.0, error=0.0):
    """A measured ramp; the real arm near it (``error`` off), the proposed arm ``edit``
    from the real one, the null arm the real actuators on other random numbers."""
    rng = np.random.default_rng(0)
    gt = np.repeat(np.linspace(0.0, 1.0, F)[:, None], 3, axis=1)

    def arm(offset):
        return (gt + offset + rng.normal(0.0, 0.05, (M, F, 3))).astype(np.float32)

    real = arm(error)
    per = {"gt": gt.astype(np.float32), "real": real, "null": arm(error)}
    return {"mse": per | {"proposed": (real + edit).astype(np.float32)}}


def test_the_four_outputs_come_from_one_ensemble(tmp_path):
    actuators = {"real": torch.zeros(F, 5), "proposed": torch.ones(F, 5)}
    doc = report.write(tmp_path, _ensemble(), _feats(), FAMILIES, META, actuators)
    with h5py.File(tmp_path / "simulation.h5") as f:
        assert f["tokens/gt/mse"].shape == (F, 2)
        assert f["tokens/real/mse"].shape == (M, F, 2)
        assert f["tokens/null/co2"].dtype == np.int16
        assert f["features/mse/gt"].shape == (F, 3)
        assert f["features/mse/proposed"].shape == (M, F, 3)
        assert "features/co2" not in f
        np.testing.assert_array_equal(f["actuators/proposed"][:], np.ones((F, 5)))
        assert f.attrs["codec_generation"] == "v4" and f.attrs["t0_s"] == 2.0
        assert list(f.attrs["held"]) == ["co2"] and f.attrs["batch"] == 2
        assert json.loads(f.attrs["seeds"]) == {"real": 0, "proposed": 0, "null": M}
        assert "mean z" in f.attrs["reduction"]
    assert json.loads((tmp_path / "metrics.json").read_text()) == doc
    assert (tmp_path / "panels" / "mse.png").stat().st_size > 0
    assert not (tmp_path / "panels" / "co2.png").exists()
    md = (tmp_path / "report.md").read_text()
    assert "| mse, intra-frame mean |" in md and "held at the placeholder: co2" in md


def test_metrics_follow_the_schema(tmp_path):
    doc = report.write(tmp_path, _ensemble(), _feats(), FAMILIES, META)
    assert set(doc) == {
        "schema", "members", "k0", "n_predict", "frame_s", "t0_s", "decode_steps",
        "temperature", "held", "modalities",
    }  # fmt: skip
    assert doc["schema"] == "shot-design-simulation-metrics-v1"
    assert (doc["members"], doc["k0"], doc["n_predict"]) == (M, K0, F - K0)
    assert doc["held"] == ["co2"] and doc["modalities"]["co2"] == {"held": True}
    mse = doc["modalities"]["mse"]
    assert set(mse) == {
        "family", "feature", "nrmse", "crps", "skill", "spread_error", "effect",
        "noise", "effect_to_noise", "resolved",
    }  # fmt: skip
    assert set(mse["nrmse"]) == {"real", "persistence", "seed_mean"}
    assert (mse["family"], mse["feature"]) == ("slowts", "intra-frame mean")


def test_an_edit_well_beyond_the_noise_floor_is_resolved_and_a_small_one_is_not(
    tmp_path,
):
    big = report.write(tmp_path / "big", _ensemble(), _feats(edit=1.0), FAMILIES, META)
    assert big["modalities"]["mse"]["resolved"] is True
    assert "(resolved)" in (tmp_path / "big" / "report.md").read_text()
    small = report.write(tmp_path / "small", _ensemble(), _feats(), FAMILIES, META)
    assert small["modalities"]["mse"]["resolved"] is False


def test_the_report_says_which_signals_do_worse_than_persistence(tmp_path):
    doc = report.write(tmp_path, _ensemble(), _feats(error=3.0), FAMILIES, META)
    assert doc["modalities"]["mse"]["skill"] < 0
    assert "Worse than persistence: mse." in (tmp_path / "report.md").read_text()


def test_undefined_scores_are_null_never_nan(tmp_path):
    feats = _feats()
    feats["mse"]["gt"][:] = 0.5  # a flat measurement: no variance, a perfect persistence
    report.write(tmp_path, _ensemble(held=()), feats, FAMILIES, META)
    text = (tmp_path / "metrics.json").read_text()
    doc = json.loads(text, parse_constant=pytest.fail)
    assert doc["modalities"]["mse"]["nrmse"]["real"] is None
    assert doc["modalities"]["mse"]["skill"] is None
    assert doc["held"] == [] and "co2" not in doc["modalities"]


def test_actuators_are_optional(tmp_path):
    report.write(tmp_path, _ensemble(), _feats(), FAMILIES, META)
    with h5py.File(tmp_path / "simulation.h5") as f:
        assert "actuators" not in f
