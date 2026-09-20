"""``shot_design.simulate.report`` -- simulation.h5, panels, and the honest report.md.

The report table's ``frac_static`` column and the qualitative-checkpoint sentence
are the two pieces the D2 brief left to controller ruling; see ``frac_static`` and
``QUALITATIVE_SENTENCE`` in ``report.py`` for the exact contract each implements.
"""

import h5py
import numpy as np
import torch

from shot_design.simulate import core, report


def _arms():
    F = 5
    z = lambda: torch.randint(0, 16, (F, 2), dtype=torch.int32)  # noqa: E731
    return core.SimulationArms(
        seed_frames=2,
        predict_frames=3,
        real={"mse": z()},
        proposed={"mse": z()},
        gt={"mse": z()},
        divergence_vs_real={"mse": 0.5},
        token_accuracy={"mse": 0.4},
        persistence_accuracy={"mse": 0.6},
    )


def _decoded():
    return {
        "mse": {
            k: np.random.rand(5, 3).astype(np.float32)
            for k in ("real", "proposed", "gt")
        }
    }


def test_report_writes_h5_panels_and_markdown(tmp_path):
    arms = _arms()
    decoded = _decoded()
    out = report.write(
        tmp_path,
        arms,
        decoded,
        {
            "design_id": "abc",
            "bundle_manifest_sha256": "0" * 64,
            "codec_generation": "v4",
            "window_s": [1.0, 5.0],
            "dynamics_step": 3200,
        },
    )
    assert out is not None
    with h5py.File(tmp_path / "simulation.h5") as f:
        assert f["decoded/mse/proposed"].shape == (5, 3)
        assert f["tokens/real/mse"].shape == (5, 2)
        assert f.attrs["codec_generation"] == "v4"
    assert (tmp_path / "panels" / "mse.png").exists()
    md = (tmp_path / "report.md").read_text()
    assert "| mse |" in md and "-0.20" in md and "qualitative" in md


def test_report_writes_actuators_when_given(tmp_path):
    arms = _arms()
    decoded = _decoded()
    actuators = {
        "real": np.zeros((5, 4), dtype=np.float32),
        "proposed": np.ones((5, 4), dtype=np.float32),
    }
    report.write(tmp_path, arms, decoded, {"dynamics_step": 1}, actuators=actuators)
    with h5py.File(tmp_path / "simulation.h5") as f:
        assert f["actuators/real"].shape == (5, 4)
        assert f["actuators/proposed"].shape == (5, 4)
        np.testing.assert_array_equal(f["actuators/proposed"][:], actuators["proposed"])


def test_report_skips_actuators_group_when_not_given(tmp_path):
    arms = _arms()
    decoded = _decoded()
    report.write(tmp_path, arms, decoded, {"dynamics_step": 1})
    with h5py.File(tmp_path / "simulation.h5") as f:
        assert "actuators" not in f


def test_report_no_qualitative_sentence_when_skill_nonnegative(tmp_path):
    arms = _arms()
    arms.token_accuracy["mse"] = 0.9
    arms.persistence_accuracy["mse"] = 0.6  # skill = 0.3 >= 0
    decoded = _decoded()
    report.write(tmp_path, arms, decoded, {"dynamics_step": 42})
    md = (tmp_path / "report.md").read_text()
    assert "qualitative" not in md


def test_frac_static_all_constant_is_one():
    codes = {"mse": torch.zeros(5, 3, dtype=torch.int64)}
    frac = report.frac_static(codes, seed_frames=2)
    assert frac["mse"] == 1.0


def test_frac_static_all_different_is_zero():
    # every predicted frame's tokens differ from the previous frame's, everywhere.
    base = torch.arange(3, dtype=torch.int64).unsqueeze(0).repeat(5, 1)  # (5, 3)
    # alternate 0/1 per frame
    parity = torch.arange(5, dtype=torch.int64).unsqueeze(1) % 2
    codes = {"mse": base + parity * 100}
    frac = report.frac_static(codes, seed_frames=2)
    assert frac["mse"] == 0.0
