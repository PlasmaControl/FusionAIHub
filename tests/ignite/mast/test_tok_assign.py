"""Settings, ladders and the selection rules (pure functions of numbers)."""

import math

import pytest

from tokamak_foundation_model.ignite.mast import tasks
from tokamak_foundation_model.ignite.mast.tokenizers import assign

IP = tasks.SignalSpec("summary-ip", 0.00025, (1,), 3.5e5, 3.0e5)
KAPPA = tasks.SignalSpec("equilibrium-elongation", 0.005, (1,), 1.6, 0.3)
LCFS = tasks.SignalSpec("equilibrium-lcfs_r", 0.005, (170,), 0.9, 0.4)
PSI = tasks.SignalSpec("equilibrium-psi", 0.005, (65, 65), 0.0, 0.1)
MIRNOV = tasks.SignalSpec(tasks.SPECTROGRAM_SIGNALS[1], 0.001024, (3, 256), 5.5, 266.0)


def test_the_kind_and_the_domains_of_a_signal():
    assert [assign.codec_kind(s) for s in (IP, KAPPA, LCFS, PSI, MIRNOV)] == [
        "frame_dct", "frame_dct", "linear", "linear", "band_dct"]
    assert assign.domains(IP, "none") == ("linear",) and assign.domains(IP, "asinh") == ("linear", "compressed")
    assert assign.domains(LCFS, "none") == ("signal", "linear")
    assert assign.domains(LCFS, "asinh") == ("signal", "linear", "compressed")
    assert assign.domains(MIRNOV, "asinh") == ("band",)


def test_the_ladders():
    assert [s.k for s in assign.rungs(IP, "linear")] == [(1,), (2,), (3,), (4,), (6,), (8,), (12,), (16,)]
    assert [s.k for s in assign.rungs(KAPPA, "linear")] == [(1,)]            # one sample per frame: K = 1 is exact
    assert [s.k for s in assign.rungs(LCFS, "signal")] == [(n, 4) for n in (0, 8, 16, 24, 32, 48, 64, 96)]
    assert assign.rungs(PSI, "signal")[4] == assign.Setting("linear", (32, 2), "signal", 64)
    assert [s.k[0] for s in assign.band_rungs(None)] == [4, 8, 16, 32] and assign.band_rungs(None)[0].k[1:] == (32,) * 7
    assert [s.k for s in assign.band_rungs(16)][1] == (16, 2, 2, 2, 2, 2, 2, 2) and assign.band_rungs(16)[0].domain == "band"


def test_the_smallest_rung_that_meets_the_target_is_chosen():
    assert assign.choose_rung([0.05, 0.02, 0.017, 0.01], 0.0176) == (2, True)
    assert assign.choose_rung([0.0176], 0.0176) == (0, True)                 # at the target counts
    assert assign.choose_rung([0.05, 0.03], 0.0176) == (1, False)            # no rung meets it: the largest, and a miss
    assert assign.choose_rung([math.nan, 0.01], 0.02) == (1, True)           # a rung that was not scored never meets
    assert assign.choose_within([0.9, 0.7, 0.685, 0.676], 0.676) == 2        # within 2% of the reference
    assert assign.choose_within([0.9, 0.7], 0.5) == 1


def test_the_reference_and_the_uniform_assignments():
    ref = assign.reference_assignment([IP, LCFS, PSI, MIRNOV])
    assert ref["summary-ip"] == assign.Setting("frame_dct", (1,))
    assert ref["equilibrium-lcfs_r"] == assign.Setting("linear", (0, 4), "signal")
    assert ref["equilibrium-psi"].k == (0, 2) and ref[MIRNOV.name] == assign.Setting("band_dct", (1,) * 8, "band")
    compress = {"summary-ip": "none", "equilibrium-lcfs_r": "none", MIRNOV.name: "asinh", "equilibrium-elongation": "none"}
    top = assign.uniform_assignment([IP, LCFS, MIRNOV], compress)
    assert top["summary-ip"].k == (16,) and top["equilibrium-lcfs_r"] == assign.Setting("linear", (96, 4), "signal")
    assert top[MIRNOV.name].k == (32,) * 8
    assert assign.uniform_assignment([IP], compress, rung=1)["summary-ip"].k == (2,)
    assert assign.uniform_assignment([KAPPA], compress, rung=3)["equilibrium-elongation"].k == (1,)


def test_an_assignment_is_stored_with_its_hash(tmp_path):
    a = {"summary-ip": assign.Setting("frame_dct", (4,)), "equilibrium-psi": assign.Setting("linear", (32, 2), "signal")}
    path = tmp_path / "v1.json"
    assign.save_assignment(a, path, notes={"summary-ip": "K=4 meets 0.0176 with 0.0161"})
    assert assign.load_assignment(path) == a
    b = dict(a, **{"summary-ip": assign.Setting("frame_dct", (6,))})
    assert assign.assignment_sha256(a) != assign.assignment_sha256(b) and len(assign.assignment_sha256(a)) == 64
    named = dict(a, **{"summary-ip": assign.Setting("frame_dct", (4,), source="basis/x.npz")})
    assert assign.Setting("frame_dct", (4,)).source == "" and assign.assignment_sha256(named) != assign.assignment_sha256(a)
    path.write_text(path.read_text().replace('"domain": "signal"', '"domain": "linear"'))
    with pytest.raises(ValueError, match="sha256"):
        assign.load_assignment(path)


def test_the_fidelity_targets_are_a_fifth_of_the_best_published_score():
    pytest.importorskip("tokamark")
    t = assign.targets()
    assert len(t) == 39
    assert t["equilibrium-lcfs_r"] == pytest.approx(0.2 * tasks.best_published("task_2-2"))
    assert t["equilibrium-psi"] == pytest.approx(0.2 * tasks.best_published("task_2-3"))
    assert t["summary-ip"] == pytest.approx(0.2 * tasks.best_published("task_2-1"))
    assert t["soft_x_rays-horizontal_cam_lower"] == pytest.approx(0.2 * tasks.best_published("task_4-1"))
    assert t["magnetics-flux_loop_flux"] == assign.INPUT_ONLY_TARGET
    assert all(t[name] is None for name in tasks.SPECTROGRAM_SIGNALS)
    assert assign.predicted_by()["summary-ip"] == ("task_2-1", "task_4-4")
