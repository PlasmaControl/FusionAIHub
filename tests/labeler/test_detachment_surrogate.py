"""The TangTV frame regression and its path into the bins (scripts/labeler/detach_*)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from labeler.events.detachment import core

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "labeler"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(SCRIPTS))
    return module


@pytest.fixture(scope="module")
def surrogate():
    try:
        yield load("detach_tv_surrogate")
    finally:
        sys.modules.pop("detach_tv_surrogate", None)


@pytest.fixture(scope="module")
def bins():
    try:
        yield load("detach_bins")
    finally:
        sys.modules.pop("detach_bins", None)


def test_features_drop_the_black_level_and_append_the_x_point_radius(surrogate):
    frames = np.full((2, 240, 720), surrogate.BLACK, dtype=np.float32)
    frames[1, :6, :6] = surrogate.BLACK + 36.0
    feats = surrogate.features(frames, np.array([1.2, 1.3]))
    assert feats.shape == (2, 40 * 120 + 1)
    assert np.all(feats[0, :-1] == 0.0)
    assert feats[1, :-1].mean() == pytest.approx(0, abs=1e-5)
    assert feats[1, :-1].std() == pytest.approx(1, abs=1e-5)
    assert feats[:, -1] == pytest.approx(surrogate.RX_SCALE * np.array([1.2, 1.3]))


def test_ridge_recovers_a_linear_target(surrogate):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(60, 8))
    y = x @ np.arange(1.0, 9.0) + 0.5
    model = surrogate.Ridge().fit(x, y, alpha=1e-6)
    fresh = rng.normal(size=(5, 8))
    assert model.predict(fresh) == pytest.approx(
        fresh @ np.arange(1.0, 9.0) + 0.5, abs=1e-3
    )


def test_kappa_is_one_for_identical_votes_and_zero_for_chance(surrogate):
    a = np.array([1, 1, 2, 2, 3, 3, 1, 2])
    assert surrogate.kappa(a, a) == pytest.approx(1.0)
    chance = surrogate.kappa(np.array([1, 1, 2, 2]), np.array([1, 2, 1, 2]))
    assert chance == pytest.approx(0.0)


def _cache(rvsod, zvsod, rxpt1, zxpt1):
    t = np.arange(0.0, 400.0, 10.0)
    return {
        "rvsod": (t, np.full(t.size, rvsod)),
        "zvsod": (t, np.full(t.size, zvsod)),
        "rxpt1": (t, np.full(t.size, rxpt1)),
        "zxpt1": (t, np.full(t.size, zxpt1)),
    }


def _with_surrogate(bins, monkeypatch, ze):
    monkeypatch.setattr(
        bins.signals, "tangtv_geometry", lambda shot, cache: (cache, "EFIT02")
    )
    frame_t = np.arange(0.0, 400.0, 20.0)
    monkeypatch.setattr(bins, "load_inversion", lambda shot: None)
    monkeypatch.setattr(
        bins,
        "load_surrogate",
        lambda shot: {
            "times_ms": frame_t,
            "ze": np.full(frame_t.size, ze),
            "valid": np.ones(frame_t.size, bool),
        },
    )
    return core.bin_edges(0.0, 400.0)


def test_surrogate_front_height_votes_on_the_shelf(bins, monkeypatch):
    edges = _with_surrogate(bins, monkeypatch, ze=-1.2)
    cache = _cache(rvsod=1.5, zvsod=-1.25, rxpt1=1.3, zxpt1=-1.1)
    indicator, source = bins.tangtv_for(1, edges, cache)
    assert source == "surrogate"
    assert indicator.valid.all()
    assert np.all(indicator.vote == core.ATTACHED)  # DZ = 1 - 0.1 / 0.15 = 0.33


def test_surrogate_never_votes_on_the_floor(bins, monkeypatch):
    edges = _with_surrogate(bins, monkeypatch, ze=-1.3)
    cache = _cache(rvsod=1.2, zvsod=-1.363, rxpt1=1.3, zxpt1=-1.1)
    indicator, source = bins.tangtv_for(1, edges, cache)
    assert source == "surrogate"
    assert not indicator.valid.any()
    assert np.all(indicator.vote == core.ABSTAIN)
    assert set(indicator.reason) == {"strike_on_floor"}


def test_no_inversion_and_no_surrogate_is_invalid_and_says_so(bins, monkeypatch):
    monkeypatch.setattr(
        bins.signals, "tangtv_geometry", lambda shot, cache: (cache, "EFIT02")
    )
    monkeypatch.setattr(bins, "load_inversion", lambda shot: None)
    monkeypatch.setattr(bins, "load_surrogate", lambda shot: None)
    edges = core.bin_edges(0.0, 400.0)
    indicator, source = bins.tangtv_for(1, edges, _cache(1.5, -1.25, 1.3, -1.1))
    assert source == "none"
    assert not indicator.valid.any()
    assert set(indicator.reason) == {"no_inversion"}


def test_features_ignore_global_brightness_after_black_subtraction(surrogate):
    tv = surrogate
    frame = np.arange(240 * 720, dtype=np.float32).reshape(1, 240, 720) % 100
    rx = np.array([1.3])
    a = tv.features(frame + tv.BLACK, rx)
    b = tv.features(2 * frame + tv.BLACK, rx)
    np.testing.assert_allclose(a, b, atol=1e-5)


def test_confinement_uses_tracked_store_without_environment(bins, monkeypatch):
    monkeypatch.delenv("LABELER_LABEL_TABLES", raising=False)
    mode, _ = bins.confinement(149993, np.array([1000.0, 1050.0, 1100.0]))
    assert np.all(mode == 2)


def test_spatial_evidence_excludes_rejected_camera_frames(bins, monkeypatch, tmp_path):
    monkeypatch.setattr(bins, "root", lambda: tmp_path)
    (tmp_path / "efit").mkdir()
    ft = np.array([10.0, 20.0, 60.0, 70.0])
    frames = np.zeros((4, 2, 2))
    frames[[0, 2], 0, 0] = 10  # Only rejected frames have an inside peak.
    frames[[1, 3], 1, 1] = 10
    inv = {
        "times_ms": ft,
        "frames": frames,
        "radii": np.array([1.3, 1.5]),
        "elevation": np.array([-1.05, -0.9]),
    }
    monkeypatch.setattr(bins, "load_inversion", lambda shot: inv)
    np.savez(
        tmp_path / "efit/1.npz",
        source="EFIT02",
        gtime_ms=ft,
        r=inv["radii"],
        z=inv["elevation"],
        ssimag=np.zeros(4),
        ssibry=np.ones(4),
        psirz=np.tile([[0.5, 1.2], [1.2, 1.2]], (4, 1, 1)),
    )
    geo = {"rxpt1": (ft, np.full(4, 1.3)), "zxpt1": (ft, np.full(4, -1.1))}
    spatial = bins.spatial_evidence(
        1, np.array([0.0, 50.0, 100.0]), geo, np.array([False, True, False, True])
    )
    assert not spatial.any()


def test_processed_reference_excludes_quality_failures_and_keeps_method_name(
    bins, monkeypatch, tmp_path
):
    monkeypatch.setattr(bins, "root", lambda: tmp_path)
    (tmp_path / "processed_probes").mkdir()
    edges = np.arange(0.0, 550.0, 50.0)
    t = core.bin_centres(edges)
    np.savez(
        tmp_path / "processed_probes/1.npz",
        p1_t_ms=t,
        p1_jsat=np.ones(10),
        p1_rz=np.array([1.5, -1.25]),
    )
    geo = _cache(1.5, -1.25, 1.3, -1.1)
    monkeypatch.setattr(bins.signals, "tangtv_geometry", lambda *args: (geo, "EFIT02"))
    monkeypatch.setattr(bins.signals, "line_density", lambda cache: (t, np.ones(10)))
    monkeypatch.setattr(
        bins.signals, "heating_power", lambda *args: (t, np.ones(10), np.ones(10))
    )
    monkeypatch.setattr(
        bins.signals,
        "corpus_group",
        lambda *args: (t, np.array([[0.0] * 7 + [10.0] * 3])),
    )
    monkeypatch.setattr(
        bins, "confinement", lambda *args: (np.ones(10, int), np.zeros(10, bool))
    )
    reason = np.array(["ramp"] * 6 + [""] * 4)
    base = core.assemble("afrac", np.ones(10), reason == "", reason, np.ones(10))
    monkeypatch.setattr(bins.afrac, "afrac_indicator", lambda *args: base)
    tv = core.assemble(
        "tangtv", np.zeros(10), np.ones(10, bool), np.full(10, ""), np.ones(10)
    )
    seen = {}

    def calibrate(jsat, positions, strike, scaling, attached, regime):
        seen["attached"] = attached
        return np.ones(10), np.ones(10, bool), np.zeros(10, int)

    monkeypatch.setattr(bins.afrac, "calibrated_ratio", calibrate)
    _, method = bins.processed_ratio(
        1, edges, {"ipmeas": (t, np.ones(10))}, base, tv, None
    )
    assert not seen["attached"][:6].any()
    assert method[-1] == "eldon_pre_puff_LH"


def test_surrogate_training_rejects_stale_efit_geometry(surrogate, monkeypatch):
    cache = _cache(1.5, -1.25, 1.3, -1.1)
    monkeypatch.setattr(
        surrogate.signals, "tangtv_geometry", lambda shot: (cache, "EFIT02")
    )
    *_, valid = surrogate.geometry(1, np.array([100.0, 600.0]))
    assert valid.tolist() == [True, False]


def test_exported_camera_geometry_excludes_efit_sentinels(bins):
    t = np.array([5.0, 15.0, 25.0])
    geo = {
        "rvsod": (t, np.full(3, 1.3)),
        "zvsod": (t, np.array([-1.363, -0.89, -0.89])),
        "rxpt1": (t, np.full(3, 1.2)),
        "zxpt1": (t, np.full(3, -1.11)),
    }
    out = bins.geometry_aux(geo, np.array([0.0, 50.0]))
    assert out["aux_zxpt1"][0] - out["aux_zvsod"][0] == pytest.approx(0.253)
