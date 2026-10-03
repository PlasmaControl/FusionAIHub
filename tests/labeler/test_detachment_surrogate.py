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
    spec.loader.exec_module(module)
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
