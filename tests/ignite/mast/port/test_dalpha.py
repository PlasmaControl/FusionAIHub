"""The D-alpha level + spikes codec (`port.dalpha.PeakLevel`) on synthetic signals."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tokamak_foundation_model.ignite.mast import cache, transform
from tokamak_foundation_model.ignite.mast.port import dalpha, framecache
from tokamak_foundation_model.ignite.mast.tokenizers import frames
from tokamak_foundation_model.ignite.mast.tokenizers.grid import GridSet

DT = 2e-5            # 50 kHz, 250 samples per 5 ms frame
L = 250
PRE, POST = 10, 90


def _template() -> np.ndarray:
    t = np.zeros(PRE + POST)
    t[:PRE] = np.linspace(0.0, 1.0, PRE + 1)[:-1]
    t[PRE:] = np.exp(-np.arange(POST) / 8.0)
    return t


def _stats(n_ch: int = 3) -> transform.ChannelStats:
    one = np.ones(n_ch)
    return transform.ChannelStats("none", one, np.zeros(n_ch), one)


def _codec(n_ch: int = 3, n_peaks: int = 2) -> dalpha.PeakLevel:
    n_val = 1 + 2 * n_peaks
    lo = np.zeros((n_ch, n_val))
    hi = np.ones((n_ch, n_val))
    width = np.full((n_ch, n_val), 100.0)
    lo[:, 0], hi[:, 0], width[:, 0] = 0.0, 5.0, 0.2
    for p in range(n_peaks):
        hi[:, 1 + 2 * p], width[:, 1 + 2 * p] = 3.0, 0.01
    return dalpha.PeakLevel(
        dalpha.SIGNAL, DT, (n_ch,), _stats(n_ch), _template(), PRE, 0.05, n_peaks,
        GridSet(lo, hi, width, n_bins=64),
    )


def _signal(n_frames: int = 12, seed: int = 0) -> tuple[np.ndarray, list[tuple[int, float]]]:
    """3 channels: a level that drops at frame 6 (an L-H drop), spikes on channel 0, small noise."""
    rng = np.random.default_rng(seed)
    n = n_frames * L
    level = np.where(np.arange(n) < 6 * L, 1.2, 0.4)
    x = np.stack([level, 0.8 * level, 0.5 * level]) + rng.normal(0.0, 0.005, (3, n))
    tmpl = _template()
    spikes = [(7 * L + 60, 1.5), (8 * L + 120, 0.9), (9 * L + 30, 1.2), (9 * L + 180, 0.6)]
    for at, h in spikes:
        lo, hi = at - PRE, min(n, at - PRE + tmpl.size)
        x[0, lo:hi] += h * tmpl[: hi - lo]
    return x, spikes


def _roundtrip(codec: dalpha.PeakLevel, x: np.ndarray) -> np.ndarray:
    fi = frames.frame_index(x.shape[-1], DT)
    n_frames = int(fi.max()) + 1
    vals = codec.encode_aligned(x, fi, n_frames)
    q = codec.grids.read(vals, "nearest")
    return codec.render(q, fi, n_frames)


def test_spikes_and_level_drop_survive_the_roundtrip():
    codec = _codec()
    x, spikes = _signal()
    y = _roundtrip(codec, x)
    assert y.shape == x.shape
    for at, h in spikes:
        # the peak sample is reproduced to within 10% of the spike height (quantized height and time)
        win = slice(at - 5, at + 6)
        assert abs(y[0, win].max() - x[0, win].max()) < 0.1 * h + 0.02
    # the level on both sides of the drop, away from the spikes and the interpolation ramp
    for c, lev in ((1, 0.8), (2, 0.5)):                  # within the level grid's resolution there
        assert abs(np.median(y[c, : 5 * L]) - 1.2 * lev) < 0.04
        assert abs(np.median(y[c, 7 * L:]) - 0.4 * lev) < 0.04
    # quantized, the codec beats an unquantized frame mean + slope (the k=2 DCT) on the spiky channel
    fi = frames.frame_index(x.shape[-1], DT)
    dct = np.empty_like(x[0])
    for f in range(int(fi.max()) + 1):
        seg = x[0, fi == f]
        tt = np.arange(seg.size) - (seg.size - 1) / 2
        a, b = np.polyfit(tt, seg, 1)
        dct[fi == f] = a * tt + b
    assert np.sqrt(np.mean((y[0] - x[0]) ** 2)) < 0.8 * np.sqrt(np.mean((dct - x[0]) ** 2))


def test_quiet_frames_have_no_spikes():
    codec = _codec()
    x, _ = _signal()
    fi = frames.frame_index(x.shape[-1], DT)
    vals = codec.encode_aligned(x, fi, int(fi.max()) + 1)
    assert vals.shape == (12, 3, 5)
    # frames 5 and 6 hold the step drop, which the level curve (linear between frame centres) cannot
    # follow, so they may carry spikes; elsewhere quiet frames code h = 0, t = 0
    quiet = np.r_[0:5, 7:12]
    assert np.all(vals[:5, :, 1:] == 0.0)
    assert np.all(vals[quiet][:, 1:, 1:] == 0.0)          # channels 1 and 2 carry no spikes
    assert vals[7, 0, 1] > 1.0 and vals[9, 0, 1] > 1.0 and vals[9, 0, 3] > 0.4


def test_a_frame_without_data_is_missing_not_zero():
    codec = _codec()
    x, _ = _signal()
    x[:, 3 * L: 4 * L] = np.nan
    fi = frames.frame_index(x.shape[-1], DT)
    vals = codec.encode_aligned(x, fi, int(fi.max()) + 1)
    assert np.isnan(vals[3]).all()
    assert np.isfinite(vals[2]).all() and np.isfinite(vals[4]).all()
    y = codec.render(codec.grids.read(vals, "nearest"), fi, int(fi.max()) + 1)
    assert np.isnan(y[:, 3 * L: 4 * L]).all()
    assert np.isfinite(y[:, : 3 * L]).all() and np.isfinite(y[:, 4 * L:]).all()


def test_codec_interface_roundtrip_matches_render():
    codec = _codec()
    x, _ = _signal()
    via_codec = codec.roundtrip(x[None], "nearest")[0]
    np.testing.assert_allclose(via_codec, _roundtrip(codec, x), atol=1e-12)


def test_framecache_codes_roundtrip(monkeypatch):
    codec = _codec()
    x, spikes = _signal(n_frames=14)
    # the cache holds model units; stats are the identity here, and the signal starts 1 frame early
    cs = cache.CachedSignal(values=x.astype(np.float32), t0=-0.005, dt=DT)
    codes, present = framecache.encode_signal_codes(codec, cs, t0=0.0, n_frames=12)
    assert present and tuple(codes.shape) == (12, 15)
    assert int(codes.max()) < 64                            # no missing code in a full window
    fi = framecache.sample_frame_index(x.shape[-1], cs.t0, DT, 0.0)
    keep = (fi >= 0) & (fi < 12)
    y = framecache.decode_signal_codes(codec, codes, sample_frame_idx=fi[keep])
    assert y.shape == (3, int(keep.sum()))
    xs = x[:, keep]
    for at, h in spikes:
        j = at - L                                           # the window starts one frame in
        if 0 <= j < xs.shape[-1]:
            assert abs(y[0, j - 5: j + 6].max() - xs[0, j - 5: j + 6].max()) < 0.1 * h + 0.02
    assert framecache.live_mask(codec).all()


def test_fit_file_roundtrip_and_sha(tmp_path):
    codec = _codec()
    lo, hi = codec.grids.lo, codec.grids.hi
    np.savez(
        tmp_path / "dalpha_fit.npz", template=codec.template, pre=np.int64(PRE),
        threshold=np.float64(0.05), local_k=np.float64(5.0), n_peaks=np.int64(2), n_bins=np.int64(64), lo=lo, hi=hi,
        width=codec.grids.width,
    )
    (tmp_path / "dalpha_fit.json").write_text(json.dumps({"version": dalpha.FIT_VERSION}))
    built = dalpha.build_dalpha_codec(tmp_path, {dalpha.SIGNAL: _stats()}, DT)
    x, _ = _signal()
    np.testing.assert_allclose(_roundtrip(built, x), _roundtrip(codec, x), atol=1e-12)
    sha = dalpha.dalpha_fit_sha256(tmp_path)
    assert dalpha.resolve_manifest_dalpha_fit({"dalpha_fit_path": str(tmp_path), "dalpha_fit_sha256": sha}) == tmp_path
    assert dalpha.resolve_manifest_dalpha_fit({}) is None
    with pytest.raises(ValueError, match="mismatch"):
        dalpha.resolve_manifest_dalpha_fit({"dalpha_fit_path": str(tmp_path), "dalpha_fit_sha256": "0" * 64})
    with pytest.raises(ValueError, match="without"):
        dalpha.resolve_manifest_dalpha_fit({"dalpha_fit_path": str(tmp_path)})


def test_noise_above_the_fitted_threshold_is_not_coded_as_spikes():
    # a channel whose fast fluctuations top the fitted threshold (0.05 V) every frame: the per-frame
    # noise threshold (local_k x the robust noise of sample-to-sample differences) keeps them out
    codec = _codec()
    rng = np.random.default_rng(1)
    x = 1.0 + rng.normal(0.0, 0.04, (3, 10 * L))
    fi = frames.frame_index(x.shape[-1], DT)
    vals = codec.encode_aligned(x, fi, 10)
    assert np.all(vals[..., 1:] == 0.0)
    # with the per-frame threshold off, the same noise fills the spike slots
    codec.local_k = 0.0
    assert (codec.encode_aligned(x, fi, 10)[..., 1] > 0).mean() > 0.9
