"""CPU tests for the per-bin log-power Mirnov spectrogram codec (`port.binspec`)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tokamak_foundation_model.ignite.mast import paths, transform
from tokamak_foundation_model.ignite.mast.port import (
    binspec,
    framecache,
    logspec,
    peakspec,
)
from tokamak_foundation_model.ignite.mast.tokenizers import frames
from tokamak_foundation_model.ignite.mast.tokenizers.grid import GridSet, fit_ranges

P, NB, POOL = 2, 64, 2
NV = NB // POOL
DT = 0.001024


def _stats() -> transform.ChannelStats:
    return transform.ChannelStats(
        "none", np.ones((P, NB)), np.zeros((P, NB)), np.ones((P, NB))
    )


def _codec(grids: GridSet | None = None) -> binspec.BinSpec:
    rng = np.random.default_rng(3)
    floor = np.power(10.0, rng.uniform(-12.0, -8.0, size=(P, NB)))
    log_mean = np.log10(floor) + rng.uniform(2.0, 4.0, size=(P, NB))
    return binspec.BinSpec("mirnov-synth", DT, (P, NB), _stats(), floor, log_mean, POOL, grids)


def _spec(line_bin: int = 20) -> np.ndarray:
    x = np.arange(NB, dtype=np.float64)
    return 0.3 * np.cos(np.pi * (x + 0.5) / NB) - 0.1 + 1.2 * np.exp(-0.5 * ((x - line_bin) / 0.8) ** 2)


def _grids() -> GridSet:
    lo, hi, width = fit_ranges(np.stack([np.full((P, NV), -2.0), np.full((P, NV), 2.0),
                                         np.zeros((P, NV))]))
    return GridSet(lo, hi, width, n_bins=binspec.BINSPEC_N_BINS)


def test_values_are_pooled_bin_means_and_render_repeats_them():
    codec = _codec()
    assert (codec.n_tok, codec.n_val) == (P, NV)
    assert isinstance(codec, peakspec.PeakSpec)  # the frame cache and rollout dispatch on this
    s = np.stack([_spec(20), _spec(41)])[None]  # (1, P, NB)
    v = codec.encode_spectra(s)
    assert v.shape == (1, P, NV)
    np.testing.assert_allclose(v[0], s[0].reshape(P, NV, POOL).mean(-1))
    back = codec.render_spectra(v)
    assert back.shape == s.shape
    np.testing.assert_allclose(back[0, :, 0::2], v[0])
    np.testing.assert_allclose(back[0, :, 1::2], v[0])
    # the line stays where it was: the brightest rendered bin pair holds bin 20 (probe 0), 41 (probe 1)
    assert int(np.argmax(back[0, 0])) // POOL == 20 // POOL
    assert int(np.argmax(back[0, 1])) // POOL == 41 // POOL


def test_quantized_values_stay_close_and_keep_the_line():
    grids = _grids()
    codec = _codec(grids)
    s = np.stack([_spec(20), _spec(41)])[None]
    v = codec.encode_spectra(s)
    idx = grids.index(v)
    assert idx.min() >= 0 and idx.max() < binspec.BINSPEC_N_BINS
    q = grids.centres()[np.arange(P)[:, None], np.arange(NV)[None, :], idx[0]][None]
    assert np.max(np.abs(q - v)) < 0.06  # 64 asinh levels over -2.2 .. 2.2
    back = codec.render_spectra(q)
    assert int(np.argmax(back[0, 0])) // POOL == 10


def test_a_probe_frame_with_too_few_finite_bins_has_no_data():
    codec = _codec()
    t = 25
    u = np.tile(_spec()[None, :, None], (P, 1, t))
    fi = frames.frame_index(t, DT)
    n = int(fi.max()) + 1
    u[1, :10, fi == 2] = np.nan  # probe 1, frame 2: 10 of 64 bins empty in every column
    u[0, 0, fi == 3] = np.nan  # probe 0, frame 3: one bin empty -> counts as 0, still data
    v = codec.encode_aligned(u, fi, n)
    assert np.isnan(v[2, 1]).all() and np.isfinite(v[2, 0]).all()
    assert np.isfinite(v[3, 0]).all()
    assert v[3, 0, 0] == pytest.approx(0.5 * _spec()[1])  # bin 0 counted as 0 in its pair
    out = codec.render_domain(v, fi, n)
    assert np.isnan(out[1][:, fi == 2]).all()
    assert np.isfinite(out[0]).all()


def test_aligned_path_matches_the_contiguous_codec_interface():
    codec = _codec(_grids())
    rng = np.random.default_rng(0)
    t = 40
    u = _spec(33)[None, :, None] + 0.02 * rng.standard_normal((P, NB, t))
    x = codec.from_domain(u)[None]  # physical (1, P, NB, T)
    fi = frames.frame_index(t, DT)
    n = int(fi.max()) + 1
    v1 = codec.encode(x)[0]
    v2 = codec.encode_aligned(codec.to_domain(x[0]), fi, n)
    np.testing.assert_allclose(v1, v2)
    d1 = codec.decode(v1[None], t)[0]
    d2 = codec.render(v2, fi, n)
    np.testing.assert_allclose(d1, d2)
    np.testing.assert_allclose(
        framecache._encode_domain_by_frame_index(codec, codec.to_domain(x), fi, n), v2
    )
    np.testing.assert_allclose(framecache._decode_domain_by_frame_index(codec, v2[None], fi, n), d2)
    assert framecache.live_mask(codec).all() and framecache.live_mask(codec).size == P * NV
    assert framecache.codec_vocab(codec) == (64, 64, 65)


def test_pool_must_divide_the_bins():
    with pytest.raises(ValueError, match="pool"):
        binspec.BinSpec("x", DT, (P, NB), _stats(), np.ones((P, NB)), np.zeros((P, NB)), 3)


def test_logspec_build_dispatches_on_a_binspec_fit(tmp_path: Path):
    codec = _codec(_grids())
    name = "magnetics-b_field_pol_probe_omv_voltage"
    fit = tmp_path / "fit"
    fit.mkdir()
    g = codec.grids
    peakspec._write_npz(fit / "logspec_fit.npz", {
        f"{name}::floor": codec.floor, f"{name}::log_mean": codec.log_mean,
        f"{name}::lo": g.lo, f"{name}::hi": g.hi, f"{name}::width": g.width,
    })
    meta = {"codec": "binspec", "n_bins": 64, "pool": POOL, "signals": {name: {"dt": DT}}}
    (fit / "logspec_fit.json").write_text(json.dumps(meta))
    built = logspec.build_logspec_codecs(fit, {name: _stats()})
    c2 = built[name]
    assert isinstance(c2, binspec.BinSpec)
    assert (c2.n_tok, c2.n_val, c2.pool) == (P, NV, POOL)
    np.testing.assert_array_equal(c2.grids.centres(), g.centres())


_CACHE_V1 = paths.cache_dir("v1")
_FRAMES = paths.mast_root() / "stage2r" / "frame_codes_v1b_logspec_cam_live_da"
_KEEP = paths.mast_root() / "stage2r" / "tokeye" / "scores_v2"


@pytest.mark.real_data
def test_real_fit_on_a_few_kept_shots(tmp_path: Path):
    for p in (_CACHE_V1 / "stats.npz", _FRAMES / "_codec_manifest.json", _KEEP / "signals.parquet",
              logspec.DEFAULT_LOGSPEC_OUT / "logspec_fit.npz"):
        if not p.exists():
            pytest.skip(f"{p} not found")
    meta = binspec.fit_binspec(
        _CACHE_V1, _FRAMES, logspec.DEFAULT_LOGSPEC_OUT, _KEEP, tmp_path / "fit",
        n_shots=4, workers=2,
    )
    assert meta["codec"] == "binspec"
    stats = transform.load_stats(_CACHE_V1 / "stats.npz")
    built = logspec.build_logspec_codecs(tmp_path / "fit", stats)
    for name in meta["signals"]:
        c = built[name]
        assert isinstance(c, binspec.BinSpec)
        assert (c.n_tok, c.n_val) == (3, 256 // binspec.BINSPEC_POOL)
        assert np.all(c.grids.hi > c.grids.lo)
