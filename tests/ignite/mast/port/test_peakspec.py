"""CPU tests for the peak + baseline Mirnov spectrogram codec (`port.peakspec`)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tokamak_foundation_model.ignite.mast import paths, transform
from tokamak_foundation_model.ignite.mast.port import framecache, logspec, peakspec
from tokamak_foundation_model.ignite.mast.tokenizers import frames
from tokamak_foundation_model.ignite.mast.tokenizers.grid import GridSet

P, NB, M, NBANDS = 2, 64, 8, 8  # band width 8 bins
DT = 0.001024


def _stats() -> transform.ChannelStats:
    return transform.ChannelStats(
        "none", np.ones((P, NB)), np.zeros((P, NB)), np.ones((P, NB))
    )


def _codec(grids: GridSet | None = None, threshold: float = 0.1) -> peakspec.PeakSpec:
    rng = np.random.default_rng(3)
    floor = np.power(10.0, rng.uniform(-12.0, -8.0, size=(P, NB)))
    log_mean = np.log10(floor) + rng.uniform(2.0, 4.0, size=(P, NB))
    return peakspec.PeakSpec(
        "mirnov-synth", DT, (P, NB), _stats(), floor, log_mean, M, NBANDS, threshold, grids
    )


def _gauss(f: float, h: float, s: float) -> np.ndarray:
    x = np.arange(NB, dtype=np.float64)
    return h * np.exp(-0.5 * ((x - f) / s) ** 2)


def _base() -> np.ndarray:
    x = np.arange(NB, dtype=np.float64)
    return 0.3 * np.cos(np.pi * (x + 0.5) / NB) - 0.1


def _grids(h_hi: float = 1.5) -> GridSet:
    # baseline coefficient ranges wide enough for the synthetic spectra
    bv = np.stack([np.full((P, M), -3.0), np.full((P, M), 3.0)])
    return peakspec.peak_grids(bv, np.array([h_hi]), M, NBANDS, NB // NBANDS, height_hi_pct=100.0)


def test_encode_finds_each_peak_in_its_band_and_renders_back():
    codec = _codec()
    spec = _base() + _gauss(13.3, 0.8, 1.2) + _gauss(40.6, 0.5, 2.0)
    s = np.stack([spec, _base()])[None]  # (1, P, NB): probe 1 has no peaks
    v = codec.encode_spectra(s)
    assert v.shape == (1, P, M + 3 * NBANDS)
    pk = v[0, :, M:].reshape(P, NBANDS, 3)
    # probe 0: band 1 (bins 8..15) and band 5 (bins 40..47)
    # the baseline (8 coefficients over 64 bins, one robust pass) still lifts a little under a
    # line, so a peak token holds most, not all, of the line's height; the render adds it back
    assert pk[0, 1, 0] + 8 == pytest.approx(13.3, abs=0.25)
    assert 0.6 < pk[0, 1, 1] <= 0.85
    assert pk[0, 1, 2] == pytest.approx(1.2, abs=0.5)
    assert pk[0, 5, 0] + 40 == pytest.approx(40.6, abs=0.25)
    assert 0.3 < pk[0, 5, 1] <= 0.55
    others = [b for b in range(NBANDS) if b not in (1, 5)]
    assert np.all(pk[0, others, 1] == 0.0)
    # an empty band codes (-0.5, 0, SIGMA_MIN); probe 1 has nothing above the threshold
    assert np.all(pk[1, :, 0] == -0.5) and np.all(pk[1, :, 1] == 0.0)
    assert np.all(pk[1, :, 2] == peakspec.SIGMA_MIN)
    back = codec.render_spectra(v)
    assert np.max(np.abs(back - s)) < 0.12
    # the robust baseline does not lift around the lines: far from them the residual is small
    far = np.r_[0:6, 22:34, 54:64]
    assert np.max(np.abs(back[0, 0, far] - s[0, 0, far])) < 0.05


def test_empty_bands_sit_on_the_lowest_grid_centre_and_quantized_peaks_stay_sharp():
    grids = _grids()
    codec = _codec(grids)
    spec = _base() + _gauss(27.4, 0.9, 1.0)
    s = np.stack([spec, _base()])[None]
    v = codec.encode_spectra(s)
    idx = grids.index(v)  # (1, P, n_val)
    pk_idx = idx[0, :, M:].reshape(P, NBANDS, 3)
    assert np.all(pk_idx[1] == 0)  # every slot of the probe without peaks is code 0
    assert np.all(pk_idx[0, [b for b in range(NBANDS) if b != 3]] == 0)
    assert pk_idx[0, 3, 1] > 0
    centres = grids.centres()
    q = centres[np.arange(P)[:, None], np.arange(codec.n_val)[None, :], idx[0]][None]
    back = codec.render_spectra(q)
    # the quantized line keeps its height (the heights are measured on the quantized baseline)
    assert back[0, 0, 27] - back[0, 0, 20] == pytest.approx(s[0, 0, 27] - s[0, 0, 20], abs=0.08)
    assert np.max(np.abs(back - s)) < 0.12


def test_a_probe_frame_with_too_few_finite_bins_has_no_data():
    codec = _codec()
    t = 25
    u = np.tile((_base() + _gauss(20.0, 0.6, 1.5))[None, :, None], (P, 1, t))
    fi = frames.frame_index(t, DT)
    n = int(fi.max()) + 1
    u[1, :10, fi == 2] = np.nan  # probe 1, frame 2: 10 of 64 bins empty in every column
    u[0, 0, fi == 3] = np.nan  # probe 0, frame 3: one bin empty -> filled with 0, still data
    v = codec.encode_aligned(u, fi, n)
    assert np.isnan(v[2, 1]).all() and np.isfinite(v[2, 0]).all()
    assert np.isfinite(v[3, 0]).all()
    out = codec.render_domain(v, fi, n)
    assert np.isnan(out[1][:, fi == 2]).all()
    assert np.isfinite(out[0]).all()


def test_aligned_path_matches_the_contiguous_codec_interface():
    codec = _codec(_grids())
    rng = np.random.default_rng(0)
    t = 40
    u = (_base() + _gauss(33.0, 0.7, 1.1))[None, :, None] + 0.02 * rng.standard_normal((P, NB, t))
    x = codec.from_domain(u)[None]  # physical (1, P, NB, T)
    fi = frames.frame_index(t, DT)
    n = int(fi.max()) + 1
    v1 = codec.encode(x)[0]
    v2 = codec.encode_aligned(codec.to_domain(x[0]), fi, n)
    np.testing.assert_allclose(v1, v2)
    d1 = codec.decode(v1[None], t)[0]
    d2 = codec.render(v2, fi, n)
    np.testing.assert_allclose(d1, d2)
    mask = codec.data_mask(x)
    assert mask.shape == x.shape and mask.all()
    # the frame-cache entry points dispatch to the same code
    np.testing.assert_allclose(
        framecache._encode_domain_by_frame_index(codec, codec.to_domain(x), fi, n), v2
    )
    np.testing.assert_allclose(framecache._decode_domain_by_frame_index(codec, v2[None], fi, n), d2)
    assert framecache.live_mask(codec).all() and framecache.live_mask(codec).size == P * codec.n_val
    assert framecache.codec_vocab(codec) == (64, 64, 65)


def test_peak_grids_layout():
    g = _grids(h_hi=2.0)
    assert (g.n_tok, g.n_val) == (P, M + 3 * NBANDS)
    c = g.centres()
    for b in range(NBANDS):
        o = M + 3 * b
        assert c[0, o, 0] == -0.5 and c[0, o, -1] == pytest.approx(NB // NBANDS - 0.5)
        assert c[0, o + 1, 0] == 0.0 and c[0, o + 1, -1] == pytest.approx(2.0)
        assert c[0, o + 2, 0] == peakspec.SIGMA_MIN and c[0, o + 2, -1] == peakspec.SIGMA_MAX
    # the offset grid is linear: sub-bin steps
    steps = np.diff(c[0, M])
    assert np.allclose(steps, steps[0], rtol=1e-3) and steps[0] < 0.15


def test_logspec_build_dispatches_on_a_peakspec_fit(tmp_path: Path):
    codec = _codec(_grids())
    name = "magnetics-b_field_pol_probe_omv_voltage"
    fit = tmp_path / "fit"
    fit.mkdir()
    g = codec.grids
    peakspec._write_npz(fit / "logspec_fit.npz", {
        f"{name}::floor": codec.floor, f"{name}::log_mean": codec.log_mean,
        f"{name}::lo": g.lo, f"{name}::hi": g.hi, f"{name}::width": g.width,
    })
    meta = {"codec": "peakspec", "n_bins": 64, "m": M, "n_bands": NBANDS, "threshold": 0.1,
            "signals": {name: {"dt": DT}}}
    (fit / "logspec_fit.json").write_text(json.dumps(meta))
    built = logspec.build_logspec_codecs(fit, {name: _stats()})
    c2 = built[name]
    assert isinstance(c2, peakspec.PeakSpec)
    assert (c2.n_tok, c2.n_val, c2.m, c2.n_bands) == (P, codec.n_val, M, NBANDS)
    np.testing.assert_array_equal(c2.grids.centres(), g.centres())
    sha1 = logspec.logspec_fit_sha256(fit)
    peakspec._write_npz(fit / "logspec_fit.npz", {
        f"{name}::floor": codec.floor, f"{name}::log_mean": codec.log_mean,
        f"{name}::lo": g.lo, f"{name}::hi": g.hi, f"{name}::width": g.width,
    })
    assert logspec.logspec_fit_sha256(fit) == sha1  # the npz is byte-for-byte deterministic


_CACHE_V1 = paths.cache_dir("v1")
_FRAMES = paths.mast_root() / "stage2r" / "frame_codes_v1b_logspec_cam_live_da"
_KEEP = paths.mast_root() / "stage2r" / "tokeye" / "scores_v2"


@pytest.mark.real_data
def test_real_fit_on_a_few_kept_shots(tmp_path: Path):
    for p in (_CACHE_V1 / "stats.npz", _FRAMES / "_codec_manifest.json", _KEEP / "signals.parquet",
              logspec.DEFAULT_LOGSPEC_OUT / "logspec_fit.npz"):
        if not p.exists():
            pytest.skip(f"{p} not found")
    meta = peakspec.fit_peakspec(
        _CACHE_V1, _FRAMES, logspec.DEFAULT_LOGSPEC_OUT, _KEEP, tmp_path / "fit",
        n_shots=4, workers=2,
    )
    assert meta["codec"] == "peakspec"
    stats = transform.load_stats(_CACHE_V1 / "stats.npz")
    built = logspec.build_logspec_codecs(tmp_path / "fit", stats)
    for name, info in meta["signals"].items():
        c = built[name]
        assert (c.n_tok, c.n_val) == (3, peakspec.PEAKSPEC_M + 3 * peakspec.PEAKSPEC_N_BANDS)
        assert 0.0 < info["band_occupancy"] < 1.0
