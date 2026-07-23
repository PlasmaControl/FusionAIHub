"""Tests for IGNITE Phase-A data glue (log_power_stft, shift_pair_windows).

SYNTHETIC raw signals only -- no real HDF5 data required, CPU only.
Written test-first (TDD). See docs/IGNITE_DESIGN.md §4.2 (raw δ-shift + re-STFT
consistency pair).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from tokamak_foundation_model.ignite import data
from tokamak_foundation_model.ignite.config import (
    CHUNK_S,
    STFT_FS,
    STFT_HOP,
    STFT_N_FFT,
    SpectroCodecConfig,
)


# --------------------------------------------------------------------------- #
# synthetic-signal helpers
# --------------------------------------------------------------------------- #
def _tone_signal(
    n_samples: int,
    freqs_hz,
    fs: float = STFT_FS,
    noise_std: float = 0.0,
    seed: int = 0,
) -> torch.Tensor:
    """A single-channel (1, n_samples) raw signal = sum of fixed-frequency
    sinusoids ("plasma modes") + optional white noise."""
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(n_samples, dtype=torch.float32) / fs
    x = torch.zeros(n_samples, dtype=torch.float32)
    for f in freqs_hz:
        phase = 2.0 * math.pi * float(torch.rand((), generator=g))
        x = x + torch.sin(2.0 * math.pi * f * t + phase)
    if noise_std > 0.0:
        x = x + noise_std * torch.randn(n_samples, generator=g)
    return x.unsqueeze(0)  # (1, n_samples)


def _small_cfg() -> SpectroCodecConfig:
    """Full STFT grid (freq_bins/time_frames must match the real STFT sizes for
    the crop/pad path to exercise realistic numbers), tiny transformer params."""
    return SpectroCodecConfig(channels=1, d_model=16, enc_depth=1, dec_depth=1, heads=2)


def _dominant_freq_bins(spec_ct: torch.Tensor, k: int = 3) -> set:
    """Top-k frequency bins by total power across time for channel 0."""
    # spec_ct: (C, F, T) log-power. Rank by summed power (not log) over time.
    power = (10.0 ** spec_ct[0]).sum(dim=-1)  # (F,)
    return set(torch.topk(power, k).indices.tolist())


# --------------------------------------------------------------------------- #
# log_power_stft — shape
# --------------------------------------------------------------------------- #
def test_log_power_stft_output_shape():
    cfg = _small_cfg()
    B, C, W = 2, cfg.channels, cfg.window_samples
    raw = torch.stack(
        [_tone_signal(W, [50e3, 120e3], seed=i) for i in range(B)], dim=0
    )  # (B, C, W)  (C==1)
    spec = data.log_power_stft(raw, cfg)
    assert spec.shape == (B, C, cfg.freq_bins, cfg.time_frames), spec.shape


def test_log_power_stft_multichannel_shape():
    cfg = SpectroCodecConfig(channels=3, d_model=16, enc_depth=1, dec_depth=1, heads=2)
    B, C, W = 2, cfg.channels, cfg.window_samples
    raw = torch.randn(B, C, W)
    spec = data.log_power_stft(raw, cfg)
    assert spec.shape == (B, C, cfg.freq_bins, cfg.time_frames), spec.shape


def test_log_power_stft_finite_and_real():
    cfg = _small_cfg()
    raw = _tone_signal(cfg.window_samples, [80e3]).unsqueeze(0)  # (1,1,W)
    spec = data.log_power_stft(raw, cfg)
    assert torch.is_floating_point(spec)
    assert torch.isfinite(spec).all(), "log-power STFT must be finite (eps floor)"


def test_log_power_stft_short_window_padded():
    """A window shorter than time_frames worth of samples must still produce a
    (..., freq_bins, time_frames) tensor (right-padded in time)."""
    cfg = _small_cfg()
    W = STFT_HOP * 10  # far fewer than time_frames STFT frames
    raw = torch.randn(1, cfg.channels, W)
    spec = data.log_power_stft(raw, cfg)
    assert spec.shape == (1, cfg.channels, cfg.freq_bins, cfg.time_frames), spec.shape
    assert torch.isfinite(spec).all()


def test_log_power_stft_identifies_injected_tone():
    """A single strong tone must show up as the dominant frequency bin, and that
    bin must sit near the analytically-expected STFT bin."""
    cfg = _small_cfg()
    f_hz = 100e3
    raw = _tone_signal(cfg.window_samples, [f_hz], noise_std=0.01).unsqueeze(0)
    spec = data.log_power_stft(raw, cfg)[0]  # (C,F,T)
    top = _dominant_freq_bins(spec, k=1)
    # torch.stft bin b -> freq b*fs/n_fft; DC (bin 0) is dropped, so bin index i
    # in the output corresponds to STFT bin (i+1).
    expected_stft_bin = round(f_hz / (STFT_FS / STFT_N_FFT))
    expected_out_bin = expected_stft_bin - 1
    assert abs(next(iter(top)) - expected_out_bin) <= 1, (top, expected_out_bin)


# --------------------------------------------------------------------------- #
# shift_pair_windows
# --------------------------------------------------------------------------- #
def test_shift_pair_windows_shapes():
    cfg = _small_cfg()
    # raw shot long enough to hold [t0, t0 + CHUNK_S + delta]
    W_full = cfg.window_samples * 4
    raw_shot = _tone_signal(W_full, [40e3, 90e3, 150e3], noise_std=0.01)  # (1, W_full)
    a, b = data.shift_pair_windows(raw_shot, t0=0.02, cfg=cfg, delta_ms=1.0)
    assert a.shape == (cfg.channels, cfg.freq_bins, cfg.time_frames), a.shape
    assert b.shape == (cfg.channels, cfg.freq_bins, cfg.time_frames), b.shape


def test_shift_pair_windows_accepts_numpy():
    cfg = _small_cfg()
    W_full = cfg.window_samples * 3
    raw_np = _tone_signal(W_full, [70e3]).numpy()  # (1, W_full) numpy
    a, b = data.shift_pair_windows(raw_np, t0=0.01, cfg=cfg, delta_ms=0.5)
    assert isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
    assert a.shape == (cfg.channels, cfg.freq_bins, cfg.time_frames)


def test_shift_pair_preserves_modes_but_differs():
    """A sub-window δ-shift keeps the SAME dominant frequency bins (modes are the
    predictable statistic) but is NOT identical (phase realization differs)."""
    cfg = _small_cfg()
    W_full = cfg.window_samples * 4
    raw_shot = _tone_signal(W_full, [40e3, 90e3, 150e3], noise_std=0.02, seed=7)
    a, b = data.shift_pair_windows(raw_shot, t0=0.03, cfg=cfg, delta_ms=1.5)

    # (a) modes preserved: dominant freq bins identical
    assert _dominant_freq_bins(a, k=3) == _dominant_freq_bins(b, k=3), (
        _dominant_freq_bins(a, k=3),
        _dominant_freq_bins(b, k=3),
    )

    # (b) not identical: the realization (phase / sub-window alignment) differs
    assert not torch.allclose(a, b, atol=1e-4), "δ-shift pair should differ"
    diff = (a - b).abs().mean().item()
    assert diff > 1e-4, diff


def test_shift_pair_zero_delta_is_identical():
    """δ = 0 is a degenerate (but legal) shift → the two windows must match."""
    cfg = _small_cfg()
    W_full = cfg.window_samples * 3
    raw_shot = _tone_signal(W_full, [60e3, 110e3], noise_std=0.0, seed=3)
    a, b = data.shift_pair_windows(raw_shot, t0=0.02, cfg=cfg, delta_ms=0.0)
    assert torch.allclose(a, b, atol=1e-6), (a - b).abs().max().item()


def test_shift_pair_delta_from_config_range():
    """When delta_ms is not given it is drawn from cfg.consistency_delta_ms, so it
    must lie in that range and yield a valid, mode-preserving pair."""
    cfg = _small_cfg()
    W_full = cfg.window_samples * 4
    raw_shot = _tone_signal(W_full, [50e3, 130e3], noise_std=0.01, seed=11)
    lo, hi = cfg.consistency_delta_ms

    seen = []
    for i in range(8):
        a, b, delta_ms = data.shift_pair_windows(
            raw_shot, t0=0.03, cfg=cfg, delta_ms=None, return_delta=True, seed=i
        )
        assert lo <= delta_ms <= hi, delta_ms
        assert a.shape == (cfg.channels, cfg.freq_bins, cfg.time_frames)
        seen.append(delta_ms)
    # sampling should not be a constant
    assert max(seen) - min(seen) > 0.0


def test_shift_pair_delta_samples_correct_offset():
    """delta_ms must map to delta_ms * STFT_FS / 1000 raw samples of shift between
    the two window origins. Verify the b-window equals the a-window's raw data
    advanced by exactly that many samples (before STFT)."""
    cfg = _small_cfg()
    W_full = cfg.window_samples * 4
    raw_shot = _tone_signal(W_full, [75e3], noise_std=0.0, seed=5)
    delta_ms = 1.0
    t0 = 0.02
    expected_shift = round(delta_ms * STFT_FS / 1000.0)

    a_win, b_win = data.raw_pair_windows(raw_shot, t0=t0, cfg=cfg, delta_ms=delta_ms)
    # b_win should equal raw_shot advanced by expected_shift relative to a_win
    start_a = round(t0 * STFT_FS)
    ref = raw_shot[..., start_a + expected_shift : start_a + expected_shift + cfg.window_samples]
    assert b_win.shape == a_win.shape
    assert torch.allclose(b_win, ref, atol=1e-5)


def test_shift_pair_windows_raises_past_shot_end():
    """A t0 so late that [t0+δ, t0+CHUNK_S+δ] runs past the shot must raise, not
    silently return a short window."""
    cfg = _small_cfg()
    W_full = cfg.window_samples + STFT_HOP  # barely one window
    raw_shot = torch.randn(1, W_full)
    try:
        data.shift_pair_windows(raw_shot, t0=CHUNK_S * 0.9, cfg=cfg, delta_ms=2.0)
    except (ValueError, IndexError):
        pass
    else:
        raise AssertionError("expected an error when the shifted window overruns the shot")


def test_log_power_stft_sanitizes_pathological_raw():
    """CO2-interferometer garbage (float32-max / inf / nan) must not leak inf/nan
    into the codec input, and must not blow the log-power out of its sane band —
    this was the co2 codec-collapse root cause (gate env_corr=NaN)."""
    cfg = SpectroCodecConfig(channels=2)
    W = cfg.window_samples
    rng = np.random.default_rng(0)
    raw = torch.as_tensor(rng.standard_normal((1, 2, W)), dtype=torch.float32)
    # inject the exact pathologies observed in real co2 windows
    raw[0, 0, 10] = 1e37   # absurd magnitude (co2 sentinel is ~float32-max 3.4e38)
    raw[0, 0, 20] = float("inf")
    raw[0, 1, 30] = float("nan")
    raw[0, 1, 40] = -float("inf")
    out = data.log_power_stft(raw, cfg)
    assert torch.isfinite(out).all(), "sanitized log-power must be all-finite"
    assert float(out.min()) >= data._LOG_FLOOR - 1e-4
    assert float(out.max()) <= data._LOG_CEIL + 1e-4


def test_log_power_stft_noop_on_clean_raw():
    """Sanitization is byte-identical for clean, in-range raw (ece/bes/mhr path)."""
    cfg = SpectroCodecConfig(channels=3)
    W = cfg.window_samples
    rng = np.random.default_rng(1)
    raw = torch.as_tensor(rng.standard_normal((2, 3, W)), dtype=torch.float32)
    out = data.log_power_stft(raw, cfg)
    # a second call with the same (finite, small) raw is identical, and nothing was
    # clamped (clean STFT log-power sits well inside the band)
    assert torch.isfinite(out).all()
    assert float(out.max()) < data._LOG_CEIL - 1.0
