"""RAW per-channel input standardization for non-O(1)-raw spectros (co2).

co2's raw is ~1e13 (unnormalized interferometer counts) -> log10(mag^2) ~ 24, which the
log-power ceiling (_LOG_CEIL=20) clips to a flat plate (100% clipped), destroying the signal
(true std ~1.08). The fix z-scores the RAW per channel BEFORE the STFT (a per-channel linear
rescale) so log-power lands un-clipped in [-10, 20] with all structure preserved. Uses the FM's
preprocessing_stats[modality]['raw'] (raw is raw -> no log-unit mismatch). No-op for ece/bes/mhr.
"""

import math

import numpy as np
import torch

import tokamak_foundation_model.ignite.train_codec as tc
from tokamak_foundation_model.ignite import data as idata
from tokamak_foundation_model.ignite.config import SpectroCodecConfig


def _write_stats(path, modality, mean, std):
    torch.save({modality: {"raw": {"mean": np.asarray(mean, np.float32),
                                   "std": np.asarray(std, np.float32)}}}, path)


def test_apply_standardization_sets_raw_fields(tmp_path):
    p = tmp_path / "stats.pt"
    _write_stats(p, "co2", [0.0, 0.0, 0.0, 0.0], [6e13, 6e13, 6e13, 6e13])
    cfg = SpectroCodecConfig(channels=4)
    tc.apply_spectro_standardization(cfg, "co2", stats_path=str(p))
    assert cfg.input_standardize is True
    assert np.asarray(cfg.raw_mean).shape == (4,)
    assert np.allclose(cfg.raw_std, 6e13)


def test_standardization_noop_for_rich_spectros(tmp_path):
    p = tmp_path / "stats.pt"
    _write_stats(p, "ece", [0.0], [1.0])
    for m in ("ece", "bes", "mhr"):
        cfg = SpectroCodecConfig(channels=40)
        tc.apply_spectro_standardization(cfg, m, stats_path=str(p))
        assert cfg.input_standardize is False
        assert cfg.raw_mean is None


def test_bad_channel_stats_sanitized(tmp_path):
    # a non-finite / zero-std channel must be neutralized (center 0, scale 1), not crash.
    p = tmp_path / "stats.pt"
    _write_stats(p, "co2", [0.0, np.inf, 5.0, np.nan], [6e13, np.nan, 0.0, 6e13])
    cfg = SpectroCodecConfig(channels=4)
    tc.apply_spectro_standardization(cfg, "co2", stats_path=str(p))
    assert cfg.input_standardize is True
    rm, rs = np.asarray(cfg.raw_mean), np.asarray(cfg.raw_std)
    assert np.isfinite(rm).all() and np.isfinite(rs).all()
    assert (rs > 0).all()
    assert rm[1] == 0.0 and rs[1] == 1.0   # inf mean / nan std -> neutral
    assert rs[2] == 1.0                     # zero std -> neutral


def test_missing_stats_leaves_off(tmp_path):
    cfg = SpectroCodecConfig(channels=4)
    tc.apply_spectro_standardization(cfg, "co2", stats_path=str(tmp_path / "nope.pt"))
    assert cfg.input_standardize is False


def _bigsig(amp):
    t = torch.arange(30_000, dtype=torch.float32) / 500_000.0
    s = torch.sin(2 * math.pi * 40e3 * t) + 0.3 * torch.sin(2 * math.pi * 90e3 * t)
    return (s * amp)[None, None, :].repeat(1, 2, 1)  # (1, 2, W)


def test_raw_standardization_unclips_large_raw():
    cfg = SpectroCodecConfig(channels=2)
    raw = _bigsig(6e13)  # co2-scale amplitude
    base = idata.log_power_stft(raw, cfg)                 # unstandardized -> clips at ceiling
    assert (base >= 19.9).float().mean() > 0.5            # mostly pinned at _LOG_CEIL=20
    cfg.input_standardize = True
    cfg.raw_mean = [0.0, 0.0]
    cfg.raw_std = [6e13, 6e13]
    out = idata.log_power_stft(raw, cfg)                  # raw z-scored BEFORE STFT
    assert float(out.max()) < 20.0                        # un-clipped now
    assert float(out.std()) > float(base.std())           # real structure recovered (base was flat)


def test_standardization_gated_off_ignores_stats():
    cfg = SpectroCodecConfig(channels=2)  # input_standardize False
    raw = _bigsig(6e13)
    a = idata.log_power_stft(raw, cfg)
    cfg.raw_mean = [0.0, 0.0]
    cfg.raw_std = [6e13, 6e13]  # set but gate OFF
    b = idata.log_power_stft(raw, cfg)
    assert torch.allclose(a, b)


def test_co2_in_standardize_registry():
    assert "co2" in tc._SPECTRO_STANDARDIZE_SIGNALS
    for m in ("ece", "bes", "mhr"):
        assert m not in tc._SPECTRO_STANDARDIZE_SIGNALS
