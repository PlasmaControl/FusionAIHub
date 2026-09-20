"""``shot_design.simulate.decode`` -- band-power frequency reduction and codec decode.

Per controller ruling on task D2: the brief left ``decode_modalities`` untested.
This file covers the pure-numpy band-power reduction (the piece the brief
specifically calls out: "10-60 kHz mean-abs reduction returns (F, C)") plus the
frequency-axis derivation it depends on, and a decode-through-a-real-codec smoke
test built on a TINY (freq_bins=16, d_model=64, depth=1) SpectroCodec -- cheap
enough to construct untrained in a unit test (no checkpoint, no GPU; see
decode.py's ``freq_axis_khz`` docstring for why the mapping needs
``cfg.band_pool == 0`` to be valid).
"""

from types import SimpleNamespace

import numpy as np
import torch

from shot_design.simulate import core, decode


def test_freq_axis_khz_matches_stft_bin_math():
    # STFT_FS=500kHz, n_fft=1024 -> bin spacing 500000/1024 = 488.28125 Hz. Bin i
    # (0-indexed, DC already dropped upstream) is raw STFT bin i+1, so
    # freq(i) = (i+1) * bin_hz.
    cfg = SimpleNamespace(stft_n_fft=1024, freq_bins=4, band_pool=0)
    axis = decode.freq_axis_khz(cfg)
    expected = np.array([1, 2, 3, 4]) * (500_000.0 / 1024) / 1000.0
    np.testing.assert_allclose(axis, expected)


def test_freq_axis_khz_none_when_band_pooled():
    # band_pool > 0 mean-pools bins unevenly from the codec's perspective -- the
    # axis is no longer recoverable from freq_bins/stft_n_fft alone, so the
    # fallback is None.
    cfg = SimpleNamespace(stft_n_fft=1024, freq_bins=4, band_pool=8)
    assert decode.freq_axis_khz(cfg) is None


def test_band_power_reduces_within_band_to_F_C():
    # (F=2, C=1, Fr=4, Tb=3): freq bins at 5, 15, 35, 65 kHz. The 10-60 kHz band
    # keeps only bins 1 and 2 (15, 35 kHz); band power is their mean |value|
    # over those bins and time.
    freq_khz = np.array([5.0, 15.0, 35.0, 65.0])
    dec = np.zeros((2, 1, 4, 3), dtype=np.float32)
    dec[:, :, 1, :] = 2.0   # 15 kHz bin
    dec[:, :, 2, :] = -6.0  # 35 kHz bin (abs -> 6.0)
    dec[:, :, 0, :] = 100.0  # outside the band -- must be excluded
    dec[:, :, 3, :] = -100.0
    out = decode.band_power(dec, freq_khz, (10.0, 60.0))
    assert out.shape == (2, 1)
    np.testing.assert_allclose(out, np.full((2, 1), (2.0 + 6.0) / 2.0))


def test_band_power_falls_back_to_full_band_when_axis_unrecoverable():
    dec = np.full((2, 1, 4, 3), 3.0, dtype=np.float32)
    out = decode.band_power(dec, None, (10.0, 60.0))
    assert out.shape == (2, 1)
    np.testing.assert_allclose(out, np.full((2, 1), 3.0))


def _tiny_spectro_codec():
    """A minuscule untrained SpectroCodec (n_tok=4, d_model=64) -- fast on CPU."""
    from tokamak_foundation_model.ignite.codec import SpectroCodec
    from tokamak_foundation_model.ignite.config import SpectroCodecConfig

    cfg = SpectroCodecConfig(
        channels=1, freq_bins=16, time_frames=4, patch_f=4, patch_t=4,
        d_model=64, enc_depth=1, dec_depth=1, heads=1,
    )
    return SpectroCodec(cfg).eval(), cfg


def test_decode_modalities_reduces_spectro_to_F_C():
    codec, cfg = _tiny_spectro_codec()
    n_tok = (cfg.freq_bins // cfg.patch_f) * (cfg.time_frames // cfg.patch_t)
    codebook = codec.quantizer.codebook_size
    F = 5

    def flat():
        return torch.randint(0, codebook, (F, n_tok), dtype=torch.int64)

    # "mhr" exercises the 10-60 kHz banded path; "co2" (not in the banded set)
    # exercises the full-band fallback -- both from the SAME tiny codec, aliased
    # under two names.
    codecs = {"mhr": (codec, cfg, "spectro"), "co2": (codec, cfg, "spectro")}
    arms = core.SimulationArms(
        seed_frames=2, predict_frames=3,
        real={"mhr": flat(), "co2": flat()},
        proposed={"mhr": flat(), "co2": flat()},
        gt={"mhr": flat(), "co2": flat()},
        divergence_vs_real={}, token_accuracy={}, persistence_accuracy={},
    )
    out = decode.decode_modalities(codecs, arms, ["mhr", "co2"])
    assert set(out) == {"mhr", "co2"}
    for m in ("mhr", "co2"):
        assert set(out[m]) == {"real", "proposed", "gt"}
        for arr in out[m].values():
            assert arr.shape == (F, cfg.channels)
            assert arr.dtype == np.float32


def test_decode_modalities_skips_names_without_a_codec():
    codec, cfg = _tiny_spectro_codec()
    n_tok = (cfg.freq_bins // cfg.patch_f) * (cfg.time_frames // cfg.patch_t)
    codebook = codec.quantizer.codebook_size
    codes = torch.randint(0, codebook, (5, n_tok), dtype=torch.int64)
    arms = core.SimulationArms(
        seed_frames=2, predict_frames=3,
        real={"mhr": codes}, proposed={"mhr": codes}, gt={"mhr": codes},
        divergence_vs_real={}, token_accuracy={}, persistence_accuracy={},
    )
    codecs = {"mhr": (codec, cfg, "spectro")}
    out = decode.decode_modalities(codecs, arms, ["mhr", "not_loaded"])
    assert set(out) == {"mhr"}
