"""Signed spectrogram reduction, frequency axes and codec decoding."""

from types import SimpleNamespace

import numpy as np
import pytest
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


def test_freq_axis_khz_puts_a_pooled_band_at_its_bins_mean_frequency():
    # band_pool=2 over 4 bins: bands {1, 2} and {3, 4} at 1.5 and 3.5 bin widths.
    cfg = SimpleNamespace(stft_n_fft=1024, freq_bins=4, band_pool=2)
    np.testing.assert_allclose(
        decode.freq_axis_khz(cfg), np.array([1.5, 3.5]) * (500_000.0 / 1024) / 1000.0
    )


def test_band_power_reduces_within_band_to_F_C():
    # (F=2, C=1, Fr=4, Tb=3): freq bins at 5, 15, 35, 65 kHz. The 10-60 kHz band
    # keeps only bins 1 and 2 (15, 35 kHz); the reduction is their signed mean
    # over those bins and time.
    freq_khz = np.array([5.0, 15.0, 35.0, 65.0])
    dec = np.zeros((2, 1, 4, 3), dtype=np.float32)
    dec[:, :, 1, :] = 2.0   # 15 kHz bin
    dec[:, :, 2, :] = -6.0  # 35 kHz bin
    dec[:, :, 0, :] = 100.0  # outside the band -- must be excluded
    dec[:, :, 3, :] = -100.0
    out = decode.band_power(dec, freq_khz, (10.0, 60.0))
    assert out.shape == (2, 1)
    np.testing.assert_allclose(out, np.full((2, 1), (2.0 - 6.0) / 2.0))


@pytest.mark.parametrize("band", [None, (200.0, 300.0)])
def test_band_power_is_the_full_band_without_a_band_or_rows_inside_it(band):
    dec = np.full((2, 3, 4, 5), -2.0, dtype=np.float32)
    dec[:, :, 0] = 2.0
    out = decode.band_power(dec, np.array([5.0, 15.0, 35.0, 65.0]), band)
    np.testing.assert_allclose(out, np.full((2, 3), -1.0))


def _tiny_spectro_codec():
    """A minuscule untrained SpectroCodec (n_tok=4, d_model=64) -- fast on CPU."""
    from tokamak_foundation_model.ignite.codec import SpectroCodec
    from tokamak_foundation_model.ignite.config import SpectroCodecConfig

    cfg = SpectroCodecConfig(
        channels=1, freq_bins=16, time_frames=4, patch_f=4, patch_t=4,
        d_model=64, enc_depth=1, dec_depth=1, heads=1,
    )
    return SpectroCodec(cfg).eval(), cfg


def _ensemble(n_tok, codebook, members=3, frames=5, held=()):
    g = torch.Generator().manual_seed(0)

    def codes(*shape):
        return torch.randint(0, codebook, (*shape, frames, n_tok), generator=g)

    names = ("mhr", "co2")
    return core.Ensemble(
        k0=2,
        gt={m: codes() for m in names},
        arms={arm: {m: codes(members) for m in names} for arm in ("real", "null")},
        seeds={"real": 0, "null": members},
        held=held,
        batch=1,
    )


def test_an_ensemble_decodes_to_one_feature_per_member_frame_and_channel():
    codec, cfg = _tiny_spectro_codec()
    n_tok = (cfg.freq_bins // cfg.patch_f) * (cfg.time_frames // cfg.patch_t)
    ens = _ensemble(n_tok, codec.quantizer.codebook_size)
    # "mhr" takes the 10-60 kHz band, "co2" the full band; one codec serves both
    codecs = {"mhr": (codec, cfg, "spectro"), "co2": (codec, cfg, "spectro")}
    out = decode.decode_ensemble(codecs, ens)
    assert set(out) == {"mhr", "co2"}
    for per in out.values():
        assert set(per) == {"gt", "real", "null"}
        assert per["gt"].shape == (5, cfg.channels)
        assert per["real"].shape == per["null"].shape == (3, 5, cfg.channels)
        assert per["real"].dtype == np.float32


def test_a_feature_is_the_same_whether_decoded_alone_or_in_a_stack():
    codec, cfg = _tiny_spectro_codec()
    n_tok = (cfg.freq_bins // cfg.patch_f) * (cfg.time_frames // cfg.patch_t)
    ens = _ensemble(n_tok, codec.quantizer.codebook_size, frames=20)  # > one chunk
    tokens = ens.arms["real"]["co2"]
    stacked = decode.features(codec, cfg, "spectro", "co2", tokens)
    alone = decode.features(codec, cfg, "spectro", "co2", tokens[1])
    np.testing.assert_allclose(stacked[1], alone, rtol=1e-5, atol=1e-6)


def test_held_modalities_and_those_without_a_codec_are_skipped():
    codec, cfg = _tiny_spectro_codec()
    n_tok = (cfg.freq_bins // cfg.patch_f) * (cfg.time_frames // cfg.patch_t)
    ens = _ensemble(n_tok, codec.quantizer.codebook_size, held=("co2",))
    codecs = {"co2": (codec, cfg, "spectro"), "mhr": (codec, cfg, "spectro")}
    assert set(decode.decode_ensemble(codecs, ens)) == {"mhr"}
    assert set(decode.decode_ensemble({"co2": codecs["co2"]}, ens)) == set()


@pytest.mark.parametrize(
    ("name", "family", "feature"),
    [
        ("mhr", "spectro", "10-60 kHz band power"),
        ("ece", "spectro", "band power"),
        ("tangtv_lower", "video", "frame mean"),
        ("mse", "slowts", "intra-frame mean"),
    ],
)
def test_each_feature_has_a_name(name, family, feature):
    assert decode.feature_name(name, family) == feature


def test_reduce_video_returns_per_frame_mean():
    # (F=2, C=1, Tv=2, H=2, W=2) -- values chosen so the mean is hand-computable.
    dec = np.zeros((2, 1, 2, 2, 2), dtype=np.float32)
    dec[0] = 1.0  # frame 0 mean = 1.0
    dec[1] = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]).reshape(1, 2, 2, 2)
    out = decode._reduce_video(dec)
    assert out.shape == (2, 1)
    np.testing.assert_allclose(out, np.array([[1.0], [7.0]]), atol=1e-6)


def test_reduce_series_returns_per_frame_channel_mean():
    # (F=2, C=2, Tt=3): each (frame, channel) row averaged over the trailing time axis.
    dec = np.array(
        [
            [[1.0, 2.0, 3.0], [4.0, 4.0, 4.0]],
            [[0.0, 0.0, 0.0], [10.0, 20.0, 30.0]],
        ],
        dtype=np.float32,
    )
    out = decode._reduce_series(dec)
    assert out.shape == (2, 2)
    np.testing.assert_allclose(out, np.array([[2.0, 4.0], [0.0, 20.0]]))


def test_reduce_series_passes_through_when_already_F_C():
    dec = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    out = decode._reduce_series(dec)
    np.testing.assert_allclose(out, dec)
