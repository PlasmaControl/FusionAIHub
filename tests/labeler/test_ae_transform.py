"""The ported AE spectrogram transform, checked against what it must be.

Hermetic: no tokeye, no data root, no real record. The pin against tokeye's
own output on a real corpus record is a script, not a test -
`outputs/labelmaker/ae/scripts/pin_transform.py`, whose measured result (max
abs diff 0.0 at every stage) is quoted in the model card.
"""
import numpy as np
import pytest
from scipy import signal

from labelmaker.ae import transform as tr


def _sine(f_khz: float, *, fs_hz: float = 5.0e5, n: int = 60_000,
          amplitude: float = 1.0) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / fs_hz
    return amplitude * np.sin(2.0 * np.pi * f_khz * 1e3 * t)


def test_a_sine_lands_in_its_own_frequency_bin():
    # 120 kHz is inside the band; on the 500 kHz grid the bins are
    # (i + 1) * fs / 1024, so 120 kHz is bin round(120e3 * 1024 / 500e3) - 1.
    spec = tr.compute_stft(_sine(120.0))
    assert spec.shape[0] == tr.N_BINS
    # Look at an interior frame, away from the padded ends.
    column = spec[:, spec.shape[1] // 2]
    peak = int(np.argmax(column))
    freqs = np.arange(tr.N_BINS) * 0.0
    freqs = (np.arange(tr.N_BINS, dtype=np.float64) + 1.0) * 500.0 / tr.N_FFT
    assert abs(freqs[peak] - 120.0) < 1.0


def test_the_stft_reproduces_scipy_directly():
    """The port is `ShortTimeFFT` plus five lines; pin it to those five lines."""
    x = _sine(150.0, n=20_000)
    win = signal.get_window("hann", tr.N_FFT)
    sft = signal.ShortTimeFFT(win, hop=tr.HOP, fs=1.0)
    want = np.log1p(np.abs(sft.stft(x[None, :])[0]))[1:, :]
    lo, hi = np.percentile(want, [tr.CLIP_LOW, tr.CLIP_HIGH])
    want = np.clip(want, lo, hi).astype(np.float32)
    np.testing.assert_array_equal(tr.compute_stft(x), want)


def test_dc_is_dropped_and_the_band_is_348_bins():
    spec = tr.spectrogram(np.stack([_sine(120.0), _sine(200.0)]))
    assert spec.shape[:2] == (2, tr.N_BINS)
    band = tr.restrict_to_band(spec)
    assert band.shape[:2] == (2, 348)
    assert tr.band_freqs_khz().shape == (348,)
    # The band's edges are the numbers the card and the labels claim.
    assert tr.band_freqs_khz()[0] == pytest.approx(80.566, abs=0.01)
    assert tr.band_freqs_khz()[-1] == pytest.approx(250.0, abs=0.01)


def test_standardisation_is_per_channel_over_the_whole_array():
    # Two channels at very different amplitudes: after standardising, each
    # must be zero-mean and unit-variance on its own, not jointly.
    raw = tr.spectrogram(np.stack([_sine(120.0), _sine(120.0, amplitude=1e4)]))
    norm = tr.standardise(raw)
    for ch in range(2):
        assert norm[ch].mean() == pytest.approx(0.0, abs=1e-4)
        assert norm[ch].std() == pytest.approx(1.0, rel=1e-3)
    assert norm.dtype == np.float32


def test_standardising_before_band_restriction_is_not_the_same_as_after():
    """The order in `model_input` is load-bearing; prove the two differ."""
    raw = tr.spectrogram(np.stack([_sine(120.0), _sine(200.0)]))
    ours = tr.restrict_to_band(tr.standardise(raw))
    other = tr.standardise(tr.restrict_to_band(raw))
    assert not np.allclose(ours, other)


def test_frame_times_match_the_task_7a_grid():
    # 1,000,001 samples over 2,000 ms is the dataset script's record, and it
    # produced exactly 7,820 frames.
    assert tr.n_frames(1_000_001) == 7820
    t = tr.frame_times_s(1_000_001, 1_000_001 / 2.0)
    assert t.size == 7820
    # The STFT pads the start, so the first frames are before sample 0.
    assert t[0] < 0.0
    np.testing.assert_allclose(np.diff(t), tr.HOP / (1_000_001 / 2.0))


def test_model_input_is_the_three_steps_composed():
    signals = np.stack([_sine(120.0 + 10.0 * ch, n=20_000) for ch in range(4)])
    got = tr.model_input(signals)
    want = tr.restrict_to_band(tr.standardise(tr.spectrogram(signals)))
    np.testing.assert_array_equal(got, want)
    assert got.shape[:2] == (4, 348)


def test_a_two_row_input_is_refused_by_compute_stft():
    """tokeye would form a cross-spectrum; task 7a never wants one."""
    with pytest.raises(ValueError, match="one channel"):
        tr.compute_stft(np.zeros((2, 4096)))
