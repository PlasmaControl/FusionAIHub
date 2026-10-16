"""labeler.ae.gpu against the CPU chain it replaces, on CPU torch.

Each test names its tolerance and the largest difference measured when it was
written (pixi labeler env: torch 2.14.0+cpu, numpy 1.26.4, scipy 1.17.1):

- resample_poly vs scipy.signal.resample_poly, float32 and float64: equal bit
  for bit (the same taps, alignment and order of rounded multiplies and adds;
  summing the taps newest first instead changes thousands of values).
- resample vs alfven.resample: equal bit for bit, and the same rate.
- percentiles vs np.percentile: equal exactly (the same float64 arithmetic).
- spectrogram vs transform.spectrogram: 1e-6 absolute (measured 0.0: the FFTs
  agree to float64 rounding, and the float32 cast hides it).
- standardise and model_input vs transform: 1e-5 absolute (measured 4.8e-7):
  numpy accumulates the per-chord mean and std in float32, gpu in float64.
- run_windows vs torch_pt.run_windows: 1e-5 absolute (measured 1.5e-8, and 0.0
  where every window is evaluated alone): batching only reorders the sums
  inside a convolution.
- read_co2 + input_signals vs full.input_signals: equal exactly (samples, the
  first sample's time, the rate), whether read_co2 read only the span or, for a
  record that stops before the window end, the whole record.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy import signal

from labeler.ae import full, gpu
from labeler.ae import transform as tr
from labeler.ae.model import AeSeldNet, AeSeldNetConfig
from labeler.events import raw
from labeler.events.review import alfven
from labeler.features.store import FeatureArray
from labeler.models.runners import torch_pt

CACHE_HZ = 1e6 / 0.6  # the raw cache's CO2 rate, ~1.667 MHz
LENGTHS = (1, 2, 5, 37, 200, 1001, 4097)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("up,down", [(3, 10), (1, 1), (2, 3), (5, 7), (1, 4), (4, 1), (10, 3), (6, 4)])
def test_resample_poly_matches_scipy(up, down, dtype):
    rng = np.random.default_rng(up * 100 + down)
    for n in (*LENGTHS, 30001):
        x = rng.standard_normal((2, n)).astype(dtype)
        ref = signal.resample_poly(x, up, down, axis=-1)
        got = gpu.resample_poly(torch.from_numpy(x), up, down)
        assert got.numpy().dtype == ref.dtype
        assert tuple(got.shape) == ref.shape, (n, tuple(got.shape), ref.shape)
        assert np.array_equal(got.numpy(), ref), n
        one = gpu.resample_poly(torch.from_numpy(x[0]), up, down)  # 1-D in, 1-D out
        assert np.array_equal(one.numpy(), ref[0])


def test_fir_is_scipys():
    # resample_poly's own filter for float32 input: firwin in float64, cast, times up in float32.
    for up, down in ((3, 10), (2, 3)):
        h = signal.firwin(2 * 10 * max(up, down) + 1, 1.0 / max(up, down), window=("kaiser", 5.0))
        h = h.astype(np.float32)
        h *= up
        got = gpu.fir(up, down)
        assert got.dtype == np.float32 and np.array_equal(got, h)


@pytest.mark.parametrize("fs_in", [CACHE_HZ, 5e5])
def test_resample_matches_alfven(fs_in):
    rng = np.random.default_rng(int(fs_in))
    for n in (100, 3001, 50000, 200001):
        t = -10.0 + np.arange(n) * 1000 / fs_in
        x = rng.standard_normal((4, n)).astype(np.float32)
        ref, fs_ref = alfven.resample(t, x)
        got, fs_got = gpu.resample(t, torch.from_numpy(x))
        assert fs_got == fs_ref
        assert got.dtype == torch.float32 and tuple(got.shape) == ref.shape
        assert np.array_equal(got.numpy(), ref), n


def test_percentiles_match_numpy():
    rng = np.random.default_rng(3)
    # 101 and 151 put the 1st percentile's virtual index on a whole and a half step;
    # the rounded draws make ties.
    for n in (1, 2, 3, 100, 101, 151, 1000, 12345):
        for v in (rng.standard_normal(n), np.round(rng.standard_normal(n), 1)):
            ref = np.percentile(v, [tr.CLIP_LOW, tr.CLIP_HIGH])
            got = gpu.percentiles(torch.from_numpy(v)).numpy()
            assert np.array_equal(got, ref), (n, got, ref)


def test_spectrogram_matches_transform():
    rng = np.random.default_rng(4)
    for n in (512, 1000, 5000, 33333):
        x = rng.standard_normal((2, n)) * np.array([[1.0], [30.0]])
        ref = tr.spectrogram(x)
        got = gpu.spectrogram(torch.from_numpy(x))
        assert got.dtype == torch.float32 and tuple(got.shape) == ref.shape
        assert ref.shape[-1] == tr.n_frames(n)
        assert np.abs(got.numpy() - ref).max() <= 1e-6, n


def test_short_record_refused_as_scipy_does():
    x = np.zeros(511)
    with pytest.raises(ValueError) as cpu:
        tr.compute_stft(x)
    with pytest.raises(ValueError) as here:
        gpu.compute_stft(torch.from_numpy(x))
    assert str(here.value) == str(cpu.value)


def test_standardise_and_band_match_transform():
    rng = np.random.default_rng(5)
    raw_spec = (3.0 + rng.standard_normal((4, 512, 300)) * np.array([1.0, 0.5, 2.0, 0.1])[:, None, None])
    raw_spec = raw_spec.astype(np.float32)
    ref = tr.standardise(raw_spec)
    got = gpu.standardise(torch.from_numpy(raw_spec))
    assert got.dtype == torch.float32 and np.abs(got.numpy() - ref).max() <= 1e-5
    band = gpu.restrict_to_band(got)
    assert tuple(band.shape) == (4, 348, 300)
    assert np.array_equal(band.numpy(), tr.restrict_to_band(got.numpy()))
    with pytest.raises(ValueError):
        gpu.restrict_to_band(got[:, :500])


def test_model_input_matches_transform():
    x = np.random.default_rng(6).standard_normal((4, 30000))
    ref = tr.model_input(x)
    got = gpu.model_input(torch.from_numpy(x))
    assert got.dtype == torch.float32 and tuple(got.shape) == ref.shape
    assert np.abs(got.numpy() - ref).max() <= 1e-5


def _small_net() -> AeSeldNet:
    torch.manual_seed(0)
    net = AeSeldNet(AeSeldNetConfig(in_channels=4, n_freq=24, pool_sizes=(2, 3, 2), conv_channels=4,
                                    rnn_sizes=(8, 8), fnn_size=8))
    for m in net.modules():  # eval-mode batch norm with statistics that are not the identity
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.uniform_(-0.5, 0.5)
            m.running_var.uniform_(0.5, 2.0)
    return net.eval()


@pytest.mark.parametrize("frames,window,context,batch", [
    (1000, 64, 16, None),   # full windows in batches, a short last window
    (256, 64, 16, 3),       # a whole number of windows, batches of 3 with a remainder
    (50, 64, 16, None),     # one window, shorter than window + 2 context: an edge window only
    (3000, 1024, 128, 1),   # the card's window and context, one window a pass
    (3000, 1024, 128, None),
])
def test_run_windows_matches_torch_pt(frames, window, context, batch):
    net = _small_net()
    x = np.random.default_rng(frames).standard_normal((4, frames, 24)).astype(np.float32)
    ref = torch_pt.run_windows(net, x, window=window, context=context)
    got = gpu.run_windows(net, torch.from_numpy(x), window=window, context=context, batch=batch)
    assert tuple(got.shape) == ref.shape
    assert np.abs(got.numpy() - ref).max() <= 1e-5


def test_window_plan_is_run_windows_plan():
    assert gpu.window_plan(2100, 1024, 128) == [
        (0, 1024, 0, 1152), (1024, 2048, 896, 2100), (2048, 2100, 1920, 2100)]
    with pytest.raises(ValueError):
        gpu.window_plan(10, 0, 0)


def _fake_co2(monkeypatch, fs_hz, start_ms, stop_ms, chords=4):
    """raw.raw_signal over one made-up record, its t_range cut as corpus_signal's
    (searchsorted on the times); returns the list of t_range each call asked for."""
    x_ms = start_ms + np.arange(int((stop_ms - start_ms) * fs_hz / 1000)) * 1000 / fs_hz
    y = (np.random.default_rng(int(fs_hz)).standard_normal((chords, x_ms.size)) * 1e19).astype(np.float32)
    asked = []

    def fake(shot, group, *, t_range=None, **kw):
        asked.append(t_range)
        if t_range is None:
            return FeatureArray(x=x_ms, y=y, attrs={"tier": "cache"})
        lo = int(np.searchsorted(x_ms, t_range[0], side="left"))
        hi = int(np.searchsorted(x_ms, t_range[1], side="right"))
        if hi <= lo:
            raise raw.WindowEmptyError("nothing in the window")
        return FeatureArray(x=x_ms[lo:hi], y=y[:, lo:hi], attrs={"tier": "cache"})

    monkeypatch.setattr(raw, "raw_signal", fake)
    return asked


@pytest.mark.parametrize("fs_hz", [CACHE_HZ, 5e5])
@pytest.mark.parametrize("end_ms,whole", [(250.0, False), (1000.0, True)])  # inside the record; past its end
def test_input_signals_match_full(monkeypatch, fs_hz, end_ms, whole):
    _fake_co2(monkeypatch, fs_hz, -50.0, 400.0)
    monkeypatch.setattr(full, "CROP_END_MS", end_ms)
    ref, t0, fs = full.input_signals(1, paths=None)
    monkeypatch.undo()
    asked = _fake_co2(monkeypatch, fs_hz, -50.0, 400.0)
    window = gpu.read_co2(1, paths=None, end_ms=end_ms)
    assert asked == [(-full.MARGIN_MS, end_ms + full.MARGIN_MS)] + ([None] if whole else [])
    assert window["chords"].dtype == np.float32
    got, t0_here, fs_here = gpu.input_signals(window)
    assert t0_here == t0 and fs_here == fs
    assert got.dtype == torch.float32 and tuple(got.shape) == ref.shape
    assert np.array_equal(got.numpy(), ref.astype(np.float32))


def test_read_co2_reads_whole_when_the_span_is_empty(monkeypatch):
    # The record starts after the window: input_signals ends in an IndexError on an
    # empty time vector, read_co2 in a ValueError naming the span.
    asked = _fake_co2(monkeypatch, 5e5, 500.0, 900.0)
    with pytest.raises(ValueError, match="no CO2 sample in"):
        gpu.read_co2(1, paths=None, end_ms=200.0)
    assert asked == [(-full.MARGIN_MS, 210.0), None]


def test_read_co2_refuses_what_full_refuses(monkeypatch):
    _fake_co2(monkeypatch, 5e5, -50.0, 100.0, chords=3)
    with pytest.raises(ValueError) as cpu:
        full.input_signals(7, paths=None)
    with pytest.raises(ValueError) as here:
        gpu.read_co2(7, paths=None)
    assert str(here.value) == str(cpu.value)
