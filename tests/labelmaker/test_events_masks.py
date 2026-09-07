"""Corpus waveform -> TokEye spectrogram -> tiles -> U-Net -> packed masks.

The whole path a mask run walks, one stage per group of tests. Two of the
stages are arithmetic that must not drift (`prep` IS the pinned TokEye
transform, and the frequency and time axes are what every event's kHz and
seconds come from), two are plumbing that must not lose anything
(`tile`/`stitch` round-trips, `pack`/`unpack` round-trips), and one is a
storage contract that later tasks read by name (`block_arrays`' key
suffixes and dtypes, `write_masks`' merge).

No GPU and no real model here: `infer`'s CUDA, AMP and out-of-memory paths
are exercised with a fake model that records what it was asked for and can
be told to run out of memory, which is the only way to test a retry policy
without a card to exhaust.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import h5py
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from labelmaker.ae import labels as ae_labels  # noqa: E402
from labelmaker.ae import transform  # noqa: E402
from labelmaker.events import masks, schema  # noqa: E402

#: Shot 198658, the pilot shot the A1 constants were measured on.
REAL_SHOT = 198658
REAL_FILE = Path(f"/scratch/gpfs/EKOLEMEN/foundation_model/{REAL_SHOT}_processed.h5")

needs_corpus = pytest.mark.skipif(
    not REAL_FILE.exists(), reason=f"corpus file not mounted: {REAL_FILE}"
)


def _corpus(path, group="mhr", *, y, x):
    """A corpus-layout file holding one group."""
    with h5py.File(path, "w") as f:
        g = f.create_group(group)
        g.create_dataset("ydata", data=np.asarray(y, dtype=np.float32))
        g.create_dataset("xdata", data=np.asarray(x, dtype=np.float32))
    return path


def _noise(n, seed=0):
    return np.random.default_rng(seed).standard_normal(n).astype(np.float32)


class FakeUNet:
    """Returns logits, like the vendored model, and remembers the batches.

    `max_batch` makes it raise `torch.cuda.OutOfMemoryError` above a batch
    size, which is how the retry policy is tested without a GPU to fill.
    The logits are `+/- gain * x`, so the returned probabilities are a known
    monotone function of the input and a wrong tile order shows up.
    """

    def __init__(self, *, max_batch: int | None = None, gain: float = 40.0):
        self.max_batch = max_batch
        self.gain = gain
        self.seen: list[int] = []
        self.dtypes: list[torch.dtype] = []

    def __call__(self, x):
        self.seen.append(int(x.shape[0]))
        self.dtypes.append(x.dtype)
        if self.max_batch is not None and x.shape[0] > self.max_batch:
            raise torch.cuda.OutOfMemoryError("fake CUDA out of memory")
        return (torch.stack([self.gain * x[:, 0], -self.gain * x[:, 0]], dim=1),)


# ---------------------------------------------------------------- constants

def test_the_threshold_is_the_one_the_ae_labels_use():
    # Not redefined here: a mask thresholded at one number and a label at
    # another would be two different masks with one name.
    assert masks.PROB_THRESHOLD is ae_labels.PROB_THRESHOLD
    assert masks.PROB_THRESHOLD == 0.2


def test_the_tiling_constants_are_the_measured_ones():
    assert (masks.TILE, masks.OVERLAP) == (512, 64)
    assert masks.STRIDE == 448
    assert masks.ZOOM_DECIM == 4
    assert masks.N_BANDS == 16
    assert masks.N_BINS == transform.N_BINS == 512


def test_the_pass_names_are_ones_an_event_row_can_carry():
    assert masks.PASS_NAMES == ("wide", "zoom")
    assert set(masks.PASS_NAMES) <= set(schema.PASS_NAMES)


# ------------------------------------------------------------ read_waveform

def test_a_trailing_nan_is_stripped_from_both_axes(tmp_path):
    # Every fast group's arrays are 2^k+1 samples and the last one is NaN.
    # Left in, the transform's percentile is NaN, the standardisation is
    # NaN, and the U-Net returns a blank mask with no error at all.
    y = np.arange(9, dtype=np.float32)[None, :].repeat(2, axis=0)
    y[:, -1] = np.nan
    path = _corpus(tmp_path / "a.h5", y=y, x=np.arange(9) / 8.0)
    got, fs, t0, t1 = masks.read_waveform(path, "mhr", 1)
    assert got.dtype == np.float32
    assert got.tolist() == list(range(8))
    assert (t0, t1) == pytest.approx((0.0, 7.0 / 8.0))
    assert fs == pytest.approx(8.0)             # (8-1) samples / (7/8) s


def test_several_trailing_non_finite_samples_all_go(tmp_path):
    y = np.zeros((1, 10), dtype=np.float32)
    y[0, -3:] = [np.nan, np.inf, -np.inf]
    path = _corpus(tmp_path / "a.h5", y=y, x=np.arange(10) / 10.0)
    got, fs, t0, t1 = masks.read_waveform(path, "mhr", 0)
    assert got.size == 7
    assert (t0, t1) == pytest.approx((0.0, 0.6))
    assert fs == pytest.approx(10.0)


def test_leading_non_finite_samples_are_stripped_too(tmp_path):
    # `mirnov` on 198658 carries 1,669,828 NaN samples BEFORE the digitiser
    # is live. The transform cannot tell them from a signal of zero, and a
    # `t0_s` taken from the file's first sample would put every column of
    # the spectrogram three seconds early.
    y = np.full((1, 10), np.nan, dtype=np.float32)
    y[0, 3:] = np.arange(7)
    path = _corpus(tmp_path / "a.h5", y=y, x=np.arange(10) / 10.0)
    got, fs, t0, t1 = masks.read_waveform(path, "mhr", 0)
    assert got.tolist() == list(range(7))
    assert (t0, t1) == pytest.approx((0.3, 0.9))
    assert fs == pytest.approx(10.0)            # (7-1) samples / 0.6 s


def test_both_ends_are_stripped_and_the_time_axis_follows(tmp_path):
    # The returned span has to be the span of the samples that came back,
    # not of the file: `fs_hz` is derived from it.
    n = 20
    y = np.full((1, n), np.nan, dtype=np.float32)
    y[0, 4:15] = np.arange(11)
    x = 0.5 + np.arange(n) / 100.0
    path = _corpus(tmp_path / "a.h5", y=y, x=x)
    got, fs, t0, t1 = masks.read_waveform(path, "mhr", 0)
    assert got.tolist() == list(range(11))
    assert (t0, t1) == pytest.approx((x[4], x[14]))
    assert fs == pytest.approx(100.0)


def test_an_interior_hole_after_a_leading_run_still_raises(tmp_path):
    # Stripping the ends must not turn a digitiser gap into a silent splice.
    y = np.full((1, 12), np.nan, dtype=np.float32)
    y[0, 3:] = 1.0
    y[0, 7] = np.nan
    path = _corpus(tmp_path / "a.h5", y=y, x=np.arange(12) / 12.0)
    with pytest.raises(ValueError, match="non-finite"):
        masks.read_waveform(path, "mhr", 0)


def test_an_interior_non_finite_sample_is_an_error_not_a_hole(tmp_path):
    # A gap in the middle is not the 2^k+1 artefact; it is a record this
    # transform cannot process, and quietly interpolating it would invent
    # spectral content.
    y = np.zeros((1, 10), dtype=np.float32)
    y[0, 4] = np.nan
    y[0, -1] = np.nan
    path = _corpus(tmp_path / "a.h5", y=y, x=np.arange(10) / 10.0)
    with pytest.raises(ValueError, match="non-finite"):
        masks.read_waveform(path, "mhr", 0)


def test_a_record_that_is_all_nan_is_an_error(tmp_path):
    path = _corpus(
        tmp_path / "a.h5",
        y=np.full((1, 8), np.nan, dtype=np.float32),
        x=np.arange(8) / 8.0,
    )
    with pytest.raises(ValueError, match="no finite"):
        masks.read_waveform(path, "mhr", 0)


def test_an_absent_group_cannot_be_read(tmp_path):
    path = _corpus(
        tmp_path / "a.h5", y=np.zeros((8, 1), dtype=np.float32), x=[0.0]
    )
    with pytest.raises(ValueError, match="absent"):
        masks.read_waveform(path, "mhr", 0)


def test_a_group_that_is_not_there_raises_a_key_error(tmp_path):
    path = _corpus(tmp_path / "a.h5", y=np.zeros((2, 8)), x=np.arange(8))
    with pytest.raises(KeyError):
        masks.read_waveform(path, "co2", 0)


def test_a_channel_beyond_the_group_raises(tmp_path):
    path = _corpus(tmp_path / "a.h5", y=np.zeros((2, 8)), x=np.arange(8))
    with pytest.raises(IndexError):
        masks.read_waveform(path, "mhr", 2)


def test_a_mismatched_time_axis_raises(tmp_path):
    path = _corpus(tmp_path / "a.h5", y=np.zeros((2, 8)), x=np.arange(7))
    with pytest.raises(ValueError, match="xdata"):
        masks.read_waveform(path, "mhr", 0)


def test_the_sample_rate_comes_from_the_span_not_from_a_constant(tmp_path):
    # 500 kHz is only nominal; `fs` is (n-1)/(t[-1]-t[0]) of what is there.
    n = 1001
    x = np.linspace(-0.1, 1.9, n)
    path = _corpus(tmp_path / "a.h5", y=np.zeros((1, n)), x=x)
    _, fs, t0, t1 = masks.read_waveform(path, "mhr", 0)
    assert fs == pytest.approx(500.0, rel=1e-5)
    assert (t0, t1) == pytest.approx((-0.1, 1.9), abs=1e-6)


# -------------------------------------------------------------------- prep

def test_prep_is_exactly_the_pinned_tokeye_transform():
    y = _noise(4096)
    spec, meta = masks.prep(y, fs_hz=5e5)
    want = transform.standardise(transform.compute_stft(y)[None])[0]
    assert spec.dtype == np.float32
    assert np.array_equal(spec, want)
    assert meta["decim"] == 1 and meta["n_cols"] == spec.shape[1]
    assert spec.shape[0] == 512


def test_prep_is_not_the_band_restricted_model_input():
    # `model_input` is the AE label path: it cuts to bins 164:512 because
    # that model was trained on the band. The event layer segments the whole
    # spectrogram, and losing 164 bins would lose every mode below 80 kHz.
    spec, _ = masks.prep(_noise(4096), fs_hz=5e5)
    assert spec.shape[0] == 512 != transform.model_input(_noise(4096)[None]).shape[1]


def test_prep_records_the_statistics_that_undo_it():
    # `raw_logpow` is not carried out of `prep`; the recorded mean and std
    # are what a caller inverts to get it back, so they have to be right.
    y = _noise(4096)
    spec, meta = masks.prep(y, fs_hz=5e5)
    raw = transform.compute_stft(y)
    back = spec * (meta["spec_std"] + transform.STD_EPS) + meta["spec_mean"]
    assert np.abs(back - raw).max() < 1e-3
    assert meta["spec_mean"] == pytest.approx(float(raw.mean()), rel=1e-6)
    assert meta["spec_std"] == pytest.approx(float(raw.std()), rel=1e-6)


def test_prep_records_the_percentile_clip_the_transform_applied():
    y = _noise(4096)
    _, meta = masks.prep(y, fs_hz=5e5)
    raw = transform.compute_stft(y)
    assert meta["clip_lo"] == pytest.approx(float(raw.min()))
    assert meta["clip_hi"] == pytest.approx(float(raw.max()))
    assert meta["clip_lo"] < meta["clip_hi"]


def test_the_zoom_pass_decimates_by_four_before_the_stft():
    from scipy import signal

    y = _noise(8192)
    spec, meta = masks.prep(y, fs_hz=5e5, decim=4)
    want_y = signal.decimate(y, 4, ftype="iir", zero_phase=True)
    want = transform.standardise(transform.compute_stft(want_y)[None])[0]
    assert np.array_equal(spec, want)
    assert meta["decim"] == 4
    assert meta["fs_hz"] == 5e5                 # the RECORD's rate, not fs/4
    assert meta["n_cols"] == spec.shape[1]
    # A quarter of the columns, to the STFT's seven-frame border.
    wide, _ = masks.prep(y, fs_hz=5e5)
    assert meta["n_cols"] == pytest.approx(wide.shape[1] / 4, abs=8)


def test_only_the_two_measured_passes_exist():
    with pytest.raises(ValueError, match="decim"):
        masks.prep(_noise(4096), fs_hz=5e5, decim=2)


def test_prep_needs_one_channel():
    with pytest.raises(ValueError):
        masks.prep(np.zeros((2, 4096), dtype=np.float32), fs_hz=5e5)


def test_the_meta_carries_the_grid_a_reader_needs():
    _, meta = masks.prep(_noise(4096), fs_hz=5e5, decim=4)
    assert set(meta) == {
        "fs_hz", "decim", "n_cols", "hop_s", "freq_khz_per_bin",
        "spec_mean", "spec_std", "clip_lo", "clip_hi",
    }
    assert meta["hop_s"] == pytest.approx(128 * 4 / 5e5)       # 1.024 ms
    assert meta["freq_khz_per_bin"] == pytest.approx(0.12207, abs=1e-5)


# ------------------------------------------------------------------- axes

def test_the_wide_frequency_axis_is_the_measured_grid():
    f = masks.freq_axis_khz(5e5, 1)
    assert f.shape == (512,)
    k = np.arange(1, 513)
    assert np.allclose(f, 0.48828 * k, atol=1e-4)
    assert f[-1] == pytest.approx(250.0)        # Nyquist of the 500 kHz grid


def test_the_zoom_frequency_axis_is_a_quarter_of_it():
    f = masks.freq_axis_khz(5e5, 4)
    k = np.arange(1, 513)
    assert np.allclose(f, 0.12207 * k, atol=1e-4)
    assert f[-1] == pytest.approx(62.5)
    # EHO and fishbones live at 2-30 kHz; the wide pass gives them bins
    # 4-61 of 512, the zoom pass gives them 16-245.
    assert int(np.searchsorted(f, 30.0)) - int(np.searchsorted(f, 2.0)) > 200


def test_the_column_times_are_the_transforms_own_frame_grid():
    # The STFT pads the start, so column 0 sits at -3 hops. Pinned against
    # `ShortTimeFFT.t()` rather than assumed, because every event's t0_s is
    # this offset plus a column index.
    n, fs = 4096, 5e5
    want = transform.frame_times_s(n, fs) + 0.25
    got = masks.col_times_s(want.size, fs, 1, 0.25)
    assert np.allclose(got, want, atol=1e-12)
    assert got.dtype == np.float64


def test_the_zoom_columns_are_four_hops_apart():
    t = masks.col_times_s(10, 5e5, 4, 0.0)
    assert np.allclose(np.diff(t), 1.024e-3)
    assert t[3] == pytest.approx(0.0)           # column 3 is sample 0


# ------------------------------------------------------------ tile / stitch

@pytest.mark.parametrize(
    ("n_cols", "n_tiles"),
    [(16391, 37), (24199, 54), (35164, 79), (24576, 55), (64000, 143),
     (4103, 10), (6055, 14), (8797, 20), (16007, 36),
     (1, 1), (512, 1), (513, 2), (960, 2), (961, 3)],
)
def test_the_tile_count_is_the_measured_one(n_cols, n_tiles):
    # The first five are the wide passes of mhr/ece/co2/bes/mirnov on the
    # pilot shot, the next four the zoom passes; the rest are the edges.
    meta = masks.tile(np.zeros((512, n_cols), dtype=np.float32))[1]
    assert meta["n_tiles"] == n_tiles
    assert meta["starts"][0] == 0
    assert meta["starts"][-1] < n_cols


def test_a_tile_is_512_columns_wide_and_the_last_one_is_zero_padded():
    spec = np.arange(512 * 700, dtype=np.float32).reshape(512, 700)
    tiles, meta = masks.tile(spec)
    assert tiles.shape == (2, 512, 512) and tiles.dtype == np.float32
    assert np.array_equal(tiles[0], spec[:, :512])
    assert np.array_equal(tiles[1][:, :252], spec[:, 448:])
    assert not tiles[1][:, 252:].any()
    assert meta["overlap"] == 64 and meta["tile"] == 512


def test_stitching_the_tiles_back_returns_the_spectrogram():
    # The identity "model": whatever comes out of `tile` goes back through
    # `stitch` unchanged, padding dropped and overlaps averaged.
    spec = _noise(512 * 1500).reshape(512, 1500)
    tiles, meta = masks.tile(spec)
    pred = np.stack([tiles, tiles], axis=1)
    got = masks.stitch(pred, meta)
    assert got.shape == (2, 512, 1500) and got.dtype == np.float32
    assert np.array_equal(got[0], spec)
    assert np.array_equal(got[1], spec)


def test_a_constant_prediction_stitches_flat_with_no_seams():
    # The seam is the thing to fear: a stitch that summed instead of
    # averaging would show a 2x stripe every 448 columns.
    _, meta = masks.tile(np.zeros((512, 2000), dtype=np.float32))
    pred = np.full((meta["n_tiles"], 2, 512, 512), 0.37, dtype=np.float32)
    got = masks.stitch(pred, meta)
    assert got.shape == (2, 512, 2000)
    assert np.allclose(got, 0.37, atol=0, rtol=0)


def test_an_overlap_column_is_the_mean_of_both_tiles():
    _, meta = masks.tile(np.zeros((512, 960), dtype=np.float32))   # exactly 2
    assert meta["n_tiles"] == 2
    pred = np.zeros((2, 2, 512, 512), dtype=np.float32)
    pred[0] = 1.0
    pred[1] = 0.0
    got = masks.stitch(pred, meta)
    assert got.shape == (2, 512, 960)
    assert np.all(got[:, :, :448] == 1.0)       # tile 0 alone
    assert np.all(got[:, :, 448:512] == 0.5)    # both tiles, averaged
    assert np.all(got[:, :, 512:] == 0.0)       # tile 1 alone


def test_stitch_drops_the_padding_of_a_ragged_last_tile():
    n_cols = 1000                               # 448 + 512 = 960 < 1000
    _, meta = masks.tile(np.zeros((512, n_cols), dtype=np.float32))
    pred = np.ones((meta["n_tiles"], 2, 512, 512), dtype=np.float32)
    assert masks.stitch(pred, meta).shape[-1] == n_cols


def test_stitch_refuses_a_prediction_that_is_not_this_tiling():
    _, meta = masks.tile(np.zeros((512, 960), dtype=np.float32))    # 2 tiles
    with pytest.raises(ValueError, match="2, C"):
        masks.stitch(np.ones((3, 2, 512, 512), dtype=np.float32), meta)
    with pytest.raises(ValueError, match="tiles of"):
        masks.stitch(np.ones((2, 2, 512, 448), dtype=np.float32), meta)


def test_tile_refuses_a_spectrogram_with_the_wrong_number_of_bins():
    with pytest.raises(ValueError, match="512"):
        masks.tile(np.zeros((348, 1000), dtype=np.float32))


# ------------------------------------------------------------------ infer

def test_infer_returns_stitched_probabilities_for_the_whole_record():
    spec = _noise(512 * 1000).reshape(512, 1000)
    model = FakeUNet()
    out = masks.infer(model, spec, "cpu", batch=8)
    assert out.shape == (2, 512, 1000) and out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
    # ch0 = sigmoid(+40 x), ch1 = sigmoid(-40 x): the two are complementary
    # and the sign of the input decides which one is lit.
    assert np.allclose(out[0] + out[1], 1.0, atol=1e-6)
    assert np.all((out[0] > 0.5) == (spec > 0))


def test_infer_batches_the_tiles():
    spec = np.zeros((512, 2304), dtype=np.float32)        # 512 + 4*448
    model = FakeUNet()
    masks.infer(model, spec, "cpu", batch=2)
    assert model.seen == [2, 2, 1]


def test_a_batch_larger_than_the_tiling_is_one_pass():
    model = FakeUNet()
    masks.infer(model, np.zeros((512, 1000), dtype=np.float32), "cpu", batch=96)
    assert model.seen == [3]                    # 1000 columns is three tiles


def test_an_out_of_memory_halves_the_batch_and_retries():
    spec = np.zeros((512, 512 + 9 * 448), dtype=np.float32)        # 10 tiles
    model = FakeUNet(max_batch=2)
    out = masks.infer(model, spec, "cpu", batch=8)
    # 8 and 4 die, 2 gets through - and the halved batch is kept for the
    # rest of the record rather than climbing back into the same wall.
    assert model.seen == [8, 4, 2, 2, 2, 2, 2]
    assert out.shape == (2, 512, 512 + 9 * 448)


def test_an_out_of_memory_at_a_batch_of_one_is_raised():
    model = FakeUNet(max_batch=0)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        masks.infer(model, np.zeros((512, 1000), dtype=np.float32), "cpu", batch=4)
    assert model.seen == [3, 2, 1]              # halved to 1, then given up


def test_an_out_of_memory_in_the_host_to_device_copy_is_retried_too(monkeypatch):
    # The copy allocates the whole batch on the card, so it is as likely a
    # place to run out as the forward pass. Left outside the `try`, an OOM
    # there escapes a policy written to survive exactly that.
    spec = np.zeros((512, 512 + 9 * 448), dtype=np.float32)        # 10 tiles
    real = masks._to_device
    moved: list[int] = []

    def greedy(x, device):
        moved.append(int(x.shape[0]))
        if x.shape[0] > 2:
            raise torch.cuda.OutOfMemoryError("fake CUDA out of memory")
        return real(x, device)

    monkeypatch.setattr(masks, "_to_device", greedy)
    model = FakeUNet()
    out = masks.infer(model, spec, "cpu", batch=8)
    assert moved == [8, 4, 2, 2, 2, 2, 2]
    assert model.seen == [2, 2, 2, 2, 2]        # the model only ever saw 2
    assert out.shape == (2, 512, 512 + 9 * 448)


def test_amp_is_not_applied_on_a_cpu_device():
    # `torch.autocast("cuda", ...)` on a CPU run would either do nothing or
    # warn; either way the fake model must see plain float32.
    model = FakeUNet()
    masks.infer(model, np.zeros((512, 600), dtype=np.float32), "cpu", amp=True)
    assert model.dtypes == [torch.float32]


def test_infer_feeds_the_model_one_channel_tiles():
    seen = []

    class Shape(FakeUNet):
        def __call__(self, x):
            seen.append(tuple(x.shape[1:]))
            return super().__call__(x)

    masks.infer(Shape(), np.zeros((512, 600), dtype=np.float32), "cpu")
    assert seen == [(1, 512, 512)]


# ----------------------------------------------------------- unstandardise

def test_unstandardise_recovers_the_pre_standardisation_log_power():
    # `raw_logpow` is what `band_logpow` and the track descriptors want and
    # what `prep` does not return; this is the one place the recovery is
    # written, so no caller has to retype the formula out of a docstring.
    y = _noise(4096)
    spec, meta = masks.prep(y, fs_hz=5e5)
    back = masks.unstandardise(spec, meta)
    assert back.dtype == np.float32
    assert np.abs(back - transform.compute_stft(y)).max() < 1e-3


def test_unstandardise_is_exactly_the_documented_formula():
    spec = np.arange(512 * 4, dtype=np.float32).reshape(512, 4) / 100.0
    meta = {"spec_mean": 1.5, "spec_std": 0.25}
    want = (spec * (0.25 + transform.STD_EPS) + 1.5).astype(np.float32)
    assert np.array_equal(masks.unstandardise(spec, meta), want)


def test_unstandardise_needs_the_statistics_that_were_applied():
    with pytest.raises(KeyError):
        masks.unstandardise(np.zeros((512, 4), np.float32), {"spec_mean": 0.0})


# ------------------------------------------------------------- band_logpow

def test_band_logpow_averages_thirty_two_rows_at_a_time():
    raw = np.arange(512, dtype=np.float32)[:, None] * np.ones((1, 5), np.float32)
    got = masks.band_logpow(raw)
    assert got.shape == (16, 5) and got.dtype == np.float16
    # Band j is rows 32j..32j+31, whose mean is 32j + 15.5.
    assert np.allclose(got[:, 0], np.arange(16) * 32 + 15.5)


def test_band_logpow_keeps_the_time_axis_whole():
    raw = np.zeros((512, 7), dtype=np.float32)
    raw[:, 3] = 4.0
    got = masks.band_logpow(raw)
    assert got[:, 3].tolist() == [4.0] * 16
    assert got[:, 2].tolist() == [0.0] * 16


def test_band_logpow_refuses_a_band_count_that_does_not_divide_512():
    with pytest.raises(ValueError, match="512"):
        masks.band_logpow(np.zeros((512, 4), dtype=np.float32), n_bands=15)


# -------------------------------------------------------------- pack/unpack

@pytest.mark.parametrize("n_rows", [512, 3])
@pytest.mark.parametrize("n_cols", [8, 9, 10, 11, 12, 13, 14, 15])
def test_packing_a_mask_round_trips_exactly(n_rows, n_cols):
    # A (512, T) mask is always a whole number of bytes, so the padding path
    # is exercised by the odd row count as well.
    rng = np.random.default_rng(n_rows * 100 + n_cols)
    mask = rng.random((n_rows, n_cols)) > 0.5
    packed = masks.pack(mask)
    assert packed.dtype == np.uint8
    assert packed.size == math.ceil(mask.size / 8)
    back = masks.unpack(packed, mask.shape)
    assert back.dtype == np.bool_
    assert np.array_equal(back, mask)


def test_packing_is_eight_times_smaller_than_the_bool_array():
    mask = np.ones((512, 16391), dtype=bool)
    assert masks.pack(mask).nbytes == mask.nbytes // 8


def test_unpacking_into_the_wrong_shape_raises():
    packed = masks.pack(np.ones((4, 8), dtype=bool))
    with pytest.raises(ValueError):
        masks.unpack(packed, (4, 9))


# ------------------------------------------------------------ block_arrays

def _block(n_cols=6, *, diag="mhr", channel=0, pass_name="wide"):
    coh = np.zeros((512, n_cols), dtype=np.float32)
    tra = np.zeros((512, n_cols), dtype=np.float32)
    coh[0, :] = 0.9                      # row 0 lit in every column
    coh[1, :3] = 0.9                     # row 1 lit in half of them
    tra[:256, 2] = 0.5                   # column 2 lit in half the rows
    raw = np.arange(512, dtype=np.float32)[:, None] * np.ones((1, n_cols), np.float32)
    return masks.MaskBlock(
        diag=diag,
        channel=channel,
        pass_name=pass_name,
        coh=coh,
        tra=tra,
        raw_logpow=raw,
        t_s=np.arange(n_cols, dtype=np.float64) / 1000.0,
        meta={
            "fs_hz": 5e5, "decim": 1, "n_cols": n_cols, "hop_s": 2.56e-4,
            "freq_khz_per_bin": 0.48828125, "spec_mean": 1.5,
            "spec_std": 0.25, "clip_lo": 0.5, "clip_hi": 3.0,
        },
    )


SHA = "a" * 64


def test_a_block_becomes_exactly_seven_keys_under_one_prefix():
    got = masks.block_arrays(_block(), unet_sha256=SHA)
    assert sorted(got) == sorted(
        f"mhr_00_wide{s}" for s in masks.KEY_SUFFIXES
    )
    assert masks.KEY_SUFFIXES == (
        "_coh_packed", "_tra_packed", "_row_lit", "_col_act",
        "_band_logpow", "_t_s", "_meta",
    )


def test_the_prefix_pads_the_channel_to_two_digits():
    got = masks.block_arrays(
        _block(diag="ece", channel=8, pass_name="zoom"), unet_sha256=SHA
    )
    assert all(k.startswith("ece_08_zoom_") for k in got)


def test_every_array_has_the_dtype_the_contract_promises():
    got = masks.block_arrays(_block(), unet_sha256=SHA)
    p = "mhr_00_wide"
    assert got[f"{p}_coh_packed"].dtype == np.uint8
    assert got[f"{p}_tra_packed"].dtype == np.uint8
    assert got[f"{p}_row_lit"].dtype == np.float32
    assert got[f"{p}_col_act"].dtype == np.float32
    assert got[f"{p}_band_logpow"].dtype == np.float16
    assert got[f"{p}_t_s"].dtype == np.float64
    assert isinstance(got[f"{p}_meta"], str)


def test_the_packed_masks_are_the_thresholded_probabilities():
    block = _block()
    got = masks.block_arrays(block, unet_sha256=SHA)
    p = "mhr_00_wide"
    coh = masks.unpack(got[f"{p}_coh_packed"], (512, 6))
    tra = masks.unpack(got[f"{p}_tra_packed"], (512, 6))
    assert np.array_equal(coh, block.coh >= masks.PROB_THRESHOLD)
    assert np.array_equal(tra, block.tra >= masks.PROB_THRESHOLD)


def test_a_different_threshold_changes_the_mask_and_is_recorded():
    got = masks.block_arrays(_block(), thr=0.6, unet_sha256=SHA)
    p = "mhr_00_wide"
    assert not masks.unpack(got[f"{p}_tra_packed"], (512, 6)).any()   # 0.5 < 0.6
    assert json.loads(got[f"{p}_meta"])["thr"] == 0.6


def test_row_lit_is_the_coherent_fraction_per_row():
    got = masks.block_arrays(_block(), unet_sha256=SHA)["mhr_00_wide_row_lit"]
    assert got.shape == (512,)
    assert got[0] == pytest.approx(1.0)          # lit in all 6 columns
    assert got[1] == pytest.approx(0.5)          # lit in 3 of 6
    assert got[2] == pytest.approx(0.0)
    # A row lit almost everywhere is receiver pickup, not a mode: this is
    # the statistic the notch is decided from.
    assert got.dtype == np.float32


def test_col_act_is_the_transient_fraction_per_column():
    got = masks.block_arrays(_block(), unet_sha256=SHA)["mhr_00_wide_col_act"]
    assert got.shape == (6,)
    assert got[2] == pytest.approx(0.5)          # 256 of 512 rows
    assert got[np.arange(6) != 2].tolist() == [0.0] * 5


def test_the_band_logpow_is_the_pre_standardisation_power():
    got = masks.block_arrays(_block(), unet_sha256=SHA)["mhr_00_wide_band_logpow"]
    assert got.shape == (16, 6)
    assert np.allclose(got[:, 0], np.arange(16) * 32 + 15.5)


def test_the_meta_json_names_the_grid_the_threshold_and_the_weights():
    meta = json.loads(masks.block_arrays(_block(), unet_sha256=SHA)["mhr_00_wide_meta"])
    assert meta["fs_hz"] == 5e5
    assert meta["decim"] == 1
    assert meta["n_cols"] == 6
    assert meta["spec_mean"] == 1.5 and meta["spec_std"] == 0.25
    assert meta["clip_lo"] == 0.5 and meta["clip_hi"] == 3.0
    assert meta["tile"] == 512 and meta["overlap"] == 64
    assert meta["thr"] == masks.PROB_THRESHOLD
    assert meta["unet_sha256"] == SHA
    assert meta["diag"] == "mhr" and meta["channel"] == 0
    assert meta["pass_name"] == "wide"
    # A JSON string, not a pickled dict: `np.load(allow_pickle=False)` has
    # to be able to read every key in the file.
    assert json.dumps(meta)


def test_a_block_whose_arrays_disagree_is_refused():
    with pytest.raises(ValueError):
        masks.MaskBlock(
            diag="mhr", channel=0, pass_name="wide",
            coh=np.zeros((512, 6), np.float32),
            tra=np.zeros((512, 5), np.float32),
            raw_logpow=np.zeros((512, 6), np.float32),
            t_s=np.zeros(6), meta={},
        )


def test_a_block_needs_a_pass_name_that_exists():
    with pytest.raises(ValueError, match="pass"):
        masks.MaskBlock(
            diag="mhr", channel=0, pass_name="medium",
            coh=np.zeros((512, 6), np.float32),
            tra=np.zeros((512, 6), np.float32),
            raw_logpow=np.zeros((512, 6), np.float32),
            t_s=np.zeros(6), meta={},
        )


# -------------------------------------------------------------- write/read

def _write(tmp_path, blocks, *, shot=198658, merge=True, name="198658_masks.npz"):
    path = tmp_path / name
    masks.write_masks(path, shot, blocks, merge=merge)
    return path


def test_a_written_file_reads_back_block_by_block(tmp_path):
    block = masks.block_arrays(_block(), unet_sha256=SHA)
    path = _write(tmp_path, [block])
    assert masks.list_blocks(path) == ["mhr_00_wide"]
    assert np.array_equal(
        masks.read_mask(path, "mhr_00_wide_row_lit"), block["mhr_00_wide_row_lit"]
    )
    meta = masks.read_mask(path, "mhr_00_wide_meta")
    assert isinstance(meta, dict) and meta["unet_sha256"] == SHA


def test_the_file_records_which_shot_it_is(tmp_path):
    path = _write(tmp_path, [masks.block_arrays(_block(), unet_sha256=SHA)])
    with np.load(path, allow_pickle=False) as z:
        assert int(z[masks.SHOT_KEY]) == 198658
    assert masks.SHOT_KEY not in masks.list_blocks(path)


def test_merging_a_second_channel_keeps_the_first(tmp_path):
    path = _write(tmp_path, [masks.block_arrays(_block(), unet_sha256=SHA)])
    other = masks.block_arrays(_block(channel=4), unet_sha256=SHA)
    masks.write_masks(path, 198658, [other])
    assert masks.list_blocks(path) == ["mhr_00_wide", "mhr_04_wide"]


def test_rewriting_a_prefix_replaces_all_of_its_keys(tmp_path):
    # A re-run with a different number of columns must not leave the old
    # `_t_s` behind beside the new `_coh_packed`.
    path = _write(tmp_path, [masks.block_arrays(_block(n_cols=6), unet_sha256=SHA)])
    masks.write_masks(
        path, 198658, [masks.block_arrays(_block(n_cols=9), unet_sha256=SHA)]
    )
    assert masks.list_blocks(path) == ["mhr_00_wide"]
    assert masks.read_mask(path, "mhr_00_wide_t_s").size == 9
    assert masks.read_mask(path, "mhr_00_wide_meta")["n_cols"] == 9


def test_merge_false_starts_the_file_again(tmp_path):
    path = _write(tmp_path, [masks.block_arrays(_block(), unet_sha256=SHA)])
    masks.write_masks(
        path, 198658, [masks.block_arrays(_block(channel=4), unet_sha256=SHA)],
        merge=False,
    )
    assert masks.list_blocks(path) == ["mhr_04_wide"]


def test_merging_into_another_shots_file_is_refused(tmp_path):
    path = _write(tmp_path, [masks.block_arrays(_block(), unet_sha256=SHA)])
    with pytest.raises(ValueError, match="shot"):
        masks.write_masks(
            path, 198659, [masks.block_arrays(_block(channel=4), unet_sha256=SHA)]
        )


def test_a_failed_write_leaves_the_old_file_whole(tmp_path, monkeypatch):
    path = _write(tmp_path, [masks.block_arrays(_block(), unet_sha256=SHA)])
    before = path.read_bytes()

    def boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(masks.np, "savez_compressed", boom)
    with pytest.raises(RuntimeError):
        masks.write_masks(
            path, 198658, [masks.block_arrays(_block(channel=4), unet_sha256=SHA)]
        )
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))


def test_the_written_name_is_the_one_the_paths_hand_out(tmp_path):
    from labelmaker.config import Paths

    path = Paths(root=tmp_path).masks_file(198658)
    masks.write_masks(path, 198658, [masks.block_arrays(_block(), unet_sha256=SHA)])
    assert path.exists() and path.name == "198658_masks.npz"


def test_write_masks_takes_the_dicts_block_arrays_makes(tmp_path):
    with pytest.raises(TypeError, match="block_arrays"):
        masks.write_masks(tmp_path / "a.npz", 1, [_block()])


def test_an_unknown_key_is_not_silently_written(tmp_path):
    with pytest.raises(ValueError, match="suffix"):
        masks.write_masks(tmp_path / "a.npz", 1, [{"mhr_00_wide_nonsense": np.zeros(3)}])


def test_reading_a_key_that_is_not_there_raises(tmp_path):
    path = _write(tmp_path, [masks.block_arrays(_block(), unet_sha256=SHA)])
    with pytest.raises(KeyError):
        masks.read_mask(path, "co2_00_wide_row_lit")


def test_a_shot_of_two_passes_holds_both(tmp_path):
    blocks = [
        masks.block_arrays(_block(pass_name="wide"), unet_sha256=SHA),
        masks.block_arrays(_block(pass_name="zoom"), unet_sha256=SHA),
    ]
    path = _write(tmp_path, blocks)
    assert masks.list_blocks(path) == ["mhr_00_wide", "mhr_00_zoom"]


# ------------------------------------------------- the pilot shot, if mounted

@needs_corpus
def test_the_pilot_shots_mhr_record_is_the_one_a1_measured():
    y, fs, t0, t1 = masks.read_waveform(REAL_FILE, "mhr", 0)
    assert y.size == 2_097_152                  # 2^21, the NaN stripped
    assert np.isfinite(y).all()
    assert fs == pytest.approx(5e5, rel=1e-4)
    assert (t1 - t0) == pytest.approx(4.194, abs=1e-3)
    assert transform.n_frames(y.size) == 16_391
    assert masks.tile(np.zeros((512, 16_391), np.float32))[1]["n_tiles"] == 37
