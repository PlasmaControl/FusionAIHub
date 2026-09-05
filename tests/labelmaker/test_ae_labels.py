"""Hermetic checks of the AE mask-cleaning and label primitives.

Every rule here is fixed by the task 7a brief, not fitted: the mode mask is
``coherent >= 0.2 and not transient >= 0.2``, a coherent mode has to persist for
``PERSIST_FRAMES = 20`` frames (5.1 ms at 0.256 ms/frame), the band is bins
164-511, and a notched bin is one lit in more than a threshold fraction of the
record. These tests pin the semantics the dataset script and the later adapter
both depend on; they touch no data root and no GPU.
"""
import numpy as np
import pytest
from scipy import ndimage

from labelmaker.ae.labels import (
    BAND_HI_BIN,
    BAND_LO_BIN,
    N_FFT,
    PERSIST_FRAMES,
    PROB_THRESHOLD,
    apply_notch,
    band_occupancy,
    bin_freqs_khz,
    centroid_khz,
    clean_mask,
    contiguous_runs,
    notch_bins,
    persist_open,
    power_weights,
)

# --- clean_mask --------------------------------------------------------------


def test_clean_mask_keeps_coherent_and_drops_pixels_the_model_calls_transient():
    coherent = np.array([[0.10, 0.30, 0.90, 0.90]])
    transient = np.array([[0.00, 0.00, 0.10, 0.50]])
    got = clean_mask(coherent, transient)
    assert got.dtype == bool
    assert got.tolist() == [[False, True, True, False]]


def test_clean_mask_threshold_is_inclusive_on_both_channels():
    thr = PROB_THRESHOLD
    coherent = np.array([thr, thr, thr - 1e-6])
    transient = np.array([thr - 1e-6, thr, 0.0])
    assert clean_mask(coherent, transient).tolist() == [True, False, False]


def test_clean_mask_threshold_is_configurable():
    coherent = np.array([0.3])
    transient = np.array([0.3])
    assert clean_mask(coherent, transient, threshold=0.5).tolist() == [False]
    assert clean_mask(np.array([0.6]), transient, threshold=0.5).tolist() == [True]


# --- persist_open ------------------------------------------------------------


def _runs_array(lengths, n_frames=80):
    """(len(lengths), n_frames) mask, one run of the given length per bin."""
    mask = np.zeros((len(lengths), n_frames), dtype=bool)
    for row, length in enumerate(lengths):
        mask[row, 10:10 + length] = True
    return mask


def test_persist_open_drops_a_19_frame_blip_and_keeps_a_20_frame_run():
    mask = _runs_array([PERSIST_FRAMES - 1, PERSIST_FRAMES])
    got = persist_open(mask, PERSIST_FRAMES)
    assert not got[0].any()
    assert got[1].sum() == PERSIST_FRAMES
    assert got[1].tolist() == mask[1].tolist()


def test_persist_open_drops_a_broadband_streak_across_every_bin():
    mask = np.zeros((512, 60), dtype=bool)
    mask[:, 30:33] = True  # an ELM-like vertical streak: all bins, 3 frames
    assert not persist_open(mask, PERSIST_FRAMES).any()


def test_persist_open_works_per_bin_and_never_joins_neighbours():
    mask = np.zeros((2, 60), dtype=bool)
    mask[0, 10:22] = True  # 12 frames
    mask[1, 22:34] = True  # 12 frames, adjacent bin, contiguous in time
    assert not persist_open(mask, PERSIST_FRAMES).any()


def test_persist_open_leaves_a_long_run_untouched_and_is_idempotent():
    mask = _runs_array([40])
    once = persist_open(mask, PERSIST_FRAMES)
    assert once.tolist() == mask.tolist()
    assert persist_open(once, PERSIST_FRAMES).tolist() == once.tolist()


def test_persist_open_matches_scipy_binary_opening_along_time():
    rng = np.random.default_rng(7)
    for density in (0.05, 0.3, 0.8):
        mask = rng.random((16, 200)) < density
        want = ndimage.binary_opening(mask, structure=np.ones((1, PERSIST_FRAMES)))
        assert persist_open(mask, PERSIST_FRAMES).tolist() == want.tolist()


def test_persist_open_handles_a_leading_channel_axis():
    mask = np.stack([_runs_array([PERSIST_FRAMES - 1, PERSIST_FRAMES])] * 3)
    got = persist_open(mask, PERSIST_FRAMES)
    assert got.shape == mask.shape
    assert not got[:, 0].any()
    assert got[:, 1].sum() == 3 * PERSIST_FRAMES


def test_persist_open_with_one_frame_is_a_no_op():
    mask = _runs_array([1, 3])
    assert persist_open(mask, 1).tolist() == mask.tolist()


# --- band_occupancy ----------------------------------------------------------


def test_band_occupancy_averages_only_the_band_bins():
    mask = np.zeros((512, 4), dtype=bool)
    mask[0:164, 0] = True          # all below the band: invisible
    mask[164:512, 1] = True        # the whole band
    mask[164:174, 2] = True        # 10 of 348 bins
    got = band_occupancy(mask, BAND_LO_BIN, BAND_HI_BIN)
    assert got.shape == (4,)
    assert got[0] == 0.0
    assert got[1] == 1.0
    assert got[2] == pytest.approx(10 / 348)
    assert got[3] == 0.0


def test_band_occupancy_keeps_the_channel_axis():
    mask = np.zeros((4, 512, 3), dtype=bool)
    mask[2, 164:512, 1] = True
    got = band_occupancy(mask, BAND_LO_BIN, BAND_HI_BIN)
    assert got.shape == (4, 3)
    assert got[2, 1] == 1.0
    assert got.sum() == 1.0


# --- notch_bins / apply_notch ------------------------------------------------


def test_notch_bins_fires_strictly_above_the_threshold():
    frac = np.array([0.0, 0.5, 0.8, 0.81, 1.0])
    assert notch_bins(frac, 0.8).tolist() == [False, False, False, True, True]


def test_notch_bins_keeps_the_channel_axis():
    frac = np.array([[0.9, 0.1], [0.1, 0.95]])
    assert notch_bins(frac, 0.8).tolist() == [[True, False], [False, True]]


def test_apply_notch_zeroes_whole_rows_and_does_not_mutate_the_input():
    mask = np.ones((3, 5), dtype=bool)
    notched = np.array([False, True, False])
    got = apply_notch(mask, notched)
    assert got[1].tolist() == [False] * 5
    assert got[0].all() and got[2].all()
    assert mask.all()


def test_apply_notch_broadcasts_over_channels():
    mask = np.ones((2, 3, 4), dtype=bool)
    notched = np.array([[False, True, False], [True, False, False]])
    got = apply_notch(mask, notched)
    assert got[0, 1].sum() == 0
    assert got[1, 0].sum() == 0
    assert got.sum() == 2 * 3 * 4 - 2 * 4


# --- centroid_khz ------------------------------------------------------------


def test_centroid_khz_is_the_intensity_weighted_mean_of_masked_pixels():
    spec = np.array([[1.0, 1.0], [3.0, 0.0]])
    mask = np.array([[True, True], [True, False]])
    khz = np.array([100.0, 200.0])
    got = centroid_khz(spec, mask, khz)
    assert got[0] == pytest.approx((1 * 100 + 3 * 200) / 4)
    assert got[1] == pytest.approx(100.0)


def test_centroid_khz_ignores_unmasked_intensity():
    spec = np.array([[1.0], [1000.0]])
    mask = np.array([[True], [False]])
    got = centroid_khz(spec, mask, np.array([100.0, 200.0]))
    assert got[0] == pytest.approx(100.0)


def test_centroid_khz_is_nan_where_the_mask_is_empty():
    spec = np.ones((2, 3))
    mask = np.zeros((2, 3), dtype=bool)
    mask[0, 1] = True
    got = centroid_khz(spec, mask, np.array([100.0, 200.0]))
    assert np.isnan(got[0]) and np.isnan(got[2])
    assert got[1] == pytest.approx(100.0)


def test_centroid_khz_pools_the_channel_axis():
    spec = np.ones((2, 2, 1))
    mask = np.zeros((2, 2, 1), dtype=bool)
    mask[0, 0, 0] = True
    mask[1, 1, 0] = True
    got = centroid_khz(spec, mask, np.array([100.0, 200.0]))
    assert got.shape == (1,)
    assert got[0] == pytest.approx(150.0)


def test_power_weights_invert_the_log1p_transform_and_square_the_amplitude():
    amplitude = np.array([1.0, np.sqrt(3.0), 10.0])
    got = power_weights(np.log1p(amplitude))
    assert got == pytest.approx([1.0, 3.0, 100.0])


def test_centroid_khz_weights_linear_power_not_the_log_values():
    """A two-bin mask with linear power weights 1 and 3 centroids at 175 kHz.

    Weighting by the stored log1p values instead would put it near the
    unweighted 150 kHz - the failure the task 6 review caught, where the log
    values span 46-63 and the centroid is effectively unweighted.
    """
    amplitude = np.array([[1.0], [np.sqrt(3.0)]])
    log_spec = np.log1p(amplitude)
    mask = np.ones((2, 1), dtype=bool)
    khz = np.array([100.0, 200.0])

    got = centroid_khz(power_weights(log_spec), mask, khz)
    assert got[0] == pytest.approx((1 * 100 + 3 * 200) / 4)  # 175 kHz

    log_weighted = centroid_khz(log_spec, mask, khz)
    assert abs(log_weighted[0] - 150.0) < abs(got[0] - 150.0)


def test_power_weights_survive_the_dynamic_range_of_the_real_transform():
    log_spec = np.array([[46.0], [63.0]])  # wider than any measured transform
    weights = power_weights(log_spec)
    assert np.all(np.isfinite(weights))
    got = centroid_khz(weights, np.ones((2, 1), dtype=bool), np.array([100.0, 200.0]))
    assert got[0] == pytest.approx(200.0, abs=1e-6)  # the bright bin dominates


# --- frequency grid ----------------------------------------------------------


def test_bin_freqs_khz_reproduces_the_task_6_band_edges():
    fs_khz = 1000001 / 2000.0
    freqs = bin_freqs_khz(fs_khz)
    assert freqs.shape == (BAND_HI_BIN,)
    assert freqs[BAND_LO_BIN] == pytest.approx(80.5665, abs=1e-3)
    assert freqs[BAND_HI_BIN - 1] == pytest.approx(250.0, abs=1e-3)
    assert freqs[0] == pytest.approx(fs_khz / N_FFT)


# --- contiguous_runs ---------------------------------------------------------


def test_contiguous_runs_finds_half_open_spans():
    flags = np.array([0, 1, 1, 0, 0, 1, 0, 1], dtype=bool)
    assert contiguous_runs(flags) == [(1, 3), (5, 6), (7, 8)]


def test_contiguous_runs_handles_the_edges_and_the_empty_case():
    assert contiguous_runs(np.ones(4, dtype=bool)) == [(0, 4)]
    assert contiguous_runs(np.zeros(4, dtype=bool)) == []
