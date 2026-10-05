"""The HL-3 full-input set: chord choice without labels, sums, clocks, stacking."""

import inspect

import numpy as np
import pytest

from labeler.sawtooth import hl3_inputs as hl3

FS = 10000.0


def sawtooth(n, period=40):
    """A 250 Hz ramp-and-crash waveform: inside the 100 Hz to 2 kHz band."""
    return (np.arange(n) % period) / period - 0.5


def fan(n=20000, seed=0):
    """32 chords: core 14 (large), 25 anti-phase, 28 in phase, rest noise."""
    rng = np.random.default_rng(seed)
    saw = sawtooth(n)
    y = rng.normal(0, 0.02, size=(32, n)) + 5.0
    y[14] += 4.0 * saw
    y[25] += -2.0 * saw
    y[28] += 1.0 * saw
    y[3] += 0.3 * saw
    return y


def test_core_is_highest_variance_and_edge_most_anticorrelated_outer_chord():
    pair = hl3.select_sxr_chords(fan())
    assert pair.core == 14
    assert pair.edge == 25
    assert pair.correlation < -0.9
    assert pair.method == hl3.METHOD
    assert len(pair.lit) == 32


def test_a_broadband_noise_chord_is_not_the_core_however_loud():
    y = fan()
    y[9] += np.random.default_rng(3).normal(0, 1.0, size=y.shape[1])
    pair = hl3.select_sxr_chords(y)
    assert pair.core == 14
    assert pair.structured_core is True
    assert pair.structured_chords >= 4


def test_without_any_structure_the_pair_falls_back_and_says_so():
    rng = np.random.default_rng(4)
    y = rng.normal(0, 1.0, size=(32, 20000)) + 5.0
    pair = hl3.select_sxr_chords(y)
    assert pair.core is not None and pair.structured_core is False
    assert pair.structured_chords == 0
    assert pair.edge is None or pair.structured_edge is False
    assert pair.to_json()["core_has_structure"] is False


def test_edge_must_lie_further_from_the_fan_centre_than_the_core():
    y = fan()
    # Chord 16 is nearer the centre than the core (14 is 1.5 from 15.5, 16 is 0.5)
    # and the most anti-correlated of all: it must still not be the edge.
    y[16] += -3.0 * sawtooth(y.shape[1])
    y[25] -= -2.0 * sawtooth(y.shape[1])
    pair = hl3.select_sxr_chords(y)
    assert pair.core == 14
    assert pair.edge not in (14, 15, 16)


def test_chord_numbers_label_the_answer_not_the_row():
    y = fan()
    numbers = 192 + np.arange(32)
    pair = hl3.select_sxr_chords(y, chords=numbers, centre=192 + hl3.FAN_CENTRE)
    assert (pair.core, pair.edge) == (192 + 14, 192 + 25)
    assert pair.to_json()["core_chord"] == 206


def test_choice_depends_on_the_signals_alone():
    y = fan()
    order = np.random.default_rng(1).permutation(32)
    shuffled = hl3.select_sxr_chords(y[order], chords=order)
    plain = hl3.select_sxr_chords(y)
    assert (shuffled.core, shuffled.edge) == (plain.core, plain.edge)
    names = set(inspect.signature(hl3.select_sxr_chords).parameters)
    assert names == {"y", "chords", "fs", "centre"}


def test_unlit_chords_and_short_fans_give_no_pair():
    y = fan()
    y[:, : y.shape[1] * 3 // 4] = np.nan  # every chord lit on only 25 %
    assert hl3.select_sxr_chords(y).core is None
    y = fan()
    y[7:] = np.nan  # seven lit chords, fewer than eight
    pair = hl3.select_sxr_chords(y)
    assert pair.core is None and pair.edge is None and len(pair.lit) == 7


def test_edge_is_none_when_the_core_is_the_outermost_chord():
    y = fan()
    y[31] += 8.0 * sawtooth(y.shape[1])  # nothing lies further from the centre
    pair = hl3.select_sxr_chords(y)
    assert pair.core == 31
    assert pair.edge is None and pair.correlation is None


def test_gaps_are_filled_for_filtering_but_never_scored():
    y = fan()
    y[14, 5000:5600] = np.nan
    y[25, 12000:12100] = np.nan
    pair = hl3.select_sxr_chords(y)
    assert (pair.core, pair.edge) == (14, 25)


def test_robust_standardise_centres_on_the_median_and_keeps_gaps():
    rng = np.random.default_rng(2)
    values = rng.normal(7.0, 3.0, size=20001)
    values[100] = np.nan
    values[200] = 1e6  # a glitch must not set the scale
    out = hl3.robust_standardise(values)
    assert np.isnan(out[100])
    assert abs(np.nanmedian(out)) < 0.05
    assert abs(np.nanmedian(np.abs(out)) * 1.4826 - 1.0) < 0.05
    flat = np.array([3.0, 3.0, 3.0])
    np.testing.assert_array_equal(hl3.robust_standardise(flat), [0.0, 0.0, 0.0])
    assert np.isnan(hl3.robust_standardise(np.full(3, np.nan))).all()


def test_uniform_clock_accepts_a_float32_axis_and_rejects_jitter_and_reversal():
    n = 400000
    x = (np.arange(n) * 1e-4 + 3.0).astype(np.float32)
    ideal = hl3.uniform_clock(x)
    assert ideal.dtype == np.float64
    assert np.max(np.abs(ideal - x)) < 5e-6
    jitter = x.copy()
    jitter[1000:] += np.float32(5e-3)
    with pytest.raises(ValueError, match="nonuniform"):
        hl3.uniform_clock(jitter)
    reversed_clock = x.copy()
    reversed_clock[50], reversed_clock[51] = reversed_clock[51], reversed_clock[50]
    with pytest.raises(ValueError, match="non-decreasing"):
        hl3.uniform_clock(reversed_clock)
    with pytest.raises(ValueError):
        hl3.uniform_clock(x[:10])


def test_uniform_clock_accepts_repeated_float32_times_of_a_fast_digitiser():
    # 1.7 MHz at 8 s: the float32 spacing is wider than the step, so times repeat.
    n = 200000
    x = (8.0 + np.arange(n) * 6e-7).astype(np.float32)
    assert (np.diff(x) == 0).any()
    ideal = hl3.uniform_clock(x)
    assert (np.diff(ideal) > 0).all()
    assert np.max(np.abs(ideal - x)) < 1e-6
    flat = np.full(100, 3.0, dtype=np.float32)
    with pytest.raises(ValueError, match="advance"):
        hl3.uniform_clock(flat)


def test_sum_sources_counts_positive_finite_power_only():
    y = np.array([[1.0, np.nan, -1.0, 2.0], [np.nan, np.nan, np.nan, np.nan]])
    np.testing.assert_array_equal(hl3.sum_sources(y), [1.0, 0.0, 0.0, 2.0])
    assert hl3.sum_sources(np.full((3, 5), np.nan)) is None
    two = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_array_equal(hl3.sum_sources(two), [4.0, 6.0])


def test_interpolate_gapped_bridges_short_holes_only():
    tx = np.array([0.0, 0.02, 0.04, 0.5, 0.52])
    y = np.array([0.0, 2.0, 4.0, 5.0, 6.0])
    t = np.array([-0.01, 0.01, 0.03, 0.2, 0.51, 0.6])
    out = hl3.interpolate_gapped(t, tx, y, max_gap_s=0.1)
    np.testing.assert_allclose(out[[1, 2, 4]], [1.0, 3.0, 5.5])
    assert np.isnan(out[[0, 3, 5]]).all()
    assert np.isnan(hl3.interpolate_gapped(t, tx[:1], y[:1], max_gap_s=0.1)).all()


def test_stack_inputs_orders_channels_and_reports_missing_ones():
    t = np.arange(100) * 1e-4
    parts = {
        "ip": np.full(100, 1.5),
        "ech_power": None,
        "stored_energy": (np.array([-1.0, 1.0]), np.array([1.0e6, 3.0e6])),
    }
    matrix, coverage = hl3.stack_inputs(t, parts)
    assert matrix.shape == (9, 100) and matrix.dtype == np.float32
    assert hl3.CHANNELS.index("ip") == 2
    np.testing.assert_array_equal(matrix[2], 1.5)
    assert coverage["ip"] == 1.0
    assert coverage["ech_power"] == 0.0 and coverage["sxr_edge"] == 0.0
    np.testing.assert_allclose(matrix[6, 0], 2.0, atol=1e-6)  # joules to MJ
    assert len(hl3.CHANNELS) == 9 and len(set(hl3.CHANNELS)) == 9


def test_stacked_units_keep_every_channel_finite_in_float16():
    # Typical DIII-D magnitudes in the units the readers produce: the benchmark
    # stores windows as float16, which overflows above 65504.
    t = np.arange(50) * 1e-4
    raw = {
        "ece_core_te": 8.0,
        "mirnov_pair_mean": 310.0,
        "ip": 1.6,
        "line_density": 4.4e14,
        "sxr_core": 100.0,
        "sxr_edge": 140.0,
        "stored_energy": 1.9e6,
        "nbi_power": 1.7e7,
        "ech_power": 2.3e6,
    }
    assert set(hl3.UNIT_SCALE) == set(hl3.CHANNELS)
    matrix, _ = hl3.stack_inputs(t, {k: np.full(50, v) for k, v in raw.items()})
    assert np.isfinite(matrix.astype(np.float16)).all()
    assert np.abs(matrix).max() < 400
    np.testing.assert_allclose(matrix[hl3.CHANNELS.index("nbi_power")], 17.0)
    np.testing.assert_allclose(matrix[hl3.CHANNELS.index("line_density")], 4.4)


def test_stack_inputs_refuses_unknown_names_and_misaligned_arrays():
    t = np.arange(10) * 1e-4
    with pytest.raises(ValueError, match="unknown"):
        hl3.stack_inputs(t, {"te": np.zeros(10)})
    with pytest.raises(ValueError, match="native grid"):
        hl3.stack_inputs(t, {"ip": np.zeros(9)})
