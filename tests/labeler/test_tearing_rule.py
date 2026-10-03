"""The whole-interval tearing-mode rule on traces built to have one answer."""

from __future__ import annotations

import numpy as np
import pytest

from labeler.events.catalog.states import attr_problems
from labeler.events.interval_tables import parse_attrs, validate_intervals
from labeler.tearing import rule

T = np.arange(0.0, 3000.0, 1.0)


def trace(*bursts, floor=0.2, n=None):
    """A noise-floor trace with trapezoid bursts `(start, rise, hold, fall, peak)`."""
    y = np.full(T.shape, floor)
    for start, rise, hold, fall, peak in bursts:
        up = np.clip((T - start) / rise, 0, 1)
        down = np.clip(1 - (T - start - rise - hold) / fall, 0, 1)
        y = np.maximum(y, floor + (peak - floor) * np.minimum(up, down))
    return y


def test_a_mode_is_present_from_a_tenth_of_its_peak_to_the_same_level_on_the_way_down():
    y = trace((1000, 100, 300, 100, 30.0))
    (item,) = rule.mode_intervals(T, y, rule.N1_RULE)
    assert item.n == 1
    assert item.peak_g == pytest.approx(30.0, abs=0.5)
    # 3 G (a tenth of 30) is crossed a tenth of the way up and down the ramps
    assert item.start_ms == pytest.approx(1000 + 100 * (3 - 0.2) / 29.8, abs=3)
    assert item.end_ms == pytest.approx(1500 - 100 * (3 - 0.2) / 29.8, abs=3)
    assert item.ended == rule.DECAY
    assert item.onset_seen and not item.locked
    assert item.release_g == pytest.approx(3.0, abs=0.1)


def test_a_spike_is_not_a_mode_and_neither_is_a_mode_below_the_onset_level():
    spike = trace((1000, 5, 20, 5, 60.0))
    assert rule.mode_intervals(T, spike, rule.N1_RULE) == []
    small = trace((1000, 100, 1000, 100, 9.0))
    assert rule.mode_intervals(T, small, rule.N1_RULE) == []


def test_a_ragged_mode_counts_but_a_train_of_short_spikes_does_not():
    ragged = trace()
    for a, b in ((1518, 1535), (1556, 1583), (1591, 1599), (1611, 1631)):
        ragged[a:b] = 25.0
    (item,) = rule.mode_intervals(T, ragged, rule.N1_RULE)
    assert item.start_ms < 1530 and item.end_ms > 1620
    spikes = trace()
    for a in range(1000, 2000, 30):  # 100 ms of ELM-like spikes, 6 ms wide, 30 ms apart
        spikes[a : a + 6] = 25.0
    assert rule.mode_intervals(T, spikes, rule.N1_RULE) == []


def test_a_median_filter_removes_a_few_sample_spike_inside_a_noise_floor():
    y = trace()
    y[1500:1503] = 90.0
    assert rule.mode_intervals(T, y, rule.N1_RULE) == []


def test_two_bursts_a_short_gap_apart_are_one_interval_and_a_long_gap_two():
    close = trace((500, 20, 100, 20, 25.0), (680, 20, 100, 20, 25.0))
    assert len(rule.mode_intervals(T, close, rule.N1_RULE)) == 1
    far = trace((500, 20, 100, 20, 25.0), (1000, 20, 100, 20, 25.0))
    first, second = rule.mode_intervals(T, far, rule.N1_RULE)
    assert first.end_ms < second.start_ms - 100


def test_a_dip_shorter_than_the_merge_gap_does_not_end_an_interval():
    y = trace((500, 20, 500, 20, 25.0))
    y[700:730] = 0.3
    (item,) = rule.mode_intervals(T, y, rule.N1_RULE)
    assert item.start_ms < 520 and item.end_ms > 1000


def test_a_mode_that_reaches_the_window_end_ends_with_the_plasma():
    y = trace((1000, 50, 5000, 50, 20.0))
    (item,) = rule.mode_intervals(T, y, rule.N1_RULE, window=(100.0, 2500.0))
    assert item.end_ms == 2500.0
    assert item.ended == rule.PLASMA_END


def test_a_mode_present_when_the_window_opens_has_no_observed_onset():
    y = trace((0, 1, 800, 50, 20.0))
    (item,) = rule.mode_intervals(T, y, rule.N1_RULE, window=(0.0, 3000.0))
    assert not item.onset_seen
    table = rule.shot_table(rule.label_shot(1, T, y, None, (0.0, 3000.0)))
    present = table[table.category == rule.PRESENT]
    assert len(present) == 1  # the span, no onset point


def test_the_window_bounds_what_is_read():
    y = trace((1000, 50, 400, 50, 20.0), (2000, 50, 400, 50, 20.0))
    found = rule.mode_intervals(T, y, rule.N1_RULE, window=(1500.0, 3000.0))
    assert [round(i.start_ms, -1) for i in found] == [2000]


def test_n2_harmonic_of_a_large_n1_mode_is_not_a_second_mode():
    n1 = trace((1000, 50, 600, 50, 40.0))
    harmonic = 0.3 * n1
    found = rule.tearing_intervals(T, n1, harmonic)
    assert [i.n for i in found] == [1]
    independent = trace((1000, 50, 600, 50, 20.0))
    both = rule.tearing_intervals(T, n1, independent)
    assert [i.n for i in both] == [1, 2]


def test_n2_alone_is_found_at_its_own_threshold():
    quiet = np.full(T.shape, 0.1)
    n2 = trace((800, 50, 300, 50, 8.0))
    (item,) = rule.tearing_intervals(T, quiet, n2)
    assert item.n == 2
    assert item.peak_g == pytest.approx(8.0, abs=0.3)


def test_locking_cuts_an_interval_at_the_detection_and_marks_it():
    y = trace((1000, 50, 800, 50, 30.0))
    found = rule.tearing_intervals(T, y)
    locked = rule.apply_locking(found, [1500.0])
    assert locked[0].locked and locked[0].ended == rule.LOCKED
    assert locked[0].end_ms == 1500.0
    # a detection after the decay, within the allowance, still marks it
    late = rule.apply_locking(found, [found[0].end_ms + 60.0], after_ms=100.0)
    assert late[0].locked and late[0].end_ms == found[0].end_ms
    # a detection long after, or none, leaves it alone
    assert not rule.apply_locking(found, [found[0].end_ms + 500.0])[0].locked
    assert not rule.apply_locking(found, None)[0].locked
    assert not rule.apply_locking(found, [10.0])[0].locked


def test_present_mask_selects_by_toroidal_number():
    y = trace((1000, 50, 400, 50, 30.0))
    found = rule.tearing_intervals(T, y)
    mask = rule.present_mask(found, T)
    assert mask[1200] and not mask[200]
    assert not rule.present_mask(found, T, n=2).any()


def test_shot_table_is_a_valid_catalog_table_with_onset_points_and_spans():
    y = trace((1000, 50, 400, 50, 30.0))
    label = rule.label_shot(185805, T, y, None, (5.0, 2900.0), start_ms=100.0)
    table = rule.shot_table(label)
    validate_intervals(table)
    span = table[(table.category == 1) & (table.t_end > table.t_start)]
    point = table[(table.category == 1) & (table.t_end == table.t_start)]
    assert len(span) == 1 and len(point) == 1
    assert parse_attrs(span["attrs"].iloc[0]) == {"iscrowd": 1, "n": 1}
    assert parse_attrs(point["attrs"].iloc[0]) == {"iscrowd": 0, "n": 1}
    assert point.t_start.iloc[0] == span.t_start.iloc[0]
    for cell in table["attrs"]:
        assert not attr_problems(rule.CATEGORY, parse_attrs(cell))
    absent = table[table.category == 0]
    assert table.t_start.min() == 5.0
    covered = (
        (absent.t_end - absent.t_start).sum()
        + (span.t_end - span.t_start).sum()
        + table[table.category == 2].eval("t_end - t_start").sum()
    )
    assert covered == pytest.approx(2900.0 - 5.0)


def test_the_ramp_up_is_uncertain_only_where_the_rule_fires_in_it():
    quiet_ramp = trace((1500, 50, 400, 50, 30.0))
    label = rule.label_shot(1, T, quiet_ramp, None, (5.0, 2900.0), start_ms=300.0)
    assert label.ramp_up == ()
    table = rule.shot_table(label)
    assert not (table.category == 2).any()
    noisy_ramp = trace((100, 20, 100, 20, 30.0), (1500, 50, 400, 50, 30.0))
    label = rule.label_shot(1, T, noisy_ramp, None, (5.0, 2900.0), start_ms=300.0)
    table = rule.shot_table(label)
    assert (table.category == 2).sum() == 1
    assert len(label.intervals) == 1  # the ramp-up's mode is not an interval


def test_a_record_that_stops_early_leaves_the_rest_of_the_window_not_observable():
    y = trace((1000, 50, 400, 50, 30.0))[:2000]
    label = rule.label_shot(1, T[:2000], y, None, (5.0, 2900.0))
    table = rule.shot_table(label)
    gone = table[table.category == 3]
    assert len(gone) == 1
    assert gone.t_start.iloc[0] == pytest.approx(1999.0, abs=2)
    assert gone.t_end.iloc[0] == 2900.0


def test_coverage_gaps_find_nan_stretches_and_ignore_short_ones():
    y = np.ones_like(T)
    y[1000:1200] = np.nan
    y[2000:2010] = np.nan
    gaps = rule.coverage_gaps(T, y, (0.0, 2999.0), min_ms=50.0)
    assert len(gaps) == 1
    assert gaps[0][0] == pytest.approx(999.0, abs=1)
    assert gaps[0][1] == pytest.approx(1200.0, abs=1)


def test_a_non_uniform_time_base_is_resampled():
    t = np.concatenate([np.arange(0, 1000, 1.0), np.arange(1000, 2000, 2.0)])
    y = np.interp(t, T, trace((500, 20, 300, 20, 30.0)))
    (item,) = rule.mode_intervals(t, y, rule.N1_RULE)
    assert 500 < item.start_ms < 520


def test_intervals_frame_lists_what_the_catalog_table_cannot():
    y = trace((1000, 50, 400, 50, 30.0))
    label = rule.label_shot(7, T, y, None, (5.0, 2900.0))
    frame = rule.intervals_frame([label])
    assert list(frame.shot) == [7]
    assert frame.peak_g.iloc[0] == pytest.approx(30.0, abs=0.5)
    assert frame.ended.iloc[0] == rule.DECAY
    assert frame.duration_ms.iloc[0] == pytest.approx(
        frame.t_end.iloc[0] - frame.t_start.iloc[0]
    )
