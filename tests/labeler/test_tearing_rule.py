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


def test_a_short_seed_on_a_lasting_sub_seed_plateau_is_not_present():
    # a 6 G plateau of 250 ms with a 40 G burst on it: the burst alone is above the
    # onset level for 8 ms, but the mode around it stays above a tenth of 40 G for the
    # plateau
    y = np.maximum(trace((1000, 100, 100, 100, 6.0)), trace((1150, 3, 6, 3, 40.0)))
    assert rule.mode_intervals(T, y, rule.N1_RULE) == []


def test_a_ragged_mode_counts_but_a_train_of_short_spikes_does_not():
    ragged = trace()
    for a, b in ((1518, 1535), (1556, 1583), (1591, 1599), (1611, 1631)):
        ragged[a:b] = 25.0
    assert rule.mode_intervals(T, ragged, rule.N1_RULE) == []
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
    table = rule.shot_table(
        rule.label_shot(
            1, T, y, None, (0.0, 3000.0), coherent={1: np.ones(T.shape, bool)}
        )
    )
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
    # An n=2 amplitude well above the calibrated harmonic ratio is its own mode.
    independent = trace((1000, 50, 600, 50, 40.0 * (rule.N2_RULE.harmonic_ratio + 0.2)))
    both = rule.tearing_intervals(T, n1, independent)
    assert [i.n for i in both] == [1, 2]


def test_n2_alone_is_found_at_its_own_threshold():
    quiet = np.full(T.shape, 0.1)
    n2 = trace((800, 50, 300, 50, 8.0))
    (item,) = rule.tearing_intervals(T, quiet, n2)
    assert item.n == 2
    assert item.peak_g == pytest.approx(8.0, abs=0.3)


def test_a_locking_in_the_last_stretch_of_an_interval_marks_it_locked():
    y = trace((1000, 50, 800, 50, 30.0))
    found = rule.tearing_intervals(T, y)
    end = found[0].end_ms
    locked = rule.apply_locking(found, [end - 40.0], confirmed_ms=[end - 40.0])
    assert locked[0].locked and locked[0].ended == rule.LOCKED
    assert locked[0].end_ms == end - 40.0
    # a drop after decay is only a candidate and cannot extend a rotating span
    assert rule.apply_locking(found, [end + 60.0])[0].locked_candidate
    # one long before the end (the mode kept rotating), long after, or none, does not
    assert not rule.apply_locking(found, [1200.0])[0].locked
    assert not rule.apply_locking(found, [end + 500.0])[0].locked
    assert not rule.apply_locking(found, None)[0].locked
    assert not rule.apply_locking(found, [10.0])[0].locked
    # by toroidal number: n = 2's locking is not n = 1's
    assert not rule.apply_locking(found, {2: [end - 40.0]})[0].locked
    assert rule.apply_locking(found, {1: [end - 40.0]})[0].locked_candidate


def test_a_plasma_ended_interval_truncates_at_a_confirmed_lock():
    y = trace((1000, 50, 5000, 50, 30.0))
    found = rule.tearing_intervals(T, y, window=(100.0, 2500.0))
    (item,) = rule.apply_locking(found, [2450.0], confirmed_ms=[2450.0])
    assert item.locked and item.ended == rule.LOCKED and item.end_ms == 2450.0


def test_frequency_locks_find_a_mode_that_spun_down_and_not_one_born_slow():
    f = np.full(T.shape, 6.0)
    f[1800:] = 0.5
    (lock,) = rule.frequency_locks(T, f)
    assert lock == 1800.0
    slow = np.full(T.shape, 0.5)
    assert len(rule.frequency_locks(T, slow)) == 0
    blip = np.full(T.shape, 6.0)
    blip[1800:1805] = 0.5  # shorter than the hold
    assert len(rule.frequency_locks(T, blip)) == 0
    gap = np.full(T.shape, 6.0)
    gap[1800:] = np.nan
    assert len(rule.frequency_locks(T, gap)) == 0


def test_present_mask_selects_by_toroidal_number():
    y = trace((1000, 50, 400, 50, 30.0))
    found = rule.tearing_intervals(T, y)
    mask = rule.present_mask(found, T)
    assert mask[1200] and not mask[200]
    assert not rule.present_mask(found, T, n=2).any()


def test_shot_table_validates_extension_intervals_with_onset_points_and_spans():
    y = trace((1000, 50, 400, 50, 30.0))
    label = rule.label_shot(
        185805,
        T,
        y,
        None,
        (5.0, 2900.0),
        start_ms=100.0,
        coherent={1: np.ones(T.shape, bool)},
        lock_ms={1: []},
    )
    table = rule.shot_table(label)
    validate_intervals(table)
    span = table[(table.category == 1) & (table.t_end > table.t_start)]
    point = table[(table.category == 1) & (table.t_end == table.t_start)]
    assert len(span) == 1 and len(point) == 1
    assert parse_attrs(span["attrs"].iloc[0]) == {
        "iscrowd": 1,
        "n": 1,
        "ended": "decay",
        "locked_known": False,
    }
    onset = parse_attrs(point["attrs"].iloc[0])
    # the weak track that opens with the mode itself leads into it by a few samples,
    # a window so short that the onset row says so
    start, window = span.t_start.iloc[0], onset.pop("onset_window_ms")
    assert window[1] == start and window[0] <= start
    assert onset == {
        "iscrowd": 0,
        "n": 1,
        "ended": "decay",
        "locked_known": False,
        "onset_window_degenerate": True,
    }
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
    label = rule.label_shot(
        1,
        T,
        quiet_ramp,
        None,
        (5.0, 2900.0),
        start_ms=300.0,
        coherent={1: quiet_ramp > 1.0},
    )
    assert label.ramp_up == ()
    table = rule.shot_table(label)
    assert not ((table.category == 2) & (table.t_start < 300)).any()
    noisy_ramp = trace((100, 20, 100, 20, 30.0), (1500, 50, 400, 50, 30.0))
    label = rule.label_shot(
        1,
        T,
        noisy_ramp,
        None,
        (5.0, 2900.0),
        start_ms=300.0,
        coherent={1: noisy_ramp > 1.0},
    )
    table = rule.shot_table(label)
    early = table[(table.category == 2) & (table.t_start < 300)]
    # the rule's own ramp-up row, and the weak-line track the flat-top screen also runs
    assert len(early) == 2 and early.t_end.max() < 300
    assert early["attrs"].eq("").sum() == 1
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
    label = rule.label_shot(
        7,
        T,
        y,
        None,
        (5.0, 2900.0),
        coherent={1: np.ones(T.shape, bool)},
        lock_ms={1: []},
    )
    frame = rule.intervals_frame([label])
    assert list(frame.shot) == [7]
    assert frame.peak_g.iloc[0] == pytest.approx(30.0, abs=0.5)
    assert frame.ended.iloc[0] == rule.DECAY
    assert frame.duration_ms.iloc[0] == pytest.approx(
        frame.t_end.iloc[0] - frame.t_start.iloc[0]
    )


def test_hold_never_counts_sub_release_gaps_or_short_seed_on_a_long_plateau():
    y = np.full(T.shape, 0.3)
    y[500:503] = 20.0
    y[548:551] = 2.5
    y[589:592] = 2.5
    assert rule.mode_intervals(T, y, rule.N1_RULE) == []
    y[1000:1300] = 6.0
    y[1100:1108] = 40.0
    assert rule.mode_intervals(T, y, rule.N1_RULE) == []


def test_acquisition_gap_stays_nan_and_is_a_hard_interval_barrier():
    t = np.r_[np.arange(1000.0), np.arange(1100.0, 2000.0)]
    grid, y, _ = rule.uniform(t, np.full(t.shape, 20.0))
    assert np.isnan(y[(grid >= 1000) & (grid < 1100)]).all()
    short = np.full(T.shape, 0.3)
    short[500:540] = 20.0
    short[540:560] = np.nan
    short[560:600] = 20.0
    assert rule.mode_intervals(T, short, rule.N1_RULE) == []


def test_missing_or_high_frequency_evidence_makes_seeded_candidates_uncertain():
    y = trace((1000, 50, 400, 50, 30.0))
    for coherent in (None, {1: np.zeros(T.shape, bool)}):
        label = rule.label_shot(1, T, y, None, (0, 2999), coherent=coherent)
        assert not label.intervals
        table = rule.shot_table(label)
        assert (table.category == rule.UNCERTAIN).any()
    label = rule.label_shot(
        1, T, y, None, (0, 2999), coherent={1: np.ones(T.shape, bool)}
    )
    assert len(label.intervals) == 1


def test_coherent_sub_seed_activity_is_uncertain_and_never_a_negative():
    y = np.full(T.shape, 0.2)
    y[1000:2000] = 3.0
    evidence = np.zeros(T.shape, bool)
    evidence[1000:2000] = True
    label = rule.label_shot(1, T, y, None, (0, 2999), coherent={1: evidence})
    table = rule.shot_table(label)
    assert not label.intervals
    assert table[(table.t_start <= 1500) & (table.t_end >= 1500)].category.eq(2).all()


def test_frequency_drop_is_only_a_candidate_until_locked_mode_confirmation():
    (item,) = rule.mode_intervals(T, trace((1000, 50, 400, 50, 30)), rule.N1_RULE)
    time = item.end_ms - 40
    (candidate,) = rule.apply_locking([item], [time])
    assert not candidate.locked and candidate.locked_candidate
    assert candidate.lock_time_ms == time and candidate.end_ms == item.end_ms
    (early,) = rule.apply_locking([item], [item.start_ms + 100, time])
    assert early.lock_candidates_ms == (item.start_ms + 100, time)
    assert early.locked_candidate and not early.locked
    (locked,) = rule.apply_locking([item], [time], confirmed_ms=[time])
    assert locked.locked and locked.end_ms == time and locked.ended == rule.LOCKED
    (unknown,) = rule.apply_locking([item], None)
    assert unknown.ended == "unknown" and not unknown.locked_known


def test_coherent_frequency_excludes_rapid_sweeps_stationary_and_high_lines():
    stable = np.full(300, 5.0)
    assert rule.coherent_frequency(stable, 1.0).all()
    for frequency in (
        np.linspace(2, 29, 75),
        np.full(300, 1.0),
        np.full(300, 40.0),
        np.full(300, np.nan),
    ):
        assert not rule.coherent_frequency(frequency, 1.0).any()
    stable[100:200] = np.nan
    mask = rule.coherent_frequency(stable, 1.0)
    assert not mask[100:200].any()


def test_rotating_seed_evidence_is_required_independently_of_release_support():
    y = trace((1000, 50, 400, 50, 30))
    label = rule.label_shot(
        1,
        T,
        y,
        None,
        (0, 2999),
        coherent={1: np.ones(T.shape, bool)},
        seed_coherent={1: np.zeros(T.shape, bool)},
    )
    assert not label.intervals
    assert rule.shot_table(label).category.eq(rule.UNCERTAIN).any()


def test_short_acquisition_gap_is_unobservable_even_inside_a_strong_mode():
    y = trace((1000, 50, 400, 50, 30))
    y[1200:1220] = np.nan
    label = rule.label_shot(
        1, T, y, None, (0, 2999), coherent={1: np.ones(T.shape, bool)}
    )
    table = rule.shot_table(label)
    middle = table[(table.t_start <= 1210) & (table.t_end >= 1210)]
    assert not middle.empty and middle.category.eq(rule.NOT_OBSERVABLE).all()


def test_weak_track_bridges_brief_coherence_interruptions_after_continuous_core():
    y = np.full(T.shape, 0.5)
    evidence = np.zeros(T.shape, bool)
    evidence[1000:2000] = True
    for start in range(1200, 2000, 100):
        evidence[start : start + 20] = False
    label = rule.label_shot(1, T, y, None, (0, 2999), weak_coherent={1: evidence})
    table = rule.shot_table(label)
    assert table[(table.t_start <= 1510) & (table.t_end >= 1510)].category.eq(2).all()
    assert any(a <= 1000 and b >= 1999 for a, b, _, _ in label.uncertain)
    y[1500:1520] = np.nan
    label = rule.label_shot(1, T, y, None, (0, 2999), weak_coherent={1: evidence})
    assert not any(a < 1500 and b > 1520 for a, b, _, _ in label.uncertain)


def test_weak_hysteresis_extends_only_tracks_with_a_continuous_high_core():
    y = np.full(T.shape, 0.5)
    seed = np.zeros(T.shape, bool)
    release = seed.copy()
    seed[1400:1500] = True
    release[1000:2000] = True
    label = rule.label_shot(
        1,
        T,
        y,
        None,
        (0, 2999),
        weak_coherent={1: seed},
        weak_release_coherent={1: release},
    )
    assert any(a == 1000 and b == 1999 for a, b, _, _ in label.uncertain)
    seed[1450:1460] = False
    label = rule.label_shot(
        1,
        T,
        y,
        None,
        (0, 2999),
        weak_coherent={1: seed},
        weak_release_coherent={1: release},
    )
    assert not label.uncertain


@pytest.mark.parametrize("confirmed", [False, True])
def test_abrupt_collapse_is_unknown_or_locked_and_tail_never_absent(confirmed):
    y = np.full(T.shape, 0.2)
    y[1000:1500] = 25.0
    amplitude = np.zeros(T.shape)
    amplitude[1500:2200] = 15.0
    label = rule.label_shot(
        1,
        T,
        y,
        None,
        (0, 2999),
        coherent={1: np.ones(T.shape, bool)},
        lock_ms={1: []},
        lock_amplitude={1: amplitude} if confirmed else None,
    )
    (item,) = label.intervals
    assert item.ended == (rule.LOCKED if confirmed else rule.UNKNOWN)
    assert item.locked == confirmed
    table = rule.shot_table(label)
    tail = table[(table.t_start < 2100) & (table.t_end > 1600)]
    assert tail.category.eq(rule.UNCERTAIN).all()
    if confirmed:
        assert table[(table.t_start < 2600) & (table.t_end > 2500)].category.eq(0).all()
    else:
        assert table[(table.t_start < 2600) & (table.t_end > 2500)].category.eq(2).all()


def test_unscreened_nonquiet_samples_are_uncertain_even_without_a_line():
    y = np.full(T.shape, 0.2)
    y[1000:2000] = 3.0
    label = rule.label_shot(
        1,
        T,
        y,
        None,
        (0, 2999),
        screened={1: np.zeros(T.shape, bool)},
    )
    table = rule.shot_table(label)
    assert table[(table.t_start < 1600) & (table.t_end > 1500)].category.eq(2).all()
    assert table[(table.t_start < 600) & (table.t_end > 500)].category.eq(0).all()


def test_brief_radial_spike_and_missing_release_do_not_confirm_decay():
    y = np.full(T.shape, 0.2)
    y[1000:1500] = 25.0
    amplitude = np.zeros(T.shape)
    amplitude[1500:1510] = 15.0
    label = rule.label_shot(
        1,
        T,
        y,
        None,
        (0, 2999),
        coherent={1: np.ones(T.shape, bool)},
        lock_amplitude={1: amplitude},
    )
    assert label.intervals[0].ended == rule.UNKNOWN
    amplitude[1500:2000] = 15.0
    amplitude[2000:] = np.nan
    label = rule.label_shot(
        1,
        T,
        y,
        None,
        (0, 2999),
        coherent={1: np.ones(T.shape, bool)},
        lock_amplitude={1: amplitude},
    )
    assert label.intervals[0].locked
    table = rule.shot_table(label)
    assert table[(table.t_start < 2600) & (table.t_end > 2500)].category.eq(2).all()


def _locked_collapse(amplitude):
    y = np.full(T.shape, 0.2)
    y[1000:1500] = 25.0
    label = rule.label_shot(
        1,
        T,
        y,
        None,
        (0, 2999),
        coherent={1: np.ones(T.shape, bool)},
        lock_ms={1: []},
        lock_amplitude={1: amplitude},
    )
    return label, rule.shot_table(label)


def _category_at(table, time):
    return int(table[(table.t_start <= time) & (table.t_end > time)].category.max())


def test_a_dip_in_the_radial_field_shorter_than_the_release_hold_is_no_release():
    amplitude = np.zeros(T.shape)
    amplitude[1500:2000] = 15.0
    amplitude[2054:2600] = 15.0
    label, table = _locked_collapse(amplitude)
    assert label.intervals[0].locked
    for time in (2020, 2100, 2500):
        assert _category_at(table, time) == rule.UNCERTAIN
    assert _category_at(table, 2800) == rule.ABSENT


def test_a_sustained_fall_of_the_radial_field_releases_the_lock_at_its_start():
    amplitude = np.zeros(T.shape)
    amplitude[1500:2000] = 15.0
    _, table = _locked_collapse(amplitude)
    assert _category_at(table, 1900) == rule.UNCERTAIN
    assert _category_at(table, 2100) == rule.ABSENT


def test_a_measured_fall_shorter_than_the_hold_then_missing_data_is_no_release():
    amplitude = np.zeros(T.shape)
    amplitude[1500:2000] = 15.0
    amplitude[2000:2100] = 1.0
    amplitude[2100:] = np.nan
    _, table = _locked_collapse(amplitude)
    assert _category_at(table, 2050) == rule.UNCERTAIN
    assert _category_at(table, 2500) == rule.UNCERTAIN


def test_n1_radial_confirmation_cannot_establish_n2_lock():
    y = np.full(T.shape, 0.2)
    y[1000:1500] = 15.0
    label = rule.label_shot(
        1,
        T,
        np.full(T.shape, 0.1),
        y,
        (0, 2999),
        coherent={2: np.ones(T.shape, bool)},
        lock_amplitude={1: np.full(T.shape, 15.0)},
    )
    assert label.intervals[0].n == 2
    assert label.intervals[0].ended == rule.UNKNOWN
    assert not label.intervals[0].locked


@pytest.mark.parametrize("shot", [176030, 176068, 176912])
def test_corrupted_radial_campaign_cannot_confirm_a_lock(shot):
    y = np.full(T.shape, 0.2)
    y[1000:1500] = 25.0
    label = rule.label_shot(
        shot,
        T,
        y,
        None,
        (0, 2999),
        coherent={1: np.ones(T.shape, bool)},
        lock_amplitude={1: np.full(T.shape, 15.0)},
    )
    assert label.intervals[0].ended == rule.UNKNOWN
    assert not label.intervals[0].locked


@pytest.mark.parametrize("drops", [[1300.0], [1020.0, 1300.0]])
def test_all_frequency_drops_precede_collapse_for_independent_confirmation(drops):
    y = np.full(T.shape, 0.2)
    y[1000:1800] = 25.0
    amplitude = np.zeros(T.shape)
    amplitude[1300:1400] = 15.0
    label = rule.label_shot(
        1,
        T,
        y,
        None,
        (0, 2999),
        coherent={1: np.ones(T.shape, bool)},
        lock_ms={1: drops},
        lock_amplitude={1: amplitude},
    )
    (item,) = label.intervals
    assert item.locked and item.ended == rule.LOCKED
    assert item.lock_time_ms == 1300.0
    assert item.end_ms == 1300.0


def test_abrupt_five_ms_collapse_can_touch_release_before_crossing_it():
    y = np.full(T.shape, 0.2)
    y[1000:1500] = 25.0
    y[1500:1504] = [12.0, 9.0, 6.0, 2.5]
    (item,) = rule.mode_intervals(T, y, rule.N1_RULE)
    assert item.abrupt_collapse_ms == 1504.0
    assert item.ended == rule.UNKNOWN


def _collapse_with(field, **kwargs):
    y = np.full(T.shape, 0.2)
    y[1000:1500] = 25.0
    label = rule.label_shot(
        1,
        T,
        y,
        None,
        (0, 2999),
        coherent={1: np.ones(T.shape, bool)},
        lock_ms={1: []},
        lock_amplitude={1: field},
        **kwargs,
    )
    return label, rule.shot_table(label)


def test_lock_confirmation_is_a_rise_over_the_pre_onset_level_not_an_absolute_level():
    # a field that sat at 53 before the mode and still does is no lock
    field = np.full(T.shape, 53.0)
    label, _ = _collapse_with(field)
    assert not label.intervals[0].locked and label.intervals[0].ended == rule.UNKNOWN
    # a rise of 6 over that level held after the collapse is one
    field[1500:2000] = 59.0
    label, _ = _collapse_with(field)
    assert label.intervals[0].locked
    # a rise of 4 over a quiet level is not, although the quiet level sits at 0
    field = np.zeros(T.shape)
    field[1500:2000] = 4.0
    label, _ = _collapse_with(field)
    assert not label.intervals[0].locked


def test_a_slowly_ramping_radial_field_is_no_step_and_confirms_no_lock():
    # 0 to 45 over the record: far above where it was when the mode began, but it
    # gains under 3 across the 200 to 20 ms before and the 20 to 120 ms after a time
    ramp = np.linspace(0.0, 45.0, T.size)
    label, _ = _collapse_with(ramp)
    assert ramp[1504] - ramp[1000] > 5.0
    (item,) = label.intervals
    assert not item.locked and item.ended == rule.UNKNOWN
    assert all(
        reason != "confirmed_locked_phase" for _, _, reason, _ in label.uncertain
    )
    # the same ramp with a step of 8 at the collapse is a lock
    stepped = ramp.copy()
    stepped[1500:] += 8.0
    label, _ = _collapse_with(stepped)
    assert label.intervals[0].locked and label.intervals[0].ended == rule.LOCKED


def test_the_baseline_is_the_field_just_before_the_candidate_not_before_the_onset():
    # high while the mode began, quiet for the 200 ms before its collapse: the step of
    # 6 over that quiet level is a lock, which the onset-time baseline would have missed
    field = np.zeros(T.shape)
    field[900:1280] = 40.0
    field[1500:2000] = 6.0
    label, _ = _collapse_with(field)
    assert label.intervals[0].locked


def test_lock_confirmation_needs_measured_field_on_both_sides_of_the_time():
    field = np.zeros(T.shape)
    field[1500:] = 9.0
    index, level = rule.lock_confirmation(field, T, 1.0, 1500.0)
    assert level == rule.LOCK_RISE and T[index] >= 1520.0
    assert rule.lock_confirmation(field, T, 1.0, 1400.0)[0] is None
    gappy = field.copy()
    gappy[1300:1485] = np.nan  # under 50 ms measured before the time
    assert rule.lock_confirmation(gappy, T, 1.0, 1500.0) == (None, None)
    assert rule.lock_confirmation(field, T, 1.0, 1500.0, floor_ms=1470.0) == (
        None,
        None,
    )


def test_a_slow_lock_after_a_plain_decay_is_found_at_the_interval_end():
    y = trace((1000, 50, 400, 50, 30.0))
    (found,) = rule.mode_intervals(T, y, rule.N1_RULE)
    assert found.ended == rule.DECAY
    field = np.zeros(T.shape)
    field[int(found.end_ms) + 20 : int(found.end_ms) + 900] = 10.0
    label = rule.label_shot(
        1,
        T,
        y,
        None,
        (0, 2999),
        coherent={1: np.ones(T.shape, bool)},
        lock_ms={1: []},
        lock_amplitude={1: field},
    )
    (item,) = label.intervals
    assert item.locked and item.ended == rule.LOCKED
    table = rule.shot_table(label)
    assert _category_at(table, found.end_ms + 400) == rule.UNCERTAIN
    assert _category_at(table, found.end_ms + 1200) == rule.ABSENT
    without = rule.label_shot(
        1, T, y, None, (0, 2999), coherent={1: np.ones(T.shape, bool)}, lock_ms={1: []}
    )
    assert not without.intervals[0].locked


def test_a_rejected_candidate_that_ends_in_a_lock_leaves_an_uncertain_tail():
    y = np.full(T.shape, 0.2)
    y[1000:1030] = 25.0  # a seed too short to be a mode
    field = np.zeros(T.shape)
    field[1040:1800] = 12.0
    label = rule.label_shot(
        1,
        T,
        y,
        None,
        (0, 2999),
        coherent={1: np.ones(T.shape, bool)},
        lock_amplitude={1: field},
    )
    assert not label.intervals
    table = rule.shot_table(label)
    assert _category_at(table, 1500) == rule.UNCERTAIN
    assert _category_at(table, 2200) == rule.ABSENT
    assert any(
        reason == "confirmed_locked_phase" for _, _, reason, _ in label.uncertain
    )


def test_a_sustained_radial_field_in_absent_flat_top_time_is_uncertain_not_absent():
    y = np.full(T.shape, 0.2)
    field = np.zeros(T.shape)
    field[1000:1300] = 12.0
    label = rule.label_shot(1, T, y, None, (0, 2999), lock_amplitude={1: field})
    assert not label.intervals
    table = rule.shot_table(label)
    assert _category_at(table, 1100) == rule.UNCERTAIN
    assert _category_at(table, 1600) == rule.ABSENT
    reasons = {reason for _, _, reason, _ in label.uncertain}
    assert reasons == {"locked_unseeded"}
    field[1000:1300] = 0.0
    field[1000:1060] = 12.0  # shorter than the 100 ms hold
    label = rule.label_shot(1, T, y, None, (0, 2999), lock_amplitude={1: field})
    assert not label.uncertain
    assert rule.shot_table(label).category.eq(rule.ABSENT).all()


def test_a_slow_drift_of_the_radial_field_in_absent_time_is_not_unseeded():
    y = np.full(T.shape, 0.2)
    field = np.linspace(0.0, 45.0, T.size)
    label = rule.label_shot(1, T, y, None, (0, 2999), lock_amplitude={1: field})
    assert not label.uncertain
    assert rule.shot_table(label).category.eq(rule.ABSENT).all()


def test_one_row_per_span_reason_and_number_whatever_the_route_that_found_it():
    label = rule.ShotLabel(
        1,
        (0.0, 100.0),
        0.0,
        (),
        uncertain=(
            (10.0, 20.0, "confirmed_locked_phase", 1),
            (10.0, 20.0, "confirmed_locked_phase", 1),
            (10.0, 20.0, "confirmed_locked_phase", 2),
        ),
    )
    table = rule.shot_table(label)
    assert not table.duplicated().any()
    assert (table.category == rule.UNCERTAIN).sum() == 2


def test_overlapping_rows_of_one_reason_and_number_merge_into_one_span():
    reason = "confirmed_locked_phase"
    label = rule.ShotLabel(
        1,
        (0.0, 100.0),
        0.0,
        (),
        uncertain=(
            (10.0, 30.0, reason, 1),
            (20.0, 40.0, reason, 1),
            (35.0, 45.0, reason, 1),
            (20.0, 40.0, reason, 2),
            (20.0, 40.0, "post_collapse_lock_unknown", 1),
            (60.0, 70.0, reason, 1),
            (70.0, 80.0, reason, 1),
        ),
    )
    rows = rule.shot_table(label).query("category == @rule.UNCERTAIN")
    spans = sorted(
        (r.t_start, r.t_end, parse_attrs(r.attrs)["reason"], parse_attrs(r.attrs)["n"])
        for r in rows.itertuples()
    )
    assert spans == [
        (10.0, 45.0, reason, 1),
        (20.0, 40.0, reason, 2),
        (20.0, 40.0, "post_collapse_lock_unknown", 1),
        (60.0, 70.0, reason, 1),  # rows that only touch stay as they were
        (70.0, 80.0, reason, 1),
    ]


def test_the_weak_screen_covers_the_ramp_up_like_the_flat_top():
    y = np.full(T.shape, 0.5)
    evidence = np.zeros(T.shape, bool)
    evidence[100:400] = True
    label = rule.label_shot(
        1, T, y, None, (5.0, 2900.0), start_ms=500.0, weak_coherent={1: evidence}
    )
    assert any(a <= 100 and b >= 399 for a, b, _, _ in label.uncertain)
    table = rule.shot_table(label)
    assert _category_at(table, 250) == rule.UNCERTAIN
    assert _category_at(table, 450) == rule.ABSENT


def test_the_n2_frequency_cap_scales_with_the_toroidal_number():
    line = np.full(300, 45.0)
    assert not rule.coherent_frequency(line, 1.0).any()
    assert rule.coherent_frequency(line, 1.0, max_khz=2 * rule.LINE_KHZ_PER_N).all()


def test_an_onset_carries_the_window_back_to_the_weak_track_that_led_into_it():
    y = trace((1000, 50, 400, 50, 30.0))
    lead = np.zeros(T.shape, bool)
    lead[400:1000] = True
    label = rule.label_shot(
        1,
        T,
        y,
        None,
        (0, 2999),
        coherent={1: np.ones(T.shape, bool)},
        weak_coherent={1: lead},
    )
    (item,) = label.intervals
    assert item.onset_window_ms == (400.0, item.start_ms)
    table = rule.shot_table(label)
    point = table[(table.category == 1) & (table.t_end == table.t_start)]
    span = table[(table.category == 1) & (table.t_end > table.t_start)]
    assert parse_attrs(point["attrs"].iloc[0])["onset_window_ms"] == [
        400.0,
        item.start_ms,
    ]
    assert "onset_window_ms" not in parse_attrs(span["attrs"].iloc[0])
    assert "onset_window_degenerate" not in parse_attrs(point["attrs"].iloc[0])
    for cell in table["attrs"]:
        assert not attr_problems(rule.CATEGORY, parse_attrs(cell))
    bare = rule.label_shot(
        1, T, y, None, (0, 2999), coherent={1: np.ones(T.shape, bool)}
    )
    # nothing led into it but the weak track that opens with the mode itself
    first, second = bare.intervals[0].onset_window_ms
    assert second == bare.intervals[0].start_ms and second - first <= 5.0
    bare_table = rule.shot_table(bare)
    bare_point = bare_table[
        (bare_table.category == 1) & (bare_table.t_end == bare_table.t_start)
    ]
    flagged = parse_attrs(bare_point["attrs"].iloc[0])
    assert flagged["onset_window_degenerate"] is True
    assert not attr_problems(rule.CATEGORY, flagged)


def test_a_window_of_exactly_five_milliseconds_is_flagged_despite_rounding():
    # shot 185961: the window [660.0, 665.0] is exactly 5 ms, but the subtraction of
    # the unrounded times can land a hair above 5.0
    for start, flagged in ((665.0, True), (665.0 + 4e-10, True), (665.01, False)):
        assert rule.onset_window_degenerate(start, 660.0) is flagged
        item = rule.Interval(
            n=1,
            start_ms=start,
            end_ms=900.0,
            peak_g=20.0,
            peak_ms=700.0,
            onset_window_ms=(660.0, start),
        )
        attrs = rule.interval_attrs(item, crowd=0)
        assert attrs.get("onset_window_degenerate", False) is flagged
