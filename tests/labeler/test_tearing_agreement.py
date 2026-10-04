"""Agreement of interval labels with reference onsets, on cases with one answer."""

from __future__ import annotations

import pandas as pd
import pytest

from labeler.tearing import agreement
from labeler.tearing.agreement import Reference


def frame(*rows):
    return pd.DataFrame(rows, columns=["shot", "n", "t_start", "t_end"])


def test_an_onset_at_the_start_of_an_interval_is_matched_with_its_error():
    intervals = frame((1, 1, 1000.0, 1500.0))
    refs = [Reference(1, 1040.0, 0.0, 6000.0)]
    onsets, compared = agreement.compare_onsets(refs, intervals)
    assert bool(onsets.matched.iloc[0])
    assert onsets.error_ms.iloc[0] == pytest.approx(40.0)
    assert bool(compared.has_onset.iloc[0])


def test_an_onset_inside_an_interval_is_matched_and_before_it_within_tolerance_too():
    intervals = frame((1, 1, 1000.0, 1500.0), (2, 1, 2000.0, 2500.0))
    refs = [Reference(1, 1300.0, 0, 6000), Reference(2, 1950.0, 0, 6000)]
    onsets, _ = agreement.compare_onsets(refs, intervals)
    assert onsets.matched.tolist() == [True, True]
    assert onsets.error_ms.tolist() == pytest.approx([300.0, -50.0])


def test_an_onset_far_from_every_interval_is_missed_and_the_interval_has_none():
    intervals = frame((1, 1, 1000.0, 1500.0))
    refs = [Reference(1, 4000.0, 0.0, 6000.0)]
    onsets, compared = agreement.compare_onsets(refs, intervals)
    assert not bool(onsets.matched.iloc[0])
    assert onsets.error_ms.isna().all()
    assert not bool(compared.has_onset.iloc[0])
    assert bool(compared.reference_has_mode.iloc[0])


def test_the_nearest_start_wins_among_intervals_that_hold_an_onset():
    intervals = frame((1, 1, 1000.0, 2000.0), (1, 1, 1450.0, 1800.0))
    onsets, _ = agreement.compare_onsets([Reference(1, 1500.0, 0, 6000)], intervals)
    assert onsets.start_ms.iloc[0] == 1450.0


def test_only_the_asked_toroidal_numbers_are_compared():
    intervals = frame((1, 2, 1000.0, 1500.0))
    refs = [Reference(1, 1000.0, 0.0, 6000.0)]
    onsets, compared = agreement.compare_onsets(refs, intervals, ns=(1,))
    assert not bool(onsets.matched.iloc[0]) and compared.empty
    onsets, _ = agreement.compare_onsets(refs, intervals, ns=None)
    assert bool(onsets.matched.iloc[0])


def test_an_interval_outside_the_references_coverage_is_not_compared():
    intervals = frame((1, 1, 7000.0, 7500.0))
    refs = [Reference(1, None, 0.0, 6000.0)]
    _, compared = agreement.compare_onsets(refs, intervals)
    assert compared.empty


def test_a_shot_with_no_reference_onset_makes_its_intervals_unmatched():
    intervals = frame((3, 1, 1000.0, 1500.0))
    refs = [Reference(3, None, 0.0, 6000.0), Reference(4, None, 0.0, 6000.0)]
    onsets, compared = agreement.compare_onsets(refs, intervals)
    assert onsets.empty
    assert not bool(compared.has_onset.iloc[0])
    assert not bool(compared.reference_has_mode.iloc[0])


def test_summarize_counts_matches_misses_errors_and_shots():
    intervals = frame(
        (1, 1, 1000.0, 1500.0),
        (2, 1, 2000.0, 2500.0),
        (3, 1, 1000.0, 1200.0),
        (5, 1, 100.0, 300.0),
    )
    refs = [
        Reference(1, 1050.0, 0, 6000),
        Reference(2, 2200.0, 0, 6000),
        Reference(4, 3000.0, 0, 6000),  # missed: no interval at all
        Reference(3, None, 0, 6000),  # ours, not the reference's
        Reference(6, None, 0, 6000),  # neither
        Reference(5, None, 0, 6000),
    ]
    onsets, compared = agreement.compare_onsets(refs, intervals)
    out = agreement.summarize(onsets, compared, refs)
    assert out["reference_onsets"] == 3 and out["matched"] == 2 and out["missed"] == 1
    assert out["matched_fraction"] == pytest.approx(2 / 3)
    assert out["error_ms"]["median"] == pytest.approx(125.0)
    assert out["error_ms"]["within_ms"]["50"] == pytest.approx(0.5)
    assert out["compared_intervals"] == 4
    assert out["intervals_without_an_onset"] == 2
    assert out["intervals_without_an_onset_on_reference_quiet_shots"] == 2
    assert out["shots"] == {
        "reference_mode_and_ours": 2,
        "reference_mode_not_ours": 1,
        "ours_not_reference": 2,
        "neither": 1,
    }


def test_summarize_survives_no_onsets_and_no_intervals():
    refs = [Reference(1, None, 0, 6000)]
    onsets, compared = agreement.compare_onsets(refs, frame())
    out = agreement.summarize(onsets, compared, refs)
    assert out["reference_onsets"] == 0 and out["matched_fraction"] is None
    assert out["compared_intervals"] == 0


def trace(*bursts, n=4000):
    """A 1 kHz RMS trace of 0.1 G with `(start, stop, level)` bursts on it."""
    import numpy as np

    t = np.arange(n, dtype=float)
    y = np.full(n, 0.1)
    for a, b, level in bursts:
        y[a:b] = level
    return t, y


def reason(bursts, onset, *, starts=(), window=(0.0, 3999.0)):
    t, y = trace(*bursts)
    return agreement.miss_reason(t, y, onset, window, starts)


def test_a_miss_outside_the_window_is_said_so():
    assert reason([], 5000.0) == "outside_window"


def test_a_burst_over_the_level_for_less_than_the_hold_is_a_short_burst():
    assert reason([(1000, 1006, 40.0)], 1000.0) == "short_burst"


def test_a_trace_that_never_reaches_the_level_is_below_it():
    assert reason([(1000, 1400, 8.0)], 1000.0) == "below_onset_level"


def test_our_interval_beginning_well_after_the_reference_is_starts_later():
    assert reason([(1200, 1600, 40.0)], 1000.0, starts=[1200.0]) == (
        "interval_starts_later"
    )


def test_an_unrelated_burst_far_from_the_onset_does_not_hide_a_below_level_miss():
    assert reason([(2500, 2506, 40.0)], 1000.0) == "below_onset_level"


def test_an_onset_inside_the_preceding_weak_window_is_said_to_be_inside_it():
    intervals = frame((1, 1, 1000.0, 1500.0), (2, 1, 2000.0, 2500.0))
    intervals["onset_window_start_ms"] = [800.0, float("nan")]
    refs = [Reference(1, 900.0, 0, 6000), Reference(2, 1950.0, 0, 6000)]
    onsets, compared = agreement.compare_onsets(refs, intervals)
    assert onsets.window_start_ms.tolist() == pytest.approx([800.0, 2000.0])
    # shot 1's onset lies in 800-1500; shot 2 has no window beyond the start, 50 ms late
    assert onsets.in_onset_window.tolist() == [True, False]
    out = agreement.summarize(onsets, compared, refs)
    assert out["error_ms"]["reference_inside_onset_window_fraction"] == 0.5
