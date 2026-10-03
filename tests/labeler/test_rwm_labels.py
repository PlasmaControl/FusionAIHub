"""RWM labels from onset points."""

from __future__ import annotations

import numpy as np

from labeler.rwm import labels as lab


def test_slice_labels_follow_the_papers_horizon():
    t = np.arange(0.0, 400.0, 10.0)
    out = lab.slice_labels(t, [200.0], horizon_ms=100.0, post_ms=50.0)
    positive = t[out == lab.POSITIVE]
    assert positive.min() == 100.0 and positive.max() == 190.0  # [o - 100, o)
    excluded = t[out == lab.EXCLUDED]
    assert excluded.min() == 200.0 and excluded.max() == 390.0
    assert (out[t < 100] == lab.NEGATIVE).all()
    assert (out[t >= 200] == lab.EXCLUDED).all()


def test_a_second_onset_keeps_its_positive_slices_over_the_first_aftermath():
    t = np.arange(0.0, 400.0, 10.0)
    out = lab.slice_labels(t, [100.0, 150.0], horizon_ms=40.0, post_ms=100.0)
    # 110-140 follow the first onset but precede the second: positive wins.
    assert (out[(t >= 110) & (t < 150)] == lab.POSITIVE).all()
    assert (out[(t >= 150) & (t < 200)] == lab.EXCLUDED).all()


def test_no_n1_onsets_means_no_primary_negatives():
    assert (lab.slice_labels(np.arange(5.0), []) == lab.EXCLUDED).all()


def test_broad_sensitivity_includes_post_last_onset_time():
    t = np.array([0.0, 100.0, 200.0, 300.0, 400.0])
    primary = lab.slice_labels(t, [200.0])
    broad = lab.slice_labels(t, [200.0], negative_scope="broad")
    assert primary.tolist() == [0, 1, -1, -1, -1]
    assert broad.tolist() == [0, 1, -1, 0, 0]


def test_primary_negative_time_ends_at_last_target_onset():
    t = np.array([0.0, 200.0, 350.0, 500.0, 600.0, 1000.0])
    out = lab.slice_labels(t, [200.0, 600.0])
    assert out.tolist() == [0, -1, 0, 1, -1, -1]


def test_growth_windows_merge_overlaps_and_keep_separate_ones():
    assert lab.growth_windows([100.0, 110.0, 500.0], growth_ms=20.0) == [
        (80.0, 110.0),
        (480.0, 500.0),
    ]


def test_window_rows_do_not_infer_physical_recovery_from_the_forecast_horizon():
    rows = lab.window_rows(
        7,
        [300.0],
        (100.0, 800.0),
        assumed_absent=True,
        growth_ms=20.0,
        horizon_ms=100.0,
        post_ms=100.0,
    )
    assert rows == [
        (7, lab.ABSENT, 100.0, 200.0),
        (7, 4, 200.0, 280.0),
        (7, lab.UNCERTAIN, 280.0, 300.0),
        (7, 4, 300.0, 800.0),
    ]
    # An unexamined shot's physical state stays explicitly unassessed.
    assert lab.window_rows(8, [], (100.0, 800.0), assumed_absent=False) == [
        (8, 4, 100.0, 800.0),
    ]


def test_window_rows_drop_a_hole_wider_than_the_flattop():
    rows = lab.window_rows(
        7,
        [150.0],
        (100.0, 200.0),
        assumed_absent=True,
        growth_ms=20.0,
        horizon_ms=100.0,
        post_ms=100.0,
    )
    assert rows == [
        (7, 4, 100.0, 130.0),
        (7, lab.UNCERTAIN, 130.0, 150.0),
        (7, 4, 150.0, 200.0),
    ]


def test_later_onsets_do_not_establish_absence_after_an_earlier_onset():
    assert lab.window_rows(7, [300.0, 600.0], (100.0, 800.0), assumed_absent=True) == [
        (7, lab.ABSENT, 100.0, 200.0),
        (7, 4, 200.0, 280.0),
        (7, lab.UNCERTAIN, 280.0, 300.0),
        (7, 4, 300.0, 580.0),
        (7, lab.UNCERTAIN, 580.0, 600.0),
        (7, 4, 600.0, 800.0),
    ]


def test_an_empty_onset_list_does_not_establish_physical_absence():
    assert lab.window_rows(7, [], (100.0, 800.0), assumed_absent=True) == [
        (7, 4, 100.0, 800.0),
    ]


def test_uncertain_onset_extent_does_not_change_the_distinct_forecast_target():
    # ONSET_TIME does not establish that the preceding 20 ms contains the mode.
    rows = lab.window_rows(7, [300.0], None, assumed_absent=False)
    assert rows == [(7, lab.UNCERTAIN, 280.0, 300.0)]
    # A forecast needs only the listed point time, not a physical mode interval.
    assert lab.slice_labels([199.0, 200.0, 280.0, 299.0, 300.0], [300.0]).tolist() == [
        lab.NEGATIVE,
        lab.POSITIVE,
        lab.POSITIVE,
        lab.POSITIVE,
        lab.EXCLUDED,
    ]


def test_other_onsets_are_excluded_but_never_positive():
    t = np.arange(0.0, 400.0, 10.0)
    out = lab.slice_labels(
        t,
        [],
        horizon_ms=40.0,
        post_ms=20.0,
        other_onsets_ms=[200.0],
        negative_scope="broad",
    )
    assert (out[(t >= 160) & (t < 220)] == lab.EXCLUDED).all()
    assert (out[(t < 160) | (t >= 220)] == lab.NEGATIVE).all()
    # A target onset inside that span still makes its own slices positive.
    both = lab.slice_labels(
        t, [210.0], horizon_ms=40.0, post_ms=20.0, other_onsets_ms=[200.0]
    )
    assert (both[(t >= 170) & (t < 210)] == lab.POSITIVE).all()


def test_merge_close_keeps_one_of_a_double_listed_onset():
    assert lab.merge_close([1617.5, 1616.9, 2500.0, 2510.0, 2530.0]) == [
        1616.9,
        2500.0,
        2530.0,
    ]
    assert lab.merge_close([]) == []
