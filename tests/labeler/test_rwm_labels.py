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
    assert excluded.min() == 200.0 and excluded.max() == 240.0  # [o, o + post)
    assert (out[t < 100] == lab.NEGATIVE).all()
    assert (out[t >= 250] == lab.NEGATIVE).all()


def test_a_second_onset_keeps_its_positive_slices_over_the_first_aftermath():
    t = np.arange(0.0, 400.0, 10.0)
    out = lab.slice_labels(t, [100.0, 150.0], horizon_ms=40.0, post_ms=100.0)
    # 110-140 follow the first onset but precede the second: positive wins.
    assert (out[(t >= 110) & (t < 150)] == lab.POSITIVE).all()
    assert (out[(t >= 150) & (t < 200)] == lab.EXCLUDED).all()


def test_no_onsets_means_all_negative():
    assert (lab.slice_labels(np.arange(5.0), []) == lab.NEGATIVE).all()


def test_growth_windows_merge_overlaps_and_keep_separate_ones():
    assert lab.growth_windows([100.0, 110.0, 500.0], growth_ms=20.0) == [
        (80.0, 110.0),
        (480.0, 500.0),
    ]


def test_window_rows_mark_growth_present_and_only_examined_shots_absent():
    rows = lab.window_rows(
        7,
        [300.0],
        (100.0, 800.0),
        examined=True,
        growth_ms=20.0,
        horizon_ms=100.0,
        post_ms=100.0,
    )
    assert rows == [
        (7, lab.ABSENT, 100.0, 200.0),
        (7, lab.PRESENT, 280.0, 300.0),
        (7, lab.ABSENT, 400.0, 800.0),
    ]
    # An unexamined shot carries no absent row, and no onset means no present row.
    assert lab.window_rows(8, [], (100.0, 800.0), examined=False) == []


def test_window_rows_drop_a_hole_wider_than_the_flattop():
    rows = lab.window_rows(
        7,
        [150.0],
        (100.0, 200.0),
        examined=True,
        growth_ms=20.0,
        horizon_ms=100.0,
        post_ms=100.0,
    )
    assert rows == [(7, lab.PRESENT, 130.0, 150.0)]


def test_other_onsets_are_excluded_but_never_positive():
    t = np.arange(0.0, 400.0, 10.0)
    out = lab.slice_labels(
        t, [], horizon_ms=40.0, post_ms=20.0, other_onsets_ms=[200.0]
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
