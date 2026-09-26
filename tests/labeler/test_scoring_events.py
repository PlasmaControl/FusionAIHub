"""Events: observed boundaries only, nearest-first matching within a tolerance."""

from __future__ import annotations

from labeler.scoring.events import boundaries, event_cells, match, within
from labeler.scoring.frames import PRESENT, UNCERTAIN, Assessment


def test_only_absent_present_changes_are_boundaries():
    read = Assessment(
        (0, 1000),
        ((100, 200, PRESENT), (200, 300, UNCERTAIN), (500, 600, 1), (600, 1000, 1)),
    )
    onsets, ends = boundaries(read)
    assert onsets.tolist() == [100, 500]  # 500-1000 runs to the window's edge
    assert ends.tolist() == []  # 200 leads into uncertain, not absent


def test_a_span_at_the_window_start_has_no_onset():
    onsets, ends = boundaries(Assessment((0, 1000), ((0, 100, PRESENT),)))
    assert onsets.tolist() == [] and ends.tolist() == [100]


def test_nearest_pairs_are_taken_first():
    m = match([0, 10], [6], 10)
    assert m.pairs.tolist() == [[1, 0]]
    assert m.unmatched_reference.tolist() == [0]
    assert m.offsets.tolist() == [4]


def test_counts_of_a_matching():
    m = match([100, 110], [104, 200], 10)
    assert m.pairs.tolist() == [[0, 0]]
    assert event_cells(m).tolist() == [1, 1, 1]  # tp, fp, fn


def test_the_tolerance_is_inclusive():
    assert len(match([0], [2.0], 2).pairs) == 1
    assert len(match([0], [2.0001], 2).pairs) == 0


def test_a_tie_goes_to_the_earlier_estimate():
    m = match([10], [8, 12], 5)
    assert m.pairs.tolist() == [[0, 0]]
    assert m.unmatched_estimate.tolist() == [1]


def test_empty_sides_leave_everything_unmatched():
    m = match([], [1, 2], 5)
    assert len(m.pairs) == 0 and m.unmatched_estimate.tolist() == [0, 1]
    assert event_cells(match([3], [], 5)).tolist() == [0, 0, 1]


def test_points_are_scored_inside_their_window_only():
    assert within([999.9, 1000, 1500, 2000], (1000, 2000)).tolist() == [1000, 1500]
