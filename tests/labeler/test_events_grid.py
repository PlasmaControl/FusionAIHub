import numpy as np
import pytest

from labeler.events.grid import (
    GRID_MS,
    UNASSESSED,
    frames,
    snap,
    snap_span,
    snap_window,
)


def test_snap_rounds_to_the_nearest_frame_edge_halves_up():
    assert GRID_MS == 10
    assert [snap(t) for t in (1004, 1004.9, 1005, 1006, -4, -5, -6)] == [
        1000, 1000, 1010, 1010, 0, 0, -10,
    ]


def test_a_short_span_claims_the_frame_of_its_midpoint():
    assert snap_span(1003, 1012) == (1000, 1010)
    assert snap_span(1001, 1004) == (1000, 1010)  # would vanish
    assert snap_span(1006, 1009) == (1000, 1010)  # would vanish, midpoint 1007.5
    assert snap_span(1013, 1014) == (1010, 1020)
    assert snap_span(1007, 1007) == (1010, 1010)  # a point stays a point
    with pytest.raises(ValueError, match="backwards"):
        snap_span(10, 0)


def test_a_window_snaps_like_a_span():
    # Not floor/ceil: 0-4573 must not gain an absent frame 4570-4580.
    assert snap_window(0, 4573) == (0, 4570)
    assert snap_window(1001, 1019) == (1000, 1020)
    assert snap_window(-3, 0) == (-10, 0)  # would vanish: its midpoint's frame


def test_frames_paint_window_then_spans():
    out = frames((995, 1046), [(1001, 1004, 2), (1012, 1031, 1)], t0_ms=980, n=10)
    assert out.dtype == np.int8
    assert out.tolist() == [-1, -1, 2, 1, 1, 0, 0, -1, -1, -1]
    assert UNASSESSED == -1
    with pytest.raises(ValueError, match="frame edge"):
        frames((0, 10), [], t0_ms=5, n=1)


def test_unknown_spans_are_unassessed_inside_the_window():
    found = frames((0, 50), [(0, 50, 1)], 0, 5, unknown=[(12, 29)])
    assert found.tolist() == [1, -1, -1, 1, 1]
