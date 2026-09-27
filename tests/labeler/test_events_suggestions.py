"""Suggestion tables: runs of frames or spans, tiled over a window, validated."""

from __future__ import annotations

import json

import pytest

from labeler.config import Paths
from labeler.events import suggestions
from labeler.events.databases import DatabaseError


def test_frames_become_one_row_per_run_with_its_mean_confidence():
    rows = suggestions.frame_rows(
        7, 5, [0, 0, 1, 1, 1, 3, 0], [0.9, 0.7, 0.6, 0.8, 1.0, 0, 1]
    )
    assert rows == [
        [7, 0, 50, 70, 0.8],
        [7, 1, 70, 100, 0.8],
        [7, 3, 100, 110, 0.0],
        [7, 0, 110, 120, 1.0],
    ]
    assert suggestions.frame_rows(7, 0, [1, 1]) == [[7, 1, 0, 20, ""]]
    assert suggestions.frame_rows(7, 0, []) == []


def test_spans_paint_in_order_over_absent_and_tile_the_window():
    rows = suggestions.span_rows(9, (0, 100), [(10, 20, 1), (15, 30, 2), (90, 150, 1)])
    assert rows == [
        [9, 0, 0, 10, ""],
        [9, 1, 10, 15, ""],
        [9, 2, 15, 30, ""],
        [9, 0, 30, 90, ""],
        [9, 1, 90, 100, ""],
    ]


def test_a_table_is_validated_sorted_and_carries_its_meta(tmp_path):
    paths = Paths(root=tmp_path)
    path = suggestions.table_path(paths, "edge_localized_mode", "elm_clock", "v1")
    assert (
        path
        == tmp_path
        / "suggestions/elm_clock/v1/edge_localized_mode_suggest_elm_clock_v1.csv"
    )
    rows = [[2, 1, 0, 10, ""], [1, 0, 0, 5, 0.5], [1, 1, 5, 9, ""]]
    frame = suggestions.write_table(path, rows, {"method": "elm_clock"})
    assert frame.shot.tolist() == [1, 1, 2]
    assert path.read_text().splitlines()[0] == "shot,category,t_start,t_end,confidence"
    meta = json.loads(path.with_suffix(".meta.json").read_text())
    assert meta == {"method": "elm_clock", "rows": 3, "shots": 2}
    with pytest.raises(DatabaseError, match="category"):
        suggestions.write_table(path, [[1, -1, 0, 5, ""]], {})
