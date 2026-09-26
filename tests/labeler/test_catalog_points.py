"""The points table: kinds per phenomenon, finite times, windows set together."""

from __future__ import annotations

import pandas as pd
import pytest

from labeler.events.catalog.points import (
    POINT_COLUMNS,
    read_points,
    validate_points,
    write_points,
)
from labeler.events.databases import DatabaseError


def _points(*rows) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=list(POINT_COLUMNS))


def test_points_are_written_sorted_and_read_back(tmp_path):
    frame = _points(
        [200, "disruption", "t_D", 3001.5, '{"intentional": false}', None, None],
        [100, "edge_localized_mode", "elm", 2000.25, "", 1990, 2100],
        [100, "edge_localized_mode", "elm", 1995.0, None, 1990, 2100],
    )
    path = tmp_path / "points.csv"
    write_points(frame, path)
    back = read_points(path)
    assert back[["shot", "t_ms"]].values.tolist() == [
        [100, 1995.0],
        [100, 2000.25],
        [200, 3001.5],
    ]
    assert back["attrs"].tolist() == ["", "", '{"intentional": false}']
    assert back.window_start_ms.isna().tolist() == [False, False, True]


def test_a_missing_or_empty_points_table_has_no_rows(tmp_path):
    assert read_points(tmp_path / "points.csv").empty
    assert validate_points(_points()).empty


def test_a_missing_points_table_has_the_same_dtypes_as_a_read_one(tmp_path):
    path = tmp_path / "points.csv"
    write_points(_points([1, "disruption", "t_D", 10, "", None, None]), path)
    written = read_points(path)
    missing = read_points(tmp_path / "missing.csv")
    validated = validate_points(_points([1, "disruption", "t_D", 10, "", 0, 20]))
    assert written.dtypes.equals(validated.dtypes)
    typed = ["shot", "t_ms", "window_start_ms", "window_end_ms"]
    assert written[typed].dtypes.equals(missing[typed].dtypes)
    assert written[typed].dtypes.equals(validated[typed].dtypes)
    assert written["shot"].dtype == "int64"
    for column in ("t_ms", "window_start_ms", "window_end_ms"):
        assert written[column].dtype == "float64"


@pytest.mark.parametrize(
    "row, problem",
    [
        ([1, "alfven_eigenmode", "crash", 10.0, "", None, None], "no point kind"),
        ([1, "disruption", "t_Q", 10.0, "", None, None], "no point kind"),
        ([1, "disruption", "t_D", float("inf"), "", None, None], "finite"),
        ([-1, "disruption", "t_D", 10.0, "", None, None], "nonnegative"),
        ([1, "disruption", "t_D", 10.0, "", 0.0, None], "together"),
        ([1, "disruption", "t_D", 10.0, "", 20.0, 20.0], "end after"),
        ([1, "disruption", "t_D", 30.0, "", 0.0, 30.0], "inside"),  # end is open
        ([1, "disruption", "t_D", 10.0, "[1]", None, None], "JSON object"),
    ],
)
def test_malformed_points_are_refused(row, problem):
    with pytest.raises(DatabaseError, match=problem):
        validate_points(_points(row))


def test_the_columns_are_fixed():
    frame = _points([1, "disruption", "t_D", 10.0, "", None, None])
    with pytest.raises(DatabaseError, match="Expected columns"):
        validate_points(frame.drop(columns="attrs"))
