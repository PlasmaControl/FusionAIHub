"""The points table: kinds per phenomenon, finite times, windows set together."""

from __future__ import annotations

import pandas as pd
import pytest

from labeler.events.catalog.points import (
    POINT_COLUMNS,
    read_points,
    validate_csv_fields,
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
    assert written.dtypes.equals(missing.dtypes)
    assert pd.concat([written, missing]).dtypes.equals(written.dtypes)
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


@pytest.mark.parametrize(
    "bounds",
    [("oops", "oops"), ("oops", ""), (-float("inf"), 20), (0, float("inf"))],
)
def test_malformed_point_windows_are_rejected(bounds):
    frame = _points([190001, "disruption", "t_D", 10.25, "", *bounds])
    with pytest.raises(DatabaseError, match="window.*finite number"):
        validate_points(frame)


@pytest.mark.parametrize("bounds", [("", ""), (None, None), (float("nan"),) * 2])
def test_blank_point_windows_remain_unchecked(bounds):
    frame = validate_points(_points([1, "disruption", "t_D", 10.25, "", *bounds]))
    assert frame.window_start_ms.isna().all()
    assert frame.window_end_ms.isna().all()


def test_points_shot_cannot_wrap_to_a_negative_int64():
    with pytest.raises(DatabaseError, match="shot"):
        validate_points(_points([2**63, "disruption", "t_D", 10.25, "", None, None]))
    frame = validate_points(
        _points([2**63 - 1, "disruption", "t_D", 10.25, "", None, None])
    )
    assert frame.shot.iloc[0] == 2**63 - 1


def test_repeated_points_name_the_first_repeat():
    row = [190001, "edge_localized_mode", "elm", 10.25, "", None, None]
    with pytest.raises(DatabaseError, match="repeat.*190001.*elm.*10.25"):
        validate_points(_points(row, row, row))
    other = [190002, *row[1:]]
    later = [*row[:3], 10.5, *row[4:]]
    assert len(validate_points(_points(row, other, later))) == 3


@pytest.mark.parametrize(
    "row, count",
    [
        ("190001,1,edge_localized_mode,elm,50,,,", 8),
        ("1,edge_localized_mode,elm,50,,", 6),
    ],
)
def test_read_points_rejects_rows_with_undeclared_or_missing_fields(
    tmp_path, row, count
):
    path = tmp_path / "points.csv"
    path.write_text(",".join(POINT_COLUMNS) + "\n" + row + "\n")
    with pytest.raises(pd.errors.ParserError, match=f"row 2:.*7 fields.*{count}"):
        read_points(path)


def test_csv_blank_lines_preserve_physical_error_line_numbers(tmp_path):
    path = tmp_path / "table.csv"
    path.write_text('a,b\n\n"two\nlines",value\n\nextra,field,here\n')
    with pytest.raises(pd.errors.ParserError, match="row 6: expected 2 fields, got 3"):
        validate_csv_fields(path)


def test_csv_skips_empty_records_but_preserves_quoted_fields(tmp_path):
    path = tmp_path / "table.csv"
    path.write_text('\na,b\n\n"two\nlines","a,b"\n\n')
    validate_csv_fields(path)


def test_csv_does_not_skip_a_record_of_empty_fields(tmp_path):
    path = tmp_path / "table.csv"
    path.write_text('a,b\n""\n')
    with pytest.raises(pd.errors.ParserError, match="row 2: expected 2 fields, got 1"):
        validate_csv_fields(path)


def test_read_points_rejects_duplicate_headers_before_pandas_renames_them(tmp_path):
    path = tmp_path / "points.csv"
    path.write_text(
        ",".join(POINT_COLUMNS) + ",kind\n1,edge_localized_mode,elm,50,,,,crash\n"
    )
    with pytest.raises(pd.errors.ParserError, match="row 1:.*duplicate.*kind"):
        read_points(path)


def test_point_attrs_refuse_duplicate_keys():
    frame = _points(
        [
            190001,
            "edge_localized_mode",
            "elm",
            1.25,
            '{"type":"I","type":"III"}',
            None,
            None,
        ]
    )
    with pytest.raises(DatabaseError, match="duplicate.*type"):
        validate_points(frame)
