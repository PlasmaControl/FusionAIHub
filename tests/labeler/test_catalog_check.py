"""The catalog checks find each kind of broken table, and the command exits by them."""

from __future__ import annotations

import pandas as pd
import pytest

from labeler.events.catalog.check import (
    CatalogError,
    check_category,
    check_table,
    main,
    require,
)
from labeler.events.catalog.points import POINT_COLUMNS
from labeler.events.interval_tables import ATTRS_COLUMN, INTERVAL_COLUMNS


def _labels(*rows) -> pd.DataFrame:
    """Rows of `shot, state, start, end`, plus attrs text when a row has five."""
    columns = list(INTERVAL_COLUMNS)
    if len(rows[0]) == 5:
        columns.append(ATTRS_COLUMN)
    return pd.DataFrame(
        [(r[0], r[1], r[2], r[3], None, *r[4:]) for r in rows], columns=columns
    )


def _points(*rows) -> pd.DataFrame:
    return pd.DataFrame(
        [(*r, "", None, None) for r in rows], columns=list(POINT_COLUMNS)
    )


GOOD = _labels(
    (1, 0, 0, 100),
    (1, 1, 100, 250),
    (1, 0, 250, 400),
    (2, 3, 50, 90),
    (2, 0, 90, 300),
)


def test_a_good_table_has_no_findings():
    assert check_table(GOOD, "alfven_eigenmode", allowed={1: (0, 400)}) == []


@pytest.mark.parametrize(
    "rows, check, detail",
    [
        ([(1, 0, 0, 100), (1, 1, 120, 200)], "tiling", "a gap from 100 ms to 120 ms"),
        (
            [(1, 0, 0, 100), (1, 1, 90, 200)],
            "tiling",
            "an overlap from 90 ms to 100 ms",
        ),
        (
            [(1, 0, 0, 100), (1, 1, 100, 100), (1, 0, 100, 200)],
            "tiling",
            "a row at 100 ms has no length",
        ),
        ([(1, 4, 0, 100)], "states", "state 4 at 0 ms is not 0-3"),
    ],
)
def test_each_broken_tiling_or_state_is_found(rows, check, detail):
    found = check_table(_labels(*rows), "alfven_eigenmode")
    assert [(f.check, f.shot, f.detail) for f in found] == [(check, 1, detail)]


def test_attributes_are_the_phenomenons():
    rows = _labels((1, 1, 0, 100, '{"m": 2, "q": 1}'), (1, 0, 100, 200, ""))
    found = check_table(rows, "neoclassical_tearing_mode")
    assert [str(f) for f in found] == [
        (
            "labels: attrs: shot 1: at 0 ms: 'q' is not an attribute of "
            "neoclassical_tearing_mode"
        )
    ]


def test_a_window_outside_the_allowed_one_is_found_and_others_left_alone():
    found = check_table(GOOD, "alfven_eigenmode", allowed={1: (10, 400)})
    assert [(f.check, f.shot) for f in found] == [("windows", 1)]  # 2 has none


def test_points_outside_their_window_or_of_another_phenomenon():
    points = _points(
        (1, "edge_localized_mode", "elm", 150.0),
        (1, "edge_localized_mode", "elm", 400.0),  # the window is [0, 400)
        (3, "edge_localized_mode", "elm", 10.0),  # shot 3 was not assessed
        (1, "disruption", "t_D", 399.0),
    )
    found = check_table(GOOD, "edge_localized_mode", points_frame=points)
    assert [(f.check, f.shot, f.detail) for f in found] == [
        ("points", 1, "elm at 400 ms is outside the assessed window"),
        ("points", 3, "elm at 10 ms is outside the assessed window"),
        ("points", 1, "t_D at 399 ms is a disruption point"),
    ]
    assert {f.where for f in found} == {"points"}


def test_a_broken_schema_is_one_finding():
    frame = GOOD.rename(columns={"confidence": "score"})
    assert [f.check for f in check_table(frame, "alfven_eigenmode")] == ["schema"]


def test_require_lists_the_findings():
    found = check_table(_labels((1, 4, 0, 100), (1, 5, 100, 200)), "alfven_eigenmode")
    with pytest.raises(CatalogError, match="state 4.*\n.*state 5"):
        require(found)
    with pytest.raises(CatalogError, match=r"\.\.\. and 1 more"):
        require(found, limit=1)
    require([])


def test_every_review_table_of_a_category_is_checked(tmp_path):
    first = tmp_path / "edge_localized_mode" / "review"
    second = first / "blind"
    second.mkdir(parents=True)
    GOOD.to_csv(first / "labels.csv", index=False)
    _labels((1, 0, 0, 100), (1, 1, 120, 200)).to_csv(second / "labels.csv", index=False)
    _points((1, "edge_localized_mode", "elm", 500.0)).to_csv(
        second / "points.csv", index=False
    )
    found, n = check_category(tmp_path / "edge_localized_mode")
    assert n == 2
    assert [(f.where, f.check) for f in found] == [
        ("edge_localized_mode/review/blind/labels.csv", "tiling"),
        ("edge_localized_mode/review/blind/points.csv", "points"),
    ]


def test_the_command_exits_one_on_any_finding(tmp_path, capsys):
    review = tmp_path / "alfven_eigenmode" / "review"
    review.mkdir(parents=True)
    GOOD.to_csv(review / "labels.csv", index=False)
    argv = ["alfven_eigenmode", "--root", str(tmp_path)]
    assert main(argv) == 0
    assert capsys.readouterr().out == "0 finding(s) in 1 table(s)\n"

    windows = tmp_path / "windows.csv"
    pd.DataFrame(
        {"shot": [1], "window_start_ms": [10.0], "window_end_ms": [400.0]}
    ).to_csv(windows, index=False)
    assert main([*argv, "--windows", str(windows)]) == 1
    assert capsys.readouterr().out.splitlines() == [
        (
            "alfven_eigenmode/review/labels.csv: windows: shot 1: "
            "window 0-400 ms is outside 10-400 ms"
        ),
        "1 finding(s) in 1 table(s)",
    ]


def test_the_command_refuses_a_category_outside_the_catalog(tmp_path):
    with pytest.raises(SystemExit) as stop:
        main(["resistive_wall_mode", "--root", str(tmp_path)])
    assert stop.value.code == 2


@pytest.mark.parametrize("problem", ["kind", "phenomenon", "attrs", None])
def test_orphan_points_are_checked_without_per_point_window_findings(tmp_path, problem):
    review = tmp_path / "edge_localized_mode" / "review" / "blind" / "reader1"
    review.mkdir(parents=True)
    frame = _points((190001, "edge_localized_mode", "elm", 10.25))
    if problem == "kind":
        frame.loc[0, "kind"] = "t_Q"
    elif problem == "phenomenon":
        frame.loc[0, ["phenomenon", "kind"]] = ["disruption", "t_D"]
    elif problem == "attrs":
        frame.loc[0, "attrs"] = '{"wrong": true}'
    frame.to_csv(review / "points.csv", index=False)
    found, n = check_category(tmp_path / "edge_localized_mode")
    assert n == 1
    assert len(found) == (1 if problem is None else 2)
    assert found[0].check == "points" and found[0].shot is None
    assert "no labels.csv beside it" in found[0].detail
    assert "no assessed window" in found[0].detail
    assert {f.where for f in found} == {
        "edge_localized_mode/review/blind/reader1/points.csv"
    }
    assert not any("outside the assessed window" in f.detail for f in found)
    assert main(["--root", str(tmp_path)]) == 1


@pytest.mark.parametrize("filename", ["labels.csv", "points.csv"])
@pytest.mark.parametrize(
    "content, error",
    [(b"", "No columns"), (b'"unclosed', "tokenizing"), (b"\xff", "decode")],
)
def test_unreadable_tables_are_schema_findings_and_checking_continues(
    tmp_path, filename, content, error
):
    review = tmp_path / "edge_localized_mode" / "review"
    broken, good = review / "a", review / "z"
    broken.mkdir(parents=True)
    good.mkdir()
    GOOD.to_csv(broken / "labels.csv", index=False)
    (broken / filename).write_bytes(content)
    _labels((1, 4, 0, 100)).to_csv(good / "labels.csv", index=False)
    found, n = check_category(review.parent)
    assert n == 2
    assert [(f.check, f.where) for f in found] == [
        ("schema", f"edge_localized_mode/review/a/{filename}"),
        ("states", "edge_localized_mode/review/z/labels.csv"),
    ]
    assert error in found[0].detail


def test_raw_points_fallback_parse_errors_are_schema_findings(tmp_path, monkeypatch):
    from labeler.events.catalog import check
    from labeler.events.databases import DatabaseError

    review = tmp_path / "edge_localized_mode" / "review"
    review.mkdir(parents=True)
    GOOD.to_csv(review / "labels.csv", index=False)
    (review / "points.csv").write_text('"unclosed')

    def invalid_schema(path):
        raise DatabaseError("bad schema")

    monkeypatch.setattr(check, "read_points", invalid_schema)
    found, n = check_category(review.parent)
    assert n == 1 and len(found) == 1
    assert found[0].check == "schema"
    assert found[0].where.endswith("/points.csv")
    assert "tokenizing" in found[0].detail


@pytest.mark.parametrize("default", [False, True])
@pytest.mark.parametrize("is_file", [False, True])
def test_missing_or_file_roots_are_usage_errors(
    tmp_path, monkeypatch, capsys, default, is_file
):
    root = tmp_path / "not-a-directory"
    if is_file:
        root.write_text("")
    monkeypatch.setenv("LABELER_LABEL_TABLES", str(root))
    with pytest.raises(SystemExit) as stop:
        main([] if default else ["--root", str(root)])
    assert stop.value.code == 2
    assert str(root) in capsys.readouterr().err


def test_missing_categories_under_an_existing_root_are_allowed(tmp_path, capsys):
    assert main(["--root", str(tmp_path)]) == 0
    assert capsys.readouterr().out == "0 finding(s) in 0 table(s)\n"


@pytest.mark.parametrize(
    "header, row, problem",
    [
        ("shot,window_start_ms", "1,0", "columns"),
        ("shot,window_start_ms,window_end_ms", "-1,0,20", "shot"),
        ("shot,window_start_ms,window_end_ms", "1.5,0,20", "shot"),
        ("shot,window_start_ms,window_end_ms", "oops,0,20", "shot"),
        ("shot,window_start_ms,window_end_ms", f"{2**63},0,20", "shot"),
        ("shot,window_start_ms,window_end_ms", "1,,", "window_start_ms"),
        ("shot,window_start_ms,window_end_ms", "1,oops,20", "window_start_ms"),
        ("shot,window_start_ms,window_end_ms", "1,0,inf", "window_end_ms"),
        ("shot,window_start_ms,window_end_ms", "1,-inf,20", "window_start_ms"),
        ("shot,window_start_ms,window_end_ms", "1,20,20", "before"),
        ("shot,window_start_ms,window_end_ms", "1,30,20", "before"),
        ("shot,window_start_ms,window_end_ms", "1,0,20\n1,1,20", "duplicate"),
    ],
)
def test_invalid_allowed_windows_are_usage_errors(
    tmp_path, capsys, header, row, problem
):
    from labeler.events.catalog.check import read_windows

    path = tmp_path / "windows.csv"
    path.write_text(f"{header}\n{row}\n")
    expected_row = 1 if problem == "columns" else 3 if problem == "duplicate" else 2
    with pytest.raises(CatalogError, match=f"row {expected_row}.*{problem}"):
        read_windows(path)
    with pytest.raises(SystemExit) as stop:
        main(["--root", str(tmp_path), "--windows", str(path)])
    assert stop.value.code == 2
    assert str(path) in capsys.readouterr().err


def test_valid_allowed_windows_keep_fractional_bounds_and_large_shots(tmp_path):
    from labeler.events.catalog.check import read_windows

    path = tmp_path / "windows.csv"
    path.write_text(
        f"shot,window_start_ms,window_end_ms\n{2**63 - 1},0.25,20.75\n0,-1,1\n"
    )
    assert read_windows(path) == {2**63 - 1: (0.25, 20.75), 0: (-1, 1)}


def test_points_findings_replace_only_the_filename():
    frame = _points((1, "edge_localized_mode", "elm", 500))
    found = check_table(
        GOOD,
        "edge_localized_mode",
        points_frame=frame,
        where="edge_localized_mode/review/labelsmith/labels.csv",
    )
    assert [f.where for f in found] == [
        "edge_localized_mode/review/labelsmith/points.csv"
    ]
