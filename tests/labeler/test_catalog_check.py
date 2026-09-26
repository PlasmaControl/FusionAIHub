"""The catalog checks find each kind of broken table, and the command exits by them."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from labeler.events.catalog.check import (
    CatalogError,
    check_category,
    check_table,
    main,
    read_windows,
    require,
    windows,
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


@pytest.mark.parametrize(
    "module", ["labeler.events.catalog.check", "labeler.events.catalog.states"]
)
def test_import_does_not_load_torch_or_toksearch(module):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                f"import sys; import {module}; "
                "assert not {'torch', 'toksearch', 'toksearch_d3d'} "
                "& sys.modules.keys()"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


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


# Disruption is always observable; keep shot 2 assessed but absent.
DISRUPTION_LABELS = GOOD.replace({"category": {3: 0}})
DISRUPTION_POINTS = _points(
    (1, "disruption", "t_D", 175.5),
    (1, "disruption", "t80", 100.25),
    (1, "disruption", "t20", 249.75),
)


def test_a_consistent_disruption_shot_passes_every_check():
    assert (
        check_table(DISRUPTION_LABELS, "disruption", points_frame=DISRUPTION_POINTS)
        == []
    )


@pytest.mark.parametrize("end", [5260.54, 5262.0])
def test_d19_final_quench_may_end_up_to_two_ms_past_the_allowed_window(end):
    labels = _labels((191389, 0, 8, 5256), (191389, 1, 5256, round(end)))
    frame = _points(
        (191389, "disruption", "t80", 5255.65),
        (191389, "disruption", "t_D", end - 0.01),
        (191389, "disruption", "t20", end),
    )
    assert (
        check_table(
            labels, "disruption", allowed={191389: (8, 5260)}, points_frame=frame
        )
        == []
    )


@pytest.mark.parametrize("kind", ["t80", "t_D", "t20"])
def test_d19_points_allow_two_ms_but_report_a_larger_overrun(kind):
    labels = _labels((191389, 0, 8, 5256), (191389, 1, 5256, 5263))
    frame = _points(
        (191389, "disruption", "t80", 5255.65),
        (191389, "disruption", "t_D", 5260.53),
        (191389, "disruption", "t20", 5260.54),
    )
    frame.loc[frame.kind == kind, "t_ms"] = 5262.01
    found = check_table(
        labels, "disruption", allowed={191389: (8, 5260)}, points_frame=frame
    )
    assert any(f.check == "windows" for f in found)
    assert any(f.check == "points" and kind in f.detail for f in found)
    for check in ("windows", "points"):
        assert any(
            f.check == check and "8-5260 ms" in f.detail and "2 ms" in f.detail
            for f in found
        )


def test_d19_points_pass_when_labels_stop_at_the_allowed_window_end():
    labels = _labels((191389, 0, 8, 5256), (191389, 1, 5256, 5260))
    frame = _points(
        (191389, "disruption", "t80", 5255.65),
        (191389, "disruption", "t_D", 5260.53),
        (191389, "disruption", "t20", 5260.54),
    )
    assert (
        check_table(
            labels, "disruption", allowed={191389: (8, 5260)}, points_frame=frame
        )
        == []
    )


@pytest.mark.parametrize("state", [0, 2])
def test_d19_does_not_extend_nonpresent_disruption_spans(state):
    labels = _labels((191389, state, 8, 5261))
    found = check_table(labels, "disruption", allowed={191389: (8, 5260)})
    assert any(f.check == "windows" for f in found)


def test_d19_does_not_extend_other_phenomena_or_the_window_start():
    labels = _labels((191389, 1, 8, 5261))
    frame = _points((191389, "edge_localized_mode", "elm", 5261))
    found = check_table(
        labels,
        "edge_localized_mode",
        allowed={191389: (8, 5260)},
        points_frame=frame,
    )
    assert {f.check for f in found} == {"windows", "points"}
    labels = _labels((191389, 1, 7, 20))
    frame = _points(
        (191389, "disruption", "t80", 7.99),
        (191389, "disruption", "t_D", 7.99),
        (191389, "disruption", "t20", 19.99),
    )
    found = check_table(
        labels, "disruption", allowed={191389: (8, 5260)}, points_frame=frame
    )
    assert any(f.check == "windows" for f in found)


def test_d19_does_not_extend_a_labelled_window_ending_before_the_allowed_end():
    labels = _labels((191389, 1, 5255, 5258))
    frame = _points(
        (191389, "disruption", "t80", 5255),
        (191389, "disruption", "t_D", 5257),
        (191389, "disruption", "t20", 5258),
    )
    found = check_table(
        labels, "disruption", allowed={191389: (8, 5260)}, points_frame=frame
    )
    assert len(found) == 1 and "outside the assessed window" in found[0].detail


@pytest.mark.parametrize("kind", ["t_D", "t80", "t20"])
def test_a_disruption_refuses_more_than_one_point_of_each_kind(kind):
    frame = pd.concat(
        [DISRUPTION_POINTS, _points((1, "disruption", kind, 180.5))], ignore_index=True
    )
    found = check_table(DISRUPTION_LABELS, "disruption", points_frame=frame)
    assert any(
        f.check == "points" and f.shot == 1 and f"more than one {kind}" in f.detail
        for f in found
    )
    assert (
        check_table(DISRUPTION_LABELS, "disruption", points_frame=DISRUPTION_POINTS)
        == []
    )


@pytest.mark.parametrize(
    "kinds",
    [("t_D",), ("t80",), ("t20",), ("t_D", "t80"), ("t_D", "t20"), ("t80", "t20")],
)
def test_a_disruption_requires_all_three_point_kinds(kinds):
    frame = DISRUPTION_POINTS[DISRUPTION_POINTS.kind.isin(kinds)]
    found = check_table(DISRUPTION_LABELS, "disruption", points_frame=frame)
    assert any(
        f.check == "points" and f.shot == 1 and "missing point kinds" in f.detail
        for f in found
    )
    assert (
        check_table(DISRUPTION_LABELS, "disruption", points_frame=DISRUPTION_POINTS)
        == []
    )


@pytest.mark.parametrize("end", [100.25, 99.25])
def test_a_disruption_requires_t80_before_t20(end):
    frame = DISRUPTION_POINTS.copy()
    frame.loc[frame.kind == "t20", "t_ms"] = end
    found = check_table(DISRUPTION_LABELS, "disruption", points_frame=frame)
    assert any(
        f.check == "points" and f.shot == 1 and "t80 must be before t20" in f.detail
        for f in found
    )
    assert (
        check_table(DISRUPTION_LABELS, "disruption", points_frame=DISRUPTION_POINTS)
        == []
    )


@pytest.mark.parametrize(
    "rows",
    [
        [(1, 0, 0, 400)],
        [(1, 0, 0, 100), (1, 1, 100, 150), (1, 0, 150, 200), (1, 1, 200, 250)],
    ],
)
def test_disruption_points_require_exactly_one_present_span(rows):
    found = check_table(_labels(*rows), "disruption", points_frame=DISRUPTION_POINTS)
    assert any(
        f.check == "points" and f.shot == 1 and "exactly one present span" in f.detail
        for f in found
    )
    assert (
        check_table(DISRUPTION_LABELS, "disruption", points_frame=DISRUPTION_POINTS)
        == []
    )


@pytest.mark.parametrize("kind, time", [("t80", 98.99), ("t20", 251.01)])
def test_disruption_present_bounds_allow_only_one_ms_of_rounding(kind, time):
    frame = DISRUPTION_POINTS.copy()
    frame.loc[frame.kind == kind, "t_ms"] = time
    found = check_table(DISRUPTION_LABELS, "disruption", points_frame=frame)
    assert any(
        f.check == "points" and f.shot == 1 and "within 1 ms" in f.detail for f in found
    )
    frame.loc[frame.kind == kind, "t_ms"] = 99 if kind == "t80" else 251
    assert check_table(DISRUPTION_LABELS, "disruption", points_frame=frame) == []


@pytest.mark.parametrize(
    "points_frame", [None, _points(), DISRUPTION_POINTS.assign(shot=2)]
)
def test_each_disruption_present_span_requires_its_own_t80_and_t20(points_frame):
    found = check_table(DISRUPTION_LABELS, "disruption", points_frame=points_frame)
    assert any(
        f.check == "points"
        and f.shot == 1
        and "present span has no t80 and t20" in f.detail
        for f in found
    )
    assert check_table(_labels((1, 0, 0, 400)), "disruption") == []


def test_disruption_present_span_is_checked_when_points_file_is_missing(tmp_path):
    review = tmp_path / "disruption" / "review"
    review.mkdir(parents=True)
    DISRUPTION_LABELS.to_csv(review / "labels.csv", index=False)
    found, n = check_category(review.parent)
    assert n == 1
    assert [(f.check, f.shot) for f in found] == [("points", 1)]
    DISRUPTION_POINTS.to_csv(review / "points.csv", index=False)
    assert check_category(review.parent) == ([], 1)


@pytest.mark.parametrize("row, count", [("190001,0,10,20", 4), ("190001,0", 2)])
def test_windows_csv_field_counts_match_the_header(tmp_path, row, count):
    path = tmp_path / "windows.csv"
    path.write_text(f"shot,window_start_ms,window_end_ms\n{row}\n")
    with pytest.raises(CatalogError, match=f"row 2:.*3 fields.*{count}") as error:
        read_windows(path)
    assert str(path) in str(error.value)


@pytest.mark.parametrize("filename, width", [("labels.csv", 5), ("points.csv", 7)])
@pytest.mark.parametrize("extra", [True, False])
def test_review_csv_field_counts_match_the_header(tmp_path, filename, width, extra):
    review = tmp_path / "edge_localized_mode" / "review"
    review.mkdir(parents=True)
    _labels((190002, 1, 0, 100)).to_csv(review / "labels.csv", index=False)
    header, row = (
        (INTERVAL_COLUMNS, "190002,1,0,100,")
        if filename == "labels.csv"
        else (POINT_COLUMNS, "190002,edge_localized_mode,elm,50,,,")
    )
    row = "190001," + row if extra else row[:-1]
    (review / filename).write_text(",".join(header) + "\n" + row + "\n")
    found, count = check_category(review.parent)
    assert count == 1
    assert [(f.check, f.where) for f in found] == [
        ("schema", f"edge_localized_mode/review/{filename}")
    ]
    assert "row 2:" in found[0].detail
    assert f"{width} fields" in found[0].detail
    assert f"got {width + (1 if extra else -1)}" in found[0].detail


def test_windows_csv_allows_named_cohort_columns_and_quoted_commas(tmp_path):
    path = tmp_path / "windows.csv"
    path.write_text(
        "shot,group,window_start_ms,window_end_ms,weight\n"
        '190001,"train,first",0.25,100.75,1\n'
    )
    assert read_windows(path) == {190001: (0.25, 100.75)}


@pytest.mark.parametrize(
    "header,row,repeated",
    [
        ("shot,shot,window_start_ms,window_end_ms", "190002,190001,0,10", "shot"),
        (
            "shot,window_start_ms,window_end_ms,window_end_ms",
            "1,10,400,50",
            "window_end_ms",
        ),
        (
            "shot,window_start_ms,window_start_ms,window_end_ms",
            "1,0,10,400",
            "window_start_ms",
        ),
        (
            "shot,group,group,window_start_ms,window_end_ms",
            "1,a,b,10,400",
            "group",
        ),
    ],
)
def test_duplicate_windows_headers_are_rejected(tmp_path, header, row, repeated):
    path = tmp_path / "windows.csv"
    path.write_text(header + "\n" + row + "\n")
    with pytest.raises(CatalogError, match=f"row 1:.*duplicate.*{repeated}") as error:
        read_windows(path)
    assert str(path) in str(error.value)


@pytest.mark.parametrize("filename", ["labels.csv", "points.csv"])
def test_duplicate_review_headers_are_schema_findings(tmp_path, filename):
    review = tmp_path / "edge_localized_mode" / "review"
    review.mkdir(parents=True)
    _labels((1, 1, 0, 100)).to_csv(review / "labels.csv", index=False)
    header, row = (
        (INTERVAL_COLUMNS, "1,1,0,100,")
        if filename == "labels.csv"
        else (POINT_COLUMNS, "1,edge_localized_mode,elm,50,,,")
    )
    (review / filename).write_text(",".join(header) + ",shot\n" + row + ",2\n")
    found, count = check_category(review.parent)
    assert count == 1
    assert [(f.check, f.where) for f in found] == [
        ("schema", f"edge_localized_mode/review/{filename}")
    ]
    assert "row 1:" in found[0].detail
    assert "duplicate" in found[0].detail and "shot" in found[0].detail


@pytest.mark.parametrize("filename", ["labels.csv", "points.csv", "windows.csv"])
def test_catalog_csv_readers_skip_blank_lines(tmp_path, filename):
    review = tmp_path / "edge_localized_mode" / "review"
    review.mkdir(parents=True)
    _labels((1, 1, 0, 100)).to_csv(review / "labels.csv", index=False)
    _points((1, "edge_localized_mode", "elm", 50)).to_csv(
        review / "points.csv", index=False
    )
    path = review / filename
    if filename == "windows.csv":
        path.write_text("shot,window_start_ms,window_end_ms\n1,0,100\n")
    path.write_text("\n" + path.read_text().replace("\n", "\n\n"))
    if filename == "windows.csv":
        assert read_windows(path) == {1: (0, 100)}
    assert check_category(review.parent) == ([], 1)


@pytest.mark.parametrize(
    "span", [(0, 400), [0.0, 400.0], np.array([0, 400]), np.array([0.0, 400.0])]
)
def test_allowed_windows_accept_finite_numeric_sequences(span):
    assert check_table(GOOD, "alfven_eigenmode", allowed={1: span}) == []


@pytest.mark.parametrize(
    "span",
    [
        np.array([False, True]),
        np.array([0, np.nan]),
        np.array([0, np.inf]),
        np.array([[0, 1], [400, 500]]),
        np.array(400),
        np.array([0, 1, 400]),
        np.array([400, 0]),
        np.array([0, 0]),
        {0, 400},
        {0: 0, 1: 400},
        iter([0, 400]),
    ],
)
def test_allowed_windows_refuse_invalid_array_spans_and_nonsequences(span):
    found = check_table(GOOD, "alfven_eigenmode", allowed={1: span})
    assert len(found) == 1 and found[0].check == "windows"
    assert "two finite numbers with start < end" in found[0].detail


def test_header_only_windows_csv_is_refused(tmp_path):
    path = tmp_path / "windows.csv"
    path.write_text("shot,window_start_ms,window_end_ms\n")
    with pytest.raises(CatalogError, match="row 1:.*no windows"):
        read_windows(path)


@pytest.mark.parametrize(
    "span",
    [
        (float("nan"), float("nan")),
        (-float("inf"), float("inf")),
        (100, 0),
        (0, 0),
        (0,),
        (0, 100, 200),
        ("0", "100"),
        (False, 100),
        None,
    ],
)
@pytest.mark.parametrize("assessed_shot", [190001, 190002])
def test_public_checker_validates_every_allowed_span(span, assessed_shot):
    # Even a shot without labels must not hide an invalid mapping entry.
    found = check_table(
        _labels((assessed_shot, 0, 0, 100)),
        "alfven_eigenmode",
        allowed={190001: span},
    )
    assert len(found) == 1 and found[0].check == "windows"
    assert "190001" in found[0].detail
    with pytest.raises(CatalogError):
        require(found)


@pytest.mark.parametrize("shot", ["190001", True, 1.0, -1, 2**63])
def test_public_checker_validates_allowed_shot_keys(shot):
    found = check_table(GOOD, "alfven_eigenmode", allowed={shot: (0, 400)})
    assert len(found) == 1 and found[0].check == "windows"
    assert repr(shot) in found[0].detail


@pytest.mark.parametrize(
    "span", [(100, 0), (float("nan"), float("nan")), (-float("inf"), float("inf"))]
)
def test_windows_itself_refuses_unordered_or_nonfinite_spans(span):
    found = windows(GOOD, {1: span})
    assert len(found) == 1 and found[0].check == "windows"
    assert found[0].shot == 1


def test_bad_labels_schema_still_checks_independent_points(tmp_path):
    review = tmp_path / "edge_localized_mode" / "review"
    review.mkdir(parents=True)
    (review / "labels.csv").write_text("shot,wrong\n190001,1\n")
    _points((190001, "edge_localized_mode", "t_Q", 50)).to_csv(
        review / "points.csv", index=False
    )
    found, count = check_category(review.parent)
    assert count == 1
    assert [(f.check, f.where) for f in found] == [
        ("schema", "edge_localized_mode/review/labels.csv"),
        ("points", "edge_localized_mode/review/points.csv"),
    ]
    assert "t_Q" in found[1].detail


def test_raw_points_fallback_also_checks_csv_field_counts(tmp_path, monkeypatch):
    from labeler.events.catalog import check
    from labeler.events.databases import DatabaseError

    review = tmp_path / "edge_localized_mode" / "review"
    review.mkdir(parents=True)
    GOOD.to_csv(review / "labels.csv", index=False)
    (review / "points.csv").write_text(
        ",".join(POINT_COLUMNS) + "\n190001,1,edge_localized_mode,elm,50,,,\n"
    )

    def invalid_schema(path):
        raise DatabaseError("bad schema")

    monkeypatch.setattr(check, "read_points", invalid_schema)
    found, count = check_category(review.parent)
    assert count == 1 and len(found) == 1
    assert found[0].check == "schema"
    assert "row 2:" in found[0].detail
    assert "7 fields" in found[0].detail and "got 8" in found[0].detail


@pytest.mark.parametrize("end, margin", [(104.25, 2), (249.75, 37.375)])
@pytest.mark.parametrize(
    "bound, offset, outside",
    [
        ("lower", -0.01, True),
        ("lower", 0, False),
        ("lower", 0.01, False),
        ("upper", -0.01, False),
        ("upper", 0, False),
        ("upper", 0.01, True),
    ],
)
def test_disruption_t_d_quench_margin_bounds(end, margin, bound, offset, outside):
    start = 100.25
    time = (start - margin if bound == "lower" else end + margin) + offset
    labels = _labels(
        (1, 0, 0, round(start)),
        (1, 1, round(start), round(end)),
        (1, 0, round(end), 400),
    )
    frame = _points(
        (1, "disruption", "t_D", time),
        (1, "disruption", "t80", start),
        (1, "disruption", "t20", end),
    )
    found = check_table(labels, "disruption", points_frame=frame)
    if outside:
        assert len(found) == 1 and found[0].check == "points"
        assert found[0].shot == 1
        assert "t_D" in found[0].detail and "outside the quench" in found[0].detail
        assert f"margin {margin:.3g} ms" in found[0].detail
        assert "D17" in found[0].detail
    else:
        assert found == []


@pytest.mark.parametrize("time", [10, 399])
def test_disruption_t_d_far_from_its_quench_is_a_finding(time):
    frame = DISRUPTION_POINTS.copy()
    frame.loc[frame.kind == "t_D", "t_ms"] = time
    found = check_table(GOOD[GOOD.shot == 1], "disruption", points_frame=frame)
    assert len(found) == 1 and found[0].check == "points"
    assert f"t_D at {time} ms" in found[0].detail
    assert "100.25-249.75 ms" in found[0].detail
    assert "margin 37.4 ms" in found[0].detail


def test_disruption_missing_t_d_does_not_hide_a_span_mismatch():
    labels = _labels((1, 0, 0, 300), (1, 1, 300, 450), (1, 0, 450, 500))
    frame = DISRUPTION_POINTS[DISRUPTION_POINTS.kind != "t_D"]
    found = check_table(labels, "disruption", points_frame=frame)
    assert [(f.check, f.shot, f.detail) for f in found] == [
        ("points", 1, "missing point kinds: t_D"),
        ("points", 1, "present span bounds must be within 1 ms of t80 and t20"),
    ]


@pytest.mark.parametrize(
    "category, key, bad, good",
    [
        ("neoclassical_tearing_mode", "m", 1, 2),
        ("neoclassical_tearing_mode", "n", 0, 1),
        ("edge_localized_mode", "frequency_hz", 0, 0.01),
        ("sawtooth_oscillation", "period_ms", 0, 0.01),
        ("sawtooth_oscillation", "inversion_channel", 0, 1),
        ("sawtooth_oscillation", "inversion_channel", 49, 48),
        ("sawtooth_oscillation", "inversion_radius_m", 0, 0.01),
    ],
)
@pytest.mark.parametrize("valid", [False, True])
def test_attribute_physical_bounds_in_the_checker(category, key, bad, good, valid):
    value = good if valid else bad
    found = check_table(_labels((1, 1, 0, 100, json.dumps({key: value}))), category)
    if valid:
        assert found == []
    else:
        assert len(found) == 1 and found[0].check == "attrs"
        assert found[0].shot == 1 and key in found[0].detail


@pytest.mark.parametrize(
    "category, always",
    [
        ("alfven_eigenmode", False),
        ("neoclassical_tearing_mode", True),
        ("high_confinement_mode", False),
        ("edge_localized_mode", False),
        ("sawtooth_oscillation", False),
        ("disruption", True),
    ],
)
def test_not_observable_state_depends_on_the_phenomenon(category, always):
    found = check_table(_labels((1, 3, 0, 100)), category)
    if always:
        assert len(found) == 1 and found[0].check == "states"
        assert found[0].shot == 1
        assert category in found[0].detail
        assert "never not observable" in found[0].detail
    else:
        assert found == []


@pytest.mark.parametrize("column", ["t_start", "t_end"])
def test_label_boundaries_must_be_whole_ms(column):
    labels = _labels((190001, 0, 1000, 2000))
    labels[column] = 1000.5 if column == "t_start" else 2000.25
    found = check_table(labels, "alfven_eigenmode")
    assert any(f.check == "whole_ms" and "row 0" in f.detail for f in found)


def test_checker_to_scoring_keeps_d19_at_original_allowed_end():
    from labeler.scoring.events import points_within
    from labeler.scoring.frames import Assessment

    rows = [(8, 5256, 0), (5256, 5261, 1)]
    labels = _labels(*[(191389, s, a, b) for a, b, s in rows])
    times = [5255.65, 5260.53, 5260.54]
    frame = _points(
        *[(191389, "disruption", k, t) for k, t in zip(["t80", "t_D", "t20"], times)]
    )
    allowed = (8, 5260)
    assert (
        check_table(labels, "disruption", allowed={191389: allowed}, points_frame=frame)
        == []
    )
    assessed = Assessment.from_checked(rows, allowed, category="disruption")
    assert assessed.window == allowed
    assert assessed.spans == ((5256, 5260, 1),)
    got = points_within(
        times + [5262.0, 5262.001], assessed.window, allowed, category="disruption"
    )
    assert got.tolist() == times + [5262.0]
    short = Assessment.from_checked([(8, 5000, 0)], allowed, category="disruption")
    assert short.window == (8, 5000)
    assert points_within(
        [4999.5, 5000, 5260], short.window, allowed, category="disruption"
    ).tolist() == [4999.5]
    with pytest.raises(ValueError, match="allowed"):
        Assessment.from_checked(rows, allowed, category="alfven_eigenmode")
