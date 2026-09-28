"""The paper's coverage figure and dataset table: AE counted, the rest coming."""

from __future__ import annotations

from labeler.paper import COMING, coverage

from . import paper_tree as tree

SUMMARY = (
    "shot,year,window_start_ms,window_end_ms,frames,present_frames,"
    "not_observable_frames,present_runs,max_prob\n"
    "201,2024,0,5000,500,0,0,0,0.1\n"
    "202,2024,0,5000,500,40,0,2,0.9\n"
    "203,2025,0,5000,500,3,0,1,0.7\n"
    "204,,0,5000,500,0,0,0,0.2\n"
)


def _inputs(tmp_path):
    event = tmp_path / "alfven_eigenmode"
    (event / "review").mkdir(parents=True)
    (event / "review" / "labels.csv").write_text(
        "shot,category,t_start,t_end,confidence\n"
        "101,0,0,300,\n101,1,300,900,\n101,0,900,2000,\n"
        "102,0,0,2000,\n"
        "103,2,0,100,\n103,1,100,400,\n103,0,400,2000,\n"
    )
    split = tmp_path / "split.csv"
    split.write_text("shot,split\n101,train\n102,val\n103,test\n104,test\n")
    summary = tmp_path / "summary.csv"
    summary.write_text(SUMMARY)
    return event, split, summary


def test_ae_counts(tmp_path):
    event, split, summary = _inputs(tmp_path)
    counts = coverage.ae_counts(event, split, summary)
    assert (counts.reviewed, counts.positive, counts.present_s) == (3, 2, 0.9)
    assert counts.split == {"train": 1, "val": 1, "test": 1}, "104 was not reviewed"
    assert counts.by_year == {coverage.UNKNOWN_YEAR: (1, 0), 2024: (2, 1), 2025: (1, 1)}
    assert (counts.suggested, counts.suggested_positive) == (4, 2)
    bare = coverage.ae_counts(event, None, tmp_path / "missing.csv")
    assert bare.split is None, "no split: the model was not chosen"
    assert bare.by_year is None, "no summary: the extension did not run"
    assert (bare.suggested, bare.suggested_positive) == (None, None)


def _coming(ax) -> int:
    return sum(t.get_text() == COMING for t in ax.texts)


def _texts(ax) -> list[str]:
    return [t.get_text() for t in ax.texts]


def test_the_coverage_figure(tmp_path):
    counts = {"alfven_eigenmode": coverage.ae_counts(*_inputs(tmp_path))}
    fig = coverage.draw_coverage(counts, tmp_path / "fig_coverage")
    assert (tmp_path / "fig_coverage.pdf").is_file()
    shots, present, split, years = fig.axes
    assert _coming(shots) == 5 and _coming(present) == 5
    assert [bar.get_width() for bar in shots.patches] == [3, 2], "reviewed, positive"
    [legend] = fig.legends
    assert [t.get_text() for t in legend.get_texts()] == [
        "reviewed",
        "with a present span",
    ]
    assert [bar.get_width() for bar in present.patches] == [0.9]
    assert "0.9" in _texts(present)
    assert [bar.get_height() for bar in split.patches] == [1, 1, 1]
    assert [t.get_text() for t in split.get_xticklabels()] == ["train", "val", "test"]
    assert [bar.get_height() for bar in years.patches] == [2, 1, 1, 1, 1, 0]
    assert [t.get_text() for t in years.get_xticklabels()] == ["2024", "2025", "?"]
    assert "suggest" in years.get_title()
    assert tree.small_text(fig) == []


def test_without_the_extension_the_year_panel_says_not_run(tmp_path):
    event, _, _ = _inputs(tmp_path)
    counts = {"alfven_eigenmode": coverage.ae_counts(event, None, None)}
    fig = coverage.draw_coverage(counts, tmp_path / "fig_coverage")
    _, _, split, years = fig.axes
    for ax in (split, years):
        assert _texts(ax) == [coverage.NOT_RUN]
        assert ax.get_legend() is None and len(ax.patches) == 0
    [legend] = fig.legends
    assert [t.get_text() for t in legend.get_texts()] == [
        "reviewed",
        "with a present span",
    ]
    assert tree.small_text(fig) == []


def test_the_datasets_table(tmp_path):
    counts = {"alfven_eigenmode": coverage.ae_counts(*_inputs(tmp_path))}
    lines = coverage.table_datasets(counts).splitlines()
    assert lines[1] == "\\begin{tabular}{lcccccc}"
    assert lines[3] == (
        "Phenomenon & Reviewed & Positive & Present (s) & Train / val / test "
        "& Suggested & Suggested positive \\\\"
    )
    assert lines[5] == "AE & 3 & 2 & 0.9 & 1 / 1 / 1 & 4 & 2 \\\\"
    assert lines[6:11] == [
        f"{name} & \\multicolumn{{6}}{{c}}{{coming}} \\\\"
        for name in ("NTM", "H-mode", "ELMing", "sawteeth", "disruption")
    ]


def test_without_the_extension_the_table_says_so(tmp_path):
    event, _, _ = _inputs(tmp_path)
    counts = {"alfven_eigenmode": coverage.ae_counts(event, None, None)}
    lines = coverage.table_datasets(counts).splitlines()
    assert lines[5] == "AE & 3 & 2 & 0.9 & -- & -- & -- \\\\"
