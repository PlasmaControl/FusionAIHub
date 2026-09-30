"""The paper's coverage figure and dataset table: AE counted, the rest coming."""

from __future__ import annotations

import dataclasses
from itertools import pairwise

import pandas as pd
import pytest

from labeler.ae.xpower.train import read_split
from labeler.events.review import labels
from labeler.paper import COMING, coverage

from . import paper_tree as tree


@pytest.fixture(autouse=True)
def paths(tmp_path, monkeypatch):
    """A temporary Paths, set before every call, though these read none."""
    return tree.temporary_paths(tmp_path, monkeypatch)


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
        "105,0,0,2000,\n"
    )
    split = tmp_path / "split.csv"
    split.write_text("shot,split\n101,train\n102,val\n103,test\n104,test\n")
    summary = tmp_path / "summary.csv"
    summary.write_text(SUMMARY)
    return labels.read_saved(event), read_split(split), pd.read_csv(summary)


def test_ae_counts(tmp_path):
    saved, split, summary = _inputs(tmp_path)
    counts = coverage.ae_counts(saved, split, summary)
    assert (counts.reviewed, counts.positive, counts.present_s) == (4, 2, 0.9)
    assert counts.split == {"train": 1, "val": 1, "test": 1}, "104 was not reviewed"
    assert counts.unsplit == 1, "105 was reviewed after the split"
    assert counts.folds is None, "a validation split, not cross-validated"
    assert counts.cross_validated is False
    folded = coverage.ae_counts(saved, split, summary, folds=3)
    assert (folded.folds, folded.cross_validated) == (3, True)
    uncounted = coverage.ae_counts(saved, split, summary, cross_validated=True)
    assert (uncounted.folds, uncounted.cross_validated) == (None, True)
    assert counts.by_year == {coverage.UNKNOWN_YEAR: (1, 0), 2024: (2, 1), 2025: (1, 1)}
    assert (counts.suggested, counts.suggested_positive) == (4, 2)
    bare = coverage.ae_counts(saved, None, None)
    assert bare.split is None, "no split: the model was not chosen"
    assert bare.unsplit is None
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
    assert _coming(shots) == 4 and _coming(present) == 4
    assert [bar.get_width() for bar in shots.patches] == [4, 2], "labelled, positive"
    legend, key = fig.legends
    assert [t.get_text() for t in legend.get_texts()] == [
        "labelled",
        "with a present span",
    ], "AE's labelled shots are its reviewed ones"
    assert [t.get_text() for t in key.get_texts()] == list(coverage.SUGGESTED)
    assert years.get_legend() is None, "the years' entries are in the shared key"
    assert [bar.get_width() for bar in present.patches] == [0.9]
    assert "0.9" in _texts(present)
    assert [bar.get_height() for bar in split.patches] == [1, 1, 1, 1]
    assert [t.get_text() for t in split.get_xticklabels()] == [
        "train",
        "val",
        "test",
        "no\nsplit",
    ]
    assert [bar.get_width() for bar in years.patches] == [2, 1, 1, 1, 1, 0]
    assert [t.get_text() for t in years.get_yticklabels()] == ["2024", "2025", "?"]
    assert "suggest" in years.get_title()
    assert tree.small_text(fig) == []


TODAY = {
    "reviewed": 180,
    "positive": 180,
    "present_s": 316.3,
    "split": {"train": 88, "val": 16, "test": 58},
    "unsplit": 18,
}  # AE's counts on 2026-09-28, so the axis is as wide as the paper's
CROSS_VALIDATED = {
    "reviewed": 198,
    "positive": 198,
    "present_s": 316.3,
    "split": {"train": 120, "val": 0, "test": 60},
    "unsplit": 18,
    "folds": 5,
    "cross_validated": True,
}  # v2 as it is made: 120 shots dealt into five folds, and 60 test shots


def test_the_fold_count_is_the_distinct_folds(tmp_path):
    (tmp_path / "folds.csv").write_text(
        "shot,split,fold\n101,train,0\n102,val,2\n103,test,\n106,train,1\n"
    )
    folds = pd.read_csv(tmp_path / "folds.csv")
    assert coverage.fold_count(folds) == 3, "folds 0-2; a test shot has none"
    ten = pd.DataFrame({"shot": range(10), "fold": [k % 5 for k in range(10)]})
    assert coverage.fold_count(ten) == 5
    unfolded = pd.DataFrame({"shot": [101], "split": ["train"]})
    assert coverage.fold_count(unfolded) is None, "no fold column: not counted"
    assert coverage.fold_count(pd.DataFrame()) is None


FOLDED_ROW = "AE & 198 & 198 & 198 & 316.3 & 120 & -- & 60 & 18 & -- & -- \\\\"


def test_a_cross_validated_split_shows_its_folds(tmp_path):
    folded = coverage.Counts(**CROSS_VALIDATED)
    fig = coverage.draw_coverage(
        {"alfven_eigenmode": folded}, tmp_path / "fig_coverage"
    )
    split = fig.axes[2]
    assert [bar.get_height() for bar in split.patches] == [120, 60, 18]
    assert [t.get_text() for t in split.get_xticklabels()] == [
        "train\n(5 folds)",
        "test",
        "no\nsplit",
    ], "the folds' shots as one group, and no validation bar"
    assert tree.small_text(fig) == []
    lines = coverage.table_datasets({"alfven_eigenmode": folded}).splitlines()
    assert lines[5] == FOLDED_ROW
    assert "AE's train shots are cross-validated over 5 folds" in lines[0]
    assert "no shot is held out for validation (--)" in lines[0]
    with_val = dataclasses.replace(folded, split={"train": 100, "val": 20, "test": 60})
    fig = coverage.draw_coverage({"alfven_eigenmode": with_val}, tmp_path / "fig")
    assert [bar.get_height() for bar in fig.axes[2].patches] == [100, 60, 18]
    assert [t.get_text() for t in fig.axes[2].get_xticklabels()] == [
        "train\n(5 folds)",
        "test",
        "no\nsplit",
    ], "no validation bar in every case (the build names such shots, `CV_VAL`)"
    lines = coverage.table_datasets({"alfven_eigenmode": with_val}).splitlines()
    assert "& 100 & -- & 60 &" in lines[5]
    uncounted = dataclasses.replace(folded, folds=None)
    fig = coverage.draw_coverage({"alfven_eigenmode": uncounted}, tmp_path / "fig")
    assert [t.get_text() for t in fig.axes[2].get_xticklabels()] == [
        "train",
        "test",
        "no\nsplit",
    ], "cross-validated, its folds not counted: no count, and still no val"
    lines = coverage.table_datasets({"alfven_eigenmode": uncounted}).splitlines()
    assert lines[5] == FOLDED_ROW
    assert lines[0].endswith(
        "; AE's train shots are cross-validated, so no shot is held out for "
        "validation (--)"
    )


@pytest.mark.parametrize("counts", [TODAY, CROSS_VALIDATED], ids=["v1", "folds"])
def test_the_split_ticks_do_not_touch(tmp_path, counts):
    counts = coverage.Counts(**counts)
    fig = coverage.draw_coverage({"alfven_eigenmode": counts}, tmp_path / "fig")
    fig.draw_without_rendering()  # lay out at the figure's own dpi again
    boxes = [t.get_window_extent() for t in fig.axes[2].get_xticklabels()]
    gaps = [(b.x0 - a.x1) * 72 / fig.dpi for a, b in pairwise(boxes)]
    assert min(gaps) >= coverage.FONT_PT - 1, f"at least an em apart (pt): {gaps}"


def test_without_the_extension_the_year_panel_says_not_run(tmp_path):
    saved, _, _ = _inputs(tmp_path)
    counts = {"alfven_eigenmode": coverage.ae_counts(saved, None, None)}
    fig = coverage.draw_coverage(counts, tmp_path / "fig_coverage")
    _, _, split, years = fig.axes
    for ax in (split, years):
        assert _texts(ax) == [coverage.NOT_RUN]
        assert ax.get_legend() is None and len(ax.patches) == 0
    [legend] = fig.legends
    assert [t.get_text() for t in legend.get_texts()] == [
        "labelled",
        "with a present span",
    ], "AE's labelled shots are its reviewed ones"
    assert tree.small_text(fig) == []


def test_the_datasets_table(tmp_path):
    counts = {"alfven_eigenmode": coverage.ae_counts(*_inputs(tmp_path))}
    lines = coverage.table_datasets(counts).splitlines()
    assert lines[0] == (
        "% Shots per phenomenon: labelled, reviewed by the owner, with any present "
        "span and their present time, the model's split (Train, Val, Test) and the "
        "extension's suggestions (not labels); -- where that run has not happened; "
        "AE's labels are the owner's reviews, so its Labelled = Reviewed = Train + "
        "Val + Test + No split (the reviewed shots saved after the model was "
        "trained)"
    ), "a validation split's comment: no frame phenomenon, no dagger"
    assert lines[1] == "\\begin{tabular}{lcccccccccc}"
    assert lines[3] == (
        "Phenomenon & Labelled & Reviewed by the owner & Positive & Present (s) "
        "& Train & Val & Test & No split & Suggested & Suggested positive \\\\"
    )
    assert lines[5] == "AE & 4 & 4 & 2 & 0.9 & 1 & 1 & 1 & 1 & 4 & 2 \\\\"
    assert lines[6:10] == [
        f"{name} & \\multicolumn{{10}}{{c}}{{coming}} \\\\"
        for name in ("NTM", "H-mode", "ELMing", "sawteeth")
    ]


def test_without_the_extension_the_table_says_so(tmp_path):
    saved, _, _ = _inputs(tmp_path)
    counts = {"alfven_eigenmode": coverage.ae_counts(saved, None, None)}
    lines = coverage.table_datasets(counts).splitlines()
    assert lines[5] == "AE & 4 & 4 & 2 & 0.9 & -- & -- & -- & -- & -- & -- \\\\"
