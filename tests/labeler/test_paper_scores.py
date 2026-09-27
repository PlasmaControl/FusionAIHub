"""The paper's score figures and tables: AE filled, the other phenomena coming."""

from __future__ import annotations

import math

import pytest

from labeler.paper import COMING, scores

from . import paper_tree as tree


def _texts(ax) -> list[str]:
    return [t.get_text() for t in ax.texts]


def test_an_undefined_estimate_is_nan():
    assert all(math.isnan(x) for x in scores.interval(tree.est(None)))
    assert scores.interval(tree.est(0.5, 0.4, 0.6)) == (0.5, 0.4, 0.6)
    assert scores.read("/nonexistent/evaluation.json") is None


def test_the_scores_grid_fills_ae_and_marks_the_rest_coming(tmp_path):
    ae = tree.ae_evaluation()
    fig = scores.draw_scores(ae, tmp_path / "fig_scores")
    assert (tmp_path / "fig_scores.pdf").stat().st_size > 0
    assert (tmp_path / "fig_scores.png").stat().st_size > 0
    first, *rest = fig.axes
    assert first.get_title() == "AE: 40 test shots, 9000 frames"
    assert [ax.get_title() for ax in rest] == [
        "NTM",
        "H-mode",
        "ELMing",
        "sawteeth",
        "disruption",
    ]
    assert all(_texts(ax) == [COMING] for ax in rest)
    assert COMING not in _texts(first)
    heights = [bar.get_height() for bar in first.patches]
    expected = [
        ae["methods"][m][metric]["value"]
        for m in scores.FIG_METHODS
        for metric in scores.METRICS
    ]
    assert heights == pytest.approx(expected)
    [legend] = fig.legends
    assert [t.get_text() for t in legend.get_texts()] == [
        "ae_xpower",
        "SELDnet",
        "TokEye",
        "always",
    ]


def test_the_mhd_figure_has_every_method_against_the_bar(tmp_path):
    ae = tree.ae_evaluation()
    fig = scores.draw_mhd(ae, tmp_path / "fig_mhd")
    [ax] = fig.axes
    widths = [bar.get_width() for bar in ax.patches]
    assert widths == pytest.approx(
        [0.02 * (i + 1) for i in range(6)] + [0.01 * (i + 1) for i in range(6)]
    )
    [line] = ax.lines
    assert list(line.get_xdata()) == [0.05, 0.05]
    assert ax.get_title() == "600 MHD frames in 12 test shots"


def test_the_segmentation_figure(tmp_path):
    seg = tree.seg_evaluation()
    fig = scores.draw_segmentation(seg, tmp_path / "fig_segmentation")
    [ax] = fig.axes
    heights = [bar.get_height() for bar in ax.patches]
    expected = [
        seg["methods"][m][metric]["value"]
        for m in scores.SEG_NAMES
        for metric in scores.SEG_METRICS
    ]
    assert heights == pytest.approx(expected)
    assert ax.get_title() == "40 test shots: 123,456 AE pixels of 7,654,321"


def test_the_tables():
    ae = tree.ae_evaluation()
    ae["methods"] = {m: ae["methods"][m] for m in ("ae_xpower", "always")}
    ae["methods"]["always"]["fp_rate_other"] = tree.est(None)
    assert scores.table_ae(ae) == (
        "% AE frame scores: 40 test shots, 9000 frames (2500 present, 600 MHD); "
        "95% shot-bootstrap intervals; band80-mhd3 at 0.42; "
        "bar A1 pass, A2 fail, A3 pass, all fail\n"
        "\\begin{tabular}{lccccc}\n"
        "\\toprule\n"
        "Method & Precision & Recall & F1 & FP (MHD) & FP (other) \\\\\n"
        "\\midrule\n"
        "ae\\_xpower & 0.90 [0.85, 0.93] & 0.88 [0.83, 0.91] & 0.89 [0.84, 0.92] "
        "& 0.02 [0.01, 0.03] & 0.01 \\\\\n"
        "always & 0.40 [0.35, 0.43] & 0.38 [0.33, 0.41] & 0.39 [0.34, 0.42] "
        "& 0.12 [0.06, 0.18] & -- \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )
    table = scores.table_segmentation(tree.seg_evaluation())
    assert table.splitlines()[3] == (
        "Method & Dice & Frame P & Frame R & Frame F1 & FP (MHD) \\\\"
    )
    assert table.splitlines()[5].startswith("ae\\_seg & 0.80 [0.76, 0.84] & ")
    assert len(table.splitlines()) == 10
