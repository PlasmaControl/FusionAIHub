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


def _legend(fig) -> list[str]:
    [legend] = fig.legends
    return [t.get_text() for t in legend.get_texts()]


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
    assert _legend(fig)[: len(scores.AE_METHODS)] == [
        scores.AE_NAMES[m] for m in scores.AE_METHODS
    ]
    assert tree.small_text(fig) == []


def test_the_scores_axis_starts_at_the_floor_and_marks_what_is_below(tmp_path):
    ae = tree.ae_evaluation()
    fig = scores.draw_scores(ae, tmp_path / "fig_scores")
    first = fig.axes[0]
    assert first.get_ylim()[0] == scores.SCORE_FLOOR
    assert f"{scores.SCORE_FLOOR:g}" in first.get_ylabel()
    dots = {c.get_label(): c.lines[0].get_ydata() for c in first.containers}
    below = []
    for m in scores.AE_METHODS:
        values = [ae["methods"][m][k]["value"] for k in scores.METRICS]
        drawn = [max(v, scores.SCORE_FLOOR) for v in values]
        assert list(dots[scores.AE_NAMES[m]]) == pytest.approx(drawn), m
        below += [f"{v:.2f}" for v in values if v < scores.SCORE_FLOOR]
    assert below and sorted(_texts(first)) == sorted(below)
    marks = [c for c in first.collections if "bar" in c.get_label()]
    levels = sorted(float(c.get_segments()[0][0][1]) for c in marks)
    assert levels == [0.75, 0.75, 0.9], "A1: precision, recall, F1"


def test_the_figures_and_the_table_show_the_same_ae_methods(tmp_path):
    ae = tree.ae_evaluation()
    names = [scores.AE_NAMES[m] for m in scores.AE_METHODS]
    fig = scores.draw_mhd(ae, tmp_path / "fig_mhd")
    assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == names
    rows = scores.table_ae(ae).splitlines()[5 : 5 + len(names)]
    assert [row.split(" & ")[0].replace("\\_", "_") for row in rows] == names


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
    assert ax.get_xlim() == (0, scores.MHD_CAP)
    assert tree.small_text(fig) == []


def test_the_mhd_figure_draws_a_bar_past_its_cap_to_the_edge(tmp_path):
    ae = tree.ae_evaluation()
    ae["methods"]["always"]["fp_rate_mhd"] = tree.est(1.0, 1.0, 1.0)
    ae["methods"]["always"]["fp_rate_other"] = tree.est(1.0, 1.0, 1.0)
    fig = scores.draw_mhd(ae, tmp_path / "fig_mhd")
    [ax] = fig.axes
    widths = [bar.get_width() for bar in ax.patches]
    assert widths[5] == widths[11] == scores.MHD_CAP
    assert _texts(ax) == ["1.00", "1.00"]
    for t in ax.texts:
        assert t.get_position()[0] <= scores.MHD_CAP


def test_the_segmentation_figure(tmp_path):
    seg = tree.seg_evaluation()
    fig = scores.draw_segmentation(seg, tmp_path / "fig_segmentation")
    [ax] = fig.axes
    bars = [p for p in ax.patches if p.get_height() > 0]
    heights = [bar.get_height() for bar in bars]
    expected = [
        seg["methods"][m][metric]["value"]
        for m in scores.SEG_NAMES
        for metric in scores.SEG_METRICS
    ]
    assert heights == pytest.approx(expected)
    n = len(scores.SEG_METRICS)
    hatched = [bool(bar.get_hatch()) for bar in bars]
    assert hatched == [False] * n + [True] * (2 * n), "recipe and TokEye: sources"
    assert ax.get_title() == "40 test shots: 123,456 AE pixels of 7,654,321"
    marks = [c for c in ax.collections if "bar" in c.get_label()]
    levels = sorted(round(float(c.get_segments()[0][0][1]), 2) for c in marks)
    assert levels == [0.05, 0.65, 0.75, 0.9], "G1 (both), G2, G3"
    assert "pseudo-mask source" in _legend(fig)
    assert tree.small_text(fig) == []


def test_the_tables():
    ae = tree.ae_evaluation()
    ae["methods"] = {m: ae["methods"][m] for m in ("ae_xpower", "always")}
    ae["methods"]["always"]["fp_rate_other"] = tree.est(None)
    ae["methods"]["always"]["fp_rate_mhd"] = tree.est(0.0488, 0.0401, 0.0508)
    assert scores.table_ae(ae) == (
        "% AE frame scores: 40 test shots, 9000 frames (2500 present, 600 MHD); "
        "95% shot-bootstrap intervals; band80-mhd3 at 0.42; "
        "bar A1 pass, A2 fail, A3 pass, all fail\n"
        "\\begin{tabular}{lccccc}\n"
        "\\toprule\n"
        "Method & Precision & Recall & F1 & FP (MHD) & FP (other) \\\\\n"
        "\\midrule\n"
        "ae\\_xpower & 0.900 [0.850, 0.930] & 0.880 [0.830, 0.910] "
        "& 0.890 [0.840, 0.920] & 0.020 [0.010, 0.030] & 0.010 \\\\\n"
        "always & 0.400 [0.350, 0.430] & 0.380 [0.330, 0.410] "
        "& 0.390 [0.340, 0.420] & 0.049 [0.040, 0.051] & -- \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )
    table = scores.table_segmentation(tree.seg_evaluation())
    assert table.splitlines()[3] == (
        "Method & Dice & Frame P & Frame R & Frame F1 & FP (MHD) \\\\"
    )
    assert table.splitlines()[5].startswith("ae\\_seg & 0.800 [0.760, 0.840] & ")
    assert len(table.splitlines()) == 10


def test_the_differences_table_gives_the_verdicts_they_decide():
    lines = scores.table_differences(
        tree.ae_evaluation(), tree.seg_evaluation()
    ).splitlines()
    assert lines[1] == "\\begin{tabular}{lcccc}"
    assert lines[3] == "Comparison & Difference & Condition & Holds & Bar \\\\"
    rows = [
        (
            "AE F1: ae\\_xpower $-$ SELDnet",
            "+0.100 [-0.020, +0.200]",
            "low $\\geq -0.03$",
            "yes",
            "A1 pass",
        ),
        (
            "AE MHD FP: ae\\_xpower $-$ SELDnet",
            "-0.020 [-0.050, +0.010]",
            "high $< 0$",
            "no",
            "A2 fail",
        ),
        (
            "AE F1: ae\\_xpower $-$ always",
            "+0.500 [+0.400, +0.600]",
            "low $> 0$",
            "yes",
            "A3 pass",
        ),
        ("Seg Dice: ae\\_seg $-$ recipe", "+0.100 [+0.050, +0.150]", "--", "--", "--"),
        (
            "Seg frame F1: ae\\_seg $-$ recipe",
            "-0.000 [-0.008, +0.007]",
            "--",
            "--",
            "--",
        ),
    ]
    assert lines[5:10] == [" & ".join(row) + " \\\\" for row in rows]
    only_ae = scores.table_differences(tree.ae_evaluation(), None).splitlines()
    assert len(only_ae) == len(lines) - 2
