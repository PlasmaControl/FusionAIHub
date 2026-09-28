"""The paper's score figures and tables: AE filled, the other phenomena coming."""

from __future__ import annotations

import math
import re

import pytest
from matplotlib.text import Text

from labeler.paper import COMING, coverage, scores

from . import paper_tree as tree


@pytest.fixture(autouse=True)
def paths(tmp_path, monkeypatch):
    """A temporary Paths, set before every call, though these read none."""
    return tree.temporary_paths(tmp_path, monkeypatch)


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
    marks = [c for c in first.collections if scores.FLOORS_LABEL in c.get_label()]
    levels = sorted(float(c.get_segments()[0][0][1]) for c in marks)
    assert levels == [0.75, 0.75, 0.9], "A1: precision, recall, F1"


def _clear(fig, text) -> None:
    """`text` lies inside the figure and over no other text drawn."""
    fig.draw_without_rendering()
    box = text.get_window_extent()
    assert fig.bbox.x0 <= box.x0 and box.x1 <= fig.bbox.x1, text.get_text()
    others = [
        t
        for t in fig.findobj(Text)
        if t is not text and t.get_visible() and t.get_text().strip()
    ]
    assert [t.get_text() for t in others if box.overlaps(t.get_window_extent())] == []


def test_the_scores_figure_states_a1s_verdict_from_the_record(tmp_path):
    passed = tree.ae_evaluation()
    passed["methods"]["ae_xpower"]["f1"] = tree.est(0.93, 0.91, 0.95)
    failed = tree.ae_evaluation()
    failed["bar"]["A1"] = False
    failed["differences"]["f1_minus_seldnet"] = tree.est(-0.017, -0.032, 0.0005)
    for ae, said in (
        (passed, "A1 pass:\nF1 − SELDnet lower bound −0.020 ≥ −0.03"),
        (failed, "A1 fail:\nF1 − SELDnet lower bound −0.032 < −0.03\nF1 0.890 < 0.9"),
    ):
        fig = scores.draw_scores(ae, tmp_path / "fig_scores")
        first = fig.axes[0]
        assert first.get_xlabel() == said
        assert _legend(fig)[-1] == "A1 floors"
        assert tree.small_text(fig) == []
        _clear(fig, first.xaxis.label)


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
    assert "MHD ≤ 0.05" in ax.get_xlabel()
    assert tree.small_text(fig) == []


def test_the_mhd_figure_states_a2s_verdict_from_the_record(tmp_path):
    passed = tree.ae_evaluation()
    passed["bar"]["A2"] = True
    passed["differences"]["mhd_fp_minus_seldnet"] = tree.est(-0.069, -0.13, -0.022)
    for ae, said in (
        (
            passed,
            (
                "A2 pass: MHD FP 0.020 ≤ 0.05\n"
                "MHD FP − SELDnet upper bound −0.022 < 0"
            ),
        ),
        (
            tree.ae_evaluation(),
            (
                "A2 fail: MHD FP 0.020 ≤ 0.05\n"
                "MHD FP − SELDnet upper bound 0.010 ≥ 0"
            ),
        ),
    ):
        fig = scores.draw_mhd(ae, tmp_path / "fig_mhd")
        [ax] = fig.axes
        assert ax.get_xlabel().endswith("\n" + said)
        assert tree.small_text(fig) == []
        _clear(fig, ax.xaxis.label)


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


def _levels(ax) -> list[float]:
    marks = [c for c in ax.collections if "bar" in c.get_label()]
    return sorted(round(float(c.get_segments()[0][0][1]), 2) for c in marks)


def test_the_segmentation_figure(tmp_path):
    seg = tree.seg_evaluation()
    fig = scores.draw_segmentation(seg, tmp_path / "fig_segmentation")
    ax, fp = fig.axes
    for axes, metrics in ((ax, scores.SEG_SCORES), (fp, (scores.SEG_FP,))):
        bars = [p for p in axes.patches if p.get_height() > 0]
        expected = [
            seg["methods"][m][metric]["value"]
            for m in scores.SEG_NAMES
            for metric in metrics
        ]
        assert [bar.get_height() for bar in bars] == pytest.approx(expected)
        n = len(metrics)
        hatched = [bool(bar.get_hatch()) for bar in bars]
        assert hatched == [False] * n + [True] * (2 * n), "recipe, TokEye: sources"
    assert fig.get_suptitle() == "40 test shots: 123,456 AE pixels of 7,654,321"
    assert _levels(ax) == [0.65, 0.75, 0.9], "G1 (both), G2"
    assert ax.get_ylim() == (0, 1)
    assert _levels(fp) == [0.05], "G3, on its own axis"
    assert fp.get_ylim() == (0, scores.SEG_FP_TOP)
    assert scores.LOWER_BETTER in _texts(fp) + [fp.get_title()]
    assert "pseudo-mask source" in _legend(fig)
    assert tree.small_text(fig) == []


def test_the_segmentation_fp_axis_reaches_a_wider_interval(tmp_path):
    seg = tree.seg_evaluation()
    seg["methods"]["tokeye"]["fp_rate_mhd"] = tree.est(0.2, 0.15, 0.27)
    fig = scores.draw_segmentation(seg, tmp_path / "fig_segmentation")
    assert fig.axes[1].get_ylim()[1] == pytest.approx(0.3)


def test_the_segmentation_fp_axis_ignores_undefined_ends():
    seg = tree.seg_evaluation()
    methods = list(scores.SEG_NAMES)
    seg["methods"][methods[0]]["fp_rate_mhd"] = tree.est(None)
    seg["methods"][methods[-1]]["fp_rate_mhd"] = tree.est(0.2, 0.15, 0.27)
    assert scores.fp_top(seg, methods) == pytest.approx(0.3), "a first NaN hides none"
    seg["methods"][methods[1]]["fp_rate_mhd"] = tree.est(0.02, 0.01, None)
    assert scores.fp_top(seg, methods) == pytest.approx(0.3)
    for m in methods:
        seg["methods"][m]["fp_rate_mhd"] = tree.est(None)
    assert scores.fp_top(seg, methods) == scores.SEG_FP_TOP, "nothing to reach"


def _score(value: str, up: str, down: str) -> str:
    """A score cell as the tables write it."""
    return rf"$\text{{{value}}}^{{+\text{{{up}}}}}_{{-\text{{{down}}}}}$"


def test_the_tables():
    ae = tree.ae_evaluation()
    ae["methods"] = {m: ae["methods"][m] for m in ("ae_xpower", "always")}
    ae["methods"]["always"]["fp_rate_other"] = tree.est(None)
    ae["methods"]["always"]["fp_rate_mhd"] = tree.est(0.0488, 0.0401, 0.0508)
    assert scores.table_ae(ae) == (
        "% AE frame scores: 40 test shots, 9000 frames (2500 present, 600 MHD); "
        f"{scores.INTERVAL_NOTE}; band80-mhd3 at 0.42; "
        "bar A1 pass, A2 fail, A3 pass, all fail; needs amsmath for its \\text{}\n"
        "\\begin{tabular}{lccccc}\n"
        "\\toprule\n"
        "Method & Precision & Recall & F1 & FP (MHD) & FP (other) \\\\\n"
        "\\midrule\n"
        f"ae\\_xpower & {_score('0.900', '0.030', '0.050')} & "
        f"{_score('0.880', '0.030', '0.050')} & {_score('0.890', '0.030', '0.050')} & "
        f"{_score('0.020', '0.010', '0.010')} & 0.010 \\\\\n"
        f"always & {_score('0.400', '0.030', '0.050')} & "
        f"{_score('0.380', '0.030', '0.050')} & {_score('0.390', '0.030', '0.050')} & "
        f"{_score('0.049', '0.002', '0.009')} & -- \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )
    assert "high - value" in scores.INTERVAL_NOTE, "the format, said once"
    assert _score("0.900", "0.030", "0.050") == (
        "$\\text{0.900}^{+\\text{0.030}}_{-\\text{0.050}}$"
    ), "the digits in the text font, the signs math symbols"
    table = scores.table_segmentation(tree.seg_evaluation())
    assert table.splitlines()[3] == (
        "Method & Dice & Frame P & Frame R & Frame F1 & FP (MHD) \\\\"
    )
    first = f"ae\\_seg & {_score('0.800', '0.040', '0.040')} & "
    assert table.splitlines()[5].startswith(first)
    assert len(table.splitlines()) == 10


def test_numbers_are_signed_with_a_minus_and_zero_is_unsigned():
    assert scores.number(-0.188) == "$-$0.188"
    assert scores.number(-0.0003) == "0.000"
    assert scores.number(-0.0003, signed=True) == "0.000"
    assert scores.number(0.0101, signed=True) == "+0.010"
    assert scores.number(0.5) == "0.500"
    assert scores.text_number(-0.0004, ".3f") == "\\text{0.000}", "as number does"
    assert scores.text_number(-0.0, "g") == "\\text{0}"
    assert scores.text_number(-0.03, "g") == "-\\text{0.03}"
    assert scores.text_number(0.0004, ".3f") == "\\text{0.000}"
    for x in (-0.188, -0.0015, -0.0004, -0.0, 0.0, 0.0004, 0.5):
        minus = scores.number(x).startswith("$-$")
        assert scores.text_number(x, ".3f").startswith("-") == minus, x


def test_the_differences_table_gives_the_verdicts_they_decide():
    lines = scores.table_differences(
        tree.ae_evaluation(), tree.seg_evaluation()
    ).splitlines()
    assert lines[1] == "\\begin{tabular}{lcccc}"
    assert lines[3] == "Comparison & Difference & Condition & Holds & Bar \\\\"
    rows = [
        (
            "AE F1: ae\\_xpower $-$ SELDnet",
            "+0.100 [$-$0.020, +0.200]",
            "low $\\geq -\\text{0.03}$",
            "yes",
            "A1 pass",
        ),
        (
            "AE MHD FP: ae\\_xpower $-$ SELDnet",
            "$-$0.020 [$-$0.050, +0.010]",
            "high $< \\text{0}$",
            "no",
            "A2 fail",
        ),
        (
            "AE F1: ae\\_xpower $-$ always",
            "+0.500 [+0.400, +0.600]",
            "low $> \\text{0}$",
            "yes",
            "A3 pass",
        ),
        ("Seg Dice: ae\\_seg $-$ recipe", "+0.100 [+0.050, +0.150]", "--", "--", "--"),
        (
            "Seg frame F1: ae\\_seg $-$ recipe",
            "0.000 [$-$0.008, +0.007]",
            "--",
            "--",
            "--",
        ),
    ]
    assert lines[5:10] == [" & ".join(row) + " \\\\" for row in rows]
    only_ae = scores.table_differences(tree.ae_evaluation(), None).splitlines()
    assert len(only_ae) == len(lines) - 2
    ae = tree.ae_evaluation()
    ae["bar_thresholds"]["f1_vs_seldnet_low"] = -0.05
    moved = scores.table_differences(ae, None).splitlines()[5].split(" & ")
    assert moved[2] == "low $\\geq -\\text{0.05}$", "A1's condition is the record's"


def _math_digits(table: str) -> list[str]:
    """The math in a table's rows that has a digit outside `\\text{}`, so in
    the math font (Computer Modern under Times)."""
    rows = "\n".join(r for r in table.splitlines() if not r.startswith("%"))
    maths = re.findall(r"\$([^$]*)\$", rows)
    return [m for m in maths if re.search(r"\d", re.sub(r"\\text\{[^{}]*\}", "", m))]


def test_the_tables_print_their_digits_in_the_text_font():
    ae, seg = tree.ae_evaluation(), tree.seg_evaluation()
    ae["bar_thresholds"]["f1_vs_seldnet_low"] = 0.02
    for table in (
        scores.table_ae(ae),
        scores.table_segmentation(seg),
        scores.table_differences(ae, seg),
        scores.table_differences(tree.ae_evaluation(), None),
    ):
        assert "$" in table
        assert _math_digits(table) == []


def test_each_table_that_uses_text_says_once_it_needs_amsmath():
    ae, seg = tree.ae_evaluation(), tree.seg_evaluation()
    for table in (
        scores.table_ae(ae),
        scores.table_segmentation(seg),
        scores.table_differences(ae, seg),
        scores.table_differences(ae, None),
    ):
        comment, *body = table.splitlines()
        assert "\\text{" in "\n".join(body)
        assert comment.startswith("% ")
        assert comment.count("amsmath") == table.count("amsmath") == 1, comment
        assert comment.endswith(f"; {scores.AMSMATH}")
    seg_only = scores.table_differences(None, seg)
    assert "\\text{" not in seg_only and "amsmath" not in seg_only, "no \\text used"
    counts = {"alfven_eigenmode": coverage.Counts(3, 2, 0.9)}
    assert "amsmath" not in coverage.table_datasets(counts)
