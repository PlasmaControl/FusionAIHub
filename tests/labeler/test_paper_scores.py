"""fig_scores (the selected models' F1 and ROC) and the score tables."""

from __future__ import annotations

import math
import re

import pytest
from matplotlib.text import Text

from labeler.paper import ORDER, PAGE_IN, coverage, roc, scores, title

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


def _roc(auroc: float) -> dict:
    """A `roc.json`'s fields fig_scores reads."""
    return {
        "auroc": auroc,
        "auprc": round(auroc - 0.1, 2),
        "positive_share": 0.25,
        "curve": {"fpr": [0.0, 0.1, 0.4, 1.0], "tpr": [0.0, 0.7, 0.9, 1.0]},
        "pr": {"recall": [0.0, 0.7, 0.9, 1.0], "precision": [1.0, 0.8, 0.5, 0.25]},
        "threshold": {
            "value": 0.5,
            "fpr": 0.1,
            "tpr": 0.7,
            "precision": 0.8,
            "recall": 0.7,
        },
    }


def _selected() -> list[scores.Selected]:
    """The five selected models: F1 0.9 - 0.1 i, AUROC 0.95 - 0.05 i."""
    out = []
    for i, (category, method) in enumerate(roc.SELECTED.items()):
        v = round(0.9 - 0.1 * i, 2)
        f1, auroc = tree.est(v, v - 0.05, v + 0.03), round(0.95 - 0.05 * i, 2)
        out.append(scores.Selected(category, method, f1, _roc(auroc)))
    return out


def _panels(fig):
    f1, curves, pr = fig.axes
    return f1, curves, pr


def _key(fig) -> list[str]:
    [legend] = fig.legends
    return [t.get_text() for t in legend.get_texts()]


def _curves(ax) -> list:
    """The panel's curves: lines with more than two points."""
    return [ln for ln in ax.get_lines() if len(ln.get_xdata()) > 2]


def test_fig_scores_has_one_f1_dot_per_phenomenon(tmp_path):
    selected = _selected()
    fig = scores.draw_scores(selected, tmp_path / "fig_scores")
    assert (tmp_path / "fig_scores.pdf").stat().st_size > 0
    assert (tmp_path / "fig_scores.png").stat().st_size > 0
    assert tuple(fig.get_size_inches()) == pytest.approx((PAGE_IN, 2.8))
    assert len(fig.axes) == 3, "F1, ROC, PR"
    f1, _, _ = _panels(fig)
    names = [title(c) for c in ORDER]
    assert [c.get_label() for c in f1.containers] == names
    dots = [float(c.lines[0].get_ydata()[0]) for c in f1.containers]
    assert dots == pytest.approx([0.9, 0.8, 0.7, 0.6, 0.5])
    assert _texts(f1) == ["0.90", "0.80", "0.70", "0.60", "0.50"]
    assert [t.get_text() for t in f1.get_xticklabels()] == names
    assert f1.get_ylim() == (0, 1)
    assert tree.small_text(fig) == []


def test_fig_scores_draws_only_the_selected_models(tmp_path):
    fig = scores.draw_scores(_selected(), tmp_path / "fig_scores")
    f1, curves, pr = _panels(fig)
    drawn = [t.get_text() for t in fig.findobj(Text)]
    for name in [*scores.AE_NAMES.values(), "baseline"]:
        assert not [t for t in drawn if name.lower() in t.lower()], name
    assert len(f1.containers) == len(ORDER)
    lines = [ln for ln in curves.get_lines() if len(ln.get_xdata()) > 1]
    assert len(lines) == len(ORDER) + 1, "a curve each and the chance diagonal"
    colours = [c.lines[0].get_color() for c in f1.containers]
    assert [ln.get_color() for ln in lines[: len(ORDER)]] == colours
    assert len(set(colours)) == len(ORDER)
    assert [ln.get_color() for ln in _curves(pr)] == colours


def test_the_key_gives_each_auroc_and_auprc(tmp_path):
    fig = scores.draw_scores(_selected(), tmp_path / "fig_scores")
    _, curves, _ = _panels(fig)
    assert _key(fig) == [
        "AE: AUROC 0.95, AUPRC 0.85",
        "NTM: AUROC 0.90, AUPRC 0.80",
        "H-mode: AUROC 0.85, AUPRC 0.75",
        "ELMing: AUROC 0.80, AUPRC 0.70",
        "sawteeth: AUROC 0.75, AUPRC 0.65",
        scores.CHANCE,
        scores.CHANCE_PR,
        scores.AT_THRESHOLD,
    ]
    [chance] = [ln for ln in curves.get_lines() if ln.get_linestyle() == "--"]
    assert list(chance.get_xydata().ravel()) == [0, 0, 1, 1]
    marks = [ln for ln in curves.get_lines() if ln.get_marker() == "o"]
    assert [tuple(m.get_xydata()[0]) for m in marks] == [(0.1, 0.7)] * len(ORDER)
    assert curves.get_xlim() == curves.get_ylim() == (0, 1)
    assert curves.get_xlabel() == "false-positive rate"
    assert curves.get_ylabel() == "true-positive rate"
    assert curves.get_aspect() == 1


def test_the_third_panel_is_the_precision_recall_curve(tmp_path):
    fig = scores.draw_scores(_selected(), tmp_path / "fig_scores")
    _, _, pr = _panels(fig)
    assert (pr.get_xlabel(), pr.get_ylabel()) == ("recall", "precision")
    assert pr.get_xlim() == pr.get_ylim() == (0, 1)
    [first, *_] = _curves(pr)
    assert list(first.get_xdata()) == [0.0, 0.7, 0.9, 1.0]
    assert list(first.get_ydata()) == [1.0, 0.8, 0.5, 0.25]
    marks = [ln for ln in pr.get_lines() if ln.get_marker() == "o"]
    assert [tuple(m.get_xydata()[0]) for m in marks] == [(0.7, 0.8)] * len(ORDER)
    ticks = [ln for ln in pr.get_lines() if ln.get_linestyle() == ":"]
    assert len(ticks) == len(ORDER), "a positive-share tick each"
    for tick in ticks:
        xs, ys = tick.get_xdata(), tick.get_ydata()
        assert xs[-1] == 1 and xs[0] > 0.5 and list(ys) == [0.25, 0.25]
    assert [t.get_color() for t in ticks] == [c.get_color() for c in _curves(pr)]
    assert tree.small_text(fig) == []


def test_a_phenomenon_without_its_roc_or_evaluation_says_so(tmp_path):
    selected = _selected()
    selected[1] = selected[1]._replace(roc=None)
    selected[2] = selected[2]._replace(f1=None)
    selected[3] = selected[3]._replace(f1=tree.est(None))
    fig = scores.draw_scores(selected, tmp_path / "fig_scores")
    f1, curves, pr = _panels(fig)
    assert [c.get_label() for c in f1.containers] == ["AE", "NTM", "sawteeth"]
    assert [t.get_text() for t in f1.get_xticklabels()] == [
        "AE",
        "NTM",
        f"H-mode\n{scores.NOT_SCORED}",
        f"ELMing\n{scores.NOT_SCORED}",
        "sawteeth",
    ]
    assert _key(fig)[1] == f"NTM: {scores.NO_ROC}"
    lines = [ln for ln in curves.get_lines() if len(ln.get_xdata()) > 1]
    assert len(lines) == len(ORDER), "four curves and the chance diagonal"
    assert len(_curves(pr)) == len(ORDER) - 1
    assert tree.small_text(fig) == []


def test_a_roc_without_a_pr_curve_draws_its_roc_and_says_no_pr(tmp_path):
    selected = _selected()
    old = {k: v for k, v in selected[2].roc.items() if k not in ("auprc", "pr")}
    old["threshold"] = {k: v for k, v in old["threshold"].items() if k[0] != "p"}
    del old["positive_share"]
    selected[2] = selected[2]._replace(roc=old)
    fig = scores.draw_scores(selected, tmp_path / "fig_scores")
    _, curves, pr = _panels(fig)
    assert _key(fig)[2] == f"H-mode: AUROC 0.85, {scores.NO_PR}"
    assert len(_curves(curves)) == len(ORDER)
    assert len(_curves(pr)) == len(ORDER) - 1
    assert len([ln for ln in pr.get_lines() if ln.get_marker() == "o"]) == 4
    assert tree.small_text(fig) == []


def test_table_ae_scores_keeps_its_six_methods():
    names = [scores.AE_NAMES[m] for m in scores.AE_METHODS]
    assert len(names) == 6
    rows = scores.table_ae(tree.ae_evaluation()).splitlines()[5 : 5 + len(names)]
    assert [row.split(" & ")[0].replace("\\_", "_") for row in rows] == names


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
