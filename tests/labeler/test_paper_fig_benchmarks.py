"""The benchmark figure (paper Figure 2): the confine-cnn bars (the Tokamak-SI score
is the ablation's protocol row, the paper's one confine-cnn score, read through a
small adapter), the tearing-mode, RWM and ELM panels (each reads its stream's final
record), the coverage row, the figure's size and text, and no typed score."""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
from pathlib import Path

import pytest
from matplotlib.figure import Figure
from matplotlib.text import Text

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/labeler/paper/fig_benchmarks.py"
spec = importlib.util.spec_from_file_location("fig_benchmarks", SCRIPT)
fig = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fig
spec.loader.exec_module(fig)

ROW = {
    "own_population": {"macro_f1": 0.7, "ci95": {"macro_f1": [0.6, 0.8]}},
    "ranking": {
        "auroc": {"macro": 0.9},
        "auprc": {"macro": 0.7},
        "ci95": {"auroc": [0.8, 0.95], "auprc": [0.6, 0.8]},
    },
}
LEGACY = {
    "paper": {c: {"f1": f} for c, f in zip(fig.CLASSES, (0.9, 1.0, 0.9, 0.8))},
    "auc_one_vs_rest": {"per_class_at_least": 0.99},
}


def test_adapter_reads_the_protocol_rows_scores_and_intervals():
    si = fig.confinement_si(ROW)
    assert si["macro"] == {"f1": 0.7, "auroc": 0.9, "auprc": 0.7}
    assert si["ci95"]["macro_f1"] == [0.6, 0.8]
    assert si["ci95"]["macro_auroc"] == [0.8, 0.95]
    assert si["ci95"]["macro_auprc"] == [0.6, 0.8]


def test_confinement_panel_draws_the_protocol_row_and_names_its_source():
    si = fig.confinement_si(ROW)
    for metric, value, key in (
        ("f1", 0.7, "own_population.macro_f1"),
        ("auroc", 0.9, "ranking.auroc.macro"),
        ("auprc", 0.7, "ranking.auprc.macro"),
    ):
        rows = fig.Rows()
        ax = Figure().add_subplot()
        fig.draw_confinement(ax, LEGACY, si, rows, "b", metric)
        drawn = [r for r in rows.rows if r["group"] == "Tokamak-SI"]
        assert len(drawn) == 1
        assert drawn[0]["value"] == value and drawn[0]["key"] == key
        assert drawn[0]["source"].endswith("ablation_rows/full_cum_abcdrgef.json")
    legacy = [r for r in rows.rows if r["group"] == "legacy"]
    assert not legacy  # the earlier paper reported no AUPRC


def test_the_figure_reads_the_table_3_row_not_the_first_retrain():
    assert fig.SOURCES["confinement_si"].name == "full_cum_abcdrgef.json"
    assert fig.SOURCES["confinement_si"].parent.name == "ablation_rows"


def test_legacy_auroc_is_the_published_bound_per_class_not_a_macro_value():
    si = fig.confinement_si(ROW)
    rows = fig.Rows()
    fig.draw_confinement(Figure().add_subplot(), LEGACY, si, rows, "b", "auroc")
    legacy = [r for r in rows.rows if r["group"] == "legacy"]
    assert len(legacy) == 1
    assert legacy[0]["value"] == 0.99
    assert "lower bound" in legacy[0]["metric"]
    assert "macro" not in legacy[0]["metric"]
    assert legacy[0]["key"] == "auc_one_vs_rest.per_class_at_least"


def test_the_published_values_are_committed_and_drawn_from_the_branch():
    published = fig.load(fig.SOURCES["confinement"])
    assert fig.SOURCES["confinement"].name == "gill_2024_published.json"
    assert {c: published["paper"][c]["f1"] for c in fig.CLASSES} == {
        "L": 0.94,
        "H": 0.97,
        "QH": 0.94,
        "WP": 0.90,
    }
    assert published["auc_one_vs_rest"]["per_class_at_least"] == 0.99
    rows = fig.Rows()
    fig.draw_confinement(
        Figure().add_subplot(), published, fig.confinement_si(ROW), rows, "b", "f1"
    )
    legacy = [r for r in rows.rows if r["group"] == "legacy"]
    assert abs(legacy[0]["value"] - 0.9375) < 1e-12


def records():
    missing = [p.name for p in fig.SOURCES.values() if not p.is_file()]
    if missing:
        pytest.skip(f"records not on this checkout: {missing}")
    return fig.read_records()


def drawn_rows(draw, *args, metric="f1", panel="x"):
    """The rows a panel function drew and the axes it drew on."""
    rows = fig.Rows()
    ax = Figure().add_subplot()
    draw(ax, *args, rows, panel, metric)
    return [r for r in rows.rows if r["drawn"]], rows.rows, ax


def bar_heights(ax):
    return {round(p.get_height(), 12) for p in ax.patches}


# ELM


def test_elm_legacy_f1_is_the_final_records_and_equals_its_counts_and_pr():
    smith = fig.load(fig.SOURCES["elm_smith"])
    value, ci = fig.smith_legacy(smith)
    block = smith["reimplemented_elmo_overlap"]
    c, p, r = block["counts"], block["point"]["precision"], block["point"]["recall"]
    assert value == pytest.approx(2 * c["tp"] / (2 * c["tp"] + c["fp"] + c["fn"]))
    assert value == pytest.approx(2 * p * r / (p + r))
    assert round(r, 3) == 0.986 and round(value, 3) == 0.992
    assert ci == tuple(block["ci95"]["f1"])
    assert value > smith["original_cached_elmo_overlap"]["point"]["f1"]


def test_a_legacy_f1_that_is_not_its_counts_or_precision_and_recall_is_refused():
    def smith(f1, precision, recall):
        return {
            "reimplemented_elmo_overlap": {
                "counts": {"tp": 90, "fp": 10, "fn": 10},
                "point": {"precision": precision, "recall": recall, "f1": f1},
                "ci95": {"f1": [0.8, 0.95]},
            }
        }

    assert fig.smith_legacy(smith(0.9, 0.9, 0.9))[0] == 0.9
    with pytest.raises(ValueError, match="counts"):
        fig.smith_legacy(smith(0.95, 0.9, 0.9))
    with pytest.raises(ValueError, match="precision and recall"):
        fig.smith_legacy(smith(0.9, 0.95, 0.85))


def test_the_elm_panel_reads_the_final_records_on_the_shots_with_bes():
    _, _, _, elm, smith, _, _ = records()
    drawn, every, ax = drawn_rows(fig.draw_elm, elm, smith)
    by = {(r["series"], r["group"]): r for r in drawn}
    assert set(by) == {
        ("elm-elmo", "legacy"),
        ("elm-elmo", "Tokamak-SI"),
        ("elm-ours", "Tokamak-SI"),
    }
    legacy = by["elm-elmo", "legacy"]
    assert legacy["key"] == "reimplemented_elmo_overlap.point.f1"
    assert legacy["source"] == "outputs/labeler/elm/smith/evaluation.json"
    bes = elm["sets"]["bes73"]["methods"]
    for model in ("elm-elmo", "elm-ours"):
        row = by[model, "Tokamak-SI"]
        assert row["value"] == bes[model]["point"]["f1"]
        assert row["key"] == f"sets.bes73.methods.{model}.point.f1"
        assert row["source"] == "outputs/labeler/elm/ours/evaluation.json"
    assert not [r for r in every if "elm/elmo/" in r["source"]]  # the older record
    kept_out = [r for r in every if not r["drawn"]]
    assert any("all119" in r["key"] for r in kept_out)
    assert {round(r["value"], 12) for r in drawn} <= bar_heights(ax)


def test_the_elm_auc_panel_has_no_legacy_bar_and_the_sweeps_values():
    _, _, _, elm, smith, _, _ = records()
    for metric in ("auroc", "auprc"):
        drawn, _, _ = drawn_rows(fig.draw_elm, elm, smith, metric=metric)
        assert {r["group"] for r in drawn} == {"Tokamak-SI"}
        for r in drawn:
            assert r["key"] == f"sets.bes73.methods.{r['series']}.point.{metric}"


# tearing modes


def test_the_tm_panel_draws_published_against_retrained_with_uncertain_diamonds():
    tm = records()[5]
    by = {r["architecture"]: r for r in tm["rows"]}
    drawn, every, ax = drawn_rows(fig.draw_tm, tm)
    series = {(r["series"], r["group"]): r for r in drawn}
    models = {
        "tm-onsetcnn": "onsetcnn",
        "tm-dsm": "dsm",
        "tm-ours": "magnetic-detector",
    }
    for name, arch in models.items():
        si = series[name, "Tokamak-SI"]
        assert si["value"] == by[arch]["tokamak_si"]["f1"]["value"]
        assert si["key"] == f"rows[{arch}].tokamak_si.f1"
        diamond = series[name, "Tokamak-SI, uncertain time scored negative"]
        unc = by[arch]["tokamak_si_uncertain_negative"]["f1"]["value"]
        assert diamond["value"] == unc and diamond["value"] < si["value"]
        legacy = series.get((name, "legacy"))
        if name == "tm-ours":
            assert legacy is None  # there is no published tm-ours
        else:
            published = by[arch]["legacy"]["f1_published_threshold"]["value"]
            assert legacy["value"] == published
            assert legacy["key"] == f"rows[{arch}].legacy.f1_published_threshold"
    assert not [r for r in drawn if "n1" in r["series"] or "two" in r["series"]]
    tuned = [r for r in every if r["group"] == "legacy, tuned threshold"]
    assert tuned and not any(r["drawn"] for r in tuned)  # in the CSV only
    markers = [ln for ln in ax.lines if ln.get_marker() == "D"]
    assert len(markers) == 3  # one open diamond on every Tokamak-SI bar


def test_the_tm_title_states_the_two_bin_widths_the_record_gives():
    tm = records()[5]
    title = fig.tm_title(tm)
    legacy = {r["legacy"]["bin_ms"] for r in tm["rows"] if r.get("legacy")}
    si = {r["tokamak_si"]["bin_ms"] for r in tm["rows"]}
    assert f"{legacy.pop():g} ms legacy" in title and f"{si.pop():g} ms" in title
    odd = {"rows": [*tm["rows"], dict(tm["rows"][0], tokamak_si={"bin_ms": 99.0})]}
    with pytest.raises(ValueError, match="bin width"):
        fig.tm_title(odd)


def test_the_tm_auc_panel_reads_the_records_auroc_and_auprc():
    tm = records()[5]
    by = {r["architecture"]: r for r in tm["rows"]}
    for metric in ("auroc", "auprc"):
        drawn, _, _ = drawn_rows(fig.draw_tm, tm, metric=metric)
        row = next(
            r for r in drawn if r["series"] == "tm-dsm" and r["group"] == "legacy"
        )
        assert row["value"] == by["dsm"]["legacy"][metric]["value"]


# resistive wall modes


def test_the_rwm_panel_has_no_legacy_value_and_a_tick_at_the_elapsed_time_baseline():
    rwm = records()[6]
    drawn, every, ax = drawn_rows(fig.draw_rwm, rwm)
    assert not [r for r in drawn if r["group"] == "legacy"]  # another machine
    legacy = [r for r in every if r["group"] == "legacy"]
    assert len(legacy) == 1 and legacy[0]["value"] is None
    assert "not comparable" in legacy[0]["note"]
    by = {r["series"]: r for r in drawn}
    brf = rwm["configs"]["rwm-brf"]["metrics"]["slice_f1"]
    clock = rwm["configs"]["rwm-rule-elapsed-time"]["metrics"]["slice_f1"]
    assert by["rwm-brf"]["value"] == brf["estimate"]
    assert by["rwm-brf"]["key"] == "configs.rwm-brf.metrics.slice_f1.estimate"
    assert by["rwm-rule-elapsed-time"]["value"] == clock["estimate"]
    assert any(
        clock["estimate"] in ln.get_ydata() for ln in ax.lines
    )  # the tick is drawn at it


def test_the_rwm_auc_panel_keeps_the_other_machines_auroc_out_of_the_figure():
    rwm = records()[6]
    _, every, _ = drawn_rows(fig.draw_rwm, rwm, metric="auroc")
    legacy = [r for r in every if r["group"] == "legacy"]
    assert legacy[0]["value"] == rwm["legacy"]["slice_auroc"] and not legacy[0]["drawn"]


# coverage


def test_the_coverage_panels_draw_the_record_and_hatch_what_has_no_value():
    cov = fig.load(fig.SOURCES["coverage"])
    for field, unit in (("shots", "labelled shots"), ("seconds", "labelled time (h)")):
        drawn, every, ax = drawn_rows(fig.draw_coverage, cov, metric=field)
        assert ax.get_yscale() == "log"
        scale = 1.0 if field == "shots" else 1 / 3600.0
        for r in drawn:
            if r["group"] == "Tokamak-SI, reviewed subset":
                continue
            side = "legacy" if r["group"] == "legacy" else "tokamak_si"
            key = next(k for k in cov["order"] if cov["sets"][k]["name"] == r["series"])
            block = cov["sets"][key][side]
            if block[field] is None:
                assert r["value"] is None  # a hatched slot, with its reason
            else:
                assert r["value"] == pytest.approx(block[field] * scale)
                assert r["key"] == f"sets.{key}.{side}.{field}"
        names = {r["series"] for r in every}
        assert names == {cov["sets"][k]["name"] for k in cov["order"]}
        assert {r["metric"] for r in every} == {unit}
        reviewed = [r for r in every if r["group"] == "Tokamak-SI, reviewed subset"]
        assert reviewed and all(
            r["value"]
            <= next(
                s["value"]
                for s in every
                if s["series"] == r["series"] and s["group"] == "Tokamak-SI"
            )
            for r in reviewed
        )


# the whole figure


def test_the_f1_figure_fits_the_main_text_and_every_text_is_at_least_7_pt():
    recs = records()
    cov = fig.load(fig.SOURCES["coverage"])
    rows = fig.Rows()
    with fig.plot_style():
        figure = fig.figure_f1(recs, cov, rows)
        twin = fig.figure_auc(recs, fig.Rows())
        for f in (figure, twin):
            assert f.get_figwidth() == pytest.approx(6.75)
            texts = [t for t in f.findobj(Text) if t.get_visible() and t.get_text()]
            assert texts
            small = {
                t.get_text(): t.get_fontsize() for t in texts if t.get_fontsize() < 7
            }
            assert not small, small
    assert figure.get_figheight() <= 5.2
    assert {r["panel"] for r in rows.rows} == set("abcdfgh")  # e: not yet scored


def test_no_version_label_and_no_misspelling_in_the_text_or_the_table():
    recs = records()
    cov = fig.load(fig.SOURCES["coverage"])
    rows = fig.Rows()
    with fig.plot_style():
        figure = fig.figure_f1(recs, cov, rows)
        figure_text = [t.get_text() for t in figure.findobj(Text)]
    cells = [str(v) for r in rows.rows for v in (r["series"], r["group"], r["note"])]
    for text in figure_text + cells:
        assert not re.search(r"\bv[0-3]\b", text), text
        assert "sawteeth" not in text.lower() and "Jalalvand" not in text


def test_sawtooth_is_drawn_as_not_yet_scored_in_both_settings():
    fig_ax = Figure().add_subplot()
    fig.draw_pending(fig_ax)
    labels = [t.get_text() for t in fig_ax.texts]
    assert labels.count("not yet scored") == 2


# the guard: a drawn score is read from a record, never typed


def test_no_score_is_typed_into_the_script():
    recs = records()
    cov = fig.load(fig.SOURCES["coverage"])
    with fig.plot_style():
        rows = fig.Rows()
        fig.figure_f1(recs, cov, rows)
        fig.figure_auc(recs, rows)
    scores = {
        round(r["value"], 3)
        for r in rows.rows
        if r["drawn"] and r["panel"] not in "gh" and 0 < (r["value"] or 0) < 1
    }
    tree = ast.parse(SCRIPT.read_text())
    typed = {
        round(n.value, 3)
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant)
        and isinstance(n.value, float)
        and 0 < n.value < 1
    }
    assert not scores & typed, scores & typed
