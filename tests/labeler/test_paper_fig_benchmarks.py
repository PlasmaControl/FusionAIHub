"""The benchmark figure (paper Figure 2): the confine-cnn bars (the Tokamak-SI score
is the ablation's protocol row, the paper's one confine-cnn score, read through a
small adapter), the tearing-mode, sawtooth, RWM and ELM panels (each reads its
stream's final record), the detachment consistency panel (one bar, read from two
records), one tick per model with paired bars, the coverage panel, plain panel
titles, the figure's size and text (none under 7 pt, none overlapping), and no
typed score."""

from __future__ import annotations

import ast
import copy
import importlib.util
import itertools
import re
import sys
from pathlib import Path

import pytest
from matplotlib import font_manager as fm
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_hex
from matplotlib.container import ErrorbarContainer
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.transforms import Bbox

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/labeler/paper/fig_benchmarks.py"
spec = importlib.util.spec_from_file_location("fig_benchmarks", SCRIPT)
fig = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fig
spec.loader.exec_module(fig)

ROW = {
    "own_population": {
        "macro_f1": 0.7,
        "ci95": {"macro_f1": [0.6, 0.8]},
        "classes": {
            c: {"support": n} for c, n in zip(("L", "H", "QH", "WP"), (40, 30, 20, 10))
        },
    },
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


def test_the_confinement_baselines_come_from_the_classes_supports():
    """A constant predictor on the windows of the row: the majority class (0.4 of
    them) scores its own F1 and the other three classes 0; the AUROC is the
    chance level; the AUPRC the classes' mean prevalence."""
    base = fig.confinement_si(ROW)["base"]
    assert base["f1"] == pytest.approx((2 * 0.4 / 1.4) / 4)
    assert base["auroc"] == 0.5
    assert base["auprc"] == pytest.approx(0.25)


def test_the_always_positive_f1_and_the_prevalence_baselines():
    assert fig.always_f1(0.5) == pytest.approx(2 / 3)
    assert fig.always_f1(1.0) == 1.0
    assert fig.prevalence_base("f1", 0.2) == pytest.approx(0.4 / 1.2)
    assert fig.prevalence_base("auroc", 0.2) == 0.5
    assert fig.prevalence_base("auprc", 0.2) == 0.2


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


# confine-ours on the protocol row's windows

ROW_FULL = copy.deepcopy(ROW)
ROW_FULL["own_population"] |= {"windows": 100, "shots": 7}
OURS = {
    "confine-cnn": {"macro_f1": 0.7, "windows": 100, "shots": 7},
    "confine-ours": {
        "macro_f1": 0.85,
        "windows": 100,
        "shots": 7,
        "ci95": {"macro_f1": [0.8, 0.9]},
        "classes": ROW["own_population"]["classes"],
        "ranking": {
            "auroc": {"macro": 0.96},
            "auprc": {"macro": 0.88},
            "ci95": {"auroc": [0.94, 0.97], "auprc": [0.84, 0.91]},
        },
    },
    "ours_minus_cnn": {
        "macro_f1": {"difference": 0.15, "ci95": [0.1, 0.2]},
        "auroc": {"difference": 0.06, "ci95": [0.03, 0.09]},
        "auprc": {"difference": 0.18, "ci95": [0.12, 0.24]},
    },
}


def test_the_ours_adapter_reads_its_scores_baselines_and_paired_difference():
    ours = fig.confinement_ours(OURS, ROW_FULL)
    assert ours["macro"] == {"f1": 0.85, "auroc": 0.96, "auprc": 0.88}
    assert ours["ci95"]["macro_auprc"] == [0.84, 0.91]
    assert ours["keys"]["f1"] == "confine-ours.macro_f1"
    # the same windows, so the same supports and baselines as confine-cnn's
    assert ours["base"] == fig.confinement_si(ROW)["base"]
    keys = ours["base_keys"]
    assert keys["f1"].startswith("confine-ours.classes")
    assert keys["auprc"].startswith("confine-ours.classes")
    assert keys["auroc"] == "chance level by definition"
    assert ours["difference"]["f1"] == (0.15, (0.1, 0.2))
    assert ours["difference"]["auprc"] == (0.18, (0.12, 0.24))


def test_an_ours_record_scored_on_other_windows_than_the_row_is_refused():
    for change in ({"windows": 99}, {"shots": 6}, {"macro_f1": 0.71}):
        record = copy.deepcopy(OURS)
        record["confine-cnn"] |= change
        with pytest.raises(ValueError, match="protocol row"):
            fig.confinement_ours(record, ROW_FULL)


def test_the_confinement_panel_draws_confine_ours_on_its_own_tick():
    si = fig.confinement_si(ROW) | {"ours": fig.confinement_ours(OURS, ROW_FULL)}
    for metric, value, diff in (
        ("f1", 0.85, 0.15),
        ("auroc", 0.96, 0.06),
        ("auprc", 0.88, 0.18),
    ):
        bars, every, ax = drawn_rows(fig.draw_confinement, LEGACY, si, metric=metric)
        assert [t.get_text() for t in ax.get_xticklabels()] == [
            fig.two_line("confine-cnn"),
            fig.two_line("confine-ours"),
        ]
        (mine,) = [r for r in bars if r["series"] == "confine-ours"]
        assert mine["value"] == value and mine["group"] == "Tokamak-SI"
        assert mine["source"].endswith(fig.SOURCES["confinement_ours"].name)
        (bar,) = [p for p in ax.patches if p.get_x() < 1 < p.get_x() + p.get_width()]
        assert bar.get_height() == pytest.approx(value)
        assert to_hex(bar.get_facecolor()).upper() == fig.SI
        # its dash across just its bar, and a row of the table for it
        span = (bar.get_x(), bar.get_x() + bar.get_width())
        assert [g[2] for g in baseline_segments(ax) if g[:2] == pytest.approx(span)]
        assert [r for r in baseline_rows(every) if r["group"].startswith("confine-o")]
        (d,) = [r for r in every if r["series"].startswith("confine-ours minus")]
        assert d["value"] == diff and not d["drawn"]
    assert fig.SOURCES["confinement_ours"].parent.name == "ours"


def records():
    missing = [p.name for p in fig.SOURCES.values() if not p.is_file()]
    if missing:
        pytest.skip(f"records not on this checkout: {missing}")
    return fig.read_records()


def drawn_rows(draw, *args, metric="f1", panel="x"):
    """The rows a panel function drew as bars (its trivial baselines, which are
    drawn as dashes, are rows of their own and not in the first list), every row
    it wrote, and the axes it drew on."""
    rows = fig.Rows()
    ax = Figure().add_subplot()
    draw(ax, *args, rows, panel, metric)
    bars = [r for r in rows.rows if r["drawn"] and r["series"] != fig.BASELINE]
    return bars, rows.rows, ax


def baseline_rows(every):
    return [r for r in every if r["series"] == fig.BASELINE]


def bar_heights(ax):
    return {round(p.get_height(), 12) for p in ax.patches}


def baseline_segments(ax):
    """The trivial-baseline dashes of a panel: (left, right, height) of each."""
    return sorted(
        (min(ln.get_xdata()), max(ln.get_xdata()), ln.get_ydata()[0])
        for ln in ax.lines
        if ln.get_linestyle() == "--" and ln.get_color() == "black"
    )


def baseline_values(ax):
    return [y for _, _, y in baseline_segments(ax)]


def reported_texts(ax):
    """The REPORTED marks written along a panel's bars."""
    return [t for t in ax.texts if t.get_text() == fig.REPORTED]


def flat(text: str) -> str:
    return " ".join(text.split())


# one tick per model


def test_each_model_is_one_tick_with_its_original_bar_left_of_its_tokamak_si_bar():
    ae = records()[0]
    _, _, ax = drawn_rows(fig.draw_ae, ae)
    assert [t.get_text() for t in ax.get_xticklabels()] == [
        fig.two_line(name) for _, name in fig.AE_MODELS
    ]
    bars = sorted(ax.patches, key=lambda p: p.get_x())
    colours = [to_hex(p.get_facecolor()).upper() for p in bars]
    assert colours == [fig.LEGACY, fig.SI] * len(fig.AE_MODELS)
    for i in range(len(fig.AE_MODELS)):  # the pair sits on its model's tick
        left, right = bars[2 * i], bars[2 * i + 1]
        centre = (left.get_x() + right.get_x() + right.get_width()) / 2
        assert centre == pytest.approx(i)
    # no setting written under the names: the legend says which bar is which
    assert not {"original", "Tokamak-SI", "legacy"} & {t.get_text() for t in ax.texts}


def test_each_ae_bar_has_the_always_present_baseline_of_its_own_truth():
    ae = records()[0]
    for metric in ("f1", "auroc", "auprc"):
        drawn, every, ax = drawn_rows(fig.draw_ae, ae, metric=metric)
        rows = baseline_rows(every)
        assert len(rows) == len(drawn) == 2 * len(fig.AE_MODELS)
        for r in rows:
            truth = next(t for s, t, *_ in fig.AE_SETTINGS if r["group"].endswith(s))
            assert r["value"] == ae["results"][truth]["always"][metric]
            assert r["key"] == f"results.{truth}.always.{metric}"
            assert r["source"] == "outputs/labeler/ae/baselines/evaluation.json"
        # one dash across each bar, at its own truth's baseline
        bars = sorted(ax.patches, key=lambda p: p.get_x())
        segments = baseline_segments(ax)
        assert len(segments) == len(bars)
        for bar, (left, right, _) in zip(bars, segments):
            assert left == pytest.approx(bar.get_x())
            assert right == pytest.approx(bar.get_x() + bar.get_width())


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
    _, _, _, elm, smith, *_ = records()
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
    # each Tokamak-SI bar has the always-present predictor's score on the same
    # bins as its baseline; the legacy bar, another dataset's, has none
    always = bes["always present"]["point"]["f1"]
    assert baseline_values(ax) == [always, always]
    base = baseline_rows(every)
    assert [r["group"] for r in base] == [
        "elm-elmo, Tokamak-SI",
        "elm-ours, Tokamak-SI",
    ]
    assert all(r["value"] == always for r in base)
    assert all(r["key"] == "sets.bes73.methods.always present.point.f1" for r in base)


def test_the_elm_auc_panel_has_no_legacy_bar_and_the_sweeps_values():
    _, _, _, elm, smith, *_ = records()
    for metric in ("auroc", "auprc"):
        drawn, _, _ = drawn_rows(fig.draw_elm, elm, smith, metric=metric)
        assert {r["group"] for r in drawn} == {"Tokamak-SI"}
        for r in drawn:
            assert r["key"] == f"sets.bes73.methods.{r['series']}.point.{metric}"


# tearing modes


def test_the_tm_panel_draws_published_against_retrained_with_no_diamonds():
    tm = records()[5]
    by = {r["architecture"]: r for r in tm["rows"]}
    drawn, every, ax = drawn_rows(fig.draw_tm, tm)
    series = {(r["series"], r["group"]): r for r in drawn}
    models = {
        "tm-cnn": "onsetcnn",
        "tm-dsm": "dsm",
        "tm-ours": "magnetic-detector",
    }
    for name, arch in models.items():
        si = series[name, "Tokamak-SI"]
        assert si["value"] == by[arch]["tokamak_si"]["f1"]["value"]
        assert si["key"] == f"rows[{arch}].tokamak_si.f1"
        unc = next(  # the same models with uncertain time scored negative: CSV only
            r
            for r in every
            if r["series"] == name
            and r["group"] == "Tokamak-SI, uncertain time scored negative"
        )
        assert not unc["drawn"] and (name, unc["group"]) not in series
        assert unc["value"] == by[arch]["tokamak_si_uncertain_negative"]["f1"]["value"]
        assert unc["value"] < si["value"]
        legacy = series.get((name, "legacy"))
        if name == "tm-ours":
            assert legacy is None  # there is no published tm-ours
        elif name == "tm-dsm":
            # the checkpoint has no published operating point: its 500 ms risk
            # peaks at 0.279 on these bins (cap 0.323), so risk >= 0.3 never fires
            # and its bar is the inner-validation threshold's
            tuned = by[arch]["legacy"]["f1_tuned"]["value"]
            assert legacy["value"] == tuned and round(tuned, 3) == 0.177
            assert legacy["key"] == f"rows[{arch}].legacy.f1_tuned"
            assert "no published operating point for this checkpoint" in legacy["note"]
            assert "inner-validation threshold" in legacy["note"]
            (zero,) = [
                r
                for r in every
                if r["series"] == name and r["group"] == "legacy, published threshold"
            ]
            assert (
                zero["value"] == by[arch]["legacy"]["f1_published_threshold"]["value"]
            )
            assert zero["value"] == 0.0 and not zero["drawn"]
            assert zero["key"] == f"rows[{arch}].legacy.f1_published_threshold"
            assert "risk >= 0.3 is an upstream helper default" in zero["note"]
            assert "0.279" in zero["note"] and "0.323" in zero["note"]
        else:  # the CNN: its card's published 0.5 threshold
            published = by[arch]["legacy"]["f1_published_threshold"]["value"]
            assert legacy["value"] == published
            assert legacy["key"] == f"rows[{arch}].legacy.f1_published_threshold"
    assert not [r for r in drawn if "n1" in r["series"] or "two" in r["series"]]
    tuned = [r for r in every if r["group"] == "legacy, tuned threshold"]
    assert [r["series"] for r in tuned] == ["tm-cnn"]  # the CNN's, CSV only
    assert not any(r["drawn"] for r in tuned)
    assert not [ln for ln in ax.lines if ln.get_marker() == "D"]  # no diamonds
    assert len(ax.patches) == 5  # tm-cnn and tm-dsm in pairs, tm-ours alone


def test_each_tm_bar_has_the_baseline_of_its_own_bins_prevalence():
    tm = records()[5]
    by = {r["architecture"]: r for r in tm["rows"]}
    always_positive = {
        "f1": fig.always_f1,
        "auroc": lambda p: 0.5,
        "auprc": lambda p: p,
    }
    for metric, expected in always_positive.items():
        _, every, ax = drawn_rows(fig.draw_tm, tm, metric=metric)
        rows = baseline_rows(every)
        assert len(rows) == len(ax.patches) == len(baseline_segments(ax)) == 5
        for r in rows:
            name = r["group"].split(", ")[0]
            arch = next(a for a, label in fig.TM_MODELS if label == name)
            side = "legacy" if r["group"].endswith("legacy") else "tokamak_si"
            p = by[arch][side]["prevalence"]["value"]  # that bar's own bins
            assert r["value"] == pytest.approx(expected(p))
            assert f"rows[{arch}].{side}.prevalence" in r["key"] or metric == "auroc"
    # the retrained and the published model are scored on different bins
    prevalences = {
        by[a][side]["prevalence"]["value"]
        for a in by
        for side in ("legacy", "tokamak_si")
        if by[a].get(side)
    }
    assert len(prevalences) > 1


def test_the_tm_auc_panel_reads_the_records_auroc_and_auprc():
    tm = records()[5]
    by = {r["architecture"]: r for r in tm["rows"]}
    for metric in ("auroc", "auprc"):
        drawn, _, _ = drawn_rows(fig.draw_tm, tm, metric=metric)
        row = next(
            r for r in drawn if r["series"] == "tm-dsm" and r["group"] == "legacy"
        )
        assert row["value"] == by["dsm"]["legacy"][metric]["value"]
        assert row["key"] == f"rows[dsm].legacy.{metric}"  # threshold-free


# resistive wall modes

RWM_MODELS = ("rwm-brf", "rwm-ours")


def test_the_rwm_f1_panel_draws_both_classifiers_with_their_bins_baseline():
    rwm = records()[6]
    drawn, every, ax = drawn_rows(fig.draw_rwm, rwm)
    # the NSTX forest has no F1: a row, not a bar
    legacy = [r for r in every if r["group"] == "legacy"]
    assert len(legacy) == 1 and legacy[0]["value"] is None and not legacy[0]["drawn"]
    assert "another machine" in legacy[0]["note"] and "forecast" in legacy[0]["note"]
    assert len(ax.patches) == 2 and not reported_texts(ax)
    assert [t.get_text() for t in ax.get_xticklabels()] == [
        fig.two_line(m) for m in RWM_MODELS
    ]
    assert [r["series"] for r in drawn] == list(RWM_MODELS)
    for row in drawn:
        model = rwm["models"][row["series"]]
        assert row["group"] == "Tokamak-SI"
        assert row["value"] == model["metrics"]["f1"]["estimate"]
        assert row["key"] == f"models.{row['series']}.metrics.f1.estimate"
    assert not any(
        "horizon" in r["note"] or "phase-controlled" in r["note"] for r in every
    )
    # one prevalence, of the bins both are scored on, under each bar
    counts = rwm["counts"]
    prevalence = counts["positive_bins"] / (
        counts["positive_bins"] + counts["negative_bins"]
    )
    assert prevalence == pytest.approx(counts["prevalence"])
    bases = baseline_rows(every)
    assert len(bases) == 2 and all(
        b["key"].endswith("counts.prevalence") for b in bases
    )
    assert baseline_values(ax) == pytest.approx([fig.always_f1(prevalence)] * 2)
    # the paired difference and the uncertain-negative view: CSV only
    (paired,) = [r for r in every if r["series"] == "rwm-ours minus rwm-brf"]
    assert not paired["drawn"]
    diff = rwm["paired"]["rwm-ours_minus_rwm-brf"]["primary"]["f1"]
    assert paired["value"] == diff["estimate"]
    unc = [r for r in every if r["group"].endswith("uncertain time scored negative")]
    assert len(unc) == 2 and not any(r["drawn"] for r in unc)


def test_the_rwm_baseline_is_refused_when_it_is_not_the_bins_counts():
    rwm = copy.deepcopy(records()[6])
    assert fig.rwm_prevalence(rwm) == rwm["counts"]["prevalence"]
    other = copy.deepcopy(rwm)
    other["counts"]["prevalence"] += 0.01
    with pytest.raises(ValueError, match="prevalence"):
        fig.rwm_prevalence(other)
    rwm["models"]["rwm-ours"]["counts"]["negative_bins"] += 1
    with pytest.raises(ValueError, match="other bins"):
        fig.rwm_prevalence(rwm)


def test_the_rwm_auroc_panel_draws_the_nstx_forecast_as_reported_beside_rwm_brf():
    rwm = records()[6]
    drawn, every, ax = drawn_rows(fig.draw_rwm, rwm, metric="auroc")
    legacy = [r for r in drawn if r["group"] == "legacy"]
    assert len(legacy) == 1 and legacy[0]["value"] == rwm["legacy"]["slice_auroc"]
    assert legacy[0]["key"] == "legacy.slice_auroc" and "REPORTED" in legacy[0]["note"]
    assert "forecast" in legacy[0]["note"] and "another machine" in legacy[0]["note"]
    assert len(reported_texts(ax)) == 1 and len(ax.patches) == 3
    bars = sorted(ax.patches, key=lambda p: p.get_x())
    colours = [to_hex(p.get_facecolor()).upper() for p in bars]
    assert colours == [fig.LEGACY, fig.SI, fig.SI]
    # the pair sits on rwm-brf's tick, rwm-ours alone on its own
    centre = (bars[0].get_x() + bars[1].get_x() + bars[1].get_width()) / 2
    assert centre == pytest.approx(0)
    assert bars[2].get_x() + bars[2].get_width() / 2 == pytest.approx(1)
    # chance-level dashes on the Tokamak-SI bars alone
    assert [b["value"] for b in baseline_rows(every)] == [0.5, 0.5]
    segments = baseline_segments(ax)
    assert [y for _, _, y in segments] == [0.5, 0.5]
    for (left, right, _), bar in zip(segments, bars[1:], strict=True):
        assert left == pytest.approx(bar.get_x())
        assert right == pytest.approx(bar.get_x() + bar.get_width())


def test_the_rwm_auprc_panel_has_no_legacy_bar():
    rwm = records()[6]
    drawn, every, ax = drawn_rows(fig.draw_rwm, rwm, metric="auprc")
    assert {r["group"] for r in drawn} == {"Tokamak-SI"}
    legacy = [r for r in every if r["group"] == "legacy"]
    assert len(legacy) == 1 and legacy[0]["value"] is None and not legacy[0]["drawn"]
    assert len(ax.patches) == 2 and not reported_texts(ax)
    assert baseline_values(ax) == pytest.approx([rwm["counts"]["prevalence"]] * 2)


# sawtooth

SAW_SOURCE = "outputs/labeler/sawtooth/fix6/benchmark.json"


def saw_block(saw, model):
    """The record's presence block the panel draws for `model`, and its key."""
    view = fig.saw_view(saw)
    entry = saw["Tokamak-SI"][model]
    entry = entry[view] if view else entry
    where = f"Tokamak-SI.{model}.{view + '.' if view else ''}crash_tolerance_2ms"
    return entry["crash_tolerance_2ms"], where


def test_the_sawtooth_panel_draws_out_of_fold_presence_f1_from_the_record():
    saw = records()[7]
    drawn, _, ax = drawn_rows(fig.draw_sawtooth, saw)
    by = {(r["series"], r["group"]): r for r in drawn}
    for model in fig.SAW_MODELS:
        row = by[model, "Tokamak-SI"]
        block, where = saw_block(saw, model)
        assert row["value"] == block["presence"]["f1"]
        assert (row["ci_lo"], row["ci_hi"]) == tuple(block["ci95"]["presence_f1"])
        assert row["key"] == f"{where}.presence.f1"
        assert row["source"] == SAW_SOURCE
        assert "out of fold" in row["note"] and "inner-selection" in row["note"]
    saw_series = dict.fromkeys(m for m, _ in by if m.startswith("saw-"))
    assert list(saw_series) == list(fig.SAW_MODELS)
    assert "saw-always-present" not in {m for m, _ in by}  # a dash, not a bar
    assert {round(r["value"], 12) for r in drawn} <= bar_heights(ax)


def test_each_sawtooth_bar_has_the_always_present_dash_on_its_own_bins():
    saw = records()[7]
    for metric in ("f1", "auroc", "auprc"):
        _, every, ax = drawn_rows(fig.draw_sawtooth, saw, metric=metric)
        base = saw_block(saw, fig.SAW_TRIVIAL)[0]["presence"][metric]
        rows = baseline_rows(every)
        assert [r["group"] for r in rows] == [
            f"{m}, Tokamak-SI" for m in fig.SAW_MODELS
        ]
        assert all(r["value"] == base for r in rows)
        assert all(r["key"].endswith(f".presence.{metric}") for r in rows)
        assert all(fig.SAW_TRIVIAL in r["key"] for r in rows)
        si_bars = sorted(
            (p for p in ax.patches if to_hex(p.get_facecolor()).upper() == fig.SI),
            key=lambda p: p.get_x(),
        )
        segments = baseline_segments(ax)
        assert len(segments) == len(si_bars) == len(fig.SAW_MODELS)
        for bar, (left, right, y) in zip(si_bars, segments):
            assert (left, right) == pytest.approx(
                (bar.get_x(), bar.get_x() + bar.get_width())
            )
            assert y == base


def test_a_sawtooth_model_scored_on_other_bins_than_the_baseline_is_refused():
    saw = copy.deepcopy(records()[7])
    view = fig.saw_view(saw)
    entry = saw["Tokamak-SI"][fig.SAW_MODELS[0]]
    block = (entry[view] if view else entry)[fig.SAW_BLOCK]
    # the 1 ms block must agree with the 2 ms one, so move both
    twin = (entry[view] if view else entry)["crash_tolerance_1ms"]
    for b in (block, twin):
        b["presence"]["positive_bins"] += 1
    with pytest.raises(ValueError, match="other bins"):
        drawn_rows(fig.draw_sawtooth, saw)


def test_the_no_q1_negatives_view_is_drawn_once_every_model_has_it():
    saw = copy.deepcopy(records()[7])
    assert fig.saw_view(saw) in (None, fig.SAW_VIEW)
    models = saw["Tokamak-SI"]
    for entry in models.values():
        entry.pop(fig.SAW_VIEW, None)
    assert fig.saw_view(saw) is None
    # a stand-in view: the old-negatives block has the same shape
    for entry in models.values():
        entry[fig.SAW_VIEW] = copy.deepcopy(entry["old_negatives_sensitivity"])
    assert fig.saw_view(saw) == fig.SAW_VIEW
    drawn, _, ax = drawn_rows(fig.draw_sawtooth, saw)
    by = {(r["series"], r["group"]): r for r in drawn}
    for model in fig.SAW_MODELS:
        view = models[model][fig.SAW_VIEW]["crash_tolerance_2ms"]
        row = by[model, "Tokamak-SI"]
        assert row["value"] == view["presence"]["f1"]
        assert row["key"] == (
            f"Tokamak-SI.{model}.{fig.SAW_VIEW}.crash_tolerance_2ms.presence.f1"
        )
        assert fig.SAW_VIEW in row["note"]
    base = models[fig.SAW_TRIVIAL][fig.SAW_VIEW]["crash_tolerance_2ms"]["presence"]
    assert baseline_values(ax) == [base["f1"]] * len(fig.SAW_MODELS)
    del models[fig.SAW_HL3_ECE][fig.SAW_VIEW]  # a view only some models have
    with pytest.raises(ValueError, match="only some"):
        fig.saw_view(saw)


def test_the_f1_panel_draws_the_hl3_papers_stated_accuracy_as_a_reported_bar():
    saw = records()[7]
    drawn, every, ax = drawn_rows(fig.draw_sawtooth, saw)
    assert not any("other machine" in flat(t.get_text()) for t in ax.texts)
    assert not [p for p in ax.patches if p.get_hatch()]
    (reported,) = [r for r in drawn if r["group"] == "legacy"]
    stated = saw["legacy"]["offline"]["accuracy_stated"]
    assert reported["series"] == fig.SAW_HL3 and round(stated, 3) == 0.956
    assert reported["value"] == stated and reported["ci_lo"] is None
    assert reported["key"] == "legacy.offline.accuracy_stated"
    assert "three-class window accuracy" in reported["metric"]
    assert "REPORTED" in reported["note"] and "not an F1" in reported["note"]
    # the real-time figure of the same paper stays in the CSV
    kept = [r for r in every if r["group"] == "legacy" and not r["drawn"]]
    assert {r["key"] for r in kept} == {"legacy.real_time.accuracy_stated"}
    assert all("not an F1" in r["note"] for r in kept)
    # it is an original (coral) bar left of saw-hl3-full's own, marked, with no dash
    (mark,) = reported_texts(ax)
    tick = fig.SAW_MODELS.index(fig.SAW_HL3)
    assert mark.get_position()[0] == pytest.approx(tick - fig.HALF)
    original = [
        p for p in ax.patches if to_hex(p.get_facecolor()).upper() == fig.LEGACY
    ]
    assert [p.get_height() for p in original] == [stated]
    assert len(baseline_segments(ax)) == len(fig.SAW_MODELS)


def test_the_sawtooth_auc_panels_have_no_hl3_paper_bar_and_no_reported_mark():
    saw = records()[7]
    for metric in ("auroc", "auprc"):
        drawn, every, ax = drawn_rows(fig.draw_sawtooth, saw, metric=metric)
        assert not [r for r in drawn if r["group"] == "legacy"]
        assert not reported_texts(ax) and len(ax.patches) == len(fig.SAW_MODELS)
        slot = [r for r in every if r["group"] == "legacy"]
        assert (
            len(slot) == 1 and slot[0]["value"] is None and slot[0]["key"] == "legacy"
        )
        assert not slot[0]["drawn"]


def test_the_sawtooth_f1_csv_names_the_threshold_policy_and_keeps_the_fixed_one_out():
    saw = records()[7]
    drawn, every, _ = drawn_rows(fig.draw_sawtooth, saw)
    assert all(
        "selected per fold on inner-selection shots" in r["note"]
        for r in drawn
        if r["group"] == "Tokamak-SI"
    )
    fixed = [r for r in every if r["group"] == "Tokamak-SI, fixed threshold"]
    assert {r["series"] for r in fixed} == set(fig.SAW_MODELS)
    for r in fixed:
        pooled = saw["Tokamak-SI"][r["series"]]["presence_fixed_threshold"]["pooled"]
        assert not r["drawn"] and r["value"] == pooled["f1"]
        assert "single threshold 0.5" in r["note"]


def test_the_sawtooth_auc_panels_read_the_records_presence_auroc_and_auprc():
    saw = records()[7]
    for metric in ("auroc", "auprc"):
        drawn, every, _ = drawn_rows(fig.draw_sawtooth, saw, metric=metric)
        for model in fig.SAW_MODELS:
            row = next(
                r for r in drawn if r["series"] == model and r["group"] == "Tokamak-SI"
            )
            block, _ = saw_block(saw, model)
            assert row["value"] == block["presence"][metric]
            assert (row["ci_lo"], row["ci_hi"]) == tuple(
                block["ci95"][f"presence_{metric}"]
            )
        assert not [r for r in every if "fixed threshold" in r["group"]]


def test_the_hl3_bar_is_the_full_input_run_and_the_four_input_run_is_csv_only():
    saw = records()[7]
    assert fig.SAW_HL3 == "saw-hl3-full" and fig.SAW_HL3_ECE == "saw-hl3-ece"
    assert fig.SAW_HL3 in fig.SAW_MODELS and fig.SAW_HL3_ECE not in fig.SAW_MODELS
    for metric in ("f1", "auroc", "auprc"):
        drawn, every, ax = drawn_rows(fig.draw_sawtooth, saw, metric=metric)
        assert fig.SAW_HL3_ECE not in {r["series"] for r in drawn}
        ece = [r for r in every if r["series"] == fig.SAW_HL3_ECE]
        presence = [r for r in ece if r["group"] == "Tokamak-SI, four-input run"]
        block, where = saw_block(saw, fig.SAW_HL3_ECE)
        assert len(presence) == 1 and not presence[0]["drawn"]
        assert presence[0]["value"] == block["presence"][metric]
        assert (presence[0]["ci_lo"], presence[0]["ci_hi"]) == tuple(
            block["ci95"][f"presence_{metric}"]
        )
        assert presence[0]["key"] == f"{where}.presence.{metric}"
        # the bars are the drawn models' values: the CSV-only run adds none
        assert {round(r["value"], 12) for r in drawn} <= bar_heights(ax)


def test_the_sawtooth_csv_keeps_three_class_accuracy_beside_its_majority_floor():
    saw = records()[7]
    _, every, _ = drawn_rows(fig.draw_sawtooth, saw)
    for model in (fig.SAW_HL3, fig.SAW_HL3_ECE):
        entry = saw["Tokamak-SI"][model]
        rows = {
            r["metric"]: r
            for r in every
            if r["series"] == model
            and r["metric"].startswith("three-class")
            and r["group"] != "legacy"
        }
        accuracy = rows["three-class window accuracy"]
        floor = rows["three-class window accuracy, majority-class floor"]
        assert accuracy["value"] == entry["three_class_window_accuracy"]
        assert floor["value"] == entry["three_class_majority_baseline"]["accuracy"]
        assert floor["value"] < accuracy["value"]
        assert not accuracy["drawn"] and not floor["drawn"]
        assert accuracy["source"] == SAW_SOURCE


def test_a_presence_block_that_differs_between_crash_tolerances_is_refused():
    def entry(one, two):
        return {
            "crash_tolerance_1ms": {"presence": {"f1": one}},
            "crash_tolerance_2ms": {"presence": {"f1": two}, "ci95": {}},
        }

    assert fig.saw_presence({"Tokamak-SI": {"m": entry(0.9, 0.9)}}, "m")[0] == {
        "f1": 0.9
    }
    with pytest.raises(ValueError, match="crash tolerances"):
        fig.saw_presence({"Tokamak-SI": {"m": entry(0.9, 0.8)}}, "m")


# detachment: detach-ours, and the TangTV vote's consistency check (CSV only)

DETACH_TE = "docs/labeler/results/detachment_te_check.json"
DETACH_SOURCE = "docs/labeler/results/detachment_source_v3.json"


def detachment_records():
    """The two records and the check read from them (the test is skipped on a
    checkout without them)."""
    keys = ("detach_te", "detach_source")
    missing = [fig.SOURCES[k].name for k in keys if not fig.SOURCES[k].is_file()]
    if missing:
        pytest.skip(f"records not on this checkout: {missing}")
    te, source = (fig.load(fig.SOURCES[k]) for k in keys)
    return te, source, fig.detachment_check(te, source)


def test_the_detachment_adapter_reads_the_auroc_and_the_agreement_from_the_records():
    te, source, det = detachment_records()
    pooled = te["by_tier"]["tangtv_vote_before_gate"][
        "auroc_neg_te_detached_vs_attached"
    ]["pooled"]
    assert det["auroc"]["value"] == pooled["value"]
    assert det["auroc"]["ci"] == tuple(pooled["ci95"])
    assert (det["auroc"]["bins"], det["auroc"]["shots"]) == (668, 23)
    states = source["agreement_vote_vs_te"]["states"]
    assert det["agreement"]["value"] == states["agreement"]
    assert det["agreement"]["kappa"] == states["kappa"]
    assert (det["agreement"]["agree"], det["agreement"]["bins"]) == (423, 448)
    # the baseline: a vote that calls Te's majority state in every compared bin
    by_te = source["agreement_vote_vs_te"]["vote_by_te"]
    te_state = {
        t: sum(by_te[v][t] for v in ("attached", "detached"))
        for t in ("attached", "detached")
    }
    assert sum(te_state.values()) == 448
    assert det["agreement"]["base"] == max(te_state.values()) / 448
    assert det["agreement"]["base"] < det["agreement"]["value"]
    # the numbers the paper states, to the digits it states them
    assert [round(x, 2) for x in (pooled["value"], *pooled["ci95"])] == [
        0.95,
        0.91,
        0.98,
    ]
    assert round(states["agreement"], 2) == 0.94 and round(states["kappa"], 2) == 0.87


def test_an_agreement_that_is_not_its_counts_is_refused():
    def source(agreement, table=None):
        return {
            "agreement_vote_vs_te": {
                "states": {
                    "agree": 90,
                    "bins": 100,
                    "agreement": agreement,
                    "shots": 5,
                    "kappa": 0.8,
                },
                "vote_by_te": table
                or {
                    "attached": {"attached": 50, "detached": 5},
                    "detached": {"attached": 5, "detached": 40},
                },
            }
        }

    te = {
        "by_tier": {
            "tangtv_vote_before_gate": {
                "auroc_neg_te_detached_vs_attached": {
                    "pooled": {
                        "value": 0.9,
                        "ci95": [0.8, 0.95],
                        "n_bins": 10,
                        "n_shots": 2,
                    }
                }
            }
        }
    }
    ok = fig.detachment_check(te, source(0.9))["agreement"]
    assert ok["value"] == 0.9 and ok["base"] == 0.55  # Te's majority state: 55 of 100
    with pytest.raises(ValueError, match="counts"):
        fig.detachment_check(te, source(0.95))
    short = {
        "attached": {"attached": 50, "detached": 5},
        "detached": {"attached": 5, "detached": 30},
    }
    with pytest.raises(ValueError, match="compared bins"):
        fig.detachment_check(te, source(0.9, short))


DETACH_OURS = "docs/labeler/results/detachment_ours.json"


def detachment_panel_records():
    """The check with detach-ours read in, as `read_records` gives the panel."""
    if not fig.SOURCES["detach_ours"].is_file():
        pytest.skip("detachment_ours.json not on this checkout")
    _, source, det = detachment_records()
    ours = fig.load(fig.SOURCES["detach_ours"])
    return source, ours, {**det, "ours": fig.detachment_ours(ours)}


def test_the_detachment_adapter_reads_detach_ours_macro_f1_and_its_majority_base():
    _, ours, det = detachment_panel_records()
    cv = ours["cv_shots"]
    assert det["ours"]["value"] == cv["macro_f1"]["value"]
    assert det["ours"]["ci"] == tuple(cv["macro_f1"]["ci95"])
    assert (det["ours"]["bins"], det["ours"]["shots"]) == (706, 22)
    # always attached (417 of 706 bins): F1 2p/(1+p) on attached, 0 on detached
    p = 417 / 706
    assert det["ours"]["base"] == pytest.approx(2 * p / (1 + p) / 2)
    assert det["ours"]["base"] < det["ours"]["value"]
    broken = {"cv_shots": {**cv, "n_bins": 700}}
    with pytest.raises(ValueError, match="class supports"):
        fig.detachment_ours(broken)


def test_the_detachment_f1_panel_is_detach_ours_alone_with_its_interval():
    source, ours, det = detachment_panel_records()
    drawn, every, ax = drawn_rows(fig.draw_detachment, det, panel="h")
    (row,) = drawn
    assert row["series"] == fig.DETACH_NAME == "detach-ours"
    assert row["group"] == "Tokamak-SI" and row["metric"] == "macro F1"
    assert row["value"] == ours["cv_shots"]["macro_f1"]["value"]
    assert (row["ci_lo"], row["ci_hi"]) == tuple(ours["cv_shots"]["macro_f1"]["ci95"])
    assert row["source"] == DETACH_OURS and row["key"] == "cv_shots.macro_f1.value"
    (base,) = baseline_rows(every)
    assert base["value"] == det["ours"]["base"] and base["source"] == DETACH_OURS
    assert baseline_values(ax) == [base["value"]]
    (bar,) = ax.patches  # no original bar
    assert to_hex(bar.get_facecolor()).upper() == fig.SI
    assert bar.get_x() + bar.get_width() / 2 == pytest.approx(0)  # on its tick
    assert [c for c in ax.containers if isinstance(c, ErrorbarContainer)]
    assert ax.get_ylabel() == "F1"  # every panel's axis names the metric
    assert [t.get_text() for t in ax.get_xticklabels()] == [
        fig.two_line(fig.DETACH_NAME)
    ]
    # the vote's consistency check stays in the CSV
    kept = {r["group"]: r for r in every if not r["drawn"]}
    states = source["agreement_vote_vs_te"]["states"]
    assert kept["Tokamak-SI, agreement"]["value"] == states["agreement"]
    assert kept["Tokamak-SI, Cohen's kappa"]["value"] == states["kappa"]
    assert {r["series"] for r in kept.values()} == {fig.DETACH_VOTE}
    text = " ".join(flat(t.get_text()) for t in ax.texts)
    assert not re.search(r"kappa|κ|agreement|\bshots\b", text), text


@pytest.mark.parametrize("metric", ["auroc", "auprc"])
def test_the_detachment_auc_panels_are_not_reported(metric):
    _, _, det = detachment_panel_records()
    drawn, every, ax = drawn_rows(fig.draw_detachment, det, metric=metric, panel="h0")
    assert not drawn and not ax.patches and not baseline_values(ax)
    assert [flat(t.get_text()) for t in ax.texts] == ["not reported"]
    assert ax.get_ylabel() == fig.METRICS[metric]
    (own,) = [r for r in every if r["series"] == fig.DETACH_NAME]
    assert own["value"] is None and not own["drawn"]
    check = [r for r in every if r["series"] == fig.DETACH_VOTE]
    if metric == "auroc":  # the -Te check, in the CSV only
        (row,) = check
        assert row["source"] == DETACH_TE and not row["drawn"]
    else:
        assert not check


def panel_title(ax):
    """A panel's title: at its left edge, but the one-tick RWM panel, narrower than
    its name, ends it at its right edge."""
    left, right = ax.get_title(loc="left"), ax.get_title(loc="right")
    assert not (left and right), (left, right)
    return left or right


def panel_titles(figure):
    """The panel titles of a figure's axes, in the order they were made."""
    return [panel_title(ax) for ax in figure.axes if panel_title(ax)]


def test_both_figures_put_detachment_last_in_the_bottom_row():
    # the owner's layout (2026-10-14): three panels over four
    recs = records()
    with fig.plot_style():
        f1 = fig.figure_f1(recs, fig.load(fig.SOURCES["coverage"]), fig.Rows())
        twin = fig.figure_auc(recs, fig.Rows())
    for figure, blocks in ((f1, 1), (twin, 2)):
        titles = panel_titles(figure)
        per_block = 7
        for b in range(blocks):
            names = titles[b * per_block : (b + 1) * per_block]
            assert names == [
                "Alfvén Eigenmodes",
                "Confinement",
                "Edge Localized Modes",
                "Tearing Modes",
                "Sawtooth",
                "Resistive Wall Modes",
                "Detachment",
            ]
        for row in (figure.axes[:3], figure.axes[3:7]):
            tops = {round(ax.get_position().y1, 6) for ax in row}
            assert len(tops) == 1  # one row
            lefts = [ax.get_position().x0 for ax in row]
            assert lefts == sorted(lefts)


def test_every_panel_title_is_plain_with_no_parentheses_or_second_line():
    recs = records()
    with fig.plot_style():
        f1 = fig.figure_f1(recs, fig.load(fig.SOURCES["coverage"]), fig.Rows())
        twin = fig.figure_auc(recs, fig.Rows())
    seen = []
    for figure in (f1, twin):
        for ax in figure.axes:
            title = panel_title(ax)
            if title:
                seen.append(title)
                assert "(" not in title and ")" not in title, title
                assert "\n" not in title, title
    assert {
        "Alfvén Eigenmodes",
        "Tearing Modes",
        "Sawtooth",
        "Resistive Wall Modes",
        "Labeled data per set",
    } <= set(seen)


def test_no_panel_title_runs_off_the_page():
    recs = records()
    with fig.plot_style():
        figures = (
            fig.figure_f1(recs, fig.load(fig.SOURCES["coverage"]), fig.Rows()),
            fig.figure_auc(recs, fig.Rows()),
        )
        for figure in figures:
            renderer = FigureCanvasAgg(figure).get_renderer()
            figure.draw(renderer)
            page = figure.bbox
            for ax in figure.axes:
                for t in (ax.title, ax._left_title, ax._right_title):
                    if t.get_text():
                        box = t.get_window_extent(renderer)
                        assert box.x0 >= page.x0 - 0.5, t.get_text()
                        assert box.x1 <= page.x1 + 0.5, t.get_text()


def test_a_tick_spacing_is_as_wide_in_every_panel_of_both_figures():
    """The three panels of the top row and the four of the bottom one share one
    tick spacing; the rows with room to spare spread their panels apart."""
    recs = records()
    with fig.plot_style():
        figures = (
            fig.figure_f1(recs, fig.load(fig.SOURCES["coverage"]), fig.Rows()),
            fig.figure_auc(recs, fig.Rows()),
        )
    for figure in figures:
        units = []
        for ax in figure.axes:
            if ax.get_yscale() == "log":  # the coverage panel has its own x axis
                continue
            lo, hi = ax.get_xlim()
            units.append(ax.get_position().width * figure.get_figwidth() / (hi - lo))
        assert units == pytest.approx([units[0]] * len(units))
        for row in (figure.axes[:3], figure.axes[3:7]):
            right = max(ax.get_position().x1 for ax in row)
            assert right <= 1 - fig.RIGHT_IN / figure.get_figwidth() + 1e-9
            for a, b in itertools.pairwise(row):
                gap = (
                    b.get_position().x0 - a.get_position().x1
                ) * figure.get_figwidth()
                assert gap >= fig.GAP_IN - 1e-9


# REPORTED and the trivial baselines


def score_panels(figure):
    """A figure's score panels (the coverage panel, on a log axis, is not one):
    seven per metric block."""
    return [ax for ax in figure.axes if ax.get_yscale() != "log"]


def interval_xs(ax):
    """Where on the x axis a panel draws an error bar."""
    return {
        round(seg[0][0], 9)
        for c in ax.containers
        if isinstance(c, ErrorbarContainer)
        for lc in c.lines[2]
        for seg in lc.get_segments()
    }


def both_figures_drawn():
    """The F1 figure and the AUC one, drawn, with their tables."""
    recs = records()
    rows = fig.Rows(), fig.Rows()
    f1 = fig.figure_f1(recs, fig.load(fig.SOURCES["coverage"]), rows[0])
    twin = fig.figure_auc(recs, rows[1])
    for figure in (f1, twin):
        figure.draw(FigureCanvasAgg(figure).get_renderer())
    return (f1, rows[0]), (twin, rows[1])


def test_reported_marks_the_original_bars_of_another_dataset_and_only_those():
    with fig.plot_style():
        (f1, _), (twin, _) = both_figures_drawn()
        seen = {}
        for name, figure in (("f1", f1), ("auc", twin)):
            for i, ax in enumerate(score_panels(figure)):
                marks = reported_texts(ax)
                if marks:
                    assert len(marks) == 1
                    seen.setdefault(name, set()).add((i // 7, panel_title(ax)))
        # F1: the confinement paper's .94, D. Smith's .992 and the HL-3 paper's
        # stated .956 -- not AE (the same set) or TM (the same cohort rescored);
        # AUROC: the confinement bound and the NSTX forest's slice AUROC
        assert seen == {
            "f1": {
                (0, "Confinement"),
                (0, "Edge Localized Modes"),
                (0, "Sawtooth"),
            },
            "auc": {(0, "Confinement"), (0, "Resistive Wall Modes")},
        }
        for figure in (f1, twin):
            renderer = figure.canvas.get_renderer()
            for ax in score_panels(figure):
                marks = reported_texts(ax)
                intervals = interval_xs(ax)
                segments = baseline_segments(ax)
                marked_bars = []
                for t in marks:
                    x, y = t.get_position()
                    (bar,) = [
                        p
                        for p in ax.patches
                        if p.get_x() < x < p.get_x() + p.get_width()
                    ]
                    marked_bars.append(bar)
                    assert to_hex(bar.get_facecolor()).upper() == fig.LEGACY
                    assert (t.get_rotation(), t.get_color()) == (90, "white")
                    assert t.get_fontweight() == "bold" and t.get_fontsize() >= 7
                    # the face of the other labels (a sans), not a serif fallback
                    face = fm.get_font(fm.findfont(t.get_fontproperties()))
                    other = next(v for v in ax.texts if v.get_text().startswith("."))
                    same = fm.get_font(fm.findfont(other.get_fontproperties()))
                    assert face.family_name == same.family_name
                    assert "serif" not in face.family_name.lower()
                    assert y == pytest.approx(bar.get_height() / 2)  # centred in it
                    inside, outer = (
                        t.get_window_extent(renderer),
                        bar.get_window_extent(renderer),
                    )
                    assert outer.x0 <= inside.x0 and inside.x1 <= outer.x1
                    assert outer.y0 <= inside.y0 and inside.y1 <= outer.y1
                    # no error bar and no baseline on a reported bar
                    assert round(bar.get_x() + bar.get_width() / 2, 9) not in intervals
                    assert not [g for g in segments if g[0] < x < g[1]]
                # every other bar has exactly one dash across just its width
                for bar in ax.patches:
                    if bar in marked_bars:
                        continue
                    span = (bar.get_x(), bar.get_x() + bar.get_width())
                    dashes = [g for g in segments if g[:2] == pytest.approx(span)]
                    assert len(dashes) == 1, (panel_title(ax), span)
                assert len(segments) == len(ax.patches) - len(marked_bars)


def test_the_legend_has_one_trivial_baseline_entry_and_no_diamond_or_hatch():
    with fig.plot_style():
        (f1, _), (twin, _) = both_figures_drawn()
        for figure, labels in (
            (f1, ["original", "Tokamak-SI", "human-reviewed subset", fig.BASELINE]),
            (twin, ["original", "Tokamak-SI", fig.BASELINE]),
        ):
            (legend,) = figure.legends
            assert [t.get_text() for t in legend.get_texts()] == labels
            (dash,) = [h for h in legend.legend_handles if h.get_linestyle() == "--"]
            assert to_hex(dash.get_color()) == "#000000"
            for h in legend.legend_handles:
                assert not (hasattr(h, "get_hatch") and h.get_hatch())
                assert not (hasattr(h, "get_marker") and h.get_marker() == "D")


def test_every_baseline_dash_is_one_row_of_the_table_with_its_source():
    with fig.plot_style():
        figures = both_figures_drawn()
    for figure, rows in figures:
        dashes = sorted(
            round(y, 9)
            for ax in score_panels(figure)
            for _, _, y in baseline_segments(ax)
        )
        table = [r for r in rows.rows if r["series"] == fig.BASELINE]
        assert dashes == sorted(round(r["value"], 9) for r in table)
        assert all(r["drawn"] and r["source"] and r["key"] for r in table)
        assert all(", " in r["group"] for r in table)  # "<model>, <setting>"


# coverage


def coverage_rows(cov):
    rows = fig.Rows()
    ax = Figure().add_subplot()
    fig.draw_coverage(ax, cov, rows, "g")
    return rows.rows, ax


def test_the_coverage_panel_draws_hours_writes_shots_and_notes_what_has_no_value():
    cov = fig.load(fig.SOURCES["coverage"])
    every, ax = coverage_rows(cov)
    assert ax.get_yscale() == "log"
    texts = [flat(t.get_text()) for t in ax.texts]
    for key in cov["order"]:
        entry = cov["sets"][key]
        for side, group in (("legacy", "legacy"), ("tokamak_si", "Tokamak-SI")):
            block = entry[side]
            mine = {
                r["metric"]: r
                for r in every
                if r["series"] == entry["name"] and r["group"] == group
            }
            hours, shots = mine["labeled time (h)"], mine["labeled shots"]
            assert hours["key"] == f"sets.{key}.{side}.seconds"
            assert shots["key"] == f"sets.{key}.{side}.shots"
            assert shots["value"] == block["shots"]
            if block["seconds"] is None:  # a grey note says why, no bar
                assert hours["value"] is None and not hours["drawn"]
                status = fig.STATUS_TEXT[block["status"]]
                assert any(t.startswith(status) for t in texts)
            else:
                assert hours["value"] == pytest.approx(block["seconds"] / 3600.0)
                assert str(block["shots"]) in texts  # written on its bar
    assert not [p for p in ax.patches if p.get_hatch()]
    reviewed = [
        r
        for r in every
        if r["group"] == "Tokamak-SI, reviewed subset" and r["metric"].endswith("(h)")
    ]
    assert reviewed and all(
        r["value"]
        <= next(
            s["value"]
            for s in every
            if s["series"] == r["series"]
            and s["group"] == "Tokamak-SI"
            and s["metric"] == r["metric"]
        )
        for r in reviewed
    )


def test_the_coverage_axis_reaches_the_smallest_reviewed_subset():
    cov = fig.load(fig.SOURCES["coverage"])
    every, ax = coverage_rows(cov)
    low = ax.get_ylim()[0]
    reviewed = [
        r for r in every if r["group"] == "Tokamak-SI, reviewed subset" and r["drawn"]
    ]
    assert reviewed and all(r["value"] >= low for r in reviewed)
    assert {r["series"] for r in reviewed} >= {"sawtooth"}  # three shots


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
    assert {r["panel"] for r in rows.rows} == set("abcdefgh")


def test_every_two_texts_keep_apart_in_either_figure():
    """Every two texts, a value's white box included, are at least 1.5 pt apart
    (the values of neighbouring pairs met in the AE AUROC panel)."""
    recs = records()
    with fig.plot_style():
        figures = (
            fig.figure_f1(recs, fig.load(fig.SOURCES["coverage"]), fig.Rows()),
            fig.figure_auc(recs, fig.Rows()),
        )
        for f in figures:
            renderer = FigureCanvasAgg(f).get_renderer()
            f.draw(renderer)
            pt = 72 / f.dpi
            boxes = []
            for t in f.findobj(Text):
                if not (t.get_visible() and t.get_text().strip()):
                    continue
                box = t.get_window_extent(renderer)
                if t.get_bbox_patch() is not None:
                    box = Bbox.union([box, t.get_bbox_patch().get_window_extent()])
                boxes.append((t.get_text(), box))
            clash = [
                (a, b)
                for (a, ba), (b, bb) in itertools.combinations(boxes, 2)
                if max(ba.x0 - bb.x1, bb.x0 - ba.x1, ba.y0 - bb.y1, bb.y0 - ba.y1) * pt
                < 1.5
            ]
            assert not clash, clash


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


def test_nothing_is_left_as_not_yet_scored_and_the_twin_has_the_sawtooth_panels():
    recs = records()
    with fig.plot_style():
        rows = fig.Rows()
        figure = fig.figure_f1(recs, fig.load(fig.SOURCES["coverage"]), rows)
        twin_rows = fig.Rows()
        twin = fig.figure_auc(recs, twin_rows)
    for f in (figure, twin):
        assert "not yet scored" not in {t.get_text() for t in f.findobj(Text)}
    assert {r["panel"] for r in twin_rows.rows if r["series"] == "saw-ours"} == {
        "e0",
        "e1",
    }
    assert {r["panel"] for r in rows.rows if r["series"] == "saw-ours"} == {"e"}


def test_the_auc_figure_heads_its_auroc_and_auprc_blocks():
    """The two blocks repeat the panel titles: a heading names each metric, above
    its block."""
    with fig.plot_style():
        twin = fig.figure_auc(records(), fig.Rows())
    headings = [t for t in twin.texts if t.get_text()]
    assert [t.get_text() for t in headings] == list(fig.AUC_HEADINGS.values())
    auroc, auprc = (t.get_position()[1] for t in headings)
    boxes = [ax.get_position() for ax in twin.axes]  # AUROC's seven, then AUPRC's
    assert auroc > max(b.y1 for b in boxes[:4])
    assert min(b.y0 for b in boxes[4:7]) > auprc > max(b.y1 for b in boxes[7:11])


# the guard: a drawn score is read from a record, never typed


def test_no_score_is_typed_into_the_script():
    recs = records()
    cov = fig.load(fig.SOURCES["coverage"])
    with fig.plot_style():
        rows = fig.Rows()
        fig.figure_f1(recs, cov, rows)
        fig.figure_auc(recs, rows)
    # a trivial baseline's AUROC is the chance level 0.5 by definition, and the
    # confinement macro AUPRC's is 0.25 (four classes' prevalences sum to 1); these
    # coincide with the layout's own tick values, so they are exempt
    definitional = {"AUROC": 0.5, "macro AUROC": 0.5, "macro AUPRC": 0.25}

    def by_definition(r):
        return r["series"] == fig.BASELINE and r["value"] == pytest.approx(
            definitional.get(r["metric"], -1)
        )

    scores = {
        round(r["value"], 3)
        for r in rows.rows
        if r["drawn"]
        and r["panel"] != "g"
        and 0 < (r["value"] or 0) < 1
        and not by_definition(r)
    }
    tree = ast.parse(SCRIPT.read_text())
    # the layout's own constants (inches, tick spacings) are not scores: a record's
    # 0.900 or 0.420 may meet HALF or GAP_IN
    layout = {
        id(n)
        for a in tree.body
        if isinstance(a, ast.Assign)
        and all(
            isinstance(t, ast.Name | ast.Tuple)
            and all(
                x.id.endswith("_IN") or x.id in ("HALF", "BAR", "MARGIN")
                for x in (t.elts if isinstance(t, ast.Tuple) else [t])
            )
            for t in a.targets
        )
        for n in ast.walk(a.value)
    }
    typed = {
        round(n.value, 3)
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant)
        and isinstance(n.value, float)
        and 0 < n.value < 1
        and id(n) not in layout
    }
    assert typed, "the guard sees the script's other constants"
    assert not scores & typed, scores & typed


def test_a_score_is_written_without_its_leading_zero():
    assert fig.score_text(0.8357) == ".836"
    assert fig.score_text(0.94, 2) == ".94"
    assert fig.score_text(1.0) == "1.000"


def test_a_name_with_two_hyphens_breaks_at_the_last_one():
    assert fig.two_line("ae-ours") == "ae-\nours"
    assert fig.two_line("saw-hl3-full") == "saw-hl3-\nfull"
    assert fig.two_line("saw-derivative") == "saw-\nderivative"
    assert fig.two_line("plain") == "plain"
