"""The benchmark figure's confine-cnn bars: the Tokamak-SI score is the ablation's
protocol row (the paper's one confine-cnn score), read through a small adapter."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from matplotlib.figure import Figure

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
LEGACY = {"paper": {c: {"f1": f} for c, f in zip(fig.CLASSES, (0.9, 1.0, 0.9, 0.8))}}


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
