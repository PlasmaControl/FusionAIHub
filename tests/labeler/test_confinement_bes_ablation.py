"""The ablation script's row table and its bookkeeping: every step names rows that
exist, the paper-training rows build the paper's network, and the per-class support
helper counts windows and shots."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts/labeler/confinement_bes_ablation.py"
)
spec = importlib.util.spec_from_file_location("confinement_bes_ablation", SCRIPT)
abl = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = abl
spec.loader.exec_module(abl)


def test_every_step_and_year_row_names_a_row_of_the_table():
    for label, name, ref in abl.STEPS:
        assert name in abl.ROWS and ref in abl.ROWS, label
    assert set(abl.YEAR_ROWS) <= set(abl.ROWS)


def test_paper_training_rows_use_no_padding_no_early_stop_and_geometry():
    for name in ("full_cum_abcdrgef", "full_cum_abcdrge", "full_leak_abcdrge"):
        row = abl.ROWS[name]
        cfg = abl.training_config(row)
        assert tuple(cfg.padding) == (0, 0, 0) and not cfg.early_stop, name
        assert cfg.steps == 60000, name
        assert row.layout == "geometry" and row.arch == "paper", name
    only_c = abl.training_config(abl.ROWS["full_only_c"])
    assert tuple(only_c.padding) == (0, 0, 0) and not only_c.early_stop
    # the earlier rows keep the first retrain's early-stopped, padded network
    old = abl.training_config(abl.ROWS["cum_abcdrgef"])
    assert tuple(old.padding) != (0, 0, 0) and old.early_stop


def test_single_factor_rows_on_the_444_shots_differ_from_only_ge_by_one_factor():
    base = abl.ROWS["only_ge"]
    assert abl.ROWS["ge_a"].gate and not base.gate
    assert abl.ROWS["ge_b"].transition_ms > 0 and base.transition_ms == 0
    assert abl.ROWS["ge_c"].optimiser == "paper" and base.optimiser == "ours"
    assert abl.ROWS["ge_r"].rows == (0, 6) and base.rows != (0, 6)
    for k in "abcr":
        row = abl.ROWS[f"ge_{k}"]
        assert (row.data, row.shots, row.protocol) == (
            base.data,
            base.shots,
            base.protocol,
        )


def test_support_counts_windows_and_shots_per_class():
    labels = np.array([0, 0, 1, 1, 1, 3])
    shots = np.array([1, 1, 1, 2, 3, 4])
    got = abl._support(labels, shots)
    assert got["L"] == {"windows": 2, "shots": 1}
    assert got["H"] == {"windows": 3, "shots": 3}
    assert got["QH"] == {"windows": 0, "shots": 0}
    assert got["WP"] == {"windows": 1, "shots": 1}
