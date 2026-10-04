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


def _scored(f1: float, shots: int) -> dict:
    return {"macro_f1": f1, "ci95_macro_f1": [f1 - 0.1, f1 + 0.1], "shots": shots}


def test_population_tables_print_rows_steps_fragmentation_mix_and_years(
    tmp_path, capsys
):
    import argparse
    import json

    classes = {
        c: {
            "intervals": 3,
            "median_ms": 20.0,
            "q25_ms": 10.0,
            "q75_ms": 40.0,
            "share_under_100_ms": 0.8,
        }
        for c in ("L", "H", "QH", "WP")
    }
    diff = {"shots": 6, "difference": 0.01, "ci95": [-0.02, 0.04]}
    record = {
        "rows": {
            "base": {
                "own": _scored(0.7, 10),
                "corpus_shots": _scored(0.7, 10),
                "other_shots": {"macro_f1": None, "shots": 0},
            }
        },
        "paired_steps": [
            {
                "step": "a gate",
                "row": "only_a",
                "versus": "base",
                "common_shots": diff,
                "corpus_shots": diff,
                "other_shots": {"shots": 0},
            }
        ],
        "fragmentation": {
            "corpus": {"classes": classes},
            "other": {"classes": classes},
        },
        "class_mix": {
            "base": {
                "as_scored": 0.7,
                "paper_test_mix": [8.3, 21.8, 17.4, 9.7],
                "reweighted_macro_f1": 0.65,
                "reweighted_f1": {"L": 0.6, "H": 0.7, "QH": 0.6, "WP": 0.7},
            }
        },
        "by_year": {"base": {"2021": _scored(0.6, 4), "2022": _scored(0.4, 3)}},
    }
    (tmp_path / "populations.json").write_text(json.dumps(record))
    abl.markdown_populations(argparse.Namespace(out_dir=tmp_path))
    out = capsys.readouterr().out
    assert "| `base` | 0.700 [0.60, 0.80] (10) | 0.700 [0.60, 0.80] (10) | - |" in out
    assert "`only_a` - `base`" in out
    assert "| Year of the shot | `base` |" in out
    assert "| 2022 | 0.400 [0.30, 0.50] (3) |" in out


def test_review_stages_name_rows_of_the_table():
    assert set(abl.REPEAT_ROWS) <= set(abl.ROWS)
    assert all(abl.ROWS[n].protocol == "paper" for n in abl.REPEAT_ROWS)
    assert set(abl.SUBSET_ROWS) <= set(abl.ROWS)
    # the failing blocks are read from a row each shot is tested in once
    assert abl.ROWS[abl.FAILURE_ROW].protocol == "cv5"


def test_compact_adds_the_mean_of_l_h_qh_without_wpqh():
    summary = {
        "windows": 10,
        "shots": 3,
        "macro_f1": 0.5,
        "ci95": {"macro_f1": [0.4, 0.6]},
        "classes": {
            "L": {"f1": 0.9, "shots": 1},
            "H": {"f1": 0.6, "shots": 1},
            "QH": {"f1": None, "shots": 0},
            "WP": {"f1": 0.0, "shots": 1},
        },
    }
    got = abl._compact(summary)
    assert abs(got["macro_f1_lhqh"] - 0.75) < 1e-12  # QH has no windows: skipped
    assert got["f1"]["WP"] == 0.0 and got["macro_f1"] == 0.5


def test_the_table_3_row_by_year_prints_f1_and_shots_per_class(tmp_path, capsys):
    import argparse
    import json

    classes = {
        c: {
            "intervals": 3,
            "median_ms": 20.0,
            "q25_ms": 10.0,
            "q75_ms": 40.0,
            "share_under_100_ms": 0.8,
        }
        for c in ("L", "H", "QH", "WP")
    }
    year = {
        "shots": 21,
        "windows": 2000,
        "macro_f1": 0.48,
        "ci95_macro_f1": [0.4, 0.9],
        "f1": {"L": 0.0, "H": None, "QH": 0.96, "WP": None},
        "shots_per_class": {"L": 1, "H": 0, "QH": 20, "WP": 0},
    }
    record = {
        "rows": {},
        "paired_steps": [],
        "fragmentation": {
            "corpus": {"classes": classes},
            "other": {"classes": classes},
        },
        "class_mix": {},
        "by_year": {"full_cum_abcdrgef": {"2014": year}},
    }
    (tmp_path / "populations.json").write_text(json.dumps(record))
    abl.markdown_populations(argparse.Namespace(out_dir=tmp_path))
    out = capsys.readouterr().out
    assert "| 2014 | 21 | 2,000 | 0.00 (1) | - | 0.96 (20) | - |" in out
    assert "a class carried by one shot counts for a half" in out


def test_block_placement_compares_the_failure_windows_with_the_shot_median(
    tmp_path, monkeypatch
):
    import pandas as pd

    monkeypatch.setattr(abl, "GEOMETRY", tmp_path)
    times = np.array([0.0, 1000.0])
    psin = np.full((2, 64), 0.9, dtype=np.float32)
    psin[:, 0:8] = 1.1  # row 0 lies outside the separatrix
    psin[1, 8:16] = 1.3  # row 1 drifts outward in the second EFIT slice
    np.savez(tmp_path / "7.npz", gtime_ms=times, psin=psin)
    blocks = pd.DataFrame(
        {"start": [0.0], "outer_psin": [1.1], "channels_in_band": [40]},
        index=pd.Index([7], name="shot"),
    )
    early = abl.block_placement(7, np.array([0.0]), blocks)
    assert abs(early["outer_psin_block"] - 1.1) < 1e-6
    assert early["channels_in_band_block"] == 40
    assert early["placement_matches_shot"] and early["outer_psin_matches_shot"]
    late = abl.block_placement(7, np.array([1000.0]), blocks)
    assert abs(late["outer_psin_block"] - 1.3) < 1e-6
    assert late["channels_in_band_block"] == 32
    assert not late["outer_psin_matches_shot"] and not late["placement_matches_shot"]
