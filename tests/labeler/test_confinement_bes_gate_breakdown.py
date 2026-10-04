"""The gate breakdown script: shares of windows passing the beam gate, by class."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts/labeler/confinement_bes_gate_breakdown.py"
)
spec = importlib.util.spec_from_file_location("confinement_bes_gate_breakdown", SCRIPT)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def test_breakdown_counts_each_reason():
    table = pd.DataFrame(
        {
            "shot": [1, 1, 1, 2, 2],
            "label": [0, 0, 0, 3, 3],
            # L: passes, beam off, between 50 and 700 kW; WP: passes, 150R too high
            "p15L_min": [800e3, 0.0, 300e3, 900e3, 900e3],
            "p15R_max": [0.0, 0.0, 0.0, 0.0, 500e3],
        }
    )
    out = gate.breakdown(table)
    assert set(out) == {"L", "WP"}
    assert out["L"]["windows"] == 3 and out["L"]["shots"] == 1
    assert np.isclose(out["L"]["passes_gate"], 1 / 3)
    assert np.isclose(out["L"]["left_below_50kW"], 1 / 3)
    assert np.isclose(out["L"]["left_50_to_700kW"], 1 / 3)
    assert out["L"]["right_above_200kW"] == 0.0
    assert np.isclose(out["WP"]["passes_gate"], 0.5)
    assert np.isclose(out["WP"]["right_above_200kW"], 0.5)
