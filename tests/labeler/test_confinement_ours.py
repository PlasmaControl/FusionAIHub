"""The segmenting steps of ``scripts/labeler/confinement_ours.py`` on small cases."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/labeler/confinement_ours.py"
spec = importlib.util.spec_from_file_location("confinement_ours", SCRIPT)
ours = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ours)


def test_bridge_gaps_closes_short_gaps_between_runs_only():
    active = np.zeros(200, dtype=bool)
    active[10:20] = active[30:40] = True  # a 10-bin gap
    active[120:130] = True  # an 80-bin gap before this run
    out = ours.bridge_gaps(active, 50)
    assert out[10:40].all()  # bridged
    assert not out[40:120].any()  # too long to bridge
    assert not out[:10].any() and not out[130:].any()  # edges stay as they were
    assert not ours.bridge_gaps(np.zeros(5, dtype=bool), 10).any()


def test_segment_shot_merges_a_short_run_into_its_neighbour_and_drops_short_islands():
    n = 400
    prob = np.zeros((4, n), dtype=np.float32)
    prob[0, :200] = 1.0  # L, then H
    prob[1, 200:] = 1.0
    active = np.ones(n, dtype=bool)
    got = ours.segment_shot(prob, active)
    assert [(c, lo, hi) for c, lo, hi, _ in got] == [(0, 0, 200), (1, 200, 400)]
    # a beam-off stretch splits the shot and a run under 20 bins is dropped
    active[300:] = False
    active[100:110] = False
    got = ours.segment_shot(prob, active)
    assert all(hi - lo >= ours.MIN_SEGMENT_MS for _, lo, hi, _ in got)
    assert all(conf > 0.99 for *_, conf in got)
