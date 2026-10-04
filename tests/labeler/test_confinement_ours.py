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


def test_neighbour_pairs_counts_close_shots_in_different_folds():
    test_of = {100: 0, 101: 1, 102: 1, 150: 2, 151: 2}
    got = ours.neighbour_pairs(test_of, within=2)
    assert got["pairs"] == 4  # (100,101), (100,102), (101,102), (150,151)
    assert got["in_different_folds"] == 2


def test_roster_rows_floor_makes_uncertain_and_tiers_the_unreviewed_classes():
    found = [(0, 0, 100, 0.95), (2, 100, 200, 0.55), (3, 200, 300, 0.9)]
    held = ours.roster_rows(1, found, curated=True, last_curated=196493)
    assert [r["category"] for r in held] == [2, 5, 4]  # QH under the floor: uncertain
    assert [r["predicted"] for r in held] == [2, 3, 4]  # the class stays on record
    assert {r["tier"] for r in held} == {"model"}
    assert {r["source"] for r in held} == {"held_out"}
    other = ours.roster_rows(200000, found, curated=False, last_curated=196493)
    assert [r["tier"] for r in other] == ["model", "unreviewed", "unreviewed"]
    assert all(r["extrapolated"] for r in other)
    assert not any(r["extrapolated"] for r in held)
    assert {r["source"] for r in other} == {"ensemble"}


def test_roster_audit_measures_the_curated_interval_coverage():
    import pandas as pd

    frame = pd.DataFrame(
        {
            "shot": [1, 1, 2],
            "category": [1, 2, 1],
            "t_start": [0.0, 100.0, 0.0],
            "t_end": [100.0, 200.0, 100.0],
            "confidence": [0.9, 0.9, 0.4],
            "predicted": [1, 2, 1],
            "source": ["held_out", "held_out", "ensemble"],
            "tier": ["model", "model", "model"],
            "extrapolated": [False, False, False],
        }
    )
    # shot 1 is curated: H over 0-50 (half of the H segment) and nothing over L
    intervals = pd.DataFrame(
        {"shot": [1], "label": [1], "t_start": [0.0], "t_end": [50.0]}
    )
    out = ours.roster_audit(frame, {1}, intervals, None, last_curated=10)
    cur = out["curated_shots"]
    assert np.isclose(cur["share_inside_a_curated_interval"], 50 / 200)
    assert np.isclose(cur["by_class"]["H"]["share_inside_one_of_the_same_class"], 0.5)
    assert cur["by_class"]["L"]["share_inside_a_curated_interval"] == 0.0
    assert out["non_curated_shots"]["shots"] == 1
    assert out["non_curated_shots"]["by_predicted_class"]["H"]["segments"] == 1


def test_by_population_splits_the_scores_on_the_corpus_bes_shots(monkeypatch):
    monkeypatch.setattr(ours, "corpus_bes_shots", lambda: {1, 2})
    rng = np.random.default_rng(0)
    ys, probs = {}, {}
    for shot in (1, 2, 3, 4, 5):
        y = rng.integers(0, 4, 300)
        ys[shot] = y.astype(np.int8)
        p = np.zeros((4, 300), dtype=np.float32)
        p[y, np.arange(300)] = 1.0  # a perfect prediction
        probs[shot] = p
    got = ours.by_population(probs, ys)
    assert got["corpus_shots"]["shots"] == 2 and got["other_shots"]["shots"] == 3
    assert got["corpus_shots"]["windows"] == 600
    assert got["other_shots"]["macro_f1"] == 1.0
