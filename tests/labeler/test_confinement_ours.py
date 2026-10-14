"""The segmenting steps of ``scripts/labeler/confinement_ours.py`` on small cases, and
the helpers that score it on confine-cnn's protocol."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/labeler/confinement_ours.py"
spec = importlib.util.spec_from_file_location("confinement_ours", SCRIPT)
ours = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ours  # its dataclass looks itself up there
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


def test_write_review_table_reads_the_roster_and_leaves_it_alone(tmp_path):
    from labeler.events.interval_tables import parse_attrs, validate_intervals

    roster = pd.DataFrame(
        {
            "shot": [1, 1],
            "category": [1, 5],
            "t_start": [0.0, 500.0],
            "t_end": [500.0, 800.0],
            "confidence": [0.9, 0.6],
            "predicted": [1, 3],
            "source": ["ensemble", "held_out"],
            "tier": ["model", "unreviewed"],
            "extrapolated": [False, True],
        }
    )
    roster.to_csv(tmp_path / "roster.csv", index=False)
    meta = {
        "columns": list(roster.columns),
        "columns_meaning": {"attrs": "old", "tier": "what tier means"},
    }
    (tmp_path / "roster.meta.json").write_text(json.dumps(meta))
    names = ("roster.csv", "roster.meta.json")
    before = [(tmp_path / name).read_bytes() for name in names]
    table = ours.write_review_table(tmp_path)
    assert [(tmp_path / name).read_bytes() for name in names] == before
    written = pd.read_csv(tmp_path / "roster_review.csv")
    assert list(written.columns) == [*roster.columns[:5], "attrs"]
    assert len(validate_intervals(written)) == 2  # the page's reader accepts it
    assert parse_attrs(table["attrs"].iloc[1])["tier"] == "unreviewed"
    review_meta = json.loads((tmp_path / "roster_review.meta.json").read_text())
    assert review_meta["columns"] == list(written.columns)
    assert review_meta["columns_meaning"]["tier"] == "what tier means"
    assert "attrs" in review_meta["columns_meaning"] and "derived_from" in review_meta


# confine-ours on confine-cnn's protocol


def test_at_windows_reads_the_bin_holding_each_window_centre():
    probs = {7: np.arange(40, dtype=np.float32).reshape(4, 10)}
    windows = pd.DataFrame({"shot": [7, 7, 7, 8], "center_ms": [0.5, 3.99, 25.0, 1.0]})
    got = ours.at_windows(probs, windows)
    assert got[0].tolist() == probs[7][:, 0].tolist()
    assert got[1].tolist() == probs[7][:, 3].tolist()
    assert got[2].tolist() == probs[7][:, 9].tolist()  # past the end: the last bin
    assert np.isnan(got[3]).all()  # a shot with no probabilities


def test_protocol_bins_counts_the_labelled_bins_the_protocol_drops():
    ys = {7: np.array([0, 0, 1, 1, -1, 2])}
    windows = pd.DataFrame(
        {"shot": [7, 7, 7], "center_ms": [0.5, 2.5, 4.5], "label": [0, 1, 0]}
    )
    out = ours.protocol_bins(ys, windows)
    assert (out["labelled_bins"], out["scored_bins"], out["dropped_bins"]) == (5, 2, 3)
    assert out["windows_checked"] == 3
    assert out["windows_whose_label_is_ours"] == 2  # bin 4 is unlabelled


def test_protocol_scores_pairs_the_two_models_on_the_windows_ours_covers():
    rng = np.random.default_rng(0)
    n = 400
    windows = pd.DataFrame(
        {"shot": np.repeat(np.arange(10), 40), "label": np.tile(np.arange(4), 100)}
    )
    noise = rng.dirichlet(np.ones(4), size=n)
    for i, c in enumerate(["L", "H", "QH", "WP"]):
        windows[f"p_{c}"] = noise[:, i]
    mine = np.full((n, 4), 0.05, dtype=np.float32)
    mine[np.arange(n), windows.label] = 0.85
    mine[:40] = np.nan  # shot 0 not covered
    out = ours.protocol_scores(windows, mine, replicates=20)
    assert out["windows"] == 360 and out["shots"] == 9
    assert out["confine-ours"]["macro_f1"] == 1.0
    assert out["confine-cnn"]["windows"] == 360
    d = out["ours_minus_cnn"]
    assert d["macro_f1"]["ci95"][0] > 0 and d["auroc"]["difference"] > 0
    assert out["confine-ours"]["ranking"]["ci95"]["auroc"] == [1.0, 1.0]


def test_clean_folds_are_those_that_never_saw_the_run_day(monkeypatch):
    records = {
        0: {"train_shots_list": [1, 2], "val_shots_list": [3]},
        1: {"train_shots_list": [4], "val_shots_list": [5]},
    }
    monkeypatch.setattr(ours, "FOLDS", 2)
    monkeypatch.setattr(ours, "fold_record", lambda variant, fold: records[fold])
    days = {1: 10, 2: 10, 3: 11, 4: 12, 5: 13, 9: 11}
    assert ours.clean_folds(None, 11, days) == [1]  # fold 0 validated on day 11
    assert ours.clean_folds(None, 14, days) == [0, 1]


def test_fold_roles_keeps_the_recorded_test_folds_and_leaves_new_shots_out(
    monkeypatch,
):
    monkeypatch.setattr(
        ours, "deal_folds", lambda ids, ys, grouping: ({s: 0 for s in ids}, {}, {})
    )
    monkeypatch.setattr(ours, "recorded_test_folds", lambda variant: {1: 3, 2: 4})
    test, _, _ = ours.fold_roles([1, 2, 3], [None] * 3)
    assert test == {1: 3, 2: 4}
    monkeypatch.setattr(ours, "recorded_test_folds", lambda variant: None)
    assert ours.fold_roles([1, 2, 3], [None] * 3)[0] == {1: 0, 2: 0, 3: 0}


def test_network_input_appends_the_counter_beams_and_zeroes_d_alpha_for_the_twin(
    monkeypatch,
):
    grid = np.ones((len(ours.zerod.CHANNELS), 5), dtype=np.float32)
    counter = np.array([0.0, 1.0, np.nan, 2.0, 20.0], dtype=np.float32)
    monkeypatch.setattr(ours, "counter_channel", lambda shot, n: counter)
    twin = ours.Variant(Path("."), counter_beams=True, d_alpha=False)
    x = ours.network_input(grid, 1, twin)
    assert x.shape == (twin.channels, 5) == (ours.zerod.N_INPUT + 2, 5)
    assert (x[:3] == 0).all() and (x[3] == 1).all()
    assert x[-2].tolist() == [0.0, 1.0, 0.0, 2.0, 8.0]  # NaN to 0, clipped at 8
    assert x[-1].tolist() == [1.0, 1.0, 0.0, 1.0, 1.0]
    plain = ours.network_input(grid, 1, ours.VARIANTS["run_day"])
    assert plain.shape == (ours.zerod.N_INPUT, 5) and (plain[:3] == 1).all()
