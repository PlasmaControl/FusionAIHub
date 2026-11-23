"""Paired gaps, verdicts, collapse and the cost of a missing channel (plan 1c)."""

import json
import re

import numpy as np
import pandas as pd
import pytest

from tokamak_foundation_model.ignite.mast import cache, registry, tasks, transform
from tokamak_foundation_model.ignite.mast.tokenizers import assign, codecs, fit, grid
from tokamak_foundation_model.ignite.mast.tokenizers.learned import compare, identity, recode, store

from . import test_lrn_command as lc

RNG = np.random.default_rng(0)
SPEC = tasks.SignalSpec("m", 0.00025, (3,), 0.0, 1.0)


def window(n=6):
    w = RNG.standard_normal((n, 3, 100)).cumsum(axis=-1) * 0.1
    return w + 0.5 * np.roll(w, 1, axis=1)


def codec(concat, n_keep):
    shape = (3,)
    st = transform.ChannelStats(compress="none", scale=np.ones(shape), mean=np.zeros(shape), std=np.ones(shape))
    front = codecs.FrameDCT("m", 0.00025, shape, st, 16)
    fv = front.encode(window(40)).reshape(-1, 3, 16)
    norm = store.fit_norm(fv)
    e = norm.apply(fv)
    r = recode.fit_linear(e.reshape(len(e), -1), 24, concat=True, n_val=4) if concat else recode.fit_linear(e.reshape(-1, 16), 16)
    return recode.RecodedCodec(front, norm, r, np.arange(3), (n_keep,))


def test_the_gap_is_paired_over_the_shots_both_sides_have():
    rng = np.random.default_rng(1)
    b = {s: float(v) for s, v in enumerate(rng.uniform(0.05, 0.15, 100))}
    a = {s: 0.8 * v for s, v in b.items()}
    a[100] = 0.01                                        # a shot only one side has
    del a[0]
    a[1] = float("nan")
    g = compare.paired_gap(a, b)
    assert g["n_shots"] == 98 and g["gap"] == pytest.approx(0.2) and g["lo"] == pytest.approx(0.2) == g["hi"]
    assert compare.paired_gap(a, b) == g                 # seeded
    assert compare.beats(g) and compare.clearly_beats(g) and not compare.beats(compare.paired_gap(b, a))
    assert not compare.clearly_beats(g, margin=0.25) and not compare.beats(g, seed_spread=0.3)
    assert compare.word(g, compare.paired_gap(b, a)) == "a clearly beats b"
    assert compare.word(compare.paired_gap(b, a), g) == "b clearly beats a"
    swing = {s: v * (1.2 if s % 2 else 0.8) for s, v in b.items()}
    n = compare.paired_gap(swing, b)
    assert n["lo"] < 0 < n["hi"] and not compare.beats(n) and compare.word(n, compare.paired_gap(b, swing)) == "tie"
    one = compare.paired_gap({1: 0.1}, {1: 0.2})
    assert one["n_shots"] == 1 and np.isnan(one["lo"]) and not compare.beats(one) and compare.word(one, one) == "no data"


def test_per_shot_errors_are_read_from_the_evaluators_table(tmp_path):
    (tmp_path / "task_x").mkdir()
    pd.DataFrame({"shot_id": [1, 1, 2], "feature_name": ["a", "b", "a"], "n_windows": 5,
                  "NRMSE_mean": [0.1, 0.2, 0.3]}).to_csv(tmp_path / "task_x" / "shots_metrics.csv", index=False)
    assert compare.per_shot_harness(tmp_path, "task_x", "a") == {1: 0.1, 2: 0.3}
    assert compare.per_shot_harness(tmp_path, "task_y", "a") == {}


def test_per_shot_cache_errors_skip_a_shot_without_the_signal():
    c = codec(False, 4)
    shots = [(30000, window(1)[0].astype(np.float32)), (30001, None), (30002, np.full((3, 100), np.nan, dtype=np.float32))]
    got = compare.per_shot_cache(c, SPEC, shots)
    assert list(got) == [30000] and 0.0 < got[30000] < 1.0


def test_values_to_reach_and_the_seed_spread():
    curve = {20: 0.05, 40: 0.02, 80: 0.012, 160: 0.010}
    assert compare.values_to_reach(curve, 0.0176) == 80 and compare.values_to_reach(curve, 0.001) is None
    assert compare.seed_spread([0.12, 0.10, 0.15]) == pytest.approx(0.05) and np.isnan(compare.seed_spread([0.1]))
    assert compare.scaled_k(2, 0.5, 1, 16) == 1 and compare.scaled_k(2, 0.25, 1, 16) is None
    assert compare.scaled_k(80, 0.125, 4, 640) == 8 and compare.scaled_k(80, 2, 8, 320) == 160
    assert compare.scaled_k(16, 2, 1, 16) is None


def test_a_constant_first_value_is_a_collapse():
    v = np.random.default_rng(0).uniform(-1, 1, (4000, 3, 4))
    u = compare.usage(v)
    assert not u["collapsed"] and u["active_values"] == 4 and u["first_value_entropy"] > 0.95
    v[:, :, 3] = 0.01                                    # an unused last value is not a collapse
    u = compare.usage(v)
    assert not u["collapsed"] and u["active_values"] == 3
    v[:, 1, 0] = -0.3                                    # token 1 never moves its first value
    v[5] = np.nan
    u = compare.usage(v)
    assert u["collapsed"] and u["first_value_entropy"] == 0.0
    assert compare.level_index(np.array([-1.0, -0.99, 0.01, 1.0]), 8).tolist() == [0, 0, 4, 7]


def test_a_missing_channel_costs_the_concatenated_codec_and_not_the_per_channel_one():
    shots = [w.astype(np.float32) for w in window(8)]
    per = [compare.missing_stream_cost(codec(False, 4), SPEC, z, 1) for z in shots[:1]]
    assert per[0][0] == pytest.approx(per[0][1], abs=1e-6)
    cat = codec(True, 12)
    costs = np.array([compare.missing_stream_cost(cat, SPEC, z, 1) for z in shots])
    assert costs[:, 1].mean() > costs[:, 0].mean()
    assert compare.missing_stream_cost(cat, SPEC, np.full((3, 100), np.nan, dtype=np.float32), 1) is None


def test_left_over_of_a_low_rank_signal():
    q, _ = np.linalg.qr(RNG.standard_normal((16, 16)))
    e = (RNG.standard_normal((8000, 3)) * [3.0, 2.0, 1.0]) @ q[:, :3].T + 0.01 * RNG.standard_normal((8000, 16))
    left = compare.left_over(e[:4000], e[4000:], [1, 3, 16])
    assert left[0] == pytest.approx(5.0 / 14.0, rel=0.1) and left[1] < 1e-3 and left[2] < 1e-8


# ------------------------------------------------------------ the rest of the library, beyond the plan's tests

def test_left_over_fits_the_rows_with_a_missing_stream_as_zeros():
    """Ruling C23: a NaN of a training row is 0.0 in the fit, as in a concatenated recoder."""
    rng = np.random.default_rng(5)
    mix = rng.standard_normal((3, 12))
    train = (rng.standard_normal((600, 3)) * [3.0, 2.0, 1.0]) @ mix + 0.05 * rng.standard_normal((600, 12))
    val = (rng.standard_normal((300, 3)) * [3.0, 2.0, 1.0]) @ mix + 0.05 * rng.standard_normal((300, 12))
    holed = train.copy()
    for row in np.flatnonzero(rng.random(600) < 0.5):
        stream = rng.integers(0, 3)
        holed[row, 4 * stream:4 * stream + 4] = np.nan                       # a whole stream of 4 values
    zeros = np.where(np.isfinite(holed), holed, 0.0)
    sizes = [1, 2, 3, 6, 12]
    assert np.isnan(holed).any(axis=1).mean() > 0.4
    assert compare.left_over(holed, val, sizes) == pytest.approx(compare.left_over(zeros, val, sizes), abs=1e-12)
    pairwise = codecs.fit_pca(holed, 12)[1].T                                 # the hashed pairwise-complete fit
    centred = val - codecs.fit_pca(holed, 12)[0]
    pairwise_left = [float(((centred - centred @ pairwise[:, :k] @ pairwise[:, :k].T) ** 2).sum() / (centred ** 2).sum())
                     for k in sizes]
    assert max(abs(a - b) for a, b in zip(pairwise_left, compare.left_over(holed, val, sizes))) > 1e-3   # the test sees C23
    rows_with_nan = val.copy()
    rows_with_nan[:10, 0] = np.nan                                            # a validation row with a NaN is dropped
    assert compare.left_over(zeros, rows_with_nan, [3]) == pytest.approx(compare.left_over(zeros, val[10:], [3]))
    assert np.isnan(compare.left_over(zeros, np.full((5, 12), np.nan), [3])).all()


def test_masked_left_over_counts_the_observed_entries_and_agrees_on_complete_rows():
    rng = np.random.default_rng(8)
    mix = rng.standard_normal((2, 8))
    train = rng.standard_normal((400, 2)) @ mix + 0.05 * rng.standard_normal((400, 8))
    val = rng.standard_normal((100, 2)) @ mix + 0.05 * rng.standard_normal((100, 8))
    drop, total = compare.left_over_sums(train, val, [1, 2, 8])
    masked, total_masked = compare.left_over_sums(train, val, [1, 2, 8], masked=True)
    assert masked == pytest.approx(drop, abs=1e-9) and total_masked == pytest.approx(total)
    holed = val.copy()
    holed[:30, 4:] = np.nan                                                   # half of each example missing in 30 rows
    assert compare.left_over(train, holed, [1]) == pytest.approx(compare.left_over(train, val[30:], [1]))   # dropped
    got, got_total = compare.left_over_sums(train, holed, [1, 2], masked=True)
    r = recode.fit_linear(train, 2, concat=True)
    ok = np.isfinite(holed)
    filled = np.where(ok, holed, 0.0)
    want = [float((np.where(ok, filled - ((filled - r.mean) @ r.basis[:, :k] @ r.basis[:, :k].T + r.mean), 0.0) ** 2).sum())
            for k in (1, 2)]
    assert got == pytest.approx(want) and got_total == pytest.approx(float((np.where(ok, filled - r.mean, 0.0) ** 2).sum()))
    assert got_total < total_masked                                            # fewer entries are counted
    nothing, none_total = compare.left_over_sums(train, np.full((5, 8), np.nan), [1], masked=True)
    assert nothing == [0.0] and none_total == 0.0


def test_a_missing_channel_that_the_others_depend_on_costs_the_concatenated_codec():
    """Near-copies of one signal: without a channel the concatenated codec codes the other two worse."""
    rng = np.random.default_rng(3)

    def shots(n):
        common = rng.standard_normal((n, 1, 100)).cumsum(axis=-1) * 0.1
        return (common + 0.05 * rng.standard_normal((n, 3, 100))).astype(np.float32)

    front = codecs.FrameDCT("m", 0.00025, (3,), transform.ChannelStats(compress="none", scale=np.ones(3), mean=np.zeros(3),
                                                                         std=np.ones(3)), 16)
    fv = front.encode(shots(60)).reshape(-1, 3, 16)
    norm = store.fit_norm(fv)
    r = recode.fit_linear(norm.apply(fv).reshape(len(fv), -1), 24, concat=True, n_val=4)
    cat = recode.RecodedCodec(front, norm, r, np.arange(3), (12,))
    per = recode.RecodedCodec(front, norm, recode.fit_linear(norm.apply(fv).reshape(-1, 16), 16), np.arange(3), (4,))
    cost = np.array([compare.missing_stream_cost(cat, SPEC, z, 1) for z in shots(12)])
    assert (cost[:, 1] > 1.5 * cost[:, 0]).all()
    same = np.array([compare.missing_stream_cost(per, SPEC, z, 1) for z in shots(12)])
    assert same[:, 1] == pytest.approx(same[:, 0], abs=1e-6)
    with pytest.raises(ValueError, match="channel"):
        compare.missing_stream_cost(cat, SPEC, shots(1)[0], 3)


def test_a_verdict_needs_a_known_spread_and_no_data_is_no_verdict():
    b = {s: 0.1 + 0.001 * s for s in range(50)}
    a = {s: 0.7 * v for s, v in b.items()}
    g, h = compare.paired_gap(a, b), compare.paired_gap(b, a)
    assert compare.word(g, h, seed_spread=0.05) == "a clearly beats b" and compare.word(g, h, seed_spread=0.5) == "tie"
    assert compare.word(g, h, seed_spread=float("nan")) == "tie"                  # an unknown spread proves nothing
    assert np.isnan(compare.seed_spread([0.1, float("nan"), 0.2]))
    empty = compare.paired_gap({1: float("nan")}, {2: 0.1})
    assert empty["n_shots"] == 0 and np.isnan(empty["mean_a"]) and np.isnan(empty["gap"]) and compare.word(empty, empty) == "no data"
    zero = compare.paired_gap({1: 0.0, 2: 0.0}, {1: 0.0, 2: 0.0})                  # no error on either side: no gap
    assert zero["n_shots"] == 2 and np.isnan(zero["gap"]) and np.isnan(zero["lo"])
    assert compare.values_to_reach({10: float("nan"), 20: 0.1}, 0.5) == 20


def test_usage_refuses_values_it_cannot_judge():
    with pytest.raises(ValueError, match="no example"):
        compare.usage(np.full((4, 2, 3), np.nan))
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        compare.usage(np.full((4, 2, 3), 1.5))
    with pytest.raises(ValueError, match="M, n_tok, n_val"):
        compare.usage(np.zeros((4, 3)))
    u = compare.usage(np.random.default_rng(0).uniform(-1, 1, (500, 2, 3)))
    assert u["n_examples"] == 500 and u["active_values"] == 3


# ------------------------------------------------------------------------ `compare` and `seeds`, end to end

ARR = "magnetics-b_field_pol_probe_ccbv_field"
ARR_SPEC = tasks.SignalSpec(ARR, 0.00025, (3,), 0.0, 1.0)
ARR_STATS = transform.ChannelStats(compress="none", scale=np.ones(3), mean=np.zeros(3), std=np.ones(3))


def array_world(tmp_path, monkeypatch):
    """Five shots of a signal of three near-identical channels (20 frames of 16 values), a fit on shots 1-3, the
    assignment ``v1`` (frame_dct, K = 2: 6 values per frame), a concatenated recoder ``pca.arr`` (8 tokens of 6) and
    the assignment ``rc`` behind it (k = 6). The trial root is `tmp_path`."""
    monkeypatch.setenv("IGNITE_MAST_ROOT", str(tmp_path))
    root = tmp_path / "cache" / "v1"
    (root / "shots").mkdir(parents=True)
    transform.save_stats({ARR: ARR_STATS}, root / "stats.npz")
    sha = transform.stats_sha256(root / "stats.npz")
    (root / "manifest.json").write_text(json.dumps({"stats_sha256": sha, "tokamark_revision": "146b0a0f8738"}))
    reg = pd.DataFrame([{"device": "mast", "signal": ARR, "channel": c, "channel_name": f"c{c}", "family": "slow",
                         "dt": ARR_SPEC.dt, "units": "T", "imas_path": "", "r": 1.0 + c, "z": 0.1 * c, "tm_mean": 0.0,
                         "tm_std": 1.0, "roles": "inp+out", "sat_abs": np.nan, "valid_fraction": 1.0,
                         "dead_fraction": 0.0, "compress": "none", "scale": 1.0, "mean": 0.0, "std": 1.0}
                        for c in range(3)])
    registry.save_registry(reg, root / "registry.csv")
    zs = {}
    for shot in (1, 2, 3, 4, 5):
        rng = np.random.default_rng(shot)
        z = (rng.standard_normal((1, 400)).cumsum(axis=-1) * 0.2 + 0.05 * rng.standard_normal((3, 400))).astype(np.float32)
        zs[shot] = z
        raw = {"time": 0.1 + ARR_SPEC.dt * np.arange(400), "values": z.astype(np.float64)}
        cache.write_shot(root / "shots" / f"{shot}.h5", shot, "train", {ARR: raw}, [ARR_SPEC], {ARR: ARR_STATS}, reg, stats_sha=sha)
    fits = {ARR: fit.fit_signal(ARR_SPEC, ARR_STATS, [zs[s] for s in (1, 2, 3)], kind="frame_dct")}
    fit.save_fit(fits, tmp_path / "stage1" / "fit" / "v1", {"stats_sha256": sha, "shots": [1, 2, 3]})
    assigns = tmp_path / "stage1" / "assignments"
    assign.save_assignment({ARR: assign.Setting("frame_dct", (2,))}, assigns / "v1.json")
    front = store.front_codec(ARR_SPEC, ARR_STATS, fits[ARR], "frame_dct", "linear")
    fv = np.concatenate([store.stream_values(front, zs[s]) for s in (1, 2, 3)])           # (60, 3, 16)
    norm = store.fit_norm(fv)
    e = norm.apply(fv).reshape(len(fv), 48)
    r = recode.fit_linear(e, 48, concat=True, n_val=6)
    ranges = grid.fit_ranges(r.encode(e, np.full(len(e), -1, dtype=np.int64)).reshape(-1, 1, 48))
    table = identity.StreamTable({ARR: (0, 3)}, np.zeros((3, identity.N_FEAT), dtype=np.float32))
    meta = {"signals": {ARR: {"kind": "frame_dct", "domain": "linear", "n_streams": 3, "values_per_token": 1}}}
    recode.save_recoder(tmp_path / "stage1" / "recoders" / "pca.arr", r, {ARR: norm}, table, meta, {ARR: ranges})
    assign.save_assignment({ARR: assign.Setting("recoded_concat", (6,), "linear", source="pca.arr")}, assigns / "rc.json")
    return root


def compare_script(monkeypatch, spec=ARR_SPEC, predicted=()):
    rt = lc.load("run_tokenizer")
    monkeypatch.setattr(rt.registry, "all_signals", lambda: [spec])
    monkeypatch.setattr(rt.tasks, "split_shots", lambda split: [4, 5] if split == "val_select" else [1, 2, 3])
    monkeypatch.setattr(rt.assign, "predicted_by", lambda: {spec.name: tuple(predicted)})
    monkeypatch.setattr(rt.tasks, "best_published", lambda task: {"task_big": 0.5, "task_small": 0.2}[task])
    return rt


def compare_argv(root, *extra, a="v1", b="rc", label="val_select-n2"):
    return ["compare", "--a", a, "--b", b, "--label", label, "--workers", "1", "--cache-dir", str(root), *extra]


def test_compare_reads_the_cache_for_a_signal_no_task_predicts(tmp_path, monkeypatch, capsys):
    root = array_world(tmp_path, monkeypatch)
    rt = compare_script(monkeypatch)
    assert rt.main(compare_argv(root)) == 0
    out = capsys.readouterr().out
    assert "not scored" not in out and f"wrote {tmp_path / 'stage1' / 'compare' / 'v1__rc.val_select-n2.csv'}" in out
    frame = pd.read_csv(tmp_path / "stage1" / "compare" / "v1__rc.val_select-n2.csv")
    assert list(frame.columns) == ["signal", "source", "values_a", "values_b", "n_shots", "mean_a", "mean_b", "gap", "lo",
                                   "hi", "gap_ba", "lo_ba", "hi_ba", "one_seed", "drop_a", "drop_b", "reach_a", "reach_b"]
    row = frame.iloc[0]
    assert row["signal"] == ARR and row["source"] == "cache" and row["values_a"] == row["values_b"] == 6 and row["n_shots"] == 2
    assert np.isfinite(row["mean_a"]) and np.isfinite(row["mean_b"]) and row["mean_a"] > 0
    assert row["gap"] == pytest.approx((row["mean_b"] - row["mean_a"]) / row["mean_b"])
    assert row[["drop_a", "drop_b", "reach_a", "reach_b"]].isna().all()                  # asked for by flags only
    expected = compare.per_shot_cache(
        rt.learned_build.build_codec(ARR_SPEC, ARR_STATS, fit.load_fit(tmp_path / "stage1" / "fit" / "v1")[0][ARR],
                                     assign.Setting("frame_dct", (2,)), tmp_path / "stage1" / "recoders"), ARR_SPEC,
        [(s, cache.read_shot(root / "shots" / f"{s}.h5", [ARR])[ARR].values) for s in (4, 5)])
    assert row["mean_a"] == pytest.approx(np.mean(list(expected.values())), rel=1e-5)     # the same number, by another path


def test_compare_reports_a_missing_channel_and_the_values_to_reach(tmp_path, monkeypatch, capsys):
    root = array_world(tmp_path, monkeypatch)
    rt = compare_script(monkeypatch)
    assert rt.main(compare_argv(root, "--drop-channel", "--reach", "1000000")) == 0
    row = pd.read_csv(tmp_path / "stage1" / "compare" / "v1__rc.val_select-n2.csv").iloc[0]
    assert row["drop_a"] == pytest.approx(1.0, abs=1e-6)                                  # per channel: nothing changes
    assert row["drop_b"] > 1.2                                                            # concatenated: the others pay
    assert row["reach_a"] == 3 and row["reach_b"] == 6        # the smallest size that exists: 1 of 16 per channel, 1 token of 6
    assert rt.main(compare_argv(root, "--reach", "0.0000000001", label="val_select-n2")) == 0
    row = pd.read_csv(tmp_path / "stage1" / "compare" / "v1__rc.val_select-n2.csv").iloc[0]
    assert np.isnan(row["reach_a"]) and np.isnan(row["reach_b"]) and np.isnan(row["drop_a"])   # never reached: empty, rewritten


def test_compare_reads_stored_floors_only_if_kept_and_says_what_is_missing(tmp_path, monkeypatch, capsys):
    root, _, fits, zs, _ = lc.toy_world(tmp_path, monkeypatch)
    lc.save_toy_recoder(tmp_path, fits, zs)
    assign.save_assignment({"summary-ip": assign.Setting("recoded", (4,), "linear", source="klt.ip")},
                           tmp_path / "stage1" / "assignments" / "r1.json")
    rt = compare_script(monkeypatch, lc.IP, predicted=("task_big", "task_small"))
    kept = {"tok.v1.unquantized", "tok.r1.unquantized"}
    monkeypatch.setattr(rt, "kept_score", lambda out_dir, task, predictor, label, prov, force: (
        ({"nrmse": 0.1}, "") if predictor in kept else (None, "no score.json")))
    scores = tmp_path / "stage1" / "scores" / "task_small"
    metrics = pd.DataFrame({"shot_id": [4, 5, 4], "feature_name": ["summary-ip", "summary-ip", "other"], "n_windows": 3,
                            "NRMSE_mean": [0.10, 0.20, 9.0]})
    for predictor, scale in (("tok.v1.unquantized", 1.0), ("tok.r1.unquantized", 0.8)):
        (scores / predictor / "val_select-n2" / "task_small").mkdir(parents=True)
        metrics.assign(NRMSE_mean=metrics["NRMSE_mean"] * scale).to_csv(
            scores / predictor / "val_select-n2" / "task_small" / "shots_metrics.csv", index=False)
    assert rt.main(compare_argv(root, b="r1")) == 0                          # the task with the smaller best score: task_small
    row = pd.read_csv(tmp_path / "stage1" / "compare" / "v1__r1.val_select-n2.csv").iloc[0]
    assert row["source"] == "task_small" and row["n_shots"] == 2 and row["mean_a"] == pytest.approx(0.15)
    assert row["mean_b"] == pytest.approx(0.12) and row["gap"] == pytest.approx(-0.25) and row["gap_ba"] == pytest.approx(0.2)
    assert row["one_seed"] == "b clearly beats a"
    kept.clear()
    capsys.readouterr()
    assert rt.main(compare_argv(root, b="r1")) == 0
    out = capsys.readouterr().out
    row = pd.read_csv(tmp_path / "stage1" / "compare" / "v1__r1.val_select-n2.csv").iloc[0]
    assert row["one_seed"] == "not scored" and np.isnan(row["mean_a"]) and np.isnan(row["gap"])
    assert row["values_a"] == 4 and row["values_b"] == 4
    for name in ("v1", "r1"):
        assert (f"score --task task_small --split val_select --n-shots 2 --assignment "
                f"{tmp_path / 'stage1' / 'assignments' / (name + '.json')} --mode unquantized") in out


def test_compare_refuses_before_it_reads_or_writes(tmp_path, monkeypatch, capsys):
    root = array_world(tmp_path, monkeypatch)
    rt = compare_script(monkeypatch)
    monkeypatch.setattr(rt.assign, "predicted_by", lambda: pytest.fail("a refusal must come before the scores are read"))
    for argv, match in ((compare_argv(root, b="nope"), "there is no assignment"),
                        (compare_argv(root, a="rc", b="rc"), "same assignment"),
                        (compare_argv(root, "--signals", "no-such-signal"), "not signals of the registry"),
                        (compare_argv(root, label="test"), "not <split>"),
                        (compare_argv(root, "--reach", "0"), "positive"),
                        (compare_argv(root, "--reach", "nan"), "positive"),
                        (compare_argv(root, a="../v1"), "not a name"),
                        ([*compare_argv(root), "--workers", "0"], "at least 1")):
        assert rt.main(argv) == 2, argv
        assert match in capsys.readouterr().err
    assert not (tmp_path / "stage1" / "compare").exists()
    assign.save_assignment({ARR: assign.Setting("frame_dct", (2,))}, tmp_path / "stage1" / "assignments" / "v1b.json")
    assert rt.main(compare_argv(root, b="v1b")) == 2 and "same setting for every signal" in capsys.readouterr().err
    monkeypatch.setattr(rt.assign, "predicted_by", lambda: {ARR: ()})
    assert rt.main([*compare_argv(root, b="v1b"), "--signals", ARR]) == 0            # named: compared anyway
    row = pd.read_csv(tmp_path / "stage1" / "compare" / "v1__v1b.val_select-n2.csv").iloc[0]
    assert row["gap"] == 0 and row["one_seed"] == "tie"                                # the same codec twice


def write_comparison(tmp_path, a, b, label, gap, lo, hi, signal="sig"):
    out = tmp_path / "stage1" / "compare"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"signal": signal, "source": "cache", "values_a": 8, "values_b": 8, "n_shots": 100, "mean_a": 0.1,
                   "mean_b": 0.1 / (1 - gap), "gap": gap, "lo": lo, "hi": hi, "gap_ba": -gap, "lo_ba": -hi, "hi_ba": -lo,
                   "one_seed": "x"}]).to_csv(out / f"{a}__{b}.{label}.csv", index=False)


def test_seeds_judges_the_first_gap_against_the_spread_of_all(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("IGNITE_MAST_ROOT", str(tmp_path))
    rt = lc.load("run_tokenizer")
    for name, gap in (("n0", 0.25), ("n1", 0.20), ("n2", 0.28)):
        write_comparison(tmp_path, name, "base", "L", gap, gap - 0.05, gap + 0.05)
    assert rt.main(["seeds", "--pairs", "n0:base,n1:base,n2:base", "--label", "L"]) == 0
    line = capsys.readouterr().out.strip()
    assert line == "sig: gaps 0.2500, 0.2000, 0.2800; spread 0.0800; a clearly beats b"
    write_comparison(tmp_path, "n3", "base", "L", -0.10, -0.15, -0.05)                    # a seed that loses: spread 0.35
    assert rt.main(["seeds", "--pairs", "n0:base,n3:base", "--label", "L"]) == 0
    assert capsys.readouterr().out.strip() == "sig: gaps 0.2500, -0.1000; spread 0.3500; tie"
    assert rt.main(["seeds", "--pairs", "n0:base,n9:base", "--label", "L"]) == 2
    err = capsys.readouterr().err
    assert "n9__base.L.csv" in err and "n0__base" not in err
    assert rt.main(["seeds", "--pairs", "n0:base", "--label", "L"]) == 2 and "at least two" in capsys.readouterr().err
    assert rt.main(["seeds", "--pairs", "n0:base,n1", "--label", "L"]) == 2
    write_comparison(tmp_path, "n4", "base", "L", float("nan"), float("nan"), float("nan"))
    assert rt.main(["seeds", "--pairs", "n0:base,n4:base", "--label", "L"]) == 0
    assert capsys.readouterr().out.strip() == "sig: gaps 0.2500, nan; spread nan; no data"


# ------------------------------------------------------------------------------ `joint`, on stored examples

def joint_world(tmp_path, monkeypatch, noise=1e-3, dead=False):
    """Two signals of 3 and 2 streams of 4 values whose joint example is one direction times a number per frame
    (plus a little noise), over 3 train shots and 2 validation shots of 20 frames; the second signal has 18 of them."""
    monkeypatch.setenv("IGNITE_MAST_ROOT", str(tmp_path))
    names, streams = ("sig-a", "sig-b"), (3, 2)
    direction = np.random.default_rng(11).standard_normal((sum(streams), 4))
    for key, shot_ids in (("train", (1, 2, 3)), ("val", (4, 5))):
        rng = np.random.default_rng(len(key))
        stores = {}
        for i, (n, s) in enumerate(zip(names, streams)):
            shots = np.repeat(np.array(shot_ids, dtype=np.int64), 20)
            level = np.random.default_rng(100 + len(key)).standard_normal(len(shots))          # one number per frame
            lo = sum(streams[:i])
            v = (level[:, None, None] * direction[None, lo:lo + s] + noise * rng.standard_normal((len(shots), s, 4))).astype(np.float32)
            if i == 1:
                keep = np.ones(len(shots), dtype=bool)
                keep[18::20] = keep[19::20] = False                                             # the last 2 frames of each shot
                v, shots = v[keep], shots[keep]
            if dead and key == "val" and i == 0:
                v[:, 1] = np.nan                                                                # a probe dead in every frame
            stores[n] = store.Store(n, v, shots)
        meta = {"signals": {n: {"kind": "frame_dct", "domain": "linear", "n_streams": s, "values_per_token": 1}
                            for n, s in zip(names, streams)}}
        store.save_store(stores, tmp_path / "stage1" / "store" / "v1" / key, meta)
    return names


def test_joint_measures_the_joint_the_own_and_the_pooled_basis(tmp_path, monkeypatch, capsys):
    names = joint_world(tmp_path, monkeypatch)
    tt = lc.load("train_tokenizer")
    capsys.readouterr()
    assert tt.main(["joint", "--signals", ",".join(names), "--sizes", "1,2,4"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].startswith("joint sig-a, sig-b: 5 streams of 4 values; training frames 54, validation frames 36 with 180 streams observed of 180")
    shares = []
    for line, k in zip(lines[1:], (1, 2, 4)):
        assert line.startswith(f"{k} values per stream ({5 * k} in all): left over joint ")
        shares.append([float(v) for v in re.findall(r"[0-9]\.[0-9]{4}", line.split("left over")[1])])
    assert len(lines) == 4 and all(len(r) == 3 for r in shares)
    assert shares[0][0] < 1e-4 and shares[0][1] < 1e-4              # one direction: the joint and each own basis hold it
    assert shares[0][2] > 0.05 and shares[2][2] < 1e-6              # a pooled per-stream basis does not at 1, and does at 4
    assert shares[0][2] > shares[1][2] > shares[2][2]
    assert tt.main(["joint", "--signals", "sig-a,sig-b", "--sizes", "5"]) == 2 and "4" in capsys.readouterr().err


def test_joint_has_data_when_a_probe_is_dead_in_every_validation_frame(tmp_path, monkeypatch, capsys):
    """On real stores some probe is dead in each shot, so no frame has every stream finite: the missing stream is
    encoded as 0 and counts in neither sum, as `linear` does."""
    names = joint_world(tmp_path, monkeypatch, dead=True)
    tt = lc.load("train_tokenizer")
    capsys.readouterr()
    assert tt.main(["joint", "--signals", ",".join(names), "--sizes", "1,4"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert "36 with 144 streams observed of 180" in lines[0] and len(lines) == 3 and "no data" not in "".join(lines)
    one, four = ([float(v) for v in re.findall(r"[0-9]\.[0-9]{4}", ln.split("left over")[1])] for ln in lines[1:])
    assert len(one) == len(four) == 3 and all(0 <= x <= 1 for x in (*one, *four))     # joint, own concatenated, pooled
    assert one[0] < one[2] and four[2] < 1e-6        # the pooled basis holds a stream at 4 values, and does not at 1


def test_joint_matches_the_frames_of_the_signals_by_shot_and_index(tmp_path, monkeypatch, capsys):
    tt = lc.load("train_tokenizer")
    assert tt._frame_keys(np.array([5, 5, 5, 7, 7], dtype=np.int64)).tolist() == [5 << 20, (5 << 20) + 1, (5 << 20) + 2,
                                                                                7 << 20, (7 << 20) + 1]
    names = joint_world(tmp_path, monkeypatch)
    train, _ = store.load_store(tmp_path / "stage1" / "store" / "v1" / "train")
    both = tt._common_frames({n: train[n].values for n in names}, train)
    assert both.shape == (54, 5 * 4)                                # sig-b has 18 frames of each shot: 3 x 18 in both
    assert np.array_equal(both[0, :12], train[names[0]].values[0].reshape(-1))
    assert np.array_equal(both[0, 12:], train[names[1]].values[0].reshape(-1))
    assert np.array_equal(both[18, :12], train[names[0]].values[20].reshape(-1))    # the first frame of shot 2 in each store
    assert np.array_equal(both[18, 12:], train[names[1]].values[18].reshape(-1))


def test_joint_refuses_what_it_cannot_measure(tmp_path, monkeypatch, capsys):
    tt = lc.load("train_tokenizer")
    monkeypatch.setenv("IGNITE_MAST_ROOT", str(tmp_path))
    assert tt.main(["joint", "--signals", "ccbv"]) == 2 and "at least two" in capsys.readouterr().err
    assert tt.main(["joint", "--signals", "ccbv,ccbv"]) == 2 and "twice" in capsys.readouterr().err
    assert tt.main(["joint", "--signals", "ccbv,obr", "--sizes", "0"]) == 2
    assert tt.main(["joint", "--signals", "ccbv,obr", "--sizes", "x"]) == 2
    assert tt.main(["joint", "--signals", "ccbv,obr"]) == 2 and "no store" in capsys.readouterr().err
    names = joint_world(tmp_path, monkeypatch)
    assert tt.main(["joint", "--signals", f"{names[0]},nope"]) == 2 and "do not hold" in capsys.readouterr().err
