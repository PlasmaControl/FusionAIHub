"""Hermetic checks of the `elm-ours` pieces: inputs, labels, scoring, onset, folds.

Synthetic records only; no data root, no GPU.
"""

import json

import numpy as np
import pandas as pd
import pytest

from labeler.elm import inputs, labels, methods, net, onset, score, train


def test_block_reduce_max_and_mean_with_empty_cell():
    t = inputs.GRID0_MS + np.array([0.01, 0.05, 0.25])
    y = np.array([[1.0, 3.0, 5.0]])
    mx, count = inputs.block_reduce(t, y, 4, "max")
    assert mx[0].tolist() == [3.0, 0.0, 5.0, 0.0]
    assert count.tolist() == [2, 0, 1, 0]
    mean, _ = inputs.block_reduce(t, y, 4, "mean")
    assert mean[0, 0] == 2.0


def _record(n_ms=300, gain=1.0, density=3e14):
    t_fs = np.arange(n_ms * 50) / 50.0
    rng = np.random.default_rng(0)
    fs = gain * 1e15 * (1 + 0.05 * rng.random((3, t_fs.size)))
    fs[:, 6000:6100] *= 5  # an ELM burst
    t_int = np.arange(n_ms * 100) / 100.0
    ne = np.full((2, t_int.size), density)
    return t_fs, fs, t_int, ne


def test_channels_shape_validity_and_gain_invariance_of_contrast():
    x = inputs.channels(*_record(gain=1.0))
    y = inputs.channels(*_record(gain=30.0))
    assert x.shape[0] == inputs.N_CHANNELS and x.dtype == np.float32
    assert x.shape[1] % inputs.CELLS_PER_MS == 0
    # nothing is sampled before the shot's clock reaches 0, 50 ms after the grid starts
    assert x[inputs.VALID][: 50 * inputs.CELLS_PER_MS].max() == 0.0
    assert x[inputs.VALID][60 * inputs.CELLS_PER_MS :].min() == 1.0
    # an ELM is an additive step in the log, so the contrast ignores the shot's gain
    c = list(inputs.FS_CONTRAST)
    assert np.allclose(x[c], y[c], atol=2e-3)
    assert not np.allclose(x[list(inputs.FS_LEVEL)], y[list(inputs.FS_LEVEL)])
    assert x[inputs.FS_CONTRAST[0]].max() > 0.2


def test_failed_chord_is_zeroed_and_gap_is_invalid():
    t_fs, fs, t_int, ne = _record()
    ne[1] = 1e18
    keep = (t_fs < 100) | (t_fs >= 120)
    x = inputs.channels(t_fs[keep], fs[:, keep], t_int, ne)
    assert not x[inputs.DENSITY[1]].any() and not x[inputs.DENSITY_HP[1]].any()
    assert x[inputs.DENSITY[0]].any()
    start, stop = inputs.valid_intervals(x[inputs.VALID])
    assert len(start) == 2 and stop[0] - start[0] >= 99
    assert start[1] - stop[0] == pytest.approx(20, abs=1.5)


def test_review_table_kinds(tmp_path):
    rows = [
        (1, 0, 100, 0, None),
        (1, 100, 120, 1, json.dumps({"iscrowd": 0})),
        (1, 120, 300, 1, json.dumps({"iscrowd": 1})),
        (1, 300, 400, 2, None),
        (1, 400, 500, 3, None),
    ]
    path = tmp_path / "labels.csv"
    pd.DataFrame(
        rows, columns=["shot", "t_start", "t_end", "category", "attrs"]
    ).to_csv(path, index=False)
    t = labels.review_table(path)
    assert t.kind.tolist() == [
        "absent",
        "non_crowd",
        "crowd",
        "uncertain",
        "not_observable",
    ]


def _table(rows):
    return pd.DataFrame(rows, columns=["t_start", "t_end", "kind"])


def test_dense_targets_ignore_uncertain_and_mask_onsets_in_crowds():
    spans = _table(
        [
            (0.0, 100.0, "absent"),
            (100.0, 110.0, "non_crowd"),
            (110.0, 200.0, "uncertain"),
            (200.0, 300.0, "crowd"),
        ]
    )
    d = labels.dense(spans, 400)
    g0 = inputs.GRID0_MS
    ms = lambda t: int(t - g0)
    assert (d.state[ms(0) : ms(100)] == 0).all()
    assert (d.state[ms(100) : ms(110)] == 1).all()
    assert (d.state[ms(110) : ms(200)] == labels.IGNORE).all()
    assert (d.state[ms(200) : ms(300)] == 1).all()
    assert d.onset_mask[ms(50)] and not d.onset_mask[ms(250)]
    assert d.onset[ms(100) - 1 : ms(100) + 1].max() > 0.9
    assert d.onset[~d.onset_mask].max() == 0.0
    assert d.window == (0.0, 300.0)


def test_scored_bins_follow_the_elmo_rule():
    spans = _table(
        [
            (0.0, 120.0, "absent"),
            (120.0, 130.0, "non_crowd"),  # too short for a whole bin
            (130.0, 400.0, "crowd"),
            (400.0, 600.0, "uncertain"),
            (600.0, 800.0, "absent"),  # under half analysed
        ]
    )
    # the last span has exactly half its time analysed: it keeps the bins in [700, 800)
    b = labels.scored_bins(spans, np.array([0.0, 700.0]), np.array([500.0, 800.0]))
    assert b.t0[b.kind == "absent"].tolist() == [0.0, 50.0, 700.0, 750.0]
    assert b.t0[b.kind == "crowd"].tolist() == [150.0, 200.0, 250.0, 300.0, 350.0]
    assert (b.truth[b.kind == "crowd"] == 1).all()
    assert (b.truth[b.kind == "absent"] == 0).all()
    # with nothing analysed in 600-800 the span has no bins at all
    b = labels.scored_bins(spans, np.array([0.0]), np.array([500.0]))
    assert b.t0.max() < 600


def test_bin_scores_mean_and_hard_hits():
    n = 400
    fine = np.zeros(n)
    fine[60:100] = 1.0
    bins = labels.Bins(
        np.array([inputs.GRID0_MS + 50.0, inputs.GRID0_MS + 100.0]),
        np.array([1, 0], dtype=np.int8),
        np.array(["crowd", "absent"], dtype=object),
        np.array([0, 1]),
    )
    s = labels.bin_scores(fine, bins)
    assert s.tolist() == pytest.approx([0.8, 0.0])
    hits = labels.hard_hits(
        np.array([10.0]),
        np.array([60.0]),
        labels.Bins(
            np.array([0.0, 50.0, 100.0]),
            np.zeros(3, np.int8),
            np.array(["absent"] * 3, object),
            np.zeros(3, int),
        ),
    )
    assert hits.tolist() == [True, True, False]


def test_runs_of_the_moving_mean():
    fine = np.zeros(300)
    fine[100:200] = 1.0
    starts, stops = labels.runs_of(fine, 0.5)
    g0 = inputs.GRID0_MS
    assert len(starts) == 1
    assert starts[0] - g0 == pytest.approx(100, abs=26)
    assert stops[0] - g0 == pytest.approx(200, abs=26)
    assert labels.runs_of(fine, 1.5)[0].size == 0


def test_roc_and_ap_known_values():
    truth = np.array([1, 1, 0, 0, 1, 0])
    s = np.array([0.9, 0.8, 0.7, 0.3, 0.6, 0.1])
    assert score.roc_auc(truth, s) == pytest.approx(8 / 9)
    assert score.average_precision(truth, s) == pytest.approx((1 + 1 + 0.75) / 3)
    tied = score.roc_auc(np.array([1, 0]), np.array([0.5, 0.5]))
    assert tied == 0.5
    assert np.isnan(score.roc_auc(np.ones(3), np.arange(3)))
    thr, f1 = score.best_threshold(truth, s)
    assert f1 == pytest.approx(6 / 7) and thr == pytest.approx(0.45)


def _part(shot, truth, sc, thr=0.5, spans=None):
    truth = np.asarray(truth, dtype=np.int8)
    sc = np.asarray(sc, dtype=float)
    kind = np.where(truth == 1, "crowd", "absent").astype(object)
    return score.ShotScore(shot, truth, kind, sc >= thr, sc, spans or {})


def test_summarise_counts_rates_and_interval_shape():
    a = _part(1, [1, 1, 0, 0], [0.9, 0.2, 0.1, 0.8])
    b = _part(2, [1, 0, 0, 0], [0.7, 0.1, 0.2, 0.3])
    out = score.summarise([a, b], score.draws(2, 50))
    assert out["counts"]["tp"] == 2 and out["counts"]["fp"] == 1
    assert out["counts"]["fn"] == 1 and out["counts"]["tn"] == 4
    assert out["point"]["precision"] == pytest.approx(2 / 3)
    assert out["point"]["crowd_bin_recall"] == pytest.approx(2 / 3)
    lo, hi = out["ci95"]["f1"]
    assert lo <= hi and out["replicates"] == 50


def test_paired_difference_needs_the_same_shots_and_is_zero_against_itself():
    a = [_part(1, [1, 0], [0.9, 0.1]), _part(2, [1, 0], [0.2, 0.6])]
    boot = score.draws(2, 30)
    d = score.paired_difference(a, a, boot, "auroc")
    assert d["value"] == 0.0 and d["ci95"] == [0.0, 0.0]
    with pytest.raises(ValueError):
        score.paired_difference(a, a[:1], boot, "f1")


def test_onset_peaks_and_match():
    x = np.zeros(200)
    x[50], x[53], x[120] = 0.9, 0.6, 0.8
    assert onset.peaks(x, 0.5).tolist() == [50, 120]  # 53 is within 10 of a higher one
    defined = np.ones(200, dtype=bool)
    defined[100:150] = False  # a crowd
    t0 = inputs.GRID0_MS
    found = np.array([50.5, 120.5, 20.5]) + t0
    tp, fp, fn = onset.match(found, np.array([52.0, 160.0]) + t0, defined, t0, 5.0)
    # 50.5 matches 52; 120.5 is in the crowd, ignored; 20.5 is false; 160 is missed
    assert (tp, fp, fn) == (1, 1, 1)


def test_unet_shapes_and_padding():
    m = net.ElmUNet(widths=(8, 8, 8, 8, 8))
    n = net.pad_to(5000)
    assert n % net.UNIT == 0 and n >= 5000 and net.pad_to(net.UNIT) == net.UNIT
    import torch

    y = m(torch.zeros(1, inputs.N_CHANNELS, n))
    assert y.shape == (1, 2, n // inputs.CELLS_PER_MS)


def _shot(shot, kinds):
    rows = []
    t = 0.0
    for k in kinds:
        rows.append((t, t + 100, k))
        t += 100
    spans = _table(rows)
    x = np.zeros((inputs.N_CHANNELS, 20_000), dtype=np.float32)
    x[inputs.VALID] = 1
    n_ms = 2000
    return train.ShotData(
        shot,
        x,
        labels.dense(spans, n_ms),
        n_ms,
        spans.assign(shot=shot),
        np.array([0.0]),
        np.array([float(n_ms)]),
        labels.scored_bins(spans, [0.0], [float(n_ms)]),
    )


def test_folds_partition_shots_and_balance_kinds():
    data = {
        s: _shot(s, ["absent", "crowd"] if s % 3 else ["absent", "non_crowd"])
        for s in range(30)
    }
    folds = train.deal_folds(data, 5, seed=1)
    flat = [s for f in folds for s in f]
    assert sorted(flat) == sorted(data) and len(set(flat)) == len(flat)
    sizes = [len(f) for f in folds]
    assert max(sizes) - min(sizes) <= 1
    rest, val = train.split_inner([s for s in data if s not in folds[0]], 4, seed=2)
    assert not set(rest) & set(val) and len(val) == 4


def test_test_shots_are_refused():
    cohort = pd.DataFrame({"shot": [1, 2, 3], "split": ["train", "val", "test"]})
    train.check_no_test([1, 2], cohort)
    with pytest.raises(ValueError, match="test"):
        train.check_no_test([1, 3], cohort)


def test_crop_loss_masks_ignored_time():
    import torch

    d = _shot(1, ["absent", "crowd", "absent"])
    rng = np.random.default_rng(0)
    cfg = train.Config(crop_ms=2560, batch=2)
    x, state, on, mask = train.crop(rng, d, cfg.crop_ms, cfg)
    assert x.shape == (inputs.N_CHANNELS, 25600) and state.shape == (2560,)
    b = {
        "x": torch.from_numpy(np.stack([x, x])),
        "state": torch.from_numpy(np.stack([state, state]).astype(np.int64)),
        "onset": torch.from_numpy(np.stack([on, on])),
        "onset_mask": torch.from_numpy(np.stack([mask, mask])),
    }
    logits = torch.zeros(2, 2, 2560)
    _, ev, _ = train.loss_of(logits, b)
    assert float(ev) == pytest.approx(np.log(2), abs=1e-5)  # zero logits: ln 2 per cell
    b["state"][:] = labels.IGNORE
    _, ev, _ = train.loss_of(logits, b)
    assert float(ev) == 0.0


def _review(rows):
    return pd.DataFrame(rows, columns=["t_start", "t_end", "kind"])


def test_span_counts_follow_the_benchmark_rule():
    review = _review(
        [
            (0.0, 100.0, "absent"),
            (100.0, 130.0, "non_crowd"),
            (130.0, 400.0, "crowd"),
            (400.0, 500.0, "absent"),  # nothing analysed here
        ]
    )
    cover = methods.cover_frame([0.0], [400.0])
    spans = methods.span_frame([20.0, 110.0], [30.0, 120.0])
    got = methods.span_counts(spans, cover, review)
    assert got == {
        "non_crowd_spans": 1,
        "non_crowd_span_hit": 1,
        "absent_spans": 1,
        "absent_span_alarm": 1,
        "crowd_spans": 1,
        "absent_spans_guard25_eligible": 1,
        "absent_span_alarm_guard25": 1,
        "absent_spans_guard25_empty": 0,
    }
    none = methods.span_counts(methods.span_frame([], []), cover, review)
    assert none["non_crowd_span_hit"] == 0 and none["absent_span_alarm"] == 0


def test_guarded_alarm_excludes_boundary_touches_and_counts_empty_interiors():
    review = _review([(0.0, 100.0, "absent"), (100.0, 140.0, "absent")])
    cover = methods.cover_frame([0.0], [140.0])
    spans = methods.span_frame([0.0, 110.0], [25.0, 120.0])
    got = methods.span_counts(spans, cover, review)
    assert got["absent_span_alarm"] == 2
    assert got["absent_span_alarm_guard25"] == 0
    assert got["absent_spans_guard25_eligible"] == 1
    assert got["absent_spans_guard25_empty"] == 1
    spans = methods.span_frame([24.0], [26.0])
    got = methods.span_counts(spans, cover, review)
    assert got["absent_span_alarm_guard25"] == 1


def test_row_part_reads_the_row_that_summarises_the_bin():
    t = np.arange(
        0, 400.0, 25.0
    )  # rows at 0, 25, ...; each summarises the 50 ms before
    s = np.zeros(t.size)
    s[6] = 1.0  # the row at 150 ms covers [100, 150)
    bins = labels.Bins(
        np.array([100.0, 150.0]),
        np.array([1, 0], dtype=np.int8),
        np.array(["crowd", "absent"], dtype=object),
        np.array([0, 1]),
    )
    review = _review([(0.0, 400.0, "crowd")])
    cover = methods.cover_frame([0.0], [400.0])
    now = methods.row_part(review, 1, bins, cover, t, s, 0.5)
    assert now.score.tolist() == [1.0, 0.0] and now.call.tolist() == [True, False]
    ahead = methods.row_part(review, 1, bins, cover, t, s, 0.5, lag_rows=-2)
    assert ahead.score.tolist() == [0.0, 1.0]  # the rows at each bin's start: 100, 150
    with pytest.raises(ValueError):
        far = labels.Bins(
            np.array([1000.0]),
            np.zeros(1, np.int8),
            np.array(["absent"], object),
            np.zeros(1, int),
        )
        methods.row_part(review, 1, far, cover, t, s, 0.5)


def test_trace_part_calls_bins_at_the_threshold():
    fine = np.zeros(400)
    fine[100:200] = 0.9
    bins = labels.Bins(
        np.array([inputs.GRID0_MS + 100.0, inputs.GRID0_MS + 300.0]),
        np.array([1, 0], dtype=np.int8),
        np.array(["crowd", "absent"], dtype=object),
        np.array([0, 1]),
    )
    review = _review([(50.0, 100.0, "absent"), (100.0, 200.0, "crowd")])
    # the review is on the shot clock: crowd 100-200 ms is cells 150-250 (GRID0 shift)
    cover = methods.cover_frame([0.0], [350.0])
    part = methods.trace_part(review, 3, bins, cover, fine, 0.5)
    assert part.call.tolist() == [True, False]
    assert part.spans["crowd_spans"] == 1
