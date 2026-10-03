"""Hermetic checks of `elm-dsm` (rows, detector, histograms) and the reference swap.

Synthetic records only; no data root, no corpus, no GPU.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from labeler.elm import compare, dsm, labels, methods, score, swap, swap_tex


def _spans(rows):
    return pd.DataFrame(
        [
            {"kind": k, "t_start": a, "t_end": b, "category": 0 if k == "absent" else 1}
            for a, b, k in rows
        ]
    )


def _bins(t0, truth, kind):
    n = len(t0)
    return labels.Bins(
        np.asarray(t0, float),
        np.asarray(truth, dtype=np.int8),
        np.asarray(kind, dtype=object),
        np.arange(n),
    )


def test_window_labels_need_the_whole_window_inside_one_span():
    spans = _spans([(0.0, 500.0, "absent"), (500.0, 1000.0, "crowd")])
    y = dsm.window_labels(spans)
    t = dsm.ROW_T_MS
    # the window [t - 50, t) lies wholly in the absent span for 50 <= t <= 500
    assert (y[(t >= 50) & (t <= 500)] == 0).all()
    assert y[t == 25][0] == -1  # [-25, 25) starts before the span
    assert y[t == 525][0] == -1  # straddles the boundary at 500
    assert (y[(t >= 550) & (t <= 1000)] == 1).all()
    assert (y[t > 1000] == -1).all()  # unlabelled time


def test_histogram_auroc_matches_the_rank_auroc():
    rng = np.random.default_rng(1)
    n = 4000
    shot = rng.integers(0, 12, n)
    case = rng.random(n) < 0.3
    risk = np.where(case, rng.normal(0.8, 1, n), rng.normal(0, 1, n))
    keep = rng.random(n) < 0.9
    shots, pos, neg = dsm.auroc_by_shot(risk, case, keep, shot, n_bins=2000)
    assert shots.tolist() == sorted(set(shot[keep].tolist()))
    exact = score.roc_auc(case[keep], risk[keep])
    assert dsm.hist_auroc(pos, neg) == pytest.approx(exact, abs=2e-3)
    assert np.isnan(dsm.hist_auroc(pos * 0, neg))


def test_detector_loads_the_published_embedding_and_gives_one_logit():
    w = np.random.default_rng(0).normal(size=(128, 60)).astype(np.float32)
    graph = SimpleNamespace(embedding=[w])
    model = dsm.Detector()
    model.load_published(graph)
    assert np.allclose(model.embedding.weight.detach().numpy(), w)
    out = dsm.predict(model, np.zeros((5, 60), dtype=np.float32))
    assert out.shape == (5,) and ((out >= 0) & (out <= 1)).all()


def test_rows_with_a_row_before_and_after_each_bin():
    usable = np.zeros(240, dtype=bool)
    usable[4:100] = True  # rows at 100 ms .. 2475 ms
    bins = _bins(
        [0.0, 50.0, 100.0, 2400.0, 2450.0, 6000.0],
        [1, 1, 1, 0, 0, 0],
        ["crowd"] * 3 + ["absent"] * 3,
    )
    # the end row of the bin at 50 ms is the row at 100 ms (index 4), usable
    kept = dsm.bins_with_rows(bins, usable).t0.tolist()
    assert kept == [50.0, 100.0, 2400.0]
    # with the forecast row too (the row at the bin's start) the first is dropped
    both = dsm.bins_with_rows(bins, usable, lags=(-2, 0)).t0.tolist()
    assert both == [100.0, 2400.0]
    assert dsm.row_index(bins, 0).tolist() == [2, 4, 6, 98, 100, 242]


def test_usable_cover_is_the_time_the_usable_rows_summarise():
    usable = np.zeros(240, dtype=bool)
    usable[4:10] = True  # rows 100..225 summarise [50, 225)
    cover = methods.cover_frame([0.0, 400.0], [200.0, 600.0])
    out = dsm.usable_cover(usable, cover)
    assert out.t_start_ms.tolist() == [50.0]
    assert out.t_end_ms.tolist() == [200.0]
    assert dsm.usable_cover(np.zeros(240, bool), cover).empty


def test_intersect_of_interval_sets():
    a0, a1 = np.array([0.0, 10.0]), np.array([5.0, 20.0])
    b0, b1 = np.array([3.0, 12.0, 18.0]), np.array([4.0, 15.0, 30.0])
    s, e = methods.intersect(a0, a1, b0, b1)
    assert s.tolist() == [3.0, 12.0, 18.0] and e.tolist() == [4.0, 15.0, 20.0]
    s, e = methods.intersect([0.0], [1.0], [], [])
    assert len(s) == 0


def test_forecast_spans_run_after_the_row():
    t = np.arange(0, 400.0, 25.0)
    on = np.zeros(t.size, bool)
    on[[4, 5, 8]] = True  # rows at 100, 125, 200
    back0, back1 = methods._row_runs(t, on)
    assert back0.tolist() == [50.0, 150.0] and back1.tolist() == [125.0, 200.0]
    ahead0, ahead1 = methods._row_runs(t, on, ahead=True)
    assert ahead0.tolist() == [100.0, 200.0] and ahead1.tolist() == [175.0, 250.0]


def test_fit_fold_learns_a_separable_detector_and_picks_a_threshold():
    rng = np.random.default_rng(0)
    rows, spans, bins = {}, {}, {}
    for shot in range(6):
        # 1000 ms blocks of absent / crowd; a row's window ends at its stamp
        y = ((25 * np.arange(240) - 1) // 1000) % 2
        x = rng.normal(size=(240, 60)).astype(np.float32)
        x[:, 3] += 3.0 * y
        usable = np.ones(240, bool)
        usable[:3] = False
        rows[shot] = dsm.Rows(shot, x, usable, usable.copy(), (), {}, ())
        spans[shot] = _spans(
            [(0.0, 1000.0, "absent"), (1000.0, 2000.0, "crowd")]
            + [(2000.0, 3000.0, "absent"), (3000.0, 4000.0, "crowd")]
            + [(4000.0, 5000.0, "absent"), (5000.0, 5975.0, "crowd")]
        )
        t0 = np.arange(50.0, 5900.0, 50.0)
        truth = ((t0 // 1000) % 2).astype(np.int8)
        bins[shot] = _bins(t0, truth, np.where(truth == 1, "crowd", "absent"))
    cfg = dsm.FitConfig(epochs=12, batch=128, lr=1e-2)
    state, thr, history = dsm.fit_fold(rows, spans, bins, [0, 1, 2, 3], [4, 5], cfg)
    assert 0.0 < thr < 1.0 and len(history) == 12
    assert max(h["val_auprc"] for h in history) > 0.95
    model = dsm.Detector()
    model.load_state_dict(state)
    sc = dsm.bin_end_scores(dsm.predict(model, rows[4].x), bins[4])
    assert score.roc_auc(bins[4].truth, sc) > 0.95


def test_rows_cache_round_trip(tmp_path):
    r = dsm.Rows(
        7,
        np.ones((240, 60), np.float32),
        np.ones(240, bool),
        np.zeros(240, bool),
        ("ip",),
        {"ece": "corpus", "ip": "archive"},
        ("a", "b"),
    )
    dsm.save_rows(r, tmp_path / "7.npz")
    back = dsm.load_rows(7, tmp_path / "7.npz")
    assert back.missing == ("ip",) and back.resolvers["ip"] == "archive"
    assert back.filled == ("a", "b") and back.x.shape == (240, 60)
    assert dsm.load_rows(8, tmp_path / "8.npz") is None
    np.savez_compressed(tmp_path / "9.npz", x=r.x)  # an older cache without resolvers
    assert dsm.load_rows(9, tmp_path / "9.npz") is None


def test_dsm_scores_round_trip(tmp_path):
    ds = compare.DsmScores(
        {1: None},
        {1: np.arange(960, dtype=np.float32).reshape(240, 4)},
        {"elm-dsm-detect": {1: np.linspace(0, 1, 240, dtype=np.float32)}},
        {"elm-dsm": {1: 0.25}, "elm-dsm-detect": {1: 0.5}},
    )
    ds.save(tmp_path)
    back = compare.DsmScores.load(tmp_path, {1: None}, variants=["elm-dsm-detect"])
    assert np.allclose(back.risk[1], ds.risk[1])
    assert np.allclose(back.scores["elm-dsm-detect"][1], ds.scores["elm-dsm-detect"][1])
    assert back.threshold["elm-dsm"][1] == 0.25


def test_table_truth_onset_truth_and_retruth():
    table = pd.DataFrame(
        {
            "shot": [1, 1, 1],
            "category": [0, 1, 0],
            "t_start": [0.0, 100.0, 150.0],
            "t_end": [100.0, 150.0, 300.0],
        }
    )
    bins = _bins(
        [0.0, 50.0, 100.0, 150.0, 300.0],
        [0, 0, 1, 1, 0],
        ["absent", "absent", "individual", "crowd", "absent"],
    )
    got = swap.table_truth(table, 1, bins)
    assert got.tolist() == [0, 0, 1, 0, -1]  # the bin at 300 ms is not covered
    assert swap.table_alignment(table, [1]) == {"rows": 3, "off_grid_edges": 0}
    on = swap.onset_truth([110.0, 160.0, 161.0], bins)
    assert on.tolist() == [0, 0, 1, 1, 0]
    part = score.ShotScore(
        1,
        bins.truth,
        bins.kind,
        np.array([0, 0, 1, 1, 0], bool),
        np.arange(5.0),
        {"x": 1},
    )
    re = swap.retruth(part, got)
    assert re.truth.tolist() == [0, 0, 1, 0] and re.call.tolist() == [0, 0, 1, 1]
    assert re.score.tolist() == [0.0, 1.0, 2.0, 3.0] and re.spans == {}
    assert re.kind.tolist() == ["absent", "absent", "present", "absent"]


def test_agreement_counts_missed_and_false_bins():
    review = np.array([1, 1, 1, 0, 0, 0], dtype=np.int8)
    kind = np.array(
        ["crowd", "crowd", "individual", "absent", "absent", "absent"], object
    )
    legacy = np.array([1, 0, 0, 1, 0, -1], dtype=np.int8)
    c = swap.agreement_counts(review, kind, legacy)
    names = dict(zip(swap.AGREEMENT_NAMES, c, strict=True))
    assert (
        names["tp"] == 1 and names["fn"] == 2 and names["fp"] == 1 and names["tn"] == 1
    )
    assert names["crowd_bins"] == 2 and names["crowd_legacy_present"] == 1
    assert names["individual_bins"] == 1 and names["individual_legacy_present"] == 0
    summary = swap.agreement_summary(np.stack([c, c]), score.draws(2, 50))
    assert summary["M"] == 4 and summary["P"] == 2 and summary["bins"] == 10
    assert summary["point"]["recall"] == pytest.approx(1 / 3)
    assert summary["point"]["precision"] == pytest.approx(0.5)
    assert summary["point"]["missed_share_of_present"] == pytest.approx(2 / 3)


def test_rankings_order_best_first_and_skip_missing_metrics():
    pts = {
        "a": {"auroc": 0.6, "f1": 0.9},
        "b": {"auroc": 0.8, "f1": 0.5},
        "c": {"f1": 0.7},
    }
    r = swap.rankings(pts)
    assert r["auroc"] == ["b", "a"] and r["f1"] == ["a", "c", "b"]


def test_swap_table_cells_and_layout():
    assert swap_tex.cell(0.9824, [0.9612, 0.99]) == "0.982 {\\scriptsize[0.96, 0.99]}"
    assert swap_tex.cell(float("nan"), None) == "--"

    def res(auroc, f1, prevalence=0.6):
        m = {
            "point": {"auroc": auroc, "f1": f1},
            "ci95": {"auroc": [0.5, 0.6], "f1": [0.4, 0.5]},
        }
        always = {"point": {"auroc": 0.5, "f1": 0.75}}
        return {
            "prevalence": prevalence,
            "methods": {"elm-ours": m, swap_tex.ALWAYS: always},
        }

    table = swap_tex.benchmark_table(
        {"reviewed": res(0.9, 0.8), "legacy": res(0.6, 0.4, 0.2)},
        "reviewed",
        "legacy",
        ("dense labels", "legacy onset table"),
        {"git": "abc1234"},
        "x.json",
    )
    assert table.count("\\\\") >= 4 and "elm-ours & 0.900" in table
    assert "(60\\% present)" in table and "(20\\% present)" in table
    assert "always present & 0.500 & 0.750 & 0.500 & 0.750" in table
