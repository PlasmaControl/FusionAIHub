"""`labeler.paper.roc`: each selected model's ROC over its evaluation's scored set."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from labeler import frames
from labeler.frames import evaluate as frames_evaluate
from labeler.frames import prepare, train
from labeler.frames.model import RowsCNN
from labeler.frames.targets import ABSENT, PRESENT_T, UNCERTAIN_T, UNKNOWN
from labeler.paper import AE, ORDER, roc, roster
from labeler.scoring import stats

from . import paper_tree as tree
from .test_round3_frames_evaluate import rechoose

NTM = "ntm_frames"


@pytest.fixture
def paths(tmp_path, monkeypatch):
    return tree.temporary_paths(tmp_path, monkeypatch)


@pytest.fixture(autouse=True)
def committed(monkeypatch):
    """The tests' code counts as committed: `write` refuses a dirty tree."""
    monkeypatch.setattr(roc, "git_dirty", lambda: False)


@pytest.mark.parametrize("dirty", [True, None])
def test_no_record_is_written_from_uncommitted_code(paths, monkeypatch, dirty):
    monkeypatch.setattr(roc, "git_dirty", lambda: dirty)
    monkeypatch.setattr(roc, "roc", lambda *a, **k: pytest.fail("computed"))
    with pytest.raises(RuntimeError, match="commit first"):
        roc.write(paths, [NTM])
    assert not roc.roc_file(paths, NTM).exists()


def _mann_whitney(score, truth) -> float:
    """P(a positive outscores a negative), a tie counted half, by every pair."""
    pos = [s for s, t in zip(score, truth, strict=True) if t]
    neg = [s for s, t in zip(score, truth, strict=True) if not t]
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def test_the_auroc_counts_a_tie_half():
    score, truth = [0.9, 0.8, 0.8, 0.3], [True, True, False, False]
    fp, tp, fpr, tpr = roc.curve(score, truth)
    assert fp.tolist() == [0, 0, 1, 2] and tp.tolist() == [0, 1, 2, 2]
    assert fpr.tolist() == [0, 0, 0.5, 1] and tpr.tolist() == [0, 0.5, 1, 1]
    assert roc.auroc(score, truth) == 0.875, "(1 + 1 + 1/2 + 1) / 4"
    rng = np.random.default_rng(0)
    score = rng.integers(0, 5, 60) / 4  # many ties
    truth = rng.random(60) < 0.4
    assert roc.auroc(score, truth) == pytest.approx(_mann_whitney(score, truth))
    with pytest.raises(ValueError, match="finite"):
        roc.curve([0.5, np.nan], [True, False])
    with pytest.raises(ValueError, match="no ROC"):
        roc.curve([0.5, 0.4], [True, True])


def test_the_thinned_curve_keeps_its_ends_and_at_most_its_points():
    rng = np.random.default_rng(1)
    truth = rng.random(5000) < 0.3
    score = rng.random(5000) + truth * 0.5
    _, _, fpr, tpr = roc.curve(score, truth)
    assert len(fpr) > roc.ROC_POINTS
    thin_fpr, thin_tpr = roc.thin(fpr, tpr)
    assert 2 < len(thin_fpr) <= roc.ROC_POINTS
    assert (thin_fpr[0], thin_tpr[0]) == (0, 0)
    assert (thin_fpr[-1], thin_tpr[-1]) == (1, 1)
    assert (np.diff(thin_fpr) >= 0).all() and (np.diff(thin_tpr) >= 0).all()
    assert set(zip(thin_fpr, thin_tpr, strict=True)) <= set(
        zip(fpr, tpr, strict=True)
    ), "points of the full curve"
    short = roc.thin([0, 0.5, 1], [0, 0.7, 1])
    assert [x.tolist() for x in short] == [[0, 0.5, 1], [0, 0.7, 1]]


def test_the_average_precision_is_the_step_sum_over_distinct_scores():
    # 0.9 P, 0.8 P, 0.8 N, 0.3 N: thresholds 0.9 (R 1/2, P 1), 0.8 (R 1, P 2/3),
    # 0.3 (R 1, P 1/2): AP = 1/2 * 1 + 1/2 * 2/3 + 0
    score, truth = [0.9, 0.8, 0.8, 0.3], [True, True, False, False]
    assert roc.average_precision(score, truth) == pytest.approx(5 / 6)
    # a tie of a positive with a negative is one threshold: P 1/2, R 1
    tied = roc.average_precision([0.5, 0.5], [True, False])
    assert tied == 0.5
    rng = np.random.default_rng(1)
    score = rng.integers(0, 6, 80) / 5  # many ties
    truth = rng.random(80) < 0.3
    ap, last = 0.0, 0.0
    for t in sorted(set(score), reverse=True):  # by every threshold
        said = score >= t
        hit = (said & truth).sum()
        recall, precision = hit / truth.sum(), hit / said.sum()
        ap += (recall - last) * precision
        last = recall
    assert roc.average_precision(score, truth) == pytest.approx(ap)


def test_the_average_precision_is_scikit_learns():
    metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(1)
    score = rng.integers(0, 6, 80) / 5  # many ties
    truth = rng.random(80) < 0.3
    x = rng.random(200)
    y = rng.random(200) < 0.2
    for a, b in ((score, truth), (x, y)):
        assert roc.average_precision(a, b) == pytest.approx(
            metrics.average_precision_score(b, a)
        )


def test_the_pr_curve_thins_like_the_roc_and_keeps_both_ends():
    rng = np.random.default_rng(2)
    score, truth = rng.random(3000), rng.random(3000) < 0.1
    fp, tp, fpr, tpr = roc.curve(score, truth)
    recall, precision = roc.pr_points(fp, tp, int(truth.sum()))
    keep = roc.thin_index(fpr, tpr)
    assert len(keep) <= roc.ROC_POINTS
    assert keep[0] == 0 and keep[-1] == len(fpr) - 1
    assert (recall[keep][0], precision[keep][0]) == (0, 1)
    assert recall[keep][-1] == 1
    assert precision[keep][-1] == pytest.approx(truth.mean())
    x, _ = roc.thin(fpr, tpr)
    assert len(x) == len(keep)


def _scored(threshold: float = 0.5) -> roc.Scored:
    score = np.array([0.9, 0.7, 0.6, 0.4, 0.3, 0.2])
    truth = np.array([True, True, False, True, False, False])
    return roc.Scored(score, truth, 3, threshold, Path("m/model.pt"), "abc", "bins")


def _frame_evaluation(f1: float, sha: str = "abc") -> dict:
    return {"scores": {NTM: {"f1": f1}}, "model": {"sha256": sha}}


def _record(scored: roc.Scored, evaluation: dict) -> dict:
    return roc.record(
        scored,
        evaluation,
        method=NTM,
        version="v1",
        evaluation_path=Path("m/evaluation.json"),
        evaluation_sha256="def",
    )


def test_the_f1_check_refuses_a_mismatch():
    scored = _scored()
    cells = roc.cells(scored.score, scored.truth, scored.threshold)
    assert cells.tolist() == [2, 1, 1, 2], "0.6 >= 0.5 is said present"
    f1 = float(stats.f1(cells))
    found = _record(scored, _frame_evaluation(f1 + 1e-12))
    assert found["f1_check"]["computed"] == f1
    assert found["f1_check"]["cells"] == [2, 1, 1, 2]
    assert found["f1_check"]["evaluation_key"] == f"scores.{NTM}.f1"
    assert (found["n_pos"], found["n_neg"], found["shots"]) == (3, 3, 3)
    assert found["threshold"] == {
        "value": 0.5,
        "fpr": 1 / 3,
        "tpr": 2 / 3,
        "precision": 2 / 3,
        "recall": 2 / 3,
    }
    p, r = found["threshold"]["precision"], found["threshold"]["recall"]
    assert 2 * p * r / (p + r) == pytest.approx(f1, abs=1e-9), "PR point gives F1"
    assert found["positive_share"] == 0.5
    assert found["pr"]["recall"][0] == 0 and found["pr"]["precision"][0] == 1
    assert found["pr"]["recall"][-1] == 1 and found["pr"]["precision"][-1] == 0.5
    assert found["auprc"] == roc.average_precision(scored.score, scored.truth)
    assert found["curve"]["fpr"][0] == 0 and found["curve"]["tpr"][-1] == 1
    assert found["phenomenon"] == "neoclassical_tearing_mode"
    assert found["note"] == roc.NOTE
    assert found["evaluation"] == {"path": "m/evaluation.json", "sha256": "def"}
    json.dumps(found, allow_nan=False)
    with pytest.raises(roc.F1Mismatch, match="not the evaluation's"):
        _record(scored, _frame_evaluation(f1 + 1e-6))
    with pytest.raises(roc.F1Mismatch):
        roc.check_f1(f1, None, "no F1 recorded")
    with pytest.raises(ValueError, match="sha256"):
        _record(scored, _frame_evaluation(f1, sha="other"))


def test_the_recorded_f1_is_where_draw_scores_reads_it():
    ae = tree.ae_evaluation()
    assert roc.recorded_f1(roc.AE_METHOD, ae) == ("methods.ae_xpower.f1.value", 0.89)
    assert roc.f1_estimate(roc.AE_METHOD, ae) == ae["methods"]["ae_xpower"]["f1"]
    assert roc.f1_key(NTM) == "f1"
    assert roc.f1_key("hmode_frames") == "f1(H)", "H-mode's ROC is for H"
    hmode = {"scores": {"hmode_frames": {"f1(H)": 0.7, "f1(L)": 0.4}}}
    assert roc.recorded_f1("hmode_frames", hmode) == ("scores.hmode_frames.f1(H)", 0.7)
    assert list(roc.SELECTED) == list(ORDER)
    assert roc.SELECTED[AE] == roc.AE_METHOD
    assert all(roc.SELECTED[c] == roster.TABLES[c] for c in ORDER if c != AE)


def test_the_bins_are_the_ones_evaluate_scores():
    states = np.array(
        [ABSENT, PRESENT_T, UNKNOWN, UNCERTAIN_T, PRESENT_T, ABSENT, PRESENT_T]
        + [ABSENT, PRESENT_T, ABSENT]
    )
    prob = np.array([0.1, 0.8, 0.9, 0.9, np.nan, 0.6, 0.5, 0.2, 0.3, 0.7])
    observed = np.array([True] * 8 + [False, True])
    score, truth = roc.scored_bins(prob, states, observed)
    assert score.tolist() == [0.1, 0.8, 0.6, 0.5, 0.2, 0.7]
    assert truth.tolist() == [False, True, False, True, False, False]
    for threshold, obs in itertools.product((0.0, 0.5, 0.65, 1.0), (observed, None)):
        s, t = roc.scored_bins(prob, states, obs)
        cells = frames_evaluate.score(prob, states, threshold, obs)["cells"]
        assert roc.cells(s, t, threshold).tolist() == cells, (threshold, obs)


def test_a_whole_window_ae_version_is_refused(paths):
    with pytest.raises(ValueError, match="whole windows"):
        roc.ae_scored(paths, "v3")


def test_the_cli_refuses_to_overwrite_a_roc(paths, monkeypatch, capsys):
    monkeypatch.setattr(torch, "set_num_threads", lambda n: None)
    target = roc.roc_file(paths, NTM)
    assert target.name == roc.ROC_FILE
    assert target.parent == roc.evaluation_file(paths, NTM).parent
    target.parent.mkdir(parents=True)
    target.write_text("{}\n")

    def computed(*args, **kwargs):
        raise AssertionError("computed before the refusal")

    monkeypatch.setattr(roc, "roc", computed)
    with pytest.raises(FileExistsError, match="recorded once"):
        roc.write(paths, [roc.AE_METHOD, NTM])
    with pytest.raises(SystemExit) as refused:
        roc.main(["--method", NTM])
    assert refused.value.code == 2
    assert "recorded once" in capsys.readouterr().err
    assert target.read_text() == "{}\n"
    assert not roc.roc_file(paths, roc.AE_METHOD).exists()
    assert roc.methods_of(None) == roc.methods_of(["all", NTM])
    assert roc.methods_of([NTM, roc.AE_METHOD]) == [roc.AE_METHOD, NTM]


def _old_record(found: dict) -> dict:
    """A record as F5 wrote it: no PR fields."""
    return {
        k: v for k, v in found.items() if k not in ("auprc", "pr", "positive_share")
    }


def _fake_roc(monkeypatch, found: dict) -> None:
    monkeypatch.setattr(roc, "roc", lambda paths, method, ae_version="v2": found)


def test_replace_without_pr_replaces_an_old_record_only(paths, monkeypatch):
    scored = _scored()
    new = _record(
        scored,
        _frame_evaluation(
            float(stats.f1(roc.cells(scored.score, scored.truth, scored.threshold)))
        ),
    )
    _fake_roc(monkeypatch, new)
    target = roc.roc_file(paths, NTM)
    target.parent.mkdir(parents=True)

    def put(record: dict) -> None:
        target.write_text(json.dumps(record))

    put(_old_record(new))
    with pytest.raises(FileExistsError, match="recorded once"):
        roc.write(paths, [NTM])
    roc.write(paths, [NTM], replace_without_pr=True)
    assert json.loads(target.read_text())["auprc"] == new["auprc"]
    before = target.read_text()  # now has auprc: refused
    with pytest.raises(FileExistsError, match="has PR"):
        roc.write(paths, [NTM], replace_without_pr=True)
    assert target.read_text() == before
    for change in ({"auroc": new["auroc"] - 1e-6}, {"n_pos": 99}, {"n_neg": 99}):
        put({**_old_record(new), **change})
        kept = target.read_text()
        with pytest.raises(ValueError, match="differ"):
            roc.write(paths, [NTM], replace_without_pr=True)
        assert target.read_text() == kept
    with pytest.raises(SystemExit):
        roc.main(["--method", NTM])
    assert target.read_text() == kept


def test_replace_without_pr_writes_nothing_if_any_is_refused(paths, monkeypatch):
    scored = _scored()
    cells = roc.cells(scored.score, scored.truth, scored.threshold)
    new = _record(scored, _frame_evaluation(float(stats.f1(cells))))
    _fake_roc(monkeypatch, new)
    for method in (NTM, roc.AE_METHOD):
        target = roc.roc_file(paths, method)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(_old_record(new)))
    bad = roc.roc_file(paths, roc.AE_METHOD)
    bad.write_text(json.dumps({**_old_record(new), "n_pos": 1}))
    with pytest.raises(ValueError, match="differ"):
        roc.write(paths, [roc.AE_METHOD, NTM], replace_without_pr=True)
    assert "auprc" not in json.loads(roc.roc_file(paths, NTM).read_text())


def _evaluation_on_disk(paths, method: str = NTM) -> str:
    """An `evaluation.json` for `method`; its sha256."""
    path = roc.evaluation_file(paths, method)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"scores": {}}\n')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_replace_for_threshold_replaces_a_record_of_an_older_evaluation(
    paths, monkeypatch
):
    monkeypatch.setattr(torch, "set_num_threads", lambda n: None)
    scored = _scored()
    cells = roc.cells(scored.score, scored.truth, scored.threshold)
    new = _record(scored, _frame_evaluation(float(stats.f1(cells))))
    _fake_roc(monkeypatch, new)
    current = _evaluation_on_disk(paths)
    target = roc.roc_file(paths, NTM)
    kept = target.with_name(roc.TRAINED_ROC_FILE)
    assert kept.name == "roc.trained-threshold.json"

    def put(record: dict) -> str:
        target.write_text(json.dumps(record))
        return target.read_text()

    at = {**new["threshold"], "value": 0.65}
    fresh = {**new, "threshold": at, "evaluation": {"path": "e", "sha256": current}}
    before = put(fresh)
    with pytest.raises(FileExistsError, match="recorded once"):
        roc.write(paths, [NTM])
    # A record of the current evaluation.json stays.
    with pytest.raises(ValueError, match="current"):
        roc.write(paths, [NTM], replace_for_threshold=True)
    assert target.read_text() == before and not kept.exists()
    stale = {**fresh, "evaluation": {"path": "e", "sha256": "0" * 64}}
    changes = (
        {"auroc": float(np.nextafter(new["auroc"], 0.0))},  # == , not a tolerance
        {"n_pos": 99},
        {"n_neg": 99},
    )
    for change in changes:
        before = put({**stale, **change})
        with pytest.raises(ValueError, match="differ"):
            roc.write(paths, [NTM], replace_for_threshold=True)
        assert target.read_text() == before and not kept.exists()
    before = put(stale)
    assert roc.main(["--method", NTM, "--replace-for-threshold"]) == 0
    assert kept.read_text() == before
    assert json.loads(target.read_text()) == new
    # The trained threshold's record is never overwritten.
    put(stale)
    with pytest.raises(FileExistsError, match="trained-threshold"):
        roc.write(paths, [NTM], replace_for_threshold=True)
    assert kept.read_text() == before
    with pytest.raises(ValueError, match="one replacement"):
        roc.write(paths, [NTM], replace_without_pr=True, replace_for_threshold=True)


def test_a_frame_roc_reads_the_threshold_through_load(paths, monkeypatch):
    out = frames.model_dir(paths, NTM, frames.VERSION)
    subs = round(10 / frames.SPECS[NTM].sub_ms)
    blob = {"threshold": 0.65, "channels": 42, "subs": subs, "width": 32}
    train.save(out, RowsCNN(42, subs).eval(), blob)
    monkeypatch.setattr(prepare, "split_shots", lambda paths_, method: {})
    found = roc.frame_scored(paths, NTM)
    assert found.threshold == 0.65
    assert found.threshold_from == {
        "threshold_source": "training",
        "threshold_sha256": None,
    }
    path = rechoose(out, 0.4)
    found = roc.frame_scored(paths, NTM)
    assert found.threshold == 0.4
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert found.threshold_from == {
        "threshold_source": "threshold.json",
        "threshold_sha256": sha,
    }
    # The record names threshold.json beside the model.
    scored = _scored()._replace(threshold_from=found.threshold_from)
    cells = roc.cells(scored.score, scored.truth, scored.threshold)
    record = _record(scored, _frame_evaluation(float(stats.f1(cells))))
    assert record["model"] == {"path": "m/model.pt", "sha256": "abc"} | (
        found.threshold_from
    )
    plain = _record(_scored(), _frame_evaluation(float(stats.f1(cells))))
    assert plain["model"] == {"path": "m/model.pt", "sha256": "abc"}, "AE's"
