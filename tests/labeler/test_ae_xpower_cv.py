"""Five-fold cross-validation over v2's snapshot, on the synthetic AE tree."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from labeler.ae import xpower
from labeler.ae.xpower import cv, evaluate, model_dir, train
from labeler.ae.xpower.data import make_split, seldnet_split
from labeler.ae.xpower.model import FrameCNN, FrameCNNConfig
from labeler.config import sha256_of
from labeler.events.review import labels

from . import ae_tree

POOL = list(range(101, 111))
TEST = [111, 112]
SPLITS = {**{s: "train" for s in POOL}, **{s: "valid" for s in TEST}}
NAMES = list(train.candidates("v2"))


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """Ten pool shots, two test shots, v2's snapshot frozen from the saved labels."""
    paths = ae_tree.build(tmp_path, SPLITS)
    ae_tree.env(monkeypatch, paths)
    digest = ae_tree.snapshot(paths, monkeypatch)
    return paths, digest


def _fake_fit(calls):
    """`train.fit` that records its shots; fold K's best epoch is K + 2."""

    def fit(train_shots, val_shots, config, log=print):
        calls.append(
            {
                "train": sorted(s.shot for s in train_shots),
                "val": sorted(s.shot for s in val_shots),
                "config": config,
            }
        )
        best = len(calls) % 5 + 2
        history = [
            {"epoch": e, "val_f1": 0.5, "kept": e <= best} for e in range(1, best + 2)
        ]
        return FrameCNN(FrameCNNConfig(width=4)), history, 0.5

    return fit


def test_the_folds_are_seeded_balanced_and_written_once(tree, capsys):
    paths, digest = tree
    models = model_dir(paths, "v2")
    assert cv.main(["--folds"]) == 0
    with (models / "cv/folds.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    split = make_split(POOL + TEST, seldnet_split(paths.root / "ae/masks"))
    assert {int(r["shot"]): r["split"] for r in rows} == split
    assert sorted(int(r["shot"]) for r in rows if r["fold"] == "") == TEST
    folds = {int(r["shot"]): int(r["fold"]) for r in rows if r["fold"] != ""}
    assert folds == cv.make_folds(POOL)
    assert sorted(np.bincount(list(folds.values())).tolist()) == [2] * 5
    record = json.loads((models / "cv/folds.json").read_text())
    assert record["labels_sha256"] == digest and record["seed"] == 20260923
    assert record["counts"] == {"test": 2, "pool": 10, "folds": [2] * 5}
    before = (models / "cv/folds.csv").read_bytes()
    assert cv.main(["--folds"]) == 0  # the same folds again: unchanged
    assert (models / "cv/folds.csv").read_bytes() == before
    # 120 pool shots, as on the real data, deal 24 to each fold.
    real = cv.make_folds(range(1000, 1120))
    assert np.bincount(list(real.values())).tolist() == [24] * 5
    assert [cv.stop_fold(k) for k in range(5)] == [1, 2, 3, 4, 0]


def test_the_folds_read_only_the_snapshot_and_refuse_another(tree, capsys):
    paths, digest = tree
    # The live labels are the owner's to keep saving; v2 never reads them.
    labels.labels_path(xpower.event_dir(paths)).write_bytes(b"poison")
    assert cv.main(["--folds"]) == 0
    snapshot = xpower.snapshot_file(paths, "v2")
    snapshot.write_bytes(snapshot.read_bytes() + b"101,0,1990,2000,\n")
    with pytest.raises(SystemExit) as error:
        cv.main(["--folds"])
    assert error.value.code != 0
    stderr = capsys.readouterr().err
    assert str(snapshot) in stderr and digest in stderr
    assert "Traceback" not in stderr
    with pytest.raises(SystemExit):
        cv.main(["--candidate", NAMES[0], "--fold", "0"])
    assert str(snapshot) in capsys.readouterr().err


def test_a_fold_task_needs_the_folds_first(tree, capsys):
    paths, _ = tree
    with pytest.raises(SystemExit) as error:
        cv.main(["--candidate", NAMES[0], "--fold", "0"])
    assert error.value.code != 0
    stderr = capsys.readouterr().err
    assert "folds.csv" in stderr and "--folds" in stderr
    assert not (model_dir(paths, "v2") / "cv" / NAMES[0]).exists()


def test_fold_k_trains_on_three_stops_on_the_next_and_predicts_its_own(
    tree, monkeypatch
):
    paths, digest = tree
    calls = []
    monkeypatch.setattr(train, "fit", _fake_fit(calls))
    frames_called = []
    real = evaluate.shot_frames

    def shot_frames(shot, **kw):
        frames_called.append(shot)
        return real(shot, **kw)

    monkeypatch.setattr(evaluate, "shot_frames", shot_frames)
    assert cv.main(["--folds"]) == 0
    folds = cv.make_folds(POOL)
    by = {k: sorted(s for s, f in folds.items() if f == k) for k in range(5)}
    assert cv.main(["--candidate", "band80-mhd10", "--fold", "4"]) == 0
    (call,) = calls
    assert call["val"] == by[0]  # stop fold (4 + 1) mod 5
    assert call["train"] == sorted(by[1] + by[2] + by[3])
    assert call["config"].mhd_weight == 10.0
    assert call["config"].epochs == train.TrainConfig.epochs
    assert call["config"].patience == train.TrainConfig.patience
    assert frames_called == by[4]
    out = model_dir(paths, "v2") / "cv" / "band80-mhd10"
    record = json.loads((out / "fold4.json").read_text())
    assert record["fold"] == 4 and record["stop_fold"] == 0
    assert record["train_folds"] == [1, 2, 3]
    assert record["shots"] == by[4] and record["stop_shots"] == by[0]
    assert record["best_epoch"] == 3 and record["version"] == "v2"
    assert record["labels_sha256"] == digest
    assert record["folds_sha256"] == sha256_of(model_dir(paths, "v2") / "cv/folds.csv")
    assert record["npz_sha256"] == sha256_of(out / "fold4.npz")
    with np.load(out / "fold4.npz") as z:
        for shot in by[4]:
            assert z[f"p{shot}"].shape == (200,)
            assert z[f"o{shot}"].tolist().count(1) == 60  # AE over 300-900 ms
            assert z[f"m{shot}"][120:150].all() and not z[f"m{shot}"][:120].any()
            assert z[f"s{shot}"].dtype == bool
    # A fold is trained once outside runs/.
    with pytest.raises(SystemExit):
        cv.main(["--candidate", "band80-mhd10", "--fold", "4"])
    assert len(calls) == 1


def test_a_fold_runs_the_real_training_loop(tree):
    paths, _ = tree
    assert cv.main(["--folds"]) == 0
    args = ["--candidate", "band80-mhd3", "--fold", "0", "--epochs", "1"]
    assert cv.main(args) == 0
    record = json.loads(
        (model_dir(paths, "v2") / "cv/band80-mhd3/fold0.json").read_text()
    )
    assert len(record["history"]) == 1 and record["history"][0]["epoch"] == 1
    assert record["config"]["epochs"] == 1 and record["config"]["mhd_weight"] == 3.0


def test_a_candidate_of_another_version_or_a_bad_fold_is_refused(tree, capsys):
    assert cv.main(["--folds"]) == 0
    for args in (
        ["--candidate", "band0-mhd3", "--fold", "0"],
        ["--candidate", NAMES[0], "--fold", "5"],
        ["--candidate", NAMES[0]],
        ["--folds", "--choose"],
    ):
        with pytest.raises(SystemExit) as error:
            cv.main(args)
        assert error.value.code != 0
    assert "Traceback" not in capsys.readouterr().err


def _designed(monkeypatch):
    """P(AE) per candidate, on each shot's real scored frames: band80-mhd3 fires
    on MHD frames up to 0.83, the others only on AE, at 0.72."""
    real = evaluate.shot_frames

    def shot_frames(shot, **kw):
        frames = real(shot, **kw)
        prob = np.full(200, 0.02, dtype=np.float32)
        if kw["blob"]["candidate"] == "band80-mhd3":
            prob[30:90], prob[120:150] = 0.93, 0.83
        else:
            prob[30:90] = 0.72
        frames.prob = prob
        return frames

    monkeypatch.setattr(evaluate, "shot_frames", shot_frames)


def _all_folds(monkeypatch, skip=()):
    monkeypatch.setattr(train, "fit", _fake_fit([]))
    assert cv.main(["--folds"]) == 0
    for name in NAMES:
        for k in range(5):
            if (name, k) not in skip:
                assert cv.main(["--candidate", name, "--fold", str(k)]) == 0


def test_the_choice_refuses_while_any_fold_is_missing(tree, monkeypatch, capsys):
    paths, _ = tree
    _all_folds(monkeypatch, skip={("band80-mhd10", 2), ("band80-mhd30", 0)})
    with pytest.raises(SystemExit) as error:
        cv.main(["--choose"])
    assert error.value.code != 0
    stderr = capsys.readouterr().err
    assert "band80-mhd10 fold 2" in stderr and "band80-mhd30 fold 0" in stderr
    assert "2 of 15" in stderr and "Traceback" not in stderr
    assert not (model_dir(paths, "v2") / "cv/choice.json").exists()


def test_the_choice_pools_the_out_of_fold_frames_and_applies_the_rule(
    tree, monkeypatch
):
    paths, digest = tree
    _designed(monkeypatch)
    _all_folds(monkeypatch)
    assert cv.main(["--choose"]) == 0
    out = model_dir(paths, "v2") / "cv"
    choice = json.loads((out / "choice.json").read_text())
    # Branch 1: MHD FP 0 and F1 1 for band80-mhd3 at 0.85-0.90 and the others at
    # 0.10-0.70; the tie goes to the lower weight, then the threshold nearest 0.5.
    assert (choice["candidate"], choice["threshold"], choice["branch"]) == (
        "band80-mhd3",
        0.85,
        1,
    )
    assert choice["version"] == "v2" and choice["labels_sha256"] == digest
    table = choice["table"]
    assert len(table) == 3 * 17
    assert {r["candidate"] for r in table} == set(NAMES)
    row = next(
        r for r in table if r["candidate"] == "band80-mhd3" and r["threshold"] == 0.5
    )
    assert row["fp_rate_mhd"] == 1.0 and row["f1"] == pytest.approx(120 / 150)
    for key in ("f1", "precision", "recall", "fp_rate_mhd"):
        assert all(key in r for r in table)
    assert choice["frames"]["shots"] == 10  # every pool shot, once
    assert choice["frames"]["present"] == 600 and choice["frames"]["mhd_absent"] == 300
    # Fold k's fake best epoch; the final trains for their median.
    assert choice["best_epochs"]["band80-mhd3"] == [3, 4, 5, 6, 2]
    assert choice["final_epochs"] == 4
    assert set(choice["folds"]["band80-mhd30"]) == {"0", "1", "2", "3", "4"}
    report = (out / "frontier.md").read_text()
    assert "branch 1" in report and "band80-mhd3" in report and "0.85" in report
    # Choosing again from the same records changes nothing.
    before = (out / "choice.json").read_bytes()
    assert cv.main(["--choose"]) == 0
    assert (out / "choice.json").read_bytes() == before


def test_the_choice_refuses_a_fold_record_that_does_not_match(
    tree, monkeypatch, capsys
):
    paths, _ = tree
    _all_folds(monkeypatch)
    out = model_dir(paths, "v2") / "cv" / "band80-mhd30"
    with np.load(out / "fold3.npz") as z:
        arrays = dict(z)
    arrays[next(iter(arrays))][0] = 0.99
    np.savez(out / "fold3.npz", **arrays)
    with pytest.raises(SystemExit) as error:
        cv.main(["--choose"])
    assert error.value.code != 0
    stderr = capsys.readouterr().err
    assert "fold3" in stderr and "Traceback" not in stderr


def test_a_pilot_writes_under_runs_and_can_run_again(tree, monkeypatch):
    paths, _ = tree
    calls = []
    monkeypatch.setattr(train, "fit", _fake_fit(calls))
    assert cv.main(["--folds", "--pilot", "5"]) == 0
    for _ in range(2):
        assert cv.main(["--candidate", NAMES[0], "--fold", "1", "--pilot", "5"]) == 0
    assert calls[0]["config"].epochs == 2
    assert len(calls[0]["train"]) == 3 and len(calls[0]["val"]) == 1
    pilot = paths.runs / "ae_xpower" / "pilot" / "v2" / "cv"
    record = json.loads((pilot / NAMES[0] / "fold1.json").read_text())
    assert record["pilot"] == 5 and len(record["shots"]) == 1
    assert not (model_dir(paths, "v2") / "cv").exists()
