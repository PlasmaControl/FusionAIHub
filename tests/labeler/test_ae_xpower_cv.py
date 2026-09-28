"""Five-fold cross-validation over v2's snapshot, on the synthetic AE tree."""

from __future__ import annotations

import copy
import csv
import dataclasses
import json

import numpy as np
import pytest
import torch

from labeler.ae import xpower
from labeler.ae.xpower import cv, evaluate, model_dir, train
from labeler.ae.xpower.data import make_split, seldnet_split, store_rows
from labeler.ae.xpower.model import FrameCNN, FrameCNNConfig
from labeler.config import sha256_of
from labeler.events.review import labels

from . import ae_tree

POOL = list(range(101, 111))
TEST = [111, 112]
SPLITS = {**{s: "train" for s in POOL}, **{s: "valid" for s in TEST}}
NAMES = list(train.candidates("v2"))


def cv_tree(tmp_path, monkeypatch):
    """Ten pool shots, two test shots, v2's snapshot frozen from the saved labels."""
    paths = ae_tree.build(tmp_path, SPLITS)
    ae_tree.env(monkeypatch, paths)
    digest = ae_tree.snapshot(paths, monkeypatch)
    return paths, digest


@pytest.fixture
def tree(tmp_path, monkeypatch):
    return cv_tree(tmp_path, monkeypatch)


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
    # v1's TrainConfig, with the candidate's MHD weight.
    assert call["config"] == train.TrainConfig(mhd_weight=10.0)
    assert frames_called == by[4]
    out = model_dir(paths, "v2") / "cv" / "band80-mhd10"
    record = json.loads((out / "fold4.json").read_text())
    assert record["fold"] == 4 and record["stop_fold"] == 0
    assert record["train_folds"] == [1, 2, 3]
    assert record["shots"] == by[4] and record["stop_shots"] == by[0]
    assert record["best_epoch"] == 3 and record["version"] == "v2"
    assert record["config"] == dataclasses.asdict(train.TrainConfig(mhd_weight=10.0))
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


def _short(monkeypatch, epochs=1):
    """A fold task's `fold_config`, for `epochs` epochs (there is no --epochs)."""
    real = cv.fold_config
    monkeypatch.setattr(
        cv,
        "fold_config",
        lambda spec, pilot: dataclasses.replace(real(spec, pilot), epochs=epochs),
    )


def test_a_fold_runs_the_real_training_loop(tree, monkeypatch):
    paths, _ = tree
    _short(monkeypatch)
    assert cv.main(["--folds"]) == 0
    assert cv.main(["--candidate", "band80-mhd3", "--fold", "0"]) == 0
    record = json.loads(
        (model_dir(paths, "v2") / "cv/band80-mhd3/fold0.json").read_text()
    )
    assert len(record["history"]) == 1 and record["history"][0]["epoch"] == 1
    assert record["config"]["epochs"] == 1 and record["config"]["mhd_weight"] == 3.0


def test_a_fold_task_takes_no_epochs(tree, monkeypatch, capsys):
    paths, _ = tree
    calls = []
    monkeypatch.setattr(train, "fit", _fake_fit(calls))
    assert cv.main(["--folds"]) == 0
    with pytest.raises(SystemExit) as error:
        cv.main(["--candidate", NAMES[0], "--fold", "0", "--epochs", "1"])
    assert error.value.code == 2 and "--epochs" in capsys.readouterr().err
    assert calls == []
    assert not (model_dir(paths, "v2") / "cv" / NAMES[0]).exists()


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
    assert calls[0]["config"] == train.TrainConfig(epochs=2, mhd_weight=3.0)
    assert len(calls[0]["train"]) == 3 and len(calls[0]["val"]) == 1
    pilot = paths.runs / "ae_xpower" / "pilot" / "v2" / "cv"
    record = json.loads((pilot / NAMES[0] / "fold1.json").read_text())
    assert record["pilot"] == 5 and len(record["shots"]) == 1
    assert not (model_dir(paths, "v2") / "cv").exists()


@pytest.mark.parametrize(
    "pilot, key, value",
    [
        (0, "epochs", 1),  # what `cv --epochs 1` trained
        (0, "patience", 5),
        (0, "mhd_weight", 30.0),  # another candidate's weight
        (5, "epochs", 60),  # a pilot's fold, trained as a full one
    ],
)
def test_the_choice_refuses_a_fold_record_trained_with_another_config(
    tree, monkeypatch, capsys, pilot, key, value
):
    paths, _ = tree
    extra = ["--pilot", str(pilot)] if pilot else []
    monkeypatch.setattr(train, "fit", _fake_fit([]))
    assert cv.main(["--folds", *extra]) == 0
    for name in NAMES:
        for k in range(5):
            assert cv.main(["--candidate", name, "--fold", str(k), *extra]) == 0
    runs = paths.runs / "ae_xpower" / "pilot" / "v2"
    out = (runs if pilot else model_dir(paths, "v2")) / "cv"
    file = out / "band80-mhd10" / "fold2.json"
    record = json.loads(file.read_text())
    assert record["config"][key] != value and record["mhd_weight"] == 10.0
    record["config"][key] = value
    file.write_text(json.dumps(record, indent=1) + "\n")
    with pytest.raises(SystemExit) as error:
        cv.main(["--choose", *extra])
    assert error.value.code != 0
    stderr = capsys.readouterr().err
    assert str(file) in stderr and "config" in stderr and "Traceback" not in stderr
    assert not (out / "choice.json").exists()


def _source_file(paths):
    return labels.source_path(xpower.event_dir(paths))


def _shorten_source(paths, shot=103, end=990):
    """The source table's window of `shot` ends at `end` ms, not 2000 ms, so
    that the frames after it lie outside it; the table's file."""
    file = _source_file(paths)
    text = file.read_text()
    old = f"{shot},0,900,2000,"
    assert text.count(old) == 1
    file.write_text(text.replace(old, f"{shot},0,900,{end},"))
    return file


def test_the_out_of_fold_frames_are_the_tests_frames(tree, monkeypatch):
    """The reviewer's Minor 5: a fold's frames go through the test's frame
    selection (`evaluate.shot_frames` with the source table's label), its
    source-window filter included, and the choice counts what that drops."""
    paths, _ = tree
    file = _shorten_source(paths)  # 103's frames 99-199 (990-2000 ms) outside
    _designed(monkeypatch)
    given, designed = {}, evaluate.shot_frames

    def shot_frames(shot, **kw):
        given[shot] = kw.get("source")
        return designed(shot, **kw)

    monkeypatch.setattr(evaluate, "shot_frames", shot_frames)
    _all_folds(monkeypatch)
    source = labels.read_labels(file)
    assert given == {s: source[s] for s in POOL}  # the test's source labels
    assert cv.main(["--choose"]) == 0
    out = model_dir(paths, "v2") / "cv"
    choice = json.loads((out / "choice.json").read_text())
    assert choice["source_sha256"] == sha256_of(file)
    # 101 frames dropped, 30 of them MHD frames the owner called absent.
    assert choice["frames"] == {
        "shots": 10,
        "scored": 10 * 200 - 101,
        "present": 600,
        "mhd_absent": 300 - 30,
        "source_dropped": 101,
    }
    k = cv.make_folds(POOL)[103]
    for name in NAMES:
        for fold in range(5):
            record = json.loads((out / name / f"fold{fold}.json").read_text())
            assert record["source_sha256"] == sha256_of(file)
            assert record["source_dropped"] == (101 if fold == k else 0)
    # The saved frames are those the test scores.
    saved = xpower.read_snapshot(paths, "v2")[1]
    blob = {"band_khz": [80.0, 250.0], "threshold": 0.5, "candidate": NAMES[0]}
    model = FrameCNN(FrameCNNConfig(width=4))
    frames = evaluate.shot_frames(
        103, paths=paths, label=saved[103], model=model, blob=blob, source=source[103]
    )
    with np.load(out / NAMES[0] / f"fold{k}.npz") as z:
        assert np.array_equal(z["s103"], frames.scored)
        assert z["s103"][:99].all() and not z["s103"][99:].any()
    assert frames.source_dropped == 101
    assert "101 frames outside the source table's window" in (
        out / "frontier.md"
    ).read_text().replace("\n", " ")


def test_a_fold_shot_the_source_table_lacks_is_refused_before_training(
    tree, monkeypatch, capsys
):
    paths, _ = tree
    file = _source_file(paths)
    kept = [x for x in file.read_text().splitlines() if not x.startswith("104,")]
    file.write_text("\n".join(kept) + "\n")
    calls = []
    monkeypatch.setattr(train, "fit", _fake_fit(calls))
    assert cv.main(["--folds"]) == 0
    k = cv.make_folds(POOL)[104]
    with pytest.raises(SystemExit) as error:
        cv.main(["--candidate", NAMES[0], "--fold", str(k)])
    assert error.value.code != 0
    stderr = capsys.readouterr().err
    assert str(file) in stderr and "104" in stderr and "Traceback" not in stderr
    assert calls == []  # refused before any training
    assert not (model_dir(paths, "v2") / "cv" / NAMES[0]).exists()


def test_the_choice_refuses_folds_scored_on_another_source_table(
    tree, monkeypatch, capsys
):
    paths, _ = tree
    _all_folds(monkeypatch)
    _shorten_source(paths)  # after the fold tasks
    with pytest.raises(SystemExit) as error:
        cv.main(["--choose"])
    assert error.value.code != 0
    stderr = capsys.readouterr().err
    assert "source_sha256" in stderr and "Traceback" not in stderr
    assert not (model_dir(paths, "v2") / "cv" / "choice.json").exists()


def test_one_check_names_a_versions_candidates(tree, tmp_path, monkeypatch, capsys):
    """The reviewer's Minor 9: cv and train refuse another version's candidate
    through one function, with one message."""
    assert train.candidate_spec("v2", "band80-mhd10") == {
        "band": (80.0, 250.0),
        "mhd_weight": 10.0,
    }
    message = (
        "candidate band0-mhd3 is not one of version v2's: band80-mhd3, "
        "band80-mhd10, band80-mhd30"
    )
    with pytest.raises(ValueError) as error:
        train.candidate_spec("v2", "band0-mhd3")
    assert str(error.value) == message
    assert cv.main(["--folds"]) == 0
    out = ["--out", str(tmp_path / "v2" / "band0-mhd3")]
    commands = (
        (cv.main, ["--candidate", "band0-mhd3", "--fold", "0"]),
        (train.main, ["--version", "v2", "--candidate", "band0-mhd3", *out]),
    )
    for main, args in commands:
        with pytest.raises(SystemExit):
            main(args)
        assert message in capsys.readouterr().err

    def refuse(version, name):
        raise ValueError(f"the one check refused {name} of {version}")

    monkeypatch.setattr(train, "candidate_spec", refuse)
    out = ["--out", str(tmp_path / "v1" / "band0-mhd3")]
    commands = (
        (cv.main, ["--candidate", NAMES[0], "--fold", "0"]),
        (train.main, ["--version", "v1", "--candidate", "band0-mhd3", *out]),
    )
    for main, args in commands:
        with pytest.raises(SystemExit):
            main(args)
        assert "the one check refused" in capsys.readouterr().err
    assert not (tmp_path / "v1").exists() and not (tmp_path / "v2").exists()


@pytest.fixture
def two_threads():
    """Two torch threads: the default, one per core, crawls on a shared node."""
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


def test_a_folds_saved_p_ae_is_its_restored_best_epochs(tree, monkeypatch, two_threads):
    """The reviewer's Minor 10, with the real `fit` on the synthetic tree: fold
    K's saved P(AE) is its best epoch's model, restored after later epochs
    trained on and predicted otherwise."""
    paths, _ = tree
    _short(monkeypatch, epochs=4)
    made, states = [], []

    class Recorded(train.FrameCNN):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            made.append(self)

    monkeypatch.setattr(train, "FrameCNN", Recorded)

    def log(line):  # after each epoch's validation, before fit keeps it or not
        states.append(copy.deepcopy(made[-1].state_dict()))

    assert cv.main(["--folds"]) == 0
    models = model_dir(paths, "v2")
    record = cv.run_fold(
        paths, models, candidate=NAMES[0], fold=3, version="v2", log=log
    )
    (model,) = made
    best, run = record["best_epoch"], record["epochs_run"]
    assert 1 <= best < run == len(states) == 4
    assert [h["kept"] for h in record["history"]].count(True) >= 1
    band = train.candidates("v2")[NAMES[0]]["band"]

    def p_ae(state, shot):
        model.load_state_dict(state)
        rows = store_rows(paths.spectrogram_file(xpower.EVENT, shot))
        prob, _ = train.probabilities(model, rows, 0, 200, band=band)
        return np.asarray(prob, dtype=np.float32)

    with np.load(models / "cv" / NAMES[0] / "fold3.npz") as z:
        saved = {s: z[f"p{s}"] for s in record["shots"]}
    assert sorted(saved) == [103, 110]
    for shot, p in saved.items():
        np.testing.assert_array_equal(p, p_ae(states[best - 1], shot))
    later = max(np.abs(p - p_ae(states[-1], s)).max() for s, p in saved.items())
    assert later > 0.01  # the last epoch's model predicts otherwise
