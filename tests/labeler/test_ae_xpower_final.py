"""v2's final model: the cross-validated choice, trained on every pool shot."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from labeler.ae import xpower
from labeler.ae.xpower import cv, evaluate, model_dir, train
from labeler.config import Paths, sha256_of
from labeler.events.review import labels

from . import ae_tree
from .test_ae_xpower_cv import POOL, TEST, _all_folds, _designed, cv_tree
from .test_ae_xpower_evaluate import _Fires
from .test_ae_xpower_model import _toy


def _untrained(train_shots, val_shots, config, log=print):
    """`train.fit` for the final model: every epoch kept, no threshold."""
    history = [
        {"epoch": e, "val_f1": None, "kept": True} for e in range(1, config.epochs + 1)
    ]
    return train.FrameCNN(train.FrameCNNConfig(width=4)), history, None


def test_fit_without_validation_keeps_every_epoch_and_never_stops():
    rng = np.random.default_rng(2)
    config = train.TrainConfig(
        epochs=3, batch=8, crop_frames=32, crops_per_shot=2, patience=1
    )
    _, history, threshold = train.fit(_toy(4, rng), [], config, log=lambda m: 0)
    assert [row["epoch"] for row in history] == [1, 2, 3]
    assert all(row["kept"] and row["val_f1"] is None for row in history)
    assert threshold is None and train.best_epoch(history) == 3


def _chosen_by_cv(tmp_path, monkeypatch):
    paths, digest = cv_tree(tmp_path, monkeypatch)
    _designed(monkeypatch)
    _all_folds(monkeypatch)
    assert cv.main(["--choose"]) == 0
    return paths, digest


def test_the_final_model_is_the_choice_on_every_pool_shot(tmp_path, monkeypatch):
    paths, digest = _chosen_by_cv(tmp_path, monkeypatch)
    models = model_dir(paths, "v2")
    choice_file = models / "cv" / "choice.json"
    calls = []

    def fit(train_shots, val_shots, config, log=print):
        calls.append((sorted(s.shot for s in train_shots), list(val_shots), config))
        history = [
            {"epoch": e, "val_f1": None, "kept": True}
            for e in range(1, config.epochs + 1)
        ]
        return train.FrameCNN(train.FrameCNNConfig(width=4)), history, None

    monkeypatch.setattr(train, "fit", fit)
    assert train.main(["--version", "v2", "--from-cv"]) == 0
    ((shots, val, config),) = calls
    assert shots == POOL and val == []  # all of the pool; nothing stops it
    assert config.epochs == 4 and config.mhd_weight == 3.0  # the median; mhd3
    out = models / "band80-mhd3"
    _, blob = train.load(out / "model.pt")
    choice_sha = sha256_of(choice_file)
    assert blob["candidate"] == "band80-mhd3" and blob["version"] == "v2"
    assert blob["threshold"] == 0.85 and blob["from_cv"] is True
    assert blob["choice_sha256"] == choice_sha
    assert blob["snapshot_sha256"] == blob["labels_sha256"] == digest
    assert blob["fixed_epochs"] == 4 and blob["cv_branch"] == 1
    snapshot = xpower.snapshot_file(paths, "v2").read_bytes()
    assert labels.labels_path(out).read_bytes() == snapshot
    split = train.read_split(out / "split.csv")
    assert split == {**dict.fromkeys(POOL, "train"), **dict.fromkeys(TEST, "test")}
    record = json.loads((out / "training.json").read_text())
    assert record["choice_sha256"] == choice_sha and record["best_epoch"] == 4
    assert record["counts"] == {"test": 2, "train": 10}
    chosen = json.loads((models / "chosen.json").read_text())
    assert chosen["candidate"] == "band80-mhd3" and chosen["version"] == "v2"
    assert chosen["threshold"] == 0.85 and chosen["choice_sha256"] == choice_sha
    assert chosen["labels_sha256"] == digest
    assert chosen["model_sha256"] == sha256_of(out / "model.pt")
    folds_sha = sha256_of(models / "cv" / "folds.csv")
    assert blob["folds_sha256"] == chosen["folds_sha256"] == folds_sha
    assert evaluate.chosen_model(models) == out / "model.pt"
    # `evaluate --choose` on a cross-validated version checks, and writes nothing.
    before = (models / "chosen.json").read_bytes()
    assert evaluate.main(["--choose", "--version", "v2"]) == 0
    assert (models / "chosen.json").read_bytes() == before
    # The final model is trained once.
    with pytest.raises(SystemExit):
        train.main(["--version", "v2", "--from-cv"])
    assert len(calls) == 1


def test_a_changed_choice_is_refused_by_the_check(tmp_path, monkeypatch, capsys):
    paths, _ = _chosen_by_cv(tmp_path, monkeypatch)
    monkeypatch.setattr(
        train,
        "fit",
        lambda s, v, c, log=print: (train.FrameCNN(), [{"epoch": 1, "kept": 1}], None),
    )
    assert train.main(["--version", "v2", "--from-cv"]) == 0
    file = model_dir(paths, "v2") / "cv" / "choice.json"
    choice = json.loads(file.read_text())
    choice["threshold"] = 0.5
    file.write_text(json.dumps(choice))
    with pytest.raises(SystemExit) as error:
        evaluate.main(["--choose", "--version", "v2"])
    assert error.value.code != 0
    stderr = capsys.readouterr().err
    assert "chosen.json" in stderr and "Traceback" not in stderr


def _swap(paths):
    """TokEye's split names of pool shot 101 and test shot 111 trade places, and
    back when called again (the reviewer's probe 3): the count is kept."""
    root = paths.root / "ae"
    for sub, suffix in (("masks", "_clean.npz"), ("dataset", ".npz")):
        now = {s: next((root / sub).glob(f"{s}_*{suffix}")) for s in (101, 111)}
        split = {s: p.name.removesuffix(suffix).split("_")[1] for s, p in now.items()}
        for s, other in ((101, 111), (111, 101)):
            now[s].rename(root / sub / f"{s}_{split[other]}{suffix}")


def _refused(capsys, main, args, *needles):
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code != 0
    stderr = capsys.readouterr().err
    assert "Traceback" not in stderr
    for needle in needles:
        assert needle in stderr, stderr


def test_a_split_changed_after_the_choice_stops_the_final_and_the_test(
    tmp_path, monkeypatch, capsys
):
    paths, _ = _chosen_by_cv(tmp_path, monkeypatch)
    models = model_dir(paths, "v2")
    folds = str(models / "cv" / "folds.csv")
    monkeypatch.setattr(train, "fit", _untrained)
    _swap(paths)  # 101 would be a test shot, 111 a pool shot
    _refused(capsys, train.main, ["--version", "v2", "--from-cv"], folds)
    assert not (models / "band80-mhd3").exists()
    assert not (models / "chosen.json").exists()
    _swap(paths)  # back: the final trains on exactly the folds' shots
    assert train.main(["--version", "v2", "--from-cv"]) == 0
    split = train.read_split(models / "band80-mhd3" / "split.csv")
    assert split == {**dict.fromkeys(POOL, "train"), **dict.fromkeys(TEST, "test")}
    ae_tree.chosen(paths, {**dict.fromkeys(POOL, "train"), 111: "test"})  # v1
    monkeypatch.setattr(
        evaluate, "load_seldnet", lambda paths: _Fires(np.ones(783, bool))
    )
    _swap(paths)  # and between the final and the test
    _refused(capsys, evaluate.main, ["--test", "--version", "v2"], folds)
    assert not (models / "evaluation.json").exists()
    _swap(paths)
    assert evaluate.main(["--test", "--version", "v2"]) == 0


def test_a_choice_made_from_other_folds_is_refused(tmp_path, monkeypatch, capsys):
    paths, _ = _chosen_by_cv(tmp_path, monkeypatch)
    models = model_dir(paths, "v2")
    file = models / "cv" / "choice.json"
    choice = json.loads(file.read_text())
    choice["folds_sha256"] = hashlib.sha256(b"other folds").hexdigest()
    file.write_text(json.dumps(choice))
    monkeypatch.setattr(train, "fit", _untrained)
    args = ["--version", "v2", "--from-cv"]
    _refused(capsys, train.main, args, str(file), "folds.csv")
    assert not (models / "band80-mhd3").exists()
    assert not (models / "chosen.json").exists()


@pytest.mark.parametrize("change", ["chosen", "split"])
def test_the_test_checks_the_final_model_against_the_folds(
    tmp_path, monkeypatch, capsys, change
):
    paths, _ = _chosen_by_cv(tmp_path, monkeypatch)
    models = model_dir(paths, "v2")
    monkeypatch.setattr(train, "fit", _untrained)
    assert train.main(["--version", "v2", "--from-cv"]) == 0
    ae_tree.chosen(paths, {**dict.fromkeys(POOL, "train"), 111: "test"})  # v1
    monkeypatch.setattr(
        evaluate, "load_seldnet", lambda paths: _Fires(np.ones(783, bool))
    )
    if change == "chosen":
        file = models / "chosen.json"
        chosen = json.loads(file.read_text())
        chosen["folds_sha256"] = hashlib.sha256(b"other folds").hexdigest()
        file.write_text(json.dumps(chosen))
        needles = (str(file), "folds_sha256")
        _refused(capsys, evaluate.main, ["--choose", "--version", "v2"], *needles)
    else:
        file = models / "band80-mhd3" / "split.csv"
        split = train.read_split(file) | {101: "test"}  # a pool shot to test
        lines = ["shot,split"] + [f"{s},{v}" for s, v in sorted(split.items())]
        file.write_text("\n".join(lines) + "\n")
        needles = (str(file), "folds")
    _refused(capsys, evaluate.main, ["--test", "--version", "v2"], *needles)
    assert not (models / "evaluation.json").exists()


def _env(tmp_path, monkeypatch) -> Paths:
    paths = ae_tree.build(tmp_path, {101: "train", 102: "valid"})
    ae_tree.env(monkeypatch, paths)
    ae_tree.snapshot(paths, monkeypatch)
    return paths


@pytest.mark.parametrize(
    "args, message",
    [
        (["--version", "v2", "--from-cv"], "choice.json"),
        (["--version", "v1", "--from-cv"], "cross-validation"),
        (["--version", "v2", "--from-cv", "--candidate", "band80-mhd3"], "choice"),
        (["--version", "v2", "--candidate", "band80-mhd3"], "--from-cv"),
    ],
)
def test_the_final_model_needs_a_choice_of_a_cross_validated_version(
    tmp_path, monkeypatch, capsys, args, message
):
    paths = _env(tmp_path, monkeypatch)
    with pytest.raises(SystemExit) as error:
        train.main(args)
    assert error.value.code != 0
    stderr = capsys.readouterr().err
    assert message in stderr and "Traceback" not in stderr
    assert not (model_dir(paths, "v2") / "band80-mhd3").exists()
    assert not (model_dir(paths, "v2") / "chosen.json").exists()


def test_a_choice_from_another_snapshot_is_refused(tmp_path, monkeypatch, capsys):
    paths = _env(tmp_path, monkeypatch)
    file = model_dir(paths, "v2") / "cv" / "choice.json"
    file.parent.mkdir(parents=True)
    other = hashlib.sha256(b"other labels").hexdigest()
    file.write_text(
        json.dumps({"version": "v2", "labels_sha256": other, "candidate": "x"})
    )
    with pytest.raises(SystemExit) as error:
        train.main(["--version", "v2", "--from-cv"])
    assert error.value.code != 0
    stderr = capsys.readouterr().err
    assert str(file) in stderr and "snapshot" in stderr
    assert not (model_dir(paths, "v2") / "chosen.json").exists()
