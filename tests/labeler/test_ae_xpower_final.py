"""v2's final model: the cross-validated choice, trained on every pool shot."""

from __future__ import annotations

import contextlib
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from labeler.ae import xpower
from labeler.ae.xpower import cv, evaluate, gallery, model_dir, train
from labeler.config import Paths, sha256_of
from labeler.events.review import labels

from . import ae_tree
from .test_ae_xpower_cv import (
    NAMES,
    POOL,
    TEST,
    _all_folds,
    _designed,
    _fake_fit,
    cv_tree,
)
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


def test_the_final_model_takes_no_epochs(tmp_path, monkeypatch, capsys):
    paths, _ = _chosen_by_cv(tmp_path, monkeypatch)
    models = model_dir(paths, "v2")
    monkeypatch.setattr(train, "fit", _untrained)
    args = ["--version", "v2", "--from-cv", "--epochs", "5"]
    _refused(capsys, train.main, args, "--epochs", "final_epochs")
    assert not (models / "band80-mhd3").exists()
    assert not (models / "chosen.json").exists()


def _final_and_v1(tmp_path, monkeypatch):
    """v2's final model from its choice, and a v1 that tests 111; the paths."""
    paths, _ = _chosen_by_cv(tmp_path, monkeypatch)
    monkeypatch.setattr(train, "fit", _untrained)
    assert train.main(["--version", "v2", "--from-cv"]) == 0
    ae_tree.chosen(paths, {**dict.fromkeys(POOL, "train"), 111: "test"})  # v1
    monkeypatch.setattr(
        evaluate, "load_seldnet", lambda paths: _Fires(np.ones(783, bool))
    )
    return paths


def test_a_copy_of_the_final_model_under_runs_is_never_scored(
    tmp_path, monkeypatch, capsys
):
    """The reviewer's probe 2: a full choice's model copied under runs/ would take
    20-shot looks at the test shots, again and again."""
    paths = _final_and_v1(tmp_path, monkeypatch)
    copy = paths.runs / "ae_xpower" / "pilot" / "v2"
    shutil.copytree(model_dir(paths, "v2"), copy)
    args = ["--test", "--version", "v2", "--models", str(copy), "--limit", "1"]
    _refused(capsys, evaluate.main, args, str(copy / "cv" / "choice.json"), "pilot")
    assert not (copy / "evaluation.json").exists()


def _files(directory: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(directory)): p.read_bytes()
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }


def test_a_copy_of_the_final_model_elsewhere_is_never_tested(
    tmp_path, monkeypatch, capsys
):
    """The reviewer's probe B: the test is taken once per models directory, so a
    copy of v2's outside runs/ would take the one look in full, again and again.
    Its own directory, by any path that resolves to it, is tested once."""
    paths = _final_and_v1(tmp_path, monkeypatch)
    real = model_dir(paths, "v2")
    copy = tmp_path / "elsewhere" / "v2"
    shutil.copytree(real, copy)
    before = _files(copy)
    args = ["--test", "--version", "v2", "--models"]
    own = ("own models directory", str(real))
    _refused(capsys, evaluate.main, [*args, str(copy)], str(copy), *own)
    assert _files(copy) == before
    assert not (real / "evaluation.json").exists()
    link = tmp_path / "link" / "v2"  # the version's own directory, by a link
    link.parent.mkdir()
    link.symlink_to(real, target_is_directory=True)
    assert evaluate.main([*args, str(link)]) == 0
    assert (real / "evaluation.json").is_file()
    _refused(capsys, evaluate.main, [*args, str(real)], "scored once")


def test_a_copy_of_the_final_model_elsewhere_is_never_drawn(
    tmp_path, monkeypatch, capsys
):
    """Probe B's gallery: a copy of v2's models directory outside runs/, made
    after the test (its evaluation names its model), would draw into the
    version's gallery. Only its own directory draws there."""
    paths = _final_and_v1(tmp_path, monkeypatch)
    real = model_dir(paths, "v2")
    assert evaluate.main(["--test", "--version", "v2"]) == 0
    copy = tmp_path / "elsewhere" / "v2"
    shutil.copytree(real, copy)
    before = _files(copy)
    args = ["--version", "v2", "--workers", "1", "--shots", "111"]
    own = ("own models directory", str(real))
    _refused(capsys, gallery.main, [*args, "--models", str(copy)], str(copy), *own)
    assert _files(copy) == before
    assert not gallery.gallery_dir(paths, "v2").exists()
    assert gallery.main(args) == 0
    assert (gallery.gallery_dir(paths, "v2") / "reviewed" / "111.jpg").is_file()


def test_a_pilot_choices_model_is_scored_and_drawn_under_runs(
    tmp_path, monkeypatch, capsys
):
    paths, _ = cv_tree(tmp_path, monkeypatch)
    _designed(monkeypatch)
    monkeypatch.setattr(train, "fit", _fake_fit([]))
    pilot = ["--pilot", "5"]  # one shot a fold
    assert cv.main(["--folds", *pilot]) == 0
    for name in NAMES:
        for k in range(5):
            assert cv.main(["--candidate", name, "--fold", str(k), *pilot]) == 0
    assert cv.main(["--choose", *pilot]) == 0
    monkeypatch.setattr(train, "fit", _untrained)
    assert train.main(["--version", "v2", "--from-cv", "--pilot", "6"]) == 0
    runs = paths.runs / "ae_xpower" / "pilot" / "v2"
    name = json.loads((runs / "chosen.json").read_text())["candidate"]
    split = train.read_split(runs / name / "split.csv")
    assert split == {**dict.fromkeys(POOL[:6], "train"), **dict.fromkeys(TEST, "test")}
    ae_tree.chosen(paths, {**dict.fromkeys(POOL, "train"), 111: "test"})  # v1
    monkeypatch.setattr(
        evaluate, "load_seldnet", lambda paths: _Fires(np.ones(783, bool))
    )
    args = ["--version", "v2", "--models", str(runs)]
    drawn = [*args, "--workers", "1", "--shots", "101", "111"]
    # Its pictures show F1 on test shots: not before the pilot's own scoring.
    _refused(capsys, gallery.main, drawn, str(runs / "evaluation.json"))
    assert evaluate.main(["--choose", *args, "--limit", "1"]) == 0
    assert evaluate.main(["--test", *args, "--limit", "1"]) == 0
    record = json.loads((runs / "evaluation.json").read_text())
    assert record["frames"]["shots"] == 1 and record["meta"]["limit"] == 1
    assert gallery.main(drawn) == 0
    # A pilot's pictures stay under runs/, never in the version's gallery.
    index = pd.read_csv(runs / "gallery" / "index.csv")
    assert sorted(index.shot) == [101, 111]
    assert (runs / "gallery" / "reviewed" / "111.jpg").is_file()
    assert not gallery.gallery_dir(paths, "v2").exists()
    assert not (model_dir(paths, "v2") / "evaluation.json").exists()


def test_the_gallery_of_a_cross_validated_version_waits_for_its_test(
    tmp_path, monkeypatch, capsys
):
    """The reviewer's probe 4: the pictures' F1 vs the owner on the test shots is
    a look at them, so it comes after the one test of the model it draws."""
    paths = _final_and_v1(tmp_path, monkeypatch)
    models = model_dir(paths, "v2")
    evaluation = models / "evaluation.json"
    args = ["--version", "v2", "--workers", "1", "--shots", "101", "111"]
    _refused(capsys, gallery.main, args, str(evaluation), "evaluate --test")
    assert not gallery.gallery_dir(paths, "v2").exists()
    assert evaluate.main(["--test", "--version", "v2"]) == 0
    assert gallery.main(args) == 0
    index = pd.read_csv(gallery.gallery_dir(paths, "v2") / "index.csv")
    assert index.set_index("shot").split.to_dict() == {101: "train", 111: "test"}
    record = json.loads(evaluation.read_text())
    record["meta"]["model_sha256"] = hashlib.sha256(b"another model").hexdigest()
    evaluation.write_text(json.dumps(record))
    _refused(capsys, gallery.main, args, str(evaluation), "another model")


@pytest.mark.parametrize("command", ["--choose", "--test"])
def test_the_checks_compare_the_models_bytes_with_chosen_json(
    tmp_path, monkeypatch, capsys, command
):
    """The reviewer's Minor 4: another model with the same record is refused."""
    paths = _final_and_v1(tmp_path, monkeypatch)
    file = model_dir(paths, "v2") / "band80-mhd3" / "model.pt"
    blob = torch.load(file, map_location="cpu", weights_only=False)
    blob["state_dict"] = {
        k: v + 1 if v.is_floating_point() else v for k, v in blob["state_dict"].items()
    }
    torch.save(blob, file)
    args = [command, "--version", "v2"]
    _refused(capsys, evaluate.main, args, str(file), "model_sha256")
    assert not (model_dir(paths, "v2") / "evaluation.json").exists()


def _crashed(tmp_path, monkeypatch):
    """v2's final model saved, and the job gone before `chosen.json` (the
    reviewer's probe 5); training again fails the test. Its models directory and
    the `chosen.json` it lost."""
    paths, _ = _chosen_by_cv(tmp_path, monkeypatch)
    models = model_dir(paths, "v2")
    monkeypatch.setattr(train, "fit", _untrained)
    assert train.main(["--version", "v2", "--from-cv"]) == 0
    before = json.loads((models / "chosen.json").read_text())
    (models / "chosen.json").unlink()

    def fit(*args, **kwargs):
        raise AssertionError("the final model was trained again")

    monkeypatch.setattr(train, "fit", fit)
    return models, before


def test_a_final_model_without_its_record_gets_the_record_not_a_second_model(
    tmp_path, monkeypatch
):
    models, before = _crashed(tmp_path, monkeypatch)
    file = models / "band80-mhd3" / "model.pt"
    data = file.read_bytes()
    assert train.main(["--version", "v2", "--from-cv"]) == 0
    assert file.read_bytes() == data
    chosen = json.loads((models / "chosen.json").read_text())
    assert {**chosen, "made_at": 0} == {**before, "made_at": 0}
    assert chosen["model_sha256"] == hashlib.sha256(data).hexdigest()
    assert evaluate.main(["--choose", "--version", "v2"]) == 0


@pytest.mark.parametrize(
    "change", ["choice", "from_cv", "choice_sha256", "folds_sha256", "snapshot_sha256"]
)
def test_a_final_model_the_current_choice_did_not_make_is_never_recorded(
    tmp_path, monkeypatch, capsys, change
):
    models, _ = _crashed(tmp_path, monkeypatch)
    file = models / "band80-mhd3" / "model.pt"
    if change == "choice":  # the choice was made again since the model
        choice_file = models / "cv" / "choice.json"
        choice = json.loads(choice_file.read_text())
        choice["made_at"] = "2026-09-29T00:00:00+00:00"
        choice_file.write_text(json.dumps(choice, indent=1) + "\n")
        needle = "choice_sha256"
    else:  # a checkpoint of another choice, folds or snapshot
        blob = torch.load(file, map_location="cpu", weights_only=False)
        other = hashlib.sha256(b"another").hexdigest()
        blob[change] = False if change == "from_cv" else other
        torch.save(blob, file)
        needle = change
    data = file.read_bytes()
    args = ["--version", "v2", "--from-cv"]
    _refused(capsys, train.main, args, str(file), "not replaced", needle)
    assert file.read_bytes() == data
    assert not (models / "chosen.json").exists()


def test_the_model_is_written_last_and_its_record_after_it(tmp_path, monkeypatch):
    """A crash while saving leaves no `model.pt`, so a rerun trains; after it,
    only `chosen.json` is left to write, which a rerun writes."""
    _chosen_by_cv(tmp_path, monkeypatch)
    monkeypatch.setattr(train, "fit", _untrained)
    written, real = [], train.atomic_path

    @contextlib.contextmanager
    def spy(path):
        written.append(Path(path).name)
        with real(path) as tmp:
            yield tmp

    monkeypatch.setattr(train, "atomic_path", spy)
    assert train.main(["--version", "v2", "--from-cv"]) == 0
    assert written == [
        "labels.csv",
        "split.csv",
        "training.json",
        "model.pt",
        "chosen.json",
    ]


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


def test_the_test_scores_with_the_choices_source_table_and_records_it(
    tmp_path, monkeypatch, capsys
):
    """The choice's frames are the test's (the reviewer's Minor 5): the check
    and the test refuse another source table than the one the fold records and
    the choice name, and the test records the one it scored with."""
    paths = _final_and_v1(tmp_path, monkeypatch)
    models = model_dir(paths, "v2")
    file = labels.source_path(xpower.event_dir(paths))
    data = file.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    choice = json.loads((models / "cv" / "choice.json").read_text())
    assert choice["source_sha256"] == digest
    file.write_bytes(data.replace(b"112,0,900,2000,", b"112,0,900,1990,"))
    for command in ("--choose", "--test"):
        args = [command, "--version", "v2"]
        _refused(capsys, evaluate.main, args, str(file), "source_sha256")
    assert not (models / "evaluation.json").exists()
    file.write_bytes(data)
    assert evaluate.main(["--test", "--version", "v2"]) == 0
    meta = json.loads((models / "evaluation.json").read_text())["meta"]
    assert meta["source_sha256"] == digest
