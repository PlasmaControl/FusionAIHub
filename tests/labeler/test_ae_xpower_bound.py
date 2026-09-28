"""No full scoring or extension from runs/, and `--version` is bound to the
models directory's name and the checkpoint's recorded version."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd
import pytest
import torch

from labeler.ae import xpower
from labeler.ae.xpower import (
    evaluate,
    extend,
    frontier,
    gallery,
    model_dir,
    suggestions_dir,
    train,
)
from labeler.config import Paths, sha256_of
from labeler.events.review import labels

from . import ae_tree
from .test_ae_xpower_evaluate import _Fires
from .test_ae_xpower_extend import _approved_model, _three_shots

OTHER = "v9"  # a version chosen on validation, as v1 is


@pytest.fixture(autouse=True)
def _other_version(monkeypatch):
    monkeypatch.setitem(train.CANDIDATES_BY_VERSION, OTHER, train.CANDIDATES)
    monkeypatch.setattr(
        evaluate, "load_seldnet", lambda paths: _Fires(np.ones(783, bool))
    )


def _refused(capsys, main, args, *needles):
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code != 0
    stderr = capsys.readouterr().err
    assert "Traceback" not in stderr
    for needle in needles:
        assert needle in stderr, stderr
    return stderr


def test_a_full_test_scoring_under_runs_is_refused(tmp_path, monkeypatch, capsys):
    paths = ae_tree.build(tmp_path, {101: "train", 102: "valid"})
    ae_tree.env(monkeypatch, paths)
    ae_tree.chosen(paths, {101: "train", 102: "test"})
    pilot = paths.runs / "ae_xpower" / "pilot" / "v1"
    shutil.copytree(model_dir(paths, "v1"), pilot)
    with pytest.raises(ValueError, match="full scoring"):
        evaluate.run_test(paths, pilot)
    _refused(capsys, evaluate.main, ["--models", str(pilot)], str(pilot), "--limit")
    _refused(capsys, evaluate.main, ["--test", "--models", str(pilot), "--limit", "0"])
    assert not (pilot / "evaluation.json").exists()
    # A pilot scoring (--limit N) there is still allowed, and repeatable.
    assert evaluate.main(["--models", str(pilot), "--limit", "1"]) == 0
    assert evaluate.main(["--models", str(pilot), "--limit", "1"]) == 0


def test_a_pilot_scoring_under_runs_is_20_test_shots_at_most(
    tmp_path, monkeypatch, capsys
):
    """The pilot rule's 20 shots: a limit as large as the split would be the
    repeatable full look that `--limit 0` is refused for."""
    paths = ae_tree.build(tmp_path, {101: "train", 102: "valid"})
    ae_tree.env(monkeypatch, paths)
    ae_tree.chosen(paths, {101: "train", 102: "test"})
    pilot = paths.runs / "ae_xpower" / "pilot" / "v1"
    shutil.copytree(model_dir(paths, "v1"), pilot)
    with pytest.raises(ValueError, match="at most 20 test shots"):
        evaluate.run_test(paths, pilot, limit=21)
    args = ["--test", "--models", str(pilot), "--limit", "60"]
    _refused(capsys, evaluate.main, args, str(pilot), "at most 20 test shots")
    assert not (pilot / "evaluation.json").exists()
    assert evaluate.main(["--test", "--models", str(pilot), "--limit", "20"]) == 0


@pytest.mark.parametrize("merging", [False, True])
def test_the_extension_gate_refuses_a_models_directory_under_runs(
    tmp_path, monkeypatch, capsys, merging
):
    paths, models = _three_shots(tmp_path, monkeypatch)
    pilot = paths.runs / "ae_xpower" / "pilot" / "v1"
    shutil.copytree(models, pilot)  # a full, passing evaluation, but under runs/
    with pytest.raises(ValueError, match="runs"):
        if merging:
            extend.merge(paths, models=pilot, of=1)
        else:
            extend.run_shard(paths, models=pilot, k=0, of=1)
    ae_tree.env(monkeypatch, paths)
    args = ["--models", str(pilot), *(["--merge"] if merging else [])]
    _refused(capsys, extend.main, args, str(pilot))
    assert not suggestions_dir(paths).exists()


def _misnamed(paths, approved: bool):
    """v1's model in a directory named for another version."""
    split = {101: "train", 102: "val", 103: "test"}
    models = _approved_model(paths, split) if approved else ae_tree.chosen(paths, split)
    other = models.with_name(OTHER)
    models.rename(other)
    return models, other


COMMANDS = {
    "choose": (evaluate.main, ["--choose"], False),
    "test": (evaluate.main, ["--test"], False),
    "frontier": (frontier.main, [], False),
    "gallery": (gallery.main, ["--workers", "1"], False),
    "shard": (extend.main, [], True),
    "merge": (extend.main, ["--merge"], True),
}


@pytest.mark.parametrize("command", sorted(COMMANDS))
def test_the_version_must_be_the_models_directorys_name(
    tmp_path, monkeypatch, capsys, command
):
    paths = ae_tree.build(tmp_path, {101: "train", 102: "train", 103: "valid"})
    ae_tree.env(monkeypatch, paths)
    main, args, approved = COMMANDS[command]
    split = {101: "train", 102: "val", 103: "test"}
    models = _approved_model(paths, split) if approved else ae_tree.chosen(paths, split)
    before = sorted(p.name for p in models.iterdir())
    args = [*args, "--version", OTHER, "--models", str(models)]
    _refused(capsys, main, args, f"--version {OTHER}", "name v1")
    assert sorted(p.name for p in models.iterdir()) == before
    assert not gallery.gallery_dir(paths, OTHER).exists()
    assert not suggestions_dir(paths, OTHER).exists()


@pytest.mark.parametrize("command", sorted(COMMANDS))
def test_the_version_must_be_the_checkpoints(tmp_path, monkeypatch, capsys, command):
    paths = ae_tree.build(tmp_path, {101: "train", 102: "train", 103: "valid"})
    ae_tree.env(monkeypatch, paths)
    main, args, approved = COMMANDS[command]
    _, models = _misnamed(paths, approved)
    before = sorted(p.name for p in models.iterdir())
    _refused(
        capsys,
        main,
        [*args, "--version", OTHER],
        f"--version {OTHER}",
        "checkpoint's version v1",
        "model.pt",
    )
    assert sorted(p.name for p in models.iterdir()) == before
    assert not gallery.gallery_dir(paths, OTHER).exists()
    assert not suggestions_dir(paths, OTHER).exists()


def test_training_is_bound_to_its_models_directory(tmp_path, monkeypatch, capsys):
    paths = ae_tree.build(tmp_path, {101: "train", 102: "valid"})
    ae_tree.env(monkeypatch, paths)
    out = model_dir(paths, "v1") / "band80-mhd3"
    args = ["--version", OTHER, "--candidate", "band80-mhd3", "--out", str(out)]
    _refused(capsys, train.main, args, f"--version {OTHER}", "name v1")
    assert not out.exists()


def test_a_checkpoint_without_a_version_is_v1s(tmp_path):
    models = tmp_path / "v1"
    xpower.check_bound("v1", models, {"candidate": "band80-mhd3"}, models / "x")
    with pytest.raises(ValueError, match="checkpoint's version v1"):
        xpower.check_bound("v2", tmp_path / "v2", {}, models / "x")
    xpower.check_bound("v2", tmp_path / "v2", {"version": "v2"}, models / "x")


def test_the_training_pilot_is_under_its_versions_pilot_directory(
    tmp_path, monkeypatch
):
    paths = ae_tree.build(tmp_path, {101: "train", 102: "train", 103: "valid"})
    ae_tree.env(monkeypatch, paths)
    monkeypatch.setattr(
        train,
        "fit",
        lambda *a, **kw: (train.FrameCNN(train.FrameCNNConfig(width=4)), [], 0.5),
    )
    for version in ("v1", OTHER):
        args = ["--version", version, "--candidate", "band80-mhd3", "--pilot", "6"]
        assert train.main(args) == 0
        out = paths.runs / "ae_xpower" / "pilot" / version / "band80-mhd3"
        assert train.load(out / "model.pt")[1]["version"] == version
    assert not paths.models.exists()


def _real_pictures(tmp_path, monkeypatch):
    paths, models = _three_shots(tmp_path, monkeypatch)
    monkeypatch.setattr(extend, "run_all", gallery.run_all)
    ae_tree.env(monkeypatch, paths)
    ae_tree.corpus(tmp_path, [201, 202, 203])
    return paths, models


def test_the_extension_pilot_draws_under_its_own_directory(tmp_path, monkeypatch):
    paths, models = _real_pictures(tmp_path, monkeypatch)
    torch.manual_seed(0)
    extend.run_shard(paths, models=models, k=0, of=2, limit=1)
    pilot = suggestions_dir(paths) / "shards" / "pilot"
    assert (pilot / "extension" / "201.jpg").is_file()
    assert json.loads((pilot / "0.json").read_text())["pictures"] == [201]
    assert not (gallery.gallery_dir(paths) / "extension").exists()


def test_merge_indexes_only_the_merged_shards_pictures(tmp_path, monkeypatch):
    paths, models = _real_pictures(tmp_path, monkeypatch)
    extend.run_shard(paths, models=models, k=0, of=2)  # 201, 203, with pictures
    extend.run_shard(paths, models=models, k=1, of=2, pictures=False)  # 202
    stale = gallery.gallery_dir(paths) / "extension" / "202.jpg"
    stale.write_bytes(b"a picture no merged shard drew")
    manifest = suggestions_dir(paths) / "shards" / "1.json"
    assert json.loads(manifest.read_text())["pictures"] == []
    merged = extend.merge(paths, models=models, of=2)
    index = pd.read_csv(gallery.gallery_dir(paths) / "index.csv")
    assert index.shot.tolist() == [201, 203] and merged["pictures"] == 2


def test_a_snapshot_versions_gallery_reads_only_its_snapshot(tmp_path, monkeypatch):
    paths = ae_tree.build(tmp_path, {101: "train", 102: "train", 103: "valid"})
    ae_tree.env(monkeypatch, paths)
    digest = ae_tree.snapshot(paths, monkeypatch)
    ae_tree.chosen(paths, {101: "train", 102: "val", 103: "test"}, "v2")
    live = labels.labels_path(xpower.event_dir(paths))
    live.write_bytes(b"the owner kept saving")  # not v2's labels
    # v2's pictures come after its test: the record of it, as evaluate writes it.
    models = model_dir(paths, "v2")
    meta = {"model_sha256": sha256_of(models / "band80-mhd3" / "model.pt")}
    (models / "evaluation.json").write_text(json.dumps({"meta": meta}))
    assert gallery.main(["--version", "v2", "--workers", "1"]) == 0
    index = pd.read_csv(gallery.gallery_dir(paths, "v2") / "index.csv")
    assert set(index.group) == {"reviewed"} and len(index) == 3
    assert set(index.snapshot_sha256) == {digest}  # the reviewer's Minor 11


def test_a_gallery_from_a_models_directory_under_runs_draws_under_runs(
    tmp_path, monkeypatch
):
    """The reviewer's Minor 6: a pilot's pictures never overwrite the version's."""
    paths = ae_tree.build(tmp_path, {101: "train", 102: "valid"})
    ae_tree.env(monkeypatch, paths)
    ae_tree.chosen(paths, {101: "train", 102: "test"})
    pilot = paths.runs / "ae_xpower" / "pilot" / "v1"
    shutil.copytree(model_dir(paths, "v1"), pilot)
    assert gallery.main(["--models", str(pilot), "--workers", "1"]) == 0
    assert (pilot / "gallery" / "reviewed" / "101.jpg").is_file()
    assert sorted(pd.read_csv(pilot / "gallery" / "index.csv").shot) == [101, 102]
    assert not gallery.gallery_dir(paths, "v1").exists()


@pytest.mark.parametrize("module", [evaluate, gallery, extend])
def test_the_models_help_names_the_versions_directory(
    tmp_path, monkeypatch, capsys, module
):
    ae_tree.env(
        monkeypatch,
        Paths(root=tmp_path / "root", corpus=tmp_path, label_tables=tmp_path),
    )
    with pytest.raises(SystemExit) as error:
        module.main(["--help"])
    assert error.value.code == 0
    text = " ".join(capsys.readouterr().out.split())
    assert "models/ae_xpower/<version>" in text and "ae_xpower/v1" not in text
