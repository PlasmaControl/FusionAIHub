"""Pilot lifecycles are repeatable only inside the temporary runs tree."""

import json
from functools import partial

import numpy as np
import pytest
import torch

from labeler.ae.seg import evaluate as seg_evaluate
from labeler.ae.seg import pseudo
from labeler.ae.seg import train as seg_train
from labeler.ae.xpower import evaluate as xp_evaluate
from labeler.ae.xpower import train as xp_train
from labeler.config import Paths, sha256_of

from . import ae_tree
from .test_ae_xpower_evaluate import _Fires


@pytest.mark.parametrize("kind", ["seg", "xpower"])
def test_script_pilot_sequence_runs_twice(tmp_path, monkeypatch, kind):
    paths = ae_tree.build(tmp_path, {101: "train", 102: "train", 103: "valid"})
    ae_tree.env(monkeypatch, paths)
    if kind == "seg":
        ae_tree.chosen(paths, {101: "train", 102: "val", 103: "test"})
        assert pseudo.main([]) == 0
        train, evaluate = seg_train, seg_evaluate
        factory = partial(train.SegNet, train.SegNetConfig(width=4))
        train_args = []
    else:
        train, evaluate = xp_train, xp_evaluate
        factory = partial(train.FrameCNN, train.FrameCNNConfig(width=4))
        train_args = ["--candidate", "band80-mhd3"]
        monkeypatch.setattr(
            evaluate, "load_seldnet", lambda paths: _Fires(np.ones(783, bool))
        )
    models = paths.runs / f"ae_{kind}" / "pilot"
    records = []
    for attempt, threshold in enumerate((0.4, 0.6)):
        torch.manual_seed(attempt)
        monkeypatch.setattr(
            train,
            "fit",
            lambda *a, threshold=threshold, **kw: (factory(), [], threshold),
        )
        assert train.main([*train_args, "--pilot", "6"]) == 0
        args = ["--models", str(models), "--limit", "3"]
        if kind == "xpower":
            assert evaluate.main([*args, "--choose"]) == 0
        assert evaluate.main(args) == 0
        record = json.loads((models / "evaluation.json").read_text())
        file = models / "model.pt" if kind == "seg" else evaluate.chosen_model(models)
        assert record["meta"]["model_sha256"] == sha256_of(file)
        assert record["meta"]["threshold"] == threshold
        records.append(record)
    assert records[0]["meta"]["model_sha256"] != records[1]["meta"]["model_sha256"]


@pytest.mark.parametrize("train", [seg_train, xp_train])
@pytest.mark.parametrize("escape", ["models", "sibling", "symlink"])
def test_pilot_output_outside_runs_is_refused(
    tmp_path, monkeypatch, capsys, train, escape
):
    paths = Paths(
        root=tmp_path / "root",
        label_tables=tmp_path / "events",
        corpus=tmp_path / "corpus",
    )
    ae_tree.env(monkeypatch, paths)
    out = paths.models / "candidate"
    out.mkdir(parents=True)
    if escape == "sibling":
        out = paths.root / "runs-other"
        out.mkdir()
    elif escape == "symlink":
        paths.runs.mkdir()
        link = paths.runs / "escape"
        link.symlink_to(out, target_is_directory=True)
        out = link
    checkpoint = out / "model.pt"
    checkpoint.write_bytes(b"preserve")
    args = ["--pilot", "6", "--out", str(out)]
    if train is xp_train:
        args += ["--candidate", "band80-mhd3"]
    with pytest.raises(SystemExit) as error:
        train.main(args)
    assert error.value.code != 0
    stderr = capsys.readouterr().err
    assert str(out) in stderr and "Traceback" not in stderr
    assert checkpoint.read_bytes() == b"preserve"


@pytest.mark.parametrize(
    "evaluate,choose",
    [(seg_evaluate, False), (xp_evaluate, False), (xp_evaluate, True)],
)
def test_limited_scoring_outside_runs_is_refused(
    tmp_path, monkeypatch, capsys, evaluate, choose
):
    paths = Paths(
        root=tmp_path / "root",
        label_tables=tmp_path / "events",
        corpus=tmp_path / "corpus",
    )
    ae_tree.env(monkeypatch, paths)
    models = paths.models / "candidate"
    run = evaluate.run_choose if choose else evaluate.run_test
    with pytest.raises(ValueError, match="limit") as error:
        run(paths, models, limit=3)
    assert str(models) in str(error.value)
    args = ["--models", str(models), "--limit", "3"]
    with pytest.raises(SystemExit) as error:
        evaluate.main(args + (["--choose"] if choose else []))
    assert error.value.code != 0
    stderr = capsys.readouterr().err
    assert str(models) in stderr and "Traceback" not in stderr
