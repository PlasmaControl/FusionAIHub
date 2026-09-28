"""v2's test, scored once on its whole split, reports v1's test shots beside it."""

from __future__ import annotations

import json

import numpy as np
import pytest

from labeler.ae.xpower import cv, evaluate, model_dir, train
from labeler.ae.xpower.model import FrameCNN, FrameCNNConfig
from labeler.config import sha256_of

from . import ae_tree
from .test_ae_xpower_cv import POOL, TEST, _all_folds, _designed, cv_tree
from .test_ae_xpower_evaluate import _Fires


def _final(tmp_path, monkeypatch, v1_test=(111,)):
    """v2's final model from its cross-validated choice, and a v1 whose chosen
    model's split tests `v1_test`; the paths."""
    paths, _ = cv_tree(tmp_path, monkeypatch)
    with monkeypatch.context() as mp:
        _designed(mp)
        _all_folds(mp)
        assert cv.main(["--choose"]) == 0
        mp.setattr(
            train,
            "fit",
            lambda s, v, c, log=print: (
                FrameCNN(FrameCNNConfig(width=4)),
                [{"epoch": 1, "val_f1": None, "kept": True}],
                None,
            ),
        )
        assert train.main(["--version", "v2", "--from-cv"]) == 0
    split = {s: "train" for s in POOL} | dict.fromkeys(v1_test, "test")
    ae_tree.chosen(paths, split)
    calls = []

    def seldnet(paths):
        calls.append(1)
        return _Fires(np.ones(783, bool))

    monkeypatch.setattr(evaluate, "load_seldnet", seldnet)
    return paths, calls


def test_the_test_scores_the_whole_split_and_v1s_shots_beside_it(tmp_path, monkeypatch):
    paths, _ = _final(tmp_path, monkeypatch)
    models = model_dir(paths, "v2")
    assert evaluate.main(["--test", "--version", "v2"]) == 0
    record = json.loads((models / "evaluation.json").read_text())
    assert record["frames"]["shots"] == len(TEST)  # the bar's split: all of it
    assert record["bar"] == evaluate.verdict(record)
    subset = record["v1_subset"]
    v1 = model_dir(paths, "v1")
    assert subset["version"] == "v1" and subset["candidate"] == "band80-mhd3"
    assert subset["shots"] == [111] and subset["frames"]["shots"] == 1
    assert subset["split_sha256"] == sha256_of(v1 / "band80-mhd3/split.csv")
    assert subset["chosen_sha256"] == sha256_of(v1 / "chosen.json")
    assert set(subset["methods"]) == set(evaluate.METHODS)
    assert "bar" not in subset  # the bar is judged on the full split only
    choice = sha256_of(models / "cv" / "choice.json")
    assert record["meta"]["version"] == "v2"
    assert record["meta"]["choice_sha256"] == choice
    text = (models / "evaluation.md").read_text()
    whole, beside = text.split("## v1's test shots")
    assert "| ae_xpower |" in whole and "The bar:" in whole
    assert "| ae_xpower |" in beside and "The bar:" not in beside
    assert "1 of v1's test shots" in beside and "not judged" in beside


def test_v1_test_shots_outside_the_test_split_are_refused_before_scoring(
    tmp_path, monkeypatch, capsys
):
    paths, calls = _final(tmp_path, monkeypatch, v1_test=(111, 104))
    with pytest.raises(SystemExit) as error:
        evaluate.main(["--test", "--version", "v2"])
    assert error.value.code != 0
    stderr = capsys.readouterr().err
    assert "104" in stderr and "split.csv" in stderr and "Traceback" not in stderr
    assert calls == []  # nothing was loaded or scored
    assert not (model_dir(paths, "v2") / "evaluation.json").exists()


def test_a_missing_v1_is_refused_before_scoring(tmp_path, monkeypatch, capsys):
    paths, calls = _final(tmp_path, monkeypatch)
    (model_dir(paths, "v1") / "chosen.json").unlink()
    with pytest.raises(SystemExit) as error:
        evaluate.main(["--test", "--version", "v2"])
    assert error.value.code != 0
    assert "v1/chosen.json" in capsys.readouterr().err
    assert calls == [] and not (model_dir(paths, "v2") / "evaluation.json").exists()


def test_the_test_and_the_choice_are_separate_commands(tmp_path, monkeypatch):
    paths = ae_tree.build(tmp_path, {101: "train"})
    ae_tree.env(monkeypatch, paths)
    with pytest.raises(SystemExit) as error:
        evaluate.main(["--test", "--choose"])
    assert error.value.code == 2
    assert not model_dir(paths).exists()
