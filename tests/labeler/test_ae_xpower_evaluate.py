"""Scoring the AE model: cells, the MHD rate, the bar, the choice, the whole run."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from labeler.ae.xpower import evaluate, train
from labeler.ae.xpower.evaluate import ShotFrames
from labeler.ae.xpower.model import FrameCNN

from . import ae_tree


def test_the_false_positive_rate_is_fp_over_fp_and_tn():
    assert evaluate.fp_rate([1, 2, 3, 6]) == pytest.approx(0.25)
    assert np.isnan(evaluate.fp_rate([4, 0, 1, 0]))
    assert evaluate.fp_rate(np.array([[0, 1, 0, 1], [0, 0, 0, 2]])).tolist() == [
        0.5,
        0.0,
    ]


def _frames(shot, owner, mhd, **said):
    owner = np.asarray(owner)
    return ShotFrames(
        shot,
        owner,
        np.asarray(mhd, bool),
        owner >= 0,
        {k: np.asarray(v, bool) for k, v in said.items()},
    )


def test_cells_count_scored_frames_and_a_where_narrows_them():
    f = _frames(1, [1, 1, 0, 0, -1], [0, 0, 1, 0, 0], m=[1, 0, 1, 0, 1])
    assert evaluate.cells([f], "m").tolist() == [[1, 1, 1, 1]]
    assert evaluate.cells([f], "m", evaluate.mhd_absent).tolist() == [[0, 1, 0, 0]]
    assert evaluate.cells([f], "m", evaluate.other_absent).tolist() == [[0, 0, 0, 1]]


def _population(ours_on_mhd: bool):
    """Twelve shots: the model right everywhere (or also firing on MHD), the
    earlier detector firing on MHD, `always` firing everywhere."""
    owner = [1] * 10 + [0] * 10
    mhd = [0] * 10 + [1] * 5 + [0] * 5
    right = owner
    fires_on_mhd = [1] * 15 + [0] * 5
    return [
        _frames(
            s,
            owner,
            mhd,
            ae_xpower=fires_on_mhd if ours_on_mhd else right,
            seldnet=fires_on_mhd,
            always=[1] * 20,
        )
        for s in range(12)
    ]


def test_the_bar_passes_a_model_that_ignores_mhd_and_fails_one_that_does_not():
    good = evaluate.score(_population(False), ("ae_xpower", "seldnet", "always"))
    assert good["methods"]["ae_xpower"]["f1"]["value"] == 1.0
    assert good["methods"]["seldnet"]["fp_rate_mhd"]["value"] == 1.0
    assert good["methods"]["seldnet"]["fp_rate_other"]["value"] == 0.0
    assert good["differences"]["mhd_fp_minus_seldnet"]["high"] == -1.0
    assert good["frames"] == {
        "shots": 12,
        "scored": 240,
        "present": 120,
        "mhd_absent": 60,
        "shots_with_mhd_absent": 12,
    }
    assert evaluate.verdict(good) == {"A1": True, "A2": True, "A3": True, "all": True}
    bad = evaluate.score(_population(True), ("ae_xpower", "seldnet", "always"))
    assert evaluate.verdict(bad) == {"A1": False, "A2": False, "A3": True, "all": False}


def _result(f1, mhd):
    return {"f1": {"value": f1}, "fp_rate_mhd": {"value": mhd}}


def test_the_choice_is_the_lowest_mhd_rate_within_the_f1_margin():
    results = {
        "band80-mhd3": _result(0.93, 0.08),
        "band0-mhd3": _result(0.92, 0.03),
        "band0-mhd10": _result(0.88, 0.01),
    }
    assert evaluate.choose(results)[0] == "band0-mhd3"
    results["band0-mhd3"] = _result(0.92, 0.08)
    assert evaluate.choose(results)[0] == "band80-mhd3"  # a tie: the first listed
    assert evaluate.choose({"band0-mhd10": _result(None, None)})[0] == "band0-mhd10"
    with pytest.raises(ValueError, match="no trained candidate"):
        evaluate.choose({})


class _Fires(torch.nn.Module):
    """A stand-in detector: over 0.5 on the columns it is told to fire on."""

    def __init__(self, fire):
        super().__init__()
        self.fire = torch.as_tensor(fire)

    def forward(self, x):
        logit = torch.where(self.fire, 5.0, -5.0)
        return torch.stack([logit, torch.zeros_like(logit)], dim=-1)[None]


def test_the_earlier_detector_says_a_frame_when_half_its_columns_fire(tmp_path):
    t = 0.5 + np.arange(40)  # 1 ms columns: frames 0-3
    np.savez(tmp_path / "s.npz", spec=np.zeros((4, 348, 40), np.float16))
    fire = np.zeros(40, bool)
    fire[10:16] = True  # six of frame 1's ten columns
    fire[20:24] = True  # four of frame 2's
    said = evaluate.seldnet_said(_Fires(fire), tmp_path / "s.npz", t, 0, 4)
    assert said.tolist() == [False, True, False, False]
    with pytest.raises(ValueError, match="columns"):
        evaluate.seldnet_said(_Fires(fire), tmp_path / "s.npz", t[:30], 0, 4)


def _candidates(paths, split):
    for name in ("band80-mhd3", "band0-mhd3"):
        torch.manual_seed(0)
        spec = train.CANDIDATES[name]
        train.save(
            evaluate.model_dir(paths) / name,
            FrameCNN(),
            threshold=0.5,
            split=split,
            history=[{"epoch": 1, "val_f1": 0.5, "kept": True}],
            config=train.TrainConfig(mhd_weight=spec["mhd_weight"]),
            band_khz=spec["band"],
            labels_file=paths.label_tables / "alfven_eigenmode/review/labels.csv",
            candidate=name,
        )


def test_the_whole_run_chooses_on_validation_then_scores_the_test_shots(
    tmp_path, monkeypatch
):
    splits = {101: "train", 102: "train", 103: "valid", 104: "valid"}
    paths = ae_tree.build(tmp_path, splits)
    split = {101: "train", 102: "val", 103: "test", 104: "test"}
    _candidates(paths, split)
    fire = np.ones(783, bool)
    monkeypatch.setattr(evaluate, "load_seldnet", lambda paths: _Fires(fire))
    models = evaluate.model_dir(paths)
    chosen = evaluate.run_choose(paths, models)
    assert chosen["candidate"] in train.CANDIDATES
    assert set(chosen["validation"]) == {"band80-mhd3", "band0-mhd3"}
    assert (
        json.loads((models / "chosen.json").read_text())["candidate"]
        == chosen["candidate"]
    )
    record = evaluate.run_test(paths, models)
    assert set(record["methods"]) == set(evaluate.METHODS)
    assert record["frames"]["shots"] == 2 and record["frames"]["scored"] == 400
    assert record["frames"]["present"] == 120 and record["frames"]["mhd_absent"] == 60
    for exact in ("source", "uci"):
        assert record["methods"][exact]["f1"]["value"] == 1.0
    assert record["methods"]["tokeye"]["fp_rate_mhd"]["value"] == 1.0  # the harmonic
    assert record["methods"]["always"]["recall"]["value"] == 1.0
    assert record["meta"]["tier"] == "suggestions"
    assert set(record["bar"]) == {"A1", "A2", "A3", "all"}
    text = (models / "evaluation.md").read_text()
    assert "| seldnet |" in text and "Tier: suggestions." in text
