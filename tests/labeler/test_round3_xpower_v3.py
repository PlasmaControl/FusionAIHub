"""ae_xpower v3: TokEye's whole-shot masks, the whole-band candidates, and
cross-validation and the one test over the owner's whole windows."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from labeler.ae import full, xpower
from labeler.ae.xpower import cv, evaluate, model_dir, train
from labeler.ae.xpower.data import FULL_BAND_KHZ, make_split, seldnet_split
from labeler.ae.xpower.model import FrameCNN, FrameCNNConfig
from labeler.config import Paths
from labeler.events.catalog.states import PRESENT

from . import ae_tree, round3_tree
from .round3_tree import AE_FRAMES, MHD_FRAMES
from .test_ae_xpower_cv import POOL, SPLITS, TEST, _fake_fit


def v3_tree(tmp_path, monkeypatch, splits=SPLITS):
    paths = round3_tree.build(tmp_path, splits)
    round3_tree.manifests(paths)  # v3's records name masks-full's and dataset-full's
    ae_tree.env(monkeypatch, paths)
    digest = ae_tree.snapshot(paths, monkeypatch, "v3")
    return paths, digest


def test_whole_window_versions_read_the_whole_shot_records(tmp_path):
    paths = Paths(root=tmp_path)
    assert xpower.WHOLE_WINDOW_VERSIONS == frozenset({"v3"})
    assert "v3" in xpower.CV_VERSIONS and len(xpower.LABEL_SNAPSHOTS["v3"]) == 64
    for version in (None, "v1", "v2"):
        assert xpower.tokeye_masks(paths, version) == tmp_path / "ae" / "masks"
        assert xpower.seldnet_inputs(paths, version) == tmp_path / "ae" / "dataset"
        assert xpower.scored_until(version) == xpower.SCORED_UNTIL_MS == 2000.0
    assert xpower.tokeye_masks(paths) == tmp_path / "ae" / "masks"
    assert xpower.tokeye_masks(paths, "v3") == full.masks_full_dir(paths)
    assert xpower.seldnet_inputs(paths, "v3") == full.dataset_full_dir(paths)
    assert xpower.scored_until("v3") is None
    assert evaluate.SUBSET_OF == {"v2": "v1", "v3": "v2", "v4": "v2"}
    assert evaluate.WHOLE_METHODS == ("ae_xpower", "seldnet", "tokeye", "always")


def test_v3s_candidates_see_the_whole_band():
    got = train.candidates("v3")
    assert list(got) == ["band0-mhd3", "band0-mhd10", "band0-mhd30"]
    assert all(spec["band"] == FULL_BAND_KHZ for spec in got.values())
    assert [spec["mhd_weight"] for spec in got.values()] == [3.0, 10.0, 30.0]
    assert list(train.candidates("v2")) == [
        "band80-mhd3",
        "band80-mhd10",
        "band80-mhd30",
    ]


def test_v3s_folds_come_from_masks_full_alone(tmp_path, monkeypatch):
    paths, digest = v3_tree(tmp_path, monkeypatch)
    shutil.rmtree(paths.root / "ae" / "masks")  # v1's masks are not read
    models = model_dir(paths, "v3")
    assert cv.main(["--version", "v3", "--folds"]) == 0
    split = make_split(POOL + TEST, seldnet_split(full.masks_full_dir(paths)))
    lines = (models / "cv/folds.csv").read_text().splitlines()[1:]
    assert {int(x.split(",")[0]): x.split(",")[1] for x in lines} == split
    assert json.loads((models / "cv/folds.json").read_text())["labels_sha256"] == digest


def test_a_shot_missing_from_masks_full_stops_the_folds(tmp_path, monkeypatch, capsys):
    paths, _ = v3_tree(tmp_path, monkeypatch)
    (full.masks_full_dir(paths) / "105_train_clean.npz").unlink()
    with pytest.raises(SystemExit) as error:
        cv.main(["--version", "v3", "--folds"])
    assert error.value.code != 0
    assert "105" in capsys.readouterr().err
    assert not (model_dir(paths, "v3") / "cv" / "folds.csv").exists()


def test_a_v3_fold_scores_the_whole_window_without_the_source_table(
    tmp_path, monkeypatch
):
    paths, _ = v3_tree(tmp_path, monkeypatch)
    # The source table covers 0-2 s only: v3 neither needs nor reads it.
    for table in (paths.label_tables / "alfven_eigenmode" / "format").glob("*.csv"):
        table.unlink()
    monkeypatch.setattr(train, "fit", _fake_fit([]))
    assert cv.main(["--version", "v3", "--folds"]) == 0
    args = ["--version", "v3", "--candidate", "band0-mhd10", "--fold", "0"]
    assert cv.main(args) == 0
    out = model_dir(paths, "v3") / "cv" / "band0-mhd10"
    record = json.loads((out / "fold0.json").read_text())
    assert record["source_sha256"] is None and record["source_dropped"] == 0
    assert record["band_khz"] == [0.0, 250.0]
    with np.load(out / "fold0.npz") as z:
        for s in record["shots"]:
            assert z[f"p{s}"].shape == (300,)
            assert z[f"s{s}"].all()  # every frame of 0-3 s is scored
            assert np.flatnonzero(z[f"o{s}"] == PRESENT).tolist() == AE_FRAMES
            assert np.flatnonzero(z[f"m{s}"]).tolist() == MHD_FRAMES


def test_whole_frames_score_every_whole_window_method(tmp_path, monkeypatch):
    paths, _ = v3_tree(tmp_path, monkeypatch, {111: "valid"})
    _, saved = xpower.read_snapshot(paths, "v3")
    model = FrameCNN(FrameCNNConfig(width=4))
    blob = {"band_khz": [0.0, 250.0], "threshold": 0.5, "candidate": "band0-mhd3"}
    frames = evaluate.whole_frames(
        111,
        paths=paths,
        label=saved[111],
        model=model,
        blob=blob,
        version="v3",
        seldnet=round3_tree.AllFire(),
        spec_path=xpower.seldnet_inputs(paths, "v3") / "111_valid.npz",
    )
    assert set(frames.said) == set(evaluate.WHOLE_METHODS)
    assert len(frames.owner) == 300 and frames.scored.all()
    # TokEye's AE call is 80-250 kHz, so the MHD mode's 98 kHz harmonic fires it.
    assert np.flatnonzero(frames.said["tokeye"]).tolist() == sorted(
        AE_FRAMES + MHD_FRAMES
    )
    assert np.flatnonzero(frames.mhd).tolist() == MHD_FRAMES
    assert frames.said["seldnet"].all() and frames.said["always"].all()
    assert frames.source_dropped == 0 and frames.prob.shape == (300,)
    early = evaluate.shot_frames(
        111, paths=paths, label=saved[111], model=model, blob=blob
    )
    assert len(early.owner) == 200  # v2's frames stay 0-2 s, from v1's masks


def _designed_whole(monkeypatch):
    """P(AE) over the whole window: every candidate fires on both AE spans at
    0.93, and band0-mhd3 on the MHD frames too, at 0.83."""
    real = evaluate.whole_frames

    def whole_frames(shot, **kw):
        frames = real(shot, **kw)
        prob = np.full(300, 0.02, dtype=np.float32)
        prob[AE_FRAMES] = 0.93
        if kw["blob"]["candidate"] == "band0-mhd3":
            prob[MHD_FRAMES] = 0.83
        frames.prob = prob
        return frames

    monkeypatch.setattr(evaluate, "whole_frames", whole_frames)


def test_v3s_test_is_whole_window_with_v2s_table_and_shots_beside(
    tmp_path, monkeypatch
):
    paths, _ = v3_tree(tmp_path, monkeypatch)
    with monkeypatch.context() as mp:
        _designed_whole(mp)
        mp.setattr(train, "fit", _fake_fit([]))
        assert cv.main(["--version", "v3", "--folds"]) == 0
        for name in train.candidates("v3"):
            for k in range(5):
                task = ["--version", "v3", "--candidate", name, "--fold", str(k)]
                assert cv.main(task) == 0
        assert cv.main(["--version", "v3", "--choose"]) == 0
        mp.setattr(
            train,
            "fit",
            lambda s, v, c, log=print: (
                FrameCNN(FrameCNNConfig(width=4)),
                [{"epoch": 1, "val_f1": None, "kept": True}],
                None,
            ),
        )
        assert train.main(["--version", "v3", "--from-cv"]) == 0
    models = model_dir(paths, "v3")
    choice = json.loads((models / "cv/choice.json").read_text())
    assert choice["candidate"] in train.candidates("v3")
    assert choice["source_sha256"] is None
    assert choice["frames"]["scored"] == len(POOL) * 300
    ae_tree.chosen(
        paths, {s: "train" for s in POOL} | dict.fromkeys(TEST, "test"), "v2"
    )
    monkeypatch.setattr(evaluate, "load_seldnet", lambda paths: round3_tree.AllFire())
    assert evaluate.main(["--test", "--version", "v3"]) == 0
    record = json.loads((models / "evaluation.json").read_text())
    assert list(record["methods"]) == list(evaluate.WHOLE_METHODS)
    assert record["frames"]["shots"] == len(TEST)
    assert record["frames"]["scored"] == 600 and record["frames"]["present"] == 200
    assert record["bar"] == evaluate.verdict(record)
    assert "window" not in record
    early = record["window_0_2s"]
    assert list(early["methods"]) == list(evaluate.METHODS)
    assert early["frames"]["scored"] == 400 and early["frames"]["present"] == 120
    assert record["meta"]["frames_window"] == "whole"
    assert record["meta"]["source_sha256"] == evaluate.source_table(paths).sha256
    subset = record["v2_subset"]
    assert subset["version"] == "v2" and subset["shots"] == TEST
    assert list(subset["methods"]) == list(evaluate.WHOLE_METHODS)
    assert list(subset["window_0_2s"]["methods"]) == list(evaluate.METHODS)
    text = (models / "evaluation.md").read_text()
    whole, beside = text.split("## v2's test shots")
    assert "600 frames of the owner's whole windows" in whole
    assert "## 0-2 s, as v2 was scored" in whole and "The bar:" in whole
    assert "| uci |" in whole and "| uci |" not in beside
