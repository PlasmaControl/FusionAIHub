"""v4, the AE models on 60-250 kHz: ae_xpower v4 (v2 with the band at 60-250 kHz)
and SegNet v4 (SegNet v1 with the band at 60-250 kHz), on the synthetic AE tree."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from labeler.ae import seg, xpower
from labeler.ae.seg import evaluate as seg_evaluate
from labeler.ae.seg import poi, pseudo
from labeler.ae.seg import train as seg_train
from labeler.ae.seg.pseudo import IGNORE
from labeler.ae.xpower import cv, evaluate, model_dir, train
from labeler.ae.xpower.data import BAND60_KHZ, band_slice, clean_path, tokeye_clean
from labeler.ae.xpower.model import FrameCNN, FrameCNNConfig
from labeler.config import sha256_of
from labeler.events.review import labels
from labeler.events.review.rows import Grid
from labeler.paper import build as paper_build
from labeler.paper import scores as paper_scores

from . import ae_tree
from .test_ae_seg_train import Small
from .test_ae_xpower_cv import NAMES, POOL, TEST, _designed, _fake_fit, cv_tree
from .test_ae_xpower_evaluate import _Fires


def _refused(main, argv, capsys) -> str:
    with pytest.raises(SystemExit) as error:
        main(argv)
    assert error.value.code != 0
    stderr = capsys.readouterr().err
    assert "Traceback" not in stderr
    return stderr


def test_v4s_candidates_bands_and_snapshot_are_v2s_but_the_band():
    assert BAND60_KHZ == (60.0, 250.0)
    got = train.candidates("v4")
    assert list(got) == ["band60-mhd3", "band60-mhd10", "band60-mhd30"]
    assert [c["band"] for c in got.values()] == [BAND60_KHZ] * 3
    v2 = train.candidates("v2")
    assert [c["mhd_weight"] for c in got.values()] == [
        c["mhd_weight"] for c in v2.values()
    ]
    assert xpower.LABEL_SNAPSHOTS["v4"] == xpower.LABEL_SNAPSHOTS["v2"]
    assert "v4" in xpower.CV_VERSIONS and "v4" not in xpower.WHOLE_WINDOW_VERSIONS
    assert xpower.scored_until("v4") == xpower.SCORED_UNTIL_MS
    assert xpower.TEST_OF == {"v4": "v2"} and evaluate.SUBSET_OF["v4"] == "v2"
    assert xpower.VERSION == "v1" and seg.VERSION == "v1"  # the defaults stay
    assert "v4" not in cv.ABLATIONS


def test_the_reuse_notes():
    assert xpower.reuse_note("v2") is None and xpower.reuse_note("v3") is None
    note = xpower.reuse_note("v4", 60)
    assert "ae_xpower v4's test is a second use of ae_xpower v2's 60 test shots" in note
    assert (
        "the 60-250 kHz band" in note and "models/ae_xpower/v2/evaluation.json" in note
    )
    assert "not an unbiased" in note and note.endswith("Tier: suggestions.")
    v4 = seg.SEG_VERSIONS["v4"]
    v1 = seg.SEG_VERSIONS["v1"]
    assert (v4.pseudo, v4.band_khz, v4.test_of) == ("pseudo-v4", BAND60_KHZ, "v1")
    assert (v4.labels, v4.ae_version, v4.whole_window, v4.gated) == (
        v1.labels,
        v1.ae_version,
        v1.whole_window,
        v1.gated,
    )
    assert seg.reuse_note("v1") is None and seg.reuse_note("v2", 60) is None
    # v3's note is as it was.
    assert seg.reuse_note("v3", 60) == (
        "SegNet v3's test is a second use of SegNet v2's 60 test shots, and its "
        "design (pseudo-v3's markers) was made after SegNet v2's test breakdown "
        "had been seen (models/ae_seg/v2/diagnosis.md): its test scores are not "
        "an unbiased estimate. Tier: suggestions."
    )
    note = seg.reuse_note("v4", 58)
    assert "SegNet v4's test is a second use of SegNet v1's 58 test shots" in note
    assert "the 60-250 kHz band" in note and "models/ae_seg/v1/evaluation.json" in note


def _v4_tree(tmp_path, monkeypatch):
    """cv_tree with v4's snapshot too: the same bytes and sha256 as v2's."""
    paths, digest = cv_tree(tmp_path, monkeypatch)
    assert ae_tree.snapshot(paths, monkeypatch, "v4") == digest
    return paths, digest


def test_v4s_snapshot_is_bound_and_its_folds_must_be_v2s(tmp_path, monkeypatch, capsys):
    paths, _ = _v4_tree(tmp_path, monkeypatch)
    v2_folds = model_dir(paths, "v2") / "cv" / "folds.csv"
    stderr = _refused(cv.main, ["--version", "v4", "--folds"], capsys)
    assert "v4's folds must be version v2's" in stderr
    assert cv.main(["--folds"]) == 0
    good = v2_folds.read_bytes()
    v2_folds.write_bytes(good.replace(b"\n", b"\r\n", 1))
    stderr = _refused(cv.main, ["--version", "v4", "--folds"], capsys)
    assert "missing or differ" in stderr
    assert not (model_dir(paths, "v4") / "cv" / "folds.csv").exists()
    v2_folds.write_bytes(good)
    assert cv.main(["--version", "v4", "--folds"]) == 0
    assert (model_dir(paths, "v4") / "cv" / "folds.csv").read_bytes() == good
    # Another snapshot is refused, as for v2.
    xpower_file = xpower.snapshot_file(paths, "v4")
    xpower_file.write_bytes(xpower_file.read_bytes() + b"\n")
    task = ["--version", "v4", "--candidate", NAMES[0], "--fold", "0"]
    assert "is not one of version v4's" in _refused(cv.main, task, capsys)
    task[3] = "band60-mhd3"
    assert "sha256" in _refused(cv.main, task, capsys)


def _files(directory):
    return {
        str(p.relative_to(directory)): sha256_of(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }


def _v4_final(tmp_path, monkeypatch, v2_test=TEST):
    """v4 cross-validated, chosen and trained, beside a v2 whose chosen model
    tests `v2_test`; the paths."""
    paths, _ = _v4_tree(tmp_path, monkeypatch)
    with monkeypatch.context() as mp:
        _designed(mp)
        mp.setattr(train, "fit", _fake_fit([]))
        assert cv.main(["--folds"]) == 0
        assert cv.main(["--version", "v4", "--folds"]) == 0
        for name in train.candidates("v4"):
            for k in range(5):
                task = ["--version", "v4", "--candidate", name, "--fold", str(k)]
                assert cv.main(task) == 0
        assert cv.main(["--version", "v4", "--choose"]) == 0
        mp.setattr(
            train,
            "fit",
            lambda s, v, c, log=print: (
                FrameCNN(FrameCNNConfig(width=4)),
                [{"epoch": 1, "val_f1": None, "kept": True}],
                None,
            ),
        )
        assert train.main(["--version", "v4", "--from-cv"]) == 0
    split = {s: "train" for s in POOL} | dict.fromkeys(v2_test, "test")
    ae_tree.chosen(paths, split, "v2")
    monkeypatch.setattr(evaluate, "load_seldnet", lambda p: _Fires(np.ones(783, bool)))
    return paths


def test_v4_runs_end_to_end_like_v2_and_carries_its_reuse_note(
    tmp_path, monkeypatch, capsys
):
    paths = _v4_final(tmp_path, monkeypatch)
    models = model_dir(paths, "v4")
    chosen = json.loads((models / "chosen.json").read_text())
    name = chosen["candidate"]
    assert name in train.candidates("v4")
    note = xpower.reuse_note("v4", len(TEST))
    assert chosen["test_reuse"] == note
    trained = json.loads((models / name / "training.json").read_text())
    assert trained["test_reuse"] == note
    _, blob = train.load(models / name / "model.pt")
    assert tuple(blob["band_khz"]) == BAND60_KHZ and blob["version"] == "v4"
    assert blob["test_reuse"] == note
    # v2's own records: a first scoring there is still v2's, and never repeated.
    v2 = model_dir(paths, "v2")
    scored = {"meta": {"made_at": "2026-09-28T00:00:00+00:00"}}
    (v2 / "evaluation.json").write_text(json.dumps(scored) + "\n")
    before = _files(v2)
    assert evaluate.main(["--test", "--version", "v4"]) == 0
    assert _files(v2) == before
    record = json.loads((models / "evaluation.json").read_text())
    assert record["meta"]["version"] == "v4" and record["meta"]["test_reuse"] == note
    assert record["frames"]["shots"] == len(TEST)
    subset = record["v2_subset"]
    assert subset["version"] == "v2" and subset["shots"] == TEST
    text = (models / "evaluation.md").read_text()
    assert note in text
    look = evaluate.second_look("v4", TEST, scored, {s: "test" for s in TEST}, chosen)
    assert look["version"] == "v2" and look["shots"] == look["of"] == len(TEST)
    stderr = _refused(evaluate.main, ["--test", "--version", "v2"], capsys)
    assert "scored once" in stderr
    assert _files(v2) == before
    stderr = _refused(evaluate.main, ["--test", "--version", "v4"], capsys)
    assert "scored once" in stderr


def test_v4_refuses_test_shots_other_than_v2s(tmp_path, monkeypatch, capsys):
    paths = _v4_final(tmp_path, monkeypatch, v2_test=TEST[:1])
    stderr = _refused(evaluate.main, ["--test", "--version", "v4"], capsys)
    assert "must be exactly" in stderr
    assert not (model_dir(paths, "v4") / "evaluation.json").exists()


def test_pseudo_v4_is_pseudo_v1_with_the_band_at_60_khz(tmp_path, monkeypatch):
    paths = ae_tree.build(tmp_path, {101: "train", 102: "valid"}, tokeye_dt=0.256)
    ae_tree.env(monkeypatch, paths)
    assert pseudo.main([]) == 0
    assert pseudo.main(["--version", "v4"]) == 0
    v1_dir, v4_dir = seg.pseudo_dir(paths), seg.pseudo_dir(paths, "v4")
    assert v4_dir.name == "pseudo-v4"
    v1_meta = json.loads((v1_dir / "meta.json").read_text())
    v4_meta = json.loads((v4_dir / "meta.json").read_text())
    assert "band_khz" not in v1_meta and v1_meta["pseudo"] == "pseudo-v1"
    assert v4_meta["pseudo"] == "pseudo-v4" and v4_meta["band_khz"] == [60.0, 250.0]
    assert v4_meta["labels_sha256"] == v1_meta["labels_sha256"]
    for shot in (101, 102):
        a = pseudo.PseudoMask.load(v1_dir / f"{shot}.npz")
        b = pseudo.PseudoMask.load(v4_dir / f"{shot}.npz")
        n_y = a.mask.shape[0]
        at80 = band_slice(a.y0_khz, a.dy_khz, n_y, (80.0, 250.0)).start
        at60 = band_slice(a.y0_khz, a.dy_khz, n_y, BAND60_KHZ).start
        assert at60 < at80
        assert (b.mask[at80:] == a.mask[at80:]).all(), "80-250 kHz as pseudo-v1's"
        assert (a.mask[at60:at80] == IGNORE).all()
        assert (b.mask[at60:at80] == 0).any(), "60-80 kHz is scored"
        assert (b.mask[:at60] == IGNORE).all()
    index = pd.read_csv(v4_dir / "index.csv")
    assert index.shot.tolist() == [101, 102]


def test_pseudo_v4_builds_60_80_khz_apart_so_80_250_is_pseudo_v1s(tmp_path):
    """Regions, MIN_AREA, rings and unlit columns do not cross 80 kHz."""
    paths = ae_tree.build(tmp_path, {101: "train"}, tokeye_dt=0.256)
    label = labels.normalise((0, 2000), [(300, 900, 1), (1000, 1100, 1)])
    made = pseudo.make(paths, 101, label)
    shape = made.mask.shape
    grid = Grid(made.t0_ms, made.dt_ms, shape[1])
    t_ms, clean, ann = tokeye_clean(clean_path(paths.root / "ae" / "masks", 101))
    clean = clean.copy()
    t = grid.t0_ms + (np.arange(grid.n) + 0.5) * grid.dt_ms
    col_of = np.floor((t_ms - grid.t0_ms) / grid.dt_ms).astype(np.int64)

    def light(rows, cols):
        at = np.isin(col_of, cols)
        for k in rows:  # page bin k takes TokEye bins 2k - 2 and 2k - 1
            clean[:2, 2 * k - 1, at] = True

    dy = 500 / 512
    at80 = band_slice(0.0, dy, shape[0], (80.0, 250.0)).start
    at60 = band_slice(0.0, dy, shape[0], BAND60_KHZ).start
    assert (at60, at80) == (62, 82)
    dark = np.flatnonzero((t > 1000) & (t < 1100))  # present, TokEye dark
    ae = np.flatnonzero((t > 310) & (t < 890))  # present, the 146 kHz line lit
    only_low = dark[len(dark) // 2 : len(dark) // 2 + 3]
    light(range(70, 73), only_low)  # a column lit only at 60-80 kHz
    joined = ae[10:12]
    light(range(78, 84), joined)  # 2 x 2 above 80 passes MIN_AREA only joined
    straddle = ae[30:33]
    light(range(76, 88), straddle)  # a region straddling 80 kHz
    below = ae[50:53]
    light(range(76, 82), below)  # just below 80: its ring would reach 82-83
    tokeye = (t_ms, clean, ann)
    v1 = pseudo.build(101, label, grid, shape[0], 0.0, dy, tokeye)
    v4 = pseudo.build(101, label, grid, shape[0], 0.0, dy, tokeye, BAND60_KHZ)
    assert (v4.mask[at80:] == v1.mask[at80:]).all(), "80-250 kHz as pseudo-v1's"
    assert (v1.mask[:at80] == IGNORE).all() and (v4.mask[:at60] == IGNORE).all()
    assert (v1.mask[:, only_low] == IGNORE).all(), "unlit in pseudo-v1's band"
    assert (v4.mask[70:73][:, only_low] == 1).all()
    assert (v4.mask[at80:][:, only_low] == IGNORE).all()
    assert v4.present_unlit == v1.present_unlit - len(only_low) > 0
    assert (v1.mask[82:84][:, joined] == IGNORE).all(), "too small alone"
    assert (v4.mask[78:82][:, joined] == 1).all()
    assert (v4.mask[82:88][:, straddle] == 1).all()
    assert (v4.mask[76:82][:, straddle] == 1).all()
    assert (v4.mask[82:84][:, below] == 0).all(), "no ring across 80 kHz"
    assert (v4.mask[76:82][:, below] == 1).all()
    # One build over 60-250 kHz would couple them, so this fixture guards it.
    lit = pseudo.pool_columns(pseudo.tokeye_rows(clean), t_ms, grid)[: shape[0]]
    state = pseudo.column_states(label, grid)
    state[~pseudo.covered_columns(t_ms, grid)] = pseudo.OUTSIDE
    coupled, _ = pseudo._band_mask(
        lit,
        state == pseudo.PRESENT,
        state == pseudo.ABSENT,
        band_slice(0.0, dy, shape[0], BAND60_KHZ),
        shape[0],
        grid.n,
    )
    for cols in (only_low, joined, below):
        assert (coupled[at80:][:, cols] != v1.mask[at80:][:, cols]).any()


def test_segnet_v4_trains_on_v1s_split_and_states_its_second_use(
    tmp_path, monkeypatch, capsys
):
    splits = {s: "train" for s in (101, 102, 103, 104)}
    paths = ae_tree.build(tmp_path, splits, tokeye_dt=0.256)
    ae_tree.chosen(paths, {101: "train", 102: "train", 103: "val", 104: "test"})
    ae_tree.env(monkeypatch, paths)
    assert pseudo.main([]) == 0
    assert pseudo.main(["--version", "v4"]) == 0
    monkeypatch.setattr(seg_train, "TrainConfig", Small)
    v1_split = seg.model_dir(paths, "v1") / "split.csv"
    stderr = _refused(seg_train.main, ["--version", "v4", "--epochs", "1"], capsys)
    assert str(v1_split) in stderr and "FileNotFoundError" in stderr
    v1_split.parent.mkdir(parents=True, exist_ok=True)
    v1_split.write_text("shot,split\n101,train\n102,val\n103,val\n104,test\n")
    stderr = _refused(seg_train.main, ["--version", "v4", "--epochs", "1"], capsys)
    assert "SegNet v1's split differs from ours" in stderr
    assert not seg.model_dir(paths, "v4").exists()
    v1_split.unlink()
    assert seg_train.main(["--epochs", "1"]) == 0  # SegNet v1
    assert seg_train.main(["--version", "v4", "--epochs", "1"]) == 0
    models = seg.model_dir(paths, "v4")
    _, blob = seg_train.load(models / "model.pt")
    assert seg_train.blob_version(blob) == "v4"
    assert seg_train.blob_band(blob) == BAND60_KHZ
    assert seg_train.read_split(models / "split.csv") == seg_train.read_split(v1_split)
    trained = json.loads((models / "training.json").read_text())
    note = seg.reuse_note("v4", 1)
    assert trained["test_reuse"] == note and trained["pseudo"] == "pseudo-v4"
    assert seg_evaluate.main([]) == 0  # SegNet v1's one scoring
    assert seg_evaluate.main(["--version", "v4"]) == 0
    record = json.loads((models / "evaluation.json").read_text())
    assert record["meta"]["version"] == "v4"
    assert record["meta"]["band_khz"] == [60.0, 250.0]
    assert record["meta"]["frames_window"] == "0-2 s"
    report = (models / "evaluation.md").read_text()
    assert note in report and "band 60-250 kHz" in report
    # The paper's segmentation table ends its comment with the note.
    assert paper_build.seg_reuse(record) == note
    assert note in paper_scores.table_segmentation(
        record, paper_build.seg_reuse(record)
    )
    v1_record = json.loads((seg.model_dir(paths) / "evaluation.json").read_text())
    assert paper_build.seg_reuse(v1_record) is None
    assert "band 60" not in (seg.model_dir(paths) / "evaluation.md").read_text()
    # Its points of interest are drawn over its band, into its own directory.
    assert poi.main(["--version", "v4", "--workers", "1", "--no-pictures"]) == 0
    meta = json.loads((seg.poi_dir(paths, "v4") / "meta.json").read_text())
    assert meta["version"] == "v4" and meta["band_khz"] == [60.0, 250.0]
    assert not seg.poi_dir(paths).exists()
