"""The paper's AE figures for a whole-window version (v3): no 80 kHz line, no
scored-window line, the F1 and the examples' pick over the owner's whole window,
and the segmentation's call over the band its blob records."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from labeler.ae.seg import train as seg_train
from labeler.ae.seg.model import SegNet, SegNetConfig
from labeler.ae.xpower import EVENT
from labeler.ae.xpower.data import FULL_BAND_KHZ
from labeler.ae.xpower.evaluate import chosen_model
from labeler.events.review.labels import Label
from labeler.paper import build, shots

from . import ae_tree, paper_tree

#: The owner's AE at 300-900 ms and again at 2050-2150 ms, in a 0-2.2 s window.
LATE = Label(window=(0, 2200), intervals=((300, 900, 1), (2050, 2150, 1)))


@pytest.fixture(autouse=True)
def paths(tmp_path, monkeypatch):
    return paper_tree.temporary_paths(tmp_path, monkeypatch)


def _says(net, rows, first, n, *, band, context=20):
    """P(AE) 1 over 300-900 ms and 0 elsewhere, whatever the rows."""
    frame = first + np.arange(n)
    prob = ((frame >= 30) & (frame < 90)).astype(np.float32)
    return prob, np.ones(n, dtype=bool)


def _v2_seg(paths) -> Path:
    """A SegNet saved as SegNet v2 saves it, its blob naming 0-250 kHz, at
    threshold 0 (every pixel of its band is AE). Its model.pt."""
    out = paths.root / "models" / "ae_seg" / "v2"
    seg_train.save(
        out,
        SegNet(SegNetConfig(width=4)),
        threshold=0.0,
        split={},
        history=[],
        config=seg_train.TrainConfig(width=4),
        inputs={},
        band_khz=FULL_BAND_KHZ,
        version="v2",
    )
    return out / "model.pt"


def test_a_whole_window_version_has_no_scored_end():
    assert shots.scored_ms() == shots.scored_ms("v1") == shots.SCORED_MS == 2000.0
    assert shots.scored_ms("v2") == 2000.0 and shots.scored_ms("v3") is None
    assert shots.window_name(shots.SCORED_MS) == "0-2 s"
    assert shots.window_name(None) == "whole window"
    assert shots.SCORED_LABEL == "scored: 0-2 s"
    assert build.no_f1(shots.SCORED_MS) == build.NO_F1
    assert "whole window" in build.no_f1(None) and "0-2 s" not in build.no_f1(None)


def test_the_examples_rule_says_the_window_it_picks_over():
    early = shots.pick_texts(shots.SCORED_MS)
    assert early.examples == shots.EXAMPLES_RULE
    assert "0-2 s" in early.examples and "whole window" not in early.examples
    whole = shots.pick_texts(None)
    assert "whole window" in whole.examples and "0-2 s" not in whole.examples
    ranked = "the reviewed test shots ranked by F1 over the whole window"
    assert whole.examples.startswith(ranked)
    assert not hasattr(shots, "interpreter_pick"), "the AE-shot pick is gone"


def test_the_f1_over_the_whole_window():
    owner = np.r_[np.ones(100), np.zeros(100), np.ones(50)].astype(np.int8)
    prob = np.r_[np.ones(200), np.zeros(50)]
    assert shots.in_scored(0, 250).sum() == 200
    assert shots.in_scored(0, 250, until_ms=None).all()
    assert shots.scored_f1(prob, owner, 0.5, first=0) == pytest.approx(2 / 3)
    whole = shots.scored_f1(prob, owner, 0.5, first=0, until_ms=None)
    assert whole == pytest.approx(4 / 7), "tp 100, fp 100, fn 50"
    assert shots.scored_f1(prob, owner, 0.5, first=100, until_ms=None) == whole


@pytest.fixture
def tree(tmp_path, monkeypatch):
    paths = ae_tree.build(tmp_path, {101: "train", 102: "valid", 103: "valid"})
    models = ae_tree.chosen(paths, {101: "train", 102: "test", 103: "test"})
    monkeypatch.setattr(shots, "probabilities", _says)
    return paths, chosen_model(models)


def _picture(paths, model_file, *, until, seg_file=None) -> shots.AEShot:
    """Shot 102 under the `LATE` label, over the review store to 2.2 s."""
    return shots.picture(
        102,
        label=LATE,
        model=shots.Model.load(model_file),
        store=paths.spectrogram_file(EVENT, 102),
        segmentation=None if seg_file is None else shots.Segmentation.load(seg_file),
        scored_until_ms=until,
    )


def test_a_whole_window_picture_scores_and_masks_as_its_version(tree):
    paths, model_file = tree
    whole = _picture(paths, model_file, until=None, seg_file=_v2_seg(paths))
    early = _picture(paths, model_file, until=shots.SCORED_MS)
    assert whole.scored_until_ms is None and early.scored_until_ms == 2000.0
    assert whole.f1 == pytest.approx(12 / 13), "the late AE is missed: fn 10"
    assert whole.f1_window == pytest.approx(12 / 13)
    assert early.f1 == early.f1_0_2s == whole.f1_0_2s == 1.0
    assert whole.mask.all(), "threshold 0 over 0-250 kHz: below 80 kHz too"
    old = _picture(paths, model_file, until=None, seg_file=paper_tree.seg_model(paths))
    freqs = old.y0 + np.arange(old.image.shape[0]) * old.dy
    assert old.mask[freqs >= 80].all(), "a blob without a band is v1's"
    assert not old.mask[freqs < 80].any()


def _scored_lines(fig) -> list:
    return [x for ax in fig.axes for x in ax.lines if x.get_label() == "_scored"]


def test_a_whole_window_figure_has_no_band_line_and_no_scored_line(tree, tmp_path):
    paths, model_file = tree
    whole = _picture(paths, model_file, until=None, seg_file=_v2_seg(paths))
    early = _picture(paths, model_file, until=shots.SCORED_MS)
    fig = shots.draw_examples([whole], tmp_path / "whole")
    spec = fig.axes[0]
    assert spec.get_title() == "shot 102 (test): F1 (whole window) 0.92"
    assert list(spec.lines) == [], "no 80 kHz line, no scored line"
    assert list(spec.texts) == [] and _scored_lines(fig) == []
    fig = shots.draw_examples([early], tmp_path / "early")
    spec = fig.axes[0]
    assert spec.get_title() == "shot 102 (test): F1 (0-2 s) 1.00"
    [line] = spec.lines
    assert list(line.get_xdata()) == [shots.SCORED_MS] * 2, "the scored line alone"
    assert [t.get_text() for t in spec.texts] == [shots.SCORED_LABEL]
    assert len(_scored_lines(fig)) == 3


def _late_copy(paths) -> None:
    """Shot 103 in v1's copy of the labels as `LATE`: AE back at 2.05 s."""
    copy = paper_tree.scored_labels(paths)
    rows = [r for r in copy.read_text().splitlines() if not r.startswith("103,")]
    late = [",".join(map(str, row)) for row in LATE.rows(103)]
    copy.write_text("\n".join([*rows, *late]) + "\n")


@pytest.fixture
def runs(tmp_path, monkeypatch):
    """test_paper_build's tree with shot 103 as `LATE`, the frame model's
    records at v1 and at v3, a SegNet v2, and the stand-in P(AE)."""
    paths = ae_tree.build(tmp_path, {101: "train", 102: "valid", 103: "valid"})
    models = ae_tree.chosen(paths, {101: "train", 102: "test", 103: "test"})
    (models / "evaluation.json").write_text(json.dumps(paper_tree.ae_evaluation()))
    _late_copy(paths)
    paper_tree.record_labels(paths)
    paper_tree.as_version(paths, "v3", keep=True)
    _v2_seg(paths)
    ae_tree.env(monkeypatch, paths)
    monkeypatch.setattr(shots, "probabilities", _says)
    return paths


def test_a_whole_window_build_picks_and_records_over_the_whole_window(
    runs, tmp_path, monkeypatch
):
    drawn = []
    draw = shots.draw_examples

    def spy(picked, stem):
        drawn.append(list(picked))
        return draw(picked, stem)

    monkeypatch.setattr(shots, "draw_examples", spy)
    out = tmp_path / "v3"
    argv = ["--out", str(out), "--version", "v3", "--seg-version", "v2"]
    assert build.main(argv) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    whole = shots.pick_texts(None)
    assert manifest["example_rule"] == whole.examples
    assert manifest["example_shots"] == [102, 103]
    late = manifest["shot_f1"]["103"]
    assert late == {"f1_0_2s": 1.0, "f1_window": 0.9231}, "12/13 at 4 decimals"
    assert "fig_examples_ae" in manifest["products"]
    assert not [k for k in manifest if k.startswith("interpreter_")]
    [picked] = drawn
    assert all(s.scored_until_ms is None and s.mask.all() for s in picked)

    assert build.main(["--out", str(tmp_path / "v1")]) == 0
    old = json.loads((tmp_path / "v1" / "manifest.json").read_text())
    assert old["example_rule"] == shots.EXAMPLES_RULE
    assert old["shot_f1"] == manifest["shot_f1"]
    assert all(s.scored_until_ms == shots.SCORED_MS for s in drawn[1])
    assert all(s.mask is None for s in drawn[1])
