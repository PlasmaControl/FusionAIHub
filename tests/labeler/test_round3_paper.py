"""The paper's AE figures for a whole-window version (v3): no 80 kHz line, no
scored-window line, the F1 and the picks over the owner's whole window, and the
segmentation's call over the band its blob records; and the interpreter's other
four tracks, as the build finds them (F9)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from labeler import frames
from labeler.ae.seg import train as seg_train
from labeler.ae.seg.model import SegNet, SegNetConfig
from labeler.ae.xpower import EVENT
from labeler.ae.xpower.data import FULL_BAND_KHZ
from labeler.ae.xpower.evaluate import chosen_model
from labeler.events import suggestions
from labeler.events.catalog.states import PRESENT
from labeler.events.review.labels import Label
from labeler.paper import COMING, ORDER, build, shots

from . import ae_tree, editor_tree, paper_tree

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


def test_the_pick_texts_say_the_window_they_pick_over():
    early = shots.pick_texts(shots.SCORED_MS)
    assert early.off_frames == shots.OFF_FRAMES
    assert early.interpreter == shots.INTERPRETER_RULE
    assert early.gap == shots.POOL_GAP and early.fallback == shots.POOL_FALLBACK
    assert early.unmarked == shots.POOL_UNMARKED
    assert early.examples == shots.EXAMPLES_RULE
    whole = shots.pick_texts(None)
    for name, text in dataclasses.asdict(whole).items():
        assert "whole window" in text and "0-2 s" not in text, name
    assert "best F1 over the whole window at 3 decimals" in whole.interpreter
    ranked = "the reviewed test shots ranked by F1 over the whole window"
    assert whole.examples.startswith(ranked)


def test_the_interpreter_pick_names_its_branch_over_the_whole_window():
    whole = shots.pick_texts(None)
    f1 = {1: 1.0, 2: 0.99}
    bare = shots.interpreter_pick(f1, None, {1: 0, 2: 0}, until_ms=None)
    assert (bare["shot"], bare["branch"]) == (1, whole.fallback)
    back = shots.interpreter_pick(f1, None, {1: 0, 2: 20}, until_ms=None)
    assert (back["shot"], back["branch"]) == (2, whole.gap)
    marked = pd.DataFrame({"shot": [1]})
    unmarked = shots.interpreter_pick(f1, marked, {1: 0, 2: 20}, until_ms=None)
    assert unmarked["branch"] == whole.unmarked.format(shots="2")
    early = shots.interpreter_pick(f1, None, {1: 0, 2: 20})
    assert early["branch"] == shots.POOL_GAP


def test_the_f1_and_the_off_period_over_the_whole_window():
    owner = np.r_[np.ones(100), np.zeros(100), np.ones(50)].astype(np.int8)
    prob = np.r_[np.ones(200), np.zeros(50)]
    assert shots.in_scored(0, 250).sum() == 200
    assert shots.in_scored(0, 250, until_ms=None).all()
    assert shots.scored_f1(prob, owner, 0.5, first=0) == pytest.approx(2 / 3)
    whole = shots.scored_f1(prob, owner, 0.5, first=0, until_ms=None)
    assert whole == pytest.approx(4 / 7), "tp 100, fp 100, fn 50"
    assert shots.scored_f1(prob, owner, 0.5, first=100, until_ms=None) == whole
    late = np.zeros(250, dtype=np.int8)
    late[30:150] = 1
    late[200:240] = 1  # AE off over 1500-2000 ms, back at 2 s
    assert shots.longest_off(late, first=0) == 0
    assert shots.longest_off(late, first=0, until_ms=None) == 50


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
    assert shots.rank_keys(whole).gap == 115, "off over 900-2050 ms, then back"
    assert shots.rank_keys(early).gap == 0, "not back before 2 s"
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
    fig = shots.draw_interpreter(whole, tmp_path / "interpreter")
    assert list(fig.axes[0].lines) == [] and _scored_lines(fig) == []
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
    draw = shots.draw_interpreter

    def spy(s, stem, tracks=None):
        drawn.append(s)
        return draw(s, stem, tracks)

    monkeypatch.setattr(shots, "draw_interpreter", spy)
    out = tmp_path / "v3"
    argv = ["--out", str(out), "--version", "v3", "--seg-version", "v2"]
    assert build.main(argv) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    whole = shots.pick_texts(None)
    assert manifest["interpreter_rule"] == whole.interpreter
    assert manifest["interpreter_branch"] == whole.gap
    assert manifest["example_rule"] == whole.examples
    assert manifest["interpreter_shot"] == 103, "AE back after 115 absent frames"
    pool = {"103": {"f1": 0.9231, "poi": 0, "longest_off_frames": 115}}
    assert manifest["interpreter_pool"] == pool
    assert manifest["example_shots"] == [102, 103]
    late = manifest["shot_f1"]["103"]
    assert late == {"f1_0_2s": 1.0, "f1_window": 0.9231}, "12/13 at 4 decimals"
    assert {"fig_interpreter", "fig_examples"} <= set(manifest["products"])
    assert drawn[0].scored_until_ms is None and drawn[0].mask.all()

    assert build.main(["--out", str(tmp_path / "v1")]) == 0
    old = json.loads((tmp_path / "v1" / "manifest.json").read_text())
    assert old["interpreter_rule"] == shots.INTERPRETER_RULE
    assert old["interpreter_branch"] == shots.POOL_FALLBACK, "not back before 2 s"
    assert old["interpreter_shot"] == 102 and old["shot_f1"] == manifest["shot_f1"]
    assert drawn[1].scored_until_ms == shots.SCORED_MS and drawn[1].mask is None


NTM, HMODE = "neoclassical_tearing_mode", "high_confinement_mode"
ELM, SAW = "edge_localized_mode", "sawtooth_oscillation"


def _frame_table(paths, method: str, rows_by_shot: dict):
    """`method`'s `frames.VERSION` suggestion table holding `rows_by_shot`."""
    spec = frames.SPECS[method]
    path = suggestions.table_path(paths, spec.event, method, frames.VERSION)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(suggestions.COLUMNS)]
    for shot, spans in rows_by_shot.items():
        lines += [f"{shot},{c},{a},{b},0.9" for a, b, c in spans]
    path.write_text("\n".join(lines) + "\n")
    return path


def test_the_interpreter_s_other_tracks_are_what_the_build_found(
    runs, tmp_path, monkeypatch
):
    # F9, on the interpreter's shot 103: the ELM model's table holds it; the
    # H-mode model's does not, though 103 has filterscopes; the sawteeth have
    # no table, and 103 has ECE; it has no Mirnov.
    elm = _frame_table(
        runs, "elm_frames", {103: [(0, 200, 0), (200, 600, PRESENT), (600, 2200, 0)]}
    )
    hmode = _frame_table(runs, "hmode_frames", {101: [(0, 1000, PRESENT)]})
    t = np.arange(0.0, 2500.0, 1.0)
    groups = {
        "filterscopes": (t, np.ones((8, len(t)))),
        "ece": (t, np.ones((4, len(t)))),
    }
    editor_tree.write(runs.corpus / "103_processed.h5", groups)
    seen, figures = [], []
    draw = shots.draw_interpreter

    def spy(s, stem, tracks=None):
        seen.append(tracks)
        figures.append(draw(s, stem, tracks))
        return figures[-1]

    monkeypatch.setattr(shots, "draw_interpreter", spy)
    out = tmp_path / "v3"
    argv = ["--out", str(out), "--version", "v3", "--seg-version", "v2"]
    assert build.main(argv) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["interpreter_shot"] == 103
    assert manifest["interpreter_tracks"] == {
        NTM: shots.NO_DATA.format("mirnov"),
        HMODE: shots.NOT_APPLIED,
        ELM: build.SUGGESTED_TRACK,
        SAW: COMING,
    }
    assert manifest["interpreter_tracks"][NTM] == "no mirnov data"
    pinned = manifest["inputs"]
    assert pinned["frames_table_elm_frames"]["path"] == str(elm)
    assert pinned["frames_table_hmode_frames"]["path"] == str(hmode)
    assert "frames_table_sawtooth_frames" not in pinned
    [tracks] = seen
    assert np.flatnonzero(tracks[ELM] == PRESENT).tolist() == list(range(20, 60))
    fig = figures[0]
    axes = dict(zip(ORDER, fig.axes[1:], strict=True))
    for category in (NTM, HMODE, SAW):
        text = manifest["interpreter_tracks"][category]
        assert [x.get_text() for x in axes[category].texts] == [text]
        assert not axes[category].collections
    assert axes[ELM].collections and not axes[ELM].texts
    legend = {t.get_text() for g in fig.legends for t in g.get_texts()}
    assert "suggested: present" in legend
    # Without anything passed, a track is coming, as before (F9's fallback).
    s = shots.picture(
        103,
        label=LATE,
        model=shots.Model.load(chosen_model(runs.root / "models" / "ae_xpower" / "v3")),
        store=runs.spectrogram_file(EVENT, 103),
    )
    plain = draw(s, tmp_path / "plain")
    texts = [[x.get_text() for x in ax.texts] for ax in plain.axes[2:]]
    assert texts == [[COMING]] * 4
