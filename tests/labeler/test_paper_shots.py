"""The one-discharge interpreter figure and the AE examples."""

from __future__ import annotations

import dataclasses
import re

import numpy as np
import pandas as pd
import pytest

from labeler.ae.seg import EVENT as SEG_EVENT
from labeler.ae.seg import train as seg_train
from labeler.ae.seg.poi import ae_pixels
from labeler.ae.xpower.data import BAND_KHZ, band_slice, store_rows
from labeler.ae.xpower.evaluate import chosen_model
from labeler.events.catalog.states import UNCERTAIN
from labeler.paper import COMING, shots
from labeler.scoring.frames import FRAME_MS

from . import ae_tree, paper_tree


@pytest.fixture(autouse=True)
def paths(tmp_path, monkeypatch):
    """A temporary Paths, set before every call; `tree` lays its own out the
    same way."""
    return paper_tree.temporary_paths(tmp_path, monkeypatch)


@pytest.fixture
def tree(tmp_path):
    paths = ae_tree.build(tmp_path, {101: "train", 102: "valid", 103: "valid"})
    models = ae_tree.chosen(paths, {101: "train", 102: "test", 103: "test"})
    return paths, chosen_model(models)


def _masked(s: shots.AEShot, t_ms: tuple[float, float], f_khz=(140.0, 152.0)):
    """`s` with a mask over `t_ms` x `f_khz` alone."""
    times = s.grid.t0_ms + (np.arange(s.grid.n) + 0.5) * s.grid.dt_ms
    freqs = s.y0 + np.arange(s.image.shape[0]) * s.dy
    mask = np.outer(
        (freqs >= f_khz[0]) & (freqs < f_khz[1]),
        (times >= t_ms[0]) & (times < t_ms[1]),
    )
    return dataclasses.replace(s, mask=mask)


def _mask_images(ax) -> list:
    return [im for im in ax.images if im.get_label().endswith(shots.MASK_LABEL)]


def test_ae_shot_reads_the_store_the_owner_and_the_model(tree):
    paths, model_file = tree
    seg_file = paper_tree.seg_model(paths)  # threshold 0: the whole band is AE
    s = shots.ae_shot(paths, 102, model_file=model_file, seg_file=seg_file)
    assert (s.first, len(s.prob), s.split, s.threshold) == (0, 200, "test", 0.5)
    assert (s.owner == 1).sum() == 60, "the owner's AE, 300-900 ms"
    assert s.image.shape[0] == 257
    assert s.grid.dt_ms == pytest.approx(0.256 * shots.PICTURE_LEVEL)
    assert (s.edges[0], s.edges[-1]) == (0, 2000)
    assert s.mask.shape == s.image.shape and s.mask.dtype == bool
    freqs = s.y0 + np.arange(s.image.shape[0]) * s.dy
    band = (freqs >= BAND_KHZ[0]) & (freqs <= BAND_KHZ[1])
    assert s.mask[band].all() and not s.mask[~band].any(), "80-250 kHz only"
    assert shots.ae_shot(paths, 103, model_file=model_file).mask is None
    with pytest.raises(KeyError, match="has not saved"):
        shots.ae_shot(paths, 999, model_file=model_file)


def test_the_mask_is_segnets_call_at_its_threshold(tree):
    """At a threshold inside SegNet's outputs the mask is `ae_pixels` of its
    P(AE) on the picture's own rows: some of the band, not all of it."""
    paths, model_file = tree
    segmentation = shots.Segmentation.load(paper_tree.seg_model(paths))
    _, values, y0, dy = store_rows(
        paths.spectrogram_file(SEG_EVENT, 102), level=shots.PICTURE_LEVEL
    )
    prob = seg_train.predict(segmentation.net, values)
    band = band_slice(y0, dy, prob.shape[0], BAND_KHZ)
    threshold = float(np.median(prob[band]))
    seg_file = paper_tree.seg_model(
        paths, "v1-median", threshold=threshold, net=segmentation.net
    )
    s = shots.ae_shot(paths, 102, model_file=model_file, seg_file=seg_file)
    expected = ae_pixels(prob, threshold, y0, dy)
    assert 0 < expected.sum() < expected[band].size
    np.testing.assert_array_equal(s.mask, expected)


F1 = {1: 0.9, 2: 0.5, 4: 0.7, 5: 0.7, 6: float("nan")}


def test_examples_run_from_the_best_test_shot_to_the_worst():
    assert shots.pick_examples(F1) == [1, 5, 2]
    assert shots.pick_examples(F1, n=5) == [1, 4, 5, 2]
    assert shots.pick_examples(F1, n=1) == [1]
    points = pd.DataFrame({"shot": [4, 2]})
    assert shots.interpreter_pick(F1, points)["shot"] == 4
    assert shots.interpreter_pick(F1, None)["shot"] == 1
    with pytest.raises(ValueError, match="no test shot"):
        shots.interpreter_pick({6: float("nan")}, None)


def test_the_interpreter_figure_has_a_track_per_phenomenon(tree, tmp_path):
    paths, model_file = tree
    s = _masked(shots.ae_shot(paths, 102, model_file=model_file), (300.0, 900.0))
    fig = shots.draw_interpreter(s, tmp_path / "fig_interpreter")
    assert (tmp_path / "fig_interpreter.pdf").is_file()
    spec, *tracks = fig.axes
    assert [ax.get_ylabel() for ax in tracks] == [
        "AE",
        "NTM",
        "H-mode",
        "ELMing",
        "sawteeth",
        "disruption",
    ]
    assert [t.get_text() for ax in tracks for t in ax.texts] == [COMING] * 5
    assert not spec.texts and not spec.patches, "no boxes, no numbers"
    [mask] = _mask_images(spec)
    assert mask.get_label() == shots.MASK_LABEL
    alpha = mask.get_array()[..., 3]
    assert mask.get_alpha() is None, "a PDF would apply an image's alpha twice"
    assert set(np.unique(alpha)) == {0.0, shots.MASK_ALPHA}, "the fill's, in the data"
    assert mask.get_extent() == spec.images[0].get_extent(), "on the picture's pixels"
    [outline] = [c for c in spec.collections if c.get_paths()]
    x0, y0, x1, y1 = outline.get_paths()[0].get_extents().extents
    assert (x0, x1) == pytest.approx((300.0, 900.0), abs=s.grid.dt_ms)
    assert (y0, y1) == pytest.approx((140.0, 152.0), abs=s.dy)
    [present] = tracks[0].collections[0].get_paths()
    extent = present.get_extents()
    assert (extent.x0, extent.x1) == (300.0, 900.0), "the owner's present frames"
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    assert labels[-1] == shots.MASK_LABEL
    assert paper_tree.small_text(fig) == []


def test_the_examples_figure(tree, tmp_path):
    paths, model_file = tree
    two = [shots.ae_shot(paths, s, model_file=model_file) for s in (102, 103)]
    fig = shots.draw_examples(two, tmp_path / "fig_examples")
    assert (tmp_path / "fig_examples.png").is_file()
    assert len(fig.axes) == 6
    assert fig.axes[0].get_title().startswith("shot 102 (test): F1 (0-2 s) ")
    assert fig.axes[3].get_title().startswith("shot 103 (test): ")
    assert shots.MASK_LABEL not in [t.get_text() for t in fig.legends[0].get_texts()]
    assert all(_mask_images(ax) == [] for ax in fig.axes), "no segmentation run"
    assert paper_tree.small_text(fig) == []


def _legend_matches_drawn(fig) -> None:
    """Every key is the label of an artist drawn inside some panel's view."""
    fig.draw_without_rendering()
    drawn = {
        a.get_label()
        for ax in fig.axes
        for a in ax.get_children()
        if isinstance(a.get_label(), str)  # an Axis's label is its Text
        and a.get_window_extent().overlaps(ax.bbox)
    }
    [legend] = fig.legends
    shown = [t.get_text() for t in legend.get_texts()]
    assert shown and set(shown) <= drawn, set(shown) - drawn


def test_the_scored_window_is_marked(tree, tmp_path):
    paths, model_file = tree
    s = _masked(shots.ae_shot(paths, 102, model_file=model_file), (300.0, 900.0))
    late = dataclasses.replace(
        s,
        prob=np.r_[s.prob, np.zeros(100)],
        owner=np.r_[s.owner, np.zeros(100, dtype=s.owner.dtype)],
    )
    for draw, name in (
        (shots.draw_interpreter, "fig_interpreter"),
        (lambda x, stem: shots.draw_examples([x], stem), "fig_examples"),
    ):
        fig = draw(late, tmp_path / name)
        spec = fig.axes[0]
        assert shots.SCORED_LABEL in [t.get_text() for t in spec.texts]
        assert any(
            list(line.get_xdata()) == [shots.SCORED_MS] * 2 for line in spec.lines
        )


def _owner(*spans: tuple[int, int, int], n: int = 200) -> np.ndarray:
    """Per-frame owner states from `(first, stop, state)` spans; absent elsewhere."""
    owner = np.zeros(n, dtype=np.int8)
    for a, b, state in spans:
        owner[a:b] = state
    return owner


def test_the_off_period_is_the_longest_run_of_absent_frames_inside_the_ae_span():
    lead_in = _owner((30, 200, 1), n=250)  # absent before breakdown, then after 2 s
    gap = _owner((30, 100, 1), (120, 200, 1))  # AE turns off for 200 ms, then on
    two = _owner((30, 100, 1), (103, 110, 1), (113, 200, 1))  # off twice, 30 ms
    unsure = _owner((30, 100, 1), (105, 106, UNCERTAIN), (120, 200, 1))
    trailing = _owner((30, 150, 1))  # AE turns off at 1.5 s and never comes back
    late = _owner((30, 190, 1), (210, 250, 1), n=250)  # back on only after 2 s
    assert shots.longest_off(lead_in, first=0) == 0
    assert shots.longest_off(gap, first=0) == 20
    assert shots.longest_off(two, first=0) == 3, "the longest run, not the sum"
    assert shots.longest_off(unsure, first=0) == 14, "an uncertain frame is not off"
    assert shots.longest_off(trailing, first=0) == 0, "off for good is not a gap"
    assert shots.longest_off(late, first=0) == 0, "the scored window is 0-2 s"
    assert shots.longest_off(_owner(), first=0) == 0, "no AE at all"
    assert shots.longest_off(gap, first=100) == 0, "frames 100.. : 1-2 s only"


def test_the_pool_needs_5_whole_absent_frames():
    assert shots.MIN_GAP_FRAMES == 5
    assert shots.MIN_GAP_FRAMES * FRAME_MS == 50
    four = _owner((30, 100, 1), (104, 200, 1))
    five = _owner((30, 100, 1), (105, 200, 1))
    threes = _owner((30, 100, 1), (103, 110, 1), (113, 200, 1))
    owners = {1: four, 2: five, 3: threes}
    gaps = {s: shots.longest_off(owner, first=0) for s, owner in owners.items()}
    assert gaps == {1: 4, 2: 5, 3: 3}
    f1 = {1: 1.0, 2: 0.99, 3: 0.995}
    poi = pd.DataFrame({"shot": [1, 2, 3]})
    pick = shots.interpreter_pick(f1, poi, gaps)
    assert (pick["shot"], pick["branch"]) == (2, shots.POOL_GAP), (
        "a 4-frame gap and two separate 3-frame gaps are out; 5 frames are in"
    )
    assert pick["pool"] == {"2": {"f1": 0.99, "poi": 1, "longest_off_frames": 5}}
    short = shots.interpreter_pick(f1, poi, {1: 4, 3: 3})
    assert (short["shot"], short["branch"]) == (1, shots.POOL_FALLBACK)


OFF_FRAMES = (
    "at least 5 whole consecutive absent 10 ms frames between the owner's first "
    "and last present frames in 0-2 s"
)


def test_the_pool_texts_say_only_what_the_code_checks():
    """Whole frames: 5 absent frames are an off-period of 50 ms or more, but one
    of 50-59 ms off the 10 ms grid can hold only 4. So each text gives the rule
    in frames, and 50 ms only as what the frames imply for a shot in the pool."""
    texts = {
        "INTERPRETER_RULE": shots.INTERPRETER_RULE,
        "POOL_GAP": shots.POOL_GAP,
        "POOL_FALLBACK": shots.POOL_FALLBACK,
        "POOL_UNMARKED": shots.POOL_UNMARKED,
    }
    for name, text in texts.items():
        assert OFF_FRAMES in text, name
    assert shots.POOL_GAP.endswith(f"{OFF_FRAMES} (so an off-period of at least 50 ms)")
    assert "so an off-period of at least 50 ms" in shots.INTERPRETER_RULE
    said = {name: re.findall(r"\d+ ms", text) for name, text in texts.items()}
    assert said == {
        "INTERPRETER_RULE": ["10 ms", "50 ms"],
        "POOL_GAP": ["10 ms", "50 ms"],
        "POOL_FALLBACK": ["10 ms"],
        "POOL_UNMARKED": ["10 ms"],
    }, "no text says what a 54 ms off-period of 4 whole frames would make false"
    named = shots.POOL_UNMARKED.format(shots="2, 3")
    assert "no point of interest (2, 3) have at least 5 whole" in named


def test_the_interpreter_shows_a_shot_where_ae_turns_off_and_on():
    lead_in, gap = _owner((30, 200, 1)), _owner((30, 100, 1), (120, 200, 1))
    trailing = _owner((30, 150, 1))  # AE ends before 2 s and never returns
    owners = {1: lead_in, 2: gap, 3: trailing}
    gaps = {s: shots.longest_off(owner, first=0) for s, owner in owners.items()}
    f1 = {1: 1.0, 2: 0.99, 3: 0.995}
    poi = pd.DataFrame({"shot": [1] * 3 + [2] * 9 + [3] * 4})
    pick = shots.interpreter_pick(f1, poi, gaps)
    assert pick["shot"] == 2, "the best F1 among the shots where AE comes back"
    assert pick["branch"] == shots.POOL_GAP
    assert pick["pool"] == {"2": {"f1": 0.99, "poi": 9, "longest_off_frames": 20}}


def test_the_fallback_says_which_case_fired():
    f1 = {1: 1.0, 2: 0.99, 3: 0.98}
    poi = pd.DataFrame({"shot": [1, 1, 2, 3]})
    none = shots.interpreter_pick(f1, poi, {1: 0, 2: 0, 3: 0})
    assert (none["shot"], none["branch"]) == (1, shots.POOL_FALLBACK)
    assert sorted(none["pool"]) == ["1", "2", "3"], "D40 as written"
    marked = pd.DataFrame({"shot": [1]})
    unmarked = shots.interpreter_pick(f1, marked, {1: 0, 2: 20, 3: 5})
    assert unmarked["branch"] == shots.POOL_UNMARKED.format(shots="2, 3")
    assert "2, 3" in unmarked["branch"], "the shots are named"
    assert (unmarked["shot"], sorted(unmarked["pool"])) == (1, ["1"]), (
        "the pool keeps D40's point of interest"
    )
    bare = shots.interpreter_pick(f1, None, {1: 0, 2: 0, 3: 0})
    assert (bare["branch"], sorted(bare["pool"])) == (
        shots.POOL_FALLBACK,
        ["1", "2", "3"],
    ), "no point of interest anywhere: every test shot"


def test_the_interpreter_ties_go_to_fewer_points_then_the_lower_shot():
    f1 = {10: 0.9, 11: 0.99, 12: 0.99004, 13: 0.99, 14: 1.0}
    gaps = {10: 5, 11: 5, 12: 5, 13: 5, 14: 0}
    poi = pd.DataFrame({"shot": [10] * 5 + [11] * 30 + [12] * 8 + [13] * 2 + [14]})
    assert shots.interpreter_pick(f1, poi, gaps)["shot"] == 13, "F1 at 3 decimals"
    even = pd.DataFrame({"shot": [11] * 3 + [13] * 3 + [12] * 3})
    assert shots.interpreter_pick(f1, even, gaps)["shot"] == 11, "then the lower shot"
    assert "absent" in shots.INTERPRETER_RULE
    assert shots.pick_examples(f1, 2) == [14, 10]


def test_the_scored_f1_is_over_0_to_2_s():
    owner = np.r_[np.ones(100), np.zeros(100), np.ones(50)].astype(np.int8)
    prob = np.r_[np.ones(200), np.zeros(50)]
    assert shots.scored_f1(prob, owner, 0.5, first=0) == pytest.approx(2 / 3)
    assert shots.scored_f1(prob[:200], owner[:200], 0.5, first=0) == pytest.approx(
        2 / 3
    )
    assert shots.scored_f1(prob, owner, 0.5, first=100) == pytest.approx(1.0)


def test_the_examples_legend_is_what_the_panels_draw(tree, tmp_path):
    paths, model_file = tree
    two = [
        _masked(shots.ae_shot(paths, 102, model_file=model_file), (300.0, 900.0)),
        shots.ae_shot(paths, 103, model_file=model_file),
    ]
    fig = shots.draw_examples(two, tmp_path / "fig_examples")
    _legend_matches_drawn(fig)
    assert paper_tree.small_text(fig) == []
    shown = [t.get_text() for t in fig.legends[0].get_texts()]
    assert "owner: uncertain" not in shown and "owner: not observable" not in shown
    assert "owner: present" in shown and "model: present" in shown
    assert fig.axes[0].get_title() == (f"shot 102 (test): F1 (0-2 s) {two[0].f1:.2f}")
    fig = shots.draw_interpreter(two[0], tmp_path / "fig_interpreter")
    _legend_matches_drawn(fig)
    assert paper_tree.small_text(fig) == []


def test_a_mask_wholly_off_the_axes_gets_no_legend_key(tree, tmp_path):
    paths, model_file = tree
    s = shots.ae_shot(paths, 102, model_file=model_file)
    late = _masked(s, (2100.0, 2180.0))  # drawn to 2050 ms; the store to 2200
    fig = shots.draw_interpreter(late, tmp_path / "fig_interpreter")
    assert fig.axes[0].get_xlim()[1] < 2100.0
    [mask] = _mask_images(fig.axes[0])
    assert mask.get_label() == "_" + shots.MASK_LABEL, "drawn, clipped away"
    shown = [t.get_text() for t in fig.legends[0].get_texts()]
    assert shots.MASK_LABEL not in shown
    _legend_matches_drawn(fig)
    both = dataclasses.replace(late, mask=late.mask | _masked(s, (300.0, 900.0)).mask)
    fig = shots.draw_interpreter(both, tmp_path / "fig_interpreter")
    shown = [t.get_text() for t in fig.legends[0].get_texts()]
    assert shots.MASK_LABEL in shown, "a pixel in view: keyed"
    _legend_matches_drawn(fig)
