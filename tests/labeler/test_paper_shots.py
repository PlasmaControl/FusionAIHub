"""The AE examples: a test shot's picture, its mask and the examples figure."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from labeler.ae.seg import EVENT as SEG_EVENT
from labeler.ae.seg import train as seg_train
from labeler.ae.seg.poi import ae_pixels
from labeler.ae.xpower.data import BAND_KHZ, band_slice, store_rows
from labeler.ae.xpower.evaluate import chosen_model
from labeler.paper import shots

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
    assert shots.pick_examples({10: 0.9, 13: 0.99, 14: 1.0}, 2) == [14, 10]


def test_the_examples_mask_is_a_fill_with_an_outline(tree, tmp_path):
    paths, model_file = tree
    s = _masked(shots.ae_shot(paths, 102, model_file=model_file), (300.0, 900.0))
    fig = shots.draw_examples([s], tmp_path / "fig_examples")
    assert (tmp_path / "fig_examples.pdf").is_file()
    spec, strip, _ = fig.axes
    assert not spec.texts and not spec.patches, "no boxes, no numbers"
    [mask] = _mask_images(spec)
    assert mask.get_label() == shots.MASK_LABEL
    alpha = mask.get_array()[..., 3]
    assert mask.get_alpha() is None, "a PDF would apply an image's alpha twice"
    assert np.unique(alpha).tolist() == [0, np.float32(shots.MASK_ALPHA)], "data"
    assert mask.get_extent() == spec.images[0].get_extent(), "on the picture's pixels"
    [outline] = [c for c in spec.collections if c.get_paths()]
    x0, y0, x1, y1 = outline.get_paths()[0].get_extents().extents
    assert (x0, x1) == pytest.approx((300.0, 900.0), abs=s.grid.dt_ms)
    assert (y0, y1) == pytest.approx((140.0, 152.0), abs=s.dy)
    [present] = strip.collections[0].get_paths()
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
    fig = shots.draw_examples([late], tmp_path / "fig_examples")
    spec = fig.axes[0]
    assert shots.SCORED_LABEL in [t.get_text() for t in spec.texts]
    assert any(list(line.get_xdata()) == [shots.SCORED_MS] * 2 for line in spec.lines)


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


def test_a_mask_wholly_off_the_axes_gets_no_legend_key(tree, tmp_path):
    paths, model_file = tree
    s = shots.ae_shot(paths, 102, model_file=model_file)
    late = _masked(s, (2100.0, 2180.0))  # drawn to 2050 ms; the store to 2200
    fig = shots.draw_examples([late], tmp_path / "fig_examples")
    assert fig.axes[0].get_xlim()[1] < 2100.0
    [mask] = _mask_images(fig.axes[0])
    assert mask.get_label() == "_" + shots.MASK_LABEL, "drawn, clipped away"
    shown = [t.get_text() for t in fig.legends[0].get_texts()]
    assert shots.MASK_LABEL not in shown
    _legend_matches_drawn(fig)
    both = dataclasses.replace(late, mask=late.mask | _masked(s, (300.0, 900.0)).mask)
    fig = shots.draw_examples([both], tmp_path / "fig_examples")
    shown = [t.get_text() for t in fig.legends[0].get_texts()]
    assert shots.MASK_LABEL in shown, "a pixel in view: keyed"
    _legend_matches_drawn(fig)
