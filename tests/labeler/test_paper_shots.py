"""The one-discharge interpreter figure and the AE examples."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
from matplotlib.patches import Rectangle

from labeler.ae.xpower.evaluate import chosen_model
from labeler.paper import COMING, shots

from . import ae_tree, paper_tree


@pytest.fixture
def tree(tmp_path):
    paths = ae_tree.build(tmp_path, {101: "train", 102: "valid", 103: "valid"})
    models = ae_tree.chosen(paths, {101: "train", 102: "test", 103: "test"})
    return paths, chosen_model(models)


def _poi(shot: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "shot": shot,
                "region": 1,
                "t_start_ms": 300.0,
                "t_end_ms": 900.0,
                "f_lo_khz": 140.0,
                "f_hi_khz": 152.0,
            }
        ]
    )


def test_ae_shot_reads_the_store_the_owner_and_the_model(tree):
    paths, model_file = tree
    s = shots.ae_shot(paths, 102, model_file=model_file, poi=_poi(102))
    assert (s.first, len(s.prob), s.split, s.threshold) == (0, 200, "test", 0.5)
    assert (s.owner == 1).sum() == 60, "the owner's AE, 300-900 ms"
    assert s.image.shape[0] == 257
    assert s.grid.dt_ms == pytest.approx(0.256 * shots.PICTURE_LEVEL)
    assert (s.edges[0], s.edges[-1]) == (0, 2000)
    assert [box["region"] for box in s.boxes] == [1]
    assert shots.ae_shot(paths, 103, model_file=model_file).boxes == ()
    with pytest.raises(KeyError, match="has not saved"):
        shots.ae_shot(paths, 999, model_file=model_file)


INDEX = pd.DataFrame(
    {
        "shot": [1, 2, 3, 4, 5, 6, 7],
        "group": ["reviewed"] * 6 + ["unreviewed"],
        "split": ["test", "test", "train", "test", "test", "test", "unreviewed"],
        "f1_vs_owner": [0.9, 0.5, 0.99, 0.7, 0.7, "", ""],
    }
)


def test_examples_run_from_the_best_test_shot_to_the_worst():
    assert shots.pick_examples(INDEX) == [1, 5, 2]
    assert shots.pick_examples(INDEX, n=5) == [1, 4, 5, 2]
    assert shots.pick_examples(INDEX, n=1) == [1]
    assert shots.interpreter_shot(INDEX, pd.DataFrame({"shot": [4, 2]})) == 4
    assert shots.interpreter_shot(INDEX, None) == 1
    with pytest.raises(ValueError, match="no reviewed test shot"):
        shots.interpreter_shot(INDEX[INDEX.split == "train"], None)


def test_the_interpreter_figure_has_a_track_per_phenomenon(tree, tmp_path):
    paths, model_file = tree
    s = shots.ae_shot(paths, 102, model_file=model_file, poi=_poi(102))
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
    assert [t.get_text() for t in spec.texts] == ["1"], "the region, numbered"
    [box] = spec.patches
    assert (box.get_x(), box.get_y(), box.get_width(), box.get_height()) == (
        300.0,
        140.0,
        600.0,
        12.0,
    )
    [present] = tracks[0].collections[0].get_paths()
    extent = present.get_extents()
    assert (extent.x0, extent.x1) == (300.0, 900.0), "the owner's present frames"
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    assert labels[-1] == "point of interest"
    assert paper_tree.small_text(fig) == []


def test_the_examples_figure(tree, tmp_path):
    paths, model_file = tree
    two = [shots.ae_shot(paths, s, model_file=model_file) for s in (102, 103)]
    fig = shots.draw_examples(two, tmp_path / "fig_examples")
    assert (tmp_path / "fig_examples.png").is_file()
    assert len(fig.axes) == 6
    assert fig.axes[0].get_title().startswith("shot 102 (test): F1 (0-2 s) ")
    assert fig.axes[3].get_title().startswith("shot 103 (test): ")
    assert "point of interest" not in [t.get_text() for t in fig.legends[0].get_texts()]
    assert paper_tree.small_text(fig) == []


def _many(shot: int, n: int) -> pd.DataFrame:
    """`n` points of interest over 0-4 s, the later half after the scored 2 s,
    the last one reaching past the right edge of the axes."""
    rows = []
    for i in range(n):
        t = 4200.0 * i / n
        rows.append(
            {
                "shot": shot,
                "region": i + 1,
                "t_start_ms": t,
                "t_end_ms": t + 150.0,
                "f_lo_khz": 90.0 + (i % 7) * 20,
                "f_hi_khz": 100.0 + (i % 7) * 20 + (i % 3) * 20,
                "pixels": 10 + i,
                "in_scored_window": t + 75.0 < 2000.0,
            }
        )
    return pd.DataFrame(rows)


def _numbers(ax) -> list:
    return [t for t in ax.texts if t.get_text().isdigit()]


def _legend_matches_drawn(fig) -> None:
    drawn = {
        a.get_label()
        for ax in fig.axes
        for a in ax.get_children()
        if isinstance(a.get_label(), str)  # an Axis's label is its Text
    }
    [legend] = fig.legends
    shown = [t.get_text() for t in legend.get_texts()]
    assert shown and set(shown) <= drawn, set(shown) - drawn


def test_points_are_thin_clipped_outlines_numbered_at_most_to_the_cap(tree, tmp_path):
    paths, model_file = tree
    many = _many(102, 3 * shots.LABEL_CAP)
    s = shots.ae_shot(paths, 102, model_file=model_file, poi=many)
    fig = shots.draw_interpreter(s, tmp_path / "fig_interpreter")
    spec = fig.axes[0]
    boxes = [p for p in spec.patches if isinstance(p, Rectangle)]
    assert len(boxes) == len(many)
    assert all(b.get_clip_on() and b.get_linewidth() <= 0.5 for b in boxes)
    assert all(b.get_alpha() is not None and b.get_alpha() < 1 for b in boxes)
    after = [b for b in boxes if b.get_linestyle() != "-"]
    assert len(after) == (~many.in_scored_window).sum(), "after 2 s: drawn dashed"
    numbers = _numbers(spec)
    assert len(numbers) == shots.LABEL_CAP, "more regions in view than the cap"
    (x0, x1), (y0, y1) = spec.get_xlim(), spec.get_ylim()
    seen = many[(many.t_start_ms >= x0) & (many.t_start_ms <= x1)]
    largest = seen.nlargest(shots.LABEL_CAP, "pixels").region.astype(str)
    assert {t.get_text() for t in numbers} == set(largest)
    for t in spec.texts:
        x, y = t.get_position()
        if t.get_transform() is spec.transData:
            assert x0 <= x <= x1 and y0 <= y <= y1, t.get_text()
        assert t.get_clip_on(), t.get_text()


def test_the_scored_window_is_marked(tree, tmp_path):
    paths, model_file = tree
    s = shots.ae_shot(paths, 102, model_file=model_file, poi=_many(102, 8))
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


def test_the_interpreter_shot_breaks_ties_by_the_recorded_rule():
    index = pd.DataFrame(
        {
            "shot": [10, 11, 12, 13, 14],
            "group": ["reviewed"] * 5,
            "split": ["test"] * 5,
            "f1_vs_owner": [0.5] * 5,
        }
    )
    f1 = {10: 1.0, 11: 1.0, 12: 0.99996, 13: 0.99, 14: 1.0}
    poi = pd.DataFrame({"shot": [10] * 5 + [11] * 30 + [12] * 8 + [13] * 2})
    mixed = {11, 12, 13}
    assert shots.interpreter_shot(index, poi, f1=f1, mixed=mixed) == 12
    assert shots.interpreter_shot(index, poi, f1=f1) == 10, "then fewer points"
    even = pd.DataFrame({"shot": [11] * 3 + [10] * 3})
    assert shots.interpreter_shot(index, even, f1=f1) == 10, "then the lower shot"
    assert shots.interpreter_shot(index, poi, f1=f1, mixed={13}) == 10, "F1 first"
    assert "mixed" in shots.INTERPRETER_RULE or "absent" in shots.INTERPRETER_RULE
    assert shots.pick_examples(index, 2, f1=f1) == [10, 13]


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
        shots.ae_shot(paths, 102, model_file=model_file, poi=_many(102, 4)),
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
