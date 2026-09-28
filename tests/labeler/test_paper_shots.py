"""The one-discharge interpreter figure and the AE examples."""

from __future__ import annotations

import dataclasses
import itertools
import re

import numpy as np
import pandas as pd
import pytest
from matplotlib.patches import Rectangle

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


def test_a_point_wholly_off_the_axes_gets_no_legend_key(tree, tmp_path):
    paths, model_file = tree
    off = _poi(102)
    off.loc[1] = [102, 3, 3734.0, 3838.0, 140.0, 152.0]  # drawn to 2050 ms
    off["in_scored_window"] = [True, False]
    s = shots.ae_shot(paths, 102, model_file=model_file, poi=off)
    fig = shots.draw_interpreter(s, tmp_path / "fig_interpreter")
    assert fig.axes[0].get_xlim()[1] < 3734.0
    assert len(fig.axes[0].patches) == 2, "both drawn, the second clipped away"
    shown = [t.get_text() for t in fig.legends[0].get_texts()]
    assert shots.POI_LABEL in shown and shots.POI_AFTER_LABEL not in shown
    _legend_matches_drawn(fig)


def test_region_numbers_never_print_over_each_other(tree, tmp_path):
    paths, model_file = tree
    same = pd.DataFrame(
        [
            {
                "shot": 102,
                "region": k + 1,
                "t_start_ms": 300.0,
                "t_end_ms": 900.0,
                "f_lo_khz": 120.0,
                "f_hi_khz": 200.0,
                "pixels": 50 - k,
            }
            for k in range(5)
        ]
    )
    s = shots.ae_shot(paths, 102, model_file=model_file, poi=same)
    fig = shots.draw_examples([s], tmp_path / "fig_examples")
    fig.draw_without_rendering()
    numbers = _numbers(fig.axes[0])
    boxes = [t.get_window_extent() for t in numbers]
    assert not any(a.overlaps(b) for a, b in itertools.combinations(boxes, 2))
    corners = {(x, y) for x in (300.0, 900.0) for y in (120.0, 200.0)}
    assert {t.get_position() for t in numbers} == corners, "each at its own box"
    assert [t.get_text() for t in numbers] == ["1", "2", "3", "4"], (
        "largest first; the fifth finds every corner taken and is left off"
    )
    # two different boxes whose top left corners meet, as 170675's 33 and 37:
    # the second number moves, and only to a corner of its own box
    two = pd.DataFrame(
        [
            {
                "shot": 102,
                "region": region,
                "t_start_ms": t0,
                "t_end_ms": t1,
                "f_lo_khz": f0,
                "f_hi_khz": f1,
                "pixels": pixels,
            }
            for region, t0, t1, f0, f1, pixels in (
                (1, 300.0, 900.0, 120.0, 200.0, 50),
                (2, 320.0, 1500.0, 100.0, 205.0, 40),
            )
        ]
    )
    s = shots.ae_shot(paths, 102, model_file=model_file, poi=two)
    fig = shots.draw_examples([s], tmp_path / "fig_two")
    fig.draw_without_rendering()
    own = {
        int(r.region): {
            (x, y) for x in (r.t_start_ms, r.t_end_ms) for y in (r.f_lo_khz, r.f_hi_khz)
        }
        for r in two.itertuples()
    }
    numbers = _numbers(fig.axes[0])
    assert sorted(t.get_text() for t in numbers) == ["1", "2"]
    assert all(t.get_position() in own[int(t.get_text())] for t in numbers), (
        "each at a corner of its own box"
    )
    boxes = [t.get_window_extent() for t in numbers]
    assert not boxes[0].overlaps(boxes[1])
