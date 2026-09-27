"""The one-discharge interpreter figure and the AE examples."""

from __future__ import annotations

import pandas as pd
import pytest

from labeler.ae.xpower.evaluate import chosen_model
from labeler.paper import COMING, shots

from . import ae_tree


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
    assert [t.get_text() for t in spec.texts] == ["AE 1"]
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


def test_the_examples_figure(tree, tmp_path):
    paths, model_file = tree
    two = [shots.ae_shot(paths, s, model_file=model_file) for s in (102, 103)]
    fig = shots.draw_examples(two, tmp_path / "fig_examples")
    assert (tmp_path / "fig_examples.png").is_file()
    assert len(fig.axes) == 6
    assert fig.axes[0].get_title().startswith("shot 102 (test): F1 against the owner")
    assert fig.axes[3].get_title().startswith("shot 103 (test): ")
    assert "point of interest" not in [t.get_text() for t in fig.legends[0].get_texts()]
