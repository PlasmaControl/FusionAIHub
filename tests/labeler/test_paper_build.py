"""The paper's build: what the inputs allow, its manifest, the manuscript's copy."""

from __future__ import annotations

import json

import pytest

from labeler.ae import xpower
from labeler.ae.xpower import gallery
from labeler.paper import build, shots

from . import ae_tree
from . import paper_tree as tree


@pytest.fixture
def runs(tmp_path, monkeypatch):
    paths = ae_tree.build(tmp_path, {101: "train", 102: "valid", 103: "valid"})
    models = ae_tree.chosen(paths, {101: "train", 102: "test", 103: "test"})
    (models / "evaluation.json").write_text(json.dumps(tree.ae_evaluation()))
    gallery.write_index(
        gallery.gallery_dir(paths) / "index.csv",
        [
            {"shot": 101, "group": "reviewed", "split": "train", "f1_vs_owner": 0.99},
            {"shot": 102, "group": "reviewed", "split": "test", "f1_vs_owner": 0.8},
            {"shot": 103, "group": "reviewed", "split": "test", "f1_vs_owner": 0.6},
        ],
    )
    ae_tree.env(monkeypatch, paths)
    return paths


def test_the_build_draws_what_its_inputs_allow(runs, tmp_path, capsys):
    out, dest = tmp_path / "paper", tmp_path / "manuscript" / "figures"
    assert build.main(["--out", str(out), "--copy-to", str(dest)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["products"] == [
        "fig_coverage",
        "fig_examples",
        "fig_interpreter",
        "fig_mhd",
        "fig_scores",
        "table_ae_scores",
        "table_datasets",
    ]
    missing = f"missing {build.inputs(runs)['seg_evaluation']}"
    assert printed["skipped"] == {
        "fig_segmentation": missing,
        "table_seg_scores": missing,
    }
    assert printed["copied"] == sorted(p.name for p in dest.iterdir())
    assert len(printed["copied"]) == 7
    assert all(name.endswith((".pdf", ".tex")) for name in printed["copied"])
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["interpreter_shot"] == 102, "the best test shot; no POI yet"
    assert manifest["example_shots"] == [102, 103]
    assert manifest["interpreter_rule"] == shots.INTERPRETER_RULE
    assert manifest["example_rule"] == shots.EXAMPLES_RULE
    assert sorted(manifest["shot_f1"]) == ["102", "103"], "the shots drawn"
    assert set(manifest["shot_f1"]["102"]) == {"f1_0_2s", "f1_window"}
    assert sorted(manifest["inputs"]) == [
        "ae_chosen",
        "ae_evaluation",
        "ae_labels",
        "gallery_index",
    ]
    for name, files in manifest["products"].items():
        assert all((out / f).is_file() for f in files), name


def test_without_the_chosen_model_the_shot_figures_wait(runs, tmp_path):
    (xpower.model_dir(runs) / "chosen.json").unlink()
    manifest = build.build(runs, tmp_path / "paper")
    missing = f"missing {build.inputs(runs)['ae_chosen']}"
    assert manifest["skipped"]["fig_interpreter"] == missing
    assert manifest["skipped"]["fig_examples"] == missing
    assert "fig_coverage" in manifest["products"], "the split is left empty"
    assert "interpreter_shot" not in manifest


def test_the_interpreter_shot_can_be_named(runs, tmp_path):
    manifest = build.build(runs, tmp_path / "paper", shot=103, examples=1)
    assert (manifest["interpreter_shot"], manifest["example_shots"]) == (103, [102])
    assert manifest["interpreter_rule"] == "named by --shot"


def test_products_drawn_without_an_input_are_recorded_as_partial(runs, tmp_path):
    found = build.inputs(runs)
    manifest = build.build(runs, tmp_path / "paper")
    no_summary, no_poi = [str(found["summary"])], [str(found["poi"])]
    assert manifest["partial"] == {
        "fig_coverage": no_summary,
        "table_datasets": no_summary,
        "fig_interpreter": no_poi,
        "fig_examples": no_poi,
    }
    assert set(manifest["partial"]) <= set(manifest["products"])
    (xpower.model_dir(runs) / "chosen.json").unlink()
    manifest = build.build(runs, tmp_path / "paper")
    no_split = [str(found["ae_chosen"]), *no_summary]
    assert manifest["partial"] == {"fig_coverage": no_split, "table_datasets": no_split}
