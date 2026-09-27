"""A version setting routes all xpower commands and their output metadata."""

import json

import numpy as np
import pandas as pd
import pytest

from labeler.ae.xpower import evaluate, extend, frontier, gallery, model_dir, train

from . import ae_tree
from .test_ae_xpower_evaluate import _Fires
from .test_ae_xpower_extend import _three_shots


@pytest.mark.parametrize("override", [False, True])
def test_version_routes_training_scoring_frontier_and_gallery(
    tmp_path, monkeypatch, override
):
    paths = ae_tree.build(tmp_path, {101: "train", 102: "valid"})
    ae_tree.env(monkeypatch, paths)
    models = tmp_path / "custom-models" if override else model_dir(paths, "v2")
    monkeypatch.setattr(
        train,
        "fit",
        lambda *a, **kw: (train.FrameCNN(train.FrameCNNConfig(width=4)), [], 0.5),
    )
    args = ["--version", "v2"]
    out_args = ["--out", str(models / "band80-mhd3")] if override else []
    assert train.main([*args, "--candidate", "band80-mhd3", *out_args]) == 0
    _, blob = train.load(models / "band80-mhd3/model.pt")
    assert blob["version"] == "v2"
    model_args = [*args, *(["--models", str(models)] if override else [])]
    monkeypatch.setattr(
        evaluate, "load_seldnet", lambda paths: _Fires(np.ones(783, bool))
    )
    assert evaluate.main([*model_args, "--choose"]) == 0
    assert evaluate.main(model_args) == 0
    assert frontier.main(model_args) == 0
    assert json.loads((models / "chosen.json").read_text())["version"] == "v2"
    assert json.loads((models / "evaluation.json").read_text())["meta"]["version"] == (
        "v2"
    )
    assert set(pd.read_csv(models / "validation_frontier.csv").version) == {"v2"}
    titles = []
    draw = gallery.draw

    def capture(path, **kwargs):
        titles.append(kwargs["title"])
        draw(path, **kwargs)

    monkeypatch.setattr(gallery, "draw", capture)
    assert gallery.main([*model_args, "--workers", "1"]) == 0
    pictures = gallery.gallery_dir(paths, "v2")
    assert (pictures / "reviewed/101.jpg").is_file()
    assert all("ae_xpower v2" in title for title in titles)
    assert set(pd.read_csv(pictures / "index.csv").version) == {"v2"}
    assert not model_dir(paths).exists()
    assert not gallery.gallery_dir(paths).exists()


def test_version_routes_extension_tables_and_pictures(tmp_path, monkeypatch):
    paths, original = _three_shots(tmp_path, monkeypatch)
    ae_tree.env(monkeypatch, paths)
    models = model_dir(paths, "v2")
    original.rename(models)
    args = ["--version", "v2", "--of", "1"]
    assert extend.main(args) == 0
    assert extend.main([*args, "--merge"]) == 0
    out = extend.suggestions_dir(paths, "v2")
    table = out / "alfven_eigenmode_suggest_ae_xpower_v2.csv"
    assert table.is_file()
    assert json.loads(table.with_suffix(".meta.json").read_text())["version"] == "v2"
    # Exercise the real extension picture worker too, on synthetic corpus CO2.
    ae_tree.corpus(tmp_path, [201])
    titles = []
    draw = extend.draw

    def capture(path, **kwargs):
        titles.append(kwargs["title"])
        draw(path, **kwargs)

    monkeypatch.setattr(extend, "draw", capture)
    with monkeypatch.context() as mp:
        mp.setattr(extend, "run_all", gallery.run_all)
        assert extend.main([*args, "--limit", "1"]) == 0
    assert "ae_xpower v2" in titles[0]
    assert (gallery.gallery_dir(paths, "v2") / "extension/201.jpg").is_file()
    assert not original.exists()
    assert not extend.suggestions_dir(paths).exists()
    assert not gallery.gallery_dir(paths).exists()
