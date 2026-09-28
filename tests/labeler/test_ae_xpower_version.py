"""A version setting routes all xpower commands and their output metadata."""

import json

import numpy as np
import pandas as pd
import pytest

from labeler.ae.xpower import evaluate, extend, frontier, gallery, model_dir, train

from . import ae_tree
from .test_ae_xpower_evaluate import _Fires
from .test_ae_xpower_extend import _three_shots

# A version chosen on validation, as v1 is, under another name. (v2 is chosen by
# cross-validation, which test_ae_xpower_cv and test_ae_xpower_final route.)
OTHER = "v9"


@pytest.fixture(autouse=True)
def _other_version(monkeypatch):
    monkeypatch.setitem(train.CANDIDATES_BY_VERSION, OTHER, train.CANDIDATES)


@pytest.mark.parametrize("override", [False, True])
def test_version_routes_training_scoring_frontier_and_gallery(
    tmp_path, monkeypatch, override
):
    paths = ae_tree.build(tmp_path, {101: "train", 102: "valid"})
    ae_tree.env(monkeypatch, paths)
    # Any directory, so long as it is named for the version.
    models = tmp_path / "custom" / OTHER if override else model_dir(paths, OTHER)
    monkeypatch.setattr(
        train,
        "fit",
        lambda *a, **kw: (train.FrameCNN(train.FrameCNNConfig(width=4)), [], 0.5),
    )
    args = ["--version", OTHER]
    out_args = ["--out", str(models / "band80-mhd3")] if override else []
    assert train.main([*args, "--candidate", "band80-mhd3", *out_args]) == 0
    _, blob = train.load(models / "band80-mhd3/model.pt")
    assert blob["version"] == OTHER
    model_args = [*args, *(["--models", str(models)] if override else [])]
    monkeypatch.setattr(
        evaluate, "load_seldnet", lambda paths: _Fires(np.ones(783, bool))
    )
    assert evaluate.main([*model_args, "--choose"]) == 0
    assert evaluate.main(model_args) == 0
    assert frontier.main(model_args) == 0
    assert json.loads((models / "chosen.json").read_text())["version"] == OTHER
    assert json.loads((models / "evaluation.json").read_text())["meta"]["version"] == (
        OTHER
    )
    assert set(pd.read_csv(models / "validation_frontier.csv").version) == {OTHER}
    titles = []
    draw = gallery.draw

    def capture(path, **kwargs):
        titles.append(kwargs["title"])
        draw(path, **kwargs)

    monkeypatch.setattr(gallery, "draw", capture)
    assert gallery.main([*model_args, "--workers", "1"]) == 0
    pictures = gallery.gallery_dir(paths, OTHER)
    assert (pictures / "reviewed/101.jpg").is_file()
    assert all(f"ae_xpower {OTHER}" in title for title in titles)
    assert set(pd.read_csv(pictures / "index.csv").version) == {OTHER}
    assert not model_dir(paths).exists()
    assert not gallery.gallery_dir(paths).exists()


def test_version_routes_extension_tables_and_pictures(tmp_path, monkeypatch):
    paths, models = _three_shots(tmp_path, monkeypatch, OTHER)
    ae_tree.env(monkeypatch, paths)
    assert models == model_dir(paths, OTHER)
    args = ["--version", OTHER, "--of", "1"]
    assert extend.main(args) == 0
    assert extend.main([*args, "--merge"]) == 0
    out = extend.suggestions_dir(paths, OTHER)
    table = out / f"alfven_eigenmode_suggest_ae_xpower_{OTHER}.csv"
    assert table.is_file()
    assert json.loads(table.with_suffix(".meta.json").read_text())["version"] == OTHER
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
    assert f"ae_xpower {OTHER}" in titles[0]
    # A pilot (--limit) draws under its own shards/pilot/, not the gallery.
    assert (out / "shards/pilot/extension/201.jpg").is_file()
    assert not (gallery.gallery_dir(paths, OTHER) / "extension/201.jpg").exists()
    assert not model_dir(paths).exists()
    assert not extend.suggestions_dir(paths).exists()
    assert not gallery.gallery_dir(paths).exists()
