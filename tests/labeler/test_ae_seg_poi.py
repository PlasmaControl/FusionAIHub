"""Points of interest: a region per AE mode the model draws, and a picture per shot."""

from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image

from labeler.ae.seg import model_dir, poi, poi_dir, pseudo, train
from labeler.ae.seg.model import SegNet, SegNetConfig
from labeler.events.review.rows import Grid

from . import ae_tree

DY = 500 / 512


def test_a_region_in_the_band_big_enough_is_a_point():
    prob = np.zeros((257, 100), dtype=np.float32)
    prob[150:153, 10:20] = 0.9  # 30 pixels at ~147 kHz
    prob[151, 15] = 0.99  # its peak
    prob[200, 50:55] = 0.9  # 5 pixels: too small
    prob[40:45, 60:80] = 0.9  # 100 pixels at ~41 kHz: below the band
    found, labelled = poi.points(7, prob, 0.5, Grid(100.0, 2.0, 100), 0.0, DY)
    [p] = found
    assert (p["region"], p["pixels"], p["shot"]) == (1, 30, 7)
    assert (p["t_start_ms"], p["t_end_ms"]) == (120.0, 140.0)
    assert (p["f_lo_khz"], p["f_hi_khz"]) == (
        round(149.5 * DY, 2),
        round(152.5 * DY, 2),
    )
    assert (p["t_peak_ms"], p["f_peak_khz"]) == (131.0, round(151 * DY, 2))
    assert p["max_prob"] == 0.99 and p["method"] == "ae_seg-v1"
    assert (labelled == 1).sum() == 30 and labelled.max() == 1


def test_the_table_keeps_other_shots_and_replaces_a_redrawn_one(tmp_path):
    path = tmp_path / "poi.csv"
    row = {c: 0 for c in poi.POI_COLUMNS}
    poi.write_points(path, [1, 2], [{**row, "shot": 1}, {**row, "shot": 2}])
    table = poi.write_points(path, [2], [{**row, "shot": 2, "region": 1}] * 2)
    assert table.shot.tolist() == [1, 2, 2]
    assert tuple(pd.read_csv(path).columns) == poi.POI_COLUMNS


def _model(paths, threshold):
    train.save(
        model_dir(paths),
        SegNet(SegNetConfig(width=4)),
        threshold=threshold,
        split={},
        history=[],
        config=train.TrainConfig(width=4),
        inputs={},
    )


def test_the_command_finds_points_on_stores_and_draws_them(tmp_path, monkeypatch):
    paths = ae_tree.build(tmp_path, {101: "train", 102: "valid"}, tokeye_dt=0.256)
    ae_tree.env(monkeypatch, paths)
    assert pseudo.main([]) == 0
    _model(paths, 0.0)  # everything is on: one region per shot, the whole band
    assert poi.main(["--workers", "1"]) == 0
    table = pd.read_csv(poi_dir(paths) / "poi.csv")
    assert table.shot.tolist() == [101, 102] and (table.source == "store").all()
    assert (table.f_lo_khz.round(0) == 80).all() and (table.f_hi_khz > 249).all()
    with Image.open(poi.gallery_dir(paths) / "101.jpg") as image:
        assert image.format == "JPEG" and image.size == (1600, 1000)


def test_the_command_reads_new_shots_from_the_corpus(tmp_path, monkeypatch):
    paths = ae_tree.build(tmp_path, {101: "train"})
    ae_tree.corpus(tmp_path, [201])
    ae_tree.env(monkeypatch, paths)
    _model(paths, 0.0)
    assert poi.main(["--from-corpus", "--shots", "201", "--workers", "1"]) == 0
    table = pd.read_csv(poi_dir(paths) / "poi.csv")
    assert table.shot.tolist() == [201] and table.source.tolist() == ["corpus"]
    with Image.open(poi.gallery_dir(paths) / "201.jpg") as image:
        assert image.size == (1600, 700), "no pseudo-mask: one panel"


def test_no_region_over_the_threshold_is_no_point(tmp_path, monkeypatch):
    paths = ae_tree.build(tmp_path, {101: "train"})
    ae_tree.env(monkeypatch, paths)
    _model(paths, 1.01)
    assert poi.main(["--workers", "1", "--no-pictures"]) == 0
    assert pd.read_csv(poi_dir(paths) / "poi.csv").empty
    assert not poi.gallery_dir(paths).exists()
