"""The AE gallery: a JPEG per shot and an index beside the folders."""

from __future__ import annotations

import csv

import numpy as np
import pandas as pd
from PIL import Image

from labeler.ae.xpower import gallery
from labeler.events.review.rows import Grid

from . import ae_tree


def test_a_picture_is_a_1600_by_1000_jpeg(tmp_path):
    grid = Grid(-50.0, 2.048, 600)
    values = np.random.default_rng(0).integers(0, 255, (3, 257, 600), dtype=np.uint8)
    prob = np.linspace(0, 1, 100)
    reference = np.r_[np.zeros(40), np.ones(30), np.full(10, 2), np.full(20, 3)]
    mhd = np.zeros(100, bool)
    mhd[5:9] = True
    for name, ref in (("1.jpg", reference), ("2.jpg", None)):
        gallery.draw(
            tmp_path / "a" / name,
            title=name,
            grid=grid,
            values=values,
            y0=0.0,
            dy=500 / 512,
            first=0,
            prob=prob,
            threshold=0.5,
            reference=ref,
            mhd=mhd,
        )
        with Image.open(tmp_path / "a" / name) as image:
            assert image.format == "JPEG" and image.size == (1600, 1000)


def test_the_index_merges_a_redrawn_shot_over_its_old_row(tmp_path):
    path = tmp_path / "index.csv"
    gallery.write_index(
        path,
        [
            {"shot": 2, "group": "reviewed", "split": "train"},
            {"shot": 1, "group": "reviewed", "split": "val"},
        ],
    )
    gallery.write_index(path, [{"shot": 2, "group": "reviewed", "split": "test"}])
    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert [(r["shot"], r["split"]) for r in rows] == [("1", "val"), ("2", "test")]
    assert tuple(rows[0]) == gallery.INDEX_COLUMNS


def test_the_gallery_draws_reviewed_and_unreviewed_shots(tmp_path, monkeypatch):
    splits = {101: "train", 102: "valid", 103: "valid"}
    paths = ae_tree.build(tmp_path, splits, reviewed={101, 102})
    ae_tree.chosen(paths, {101: "train", 102: "test"})
    ae_tree.env(monkeypatch, paths)
    assert gallery.main(["--workers", "1"]) == 0
    root = gallery.gallery_dir(paths)
    assert (root / "reviewed" / "101.jpg").is_file()
    assert (root / "unreviewed" / "103.jpg").is_file()
    index = pd.read_csv(root / "index.csv", keep_default_na=False)
    assert index.shot.tolist() == [101, 102, 103]
    assert index.group.tolist() == ["reviewed", "reviewed", "unreviewed"]
    assert index.split.tolist() == ["train", "test", "unreviewed"]
    assert index.reference_present_frames.tolist() == [60, 60, 60]
    assert index.f1_vs_owner.iloc[2] == ""
