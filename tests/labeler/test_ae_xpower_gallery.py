"""The AE gallery: a JPEG per shot and an index beside the folders."""

from __future__ import annotations

import csv

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from labeler.ae.xpower import gallery
from labeler.events.review import labels
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
    assert index.snapshot_sha256.tolist() == ["", "", ""]  # v1 has no snapshot


@pytest.mark.parametrize("with_extension", [False, True])
def test_a_newly_reviewed_shot_replaces_its_row_and_picture(
    tmp_path, monkeypatch, with_extension
):
    paths = ae_tree.build(tmp_path, {101: "train", 102: "valid"}, reviewed={101})
    ae_tree.chosen(paths, {101: "train"})
    ae_tree.env(monkeypatch, paths)
    args = ["--workers", "1", "--shots", "102"]
    assert gallery.main(args) == 0
    root = gallery.gallery_dir(paths)
    index_path = root / "index.csv"
    old_picture = root / "unreviewed" / "102.jpg"
    assert old_picture.is_file()
    extension = {"shot": 102, "group": "extension", "file": "extension/102.jpg"}
    if with_extension:
        picture = root / extension["file"]
        picture.parent.mkdir()
        extension_bytes = old_picture.read_bytes()
        picture.write_bytes(extension_bytes)
        gallery.write_index(index_path, [extension])

    event = gallery.event_dir(paths)
    labels.save(event, 102, labels.read_source(event)[102], source="test")
    assert gallery.main(args) == 0
    index = pd.read_csv(index_path, dtype=str, keep_default_na=False)
    expected = {"reviewed/102.jpg"}
    if with_extension:
        expected.add("extension/102.jpg")
    assert len(index) == len(expected)
    assert set(index.file) == expected
    assert {str(p.relative_to(root)) for p in root.glob("*/*.jpg")} == expected
    reviewed = index[index.group == "reviewed"].iloc[0]
    assert reviewed.split == "after training" and reviewed.f1_vs_owner != ""

    if with_extension:
        assert picture.read_bytes() == extension_bytes
        gallery.write_index(index_path, [{**extension, "model_present_frames": 7}])
        redrawn = pd.read_csv(index_path, dtype=str, keep_default_na=False)
        assert len(redrawn) == 2 and set(redrawn.file) == expected
        assert redrawn[redrawn.group == "reviewed"].iloc[0].equals(reviewed)
        extended = redrawn[redrawn.group == "extension"].iloc[0]
        assert int(extended.model_present_frames) == 7


def test_the_title_gives_the_f1_over_the_scored_0_to_2_s(tmp_path, monkeypatch):
    """The owner's window runs to 2.2 s, AE present again after 2 s; every
    frame is called present, so the whole window's F1 (80 of 220 frames
    present) differs from the scored 0-2 s's (60 of 200)."""
    paths = ae_tree.build(tmp_path, {101: "train", 102: "valid"}, reviewed={101})
    spans = ((0, 300, 0), (300, 900, 1), (900, 2000, 0), (2000, 2200, 1))
    saved = labels.labels_path(gallery.event_dir(paths))
    saved.write_text(
        "shot,category,t_start,t_end,confidence\n"
        + "".join(f"101,{c},{a},{b},\n" for a, b, c in spans)
    )
    models = ae_tree.chosen(paths, {101: "train"})
    file = models / "band80-mhd3" / "model.pt"
    torch.save(torch.load(file, weights_only=False) | {"threshold": 0.0}, file)
    ae_tree.env(monkeypatch, paths)
    titles, draw = {}, gallery.draw
    monkeypatch.setattr(
        gallery, "draw", lambda f, **kw: draw(f, **kw) or titles.update({f.stem: kw})
    )
    assert gallery.main(["--workers", "1"]) == 0
    index = pd.read_csv(gallery.gallery_dir(paths) / "index.csv", dtype=str)
    index = index.fillna("").set_index("shot")
    assert tuple(index.reset_index()) == gallery.INDEX_COLUMNS
    assert index.loc["101", "f1_vs_owner"] == str(round(160 / 300, 4))
    assert index.loc["101", "f1_0_2s"] == str(round(120 / 260, 4))
    assert titles["101"]["title"].endswith(", F1 vs owner, 0-2 s 0.46")
    assert index.loc["102", "f1_0_2s"] == "", "not reviewed"
