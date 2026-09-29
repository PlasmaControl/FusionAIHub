"""v3's pictures: no 80 kHz line anywhere, no 0-2 s line for a whole-window
version, and the gallery's F1 over the window the version is scored on."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from matplotlib.figure import Figure

from labeler.ae import xpower
from labeler.ae.seg import poi
from labeler.ae.xpower import evaluate, extend, gallery, model_dir, train
from labeler.ae.xpower.data import BAND_KHZ, FULL_BAND_KHZ
from labeler.ae.xpower.model import FrameCNN, FrameCNNConfig
from labeler.events.review.rows import Grid

from . import ae_tree, round3_tree


@pytest.fixture
def figures(monkeypatch):
    drawn = []

    def savefig(fig, path, **kwargs):
        drawn.append(fig)
        path.write_bytes(b"picture")

    monkeypatch.setattr(Figure, "savefig", savefig)
    return drawn


def _at_80_khz(fig) -> list:
    return [
        line
        for ax in fig.axes
        for line in ax.lines
        if list(line.get_ydata()) == [80.0, 80.0]
    ]


def _scored(fig) -> list:
    return [
        line
        for ax in fig.axes
        for line in ax.lines
        if str(line.get_label()).startswith("scored")
    ]


def _draw(path, **kwargs):
    gallery.draw(
        path,
        title="t",
        grid=Grid(0, 10, 300),
        values=np.zeros((3, 257, 300), dtype=np.uint8),
        y0=0,
        dy=500 / 512,
        first=0,
        prob=np.zeros(300),
        threshold=0.5,
        **kwargs,
    )


def test_the_gallery_draws_no_80_khz_line(tmp_path, figures):
    _draw(tmp_path / "a.jpg")
    assert _at_80_khz(figures[0]) == []
    assert not hasattr(gallery, "BAND_LINE_KHZ")


def test_a_whole_window_picture_has_no_scored_line(tmp_path, figures):
    _draw(tmp_path / "a.jpg", scored_until_ms=None)
    _draw(tmp_path / "b.jpg")
    assert _scored(figures[0]) == []
    texts = [t.get_text() for ax in figures[0].axes for t in ax.texts]
    assert not any(text.startswith("scored") for text in texts)
    lines = _scored(figures[1])
    assert len(lines) == len(figures[1].axes) == 5
    assert {line.get_label() for line in lines} == {"scored: 0-2 s"}


def test_the_poi_pictures_draw_no_80_khz_line(tmp_path, figures):
    labelled = np.zeros((257, 32), dtype=int)
    poi.draw(
        tmp_path / "p.jpg",
        title="t",
        grid=Grid(0.0, 100.0, 32),
        row=np.zeros_like(labelled),
        y0=0.0,
        dy=ae_tree.DY,
        labelled=labelled,
        found=[],
    )
    assert _at_80_khz(figures[0]) == []
    assert _scored(figures[0])  # SegNet v1's pictures keep their 0-2 s line
    assert not hasattr(poi, "BAND_LINE_KHZ")


def _model(paths, version, candidate, band):
    torch.manual_seed(0)
    out = model_dir(paths, version) / candidate
    train.save(
        out,
        FrameCNN(FrameCNNConfig(width=4)),
        threshold=0.5,
        split={101: "test"},
        history=[{"epoch": 1, "val_f1": 0.5, "kept": True}],
        config=train.TrainConfig(),
        band_khz=band,
        labels_file=xpower.snapshot_file(paths, version),
        candidate=candidate,
        version=version,
    )
    return out / "model.pt"


def _fires_on_the_first_ae(model, rows, first, n, *, band, context=20):
    """P(AE) 0.9 on the first AE span's frames (30-89) only, 0.1 elsewhere."""
    frame = first + np.arange(n)
    prob = np.where((frame >= 30) & (frame < 90), 0.9, 0.1).astype(np.float32)
    return prob, np.ones(n, dtype=bool)


@pytest.mark.parametrize(
    "version, candidate, band, window, f1, title, until",
    [
        # 60 of 100 present frames over 0-3 s: F1 120 / 160; all 60 of 0-2 s.
        ("v3", "band0-mhd3", FULL_BAND_KHZ, "whole", 0.75, "whole window 0.75", None),
        ("v2", "band80-mhd3", BAND_KHZ, "0-2 s", 1.0, "0-2 s 1.00", 2000.0),
    ],
)
def test_a_picture_gives_the_f1_of_the_versions_scored_window(
    tmp_path, monkeypatch, version, candidate, band, window, f1, title, until
):
    paths = round3_tree.build(tmp_path, {101: "valid"})
    ae_tree.env(monkeypatch, paths)
    ae_tree.snapshot(paths, monkeypatch, version)
    model_file = _model(paths, version, candidate, band)
    for module in (gallery, evaluate):
        monkeypatch.setattr(module, "probabilities", _fires_on_the_first_ae)
    seen = []
    monkeypatch.setattr(gallery, "draw", lambda path, **kw: seen.append(kw))
    gallery._init(
        str(model_file),
        str(paths.root),
        str(paths.label_tables),
        str(paths.corpus),
        version,
        str(tmp_path / "out"),
    )
    row = gallery.picture(101)
    assert row["scored_window"] == window and row["f1_scored"] == f1
    assert row["f1_0_2s"] == 1.0 and row["f1_vs_owner"] == 0.75
    assert f"F1 vs owner, {title}" in seen[0]["title"]
    assert seen[0]["scored_until_ms"] == until
    assert gallery.INDEX_COLUMNS[-3:] == ("f1_0_2s", "scored_window", "f1_scored")


def test_the_extension_draws_the_versions_scored_window(tmp_path, monkeypatch):
    paths = round3_tree.build(tmp_path, {101: "train"})
    ae_tree.env(monkeypatch, paths)
    ae_tree.snapshot(paths, monkeypatch, "v3")
    model_file = _model(paths, "v3", "band0-mhd3", FULL_BAND_KHZ)
    ae_tree.corpus(tmp_path, [201])
    seen = []
    monkeypatch.setattr(extend, "draw", lambda path, **kw: seen.append(kw))
    for version in ("v3", "v2"):
        extend._init_shard(
            str(model_file),
            str(paths.root),
            str(paths.corpus),
            True,
            version,
            str(tmp_path / version),
        )
        extend.label_shot((201, 2024, 0, 600))
    assert [kw["scored_until_ms"] for kw in seen] == [None, 2000.0]
