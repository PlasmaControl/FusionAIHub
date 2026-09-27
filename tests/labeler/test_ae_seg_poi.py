"""Points of interest: a region per AE mode the model draws, and a picture per shot."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from labeler.ae.seg import model_dir, poi, poi_dir, pseudo, train
from labeler.ae.seg.model import SegNet, SegNetConfig
from labeler.config import Paths, sha256_of
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
    meta = json.loads((poi_dir(paths) / "meta.json").read_text())
    assert meta["model_sha256"] == sha256_of(model_dir(paths) / "model.pt")
    assert meta["threshold"] == 0.0 and meta["MIN_POI_PIXELS"] == 20
    assert meta["shots"] == [101, 102]
    assert meta["points_before_2s"] == int((table.t_peak_ms < 2000).sum())
    assert meta["points_after_2s"] == int((table.t_peak_ms >= 2000).sum())
    assert meta["git_sha"] and meta["made_at"]
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


@pytest.mark.parametrize("review", [None, (2, 5)])
def test_pictures_explain_review_number_only_the_largest_and_mark_scored_time(
    tmp_path, monkeypatch, review
):
    from matplotlib.figure import Figure

    figures = []
    grid = Grid(0.0, 100.0, 32)
    labelled = np.zeros((257, 32), dtype=int)
    labelled[100, ::2] = 1
    found = [
        {
            "region": k,
            "pixels": 100 if k == 32 else 20,
            "t_start_ms": k * 90,
            "f_hi_khz": 101,
        }
        for k in range(1, 33)
    ]

    # atomic_path expects its temporary file to have been written by savefig.
    def savefig(fig, path, **kwargs):
        figures.append(fig)
        path.write_bytes(b"picture")

    monkeypatch.setattr(Figure, "savefig", savefig)
    poi.draw(
        tmp_path / "picture.jpg",
        title="test",
        grid=grid,
        row=np.zeros_like(labelled),
        y0=0.0,
        dy=DY,
        labelled=labelled,
        found=found,
        pseudo=np.zeros_like(labelled),
        pseudo_review=review,
    )
    [fig] = figures
    assert fig.axes[0].get_title(loc="left") == (
        "32 regions the model draws (the 30 largest numbered; all in poi.csv)"
    )
    numbered = {int(t.get_text()) for t in fig.axes[0].texts if t.get_text().isdigit()}
    assert numbered == {*range(1, 30), 32}
    assert len(fig.axes[0].collections) == 1, "every region is still contoured"
    expected = (
        "pseudo-mask (TokEye inside the owner's AE frames; regions not reviewed)"
        if review is None
        else "pseudo-mask after the owner's region review (2 of 5 rejected)"
    )
    assert fig.axes[1].get_title(loc="left") == expected
    for ax in fig.axes:
        scored = [line for line in ax.lines if line.get_label() == "scored: 0-2 s"]
        assert len(scored) == 1
        assert list(scored[0].get_xdata()) == [2000, 2000]
        assert scored[0].get_linestyle() == "--"
        assert any(t.get_text() == "scored: 0-2 s" for t in ax.texts)


@pytest.mark.parametrize("current", [False, True])
def test_only_a_current_region_decision_is_called_reviewed(
    tmp_path, monkeypatch, current
):
    from labeler.ae.seg import regions

    paths = ae_tree.build(tmp_path, {101: "train"}, tokeye_dt=0.256)
    ae_tree.env(monkeypatch, paths)
    assert pseudo.main([]) == 0
    _model(paths, 1.01)
    regions.save_decision(
        paths.label_tables / "alfven_eigenmode",
        101,
        [1],
        pseudo_sha256=(
            regions.file_sha256(regions.pseudo_file(paths, 101)) if current else "old"
        ),
    )
    captured = []
    monkeypatch.setattr(poi, "draw", lambda path, **kwargs: captured.append(kwargs))
    assert poi.main(["--workers", "1"]) == 0
    assert captured[0]["pseudo_review"] == ((1, 1) if current else None)


@pytest.mark.parametrize("g3", [False, None])
def test_points_keep_their_regions_and_gain_window_flags(tmp_path, monkeypatch, g3):
    paths = Paths(
        root=tmp_path / "root",
        label_tables=tmp_path / "events",
        corpus=tmp_path / "corpus",
    )
    ae_tree.env(monkeypatch, paths)
    _model(paths, 0.5)
    if g3 is not None:
        (model_dir(paths) / "evaluation.json").write_text(
            json.dumps({"bar": {"G1": True, "G2": True, "G3": g3}})
        )
    paths.catalog.mkdir()
    (paths.catalog / "population.csv").write_text(
        "shot,window_start_ms,window_end_ms\n101,0,2000\n102,0,3000\n104,,\n"
    )
    # The AE Ip log fills only shots the population lacks: 103, not 101 or 104.
    log = poi.ae_ip_log(paths)
    log.parent.mkdir(parents=True)
    log.write_text(
        "".join(
            json.dumps(
                {
                    "shot": shot,
                    "version": 3,
                    "status": "ok",
                    "window_start_ms": start,
                    "window_end_ms": 3000,
                    "ip_peak_ma": 1.0,
                    "ip_sha256": "a" * 64,
                    "run": "b" * 32,
                }
            )
            + "\n"
            for shot, start in ((101, 1500), (103, 0), (104, 0))
        )
    )
    log.with_name("ip_runs.jsonl").write_text(json.dumps({"run": "b" * 32}) + "\n")
    grid = Grid(-1000.5, 1, 6100)
    prob = np.zeros((257, grid.n), dtype=np.float32)
    peaks = [-1, 0, 1999, 2000, 4000]
    for row, peak in zip(range(100, 115, 3), peaks):
        column = peak + 1000
        prob[row, column - 10 : column + 10] = 0.8
        prob[row, column] = 0.9
    expected, _ = poi.points(101, prob, 0.5, grid, 0, DY)
    monkeypatch.setattr(poi, "predict", lambda *a: prob)
    monkeypatch.setattr(
        poi,
        "shot_rows",
        lambda *a: (grid, np.zeros((3, 257, grid.n), np.uint8), 0, DY),
    )
    assert (
        poi.main(
            ["--shots", "101", "102", "103", "104", "--no-pictures", "--workers", "1"]
        )
        == 0
    )
    table = pd.read_csv(poi_dir(paths) / "poi.csv")
    assert tuple(table.columns[-2:]) == ("in_scored_window", "in_plasma")
    assert len(table) == 20 and table.pixels.tolist() == [20] * 20
    for shot, frame in table.groupby("shot"):
        assert frame.t_peak_ms.tolist() == peaks
        assert frame.in_scored_window.tolist() == [False, True, True, False, False]
        if shot in (101, 102, 103):
            assert frame.in_plasma.tolist() == [False, True, True, shot > 101, False]
        else:
            assert frame.in_plasma.isna().all()
    # All pre-existing point values survive, including the regions outside windows.
    for before, after in zip(expected, table[table.shot == 101].to_dict("records")):
        for key, value in before.items():
            if key not in ("in_scored_window", "in_plasma"):
                assert after[key] == value
    meta = json.loads((poi_dir(paths) / "meta.json").read_text())
    assert meta["points_in_plasma"] == 8
    assert meta["points_outside_plasma"] == 7
    assert meta["points_unknown_plasma"] == 5
    population = paths.catalog / "population.csv"
    assert meta["plasma_window_sources"] == {
        str(population): {"sha256": sha256_of(population), "shots": [101, 102]},
        str(log): {"sha256": sha256_of(log), "shots": [103]},
    }
    assert meta["G3"] is g3
