"""SegNet v2: pseudo-v2 over 0-250 kHz, trained on v3's snapshot with v3's test
shots, scored once over the owner's whole windows with the 0-2 s table beside,
and its points of interest over the band its blob records."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch
from matplotlib.figure import Figure

from labeler.ae import seg, xpower
from labeler.ae.seg import evaluate, poi, pseudo, regions, train
from labeler.ae.seg.model import SegNet, SegNetConfig
from labeler.ae.xpower.data import BAND_KHZ, FULL_BAND_KHZ
from labeler.ae.xpower.train import load as load_ae
from labeler.config import Paths, sha256_of
from labeler.events.review.rows import Grid

from . import ae_tree, round3_tree
from .test_ae_seg_train import Small

POOL = [101, 102, 103, 104]
TEST = [111]
SPLITS = {**dict.fromkeys(POOL, "train"), **dict.fromkeys(TEST, "valid")}
DY = 500 / 512
LOW_BIN = 143  # TokEye's bin at ~70 kHz (page row 72): below 80 kHz, above 60


def _save(out, **kwargs) -> dict:
    train.save(
        out,
        SegNet(SegNetConfig(width=4)),
        threshold=0.5,
        split={},
        history=[],
        config=train.TrainConfig(width=4),
        inputs={},
        **kwargs,
    )
    return train.load(out / "model.pt")[1]


def test_a_blob_names_its_band_and_version_and_one_without_is_v1s(tmp_path):
    assert train.blob_band({}) == BAND_KHZ and train.blob_version({}) == "v1"
    v1 = _save(tmp_path / "v1")
    assert "band_khz" not in v1 and "version" not in v1, "v1's blob as it was"
    assert train.blob_band(v1) == BAND_KHZ and train.blob_version(v1) == "v1"
    v2 = _save(tmp_path / "v2", band_khz=FULL_BAND_KHZ, version="v2")
    assert train.blob_band(v2) == FULL_BAND_KHZ and train.blob_version(v2) == "v2"
    record = json.loads((tmp_path / "v2" / "training.json").read_text())
    assert record["band_khz"] == [0.0, 250.0] and record["version"] == "v2"
    assert record["pseudo"] == "pseudo-v2"
    old = json.loads((tmp_path / "v1" / "training.json").read_text())
    assert "version" not in old and "band_khz" not in old


def test_a_versions_masks_and_pictures_have_their_own_directories(tmp_path):
    paths = Paths(root=tmp_path)
    assert regions.pseudo_file(paths, 7) == seg.pseudo_dir(paths) / "7.npz"
    assert regions.pseudo_file(paths, 7, "v2") == seg.pseudo_dir(paths, "v2") / "7.npz"
    pictures = tmp_path / "gallery" / "alfven_eigenmode"
    assert poi.gallery_dir(paths) == pictures / "ae_seg-v1"
    assert poi.gallery_dir(paths, "v2") == pictures / "ae_seg-v2"


def test_the_pixels_and_points_cover_the_band_the_blob_names():
    prob = np.zeros((257, 100), dtype=np.float32)
    prob[150:153, 10:20] = 0.9  # 30 pixels at ~147 kHz
    prob[40:45, 60:80] = 0.9  # 100 pixels at ~41 kHz
    assert poi.ae_pixels(prob, 0.5, 0.0, DY).sum() == 30
    assert poi.ae_pixels(prob, 0.5, 0.0, DY, band=FULL_BAND_KHZ).sum() == 130
    grid = Grid(100.0, 2.0, 100)
    [old] = poi.points(7, prob, 0.5, grid, 0.0, DY)[0]
    assert old["method"] == "ae_seg-v1" and old["pixels"] == 30
    found, labelled = poi.points(
        7, prob, 0.5, grid, 0.0, DY, band=FULL_BAND_KHZ, method="ae_seg-v2"
    )
    assert [p["pixels"] for p in found] == [100, 30] and labelled.max() == 2
    assert {p["method"] for p in found} == {"ae_seg-v2"}
    assert found[0]["f_peak_khz"] == round(40 * DY, 2)


def test_a_point_is_in_its_shots_scored_window_or_blank(tmp_path):
    peaks = (-1.0, 0.0, 2500.0, 3000.0)
    rows = [
        {**dict.fromkeys(poi.POI_COLUMNS, 0), "shot": shot, "region": k, "t_peak_ms": t}
        for shot in (101, 102)
        for k, t in enumerate(peaks, start=1)
    ]
    file = tmp_path / "poi.csv"
    table = poi.write_points(file, [101, 102], rows, scored={101: (0.0, 3000.0)})
    assert table[table.shot == 101].in_scored_window.tolist() == [
        False,
        True,
        True,
        False,
    ]
    assert table[table.shot == 102].in_scored_window.tolist() == [""] * 4
    assert pd.read_csv(file).query("shot == 102").in_scored_window.isna().all()
    v1 = poi.write_points(tmp_path / "v1.csv", [101], rows[:4])
    assert v1.in_scored_window.tolist() == [False, True, False, False], "0-2 s"


@pytest.fixture
def figures(monkeypatch):
    drawn = []

    def savefig(fig, path, **kwargs):
        drawn.append(fig)
        path.write_bytes(b"picture")

    monkeypatch.setattr(Figure, "savefig", savefig)
    return drawn


def test_a_whole_window_poi_picture_has_no_scored_line(tmp_path, figures):
    labelled = np.zeros((257, 32), dtype=int)
    for until in (None, 2000.0):
        poi.draw(
            tmp_path / f"{until}.jpg",
            title="t",
            grid=Grid(0.0, 100.0, 32),
            row=np.zeros_like(labelled),
            y0=0.0,
            dy=DY,
            labelled=labelled,
            found=[],
            scored_until_ms=until,
        )
    scored = [
        [x for ax in fig.axes for x in ax.lines if x.get_label().startswith("scored")]
        for fig in figures
    ]
    assert scored[0] == [] and len(scored[1]) == 1
    assert scored[1][0].get_label() == "scored: 0-2 s"


def _low_line(paths, shot: int) -> None:
    """Light the shot's whole-shot TokEye mask at ~70 kHz over `AE_MS` too: an AE
    line below 80 kHz that pseudo-v2 keeps, as it lies above 60 kHz (not steady)
    and inside the owner's present frames (no absent run)."""
    [file] = xpower.tokeye_masks(paths, "v3").glob(f"{shot}_*_clean.npz")
    with np.load(file) as npz:
        saved = dict(npz)
    t = saved["t_ms"]
    lit = np.unpackbits(saved["mask_clean"], axis=-1, count=len(t)).astype(bool)
    lit[:, LOW_BIN, (t >= ae_tree.AE_MS[0]) & (t < ae_tree.AE_MS[1])] = True
    saved["mask_clean"] = np.packbits(lit, axis=-1)
    np.savez(file, **saved)


def _v2_tree(tmp_path, monkeypatch):
    """Task 1.6's tree: pseudo-v2 written from v3's snapshot, one val shot, and
    test shot 111 with an AE line below 80 kHz (`_low_line`)."""
    paths = round3_tree.build(tmp_path, SPLITS, tokeye_dt=0.256)
    _low_line(paths, 111)
    ae_tree.env(monkeypatch, paths)
    digest = ae_tree.snapshot(paths, monkeypatch, "v3")
    monkeypatch.setattr(pseudo, "N_VAL", 1)
    assert pseudo.main(["--version", "v2"]) == 0
    return paths, digest


def test_segnet_v2_refuses_test_shots_that_are_not_v3s(tmp_path, monkeypatch, capsys):
    paths, _ = _v2_tree(tmp_path, monkeypatch)
    models = ae_tree.chosen(paths, dict.fromkeys(POOL + TEST, "train"), "v3")
    with pytest.raises(SystemExit) as error:
        train.main(["--version", "v2"])
    assert error.value.code != 0
    stderr = capsys.readouterr().err
    assert str(models / "band80-mhd3" / "split.csv") in stderr
    assert "test shots" in stderr and "Traceback" not in stderr
    assert not seg.model_dir(paths, "v2").exists()


def _two_regions(model, values):
    """P(AE) at ~40 kHz and ~147 kHz over columns 200-219: one region below
    80 kHz and one above. It is 0.99, above every value of `train.THRESHOLDS`, so
    the regions are drawn at whatever threshold training picked."""
    prob = np.zeros(values.shape[1:], dtype=np.float32)
    prob[40:43, 200:220] = 0.99
    prob[150:153, 200:220] = 0.99
    return prob


def _and_a_late_one(model, values):
    """`_two_regions` and a third region after 3.1 s, past the windows (0-3000 ms)."""
    prob = _two_regions(model, values)
    prob[150:153, -25:-5] = 0.99
    return prob


class _AllAE(torch.nn.Module):
    """A stand-in SegNet calling every pixel AE."""

    def forward(self, x):
        return torch.full_like(x[:, :1], 10.0)


def test_segnet_v2_trains_scores_and_draws_over_the_whole_band_and_window(
    tmp_path, monkeypatch, capsys
):
    paths, digest = _v2_tree(tmp_path, monkeypatch)
    split = pseudo.v2_split(POOL + TEST, xpower.tokeye_masks(paths, "v3"))
    models = ae_tree.chosen(paths, {**dict.fromkeys(POOL, "train"), 111: "test"}, "v3")
    monkeypatch.setattr(train, "TrainConfig", Small)
    assert train.main(["--version", "v2", "--epochs", "1"]) == 0
    out = seg.model_dir(paths, "v2")
    _, blob = train.load(out / "model.pt")
    assert train.blob_band(blob) == FULL_BAND_KHZ and train.blob_version(blob) == "v2"
    inputs = blob["inputs"]
    ae_file = models / "band80-mhd3" / "model.pt"
    assert inputs["labels_sha256"] == digest
    assert inputs["ae_model_sha256"] == sha256_of(ae_file)
    assert inputs["ae_split_sha256"] == sha256_of(ae_file.parent / "split.csv")
    snapshot = xpower.snapshot_file(paths, "v3").read_bytes()
    assert (out / "review" / "labels.csv").read_bytes() == snapshot
    assert train.read_split(out / "split.csv") == split
    manifest = json.loads((out / "pseudo_masks.json").read_text())
    assert set(manifest) == {"index.csv", *(f"{s}.npz" for s in POOL + TEST)}
    for name, digest_of in manifest.items():
        assert digest_of == sha256_of(seg.pseudo_dir(paths, "v2") / name)

    # v1's evaluation and POI refuse a v2 model before scoring or drawing.
    for command in (evaluate.main, poi.main):
        with pytest.raises(SystemExit):
            command(["--models", str(out)])
        assert str(out / "model.pt") in capsys.readouterr().err
    assert not (out / "evaluation.json").exists()

    assert evaluate.main(["--version", "v2"]) == 0
    record = json.loads((out / "evaluation.json").read_text())
    counts = record["counts"]
    assert counts["shots"] == 1 and counts["frames"] == 300
    assert counts["present_frames"] == 100 and counts["mhd_absent_frames"] == 30
    early = record["window_0_2s"]["counts"]
    assert (early["frames"], early["present_frames"]) == (200, 60)
    assert early["mhd_absent_frames"] == 30
    # 0-2 s scores a part of the whole window's pixels: the late AE span is not in it.
    assert early["scored_pixels"] < counts["scored_pixels"]
    assert 0 < early["ae_pixels"] < counts["ae_pixels"]
    assert set(record["window_0_2s"]["methods"]) == {"ae_seg", "recipe", "tokeye"}
    assert record["methods"]["tokeye"]["frame_recall"]["value"] == 1.0
    meta = record["meta"]
    assert meta["version"] == "v2" and meta["band_khz"] == [0.0, 250.0]
    assert meta["frames_window"] == "whole" and meta["tier"] == "suggestions"
    report = (out / "evaluation.md").read_text()
    assert "frames of the owner's whole windows" in report and "## 0-2 s" in report
    assert "still pseudo-v2's over 0-250 kHz" in report

    # Every method is cut to the band the blob records: shot 111's line at
    # ~70 kHz is AE in pseudo-v2, and only an 80-250 kHz band misses it.
    ex = train.load_example(paths, 111, {}, margin=None, version="v2")
    khz = ex.y0_khz + np.arange(ex.y.shape[0]) * ex.dy_khz
    below = int(((ex.y == 1) & (khz < 80)[:, None]).sum())
    assert below > 0
    label = xpower.read_snapshot(paths, "v3")[1][111]
    ae = load_ae(ae_file)
    for band, missed in ((FULL_BAND_KHZ, 0), (BAND_KHZ, below)):
        pixels = evaluate.shot_scores(
            paths,
            111,
            label=label,
            decisions={},
            seg=(_AllAE(), {**blob, "band_khz": band}),
            ae=ae,
            version="v2",
        )["pixels"]
        assert pixels["ae_seg"][2] == pixels["tokeye"][2] == missed, band

    monkeypatch.setattr(poi, "predict", _two_regions)
    assert poi.main(["--version", "v2", "--workers", "1", "--no-pictures"]) == 0
    table = pd.read_csv(seg.poi_dir(paths, "v2") / "poi.csv")
    assert table.shot.unique().tolist() == POOL + TEST and len(table) == 10
    assert (table.method == "ae_seg-v2").all() and table.in_scored_window.all()
    peaks = sorted(table.f_peak_khz.unique())
    assert peaks == [round(40 * DY, 2), round(150 * DY, 2)], "below 80 kHz too"
    meta_file = seg.poi_dir(paths, "v2") / "meta.json"
    drawn = json.loads(meta_file.read_text())
    assert drawn["version"] == "v2" and drawn["band_khz"] == [0.0, 250.0]
    assert drawn["shots"] == POOL + TEST
    assert drawn["points_in_scored_window"] == 10
    assert drawn["points_outside_scored_window"] == 0
    assert drawn["points_unknown_scored_window"] == 0
    # A point after 111's window (0-3000 ms) is counted outside it.
    monkeypatch.setattr(poi, "predict", _and_a_late_one)
    args = ["--version", "v2", "--workers", "1", "--no-pictures", "--shots", "111"]
    assert poi.main(args) == 0
    drawn = json.loads(meta_file.read_text())
    assert drawn["points_in_scored_window"] == 2
    assert drawn["points_outside_scored_window"] == 1
    assert not seg.poi_dir(paths).exists(), "v1's points untouched"

    # A region decision counts only on the bytes it was made on.
    sha = regions.file_sha256(regions.pseudo_file(paths, 101, "v2"))
    current = {101: {"rejected": [1], "pseudo_sha256": sha}}
    stale = {101: {"rejected": [1], "pseudo_sha256": "0" * 64}}
    kept = train.load_example(paths, 101, stale, version="v2")
    cut = train.load_example(paths, 101, current, version="v2")
    assert (kept.y == 1).sum() > (cut.y == 1).sum() > 0
