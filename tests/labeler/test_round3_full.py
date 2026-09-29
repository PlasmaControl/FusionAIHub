"""Whole-shot TokEye's input, frame grid, annotation and tiles, and the check
against v1's 0-2 s masks."""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest
import torch
from scipy import signal

from labeler.ae import full
from labeler.ae.labels import HOP, N_BINS, N_FFT, bin_freqs_khz
from labeler.ae.xpower import evaluate
from labeler.config import Paths
from labeler.events import raw
from labeler.events.review import alfven


def _cache(paths, shot, t0_ms, t1_ms, rate_hz):
    """A raw-cache CO2 record of four chords."""
    n = round((t1_ms - t0_ms) / 1000 * rate_hz) + 1
    t = t0_ms + np.arange(n) * 1000 / rate_hz
    y = np.stack([np.sin(2 * np.pi * (10 + k) * t) for k in range(4)])
    path = paths.raw_cache / f"{shot}_processed.h5"
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        g = f.create_group("co2")
        g.create_dataset("xdata", data=(t / 1000).astype("float32"))
        g.create_dataset("ydata", data=y.astype("float32"))


@pytest.fixture
def paths(tmp_path):
    return Paths(
        root=tmp_path / "root", corpus=tmp_path / "corpus", raw_cache=tmp_path / "cache"
    )


def _last(x, t0, fs):
    return t0 + (x.shape[1] - 1) / fs * 1000


def test_the_input_runs_from_0_to_the_record_end_at_500_khz(paths):
    _cache(paths, 170720, -20.0, 50.0, 1_000_000)
    x, t0, fs = full.input_signals(170720, paths=paths)
    assert fs == pytest.approx(500_000.0, rel=1e-6)
    assert x.shape[0] == 4 and x.dtype == np.float64
    assert 0.0 <= t0 < 0.002 + 1e-6
    assert 50.0 - 0.002 - 1e-6 <= _last(x, t0, fs) <= 50.0 + 1e-6


def test_the_input_stops_at_crop_end(paths, monkeypatch):
    monkeypatch.setattr(full, "CROP_END_MS", 30.0)
    _cache(paths, 170721, -20.0, 50.0, 1_000_000)
    x, t0, fs = full.input_signals(170721, paths=paths)
    assert 30.0 - 0.002 - 1e-6 <= _last(x, t0, fs) <= 30.0 + 1e-6


def test_the_input_is_the_review_rows_resampling_of_the_margined_record(paths):
    _cache(paths, 170722, -20.0, 50.0, 1_666_666.7)
    x, t0, fs = full.input_signals(170722, paths=paths)
    a = raw.raw_signal(170722, "co2", paths=paths)
    keep = (a.x >= -full.MARGIN_MS) & (a.x <= min(a.x[-1], 6000.0) + full.MARGIN_MS)
    want, want_fs = alfven.resample(a.x[keep], a.y[:, keep])
    t = a.x[keep][0] + np.arange(want.shape[1]) * 1000 / want_fs
    inside = (t >= 0) & (t <= min(a.x[-1], 6000.0))
    assert fs == want_fs and t0 == t[inside][0]
    assert np.array_equal(x, want[:, inside].astype(np.float64))


def test_frame_times_are_v1s_rule_on_any_record():
    window = signal.get_window("hann", N_FFT)
    fs_khz = 1_000_001 / 2000.0  # v1's frame_grid
    want = signal.ShortTimeFFT(window, hop=HOP, fs=fs_khz).t(1_000_001)
    got = full.frame_times(1_000_001, 0.0, fs_khz * 1000)
    assert len(got) == 7820 and np.allclose(got, want, rtol=0, atol=1e-9)
    later = full.frame_times(3_000_000, 0.004, 500_000.0)
    want = signal.ShortTimeFFT(window, hop=HOP, fs=500.0).t(3_000_000)
    assert np.allclose(later - 0.004, want, rtol=0, atol=1e-9)


def test_nearest_takes_the_earlier_on_a_tie():
    t_from = np.array([0.0, 1.0, 2.0])
    got = full.nearest(t_from, np.array([-5.0, 0.4, 0.5, 0.6, 1.5, 9.0]))
    assert got.tolist() == [0, 0, 0, 1, 1, 2]


def _v1_clean(path, t_ms, ann, lfm):
    frame_labels = np.zeros((5, len(t_ms)), bool)
    frame_labels[0], frame_labels[2] = lfm, ann
    np.savez_compressed(
        path, t_ms=t_ms.astype(np.float32), ann=ann, lfm=lfm, frame_labels=frame_labels
    )


def test_the_annotation_is_v1s_at_the_nearest_frame_and_false_after_it(tmp_path):
    t1 = (-0.768 + 0.256 * np.arange(20)).astype(np.float32).astype(np.float64)
    ann = np.zeros(20, bool)
    ann[5:9] = True
    lfm = np.zeros(20, bool)
    lfm[-1] = True
    _v1_clean(tmp_path / "1_train_clean.npz", t1, ann, lfm)
    t2 = np.arange(-0.6, 8.0, 0.25)
    got = full.annotation(tmp_path / "1_train_clean.npz", t2)
    near = np.abs(t2[:, None] - t1[None, :]).argmin(axis=1)
    inside = t2 <= t1[-1]
    assert np.array_equal(got["ann"], np.where(inside, ann[near], False))
    assert np.array_equal(got["lfm"], np.where(inside, lfm[near], False))
    assert got["frame_labels"].shape == (5, len(t2))
    assert np.array_equal(got["frame_labels"][2], got["ann"])
    assert got["ann_until_ms"] == pytest.approx(t1[-1])
    assert got["ann"][inside].any() and not got["ann"][~inside].any()


def test_one_tile_is_the_whole_record():
    assert full.tiles(100, 0) == [(0, 100, 0, 100)]


def test_tiles_keep_each_frame_once_and_add_margins():
    assert full.tiles(1000, 300, margin=50) == [
        (0, 350, 0, 300),
        (250, 650, 300, 600),
        (550, 950, 600, 900),
        (850, 1000, 900, 1000),
    ]


def test_tiled_probs_stitch_each_tiles_middle():
    norm = np.zeros((2, 3, 1000), np.float32)
    widths = []

    def fake_mask_probs(model, chunk, device, batch):
        widths.append(chunk.shape[-1])
        out = np.zeros((chunk.shape[0], 2, chunk.shape[1], chunk.shape[2]), np.uint8)
        out[:, 0] = np.arange(chunk.shape[2]) % 256
        return out

    cuts = full.tiles(1000, 300, margin=50)
    got = full.tiled_probs(fake_mask_probs, None, norm, "cpu", 2, cuts)
    assert widths == [350, 400, 400, 150]
    want = (
        np.concatenate(
            [
                np.arange(0, 300),
                np.arange(50, 350),
                np.arange(50, 350),
                np.arange(50, 150),
            ]
        )
        % 256
    )
    assert got.shape == (2, 2, 3, 1000) and np.array_equal(got[0, 0, 0], want)


def _clean(path, t_ms, lines, notch_bin=None):
    """A clean file: each (b0, b1, t0, t1) line lit on chords r0 and v1."""
    mask = np.zeros((4, N_BINS, len(t_ms)), bool)
    for b0, b1, t0, t1 in lines:
        cols = np.flatnonzero((t_ms >= t0) & (t_ms < t1))
        mask[:2, b0:b1, cols[0] : cols[-1] + 1] = True
    if notch_bin is not None:
        mask[0, notch_bin, :] = True  # one chord only: notched, never "lit"
    np.savez_compressed(
        path, t_ms=t_ms.astype(np.float32), mask_clean=np.packbits(mask, axis=-1)
    )


def test_the_check_compares_lit_pixels_before_2_s_in_both_bands(tmp_path):
    t1 = -0.768 + 0.256 * np.arange(7820)
    t2 = 0.1 + 0.256 * np.arange(12000)
    _clean(tmp_path / "v1.npz", t1, [(200, 210, 500, 900), (80, 85, 1000, 1500)])
    _clean(
        tmp_path / "full.npz",
        t2,
        [(200, 210, 500, 900), (200, 210, 2500, 3000)],
        notch_bin=300,
    )
    got = full.check_shot(tmp_path / "v1.npz", tmp_path / "full.npz")
    t1_32 = t1.astype(np.float32)
    t2_32 = t2.astype(np.float32).astype(np.float64)
    assert got["frames"] == int(
        ((t2_32 >= t1_32[0]) & (t2_32 <= t1_32[-1]) & (t2_32 < 2000.0)).sum()
    )
    assert got["iou"]["80-250"] == pytest.approx(1.0, abs=0.01)
    ae, line = 10 * 400, 5 * 500  # pixel-ms of the AE and of v1's 40 kHz line
    assert got["iou"]["0-250"] == pytest.approx(ae / (ae + line), abs=0.01)
    assert got["notch_full"]["r0"] == [round(float(bin_freqs_khz(500.0)[300]), 2)]
    assert got["notch_v1"] == {"r0": [], "v1": [], "v2": [], "v3": []}


def test_the_bar_needs_both_bands():
    good = {"iou": {"0-250": 0.8, "80-250": 0.9}}
    low = {"iou": {"0-250": 0.4, "80-250": 0.9}}
    empty = {"iou": {"0-250": float("nan"), "80-250": float("nan")}}
    assert full.check_bar([good] * 10 + [empty])["passed"]
    bar = full.check_bar([good] * 9 + [low] * 2)
    assert bar["bands"]["80-250"]["passed"] and not bar["bands"]["0-250"]["passed"]
    assert bar["bands"]["0-250"]["below_half"] == 2 and not bar["passed"]
    soft = full.check_bar([{"iou": {"0-250": 0.7, "80-250": 0.7}}] * 5)
    assert not soft["passed"] and soft["bands"]["0-250"]["median"] == pytest.approx(0.7)


class _AllFire(torch.nn.Module):
    def forward(self, x):
        logit = torch.full((x.shape[2],), 5.0)
        return torch.stack([logit, torch.zeros_like(logit)], -1)[None]


def test_the_check_cli_writes_json_and_md(tmp_path, monkeypatch):
    root = tmp_path / "root"
    masks, masks_full = root / "ae" / "masks", root / "ae" / "masks-full"
    dataset, dataset_full = root / "ae" / "dataset", root / "ae" / "dataset-full"
    for d in (masks, masks_full, dataset, dataset_full):
        d.mkdir(parents=True)
    t1 = -0.768 + 0.256 * np.arange(400)
    t2 = 0.1 + 0.256 * np.arange(800)
    for stem in ("101_train", "102_valid"):
        _clean(masks / f"{stem}_clean.npz", t1, [(200, 210, 20, 80)])
        (masks / f"{stem}_probs.npz").write_bytes(b"")
        np.savez(dataset / f"{stem}.npz", spec=np.zeros((4, 348, 400), np.float16))
    _clean(masks_full / "101_train_clean.npz", t2, [(200, 210, 20, 80)])
    np.savez(dataset_full / "101_train.npz", spec=np.zeros((4, 348, 800), np.float16))
    monkeypatch.setenv("LABELER_ROOT", str(root))
    monkeypatch.setattr(evaluate, "load_seldnet", lambda paths: _AllFire())
    assert full.main(["--check"]) == 0
    got = json.loads((masks_full / "check.json").read_text())
    assert [s["shot"] for s in got["shots"]] == [101]
    assert got["missing"] == ["102_valid"]
    assert got["shots"][0]["split"] == "train"
    assert got["shots"][0]["seldnet_agreement"] == pytest.approx(1.0)
    assert got["shots"][0]["iou"]["80-250"] == pytest.approx(1.0, abs=0.02)
    assert "passed" in got["bar"]
    assert "101" in (masks_full / "check.md").read_text()
