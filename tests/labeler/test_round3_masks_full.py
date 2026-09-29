"""The whole-shot TokEye runner, with TokEye's three functions faked."""

from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy import signal

from labeler.ae import full
from labeler.ae.labels import HOP, N_FFT

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "labeler"
V1_KEYS = {
    "mask_clean",
    "mask_raw",
    "active_frac_clean",
    "active_frac_raw",
    "n_pixels_raw",
    "n_pixels_no_transient",
    "n_pixels_clean",
    "protected",
    "window_centroids_khz",
    "ann",
    "lfm",
    "frame_labels",
    "t_ms",
    "freqs",
    "spec_mean",
    "spec_std",
}
DATASET_KEYS = {
    "spec",
    "annotated",
    "split",
    "spec_mean",
    "spec_std",
    "freq_khz_bins",
}


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "ae_masks_full", SCRIPTS / "ae_masks_full.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "ae_masks_full", module)
    spec.loader.exec_module(module)
    calls = {"stft": 0}

    def fake_stft(signals):
        calls["stft"] += 1
        sft = signal.ShortTimeFFT(signal.get_window("hann", N_FFT), hop=HOP, fs=1.0)
        raw = np.stack(
            [np.log1p(np.abs(sft.stft(s)))[1:].astype(np.float32) for s in signals]
        )
        mean, std = raw.mean(axis=(1, 2)), raw.std(axis=(1, 2))
        norm = (raw - mean[:, None, None]) / (std[:, None, None] + 1e-6)
        return raw, norm.astype(np.float32), mean, std

    def fake_probs(model, norm, device, batch):
        out = np.zeros((norm.shape[0], 2) + norm.shape[1:], np.uint8)
        out[:, 0] = np.where(norm > 3.0, 255, 0)
        return out

    monkeypatch.setattr(module.ae_dataset, "spectrogram_tokeye", fake_stft)
    monkeypatch.setattr(module.ae_dataset, "mask_probs", fake_probs)
    monkeypatch.setattr(module.ae_dataset, "load_model", lambda device: None)
    module.calls = calls
    return module


def _tree(tmp_path, monkeypatch):
    """Two AE shots: a raw-cache record with a 100 kHz burst at 100-300 ms, and
    v1's clean file with UCI's annotation over 150-250 ms."""
    root, cache = tmp_path / "root", tmp_path / "cache"
    masks = root / "ae" / "masks"
    masks.mkdir(parents=True)
    rate = 1_000_000
    t = -20.0 + np.arange(int(420 * rate / 1000) + 1) * 1000 / rate
    burst = np.where((t >= 100) & (t < 300), 5 * np.sin(2 * np.pi * 100 * t), 0.0)
    noise = np.random.default_rng(1).normal(scale=0.1, size=(4, len(t)))
    for shot, split in ((201, "train"), (202, "valid")):
        path = cache / f"{shot}_processed.h5"
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as f:
            g = f.create_group("co2")
            g.create_dataset("xdata", data=(t / 1000).astype("float32"))
            g.create_dataset("ydata", data=(noise + burst).astype("float32"))
        t1 = (-0.768 + 0.256 * np.arange(1570)).astype(np.float32)
        ann = (t1 >= 150) & (t1 < 250)
        labels = np.zeros((5, len(t1)), bool)
        labels[1] = ann
        np.savez_compressed(
            masks / f"{shot}_{split}_clean.npz",
            t_ms=t1,
            ann=ann,
            lfm=np.zeros(len(t1), bool),
            frame_labels=labels,
            mask_clean=np.packbits(np.zeros((4, 512, len(t1)), bool), axis=-1),
        )
        (masks / f"{shot}_{split}_probs.npz").write_bytes(b"")
    monkeypatch.setenv("LABELER_ROOT", str(root))
    monkeypatch.setenv("LABELER_RAW_CACHE", str(cache))
    monkeypatch.setenv("LABELER_CORPUS", str(tmp_path / "corpus"))
    monkeypatch.setenv("LABELER_NO_FETCH", "1")
    return root


def test_one_shot_writes_v1s_keys_and_the_extras(runner, tmp_path, monkeypatch):
    root = _tree(tmp_path, monkeypatch)
    args = ["--shots", "201", "--device", "cpu", "--workers", "0"]
    assert runner.main(args) == 0
    out = root / "ae" / "masks-full"
    with np.load(out / "201_train_clean.npz") as z:
        assert set(z.files) == V1_KEYS | set(full.EXTRA_KEYS)
        t_ms = z["t_ms"]
        clean = np.unpackbits(z["mask_clean"], axis=-1, count=len(t_ms)).astype(bool)
        assert clean.shape == (4, 512, len(t_ms))
        assert float(z["tile_ms"]) == 0.0
        assert float(z["fs_hz"]) == pytest.approx(500_000.0, rel=1e-6)
        fs, t0 = float(z["fs_hz"]), float(z["t0_input_ms"])
        window = signal.get_window("hann", N_FFT)
        first = signal.ShortTimeFFT(window, hop=HOP, fs=fs / 1000).t(10_000)[0]
        assert t_ms[0] == pytest.approx(t0 + first, abs=1e-3)
        assert np.allclose(np.diff(t_ms), HOP / fs * 1000, atol=1e-3)
        assert t_ms[-1] > 395
        ann, until = z["ann"], float(z["ann_until_ms"])
        assert until == pytest.approx(-0.768 + 0.256 * 1569, abs=1e-3)
    lit = clean.any(axis=0)
    rows = np.flatnonzero(lit.any(axis=1))
    freqs = (rows + 1) * 500.0 / N_FFT
    assert rows.size and freqs.min() > 90 and freqs.max() < 110
    cols = np.flatnonzero(lit.any(axis=0))
    assert t_ms[cols].min() > 95 and t_ms[cols].max() < 305
    assert ann[(t_ms > 160) & (t_ms < 240)].all() and not ann[t_ms > 260].any()
    with np.load(out / "201_train_probs.npz") as z:
        assert set(z.files) == {"prob", "ann"}
        assert z["prob"].shape == (4, 2, 512, len(t_ms)) and z["prob"].dtype == np.uint8
    with np.load(root / "ae" / "dataset-full" / "201_train.npz") as z:
        assert set(z.files) == DATASET_KEYS
        assert z["spec"].shape == (4, 348, len(t_ms)) and z["spec"].dtype == np.float16
        assert str(z["split"]) == "train"
    assert not (out / "202_valid_clean.npz").exists()


def test_existing_shots_are_skipped_unless_overwrite(runner, tmp_path, monkeypatch):
    _tree(tmp_path, monkeypatch)
    args = ["--shots", "201,202", "--device", "cpu", "--workers", "0"]
    runner.main(args)
    assert runner.calls["stft"] == 2
    runner.main(args)
    assert runner.calls["stft"] == 2
    runner.main(args + ["--overwrite"])
    assert runner.calls["stft"] == 4


def test_tiles_give_the_same_mask_as_one_pass_here(runner, tmp_path, monkeypatch):
    root = _tree(tmp_path, monkeypatch)
    out = root / "ae" / "masks-full"
    runner.main(["--shots", "201", "--device", "cpu", "--workers", "0"])
    with np.load(out / "201_train_clean.npz") as z:
        whole = z["mask_clean"].copy()
    runner.main(
        [
            "--shots",
            "201",
            "--device",
            "cpu",
            "--workers",
            "0",
            "--tile-ms",
            "50",
            "--overwrite",
        ]
    )
    with np.load(out / "201_train_clean.npz") as z:
        assert float(z["tile_ms"]) == 50.0
        assert np.array_equal(z["mask_clean"], whole)  # the fake is per pixel


def test_the_pilot_starts_with_the_owners_three(runner):
    stems = [f"{s}_train" for s in range(170000, 170030)] + [
        "170720_valid",
        "176041_train",
        "176053_valid",
    ]
    got = runner.pilot(sorted(stems))
    assert got[:3] == ["170720_valid", "176053_valid", "176041_train"]
    assert len(got) == runner.PILOT_SIZE and len(set(got)) == runner.PILOT_SIZE


def test_an_unknown_shot_is_refused(runner, tmp_path, monkeypatch):
    _tree(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="999"):
        runner.main(["--shots", "999", "--device", "cpu", "--workers", "0"])


def test_compressed_files_read_back_whole_at_the_fast_level(
    runner, tmp_path, monkeypatch
):
    levels = []
    compressobj = zipfile.zlib.compressobj

    def spy(level, *args, **kwargs):
        levels.append(level)
        return compressobj(level, *args, **kwargs)

    monkeypatch.setattr(zipfile.zlib, "compressobj", spy)
    arrays = {
        "prob": np.random.default_rng(0).integers(0, 256, (2, 2, 8, 50), np.uint8),
        "ann": np.arange(50) % 3 == 0,
        "ann_until_ms": np.float64(1234.5),
    }
    path = tmp_path / "201_train_probs.npz"
    runner._savez_compressed(path, **arrays)
    with np.load(path) as z:
        assert sorted(z.files) == sorted(arrays)
        for name, array in arrays.items():
            assert z[name].dtype == array.dtype
            assert np.array_equal(z[name], array)
    with zipfile.ZipFile(path) as archive:
        assert {i.compress_type for i in archive.infolist()} == {zipfile.ZIP_DEFLATED}
    assert levels == [runner.ZLIB_LEVEL] * len(arrays)
    assert list(tmp_path.iterdir()) == [path]


def test_a_refused_write_leaves_the_old_file_and_no_temporary(runner, tmp_path):
    path = tmp_path / "201_train_probs.npz"
    runner._savez_compressed(path, prob=np.zeros(3, np.uint8))
    before = path.read_bytes()
    with pytest.raises(ValueError):
        runner._savez_compressed(path, prob=np.array([{}, None], dtype=object))
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]
