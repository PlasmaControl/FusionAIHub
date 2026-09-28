"""A synthetic AE tree for the `labeler.ae.xpower` and `labeler.ae.seg` tests.

Each shot gets what the real ones have: a review store of three cross-power
rows (`spectrograms/alfven_eigenmode/<shot>.h5`), TokEye's cleaned mask
(`ae/masks/<shot>_<split>_clean.npz`, coarser in time than the real 0.256 ms),
the earlier detector's input (`ae/dataset/<shot>_<split>.npz`), and the owner's
and the source table's labels. Every shot has an AE line at ~146 kHz over
`AE_MS` and an MHD mode (a 19.5 kHz line and its harmonics) over `MHD_MS`, and
the owner calls the AE present and everything else absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from labeler.ae.xpower import data
from labeler.config import Paths
from labeler.events.review import rows

AE_MS = (300, 900)
MHD_MS = (1200, 1500)
DY = 500 / 512  # kHz per row bin
TOKEYE_DT = 2.56  # ms: ten real TokEye columns


def _store(path: Path, rng) -> None:
    grid = rows.Grid(t0_ms=-100.0, dt_ms=0.256, n=int(2300 / 0.256))
    t = grid.t0_ms + (np.arange(grid.n) + 0.5) * grid.dt_ms
    ae = (t >= AE_MS[0]) & (t < AE_MS[1])
    mhd = (t >= MHD_MS[0]) & (t < MHD_MS[1])
    built = []
    for name in data.CROSS_ROWS:
        values = rng.integers(0, 40, (257, grid.n), dtype=np.uint8)
        values[150, ae] = 220  # 146 kHz
        for k in (1, 5, 6):  # 19.5 kHz and its 98 and 117 kHz harmonics
            values[20 * k, mhd] = 200
        built.append(
            rows.ImageRow(
                name,
                name,
                values,
                y0=0.0,
                dy=DY,
                y_units="kHz",
                z_lo=-3.0,
                z_hi=27.0,
                z_units="dB",
                band=(80.0, 250.0),
            )
        )
    rows.write(path, grid, built, event="alfven_eigenmode", shot=int(path.stem))


def _tokeye(root: Path, shot: int, split: str, dt: float = TOKEYE_DT) -> None:
    t = -0.768 + dt * np.arange(int(2002.8 // dt) + 1)  # to 2002 ms
    clean = np.zeros((4, 512, len(t)), dtype=bool)
    ae = (t >= AE_MS[0]) & (t < AE_MS[1])
    mhd = (t >= MHD_MS[0]) & (t < MHD_MS[1])
    clean[:, 299, ae] = True  # (299 + 1) * 0.488 = 146 kHz
    clean[:, 39, mhd] = True  # 19.5 kHz
    clean[:, 199, mhd] = True  # 98 kHz, the fifth harmonic
    masks = root / "ae" / "masks"
    masks.mkdir(parents=True, exist_ok=True)
    np.savez(
        masks / f"{shot}_{split}_clean.npz",
        t_ms=t,
        mask_clean=np.packbits(clean, axis=-1),
        ann=ae.astype(np.uint8),
    )
    dataset = root / "ae" / "dataset"
    dataset.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dataset / f"{shot}_{split}.npz",
        spec=np.zeros((4, 348, len(t)), dtype=np.float16),
    )


def _table(path: Path, shots, spans) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["shot,category,t_start,t_end,confidence"]
    for shot in shots:
        lines += [f"{shot},{c},{a},{b}," for a, b, c in spans]
    path.write_text("\n".join(lines) + "\n")


def build(
    tmp_path: Path,
    splits: dict[int, str],
    seed: int = 0,
    reviewed=None,
    tokeye_dt: float = TOKEYE_DT,
) -> Paths:
    """The tree under `tmp_path`, `splits` giving each shot's SELDNet split;
    the owner has saved `reviewed` (default: every shot). `tokeye_dt=0.256` gives
    TokEye the real column spacing."""
    root, events = tmp_path / "root", tmp_path / "events"
    rng = np.random.default_rng(seed)
    store_dir = root / "spectrograms" / "alfven_eigenmode"
    store_dir.mkdir(parents=True, exist_ok=True)
    for shot, split in sorted(splits.items()):
        _store(store_dir / f"{shot}.h5", rng)
        _tokeye(root, shot, split, tokeye_dt)
    spans = [(0, AE_MS[0], 0), (*AE_MS, 1), (AE_MS[1], 2000, 0)]
    event = events / "alfven_eigenmode"
    saved = sorted(splits) if reviewed is None else sorted(reviewed)
    _table(event / "review" / "labels.csv", saved, spans)
    _table(
        event / "format" / "alfven_eigenmode_format_2026_v1.csv", sorted(splits), spans
    )
    return Paths(root=root, label_tables=events, corpus=tmp_path / "corpus")


def corpus(tmp_path: Path, shots, *, seconds=(-0.1, 0.7), tone_s=(0.2, 0.4)) -> Path:
    """Corpus files with a `co2` group: four 500 kHz chords of noise, and a
    coherent 150 kHz mode on all four over `tone_s`."""
    import h5py

    out = tmp_path / "corpus"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    t = np.arange(seconds[0], seconds[1], 1 / 500_000)
    tone = np.sin(2 * np.pi * 150e3 * t) * ((t >= tone_s[0]) & (t < tone_s[1]))
    for shot in shots:
        y = (rng.normal(0, 1, (4, t.size)) + 4 * tone).astype(np.float32)
        with h5py.File(out / f"{int(shot)}_processed.h5", "w") as f:
            f.create_dataset("co2/xdata", data=t)
            f.create_dataset("co2/ydata", data=y)
    return out


def chosen(paths: Paths, split: dict[int, str], version: str = "v1") -> Path:
    """An untrained `band80-mhd3` model of `version` saved as the chosen one,
    threshold 0.5; its models directory."""
    import json

    import torch

    from labeler.ae.xpower import train
    from labeler.ae.xpower.model import FrameCNN

    torch.manual_seed(0)
    models = paths.root / "models" / "ae_xpower" / version
    train.save(
        models / "band80-mhd3",
        FrameCNN(),
        threshold=0.5,
        split=split,
        history=[{"epoch": 1, "val_f1": 0.5, "kept": True}],
        config=train.TrainConfig(),
        band_khz=(80.0, 250.0),
        labels_file=paths.label_tables / "alfven_eigenmode/review/labels.csv",
        candidate="band80-mhd3",
        version=version,
    )
    (models / "chosen.json").write_text(json.dumps({"candidate": "band80-mhd3"}))
    return models


def env(monkeypatch, paths: Paths) -> None:
    """Point `Paths.from_env` at the tree."""
    monkeypatch.setenv("LABELER_ROOT", str(paths.root))
    monkeypatch.setenv("LABELER_LABEL_TABLES", str(paths.label_tables))
    monkeypatch.setenv("LABELER_CORPUS", str(paths.corpus))


def snapshot(paths: Paths, monkeypatch, version: str = "v2") -> str:
    """Freeze the tree's saved labels as `version`'s snapshot, as the controller
    froze the owner's (`models/ae_xpower/<version>/review/labels.csv`), and make
    its sha256 the one the version expects; the sha256."""
    import hashlib

    from labeler.ae import xpower
    from labeler.events.review import labels

    data = labels.labels_path(xpower.event_dir(paths)).read_bytes()
    file = xpower.snapshot_file(paths, version)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    monkeypatch.setitem(xpower.LABEL_SNAPSHOTS, version, digest)
    return digest
