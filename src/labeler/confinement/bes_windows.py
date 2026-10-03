"""Windows of the curated confinement intervals, as features plus metadata.

``shot_windows`` cuts a shot's BES into the 1024-sample windows that lie wholly inside
one curated interval (``labeler.confinement.bes_features``), and ``window_beam_power``
reads a beam trace over each. ``load_windows`` reads the per-shot ``.npz`` files that
the fetch script (native 1 MHz) and the corpus script (500 kHz) write, and joins the
interval bounds, so a window knows its distance to the ends of its interval and to the
previous one, which is what the paper's transition exclusion and beam gating need.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import bes_features as bf

CLASSES = ("L", "H", "QH", "WP")
CHANNELS = 64
STRIDE = 2048
_MAIN = "/scratch/gpfs/nc1514/FusionAIHub"
INTERVALS = Path(
    os.environ.get(
        "CONFINEMENT_INTERVALS",
        f"{_MAIN}/runs/labeler/confinement/v1/merged_intervals.csv",
    )
)
COHORT = Path(
    os.environ.get(
        "CONFINEMENT_COHORT",
        "/scratch/gpfs/nc1514/FusionAIHub/data/events/catalog/cohort.csv",
    )
)


def curated_intervals(path: Path = INTERVALS, cohort: Path = COHORT) -> pd.DataFrame:
    """The L/H/QH/WP intervals of the shots outside the cohort's blind test split.

    ``interval`` numbers a shot's intervals from 0 in table order; ``label`` is the
    index in ``CLASSES``. The one H/L conflict interval is left out.
    """
    mi = pd.read_csv(path)
    blind = set(pd.read_csv(cohort).query("split == 'test'").shot)
    mi = mi[mi.regimes.isin(CLASSES) & ~mi.shot.isin(blind)].copy()
    mi["interval"] = mi.groupby("shot").cumcount()
    mi["label"] = mi.regimes.map({c: i for i, c in enumerate(CLASSES)})
    return mi.reset_index(drop=True)


def window_beam_power(
    t_ms: np.ndarray | None, y: np.ndarray | None, start_ms: np.ndarray, span_ms: float
) -> dict[str, np.ndarray]:
    """Minimum, mean and maximum of a beam trace over each window (9 points across it).

    NaN throughout for a shot with no record of the beam.
    """
    if y is None:
        nan = np.full(start_ms.shape, np.nan, dtype=np.float32)
        return {"min": nan, "mean": nan.copy(), "max": nan.copy()}
    at = start_ms[:, None] + np.linspace(0.0, span_ms, 9)[None, :]
    vals = np.interp(at, t_ms, y.astype(np.float64))
    return {
        "min": vals.min(axis=1).astype(np.float32),
        "mean": vals.mean(axis=1).astype(np.float32),
        "max": vals.max(axis=1).astype(np.float32),
    }


def shot_windows(
    intervals: pd.DataFrame,
    bes: np.ndarray,
    t0: float,
    dt: float,
    fs: float,
    stride: int = STRIDE,
    channels: slice | None = None,
) -> dict[str, np.ndarray]:
    """Features and metadata of the windows lying wholly in one shot's intervals.

    ``bes`` is ``(C, T)``; ``t0`` and ``dt`` are the first sample's time and the step in
    ms. A NaN channel (a dead or absent one) is zeroed so the window keeps the shot's
    other channels; a window whose every channel is NaN is dropped and counted.
    """
    sos = bf.band_sos(fs)
    n = bes.shape[1]
    feats, power, start, interval, label, dropped = [], [], [], [], [], 0
    for r in intervals.itertuples():
        lo = max(0, int(np.ceil((r.t_start - t0) / dt)))
        hi = min(n, int(np.floor((r.t_end - t0) / dt)))
        starts = bf.window_starts(lo, hi, stride)
        for c0 in range(0, starts.size, 256):
            chunk = starts[c0 : c0 + 256]
            w = bf.filtered_windows(bes, chunk, sos)
            nan_ch = ~np.isfinite(w).all(axis=2)
            if nan_ch.any():
                w = np.where(nan_ch[:, :, None], 0.0, w)
            ok = ~nan_ch.all(axis=1)
            dropped += int((~ok).sum())
            if not ok.any():
                continue
            w, chunk = w[ok], chunk[ok]
            feats.append(bf.spectral_features(w))
            power.append((w**2).mean(axis=2).astype(np.float32))
            start.append(t0 + chunk * dt)
            interval.append(np.full(chunk.size, r.interval, dtype=np.int32))
            label.append(np.full(chunk.size, r.label, dtype=np.int8))
    if not feats:
        return {}
    return {
        "feats": np.concatenate(feats),
        "power": np.concatenate(power),
        "start_ms": np.concatenate(start),
        "interval": np.concatenate(interval),
        "label": np.concatenate(label),
        "dropped": np.array(dropped),
    }


SMALL_KEYS = (
    "start_ms",
    "label",
    "interval",
    "p15L_min",
    "p15L_mean",
    "p15R_max",
    "p15R_mean",
)


def _shot_table(shot: int, d, dt_ms: float) -> pd.DataFrame:
    frame = pd.DataFrame({k: d[k] for k in SMALL_KEYS})
    frame["label"] = frame.label.astype(int)
    frame["interval"] = frame.interval.astype(int)
    frame.insert(0, "shot", shot)
    frame["dt_ms"] = dt_ms
    frame["center_ms"] = frame.start_ms + 0.5 * bf.WINDOW * dt_ms
    return frame


def _join_intervals(table: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
    """Add the interval bounds, status and the previous interval's label and gap."""
    key = pd.MultiIndex.from_arrays([table.shot, table.interval])
    ivs = intervals.set_index(["shot", "interval"]).loc[key]
    table["t_start"] = ivs.t_start.to_numpy()
    table["t_end"] = ivs.t_end.to_numpy()
    table["status"] = ivs.status.to_numpy()
    table["regime"] = ivs.regimes.to_numpy()
    ordered = intervals.sort_values(["shot", "t_start"]).copy()
    ordered["prev_label"] = ordered.groupby("shot").label.shift(1)
    ordered["gap_ms"] = ordered.t_start - ordered.groupby("shot").t_end.shift(1)
    prev = ordered.set_index(["shot", "interval"])[["prev_label", "gap_ms"]].loc[key]
    table["prev_label"] = prev.prev_label.to_numpy()
    table["gap_ms"] = prev.gap_ms.to_numpy()
    return table


def consolidate(
    directory: Path,
    out_dir: Path,
    intervals: pd.DataFrame,
    shots: list[int] | None = None,
) -> pd.DataFrame:
    """Join the per-shot ``.npz`` files of a directory into one memory-mappable dataset.

    Writes ``features.npy`` (n, 2, 64, 128) float16, ``power.npy`` (n, 64) float32,
    ``windows.csv`` (one row per window, see ``_join_intervals``) and ``channels.csv``
    (the per-shot channel record) to ``out_dir``, and returns the window table.
    """
    directory, out_dir = Path(directory), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = [
        f
        for f in sorted(directory.glob("*.npz"))
        if not f.name.endswith(".tmp.npz") and (shots is None or int(f.stem) in shots)
    ]
    tables, sizes, chan = [], [], []
    for f in files:
        with np.load(f) as d:
            rec = d["record"]
            tables.append(_shot_table(int(f.stem), d, float(rec[1])))
            sizes.append(len(tables[-1]))
            chan.append(
                {
                    "shot": int(f.stem),
                    "empty_channels": int(rec[3]),
                    **{f"std{c}": float(v) for c, v in enumerate(d["channel_std"])},
                }
            )
    n = int(sum(sizes))
    feats = np.lib.format.open_memmap(
        out_dir / "features.npy",
        mode="w+",
        dtype=np.float16,
        shape=(n, 2, CHANNELS, bf.FREQS),
    )
    power = np.zeros((n, CHANNELS), dtype=np.float32)
    at = 0
    for f, size in zip(files, sizes, strict=True):
        with np.load(f) as d:
            feats[at : at + size] = d["feats"]
            power[at : at + size] = d["power"]
        at += size
    feats.flush()
    np.save(out_dir / "power.npy", power)
    table = _join_intervals(pd.concat(tables, ignore_index=True), intervals)
    table.to_csv(out_dir / "windows.csv", index=False)
    pd.DataFrame(chan).to_csv(out_dir / "channels.csv", index=False)
    return table


def open_dataset(out_dir: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """The window table, memory-mapped features and channel power of a dataset."""
    out_dir = Path(out_dir)
    return (
        pd.read_csv(out_dir / "windows.csv"),
        np.load(out_dir / "features.npy", mmap_mode="r"),
        np.load(out_dir / "power.npy"),
    )
