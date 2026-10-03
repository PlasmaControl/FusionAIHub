"""A shot's 0D traces on one 1 ms grid, and the labels on the same grid.

The confinement segmenter (``labeler.confinement.unet``) reads seven channels per
millisecond bin, all cut from signals the corpus, the raw cache or
``scripts/labeler/confinement_zerod_fetch.py`` hold for a shot:

====== ================================================================
da_mean D-alpha, bin mean:  log10(mean / reference + 0.01)
da_max  D-alpha, bin maximum, the same transform (ELM spikes)
da_std  D-alpha, bin standard deviation over the reference, at most 3
dens    CO2 V2 line-integrated density, / 1e14
betan   EFIT01 normalised beta, / 1.5
wmhd    EFIT01 stored energy, / 5e5 J
pinj    total injected neutral-beam power, / 3e6 W
====== ================================================================

The reference is the shot's own 95th percentile of D-alpha, the only per-shot scaling;
every other channel has a fixed scale, so absolute levels (a beta_N of 2) mean the same
on every shot. A bin with no value is NaN here and becomes zero, with a 1 on a mask
channel, at the input of the network (``to_input``). The EFIT traces are interpolated
across gaps of at most ``MAX_GAP_MS``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

CHANNELS = ("da_mean", "da_max", "da_std", "dens", "betan", "wmhd", "pinj")
#: Channels that can be missing on a shot, each with a mask channel at the network
#: input.
SPARSE = ("dens", "betan", "wmhd", "pinj")
N_INPUT = len(CHANNELS) + len(SPARSE)
CLASSES = ("L", "H", "QH", "WP")
BIN_MS = 1.0
MAX_GAP_MS = 100.0
CORPUS = Path("/scratch/gpfs/EKOLEMEN/foundation_model")
SCALES = {"dens": 1e14, "betan": 1.5, "wmhd": 5e5, "pinj": 3e6}
D_ALPHA_ROWS = 8


def bin_stats(
    t_ms: np.ndarray, y: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean, maximum and standard deviation of ``y`` in the 1 ms bins ``[k, k + 1)``.

    NaN where a bin holds no finite sample. ``t_ms`` must be ascending.
    """
    ok = np.isfinite(y) & np.isfinite(t_ms) & (t_ms >= 0) & (t_ms < n * BIN_MS)
    t_ms, y = t_ms[ok], y[ok].astype(np.float64)
    nan = np.full(n, np.nan, dtype=np.float32)
    if y.size == 0:
        return nan, nan.copy(), nan.copy()
    idx = np.floor(t_ms / BIN_MS).astype(np.int64)
    count = np.bincount(idx, minlength=n)[:n]
    total = np.bincount(idx, weights=y, minlength=n)[:n]
    square = np.bincount(idx, weights=y * y, minlength=n)[:n]
    starts = np.flatnonzero(np.r_[True, np.diff(idx) > 0])
    peak = np.full(n, np.nan)
    peak[idx[starts]] = np.maximum.reduceat(y, starts)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, total / count, np.nan)
        var = np.where(count > 0, square / count - mean**2, np.nan)
    std = np.sqrt(np.clip(var, 0.0, None))
    return mean.astype(np.float32), peak.astype(np.float32), std.astype(np.float32)


def interpolate_sparse(
    t_ms: np.ndarray, y: np.ndarray, n: int, max_gap_ms: float = MAX_GAP_MS
) -> np.ndarray:
    """A slowly sampled trace on the bin centres; NaN outside its span or across a long
    gap."""
    ok = np.isfinite(t_ms) & np.isfinite(y)
    t_ms, y = t_ms[ok], y[ok].astype(np.float64)
    out = np.full(n, np.nan, dtype=np.float32)
    if t_ms.size < 2:
        return out
    order = np.argsort(t_ms, kind="stable")
    t_ms, y = t_ms[order], y[order]
    centre = (np.arange(n) + 0.5) * BIN_MS
    value = np.interp(centre, t_ms, y, left=np.nan, right=np.nan)
    right = np.clip(np.searchsorted(t_ms, centre), 1, t_ms.size - 1)
    gap = t_ms[right] - t_ms[right - 1]
    value[gap > max_gap_ms] = np.nan
    out[:] = value
    return out


def _h5_rows(
    path: Path, group: str, rows: slice
) -> tuple[np.ndarray, np.ndarray] | None:
    """Time (ms) and the rows of a corpus-style group, or None if the group holds
    nothing."""
    import h5py

    if not path.exists():
        return None
    with h5py.File(path, "r") as f:
        if group not in f or f[group]["ydata"].shape[-1] <= 1:
            return None
        return (
            f[group]["xdata"][:].astype(np.float64) * 1000.0,
            f[group]["ydata"][rows].astype(np.float32),
        )


def read_d_alpha(
    shot: int, raw_dir: Path, corpus: Path = CORPUS
) -> tuple[np.ndarray, np.ndarray, str] | None:
    """Time (ms), one live D-alpha chord and where it came from, or None.

    The chord is the first of FS01-FS08 that has finite samples and varies; the corpus
    group is tried before the raw cache.
    """
    for source, base in (("corpus", corpus), ("raw", raw_dir)):
        got = _h5_rows(
            base / f"{shot}_processed.h5", "filterscopes", slice(0, D_ALPHA_ROWS)
        )
        if got is None:
            continue
        t_ms, rows = got
        for i, row in enumerate(rows):
            finite = np.isfinite(row)
            if (
                finite.mean() > 0.5
                and np.nanstd(row[finite][:: max(1, finite.sum() // 20000)]) > 0
            ):
                return t_ms, row, f"{source}:FS{i + 1:02d}"
    return None


def read_beams(
    shot: int, raw_dir: Path, zerod_dir: Path, corpus: Path = CORPUS
) -> tuple[np.ndarray, np.ndarray] | None:
    """Time (ms) and the eight beams' power ``(8, n)`` in watts (15L, 15R, ...), or None.

    The corpus is tried first, then the raw cache, then the 0D fetch's own pinj record.
    """
    for base in (corpus, raw_dir):
        got = _h5_rows(base / f"{shot}_processed.h5", "pinj", slice(None))
        if got is not None:
            return got
    path = zerod_dir / f"{shot}.npz"
    if path.exists():
        with np.load(path) as d:
            if "pinj" in d.files:
                grid = d["pinj"]
                t_ms = float(d["pinj_t0_ms"]) + (np.arange(grid.shape[1]) + 0.5)
                return t_ms, grid
    return None


def read_beam_power(shot: int, raw_dir: Path, zerod_dir: Path, corpus: Path = CORPUS):
    """Time (ms) and the eight beams' total power (W, NaN-safe sum), or None."""
    got = read_beams(shot, raw_dir, zerod_dir, corpus)
    return None if got is None else (got[0], np.nansum(got[1], axis=0))


def assemble(
    shot: int, raw_dir: Path, zerod_dir: Path, corpus: Path = CORPUS
) -> dict | None:
    """The ``(len(CHANNELS), n)`` grid of a shot, or None without D-alpha.

    ``n`` runs to the end of the D-alpha record. Also returns which D-alpha chord was
    read and the channels the shot lacks.
    """
    da = read_d_alpha(shot, raw_dir, corpus)
    if da is None:
        return None
    t_ms, y, source = da
    n = int(np.floor(t_ms[-1] / BIN_MS)) + 1
    mean, peak, std = bin_stats(t_ms, y, n)
    ref = (
        float(np.nanpercentile(y[np.isfinite(y) & (y > 0)], 95))
        if (y > 0).any()
        else 1.0
    )
    ref = max(ref, 1e-30)
    grid = np.full((len(CHANNELS), n), np.nan, dtype=np.float32)
    grid[0] = np.log10(np.maximum(mean, 0) / ref + 0.01)
    grid[1] = np.log10(np.maximum(peak, 0) / ref + 0.01)
    grid[2] = np.minimum(std / ref, 3.0)
    lacks: list[str] = []
    zpath = zerod_dir / f"{shot}.npz"
    if zpath.exists():
        with np.load(zpath) as d:
            if "dens" in d.files:
                dens = d["dens"]
                lo = round(float(d["dens_t0_ms"]) / BIN_MS)
                place = np.full(n, np.nan, dtype=np.float32)
                src = slice(max(0, -lo), min(len(dens), n - lo))
                if src.stop > src.start:
                    place[lo + src.start : lo + src.stop] = dens[src]
                grid[3] = place / SCALES["dens"]
            for name, row in (("betan", 4), ("wmhd", 5)):
                if name in d.files:
                    grid[row] = (
                        interpolate_sparse(d[f"{name}_t_ms"], d[name], n) / SCALES[name]
                    )
    power = read_beam_power(shot, raw_dir, zerod_dir, corpus)
    if power is not None:
        pmean, _, _ = bin_stats(power[0], power[1], n)
        grid[6] = pmean / SCALES["pinj"]
    for i, name in enumerate(CHANNELS):
        if name in SPARSE and not np.isfinite(grid[i]).any():
            lacks.append(name)
    return {"grid": grid, "da_source": source, "da_reference": ref, "lacks": lacks}


def to_input(grid: np.ndarray) -> np.ndarray:
    """Network input ``(N_INPUT, n)``: NaN to 0 on every channel, plus a mask per sparse
    one."""
    values = np.nan_to_num(grid, nan=0.0, posinf=0.0, neginf=0.0)
    values[0:2] = np.clip(values[0:2], -2.5, 2.5)
    values[3:] = np.clip(values[3:], -1.0, 8.0)
    masks = np.stack(
        [np.isfinite(grid[CHANNELS.index(c)]).astype(np.float32) for c in SPARSE]
    )
    return np.concatenate([values.astype(np.float32), masks])


def interval_labels(intervals: pd.DataFrame, n: int) -> np.ndarray:
    """Per-bin class index (L 0, H 1, QH 2, WP 3), -1 where no interval lies.

    ``intervals`` carries ``t_start``, ``t_end`` (ms) and ``label``; bin ``k`` is
    labelled when it lies wholly inside an interval.
    """
    out = np.full(n, -1, dtype=np.int8)
    for r in intervals.itertuples():
        lo = max(0, int(np.ceil(r.t_start / BIN_MS)))
        hi = min(n, int(np.floor(r.t_end / BIN_MS)))
        if hi > lo:
            out[lo:hi] = int(r.label)
    return out


def save_grid(path: Path, result: dict) -> None:
    tmp = path.with_suffix(".tmp.npz")
    np.savez(
        tmp,
        grid=result["grid"],
        meta=np.array(json.dumps({k: v for k, v in result.items() if k != "grid"})),
    )
    tmp.rename(path)


def load_grid(path: Path) -> tuple[np.ndarray, dict]:
    with np.load(path) as d:
        return d["grid"], json.loads(str(d["meta"]))
