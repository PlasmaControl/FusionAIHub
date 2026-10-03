"""Local detachment context and validity-gated producer indicators.

Raw TPLANG sweeps and medians of uncalibrated bolometer voltages are omitted:
neither measures Isat nor radiated power. This view never fetches or votes.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from ...config import Paths
from ..verify import Panel
from ._shared import plasma_window

log = logging.getLogger(__name__)
# Channel order from modalities.yaml. FS calibration is absent from the corpus.
TRACES = (
    (
        "filterscopes",
        "D-alpha filterscopes",
        [f"FS{i:02d}" for i in range(1, 9)],
        "a.u.",
        1.0,
    ),
    ("co2", "CO2 R0 line-averaged density", ["R0 DENUF"], "cm^-3", 1.0),
    (
        "gas_flow",
        "Gas flow",
        [
            "GASA",
            "GASB",
            "GASC",
            "GASD",
            "GASE",
            "LOB1",
            "LOB2",
            "PFX1",
            "PFX2",
            "PFX3",
            "UOB",
        ],
        "Torr L/s",
        1.0,
    ),
)
CSV_INDICATORS = (
    ("afrac", "Afrac", "dimensionless"),
    ("prad_div", "Divertor radiated power", "MW"),
    ("tangtv_front_height", "TangTV front height", "m"),
)
BIN_INDICATORS = (
    ("afrac", "Afrac", "dimensionless"),
    ("prad", "Lower-divertor radiation fraction", "dimensionless"),
    ("tangtv", "TangTV normalized front DZ", "dimensionless"),
)


def indicator_panels(shot, paths, t_range=None):
    """Read detach_bins' 50 ms NPZ/CSV, or the original CSV interchange.

    Producer values are Afrac, Prad_divL/P_in, and normalized DZ, NOT MW and
    metres. Preserve its validity gates, reasons and votes; never recompute them.
    Configure LABELER_DETACHMENT_INDICATORS, otherwise use a generic local root.
    """
    root = Path(
        os.environ.get(
            "LABELER_DETACHMENT_INDICATORS", str(paths.root / "indicators/detachment")
        )
    )
    path = root / f"{int(shot)}.npz"
    if not path.is_file():
        path = root / f"{int(shot)}.csv"
    if not path.is_file():
        return []
    try:
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as source:
                data = {name: source[name] for name in source.files}
        else:
            data = {
                name: series.to_numpy() for name, series in pd.read_csv(path).items()
            }
        bins = "start_ms" in data
        # detach_bins.py uses fixed 50 ms bins, including when plasma gaps exist.
        x = np.asarray(data["start_ms" if bins else "t_ms"], dtype=float)
        if bins:
            x = x + 25.0
        if x.ndim != 1 or not np.isfinite(x).all() or np.any(np.diff(x) <= 0):
            raise ValueError("indicator clock must be finite and increasing")
        keep = np.ones(len(x), dtype=bool)
        if t_range is not None:
            keep = (x >= t_range[0]) & (x <= t_range[1])
        built = []
        schema = "detach_bins_50ms" if bins else "indicator_csv"
        for name, title, unit in BIN_INDICATORS if bins else CSV_INDICATORS:
            value = f"{name}_value" if bins else name
            if value not in data or f"{name}_valid" not in data:
                continue
            y = np.asarray(data[value], dtype=float).copy()
            valid = np.asarray(data[f"{name}_valid"], dtype=float) == 1
            if y.shape != x.shape or valid.shape != x.shape:
                raise ValueError(f"{name}: value/valid shape disagrees with clock")
            y[~valid] = np.nan
            metadata = {"source": str(path), "schema": schema}
            for suffix in ("reason", "vote"):
                key = f"{name}_{suffix}"
                if key in data:
                    values = np.asarray(data[key])
                    if values.shape != x.shape:
                        raise ValueError(f"{key}: shape disagrees with clock")
                    metadata[suffix] = values[keep].tolist()
            if np.isfinite(y[keep]).any():
                built.append(
                    Panel(
                        title=title,
                        x=x[keep],
                        y=y[None, keep],
                        ylabel=unit,
                        legend=[f"{title} ({unit})"],
                        metadata=metadata,
                    )
                )
        return built
    except (ValueError, KeyError, OSError) as error:
        log.warning("shot %s: cannot read detachment indicators: %s", shot, error)
        return []


def _block_means(data, x, ids, start, stop, step):
    """Average contiguous native samples, including a final partial block.

    HDF5 reads are bounded to ~262k samples per channel. Float32 clocks use the
    span for spacing, avoiding quantized median-diff aliases at late shot times.
    """
    xs, ys = [], []
    chunk = max(1, 262144 // step) * step
    for lo in range(start, stop, chunk):
        hi = min(stop, lo + chunk)
        starts = np.arange(0, hi - lo, step)
        counts = np.minimum(step, hi - lo - starts)
        native = np.asarray(data[ids, lo:hi], dtype=np.float64)
        finite = np.isfinite(native)
        sums = np.add.reduceat(np.where(finite, native, 0), starts, axis=1)
        ns = np.add.reduceat(finite.astype(np.int32), starts, axis=1)
        mean = np.full(sums.shape, np.nan)
        np.divide(sums, ns, out=mean, where=ns > 0)
        xs.append(np.add.reduceat(x[lo:hi], starts) / counts)
        ys.append(mean)
    return np.concatenate(xs), np.concatenate(ys, axis=1)


def panels(shot, *, t_range=None, paths=None):
    """Block-mean context over the catalog plasma window, when available."""
    paths = Paths.from_env() if paths is None else paths
    if t_range is None:
        t_range = plasma_window(int(shot), paths)
    built = indicator_panels(shot, paths, t_range)
    path = paths.corpus_file(int(shot))
    if not path.is_file():
        return built
    with h5py.File(path, "r") as source:
        for name, title, names, unit, spacing in TRACES:
            if name not in source:
                continue
            group = source[name]
            if "xdata" not in group or "ydata" not in group:
                continue
            x = np.asarray(group["xdata"], dtype=float) * 1000
            data = group["ydata"]
            if (
                x.ndim != 1
                or len(x) < 2
                or data.ndim != 2
                or data.shape[1] != len(x)
                or not np.isfinite(x).all()
                or np.any(np.diff(x) <= 0)
            ):
                continue
            dt = float((x[-1] - x[0]) / (len(x) - 1))
            step = max(1, int(np.ceil(spacing / dt - 1e-6)))
            lo, hi = t_range if t_range is not None else (0.0, x[-1])
            start = int(np.searchsorted(x, lo))
            stop = int(np.searchsorted(x, hi, side="right"))
            ids = list(range(min(len(names), data.shape[0])))
            if stop <= start or not ids:
                continue
            bx, y = _block_means(data, x, ids, start, stop, step)
            live = np.isfinite(y).any(axis=1)
            if not live.any():
                continue
            legend = [
                f"{names[c]} ({unit})" for c, ok in zip(ids, live, strict=True) if ok
            ]
            built.append(
                Panel(
                    title=title,
                    x=bx,
                    y=y[live],
                    legend=legend,
                    ylabel=unit,
                    metadata={
                        "reduction": "block mean",
                        "samples_per_block": step,
                        "block_ms": step * dt,
                    },
                )
            )
    return built
