"""Local detachment context, with optional independently extracted indicators.

The corpus's uncalibrated Langmuir/bolometer channels are explicitly raw
context; their median is neither Afrac nor divertor radiated power. Nothing
in this view fetches diagnostics or infers a detachment state.
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

log = logging.getLogger(__name__)
# group, title, channels (None means a median across channels), spacing in ms.
TRACES = (
    ("filterscopes", "D-alpha filterscopes (corpus)", range(8), 1.0),
    ("co2", "CO2 R0 line-averaged density (corpus)", [0], 1.0),
    ("gas_flow", "Gas flow (corpus channels)", range(11), 1.0),
    ("langmuir", "Langmuir raw channel median (10 ms samples)", None, 10.0),
    ("bolo", "Bolometer raw channel median (1 ms samples)", None, 1.0),
)
INDICATORS = (
    ("afrac", "Afrac", ""),
    ("prad_div", "Divertor radiated power", "MW"),
    ("tangtv_front_height", "TangTV front height", "m"),
)


def indicator_panels(shot, paths, t_range=None):
    """Optional CSV: t_ms, indicator, indicator_valid (0/1), in named units.

    An explicit validity mask is required: invalid geometry/uncalibrated
    quantities must not appear as measured detachment evidence.
    """
    root = Path(
        os.environ.get(
            "LABELER_DETACHMENT_INDICATORS",
            str(paths.root / "round4/detach/indicators"),
        )
    )
    path = root / f"{int(shot)}.csv"
    if not path.is_file():
        return []
    try:
        data = pd.read_csv(path)
        x = pd.to_numeric(data.t_ms).to_numpy(dtype=float)
        if not np.isfinite(x).all() or np.any(np.diff(x) <= 0):
            raise ValueError("indicator clock must be finite and increasing")
        keep = np.ones(len(x), dtype=bool)
        if t_range is not None:
            keep = (x >= t_range[0]) & (x <= t_range[1])
        built = []
        for name, title, unit in INDICATORS:
            if name not in data or f"{name}_valid" not in data:
                continue
            y = pd.to_numeric(data[name]).to_numpy(dtype=float, copy=True)
            valid = pd.to_numeric(data[f"{name}_valid"]).to_numpy() == 1
            y[~valid] = np.nan
            if np.isfinite(y[keep]).any():
                built.append(
                    Panel(
                        title=title,
                        x=x[keep],
                        y=y[None, keep],
                        ylabel=unit,
                        metadata={"source": str(path)},
                    )
                )
        return built
    except (ValueError, KeyError, AttributeError, OSError) as error:
        log.warning("shot %s: cannot read detachment indicators: %s", shot, error)
        return []


def panels(shot, *, t_range=None, paths=None):
    """Read bounded, sampled scalar context directly from the read-only corpus."""
    paths = Paths.from_env() if paths is None else paths
    built = indicator_panels(shot, paths, t_range)
    path = paths.corpus_file(int(shot))
    if not path.is_file():
        return built
    with h5py.File(path, "r") as source:
        for name, title, channels, spacing in TRACES:
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
            dt = float(np.median(np.diff(x)))
            step = max(1, int(np.ceil(spacing / dt)))
            # Drop pre-shot baselines from the default discharge view.
            lo, hi = t_range if t_range is not None else (0.0, x[-1])
            start, stop = np.searchsorted(x, [lo, hi], side="left")
            selection = slice(int(start), int(stop), step)
            x = x[selection]
            if not len(x):
                continue
            ids = (
                list(range(data.shape[0]))
                if channels is None
                else [c for c in channels if c < data.shape[0]]
            )
            if not ids:
                continue
            y = np.asarray(data[ids, selection], dtype=np.float32)
            live = np.isfinite(y).any(axis=1)
            y = y[live]
            if not len(y):
                continue
            legend = [f"ch {c}" for c, ok in zip(ids, live, strict=True) if ok]
            if channels is None:
                y = np.ma.median(np.ma.masked_invalid(y), axis=0).filled(np.nan)
                y = y[None, :]
                legend = ["raw channel median"]
            built.append(
                Panel(title=title, x=x, y=y, legend=legend, ylabel="corpus units")
            )
    return built
