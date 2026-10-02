"""RWM review: magnetic amplitudes, beta_N/inductance and curated onset markers."""

from __future__ import annotations

import numpy as np

from ...config import Paths
from .. import equilibrium, rwm
from ..verify import Panel
from ._shared import optional


def onset_panels(shot, *, paths, t_range=None):
    points = rwm.onsets(shot, paths)
    if not points:
        return []
    times = np.array([p["t_ms"] for p in points])
    # Every point has an exact-time peak bracketed by zero-valued support
    # samples. The trace draws an annotation, not a label with that duration.
    x = np.unique(np.r_[np.arange(min(0, times.min() - 1), times.max() + 501, 1.0),
                        times - 1, times, times + 1])
    y = np.isin(x, times).astype(float)[None]
    if t_range is not None:
        keep = (x >= t_range[0]) & (x <= t_range[1])
        x, y = x[keep], y[:, keep]
    return [Panel("curated RWM onsets (times only)", x=x, y=y,
                  legend=["database onset"], ylabel="onset")]


def panels(shot, *, t_range=None, paths=None):
    paths = Paths.from_env() if paths is None else paths
    kwargs = {"paths": paths, "t_range": t_range}
    built = []
    for name, title, unit in (
        ("n1rms", "n=1 magnetic RMS (not RWM-specific)", "G"),
        ("n2rms", "n=2 magnetic RMS (not RWM-specific)", "G"),
        ("betan", "beta_N", "β_N"), ("li", "internal inductance", "l_i"),
        ("ip", "ip", "A"),
    ):
        built += optional(name, shot, lambda name=name, title=title, unit=unit:
                          equilibrium.line_panel(shot, name, title, ylabel=unit,
                                                 **kwargs))
    return built + onset_panels(shot, **kwargs)
