"""RWM review: the n = 1 delta-Bp, magnetic amplitudes, beta_N/inductance and
curated onset markers."""

from __future__ import annotations

import numpy as np

from ...config import Paths
from ...rwm import bp
from .. import equilibrium, rwm
from ..verify import NoDataError, Panel
from ._shared import optional, with_caveat


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
    return [with_caveat(
        Panel("RWM onsets (curated)", x=x, y=y, legend=["database onset"], ylabel="onset"),
        "Curated database onset times only: a mark at each time, not a label with a duration",
    )]


def bp_panels(shot, *, paths, t_range=None):
    """The v5 sensor: the centred n = 1 delta-Bp with its screen threshold
    (`rwm.bp_series`), each probe and coil fetched live when not cached (the
    single probes only for a difference that is missing, as from 2021); the
    plasma's own threshold line. NoDataError without the four differences."""
    missing = set()
    for name in rwm.BP_INPUTS:
        needed_by = [d for d, pair in bp.SINGLES.items() if name in pair]
        if needed_by and needed_by[0] not in missing:
            continue
        try:
            equilibrium.signal(int(shot), name, paths, fetch=True)
        except (NoDataError, ValueError, KeyError, OSError):
            if name in bp.ANGLES_DEG:
                missing.add(name)
    _, _, _, threshold, info = rwm.bp_series(int(shot), paths)
    got = bp.amplitude(rwm.bp_signals(int(shot), paths))
    x, y = got["t_ms"], got["amp_g"][None]
    if t_range is not None:
        keep = (x >= t_range[0]) & (x <= t_range[1])
        x, y = x[keep], y[:, keep]
    removed = info["pickup"]["regressors"]
    return [with_caveat(
        Panel("n=1 delta-Bp (midplane probes)", x=x, y=y, ylabel="G",
              hlines=[threshold]),
        "The RWM's own sensor (Okabayashi 2005): MPID067U-157U less a 100 ms "
        "running median, n = 1 fit; a locked or slow tearing mode reaches it too",
        *([("Probe differences formed from the single probes MPI66M*F (no MPID "
            "points from 2021)")] if got["probes"] != "MPID" else []),
        f"Coil pickup removed ({', '.join(removed)})" if removed
        else "No coil record: pickup not removed",
        f"Dashed: the v5 screen threshold, {threshold:.1f} G",
    )]


def panels(shot, *, t_range=None, paths=None):
    paths = Paths.from_env() if paths is None else paths
    kwargs = {"paths": paths, "t_range": t_range}
    built = optional("bp_n1", shot, lambda: bp_panels(shot, **kwargs))
    for name, title, unit, caveat in (
        ("n1rms", "n=1 magnetic RMS", "G", "Not RWM-specific"),
        ("n2rms", "n=2 magnetic RMS", "G", "Not RWM-specific"),
        ("betan", "beta_N", "β_N", ""), ("li", "internal inductance", "l_i", ""),
        ("ip", "ip", "A", ""),
    ):
        built += optional(
            name, shot,
            lambda name=name, title=title, unit=unit, caveat=caveat: [
                with_caveat(panel, caveat)
                for panel in equilibrium.line_panel(shot, name, title, ylabel=unit,
                                                    **kwargs)
            ],
        )
    return built + onset_panels(shot, **kwargs)
