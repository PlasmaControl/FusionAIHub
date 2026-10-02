"""qmin, against the rule's own class thresholds."""

from __future__ import annotations

import numpy as np

from ...config import Paths
from ...features.store import FeatureArray
from ..verify import NoDataError, Panel
from .. import equilibrium
from ._shared import optional

#: The rule's class boundaries, drawn as dashed lines. A `bands` entry 0.01
#: tall at 8% opacity, on an axis spanning ~0.8-3, is invisible; a threshold
#: is a line, so `hlines` draws one.
THRESHOLDS = list(equilibrium.QMIN_THRESHOLDS)


def panels(shot, *, t_range=None, paths=None):
    paths = Paths.from_env() if paths is None else paths
    def read(name):
        try:
            return equilibrium.signal(int(shot), name, paths, fetch=True)
        except ValueError as error:
            raise NoDataError(f"shot {shot}: malformed {name}: {error}") from error

    def window(x, y):
        """Seconds on disk, milliseconds on screen, clipped to the window."""
        x, y = equilibrium.plot_arrays(FeatureArray(np.asarray(x), y))
        if t_range is None:
            return x, y
        keep = (x >= t_range[0]) & (x <= t_range[1])
        return x[keep], y[:, keep]

    def scalar(name, title, ylabel, hlines=()):
        array = read(name)
        x, y = window(array.x, array.y)
        return [Panel(title, x=x, y=y, ylabel=ylabel, hlines=hlines)]

    def profile():
        array = read("qpsi")
        x, y = window(array.x, array.y)
        # EFIT profiles use normalized psi, not rho.
        return [Panel("q profile", kind="heatmap", x=x,
                      y=np.linspace(0, 1, y.shape[0]), z=y, ylabel="psi_n")]

    return (
        optional("qmin", shot, lambda: scalar("qmin", "qmin (EFIT01 aeqdsk)",
                                             "q", THRESHOLDS))
        + optional("q profile", shot, profile)
        + optional("ip", shot, lambda: scalar("ip", "ip", "A"))
    )
