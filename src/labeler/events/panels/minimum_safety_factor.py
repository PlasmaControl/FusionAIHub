"""qmin, against the rule's own class thresholds."""

from __future__ import annotations

import numpy as np

from ...config import Paths
from ...features.store import read_feature
from ..verify import Panel

#: The rule's class boundaries, drawn as dashed lines. A `bands` entry 0.01
#: tall at 8% opacity, on an axis spanning ~0.8-3, is invisible; a threshold
#: is a line, so `hlines` draws one.
THRESHOLDS = [0.95, 1.5, 2.0]

GUIDANCE = (
    "<b>What you are looking for:</b> where the qmin trace crosses 0.95, 1.5 "
    "and 2.0 (the dashed lines), and whether the label boundary sits on the "
    "crossing. The q profile below is on normalized psi, <b>not rho</b>."
)


def panels(shot, *, t_range=None, paths=None):
    paths = Paths.from_env() if paths is None else paths
    features = paths.features_file(int(shot))
    qmin = read_feature(features, "qmin")
    qpsi = read_feature(features, "qpsi")
    ip = read_feature(features, "ip")

    def window(x, y):
        """Seconds on disk, milliseconds on screen, clipped to the window."""
        x = np.asarray(x) * 1000.0
        if t_range is None:
            return x, y
        keep = (x >= t_range[0]) & (x <= t_range[1])
        return x[keep], y[:, keep]

    qmin_x, qmin_y = window(qmin.x, qmin.y)
    qpsi_x, qpsi_y = window(qpsi.x, qpsi.y)
    ip_x, ip_y = window(ip.x, ip.y)
    return [
        Panel(
            title="qmin (EFIT01 aeqdsk)",
            x=qmin_x,
            y=qmin_y,
            ylabel="q",
            hlines=THRESHOLDS,
        ),
        Panel(
            title="q profile",
            kind="heatmap",
            x=qpsi_x,
            y=np.linspace(0.0, 1.0, qpsi_y.shape[0]),
            z=qpsi_y,
            # normalized psi, NOT rho - see features/namespace.py's note on
            # the qpsi radial axis
            ylabel="psi_n",
        ),
        Panel(title="ip", x=ip_x, y=ip_y, ylabel="A"),
    ]
