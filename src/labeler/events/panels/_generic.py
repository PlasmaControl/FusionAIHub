"""Panels for an event whose own traces have not been decided yet.

`ip`, `betan` and `pinj_total` show that the shot exists and that its labels
sit inside the discharge. They do NOT show whether the phenomenon happened.
An event still on these is an event waiting for someone who knows which
traces settle it.
"""

from __future__ import annotations

from ...config import Paths
from ...features.store import read_feature
from ..verify import Panel

GUIDANCE = (
    "<b>These are generic panels.</b> <code>ip</code>, <code>betan</code> and "
    "<code>pinj_total</code> show that the shot exists and that its labels sit "
    "inside the discharge. They do not show whether this phenomenon actually "
    "happened. Until someone adds a builder for this event under "
    "<code>labeler/events/panels/</code>, treat a review here as provisional."
)

TRACES = (("ip", "A"), ("betan", ""), ("pinj_total", "kW"))


def panels(shot, *, t_range=None, paths=None):
    """One panel per scalar feature; seconds on disk, milliseconds on screen."""
    paths = Paths.from_env() if paths is None else paths
    features = paths.features_file(int(shot))
    built = []
    for name, ylabel in TRACES:
        array = read_feature(features, name)
        x = array.x * 1000.0
        y = array.y
        if t_range is not None:
            keep = (x >= t_range[0]) & (x <= t_range[1])
            x, y = x[keep], y[:, keep]
        built.append(Panel(title=name, x=x, y=y, ylabel=ylabel))
    return built
