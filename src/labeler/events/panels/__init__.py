"""Which traces settle which phenomenon.

Deciding that is case-by-case work that does not generalise, which is why it
used to live in sixteen notebook cells. It lives here instead so it can be
imported, tested, and called by a server - and so there is one copy of it.

A builder module exposes `panels(shot, *, t_range, paths) -> list[Panel]` and
a `GUIDANCE` string, the prose a reviewer needs beside the figure. An event
with no module gets `_generic`, which is what twelve of those notebooks
already were.
"""

from __future__ import annotations

from ...config import Paths
from ..verify import Panel
from . import (
    _generic,
    alfven_eigenmode,
    fishbone,
    minimum_safety_factor,
    sawtooth_oscillation,
)

BUILDERS = {
    "alfven_eigenmode": alfven_eigenmode,
    "fishbone": fishbone,
    "minimum_safety_factor": minimum_safety_factor,
    "sawtooth_oscillation": sawtooth_oscillation,
}


def build(
    event: str,
    shot: int,
    *,
    t_range: tuple[float, float] | None = None,
    paths: Paths | None = None,
) -> list[Panel]:
    """The panels for one shot of one event, over one window."""
    builder = BUILDERS.get(event, _generic)
    return list(builder.panels(int(shot), t_range=t_range, paths=paths))


def guidance(event: str) -> str:
    """What a reviewer of this event needs to be told, as HTML."""
    return BUILDERS.get(event, _generic).GUIDANCE
