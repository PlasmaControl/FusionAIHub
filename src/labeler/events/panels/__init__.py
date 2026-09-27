"""Which traces settle which phenomenon.

Deciding that is case-by-case work that does not generalise, which is why it
used to live in sixteen notebook cells. It lives here instead so it can be
imported, tested, and called by a server - and so there is one copy of it.

A builder module exposes `panels(shot, *, t_range, paths) -> list[Panel]`.
An event with no module gets `_generic`, which is what twelve of those
notebooks already were. Alfven eigenmodes are not here: the review store
builds them with the Heidbrink recipe, `review/alfven.py`.
"""

from __future__ import annotations

from ...config import Paths
from ..verify import Panel
from . import (
    _generic,
    edge_localized_mode,
    fishbone,
    high_confinement_mode,
    minimum_safety_factor,
    neoclassical_tearing_mode,
    sawtooth_oscillation,
)

BUILDERS = {
    "edge_localized_mode": edge_localized_mode,
    "fishbone": fishbone,
    "high_confinement_mode": high_confinement_mode,
    "minimum_safety_factor": minimum_safety_factor,
    "neoclassical_tearing_mode": neoclassical_tearing_mode,
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
