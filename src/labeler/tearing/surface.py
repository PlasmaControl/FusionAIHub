"""Which poloidal number a labelled mode can be given, from the EFIT safety factor.

A tearing mode of toroidal number n sits on the rational surface q = m/n, so m = n q
there. The magnetics alone give n, not where the mode is; EFIT's q profile says which
rational surfaces the plasma has at all. `supported_m` gives m only when the profile
admits exactly one surface with m > n (the surfaces at q <= 1 are the sawtooth and
fishbone kinks, not tearing modes) over the interval; with two or more candidates
nothing in q picks one, and m is left empty. Radial evidence (the ECE island flattening
or the Mirnov poloidal array) is what would choose between candidates; none is used.
"""

from __future__ import annotations

import numpy as np

#: The profile is read inside this normalised radius: q rises towards the separatrix
#: and its last points carry no rational surface a mode lives on.
RHO_MAX = 0.95


def candidate_m(n: int, profile, rho, rho_max: float = RHO_MAX) -> list[int]:
    """The `m > n` whose surface `q = m / n` lies inside the profile's range.

    `profile` is q on `rho`; NaNs are skipped. An empty list when the profile is empty.
    """
    q = np.asarray(profile, dtype=float)
    inside = np.asarray(rho, dtype=float) <= rho_max
    q = q[inside & np.isfinite(q)]
    if q.size == 0:
        return []
    low, high = float(q.min()), float(q.max())
    return [
        m for m in range(n + 1, int(np.floor(high * n)) + 1) if low <= m / n <= high
    ]


def supported_m(
    n: int, t_ms, q, rho, start_ms: float, end_ms: float, rho_max: float = RHO_MAX
) -> int | None:
    """m of a mode of toroidal number `n` over `[start_ms, end_ms]`, or None.

    `q` is `(len(rho), len(t_ms))`. The profile is the median over the samples inside
    the interval; m is returned only if exactly one `m > n` has its surface in it.
    """
    t = np.asarray(t_ms, dtype=float)
    q = np.asarray(q, dtype=float)
    inside = (t >= start_ms) & (t <= end_ms)
    if not inside.any():
        return None
    with np.errstate(all="ignore"):
        profile = np.nanmedian(q[:, inside], axis=1)
    found = candidate_m(n, profile, rho, rho_max)
    return found[0] if len(found) == 1 else None
