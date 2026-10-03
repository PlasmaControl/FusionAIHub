"""Which poloidal number a labelled mode can be given, from the EFIT safety factor.

A tearing mode of toroidal number n sits on the rational surface q = m/n, so m = n q
there. The magnetics alone give n, not where the mode is; EFIT's q profile says which
rational surfaces the plasma has at all. `candidate_m` lists those possibilities,
but even a unique candidate does not locate an island. `supported_m` requires an
independently observed island radius, such as ECE flattening, and evaluates q there.
Without that radial evidence m is left empty.
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
    n: int,
    t_ms,
    q,
    rho,
    start_ms: float,
    end_ms: float,
    rho_max: float = RHO_MAX,
    *,
    island_rho: float | None = None,
) -> int | None:
    """m of a mode of toroidal number `n` over `[start_ms, end_ms]`, or None.

    `q` is `(len(rho), len(t_ms))`. At an independently measured `island_rho`,
    median n*q over the interval must be within 0.1 of an integer m > n. No radial
    evidence, no finite q, or a radius outside the observed interior returns None.
    """
    if island_rho is None or not np.isfinite(island_rho):
        return None
    if not 0 <= island_rho <= rho_max:
        return None
    t = np.asarray(t_ms, dtype=float)
    q = np.asarray(q, dtype=float)
    rho = np.asarray(rho, dtype=float)
    inside = (t >= start_ms) & (t <= end_ms)
    if not inside.any():
        return None
    values = []
    for profile in q[:, inside].T:
        finite = np.isfinite(profile) & np.isfinite(rho) & (rho <= rho_max)
        r, p = rho[finite], profile[finite]
        if len(r) < 2:
            continue
        order = np.argsort(r)
        r, p = r[order], p[order]
        if r[0] <= island_rho <= r[-1]:
            values.append(n * np.interp(island_rho, r, p))
    if not values:
        return None
    nq = float(np.median(values))
    m = round(nq)
    return m if m > n and abs(nq - m) <= 0.1 else None
