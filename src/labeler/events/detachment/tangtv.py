"""Geometry-gated C-III front with explicit MARFE evidence requirements.

The upper shelf is Z=-1.25 m, R>=1.37 m. Lower-shelf inversions use the
plasma_tv 2026 window rx<=R<1.37 m and their own lower-shelf strike height.
Height alone is a candidate MARFE, never a MARFE vote.
"""

from __future__ import annotations

import itertools

import numpy as np

from . import thresholds as th
from .core import (
    ABSTAIN,
    ATTACHED,
    DETACHED,
    MARFE,
    Indicator,
    assemble,
    bin_centres,
)


def outer_leg_ze(
    frames: np.ndarray,
    radii: np.ndarray,
    elevation: np.ndarray,
    rx_m: np.ndarray,
    *,
    r_max: float | None = None,
) -> np.ndarray:
    """Weighted C-III height (m) outboard of the X-point, one per inverted frame.

    `frames` is `(T, nZ, nR)`; `radii` (nR,) and `elevation` (nZ,) are the grid axes
    in metres; `rx_m` (T,) is the X-point major radius at each frame. Pixels below
    EMISSION_THRESHOLD are zeroed, the rest are summed over the columns at or
    outboard of `rx_m`, and the height is the SSA estimator
    `sqrt(sum_i (i w_i)^2 / sum_i w_i^2)` on the row index, mapped to metres by
    linear interpolation of `elevation`. NaN where the window holds no emission or
    `rx_m` is not finite.
    """
    frames = np.asarray(frames, dtype=np.float32)
    radii = np.asarray(radii, dtype=float)
    elevation = np.asarray(elevation, dtype=float)
    rows = np.arange(len(elevation), dtype=float)
    out = np.full(len(frames), np.nan)
    for k, frame in enumerate(frames):
        rx = rx_m[k]
        if not np.isfinite(rx):
            continue
        column = int(np.clip(np.searchsorted(radii, rx), 0, len(radii) - 1))
        bright = np.where(frame > th.EMISSION_THRESHOLD, frame, 0.0)
        stop = len(radii) if r_max is None else np.searchsorted(radii, r_max)
        weight = bright[:, column:stop].sum(axis=1)
        denom = float(np.sum(weight**2))
        if denom <= 0.0:
            continue
        index = np.sqrt(np.sum((rows * weight) ** 2) / denom)
        out[k] = np.interp(index, rows, elevation)
    return out


def front_dz(ze: np.ndarray, zx: np.ndarray, zs: np.ndarray) -> np.ndarray:
    """DZ = 1 - (ZX - ZE)/(ZX - ZS); NaN where the leg is shorter than 10 cm."""
    ze, zx, zs = (np.asarray(a, dtype=float) for a in (ze, zx, zs))
    leg = zx - zs
    with np.errstate(invalid="ignore", divide="ignore"):
        dz = 1.0 - (zx - ze) / leg
    return np.where(leg >= th.MIN_LEG_M, dz, np.nan)


def _is_real(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """True where x is finite, inside (lo, hi) and not an EFIT sentinel value."""
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x) & (x > lo) & (x < hi)
    for s in th.EFIT_SENTINELS:
        ok &= ~np.isclose(x, s)
    return ok


def shelf_gate(
    rvsod: np.ndarray,
    zvsod: np.ndarray,
    rxpt1: np.ndarray,
    zxpt1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Where the TangTV method applies: (valid, reason), by EFIT geometry.

    Valid when the plasma is lower single null (primary X-point below
    `LSN_ZX_MAX`) with a real outer strike point on the shelf: R at least
    SHELF_WALL_R and Z within SHELF_Z_TOL of SHELF_Z. The reason names the first
    failing test: `efit_missing`, `not_lower_null`, `strike_on_floor` (R < 1.37 m
    at floor height), `strike_not_on_shelf` (anything else).
    """
    rvsod, zvsod, rxpt1, zxpt1 = (
        np.asarray(a, dtype=float) for a in (rvsod, zvsod, rxpt1, zxpt1)
    )
    have = (
        _is_real(rvsod, 0.8, 2.5)
        & _is_real(zvsod, -1.6, -0.9)
        & _is_real(rxpt1, 0.8, 2.5)
        & _is_real(zxpt1, -1.6, 1.6)
    )
    lsn = have & (zxpt1 < th.LSN_ZX_MAX)
    on_shelf = (rvsod >= th.SHELF_WALL_R) & (
        np.abs(zvsod - th.SHELF_Z) <= th.SHELF_Z_TOL
    )
    on_floor = (rvsod < th.SHELF_WALL_R) & (zvsod < th.SHELF_Z - th.SHELF_Z_TOL)
    valid = lsn & on_shelf & (zxpt1 - zvsod >= th.MIN_LEG_M)
    reason = np.full(rvsod.shape, "", dtype=object)
    reason[valid] = ""
    reason[have & ~lsn] = "not_lower_null"
    reason[have & lsn & ~on_shelf] = "strike_not_on_shelf"
    reason[have & lsn & on_floor] = "strike_on_floor"
    reason[lsn & on_shelf & (zxpt1 - zvsod < th.MIN_LEG_M)] = "short_leg"
    reason[~have] = "efit_missing"
    return valid, reason


def dz_vote(dz: np.ndarray) -> np.ndarray:
    """Vote on a DZ value: attached, detached, marfe, or abstain in the cliff band."""
    dz = np.asarray(dz, dtype=float)
    vote = np.full(dz.shape, ABSTAIN, dtype=np.int8)
    vote[dz < th.DZ_ATTACHED_MAX] = ATTACHED
    vote[(dz >= th.DZ_DETACHED_MIN) & (dz < th.DZ_MARFE_MIN)] = DETACHED
    # High fronts abstain until persistence, spatial evidence and a second cue.
    vote[~np.isfinite(dz)] = ABSTAIN
    return vote


def shelf_tier(edges, frame_t_ms, rvsod, zvsod, accepted):
    """Geometry provenance of accepted frames; mixed bins cannot be upper tier."""
    from .core import bin_mean

    upper = (np.asarray(rvsod) >= th.SHELF_WALL_R) & (
        np.abs(np.asarray(zvsod) - th.SHELF_Z) <= th.SHELF_Z_TOL
    )
    lower = (np.asarray(rvsod) < th.SHELF_WALL_R) & (np.asarray(zvsod) < -1.30)
    upper_share, count = bin_mean(frame_t_ms, upper, edges, keep=accepted)
    lower_share, _ = bin_mean(frame_t_ms, lower, edges, keep=accepted)
    tier = np.full(len(edges) - 1, "none", dtype="U24")
    tier[(count > 0) & (upper_share == 1)] = "upper_shelf"
    tier[(count > 0) & (lower_share == 1)] = "lower_shelf_window"
    tier[(upper_share > 0) & (lower_share > 0)] = "mixed_shelf"
    return tier


def tangtv_indicator(
    edges: np.ndarray,
    frame_t_ms: np.ndarray,
    ze: np.ndarray,
    efit_t_ms: np.ndarray,
    rvsod: np.ndarray,
    zvsod: np.ndarray,
    rxpt1: np.ndarray,
    zxpt1: np.ndarray,
    *,
    max_efit_gap_ms: float = 40.0,
    lower_shelf: bool | np.ndarray = False,
    frame_valid: np.ndarray | None = None,
    elm_t_ms: np.ndarray | None = None,
    elm_flag: np.ndarray | None = None,
    return_frame_mask: bool = False,
) -> Indicator | tuple[Indicator, np.ndarray]:
    """TangTV indicator on a bin grid from per-frame ZE and the EFIT geometry.

    Each inverted frame takes the nearest EFIT slice (at most `max_efit_gap_ms`
    away); the gate, DZ and the vote are evaluated per frame and the bin's value is
    the median DZ of its valid frames. A bin is valid when at least half of the
    frames inside it are, and its vote is the vote of that median DZ. The reason on
    an invalid bin is the most common reason among its frames, or `no_frames` for a
    bin with none.
    """
    from .core import bin_median

    centres = bin_centres(edges)
    n = len(centres)
    efit_t_ms = np.asarray(efit_t_ms, dtype=float)
    near = np.full(len(frame_t_ms), -1, dtype=int)
    if len(efit_t_ms):
        order = np.searchsorted(efit_t_ms, frame_t_ms)
        left = np.clip(order - 1, 0, len(efit_t_ms) - 1)
        right = np.clip(order, 0, len(efit_t_ms) - 1)
        pick = np.where(
            np.abs(efit_t_ms[left] - frame_t_ms)
            <= np.abs(efit_t_ms[right] - frame_t_ms),
            left,
            right,
        )
        close = np.abs(efit_t_ms[pick] - frame_t_ms) <= max_efit_gap_ms
        near = np.where(close, pick, -1)
    have = near >= 0

    def take(a):
        return np.where(have, np.asarray(a, dtype=float)[near], np.nan)

    rv, zv, rx, zx = take(rvsod), take(zvsod), take(rxpt1), take(zxpt1)
    gate, why = shelf_gate(rv, zv, rx, zx)
    if np.any(lower_shelf):
        lower = (
            _is_real(rv, 0.8, th.SHELF_WALL_R)
            & _is_real(zv, -1.45, -1.30)
            & _is_real(rx, 0.8, 2.5)
            & _is_real(zx, -1.3, th.LSN_ZX_MAX)
            & (zx - zv >= th.MIN_LEG_M)
        )
        use_lower = np.broadcast_to(lower_shelf, gate.shape)
        gate = np.where(use_lower, lower, gate)
        why = np.where(use_lower, np.where(lower, "", "lower_shelf_geometry"), why)
    if frame_valid is not None:
        why = np.where(~frame_valid, "surrogate_domain", why)
        gate &= frame_valid

    why = np.where(have, why, "efit_gap")
    dz = front_dz(ze, zx, zv)
    from .core import elm_bin_known

    # Chen 2026's 30 Hz exposures integrate ELMs. Keep their emission fronts;
    # ELM coverage is nevertheless required to distinguish known from missing.
    known = elm_bin_known(edges, elm_t_ms, elm_flag)
    index = np.searchsorted(edges, frame_t_ms, side="right") - 1
    frame_known = np.zeros(len(frame_t_ms), bool)
    inside = (index >= 0) & (index < n)
    frame_known[inside] = known[index[inside]]
    why = np.where(~frame_known & gate, "elm_unknown", why)
    gate &= frame_known
    ok = gate & np.isfinite(dz) & (dz >= th.DZ_UNPHYSICAL_MIN)
    why = np.where(gate & ~np.isfinite(dz), "no_emission", why)
    why = np.where(gate & np.isfinite(dz) & ~ok, "dz_unphysical", why)

    value, count = bin_median(frame_t_ms, dz, edges, keep=ok)
    _, total = bin_median(frame_t_ms, np.ones(len(dz)), edges)
    good = count > 0
    valid = good & (count * 2 >= total)
    reason = np.full(n, "no_frames", dtype=object)
    for b in np.unique(index[(index >= 0) & (index < n)]):
        reasons = [r for r in why[index == b] if r]
        reason[b] = (
            max(set(reasons), key=reasons.count) if reasons else "minority_valid"
        )
    vote = dz_vote(value)
    result = assemble("tangtv", value, valid, reason, vote)
    return (result, ok) if return_frame_mask else result


def evidence_votes(dz, valid, spatial, second_cue):
    """DZ>=1.2 for >=2 adjacent valid bins, psiN<1 peak and independent cue."""
    dz = np.asarray(dz)
    candidate = np.asarray(valid) & (dz >= th.DZ_MARFE_MIN)
    evidence = candidate & np.asarray(spatial) & np.asarray(second_cue)
    edges = np.flatnonzero(np.r_[True, evidence[1:] != evidence[:-1], True])
    sustained = np.zeros(len(dz), bool)
    for start, end in itertools.pairwise(edges):
        if evidence[start] and end - start >= 2:
            sustained[start:end] = True
    vote = dz_vote(dz)
    vote[sustained] = MARFE
    vote[~np.asarray(valid)] = ABSTAIN
    return vote, candidate
