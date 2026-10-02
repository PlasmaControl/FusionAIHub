"""ECE localization for sawtooth review, with conservative missing-data handling.

Local ece_psi is calibrated normalized poloidal flux per ECE channel, on the
same channel axis as ece. EFIT qpsi establishes the innermost q=1 surface
connected to a q<1 magnetic axis. Local ece_q alone supports q-band comparisons,
but cannot establish magnetic-core membership. Neither signal is guessed or
fetched from an unverified diagnostic node.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...config import Paths
from ...features.store import read_feature
from .. import equilibrium
from ..verify import NoDataError


@dataclass(frozen=True)
class Geometry:
    time_ms: np.ndarray
    q: np.ndarray
    psi: np.ndarray | None = None
    q1_psi: np.ndarray | None = None


def _clock(values, name):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all() or (
        np.diff(x) <= 0
    ).any():
        raise ValueError(f"{name} must have finite, increasing sample times")
    return x


def _positions(target, source):
    """Left/right indices, exact samples and contiguous interpolation coverage."""
    x = np.asarray(target, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("target times must be a finite vector")
    t = _clock(source, "geometry")
    left = np.clip(np.searchsorted(t, x, side="right") - 1, 0, len(t) - 1)
    right = np.minimum(left + 1, len(t) - 1)
    exact = x == t[left]
    limit = 1.5 * np.median(np.diff(t))
    between = (
        (x > t[left]) & (x < t[right]) & (t[right] - t[left] <= limit)
    )
    return left, right, exact, between


def classify_q1(q, *, tolerance=1e-3):
    """q<1 and q>1 masks; q-only data make no spatial core claim."""
    values = np.asarray(q, dtype=float)
    if values.ndim != 2:
        raise ValueError("q must have shape (channels, times)")
    valid = np.isfinite(values) & (values > 0)
    return valid & (values < 1 - tolerance), valid & (values > 1 + tolerance)


def q1_surface(psi_grid, qpsi):
    """Innermost q=1 crossing of a continuous positive profile with axis q<1.

    Reverse-shear profiles with q0>=1, radial gaps before the crossing, and
    profiles without a crossing do not establish an axis-connected surface.
    """
    grid = np.asarray(psi_grid, dtype=float)
    profiles = np.asarray(qpsi, dtype=float)
    if grid.ndim != 1 or len(grid) < 2 or not np.isfinite(grid).all() or (
        np.diff(grid) <= 0
    ).any() or grid[0] != 0:
        raise ValueError("qpsi needs an increasing normalized-psi grid from the axis")
    if profiles.ndim != 2 or profiles.shape[1] != len(grid):
        raise ValueError("qpsi must have shape (times, psi)")
    surfaces = np.full(len(profiles), np.nan)
    for time, profile in enumerate(profiles):
        if not np.isfinite(profile[0]) or not 0 < profile[0] < 1:
            continue
        for i in range(1, len(grid)):
            if not np.isfinite(profile[i]) or profile[i] <= 0:
                break
            if profile[i] >= 1:
                fraction = (1 - profile[i - 1]) / (profile[i] - profile[i - 1])
                surfaces[time] = grid[i - 1] + fraction * (grid[i] - grid[i - 1])
                break
    return surfaces


def surface_membership(psi, q1_psi, *, tolerance=1e-3):
    """Calibrated channels inside/outside the axis-connected q=1 surface."""
    positions = np.asarray(psi, dtype=float)
    surface = np.asarray(q1_psi, dtype=float)[None, :]
    if positions.ndim != 2 or positions.shape[1] != surface.shape[1]:
        raise ValueError("ECE psi and q=1 surface sample counts do not agree")
    valid = (
        np.isfinite(positions) & (positions >= 0) & (positions <= 1)
        & np.isfinite(surface) & (surface > 0) & (surface <= 1)
    )
    return (
        valid & (positions < surface - tolerance),
        valid & (positions > surface + tolerance),
    )


def align_q(x_ms, q_x_ms, q):
    """Linear interpolation within finite neighboring samples and native cadence."""
    times = _clock(q_x_ms, "geometry")
    values = np.asarray(q, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(times):
        raise ValueError("geometry must have shape (channels, times)")
    left, right, exact, between = _positions(x_ms, times)
    output = np.full((len(values), len(left)), np.nan)
    span = times[right] - times[left]
    fraction = np.divide(
        np.asarray(x_ms) - times[left], span,
        out=np.zeros(len(left)), where=span > 0,
    )
    for row, trace in enumerate(values):
        good = between & np.isfinite(trace[left]) & np.isfinite(trace[right])
        output[row, good] = (
            trace[left[good]] * (1 - fraction[good])
            + trace[right[good]] * fraction[good]
        )
        output[row, exact] = trace[left[exact]]
    return output


def align_mask(x_ms, geometry_x_ms, mask):
    """Keep membership only where both neighboring geometry slices agree.

    Exact geometry samples remain known. A gap, missing slice, or changing
    membership leaves the intervening ECE samples unassigned; nothing is
    extrapolated outside the geometry record.
    """
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 2 or values.shape[1] != len(geometry_x_ms):
        raise ValueError("geometry mask and times have incompatible shapes")
    left, right, exact, between = _positions(x_ms, geometry_x_ms)
    output = (
        values[:, left] & values[:, right] & between[None, :]
    )
    output[:, exact] = values[:, left[exact]]
    return output


def q_from_psi(psi, psi_grid, qpsi):
    """Channel normalized psi mapped into each native EFIT profile; gaps stay gaps."""
    channels = np.asarray(psi, dtype=float)
    grid = _clock(psi_grid, "normalized psi")
    profiles = np.asarray(qpsi, dtype=float)
    if channels.ndim != 2 or profiles.shape != (channels.shape[1], len(grid)):
        raise ValueError("psi must be (channels,times), qpsi must be (times,psi)")
    mapped = np.full(channels.shape, np.nan)
    for time, profile in enumerate(profiles):
        finite = np.isfinite(channels[:, time])
        if finite.any():
            mapped[finite, time] = align_q(
                channels[finite, time], grid, profile[None, :]
            )[0]
    return mapped


def _local(shot, name, paths):
    for path in (
        paths.features_file(shot), paths.corpus_file(shot),
        paths.raw_cache / f"{int(shot)}_processed.h5",
    ):
        try:
            array = read_feature(path, name)
        except (OSError, KeyError):
            continue
        x = _clock(np.asarray(array.x) * 1000, name)
        y = np.asarray(array.y, dtype=float)
        if y.ndim != 2 or not len(y) or y.shape[1] != len(x):
            raise ValueError(f"{name} must have shape (ECE channels, times)")
        return x, y
    raise NoDataError(f"shot {int(shot)} has no local {name} calibration")


def load_q(shot, paths=None):
    """Local ece_q, whose rows explicitly match the corpus ECE channel axis."""
    return _local(int(shot), "ece_q", Paths.from_env() if paths is None else paths)


def load_geometry(shot, paths=None):
    """Prefer calibrated psi + EFIT; q-only metadata gives explicitly named bands."""
    paths = Paths.from_env() if paths is None else paths
    try:
        try:
            psi_x, psi = _local(int(shot), "ece_psi", paths)
        except NoDataError:
            x, q = load_q(shot, paths)
            _clock(x, "ece_q")
            if np.asarray(q).ndim != 2 or np.asarray(q).shape[1] != len(x):
                raise ValueError("ece_q must have shape (ECE channels, times)")
            return Geometry(x, q)
        array = equilibrium.signal(int(shot), "qpsi", paths)
        times = np.asarray(array.x) * 1000
        profiles = align_q(psi_x, times, array.y).T
        # Canonical EFIT qpsi is on uniform normalized psi, not rho.
        grid = np.linspace(0, 1, profiles.shape[1])
        q = q_from_psi(psi, grid, profiles)
        return Geometry(psi_x, q, psi, q1_surface(grid, profiles))
    except ValueError as error:
        raise NoDataError(f"shot {int(shot)}: malformed ECE geometry: {error}") from error


def inversion_difference(x_ms, y, *, channels=tuple(range(20, 36))):
    """Subtract each channel's full-record mean, then compare two channel bands."""
    values = np.asarray(y, dtype=float)
    selected = np.asarray(channels, dtype=int)
    if values.ndim != 2 or len(selected) != 16 or values.shape[0] <= selected.max():
        raise ValueError("ECE inversion comparison needs sixteen selected channels")

    def demean(group):
        finite = np.isfinite(group)
        count = finite.sum(axis=1)
        baseline = np.divide(
            np.where(finite, group, 0).sum(axis=1), count,
            out=np.full(len(group), np.nan), where=count > 0,
        )[:, None]
        centered = group - baseline
        finite = np.isfinite(centered)
        count = finite.sum(axis=0)
        return np.divide(
            np.where(finite, centered, 0).sum(axis=0), count,
            out=np.full(group.shape[1], np.nan), where=count > 0,
        )

    result = np.vstack([demean(values[selected[:8]]), demean(values[selected[8:]])])
    return np.asarray(x_ms), result
