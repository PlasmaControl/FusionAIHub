"""Profiles for the dock's panels (plan section B): one quantity against a radial
coordinate, at every time the shot has it; the page slices it at the cursor.

`/api/profile` sends a profile as little-endian float32: `t` (n_t, ms), then `x`
(n_x, or n_t x n_x when `x_per_time`), then `y` (n_t x n_x, C order), with the header
`{n_t, n_x, x_kind, x_per_time, label, units, x_units, source}`.

The readers (`recipe` sources in configs/labeler/signals.yaml):

- `park_q`: |q|(psi_N) from the MHD park's EFIT02 file (else its fallback tree), 65
  points, one psi_N axis.
- `efit01_q`: the same from EFIT01 through the fetch worker (`fetch_worker`), cached
  in the signals cache.
- `thomson_te`, `thomson_ne`: the corpus' core Thomson chords (`ts_core_temp` eV ->
  keV, `ts_core_density` m^-3 -> 1e19 m^-3). The core chords are vertical, at
  R = 1.94 m, so each chord's x is the major radius where its flux surface crosses
  the outboard midplane, from the nearest EFIT slice (`midplane_r`): `R_m`, per
  time. Without the park's equilibrium or `mhd/fusion/ts_geom.csv` the x is the
  chord number (`channel`).
- `zipfit_rotation`: ZIPFIT toroidal rotation (krad/s) against rho, from the park.
- `ece_te`: ECE Te (keV) of channels 1-40 at 1 ms (the features store, else the
  corpus or raw cache averaged to 1 ms) against each channel's major radius from the
  park's equilibrium (`mhd.equilibrium.Equilibrium.slice`, the second-harmonic
  resonance with the F(psi) correction), per time; `channel` without it.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from ...config import Paths
from ...mhd import equilibrium as mhd_eq
from .. import raw
from .fetch_worker import FetchError
from .signals import CHUNK, Pending, Signals, invalid

#: ECE channels shown (the park's `ece_freq` gives their resonance for these).
N_ECE = 40
ECE_STEP_S = 1e-3
TS_GROUPS = {"thomson_te": ("ts_core_temp", 1e-3), "thomson_ne": ("ts_core_density",
                                                                  1e-19)}
#: A time farther than this from every EFIT slice gets no radius.
EFIT_NEAREST_MS = 100.0
#: Points along the outboard midplane where psi_N is sampled.
MIDPLANE_POINTS = 200


@dataclass
class Profile:
    t_ms: np.ndarray
    x: np.ndarray
    y: np.ndarray
    x_kind: str
    x_units: str
    source: str
    tag: str = ""

    @property
    def x_per_time(self) -> bool:
        return self.x.ndim == 2

    def body(self) -> bytes:
        return b"".join(np.ascontiguousarray(a, dtype="<f4").tobytes()
                        for a in (self.t_ms, self.x, self.y))

    def header(self, entry) -> dict:
        n_t, n_x = self.y.shape
        return {"n_t": int(n_t), "n_x": int(n_x), "x_kind": self.x_kind,
                "x_per_time": self.x_per_time,
                "label": entry.label + (f" ({self.tag})" if self.tag else ""),
                "units": entry.units, "x_units": self.x_units, "source": self.source,
                "id": entry.id}


def read(signals: Signals, shot: int, ident) -> Profile | Pending:
    """The profile `ident` names, or the fetch it waits on. Raises FetchError."""
    entry = signals.resolve(ident)
    if entry.kind != "profile":
        raise invalid(f"{entry.id} is a trace: add it as a row")
    for source in entry.sources:
        if source.fetches:
            got = signals.fetched_profile(shot, entry, source)
            if isinstance(got, Pending):
                return got
            if got is not None:
                return Profile(np.asarray(got["t_ms"]), np.asarray(got["x"]),
                               np.asarray(got["y"]), entry.axis, "", source.wire,
                               "EFIT01")
            continue
        found = READERS[source.name](signals.paths, int(shot))
        if found is not None:
            return found
    wheres = "; ".join(s.where() for s in entry.sources)
    raise FetchError("missing", f"shot {shot} has no {entry.label} profile ({wheres})")


def available(paths: Paths, shot: int, recipe: str) -> bool:
    """Whether a local reader would find something for the shot: metadata only."""
    if recipe == "park_q":
        return mhd_eq.equilibrium_file(paths, shot) is not None
    if recipe == "zipfit_rotation":
        return mhd_eq.rot_path(paths, shot).is_file()
    if recipe in TS_GROUPS:
        return _has_group(paths.corpus_file(shot), TS_GROUPS[recipe][0])
    if recipe == "ece_te":
        return (_has_group(paths.features_file(shot), "ece", features=True)
                or _has_group(paths.corpus_file(shot), "ece")
                or _has_group(raw.cache_path(shot, paths=paths), "ece"))
    return False


def _has_group(path, name: str, features: bool = False) -> bool:
    if not path.is_file():
        return False
    try:
        with h5py.File(path, "r") as f:
            if name not in f:
                return False
            if features:
                return True
            return f[name]["ydata"].shape[-1] > raw.SENTINEL_WIDTH
    except (OSError, KeyError):
        return False


# ---- the readers ----


def park_q(paths: Paths, shot: int) -> Profile | None:
    eq = mhd_eq.equilibrium(paths, shot)
    if eq is None:
        return None
    return Profile(np.asarray(eq.t, dtype=np.float64), eq.psin, eq.q, "psi_n", "",
                   "park", eq.tree.upper())


def zipfit_rotation(paths: Paths, shot: int) -> Profile | None:
    got = mhd_eq.rot_profile(paths, shot)
    if got is None:
        return None
    omega, rho, t_ms = got
    order = np.argsort(t_ms, kind="stable")
    return Profile(np.asarray(t_ms, dtype=np.float64)[order], np.asarray(rho),
                   np.asarray(omega)[order], "rho", "", "park", "ZIPFIT")


def _thomson(paths: Paths, shot: int, recipe: str) -> Profile | None:
    group, scale = TS_GROUPS[recipe]
    path = paths.corpus_file(shot)
    if not _has_group(path, group):
        return None
    with h5py.File(path, "r") as f:
        t_ms = np.asarray(f[group]["xdata"][:], dtype=np.float64) * 1e3
        values = np.asarray(f[group]["ydata"][:], dtype=np.float64)
    y = (values.T * scale).astype(np.float32)
    y[~np.isfinite(y)] = np.nan
    keep = np.isfinite(t_ms)
    t_ms, y = t_ms[keep], y[keep]
    if t_ms.size == 0 or not np.isfinite(y).any():
        return None
    geo, _ = mhd_eq.ts_geometry(paths, shot)
    eq = mhd_eq.equilibrium(paths, shot)
    if geo is None or eq is None:
        return Profile(t_ms, np.arange(1, y.shape[1] + 1, dtype=np.float32), y,
                       "channel", "", "corpus")
    k = min(len(geo), y.shape[1])
    radii = _midplane_cached(eq, tuple(geo.R.values[:k]), tuple(geo.Z.values[:k]))
    x = np.full(y.shape, np.nan, dtype=np.float32)
    x[:, :k] = per_time(radii, eq.t, t_ms)
    return Profile(t_ms, x, y, "R_m", "m", "corpus", f"R from {eq.tree.upper()}")


def thomson_te(paths: Paths, shot: int) -> Profile | None:
    return _thomson(paths, shot, "thomson_te")


def thomson_ne(paths: Paths, shot: int) -> Profile | None:
    return _thomson(paths, shot, "thomson_ne")


def ece_te(paths: Paths, shot: int) -> Profile | None:
    got = _ece_values(paths, shot)
    if got is None:
        return None
    t_ms, y = got
    eq = mhd_eq.equilibrium(paths, shot)
    if eq is None or eq.freq is None:
        return Profile(t_ms, np.arange(1, y.shape[1] + 1, dtype=np.float32), y,
                       "channel", "", "corpus")
    radii = _ece_radii(eq)
    x = np.full(y.shape, np.nan, dtype=np.float32)
    k = min(radii.shape[1], y.shape[1])
    x[:, :k] = per_time(radii[:, :k], eq.t, t_ms)
    return Profile(t_ms, x, y, "R_m", "m", "corpus", f"R from {eq.tree.upper()}")


READERS = {"park_q": park_q, "thomson_te": thomson_te, "thomson_ne": thomson_ne,
           "zipfit_rotation": zipfit_rotation, "ece_te": ece_te}


# ---- geometry ----


def per_time(by_slice: np.ndarray, slice_ms, t_ms) -> np.ndarray:
    """`(n_t, C)`: each time's row of `by_slice` (K, C), from its nearest slice; NaN
    farther than `EFIT_NEAREST_MS` from every slice."""
    slice_ms = np.asarray(slice_ms, dtype=np.float64)
    order = np.argsort(slice_ms, kind="stable")
    sorted_ms = slice_ms[order]
    right = np.clip(np.searchsorted(sorted_ms, t_ms), 1, sorted_ms.size - 1) if (
        sorted_ms.size > 1) else np.zeros(len(t_ms), dtype=np.int64)
    left = np.maximum(right - 1, 0)
    pick = np.where(np.abs(sorted_ms[left] - t_ms) <= np.abs(sorted_ms[right] - t_ms),
                    left, right)
    out = by_slice[order[pick]].astype(np.float32)
    out[np.abs(sorted_ms[pick] - t_ms) > EFIT_NEAREST_MS] = np.nan
    return out


def midplane_r(eq, R, Z) -> np.ndarray:
    """`(K, C)`: per EFIT slice, the major radius at which the flux surface through
    each point `(R, Z)` crosses the outboard midplane (the axis' height); NaN off
    the monotonic part of psi_N along it."""
    R, Z = np.asarray(R, dtype=np.float64), np.asarray(Z, dtype=np.float64)
    out = np.full((eq.t.size, R.size), np.nan, dtype=np.float32)
    for k in range(eq.t.size):
        span = float(eq.sb[k] - eq.sa[k])
        if not np.isfinite(span) or span == 0:
            continue
        psi = RegularGridInterpolator((eq.z, eq.r), eq.psirz[k], bounds_error=False,
                                      fill_value=np.nan)
        points = (psi(np.c_[Z, R]) - eq.sa[k]) / span
        column = int(np.argmin(np.abs(eq.r - eq.ra[k])))
        z0 = float(eq.z[int(np.nanargmin(np.abs(eq.psirz[k][:, column] - eq.sa[k])))])
        rr = np.linspace(float(eq.ra[k]), float(eq.r[-1]), MIDPLANE_POINTS)
        line = (psi(np.c_[np.full_like(rr, z0), rr]) - eq.sa[k]) / span
        steps = np.diff(line)
        bad = np.flatnonzero(~np.isfinite(steps) | (steps <= 0))
        stop = int(bad[0]) + 1 if bad.size else line.size
        if stop < 2:
            continue
        out[k] = np.interp(points, line[:stop], rr[:stop], left=np.nan, right=np.nan)
    return out


_GEOMETRY: OrderedDict = OrderedDict()
_GEOMETRY_LOCK = Lock()


def _remember(key, build):
    with _GEOMETRY_LOCK:
        if key in _GEOMETRY:
            _GEOMETRY.move_to_end(key)
            return _GEOMETRY[key]
    value = build()
    with _GEOMETRY_LOCK:
        _GEOMETRY[key] = value
        while len(_GEOMETRY) > 32:
            _GEOMETRY.popitem(last=False)
    return value


def _eq_key(eq) -> tuple:
    """The equilibrium's file and its stamp, so a rewritten park file is a new key."""
    return (eq.path, tuple(mhd_eq.stamp(eq.path) or ()) if eq.path else id(eq))


def _midplane_cached(eq, R: tuple, Z: tuple) -> np.ndarray:
    return _remember(("midplane", *_eq_key(eq), R, Z), lambda: midplane_r(eq, R, Z))


def _ece_radii(eq) -> np.ndarray:
    """`(K, n)`: each ECE channel's major radius per EFIT slice."""

    def build():
        return np.stack([np.asarray(eq.slice(t)["R"], dtype=np.float32)
                         for t in eq.t])

    return _remember(("ece", *_eq_key(eq)), build)


# ---- ECE values ----


def _ece_values(paths: Paths, shot: int):
    """`(t_ms, y (n_t, N_ECE))` at 1 ms: the features store, else the corpus or the
    raw cache averaged."""
    from ...features.store import read_feature
    from ..equilibrium import canonical

    try:
        array = canonical(read_feature(paths.features_file(shot), "ece"), "ece")
    except (OSError, KeyError, ValueError):
        array = None
    if array is not None and array.y.shape[0] >= 1 and len(array.x) > 1:
        y = np.asarray(array.y[:N_ECE], dtype=np.float32).T
        return np.asarray(array.x, dtype=np.float64) * 1e3, y
    for path in (paths.corpus_file(shot), raw.cache_path(shot, paths=paths)):
        if _has_group(path, "ece"):
            st = path.stat()
            return _remember(("ece_values", str(path), st.st_mtime_ns, st.st_size),
                             lambda path=path: _binned(path, "ece", N_ECE))
    return None


def _binned(path, group: str, rows: int, step_s: float = ECE_STEP_S):
    """The first `rows` rows of a corpus-layout group averaged over `step_s` bins,
    from chunked reads: `(t_ms, y (n_t, rows))`."""
    with h5py.File(path, "r") as f:
        x, y = f[group]["xdata"], f[group]["ydata"]
        rows = min(rows, y.shape[0])
        n = x.shape[0]
        first = float(x[0])
        bins = int((float(x[n - 1]) - first) / step_s) + 1
        sums = np.zeros((rows, bins))
        counts = np.zeros((rows, bins))
        for a in range(0, n, CHUNK):
            b = min(n, a + CHUNK)
            index = np.clip(((x[a:b] - first) / step_s).astype(np.int64), 0, bins - 1)
            values = np.asarray(y[:rows, a:b], dtype=np.float64)
            finite = np.isfinite(values)
            values[~finite] = 0.0
            for c in range(rows):
                sums[c] += np.bincount(index, weights=values[c], minlength=bins)
                counts[c] += np.bincount(index, weights=finite[c], minlength=bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(counts > 0, sums / counts, np.nan)
    t_ms = (first + (np.arange(bins) + 0.5) * step_s) * 1e3
    return t_ms, mean.T.astype(np.float32)
