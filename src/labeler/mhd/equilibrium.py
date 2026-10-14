"""The method's parked inputs: the equilibrium with the ECE setup, ZIPFIT rotation, TS geometry.

All under `$LABELER_ROOT/mhd` (`root`): `efit02/<shot>.npz` (dev/mhd/corpus/fetch_eq.py:
the G-EQDSK profiles and flux map, the A-EQDSK q_min and q95, ECE FREQ and ECEZH),
`efit_fallback/<shot>.npz` (the same keys plus `tree`, EFIT02ER or EFIT01, for shots with
no usable EFIT02), `fusion/rot/<shot>.npz` (dev/mhd/fusion/fetch_rot.py: ZIPFIT01/02
TROTFIT, krad/s) and `fusion/ts_geom.csv` (dev/mhd/fusion/fetch_geom.py: the core Thomson
chords' R, Z). Nothing here fetches; a shot without a file has no such input.

`equilibrium` is the one reader the method labels with (owner 2026-10-14: "if efit02
doesnt exist, use efit01 or efit02er"): the EFIT02 file, else the fallback file, each only
with at least `MIN_SLICES` time slices; the `Equilibrium` carries its `tree` and `path`.
EFIT02ER is MSE-constrained like EFIT02 (`TRUSTED`); EFIT01 is magnetics-only and its q_min
reads about 0.1 high at the median and 0.5 at the 90th percentile against EFIT02, which
`labeler.mhd.rows` allows for. `efit02` stays EFIT02-only: the freeze was fit on it.

`Equilibrium` is dev/mhd/corpus/tracks.py's (the q profile, the outermost crossing of each
rational surface, and the ECE channels' rho by the 2nd-harmonic X-mode resonance with the
F(psi) correction); `rot_profile` is dev/mhd/fusion/unified.py's and `rotation`
dev/mhd/fusion/resonance.py's (they choose between ZIPFIT01 and 02 differently, and each
keeps its own choice).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from ..config import Paths

N_ECE = 40  # the radiometer; channels 41-48 are not on the same grid
GHZ_PER_T = 55.98  # 2nd-harmonic X-mode
PAIR_DRHO = 0.04
SURFACES = (("q1", 1.0), ("q43", 4 / 3), ("q32", 1.5), ("q2", 2.0), ("q3", 3.0))
#: ZIPFIT values at or above this (krad/s) are a diverged fit, not rotation.
ROT_DIVERGED = 500.0
#: A shot with no TS geometry of its own borrows the nearest shot's within this.
TS_NEAREST = 2000
#: The trees in the order they are taken (EFIT02 in `efit02/`, the others in
#: `efit_fallback/` under the file's `tree`), and those with an MSE-constrained q.
TREES = ("efit02", "efit02er", "efit01")
TRUSTED = ("efit02", "efit02er")
#: A tree counts only with at least this many time slices (some EFIT02 files hold one).
MIN_SLICES = 2


def root(paths: Paths) -> Path:
    return paths.root / "mhd"


def efit02_path(paths: Paths, shot: int) -> Path:
    return root(paths) / "efit02" / f"{int(shot)}.npz"


def fallback_path(paths: Paths, shot: int) -> Path:
    return root(paths) / "efit_fallback" / f"{int(shot)}.npz"


def rot_path(paths: Paths, shot: int) -> Path:
    return root(paths) / "fusion" / "rot" / f"{int(shot)}.npz"


def ts_geom_path(paths: Paths) -> Path:
    return root(paths) / "fusion" / "ts_geom.csv"


class Equilibrium:
    """One shot's equilibrium (dev/mhd/corpus/tracks.py), sliced at the nearest time;
    `tree` names the EFIT tree and `path` the file it was read from."""

    def __init__(self, d, tree: str = "efit02", path: str | None = None):
        self.tree, self.path = tree, path
        self.t = d["times"]
        self.q = np.abs(d["qpsi"])
        self.F = np.abs(d["fpol"])
        self.psirz, self.r, self.z = d["psirz"], d["r"], d["z"]
        self.sa, self.sb, self.ra = d["ssimag"], d["ssibry"], d["rmaxis"]
        fq = d["ece_freq"] if "ece_freq" in d.files else np.array([np.nan])
        self.freq = fq[:N_ECE].astype(float) if len(fq) >= N_ECE else None
        zh = float(np.ravel(d["ece_zh"])[0]) if "ece_zh" in d.files else np.nan
        self.zh = (
            zh if np.isfinite(zh) else 0.0
        )  # missing ECEZH: the midplane sightline
        self.psin = np.linspace(0, 1, self.q.shape[1])
        self.cache = {}

    def slice(self, t):
        k = int(np.argmin(np.abs(self.t - t)))
        if k in self.cache:
            return self.cache[k]
        q = self.q[k]
        out = {
            "k": k,
            "dt": float(abs(self.t[k] - t)),
            "qmin": float(q.min()),
            "q95": float(np.interp(0.95, self.psin, q)),
            "rho_qmin": float(np.sqrt(self.psin[int(np.argmin(q))])),
        }
        for name, qv in SURFACES:  # outermost crossing of each rational surface
            out[f"rho_{name}"] = np.nan
            below = np.flatnonzero(q < qv)
            if len(below) and below[-1] + 1 < len(q):
                i = below[-1]
                x = np.interp(qv, [q[i], q[i + 1]], [self.psin[i], self.psin[i + 1]])
                out[f"rho_{name}"] = float(np.sqrt(x))
        if self.freq is not None:
            interp = RegularGridInterpolator(
                (self.z, self.r), self.psirz[k], bounds_error=False, fill_value=np.nan
            )

            def psin_at(R):
                ps = interp(np.c_[np.full_like(R, self.zh), R])
                return (ps - self.sa[k]) / (self.sb[k] - self.sa[k])

            R = GHZ_PER_T * self.F[k][-1] / self.freq
            for _ in range(2):  # F(psi) correction inside the plasma
                pn = psin_at(R)
                Fl = np.where(
                    np.isfinite(pn) & (pn < 1),
                    np.interp(np.clip(pn, 0, 1), self.psin, self.F[k]),
                    self.F[k][-1],
                )
                R = GHZ_PER_T * Fl / self.freq
            pn = psin_at(R)
            rho = np.where(np.isfinite(pn), np.sqrt(np.clip(pn, 0, None)), np.nan)
            side = np.where(R < self.ra[k], -1, 1)
            inside = np.isfinite(pn) & (pn < 0.9)
            seen = np.isfinite(pn) & (pn < 1)
            pairs = []
            for h in np.flatnonzero(inside & (side < 0)):
                cand = np.flatnonzero(inside & (side > 0))
                if len(cand):
                    j = cand[np.argmin(np.abs(rho[cand] - rho[h]))]
                    if abs(rho[j] - rho[h]) < PAIR_DRHO:
                        pairs.append((h, j))
            out.update(
                rho=rho,
                side=side,
                seen=seen,
                pairs=np.array(pairs, int).reshape(-1, 2),
                R=R,
                rho_seen=float(rho[seen].max()) if seen.any() else np.nan,
            )
        self.cache[k] = out
        return out


def _stamp(path: Path):
    """`(path, mtime_ns, size)` of a file, or None: the caches' key."""
    try:
        st = path.stat()
    except OSError:
        return None
    return str(path), st.st_mtime_ns, st.st_size


def stamp(path: Path) -> list | None:
    """The file's mtime and size, for a cache key; None when it is not there."""
    got = _stamp(Path(path))
    return None if got is None else [got[1], got[2]]


@lru_cache(maxsize=4)
def _efit02(key, tree="efit02") -> Equilibrium:
    with np.load(key[0]) as d:
        return Equilibrium(d, tree, key[0])


def efit02(paths: Paths, shot: int) -> Equilibrium | None:
    """The shot's EFIT02 file whatever its slices, or None when it has none parked (the
    freeze's input; labelling reads `equilibrium`)."""
    key = _stamp(efit02_path(paths, shot))
    return None if key is None else _efit02(key)


@lru_cache(maxsize=64)
def _probe(key) -> tuple[str, int]:
    """The file's tree (its `tree`, else the directory's: EFIT02) and slice count."""
    with np.load(key[0]) as d:
        tree = str(d["tree"]).strip().lower() if "tree" in d.files else "efit02"
        return tree, int(np.size(d["times"]))


def equilibrium_file(paths: Paths, shot: int) -> tuple[tuple, str] | None:
    """`((path, mtime_ns, size), tree)` of the file the shot's equilibrium comes from: the
    EFIT02 file, else the fallback file, each only with at least `MIN_SLICES` slices; None
    when neither has them (no tree)."""
    for path in (efit02_path(paths, shot), fallback_path(paths, shot)):
        key = _stamp(path)
        if key is None:
            continue
        tree, slices = _probe(key)
        if slices >= MIN_SLICES:
            return key, tree
    return None


def equilibrium(paths: Paths, shot: int) -> Equilibrium | None:
    """The shot's equilibrium (`equilibrium_file`), or None when it has no tree."""
    got = equilibrium_file(paths, shot)
    return None if got is None else _efit02(*got)


@lru_cache(maxsize=4)
def _rot_arrays(key) -> dict:
    """The file's arrays; a one-time ZIPFIT profile (saved 1-D) as one row of [time x rho]."""
    with np.load(key[0]) as d:
        out = {k: d[k] for k in d.files}
    for k in ("01", "02"):
        if f"w{k}" in out and out[f"w{k}"].ndim == 1:
            out[f"w{k}"] = out[f"w{k}"][None, :]
            out[f"t{k}"] = np.atleast_1d(out[f"t{k}"])
    return out


def _rot(paths: Paths, shot: int) -> dict | None:
    key = _stamp(rot_path(paths, shot))
    return None if key is None else _rot_arrays(key)


def rot_profile(paths: Paths, shot: int):
    """(omega krad/s [time x rho], rho_tor, t ms) from ZIPFIT01, else ZIPFIT02, inside
    rho 1; None when neither exists or both diverge (|omega| >= 500 krad/s)
    (dev/mhd/fusion/unified.py)."""
    d = _rot(paths, shot)
    if d is None:
        return None
    for k in ("01", "02"):
        if f"w{k}" in d:
            w, r, tt = d[f"w{k}"], d[f"rho{k}"], d[f"t{k}"]
            core = w[:, r <= 1.0]
            if np.nanmax(np.abs(core)) < ROT_DIVERGED:
                return core, r[r <= 1.0], tt
    return None


def rotation(paths: Paths, shot: int):
    """ZIPFIT01 then ZIPFIT02 as [(omega krad/s [time x rho], rho_tor, t ms)], each keeping
    only its times with |omega| < 500 krad/s inside rho 1; the sign of the core rotation
    (dev/mhd/fusion/resonance.py)."""
    d = _rot(paths, shot)
    if d is None:
        return [], 1.0
    out = []
    for k in ("01", "02"):
        if f"w{k}" in d:
            w, r, tt = d[f"w{k}"], d[f"rho{k}"], d[f"t{k}"]
            w = w[:, r <= 1.0]
            ok = np.isfinite(w).all(axis=1) & (
                np.nanmax(np.abs(w), axis=1) < ROT_DIVERGED
            )
            if ok.any():
                out.append((w[ok], r[r <= 1.0], tt[ok]))
    if not out:
        return [], 1.0
    w, r, _ = out[0]
    return out, (-1.0 if np.nanmedian(w[:, r < 0.2]) < 0 else 1.0)


def rho_tor_of(eq: Equilibrium, k: int, grid):
    """rho_tor at the rho_pol values `grid`, from EFIT's q: Phi = integral of q dpsi."""
    q = eq.q[k]
    phi = np.concatenate([[0], np.cumsum(0.5 * (q[1:] + q[:-1]) * np.diff(eq.psin))])
    return np.sqrt(np.interp(np.asarray(grid) ** 2, eq.psin, phi / phi[-1]))


def ts_geometry(paths: Paths, shot: int) -> tuple[pd.DataFrame | None, int | None]:
    """The core Thomson chords' (ch, R, Z) for the shot, else the nearest shot's within
    `TS_NEAREST` (same campaign), and whose they are; (None, None) without either
    (dev/mhd/fusion/features.py)."""
    path = ts_geom_path(paths)
    if not path.is_file():
        return None, None
    geo = pd.read_csv(path)
    g = geo[geo.shot == shot].sort_values("ch")
    if len(g):
        return g, int(shot)
    have = geo.shot.unique()
    if not len(have):
        return None, None
    near = int(have[np.argmin(np.abs(have - shot))])
    if abs(near - shot) < TS_NEAREST:
        return geo[geo.shot == near].sort_values("ch"), near
    return None, None
