"""The rotating-resonance gate on the m/n label (port of dev/mhd/fusion/resonance.py).

A tearing or resonant kink mode m/n sits on q = m/n and the plasma there carries it, so its
f/n follows the rotation at that surface: f/n = a f_rot(rho) + b where q(rho) = m/n. For a
track and a candidate m/n the gate measures how far (m/n, f/n) lies from the plasma's curve
{(q(rho), a f_rot(rho) + b)} in noise units s_q, s_f, on each EFIT02 slice inside the track
(ZIPFIT01 carbon, else ZIPFIT02, within 60 ms), median over the slices; it passes at
r* = 1.96. a, b, s_f and s_q are fit once on the anchors 2/1, 3/2, 4/3 (`freeze.py`).
With the label model's posterior P (inside the rules 1-4 set) and G_c = [r_c <= r*],
P(resonant) = sum_c P_c G_c and the status is
  m/n            the class with the largest P_c G_c holds P_c >= 0.9 and q_min <= m/n + 0.1
  q_min above    as m/n, but q_min is more than 0.1 above m/n
  not resonant   P(resonant) <= 0.1 (above rotation, no surface, views vs rotation)
  uncertain      the rest
  no rotation    no ZIPFIT profile within 60 ms of the track's EFIT02 slices
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import equilibrium as EQ
from .labelmodel import CLASSES, NAMES

GRID = np.linspace(0, 1, 101)  # rho_pol
MAX_SLICES = 30
DT_ROT = 60.0  # ms; ZIPFIT time within this of the EFIT slice
DT_F = 12.0  # ms; the track's frames within this of the slice
R_STAR = 1.96  # |N(0, 1)| at 5%: the gate rejects 5% of true modes
P_CONF = 0.9
Q_TOL = 0.1  # EFIT02's q_min error, taken as 0.1 (not fitted)
STATUSES = ("m/n", "q_min above", "uncertain", "not resonant", "no rotation")
LAW_CLASSES = ("2/1", "3/2", "4/3")
NAMES_A = np.array(NAMES)
MN = np.array([m / n for n, m in CLASSES])
NOF = np.array([n for n, _ in CLASSES])


def slices(tab, frames, eqs: dict, rots: dict):
    """Per track (row position in tab), the EFIT02 slices inside [t0, t1] (at most MAX_SLICES,
    else the one nearest the mid-time): q and f_rot (kHz, sign of the core rotation) on GRID and
    f/n then. `rots[shot]` is `equilibrium.rotation`'s (list, sign)."""
    rows, Q, FR, FN = [], [], [], []
    fr_by = {k: g for k, g in frames.groupby(["shot", "family_id"])}
    pos = pd.Series(np.arange(len(tab)), index=tab.index)
    for shot, g in tab.groupby("shot"):
        rot, sign = rots.get(int(shot)) or ([], 1.0)
        eq = eqs.get(int(shot))
        if not rot or eq is None:
            continue
        for i, r in zip(pos[g.index].values, g.itertuples()):
            tr = fr_by.get((shot, r.family_id))
            ks = np.flatnonzero((eq.t >= r.t0) & (eq.t <= r.t1))
            if not len(ks):
                ks = np.array([int(np.argmin(np.abs(eq.t - (r.t0 + r.t1) / 2)))])
            if len(ks) > MAX_SLICES:
                ks = ks[np.linspace(0, len(ks) - 1, MAX_SLICES).astype(int)]
            for k in ks:
                for w, rho_t, tt in rot:
                    j = int(np.argmin(np.abs(tt - eq.t[k])))
                    if abs(tt[j] - eq.t[k]) <= DT_ROT:
                        break
                else:
                    continue
                near = tr[np.abs(tr.t_ms - eq.t[k]) <= DT_F] if tr is not None else ()
                f = near.f0.median() if len(near) else r.f
                rows.append(i)
                Q.append(np.abs(np.interp(GRID**2, eq.psin, eq.q[k])))
                FR.append(
                    sign
                    * np.interp(EQ.rho_tor_of(eq, k, GRID), rho_t, w[j])
                    / (2 * np.pi)
                )
                FN.append(f / r.n)
    nG = len(GRID)
    return {
        "i": np.array(rows, int),
        "q": np.array(Q).reshape(-1, nG),
        "fr": np.array(FR).reshape(-1, nG),
        "fn": np.array(FN, float),
    }


def _robust(x, y, it=5):
    """y = a x + b by least squares with 3-MAD trimming; returns a, b, the residual scale."""
    keep = np.isfinite(x) & np.isfinite(y)
    for _ in range(it):
        a, b = np.polyfit(x[keep], y[keep], 1)
        res = y - a * x - b
        s = 1.4826 * np.median(np.abs(res[keep]))
        keep = np.isfinite(res) & (np.abs(res) < 3 * s)
    return float(a), float(b), float(s)


def fit_law(sl, tab):
    """a, b, s_f, s_q from the anchors 2/1, 3/2, 4/3."""
    an = tab.anchor4.values[sl["i"]]
    ok = np.isin(an, LAW_CLASSES)
    q, fr, fn = sl["q"][ok], sl["fr"][ok], sl["fn"][ok]
    mn = np.array([int(c.split("/")[0]) / int(c.split("/")[1]) for c in an[ok]])
    # f_rot at the true surface: the outermost crossing of q = m/n, else q_min
    xs = np.full(len(q), np.nan)
    for s in range(len(q)):
        c = np.flatnonzero(np.diff(np.sign(q[s] - mn[s])) != 0)
        if len(c):
            c = c[-1]
            u = (mn[s] - q[s, c]) / (q[s, c + 1] - q[s, c])
            xs[s] = fr[s, c] + u * (fr[s, c + 1] - fr[s, c])
        else:
            xs[s] = fr[s, np.argmin(q[s])]
    a, b, s_f = _robust(xs, fn)
    # s_q: q - m/n at the co-rotating radius nearest the true surface
    dq = np.full(len(q), np.nan)
    for s in range(len(q)):
        g = a * fr[s] + b - fn[s]
        c = np.flatnonzero(np.diff(np.sign(g)) != 0)
        if len(c):
            qc = q[s, c] + (q[s, c + 1] - q[s, c]) * g[c] / (g[c] - g[c + 1])
            dq[s] = qc[np.argmin(np.abs(qc - mn[s]))] - mn[s]
    s_q = 1.4826 * np.nanmedian(np.abs(dq - np.nanmedian(dq)))
    return {
        "a": a,
        "b": b,
        "s_f": s_f,
        "s_q": float(s_q),
        "N": int(ok.sum()),
        "frac_corot": float(np.isfinite(dq).mean()),
    }


def distances(sl, tab, law):
    """Per track: r for every class of its n (NaN for the other n), the median (q_min - m/n)
    per class, the median f/n and the median of a max f_rot + b."""
    nT = len(tab)
    R = np.full((nT, len(MN)), np.nan)
    A = np.full((nT, len(MN)), np.nan)
    fn = np.full(nT, np.nan)
    top = np.full(nT, np.nan)
    idx = sl["i"]
    if not len(idx):
        return R, A, fn, top
    nS = NOF[None] == tab.n.values[idx][:, None]
    D = np.full((len(idx), len(MN)), np.nan)
    pred = law["a"] * sl["fr"] + law["b"]
    for j, mn in enumerate(MN):
        d2 = ((sl["q"] - mn) / law["s_q"]) ** 2 + (
            (sl["fn"][:, None] - pred) / law["s_f"]
        ) ** 2
        D[:, j] = np.sqrt(np.nanmin(d2, axis=1))
    D[~nS] = np.nan
    above = np.nanmin(sl["q"], axis=1)[:, None] - MN[None]
    df = pd.DataFrame(np.c_[idx, D, above, sl["fn"], np.nanmax(pred, axis=1)])
    med = df.groupby(0).median()
    k = med.index.values.astype(int)
    R[k] = med.iloc[:, : len(MN)].values
    A[k] = med.iloc[:, len(MN) : 2 * len(MN)].values
    fn[k], top[k] = med.iloc[:, -2].values, med.iloc[:, -1].values
    R[NOF[None] != tab.n.values[:, None]] = np.nan
    return R, A, fn, top


def decide(P, R, A, fn, top, law, rstar=R_STAR):
    """label, status, reason, cand, p, p_res per track (the module docstring's rules)."""
    G = np.nan_to_num(R, nan=np.inf) <= rstar
    Q = np.where(G, P, 0.0)
    j = Q.argmax(axis=1)
    p = Q[np.arange(len(Q)), j]
    p_res = Q.sum(axis=1)
    above = A[np.arange(len(A)), j] > Q_TOL
    has_r = np.isfinite(R).any(axis=1)
    status = np.select(
        [~has_r, p_res <= 1 - P_CONF, (p >= P_CONF) & above, p >= P_CONF],
        ["no rotation", "not resonant", "q_min above", "m/n"],
        "uncertain",
    )
    label = np.where(status == "m/n", NAMES_A[j], "")
    near = np.where(G, np.nan_to_num(R, nan=np.inf), np.inf).argmin(axis=1)
    cand = np.where(p > 0, NAMES_A[j], np.where(G.any(axis=1), NAMES_A[near], ""))
    with np.errstate(invalid="ignore"):
        reason = np.select(
            [status != "not resonant", fn > top + 2 * law["s_f"], ~G.any(axis=1)],
            ["", "above rotation", "no surface"],
            "views vs rotation",
        )
    return label, status, reason, cand, p, p_res


def load_rotation(paths, shots) -> dict:
    """`equilibrium.rotation` per shot."""
    return {int(s): EQ.rotation(paths, int(s)) for s in shots}
