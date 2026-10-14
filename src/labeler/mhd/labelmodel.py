"""The unified label model: one latent m per track, every view one law shared by all classes
(port of dev/mhd/fusion/labelmodel.py and unified.py, the label model only; the network is
not ported).

For a non-harmonic n = 1-3 track and a candidate m/n (12 classes: m in 1-4, 2-5, 3,4,5,7 for
n = 1, 2, 3), each diagnostic is read under that hypothesis:
  q              the surface must exist (fixed: Phi((m/n - q_min + 0.05)/0.15) Phi((q95 - m/n)
                 /0.15), floor 0.02, inside q_min - 0.2 < m/n < q95 + 0.2)
  wall parity    MPI1A+1B against MPI66M should show (-1)^m
  mirror pairs   3A+3B, 5A+5B, 89A+89B against MPI66M should show sign cos(m theta*)
  ECE parity     HFS/LFS pairs at rho_s +- 0.1 should show (-1)^m
  ECE location   the located rho (L_jump, else L_amp) should follow rho_s
  CO2 extent     V1 and V3 see the mode only if their chords reach rho_s
  rotation       f/n should follow the ZIPFIT carbon rotation at rho_s
  ordering       an outer surface rotates slower than an overlapping anchor's
The learned parts (the prior, the tables p(observed | predicted), the location and rotation
laws, the ECE-parity rates, the ordering error) are fit once by EM on the dev corpus with the
rules 1-4 sets as partial labels (`freeze.py`) and read from freeze.json when labelling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import norm

from ..config import Paths
from . import equilibrium as EQ

M_OF_N = {1: (1, 2, 3, 4), 2: (2, 3, 4, 5), 3: (3, 4, 5, 7)}
CLASSES = [(n, m) for n in (1, 2, 3) for m in M_OF_N[n]]
NAMES = [f"{m}/{n}" for n, m in CLASSES]
CI = {c: i for i, c in enumerate(CLASSES)}
PAIRS = {"M3": (1, 9), "M5": (2, 10), "M89": (3, 11)}
SIG_Q, FLOOR_Q = 0.15, 0.02
Q_MARGIN = 0.2
W_MIR = 0.5
SIG_LOC = (0.06, 0.3)
EPS_LOC = (0.05, 0.5)
WEIGHT = {"W": 1.0, "M3": W_MIR, "M5": W_MIR, "M89": W_MIR, "Es": 1.0, "El": 0.5}
WEIGHT.update(Ce=1.0, Rot=1.0)
LAM = 0.4
RING_RZ = np.array(
    [
        [0.974, 0.182],
        [0.975, 0.85],
        [1.051, 1.33],
        [1.402, 1.407],
        [1.783, 1.323],
        [2.067, 1.09],
        [2.27, 0.746],
        [2.416, 0.249],
        [0.974, -0.187],
        [0.974, -0.854],
        [1.048, -1.33],
        [1.254, -1.406],
        [1.894, -1.102],
        [2.212, -0.873],
        [2.315, -0.624],
        [2.418, -0.001],
    ]
)  # corpus mirnov rows 0-15
CHORD_R = (1.48, 1.94, 2.10)  # CO2 V1, V2, V3 (vertical)
NPRED = {"W": 2, "M3": 3, "M5": 3, "M89": 3, "Ce": 4}
NOBS = {"W": 3, "M3": 3, "M5": 3, "M89": 3, "Ce": 4}
ROT_RANGE = 40.0  # kHz; the outlier density of the rotation law is e / ROT_RANGE
ALL2 = ["W", "M3", "M5", "M89", "Es", "Ce", "Rot", "R", "El"]
SURFQ = {"2/1": 2.0, "3/2": 1.5, "4/3": 4 / 3, "3/1": 3.0}
ITERS = 60


def surfaces(tab: pd.DataFrame, eqs: dict):
    """rho of every class's q = m/n surface (outermost crossing) at each track's mid-time,
    rho_qmin where q_min is above it, 1 where q stays below it; NaN without EFIT02."""
    qs = sorted({m / n for n, m in CLASSES})
    out = np.full((len(tab), len(qs)), np.nan)
    pos = pd.Series(np.arange(len(tab)), index=tab.index)
    for shot, g in tab.groupby("shot"):
        eq = eqs.get(int(shot))
        if eq is None:
            continue
        for i, tm in zip(pos[g.index].values, (g.t0 + g.t1) / 2):
            k = int(np.argmin(np.abs(eq.t - tm)))
            q = eq.q[k]
            for j, qv in enumerate(qs):
                below = np.flatnonzero(q < qv)
                if not len(below):
                    out[i, j] = np.sqrt(eq.psin[int(np.argmin(q))])
                elif below[-1] + 1 < len(q):
                    b = below[-1]
                    out[i, j] = np.sqrt(
                        np.interp(qv, [q[b], q[b + 1]], [eq.psin[b], eq.psin[b + 1]])
                    )
                else:
                    out[i, j] = 1.0
    return {qv: out[:, j] for j, qv in enumerate(qs)}


def mag_views(tab: pd.DataFrame, F: dict) -> dict:
    """The magnetic views: wall parity (even/odd/blind) and the mirror pairs (+/-/blind)."""
    V = {"W": tab.par.map({"even": 0, "odd": 1}).fillna(2).astype(int).values}
    mc, mg = F["mir_c"], F["mir_g"]
    for name, (a, b) in PAIRS.items():
        S = mc[:, a] + mc[:, b]
        with np.errstate(invalid="ignore", divide="ignore"):
            strength = np.abs(S) / (np.abs(mc[:, a]) + np.abs(mc[:, b]))
            cs = np.real(S) / np.abs(S)
            ok = (mg[:, a] >= 0.3) & (mg[:, b] >= 0.3) & (strength >= 0.3)
            V[name] = np.where(ok & (cs > 0.5), 0, np.where(ok & (cs < -0.5), 1, 2))
    return V


def surface_parity(F, rs, tab):
    """ECE across-axis parity at each class's own surface (rho_s +- 0.1): 0 agrees with (-1)^m,
    1 disagrees, 2 blind."""
    nT = len(tab)
    out = np.full((nT, len(CLASSES)), 2, np.int8)
    c_, g_, rho_, side_ = F["ece_c"], F["ece_g"], F["rho"], F["side"]
    for i in range(nT):
        c, g, rho, side = c_[i], g_[i], rho_[i], side_[i]
        with np.errstate(invalid="ignore"):
            ok0 = np.isfinite(c) & (g >= 0.1) & np.isfinite(rho)
        if not ok0.any():
            continue
        for j, (n, m) in enumerate(CLASSES):
            if n != tab.n.values[i] or not np.isfinite(rs[i, j]):
                continue
            ok = ok0 & (np.abs(rho - rs[i, j]) <= 0.1)
            h, l_ = np.flatnonzero(ok & (side < 0)), np.flatnonzero(ok & (side > 0))
            w, cs = [], []
            for a in h:
                if not len(l_):
                    break
                b = l_[np.argmin(np.abs(rho[l_] - rho[a]))]
                if abs(rho[b] - rho[a]) < 0.04:
                    w.append(abs(c[a] * c[b]))
                    cs.append(np.cos(np.angle(c[a] * np.conj(c[b]))))
            if w:
                v = np.average(cs, weights=w)
                if abs(v) >= 0.3:
                    out[i, j] = 0 if np.sign(v) == (1 if m % 2 == 0 else -1) else 1
    return out


def exists(tab):
    """(tracks x classes): the track's n and a surface q_min - 0.2 < m/n < q95 + 0.2 on EFIT02;
    unknown q, or no class left, keeps the n."""
    qv = np.array([m / n for n, m in CLASSES])
    same_n = tab.n.values[:, None] == np.array([n for n, _ in CLASSES])
    out = (
        same_n
        & ~(tab.qmin.values[:, None] - Q_MARGIN >= qv)
        & ~(tab.q95.values[:, None] + Q_MARGIN <= qv)
    )
    none = ~out.any(axis=1)
    out[none] = same_n[none]
    return out


def sets_of(tab, col="list4"):
    return [s.split("|") if isinstance(s, str) else [] for s in tab[col]]


def set_mask(tab, allowed, col="list4"):
    """Rules 1-4 candidate sets as a (tracks x classes) mask; 'other' = the classes the rules
    do not name; no set = every class of the track's n."""
    named = {"1/1", "2/1", "3/1", "3/2", "4/3"}
    M = np.zeros_like(allowed)
    for i, s in enumerate(sets_of(tab, col)):
        for c in s:
            if c in NAMES:
                M[i, NAMES.index(c)] = True
            elif c == "other":
                M[i] |= allowed[i] & np.array([x not in named for x in NAMES])
    M &= allowed
    M[~M.any(axis=1)] = allowed[~M.any(axis=1)]
    return M


def rotation_refs(tab, rho_s):
    """Overlapping rules 1-4 anchors (same shot, half the shorter track) as rotation refs."""
    refs = [[] for _ in range(len(tab))]
    pos = pd.Series(np.arange(len(tab)), index=tab.index)
    for _, g in tab.groupby("shot"):
        A = g[g.anchor4.isin(list(SURFQ))]
        for i, r in g.iterrows():
            for j, a in A.iterrows():
                if i == j:
                    continue
                ov = min(r.t1, a.t1) - max(r.t0, a.t0)
                if ov > 0.5 * min(r.t1 - r.t0, a.t1 - a.t0):
                    rho_a = rho_s[SURFQ[a.anchor4]][pos[j]]
                    if np.isfinite(rho_a):
                        refs[pos[i]].append((rho_a, a.f / a.n))
    return refs


def geometry(tab, eqs: dict, rots: dict):
    """Per track: theta* of the 16 ring probes, the CO2 chords' smallest rho, and the rotation
    profile at the track's mid-time (f_rot in kHz on a rho_pol grid)."""
    nT = len(tab)
    th = np.full((nT, 16), np.nan)
    chord = np.full((nT, 3), np.nan)
    grid = np.linspace(0, 1, 101)
    frot = np.full((nT, len(grid)), np.nan)
    pos = pd.Series(np.arange(nT), index=tab.index)
    for shot, g in tab.groupby("shot"):
        eq = eqs.get(int(shot))
        if eq is None:
            continue
        rot = rots.get(int(shot))
        for i, tm in zip(pos[g.index].values, (g.t0 + g.t1) / 2):
            k = int(np.argmin(np.abs(eq.t - tm)))
            ps = (eq.psirz[k] - eq.sa[k]) / (eq.sb[k] - eq.sa[k])
            interp = RegularGridInterpolator(
                (eq.z, eq.r), ps, bounds_error=False, fill_value=np.nan
            )
            zz = np.linspace(-0.6, 0.6, 121)
            zax = zz[int(np.nanargmin(interp(np.c_[zz, np.full_like(zz, eq.ra[k])])))]
            t_geo = np.arctan2(RING_RZ[:, 1] - zax, RING_RZ[:, 0] - eq.ra[k])
            th[i] = t_geo - LAM * np.sin(t_geo)
            for j, R in enumerate(CHORD_R):
                v = interp(np.c_[np.linspace(-1, 1, 201), np.full(201, R)])
                chord[i, j] = (
                    np.sqrt(np.nanmin(v))
                    if np.isfinite(v).any() and np.nanmin(v) < 1
                    else 1.0
                )
            if rot is not None and rot[2][0] - 60 <= tm <= rot[2][-1] + 60:
                rho_tor = EQ.rho_tor_of(eq, k, grid)
                j = int(np.argmin(np.abs(rot[2] - tm)))
                w = np.interp(rho_tor, rot[1], rot[0][j])
                frot[i] = w / (2 * np.pi)  # krad/s -> kHz
    # a shot spun counter to the current: |omega| through the sign of the core rotation
    for shot, g in tab.groupby("shot"):
        rows = pos[g.index].values
        sub = frot[rows, :20]
        if not np.isfinite(sub).any():
            continue
        core = np.nanmedian(sub)
        if np.isfinite(core) and core < 0:
            frot[rows] *= -1
    return th, chord, grid, frot


def at_surface(grid, prof, rs):
    out = np.full(rs.shape, np.nan)
    for j in range(rs.shape[1]):
        ok = np.isfinite(rs[:, j]) & np.isfinite(prof).all(axis=1)
        out[ok, j] = [np.interp(r, grid, p) for r, p in zip(rs[ok, j], prof[ok])]
    return out


def predictions(RS, chord, TH):
    """What each class says every categorical view should show, (tracks x classes)."""
    nT, nC = RS.shape
    m = np.array([c[1] for c in CLASSES])
    P = {"W": np.broadcast_to((m % 2)[None], (nT, nC)).copy()}
    for v, (a, _) in PAIRS.items():
        c = np.cos(m[None] * TH[:, a : a + 1])
        P[v] = np.where(c > 0.3, 0, np.where(c < -0.3, 1, 2))
    reach = (chord[:, [0, 2]][:, None, :] < RS[:, :, None]).sum(axis=2)
    P["Ce"] = np.where(np.isfinite(RS), reach, 3)
    return P


def observations(F, V):
    cc, cg = F["co2_c"], F["co2_g"]
    amp = np.abs(cc)
    with np.errstate(invalid="ignore", divide="ignore"):
        have = (
            (F["co2_n"] > 0).all(axis=1)
            & (np.nanmax(np.where(np.isfinite(cg), cg, -np.inf), axis=1) >= 0.1)
            & np.isfinite(amp).all(axis=1)
        )
        edge = (amp[:, 1] + amp[:, 3]) / (2 * amp[:, 2])
    O = {v: V[v] for v in ("W", "M3", "M5", "M89")}
    O["Ce"] = np.where(have, np.digitize(edge, [0.6, 0.9]), 3)
    return O


def prepare(tab, F, loc, eqs, rots):
    """Every view of every track, read under each class."""
    V = mag_views(tab, F)
    rho_s = surfaces(tab, eqs)
    refs = rotation_refs(tab, rho_s)
    RS = np.stack([rho_s[m / n] for n, m in CLASSES], axis=1)
    ES = surface_parity(F, RS, tab)
    TH, chord, grid, frot = geometry(tab, eqs, rots)
    FRS = at_surface(grid, frot, RS)
    PRED = predictions(RS, chord, TH)
    OBS = observations(F, V)
    return {
        "V": V,
        "rho_s": rho_s,
        "refs": refs,
        "RS": RS,
        "ES": ES,
        "FRS": FRS,
        "PRED": PRED,
        "OBS": OBS,
    }


class LabelModel:
    """dev's labelmodel.LabelModel with unified.Unified's tied views, merged; only what the
    unified posterior reads is kept (dev's per-view tables and blind rates are unused there)."""

    def __init__(self, use=ALL2):
        self.use = list(use)

    def setup(self, tab, X, loc):
        self.n = tab.n.values
        nT, nC = len(tab), len(CLASSES)
        logq = np.full((nT, nC), -np.inf)
        self.rs = np.full((nT, nC), np.nan)
        with np.errstate(invalid="ignore"):
            for j, (n, m) in enumerate(CLASSES):
                sel = self.n == n
                qv = m / n
                ex = norm.cdf((qv - tab.qmin.values + 0.05) / SIG_Q) * norm.cdf(
                    (tab.q95.values - qv) / SIG_Q
                )
                logq[sel, j] = np.log(np.maximum(np.nan_to_num(ex, nan=0.5), FLOOR_Q))[
                    sel
                ]
                self.rs[:, j] = X["rho_s"][qv]
        self.allowed = exists(tab)
        self.logq, self.loc = logq, loc
        self.fn = (tab.f / tab.n).values
        self.set_rotation(X["refs"])
        self.Es = X["ES"]
        self.PRED, self.OBS = X["PRED"], X["OBS"]
        self.fn_obs, self.frs = self.fn, X["FRS"]
        self.rp = {"mu": 0.0, "b": 0.0, "s": 2.0, "e": 0.2}

    def set_rotation(self, refs):
        """For each class, the number of references its surface is ordered consistently /
        inconsistently with (outer surface <-> lower f/n)."""
        nT, nC = self.allowed.shape
        self.rot_ok = np.zeros((nT, nC))
        self.rot_bad = np.zeros((nT, nC))
        for i, rr in enumerate(refs):
            for rho_a, fn_a in rr:
                fn = self.fn[i]
                if abs(fn - fn_a) < 0.05 * max(fn, fn_a):
                    continue
                d = self.rs[i] - rho_a
                with np.errstate(invalid="ignore"):
                    use = np.isfinite(d) & (np.abs(d) >= 0.03)
                ok = use & (np.sign(d) == -np.sign(fn - fn_a))
                self.rot_ok[i] += ok
                self.rot_bad[i] += use & ~ok
        k = self.rot_ok + self.rot_bad
        self.rot_t = np.where(k > 0, np.minimum(1.0, 3.0 / np.maximum(k, 1)), 0.0)
        self.eta = 0.15

    def init_from_sets(self, sets):
        R = np.where(self.allowed, 1.0, 0.0)
        for i, s in enumerate(sets):
            if s:
                z = np.zeros(len(CLASSES))
                for c in s:
                    if c in NAMES:
                        z[NAMES.index(c)] = 1
                if z.any():
                    R[i] = z
        return R / R.sum(axis=1, keepdims=True)

    def mstep(self, R, rows=None):
        rows = np.arange(len(R)) if rows is None else rows
        Rr = R[rows]
        self.prior = np.log(np.maximum(Rr.sum(axis=0) + 1.0, 1e-9))
        for n in (1, 2, 3):
            js = [CI[(n, m)] for m in M_OF_N[n]]
            self.prior[js] -= np.log(np.exp(self.prior[js]).sum())
        if not hasattr(self, "lp"):
            self.lp = {v: {"mu": 0.0, "b": 0.0, "s": 0.1, "e": 0.2} for v in self.loc}
        for v, locv in self.loc.items():
            lp = self.lp[v]
            has = np.isfinite(locv[rows])
            d = locv[rows][:, None] - self.rs[rows]
            w = Rr * has[:, None] * np.isfinite(d)
            dd = np.nan_to_num(d)
            rsr = np.nan_to_num(self.rs[rows])
            for _ in range(5):  # offset a + b rho_s, the same law for every class
                mu = lp["mu"] + lp["b"] * rsr
                g = (1 - lp["e"]) * norm.pdf(dd, mu, lp["s"])
                inl = g / (g + lp["e"])
                wi = (w * inl).ravel()
                A = np.c_[np.ones(wi.size), rsr.ravel()]
                coef = np.linalg.solve(
                    A.T @ (A * wi[:, None]) + 1e-6 * np.eye(2), A.T @ (wi * dd.ravel())
                )
                lp["mu"], lp["b"] = float(coef[0]), float(np.clip(coef[1], -0.8, 0.8))
                res = dd - lp["mu"] - lp["b"] * rsr
                lp["s"] = float(
                    np.clip(
                        np.sqrt((w * inl * res**2).sum() / (w * inl).sum()), *SIG_LOC
                    )
                )
                lp["e"] = float(np.clip((w * (1 - inl)).sum() / w.sum(), *EPS_LOC))
        self.ttab = {}
        for v in self.PRED:
            cnt = np.ones((NPRED[v], NOBS[v]))
            pr = self.PRED[v][rows]
            ob = np.broadcast_to(self.OBS[v][rows][:, None], pr.shape)
            np.add.at(cnt, (pr, ob), Rr)
            self.ttab[v] = np.log(cnt / cnt.sum(axis=1, keepdims=True))
        rp = self.rp
        y = self.fn_obs[rows][:, None]
        yh = self.frs[rows]
        has = np.isfinite(yh)
        w = Rr * has
        yh0 = np.nan_to_num(yh)
        d = np.nan_to_num(y - yh)
        for _ in range(
            5
        ):  # f/n = f_rot(rho_s) (1 + b) + mu, the same law for every class
            g = (1 - rp["e"]) * norm.pdf(d, rp["mu"] + rp["b"] * yh0, rp["s"])
            inl = g / (g + rp["e"] / ROT_RANGE)
            wi = (w * inl).ravel()
            A = np.c_[np.ones(wi.size), yh0.ravel()]
            coef = np.linalg.solve(
                A.T @ (A * wi[:, None]) + 1e-6 * np.eye(2), A.T @ (wi * d.ravel())
            )
            rp["mu"], rp["b"] = float(coef[0]), float(np.clip(coef[1], -0.9, 1.0))
            res = d - rp["mu"] - rp["b"] * yh0
            rp["s"] = float(
                np.clip(
                    np.sqrt((w * inl * res**2).sum() / max((w * inl).sum(), 1e-9)),
                    0.5,
                    20,
                )
            )
            rp["e"] = float(
                np.clip((w * (1 - inl)).sum() / max(w.sum(), 1e-9), 0.02, 0.5)
            )

    def mstep_es(self, R, rows):
        E, Rr, al = self.Es[rows], R[rows], self.allowed[rows]
        t = np.ones(3)
        b = np.ones(3)
        for o in range(3):
            t[o] += (Rr * (E == o)).sum()
            b[o] += ((1 - Rr) * al * (E == o)).sum()
        self.es_true, self.es_bg = np.log(t / t.sum()), np.log(b / b.sum())

    def mstep_rot(self, R, rows):
        w = R[rows] * self.rot_t[rows]
        bad = (w * self.rot_bad[rows]).sum()
        tot = (w * (self.rot_ok[rows] + self.rot_bad[rows])).sum()
        self.eta = float(np.clip(bad / max(tot, 1e-9), 0.02, 0.45))

    def fit(self, R0, iters=ITERS, rows=None, clamp=None):
        """EM; with `clamp` the posterior of a track is kept inside its candidate set."""
        rows = np.arange(len(R0)) if rows is None else rows
        self.mstep(R0, rows)
        self.mstep_es(R0, rows)
        for _ in range(iters):
            R = self.posterior()
            if clamp is not None:
                R = np.where(clamp, R, 0.0)
                R = R / np.maximum(R.sum(axis=1, keepdims=True), 1e-300)
            self.mstep(R, rows)
            self.mstep_rot(R, rows)
            self.mstep_es(R, rows)
        return self.posterior()

    def loglik(self, use=None):
        use = self.use if use is None else use
        L = self.logq + self.prior[None]
        for v in use:
            if v in self.PRED:
                L = L + WEIGHT[v] * self.ttab[v][self.PRED[v], self.OBS[v][:, None]]
            elif v == "Rot":
                rp = self.rp
                d = self.fn_obs[:, None] - self.frs
                ll = np.log(
                    (1 - rp["e"])
                    * norm.pdf(d, rp["mu"] + rp["b"] * np.nan_to_num(self.frs), rp["s"])
                    + rp["e"] / ROT_RANGE
                )
                L = L + WEIGHT[v] * np.where(np.isfinite(d), ll, 0.0)
            elif v in self.loc:  # the ECE location law; a missing location adds nothing
                lp, locv = self.lp[v], self.loc[v]
                d = np.nan_to_num(locv[:, None] - self.rs)
                ll = np.log(
                    (1 - lp["e"])
                    * norm.pdf(d, lp["mu"] + lp["b"] * np.nan_to_num(self.rs), lp["s"])
                    + lp["e"]
                )
                ok = np.isfinite(locv)[:, None] & np.isfinite(self.rs)
                L = L + WEIGHT[v] * np.where(ok, ll, 0.0)
            elif v == "Es":
                L = L + WEIGHT[v] * (self.es_true[self.Es] - self.es_bg[self.Es])
            elif v == "R":
                L = L + self.rot_t * (
                    self.rot_ok * np.log(1 - self.eta) + self.rot_bad * np.log(self.eta)
                )
        return np.where(self.allowed, L, -np.inf)

    def posterior(self, use=None):
        L = self.loglik(use)
        L = L - L.max(axis=1, keepdims=True)
        P = np.exp(L)
        return P / P.sum(axis=1, keepdims=True)

    def params(self) -> dict:
        """The learned parameters, JSON-ready."""
        return {
            "prior": self.prior.tolist(),
            "ttab": {v: t.tolist() for v, t in self.ttab.items()},
            "rp": dict(self.rp),
            "lp": {v: dict(p) for v, p in self.lp.items()},
            "es_true": self.es_true.tolist(),
            "es_bg": self.es_bg.tolist(),
            "eta": self.eta,
        }

    def load(self, p: dict):
        self.prior = np.asarray(p["prior"], float)
        self.ttab = {v: np.asarray(t, float) for v, t in p["ttab"].items()}
        self.rp = dict(p["rp"])
        self.lp = {v: dict(x) for v, x in p["lp"].items()}
        self.es_true = np.asarray(p["es_true"], float)
        self.es_bg = np.asarray(p["es_bg"], float)
        self.eta = float(p["eta"])


def location(tab) -> dict:
    """The ECE location view: L_jump, else L_amp (dev's "chain" locator)."""
    return {"El": tab.L_jump.fillna(tab.L_amp).values.astype(float)}


def posterior(tab, F, eqs, rots, params) -> np.ndarray:
    """The frozen model's posterior inside each track's rules 1-4 set, (tracks x classes)."""
    loc = location(tab)
    X = prepare(tab, F, loc, eqs, rots)
    lm = LabelModel()
    lm.setup(tab, X, loc)
    lm.load(params)
    S = set_mask(tab, lm.allowed)
    P = np.where(S, lm.posterior(), 0)
    return P / P.sum(axis=1, keepdims=True)


def fit(tab, F, eqs, rots):
    """EM on every row with the rules 1-4 sets clamped (dev's resonance.corpus_posterior):
    the fitted model and its posterior inside the sets."""
    loc = location(tab)
    X = prepare(tab, F, loc, eqs, rots)
    lm = LabelModel()
    lm.setup(tab, X, loc)
    S = set_mask(tab, lm.allowed)
    lm.fit(lm.init_from_sets(sets_of(tab)), rows=np.arange(len(tab)), clamp=S)
    P = np.where(S, lm.posterior(), 0)
    return lm, P / P.sum(axis=1, keepdims=True)


def load_inputs(paths: Paths, shots) -> tuple[dict, dict]:
    """EFIT02 and the unified rotation profile per shot (None where missing)."""
    eqs = {int(s): EQ.efit02(paths, int(s)) for s in shots}
    rots = {int(s): EQ.rot_profile(paths, int(s)) for s in shots}
    return eqs, rots


def credible(P, names):
    """MAP, its probability, and the classes holding 90% of the posterior ('a|b' for two or
    more) (dev/mhd/fusion/cases.py)."""
    o = np.argsort(-P, axis=1)
    out = []
    for i in range(len(P)):
        c = np.cumsum(P[i, o[i]])
        k = int(np.searchsorted(c, 0.9) + 1)
        out.append("|".join(names[i][j] for j in o[i][:k]))
    return names[np.arange(len(P)), o[:, 0]], P[np.arange(len(P)), o[:, 0]], out
