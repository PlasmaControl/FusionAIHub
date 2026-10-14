"""Every view at the track frequency, summed over the track (port of dev/mhd/fusion/features.py,
aggregate.py and the audit.py locators).

Per frame, at the frame's f0 (`frame_features`):
- the five 2.56 ms sub-windows of tracks.py, each linearly detrended per row before the Hann
  window;
- every channel's cross-spectrum with the reference probe MPI66M322 (the first live toroidal
  probe when it is dead): C = sum_j X_j conj(M_j) / sqrt(K sum_j |M_j|^2), and the coherence
  g = |sum X M*|^2 / (sum |X|^2 sum |M|^2), for 29 Mirnov rows, 40 ECE and 4 CO2 channels;
- the floor at f0 + FLOOR_K/WIN on the 5.12 ms window;
- for ECE: the 1 ms Te profile, R, rho and side (EFIT02), and the X2 cutoff (the largest
  right-hand cutoff between the channel's resonance and the LFS separatrix over the channel's
  frequency, from the corpus Thomson core densities mapped through EFIT02);
- for CO2: the 1 ms line densities; a sawtooth flag (the largest 1 ms core Te fall).
Per track (`track_features`): each channel's cross-spectrum summed over the track's frames
where it is usable (alive, inside the separatrix, Te above 50 eV and, for ECE, cutoff ratio
below CUT = 0.95).
Locators (`locate`, `across_axis`): the island's phase flip between two opposite lobes
(L_jump, LFS first), else the coherent amplitude's peak (L_amp), on channels with coherence
>= 0.1; the across-axis ECE parity at matched rho.
"""

from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from . import tracks as T
from .equilibrium import N_ECE, Equilibrium

N_MIR, N_CO2 = 29, 4
REF_ROWS = [15, 20, 21, 16, 18, 22]  # MPI66M322 first, then the other midplane probes
E, ME, EPS0 = 1.602176634e-19, 9.1093837e-31, 8.8541878128e-12
K_SUB = len(T.SUB_OFF)
TE_MIN = 0.05
CUT = 0.95
THR = 0.1
GROUPS = {"mir": N_MIR, "ece": N_ECE, "co2": N_CO2}


def detrend(s):
    tt = np.arange(s.shape[1]) - (s.shape[1] - 1) / 2
    return s - s.mean(axis=1, keepdims=True) - np.outer(s @ tt / (tt @ tt), tt)


def subwins(X, t0, t, f0):
    """(rows, K) complex amplitudes at f0 on the detrended sub-windows, or None near the ends."""
    out = []
    for off in T.SUB_OFF:
        s = T.seg(X, t0, t + off, T.SUB)
        if s is None:
            return None
        out.append(detrend(s) @ T.kern(s.shape[1], f0)[:, 0])
    return np.stack(out, axis=1)


def floor(X, t0, t, f0):
    s = T.seg(X, t0, t, T.WIN)
    if s is None:
        return None
    Z = detrend(s) @ T.kern(s.shape[1], f0 + T.FLOOR_K / T.WIN)
    return np.median(np.abs(Z), axis=1)


def cross(S, M):
    pm = (np.abs(M) ** 2).sum()
    num = S @ M.conj()
    c = num / np.sqrt(K_SUB * pm)
    g = np.abs(num) ** 2 / np.maximum((np.abs(S) ** 2).sum(axis=1) * pm, 1e-300)
    return c, g


class Cutoff:
    """X2 accessibility of each ECE channel on one EFIT02 slice and one Thomson profile."""

    def __init__(self, eq, ts_R, ts_Z):
        self.eq, self.ts_R, self.ts_Z = eq, ts_R, ts_Z
        self.w_ch = 2 * np.pi * eq.freq * 1e9  # the channels' angular frequencies

    def __call__(self, k, ne, R_ch):
        eq = self.eq
        interp = RegularGridInterpolator(
            (eq.z, eq.r), eq.psirz[k], bounds_error=False, fill_value=np.nan
        )

        def norm(ps):
            return (ps - eq.sa[k]) / (eq.sb[k] - eq.sa[k])

        ok = np.isfinite(ne) & (ne > 0)
        if ok.sum() < 5:
            return np.full(len(R_ch), np.nan)
        pts = norm(interp(np.c_[self.ts_Z[ok], self.ts_R[ok]]))
        good = np.isfinite(pts) & (pts < 1)
        if good.sum() < 5:
            return np.full(len(R_ch), np.nan)
        o = np.argsort(pts[good])
        px, py = pts[good][o], ne[ok][good][o]
        py = pd.Series(py).rolling(3, center=True, min_periods=1).median().values
        Rg = np.linspace(eq.r[0], eq.r[-1], 400)
        pg = norm(interp(np.c_[np.full_like(Rg, eq.zh), Rg]))
        inside = np.isfinite(pg) & (pg < 1)
        if not inside.any():
            return np.full(len(R_ch), np.nan)
        neg = np.where(
            inside,
            np.interp(np.clip(pg, 0, 1), np.r_[px, 1.0], np.r_[py, 0.2 * py[-1]]),
            0.0,
        )
        F = np.where(
            inside, np.interp(np.clip(pg, 0, 1), eq.psin, eq.F[k]), eq.F[k][-1]
        )
        wce = E * F / Rg / ME
        wR = wce / 2 + np.sqrt(wce**2 / 4 + neg * E**2 / (EPS0 * ME))
        wR = np.where(
            Rg <= Rg[inside].max() + 1e-6, wR, 0.0
        )  # nothing beyond the separatrix
        tail = np.maximum.accumulate(wR[::-1])[::-1]  # max over R' >= R
        return np.interp(R_ch, Rg, tail) / self.w_ch


def frame_features(path, fr: pd.DataFrame, eq: Equilibrium | None, ts_geo):
    """dev's features.run_shot on the frames `fr` (shot, family_id, t_ms, f0): `(dict, reason)`.
    Without EFIT02 or its ECE frequencies the ECE views stay empty (dev skipped such shots)."""
    w0, w1 = fr.t_ms.min() - 20, fr.t_ms.max() + 20
    with_ece = eq is not None and eq.freq is not None
    with h5py.File(path, "r") as f:
        rm = T.read(f, "mirnov", list(range(N_MIR)), w0, w1)
        re_ = (
            T.read(f, "ece", list(range(N_ECE)), w0, w1, slow=True)
            if with_ece and "ece" in f
            else None
        )
        rc = (
            T.read(f, "co2", list(range(N_CO2)), w0, w1, slow=True)
            if "co2" in f and f["co2/xdata"].shape[0] > 1000
            else None
        )
        tsx = f["ts_core_density/xdata"][:] * 1e3 if "ts_core_density" in f else None
        tsy = f["ts_core_density/ydata"][:] if tsx is not None else None
    if rm is None:
        return None, "no mirnov"
    mt0, M, malive, _ = rm
    ref = next((r for r in REF_ROWS if malive[r]), None)
    if ref is None:
        return None, "no reference probe"
    cut = None
    if with_ece and ts_geo is not None and len(ts_geo) and tsy is not None:
        k = min(len(ts_geo), tsy.shape[0])
        tsy = tsy[:k]
        cut = Cutoff(eq, ts_geo.R.values[:k], ts_geo.Z.values[:k])
    nF = len(fr)

    def full(shape, dtype=np.float32):
        return np.full(shape, np.nan, dtype)

    out = {
        "t_ms": fr.t_ms.values,
        "f0": fr.f0.values,
        "family_id": fr.family_id.values,
        "mir_c": full((nF, N_MIR), np.complex64),
        "mir_g": full((nF, N_MIR)),
        "mir_floor": full((nF, N_MIR)),
        "ref_amp": full(nF),
        "ece_c": full((nF, N_ECE), np.complex64),
        "ece_g": full((nF, N_ECE)),
        "ece_floor": full((nF, N_ECE)),
        "te": full((nF, N_ECE)),
        "R": full((nF, N_ECE)),
        "rho": full((nF, N_ECE)),
        "side": np.zeros((nF, N_ECE), np.int8),
        "seen": np.zeros((nF, N_ECE), bool),
        "cutoff": full((nF, N_ECE)),
        "crash": full(nF),
        "co2_c": full((nF, N_CO2), np.complex64),
        "co2_g": full((nF, N_CO2)),
        "co2_floor": full((nF, N_CO2)),
        "co2_lvl": full((nF, N_CO2)),
        "mir_alive": malive,
        "ref_row": ref,
    }
    if re_ is not None:
        et0, EC, ealive, Eslow = re_
        out["ece_alive"] = ealive
    if rc is not None:
        ct0, CC, calive, Cslow = rc
        out["co2_alive"] = calive
    # A 1 ms Te or CO2 block inside a NaN stretch of the record reduces to NaN, as in dev,
    # without numpy's all-NaN RuntimeWarnings (only those).
    with T.quiet_all_nan():
        for i, (t, f0) in enumerate(zip(fr.t_ms.values, fr.f0.values)):
            S = subwins(M, mt0, t, f0)
            if S is None:
                continue
            Mr = S[ref]
            out["ref_amp"][i] = np.sqrt((np.abs(Mr) ** 2).mean())
            c, gg = cross(S, Mr)
            out["mir_c"][i], out["mir_g"][i] = c, gg
            fl = floor(M, mt0, t, f0)
            if fl is not None:
                out["mir_floor"][i] = fl
            if re_ is not None:
                geo_k = eq.slice(t)
                SE = subwins(EC, et0, t, f0)
                if SE is not None:
                    c, gg = cross(SE, Mr)
                    out["ece_c"][i] = np.where(ealive, c, np.nan)
                    out["ece_g"][i] = np.where(ealive, gg, np.nan)
                    fl = floor(EC, et0, t, f0)
                    if fl is not None:
                        out["ece_floor"][i] = fl
                j0 = max(0, int((t - T.WIN / 2 - et0) / (T.SLOW_N * 0.002)))
                blk = Eslow[:, j0 : j0 + max(1, int(T.WIN))]
                out["te"][i] = np.nanmean(blk, axis=1)
                out["R"][i], out["rho"][i] = geo_k["R"], geo_k["rho"]
                out["side"][i], out["seen"][i] = geo_k["side"], geo_k["seen"]
                core = ealive & geo_k["seen"] & (geo_k["rho"] < 0.3)
                span = Eslow[:, max(0, j0 - 2) : j0 + int(T.WIN) + 3]
                if core.any() and span.shape[1] >= 2:
                    drop = -np.diff(span[core], axis=1) / np.maximum(
                        np.nanmean(span[core], axis=1, keepdims=True), 1e-6
                    )
                    out["crash"][i] = (
                        float(np.nanmax(drop)) if np.isfinite(drop).any() else np.nan
                    )
                if cut is not None:
                    kt = int(np.argmin(np.abs(tsx - t)))
                    out["cutoff"][i] = cut(
                        geo_k["k"], tsy[:, kt].astype(float), geo_k["R"]
                    )
            if rc is not None:
                SC = subwins(CC, ct0, t, f0)
                if SC is not None:
                    c, gg = cross(SC, Mr)
                    out["co2_c"][i] = np.where(calive, c, np.nan)
                    out["co2_g"][i] = np.where(calive, gg, np.nan)
                    fl = floor(CC, ct0, t, f0)
                    if fl is not None:
                        out["co2_floor"][i] = fl
                j0 = max(0, int((t - T.WIN / 2 - ct0) / (T.SLOW_N * 0.002)))
                out["co2_lvl"][i] = np.nanmean(
                    Cslow[:, j0 : j0 + max(1, int(T.WIN))], axis=1
                )
    return out, "ok"


def track_sums(d, idx, group, use):
    """Coherent amplitude (complex), coherence and frame count per channel over frames idx."""
    c, g, ref = d[f"{group}_c"][idx], d[f"{group}_g"][idx], d["ref_amp"][idx][:, None]
    ok = use & np.isfinite(c) & np.isfinite(g) & (g > 0) & np.isfinite(ref)
    num = np.where(ok, c * K_SUB * ref, 0)
    pm = np.where(ok, K_SUB * ref**2, 0)
    with np.errstate(invalid="ignore", divide="ignore"):  # NaN where not ok, not used
        px = np.where(ok, np.abs(num) ** 2 / np.maximum(g * K_SUB * ref**2, 1e-300), 0)
    N = ok.sum(axis=0)
    snum, spm, spx = num.sum(axis=0), pm.sum(axis=0), px.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        coh = np.abs(snum) ** 2 / (spx * spm)
        amp = snum / np.sqrt(np.maximum(N, 1) * K_SUB * spm)
    return np.where(N > 0, amp, np.nan), np.where(N > 0, coh, np.nan), N


def track_features(d, cut=CUT):
    """dev's aggregate.one on one shot's frame features: one dict per track."""
    fam = d["family_id"]
    rows = []
    nF = len(fam)
    if nF == 0:
        return rows
    ealive = d.get("ece_alive", np.zeros(N_ECE, bool))
    calive = d.get("co2_alive", np.zeros(N_CO2, bool))
    cutoff = d["cutoff"]
    with np.errstate(invalid="ignore"):
        use_e = (
            ealive[None]
            & d["seen"]
            & (d["te"] > TE_MIN)
            & np.where(np.isfinite(cutoff), cutoff < cut, False)
        )
        use_e_nocut = ealive[None] & d["seen"] & (d["te"] > TE_MIN)
    use_m = np.broadcast_to(d["mir_alive"][None], (nF, N_MIR))
    use_c = np.broadcast_to(calive[None], (nF, N_CO2))
    for fid in np.unique(fam):
        idx = np.flatnonzero(fam == fid)
        r = {"family_id": int(fid), "nfr": len(idx)}
        for grp, use in (("mir", use_m), ("co2", use_c)):
            r[f"{grp}_c"], r[f"{grp}_g"], r[f"{grp}_n"] = track_sums(
                d, idx, grp, use[idx]
            )
        r["ece_c"], r["ece_g"], r["ece_n"] = track_sums(d, idx, "ece", use_e[idx])
        r["ece_c0"], r["ece_g0"], _ = track_sums(d, idx, "ece", use_e_nocut[idx])
        with T.quiet_all_nan():  # all-NaN medians
            for k in ("te", "rho", "R", "cutoff"):
                r[k] = np.nanmedian(d[k][idx], axis=0)
            r["side"] = np.sign(np.nanmedian(d["side"][idx], axis=0))
            r["use_frac"] = use_e[idx].mean(axis=0)
            r["ece_floor"] = np.nanmedian(d["ece_floor"][idx], axis=0)
            r["co2_floor"] = np.nanmedian(d["co2_floor"][idx], axis=0)
            r["co2_lvl"] = np.nanmedian(d["co2_lvl"][idx], axis=0)
            r["crash"] = np.nanmedian(d["crash"][idx])
            r["ref_amp"] = np.nanmedian(d["ref_amp"][idx])
        rows.append(r)
    return rows


def stack(rows: list[dict]) -> dict:
    """Track dicts -> arrays, one row per track."""
    keys = [k for k in rows[0] if k not in ("shot", "family_id")]
    return {k: np.stack([np.asarray(r[k]) for r in rows]) for k in keys}


def lobes(v, rho, i, sgn, inner):
    if inner:
        sel = (rho <= rho[i]) & (rho >= rho[i] - 0.2)
    else:
        sel = (rho >= rho[i]) & (rho <= rho[i] + 0.2)
    w = v[sel] * sgn
    return w.max() if len(w) else 0.0


def jump_side(c, g, rho, ok):
    s = ok & np.isfinite(c) & (g >= THR / 3)
    if s.sum() < 4:
        return np.nan, 0.0
    o = np.flatnonzero(s)[np.argsort(rho[s])]
    cc, rr, gg = c[o], rho[o], g[o]
    strong = gg >= THR
    if strong.sum() < 2:
        return np.nan, 0.0
    phi0 = 0.5 * np.angle((cc[strong] ** 2).sum())
    v = (cc * np.exp(-1j * phi0)).real
    vmax = np.abs(v).max()
    best, where = 0.0, np.nan
    for i in range(len(v) - 1):
        if np.sign(v[i]) == np.sign(v[i + 1]) or v[i] == 0:
            continue
        a = lobes(v, rr, i, np.sign(v[i]), True)
        b = lobes(v, rr, i + 1, np.sign(v[i + 1]), False)
        sc = min(a, b) / vmax
        if sc > best:
            best = sc
            where = rr[i] + (rr[i + 1] - rr[i]) * abs(v[i]) / (
                abs(v[i]) + abs(v[i + 1])
            )
    return (where, best) if best >= 0.2 else (np.nan, best)


def locate(c, g, te, rho, R, side):
    """L_amp and L_jump (the only locators the method uses) and the coherent channel count."""
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(c) & np.isfinite(rho) & (rho < 0.95)
        sig = ok & (g >= THR)
    out = {"L_amp": np.nan, "L_jump": np.nan, "nsig": int(sig.sum())}
    if sig.sum() < 2:
        return out
    a = np.abs(c)
    out["L_amp"] = rho[sig][np.argmax(a[sig])]
    for sd in (1, -1):
        with np.errstate(invalid="ignore"):
            j, _ = jump_side(c, g, rho, ok & (side == sd))
        if np.isfinite(j):
            out["L_jump"] = j
            break
    return out


def across_axis(c, g, rho, side):
    """Track-level ECE parity: weighted cos of the HFS-LFS phase at matched rho (< 0.04 apart)."""
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(c) & (g >= THR) & np.isfinite(rho) & (rho < 0.9)
    h = np.flatnonzero(ok & (side < 0))
    l_ = np.flatnonzero(ok & (side > 0))
    if not len(h) or not len(l_):
        return np.nan, 0
    w, cs = [], []
    for i in h:
        j = l_[np.argmin(np.abs(rho[l_] - rho[i]))]
        if abs(rho[j] - rho[i]) < 0.04:
            w.append(abs(c[i]) * abs(c[j]))
            cs.append(np.cos(np.angle(c[i] * np.conj(c[j]))))
    if not w:
        return np.nan, 0
    return float(np.average(cs, weights=w)), len(w)


def locators(F: dict) -> pd.DataFrame:
    """L_jump, L_amp and eceP_new per track row of F."""
    rows = []
    for i in range(len(F["ece_c"])):
        r = locate(
            F["ece_c"][i],
            F["ece_g"][i],
            F["te"][i],
            F["rho"][i],
            F["R"][i],
            F["side"][i],
        )
        r["eceP_new"], r["ece_npair"] = across_axis(
            F["ece_c"][i], F["ece_g"][i], F["rho"][i], F["side"][i]
        )
        rows.append(r)
    return pd.DataFrame(rows)
