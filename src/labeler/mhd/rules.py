"""Gates, harmonic lock, the track table and rules 1-4 (port of dev/mhd/corpus/analyze.py and
census.py).

- `prepare`: the frame gates. Frames farther than 30 ms from an EFIT02 slice lose their
  equilibrium and ECE columns; the wall pair passes with SNR >= 3 on both and coherence >= 0.6.
- `harmonic_lock`: an n = k line at k f of an n = 1 line, phase-locked (>= 0.7 over the five
  sub-windows), is that line's harmonic.
- `track_table`: one row per track (median frequency, wall parity, harmonic, EFIT02 q_min and
  q95, the ECE peak radius).
- `shortlist` rules 1-4: fixed-frequency lines leave as "line" (>= 10 frames, frequency IQR
  <= 0.3% of the median); a kept harmonic is "harmonic"; the candidates are the m/n with
  q_min - 0.2 < m/n < q95 + 0.2 and the wall parity (-1)^m; out-of-scope m/n are "other",
  dropped where an in-scope class of the same parity remains. The method uses rules 1-4 only
  (dev's `loc_rule=False`, `below=False`): `anchor4` is the single surviving class, `list4`
  the set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SNR, COH = 3.0, 0.6
EQ_MAX_DT = 30.0  # ms; frames farther from an EFIT02 slice get no equilibrium
LOCK = 0.7
INSCOPE = {
    "1/1": "rho_q1",
    "2/1": "rho_q2",
    "3/1": "rho_q3",
    "3/2": "rho_q32",
    "4/3": "rho_q43",
}
MARGIN, DELTA = 0.2, 0.25
LINE_FRAMES, LINE_IQR = 10, 0.003
SURFACE_COLUMNS = ("rho_q1", "rho_q43", "rho_q32", "rho_q2", "rho_q3", "rho_qmin")

#: Every column dev/mhd/corpus/tracks.py writes; a shot whose frames lack some (no wall pair,
#: no EFIT02, no ECE) gets them empty, as dev's concatenation over shots did.
FRAME_COLUMNS = (
    [
        "shot",
        "family_id",
        "t_ms",
        "f0",
        "n_frame",
        "n_score",
        "n_track",
        "peak_snr",
        "amp",
    ]
    + ["snr66", "snrab", "dphiab", "dphi1a", "dphi1b", "cohab"]
    + [f"sw{j}_{p}" for j in range(5) for p in ("re", "im")]
    + [
        "qmin",
        "q95",
        "eq_dt",
        "rho_qmin",
        "rho_q1",
        "rho_q43",
        "rho_q32",
        "rho_q2",
        "rho_q3",
    ]
    + ["ece_rho_seen", "ece_step", "ece_axis_ratio", "ece_te_max", "ece_valid"]
    + ["ece_pair_rho_max", "eceP", "ece_np", "ece_rho_pair"]
    + [f"eceP_b{b}" for b in range(4)]
    + ["ece_nsig", "ece_rho_peak", "ece_side_peak", "ece_snr_peak", "ece_rho_outer"]
    + ["ece_rho_wmean", "eceP_loc"]
)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """dev's analyze.load on frames already in memory."""
    df = df.reindex(
        columns=list(dict.fromkeys(FRAME_COLUMNS + list(df.columns)))
    ).copy()
    df["n"] = df.n_track.astype(int)
    far = df.eq_dt > EQ_MAX_DT
    eqcols = [
        c
        for c in df.columns
        if c.startswith(("qmin", "q95", "rho_q", "ece_", "eceP")) and c != "ece_valid"
    ]
    df.loc[far, eqcols] = np.nan
    df["ece_valid"] = df.ece_valid.astype(object).where(~far, False)
    df["cw"] = np.cos(df.dphiab.astype(float))
    df["wall_ok"] = (df.snrab >= SNR) & (df.snr66 >= SNR) & (df.cohab >= COH)
    df["ece_valid"] = df.ece_valid.fillna(False).astype(bool)
    df["ece_ok"] = df.ece_valid & (df.ece_np >= 2) & (df.eceP.abs() >= 0.3)
    return df


def harmonic_lock(df: pd.DataFrame) -> pd.DataFrame:
    """k*f and phase lock (5 sub-windows on MPI66M322) of n=k lines to an n=1 line; fund_id is
    that line's track, so a harmonic can be named after its fundamental."""
    sw = np.stack(
        [df[f"sw{j}_re"].values + 1j * df[f"sw{j}_im"].values for j in range(5)], axis=1
    ).astype(complex)
    ang = np.angle(sw)
    harm_k = np.zeros(len(df), int)
    harm_lock = np.full(len(df), np.nan)
    fund_cw = np.full(len(df), np.nan)
    fund_id = np.full(len(df), -1)
    nn, ff = df.n.values, df.f0.values
    for g in df.groupby(["shot", "t_ms"]).indices.values():
        fund = g[nn[g] == 1]
        if len(fund) == 0:
            continue
        for i in g[nn[g] >= 2]:
            k = nn[i]
            ok = np.abs(ff[fund] * k - ff[i]) <= max(0.3, 0.02 * ff[i])
            best, bj = 0.0, None
            for j in fund[ok]:
                if not (np.isfinite(ang[i]).all() and np.isfinite(ang[j]).all()):
                    continue
                lock = float(np.abs(np.exp(1j * (ang[i] - k * ang[j])).mean()))
                if lock > best:
                    best, bj = lock, j
            if bj is not None:
                harm_lock[i] = best
                if best >= LOCK:
                    harm_k[i] = k
                    fund_cw[i] = df.cw.values[bj] if df.wall_ok.values[bj] else np.nan
                    fund_id[i] = df.family_id.values[bj]
    df["harm_k"], df["harm_lock"] = harm_k, harm_lock
    df["fund_cw"], df["fund_id"] = fund_cw, fund_id
    return df


def track_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (s, fid), x in df.groupby(["shot", "family_id"]):
        w = x[x.wall_ok]
        par = None
        if len(w) >= 3:
            m = np.average(w.cw, weights=w.cohab)
            if abs(m) >= 0.3:
                par = "even" if m > 0 else "odd"
        hk = x.harm_k.values
        locked = (hk > 0).mean() >= 0.5
        harm = False
        fid0 = -1
        if locked:
            k = int(np.median(hk[hk > 0]))
            fid0 = int(x[x.harm_k > 0].fund_id.mode().iloc[0])
            fund = x[x.harm_k > 0].fund_cw.dropna()
            if k == 2:
                harm = par != "odd"  # a true 2f harmonic is even
            elif k == 3:
                fp = ("even" if fund.mean() > 0 else "odd") if len(fund) else None
                harm = par is None or fp is None or par == fp
        e = x[
            x.ece_valid
            & (x.ece_snr_peak >= 5)
            & (x.ece_nsig >= 2)
            & x.ece_rho_peak.notna()
        ]
        e_seen = x[x.ece_valid].ece_rho_seen
        ece = x[x.ece_ok]
        ece_par = None
        if len(ece) >= 3 and abs(ece.eceP.mean()) >= 0.3:
            ece_par = "even" if ece.eceP.mean() > 0 else "odd"
        rows.append(
            dict(
                shot=s,
                family_id=fid,
                n=int(x.n.iloc[0]),
                frames=len(x),
                t0=x.t_ms.min(),
                t1=x.t_ms.max(),
                f=x.f0.median(),
                fiqr=x.f0.quantile(0.75) - x.f0.quantile(0.25),
                amp=x.amp.median(),
                par=par,
                ngated=len(w),
                locked=locked,
                harm=harm,
                fund_id=fid0,
                qmin=x.qmin.median(),
                q95=x.q95.median(),
                loc=e.ece_rho_peak.median() if len(e) >= 3 else np.nan,
                loc_w=e.ece_rho_wmean.median() if len(e) >= 3 else np.nan,
                nloc=len(e),
                outer=(
                    e.ece_rho_outer.median()
                    if e.ece_rho_outer.notna().sum() >= 3
                    else np.nan
                ),
                seen=e_seen.median() if len(e_seen) else np.nan,
                ece_par=ece_par,
                ece_pair_rho_max=x.ece_pair_rho_max.median(),
                coh=x.n_score.median(),
                **{c: x[c].median() for c in SURFACE_COLUMNS},
            )
        )
    return pd.DataFrame(rows)


def is_line(r) -> bool:
    """A fixed-frequency line: pickup, no rotation drift."""
    return r.frames >= LINE_FRAMES and r.fiqr <= LINE_IQR * r.f


def shortlist(r, d=MARGIN):
    """Rules 1-4 (dev's census.shortlist with loc_rule=False, below=False)."""
    if is_line(r):
        return ("line",)
    if r.harm:
        return ("harmonic",)
    if not (np.isfinite(r.qmin) and np.isfinite(r.q95)):
        return ("none",)
    par = r.par
    cands = []
    for m in range(1, 13):
        qs = m / r.n
        if r.qmin > qs + d or r.q95 < qs - d:
            continue
        if isinstance(par, str) and (m % 2 == 0) != (par == "even"):
            continue
        name = f"{m}/{r.n}"
        inscope = name in INSCOPE and np.gcd(m, r.n) == 1
        cands.append((name if inscope else "other", m % 2))
    inpar = {p for c, p in cands if c != "other"}
    S = tuple(sorted({c for c, p in cands if c != "other" or p not in inpar}))
    return S if S else ("none",)


def census(T: pd.DataFrame) -> pd.DataFrame:
    """`anchor4` (the one class rules 1-4 leave, else "") and `list4` per track."""
    S = [shortlist(r) for r in T.itertuples()]
    T = T.copy()
    T["anchor4"] = np.array([s[0] if len(s) == 1 else "" for s in S], dtype=object)
    T["list4"] = ["|".join(s) for s in S]
    return T


def harmonic_name(n, fund):
    """A harmonic at toroidal number n of a fundamental m0/n0 is k m0 / k n0 (k = n / n0);
    None when n is no multiple of n0 or the fundamental has no m/n (dev/mhd/fusion/cases.py)."""
    if not isinstance(fund, str) or "/" not in fund:
        return None
    m0, n0 = (int(v) for v in fund.split("/"))
    k = n / n0
    return f"{round(k * m0)}/{n}" if k == round(k) else None
