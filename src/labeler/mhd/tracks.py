"""Mode tracks and per-frame measurements (port of dev/mhd/corpus/tracks.py).

- Peaks: every 4 ms, a 5.12 ms Hann window on the six midplane probes of the corpus
  `mirnov` group (decimated to 250 kHz), up to 12 local maxima of their mean amplitude
  between 1 and 100 kHz at least 3x the median of the neighbour bins.
- n: the fit at the refined peak frequency (pyspecview's way); a frame's n counts when its
  score is >= 0.85.
- Tracks: peaks linked frame to frame within max(0.5 kHz, 6%) and one missed frame, never
  across two different fitted n; kept with >= 5 frames. The track's n is the mode of its
  fitted |n| (>= 3 fitted frames, >= 60% agreement), and only n = 1, 2, 3 are kept.
- Per frame: the wall-parity proxy MPI1A322+MPI1B322 against MPI66M322, five sub-windows on
  MPI66M322 for the harmonic lock, ECE across the axis and the ECE peak radius (corpus
  TECEF, channels 1-40), and EFIT02 (q_min, q95, the radius of each rational surface).

Three changes from dev, all coverage only: the grid runs over the labeler's window, not the
EFIT02 flat-top; a shot without EFIT02 or without ECE still gets its Mirnov tracks (the n
events need only those), with its equilibrium and ECE columns left empty; and a record that
is NaN over part of the window (every row together, as some corpus Mirnov records are before
or after their digitizer's time) is used where it is finite (`read`): a row is judged alive
on the time the record covers, the decimated samples within the filter's reach of a gap are
NaN, and a frame whose window touches them is skipped (`seg`). On a record with no NaN in
the window all three give dev's numbers.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager

import h5py
import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from ..config import Paths
from .equilibrium import N_ECE, SURFACES, Equilibrium

#: numpy's RuntimeWarnings for a NaN reduction over an all-NaN (or empty) slice, whose NaN
#: is the intended result (`quiet_all_nan`); every other RuntimeWarning is shown.
ALL_NAN = (
    r"Mean of empty slice|All-NaN (slice|axis) encountered"
    r"|Degrees of freedom <= 0 for slice"
)


@contextmanager
def quiet_all_nan():
    """Silence numpy's all-NaN RuntimeWarnings (`ALL_NAN`) only."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=ALL_NAN, category=RuntimeWarning)
        yield


#: corpus `mirnov` rows -> toroidal angle (deg), the labeler's table (pyspecview's R0 probes)
TOR = {15: 317.4, 16: 132.5, 18: 312.4, 20: 19.5, 21: 97.4, 22: 307.0}
ROW_1A, ROW_1B, ROW_66 = 0, 8, 15
ROWS = sorted({ROW_1A, ROW_1B, *TOR})
N_VALUES = np.arange(-4, 6)
DEC = 2  # 500 kHz -> 250 kHz
DT = 0.004  # ms after decimation
WIN, SUB, STEP = 5.12, 2.56, 4.0
SUB_OFF = np.array([-2.56, -1.28, 0.0, 1.28, 2.56])
FLOOR_K = np.r_[-9:-2, 3:10]  # neighbour bins (1/WIN) for the local floor
BAND = (1.0, 100.0)  # kHz
PEAK_SNR, MAX_PEAKS, MIN_SEP = 3.0, 12, 3
N_SCORE = 0.85
LINK_KHZ, LINK_FRAC, GAP = 0.5, 0.06, 2
MIN_FRAMES = 5
MIN_PROBES = 4
SLOW_N = 500  # 1 ms at 500 kHz, for the Te profile
TE_MIN, STEP_MAX, AXIS_MIN = 0.05, 2.0, 0.6  # keV; ECE validity inside rho 0.8
PAIR_BINS = (0.0, 0.2, 0.4, 0.6, 0.9)
#: The grid stays this far inside the window (dev: inside the flat-top).
EDGE_MS = 10.0
#: Mirnov is read this far beyond the grid.
READ_PAD_MS = 20.0
#: A row is alive when it is finite over more than this fraction of the time the record
#: covers (the samples where any row read is finite) and not flat (dev: of the whole read).
ALIVE_FRAC = 0.99
#: resample_poly's filter half-length in input samples at DEC (scipy: 10 x max(up, down)).
FIR_HALF = 10 * DEC


def corpus_path(paths: Paths, shot: int):
    return paths.corpus / f"{int(shot)}_processed.h5"


def kern(n, f_khz):
    tt = (np.arange(n) - (n - 1) / 2) * DT
    w = np.hanning(n)
    return (
        np.exp(1j * 2 * np.pi * np.outer(tt, np.atleast_1d(f_khz)))
        * w[:, None]
        * (2.0 / w.sum())
    )


def seg(X, t0, tc, win):
    """The `win` ms of X centred on tc, or None near the ends or over a gap (NaN)."""
    n = int(round(win / DT))  # noqa: RUF046
    i0 = int(round((tc - t0) / DT - (n - 1) / 2))  # noqa: RUF046
    if i0 < 0 or i0 + n > X.shape[1]:
        return None
    s = X[:, i0 : i0 + n]
    return s if np.isfinite(s).all() else None


def coh(a, b):
    den = (np.abs(a) ** 2).sum() * (np.abs(b) ** 2).sum()
    return float(np.abs((a * b.conj()).sum()) ** 2 / den) if den > 0 else np.nan


def read(f, group, rows, w0, w1, slow=False):
    """Rows of a corpus group over [w0, w1] ms, decimated by DEC; (t0_ms, X, alive, S),
    S the 1 ms means before the mean is removed (slow=True), else None. None when the
    record holds fewer than 10 x DEC finite samples there.

    A row is alive on the time the record covers (`ALIVE_FRAC`). Where the record has no
    finite sample, or a live row is not finite, X is NaN in every row over the decimated
    samples the filter reaches, so `seg` skips any window over them."""
    x = f[f"{group}/xdata"]
    n = x.shape[0]
    xa, xb = float(x[0]) * 1e3, float(x[-1]) * 1e3
    dt = (xb - xa) / (n - 1)
    i0 = max(0, int((w0 - xa) / dt))
    i1 = min(n, int((w1 - xa) / dt) + 1)
    if i1 - i0 < 10 * DEC:
        return None
    Y = f[f"{group}/ydata"][rows, i0:i1].astype(np.float64)
    fin = np.isfinite(Y)
    covered = fin.any(axis=0)
    if covered.sum() < 10 * DEC:
        return None
    with quiet_all_nan():  # an all-NaN row or 1 ms block is NaN, silently
        alive = (fin[:, covered].mean(axis=1) > ALIVE_FRAC) & (
            np.nanstd(np.where(fin, Y, np.nan), axis=1) > 0
        )
        S = None
        if slow:
            nb = Y.shape[1] // SLOW_N
            S = np.nanmean(
                np.where(fin, Y, np.nan)[:, : nb * SLOW_N].reshape(
                    len(rows), nb, SLOW_N
                ),
                axis=2,
            )
    Y = np.where(fin, Y, 0.0)
    mean = Y.mean(axis=1)
    part = fin.any(axis=1) & ~fin.all(axis=1)
    mean[part] = Y[part].sum(axis=1) / fin[part].sum(axis=1)  # the finite samples' mean
    Y -= mean[:, None]
    Y[~fin] = 0.0
    X = resample_poly(Y, 1, DEC, axis=1)
    gap = ~covered | (~fin[alive]).any(axis=0)
    if gap.any():
        reach = np.convolve(gap.astype(float), np.ones(2 * FIR_HALF + 1), "same") > 0
        X[:, reach[::DEC][: X.shape[1]]] = np.nan
    return xa + i0 * dt, X, alive, S


def coverage(t0, X, window):
    """The finite stretch of X inside `window` (ms): `([first, last], [[a, b], ...] gaps
    inside it)`, or `(None, [])` when X has no finite sample there."""
    t = t0 + np.arange(X.shape[1]) * DT
    ok = np.isfinite(X).all(axis=0) & (t >= window[0]) & (t <= window[1])
    if not ok.any():
        return None, []
    first, last = np.flatnonzero(ok)[[0, -1]]
    bad = np.r_[False, ~ok[first : last + 1], False]
    edges = np.flatnonzero(np.diff(bad.astype(int)))
    gaps = [
        [float(t[first + a - 1]), float(t[first + b])]
        for a, b in zip(edges[::2], edges[1::2])
    ]
    return [float(t[first]), float(t[last])], gaps


def peaks(M, t0, grid, tor_idx, phi):
    """Per frame: (frame index, f0 kHz, amplitude, snr, |n| or nan, n score)."""
    n = int(round(WIN / DT))  # noqa: RUF046
    f = np.fft.rfftfreq(n, DT)  # kHz
    band = np.flatnonzero((f >= BAND[0]) & (f <= BAND[1]))
    w = np.hanning(n)
    out = []
    for it, tc in enumerate(grid):
        s = seg(M, t0, tc, WIN)
        if s is None:
            continue
        A = np.sqrt((np.abs(np.fft.rfft(s[tor_idx] * w, axis=1)) ** 2).mean(axis=0))
        k = band
        nb = np.stack([A[np.clip(k + d, 0, len(A) - 1)] for d in FLOOR_K])
        snr = A[k] / np.median(nb, axis=0)
        loc = (
            (A[k] >= A[k - 1])
            & (A[k] >= A[np.minimum(k + 1, len(A) - 1)])
            & (snr >= PEAK_SNR)
        )
        cand = k[loc][np.argsort(-A[k][loc])]
        kept = []
        for c in cand:
            if all(abs(c - o) >= MIN_SEP for o in kept):
                kept.append(c)
            if len(kept) == MAX_PEAKS:
                break
        if not kept:
            continue
        la = np.log(A[np.array(kept)[:, None] + np.array([-1, 0, 1])] + 1e-30)
        den = la[:, 0] - 2 * la[:, 1] + la[:, 2]
        off = np.where(
            den < 0, 0.5 * (la[:, 0] - la[:, 2]) / np.where(den < 0, den, 1), 0.0
        )
        f0 = f[np.array(kept)] + np.clip(off, -0.5, 0.5) * (f[1] - f[0])
        Y = s[tor_idx] @ kern(n, f0)  # (P, npk)
        u = Y / np.maximum(np.abs(Y), 1e-30)
        sc = np.abs(np.exp(-1j * np.outer(N_VALUES, phi)) @ u) / len(phi)  # (nN, npk)
        best = sc.argmax(axis=0)
        for j, c in enumerate(kept):
            score = float(sc[best[j], j])
            nn = abs(int(N_VALUES[best[j]])) if score >= N_SCORE else np.nan
            out.append(
                (
                    it,
                    float(f0[j]),
                    float(A[c]),
                    float(snr[np.flatnonzero(k == c)[0]]),
                    nn,
                    score,
                )
            )
    return out


def link(pk):
    """Greedy frame-to-frame linking; returns a track id per peak (-1 = unlinked)."""
    tid = np.full(len(pk), -1)
    tracks = []  # [last frame, last f, n votes]
    by_frame = {}
    for i, p in enumerate(pk):
        by_frame.setdefault(p[0], []).append(i)
    for it in sorted(by_frame):
        used = set()
        for i in sorted(by_frame[it], key=lambda i: -pk[i][2]):
            f0, nn = pk[i][1], pk[i][4]
            best, bd = None, None
            for j, tr in enumerate(tracks):
                if j in used or it - tr[0] > GAP or it == tr[0]:
                    continue
                d = abs(f0 - tr[1])
                if d > max(LINK_KHZ, LINK_FRAC * f0):
                    continue
                votes = tr[2]
                if np.isfinite(nn) and votes and max(votes, key=votes.get) != nn:
                    continue
                if bd is None or d < bd:
                    best, bd = j, d
            if best is None:
                tracks.append([it, f0, {}])
                best = len(tracks) - 1
            tr = tracks[best]
            tr[0], tr[1] = it, f0
            if np.isfinite(nn):
                tr[2][nn] = tr[2].get(nn, 0) + 1
            tid[i] = best
            used.add(best)
    return tid


def track_n(ns):
    ns = ns[np.isfinite(ns)]
    if len(ns) < 3:
        return np.nan
    v, c = np.unique(ns, return_counts=True)
    return float(v[c.argmax()]) if c.max() / len(ns) >= 0.6 else np.nan


def _ece_columns(row, t, geo, E, et0, ealive, Eslow, fq):
    """The frame's ECE columns (dev's run_shot, unchanged)."""
    sE = seg(E, et0, t, WIN)
    if sE is None:
        return
    Z = sE @ kern(sE.shape[1], fq)
    za = np.where(ealive, np.abs(Z[:, 0]), 0.0)
    zs = za / np.maximum(np.median(np.abs(Z[:, 1:]), axis=1), 1e-30)
    rho = geo["rho"]
    row["ece_rho_seen"] = geo["rho_seen"]
    # validity: Te rises toward the axis without jumps (cutoff, harmonic overlap)
    j0 = max(0, int((t - WIN / 2 - et0) / (SLOW_N * 0.002)))
    with quiet_all_nan():  # an all-NaN 1 ms block is NaN, silently
        te = np.nanmean(Eslow[:, j0 : j0 + max(1, int(WIN))], axis=1)
    use = geo["seen"] & ealive & (te > TE_MIN) & (rho < 0.8)
    if use.sum() >= 5:
        o = np.flatnonzero(use)[np.argsort(geo["R"][use])]
        tt = te[o]
        step = float(np.max(np.maximum(tt[1:] / tt[:-1], tt[:-1] / tt[1:])))
        ax = float(te[o[np.argmin(rho[o])]] / tt.max())
        row.update(
            ece_step=step,
            ece_axis_ratio=ax,
            ece_te_max=float(tt.max()),
            ece_valid=bool(step <= STEP_MAX and ax >= AXIS_MIN),
        )
    else:
        row["ece_valid"] = False
    Pp = geo["pairs"]
    if len(Pp):
        row["ece_pair_rho_max"] = float(rho[Pp[:, 0]].max())
        good = (zs[Pp[:, 0]] >= 3) & (zs[Pp[:, 1]] >= 3)
        Pg = Pp[good]
        if len(Pg):
            w = za[Pg[:, 0]] * za[Pg[:, 1]]
            c = np.cos(np.angle(Z[Pg[:, 0], 0] * np.conj(Z[Pg[:, 1], 0])))
            rp = rho[Pg[:, 0]]
            row.update(
                eceP=float((w * c).sum() / w.sum()),
                ece_np=len(Pg),
                ece_rho_pair=float(np.average(rp, weights=w)),
            )
            for b in range(len(PAIR_BINS) - 1):
                sel = (rp >= PAIR_BINS[b]) & (rp < PAIR_BINS[b + 1])
                if sel.any():
                    row[f"eceP_b{b}"] = float((w[sel] * c[sel]).sum() / w[sel].sum())
    sig = (zs >= 3) & geo["seen"]
    row["ece_nsig"] = int(sig.sum())
    if sig.any():
        i = int(np.argmax(np.where(sig, za, 0)))
        strong = (zs >= 5) & geo["seen"]
        row.update(
            ece_rho_peak=float(rho[i]),
            ece_side_peak=int(geo["side"][i]),
            ece_snr_peak=float(zs[i]),
            ece_rho_outer=float(np.nanmax(rho[strong]))
            if strong.sum() >= 2
            else np.nan,
            ece_rho_wmean=float(np.average(rho[sig], weights=za[sig])),
        )
        if len(Pp) and len(Pg):
            sel = np.abs(rp - rho[i]) <= 0.1
            if sel.any():
                row["eceP_loc"] = float((w[sel] * c[sel]).sum() / w[sel].sum())


def frames(shot: int, paths: Paths, window, eq: Equilibrium | None):
    """The shot's track frames over `window` (ms): `(DataFrame or None, reason, info)`.

    reason "ok" (tracks found), "no peaks" or "no n=1-3 tracks" (the Mirnov was read and
    holds no mode), or why the Mirnov could not be read (no record, no `mirnov` group or
    no finite sample in the window, fewer than `MIN_PROBES` live toroidal probes).
    `info["mirnov_ms"]` is the stretch the Mirnov record covers inside the window (its
    first and last finite sample) and `info["mirnov_gaps"]` the non-finite stretches
    inside it (`coverage`).
    """
    shot = int(shot)
    w0, w1 = float(window[0]) + EDGE_MS, float(window[1]) - EDGE_MS
    info = {}
    path = corpus_path(paths, shot)
    if not path.is_file():
        return None, "no corpus record", info
    if w1 - w0 < 4 * STEP:
        return None, "window too short", info
    try:
        with h5py.File(path, "r") as f:
            if "mirnov" not in f:
                return None, "no mirnov", info
            rm = read(f, "mirnov", ROWS, w0 - READ_PAD_MS, w1 + READ_PAD_MS)
            re_ = None
            if eq is not None and eq.freq is not None and "ece" in f:
                re_ = read(
                    f,
                    "ece",
                    list(range(N_ECE)),
                    w0 - READ_PAD_MS,
                    w1 + READ_PAD_MS,
                    slow=True,
                )
    except (OSError, KeyError) as err:
        return None, f"read error {err}", info
    if rm is None:
        return None, "no mirnov", info
    mt0, M, malive, _ = rm
    cover, gaps = coverage(mt0, M, (float(window[0]), float(window[1])))
    if cover is None:
        return None, "no mirnov", info
    info["mirnov_ms"], info["mirnov_gaps"] = cover, gaps
    idx = {r: i for i, r in enumerate(ROWS)}
    tor = [r for r in TOR if malive[idx[r]]]
    info["probes"] = len(tor)
    if len(tor) < MIN_PROBES:
        return None, f"only {len(tor)} toroidal probes", info
    tor_idx = [idx[r] for r in tor]
    phi = np.deg2rad([TOR[r] for r in tor])
    wall = malive[idx[ROW_1A]] and malive[idx[ROW_1B]] and malive[idx[ROW_66]]
    info["wall_pair"] = bool(wall)
    grid = np.arange(w0, w1, STEP)
    pk = peaks(M, mt0, grid, tor_idx, phi)
    if not pk:
        return None, "no peaks", info
    tid = link(pk)
    P = pd.DataFrame(
        pk, columns=["it", "f0", "amp", "snr", "n_frame", "n_score"]
    ).assign(family_id=tid)
    size = P.groupby("family_id").size()
    P = P[P.family_id.map(size) >= MIN_FRAMES]
    tn = (
        P.groupby("family_id")
        .n_frame.apply(lambda s: track_n(s.values))
        .rename("n_track")
    )
    P = P.join(tn, on="family_id")
    P = P[P.n_track.isin([1, 2, 3])]
    if P.empty:
        return None, "no n=1-3 tracks", info

    E = None
    if re_ is not None:
        et0, E, ealive, Eslow = re_
    freqs_off = FLOOR_K / WIN
    i66, i1a, i1b = idx[ROW_66], idx[ROW_1A], idx[ROW_1B]
    rows = []
    for r in P.itertuples():
        t = float(grid[r.it])
        f0 = r.f0
        fq = np.r_[f0, f0 + freqs_off]
        row = {
            "shot": shot,
            "family_id": int(r.family_id),
            "t_ms": t,
            "f0": f0,
            "n_frame": r.n_frame,
            "n_score": r.n_score,
            "n_track": r.n_track,
            "peak_snr": r.snr,
            "amp": r.amp,
        }
        sM = seg(M, mt0, t, WIN)
        if wall and sM is not None:
            Y = sM[[i66, i1a, i1b]] @ kern(sM.shape[1], fq)
            fl = np.median(np.abs(Y[:, 1:]), axis=1)
            yab = Y[1] + Y[2]
            row.update(
                snr66=float(abs(Y[0, 0]) / fl[0]),
                snrab=float(abs(yab[0]) / np.median(np.abs(yab[1:]))),
                dphiab=float(np.angle(yab[0] * np.conj(Y[0, 0]))),
                dphi1a=float(np.angle(Y[1, 0] * np.conj(Y[0, 0]))),
                dphi1b=float(np.angle(Y[2, 0] * np.conj(Y[0, 0]))),
            )
            S = []
            for off in SUB_OFF:
                s = seg(M, mt0, t + off, SUB)
                if s is None:
                    break
                S.append(s[[i66, i1a, i1b]] @ kern(s.shape[1], f0)[:, 0])
            if len(S) == len(SUB_OFF):
                S = np.array(S)
                row["cohab"] = coh(S[:, 1] + S[:, 2], S[:, 0])
                for j in range(len(SUB_OFF)):
                    row[f"sw{j}_re"], row[f"sw{j}_im"] = S[j, 0].real, S[j, 0].imag
        if eq is not None:
            geo = eq.slice(t)
            row.update(
                qmin=geo["qmin"],
                q95=geo["q95"],
                eq_dt=geo["dt"],
                rho_qmin=geo["rho_qmin"],
                **{f"rho_{n}": geo[f"rho_{n}"] for n, _ in SURFACES},
            )
            if E is not None:
                _ece_columns(row, t, geo, E, et0, ealive, Eslow, fq)
        rows.append(row)
    return pd.DataFrame(rows), "ok", info
