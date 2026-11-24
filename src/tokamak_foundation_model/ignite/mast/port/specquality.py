"""TokEye-informed quality scores for the two MAST Mirnov spectrograms.

What this is for: deciding which spectrogram data a codec is fitted on and which a dynamics
model is trained on. dev-peter's advice for the spectrogram tokenizer was to find shots with
enough mode content and to inspect the training data carefully; his own selection used the
spread of the time-averaged spectrum (``pick_mode_rich_spectro``), the lag-1 autocorrelation
along time (``ac1``) and the share of a band's variance that survives a 5-frame boxcar
(``coh_frac``). This module adds TokEye's segmenter (the vendored ``big_tf_unet`` of
``labeler.events.unet``, sha-pinned) on top, so that "is there a mode here" is answered by a
model trained to see modes, not only by variance.

**The input is TokEye's transform, applied to what the cache holds.** Cache v1 stores
TokaMark's spectrogram: non-overlapping 512-sample Hann windows of the 500 kHz signal,
``|FFT|^2``, 256 bins (0.977 kHz/bin), one column every 1.024 ms. TokEye's own transform is
``log1p|STFT|`` (Hann 1024, hop 128), DC dropped, clipped to the 1st/99th percentile, then
z-scored. Here: log magnitude (``0.5 log P``; for magnitudes well above one ``log1p`` is
``log``, and the clip and z-score remove scale and offset), DC dropped, the frequency axis
interpolated x2 so a row is 0.488 kHz as in TokEye's wide pass, then the same clip and
z-score over the valid columns only. Time is left at 1.024 ms, which is TokEye's zoom-pass
column width. Checked by eye on shots 24934, 23403 and 20384 (2026-11-24): the coherent mask
follows the chirping modes on OMV, and x4 in time as well gave sharper but more fragmented
tracks at four times the cost.

**Pickup lines are not modes.** TokEye lights any line with a track, including the
stationary instrument pickup lines that MAST's Mirnov spectrograms carry before, during and
after the plasma (shot 23403's CC probe: lines at 20, 40 and 80 kHz from 0 to 0.47 s, while
OMV shows the chirping modes). A row is pickup if it is lit for at least
``PICKUP_OFF_FRAC`` of the valid columns outside the plasma window (when there are at least
``MIN_OFF_COLUMNS`` of them), or for at least ``PICKUP_STATIONARY_FRAC`` of the plasma
window's valid columns. Pickup rows are dilated by ``PICKUP_DILATE`` rows and excluded from
the coherent activity.

Nothing here decides keep or drop on its own: `score_shot` stores the measurements, and
`verdict` turns them into a decision with thresholds that are stated beside it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_dilation, zoom

from .. import tasks

#: Frequency interpolation factor: 0.977 -> 0.488 kHz per row, TokEye's wide-pass spacing.
F_UP: int = 2
#: TokEye's percentile clip (``tokeye.transforms.compute_stft``).
CLIP_LOW, CLIP_HIGH = 1.0, 99.0
#: A sigmoid output at or above this is "lit" (``labeler.ae.labels.PROB_THRESHOLD``).
LIT: float = 0.5
#: A column is valid when at least this share of its bins are finite and positive.
VALID_BIN_FRAC: float = 0.9
#: Pickup: lit for this share of the valid non-plasma columns ...
PICKUP_OFF_FRAC: float = 0.3
#: ... provided there are at least this many of them (~20 ms) ...
MIN_OFF_COLUMNS: int = 20
#: ... or lit for this share of the plasma window's valid columns.
PICKUP_STATIONARY_FRAC: float = 0.6
#: Rows of dilation around a pickup row (x2 rows, so +-1 original bin).
PICKUP_DILATE: int = 2
#: dev-peter's boxcar for coh_frac, in spectrogram columns.
SMOOTH: int = 5

SIGNALS: tuple[str, ...] = tuple(tasks.SPECTROGRAM_SIGNALS)
#: Short names used in stored keys.
SHORT: dict[str, str] = {s: s.split("_")[-2] for s in SIGNALS}   # "cc", "omv"


def valid_columns(power: np.ndarray) -> np.ndarray:
    """``(n_bins, T)`` power -> ``(T,)`` bool: at least `VALID_BIN_FRAC` finite positive bins."""
    p = np.asarray(power, dtype=np.float64)
    ok = np.isfinite(p) & (p > 0.0)
    return ok.mean(axis=0) >= VALID_BIN_FRAC


def tokeye_input(power: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """``(n_bins, T)`` power -> ``(F_UP * (n_bins - 1), T)`` float32 TokEye input.

    Invalid columns are set to the valid mean (0 after the z-score), so a missing stretch is
    flat, not a cliff the segmenter could call a transient.
    """
    p = np.asarray(power, dtype=np.float64)
    if not valid.any():
        return np.zeros((F_UP * (p.shape[0] - 1), p.shape[1]), dtype=np.float32)
    pos = p[:, valid]
    pos = pos[np.isfinite(pos) & (pos > 0.0)]
    floor = float(np.percentile(pos, 0.1)) if pos.size else 1e-30
    x = 0.5 * np.log(np.maximum(np.where(np.isfinite(p), p, floor), floor))[1:]
    if F_UP > 1:
        x = zoom(x, (F_UP, 1), order=1)
    xv = x[:, valid]
    lo, hi = np.percentile(xv, [CLIP_LOW, CLIP_HIGH])
    x = np.clip(x, lo, hi)
    xv = x[:, valid]
    x = (x - xv.mean()) / (xv.std() + 1e-6)
    x[:, ~valid] = 0.0
    return x.astype(np.float32)


def segment(model: torch.nn.Module, x: np.ndarray, device: str = "cpu") -> np.ndarray:
    """``(H, W)`` input -> ``(2, H, W)`` float32 sigmoid masks (coherent, transient).

    The U-Net is fully convolutional; the input is reflect-padded to a multiple of 16 (four
    max-pools) and the pad cropped off.
    """
    H, W = x.shape
    Hp, Wp = -(-H // 16) * 16, -(-W // 16) * 16
    xp = np.pad(x, ((0, Hp - H), (0, Wp - W)), mode="reflect")
    with torch.inference_mode():
        logits = model(torch.from_numpy(xp)[None, None].to(device))[0]
        pr = torch.sigmoid(logits)[0].float().cpu().numpy()
    return pr[:, :H, :W]


def pickup_rows(coh_lit: np.ndarray, valid: np.ndarray, in_plasma: np.ndarray) -> np.ndarray:
    """``(H,)`` bool: rows lit like a stationary line (see the module docstring)."""
    H = coh_lit.shape[0]
    rows = np.zeros(H, dtype=bool)
    off = valid & ~in_plasma
    on = valid & in_plasma
    if off.sum() >= MIN_OFF_COLUMNS:
        rows |= coh_lit[:, off].mean(axis=1) >= PICKUP_OFF_FRAC
    if on.any():
        rows |= coh_lit[:, on].mean(axis=1) >= PICKUP_STATIONARY_FRAC
    if PICKUP_DILATE > 0 and rows.any():
        rows = binary_dilation(rows, iterations=PICKUP_DILATE)
    return rows


def band_ac1_coh(logp: np.ndarray, valid: np.ndarray, smooth: int = SMOOTH) -> tuple[float, float]:
    """dev-peter's ``ac1`` and ``coh_frac`` over the longest valid run of a ``(F, T)`` log power.

    Mean-removed per row; ``ac1 = sum(x_t x_{t+1}) / sum(x_t^2)``; ``coh_frac`` = mean square of
    the ``smooth``-column boxcar over the mean square (only consecutive valid columns are used,
    as in ``spectro_final_fig.band_stats``).
    """
    idx = np.flatnonzero(valid)
    if idx.size < smooth + 2:
        return float("nan"), float("nan")
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks + 1, idx.size]
    j = int(np.argmax(ends - starts))
    cols = idx[starts[j]:ends[j]]
    if cols.size < smooth + 2:
        return float("nan"), float("nan")
    x = logp[:, cols].astype(np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    den = float((x * x).sum())
    if den <= 0:
        return float("nan"), float("nan")
    ac1 = float((x[:, :-1] * x[:, 1:]).sum()) / den
    k = np.ones(smooth) / smooth
    xs = np.apply_along_axis(lambda r: np.convolve(r, k, mode="valid"), 1, x)
    coh = float((xs * xs).mean()) / float((x * x).mean())
    return ac1, coh


#: Frequency bands for the per-band statistics: 255 bins (DC dropped) in 8 bands of ~31 kHz.
N_BANDS: int = 8


def band_ac1_coh_per_band(logp: np.ndarray, valid: np.ndarray, n_bands: int = N_BANDS,
                          smooth: int = SMOOTH) -> tuple[np.ndarray, np.ndarray]:
    """`band_ac1_coh` in each of ``n_bands`` equal frequency bands -> ``(ac1, coh5)``, each ``(n_bands,)``.

    The whole-band numbers are dominated by the many bins at the noise floor: a MAST OMV probe
    whose plasma signal sits below ~30 kHz reads ac1 ~0.1 over all 255 bins (shots 13780-16737,
    contact sheet 2026-11-24), while its lowest band is clearly coherent. A probe is noise only
    if no band is.
    """
    edges = np.linspace(0, logp.shape[0], n_bands + 1).round().astype(int)
    a = np.full(n_bands, np.nan)
    c = np.full(n_bands, np.nan)
    for b in range(n_bands):
        a[b], c[b] = band_ac1_coh(logp[edges[b]:edges[b + 1]], valid, smooth)
    return a, c


@dataclass
class ProbeScore:
    """One probe of one spectrogram signal of one shot, on the spectrogram's own columns."""

    valid: np.ndarray        # (T,) bool
    coh: np.ndarray          # (T,) float32: share of non-pickup rows lit coherent
    tra: np.ndarray          # (T,) float32: share of rows lit transient
    pickup: np.ndarray       # (H,) bool
    pickup_frac: float       # lit coherent pixels in pickup rows / all lit (plasma window)
    contrast_db: float       # 10 log10 median plasma power / median off-plasma power (NaN if none)
    ac1: float               # plasma window, log10 power, all bins
    coh5: float              # dev-peter's coh_frac, plasma window
    band_ac1: np.ndarray     # (N_BANDS,) the same per frequency band
    band_coh5: np.ndarray    # (N_BANDS,)


def score_probe(model, power: np.ndarray, in_plasma: np.ndarray, device: str = "cpu") -> ProbeScore:
    """Score one ``(n_bins, T)`` power spectrogram; ``in_plasma`` is ``(T,)`` bool."""
    valid = valid_columns(power)
    x = tokeye_input(power, valid)
    pr = segment(model, x, device)
    coh_lit = pr[0] >= LIT
    tra_lit = pr[1] >= LIT
    rows = pickup_rows(coh_lit, valid, in_plasma)
    keep_rows = ~rows
    n_keep = max(int(keep_rows.sum()), 1)
    coh = coh_lit[keep_rows].sum(axis=0) / n_keep
    tra = tra_lit.mean(axis=0)
    coh[~valid] = np.nan
    tra[~valid] = np.nan
    on = valid & in_plasma
    lit_on = coh_lit[:, on]
    n_lit = int(lit_on.sum())
    pickup_frac = float(lit_on[rows].sum() / n_lit) if n_lit else float("nan")
    p = np.asarray(power, dtype=np.float64)
    band = np.where(np.isfinite(p[1:]) & (p[1:] > 0), p[1:], np.nan)
    col_pow = np.full(p.shape[1], np.nan)
    if valid.any():                      # an invalid column may hold no positive bin at all
        col_pow[valid] = np.nanmedian(band[:, valid], axis=0)
    off = valid & ~in_plasma
    if on.any() and off.sum() >= MIN_OFF_COLUMNS:
        contrast = float(10.0 * np.log10(np.nanmedian(col_pow[on]) / np.nanmedian(col_pow[off])))
    else:
        contrast = float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.log10(np.maximum(np.where(np.isfinite(p), p, 0.0), 1e-30))[1:]
    ac1, coh5 = band_ac1_coh(logp, on)
    b_ac1, b_coh5 = band_ac1_coh_per_band(logp, on)
    return ProbeScore(valid, coh.astype(np.float32), tra.astype(np.float32), rows, pickup_frac,
                      contrast, ac1, coh5, b_ac1, b_coh5)


def column_frames(n_cols: int, sig_t0: float, sig_dt: float, frame_t0: float, frame_dt: float) -> np.ndarray:
    """Frame index (relative to ``frame_t0``) of each spectrogram column, by column centre."""
    tc = float(sig_t0) + (np.arange(n_cols) + 0.5) * float(sig_dt)
    return np.floor((tc - float(frame_t0)) / float(frame_dt)).astype(np.int64)


def to_frames(per_col: np.ndarray, fidx: np.ndarray, n_frames: int) -> np.ndarray:
    """Mean of finite per-column values in each frame ``0..n_frames-1``; NaN where none."""
    out = np.full(n_frames, np.nan, dtype=np.float32)
    keep = (fidx >= 0) & (fidx < n_frames) & np.isfinite(per_col)
    if keep.any():
        s = np.bincount(fidx[keep], weights=per_col[keep], minlength=n_frames)
        c = np.bincount(fidx[keep], minlength=n_frames)
        np.divide(s, c, out=out, where=c > 0, casting="unsafe")
    return out


def score_shot(
    model,
    signals: Mapping[str, tuple[np.ndarray, float, float]],
    frame_t0: float,
    n_frames: int,
    frame_dt: float,
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """Score every probe of every spectrogram signal of a shot.

    ``signals`` maps a signal name to ``(power (P, n_bins, T), sig_t0, sig_dt)``. Returns a
    flat dict of arrays (``np.savez``-ready): per signal ``<s>_coh``, ``<s>_tra``,
    ``<s>_valid`` as ``(P, n_frames)`` frame means, per-probe scalars ``<s>_pickup_frac``,
    ``<s>_contrast_db``, ``<s>_ac1``, ``<s>_coh5``, ``<s>_n_pickup_rows`` as ``(P,)``, and
    ``<s>_band_ac1``, ``<s>_band_coh5`` as ``(P, N_BANDS)``.
    """
    out: dict[str, np.ndarray] = {"n_frames": np.int64(n_frames), "frame_t0": np.float64(frame_t0)}
    for sig, (power, sig_t0, sig_dt) in signals.items():
        s = SHORT[sig]
        P, _, T = power.shape
        fidx = column_frames(T, sig_t0, sig_dt, frame_t0, frame_dt)
        in_plasma = (fidx >= 0) & (fidx < n_frames)
        rows = {k: [] for k in ("coh", "tra", "valid", "pickup_frac", "contrast_db", "ac1", "coh5",
                                "n_pickup_rows", "band_ac1", "band_coh5")}
        for p in range(P):
            ps = score_probe(model, power[p], in_plasma, device)
            rows["coh"].append(to_frames(ps.coh, fidx, n_frames))
            rows["tra"].append(to_frames(ps.tra, fidx, n_frames))
            rows["valid"].append(to_frames(ps.valid.astype(np.float32), fidx, n_frames))
            rows["pickup_frac"].append(ps.pickup_frac)
            rows["contrast_db"].append(ps.contrast_db)
            rows["ac1"].append(ps.ac1)
            rows["coh5"].append(ps.coh5)
            rows["n_pickup_rows"].append(int(ps.pickup.sum()))
            rows["band_ac1"].append(ps.band_ac1)
            rows["band_coh5"].append(ps.band_coh5)
        for k, v in rows.items():
            dtype = np.int64 if k == "n_pickup_rows" else np.float32
            out[f"{s}_{k}"] = np.asarray(v, dtype=dtype)
    return out


#: Tolerance of `mode_agreement`, in TokEye pixels (rows of 0.488 kHz, columns of 1.024 ms):
#: a lit pixel counts as matched if the other mask is lit within this many pixels.
AGREE_TOL: int = 1


def mode_agreement(model, truth: np.ndarray, other: np.ndarray, device: str = "cpu",
                   tol: int = AGREE_TOL) -> dict[str, float]:
    """TokEye coherent-mask agreement of ``other`` (a decode or a rollout) with ``truth``.

    Both are ``(n_bins, T)`` power on the same columns. Each is put through `tokeye_input`
    and `segment` on its own; pixels are compared on the columns valid in both and on the
    rows that are not pickup in the truth (every column counts as plasma here). With
    ``tol`` pixels of slack: ``recall`` = share of the truth's lit pixels with a lit pixel of
    ``other`` nearby, ``precision`` the converse, ``f1`` their harmonic mean. ``truth_lit``
    and ``other_lit`` are the lit shares. NaN where the truth has no lit pixel (recall,
    f1) or ``other`` has none (precision).
    """
    vt, vo = valid_columns(truth), valid_columns(other)
    cols = vt & vo
    nan = float("nan")
    out = dict(recall=nan, precision=nan, f1=nan, truth_lit=nan, other_lit=nan, n_cols=int(cols.sum()))
    if cols.sum() < 4:
        return out
    lt = segment(model, tokeye_input(truth, vt), device)[0] >= LIT
    lo = segment(model, tokeye_input(other, vo), device)[0] >= LIT
    rows = ~pickup_rows(lt, vt, np.ones(lt.shape[1], dtype=bool))
    lt, lo = lt[rows][:, cols], lo[rows][:, cols]
    st = np.ones((2 * tol + 1, 2 * tol + 1), dtype=bool)
    lt_d = binary_dilation(lt, structure=st) if tol > 0 else lt
    lo_d = binary_dilation(lo, structure=st) if tol > 0 else lo
    nt, no = int(lt.sum()), int(lo.sum())
    rec = float((lt & lo_d).sum() / nt) if nt else nan
    pre = float((lo & lt_d).sum() / no) if no else nan
    f1 = 2 * rec * pre / (rec + pre) if nt and no and (rec + pre) > 0 else (0.0 if nt else nan)
    out.update(recall=rec, precision=pre, f1=float(f1), truth_lit=float(lt.mean()), other_lit=float(lo.mean()))
    return out


def load_scores(path: Path | str) -> dict[str, np.ndarray]:
    """Read one ``<shot>.npz`` written from `score_shot`."""
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


# --------------------------------------------------------------------------------------------
# Verdicts. The thresholds are set from the score distributions of all 11,177 shots and checked
# on contact sheets of each class (docs/models/ignite-mast-spectrogram-quality.md).
# --------------------------------------------------------------------------------------------

#: A probe with fewer valid plasma frames than this is "absent" (no data, or stored as zeros).
ABSENT_VALID: float = 0.2
#: A probe is "noise" when its log power is white in time in every frequency band: the
#: largest per-band lag-1 autocorrelation is below `NOISE_AC1` AND the largest per-band
#: coh_frac (dev-peter's, 5 columns) is below `NOISE_COH5`. White noise gives ac1 = 0 and
#: coh_frac = 1/5.
NOISE_AC1: float = 0.15
NOISE_COH5: float = 0.3
#: A frame is mode-active when at least this share of a good probe's non-pickup rows is lit
#: coherent (frame mean over the good probes).
MODE_COH: float = 0.02

#: TokEye rescue of a probe the statistics call noise: weak chirps on a white background
#: (MAST OMV, shots ~29000-30500) read ac1 ~0.1 in every band but TokEye lights them for part of
#: the shot. Rescued when the 90th percentile of the per-frame coherent activity is at least
#: `RESCUE_P90` and the median is at most `RESCUE_RATIO` of it (activity that comes and goes, as a
#: mode does). Only for the signals in `RESCUE_SIGNALS`: on CC the same rule mostly keeps probes
#: dominated by broadband interference and intermittent lines (contact sheets, 2026-11-24), so
#: TokEye's activity there is not evidence of plasma signal.
RESCUE_P90: float = 0.02
RESCUE_RATIO: float = 0.3
RESCUE_SIGNALS: tuple[str, ...] = ("omv",)

PROBE_CLASSES = ("ok", "noise", "absent")


def tokeye_rescues(coh_frames: np.ndarray) -> bool:
    """True when a probe's per-frame TokEye coherent activity looks like an intermittent mode."""
    c = np.asarray(coh_frames, dtype=np.float64)
    c = c[np.isfinite(c)]
    if not c.size:
        return False
    p90 = float(np.percentile(c, 90))
    return p90 >= RESCUE_P90 and float(np.median(c)) <= RESCUE_RATIO * p90


def probe_class(valid: float, ac1: float, coh5: float, coh_frames: np.ndarray | None = None,
                short: str | None = None) -> str:
    """``"absent"``, ``"noise"`` or ``"ok"`` for one probe over its plasma window.

    ``ac1`` and ``coh5`` are the largest per-band values (`band_ac1_coh_per_band`). With
    ``coh_frames`` (the probe's ``<s>_coh`` row) and ``short`` in `RESCUE_SIGNALS`, a noise
    probe that `tokeye_rescues` is ``ok``.
    """
    if not np.isfinite(valid) or valid < ABSENT_VALID or not np.isfinite(ac1):
        return "absent"
    if ac1 < NOISE_AC1 and (not np.isfinite(coh5) or coh5 < NOISE_COH5):
        if coh_frames is not None and short in RESCUE_SIGNALS and tokeye_rescues(coh_frames):
            return "ok"
        return "noise"
    return "ok"


def signal_verdict(classes: list[str]) -> bool:
    """Keep a spectrogram signal of a shot in the loss when its probes carry signal.

    Kept when at least one probe is ``ok`` and no more probes are ``noise`` than ``ok``: an
    absent probe codes a constant, which costs nothing to predict, while a noise probe is a
    target nothing can predict.
    """
    n_ok = classes.count("ok")
    return n_ok >= 1 and classes.count("noise") <= n_ok


def mode_frames(scores: Mapping[str, np.ndarray], short: str, classes: list[str]) -> np.ndarray:
    """``(n_frames,)`` float32: mean coherent activity over the ``ok`` probes (NaN if none)."""
    coh = np.asarray(scores[f"{short}_coh"], dtype=np.float64)
    ok = np.array([c == "ok" for c in classes])
    n = int(scores["n_frames"])
    if not ok.any():
        return np.full(n, np.nan, dtype=np.float32)
    sel = coh[ok]
    fin = np.isfinite(sel)
    cnt = fin.sum(axis=0)
    out = np.full(n, np.nan)
    np.divide(np.where(fin, sel, 0.0).sum(axis=0), cnt, out=out, where=cnt > 0)
    return out.astype(np.float32)
