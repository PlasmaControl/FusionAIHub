"""IGNITE Phase-A **acceptance instrument** (the oracle gate).

This module implements the codec-acceptance metrics of ``docs/IGNITE_DESIGN.md``
§4.4. It decides whether a fresh statistics-first codec's *codes* are good enough to
train a MaskGIT world model on (Phase B). Because Phase-A/Phase-B coupling is
**gate-only** (§6) — there is no downstream gradient pulling the codes toward
predictability — the correctness of these metrics matters a great deal. Every metric
here is validated against synthetic ground truth in ``tests/ignite/test_gate.py``.

Design mandate (§4.4):
  * **Stability ≥ 0.80** — fraction of code positions unchanged under the nuisance
    (raw δ-shift + re-STFT). A stable codec encodes the *statistic*, not the STFT-phase
    *realization*.
  * **Persistence ≥ 0.50** on steady segments (floor 0.10 = old-codec churn), and
    *measurably lower* on transitions.
  * **Forecastability** — a *cheap* probe (per-position first-order Markov) beats a
    persistence baseline at predicting the next frame's code **distribution**, reported
    separately for the transition vs stable strata.
  * **Mode-bearing decode** — reconstructed spectrogram matches GT mode structure
    (per-frequency power-envelope correlation + peak-overlap F1) and is not blurred
    (high-frequency gradient-energy ratio ≈ 1).

Reuse boundary (§7): this file reuses **no** FAITH model code. It depends only on
``config`` (shape contract), ``numpy``, and — where handed a ``torch.Tensor`` — accepts
it framework-agnostically by converting to numpy. No ``x-transformers`` / FSQ / e2e /
mode_audit imports.

All functions operate on small tensors and run on CPU.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

__all__ = [
    "stability",
    "persistence",
    "forecastability",
    "decode_fidelity",
    "video_decode_fidelity",
    "slowts_decode_fidelity",
    "fastts_decode_fidelity",
    "utilization",
]


# --------------------------------------------------------------------------------------
# tensor helpers (framework-agnostic: accept numpy or torch, return numpy)
# --------------------------------------------------------------------------------------
def _to_numpy(x) -> np.ndarray:
    """Convert a numpy array or torch tensor (or array-like) to a numpy array.

    We deliberately avoid importing torch at module scope so the gate has no hard torch
    dependency; if a torch tensor is passed we detect it duck-typed via ``.detach``.
    """
    if x is None:
        return None  # type: ignore[return-value]
    if isinstance(x, np.ndarray):
        return x
    # torch.Tensor (and anything else with .detach().cpu().numpy())
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _as_int_codes(codes) -> np.ndarray:
    """Coerce codes to an integer numpy array of shape (..., fsq_dim)."""
    arr = _to_numpy(codes)
    if not np.issubdtype(arr.dtype, np.integer):
        # FSQ indices are integral; round defensively (they should already be exact ints).
        arr = np.rint(arr).astype(np.int64)
    return arr


# --------------------------------------------------------------------------------------
# 1. stability — invariance under the nuisance (δ-shift) transform
# --------------------------------------------------------------------------------------
def stability(codes_x, codes_shifted) -> float:
    """Fraction of code positions **unchanged** under the nuisance transform.

    The nuisance is the raw δ-shift + re-STFT pair of §4.2. A statistics-first codec
    should map both members of the pair to (nearly) the same discrete codes. We measure
    the fraction of ``(batch, token, fsq_dim)`` entries that are identical between the
    two code tensors.

    A code *position* here is a single entry of the ``fsq_dim``-vector at one token —
    the finest granularity, so a partial flip of some FSQ dims at a token counts
    proportionally (this is the natural, monotone definition and matches "fraction of
    code positions unchanged").

    Args:
        codes_x:        (B, n_tok, fsq_dim) int codes for the original window.
        codes_shifted:  (B, n_tok, fsq_dim) int codes for the δ-shifted window.

    Returns:
        Float in [0, 1]; 1.0 iff every position matches.
    """
    a = _as_int_codes(codes_x)
    b = _as_int_codes(codes_shifted)
    if a.shape != b.shape:
        raise ValueError(f"stability: shape mismatch {a.shape} vs {b.shape}")
    if a.size == 0:
        raise ValueError("stability: empty code tensor")
    return float(np.mean(a == b))


# --------------------------------------------------------------------------------------
# 2. persistence — frame-to-frame code stickiness
# --------------------------------------------------------------------------------------
def persistence(codes_t, codes_tp1, steady_mask: Optional[object] = None) -> float:
    """Fraction of code positions **unchanged** from frame *t* to frame *t+1*.

    Persistence measures how "sticky" the codes are across the world-model stepping axis
    (one 50 ms frame). On steady plasma segments a good statistics-first codec should
    largely repeat its codes (mandate ≥ 0.50; old-codec churn floor ≈ 0.10). On
    transitions it should — and is expected to — be measurably lower.

    Args:
        codes_t:     (B, n_tok, fsq_dim) int codes at frame t.
        codes_tp1:   (B, n_tok, fsq_dim) int codes at frame t+1.
        steady_mask: optional boolean array selecting the *steady* subset. Broadcastable
                     against the leading dims of the code tensors; typical shapes are
                     (B,) — per-sample steady flag — or the full (B, n_tok, fsq_dim).
                     If given, persistence is computed only over positions where the mask
                     is True.

    Returns:
        Fraction in [0, 1] of unchanged positions (over the masked subset if provided).
    """
    a = _as_int_codes(codes_t)
    b = _as_int_codes(codes_tp1)
    if a.shape != b.shape:
        raise ValueError(f"persistence: shape mismatch {a.shape} vs {b.shape}")
    if a.size == 0:
        raise ValueError("persistence: empty code tensor")

    unchanged = a == b  # (B, n_tok, fsq_dim) bool

    if steady_mask is None:
        return float(np.mean(unchanged))

    mask = _to_numpy(steady_mask).astype(bool)
    # Broadcast a leading-dim mask (e.g. (B,) or (B, n_tok)) up to the full code shape.
    try:
        mask_b = np.broadcast_to(
            mask.reshape(mask.shape + (1,) * (unchanged.ndim - mask.ndim)),
            unchanged.shape,
        )
    except ValueError as e:
        raise ValueError(
            f"persistence: steady_mask shape {mask.shape} not broadcastable to "
            f"codes shape {unchanged.shape}"
        ) from e
    if mask_b.sum() == 0:
        raise ValueError("persistence: steady_mask selects no positions")
    return float(unchanged[mask_b].mean())


# --------------------------------------------------------------------------------------
# 3. forecastability — cheap next-frame probe vs a persistence baseline
# --------------------------------------------------------------------------------------
def _stack_frame_transitions(codes_seq: np.ndarray):
    """Flatten (B, n_frames, n_tok, fsq_dim) into aligned (t -> t+1) code-index pairs.

    Returns per-position source and target FSQ code-index arrays plus a per-transition
    "is a transition" flag (any position changed t -> t+1). We treat each
    ``(token, fsq_dim)`` entry as its own categorical stream (per-position Markov table,
    §4.4: "per-code Markov"), which keeps the probe cheap and honest.

    Shapes returned:
        src:   (N_trans, P) int   — code index at frame t
        tgt:   (N_trans, P) int   — code index at frame t+1
        moved: (N_trans,)  bool   — did *any* position change at this transition
    where P = n_tok * fsq_dim and N_trans = B * (n_frames - 1).
    """
    B, Fr, n_tok, fd = codes_seq.shape
    P = n_tok * fd
    flat = codes_seq.reshape(B, Fr, P)
    src = flat[:, :-1, :].reshape(-1, P)   # (B*(Fr-1), P)
    tgt = flat[:, 1:, :].reshape(-1, P)    # (B*(Fr-1), P)
    moved = np.any(src != tgt, axis=1)     # (B*(Fr-1),)
    return src, tgt, moved


def forecastability(
    codes_seq,
    transition_mask: Optional[object] = None,
    laplace: float = 1.0,
) -> Dict[str, float]:
    """Cheap next-frame-code-**distribution** probe vs a persistence baseline.

    This is the gate's forecastability instrument (§4.4). It is deliberately *not* the
    real dynamics: a per-position **first-order Markov** table is fit from the transition
    counts of ``codes_seq`` and scored by mean predictive log-likelihood of the true
    next code, against a **persistence** baseline (predict next == current, smoothed to a
    valid distribution). The reported ``margin`` is (probe − persistence) mean
    log-likelihood; **positive means the probe beats persistence**.

    Stratification (§4.4 "transition vs stable"): a *transition* is a frame step where at
    least one code position changed; a *stable* step is one where nothing changed. We
    report skill separately on each stratum. On stable steps persistence is already
    optimal (nothing moves), so the probe can at best tie (margin ≈ 0). On predictable
    transitions (e.g. a drifting index) the Markov probe should strictly beat persistence
    (margin > 0). On a frozen sequence there are no transitions and the transition margin
    is reported as 0.0.

    Args:
        codes_seq:       (B, n_frames, n_tok, fsq_dim) int codes.
        transition_mask: optional boolean per-transition override of shape
                         (B, n_frames - 1). If None, the transition/stable split is
                         derived from the data (any-position-moved).
        laplace:         additive (Laplace) smoothing count for both models, so unseen
                         symbols get finite probability.

    Returns:
        dict with keys:
          ``margin_transition``  probe − persistence mean log-lik on transition steps.
          ``margin_stable``      same on stable steps (≈ 0 by construction).
          ``margin_overall``     over all steps.
          ``probe_ll_transition`` / ``persistence_ll_transition``  raw mean log-liks.
          ``probe_ll_stable``     / ``persistence_ll_stable``.
          ``n_transition`` / ``n_stable``   step counts in each stratum.
          ``beats_persistence``  bool: margin_transition > 0 (the gate's pass condition).
    """
    codes = _as_int_codes(codes_seq)
    if codes.ndim != 4:
        raise ValueError(
            f"forecastability expects (B, n_frames, n_tok, fsq_dim); got {codes.shape}"
        )
    B, Fr, n_tok, fd = codes.shape
    if Fr < 2:
        raise ValueError("forecastability: need at least 2 frames")

    src, tgt, moved = _stack_frame_transitions(codes)
    P = src.shape[1]
    n_trans = src.shape[0]

    if transition_mask is not None:
        tm = _to_numpy(transition_mask).astype(bool)
        if tm.shape != (B, Fr - 1):
            raise ValueError(
                f"transition_mask must be (B, n_frames-1)={ (B, Fr - 1) }; got {tm.shape}"
            )
        moved = tm.reshape(-1)

    # Persistence baseline needs a small "leak" probability so it is a proper
    # distribution (never log(0)) yet still strongly favours next==current. We tie it to
    # the data size so it shrinks as evidence grows (like Laplace smoothing), and clamp
    # it to a small value so the baseline stays a genuine persistence predictor.
    eps = min(0.05, laplace / (laplace + n_trans))

    # Fit a per-position first-order Markov table over ALL transitions (the probe is
    # allowed to learn from the sequence — it is a fit predictor, not an oracle) and
    # accumulate the mean-over-positions log-likelihood of the true next code per step.
    probe_ll = np.zeros(n_trans, dtype=np.float64)
    persist_ll = np.zeros(n_trans, dtype=np.float64)

    for p in range(P):
        s = src[:, p]
        t = tgt[:, p]
        vmax = int(max(s.max(), t.max())) + 1  # symbols seen at this position

        # ---- Markov probe: P(next=j | cur=i) from Laplace-smoothed counts ----
        counts = np.zeros((vmax, vmax), dtype=np.float64)
        np.add.at(counts, (s, t), 1.0)
        counts += laplace
        trans_prob = counts / counts.sum(axis=1, keepdims=True)  # rows sum to 1
        probe_ll += np.log(trans_prob[s, t])

        # ---- persistence baseline: put (1-eps) mass on next==cur, eps spread over rest.
        if vmax == 1:
            persist_ll += 0.0  # single symbol: both models certain -> log(1) = 0
        else:
            same = t == s
            p_same = 1.0 - eps
            p_diff = eps / (vmax - 1)
            persist_ll += np.where(same, np.log(p_same), np.log(p_diff))

    probe_ll /= P
    persist_ll /= P

    def _stratum(sel):
        n = int(sel.sum())
        if n == 0:
            return 0.0, 0.0, 0.0, 0
        pll = float(probe_ll[sel].mean())
        qll = float(persist_ll[sel].mean())
        return pll - qll, pll, qll, n

    m_tr, p_tr, q_tr, n_tr = _stratum(moved)
    m_st, p_st, q_st, n_st = _stratum(~moved)
    m_all, p_all, q_all, _ = _stratum(np.ones(n_trans, dtype=bool))

    return {
        "margin_transition": m_tr,
        "margin_stable": m_st,
        "margin_overall": m_all,
        "probe_ll_transition": p_tr,
        "persistence_ll_transition": q_tr,
        "probe_ll_stable": p_st,
        "persistence_ll_stable": q_st,
        "n_transition": n_tr,
        "n_stable": n_st,
        "beats_persistence": bool(m_tr > 0.0),
    }


# --------------------------------------------------------------------------------------
# 4. decode_fidelity — mode-structure match + sharpness of reconstructed spectrogram
# --------------------------------------------------------------------------------------
def _power_envelope(spec: np.ndarray) -> np.ndarray:
    """Per-frequency power envelope: mean power over time, per (batch, channel, freq).

    Input is a log-power spectrogram (B, C, F, T). We average over the time axis to get
    the (B, C, F) profile of "which frequencies carry power" — the mode-presence
    statistic the codec is supposed to preserve.
    """
    return spec.mean(axis=-1)  # (B, C, F)


def _envelope_correlation(recon: np.ndarray, target: np.ndarray) -> float:
    """Mean Pearson correlation of the per-frequency power envelope over (B, C)."""
    er = _power_envelope(recon)   # (B, C, F)
    et = _power_envelope(target)
    B, C, F = er.shape
    er = er.reshape(B * C, F)
    et = et.reshape(B * C, F)
    corrs = []
    for i in range(B * C):
        a = er[i] - er[i].mean()
        b = et[i] - et[i].mean()
        denom = np.sqrt((a * a).sum() * (b * b).sum())
        if denom <= 1e-12:
            # A flat envelope has no structure to correlate; treat as 0 (no match info).
            corrs.append(0.0)
        else:
            corrs.append(float((a * b).sum() / denom))
    return float(np.mean(corrs))


def _peak_mask(env: np.ndarray, k: float) -> np.ndarray:
    """Boolean per-frequency 'mode present' mask: env above mean + k * std, per row.

    ``env`` is (N, F). A frequency is a detected peak/mode if its time-averaged power is
    k standard deviations above that row's mean — a simple, threshold-based mode detector
    that needs no external code.
    """
    mu = env.mean(axis=1, keepdims=True)
    sd = env.std(axis=1, keepdims=True)
    return env > (mu + k * sd)


def _peak_overlap_f1(recon: np.ndarray, target: np.ndarray, k: float = 1.0) -> float:
    """F1 of detected-mode (peak) overlap between recon and target envelopes."""
    er = _power_envelope(recon)
    et = _power_envelope(target)
    B, C, F = er.shape
    er = er.reshape(B * C, F)
    et = et.reshape(B * C, F)
    pr = _peak_mask(er, k)
    pt = _peak_mask(et, k)
    tp = np.logical_and(pr, pt).sum(axis=1).astype(np.float64)
    fp = np.logical_and(pr, ~pt).sum(axis=1).astype(np.float64)
    fn = np.logical_and(~pr, pt).sum(axis=1).astype(np.float64)
    denom = 2 * tp + fp + fn
    # If a row has no peaks in either recon or target, overlap is trivially perfect.
    f1_rows = np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-12), 1.0)
    return float(f1_rows.mean())


def _hf_gradient_energy(spec: np.ndarray) -> float:
    """Total high-frequency gradient energy: sum of squared finite differences.

    Sharpness proxy. Along both the frequency and time axes we take first differences
    (a high-pass filter) and sum their squared magnitude. A blurred / mean-collapsed
    spectrogram has far less gradient energy than a sharp, mode-bearing one.
    """
    df = np.diff(spec, axis=-2)  # along frequency
    dt = np.diff(spec, axis=-1)  # along time
    return float((df ** 2).sum() + (dt ** 2).sum())


def decode_fidelity(recon, target, peak_k: float = 1.0) -> Dict[str, float]:
    """Mode-structure match + sharpness between a reconstructed and target spectrogram.

    Implements the §4.4 "mode-bearing decode" checks on log-power spectrograms
    ``(B, C, F, T)``:

      * ``envelope_corr``  — mean Pearson correlation of the per-frequency power envelope
        (which frequencies carry power). ≈ 1 when mode structure is preserved.
      * ``peak_f1``        — F1 of detected-mode (peak) overlap between recon and target.
        ≈ 1 when the same modes are present; drops when modes are missed/added.
      * ``sharpness``      — ratio of high-frequency gradient energy (recon / target).
        ≈ 1 = not blurred; < 1 = collapsed/low-pass; note it can exceed 1 if the recon is
        *noisier/sharper* than the target.

    Args:
        recon:  (B, C, F, T) reconstructed log-power spectrogram.
        target: (B, C, F, T) ground-truth log-power spectrogram.
        peak_k: std-multiplier threshold for the peak/mode detector.

    Returns:
        dict with ``envelope_corr``, ``peak_f1``, ``sharpness``, and the raw
        ``hf_energy_recon`` / ``hf_energy_target``.
    """
    r = _to_numpy(recon).astype(np.float64)
    t = _to_numpy(target).astype(np.float64)
    if r.shape != t.shape:
        raise ValueError(f"decode_fidelity: shape mismatch {r.shape} vs {t.shape}")
    if r.ndim != 4:
        raise ValueError(
            f"decode_fidelity expects (B, C, F, T); got {r.shape}"
        )

    env_corr = _envelope_correlation(r, t)
    peak_f1 = _peak_overlap_f1(r, t, k=peak_k)
    hf_r = _hf_gradient_energy(r)
    hf_t = _hf_gradient_energy(t)
    sharpness = float(hf_r / hf_t) if hf_t > 1e-12 else float("nan")

    return {
        "envelope_corr": env_corr,
        "peak_f1": peak_f1,
        "sharpness": sharpness,
        "hf_energy_recon": hf_r,
        "hf_energy_target": hf_t,
    }


# --------------------------------------------------------------------------------------
# 4b. video_decode_fidelity — mode-structure match + sharpness for VIDEO reconstructions
# --------------------------------------------------------------------------------------
def _video_pixel_profile(vid: np.ndarray) -> np.ndarray:
    """Per-pixel time-averaged intensity: mean over the time axis, per (batch, channel, H, W).

    Input is a video ``(B, C, T, H, W)``. Averaging over time gives the ``(B, C, H, W)``
    spatial *intensity map* — "where the divertor light sits" — the persistent spatial
    statistic the video codec is supposed to preserve (the divertor-video analogue of the
    spectrogram's per-frequency power envelope in :func:`decode_fidelity`).
    """
    return vid.mean(axis=2)  # (B, C, H, W)


def _flatten_profile(prof: np.ndarray) -> np.ndarray:
    """(B, C, H, W) intensity map -> (B*C, H*W) rows for per-row correlation / peak stats."""
    B, C, H, W = prof.shape
    return prof.reshape(B * C, H * W)


def _profile_correlation(recon: np.ndarray, target: np.ndarray) -> float:
    """Mean Pearson correlation of the per-pixel time-averaged intensity map over (B, C)."""
    er = _flatten_profile(_video_pixel_profile(recon))
    et = _flatten_profile(_video_pixel_profile(target))
    corrs = []
    for i in range(er.shape[0]):
        a = er[i] - er[i].mean()
        b = et[i] - et[i].mean()
        denom = np.sqrt((a * a).sum() * (b * b).sum())
        corrs.append(0.0 if denom <= 1e-12 else float((a * b).sum() / denom))
    return float(np.mean(corrs))


def _profile_peak_f1(recon: np.ndarray, target: np.ndarray, k: float = 1.0) -> float:
    """F1 of bright-region (peak) overlap between recon and target intensity maps."""
    er = _flatten_profile(_video_pixel_profile(recon))
    et = _flatten_profile(_video_pixel_profile(target))
    pr = _peak_mask(er, k)  # reuse the spectro peak detector: > row-mean + k*row-std
    pt = _peak_mask(et, k)
    tp = np.logical_and(pr, pt).sum(axis=1).astype(np.float64)
    fp = np.logical_and(pr, ~pt).sum(axis=1).astype(np.float64)
    fn = np.logical_and(~pr, pt).sum(axis=1).astype(np.float64)
    denom = 2 * tp + fp + fn
    f1_rows = np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-12), 1.0)
    return float(f1_rows.mean())


def _video_hf_gradient_energy(vid: np.ndarray) -> float:
    """Total spatial+temporal high-frequency gradient energy (sum of squared diffs).

    Sharpness proxy for video: first differences along the two spatial axes AND the time
    axis. A blurred / mean-collapsed reconstruction (the failure a strong pixel-MSE would
    cause) has far less gradient energy than the sharp, structure-bearing target.
    """
    dh = np.diff(vid, axis=-2)   # height
    dw = np.diff(vid, axis=-1)   # width
    dt = np.diff(vid, axis=2)    # time
    return float((dh ** 2).sum() + (dw ** 2).sum() + (dt ** 2).sum())


def video_decode_fidelity(recon, target, peak_k: float = 1.0) -> Dict[str, float]:
    """Video analogue of :func:`decode_fidelity` — spatial-structure match + sharpness.

    Operates on ``(B, C, T, H, W)`` frames. Returns the SAME keys as :func:`decode_fidelity`
    (``envelope_corr`` / ``peak_f1`` / ``sharpness`` / raw ``hf_energy_*``) so the composite
    :func:`~ignite.spike.gate_score` and the gate plumbing consume it unchanged, but the
    "envelope" here is the per-pixel time-averaged intensity map (the persistent spatial
    statistic) rather than the per-frequency power envelope:

      * ``envelope_corr`` — mean Pearson correlation of the time-averaged intensity map
        (where the light sits). ≈ 1 when spatial structure is preserved.
      * ``peak_f1``       — F1 of bright-region overlap between recon and target maps.
      * ``sharpness``     — ratio of spatial+temporal HF gradient energy (recon / target).
        ≈ 1 = not blurred; < 1 = mean-collapsed; > 1 = noisier/sharper than the target.
    """
    r = _to_numpy(recon).astype(np.float64)
    t = _to_numpy(target).astype(np.float64)
    if r.shape != t.shape:
        raise ValueError(f"video_decode_fidelity: shape mismatch {r.shape} vs {t.shape}")
    if r.ndim != 5:
        raise ValueError(f"video_decode_fidelity expects (B, C, T, H, W); got {r.shape}")

    env_corr = _profile_correlation(r, t)
    peak_f1 = _profile_peak_f1(r, t, k=peak_k)
    hf_r = _video_hf_gradient_energy(r)
    hf_t = _video_hf_gradient_energy(t)
    sharpness = float(hf_r / hf_t) if hf_t > 1e-12 else float("nan")

    return {
        "envelope_corr": env_corr,
        "peak_f1": peak_f1,
        "sharpness": sharpness,
        "hf_energy_recon": hf_r,
        "hf_energy_target": hf_t,
    }


# --------------------------------------------------------------------------------------
# 4c. slowts_decode_fidelity — profile-structure match + sharpness for slow-TS recons
# --------------------------------------------------------------------------------------
def _slowts_profile(sig: np.ndarray) -> np.ndarray:
    """Per-position time-averaged value: mean over the time axis, per (batch, position).

    Input is a slow-TS window ``(B, C, T)`` — ``C`` profile positions × ``T`` time samples.
    Averaging over time gives the ``(B, C)`` *profile shape* — "the value at each position" —
    the persistent statistic the slow-TS codec is supposed to preserve (the smooth-profile
    analogue of the spectrogram's per-frequency power envelope in :func:`decode_fidelity` and
    the video's per-pixel intensity map in :func:`video_decode_fidelity`).
    """
    return sig.mean(axis=-1)  # (B, C)


def _slowts_profile_masked(sig: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """Per-position time-average over VALID samples only; falls back to the plain mean.

    ``mask`` is a (B, C, T) validity mask (1 = valid). A position with no valid samples in
    the window gets its plain time-mean (which is the zero/NaN-fill value) — harmless, since
    the correlation is over positions and a fully-missing position is a constant either way.
    """
    if mask is None:
        return _slowts_profile(sig)
    m = mask.astype(np.float64)
    denom = m.sum(axis=-1)                       # (B, C)
    num = (sig * m).sum(axis=-1)                 # (B, C)
    with np.errstate(invalid="ignore", divide="ignore"):
        prof = np.where(denom > 0, num / np.maximum(denom, 1e-12), sig.mean(axis=-1))
    return prof


def slowts_decode_fidelity(recon, target, mask=None, peak_k: float = 1.0) -> Dict[str, float]:
    """Slow-TS analogue of :func:`decode_fidelity` — profile-structure match + sharpness.

    Operates on ``(B, C, T)`` slow-TS windows. Returns the SAME keys as
    :func:`decode_fidelity` (``envelope_corr`` / ``peak_f1`` / ``sharpness`` / raw
    ``hf_energy_*``) so the composite :func:`~ignite.spike.gate_score` and the gate plumbing
    consume it unchanged, but the "envelope" here is the per-position time-averaged profile
    (the smooth statistic) rather than a per-frequency power envelope:

      * ``envelope_corr`` — mean Pearson correlation of the per-position time-averaged profile
        (the profile shape). ≈ 1 when profile structure is preserved.
      * ``peak_f1``       — F1 of high-value-position overlap between recon and target profiles.
      * ``sharpness``     — ratio of position+time HF gradient energy (recon / target).
        ≈ 1 = profile detail preserved; < 1 = over-smoothed / mean-collapsed.

    ``mask`` (optional, (B, C, T) validity, 1 = valid) makes the per-position profile average
    over VALID samples only, so beam-off / not-firing missing samples do not corrupt the
    statistic. The HF-gradient sharpness is computed on the raw (zero/NaN-filled) window: it is
    a relative recon/target ratio, so the shared fill cancels.
    """
    r = _to_numpy(recon).astype(np.float64)
    t = _to_numpy(target).astype(np.float64)
    if r.shape != t.shape:
        raise ValueError(f"slowts_decode_fidelity: shape mismatch {r.shape} vs {t.shape}")
    if r.ndim != 3:
        raise ValueError(f"slowts_decode_fidelity expects (B, C, T); got {r.shape}")
    m = _to_numpy(mask) if mask is not None else None
    if m is not None and m.shape != r.shape:
        raise ValueError(
            f"slowts_decode_fidelity: mask shape {m.shape} != signal shape {r.shape}"
        )

    er = _slowts_profile_masked(r, m)   # (B, C)
    et = _slowts_profile_masked(t, m)
    # per-row (per-sample) Pearson correlation of the (C,) profile vectors.
    corrs = []
    for i in range(er.shape[0]):
        a = er[i] - er[i].mean()
        b = et[i] - et[i].mean()
        denom = np.sqrt((a * a).sum() * (b * b).sum())
        corrs.append(0.0 if denom <= 1e-12 else float((a * b).sum() / denom))
    env_corr = float(np.mean(corrs))

    # peak/high-value-position overlap F1 (reuse the shared threshold detector on the profiles).
    pr = _peak_mask(er, peak_k)
    pt = _peak_mask(et, peak_k)
    tp = np.logical_and(pr, pt).sum(axis=1).astype(np.float64)
    fp = np.logical_and(pr, ~pt).sum(axis=1).astype(np.float64)
    fn = np.logical_and(~pr, pt).sum(axis=1).astype(np.float64)
    denom = 2 * tp + fp + fn
    f1_rows = np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-12), 1.0)
    peak_f1 = float(f1_rows.mean())

    # sharpness: position + time HF gradient energy ratio (recon / target).
    def _hf(v: np.ndarray) -> float:
        dc = np.diff(v, axis=1)   # along position
        dt = np.diff(v, axis=2)   # along time
        return float((dc ** 2).sum() + (dt ** 2).sum())

    hf_r = _hf(r)
    hf_t = _hf(t)
    sharpness = float(hf_r / hf_t) if hf_t > 1e-12 else float("nan")

    return {
        "envelope_corr": env_corr,
        "peak_f1": peak_f1,
        "sharpness": sharpness,
        "hf_energy_recon": hf_r,
        "hf_energy_target": hf_t,
    }


# --------------------------------------------------------------------------------------
# 4d. fastts_decode_fidelity — ELM-envelope-structure match + sharpness for fast-TS recons
# --------------------------------------------------------------------------------------
def fastts_decode_fidelity(recon, target, peak_k: float = 1.0) -> Dict[str, float]:
    """Fast-TS analogue of :func:`decode_fidelity` — ELM-envelope match + sharpness.

    Operates on the ELM ACTIVITY ENVELOPE ``(B, C, E)`` (E = envelope-time bins), NOT the raw
    spike waveform (docs/IGNITE_DESIGN.md §4.3). Returns the SAME keys as
    :func:`decode_fidelity` (``envelope_corr`` / ``peak_f1`` / ``sharpness`` / raw
    ``hf_energy_*``) so the composite :func:`~ignite.spike.gate_score` and the gate plumbing
    consume it unchanged. Unlike :func:`slowts_decode_fidelity` — which time-averages to a
    per-position profile — fast-TS's whole statistic IS the WHEN-and-how-much of ELM activity
    over time, so the metrics operate on the full per-channel envelope-time CURVE:

      * ``envelope_corr`` — mean Pearson correlation of the per-channel envelope curve over the
        envelope-time axis (does the recon put activity at the same bins, with the same
        relative amplitude?). ≈ 1 when the ELM activity timing/amplitude is preserved.
      * ``peak_f1``       — F1 of high-activity-bin (burst) overlap between recon and target
        envelope curves (the ELM-burst-detection analogue of spectro mode-peak overlap).
      * ``sharpness``     — ratio of time-axis HF gradient energy (recon / target). ≈ 1 =
        burst edges preserved; < 1 = over-smoothed / mean-collapsed envelope.

    A mode-bearing (burst-detection) metric is natural here (``peak_f1`` over the envelope's
    high-activity bins), satisfying §4.4's "a burst/ELM-relevant metric if natural".
    """
    r = _to_numpy(recon).astype(np.float64)
    t = _to_numpy(target).astype(np.float64)
    if r.shape != t.shape:
        raise ValueError(f"fastts_decode_fidelity: shape mismatch {r.shape} vs {t.shape}")
    if r.ndim != 3:
        raise ValueError(f"fastts_decode_fidelity expects (B, C, E); got {r.shape}")

    B, C, E = r.shape
    rr = r.reshape(B * C, E)
    tt = t.reshape(B * C, E)

    # per-channel Pearson correlation of the envelope-time CURVE (not time-averaged).
    corrs = []
    for i in range(B * C):
        a = rr[i] - rr[i].mean()
        b = tt[i] - tt[i].mean()
        denom = np.sqrt((a * a).sum() * (b * b).sum())
        # A flat envelope curve (no activity) has no structure to correlate; treat as 0.
        corrs.append(0.0 if denom <= 1e-12 else float((a * b).sum() / denom))
    env_corr = float(np.mean(corrs))

    # burst-overlap F1: a bin is a "burst" if its activity is > row-mean + k*row-std
    # (reuse the shared threshold detector on the envelope curves).
    pr = _peak_mask(rr, peak_k)
    pt = _peak_mask(tt, peak_k)
    tp = np.logical_and(pr, pt).sum(axis=1).astype(np.float64)
    fp = np.logical_and(pr, ~pt).sum(axis=1).astype(np.float64)
    fn = np.logical_and(~pr, pt).sum(axis=1).astype(np.float64)
    denom = 2 * tp + fp + fn
    f1_rows = np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-12), 1.0)
    peak_f1 = float(f1_rows.mean())

    # sharpness: time-axis HF gradient energy ratio (recon / target) — burst-edge preservation.
    def _hf(v: np.ndarray) -> float:
        dt = np.diff(v, axis=-1)   # along the envelope-time axis
        return float((dt ** 2).sum())

    hf_r = _hf(r)
    hf_t = _hf(t)
    sharpness = float(hf_r / hf_t) if hf_t > 1e-12 else float("nan")

    return {
        "envelope_corr": env_corr,
        "peak_f1": peak_f1,
        "sharpness": sharpness,
        "hf_energy_recon": hf_r,
        "hf_energy_target": hf_t,
    }


# --------------------------------------------------------------------------------------
# 5. utilization — codebook-usage / anti-collapse detector
# --------------------------------------------------------------------------------------
def utilization(
    codes,
    cfg=None,
    *,
    codebook_size: Optional[int] = None,
    min_utilization: Optional[float] = None,
    min_code_entropy: Optional[float] = None,
) -> Dict[str, object]:
    """Codebook-utilization / **collapse** detector for a set of FSQ codes.

    A gate-spike revealed the codec can posterior-collapse to a single FSQ code (the
    shift-consistency loss has a trivial minimum at encoder ≡ constant). This metric flags
    that failure directly from the eval codes, independently of the differentiable training
    regularizer.

    Signals:

      * ``min_dim_entropy``  — the MINIMUM over FSQ dimensions of that dimension's
        normalized code entropy ``H(dim) / log(levels[dim])`` (in [0, 1]). PRIMARY collapse
        signal: eval-size-robust (a collapsed dimension drives it to 0 regardless of how
        many tokens were evaluated).
      * ``frac_of_observable`` — distinct flat codes divided by the MAX achievable given the
        eval set, ``min(codebook_size, n_observed_tokens)``. Catches total collapse without
        the eval-size confound.
      * ``frac_codes_used``  — distinct flat codes / total ``codebook_size``. INFORMATIONAL
        ONLY (do NOT threshold it): for a large codebook + small eval set the max achievable
        value is ``n_observed / codebook_size`` (e.g. 576/32768 ≈ 0.018), so an absolute
        threshold on it fires spuriously even for a perfectly healthy codec.

    A codec is ``collapsed`` if ``min_dim_entropy < min_code_entropy`` OR
    ``frac_of_observable < min_utilization``.

    Args:
        codes:            (B, n_tok, fsq_dim) OR (B, n_frames, n_tok, fsq_dim) int codes
                          (frames are flattened into the sample axis). Each entry ``[..., i]``
                          is a level index in ``[0, levels[i])``.
        cfg:              optional ``SpectroCodecConfig``; if given, supplies
                          ``codebook_size`` and the ``gate_min_utilization`` /
                          ``gate_min_code_entropy`` thresholds (and ``fsq_levels`` for the
                          per-dim entropy normalization).
        codebook_size:    total codebook size (prod of per-dim levels). Overrides ``cfg``.
        min_utilization:  threshold on ``frac_codes_used``. Overrides ``cfg``.
        min_code_entropy: threshold on ``min_dim_entropy``. Overrides ``cfg``.

    Returns:
        dict with ``frac_codes_used`` (float), ``min_dim_entropy`` (float),
        ``collapsed`` (bool), plus ``n_distinct_codes`` and ``codebook_size`` for logging.
    """
    arr = _as_int_codes(codes)
    if arr.ndim == 4:
        # (B, n_frames, n_tok, fsq_dim) -> flatten frames into the sample axis.
        B, Fr, n_tok, fd = arr.shape
        arr = arr.reshape(B * Fr, n_tok, fd)
    if arr.ndim != 3:
        raise ValueError(
            f"utilization expects (B, n_tok, fsq_dim) or (B, n_frames, n_tok, fsq_dim); "
            f"got {arr.shape}"
        )
    if arr.size == 0:
        raise ValueError("utilization: empty code tensor")

    fsq_dim = arr.shape[-1]

    # resolve config-derived quantities (explicit kwargs win over cfg).
    if codebook_size is None:
        if cfg is None:
            raise ValueError("utilization: provide cfg or codebook_size")
        codebook_size = int(cfg.codebook_size)
    if min_utilization is None:
        min_utilization = float(getattr(cfg, "gate_min_utilization", 0.02))
    if min_code_entropy is None:
        min_code_entropy = float(getattr(cfg, "gate_min_code_entropy", 0.3))

    # per-dim level counts for the entropy normalization; fall back to the observed
    # per-dim max+1 if cfg (hence fsq_levels) is unavailable.
    if cfg is not None and getattr(cfg, "fsq_levels", None) is not None:
        levels = [int(v) for v in cfg.fsq_levels]
        if len(levels) != fsq_dim:
            raise ValueError(
                f"utilization: cfg.fsq_levels has {len(levels)} dims but codes have {fsq_dim}"
            )
    else:
        levels = [int(arr[..., i].max()) + 1 for i in range(fsq_dim)]

    # --- distinct flat codes / codebook_size -------------------------------------------
    flat = arr.reshape(-1, fsq_dim)                       # (M, fsq_dim)
    n_distinct = int(np.unique(flat, axis=0).shape[0])
    frac_codes_used = float(n_distinct / codebook_size)   # INFORMATIONAL only (eval-size-confounded)
    n_observed = int(flat.shape[0])
    # eval-size-RELATIVE utilization: you cannot observe more distinct codes than tokens,
    # so normalise by the achievable max. This is the sound collapse signal for a large
    # codebook evaluated on a small token set.
    frac_of_observable = float(n_distinct / max(1, min(int(codebook_size), n_observed)))

    # --- per-dim normalized entropy; take the minimum over dims ------------------------
    dim_entropies = []
    for i in range(fsq_dim):
        vals = flat[:, i]
        counts = np.bincount(vals, minlength=levels[i]).astype(np.float64)
        probs = counts / counts.sum()
        nz = probs[probs > 0]
        h = float(-(nz * np.log(nz)).sum())               # nats
        norm = np.log(levels[i]) if levels[i] > 1 else 1.0
        dim_entropies.append(h / norm if norm > 0 else 0.0)
    min_dim_entropy = float(min(dim_entropies)) if dim_entropies else 0.0

    collapsed = bool(
        min_dim_entropy < min_code_entropy or frac_of_observable < min_utilization
    )

    return {
        "frac_codes_used": frac_codes_used,          # informational (eval-size-confounded)
        "frac_of_observable": frac_of_observable,    # eval-size-relative (used for collapse)
        "min_dim_entropy": min_dim_entropy,
        "collapsed": collapsed,
        "n_distinct_codes": n_distinct,
        "n_observed_tokens": n_observed,
        "codebook_size": int(codebook_size),
    }
