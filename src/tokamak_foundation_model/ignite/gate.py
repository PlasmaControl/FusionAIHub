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

from typing import Dict, Optional, Sequence

import numpy as np

__all__ = [
    "stability",
    "persistence",
    "forecastability",
    "decode_fidelity",
    "full_spectro_metrics",
    "trivial_spectro_baselines",
    "patch_lattice_metrics",
    "remove_patch_lattice",
    "video_decode_fidelity",
    "slowts_decode_fidelity",
    "fastts_decode_fidelity",
    "full_fastts_metrics",
    "trivial_fastts_baselines",
    "full_slowts_metrics",
    "trivial_slowts_baselines",
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


# ---------------------------------------------------------------------------------------
# 4a. PATCH-LATTICE ARTIFACT metrics — the checkerboard / patch-seam detector
# ---------------------------------------------------------------------------------------
# The spectro decoder paints every ``(patch_f x patch_t)`` patch from its OWN token through a
# SHARED linear map (``nets.SpectroDecoder.to_pixels``: d_model -> C*patch_f*patch_t, plus a
# shared bias) and then tiles the patches. Every patch therefore draws its texture from the same
# 256-column basis, so whatever high-frequency texture that basis carries is REPLICATED on a
# ``patch_f x patch_t`` lattice — a visible regular grid in the decoded spectrogram.
#
# Neither ``envelope_corr`` (time-averaged per-frequency profile), ``peak_f1`` (mode overlap) nor
# ``sharpness`` (total HF gradient energy) can see this: the lattice texture IS high-frequency
# gradient energy, so it can even INFLATE sharpness. These two metrics detect it directly.
#
# MEASURED on the mhr v1k ms2_s1 codec (step 12001, patch 16x16), 32 held-out windows:
#     ground truth      patch_lattice_ratio 1.151   lattice_energy_frac 0.034
#     reconstruction    patch_lattice_ratio 61.6    lattice_energy_frac 0.559
# i.e. 56 % of the reconstruction's 2-D spectral energy sits EXACTLY on the patch lattice
# (ground truth: 3.4 %), and no single held-out window overlapped between the two populations
# (GT 1.05-1.21 vs recon 43.6-76.0).
def _lattice_harmonics(F: int, T: int, patch_f: int, patch_t: int):
    """The (kf, kt) DFT bins of a ``patch_f x patch_t`` tiling of a (F, T) image.

    Tiling one shared patch texture ``P`` times along an axis of length ``F`` puts all of its
    energy on the multiples of ``F // patch_f`` — the lattice harmonics. Returns the sorted set
    of harmonic bins EXCLUDING DC (which is just the image mean).
    """
    nf, nt = F // patch_f, T // patch_t
    harm = {((m * nf) % F, (n * nt) % T) for m in range(patch_f) for n in range(patch_t)}
    harm.discard((0, 0))
    return sorted(harm)


def patch_lattice_metrics(
    spec: np.ndarray,
    patch_f: int,
    patch_t: int,
    ring: int = 3,
) -> Dict[str, float]:
    """Patch-lattice (checkerboard / seam) artifact strength of a spectrogram batch.

    Args:
        spec:    (B, C, F, T) log-power spectrogram (reconstruction OR ground truth).
        patch_f: codec patch height in frequency bins (``cfg.patch_f``).
        patch_t: codec patch width in STFT time frames (``cfg.patch_t``).
        ring:    half-width of the local background window around each harmonic bin.

    Returns:
        ``patch_lattice_ratio`` — the PRIMARY artifact scalar. Per-image mean removed, 2-D FFT
            magnitudes averaged over (B, C); for every patch-lattice harmonic bin we take
            ``|X(harmonic)| / median(|X| over the (2*ring+1)^2 neighbourhood, harmonics
            excluded)`` and return the GEOMETRIC mean over harmonics. **1.0 = no lattice**
            (the harmonic bins are indistinguishable from their neighbours); > 1 = a
            ``patch_f x patch_t`` grid texture. Local (not global) normalization is what makes
            the ideal 1.0 hold on real spectrograms, whose spectral energy falls off steeply
            with spatial frequency.
        ``lattice_energy_frac`` — share of the mean power spectrum (DC excluded) that sits on
            the lattice harmonics. Scale-free companion; ~0.03 on real mhr spectrograms.
        ``seam_ratio_freq`` / ``seam_ratio_time`` — mean |first difference| ACROSS patch
            boundaries divided by a LOCAL (+-ring, boundaries excluded) interior baseline, per
            axis. 1.0 = no seam. These read the BOUNDARY only, so they are diagnostic rather
            than primary: a decoder whose shared per-patch texture is smooth at its edges but
            replicated everywhere shows a strong lattice with seam ratios near (or below) 1.

    READ IT BESIDE ``envelope_corr``. A COLLAPSED codec (one code for every token) emits the
    same patch everywhere, which is a perfect lattice — measured 1.4e5 on a 1-code arm at step
    0. A high lattice ratio therefore means "grid artifact" only for a codec that is actually
    reconstructing; on a dead codec it is just restating the collapse.

    AND BESIDE ``sharpness``. The metric is NOT blur-proof on its own: Gaussian-blurring the
    mhr baseline recon drives it 61.7 -> 24.8 -> 3.1 at sigma 0.5 / 1.0 / 2.0, i.e. the trivial
    "fix" scores well. What exposes that is the HF-gradient ratio falling in lockstep
    (0.820 -> 0.024 -> 0.0024). The honest verdict is always the PAIR: a real fix lowers the
    lattice ratio while HOLDING ``sharpness``.
    """
    x = np.asarray(spec, dtype=np.float64)
    if x.ndim != 4:
        raise ValueError(f"patch_lattice_metrics expects (B, C, F, T); got {x.shape}")
    F, T = x.shape[-2], x.shape[-1]
    if F % patch_f or T % patch_t:
        raise ValueError(
            f"patch_lattice_metrics: (F, T) = ({F}, {T}) is not divisible by "
            f"(patch_f, patch_t) = ({patch_f}, {patch_t})"
        )

    # --- 2-D FFT lattice-harmonic ratio -------------------------------------------------
    y = x - x.mean(axis=(-2, -1), keepdims=True)
    mag = np.abs(np.fft.fft2(y, axes=(-2, -1))).reshape(-1, F, T).mean(axis=0)
    harm = _lattice_harmonics(F, T, patch_f, patch_t)
    hmask = np.zeros((F, T), dtype=bool)
    for a, b in harm:
        hmask[a, b] = True
    ratios = []
    for a, b in harm:
        aa = np.arange(a - ring, a + ring + 1) % F
        bb = np.arange(b - ring, b + ring + 1) % T
        sub = mag[np.ix_(aa, bb)]
        bg = np.median(sub[~hmask[np.ix_(aa, bb)]])
        if bg > 0:
            ratios.append(mag[a, b] / bg)
    lattice_ratio = float(np.exp(np.mean(np.log(ratios)))) if ratios else float("nan")

    power = mag ** 2
    power[0, 0] = 0.0
    total = power.sum()
    lattice_frac = float(power[hmask].sum() / total) if total > 0 else float("nan")

    # --- locally-normalized boundary seam ratios ---------------------------------------
    def _seam(axis: int, P: int) -> float:
        d = np.abs(np.diff(x, axis=axis))
        other = tuple(i for i in range(x.ndim) if i != axis)
        prof = d.mean(axis=other)                      # (N-1,) mean |diff| per position
        L = prof.shape[0]
        # diff index i sits between bins i and i+1, so patch boundaries are at i = k*P - 1.
        bnd = np.arange(1, (L + 1) // P) * P - 1
        is_b = np.zeros(L, dtype=bool)
        is_b[bnd] = True
        vals = []
        for b in bnd:
            sel = np.arange(max(0, b - ring), min(L, b + ring + 1))
            sel = sel[~is_b[sel]]
            if sel.size and prof[sel].mean() > 0:
                vals.append(prof[b] / prof[sel].mean())
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "patch_lattice_ratio": lattice_ratio,
        "lattice_energy_frac": lattice_frac,
        "seam_ratio_freq": _seam(x.ndim - 2, patch_f),
        "seam_ratio_time": _seam(x.ndim - 1, patch_t),
    }


def remove_patch_lattice(spec: np.ndarray, patch_f: int, patch_t: int) -> np.ndarray:
    """Zero the patch-lattice DFT bins of ``spec`` (B, C, F, T) — an ideal lattice notch.

    DIAGNOSTIC / RENDER-TIME ONLY. It is not part of any training loss and no codec calls it;
    it exists to answer "how much of this reconstruction IS the artifact?" by deleting exactly
    the bins :func:`patch_lattice_metrics` scores and re-measuring everything else.

    MEASURED on the mhr v1k ms2_s1 baseline (32 held-out windows, patch 16x16):

        recon            hf_ratio 0.8204   env_corr 0.6966   peak_f1 0.6146   pixel-L1 1.3071
        recon NOTCHED    hf_ratio 0.0634   env_corr 0.7474   peak_f1 0.6704   pixel-L1 1.1306

    i.e. 92 % of that codec's high-frequency gradient energy — the thing ``sharpness``
    measures — is the patch lattice, and deleting it IMPROVES envelope correlation (+0.051),
    peak F1 (+0.056) and pixel L1 (-13 %). Read the ``sharpness`` / "HF ratio" of a
    lattice-bearing codec with that in mind: it is mostly counting the artifact.
    """
    x = np.asarray(spec, dtype=np.float64)
    F, T = x.shape[-2], x.shape[-1]
    keep = np.ones((F, T), dtype=np.float64)
    for a, b in _lattice_harmonics(F, T, patch_f, patch_t):
        keep[a, b] = 0.0
    return np.real(np.fft.ifft2(np.fft.fft2(x, axes=(-2, -1)) * keep, axes=(-2, -1)))


# ---------------------------------------------------------------------------------------
# 4c. FULL-SPECTROGRAM reconstruction metrics — the honest recon check (NO axis collapsed)
# ---------------------------------------------------------------------------------------
# WHY THIS EXISTS. The three metrics above all look at a TIME-COLLAPSED view or at a single
# scalar ratio, so none of them compares the spectrogram to the spectrogram:
#   * ``envelope_corr`` / ``peak_f1`` run on ``_power_envelope`` = ``spec.mean(axis=-1)``, which
#     AVERAGES THE WHOLE 50 ms TIME AXIS AWAY. What survives is a 512-number "which frequencies
#     carry power" profile per (window, channel); every mode track, burst edge and chirp is gone
#     before the correlation is taken.
#   * ``sharpness`` is a single HF-gradient-energy RATIO with ideal 1.0, and it is dominated by
#     the patch-lattice artifact: notching only the lattice DFT bins of the best mhr codec moves
#     it 0.729 -> 0.044 while IMPROVING envelope_corr (see :func:`remove_patch_lattice`).
# The two metrics below operate on the full ``(F, T)`` array per (window, channel).
#
# READ THEM WITH THE TRIVIAL BASELINES of :func:`trivial_spectro_baselines`, which are in the
# SAME units — a codec number is meaningless until you know what the time-averaged envelope
# alone (what ``envelope_corr`` rewards) already scores.
#
# MEASURED 2026-09-02 — all five saved mhr codecs, 360 held-out windows of shots 204984-204987
# (analysis/render_codec_recon_figs.py --audit; rows in
#  eval_runs/codec_recon_figs/mhr_fullspec_rescore.json):
#
#     arm         spec_nrmse  spec_corr2d   nrmse_band  corr2d_band  | envelope_corr (old)
#     production      1.1333       0.3236       1.2822       0.2048  |        0.6968
#     je1             1.0226       0.2876       1.1246       0.1868  |        0.6789
#     ms2_s1          1.0990       0.2721       1.2150       0.1775  |        0.6997
#     j1ms2           1.0055       0.3206       1.1276       0.1861  |        0.6989
#     refd4_s1        1.2530       0.2666       1.6532       0.1842  |        0.6709
#     ~tmean baseline 0.8799       0.4443       0.9038       0.3949  |   1.0000 (by constr.)
#     ~wcmean anchor  1.0000       0.0000       1.0580       0.0000
#
# EVERY codec reads spec_nrmse >= 1.0, i.e. its reconstruction is FARTHER from the target than
# that window's own constant mean, and every codec is beaten on BOTH metrics by the trivial
# time-averaged-envelope predictor (0.8799 / 0.4443) — the very thing envelope_corr scores 1.0.
# corr2d 0.27-0.32 means 7-10 % of the spectrogram's variance is explained; the gap to
# envelope_corr ~0.70 is the time axis that envelope_corr deletes. The result is not an offset
# or gain error either: the best nrmse reachable with a free per-window affine rescale is
# sqrt(1 - corr2d^2) = 0.946-0.964, still worse than the 0.8799 baseline.

#: Mode-band cut for the ``*_band`` variants: STFT bin index (exclusive) of the upper edge.
#: The spectro grid is 512 bins over 0-250 kHz (STFT_FS 500 kHz / 2), i.e. 488.28 Hz per bin,
#: so the default 120 bins = 0-58.6 kHz — the band that carries the coherent MHD/mode tracks.
DEFAULT_MODE_BAND_BINS: int = 120

_SPEC_STD_EPS: float = 1e-8


def _spectro_mask_bct(mask, shape) -> Optional[np.ndarray]:
    """Normalise a spectro validity mask to a ``(B, C, T)`` bool array, or None.

    Accepts ``(B, T)`` (broadcast over channels) or ``(B, C, T)``; ``shape`` is the
    ``(B, C, F, T)`` spectrogram shape. Returns None when the mask is absent OR selects
    everything, so every caller then takes the ORIGINAL unmasked code path and the default
    is bit-identical. Mirrors ``_video_mask_bct``.
    """
    if mask is None:
        return None
    B, C, _F, T = shape
    m = _to_numpy(mask)
    if m.ndim == 2:
        m = np.broadcast_to(m[:, None, :], (B, C, T))
    if m.shape != (B, C, T):
        raise ValueError(f"spectro mask must be (B,T) or (B,C,T)=({B},{C},{T}); got {m.shape}")
    m = m > 0.5
    return None if m.all() else m


def _nrmse_corr2d(recon: np.ndarray, target: np.ndarray, mask=None):
    """(nrmse, corr2d, valid_frac) over the FULL (F, T) array, per (window, channel).

    Both arrays are (B, C, F, T). For each of the ``B*C`` (window, channel) pairs, with
    ``N = F*T`` elements::

        nrmse = sqrt( mean_N (recon - target)^2 ) / std_N(target)
        corr2d = Pearson(recon.flatten(), target.flatten())

    Pairs whose TARGET is constant (``std <= 1e-8``: an absent / dead diagnostic channel)
    carry no reconstructable structure and would divide by zero; they are EXCLUDED from both
    means and counted in ``valid_frac`` instead (measured 1.000 on the 4 held-out mhr shots,
    i.e. no dead channel there — but other modalities do carry them). A pair whose RECON is
    constant while its target is not (mean collapse) IS counted: corr2d 0.0 and nrmse >= 1.0
    (exactly 1.0 when that constant happens to be the target's own mean), which is the honest
    reading.
    """
    B, C, F, T = recon.shape
    r = recon.reshape(B * C, -1)
    t = target.reshape(B * C, -1)
    m = _spectro_mask_bct(mask, recon.shape)
    if m is None:
        # UNMASKED path, untouched: every statistic over all F*T elements.
        w = None
        n = np.full(B * C, float(r.shape[1]))
        rmean = r.mean(axis=1, keepdims=True)
        tmean = t.mean(axis=1, keepdims=True)
    else:
        # MASKED path. Missingness is per (window, channel, FRAME) and has no frequency
        # structure, so the (B, C, T) mask broadcasts over F. EVERY statistic below --
        # each side's own mean, the target std that normalises the RMSE, the RMSE itself
        # and both correlation sums -- is then taken over VALID elements only, which is the
        # standard the slow-TS gate already meets.
        w = np.broadcast_to(m[:, :, None, :], (B, C, F, T)).reshape(B * C, -1).astype(np.float64)
        n = w.sum(axis=1)
        safe = np.maximum(n, 1.0)[:, None]
        rmean = (r * w).sum(axis=1, keepdims=True) / safe
        tmean = (t * w).sum(axis=1, keepdims=True) / safe
    rc = r - rmean
    tc = t - tmean
    if w is None:
        t_std = np.sqrt((tc * tc).mean(axis=1))
        rmse = np.sqrt(((r - t) ** 2).mean(axis=1))
        num = (rc * tc).sum(axis=1)
        denom = np.sqrt((rc * rc).sum(axis=1) * (tc * tc).sum(axis=1))
    else:
        rc, tc = rc * w, tc * w                      # zero the invalid elements everywhere
        safe1 = np.maximum(n, 1.0)
        t_std = np.sqrt((tc * tc).sum(axis=1) / safe1)
        rmse = np.sqrt((((r - t) ** 2) * w).sum(axis=1) / safe1)
        num = (rc * tc).sum(axis=1)
        denom = np.sqrt((rc * rc).sum(axis=1) * (tc * tc).sum(axis=1))
    # A pair needs a non-constant target AND (masked) at least 2 real elements to score.
    valid = (t_std > _SPEC_STD_EPS) & (n >= 2)
    if not valid.any():
        return float("nan"), float("nan"), 0.0
    nrmse = rmse[valid] / t_std[valid]
    # denom == 0 means one side is constant: no structure to correlate -> 0.0 (the same
    # convention _envelope_correlation uses). np.maximum only guards the divide-by-zero
    # that np.where would still evaluate.
    corr = np.where(denom > 0.0, num / np.maximum(denom, 1e-300), 0.0)
    return float(nrmse.mean()), float(corr[valid].mean()), float(valid.mean())


def full_spectro_metrics(
    recon,
    target,
    band_bins: Optional[int] = DEFAULT_MODE_BAND_BINS,
    prefix: str = "",
    mask=None,
) -> Dict[str, float]:
    """FULL-array (B, C, F, T) reconstruction fidelity — no axis averaged away.

    Args:
        recon:  (B, C, F, T) reconstructed log-power spectrogram.
        target: (B, C, F, T) ground-truth log-power spectrogram.
        band_bins: if not None, ALSO report the same two metrics restricted to frequency bins
            ``[0, band_bins)`` (default :data:`DEFAULT_MODE_BAND_BINS` = 120 bins = 0-58.6 kHz,
            the coherent-mode band). Reported SEPARATELY with a ``_band`` suffix; it never
            changes the full-band numbers.
        prefix: prepended to every key (used for the trivial baselines).

    Returns (keys prefixed by ``prefix``):
        ``spec_nrmse``   — RMSE(recon, target) / std(target), computed per (window, channel)
            over the flattened (F, T) array and then averaged over pairs.
            NORMALISATION, stated explicitly: the divisor is the standard deviation of THAT
            SAME (window, channel) target, not a global std and not its range. That choice
            makes the scale of the metric an interpretable baseline rather than an arbitrary
            unit: predicting the per-(window, channel) constant mean scores EXACTLY 1.0, so
            ``spec_nrmse < 1`` = "beats the window's own DC level", ``>= 1`` = "does not".
            0.0 = perfect. It is invariant to the log-power offset and gain of each window,
            which matters because these windows are z-scored per modality, not per window.
        ``spec_corr2d`` — Pearson correlation over the flattened (F, T) array per
            (window, channel), averaged over pairs. 1.0 = perfect; this is the honest
            analogue of ``envelope_corr``, which computes the same statistic AFTER deleting
            the time axis. Expect ``spec_corr2d`` << ``envelope_corr`` for a codec that only
            reproduces the time-averaged spectrum.
        ``spec_valid_frac`` — fraction of (window, channel) pairs with a non-constant target
            (the rest are absent/dead channels and are excluded from both means above).
        ``spec_nrmse_band`` / ``spec_corr2d_band`` — same two metrics on bins
            ``[0, band_bins)`` only (present iff ``band_bins`` is not None).

    Validated in ``tests/ignite/test_gate.py``: target vs itself gives nrmse 0.0 / corr2d 1.0,
    and a mean-collapsed recon gives nrmse 1.0 / corr2d 0.0.
    """
    r = _to_numpy(recon).astype(np.float64)
    t = _to_numpy(target).astype(np.float64)
    if r.shape != t.shape:
        raise ValueError(f"full_spectro_metrics: shape mismatch {r.shape} vs {t.shape}")
    if r.ndim != 4:
        raise ValueError(f"full_spectro_metrics expects (B, C, F, T); got {r.shape}")
    nrmse, corr, valid = _nrmse_corr2d(r, t, mask)
    out = {
        f"{prefix}spec_nrmse": nrmse,
        f"{prefix}spec_corr2d": corr,
        f"{prefix}spec_valid_frac": valid,
    }
    if band_bins is not None:
        k = int(min(int(band_bins), r.shape[-2]))
        b_nrmse, b_corr, _ = _nrmse_corr2d(r[..., :k, :], t[..., :k, :], mask)
        out[f"{prefix}spec_nrmse_band"] = b_nrmse
        out[f"{prefix}spec_corr2d_band"] = b_corr
    return out


def trivial_spectro_baselines(
    target,
    band_bins: Optional[int] = DEFAULT_MODE_BAND_BINS,
    mask=None,
) -> Dict[str, float]:
    """The reference points that make :func:`full_spectro_metrics` interpretable.

    Every baseline is scored with the SAME function and the same units, against the same
    target batch, so the codec row and the baseline rows are directly comparable:

      * ``base_self_*``   — the target predicting itself. MUST read nrmse 0.0 / corr2d 1.0;
        it is the self-check that the metric is wired correctly.
      * ``base_tmean_*``  — the target's own time-averaged envelope ``target.mean(-1)``
        broadcast back over all T frames. This is the "perfect envelope, zero temporal
        structure" predictor: it is EXACTLY what ``envelope_corr`` scores 1.0 and what a codec
        that only reproduces the per-frequency power profile achieves. Its ``spec_corr2d`` is
        the bar any honest reconstruction has to clear.
      * ``base_cfmean_*`` — the per-(channel, frequency) mean over the whole evaluated batch
        AND over time, broadcast back. A single dataset-level spectrum, no per-window
        information at all.
      * ``base_wcmean_*`` — the per-(window, channel) scalar mean broadcast back. Scores
        nrmse EXACTLY 1.0 by construction (that is the normalisation anchor) and corr2d 0.0.
        Its ``_band`` variant is only APPROXIMATELY 1.0 (measured 1.025 on synthetic data)
        because the predicted constant is the whole window's mean while the band metric
        normalises by the in-band std — the anchor is exact for the full-band numbers only.

    Returns one flat dict with all four sets of keys.
    """
    t = _to_numpy(target).astype(np.float64)
    if t.ndim != 4:
        raise ValueError(f"trivial_spectro_baselines expects (B, C, F, T); got {t.shape}")
    mb = _spectro_mask_bct(mask, t.shape)
    if mb is None:
        tmean = np.broadcast_to(t.mean(axis=-1, keepdims=True), t.shape)
        cfmean = np.broadcast_to(t.mean(axis=(0, -1))[None, :, :, None], t.shape)
        wcmean = np.broadcast_to(t.mean(axis=(-2, -1), keepdims=True), t.shape)
    else:
        # EACH BASELINE'S OWN MEAN over valid samples only -- otherwise the reference points
        # that make the codec row interpretable are themselves computed on fill, and
        # base_wcmean stops being the exact 1.0 anchor the normalisation is defined by.
        w = np.broadcast_to(mb[:, :, None, :], t.shape).astype(np.float64)
        tw = t * w
        tmean = tw.sum(axis=-1, keepdims=True) / np.maximum(w.sum(axis=-1, keepdims=True), 1.0)
        tmean = np.broadcast_to(tmean, t.shape)
        cfnum = tw.sum(axis=(0, -1))[None, :, :, None]
        cfden = np.maximum(w.sum(axis=(0, -1))[None, :, :, None], 1.0)
        cfmean = np.broadcast_to(cfnum / cfden, t.shape)
        wcnum = tw.sum(axis=(-2, -1), keepdims=True)
        wcden = np.maximum(w.sum(axis=(-2, -1), keepdims=True), 1.0)
        wcmean = np.broadcast_to(wcnum / wcden, t.shape)
    out: Dict[str, float] = {}
    for name, pred in (("self", t), ("tmean", tmean), ("cfmean", cfmean), ("wcmean", wcmean)):
        m = full_spectro_metrics(pred, t, band_bins=band_bins, prefix=f"base_{name}_",
                                 mask=mask)
        m.pop(f"base_{name}_spec_valid_frac", None)  # identical for every baseline (same target)
        out.update(m)
    return out


def decode_fidelity(recon, target, peak_k: float = 1.0,
                    patch_f: Optional[int] = None,
                    patch_t: Optional[int] = None,
                    full_spec: bool = False,
                    band_bins: Optional[int] = DEFAULT_MODE_BAND_BINS,
                    mode_baseline: bool = True,
                    detrended: bool = False,
                    mask=None) -> Dict[str, float]:
    # ``mode_baseline`` gates EVERY target-only baseline (the trivial spectro baselines and the
    # time-mean envelope's mode-structure scores). They are identical for all arms on a given
    # window set, so the audit computes them for the first arm and shares them across rows.
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

    When ``patch_f`` / ``patch_t`` are given, the PATCH-LATTICE artifact metrics of
    :func:`patch_lattice_metrics` are added for BOTH the reconstruction and the target
    (``patch_lattice_ratio`` / ``lattice_energy_frac`` / ``seam_ratio_freq`` /
    ``seam_ratio_time``, plus the same four with a ``target_`` prefix). The target values are
    the CONTROL: they say what the metric reads on data with no codec in the loop, so a
    reconstruction number is only meaningful beside them. None of the three metrics above can
    see a patch lattice — the lattice IS high-frequency gradient energy, so it can even
    INFLATE ``sharpness`` — which is exactly why they are reported together.

    When ``full_spec=True`` the FULL-ARRAY metrics of :func:`full_spectro_metrics` are added
    (``spec_nrmse`` / ``spec_corr2d`` / ``spec_valid_frac`` and the ``_band`` variants),
    together with the trivial baselines of :func:`trivial_spectro_baselines` (``base_self_*``,
    ``base_tmean_*``, ``base_cfmean_*``, ``base_wcmean_*``) in the same units. These are the
    only keys here that compare the spectrogram to the spectrogram; the three metrics above
    all collapse the time axis first. DEFAULT OFF, so every existing caller returns a
    byte-identical dict.

    Args:
        recon:  (B, C, F, T) reconstructed log-power spectrogram.
        target: (B, C, F, T) ground-truth log-power spectrogram.
        peak_k: std-multiplier threshold for the peak/mode detector.
        patch_f / patch_t: codec patch size. Both None (default) = skip the lattice metrics
            (keys absent), so every existing caller is unchanged.
        full_spec: add the full-spectrogram metrics + trivial baselines (default False).
        band_bins: mode-band cut passed through to :func:`full_spectro_metrics` when
            ``full_spec`` is on; None disables the ``_band`` variants.
        mask: optional ``(B, T)`` / ``(B, C, T)`` validity mask (see
            ``data.spectro_frame_mask``). When given, ``spec_nrmse`` / ``spec_corr2d`` and
            EVERY trivial baseline -- including each baseline's own mean -- are taken over
            VALID samples only, which is the standard the slow-TS gate already meets. Also
            excludes fully-invalid windows from ``envelope_corr`` / ``peak_f1`` / the
            sharpness and patch-lattice energies, so no statistic is computed on fill.
            ``None`` (default) = every existing caller returns a byte-identical dict.

    Returns:
        dict with ``envelope_corr``, ``peak_f1``, ``sharpness``, the raw
        ``hf_energy_recon`` / ``hf_energy_target``, — when the patch size is given — the
        patch-lattice keys described above, and — when ``full_spec`` is on — the
        full-spectrogram keys and their baselines.
    """
    r = _to_numpy(recon).astype(np.float64)
    t = _to_numpy(target).astype(np.float64)
    if r.shape != t.shape:
        raise ValueError(f"decode_fidelity: shape mismatch {r.shape} vs {t.shape}")
    if r.ndim != 4:
        raise ValueError(
            f"decode_fidelity expects (B, C, F, T); got {r.shape}"
        )

    # WINDOW-LEVEL exclusion for the structure metrics. `_envelope_correlation`,
    # `_peak_overlap_f1`, `_hf_gradient_energy` and `patch_lattice_metrics` all operate on the
    # (F, T) plane and cannot take a per-frame mask, so a window with ANY missing channel is
    # dropped whole -- the same rule the trainer's loss uses (SpectroCodec._valid_windows).
    # Spectro missingness is a per-(shot, channel) property, so this loses almost nothing.
    _mb = _spectro_mask_bct(mask, r.shape)
    if _mb is not None:
        _keep = _mb.all(axis=(1, 2))                  # (B,) fully-valid windows
        if _keep.any():
            r, t = r[_keep], t[_keep]
            mask = _mb[_keep]
        # if NOTHING is fully valid, fall through on the full arrays rather than emit NaNs.

    env_corr = _envelope_correlation(r, t)
    peak_f1 = _peak_overlap_f1(r, t, k=peak_k)
    hf_r = _hf_gradient_energy(r)
    hf_t = _hf_gradient_energy(t)
    sharpness = float(hf_r / hf_t) if hf_t > 1e-12 else float("nan")

    out = {
        "envelope_corr": env_corr,
        "peak_f1": peak_f1,
        "sharpness": sharpness,
        "hf_energy_recon": hf_r,
        "hf_energy_target": hf_t,
    }
    if patch_f is not None and patch_t is not None:
        out.update(patch_lattice_metrics(r, int(patch_f), int(patch_t)))
        for k, v in patch_lattice_metrics(t, int(patch_f), int(patch_t)).items():
            out[f"target_{k}"] = v
    if full_spec:
        out.update(full_spectro_metrics(r, t, band_bins=band_bins, mask=mask))
        if mode_baseline:
            # trivial_spectro_baselines depends ONLY on the target, so it is identical for
            # every arm scored against the same windows -- computing it per arm was ~9.5 s per
            # chunk of pure duplication (profiled; it is the single most expensive call here).
            out.update(trivial_spectro_baselines(t, band_bins=band_bins, mask=mask))
        # MODE-TRACK metrics (see the section at the end of this module). Reported over the
        # FULL band, not `band_bins`: the mhr mode tracks run to ~90 kHz, above the 58.6 kHz
        # mode-band cut, and the deliverable is the whole 0-250 kHz range. These are the keys
        # arms are RANKED on -- `spec_nrmse` ranks the blur best and cannot be the ranking key.
        out.update(mode_structure_metrics(r, t, band_bins=None, detrended=detrended))
        if mode_baseline:
            # The time-mean envelope's own mode-structure scores. Identical for every arm on a
            # given target, so the audit computes it for the FIRST arm only -- doing it per arm
            # doubled the metric cost for no extra information.
            _tm = np.broadcast_to(t.mean(axis=-1, keepdims=True), t.shape)
            out.update(mode_structure_metrics(_tm, t, band_bins=None, prefix="base_tmean_",
                                              detrended=detrended))
    return out


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
        threshold on it fires spuriously even for a perfectly healthy codec. Even when
        ``codebook_size <= n_observed`` the ceiling is a coupon-collector bound, not 1.0: with
        ``M`` tokens drawn uniformly from ``K`` codes the expected distinct count is
        ``K(1 - (1 - 1/K)^M)``, i.e. 900/1000 at the default M=2304 eval size — so
        ``frac_codes_used = 0.90`` at cb=1000 means essentially PERFECTLY uniform usage.
      * ``code_entropy_nats`` / ``effective_codes`` — the JOINT code entropy and its
        perplexity ``e^H``. The per-dim entropies above are MARGINAL and a rank-1 encoder
        maximizes all of them while using almost no joint codes, so this is the signal that
        catches that failure. ``dim_redundancy_nats`` = ``sum(dim_entropy_nats) -
        code_entropy_nats`` is 0 for independent dims and large when they replicate.

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

    # --- JOINT code entropy: the quantity the per-dim entropies CANNOT see ---------------
    # The per-dim entropies are MARGINAL. A rank-1 encoder (all FSQ dims a copy of one scalar)
    # drives every marginal to its ceiling while the JOINT distribution stays tiny — measured
    # on the prod mhr codec: min_dim_entropy 0.921 (93% of ln 8) with a joint entropy of only
    # 2.656 nats = 14.2 effective codes out of 32768. ``effective_codes`` = exp(H_joint) is the
    # perplexity, i.e. the size of the uniform codebook that would have this much diversity, and
    # is the honest companion to ``n_distinct_codes`` (which counts a code used ONCE the same as
    # one used a thousand times). ``dim_redundancy_nats`` = sum(marginal H) - H_joint is 0 for
    # independent dims and large when they replicate each other.
    dim_entropy_nats = []
    for i in range(fsq_dim):
        counts = np.bincount(flat[:, i], minlength=levels[i]).astype(np.float64)
        probs = counts / counts.sum()
        nz = probs[probs > 0]
        dim_entropy_nats.append(float(-(nz * np.log(nz)).sum()))
    _, joint_counts = np.unique(flat, axis=0, return_counts=True)
    jp = joint_counts.astype(np.float64) / joint_counts.sum()
    code_entropy_nats = float(-(jp * np.log(jp)).sum())

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
        "code_entropy_nats": code_entropy_nats,      # JOINT code entropy (nats)
        "effective_codes": float(np.exp(code_entropy_nats)),   # perplexity e^H_joint
        "dim_entropy_nats": dim_entropy_nats,        # per-dim MARGINAL entropies (nats)
        "dim_redundancy_nats": float(sum(dim_entropy_nats) - code_entropy_nats),
    }


# --------------------------------------------------------------------------------------
# 6. full_slowts_metrics — the SLOW-TS analogue of full_spectro_metrics (2026-09-03)
#
# Same normalisation CONVENTION as full_spectro_metrics, deliberately not a new one: RMSE
# divided by the standard deviation of THAT SAME target group, so predicting the group's own
# constant mean scores EXACTLY 1.0000 and "< 1.0" means "beats that group's own DC level".
#
# The GROUP differs, because the arrays differ. A spectrogram is (B, C, F, T) and the spectro
# metric groups per (window, channel), reducing over the (F, T) structure+time plane. A slow-TS
# window is (B, C, T) where C (profile position) is the STRUCTURE axis — the analogue of F, not
# of the spectro channel axis — and T is time. So the slow-TS metric groups per WINDOW and
# reduces over the (C, T) plane. Grouping per (window, position) instead would reduce over T
# alone (5 samples) and make `tmean` identical to `wcmean`, i.e. collapse the two reference
# points the metric exists to separate.
#
# MASKING. Slow-TS is the one MASKED codec family: Thomson zero_is_missing zeros and CER/MSE
# beam-off NaNs are filled with 0 in both the input and the target, and for ts_core_density the
# MEDIAN window is ~7% present. An unmasked RMSE would therefore be dominated by how well the
# codec reproduces a constant fill value. Every statistic below (mean, std, RMSE, correlation,
# and every baseline's own mean) is taken over VALID samples only when a mask is given.
# --------------------------------------------------------------------------------------
_SLOWTS_STD_EPS = 1e-8
_SLOWTS_MIN_VALID = 2      # a window needs >= 2 valid samples to have a std at all


def _slowts_nrmse_corr(recon: np.ndarray, target: np.ndarray, mask: Optional[np.ndarray]):
    """(nrmse, corr, valid_frac, present_frac) per WINDOW over the flattened (C, T) plane.

    All arrays are (B, C, T). For each of the ``B`` windows, over its VALID samples only::

        nrmse = sqrt( mean_valid (recon - target)^2 ) / std_valid(target)
        corr  = Pearson(recon, target) over the same valid samples

    Windows whose target is constant over its valid samples (``std <= 1e-8``: a dead / absent
    profile) carry no reconstructable structure and would divide by zero; they are EXCLUDED
    from both means and counted in ``valid_frac``. ``present_frac`` is the mean validity-mask
    occupancy, reported because a slow-TS number is uninterpretable without knowing how much
    of the window was actually measured.
    """
    B = recon.shape[0]
    r = recon.reshape(B, -1).astype(np.float64)
    t = target.reshape(B, -1).astype(np.float64)
    if mask is None:
        m = np.ones_like(t)
    else:
        m = (mask.reshape(B, -1) > 0.5).astype(np.float64)
    n = m.sum(axis=1)                                             # (B,) valid samples per window
    present_frac = float(m.mean())
    ok = n >= _SLOWTS_MIN_VALID
    safe_n = np.maximum(n, 1.0)
    t_mu = (t * m).sum(axis=1) / safe_n
    r_mu = (r * m).sum(axis=1) / safe_n
    tc = (t - t_mu[:, None]) * m
    rc = (r - r_mu[:, None]) * m
    t_std = np.sqrt((tc * tc).sum(axis=1) / safe_n)
    rmse = np.sqrt((((r - t) ** 2) * m).sum(axis=1) / safe_n)
    valid = ok & (t_std > _SLOWTS_STD_EPS)
    if not valid.any():
        return (float("nan"), float("nan"), float("nan"), float("nan"), 0.0, present_frac)
    nrmse = rmse[valid] / t_std[valid]
    denom = np.sqrt((rc * rc).sum(axis=1) * (tc * tc).sum(axis=1))
    corr = np.where(denom > 0.0, (rc * tc).sum(axis=1) / np.maximum(denom, 1e-300), 0.0)
    # POOLED (variance-weighted) nRMSE — see the note in full_slowts_metrics. Same 1.0 anchor,
    # but it sums the squared error and the squared deviation over ALL windows before dividing,
    # so a nearly-flat window (tiny t_std) contributes its tiny error to the numerator instead
    # of a huge ratio to the mean. Equals sqrt(1 - R^2) over the pooled valid samples.
    sse = (((r - t) ** 2) * m).sum(axis=1)[valid].sum()
    sst = ((tc * tc).sum(axis=1))[valid].sum()
    pooled = float(np.sqrt(sse / sst)) if sst > 0 else float("nan")
    return (float(nrmse.mean()), float(np.median(nrmse)), pooled, float(corr[valid].mean()),
            float(valid.mean()), present_frac)


def _slowts_std_ratio(recon: np.ndarray, target: np.ndarray, mask: Optional[np.ndarray]):
    """AMPLITUDE diagnostic: std(recon) / std(target), ideal EXACTLY 1.0. (B, C, T) arrays.

    WHY THIS EXISTS (2026-09-03). ``slowts_nrmse`` is a squared error, and the exact minimizer
    of a squared error is the CONDITIONAL MEAN — so an under-capacity decoder minimises it by
    shrinking toward the mean, and a variance-collapsed reconstruction is *rewarded* rather than
    punished. Measured on the retrained codecs: ``cer_ti``'s reconstruction swings 0.2 -> 0.7
    where the ground truth swings 0 -> 2.3, and ``mse``'s is a near-flat line against a +-3.5
    oscillating target — both score a respectable nrmse (0.8752 / 0.7956) because a shrunken
    prediction has a small squared error. Nothing in the nRMSE/corr pair reports it: correlation
    is scale-INVARIANT and nRMSE is scale-*preferring*. This function makes the collapse a
    NUMBER instead of a visual judgement.

    It is a DIAGNOSTIC, never a ranking key: 1.0 is achievable by pure noise of the right
    amplitude, so it is only meaningful read beside ``slowts_nrmse`` / ``slowts_corr``.
    Interpretation: << 1 = dynamic-range compression (the failure mode above); >> 1 = the
    reconstruction invents amplitude the target does not have.

    Two GROUPINGS are returned because they answer different questions:

      * per-WINDOW over the flattened (C, T) plane — the SAME grouping as ``slowts_nrmse``, so
        the two numbers are directly comparable. Mixes profile-shape spread with temporal
        spread.
      * per-(WINDOW, CHANNEL) over the TIME axis only — the pure TEMPORAL amplitude. This is
        the one that catches a flat line whose profile shape is right. It is variance-WEIGHTED
        across rows rather than a mean of per-row ratios because a slow-TS window is only
        T=5 samples at 100 Hz, so a single row's std is far too noisy to average as a ratio.

    Returns ``(ratio_mean, ratio_med, ratio_pooled, ratio_time)``; all NaN when no window
    qualifies. Rows/windows need >= ``_SLOWTS_MIN_VALID`` valid samples and a non-constant
    target (the ``slowts_nrmse`` convention) to contribute.
    """
    B = recon.shape[0]
    r = recon.reshape(B, -1).astype(np.float64)
    t = target.reshape(B, -1).astype(np.float64)
    m = (np.ones_like(t) if mask is None
         else (mask.reshape(B, -1) > 0.5).astype(np.float64))
    n = m.sum(axis=1)
    safe_n = np.maximum(n, 1.0)
    t_mu = (t * m).sum(axis=1) / safe_n
    r_mu = (r * m).sum(axis=1) / safe_n
    ss_t = ((((t - t_mu[:, None]) * m) ** 2)).sum(axis=1)
    ss_r = ((((r - r_mu[:, None]) * m) ** 2)).sum(axis=1)
    valid = (n >= _SLOWTS_MIN_VALID) & (np.sqrt(ss_t / safe_n) > _SLOWTS_STD_EPS)
    if not valid.any():
        return (float("nan"),) * 4
    ratio = np.sqrt(ss_r[valid] / ss_t[valid])
    pooled = float(np.sqrt(ss_r[valid].sum() / ss_t[valid].sum()))

    # --- per-(window, channel) over the TIME axis only -------------------------------
    r3 = recon.astype(np.float64)
    t3 = target.astype(np.float64)
    m3 = (np.ones_like(t3) if mask is None else (mask > 0.5).astype(np.float64))
    n_ct = m3.sum(axis=-1)                                        # (B, C)
    safe_ct = np.maximum(n_ct, 1.0)
    t_mu_ct = (t3 * m3).sum(axis=-1) / safe_ct
    r_mu_ct = (r3 * m3).sum(axis=-1) / safe_ct
    ss_t_ct = ((((t3 - t_mu_ct[..., None]) * m3) ** 2)).sum(axis=-1)
    ss_r_ct = ((((r3 - r_mu_ct[..., None]) * m3) ** 2)).sum(axis=-1)
    ok = (n_ct >= _SLOWTS_MIN_VALID) & (np.sqrt(ss_t_ct / safe_ct) > _SLOWTS_STD_EPS)
    if ok.any() and ss_t_ct[ok].sum() > 0:
        ratio_time = float(np.sqrt(ss_r_ct[ok].sum() / ss_t_ct[ok].sum()))
    else:
        ratio_time = float("nan")
    return float(ratio.mean()), float(np.median(ratio)), pooled, ratio_time


def full_slowts_metrics(recon, target, mask=None, prefix: str = "") -> Dict[str, float]:
    """FULL-array (B, C, T) slow-TS reconstruction fidelity — no axis averaged away.

    The slow-TS analogue of :func:`full_spectro_metrics`, with the SAME normalisation
    convention (see the section note above). Unlike ``slowts_decode_fidelity``'s
    ``envelope_corr``, which deletes the time axis first and therefore scores 1.0 for a
    predictor with zero temporal structure, this keeps every sample.

    Args:
        recon:  (B, C, T) reconstructed slow-TS window (in the codec's standardized space).
        target: (B, C, T) ground-truth window (same space).
        mask:   optional (B, C, T) validity mask (1 = real sample). Every statistic is taken
                over valid samples only; ``None`` = all valid.
        prefix: prepended to every key (used for the trivial baselines).

    Returns (keys prefixed by ``prefix``):
        ``slowts_nrmse``       — RMSE/std(target) per window over the flattened (C, T) plane,
            averaged over windows. 0.0 = perfect; EXACTLY 1.0 = predicting that window's own
            constant (valid-sample) mean; > 1.0 = worse than that constant.
        ``slowts_nrmse_med``   — the MEDIAN of the same per-window ratios. Reported because the
            per-window MEAN has a heavy tail that the spectrogram version does not: a slow-TS
            window whose valid samples happen to be nearly flat has a near-zero divisor, so a
            handful of such windows dominate the mean (measured on prod ts_core_density: mean
            2.6592 while the typical window sits near 1.0).
        ``slowts_nrmse_pooled``— the VARIANCE-WEIGHTED form: sqrt( sum_windows SSE /
            sum_windows SST ), i.e. sqrt(1 - R^2) over the pooled valid samples. Same
            convention and the SAME exact 1.0 anchor (predicting each window's own mean gives
            exactly 1.0), but a flat window contributes its small error to the numerator rather
            than a huge ratio to an average. This is the robust headline number; the per-window
            mean above is kept because it is the spectro side's literal convention.
        ``slowts_corr``        — Pearson correlation over the same valid samples, per window,
            averaged. 1.0 = perfect.
        ``slowts_std_ratio``   — AMPLITUDE diagnostic std(recon)/std(target) per window over the
            same flattened (C, T) plane, averaged. IDEAL EXACTLY 1.0; << 1 means the
            reconstruction is dynamic-range COMPRESSED, which neither nRMSE (a squared error,
            minimised by shrinking toward the mean) nor corr (scale-invariant) reports. Never a
            ranking key — see :func:`_slowts_std_ratio`.
        ``slowts_std_ratio_med``    — median of the same per-window ratios.
        ``slowts_std_ratio_pooled`` — the variance-weighted form of the same (robust headline).
        ``slowts_std_ratio_t`` — the same ratio taken per (window, CHANNEL) over the TIME axis
            only, variance-weighted. The pure TEMPORAL amplitude: a reconstruction with the
            right profile shape but a flat time trace scores ~0 here and near-1 above. The
            ``tmean`` baseline reads EXACTLY 0.0 by construction (it has no time variation).
        ``slowts_valid_frac``  — fraction of windows with a non-constant target (the rest are
            absent/dead and excluded from both means).
        ``slowts_present_frac``— mean validity-mask occupancy of the evaluated windows.
    """
    r = _to_numpy(recon).astype(np.float64)
    t = _to_numpy(target).astype(np.float64)
    if r.shape != t.shape:
        raise ValueError(f"full_slowts_metrics: shape mismatch {r.shape} vs {t.shape}")
    if r.ndim != 3:
        raise ValueError(f"full_slowts_metrics expects (B, C, T); got {r.shape}")
    m = _to_numpy(mask) if mask is not None else None
    if m is not None and m.shape != r.shape:
        raise ValueError(f"full_slowts_metrics: mask shape {m.shape} != signal shape {r.shape}")
    nrmse, nrmse_med, pooled, corr, valid, present = _slowts_nrmse_corr(r, t, m)
    sr, sr_med, sr_pooled, sr_time = _slowts_std_ratio(r, t, m)
    return {
        f"{prefix}slowts_nrmse": nrmse,
        f"{prefix}slowts_nrmse_med": nrmse_med,
        f"{prefix}slowts_nrmse_pooled": pooled,
        f"{prefix}slowts_corr": corr,
        f"{prefix}slowts_std_ratio": sr,
        f"{prefix}slowts_std_ratio_med": sr_med,
        f"{prefix}slowts_std_ratio_pooled": sr_pooled,
        f"{prefix}slowts_std_ratio_t": sr_time,
        f"{prefix}slowts_valid_frac": valid,
        f"{prefix}slowts_present_frac": present,
    }


def _slowts_masked_mean(t: np.ndarray, m: Optional[np.ndarray], axis) -> np.ndarray:
    """Mean of ``t`` over ``axis`` using only ``m``-valid samples (0 where nothing is valid)."""
    if m is None:
        return t.mean(axis=axis, keepdims=True)
    num = (t * m).sum(axis=axis, keepdims=True)
    den = m.sum(axis=axis, keepdims=True)
    return np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)


def trivial_slowts_baselines(target, mask=None) -> Dict[str, float]:
    """The reference points that make :func:`full_slowts_metrics` interpretable.

    The slow-TS analogue of :func:`trivial_spectro_baselines`. Every baseline is scored with
    the SAME function, the same mask and the same units against the same target batch, so the
    codec row and the baseline rows are directly comparable:

      * ``base_self_*``   — the target predicting itself. MUST read nrmse 0.0 / corr 1.0; the
        self-check that the metric is wired correctly.
      * ``base_tmean_*``  — the target's own per-(window, position) time-averaged value,
        broadcast back over all T samples. This is the "perfect profile shape, ZERO temporal
        structure" predictor — exactly what ``slowts_decode_fidelity``'s ``envelope_corr``
        scores 1.0. **THE bar any honest slow-TS reconstruction has to clear.**
      * ``base_cmean_*``  — the per-POSITION mean over the whole evaluated batch and over time,
        broadcast back: one dataset-level profile, no per-window information at all. (The
        analogue of the spectro ``base_cfmean_*``.)
      * ``base_wcmean_*`` — the per-window scalar mean broadcast back. Scores nrmse EXACTLY
        1.0000 by construction — that is the normalisation anchor — and corr 0.0.
    """
    t = _to_numpy(target).astype(np.float64)
    if t.ndim != 3:
        raise ValueError(f"trivial_slowts_baselines expects (B, C, T); got {t.shape}")
    m = _to_numpy(mask).astype(np.float64) if mask is not None else None
    if m is not None:
        m = (m > 0.5).astype(np.float64)
    tmean = np.broadcast_to(_slowts_masked_mean(t, m, -1), t.shape)          # (B, C, 1)
    cmean = np.broadcast_to(_slowts_masked_mean(t, m, (0, -1)), t.shape)     # (1, C, 1)
    wcmean = np.broadcast_to(_slowts_masked_mean(t, m, (-2, -1)), t.shape)   # (B, 1, 1)
    out: Dict[str, float] = {}
    for name, pred in (("self", t), ("tmean", tmean), ("cmean", cmean), ("wcmean", wcmean)):
        mm = full_slowts_metrics(pred, t, mask=m, prefix=f"base_{name}_")
        mm.pop(f"base_{name}_slowts_valid_frac", None)    # identical for every baseline
        mm.pop(f"base_{name}_slowts_present_frac", None)  # (same target + mask)
        out.update(mm)
    return out


# --------------------------------------------------------------------------------------
# 7. full_fastts_metrics — the FAST-TS analogue of full_spectro_metrics (2026-09-03)
# --------------------------------------------------------------------------------------
# Same normalisation CONVENTION as full_spectro_metrics, deliberately not a new one: RMSE
# divided by the standard deviation of THAT SAME target group, so predicting the group's own
# constant mean scores EXACTLY 1.0000 and "< 1.0" means "beats that group's own DC level".
#
# The GROUP is the (window, channel) pair and the reduction is over the SAMPLE axis, exactly
# as specified for the raw fast-TS codec:
#
#     for each (window w, channel c):
#         nrmse[w, c] = RMSE(recon[w, c] - target[w, c]) / std(target[w, c])
#     result = mean over (w, c)
#
# A fast-TS window is (B, C, W) with W = 500 raw 10 kHz samples and no structure axis, so
# ``tmean`` (the time-average broadcast back over the window) and ``wcmean`` (the
# per-(window, channel) scalar mean) are THE SAME PREDICTOR here — unlike spectro, where
# tmean keeps the frequency profile. Both are reported anyway, and both MUST read exactly
# 1.0000: that identity is the metric's self-check, and a fast-TS codec beating "tmean" and
# beating "a flat line" are the same claim.
#
# WHAT THE NUMBER MEANS PHYSICALLY. The per-(window, channel) std of the standardized raw
# filterscope signal is ~0.007, while the between-window spread of the window MEAN is ~25x
# larger. So this metric deliberately ignores the (large, easy) light LEVEL and scores only
# the (small, broadband, hard) within-window waveform. ``fastts_global_nrmse`` is reported
# alongside as the complementary view in which the level DOES count.
_FASTTS_STD_EPS = 1e-8


def _fastts_nrmse_corr(recon: np.ndarray, target: np.ndarray):
    """(nrmse, corr, valid_frac) per (window, channel) over the SAMPLE axis.

    Both arrays are (B, C, W). Pairs whose TARGET is constant (``std <= 1e-8``: an absent /
    dead channel) carry no reconstructable structure and would divide by zero; they are
    EXCLUDED from both means and counted in ``valid_frac`` instead. A pair whose RECON is
    constant while its target is not IS counted: corr 0.0 and nrmse >= 1.0 (exactly 1.0 when
    that constant is the target's own mean), which is the honest reading.
    """
    B, C = recon.shape[0], recon.shape[1]
    r = recon.reshape(B * C, -1)
    t = target.reshape(B * C, -1)
    rc = r - r.mean(axis=1, keepdims=True)
    tc = t - t.mean(axis=1, keepdims=True)
    t_std = np.sqrt((tc * tc).mean(axis=1))
    rmse = np.sqrt(((r - t) ** 2).mean(axis=1))
    valid = t_std > _FASTTS_STD_EPS
    if not valid.any():
        return float("nan"), float("nan"), 0.0
    nrmse = rmse[valid] / t_std[valid]
    denom = np.sqrt((rc * rc).sum(axis=1) * (tc * tc).sum(axis=1))
    corr = np.where(denom > 0.0, (rc * tc).sum(axis=1) / np.maximum(denom, 1e-300), 0.0)
    return float(nrmse.mean()), float(corr[valid].mean()), float(valid.mean())


def full_fastts_metrics(recon, target, prefix: str = "") -> Dict[str, float]:
    """FULL-array (B, C, W) fast-TS RAW-SAMPLE reconstruction fidelity.

    The fast-TS analogue of :func:`full_spectro_metrics` / :func:`full_slowts_metrics`, with
    the SAME normalisation convention (see the section note above). Nothing is pooled,
    rectified or time-averaged first — this scores the raw 10 kHz samples.

    Args:
        recon:  (B, C, W) reconstructed raw window (in the codec's standardized space).
        target: (B, C, W) ground-truth raw window (same space).
        prefix: prepended to every key (used for the trivial baselines).

    Returns (keys prefixed by ``prefix``):
        ``fastts_nrmse``      — RMSE/std(target) per (window, channel) over the SAMPLE axis,
            averaged. 0.0 = perfect; EXACTLY 1.0 = predicting that (window, channel)'s own
            constant mean; > 1.0 = worse than that constant.
        ``fastts_corr``       — Pearson correlation over the same samples, per
            (window, channel), averaged. 1.0 = perfect.
        ``fastts_valid_frac`` — fraction of (window, channel) pairs with a non-constant target
            (the rest are absent/dead channels, excluded from both means above).
        ``fastts_global_nrmse`` — the complementary UN-normalised view: RMSE over the whole
            batch divided by the per-CHANNEL std taken over the whole batch. Here the absolute
            light LEVEL counts (it does not in ``fastts_nrmse``, which removes each window's
            own scale). Reported because the two can disagree by a factor of ~25 for this
            signal and quoting only one of them is how a codec looks better than it is.
    """
    r = _to_numpy(recon).astype(np.float64)
    t = _to_numpy(target).astype(np.float64)
    if r.shape != t.shape:
        raise ValueError(f"full_fastts_metrics: shape mismatch {r.shape} vs {t.shape}")
    if r.ndim != 3:
        raise ValueError(f"full_fastts_metrics expects (B, C, W); got {r.shape}")
    nrmse, corr, valid = _fastts_nrmse_corr(r, t)
    # global (level-aware) view: per-channel std over the whole evaluated batch.
    g_std = t.transpose(1, 0, 2).reshape(t.shape[1], -1).std(axis=1)
    g_rmse = np.sqrt(((r - t) ** 2).transpose(1, 0, 2).reshape(t.shape[1], -1).mean(axis=1))
    ok = g_std > _FASTTS_STD_EPS
    g_nrmse = float((g_rmse[ok] / g_std[ok]).mean()) if ok.any() else float("nan")
    return {
        f"{prefix}fastts_nrmse": nrmse,
        f"{prefix}fastts_corr": corr,
        f"{prefix}fastts_valid_frac": valid,
        f"{prefix}fastts_global_nrmse": g_nrmse,
    }


def trivial_fastts_baselines(target) -> Dict[str, float]:
    """The reference points that make :func:`full_fastts_metrics` interpretable.

    Every baseline is scored with the SAME function, the same units, against the same target
    batch, so the codec row and the baseline rows are directly comparable:

      * ``base_self_*``   — the target predicting itself. MUST read nrmse 0.0000 / corr 1.0000;
        the self-check that the metric is wired correctly.
      * ``base_wcmean_*`` — the per-(window, channel) scalar mean broadcast back. Scores nrmse
        EXACTLY 1.0000 by construction — that IS the normalisation anchor — and corr 0.0.
      * ``base_tmean_*``  — the target's own time-averaged value broadcast back over the
        window. **THE baseline to beat.** For a (B, C, W) array with no structure axis this is
        the SAME predictor as ``wcmean``, so it must also read exactly 1.0000; the two keys are
        kept separate because the two self-checks answer different questions and their
        coincidence here is itself worth asserting.
      * ``base_cmean_*``  — the per-CHANNEL mean over the whole evaluated batch, broadcast
        back: one dataset-level constant per channel, no per-window information at all.
        Expect it to be MUCH worse than 1.0 (measured ~25.8 on 4 held-out shots), because the
        window-to-window light level dominates the within-window fluctuation.
      * ``base_binmean<P>_*`` — the target's own mean within each non-overlapping ``P``-sample
        bin, broadcast back inside the bin: a piecewise-constant predictor at the resolution
        the OLD ELM-envelope codec worked at. ``P = 100`` (10 ms, the shipped envelope bin)
        measured 0.9630 on held-out shots — i.e. the envelope resolution captures only ~7% of
        the within-window variance, which is why the envelope codec could never reconstruct
        this signal.
    """
    t = _to_numpy(target).astype(np.float64)
    if t.ndim != 3:
        raise ValueError(f"trivial_fastts_baselines expects (B, C, W); got {t.shape}")
    B, C, W = t.shape
    wcmean = np.broadcast_to(t.mean(axis=-1, keepdims=True), t.shape)
    cmean = np.broadcast_to(t.mean(axis=(0, -1))[None, :, None], t.shape)
    preds = [("self", t), ("tmean", wcmean), ("wcmean", wcmean), ("cmean", cmean)]
    for pool in (100, 10):
        if W % pool == 0:
            bm = t.reshape(B, C, W // pool, pool).mean(-1, keepdims=True)
            preds.append((f"binmean{pool}",
                          np.broadcast_to(bm, (B, C, W // pool, pool)).reshape(t.shape)))
    out: Dict[str, float] = {}
    for name, pred in preds:
        m = full_fastts_metrics(pred, t, prefix=f"base_{name}_")
        m.pop(f"base_{name}_fastts_valid_frac", None)  # identical for every baseline
        out.update(m)
    return out


# ========================================================================================= #
# FAST-TS ENVELOPE metrics (2026-09-03) — the POOLED, GLOBAL-std read the deliverable uses.
#
# WHY A SECOND nRMSE. ``full_fastts_metrics``'s ``fastts_nrmse`` divides each (window, channel)
# by its OWN std, which for the ELM ENVELOPE throws away the very thing that dominates it:
# 96.2% of the envelope's variance is the BETWEEN-window DC level, and a per-window
# normalisation makes that level free. The number the envelope codec is actually judged on --
# and the number the reconstruction figure reports -- normalises by each channel's GLOBAL std
# over the whole held-out pool, so a constant predictor scores 1.0000 and the level counts.
# These are different quantities, not two estimates of one; both are reported, prefixed.
#
# THE BAR. ``env_base_ratematched`` is a trivial encoder that transmits ONLY each channel's
# window mean, quantised to the CODEC'S OWN bit budget (n_tok * log2(codebook) bits, split
# evenly over the channels). Measured 0.1935 on 4 held-out shots against the shipped codec's
# 0.5252 -- a codec that loses to "transmit 8 numbers" at its own rate is not doing its job,
# so this row belongs in every fast-TS table. ``env_base_wmean`` is the same predictor
# UNQUANTISED (0.1919), i.e. the floor of any level-only code.
#
# AND THE AMPLITUDE. nRMSE's exact minimiser is the conditional mean, so it is a FLOOR and
# never the ranking key on its own: ``env_std_ratio`` (ideal 1.0) and ``env_burst_keep``
# (fraction of a real ELM burst's height that survives) are what detect the amplitude collapse
# an element-wise error rewards.
# ========================================================================================= #


def fastts_envelope_metrics(recon, target, bits: float = 0.0,
                            prefix: str = "") -> Dict[str, float]:
    """Pooled, GLOBAL-std envelope metrics + the trivial anchors, on ``(B, C, E)`` arrays.

    Parameters
    ----------
    recon, target : (B, C, E)
        The reconstruction and the ELM-activity envelope it targets.
    bits : float
        The codec's token budget in bits (``n_tok * log2(codebook_size)``). Used ONLY to build
        ``env_base_ratematched`` at the codec's own rate; 0 disables that row.

    Returns (all averaged over the C channels)
    -------
    ``env_nrmse``            RMS(recon-gt)/std(gt), pooled over (window, bin) per channel.
                             A global constant scores 1.0000.
    ``env_std_ratio``        std(recon)/std(gt) pooled over (window, bin). Ideal 1.0, but
                             DOMINATED by the between-window level -- it reads 0.9926 on a
                             0.4x-shrunk reference, so it is NOT the burst-amplitude check.
    ``env_ac_std_ratio``     the same ratio on the MEAN-REMOVED (within-window) part. THE
                             amplitude check on this signal; reads 0.4000 on that reference.
    ``env_level_corr``       Pearson r of the per-(window, channel) LEVEL (mean over bins).
    ``env_burst_keep``       burst height retained: on the top-1% most active windows, the
                             mean peak-above-median of the recon over that of the target.
    ``env_base_gmean``       global-constant anchor (1.0000 by construction; a self-check).
    ``env_base_wmean``       exact per-window-mean anchor -- the level-only FLOOR.
    ``env_base_ratematched`` the same predictor quantised to ``bits`` -- THE BAR.
    """
    r = _to_numpy(recon).astype(np.float64)
    t = _to_numpy(target).astype(np.float64)
    if r.shape != t.shape:
        raise ValueError(f"fastts_envelope_metrics: shape mismatch {r.shape} vs {t.shape}")
    if t.ndim != 3:
        raise ValueError(f"fastts_envelope_metrics expects (B, C, E); got {t.shape}")
    B, C, E = t.shape

    def _pooled(pred: np.ndarray) -> float:
        vals = [np.sqrt(((pred[:, c] - t[:, c]) ** 2).mean()) / (t[:, c].std() + 1e-12)
                for c in range(C)]
        return float(np.mean(vals))

    lev_t, lev_r = t.mean(-1), r.mean(-1)
    ac_t, ac_r = t - lev_t[..., None], r - lev_r[..., None]
    corrs, sratio, acratio, burst = [], [], [], []
    for c in range(C):
        g, q = t[:, c], r[:, c]
        sratio.append(q.std() / (g.std() + 1e-12))
        # AC-ONLY amplitude ratio. The pooled std_ratio above is DOMINATED by the between-
        # window level (96.2% of the variance) and is therefore nearly blind to a within-
        # window amplitude collapse: measured on the validation ladder, a 0.4x-shrunk
        # reference reads pooled std_ratio 0.9926 -- it LOOKS like correct amplitude -- while
        # this AC ratio reads 0.4000, which is the truth. Never judge burst amplitude on
        # env_std_ratio alone.
        acratio.append(ac_r[:, c].std() / (ac_t[:, c].std() + 1e-12))
        lt, lr = lev_t[:, c], lev_r[:, c]
        corrs.append(0.0 if lt.std() < 1e-12 or lr.std() < 1e-12
                     else float(np.corrcoef(lt, lr)[0, 1]))
        # burst height retained: the ELM bursts are the whole point of a fast-TS channel and
        # the quiet 99% of windows dominate nRMSE, which therefore barely notices flattening.
        gm, rm = g.max(1), q.max(1)
        k = max(1, int(0.01 * len(gm)))
        idx = np.argsort(gm)[-k:]
        base = np.median(g)
        burst.append(float((rm[idx] - base).mean() / ((gm[idx] - base).mean() + 1e-12)))

    wmean = np.broadcast_to(lev_t[..., None], t.shape)
    out = {
        f"{prefix}env_nrmse": _pooled(r),
        f"{prefix}env_std_ratio": float(np.mean(sratio)),
        f"{prefix}env_ac_std_ratio": float(np.mean(acratio)),
        f"{prefix}env_level_corr": float(np.mean(corrs)),
        f"{prefix}env_burst_keep": float(np.mean(burst)),
        f"{prefix}env_base_gmean": _pooled(
            np.broadcast_to(t.mean(axis=(0, -1))[None, :, None], t.shape)),
        f"{prefix}env_base_wmean": _pooled(wmean),
    }
    if bits and bits > 0:
        lv = max(2, int(2 ** (float(bits) / C)))
        q = np.empty_like(lev_t)
        for c in range(C):
            lo, hi = lev_t[:, c].min(), lev_t[:, c].max()
            q[:, c] = (np.round((lev_t[:, c] - lo) / (hi - lo + 1e-12) * (lv - 1))
                       / (lv - 1) * (hi - lo) + lo)
        out[f"{prefix}env_base_ratematched"] = _pooled(np.broadcast_to(q[..., None], t.shape))
        out[f"{prefix}env_rate_levels"] = float(lv)
    return out


# ========================================================================================= #
# MODE-TRACK metrics (2026-09-03) — built because spec_nrmse RANKS THE BLUR BEST.
#
# Measured on 720 held-out mhr windows: lr1e4d6 scores spec_nrmse 0.9010 (the best of any
# codec) with hf_ratio 0.0143 and renders a featureless smear, while ms2_s1 scores 1.0972 /
# 0.8258 and visibly reproduces the rising 50->65 kHz mode track. An element-wise error is
# minimised by the conditional mean, so it cannot be the ranking key for "recognisable".
#
# The defining property of a mode track is a NARROW-BAND RIDGE IN FREQUENCY THAT PERSISTS
# COHERENTLY ACROSS TIME. Every metric below therefore evaluates the frequency profile
# PER TIME FRAME and never averages the time axis — which is exactly the defect in
# ``peak_f1`` / ``envelope_corr`` / ``sharpness`` above, all of which run on
# ``_power_envelope`` (= ``spec.mean(axis=-1)``) and so award a static smear a perfect score.
#
# None of these enters ``spike.gate_score``: checkpoint selection must not shift silently.
# ========================================================================================= #
def _as_nft(x) -> np.ndarray:
    """(B, C, F, T) -> (B*C, F, T) float32.

    float32 deliberately: every metric built on this is a ratio, a correlation or an F1, none
    of which needs float64, and the float64 version made the audit ~4x slower per chunk (it is
    called several times per arm per chunk).
    """
    a = _to_numpy(x).astype(np.float32)
    if a.ndim != 4:
        raise ValueError(f"expected (B, C, F, T); got {a.shape}")
    return a.reshape(a.shape[0] * a.shape[1], a.shape[2], a.shape[3])


# Absolute floor on a frame's frequency-profile spread, below which it has NO peaks.
# The prominence test is RELATIVE to each frame's own std, which means a frame carrying only
# numerical noise still yields "peaks": the detrended time-mean envelope is analytically zero,
# but in float32 its residual is ~1.8e-07 (3.3e-16 in float64), and peak-picking that noise
# scored the envelope 0.17 instead of the 0.00 it must have by construction. Real log-power
# z-scored spectrogram frames have sd ~0.5-1.5, so this floor is a no-op on actual data.
_PEAK_MIN_SD: float = 1e-3


def _peaks_per_frame(X: np.ndarray, prom: float) -> np.ndarray:
    """(N, F, T) -> bool mask of frequency-local peaks, computed INDEPENDENTLY per time frame.

    A bin is a peak when it is a strict local maximum along FREQUENCY and rises at least
    ``prom`` times that frame's own frequency-profile standard deviation above that frame's
    median. Normalising by the frame's own spread makes the threshold scale-free, so it does
    not have to be retuned per modality or per normalisation convention -- but it also means a
    frame of pure rounding noise would qualify, hence the ``_PEAK_MIN_SD`` absolute floor.
    """
    lm = np.zeros(X.shape, dtype=bool)
    lm[:, 1:-1, :] = (X[:, 1:-1, :] > X[:, :-2, :]) & (X[:, 1:-1, :] >= X[:, 2:, :])
    med = np.median(X, axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    return lm & ((X - med) >= prom * (sd + 1e-12)) & (sd > _PEAK_MIN_SD)


def _dilate_freq(mask: np.ndarray, tol: int) -> np.ndarray:
    """Widen a (N, F, T) peak mask by +/- ``tol`` bins along the FREQUENCY axis."""
    out = mask.copy()
    for d in range(1, int(tol) + 1):
        out[:, d:, :] |= mask[:, :-d, :]
        out[:, :-d, :] |= mask[:, d:, :]
    return out


def mode_track_f1(recon, target, prom: float = 1.5, tol: int = 2,
                  band_bins: Optional[int] = None, prefix: str = "",
                  detrend: bool = False) -> Dict[str, float]:
    """Per-time-frame spectral-peak F1 — "are the same modes present at the same times?".

    For every (window, channel, TIME FRAME) the frequency profile of both arrays is peak-picked
    (:func:`_peaks_per_frame`); a ground-truth peak counts as recovered when the reconstruction
    has a peak within ``+/- tol`` frequency bins of it. F1 is pooled over all (frame, peak)
    pairs, so a codec that reproduces the average spectrum but not its time evolution scores
    LOW — unlike ``peak_f1``, which deletes the time axis first.

    Returns ``{prefix}mode_track_f1`` plus ``{prefix}mode_track_precision`` /
    ``_recall`` / ``_median_df`` (median |delta f| in bins over matched GT peaks) and
    ``{prefix}mode_track_n_gt_peaks`` (peaks per frame in the target — the metric is
    meaningless if this is ~0, so it is reported).
    """
    r, t = _as_nft(recon), _as_nft(target)
    if band_bins is not None:
        k = int(min(int(band_bins), t.shape[1]))
        r, t = r[:, :k, :], t[:, :k, :]
    if detrend:
        # MEASURED MOTIVATION: at detrend=False the TIME-MEAN ENVELOPE scores HIGHEST of every
        # candidate (0.4952 on 240 held-out mhr windows, above every trained codec), because the
        # metric then measures peak PRESENCE in the frequency marginal and mhr's peaks are
        # largely static within a 50 ms window. Subtracting each (window, channel)'s own
        # time-average profile from every frame removes exactly the part the envelope can
        # supply: the detrended envelope is IDENTICALLY ZERO and therefore has no peaks at all,
        # so it must score 0. What survives is peak structure BEYOND the static envelope, which
        # is the quantity "does the mode track move with time" actually asks about.
        r = r - r.mean(axis=2, keepdims=True)
        t = t - t.mean(axis=2, keepdims=True)
    pg, pr = _peaks_per_frame(t, prom), _peaks_per_frame(r, prom)
    dg, dr = _dilate_freq(pg, tol), _dilate_freq(pr, tol)
    tp = float((pg & dr).sum())
    fn = float((pg & ~dr).sum())
    fp = float((pr & ~dg).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * tp / (2.0 * tp + fp + fn)) if (2.0 * tp + fp + fn) > 0 else 0.0
    # median |delta f| of matched GT peaks: smallest shift at which a recon peak appears
    delta = np.full(pg.shape, -1, dtype=np.int16)
    remaining = pg.copy()
    for d in range(0, int(tol) + 1):
        for sgn in ((0,) if d == 0 else (-1, 1)):
            sh = np.zeros(pr.shape, dtype=bool)
            if d == 0:
                sh = pr
            elif sgn < 0:
                sh[:, d:, :] = pr[:, :-d, :]
            else:
                sh[:, :-d, :] = pr[:, d:, :]
            hit = remaining & sh
            delta[hit] = d
            remaining &= ~hit
    med_df = float(np.median(delta[delta >= 0])) if bool((delta >= 0).any()) else float("nan")
    n_frames = float(t.shape[0] * t.shape[2])
    return {
        f"{prefix}mode_track_f1": float(f1),
        f"{prefix}mode_track_precision": float(prec),
        f"{prefix}mode_track_recall": float(rec),
        f"{prefix}mode_track_median_df": med_df,
        f"{prefix}mode_track_n_gt_peaks": float(pg.sum() / max(n_frames, 1.0)),
    }


def ridge_traj_corr(recon, target, band_bins: Optional[int] = DEFAULT_MODE_BAND_BINS,
                    prefix: str = "") -> Dict[str, float]:
    """Correlation of the DOMINANT-FREQUENCY TRAJECTORY over time.

    Per (window, channel) the argmax frequency inside the mode band is taken for every time
    frame, giving a trajectory ``f*(t)``; the metric is the Pearson correlation of the target's
    trajectory against the reconstruction's, averaged over pairs. A blur or a static envelope
    produces a constant (or noise-driven) trajectory whose correlation with the real one is ~0;
    a reconstruction that actually follows the track scores high. Pairs where either trajectory
    is constant contribute 0.0 (no trajectory to correlate), which is the honest reading.
    """
    r, t = _as_nft(recon), _as_nft(target)
    if band_bins is not None:
        k = int(min(int(band_bins), t.shape[1]))
        r, t = r[:, :k, :], t[:, :k, :]
    ag = t.argmax(axis=1).astype(np.float64)          # (N, T)
    ar = r.argmax(axis=1).astype(np.float64)
    gc = ag - ag.mean(axis=1, keepdims=True)
    rc = ar - ar.mean(axis=1, keepdims=True)
    den = np.sqrt((gc * gc).sum(axis=1) * (rc * rc).sum(axis=1))
    corr = np.where(den > 0.0, (gc * rc).sum(axis=1) / np.maximum(den, 1e-300), 0.0)
    return {f"{prefix}ridge_traj_corr": float(corr.mean())}


def spectral_contrast_ratio(recon, target, band_bins: Optional[int] = None,
                            prefix: str = "") -> Dict[str, float]:
    """Per-frame frequency CONTRAST, reconstruction / target. IDEAL 1.0.

    Contrast of one frequency profile is ``max - median``: how far the strongest band rises
    above the typical level in THAT time frame. A blur — whose profile is flat — drives it
    toward 0; values > 1 mean the reconstruction is MORE peaked than the target.

    LATTICE ROBUSTNESS, MEASURED — robust, NOT immune. Adding a synthetic checkerboard to a
    blurred reconstruction multiplies the metrics by these factors, over two moving-ridge
    targets of different slope (see ``tests/ignite/test_gate.py``)::

        artifact added        sharpness (hf_ratio)   this metric      mode_track_f1
        frequency-periodic       x12.5 - x19.2       x2.2 - x3.3       x0.51 - x0.53
        time-periodic            x12.0 - x18.7       x1.00 (exact)     x1.00 (exact)

    ``sharpness`` is inflated by an ORDER OF MAGNITUDE and is therefore unusable as a quality
    signal — consistent with the measured 92-94 % of the mhr reconstruction's HF energy sitting
    on the patch lattice. This metric moves 2-3x, i.e. ~6x less, and a purely TIME-periodic
    lattice leaves it exactly unchanged. But it is genuinely NOT immune: a frequency-periodic
    lattice is by construction a set of narrow frequency peaks, which is the quantity this
    metric rewards, and the inflation grows as the target's own contrast falls.

    ``mode_track_f1`` and ``ms_ssim`` are the artifact-SAFE members of the family — a lattice
    makes both WORSE (f1 roughly halves), because the spurious peaks are false positives and
    the tiled texture is not the target's structure. Rank on those; read this one alongside.
    """
    r, t = _as_nft(recon), _as_nft(target)
    if band_bins is not None:
        k = int(min(int(band_bins), t.shape[1]))
        r, t = r[:, :k, :], t[:, :k, :]
    cg = t.max(axis=1) - np.median(t, axis=1)         # (N, T)
    cr = r.max(axis=1) - np.median(r, axis=1)
    mg = float(cg.mean())
    return {
        f"{prefix}spectral_contrast_ratio": float(cr.mean() / mg) if mg > 0 else float("nan"),
        f"{prefix}spectral_contrast_gt": mg,
    }


def _box2d(x: np.ndarray, k: int) -> np.ndarray:
    """(N, H, W) -> box mean over a k x k window, 'valid', via an integral image.

    Deliberately float32: this is called 5x per scale per call and the float64 version was
    the measured bottleneck of the whole metric suite (~2 min per 24-window chunk). SSIM is a
    ratio of second moments and needs nothing like float64 precision here.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    cs = np.cumsum(np.cumsum(
        np.pad(x, ((0, 0), (1, 0), (1, 0)), mode="constant"), axis=1, dtype=np.float32),
        axis=2, dtype=np.float32)
    return ((cs[:, k:, k:] - cs[:, :-k, k:] - cs[:, k:, :-k] + cs[:, :-k, :-k])
            / np.float32(k * k))


def ms_ssim(recon, target, win: int = 7, scales: Sequence[int] = (1, 2, 4),
            prefix: str = "") -> Dict[str, float]:
    """Multi-scale SSIM on the spectrogram — the canonical "MSE selects blur" cross-check.

    Structural similarity is computed with a ``win x win`` box window at each avg-pooled scale
    and averaged over scales, using each (window, channel) target's own dynamic range for the
    stabilising constants. Included as an INDEPENDENT check: it is a standard image metric with
    a standard bias (it penalises loss of local structure), so if it agrees with the bespoke
    mode-track metrics that is evidence they are measuring something real.
    """
    r = _as_nft(recon).astype(np.float32, copy=False)
    t = _as_nft(target).astype(np.float32, copy=False)
    rng = (t.max(axis=(1, 2)) - t.min(axis=(1, 2)))[:, None, None]
    rng = np.maximum(rng, 1e-12)
    c1, c2 = (0.01 * rng) ** 2, (0.03 * rng) ** 2
    vals = []
    for s in scales:
        a, b = (r, t) if s == 1 else (_box2d(r, s)[:, ::s, ::s], _box2d(t, s)[:, ::s, ::s])
        if min(a.shape[1], a.shape[2]) <= win:
            continue
        mu_a, mu_b = _box2d(a, win), _box2d(b, win)
        sa = _box2d(a * a, win) - mu_a * mu_a
        sb = _box2d(b * b, win) - mu_b * mu_b
        sab = _box2d(a * b, win) - mu_a * mu_b
        ssim = (((2 * mu_a * mu_b + c1) * (2 * sab + c2))
                / ((mu_a ** 2 + mu_b ** 2 + c1) * (sa + sb + c2)))
        vals.append(float(np.nanmean(ssim)))
    return {f"{prefix}ms_ssim": float(np.mean(vals)) if vals else float("nan")}


def mode_structure_metrics(recon, target, band_bins: Optional[int] = DEFAULT_MODE_BAND_BINS,
                           prefix: str = "", detrended: bool = False) -> Dict[str, float]:
    """All four mode-track metrics in one call (what ``decode_fidelity`` folds in)."""
    out: Dict[str, float] = {}
    out.update(mode_track_f1(recon, target, band_bins=band_bins, prefix=prefix))
    if detrended:
        # OPT-IN (costs a second full peak-picking pass): the same F1 after removing each
        # pair's own time-average profile, so the static envelope scores 0 by construction.
        _d = mode_track_f1(recon, target, band_bins=band_bins, prefix=prefix, detrend=True)
        out[f"{prefix}mode_track_f1_detr"] = _d[f"{prefix}mode_track_f1"]
        out[f"{prefix}mode_track_n_gt_peaks_detr"] = _d[f"{prefix}mode_track_n_gt_peaks"]
    out.update(ridge_traj_corr(recon, target, band_bins=band_bins, prefix=prefix))
    out.update(spectral_contrast_ratio(recon, target, band_bins=band_bins, prefix=prefix))
    out.update(ms_ssim(recon, target, prefix=prefix))
    return out


__all__ += [
    "mode_track_f1",
    "ridge_traj_corr",
    "spectral_contrast_ratio",
    "ms_ssim",
    "mode_structure_metrics",
]


# --------------------------------------------------------------------------------------
# 8. FAST-TS 1-D SSIM — the amplitude/dynamic-range check nRMSE cannot make (2026-09-03)
# --------------------------------------------------------------------------------------
# WHY THIS EXISTS. nRMSE's exact minimiser is the conditional mean, so a codec trained or
# ranked on it shrinks toward the local mean under uncertainty: the nRMSE-optimal scaling of a
# prediction whose correlation with the truth is r is a factor of r, which for r = 0.4 means an
# output whose amplitude is 60% COMPRESSED while nRMSE reads a respectable 0.917. On the
# spectro side this was measured directly: the codec nRMSE ranked BEST (0.9010) was a
# featureless blur, and one it ranked WORSE (1.0972) visibly carried the physics. For a spiky
# 10 kHz filterscope trace, variance collapse is the single most likely outcome of any
# squared-error sweep, and nRMSE alone cannot see it -- MSE has no variance term at all.
#
# SSIM does. Its canonical factorisation is
#
#     SSIM = luminance x CONTRAST x STRUCTURE
#     luminance = (2 mu_r mu_t + C1) / (mu_r^2 + mu_t^2 + C1)
#     contrast  = (2 sig_r sig_t + C2) / (sig_r^2 + sig_t^2 + C2)
#     structure = (sig_rt + C2/2) / (sig_r sig_t + C2/2)
#
# and the CONTRAST factor collapses exactly when the reconstruction's local std collapses:
# a recon with the right shape at half the amplitude scores contrast 0.8, at 0.4x it scores
# 0.69, and a flat line scores ~0. This is the NVIDIA Spectral Codec reference's family of
# choice (arXiv 2406.05298 judges reconstruction with ViSQOL, built on NSIM, an SSIM-family
# structural measure) -- that paper reports NO normalised element-wise error anywhere and its
# own codecs score SI-SDR -23 dB while being perceptually excellent.
#
# HOW IT IS USED HERE. nRMSE stays a FLOOR (a codec must beat the 1.0000 constant-mean
# anchor); it is NOT the ranking key. These structural numbers are what distinguish two codecs
# that both clear the floor. It is ALSO a differentiable loss -- see
# ``FastTSCodec._ssim_loss`` -- and note this attacks spike preservation through LOCAL VARIANCE,
# a different mechanism from the per-sample spike loss-WEIGHTING that is a recorded dead end
# for this modality.
#
# CHOICES, stated explicitly:
#   * 1-D windows along the SAMPLE axis, per (window, channel) -- the axis the physics lives on.
#   * BOX window of ``win = 17`` samples = 1.7 ms at 10 kHz. Filterscope ELM bursts recur every
#     ~1.6 ms, so one window spans about one burst period: long enough for a local variance to
#     mean something, short enough that a single burst is not averaged away. (A box window is
#     the original SSIM paper's own choice and is exact + dependency-free via cumsum.)
#   * 'VALID' positions only (no padding), so no edge artefact enters the mean.
#   * The stabilising constants use each (window, channel) TARGET's OWN std as the dynamic
#     range L, so the metric is invariant to the per-channel standardization and to each
#     window's own scale -- the same convention ``full_fastts_metrics`` uses.
#   * Both the FULL SSIM and the luminance-free CS form are reported. For a SIGNED signal the
#     luminance factor is ill-behaved (``2 mu_r mu_t`` goes negative when the two means
#     straddle zero), which is why MS-SSIM itself uses CS at every scale but the coarsest;
#     the DC level is already accounted for by nRMSE, which is not mean-removed.
_FASTTS_SSIM_WIN: int = 17
_FASTTS_SSIM_SCALES = (1, 2, 4)


def _box1d(x: np.ndarray, win: int) -> np.ndarray:
    """Mean over every FULLY-INSIDE length-``win`` window along the last axis (exact, cumsum).

    ``(..., L) -> (..., L - win + 1)``. No padding, so no edge artefact.
    """
    c = np.cumsum(x, axis=-1)
    z = np.zeros(x.shape[:-1] + (1,), dtype=c.dtype)
    c = np.concatenate([z, c], axis=-1)
    return (c[..., win:] - c[..., :-win]) / float(win)


def _pool1d(x: np.ndarray, k: int) -> np.ndarray:
    """Non-overlapping length-``k`` average pool along the last axis (the MS-SSIM downsample)."""
    if k == 1:
        return x
    L = (x.shape[-1] // k) * k
    return x[..., :L].reshape(*x.shape[:-1], L // k, k).mean(axis=-1)


def _ssim1d_terms(r: np.ndarray, t: np.ndarray, win: int):
    """Per-(row, position) SSIM factor maps for (N, L) arrays: (luminance, contrast, structure).

    ``C1 = (0.01 L)^2``, ``C2 = (0.03 L)^2`` with ``L`` = each ROW's target std (see the
    section note). Rows whose target is constant get L clamped, which sends their contrast to
    ~0 for any non-constant recon -- the honest reading.
    """
    L = np.maximum(t.std(axis=-1, keepdims=True), 1e-12)
    c1, c2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    mu_r, mu_t = _box1d(r, win), _box1d(t, win)
    var_r = np.maximum(_box1d(r * r, win) - mu_r * mu_r, 0.0)
    var_t = np.maximum(_box1d(t * t, win) - mu_t * mu_t, 0.0)
    cov = _box1d(r * t, win) - mu_r * mu_t
    sd_r, sd_t = np.sqrt(var_r), np.sqrt(var_t)
    lum = (2 * mu_r * mu_t + c1) / (mu_r ** 2 + mu_t ** 2 + c1)
    con = (2 * sd_r * sd_t + c2) / (var_r + var_t + c2)
    stru = (cov + c2 / 2.0) / (sd_r * sd_t + c2 / 2.0)
    return lum, con, stru


def fastts_ssim_metrics(recon, target, win: int = _FASTTS_SSIM_WIN,
                        scales: Sequence[int] = _FASTTS_SSIM_SCALES,
                        prefix: str = "") -> Dict[str, float]:
    """1-D structural similarity of the raw fast-TS waveform. See the section note above.

    Args:
        recon / target: (B, C, W) raw windows in the codec's standardized space.
        win: box-window length in SAMPLES (default 17 = 1.7 ms at 10 kHz, ~one ELM period).
        scales: average-pool factors for the multi-scale CS average.
        prefix: prepended to every key.

    Returns (keys prefixed by ``prefix``):
        ``fastts_ssim``           full SSIM (luminance x contrast x structure), 1.0 = perfect.
        ``fastts_ssim_cs``        contrast x structure -- the luminance-free form; THE ranking
                                  number, because the DC level is already in nRMSE.
        ``fastts_ssim_contrast``  the contrast factor ALONE = the amplitude / dynamic-range
                                  term. **This is the variance-collapse detector**: right
                                  shape at half amplitude scores 0.80, at 0.4x scores 0.69,
                                  a flat line scores ~0.
        ``fastts_ssim_structure`` the structure factor alone (shape + spike timing, amplitude
                                  divided out).
        ``fastts_ms_ssim``        CS averaged over ``scales`` (multi-scale).
        ``fastts_std_ratio``      mean over (window, channel) of std(recon)/std(target). The
                                  bluntest amplitude read there is: 1.0 = right dynamic range,
                                  < 1 = compressed (blurred), > 1 = over-driven. Reported
                                  because it needs no interpretation at all.
    """
    r = _to_numpy(recon).astype(np.float64)
    t = _to_numpy(target).astype(np.float64)
    if r.shape != t.shape:
        raise ValueError(f"fastts_ssim_metrics: shape mismatch {r.shape} vs {t.shape}")
    if r.ndim != 3:
        raise ValueError(f"fastts_ssim_metrics expects (B, C, W); got {r.shape}")
    rf = r.reshape(-1, r.shape[-1])
    tf = t.reshape(-1, t.shape[-1])
    live = tf.std(axis=-1) > _FASTTS_STD_EPS      # exclude dead/absent channels, as nRMSE does
    if not live.any():
        return {f"{prefix}fastts_ssim": float("nan"), f"{prefix}fastts_ssim_cs": float("nan"),
                f"{prefix}fastts_ssim_contrast": float("nan"),
                f"{prefix}fastts_ssim_structure": float("nan"),
                f"{prefix}fastts_ms_ssim": float("nan"),
                f"{prefix}fastts_std_ratio": float("nan")}
    rf, tf = rf[live], tf[live]

    lum, con, stru = _ssim1d_terms(rf, tf, win)
    out = {
        f"{prefix}fastts_ssim": float(np.nanmean(lum * con * stru)),
        f"{prefix}fastts_ssim_cs": float(np.nanmean(con * stru)),
        f"{prefix}fastts_ssim_contrast": float(np.nanmean(con)),
        f"{prefix}fastts_ssim_structure": float(np.nanmean(stru)),
        f"{prefix}fastts_std_ratio": float(np.nanmean(
            rf.std(axis=-1) / np.maximum(tf.std(axis=-1), 1e-12))),
    }
    cs_scales = []
    for s in scales:
        a, b = _pool1d(rf, s), _pool1d(tf, s)
        if a.shape[-1] <= win:
            continue
        _, c_s, s_s = _ssim1d_terms(a, b, win)
        cs_scales.append(float(np.nanmean(c_s * s_s)))
    out[f"{prefix}fastts_ms_ssim"] = float(np.mean(cs_scales)) if cs_scales else float("nan")
    return out


def fastts_structural_references(target, win: int = _FASTTS_SSIM_WIN) -> Dict[str, float]:
    """VALIDATION references for :func:`fastts_ssim_metrics` — run these BEFORE ranking on it.

    Constructed from the target alone, so they are available on any eval batch. A metric that
    does not order these correctly is rejected, not interpreted:

      * ``ref_self_*``      target vs itself. MUST be 1.0000 on every factor.
      * ``ref_noise_*``     target + 0.05 sigma white noise. A nearly-perfect reconstruction:
        must score WELL (high contrast, high structure) even though its nRMSE is 0.05.
      * ``ref_smooth_*``    target box-smoothed over 9 samples (0.9 ms). The BLUR failure mode:
        must score BADLY on contrast (its local variance is destroyed) even though smoothing
        LOWERS nRMSE relative to many honest reconstructions.
      * ``ref_shrunk_*``    0.4 x the target, i.e. perfect shape at 40% amplitude -- exactly
        what an nRMSE-optimal shrinkage produces at correlation 0.4. Structure must stay ~1.0
        while CONTRAST drops to ~0.69: this is the metric proving it separates 'right shape'
        from 'right amplitude'.
      * ``ref_flat_*``      the per-(window, channel) constant mean (the nRMSE 1.0000 anchor).
        Must score ~0 on contrast and structure.
    """
    t = _to_numpy(target).astype(np.float64)
    if t.ndim != 3:
        raise ValueError(f"fastts_structural_references expects (B, C, W); got {t.shape}")
    rng = np.random.default_rng(0)
    sd = t.std(axis=-1, keepdims=True)
    mu = t.mean(axis=-1, keepdims=True)
    sm = _box1d(t, 9)
    smooth = np.concatenate([sm, np.repeat(sm[..., -1:], t.shape[-1] - sm.shape[-1], -1)], -1)
    refs = {
        "self": t,
        "noise": t + 0.05 * sd * rng.standard_normal(t.shape),
        "smooth": smooth,
        "shrunk": mu + 0.4 * (t - mu),
        "flat": np.broadcast_to(mu, t.shape),
    }
    out: Dict[str, float] = {}
    for name, pred in refs.items():
        out.update(fastts_ssim_metrics(pred, t, win=win, prefix=f"ref_{name}_"))
        out[f"ref_{name}_fastts_nrmse"] = full_fastts_metrics(pred, t)["fastts_nrmse"]
    return out


__all__ += [
    "full_fastts_metrics",
    "fastts_envelope_metrics",
    "trivial_fastts_baselines",
    "fastts_ssim_metrics",
    "fastts_structural_references",
]


# --------------------------------------------------------------------------------------
# 9. VIDEO full-array metrics — the tangtv analogue of full_slowts_metrics (2026-09-03)
# --------------------------------------------------------------------------------------
# Same normalisation CONVENTION as full_spectro_metrics / full_slowts_metrics, deliberately
# not a new one: RMSE divided by the standard deviation of THAT SAME group, so predicting the
# group's own constant mean scores EXACTLY 1.0000 and "< 1.0" means "beats that group's own
# DC level".
#
# The GROUP is per (WINDOW, CHANNEL) over the flattened (T, H, W) volume. A tangtv channel is
# a separate CAMERA with its own optical filter and its own brightness scale, so pooling the
# two cameras of a divertor into one group would let a bright camera's variance hide a dead
# one. This mirrors full_spectro_metrics, which groups per (window, channel) over (F, T).
#
# MASKING. Video is a HEAVILY masked family -- more so than slow-TS. Measured over all 8753
# shots (foundation_model_meta/video_channel_liveness.pt): tangtv_lower has NO live camera in
# 51.30% of shots and tangtv_upper in 68.58%, and among the shots that do have one, a further
# 19.0% / 16.9% of channel-slots are a zero slab. An unmasked video metric therefore mostly
# scores how well the codec reproduces the number zero. Every statistic below -- mean, std,
# RMSE, correlation, std-ratio and each baseline's own mean -- is taken over VALID (channel,
# frame) entries only when a mask is given, matching the slow-TS standard exactly.
#
# READ nRMSE AS A FLOOR, NEVER A RANKING KEY. Its exact minimiser is the conditional mean, so
# it rewards blur and amplitude collapse; ``video_std_ratio`` (ideal EXACTLY 1.0) is reported
# beside it precisely to expose that, and ``video_std_ratio_t`` catches the "correct still
# image, frozen in time" failure that the spatial ratio cannot see.
# --------------------------------------------------------------------------------------
_VIDEO_STD_EPS = 1e-8
_VIDEO_MIN_VALID = 2


def _video_mask_bct(mask, shape) -> Optional[np.ndarray]:
    """Normalise a video validity mask to ``(B, C, T)``; ``None`` passes through.

    Both layouts the loader can emit are accepted: the PER-FRAME ``(B, T)`` mask (the legacy
    ``VideoCodecPairDataset`` shape, which was unconditionally all-ones) and the PER-(channel,
    frame) ``(B, C, T)`` mask that ``cfg.mask_missing`` turns on. A ``(B, T)`` mask is
    broadcast across channels, which is exactly its meaning: "this whole frame is valid".
    """
    if mask is None:
        return None
    m = _to_numpy(mask)
    B, C, T = int(shape[0]), int(shape[1]), int(shape[2])
    if m.ndim == 2 and m.shape == (B, T):
        return np.broadcast_to(m[:, None, :], (B, C, T)).copy()
    if m.ndim == 3 and m.shape == (B, C, T):
        return m
    raise ValueError(f"video mask must be (B,T)={(B, T)} or (B,C,T)={(B, C, T)}; got {m.shape}")


def _video_flat(x: np.ndarray, m: Optional[np.ndarray]):
    """(B,C,T,H,W) [+ (B,C,T) mask] -> (B*C, T*H*W) rows + a broadcast row mask."""
    B, C, T, H, W = x.shape
    r = x.reshape(B * C, T * H * W).astype(np.float64)
    if m is None:
        return r, np.ones_like(r)
    mm = (m > 0.5).astype(np.float64)                    # (B, C, T)
    mm = np.repeat(mm.reshape(B * C, T, 1), H * W, axis=2).reshape(B * C, T * H * W)
    return r, mm


def _video_nrmse_corr(recon: np.ndarray, target: np.ndarray, mask: Optional[np.ndarray]):
    """(nrmse_mean, nrmse_med, nrmse_pooled, corr, valid_frac, present_frac) per (window, channel).

    Identical arithmetic to :func:`_slowts_nrmse_corr`, on the (T, H, W) volume instead of
    (C, T). Groups whose target is constant over its valid entries (a dead camera) are
    EXCLUDED from both means and counted in ``valid_frac``.
    """
    r, m = _video_flat(recon, mask)
    t, _ = _video_flat(target, mask)
    n = m.sum(axis=1)
    present_frac = float(m.mean())
    safe_n = np.maximum(n, 1.0)
    t_mu = (t * m).sum(axis=1) / safe_n
    r_mu = (r * m).sum(axis=1) / safe_n
    tc = (t - t_mu[:, None]) * m
    rc = (r - r_mu[:, None]) * m
    t_std = np.sqrt((tc * tc).sum(axis=1) / safe_n)
    rmse = np.sqrt((((r - t) ** 2) * m).sum(axis=1) / safe_n)
    valid = (n >= _VIDEO_MIN_VALID) & (t_std > _VIDEO_STD_EPS)
    if not valid.any():
        return (float("nan"),) * 4 + (0.0, present_frac)
    nrmse = rmse[valid] / t_std[valid]
    denom = np.sqrt((rc * rc).sum(axis=1) * (tc * tc).sum(axis=1))
    corr = np.where(denom > 0.0, (rc * tc).sum(axis=1) / np.maximum(denom, 1e-300), 0.0)
    sse = (((r - t) ** 2) * m).sum(axis=1)[valid].sum()
    sst = ((tc * tc).sum(axis=1))[valid].sum()
    pooled = float(np.sqrt(sse / sst)) if sst > 0 else float("nan")
    return (float(nrmse.mean()), float(np.median(nrmse)), pooled,
            float(corr[valid].mean()), float(valid.mean()), present_frac)


def _video_std_ratio(recon: np.ndarray, target: np.ndarray, mask: Optional[np.ndarray]):
    """AMPLITUDE diagnostic std(recon)/std(target), ideal EXACTLY 1.0. Never a ranking key.

    Returns ``(mean, median, pooled, temporal)``:

      * the first three group per (window, channel) over the (T, H, W) volume -- the SAME
        grouping as the nRMSE, so the two numbers are directly comparable. This mixes the
        spatial contrast of the frame with the temporal swing.
      * ``temporal`` groups per (window, channel, PIXEL) over the TIME axis only, variance-
        weighted. This is the one that catches a reconstruction whose still image is right but
        whose video is FROZEN: it reads ~0 there while the spatial ratio reads ~1. The
        ``tmean`` baseline scores EXACTLY 0.0 on it by construction.
    """
    r, m = _video_flat(recon, mask)
    t, _ = _video_flat(target, mask)
    n = m.sum(axis=1)
    safe_n = np.maximum(n, 1.0)
    t_mu = (t * m).sum(axis=1) / safe_n
    r_mu = (r * m).sum(axis=1) / safe_n
    ss_t = ((((t - t_mu[:, None]) * m) ** 2)).sum(axis=1)
    ss_r = ((((r - r_mu[:, None]) * m) ** 2)).sum(axis=1)
    valid = (n >= _VIDEO_MIN_VALID) & (np.sqrt(ss_t / safe_n) > _VIDEO_STD_EPS)
    if not valid.any():
        return (float("nan"),) * 4
    ratio = np.sqrt(ss_r[valid] / ss_t[valid])
    pooled = float(np.sqrt(ss_r[valid].sum() / ss_t[valid].sum()))

    # --- per (window, channel, pixel) over TIME only ---------------------------------
    B, C, T, H, W = recon.shape
    r5 = recon.astype(np.float64)
    t5 = target.astype(np.float64)
    if mask is None:
        m5 = np.ones((B, C, T, 1, 1))
    else:
        m5 = (mask > 0.5).astype(np.float64)[..., None, None]        # (B, C, T, 1, 1)
    n_t = m5.sum(axis=2)                                             # (B, C, 1, 1)
    safe_t = np.maximum(n_t, 1.0)
    t_mu_t = (t5 * m5).sum(axis=2, keepdims=True) / safe_t[:, :, None]
    r_mu_t = (r5 * m5).sum(axis=2, keepdims=True) / safe_t[:, :, None]
    ss_t_t = ((((t5 - t_mu_t) * m5) ** 2)).sum(axis=2)               # (B, C, H, W)
    ss_r_t = ((((r5 - r_mu_t) * m5) ** 2)).sum(axis=2)
    ok = np.broadcast_to(n_t >= _VIDEO_MIN_VALID, ss_t_t.shape) & (ss_t_t > _VIDEO_STD_EPS)
    if ok.any() and ss_t_t[ok].sum() > 0:
        ratio_time = float(np.sqrt(ss_r_t[ok].sum() / ss_t_t[ok].sum()))
    else:
        ratio_time = float("nan")
    return float(ratio.mean()), float(np.median(ratio)), pooled, ratio_time


def _video_hf_ratio(recon: np.ndarray, target: np.ndarray, mask: Optional[np.ndarray]) -> float:
    """SHARPNESS ratio: recon HF gradient energy / target HF gradient energy. Ideal 1.0.

    The masked twin of ``video_decode_fidelity``'s ``sharpness``. Spatial (H and W) first
    differences are taken only on VALID frames, and temporal first differences only on
    consecutive frame PAIRS that are both valid -- otherwise the step from a real frame into a
    zero-filled dead one is counted as signal.

    READ IT AS A PAIR WITH ``patch_lattice_ratio``. A high HF ratio can be pure checkerboard: on
    the old mhr codec 92% of the measured HF energy was the patch lattice, and notching the
    lattice bins IMPROVED every other metric. Low always means smooth; high means EITHER real
    texture OR artifact, and only the lattice ratio separates the two.
    """
    r = recon.astype(np.float64)
    t = target.astype(np.float64)
    B, C, T, H, W = r.shape
    if mask is None:
        fm = np.ones((B, C, T), dtype=bool)
    else:
        fm = mask > 0.5
    sp = fm[..., None, None]                                # (B, C, T, 1, 1) valid frames
    tp = (fm[:, :, :-1] & fm[:, :, 1:])[..., None, None]    # consecutive-valid pairs

    def _e(x):
        return (float(((np.diff(x, axis=-2) * sp) ** 2).sum())
                + float(((np.diff(x, axis=-1) * sp) ** 2).sum())
                + float(((np.diff(x, axis=2) * tp) ** 2).sum()))

    e_t = _e(t)
    return float(_e(r) / e_t) if e_t > 1e-12 else float("nan")


def full_video_metrics(recon, target, mask=None, prefix: str = "") -> Dict[str, float]:
    """FULL-array (B, C, T, H, W) video reconstruction fidelity -- no axis averaged away.

    The video analogue of :func:`full_slowts_metrics`. Unlike ``video_decode_fidelity``'s
    ``envelope_corr``, which averages the whole 50 ms time axis away first and therefore scores
    1.0 for a predictor with ZERO temporal structure, this keeps every pixel of every frame.

    Args:
        recon:  (B, C, T, H, W) reconstruction, in the codec's standardized frame space.
        target: (B, C, T, H, W) ground truth, same space.
        mask:   optional (B, C, T) per-(channel, frame) validity (1 = camera recording).
                Every statistic is over valid entries only; ``None`` = all valid.
        prefix: prepended to every key (used by the trivial baselines).

    Returns (keys prefixed by ``prefix``):
        ``video_nrmse`` / ``_med`` / ``_pooled`` -- RMSE/std(target) per (window, channel) over
            the flattened (T, H, W) volume; mean, median and the variance-weighted (pooled)
            form. 0.0 = perfect, EXACTLY 1.0 = predicting that camera's own constant mean.
        ``video_corr``          -- Pearson over the same valid entries.
        ``video_std_ratio`` / ``_med`` / ``_pooled`` / ``_t`` -- AMPLITUDE, ideal EXACTLY 1.0;
            ``_t`` is the purely TEMPORAL ratio (frozen video reads ~0).
        ``video_hf_ratio``      -- masked sharpness ratio, ideal 1.0. Read with the lattice.
        ``video_valid_frac``    -- fraction of (window, channel) groups with a non-constant
            target (the rest are dead cameras, excluded from every mean).
        ``video_present_frac``  -- mean validity-mask occupancy of the evaluated windows.
    """
    r = _to_numpy(recon).astype(np.float64)
    t = _to_numpy(target).astype(np.float64)
    if r.shape != t.shape:
        raise ValueError(f"full_video_metrics: shape mismatch {r.shape} vs {t.shape}")
    if r.ndim != 5:
        raise ValueError(f"full_video_metrics expects (B, C, T, H, W); got {r.shape}")
    m = _video_mask_bct(mask, r.shape)
    nrmse, nrmse_med, pooled, corr, valid, present = _video_nrmse_corr(r, t, m)
    sr, sr_med, sr_pooled, sr_t = _video_std_ratio(r, t, m)
    return {
        f"{prefix}video_nrmse": nrmse,
        f"{prefix}video_nrmse_med": nrmse_med,
        f"{prefix}video_nrmse_pooled": pooled,
        f"{prefix}video_corr": corr,
        f"{prefix}video_std_ratio": sr,
        f"{prefix}video_std_ratio_med": sr_med,
        f"{prefix}video_std_ratio_pooled": sr_pooled,
        f"{prefix}video_std_ratio_t": sr_t,
        f"{prefix}video_hf_ratio": _video_hf_ratio(r, t, m),
        f"{prefix}video_valid_frac": valid,
        f"{prefix}video_present_frac": present,
    }


def _video_masked_mean(t: np.ndarray, m: Optional[np.ndarray], axis) -> np.ndarray:
    """Mean of ``t`` (B,C,T,H,W) over ``axis`` using only ``m``-valid (B,C,T) frames."""
    if m is None:
        return t.mean(axis=axis, keepdims=True)
    m5 = (m > 0.5).astype(np.float64)[..., None, None]
    num = (t * m5).sum(axis=axis, keepdims=True)
    den = np.broadcast_to(m5, t.shape).sum(axis=axis, keepdims=True)
    return np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)


def trivial_video_baselines(target, mask=None) -> Dict[str, float]:
    """The reference points that make :func:`full_video_metrics` interpretable.

    Every baseline is scored with the SAME function, the same mask and the same units against
    the same target batch, so the codec row and the baseline rows are directly comparable:

      * ``base_self_*``   -- the target predicting itself. MUST read nrmse 0.0 / corr 1.0 /
        std_ratio 1.0; the self-check that the metric is wired correctly.
      * ``base_tmean_*``  -- the target's own per-PIXEL time-averaged frame, broadcast back over
        all T frames. "Perfect still image, ZERO temporal structure" -- exactly what
        ``video_decode_fidelity``'s ``envelope_corr`` scores 1.0, and exactly what a frozen
        reconstruction is. **THE bar any honest video reconstruction has to clear.**
      * ``base_cmean_*``  -- the per-(channel, pixel) mean over the WHOLE evaluated batch: one
        static image per camera for every window, no per-window information at all.
      * ``base_wcmean_*`` -- the per-(window, channel) scalar mean broadcast back. Scores nrmse
        EXACTLY 1.0000 by construction -- that is the normalisation anchor -- and corr 0.0.
    """
    t = _to_numpy(target).astype(np.float64)
    if t.ndim != 5:
        raise ValueError(f"trivial_video_baselines expects (B, C, T, H, W); got {t.shape}")
    m = _video_mask_bct(mask, t.shape)
    if m is not None:
        m = (m > 0.5).astype(np.float64)
    tmean = np.broadcast_to(_video_masked_mean(t, m, 2), t.shape)             # (B,C,1,H,W)
    cmean = np.broadcast_to(_video_masked_mean(t, m, (0, 2)), t.shape)        # (1,C,1,H,W)
    wcmean = np.broadcast_to(_video_masked_mean(t, m, (2, 3, 4)), t.shape)    # (B,C,1,1,1)
    out: Dict[str, float] = {}
    for name, pred in (("self", t), ("tmean", tmean), ("cmean", cmean), ("wcmean", wcmean)):
        mm = full_video_metrics(pred, t, mask=m, prefix=f"base_{name}_")
        mm.pop(f"base_{name}_video_valid_frac", None)
        mm.pop(f"base_{name}_video_present_frac", None)
        out.update(mm)
    return out


def video_patch_lattice(
    vid, patch_h: int, patch_w: int, mask=None, ring: int = 3
) -> Dict[str, float]:
    """Patch-lattice (checkerboard) strength of a VIDEO batch -- a thin adapter, not a new metric.

    The video decoder paints every ``patch_h x patch_w`` spatial patch from its own token
    through the SHARED ``to_pixels`` linear and tiles them, exactly as the spectro decoder does
    over (freq, time). So the artifact is the same object and the DETECTOR is reused verbatim:
    each valid frame of each channel is handed to :func:`patch_lattice_metrics` as a
    ``(N, 1, H, W)`` batch with ``(patch_f, patch_t) = (patch_h, patch_w)``.

    **1.0 = no lattice.** Validated reference points (2026-09-03, 720 held-out tangtv_lower
    frames): ground truth reads 1.10; the production 64k-code video codec's reconstruction
    reads 24.99. Read it as a PAIR with ``video_hf_ratio`` -- blurring drives the lattice down
    for free, and only holding the HF ratio while the lattice falls is a real fix.

    ``mask`` (B, C, T) drops dead-camera frames; a zero plate has no lattice and would dilute
    the measurement toward 1.0, flattering a codec exactly where it is doing nothing.
    """
    x = _to_numpy(vid).astype(np.float64)
    if x.ndim != 5:
        raise ValueError(f"video_patch_lattice expects (B, C, T, H, W); got {x.shape}")
    B, C, T, H, W = x.shape
    frames = x.reshape(B * C * T, 1, H, W)
    mask = _video_mask_bct(mask, x.shape)
    if mask is not None:
        keep = (np.asarray(mask) > 0.5).reshape(B * C * T)
        if keep.any():
            frames = frames[keep]
    return patch_lattice_metrics(frames, patch_h, patch_w, ring=ring)


def code_rate_bits(codes, codebook_size: int, n_tok: Optional[int] = None) -> Dict[str, float]:
    """Codebook utilization expressed as a BIT RATE, not just a distinct-code count.

    ``frac_codes_used`` answers "how many symbols did we ever emit"; it counts a code used once
    the same as one used a million times, and its ceiling is a coupon-collector bound rather
    than 1.0. The honest question for a tokenizer is how many of the ``n_tok * log2(K)`` bits
    the frame layout PAYS FOR are actually delivered.

    Args:
        codes: (B, n_tok, fsq_dim) or (B, n_frames, n_tok, fsq_dim) integer FSQ codes.
        codebook_size: prod(fsq_levels) -- the K whose log2 is the per-token budget.
        n_tok: tokens per frame; inferred from ``codes`` when omitted.

    Returns:
        ``bits_available_per_frame`` = n_tok * log2(K) -- what the world-model frame is charged.
        ``bits_pooled_per_token``    = H(joint code, ALL positions pooled) / ln 2.
        ``bits_positional_per_token``= mean over the n_tok POSITIONS of that position's own code
            entropy, in bits. This is the stricter, honest rate: a codec whose token 7 always
            emits the same code delivers zero bits there however diverse the pooled histogram
            looks. ``bits_pooled`` >= ``bits_positional`` always.
        ``rate_pooled`` / ``rate_positional`` = those divided by log2(K), in [0, 1]. **This is
            the ">= 90% utilization" number** -- but read it against the ceilings below.
        ``rate_positional_ceiling`` / ``rate_pooled_ceiling`` -- the LARGEST value the estimate
            could take on this eval set, ``log2(n_samples)/log2(K)``. An entropy estimated from
            M samples cannot exceed log2(M): at 320 windows and K=1000 the positional ceiling
            is 8.32/9.97 = 0.835, so a perfect codec CANNOT read 0.90 positionally on 320
            windows. The pooled rate uses n_windows * n_tok samples and is unconstrained in
            practice. Quote a rate as a fraction OF ITS CEILING when the two are close.
        ``n_distinct_codes`` / ``frac_codes_used`` for continuity with :func:`utilization`.
    """
    arr = _as_int_codes(codes)
    if arr.ndim == 4:
        B, Fr, nt, fd = arr.shape
        arr = arr.reshape(B * Fr, nt, fd)
    if arr.ndim != 3:
        raise ValueError(f"code_rate_bits expects (B, n_tok, fsq_dim); got {arr.shape}")
    N, nt, fd = arr.shape
    n_tok = int(n_tok) if n_tok is not None else nt
    log2K = float(np.log2(max(2, int(codebook_size))))

    flat = arr.reshape(-1, fd)
    _, cnt = np.unique(flat, axis=0, return_counts=True)
    pp = cnt.astype(np.float64) / cnt.sum()
    pooled_bits = float(-(pp * np.log2(pp)).sum())

    per_pos = []
    for j in range(nt):
        _, c = np.unique(arr[:, j, :], axis=0, return_counts=True)
        q = c.astype(np.float64) / c.sum()
        per_pos.append(float(-(q * np.log2(q)).sum()))
    positional_bits = float(np.mean(per_pos))

    n_distinct = int(np.unique(flat, axis=0).shape[0])
    # EVAL-SIZE CEILINGS. An entropy estimated from M samples cannot exceed log2(M), so the
    # positional rate has a hard ceiling of log2(N_windows)/log2(K) -- at 320 windows and
    # K=1000 that is 8.32/9.97 = 0.835, i.e. a PERFECT codec cannot read 0.90 here. Report
    # both ceilings so a rate is never mistaken for a shortfall of the codec.
    ceil_pos = float(np.log2(max(2, N))) / log2K
    ceil_pool = float(np.log2(max(2, flat.shape[0]))) / log2K
    return {
        "bits_available_per_frame": float(n_tok) * log2K,
        "rate_positional_ceiling": min(1.0, ceil_pos),
        "rate_pooled_ceiling": min(1.0, ceil_pool),
        "n_windows": int(N),
        "bits_pooled_per_token": pooled_bits,
        "bits_positional_per_token": positional_bits,
        "rate_pooled": pooled_bits / log2K,
        "rate_positional": positional_bits / log2K,
        "bits_delivered_per_frame_pooled": float(n_tok) * pooled_bits,
        "bits_delivered_per_frame_positional": float(n_tok) * positional_bits,
        "n_distinct_codes": n_distinct,
        "frac_codes_used": float(n_distinct / max(1, int(codebook_size))),
        "n_observed_tokens": int(flat.shape[0]),
    }


__all__ += [
    "full_video_metrics",
    "trivial_video_baselines",
    "video_patch_lattice",
    "code_rate_bits",
]
