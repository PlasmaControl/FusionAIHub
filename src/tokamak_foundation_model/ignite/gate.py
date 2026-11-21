"""Codec acceptance metrics (the oracle gate), ``docs/IGNITE_DESIGN.md`` §4.4.

The gate decides whether a codec's codes are good enough to train the Phase-B
dynamics model on. Phase A and Phase B are coupled only through this gate (§6): no
gradient from the dynamics model reaches the codes. Each metric is checked against
synthetic ground truth in ``tests/ignite/test_gate.py``.

Targets from §4.4:

- Stability >= 0.80: fraction of code positions unchanged under the nuisance
  transform (raw δ-shift plus re-STFT). A stable codec encodes the statistic, not
  the STFT-phase realization.
- Persistence >= 0.50 on steady segments (old codecs churned at about 0.10), and
  lower on transitions.
- Forecastability: a per-position first-order Markov probe beats persistence at
  predicting the next frame's code distribution, reported separately on transition
  and stable steps.
- Mode-bearing decode: the reconstruction keeps the target's mode structure
  (per-frequency envelope correlation, peak-overlap F1) and is not blurred
  (high-frequency gradient-energy ratio near 1).

Following §7 this module reuses no FAITH model code. It needs only numpy; torch
tensors are accepted and converted. Everything runs on CPU on small arrays.
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


# Tensor helpers: accept numpy or torch, return numpy.
def _to_numpy(x) -> np.ndarray:
    """Convert a numpy array, torch tensor or array-like to a numpy array.

    Torch is not imported here; tensors are recognised by their ``.detach`` method.
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
        # FSQ indices are integers; round in case they arrive as floats.
        arr = np.rint(arr).astype(np.int64)
    return arr


# 1. Stability under the nuisance (δ-shift) transform
def stability(codes_x, codes_shifted) -> float:
    """Fraction of code positions unchanged under the nuisance transform.

    The nuisance is the raw δ-shift plus re-STFT pair of §4.2; a statistics-first
    codec maps both members to nearly the same codes. A position is one entry of the
    ``fsq_dim`` vector at one token, so a token whose FSQ dimensions flip partly
    counts partly.

    Parameters
    ----------
    codes_x : array of shape (B, n_tok, fsq_dim)
        Integer codes for the original window.
    codes_shifted : array of shape (B, n_tok, fsq_dim)
        Integer codes for the δ-shifted window.

    Returns
    -------
    float
        Fraction in [0, 1]; 1.0 when every position matches.
    """
    a = _as_int_codes(codes_x)
    b = _as_int_codes(codes_shifted)
    if a.shape != b.shape:
        raise ValueError(f"stability: shape mismatch {a.shape} vs {b.shape}")
    if a.size == 0:
        raise ValueError("stability: empty code tensor")
    return float(np.mean(a == b))


# 2. Persistence: frame-to-frame code stickiness
def persistence(codes_t, codes_tp1, steady_mask: Optional[object] = None) -> float:
    """Fraction of code positions unchanged from frame t to frame t+1 (50 ms).

    On steady segments a statistics-first codec should mostly repeat its codes
    (target >= 0.50; old codecs churned at about 0.10). On transitions it is
    expected to be lower.

    Parameters
    ----------
    codes_t, codes_tp1 : array of shape (B, n_tok, fsq_dim)
        Integer codes at frames t and t+1.
    steady_mask : bool array, optional
        Selects the steady subset. Broadcast against the leading dims of the codes,
        typically (B,) or the full (B, n_tok, fsq_dim).

    Returns
    -------
    float
        Fraction in [0, 1] of unchanged positions, over the masked subset if given.
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


# 3. Forecastability: cheap next-frame probe against persistence
def _stack_frame_transitions(codes_seq: np.ndarray):
    """Flatten (B, n_frames, n_tok, fsq_dim) codes into aligned t -> t+1 pairs.

    Each ``(token, fsq_dim)`` entry is its own categorical stream (the per-code
    Markov table of §4.4).

    Returns
    -------
    src, tgt : int arrays of shape (N_trans, P)
        Codes at frames t and t+1, with ``P = n_tok * fsq_dim`` and
        ``N_trans = B * (n_frames - 1)``.
    moved : bool array of shape (N_trans,)
        Whether any position changed at that transition.
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
    """Next-frame code-distribution probe against a persistence baseline (§4.4).

    This is not the dynamics model. A per-position first-order Markov table is fit
    to the transition counts of ``codes_seq`` and scored by the mean log-likelihood
    of the true next code. The baseline is persistence (next == current), smoothed
    into a proper distribution. A margin is probe minus persistence; positive means
    the probe wins.

    A transition step is one where at least one position changed; a stable step is
    one where nothing did. On stable steps persistence is already optimal, so the
    probe can at best tie. On predictable transitions (e.g. a drifting index) the
    probe should win. A frozen sequence has no transitions and reports a transition
    margin of 0.0.

    Parameters
    ----------
    codes_seq : array of shape (B, n_frames, n_tok, fsq_dim)
        Integer codes.
    transition_mask : bool array of shape (B, n_frames - 1), optional
        Overrides the data-derived transition/stable split.
    laplace : float
        Additive smoothing count for both models, so unseen symbols stay finite.

    Returns
    -------
    dict
        ``margin_transition``, ``margin_stable`` (about 0 by construction) and
        ``margin_overall``; the raw mean log-likelihoods ``probe_ll_*`` and
        ``persistence_ll_*`` for both strata; the step counts ``n_transition`` and
        ``n_stable``; and ``beats_persistence`` (``margin_transition > 0``, the pass
        condition).
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

    # The persistence baseline leaks a little probability to other symbols so it is
    # a proper distribution. The leak shrinks with the data size, like Laplace
    # smoothing, and is capped at 0.05 so the baseline stays a persistence predictor.
    eps = min(0.05, laplace / (laplace + n_trans))

    # Fit the Markov table on all transitions (the probe is a fitted predictor, not
    # an oracle) and accumulate the per-step log-likelihood averaged over positions.
    probe_ll = np.zeros(n_trans, dtype=np.float64)
    persist_ll = np.zeros(n_trans, dtype=np.float64)

    for p in range(P):
        s = src[:, p]
        t = tgt[:, p]
        vmax = int(max(s.max(), t.max())) + 1  # symbols seen at this position

        # Markov probe: P(next=j | cur=i) from Laplace-smoothed counts.
        counts = np.zeros((vmax, vmax), dtype=np.float64)
        np.add.at(counts, (s, t), 1.0)
        counts += laplace
        trans_prob = counts / counts.sum(axis=1, keepdims=True)  # rows sum to 1
        probe_ll += np.log(trans_prob[s, t])

        # Persistence: 1 - eps on next == cur, eps spread over the other symbols.
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


# 4. Decode fidelity of spectrogram reconstructions
def _power_envelope(spec: np.ndarray) -> np.ndarray:
    """Mean power over time per (batch, channel, freq) of a (B, C, F, T) spectrogram.

    This "which frequencies carry power" profile is the mode-presence statistic the
    codec should preserve.
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
            # A flat envelope has nothing to correlate; count it as 0.
            corrs.append(0.0)
        else:
            corrs.append(float((a * b).sum() / denom))
    return float(np.mean(corrs))


def _peak_mask(env: np.ndarray, k: float) -> np.ndarray:
    """Per-row mode mask of an (N, F) envelope: bins above the row mean + k * std."""
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
    """Sum of squared first differences along frequency and time.

    A sharpness proxy: a blurred or mean-collapsed spectrogram has much less of this
    than a sharp, mode-bearing one.
    """
    df = np.diff(spec, axis=-2)  # along frequency
    dt = np.diff(spec, axis=-1)  # along time
    return float((df ** 2).sum() + (dt ** 2).sum())


# 4a. Patch-lattice artifact metrics
#
# The spectro decoder paints each (patch_f x patch_t) patch from its own token
# through one shared linear map (``nets.SpectroDecoder.to_pixels`` plus a shared
# bias) and tiles the patches. Any high-frequency texture in that shared basis is
# repeated on a patch-sized grid, visible as a regular lattice in the decoded
# spectrogram. ``envelope_corr``, ``peak_f1`` and ``sharpness`` cannot see it; the
# lattice is high-frequency energy, so it can even raise ``sharpness``. On one mhr
# codec with 16x16 patches, 56% of the reconstruction's 2-D spectral energy sat on
# the lattice, against 3.4% for the ground truth.
def _lattice_harmonics(F: int, T: int, patch_f: int, patch_t: int):
    """DFT bins (kf, kt) of a ``patch_f x patch_t`` tiling of an (F, T) image.

    Tiling one patch texture along an axis of length F puts all its energy on the
    multiples of ``F // patch_f``. Returns the sorted harmonic bins, without DC.
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

    Parameters
    ----------
    spec : array of shape (B, C, F, T)
        Log-power spectrogram, reconstruction or ground truth.
    patch_f, patch_t : int
        Codec patch size in frequency bins and STFT frames.
    ring : int
        Half-width of the local background window around each harmonic bin.

    Returns
    -------
    dict
        ``patch_lattice_ratio``: the main scalar. With each image's mean removed and
        2-D FFT magnitudes averaged over (B, C), each lattice harmonic is divided by
        the median of its (2 * ring + 1)^2 neighbourhood (harmonics excluded); the
        result is the geometric mean over harmonics. 1.0 means no lattice. The local
        normalisation is what keeps real spectrograms, whose spectral energy falls
        steeply with frequency, near 1.0.

        ``lattice_energy_frac``: share of the mean power spectrum (DC excluded) on
        the lattice harmonics; about 0.03 on real mhr spectrograms.

        ``seam_ratio_freq`` / ``seam_ratio_time``: mean |first difference| across
        patch boundaries over a local interior baseline (+-ring, boundaries
        excluded); 1.0 means no seam. Diagnostic only: a decoder whose shared
        texture is smooth at its edges shows a strong lattice with seam ratios
        near 1.

    Notes
    -----
    Read it beside ``envelope_corr``: a collapsed codec emits the same patch
    everywhere, which is a perfect lattice (1.4e5 on a one-code codec). A high ratio
    means "grid artifact" only for a codec that reconstructs.

    Read it beside ``sharpness`` too: blurring lowers the ratio (61.7 to 3.1 at a
    Gaussian sigma of 2 on an mhr reconstruction) while the HF-gradient ratio falls
    with it (0.82 to 0.0024). A real fix lowers the lattice ratio and holds
    ``sharpness``.
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

    # 2-D FFT lattice-harmonic ratio
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

    # Locally normalised seam ratios at patch boundaries
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
    """Zero the patch-lattice DFT bins of a (B, C, F, T) spectrogram.

    A diagnostic for rendering and audits; no loss or codec uses it. It removes
    exactly the bins :func:`patch_lattice_metrics` scores, to show how much of a
    reconstruction is the artifact. On one mhr codec (16x16 patches) the notch cut
    the HF-gradient ratio from 0.82 to 0.063, while envelope correlation rose by
    0.05, peak F1 by 0.06 and pixel L1 fell by 13%. The ``sharpness`` of a
    lattice-bearing codec mostly counts the artifact.
    """
    x = np.asarray(spec, dtype=np.float64)
    F, T = x.shape[-2], x.shape[-1]
    keep = np.ones((F, T), dtype=np.float64)
    for a, b in _lattice_harmonics(F, T, patch_f, patch_t):
        keep[a, b] = 0.0
    return np.real(np.fft.ifft2(np.fft.fft2(x, axes=(-2, -1)) * keep, axes=(-2, -1)))


# 4c. Full-spectrogram reconstruction metrics (no axis averaged away)
#
# The metrics above compare a time-collapsed view or a single ratio.
# ``envelope_corr`` and ``peak_f1`` run on ``spec.mean(axis=-1)``, which drops the
# whole 50 ms time axis and with it every mode track, burst edge and chirp.
# ``sharpness`` is dominated by the patch lattice (see :func:`remove_patch_lattice`).
# The metrics below compare the full (F, T) array per (window, channel). Read them
# beside :func:`trivial_spectro_baselines`, in the same units: on held-out mhr
# windows every codec scored ``spec_nrmse >= 1`` and lost on both metrics to the
# time-averaged envelope (nrmse 0.88, corr2d 0.44), the predictor that
# ``envelope_corr`` scores 1.0.

#: Upper edge (exclusive) of the mode band for the ``*_band`` variants, in STFT
#: bins. 512 bins span 0-250 kHz (488.28 Hz per bin), so 120 bins = 0-58.6 kHz, the
#: band of the coherent MHD mode tracks.
DEFAULT_MODE_BAND_BINS: int = 120

_SPEC_STD_EPS: float = 1e-8


def _spectro_mask_bct(mask, shape) -> Optional[np.ndarray]:
    """Normalise a spectro validity mask to a (B, C, T) bool array, or None.

    Accepts (B, T), broadcast over channels, or (B, C, T); ``shape`` is the
    (B, C, F, T) spectrogram shape. Returns None when there is no mask or it selects
    everything, so callers take the unmasked path and give identical results.
    Mirrors ``_video_mask_bct``.
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
    """(nrmse, corr2d, valid_frac) over the full (F, T) array per (window, channel).

    For each of the ``B * C`` pairs, over its ``F * T`` elements::

        nrmse = sqrt(mean((recon - target)^2)) / std(target)
        corr2d = Pearson(recon.ravel(), target.ravel())

    Pairs with a constant target (std <= 1e-8, an absent or dead channel) are left
    out of both means and counted in ``valid_frac``. A constant recon against a live
    target is counted: corr2d 0 and nrmse >= 1 (exactly 1 when the constant is the
    target's mean).
    """
    B, C, F, T = recon.shape
    r = recon.reshape(B * C, -1)
    t = target.reshape(B * C, -1)
    m = _spectro_mask_bct(mask, recon.shape)
    if m is None:
        # Unmasked: statistics over all F * T elements.
        w = None
        n = np.full(B * C, float(r.shape[1]))
        rmean = r.mean(axis=1, keepdims=True)
        tmean = t.mean(axis=1, keepdims=True)
    else:
        # Masked. Missingness is per (window, channel, frame), so the (B, C, T) mask
        # broadcasts over F, and every statistic (means, target std, RMSE,
        # correlation sums) is taken over valid elements only, as in the slow-TS
        # gate.
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
    # A pair needs a non-constant target and, when masked, at least 2 valid elements.
    valid = (t_std > _SPEC_STD_EPS) & (n >= 2)
    if not valid.any():
        return float("nan"), float("nan"), 0.0
    nrmse = rmse[valid] / t_std[valid]
    # denom == 0 means one side is constant: correlation 0, as in
    # _envelope_correlation. np.maximum only avoids the division np.where still runs.
    corr = np.where(denom > 0.0, num / np.maximum(denom, 1e-300), 0.0)
    return float(nrmse.mean()), float(corr[valid].mean()), float(valid.mean())


def full_spectro_metrics(
    recon,
    target,
    band_bins: Optional[int] = DEFAULT_MODE_BAND_BINS,
    prefix: str = "",
    mask=None,
) -> Dict[str, float]:
    """Full-array (B, C, F, T) reconstruction fidelity, no axis averaged away.

    Parameters
    ----------
    recon, target : array of shape (B, C, F, T)
        Reconstructed and ground-truth log-power spectrograms.
    band_bins : int or None
        If set, also report both metrics on frequency bins ``[0, band_bins)``
        (default :data:`DEFAULT_MODE_BAND_BINS`, 0-58.6 kHz) under a ``_band``
        suffix. The full-band numbers are unaffected.
    prefix : str
        Prepended to every key (used for the trivial baselines).
    mask : array, optional
        (B, T) or (B, C, T) validity mask; statistics use valid samples only.

    Returns
    -------
    dict
        Keys, each with ``prefix``:

        - ``spec_nrmse``: RMSE / std(target) per (window, channel) over the
          flattened (F, T) array, averaged over pairs. The divisor is that pair's
          own target std, so predicting the pair's constant mean scores exactly 1.0:
          below 1 beats the window's DC level, 0 is perfect. It ignores each
          window's log-power offset and gain, which matters because windows are
          z-scored per modality, not per window.
        - ``spec_corr2d``: Pearson correlation over the flattened (F, T) array,
          averaged over pairs. It is ``envelope_corr`` without first deleting the
          time axis; a codec that only reproduces the time-averaged spectrum scores
          much lower here.
        - ``spec_valid_frac``: fraction of pairs with a non-constant target.
        - ``spec_nrmse_band`` / ``spec_corr2d_band``: the same on bins
          ``[0, band_bins)``, present when ``band_bins`` is set.

    ``tests/ignite/test_gate.py`` checks that the target scores nrmse 0 / corr2d 1
    against itself and a mean-collapsed recon scores 1 / 0.
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
    """Reference points for :func:`full_spectro_metrics`, scored by it on one target.

    - ``base_self_*``: the target against itself. Must read nrmse 0 / corr2d 1; it
      checks the wiring.
    - ``base_tmean_*``: the target's own time-averaged envelope broadcast over all T
      frames, a perfect envelope with no temporal structure. ``envelope_corr``
      scores it 1.0, so its ``spec_corr2d`` is the bar a reconstruction has to
      clear.
    - ``base_cfmean_*``: the per-(channel, frequency) mean over the batch and time,
      a single dataset-level spectrum.
    - ``base_wcmean_*``: the per-(window, channel) scalar mean. Scores nrmse exactly
      1.0 (the normalisation anchor) and corr2d 0. Its ``_band`` variant is only
      about 1.0 (1.025 on synthetic data), because the constant is the whole
      window's mean while the band metric divides by the in-band std.

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
        # Each baseline's own mean uses valid samples only; otherwise the baselines
        # are computed on fill and base_wcmean is no longer exactly 1.0.
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
        m.pop(f"base_{name}_spec_valid_frac", None)  # same for every baseline
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
    """Mode-structure match and sharpness of a reconstructed spectrogram (§4.4).

    Base keys, on (B, C, F, T) log-power spectrograms:

    - ``envelope_corr``: mean Pearson correlation of the per-frequency power
      envelope; near 1 when mode structure is kept.
    - ``peak_f1``: F1 of detected-mode overlap; drops when modes are missed or
      added.
    - ``sharpness``: high-frequency gradient energy, recon over target. Near 1 means
      not blurred, below 1 low-passed, above 1 noisier than the target.
    - ``hf_energy_recon`` / ``hf_energy_target``: the raw energies.

    With ``patch_f`` and ``patch_t`` the :func:`patch_lattice_metrics` keys are added
    for the reconstruction, and again with a ``target_`` prefix for the target as
    the control. The three metrics above cannot see a patch lattice, which is why
    these are reported beside them.

    With ``full_spec`` the :func:`full_spectro_metrics` and
    :func:`mode_structure_metrics` keys are added, the latter being what codecs are
    ranked on. These are the only keys that compare the full arrays rather than a
    time-collapsed view.

    Parameters
    ----------
    recon, target : array of shape (B, C, F, T)
        Reconstructed and ground-truth log-power spectrograms.
    peak_k : float
        Threshold of the peak detector, in standard deviations.
    patch_f, patch_t : int, optional
        Codec patch size. Both None (default) skips the lattice keys.
    full_spec : bool
        Add the full-array and mode-track metrics.
    band_bins : int or None
        Mode-band cut passed to :func:`full_spectro_metrics`; None drops the
        ``_band`` keys.
    mode_baseline : bool
        With ``full_spec``, also add the target-only baselines:
        :func:`trivial_spectro_baselines` and the time-mean envelope's mode-track
        scores (prefix ``base_tmean_``). They are the same for every codec on a
        given window set, so an audit computes them for the first codec only.
    detrended : bool
        Passed to :func:`mode_structure_metrics`.
    mask : array, optional
        (B, T) or (B, C, T) validity mask (see ``data.spectro_frame_mask``). The
        full-array metrics and every baseline mean use valid samples only, and
        windows with any invalid frame are dropped from the envelope, peak,
        sharpness and lattice metrics. None leaves the output unchanged.

    Returns
    -------
    dict
        The keys described above.
    """
    r = _to_numpy(recon).astype(np.float64)
    t = _to_numpy(target).astype(np.float64)
    if r.shape != t.shape:
        raise ValueError(f"decode_fidelity: shape mismatch {r.shape} vs {t.shape}")
    if r.ndim != 4:
        raise ValueError(
            f"decode_fidelity expects (B, C, F, T); got {r.shape}"
        )

    # The envelope, peak, sharpness and lattice metrics work on the whole (F, T)
    # plane and take no per-frame mask, so a window with any missing frame is
    # dropped, the same rule as the codec loss (SpectroCodec._valid_windows).
    # Spectro missingness is per (shot, channel), so this loses almost nothing.
    _mb = _spectro_mask_bct(mask, r.shape)
    if _mb is not None:
        _keep = _mb.all(axis=(1, 2))                  # (B,) fully-valid windows
        if _keep.any():
            r, t = r[_keep], t[_keep]
            mask = _mb[_keep]
        # If no window is fully valid, use the full arrays rather than return NaN.

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
            # Depends only on the target, so callers share it across codecs; it is
            # the most expensive call here.
            out.update(trivial_spectro_baselines(t, band_bins=band_bins, mask=mask))
        # Mode-track metrics (end of this module), over the full band: mhr mode
        # tracks reach about 90 kHz, above the 58.6 kHz band cut. Codecs are ranked
        # on these, since ``spec_nrmse`` ranks a blur best.
        out.update(mode_structure_metrics(r, t, band_bins=None, detrended=detrended))
        if mode_baseline:
            # The time-mean envelope's own mode-track scores, also target-only.
            _tm = np.broadcast_to(t.mean(axis=-1, keepdims=True), t.shape)
            out.update(mode_structure_metrics(_tm, t, band_bins=None, prefix="base_tmean_",
                                              detrended=detrended))
    return out


# 4b. Decode fidelity of video reconstructions
def _video_pixel_profile(vid: np.ndarray) -> np.ndarray:
    """Per-pixel time-mean intensity (B, C, H, W) of a (B, C, T, H, W) video.

    Where the divertor light sits; the video counterpart of the spectrogram's
    per-frequency power envelope.
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
    """Sum of squared first differences along height, width and time.

    A sharpness proxy: a blurred or mean-collapsed reconstruction has much less of
    this than the target.
    """
    dh = np.diff(vid, axis=-2)   # height
    dw = np.diff(vid, axis=-1)   # width
    dt = np.diff(vid, axis=2)    # time
    return float((dh ** 2).sum() + (dw ** 2).sum() + (dt ** 2).sum())


def video_decode_fidelity(recon, target, peak_k: float = 1.0) -> Dict[str, float]:
    """Video counterpart of :func:`decode_fidelity`, on (B, C, T, H, W) frames.

    Returns the same keys, so :func:`~ignite.spike.gate_score` consumes it
    unchanged, but the envelope is the per-pixel time-mean intensity map:

    - ``envelope_corr``: mean Pearson correlation of the intensity maps.
    - ``peak_f1``: F1 of bright-region overlap.
    - ``sharpness``: spatial plus temporal HF gradient energy, recon over target.
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


# 4c. Decode fidelity of slow-TS reconstructions
def _slowts_profile(sig: np.ndarray) -> np.ndarray:
    """Per-position time mean (B, C) of a (B, C, T) slow-TS window: the profile."""
    return sig.mean(axis=-1)  # (B, C)


def _slowts_profile_masked(sig: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """Per-position time mean over valid samples (``mask`` (B, C, T), 1 = valid).

    A position with no valid sample gets its plain time mean (the fill value). That
    is harmless: the correlation is over positions and such a position is constant
    either way.
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
    """Slow-TS counterpart of :func:`decode_fidelity`, on (B, C, T) windows.

    Returns the same keys, so :func:`~ignite.spike.gate_score` consumes it
    unchanged, but the envelope is the per-position time-mean profile:

    - ``envelope_corr``: mean Pearson correlation of the profiles.
    - ``peak_f1``: F1 of high-value-position overlap.
    - ``sharpness``: position plus time HF gradient energy, recon over target; below
      1 means over-smoothed.

    ``mask`` ((B, C, T), 1 = valid) restricts the profile average to valid samples,
    so beam-off and missing samples do not distort it. Sharpness uses the filled
    window: it is a recon/target ratio and the shared fill cancels.
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
    # Per-sample Pearson correlation of the (C,) profiles.
    corrs = []
    for i in range(er.shape[0]):
        a = er[i] - er[i].mean()
        b = et[i] - et[i].mean()
        denom = np.sqrt((a * a).sum() * (b * b).sum())
        corrs.append(0.0 if denom <= 1e-12 else float((a * b).sum() / denom))
    env_corr = float(np.mean(corrs))

    # High-value-position overlap F1, with the shared threshold detector.
    pr = _peak_mask(er, peak_k)
    pt = _peak_mask(et, peak_k)
    tp = np.logical_and(pr, pt).sum(axis=1).astype(np.float64)
    fp = np.logical_and(pr, ~pt).sum(axis=1).astype(np.float64)
    fn = np.logical_and(~pr, pt).sum(axis=1).astype(np.float64)
    denom = 2 * tp + fp + fn
    f1_rows = np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-12), 1.0)
    peak_f1 = float(f1_rows.mean())

    # Sharpness: position + time HF gradient energy, recon over target.
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


# 4d. Decode fidelity of fast-TS (ELM envelope) reconstructions
def fastts_decode_fidelity(recon, target, peak_k: float = 1.0) -> Dict[str, float]:
    """Fast-TS counterpart of :func:`decode_fidelity`, on the ELM activity envelope.

    Works on the (B, C, E) envelope (E envelope-time bins), not the raw spike
    waveform (§4.3), and returns the same keys so :func:`~ignite.spike.gate_score`
    consumes it unchanged. The statistic here is when and how much ELM activity
    there is, so unlike :func:`slowts_decode_fidelity` nothing is time-averaged:

    - ``envelope_corr``: mean Pearson correlation of each channel's envelope curve.
    - ``peak_f1``: F1 of high-activity-bin (burst) overlap, the burst-detection
      metric §4.4 asks for.
    - ``sharpness``: time-axis HF gradient energy, recon over target; below 1 means
      burst edges are smoothed.
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

    # Per-channel Pearson correlation of the envelope curve.
    corrs = []
    for i in range(B * C):
        a = rr[i] - rr[i].mean()
        b = tt[i] - tt[i].mean()
        denom = np.sqrt((a * a).sum() * (b * b).sum())
        # A flat envelope curve (no activity) has no structure to correlate; treat as 0.
        corrs.append(0.0 if denom <= 1e-12 else float((a * b).sum() / denom))
    env_corr = float(np.mean(corrs))

    # Burst-overlap F1: a bin is a burst if it exceeds row mean + k * row std.
    pr = _peak_mask(rr, peak_k)
    pt = _peak_mask(tt, peak_k)
    tp = np.logical_and(pr, pt).sum(axis=1).astype(np.float64)
    fp = np.logical_and(pr, ~pt).sum(axis=1).astype(np.float64)
    fn = np.logical_and(~pr, pt).sum(axis=1).astype(np.float64)
    denom = 2 * tp + fp + fn
    f1_rows = np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-12), 1.0)
    peak_f1 = float(f1_rows.mean())

    # Sharpness: time-axis HF gradient energy, recon over target.
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


# 5. Utilization: codebook usage and collapse
def utilization(
    codes,
    cfg=None,
    *,
    codebook_size: Optional[int] = None,
    min_utilization: Optional[float] = None,
    min_code_entropy: Optional[float] = None,
) -> Dict[str, object]:
    """Codebook utilization and collapse detector for a set of FSQ codes.

    The codec can collapse to a single FSQ code (the shift-consistency loss has a
    trivial minimum at a constant encoder). This flags it from the evaluation codes,
    independently of the training regulariser.

    Signals:

    - ``min_dim_entropy``: minimum over FSQ dimensions of the normalised entropy
      ``H(dim) / log(levels[dim])``, in [0, 1]. The main collapse signal; a
      collapsed dimension drives it to 0 whatever the evaluation size.
    - ``frac_of_observable``: distinct codes over the most that could be seen,
      ``min(codebook_size, n_observed_tokens)``.
    - ``frac_codes_used``: distinct codes over ``codebook_size``. Informational
      only; do not threshold it. With a large codebook and few tokens its maximum is
      ``n_observed / codebook_size`` (e.g. 576 / 32768 = 0.018). Even with enough
      tokens the ceiling is the coupon-collector bound of ``K(1 - (1 - 1/K)^M)``
      codes, about 900 of 1000 at the default M = 2304, so 0.90 at K = 1000 means
      uniform use.
    - ``code_entropy_nats`` / ``effective_codes``: the joint code entropy and its
      perplexity ``e^H``. A rank-1 encoder maxes every per-dim (marginal) entropy
      while using few joint codes; this catches it. ``dim_redundancy_nats`` =
      ``sum(dim_entropy_nats) - code_entropy_nats``, 0 for independent dimensions.

    A codec is ``collapsed`` if ``min_dim_entropy < min_code_entropy`` or
    ``frac_of_observable < min_utilization``.

    Parameters
    ----------
    codes : int array of shape (B, n_tok, fsq_dim) or (B, n_frames, n_tok, fsq_dim)
        FSQ level indices, ``[..., i]`` in ``[0, levels[i])``. Frames are folded
        into the sample axis.
    cfg : SpectroCodecConfig, optional
        Supplies ``codebook_size``, ``fsq_levels`` and the ``gate_min_utilization``
        / ``gate_min_code_entropy`` thresholds.
    codebook_size : int, optional
        Total codebook size (product of the levels). Overrides ``cfg``.
    min_utilization : float, optional
        Threshold on ``frac_of_observable``. Overrides ``cfg``.
    min_code_entropy : float, optional
        Threshold on ``min_dim_entropy``. Overrides ``cfg``.

    Returns
    -------
    dict
        The signals above plus ``collapsed``, ``n_distinct_codes``,
        ``n_observed_tokens``, ``codebook_size`` and ``dim_entropy_nats``.
    """
    arr = _as_int_codes(codes)
    if arr.ndim == 4:
        # Fold frames into the sample axis.
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

    # Explicit arguments win over cfg.
    if codebook_size is None:
        if cfg is None:
            raise ValueError("utilization: provide cfg or codebook_size")
        codebook_size = int(cfg.codebook_size)
    if min_utilization is None:
        min_utilization = float(getattr(cfg, "gate_min_utilization", 0.02))
    if min_code_entropy is None:
        min_code_entropy = float(getattr(cfg, "gate_min_code_entropy", 0.3))

    # Per-dim level counts for the entropy normalisation; without cfg.fsq_levels
    # use the observed max + 1.
    if cfg is not None and getattr(cfg, "fsq_levels", None) is not None:
        levels = [int(v) for v in cfg.fsq_levels]
        if len(levels) != fsq_dim:
            raise ValueError(
                f"utilization: cfg.fsq_levels has {len(levels)} dims but codes have {fsq_dim}"
            )
    else:
        levels = [int(arr[..., i].max()) + 1 for i in range(fsq_dim)]

    # Distinct codes
    flat = arr.reshape(-1, fsq_dim)                       # (M, fsq_dim)
    n_distinct = int(np.unique(flat, axis=0).shape[0])
    frac_codes_used = float(n_distinct / codebook_size)  # informational only
    n_observed = int(flat.shape[0])
    # There cannot be more distinct codes than tokens, so also normalise by the
    # achievable maximum; this is the collapse signal for a large codebook.
    frac_of_observable = float(n_distinct / max(1, min(int(codebook_size), n_observed)))

    # Per-dim normalised entropy, minimum over dims
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

    # Joint code entropy. The per-dim entropies are marginal: a rank-1 encoder
    # (every FSQ dim a copy of one scalar) maxes them all while the joint
    # distribution stays small (one mhr codec: min_dim_entropy 0.92 but 14 effective
    # codes of 32768). exp(H_joint) is the size of a uniform codebook with the same
    # diversity; unlike n_distinct_codes it does not count a code used once like one
    # used often.
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
        "frac_codes_used": frac_codes_used,          # depends on the eval size
        "frac_of_observable": frac_of_observable,    # used for the collapse flag
        "min_dim_entropy": min_dim_entropy,
        "collapsed": collapsed,
        "n_distinct_codes": n_distinct,
        "n_observed_tokens": n_observed,
        "codebook_size": int(codebook_size),
        "code_entropy_nats": code_entropy_nats,      # joint code entropy, nats
        "effective_codes": float(np.exp(code_entropy_nats)),   # perplexity e^H_joint
        "dim_entropy_nats": dim_entropy_nats,        # per-dim marginal entropies, nats
        "dim_redundancy_nats": float(sum(dim_entropy_nats) - code_entropy_nats),
    }


# 6. Full-array slow-TS metrics
#
# Same convention as full_spectro_metrics: RMSE over the std of the same target
# group, so the group's own constant mean scores exactly 1.0.
#
# The group differs. In a slow-TS window (B, C, T) the profile position C is the
# structure axis, the counterpart of F rather than of the spectro channel, so the
# metric groups per window and reduces over (C, T). Grouping per (window, position)
# would reduce over T alone (5 samples) and make ``tmean`` the same as ``wcmean``.
#
# Masking: Thomson zero_is_missing zeros and CER/MSE beam-off NaNs are filled with
# 0 in input and target, and the median ts_core_density window is about 7% present,
# so an unmasked RMSE would mostly score the fill. With a mask, every statistic and
# every baseline mean below uses valid samples only.
_SLOWTS_STD_EPS = 1e-8
_SLOWTS_MIN_VALID = 2      # a window needs >= 2 valid samples to have a std at all


def _slowts_nrmse_corr(recon: np.ndarray, target: np.ndarray, mask: Optional[np.ndarray]):
    """Per-window nRMSE and correlation over the flattened (C, T) plane.

    All arrays are (B, C, T). Over each window's valid samples::

        nrmse = sqrt(mean((recon - target)^2)) / std(target)
        corr = Pearson(recon, target)

    Windows with a constant target (std <= 1e-8, a dead or absent profile) are left
    out and counted in ``valid_frac``. Returns ``(nrmse_mean, nrmse_median,
    nrmse_pooled, corr, valid_frac, present_frac)``; ``present_frac`` is the mean
    mask occupancy, without which a slow-TS number cannot be read.
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
    # Pooled nRMSE: squared error and squared deviation are summed over all windows
    # before dividing, so a nearly flat window adds its small error instead of a
    # huge ratio. Same 1.0 anchor; equals sqrt(1 - R^2) over the pooled samples.
    sse = (((r - t) ** 2) * m).sum(axis=1)[valid].sum()
    sst = ((tc * tc).sum(axis=1))[valid].sum()
    pooled = float(np.sqrt(sse / sst)) if sst > 0 else float("nan")
    return (float(nrmse.mean()), float(np.median(nrmse)), pooled, float(corr[valid].mean()),
            float(valid.mean()), present_frac)


def _slowts_std_ratio(recon: np.ndarray, target: np.ndarray, mask: Optional[np.ndarray]):
    """Amplitude diagnostic std(recon) / std(target) on (B, C, T) arrays; ideal 1.0.

    ``slowts_nrmse`` is a squared error, minimised by the conditional mean, so a
    decoder that shrinks toward the mean is rewarded. On one set of codecs the
    cer_ti reconstruction swung 0.2 to 0.7 against a target swinging 0 to 2.3, and
    mse was nearly flat against a +-3.5 oscillation, yet both scored nrmse about
    0.8. Correlation is scale-invariant, so neither number shows it; this ratio does.

    It is a diagnostic, not a ranking key: noise of the right amplitude also scores
    1.0. Well below 1 is dynamic-range compression; well above 1 is invented
    amplitude.

    Two groupings:

    - per window over the flattened (C, T) plane, as ``slowts_nrmse``, mixing
      profile spread with temporal spread;
    - per (window, channel) over time only, the temporal amplitude, which catches a
      flat trace with the right profile shape. It is variance-weighted across rows
      rather than a mean of ratios, because a row is only T = 5 samples at 100 Hz.

    Returns ``(ratio_mean, ratio_med, ratio_pooled, ratio_time)``, all NaN when no
    window qualifies. Rows need ``_SLOWTS_MIN_VALID`` valid samples and a
    non-constant target.
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

    # Per (window, channel) over time only
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
    """Full-array (B, C, T) slow-TS reconstruction fidelity, no axis averaged away.

    The slow-TS counterpart of :func:`full_spectro_metrics` (see the section note).
    ``slowts_decode_fidelity``'s ``envelope_corr`` deletes the time axis and so
    scores 1.0 for a predictor with no temporal structure; this keeps every sample.

    Parameters
    ----------
    recon, target : array of shape (B, C, T)
        Reconstruction and ground truth in the codec's standardised space.
    mask : array of shape (B, C, T), optional
        Validity (1 = real sample); every statistic uses valid samples only.
    prefix : str
        Prepended to every key (used for the trivial baselines).

    Returns
    -------
    dict
        Keys, each with ``prefix``:

        - ``slowts_nrmse``: RMSE / std(target) per window over (C, T), averaged.
          0 is perfect; exactly 1.0 is the window's own constant mean.
        - ``slowts_nrmse_med``: median of the same ratios. The mean has a heavy
          tail, since a window whose valid samples are nearly flat has a tiny
          divisor (mean 2.66 on ts_core_density while a typical window is near 1).
        - ``slowts_nrmse_pooled``: sqrt(sum SSE / sum SST) over windows, i.e.
          sqrt(1 - R^2) over the pooled valid samples, with the same exact 1.0
          anchor. The robust headline number; the per-window mean is kept to match
          the spectro convention.
        - ``slowts_corr``: Pearson correlation per window, averaged.
        - ``slowts_std_ratio``: std(recon) / std(target) per window over (C, T),
          averaged; ideal 1.0. Below 1 is compressed amplitude, which neither nRMSE
          nor correlation shows. Not a ranking key (see :func:`_slowts_std_ratio`).
        - ``slowts_std_ratio_med`` / ``slowts_std_ratio_pooled``: median and
          variance-weighted forms.
        - ``slowts_std_ratio_t``: the ratio per (window, channel) over time only,
          variance-weighted. A flat trace with the right profile scores about 0
          here; the ``tmean`` baseline scores exactly 0.
        - ``slowts_valid_frac``: fraction of windows with a non-constant target.
        - ``slowts_present_frac``: mean mask occupancy.
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
    """Reference points for :func:`full_slowts_metrics`, scored with the same mask.

    - ``base_self_*``: the target against itself; must read nrmse 0 / corr 1.
    - ``base_tmean_*``: each (window, position)'s time mean broadcast over T, a
      perfect profile with no temporal structure (what ``envelope_corr`` scores
      1.0). The bar a slow-TS reconstruction has to clear.
    - ``base_cmean_*``: the per-position mean over the batch and time, one profile
      for every window (the counterpart of spectro ``base_cfmean_*``).
    - ``base_wcmean_*``: the per-window scalar mean; nrmse exactly 1.0 (the
      normalisation anchor) and corr 0.
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


# 7. Full-array fast-TS metrics
#
# Same convention as full_spectro_metrics: RMSE over the std of the same target
# group, so its constant mean scores exactly 1.0. The group is (window, channel)
# and the reduction is over the sample axis:
#
#     nrmse[w, c] = RMSE(recon[w, c] - target[w, c]) / std(target[w, c])
#     result = mean over (w, c)
#
# A fast-TS window (B, C, W) holds W = 500 raw 10 kHz samples and has no structure
# axis, so ``tmean`` and ``wcmean`` are the same predictor here. Both must read
# exactly 1.0, which is a self-check.
#
# The per-(window, channel) std of the standardised filterscope signal is about
# 0.007, while the window means spread about 25 times more. This metric therefore
# ignores the light level and scores only the within-window waveform;
# ``fastts_global_nrmse`` is the view in which the level counts.
_FASTTS_STD_EPS = 1e-8


def _fastts_nrmse_corr(recon: np.ndarray, target: np.ndarray):
    """(nrmse, corr, valid_frac) per (window, channel) over the sample axis.

    Both arrays are (B, C, W). Pairs with a constant target (std <= 1e-8, a dead
    channel) are left out and counted in ``valid_frac``. A constant recon against a
    live target is counted: corr 0 and nrmse >= 1.
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
    """Full-array (B, C, W) fast-TS reconstruction fidelity on the raw samples.

    Nothing is pooled, rectified or time-averaged first; see the section note for
    the normalisation.

    Parameters
    ----------
    recon, target : array of shape (B, C, W)
        Raw windows in the codec's standardised space.
    prefix : str
        Prepended to every key (used for the trivial baselines).

    Returns
    -------
    dict
        Keys, each with ``prefix``:

        - ``fastts_nrmse``: RMSE / std(target) per (window, channel), averaged. 0 is
          perfect; exactly 1.0 is the pair's own constant mean.
        - ``fastts_corr``: Pearson correlation per (window, channel), averaged.
        - ``fastts_valid_frac``: fraction of pairs with a non-constant target.
        - ``fastts_global_nrmse``: RMSE over the batch divided by each channel's std
          over the batch, so the light level counts. The two nRMSEs can differ by a
          factor of about 25 on this signal, so quote both.
    """
    r = _to_numpy(recon).astype(np.float64)
    t = _to_numpy(target).astype(np.float64)
    if r.shape != t.shape:
        raise ValueError(f"full_fastts_metrics: shape mismatch {r.shape} vs {t.shape}")
    if r.ndim != 3:
        raise ValueError(f"full_fastts_metrics expects (B, C, W); got {r.shape}")
    nrmse, corr, valid = _fastts_nrmse_corr(r, t)
    # Level-aware view: per-channel std over the whole batch.
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
    """Reference points for :func:`full_fastts_metrics`, scored by it on one target.

    - ``base_self_*``: the target against itself; must read nrmse 0 / corr 1.
    - ``base_wcmean_*``: the per-(window, channel) mean; nrmse exactly 1.0 (the
      anchor) and corr 0.
    - ``base_tmean_*``: the time mean broadcast over the window, the baseline to
      beat. With no structure axis it is the same predictor as ``wcmean`` and must
      also read 1.0; both are kept so that identity is checked.
    - ``base_cmean_*``: the per-channel mean over the batch. Much worse than 1.0
      (about 25.8 on held-out shots), because the window-to-window light level
      dwarfs the within-window fluctuation.
    - ``base_binmean<P>_*``: the mean within each non-overlapping P-sample bin, a
      piecewise-constant predictor. P = 100 (10 ms, the envelope codec's bin) scored
      0.963, i.e. that resolution holds only about 7% of the within-window variance.
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


# Fast-TS envelope metrics: pooled, with each channel's global std
#
# ``fastts_nrmse`` divides each (window, channel) by its own std. For the ELM
# envelope that discards what dominates it: 96% of the envelope's variance is the
# between-window level. The envelope codec is judged instead with each channel's
# global std over the held-out pool, so a global constant scores 1.0 and the level
# counts. The two are different quantities and both are reported.
#
# ``env_base_ratematched`` is the bar: a trivial encoder that sends only each
# channel's window mean, quantised to the codec's own bit budget
# (n_tok * log2(codebook) bits, split over channels). It scored 0.19 against the
# codec's 0.53 on held-out shots. ``env_base_wmean`` is the same predictor
# unquantised, the floor of any level-only code.
#
# nRMSE is minimised by the conditional mean, so it is a floor, not a ranking key:
# ``env_ac_std_ratio`` (ideal 1.0) and ``env_burst_keep`` (fraction of an ELM
# burst's height kept) detect the amplitude collapse it rewards.


def fastts_envelope_metrics(recon, target, bits: float = 0.0,
                            prefix: str = "") -> Dict[str, float]:
    """Pooled envelope metrics with per-channel global std, on (B, C, E) arrays.

    Parameters
    ----------
    recon, target : array of shape (B, C, E)
        The reconstruction and the ELM activity envelope it targets.
    bits : float
        The codec's token budget in bits (``n_tok * log2(codebook_size)``), used
        only for ``env_base_ratematched``; 0 skips that row.
    prefix : str
        Prepended to every key.

    Returns
    -------
    dict
        Each averaged over the C channels:

        - ``env_nrmse``: RMS(recon - gt) / std(gt), pooled over (window, bin) per
          channel; a global constant scores 1.0.
        - ``env_std_ratio``: std(recon) / std(gt) pooled over (window, bin).
          Dominated by the between-window level: a 0.4x-shrunk reference reads 0.99,
          so it is not the burst-amplitude check.
        - ``env_ac_std_ratio``: the same ratio on the within-window (mean-removed)
          part; the amplitude check here (0.40 on that reference).
        - ``env_level_corr``: Pearson r of the per-(window, channel) level.
        - ``env_burst_keep``: on the 1% most active windows, the recon's mean peak
          above the median over the target's.
        - ``env_base_gmean``: global-constant anchor, 1.0 by construction.
        - ``env_base_wmean``: exact per-window mean, the level-only floor.
        - ``env_base_ratematched`` and ``env_rate_levels``: that predictor quantised
          to ``bits`` (the bar), and its number of levels.
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
        # Amplitude ratio on the within-window part. The pooled ratio above is
        # dominated by the between-window level and barely sees a within-window
        # collapse, so judge burst amplitude on this one.
        acratio.append(ac_r[:, c].std() / (ac_t[:, c].std() + 1e-12))
        lt, lr = lev_t[:, c], lev_r[:, c]
        corrs.append(0.0 if lt.std() < 1e-12 or lr.std() < 1e-12
                     else float(np.corrcoef(lt, lr)[0, 1]))
        # Burst height kept. The quiet 99% of windows dominate nRMSE, which
        # therefore barely notices flattened bursts.
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


# Mode-track metrics
#
# spec_nrmse ranks a blur best: on held-out mhr windows the codec with the best
# spec_nrmse (0.90) had an HF ratio of 0.014 and rendered a featureless smear, while
# one at 1.10 visibly kept the rising 50-65 kHz mode track. An element-wise error is
# minimised by the conditional mean, so it cannot rank "recognisable".
#
# A mode track is a narrow ridge in frequency that persists across time. The
# metrics below look at the frequency profile of each time frame and never average
# time away, unlike ``peak_f1`` and ``envelope_corr``, which run on
# ``spec.mean(axis=-1)`` and give a static smear a perfect score.
#
# None of these enters ``spike.gate_score``, so checkpoint selection is unchanged.
def _as_nft(x) -> np.ndarray:
    """(B, C, F, T) -> (B*C, F, T) float32.

    float32 is enough for the ratios, correlations and F1 scores built on this, and
    float64 made the audit about 4x slower.
    """
    a = _to_numpy(x).astype(np.float32)
    if a.ndim != 4:
        raise ValueError(f"expected (B, C, F, T); got {a.shape}")
    return a.reshape(a.shape[0] * a.shape[1], a.shape[2], a.shape[3])


# Floor on a frame's frequency-profile std below which it has no peaks. The
# prominence test is relative to the frame's own std, so rounding noise would
# otherwise yield peaks: the detrended time-mean envelope is zero analytically but
# about 1.8e-7 in float32, and peak-picking that noise gave it 0.17 instead of 0.
# Real z-scored log-power frames have std 0.5-1.5, so the floor never bites there.
_PEAK_MIN_SD: float = 1e-3


def _peaks_per_frame(X: np.ndarray, prom: float) -> np.ndarray:
    """Bool mask of frequency-local peaks of an (N, F, T) array, per time frame.

    A bin is a peak when it is a local maximum along frequency and rises at least
    ``prom`` times the frame's frequency-profile std above the frame's median. The
    relative threshold needs no tuning per modality or normalisation; the
    ``_PEAK_MIN_SD`` floor keeps noise-only frames from qualifying.
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
    """Per-frame spectral-peak F1: are the same modes present at the same times?

    The frequency profile of each (window, channel, time frame) is peak-picked in
    both arrays (:func:`_peaks_per_frame`). A target peak counts as recovered when
    the recon has a peak within ``tol`` bins. F1 is pooled over all (frame, peak)
    pairs, so reproducing the average spectrum without its time evolution scores
    low, unlike ``peak_f1``.

    With ``detrend`` each (window, channel)'s time-mean profile is subtracted from
    every frame first. Without it the time-mean envelope scores higher than any
    trained codec (0.50 on mhr), because mhr peaks are mostly static within a 50 ms
    window. After detrending the envelope is zero and scores 0, and what remains is
    peak structure that moves with time.

    Returns ``{prefix}mode_track_f1``, ``_precision``, ``_recall``, ``_median_df``
    (median |delta f| in bins over matched target peaks) and ``_n_gt_peaks``
    (target peaks per frame; the F1 means nothing when this is near 0).
    """
    r, t = _as_nft(recon), _as_nft(target)
    if band_bins is not None:
        k = int(min(int(band_bins), t.shape[1]))
        r, t = r[:, :k, :], t[:, :k, :]
    if detrend:
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
    # Median |delta f| of matched target peaks: the smallest shift to a recon peak.
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
    """Correlation of the dominant-frequency trajectory over time.

    For each (window, channel) the argmax frequency in the mode band is taken per
    frame, giving ``f*(t)``; the result is the Pearson correlation of target and
    recon trajectories, averaged over pairs. A blur or static envelope gives a
    constant or noisy trajectory and scores about 0. Pairs with a constant
    trajectory score 0.
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
    """Per-frame frequency contrast, recon over target; ideal 1.0.

    Contrast of one frequency profile is ``max - median``. A blur has a flat profile
    and drives it toward 0; above 1 means the recon is more peaked than the target.

    Patch lattices inflate it far less than ``sharpness``, but they do inflate it.
    Adding a synthetic checkerboard to a blurred recon of two moving-ridge targets
    (``tests/ignite/test_gate.py``) multiplies the metrics by::

        artifact added        sharpness (hf_ratio)   this metric      mode_track_f1
        frequency-periodic       x12.5 - x19.2       x2.2 - x3.3       x0.51 - x0.53
        time-periodic            x12.0 - x18.7       x1.00 (exact)     x1.00 (exact)

    A frequency-periodic lattice is a set of narrow frequency peaks, which this
    metric rewards, and the inflation grows as the target's own contrast falls.
    ``mode_track_f1`` and ``ms_ssim`` get worse under a lattice, so rank on those
    and read this one beside them.
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
    """Box mean of an (N, H, W) array over a k x k window ('valid'), by integral image.

    float32 on purpose: SSIM needs no more, and float64 here was the slowest part of
    the metric suite.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    cs = np.cumsum(np.cumsum(
        np.pad(x, ((0, 0), (1, 0), (1, 0)), mode="constant"), axis=1, dtype=np.float32),
        axis=2, dtype=np.float32)
    return ((cs[:, k:, k:] - cs[:, :-k, k:] - cs[:, k:, :-k] + cs[:, :-k, :-k])
            / np.float32(k * k))


def ms_ssim(recon, target, win: int = 7, scales: Sequence[int] = (1, 2, 4),
            prefix: str = "") -> Dict[str, float]:
    """Multi-scale SSIM of the spectrogram, a standard cross-check against blur.

    SSIM with a ``win x win`` box window at each average-pooled scale, averaged over
    scales; the constants use each (window, channel) target's own range. Being a
    standard image metric, it is an independent check on the mode-track metrics.
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
        # Opt-in, as it costs a second peak-picking pass: the F1 after removing each
        # pair's time-mean profile, so the static envelope scores 0.
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


# 8. Fast-TS 1-D SSIM: the amplitude check nRMSE cannot make
#
# nRMSE is minimised by the conditional mean, so under uncertainty a codec trained
# or ranked on it shrinks toward the local mean: for correlation r the
# nRMSE-optimal scale is r, so at r = 0.4 the output is 60% compressed while nRMSE
# reads 0.917. For a spiky 10 kHz filterscope trace this variance collapse is the
# likely outcome of a squared-error sweep, and MSE has no variance term to show it.
#
# SSIM factors as luminance x contrast x structure:
#
#     luminance = (2 mu_r mu_t + C1) / (mu_r^2 + mu_t^2 + C1)
#     contrast  = (2 sig_r sig_t + C2) / (sig_r^2 + sig_t^2 + C2)
#     structure = (sig_rt + C2/2) / (sig_r sig_t + C2/2)
#
# The contrast factor falls with the recon's local std: the right shape at half
# amplitude scores 0.8, at 0.4x 0.69, and a flat line about 0. (The NVIDIA spectral
# codec paper, arXiv 2406.05298, also judges reconstruction with an SSIM-family
# measure rather than an element-wise error.)
#
# nRMSE stays a floor (beat the 1.0 constant-mean anchor); these structural numbers
# rank codecs that clear it. ``FastTSCodec._ssim_loss`` uses the same idea as a
# loss, which targets spikes through local variance rather than per-sample loss
# weighting (a known dead end for this modality).
#
# Choices:
# - 1-D windows along the sample axis, per (window, channel).
# - Box window of 17 samples = 1.7 ms at 10 kHz. ELM bursts recur about every
#   1.6 ms, so a window spans about one burst period. A box window is the original
#   SSIM paper's choice and is exact via cumsum.
# - Valid positions only (no padding), so no edge effects.
# - C1 and C2 use each (window, channel) target's own std as the range L, so the
#   metric ignores the per-channel standardisation and each window's scale, like
#   full_fastts_metrics.
# - Both full SSIM and the luminance-free CS form are reported. For a signed signal
#   the luminance term misbehaves (2 mu_r mu_t < 0 when the means straddle zero),
#   which is why MS-SSIM uses CS at all but the coarsest scale; nRMSE already
#   covers the DC level.
_FASTTS_SSIM_WIN: int = 17
_FASTTS_SSIM_SCALES = (1, 2, 4)


def _box1d(x: np.ndarray, win: int) -> np.ndarray:
    """Mean over each length-``win`` window inside the last axis, via cumsum.

    ``(..., L) -> (..., L - win + 1)``; no padding.
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
    """Per-(row, position) SSIM factor maps of (N, L) arrays.

    Returns (luminance, contrast, structure). ``C1 = (0.01 L)^2`` and
    ``C2 = (0.03 L)^2`` with ``L`` each row's target std. A constant target row gets
    a clamped L, which sends its contrast to about 0 for any non-constant recon.
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
    """1-D structural similarity of the raw fast-TS waveform (see the section note).

    Parameters
    ----------
    recon, target : array of shape (B, C, W)
        Raw windows in the codec's standardised space.
    win : int
        Box-window length in samples (default 17 = 1.7 ms at 10 kHz, about one ELM
        period).
    scales : sequence of int
        Average-pool factors for the multi-scale CS average.
    prefix : str
        Prepended to every key.

    Returns
    -------
    dict
        Keys, each with ``prefix``:

        - ``fastts_ssim``: full SSIM; 1.0 is perfect.
        - ``fastts_ssim_cs``: contrast x structure, without luminance. The ranking
          number, since nRMSE already covers the DC level.
        - ``fastts_ssim_contrast``: the contrast factor alone, the variance-collapse
          detector (0.80 at half amplitude, 0.69 at 0.4x, about 0 for a flat line).
        - ``fastts_ssim_structure``: the structure factor alone (shape and spike
          timing, amplitude divided out).
        - ``fastts_ms_ssim``: CS averaged over ``scales``.
        - ``fastts_std_ratio``: mean std(recon) / std(target) over (window,
          channel); 1.0 is the right range, below 1 compressed, above 1 over-driven.
    """
    r = _to_numpy(recon).astype(np.float64)
    t = _to_numpy(target).astype(np.float64)
    if r.shape != t.shape:
        raise ValueError(f"fastts_ssim_metrics: shape mismatch {r.shape} vs {t.shape}")
    if r.ndim != 3:
        raise ValueError(f"fastts_ssim_metrics expects (B, C, W); got {r.shape}")
    rf = r.reshape(-1, r.shape[-1])
    tf = t.reshape(-1, t.shape[-1])
    live = tf.std(axis=-1) > _FASTTS_STD_EPS      # skip dead channels, as nRMSE does
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
    """Reference cases for :func:`fastts_ssim_metrics`, built from the target alone.

    Check them before ranking on the metric; one that orders them wrongly should
    not be used.

    - ``ref_self_*``: the target against itself; 1.0 on every factor.
    - ``ref_noise_*``: the target plus 0.05 sigma white noise, a near-perfect recon;
      must score well.
    - ``ref_smooth_*``: the target box-smoothed over 9 samples (0.9 ms), the blur
      failure; must score badly on contrast, although smoothing lowers nRMSE.
    - ``ref_shrunk_*``: 0.4 x the target around its mean, the right shape at 40%
      amplitude (what nRMSE-optimal shrinkage gives at r = 0.4). Structure stays
      near 1 while contrast drops to about 0.69.
    - ``ref_flat_*``: the per-(window, channel) mean, the nRMSE 1.0 anchor; about 0
      on contrast and structure.

    Each set also carries ``ref_<name>_fastts_nrmse``.
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


# 9. Full-array video metrics
#
# Same convention as full_spectro_metrics: RMSE over the std of the same group, so
# the group's constant mean scores exactly 1.0.
#
# The group is (window, channel) over the flattened (T, H, W) volume. Each tangtv
# channel is a separate camera with its own filter and brightness, so pooling a
# divertor's two cameras would let a bright one hide a dead one.
#
# Masking: tangtv_lower has no live camera in 51% of shots and tangtv_upper in 69%,
# and in the shots that have one a further 17-19% of channel slots are zero. An
# unmasked metric would mostly score how well the codec reproduces zero. With a
# mask, every statistic and every baseline mean uses valid (channel, frame) entries
# only, as for slow-TS.
#
# nRMSE is a floor, not a ranking key: it rewards blur and amplitude collapse.
# ``video_std_ratio`` (ideal 1.0) shows that, and ``video_std_ratio_t`` catches a
# right still image frozen in time.
_VIDEO_STD_EPS = 1e-8
_VIDEO_MIN_VALID = 2


def _video_mask_bct(mask, shape) -> Optional[np.ndarray]:
    """Normalise a video validity mask to (B, C, T); None passes through.

    Accepts the per-frame (B, T) mask, broadcast across channels ("the whole frame
    is valid"), and the per-(channel, frame) (B, C, T) mask that
    ``cfg.mask_missing`` produces.
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
    """Per-(window, channel) nRMSE and correlation over the (T, H, W) volume.

    Same arithmetic as :func:`_slowts_nrmse_corr`, with the same six return values.
    Groups with a constant target over their valid entries (a dead camera) are left
    out and counted in ``valid_frac``.
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
    """Amplitude diagnostic std(recon) / std(target), ideal 1.0; not a ranking key.

    Returns ``(mean, median, pooled, temporal)``. The first three group per
    (window, channel) over (T, H, W), as the nRMSE does, mixing spatial contrast
    with temporal swing. ``temporal`` groups per (window, channel, pixel) over time
    only, variance-weighted: a right still image that is frozen reads about 0 there
    and about 1 on the others. The ``tmean`` baseline scores exactly 0 on it.
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

    # Per (window, channel, pixel) over time only
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
    """Sharpness ratio: recon HF gradient energy over the target's; ideal 1.0.

    The masked version of ``video_decode_fidelity``'s ``sharpness``. Spatial
    differences use valid frames only and temporal differences only pairs of
    consecutive valid frames, so the step into a zero-filled dead frame is not
    counted.

    Read it beside ``patch_lattice_ratio``: a high HF ratio can be checkerboard
    (92% of one mhr codec's HF energy was the lattice). Low means smooth; high means
    real texture or artifact, and only the lattice ratio tells them apart.
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
    """Full-array (B, C, T, H, W) video reconstruction fidelity, no axis averaged away.

    The video counterpart of :func:`full_slowts_metrics`. ``video_decode_fidelity``'s
    ``envelope_corr`` averages the 50 ms time axis away and scores 1.0 for a
    predictor with no temporal structure; this keeps every pixel of every frame.

    Parameters
    ----------
    recon, target : array of shape (B, C, T, H, W)
        Reconstruction and ground truth in the codec's standardised frame space.
    mask : array of shape (B, C, T) or (B, T), optional
        Validity per (channel, frame), 1 = camera recording; statistics use valid
        entries only.
    prefix : str
        Prepended to every key (used by the trivial baselines).

    Returns
    -------
    dict
        Keys, each with ``prefix``:

        - ``video_nrmse`` / ``_med`` / ``_pooled``: RMSE / std(target) per (window,
          channel) over (T, H, W); mean, median and variance-weighted. 0 is perfect,
          exactly 1.0 is that camera's own constant mean.
        - ``video_corr``: Pearson correlation over the same entries.
        - ``video_std_ratio`` / ``_med`` / ``_pooled`` / ``_t``: amplitude, ideal
          1.0; ``_t`` is the temporal ratio (a frozen video reads about 0).
        - ``video_hf_ratio``: masked sharpness ratio, ideal 1.0; read it with the
          lattice ratio.
        - ``video_valid_frac``: fraction of groups with a non-constant target.
        - ``video_present_frac``: mean mask occupancy.
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
    """Reference points for :func:`full_video_metrics`, scored with the same mask.

    - ``base_self_*``: the target against itself; must read nrmse 0, corr 1 and
      std_ratio 1.
    - ``base_tmean_*``: each pixel's time-mean frame broadcast over T, a right still
      image with no temporal structure (what ``envelope_corr`` scores 1.0, and what
      a frozen recon is). The bar a video reconstruction has to clear.
    - ``base_cmean_*``: the per-(channel, pixel) mean over the batch, one static
      image per camera.
    - ``base_wcmean_*``: the per-(window, channel) mean; nrmse exactly 1.0 (the
      anchor) and corr 0.
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
    """Patch-lattice strength of a (B, C, T, H, W) video batch.

    The video decoder tiles ``patch_h x patch_w`` patches through a shared
    ``to_pixels`` linear, as the spectro decoder does over (freq, time), so the same
    detector applies: each valid frame of each channel goes to
    :func:`patch_lattice_metrics` as an (N, 1, H, W) batch with
    ``(patch_f, patch_t) = (patch_h, patch_w)``.

    1.0 means no lattice; tangtv_lower ground truth reads 1.10 and the 64k-code
    video codec's reconstruction 25. Read it beside ``video_hf_ratio``: blurring
    lowers the lattice for free, so a real fix lowers it while holding the HF ratio.

    ``mask`` (B, C, T) drops dead-camera frames, whose flat plates would pull the
    result toward 1.0.
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
    """Codebook utilization as a bit rate rather than a distinct-code count.

    ``frac_codes_used`` counts a code used once like one used a million times, and
    its ceiling is a coupon-collector bound. This asks how many of the
    ``n_tok * log2(K)`` bits a frame pays for are delivered.

    Parameters
    ----------
    codes : int array of shape (B, n_tok, fsq_dim) or (B, n_frames, n_tok, fsq_dim)
        FSQ codes.
    codebook_size : int
        K = prod(fsq_levels).
    n_tok : int, optional
        Tokens per frame; inferred from ``codes`` when omitted.

    Returns
    -------
    dict
        - ``bits_available_per_frame``: n_tok * log2(K).
        - ``bits_pooled_per_token``: entropy of the code over all positions pooled,
          in bits.
        - ``bits_positional_per_token``: mean over positions of each position's own
          code entropy, in bits. The stricter rate: a token that always emits the
          same code delivers 0 bits however diverse the pooled histogram is. Never
          above the pooled rate.
        - ``rate_pooled`` / ``rate_positional``: those over log2(K), in [0, 1]; the
          ">= 90% utilization" number, read against the ceilings.
        - ``rate_positional_ceiling`` / ``rate_pooled_ceiling``: the largest value
          the estimate can take here, log2(n_samples) / log2(K). An entropy from M
          samples cannot exceed log2(M): at 320 windows and K = 1000 the positional
          ceiling is 0.835. The pooled rate uses n_windows * n_tok samples and is
          rarely limited.
        - ``bits_delivered_per_frame_pooled`` / ``_positional``, ``n_windows``,
          ``n_distinct_codes``, ``frac_codes_used`` and ``n_observed_tokens``.
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
    # Eval-size ceilings (see the docstring), so a low rate is not mistaken for a
    # shortfall of the codec.
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
