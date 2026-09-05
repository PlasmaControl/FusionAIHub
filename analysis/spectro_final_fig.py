"""FINAL spectrogram codec deliverable: arm table + the one figure, per modality.

Spectro sibling of ``analysis/video_final_fig.py`` (tangtv) and
``analysis/fastts_env_final_fig.py`` (filterscopes). EVALUATION + RENDERING ONLY -- every
checkpoint is opened read-only and nothing is trained here.

Two modes, both on HELD-OUT shots (the trainer's eval split is ``all_shots[-eval_n_shots:]``,
so no arm has seen them):

  --mode score    The arm table: nRMSE with its trivial baselines (self 0.0, wcmean exactly
                  1.0, tmean, cfmean), corr2d, patch_lattice_ratio WITH its ground-truth
                  control, hf_ratio, std_ratio, codebook utilization AS A BIT RATE, and the
                  forecastability margins with n_transition.

  --mode figure   The deliverable figure: GT vs reconstruction across a whole shot over the
                  FULL 0-250 kHz band, missing data left as GAPS (never filled), % data
                  coverage per panel, and per-panel metrics so each panel stands alone.
                  ``--band_khz lo,hi`` zooms the frequency axis onto the coherent-mode band
                  (see --mode structure): at the full 0-250 kHz a mode line 1-2 bins wide is
                  ONE PIXEL, so the full-band panel cannot show whether tracks were resolved
                  and a codec is judged "smooth" by rendering artefact.

  --mode structure  WHERE THE MODE STRUCTURE IS, and how much of it each arm reproduces.
                  Added 2026-09-04 because the arm table cannot answer the question the
                  codecs are FOR. These feed a world model that predicts how modes EVOLVE, so
                  ``spec_nrmse`` is a floor (its exact minimiser is the blur) and ~tmean --
                  the time-AVERAGED spectrum -- is not a target at all: it has zero temporal
                  structure by construction, so an arm converging on it has thrown away
                  exactly the signal the world model needs. Reports, per frequency band:

                    ac1        the GT's lag-1 autocorrelation along the STFT-frame axis.
                               ~0 = the band is realization speckle no codec can reproduce;
                               high = a temporally coherent (predictable) band, i.e. modes.
                    coh_frac   var(GT smoothed over `smooth` frames) / var(GT). The share of
                               the band's variance that is coherent rather than speckle.
                    hf_gt/arm  HF gradient energy of GT and of each arm, so "sharpness" is
                               read where the modes actually are instead of pooled over a
                               band that is 80% noise.

                  Plus two ORACLES that bound the token grid itself:
                    patchmean  the exact per-(channel, patch_f x patch_t) mean. This is the
                               most a codec can say if a token carries only its patch's
                               level. If mode tracks are already invisible here, the blocker
                               is the PATCH GRID (and therefore FRAME_LAYOUT), not the codec.
                    tsmooth    GT smoothed along time. The ceiling for any predictor that
                               cannot reproduce speckle.

MASKING. Every statistic -- mean, std, RMSE, correlation, and each baseline's OWN mean -- is
taken over valid samples only, via ``data.spectro_frame_mask`` and the ``mask=`` argument of
``gate.decode_fidelity``. This is not cosmetic: measured over all 8753 shots
(foundation_model_meta/spectro_channel_liveness.pt) the fraction of the streamed tensor that
is REAL diagnostic data is bes 0.376, mirnov 0.356, co2 0.547, ece 0.872, mhr 0.874.

nRMSE IS A FLOOR, NOT A RANKING KEY -- its exact minimiser is the conditional mean, so it
rewards blur and amplitude collapse. ``std_ratio`` (ideal EXACTLY 1.0) is printed beside it,
``patch_lattice_ratio`` is always read as a PAIR with ``hf_ratio`` and against the GT control,
and utilization is a HARD GATE.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "16")))

from tokamak_foundation_model.ignite import gate                      # noqa: E402
from tokamak_foundation_model.ignite import train_codec as tc         # noqa: E402
from tokamak_foundation_model.ignite.codec import SpectroCodec        # noqa: E402
from tokamak_foundation_model.ignite.config import STFT_FS            # noqa: E402

DATA = tc.DEFAULT_DATA_DIR
CACHE = Path("/lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/eval_runs/"
             "codec_recon_figs/_cache")


# ------------------------------------------------------------------------------------ #
# data / model
# ------------------------------------------------------------------------------------ #
def load_codec(ckpt_path: str, device="cpu"):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    codec = SpectroCodec(cfg).to(device)
    codec.load_state_dict(ck["codec"])
    codec.eval()
    return codec, cfg, ck


def held_out_shots(modality: str, eval_n_shots: int = 16, live_only: bool = True) -> List[str]:
    """The trainer's eval split, after the SAME presence filter the arms trained under.

    Without the filter the held-out pool for bes/co2 is dominated by shots whose HDF5 group
    is an empty stub, whose only possible window is the eps-floor constant plate -- scoring
    against that measures nothing about the codec.
    """
    shots = tc.spike.discover_shots(DATA)
    if live_only:
        shots = tc.spectro_live_shots(modality, shots, log_fn=print)
    return [str(s) for s in shots[-eval_n_shots:]]


def window_pool(modality: str, cfg, shots: List[str], n_windows: int, per_shot: int = 0):
    """``(N, C, F, T)`` held-out windows + their ``(N, C, T)`` validity masks."""
    X, M = [], []
    for sh in shots:
        if len(X) >= n_windows:
            break
        try:
            ds = tc.CodecPairDataset(modality, [sh], cfg, data_dir=DATA,
                                     lengths_cache_path=None, emit_mask=True)
        except ValueError:
            continue
        # The AUDIT POOL must not be activity-stratified either: cfg.min_activity /
        # cfg.active_bias are baked into the checkpoint's cfg for co2 and mirnov, so without
        # this the >=300-window table would be scored on a sample deliberately biased toward
        # active windows. Zeroed on the instance so the codec's own cfg is untouched.
        ds.min_activity, ds.active_bias = 0.0, 0.0
        n = len(ds)
        if n == 0:
            continue
        k = per_shot or max(1, math.ceil(n_windows / max(1, len(shots))))
        for i in np.linspace(0, n - 1, min(k, n)).astype(int):
            a, _b, m = ds[int(i)]
            X.append(a.numpy())
            M.append(m.numpy())
            if len(X) >= n_windows:
                break
    if not X:
        raise RuntimeError(f"no held-out windows for {modality}")
    return np.stack(X, 0), np.stack(M, 0)


def seq_pool(modality: str, cfg, shots: List[str], seq_len: int = 8, per_shot: int = 2):
    """``(B, seq_len, C, F, T)`` blocks of CONSECUTIVE windows, for forecastability."""
    blocks = []
    for sh in shots:
        try:
            ds = tc.CodecPairDataset(modality, [sh], cfg, data_dir=DATA,
                                     lengths_cache_path=None, emit_mask=True)
        except ValueError:
            continue
        # The AUDIT POOL must not be activity-stratified either: cfg.min_activity /
        # cfg.active_bias are baked into the checkpoint's cfg for co2 and mirnov, so without
        # this the >=300-window table would be scored on a sample deliberately biased toward
        # active windows. Zeroed on the instance so the codec's own cfg is untouched.
        ds.min_activity, ds.active_bias = 0.0, 0.0
        n = len(ds)
        if n < seq_len:
            continue
        got = 0
        for s0 in np.unique(np.linspace(0, n - seq_len, per_shot * 3).astype(int)):
            if got >= per_shot:
                break
            blk, ok = [], True
            for k in range(seq_len):
                a, _b, m = ds[int(s0) + k]
                if float(m.min()) <= 0.0:          # only fully-real blocks
                    ok = False
                    break
                blk.append(a.numpy())
            if ok:
                blocks.append(np.stack(blk, 0))
                got += 1
    if not blocks:
        return None
    return np.stack(blocks, 0)


@torch.no_grad()
def reconstruct(codec, X: np.ndarray, batch: int = 8, device="cpu"):
    recon, codes = [], []
    for i in range(0, X.shape[0], batch):
        xb = torch.from_numpy(X[i:i + batch]).to(device, torch.float32)
        out = codec.forward(xb)
        recon.append(out["recon"].cpu().numpy())
        codes.append(out["codes"].cpu())
    return np.concatenate(recon, 0), torch.cat(codes, 0)


# ------------------------------------------------------------------------------------ #
# metrics the gate does not carry for spectro
# ------------------------------------------------------------------------------------ #
def masked_std_ratio(recon: np.ndarray, target: np.ndarray, mask: Optional[np.ndarray]):
    """mean over (window, channel) of std(recon)/std(target), over VALID samples only.

    Reported BESIDE nRMSE because nRMSE's exact minimiser is the conditional mean: a codec
    that shrinks amplitude toward the mean improves nRMSE while destroying the signal. Ideal
    is EXACTLY 1.0; < 1 is amplitude collapse, > 1 is added noise.
    """
    B, C, F, T = recon.shape
    r = recon.reshape(B * C, -1).astype(np.float64)
    t = target.reshape(B * C, -1).astype(np.float64)
    if mask is None:
        rs, ts = r.std(axis=1), t.std(axis=1)
    else:
        w = np.broadcast_to(np.asarray(mask)[:, :, None, :] > 0.5,
                            (B, C, F, T)).reshape(B * C, -1).astype(np.float64)
        n = np.maximum(w.sum(axis=1), 1.0)
        rm = (r * w).sum(axis=1) / n
        tm = (t * w).sum(axis=1) / n
        rs = np.sqrt((((r - rm[:, None]) ** 2) * w).sum(axis=1) / n)
        ts = np.sqrt((((t - tm[:, None]) ** 2) * w).sum(axis=1) / n)
    ok = ts > 1e-8
    return float((rs[ok] / ts[ok]).mean()) if ok.any() else float("nan")


def _boxcar_last(x: np.ndarray, k: int, mode: str = "valid") -> np.ndarray:
    """Vectorized k-tap boxcar moving average along the LAST axis, via a cumulative sum.

    ``np.apply_along_axis(np.convolve, ...)`` here means ~650k independent 96-sample
    convolutions in Python for one modality's pool, which takes minutes; the cumsum form is
    one pass. ``mode="valid"`` returns ``T - k + 1`` samples (used for the variance ratio);
    ``mode="same"`` edge-pads first and returns ``T`` (used by the smoothing oracle, so it
    stays shape-compatible with the target).
    """
    if k <= 1:
        return x
    if mode == "same":
        pad = k // 2
        x = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(pad, k - 1 - pad)], mode="edge")
    c = np.cumsum(x, axis=-1, dtype=np.float64)
    head = c[..., k - 1:]
    tail = np.concatenate([np.zeros(c.shape[:-1] + (1,)), c[..., :-k]], axis=-1)
    return (head - tail) / float(k)


def bin_to_khz(cfg, b: int) -> float:
    """STFT bin index -> kHz. Bin width is STFT_FS / stft_n_fft (488 Hz at 500 kHz / 1024)."""
    return float(b) * (STFT_FS / float(getattr(cfg, "stft_n_fft", 1024))) / 1e3


def khz_to_bin(cfg, khz: float) -> int:
    return int(round(khz * 1e3 / (STFT_FS / float(getattr(cfg, "stft_n_fft", 1024)))))


def _masked_bct(mask: Optional[np.ndarray], shape):
    """(B, C, T) bool from a (B,C,T) mask, or None when everything is valid."""
    if mask is None:
        return None
    m = np.asarray(mask) > 0.5
    return None if m.all() else m


def band_structure(X: np.ndarray, M: Optional[np.ndarray], n_bands: int = 16,
                   smooth: int = 5) -> List[Dict]:
    """Per-frequency-band temporal-coherence profile of the GROUND TRUTH.

    ``X`` is (B, C, F, T). For each band of ``F // n_bands`` bins, over VALID (b, c, t) only:

      ac1       lag-1 autocorrelation along T of the per-(b, c, f) mean-removed series,
                pooled over (b, c, f) as sum(x_t x_{t+1}) / sum(x_t^2). This is the same
                estimator the spectrogram trend/noise audit used; it separates a coherent
                band (ac1 -> 1) from realization speckle (ac1 -> 0).
      coh_frac  var of the ``smooth``-frame moving average / var of the band. The share of
                the band's variance that survives low-passing in TIME, i.e. the share a
                codec could even in principle carry.
      std       the band's own std (so a band with ac1 ~ 1 but no amplitude is not mistaken
                for the signal).
    """
    B, C, F, T = X.shape
    step = max(1, F // n_bands)
    valid = _masked_bct(M, X.shape)
    out: List[Dict] = []
    k = int(smooth)
    for b0 in range(0, F, step):
        b1 = min(F, b0 + step)
        x = X[:, :, b0:b1, :].astype(np.float64)                     # (B, C, f, T)
        if valid is not None:
            # A window/channel with ANY invalid frame is dropped whole: the ac1 and the
            # moving average both need CONSECUTIVE frames, and gathering a subset of frames
            # would splice non-adjacent times into an autocorrelation.
            keep = valid.all(axis=2)                                  # (B, C)
            if not keep.any():
                continue
            x = x[keep]                                               # (n, f, T)
        else:
            x = x.reshape(B * C, b1 - b0, T)
        x = x - x.mean(axis=-1, keepdims=True)
        num = float((x[..., :-1] * x[..., 1:]).sum())
        den = float((x * x).sum())
        ac1 = num / den if den > 0 else float("nan")
        if k > 1 and T >= k:
            xs = _boxcar_last(x, k, mode="valid")
            coh = float((xs * xs).mean()) / (float((x * x).mean()) or float("nan"))
        else:
            coh = float("nan")
        out.append({"b0": b0, "b1": b1, "ac1": ac1, "coh_frac": coh,
                    "std": float(np.sqrt(den / max(x.size, 1)))})
    return out


def patchmean_oracle(X: np.ndarray, patch_f: int, patch_t: int) -> np.ndarray:
    """The EXACT per-(channel, patch) mean, broadcast back — the token grid's own ceiling.

    A codec token owns one (patch_f x patch_t) patch across all channels. This predictor is
    handed that patch's exact mean for FREE, per channel, at infinite precision. Anything a
    real codec resolves INSIDE a patch it must synthesise; anything invisible here is
    invisible to any codec on this grid, which makes ``patch_f`` / ``patch_t`` (and therefore
    FRAME_LAYOUT) the blocker rather than the objective.
    """
    B, C, F, T = X.shape
    nf, nt = F // patch_f, T // patch_t
    x = X[:, :, : nf * patch_f, : nt * patch_t]
    m = x.reshape(B, C, nf, patch_f, nt, patch_t).mean(axis=(3, 5), keepdims=True)
    out = np.broadcast_to(m, (B, C, nf, patch_f, nt, patch_t)).reshape(
        B, C, nf * patch_f, nt * patch_t)
    if out.shape != X.shape:                     # pad the cropped tail with the edge patch
        full = X.copy()
        full[:, :, : out.shape[2], : out.shape[3]] = out
        return full
    return np.ascontiguousarray(out)


def tsmooth_oracle(X: np.ndarray, k: int = 5) -> np.ndarray:
    """GT low-passed along TIME with a k-frame boxcar — the no-speckle ceiling."""
    return _boxcar_last(X.astype(np.float64), k, mode="same")


def band_slices(cfg, F: int, bands_khz: List[tuple]) -> List[tuple]:
    """[(name, b0, b1)] for the requested kHz windows, clipped to F."""
    out = []
    for lo, hi in bands_khz:
        b0 = max(0, khz_to_bin(cfg, lo))
        b1 = min(F, max(b0 + 1, khz_to_bin(cfg, hi)))
        out.append((f"{lo:g}-{hi:g}kHz", b0, b1))
    return out


def coherent_rank(X: np.ndarray, M: Optional[np.ndarray], smooth: int = 5) -> Dict[str, float]:
    """How many TEMPORAL PATTERNS describe one window's coherent dynamics.

    This is the number that decides whether 192 tokens x log2(1000) = 1914 bits/window is
    the binding constraint or an excuse. Per window: low-pass along time (drop the speckle no
    codec should reproduce), remove the per-(channel, frequency) time mean (that is the static
    envelope, which the gain path codes separately), and take the spectrum of the resulting
    ``(C*F, T)`` matrix. Its rank is the count of independent time-courses the window's
    coherent content actually has -- if the 29 mirnov channels and 512 frequency bins all
    ride a handful of shared mode envelopes, that count is small and the bit budget is ample.

    Reported:
      rank90  smallest k with >= 90% of the coherent variance in the top k singular values.
      pr      participation ratio (sum lam)^2 / sum lam^2 -- a soft, threshold-free rank.
      coh_var share of the window's total variance that is coherent (post-smoothing).

    The Gram matrix is only (T, T), so the eigendecomposition is microseconds -- and it runs
    on the HOST in numpy because torch.linalg.eigh SIGKILLs silently on this machine.
    """
    B, C, F, T = X.shape
    valid = _masked_bct(M, X.shape)
    r90, prs, cvs = [], [], []
    for b in range(B):
        x = X[b].astype(np.float64)                                  # (C, F, T)
        if valid is not None:
            keep = valid[b].all(axis=-1)                             # (C,)
            if not keep.any():
                continue
            x = x[keep]
        tot = float(((x - x.mean(-1, keepdims=True)) ** 2).mean())
        xs = _boxcar_last(x, smooth, mode="same")
        xs = xs - xs.mean(-1, keepdims=True)                         # drop the envelope
        cvs.append(float((xs * xs).mean()) / tot if tot > 0 else np.nan)
        A = xs.reshape(-1, T)
        lam = np.linalg.eigvalsh(A.T @ A)[::-1].clip(min=0.0)        # (T,) descending
        tl = lam.sum()
        if tl <= 0:
            continue
        r90.append(int(np.searchsorted(np.cumsum(lam) / tl, 0.90) + 1))
        prs.append(float(tl ** 2 / (lam ** 2).sum()))
    return {"rank90": float(np.mean(r90)) if r90 else float("nan"),
            "pr": float(np.mean(prs)) if prs else float("nan"),
            "coh_var": float(np.nanmean(cvs)) if cvs else float("nan"),
            "T": T, "n_windows": len(r90)}


_CHUNK = 24

# Set by _decode_fidelity_chunked immediately before forking the pool; workers read it
# copy-on-write. Module-level (not a closure) so the fork context can reach it.
_PAR = None


def _chunk_fidelity(args):
    """One chunk of gate.decode_fidelity. Returns (n_windows, metrics dict)."""
    i, chunk = args
    recon, target, mask, pf, pt = _PAR
    sl = slice(i, i + chunk)
    d = gate.decode_fidelity(
        recon[sl], target[sl], patch_f=pf, patch_t=pt, full_spec=True, band_bins=None,
        mask=(None if mask is None else mask[sl]))
    return int(recon[sl].shape[0]), d


def _hf_chunked(a: np.ndarray, b0: int, b1: int, chunk: int = _CHUNK) -> float:
    """gate._hf_gradient_energy over a frequency slice, accumulated in window chunks.

    The metric is a SUM of squared finite differences, so chunking is exact (not an
    approximation) as long as the split is along the WINDOW axis, which carries no
    differences. Chunking matters: at 40 channels the full-band float64 copy of a 320-window
    pool is 5 GB and the metric's temporaries multiply that -- the same trap that got
    _specport_plateau's host process SIGKILLed.
    """
    tot = 0.0
    for i in range(0, a.shape[0], chunk):
        tot += gate._hf_gradient_energy(a[i:i + chunk, :, b0:b1, :].astype(np.float64))
    return tot


def _metrics_chunked(r: np.ndarray, t: np.ndarray, m: Optional[np.ndarray],
                     b0: int, b1: int, chunk: int = _CHUNK) -> Dict[str, float]:
    """gate.full_spectro_metrics over a frequency slice, averaged over window chunks.

    Same convention as the trainer's gate and analysis/_specport_plateau.py: a per-chunk mean
    of the per-(window, channel) means. Exact when chunks are equal-sized; the last chunk is
    weighted by its window count so an uneven tail cannot bias the result.
    """
    acc, wts = [], []
    for i in range(0, r.shape[0], chunk):
        sl = slice(i, i + chunk)
        met = gate.full_spectro_metrics(
            r[sl, :, b0:b1, :].astype(np.float64), t[sl, :, b0:b1, :].astype(np.float64),
            band_bins=None, mask=(None if m is None else m[sl]))
        acc.append((met["spec_nrmse"], met["spec_corr2d"]))
        wts.append(r[sl].shape[0])
    a = np.average(np.array(acc), axis=0, weights=wts)
    return {"spec_nrmse": float(a[0]), "spec_corr2d": float(a[1])}


def structure_report(modality: str, X: np.ndarray, M: np.ndarray, cfg,
                     arms: List[tuple], device="cpu", batch: int = 8,
                     n_bands: int = 16, smooth: int = 5,
                     bands_khz: Optional[List[tuple]] = None,
                     patch_aspects: Optional[List[tuple]] = None) -> Dict:
    """Print the GT coherence profile, the two oracles, and each arm's per-band sharpness."""
    B, C, F, T = X.shape
    print(f"\n=== {modality}: WHERE THE MODE STRUCTURE IS  ({B} held-out windows, "
          f"{C} ch, {F} bins x {T} frames, bin {bin_to_khz(cfg, 1) * 1e3:.0f} Hz) ===",
          flush=True)
    prof = band_structure(X, M, n_bands=n_bands, smooth=smooth)
    print(f"{'band (kHz)':>16}{'GT std':>9}{'ac1':>8}{'coh_frac':>10}   "
          f"(ac1 ~ 0 = realization speckle; high = temporally coherent = MODES)", flush=True)
    for r in prof:
        print(f"{bin_to_khz(cfg, r['b0']):7.1f}-{bin_to_khz(cfg, r['b1']):6.1f}"
              f"{r['std']:>9.3f}{r['ac1']:>8.3f}{r['coh_frac']:>10.3f}", flush=True)

    rk = coherent_rank(X, M, smooth=smooth)
    print(f"\ncoherent DIMENSION per window (time-smoothed over {smooth} frames, envelope "
          f"removed): rank90 {rk['rank90']:.1f} of T={rk['T']}   participation-ratio "
          f"{rk['pr']:.1f}   coherent share of variance {rk['coh_var']:.3f}", flush=True)
    _bits = cfg.n_tok * math.log2(cfg.codebook_size)
    print(f"  -> the window's coherent content rides ~{rk['rank90']:.0f} independent "
          f"time-courses; the token budget is {_bits:.0f} bits "
          f"({_bits / max(rk['rank90'], 1e-9):.0f} bits per coherent time-course over "
          f"{C}x{F} = {C * F} (channel, freq) cells). Bits are NOT obviously the binding "
          f"constraint; read the arm rows below as a statement about the OBJECTIVE.",
          flush=True)

    # ORACLES + arms, scored on the FULL band and on each requested sub-band.
    bands = [("full", 0, F)] + (band_slices(cfg, F, bands_khz) if bands_khz else [])
    preds: List[tuple] = [
        ("patchmean", patchmean_oracle(X, cfg.patch_f, cfg.patch_t)),
        (f"tsmooth{smooth}", tsmooth_oracle(X, smooth)),
    ]
    # PATCH ASPECT AT CONSTANT n_tok. ``n_tok = (freq_bins // patch_f) * (time_frames //
    # patch_t)``, so 8x32, 16x16, 32x8 and 64x4 ALL give 192 tokens on the production spectro
    # geometry (512 bins x 96 frames): they are a REALLOCATION of the same 192 dimensions
    # between frequency and time, not a token-count change, and FRAME_LAYOUT / the Phase-B
    # vocab are untouched. That matters because the headline ranking key, ``peak_f1``, is
    # computed on ``_power_envelope`` = ``spec.mean(-1)`` -- the window's per-FREQUENCY profile
    # with time averaged away -- and modes are thin in frequency and extended in time. Each
    # extra row is the patch-grid ceiling AT THAT ASPECT: if 8x32 lifts patchmean's peak_f1
    # well above the 16x16 value, the current grid, not the objective, is what the arms are
    # stuck against. Rows are oracles (an avg-pool), so they cost no codec decode.
    for pf, pt in (patch_aspects or []):
        if X.shape[2] % pf or X.shape[3] % pt:
            print(f"  SKIP patchmean {pf}x{pt}: does not divide {X.shape[2]}x{X.shape[3]}")
            continue
        ntok = (cfg.freq_bins // pf) * (cfg.time_frames // pt) * getattr(cfg, "channel_groups", 1)
        preds.append((f"patchmean{pf}x{pt}[n_tok {ntok}]", patchmean_oracle(X, pf, pt)))
    for label, ckpt in arms:
        if not Path(ckpt).exists():
            print(f"  SKIP {label}: {ckpt} missing")
            continue
        codec, ccfg, _ck = load_codec(ckpt, device=device)
        r, _c = reconstruct(codec, X, batch=batch, device=device)
        preds.append((label, r))
        del codec

    # peak_f1 for the ORACLES is the number that decides whether the mode TRACKS are
    # reachable at all: patchmean is what a patch-level code gives for free, tsmooth5 is what
    # perfect coherent structure gives. An arm sitting at patchmean's peak_f1 is carrying no
    # track information beyond the patch grid, however good its hf_ratio looks.
    print("\npeak_f1 (top-k spectral peak OVERLAP) is the TRACK metric and rides at the end of "
          "each row:\n  patchmean = what the patch grid gives for FREE; tsmooth5 = what "
          "PERFECT coherent structure gives. An arm sitting at patchmean's peak_f1 carries no "
          "track information beyond the grid, however good its hf_ratio looks.", flush=True)
    print(f"\n{'predictor':<18}" + "".join(
        f"{nm + ' hf':>16}{nm + ' nRMSE':>17}" for nm, _a, _b in bands), flush=True)
    gt_line = f"{'GROUND TRUTH':<18}"
    for _nm, b0, b1 in bands:
        gt_line += f"{_hf_chunked(X, b0, b1):>16.4g}{0.0:>17.4f}"
    print(gt_line, flush=True)
    rows = []
    for label, r in preds:
        line = f"{label:<18}"
        rec = {"label": label}
        for nm, b0, b1 in bands:
            hf = _hf_chunked(r, b0, b1) / max(_hf_chunked(X, b0, b1), 1e-12)
            met = _metrics_chunked(r, X, M, b0, b1)
            line += f"{hf:>16.3f}{met['spec_nrmse']:>17.4f}"
            rec[f"{nm}_hf_ratio"] = hf
            rec[f"{nm}_nrmse"] = met["spec_nrmse"]
            rec[f"{nm}_corr2d"] = met["spec_corr2d"]
        # peak_f1 on the FULL band, chunked (it pools over windows internally, so a per-chunk
        # mean is the same convention the trainer's gate uses).
        pf = float(np.mean([
            gate.decode_fidelity(r[i:i + _CHUNK], X[i:i + _CHUNK],
                                 mask=(None if M is None else M[i:i + _CHUNK]))["peak_f1"]
            for i in range(0, X.shape[0], _CHUNK)]))
        rec["peak_f1"] = pf
        print(line + f"   peak_f1 {pf:.4f}", flush=True)
        rows.append(rec)
    print("\n  hf here is recon HF-gradient energy / GT HF-gradient energy IN THAT BAND "
          "(ideal 1.0). nRMSE is a FLOOR (< 1.0), never the ranking key.")
    return {"bands_profile": prof, "coherent_rank": rk, "rows": rows,
            "bands": [{"name": nm, "b0": b0, "b1": b1} for nm, b0, b1 in bands]}


def _decode_fidelity_chunked(recon: np.ndarray, target: np.ndarray, mask: Optional[np.ndarray],
                             cfg, chunk: int = 24) -> Dict[str, float]:
    """``gate.decode_fidelity(full_spec=True)`` accumulated over WINDOW chunks.

    WHY THIS IS NOT OPTIONAL. ``decode_fidelity`` casts to float64 and evaluates
    ``_nrmse_corr2d`` + ``_envelope_correlation`` + ``_peak_overlap_f1`` +
    ``patch_lattice_metrics`` on the WHOLE array. At the required >= 300 windows that is
    320 x 40 x 512 x 96 = 629 M elements = 5 GB per copy for ece, and the temporaries multiply
    it: MEASURED 2026-09-04, three concurrent 320-window audits sat at 31 GB RSS each and
    scored ZERO arms in 38 minutes. Chunked, the same table is minutes.

    COMBINATION CONVENTION, per key -- the same one ``analysis/_specport_plateau.py``
    already documents as "identical convention to the audit":
      * every value is combined as a WINDOW-COUNT-WEIGHTED MEAN over chunks, so an uneven
        last chunk cannot bias it;
      * EXCEPT ``hf_energy_recon`` / ``hf_energy_target``, which are SUMS of squared finite
        differences over the array -- those are accumulated as sums, and ``sharpness`` is then
        recomputed as their exact ratio rather than averaged.
    ``spec_nrmse`` / ``spec_corr2d`` / ``peak_f1`` / ``envelope_corr`` are already per-window
    (or per window-channel) means inside the function, so the weighted mean is exact up to the
    per-chunk distribution of dead channels. The lattice ratios are ratios of energies, so
    their chunked mean is an approximation -- fine, they are read as orders of magnitude
    (4 vs 40 vs 130 against a GT control near 1.1), never to two decimals.
    """
    # PARALLEL over chunks. decode_fidelity is host-numpy with Python-level loops, so it is
    # single-core bound: MEASURED 2026-09-04, a 40-channel ece arm took over 2 HOURS for ONE
    # checkpoint on one core (and it was doing that on a shared login node, which is the other
    # reason this is now a process pool inside an allocation). Chunks are independent by
    # construction -- the combination is a weighted mean plus two summed energies -- so this
    # is the same arithmetic, just concurrent. Arrays are inherited copy-on-write through
    # fork; workers index the globals rather than receiving pickled slices.
    global _PAR
    _PAR = (recon, target, mask, cfg.patch_f, cfg.patch_t)
    idx = list(range(0, recon.shape[0], chunk))
    nproc = max(1, min(len(idx), int(os.environ.get("AUDIT_WORKERS", "16"))))
    if nproc > 1:
        import multiprocessing as _mp
        with _mp.get_context("fork").Pool(nproc) as pool:
            results = pool.map(_chunk_fidelity, [(i, chunk) for i in idx])
    else:
        results = [_chunk_fidelity((i, chunk)) for i in idx]
    _PAR = None
    acc: Dict[str, list] = {}
    wts: list = []
    hf_r = hf_t = 0.0
    for n, d in results:
        wts.append(n)
        hf_r += float(d["hf_energy_recon"])
        hf_t += float(d["hf_energy_target"])
        for k, v in d.items():
            if isinstance(v, (int, float)):
                acc.setdefault(k, []).append(float(v))
    w = np.asarray(wts, dtype=np.float64)
    out: Dict[str, float] = {}
    for k, vals in acc.items():
        a = np.asarray(vals, dtype=np.float64)
        ok = np.isfinite(a)
        out[k] = float(np.average(a[ok], weights=w[ok])) if ok.any() else float("nan")
    out["hf_energy_recon"], out["hf_energy_target"] = hf_r, hf_t
    out["sharpness"] = float(hf_r / hf_t) if hf_t > 1e-12 else float("nan")
    return out


def score_arm(label: str, ckpt: str, X: np.ndarray, M: np.ndarray, seq: Optional[np.ndarray],
              device="cpu", batch: int = 8) -> Dict:
    codec, cfg, ck = load_codec(ckpt, device=device)
    recon, codes = reconstruct(codec, X, batch=batch, device=device)
    out = _decode_fidelity_chunked(recon, X, M, cfg, chunk=_CHUNK)
    row = {
        "label": label, "ckpt": ckpt, "step": ck.get("step"),
        "nrmse": out["spec_nrmse"], "corr2d": out["spec_corr2d"],
        "base_self": out["base_self_spec_nrmse"],
        "base_tmean": out["base_tmean_spec_nrmse"],
        "base_cfmean": out["base_cfmean_spec_nrmse"],
        "base_wcmean": out["base_wcmean_spec_nrmse"],
        # peak_f1 is the TRACK metric -- the overlap of the top-k spectral peaks. It is the
        # only column that separates "looks like a spectrogram" from "is THIS spectrogram":
        # measured, a co2 arm at adversarial 0.2 + ms_ssim 50 reaches hf 82% of its coherent
        # ceiling and std_ratio 0.964 with GT-like granularity and correctly placed burst
        # columns, and still does not reproduce the coherent 10-20 kHz mode track.
        "peak_f1": out["peak_f1"],
        "envelope_corr": out["envelope_corr"],
        "lattice": out["patch_lattice_ratio"],
        "gt_lattice": out["target_patch_lattice_ratio"],
        "hf_ratio": out["sharpness"],
        "std_ratio": masked_std_ratio(recon, X, M),
        "n_windows": int(X.shape[0]),
    }
    row.update(gate.code_rate_bits(codes, cfg.codebook_size, n_tok=cfg.n_tok))
    ut = gate.utilization(codes, cfg=cfg)
    row["n_distinct_codes"] = ut.get("n_distinct_codes")
    row["codebook_size"] = cfg.codebook_size
    row["effective_codes"] = ut.get("effective_codes")
    row["min_dim_entropy"] = ut.get("min_dim_entropy")
    row["n_tok"] = cfg.n_tok
    row["fsq_levels"] = list(cfg.fsq_levels)
    if seq is not None:
        B, L = seq.shape[0], seq.shape[1]
        _, cs = reconstruct(codec, seq.reshape(B * L, *seq.shape[2:]), batch=batch,
                            device=device)
        cs = cs.reshape(B, L, cfg.n_tok, cfg.fsq_dim)
        fc = gate.forecastability(cs)
        row.update({"margin_overall": fc["margin_overall"],
                    "margin_transition": fc["margin_transition"],
                    "n_transition": fc["n_transition"], "n_stable": fc["n_stable"]})
    return row


def hf_references(X: np.ndarray, cfg, smooth: int = 5) -> Dict[str, float]:
    """The two hf_ratio CALIBRATION points every arm must be read against.

    ``hf_ratio`` 1.0 is NOT the target. Most of a spectrogram's HF gradient energy is
    frame-to-frame realization speckle, which no codec can or should reproduce (the recorded
    turbulent-tokenization finding: 87-91% of these codes flip per frame). So:

      hf_tsmooth   GT low-passed over ``smooth`` STFT frames. The ceiling for a codec that
                   reproduces ALL temporally coherent structure and NO speckle -- measured
                   0.250 on co2. This is the number an arm is trying to approach.
      hf_patchmean the exact per-(channel, patch) mean at infinite precision. The floor a
                   token grid gives you for free if a token says only "this patch's level"
                   -- measured 0.015 on co2. An arm below this has learned nothing the grid
                   did not already imply.
    """
    # CHUNKED over windows (see _hf_chunked): both oracles are full-size arrays, and at 40
    # channels a float64 copy of a 320-window pool is 5 GB before the metric's temporaries.
    gt = 0.0
    ts = pm = 0.0
    for i in range(0, X.shape[0], _CHUNK):
        c = X[i:i + _CHUNK]
        gt += gate._hf_gradient_energy(c.astype(np.float64))
        ts += gate._hf_gradient_energy(tsmooth_oracle(c, smooth))
        pm += gate._hf_gradient_energy(
            patchmean_oracle(c, cfg.patch_f, cfg.patch_t).astype(np.float64))
    return {"hf_tsmooth": ts / max(gt, 1e-12), "hf_patchmean": pm / max(gt, 1e-12)}


def print_table(rows: List[Dict], floor: Optional[float] = None,
                hf_ref: Optional[Dict[str, float]] = None):
    hdr = (f"{'arm':<16}{'step':>7}{'nRMSE':>9}{'corr2d':>8}{'std_r':>7}{'hf_r':>7}"
           f"{'peakF1':>8}{'envcor':>8}{'lattice':>9}{'GTlat':>7}{'bits/win':>10}"
           f"{'%ceil':>7}{'codes':>10}{'m_over':>9}{'m_trans':>9}{'n_tr':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        # UTILIZATION AS A BIT RATE. `bits_delivered_per_frame_positional` is what the tokens
        # actually carry; `bits_available_per_frame` is n_tok * log2(K). The %-of-ceiling
        # column divides by the EVAL-SIZE ceiling (an entropy from N windows cannot exceed
        # log2 N), so a perfect codec reads ~100 rather than an unreachable number.
        used = r.get("bits_delivered_per_frame_positional", float("nan"))
        avail = r.get("bits_available_per_frame", float("nan"))
        ceil = r.get("rate_positional_ceiling", 1.0) or 1.0
        pct = (100.0 * used / (avail * ceil)
               if avail and np.isfinite(avail) and avail > 0 else float("nan"))
        print(f"{r['label']:<16}{str(r.get('step')):>7}{r['nrmse']:>9.4f}{r['corr2d']:>8.4f}"
              f"{r['std_ratio']:>7.3f}{r['hf_ratio']:>7.3f}{r['peak_f1']:>8.4f}"
              f"{r['envelope_corr']:>8.4f}{r['lattice']:>9.2f}{r['gt_lattice']:>7.2f}"
              f"{used:>10.1f}{pct:>7.1f}"
              f"{str(r.get('n_distinct_codes')) + '/' + str(r.get('codebook_size')):>10}"
              f"{r.get('margin_overall', float('nan')):>9.4f}"
              f"{r.get('margin_transition', float('nan')):>9.4f}"
              f"{str(r.get('n_transition')):>7}")
    b = rows[0]
    print(f"  bits available/frame {b.get('bits_available_per_frame', float('nan')):.0f} "
          f"(n_tok {b['n_tok']} x log2 {b['codebook_size']}); eval-size rate ceiling "
          f"{b.get('rate_positional_ceiling', float('nan')):.3f} at {b['n_windows']} windows")
    print(f"\ntrivial baselines (same windows, same mask, each with its OWN masked mean): "
          f"self {b['base_self']:.4f}  wcmean {b['base_wcmean']:.4f} (the 1.0 anchor)  "
          f"tmean {b['base_tmean']:.4f}  cfmean {b['base_cfmean']:.4f}")
    if floor is not None:
        print(f"out-of-sample LINEAR FLOOR at k=n_tok: {floor:.4f}  (context only -- nRMSE is "
              f"a floor to CLEAR, not the ranking key)")
    if hf_ref:
        print(f"\nhf_ratio CALIBRATION on these windows: coherent ceiling (GT smoothed over 5 "
              f"STFT frames) {hf_ref['hf_tsmooth']:.3f}  |  patch-level floor (exact per-patch "
              f"mean) {hf_ref['hf_patchmean']:.3f}")
    print("RANKING KEYS, in order: (1) peakF1 -- do the top-k spectral peaks COINCIDE, i.e. "
          "are the mode tracks in the right place; (2) hf_ratio toward the coherent ceiling, "
          "read WITH lattice vs GTlat (a high hf at a high lattice is checkerboard, and a "
          "high hf at the RIGHT amplitude can still be the wrong texture in the wrong place "
          "-- hence peakF1 leads); (3) std_r toward 1.0. nRMSE < 1.0 is a FLOOR; do NOT rank "
          "on it, and do NOT treat tmean as a target -- it has zero temporal structure by "
          "construction.")


# ------------------------------------------------------------------------------------ #
# THE FIGURE
# ------------------------------------------------------------------------------------ #
def build_panels(modality: str, ckpt: str, shot: str, channels: List[int], device="cpu",
                 max_windows: int = 0):
    """One panel per requested channel: GT / recon strips over the WHOLE shot + a metric trace."""
    codec, cfg, _ck = load_codec(ckpt, device=device)
    ds = tc.CodecPairDataset(modality, [shot], cfg, data_dir=DATA,
                             lengths_cache_path=None, emit_mask=True)
    # A FIGURE must show consecutive real windows, never an activity-biased re-draw (which
    # would silently substitute a different chunk for a quiet one and break the time axis).
    ds.min_activity, ds.active_bias = 0.0, 0.0
    n = len(ds)
    if max_windows:
        n = min(n, max_windows)
    X, M = [], []
    for i in range(n):
        a, _b, m = ds[i]
        X.append(a.numpy())
        M.append(m.numpy())
    X = np.stack(X, 0)
    M = np.stack(M, 0)
    recon, _codes = reconstruct(codec, X, device=device)
    t0 = getattr(ds, "warmup_s", 1.0)
    step = getattr(ds, "step_size_s", 0.05)
    times = t0 + np.arange(n) * step
    return X, M, recon, cfg, times, ds


def make_figure(out_png: str, modality: str, shot: str, X, M, recon, cfg, times,
                channels: List[int], title: str, band_khz: Optional[tuple] = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N, C, F, T = X.shape
    # FREQUENCY ZOOM. At the full 0-250 kHz a coherent mode line 1-2 STFT bins wide occupies
    # well under one screen pixel, so a full-band panel cannot distinguish "resolved several
    # mode lines" from "painted one smooth band" -- the exact discrimination this figure is
    # the deliverable for. `band_khz` crops BOTH rows to the same window (see --mode
    # structure for where each modality's coherent band is). None = the full band, unchanged.
    f0, f1 = 0, F
    if band_khz is not None:
        f0 = max(0, khz_to_bin(cfg, band_khz[0]))
        f1 = min(F, max(f0 + 1, khz_to_bin(cfg, band_khz[1])))
        X, recon = X[:, :, f0:f1, :], recon[:, :, f0:f1, :]
    fmin_khz = bin_to_khz(cfg, f0)
    fmax_khz = bin_to_khz(cfg, f1)
    F = X.shape[2]
    # Stitch the per-window (F, T) tiles into one (F, N*T) shot-long spectrogram per channel.
    n_ch = len(channels)
    fig = plt.figure(figsize=(19, 3.5 * n_ch))
    gs = fig.add_gridspec(n_ch, 1, hspace=0.55)
    for pi, c in enumerate(channels):
        gt = np.concatenate([X[i, c] for i in range(N)], axis=-1)       # (F, N*T)
        rc = np.concatenate([recon[i, c] for i in range(N)], axis=-1)
        live = np.repeat(M[:, c, :].reshape(-1) > 0.5, 1)              # (N*T,)
        cov = 100.0 * float(live.mean())
        # per-panel metrics over the LIVE columns only
        g, r = gt[:, live], rc[:, live]
        if g.size:
            nr = float(np.sqrt(((r - g) ** 2).mean()) / max(g.std(), 1e-9))
            sr = float(r.std() / max(g.std(), 1e-9))
            cc = float(np.corrcoef(r.ravel(), g.ravel())[0, 1])
        else:
            nr = sr = cc = float("nan")
        vmin, vmax = np.percentile(g, [1, 99]) if g.size else (0.0, 1.0)
        # GAPS: missing columns are painted as a flat grey band in BOTH rows, never filled
        # with a reconstruction.
        gtm = np.where(live[None, :], gt, np.nan)
        rcm = np.where(live[None, :], rc, np.nan)
        sub = gs[pi].subgridspec(3, 1, hspace=0.10, height_ratios=[0.22, 1, 1])
        cap = fig.add_subplot(sub[0]); cap.axis("off")
        cap.text(0.0, 0.5,
                 f"{modality}  shot {shot}  channel {c}   |   data coverage {cov:5.1f}% "
                 f"(grey = NO DATA, never filled)   |   nRMSE {nr:.3f}   "
                 f"std(recon)/std(GT) {sr:.3f}   corr {cc:.3f}   |   "
                 f"{fmin_khz:.0f}-{fmax_khz:.0f} kHz, {N} windows x {T} STFT frames",
                 fontsize=9, family="monospace", ha="left", va="center")
        for row, img, tag in ((1, gtm, "GT"), (2, rcm, "recon")):
            ax = fig.add_subplot(sub[row])
            ax.set_facecolor("0.85")                                    # the GAP colour
            ax.imshow(img, aspect="auto", origin="lower", cmap="inferno",
                      vmin=vmin, vmax=vmax, interpolation="nearest",
                      extent=[times[0], times[-1] + T * 0.0005, fmin_khz, fmax_khz])
            ax.set_ylabel(f"{tag}\nkHz", fontsize=8)
            ax.tick_params(labelsize=7)
            if row == 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("time (s)", fontsize=8)
    fig.suptitle(title, fontsize=12, y=0.995)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    print("wrote", out_png, flush=True)


# ------------------------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["score", "figure", "structure"], required=True)
    ap.add_argument("--modality", required=True, choices=list(tc.SPECTRO_MODALITIES))
    ap.add_argument("--arms", default="", help="label=ckpt,label=ckpt,...")
    ap.add_argument("--eval_n_shots", type=int, default=16)
    ap.add_argument("--n_windows", type=int, default=320,
                    help=">= 300: the 32-window training gate has misled repeatedly.")
    ap.add_argument("--seq_len", type=int, default=8)
    ap.add_argument("--no_seq", action="store_true", help="skip forecastability")
    ap.add_argument("--floor", type=float, default=None,
                    help="this modality's out-of-sample linear floor (analysis/_specport_plateau.py)")
    ap.add_argument("--fig_shot", default=None)
    ap.add_argument("--channels", default="0", help="comma list, --mode figure")
    ap.add_argument("--max_windows", type=int, default=0, help="cap windows in --mode figure")
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--title", default=None)
    ap.add_argument("--band_khz", default=None,
                    help="--mode figure: 'lo,hi' frequency zoom in kHz (default full band). "
                         "A 1-2 bin mode line is invisible at 0-250 kHz.")
    ap.add_argument("--bands_khz", default=None,
                    help="--mode structure: 'lo-hi,lo-hi,...' sub-bands to score separately, "
                         "on top of the always-reported full band.")
    ap.add_argument("--n_bands", type=int, default=16,
                    help="--mode structure: equal-width bands for the GT ac1 profile.")
    ap.add_argument("--smooth", type=int, default=5,
                    help="--mode structure: frames in the time-smoothing oracle / coh_frac.")
    ap.add_argument("--patch_aspects", default="",
                    help="--mode structure: extra patchmean oracles at 'FxT,FxT,...' patch "
                         "sizes, e.g. '8x32,32x8'. On the production spectro geometry every "
                         "aspect whose (512//F)*(96//T) is 192 keeps n_tok and FRAME_LAYOUT "
                         "unchanged -- these rows say whether the frequency/time SPLIT of the "
                         "same 192 dimensions is what caps peak_f1.")
    args = ap.parse_args()

    specs = []
    for tok in args.arms.split(","):
        tok = tok.strip()
        if not tok:
            continue
        label, _, path = tok.partition("=")
        specs.append((label if path else Path(label).parent.name, path or label))
    if not specs:
        raise SystemExit("--arms is required (label=ckpt,...)")

    # ---- POOL-CONSISTENCY GUARD -------------------------------------------------------- #
    # window_pool() builds the GROUND TRUTH from specs[0]'s cfg, so every arm in one run is
    # scored against a GT standardised the FIRST arm's way. Mixing arms whose input transform
    # differs silently corrupts the whole table.
    #
    # MEASURED 2026-09-04: production ece (logpow_freq_mean=None, raw per-channel z) was listed
    # first alongside arms trained with per-frequency log-z. The GT changed underneath every-
    # thing -- tmean 0.9688 -> 0.7258, GT patch-lattice 1.05 -> 1.13 -- and the log-z arms, fed
    # a distribution they had never seen, read nRMSE 3.2-3.9 instead of ~1.1. Three conclusions
    # were drawn from that table and all three were wrong ("ece diverges", "arm at 77% of its
    # ceiling", "aspect buys ece nothing"). The one finding that survived was the production
    # codec's own collapse, because it alone was scored in its native normalisation.
    #
    # Codecs with DIFFERENT standardisation must be scored in SEPARATE runs and their numbers
    # never placed in one table.
    if len(specs) > 1:
        _sig = {}
        for _lab, _ck in specs:
            if not Path(_ck).exists():
                continue
            try:
                _c = load_codec(_ck, device="cpu")[1]
            except Exception:
                continue
            _sig.setdefault(getattr(_c, "logpow_freq_mean", None) is not None, []).append(_lab)
        if len(_sig) > 1:
            _on = _sig.get(True, []); _off = _sig.get(False, [])
            raise SystemExit(
                "REFUSING TO SCORE: the arms disagree on INPUT STANDARDISATION, so one pool "
                "cannot serve them.\n"
                f"  per-freq log-z ON  ({len(_on)}): {', '.join(_on[:6])}{' ...' if len(_on)>6 else ''}\n"
                f"  per-freq log-z OFF ({len(_off)}): {', '.join(_off[:6])}{' ...' if len(_off)>6 else ''}\n"
                "The ground truth is built from the FIRST arm's cfg; the others would be fed an "
                "input distribution they never saw. Score them in separate runs."
            )

    shots = held_out_shots(args.modality, args.eval_n_shots)
    print(f"[{args.modality}] held-out shots: {shots}", flush=True)

    if args.mode == "structure":
        _c0, cfg0 = load_codec(specs[0][1], device="cpu")[:2]
        X, M = window_pool(args.modality, cfg0, shots, args.n_windows)
        print(f"[{args.modality}] structure on {X.shape[0]} held-out windows {X.shape[1:]}; "
              f"real-data fraction {float((M > 0.5).mean()):.4f}", flush=True)
        bands = None
        if args.bands_khz:
            bands = []
            for tok in args.bands_khz.split(","):
                lo, _, hi = tok.strip().partition("-")
                bands.append((float(lo), float(hi)))
        aspects = []
        for tok in (args.patch_aspects or "").split(","):
            tok = tok.strip()
            if not tok:
                continue
            a, _, b = tok.partition("x")
            aspects.append((int(a), int(b)))
        rep = structure_report(args.modality, X, M, cfg0, specs, device=args.device,
                               batch=args.batch_size, n_bands=args.n_bands,
                               smooth=args.smooth, bands_khz=bands,
                               patch_aspects=aspects)
        if args.json:
            Path(args.json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.json).write_text(json.dumps(rep, indent=1, default=float))
            print("wrote", args.json)
        return

    if args.mode == "figure":
        shot = args.fig_shot or shots[0]
        chans = [int(c) for c in args.channels.split(",") if c.strip()]
        label, ckpt = specs[0]
        X, M, recon, cfg, times, _ds = build_panels(
            args.modality, ckpt, shot, chans, device=args.device,
            max_windows=args.max_windows)
        out = args.out or f"eval_runs/codec_recon_figs/{args.modality}_FINAL_fullshot.png"
        bk = None
        if args.band_khz:
            lo, _, hi = args.band_khz.partition(",")
            bk = (float(lo), float(hi))
        title = args.title or (
            f"IGNITE {args.modality} spectrogram codec [{label}] - GT vs reconstruction, "
            + (f"{bk[0]:g}-{bk[1]:g} kHz zoom" if bk else "full 0-250 kHz")
            + ", gaps never filled")
        make_figure(out, args.modality, shot, X, M, recon, cfg, times, chans, title,
                    band_khz=bk)
        return

    # --mode score
    _c0, cfg0 = load_codec(specs[0][1], device="cpu")[:2]
    X, M = window_pool(args.modality, cfg0, shots, args.n_windows)
    print(f"[{args.modality}] scored on {X.shape[0]} held-out windows "
          f"{X.shape[1:]}; real-data fraction of the pool {float((M > 0.5).mean()):.4f}",
          flush=True)
    seq = None if args.no_seq else seq_pool(args.modality, cfg0, shots, seq_len=args.seq_len)
    rows = []
    for label, ckpt in specs:
        if not Path(ckpt).exists():
            print(f"  SKIP {label}: {ckpt} missing", flush=True)
            continue
        rows.append(score_arm(label, ckpt, X, M, seq, device=args.device,
                              batch=args.batch_size))
        print(f"  scored {label}", flush=True)
    if not rows:
        raise SystemExit("no arms scored")
    print_table(rows, floor=args.floor, hf_ref=hf_references(X, cfg0))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(rows, indent=1, default=float))
        print("wrote", args.json)


if __name__ == "__main__":
    main()
