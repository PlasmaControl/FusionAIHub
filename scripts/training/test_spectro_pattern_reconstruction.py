#!/usr/bin/env python
"""Objective thin-pattern retention test for the spectrogram encoder/decoder.

WHY
---
The deterministic spectro head regressed thin mode structure to a blurry
conditional-mean envelope (mean-collapse), and the full-frequency patch
``(512, 4)`` (1 frequency token, cond_map bilinear-upsampled 1->512) gave the
generative flow head no frequency localization to PLACE modes. The fix:
  * reallocate the patch to ``(64, 32)`` -> 8 frequency tokens x 3 time tokens
    = the SAME 24-token budget but 8x the frequency localization, and
  * add a sinusoidal FREQUENCY positional embedding to the flow velocity net so
    its (translation-equivariant) convs gain absolute-frequency awareness.

This test proves OBJECTIVELY that the new encoder/decoder RETAINS thin, mode-
like patterns that the old configuration blurs away. It overfits the
encode (SpectrogramTokenizer) -> decode (SpectrogramFlowHead) round-trip on a
set of synthetic mode-like spectrograms and measures reconstruction quality
per pattern family for the OLD vs NEW configuration at the IDENTICAL token
budget, so the comparison isolates the patch/PE change.

PATTERNS (all THIN, high-contrast over a quiet background) - chosen to look
like real tokamak MHD structure:
  * steady    : 1-2 horizontal lines (constant-frequency mode) + a harmonic
  * chirp      : diagonal line (frequency sweep)
  * drift      : a wavy horizontal band (tearing-mode-like frequency drift)
  * burst      : vertical line(s) (broadband ELM transient)
  * intermittent: a horizontal mode amplitude-modulated on/off in time

METRICS (per family, reported for both mu and a flow sample):
  * PSNR (dB)            - overall reconstruction fidelity
  * SSIM (windowed)      - structural similarity (sensitive to thin structure)
  * line-contrast ratio  - (recon[pattern]-recon[bg]) / (gt[pattern]-gt[bg]);
                           ~1 = thin contrast preserved, ~0 = blurred to mean
  * Pearson correlation  - GT-vs-recon structure agreement

PASS: NEW reconstruction retains thin patterns at high quality (mean SSIM and
line-contrast across families above threshold) AND clearly beats OLD.

Run on 1 GPU (falls back to CPU). Writes a metrics table + a GT|OLD|NEW figure
to ``--out_dir``.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# repo src on path (sbatch also sets PYTHONPATH; this makes the file runnable
# directly from the repo root too).
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from tokamak_foundation_model.e2e.tokenizers.spectrogram import (  # noqa: E402
    SpectrogramTokenizer,
)
from tokamak_foundation_model.e2e.output_heads import (  # noqa: E402
    SpectrogramFlowHead,
    SpectrogramOutputHead,
)
from tokamak_foundation_model.e2e.quantizers import FSQ, FSQBottleneck  # noqa: E402,F401

FAMILIES = ["steady", "chirp", "drift", "burst", "intermittent"]


# --------------------------------------------------------------------------- #
# Synthetic mode-like spectrograms                                            #
# --------------------------------------------------------------------------- #
def _blank(C, F, T, rng, bg=0.10, noise=0.02):
    x = bg + noise * rng.standard_normal((C, F, T)).astype(np.float32)
    m = np.zeros((F, T), dtype=bool)
    return x, m


def _paint_row(x, m, f0, t0, t1, amp, width):
    F_ = x.shape[1]
    lo, hi = max(0, f0 - width), min(F_, f0 + width + 1)
    x[:, lo:hi, t0:t1] += amp
    m[lo:hi, t0:t1] = True


def make_dataset(n_per: int, C: int, F: int, T: int, seed: int = 0):
    """Return (X (N,C,F,T) float32, masks (N,F,T) bool, types (N,) int)."""
    rng = np.random.default_rng(seed)
    X, M, TY = [], [], []
    amp, w = 1.0, 1                       # thin (half-width 1 -> 3 bins) bright lines
    for ti, fam in enumerate(FAMILIES):
        for _ in range(n_per):
            x, m = _blank(C, F, T, rng)
            if fam == "steady":
                f0 = int(rng.integers(40, F - 120))
                _paint_row(x, m, f0, 0, T, amp, w)
                if rng.random() < 0.8:                 # a harmonic
                    _paint_row(x, m, min(F - 2, 2 * f0), 0, T, 0.7 * amp, w)
            elif fam == "chirp":
                f0 = int(rng.integers(30, F // 2))
                f1 = int(rng.integers(F // 2, F - 30))
                for t in range(T):
                    f = int(f0 + (f1 - f0) * t / max(1, T - 1))
                    _paint_row(x, m, f, t, t + 1, amp, w)
            elif fam == "drift":
                f0 = int(rng.integers(120, F - 120))
                A = float(rng.integers(30, 90)); k = float(rng.integers(1, 4))
                for t in range(T):
                    f = int(f0 + A * math.sin(2 * math.pi * k * t / T))
                    _paint_row(x, m, f, t, t + 1, amp, w)
            elif fam == "burst":
                for _ in range(int(rng.integers(1, 3))):
                    t0 = int(rng.integers(5, T - 5))
                    x[:, :, t0:t0 + 1] += amp; m[:, t0:t0 + 1] = True
            elif fam == "intermittent":
                f0 = int(rng.integers(60, F - 60))
                period = int(rng.integers(8, 20))
                for t in range(T):
                    if (t // period) % 2 == 0:
                        _paint_row(x, m, f0, t, t + 1, amp, w)
            X.append(x); M.append(m); TY.append(ti)
    return (
        torch.from_numpy(np.stack(X)).float(),
        torch.from_numpy(np.stack(M)),
        torch.tensor(TY, dtype=torch.long),
    )


# --------------------------------------------------------------------------- #
# Real-data spectrogram windows (single shot)                                 #
# --------------------------------------------------------------------------- #
def load_real_shot_spectro(shot, data_dir, stats_path, modality, n_windows,
                           n_channels, patch_t=32):
    """Load MODEL-NORMALIZED spectrogram windows for ONE real shot.

    Reuses ``TokamakMultiFileDataset`` -- the SAME class the trainer uses -- so
    the normalization is byte-for-byte the training path: torch.stft(n_fft=1024,
    hop=256) magnitude -> log10(clip(x,-0.99)+1) -> per-channel standardize from
    ``preprocessing_stats.pt``. This deliberately AVOIDS
    ``eval_e2e_animation_tokamak.load_and_spectrogram`` (scipy power spectrum,
    ~140x off the model scale).

    Returns (X (N,C,F,T) float32, types (N,) long all-zero). On real data there
    are no planted ground-truth modes, so the caller derives the "true" mode
    masks as ``mode_hard(X)`` (binarized GT) -- the validity ceiling is then
    trivially 1.0 and the meaningful number is mDice(pred) vs mDice(GT).
    """
    from tokamak_foundation_model.data.multi_file_dataset import (
        TokamakMultiFileDataset,
    )
    shot_file = os.path.join(data_dir, f"{shot}_processed.h5")
    if not os.path.exists(shot_file):
        raise FileNotFoundError(shot_file)
    stats = torch.load(stats_path, weights_only=False)
    ds = TokamakMultiFileDataset(
        hdf5_paths=[shot_file],
        chunk_duration_s=0.05,
        prediction_mode=True,
        prediction_horizon_s=0.05,
        step_size_s=0.01,
        warmup_s=1.0,
        n_fft=1024,
        hop_length=256,
        preprocessing_stats=stats,
        input_signals=[modality],
        target_signals=[modality],
    )
    n_total = len(ds)
    if n_total == 0:
        raise RuntimeError(f"no windows for shot {shot} modality {modality}")
    # evenly spaced indices across the shot (windows overlap at 10ms stride, so
    # sub-sample to span the discharge rather than take N adjacent windows).
    k = max(1, n_total // max(1, n_windows))
    idxs = list(range(0, n_total, k))[:n_windows]
    xs = []
    for i in idxs:
        s = ds[i]
        x = s["inputs"][modality]                     # (C, F, T) model-norm
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        x = torch.nan_to_num(x.float(), nan=0.0)
        if float(x.std()) < 1e-4:                     # skip dead/missing windows
            continue
        xs.append(x)
    if not xs:
        raise RuntimeError(f"all windows empty for shot {shot}/{modality}")
    X = torch.stack(xs)                                # (N, C, F, T)
    T = X.shape[-1]
    X = X[..., : (T // patch_t) * patch_t]             # T -> multiple of patch_t
    if n_channels and n_channels < X.shape[1]:
        X = X[:, :n_channels]
    types = torch.zeros(X.shape[0], dtype=torch.long)
    print(f"[real] shot {shot} modality {modality}: {n_total} windows total, "
          f"using {X.shape[0]} (every {k}th), X={tuple(X.shape)}", flush=True)
    return X, types


# --------------------------------------------------------------------------- #
# Mode-survival / decorrelation analysis (#1 — MODEL-FREE predictability test) #
# How much of the target-window mode structure is determined by the recent     #
# past: binarize each shot's spectrogram (production rule) into a (C,F,T) mode  #
# field, then measure Dice(mode field, mode field shifted by Δ) vs the time     #
# gap Δ. The curve decays from 1 (Δ=0) toward the chance floor (= mode density).#
# Compared to the 50 ms forecast horizon: survives ≫ chance at 50 ms => modes   #
# are PREDICTABLE (a forecast can get them); decayed to ~chance within 50 ms => #
# STOCHASTIC (no forecast-mean / head fix recovers them). Reuses the prod        #
# binarization (mode_hard, _USE_PROD_BIN). GPU-accelerated.                     #
# --------------------------------------------------------------------------- #
def load_contiguous_spectro(shot, data_dir, stats_path, modality, max_windows=0):
    """Full-shot MODEL-NORMALIZED spectrogram as a CONTIGUOUS (C,F,T) tensor.

    Non-overlapping windows (step = chunk) tiled across the shot, concatenated
    along time → the model's exact spectro frames in order. Returns
    (X (C,F,T_full), frame_dt_s). Reuses TokamakMultiFileDataset (training path).
    """
    from tokamak_foundation_model.data.multi_file_dataset import (
        TokamakMultiFileDataset,
    )
    shot_file = os.path.join(data_dir, f"{shot}_processed.h5")
    if not os.path.exists(shot_file):
        raise FileNotFoundError(shot_file)
    stats = torch.load(stats_path, weights_only=False)
    ds = TokamakMultiFileDataset(
        hdf5_paths=[shot_file], chunk_duration_s=0.05, prediction_mode=True,
        prediction_horizon_s=0.05, step_size_s=0.05,   # NON-overlapping tiles
        warmup_s=1.0, n_fft=1024, hop_length=256,
        preprocessing_stats=stats, input_signals=[modality],
        target_signals=[modality],
    )
    n = len(ds)
    if n == 0:
        raise RuntimeError(f"no windows for {shot}/{modality}")
    if max_windows:
        n = min(n, max_windows)
    frames = []
    for i in range(n):
        x = ds[i]["inputs"][modality]
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        x = torch.nan_to_num(x.float(), nan=0.0)        # (C,F,T_win)
        if float(x.std()) < 1e-4:
            continue
        frames.append(x)
    if not frames:
        raise RuntimeError(f"all windows empty {shot}/{modality}")
    X = torch.cat(frames, dim=-1)                        # (C,F,T_full)
    frame_dt_s = 0.05 / frames[0].shape[-1]              # 50 ms window / frames
    return X, frame_dt_s


def _freq_dilate(mask, tol):
    """Max-pool the mode mask over FREQUENCY by ±tol bins, so a mode that DRIFTS
    in frequency (or whose exact bin flickers) still counts as 'present'. This
    turns the strict exact-pixel survival into a band/mode-presence survival —
    the predictability the model can actually use (a coherent mode in the input
    window persists as a band even as its exact bin moves). x: (C,F,T)."""
    if tol <= 0:
        return mask
    m = mask.unsqueeze(1)                                   # (C,1,F,T)
    m = F.max_pool2d(m, kernel_size=(2 * tol + 1, 1), stride=1, padding=(tol, 0))
    return m.squeeze(1)


def _survival_accum(mask, max_lag):
    """Per-lag Dice numerator/denominator for a (C,F,T) {0,1} mask (on device).
    Returns (num[L+1], den[L+1]) tensors; lag d uses overlap of t vs t+d."""
    T = mask.shape[-1]
    L = min(max_lag, T - 2)
    num = torch.zeros(L + 1, device=mask.device)
    den = torch.zeros(L + 1, device=mask.device)
    base = mask.sum()
    for d in range(L + 1):
        a = mask[..., : T - d] if d > 0 else mask
        b = mask[..., d:]
        num[d] = 2.0 * (a * b).sum()
        den[d] = a.sum() + b.sum()
    return num, den


def run_survival(args, device):
    """#1 mode-survival/decorrelation curve for spectro modalities, model-free."""
    import glob
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    global _USE_PROD_BIN
    _USE_PROD_BIN = True
    os.makedirs(args.out_dir, exist_ok=True)
    mods = [m.strip() for m in args.survival_modalities.split(",") if m.strip()]
    kbm = {"ece": 2.5, "co2": 2.0, "bes": 2.0}
    files = sorted(glob.glob(os.path.join(args.data_dir, "*_processed.h5")))
    shots = [int(os.path.basename(f).split("_")[0]) for f in files
             if os.path.basename(f).split("_")[0].isdigit()]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(shots)
    pick = []
    if args.real_shot and args.real_shot in shots:
        pick.append(args.real_shot)
    for s in shots:
        if len(pick) >= args.survival_shots:
            break
        if s not in pick:
            pick.append(s)
    print(f"[survival] device={device} shots={sorted(pick)} modalities={mods}",
          flush=True)
    tols = [int(t) for t in args.survival_freq_tols.split(",") if t.strip() != ""]
    results = {}
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for mod in mods:
        k = kbm.get(mod, 2.0)
        acc = {t: [None, None, 0.0, 0.0] for t in tols}   # tol -> [num,den,modepix,totpix]
        frame_dt = None; nshot = 0
        for shot in sorted(pick):
            try:
                X, frame_dt = load_contiguous_spectro(
                    shot, args.data_dir, args.stats_path, mod,
                    max_windows=args.survival_max_windows)
            except Exception as e:
                print(f"  [survival] {shot}/{mod} skip: {type(e).__name__}: {e}",
                      flush=True)
                continue
            base = mode_hard(X[None].to(device), None, None, k)[0]   # (C,F,T) strict
            for t in tols:
                m = _freq_dilate(base, t)                  # ±t-bin freq tolerance
                num, den = _survival_accum(m, args.survival_max_lag)
                num, den = num.cpu(), den.cpu()
                a = acc[t]
                if a[0] is None:
                    a[0], a[1] = num.clone(), den.clone()
                else:
                    L = min(len(a[0]), len(num))
                    a[0] = a[0][:L] + num[:L]; a[1] = a[1][:L] + den[:L]
                a[2] += float(m.sum()); a[3] += float(m.numel())
            nshot += 1
        for t in tols:
            a = acc[t]
            if a[0] is None:
                print(f"  [survival] {mod} ±{t}: no usable shots", flush=True); continue
            surv = (a[0] / a[1].clamp_min(1.0)).numpy()
            dens = a[2] / max(a[3], 1.0)                    # chance Dice floor
            lags_ms = np.arange(len(surv)) * frame_dt * 1000.0
            half = dens + (1.0 - dens) / 2.0
            tau_ms = float(lags_ms[np.argmax(surv <= half)] if np.any(surv <= half)
                           else lags_ms[-1])
            hidx = int(np.argmin(np.abs(lags_ms - 50.0)))
            surv_50 = float(surv[hidx])
            norm_50 = (surv_50 - dens) / max(1.0 - dens, 1e-6)  # 1=survives, 0=chance
            verdict = ("PREDICTABLE" if norm_50 >= 0.5 else
                       "STOCHASTIC" if norm_50 <= 0.2 else "PARTIAL")
            results[f"{mod}±{t}bin"] = dict(chance=dens, tau_ms=tau_ms,
                                            surv_50=surv_50, norm_50=norm_50,
                                            verdict=verdict, nshot=nshot)
            line, = ax.plot(lags_ms, surv,
                            label=f"{mod} ±{t}bin: τ½={tau_ms:.0f}ms "
                                  f"surv@50ms={surv_50:.2f}(n{norm_50:.2f})→{verdict}")
            ax.axhline(dens, ls=":", lw=0.7, color=line.get_color())
    ax.axvline(50.0, ls="--", color="k", lw=1.0, label="50 ms forecast horizon")
    ax.set_xlabel("time gap Δ (ms)"); ax.set_ylabel("mode-mask Dice (survival)")
    ax.set_ylim(0, 1)
    ax.set_title(f"Spectro mode survival / decorrelation ({len(pick)} shots)\n"
                 "dotted = chance floor (mode density); above it at 50 ms = predictable")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    figp = os.path.join(args.out_dir, "mode_survival.png")
    fig.savefig(figp, dpi=120); fig.savefig(figp.replace(".png", ".pdf"))
    with open(os.path.join(args.out_dir, "mode_survival.txt"), "w") as fh:
        fh.write(f"shots={sorted(pick)}\n")
        for mod, r in results.items():
            ln = (f"{mod}: chance={r['chance']:.3f}  tau_half={r['tau_ms']:.1f}ms  "
                  f"surv@50ms={r['surv_50']:.3f}  norm@50ms={r['norm_50']:.3f}  "
                  f"({r['nshot']} shots) -> {r['verdict']}")
            fh.write(ln + "\n"); print("[survival] " + ln, flush=True)
    print(f"[survival] figure -> {figp}", flush=True)


# --------------------------------------------------------------------------- #
# Model: encode (tokenizer) -> decode (flow head)                              #
# --------------------------------------------------------------------------- #
_VAR_DEFAULTS = {"pf": 512, "pt": 4, "fpe": 0, "tpe": 0, "fstem": 0, "istem": 0,
                 # coherent-mode experiment flags (all default OFF):
                 #   struct (A): soft-binarized structural Dice loss on mu
                 #   mask   (B): auxiliary binary mode-mask head (deterministic)
                 #   cond   (C): mask-gated flow residual at eval
                 "struct": 0, "mask": 0, "cond": 0,
                 # fsq (D): insert a discrete FSQ bottleneck (levels from
                 # --fsq_levels) between encoder tokens and decoder -> tests
                 # whether modes survive quantization (Stage-A gate for the
                 # VQ/FSQ pivot). 0 = off (continuous, the original test).
                 # fsqd>0 overrides --fsq_levels with [fsqL]*fsqd (per-variant
                 # dim sweep: more dims = higher fidelity, more Stage-B heads).
                 "fsq": 0, "fsqd": 0, "fsqL": 8,
                 # per-variant struct-loss weight (None -> use the global
                 # --lam_struct). Lets one job SCAN lambda across variants.
                 "lam": None}


def parse_variants(s: str):
    """Parse a ';'-separated variant spec into dicts. Each variant:
    ``name:pf=64,pt=32,fpe=16,tpe=8,fstem=1,istem=1`` (omitted keys default to
    the OLD baseline). Example default sweep below."""
    out = []
    for tokn in s.split(";"):
        tokn = tokn.strip()
        if not tokn:
            continue
        name, _, kvs = tokn.partition(":")
        d = dict(_VAR_DEFAULTS); d["name"] = name.strip()
        for kv in kvs.split(","):
            kv = kv.strip()
            if kv:
                k, v = kv.split("=")
                k = k.strip()
                d[k] = float(v) if k == "lam" else int(v)
        out.append(d)
    return out


def build_variant(spec: dict, C: int, F: int, T: int, d_model: int,
                  base_ch: int, flow_steps: int, fsq_levels=None):
    """Build (tokenizer, head, mask_head_or_None, fsq_or_None, label)."""
    pf, pt = int(spec["pf"]), int(spec["pt"])
    fpe, tpe = int(spec["fpe"]), int(spec["tpe"])
    fstem, istem = bool(spec["fstem"]), bool(spec["istem"])
    struct = bool(spec.get("struct", 0))
    mask = bool(spec.get("mask", 0))
    cond = bool(spec.get("cond", 0))
    npf, npt = F // pf, T // pt
    tok = SpectrogramTokenizer(
        n_channels=C, d_model=d_model, patch_f=pf, patch_t=pt,
        freq_bins=F, time_frames=T, enable_freq_stem=fstem,
    )
    head = SpectrogramFlowHead(
        n_channels=C, d_model=d_model, patch_f=pf, patch_t=pt,
        n_patches_f=npf, n_patches_t=npt, flow_base_ch=base_ch,
        flow_sample_steps=flow_steps, flow_lambda=1.0,
        flow_freq_pe_ch=fpe, flow_time_pe_ch=tpe, enable_inv_stem=istem,
    )
    # Optional binary mode-mask head (B): a SEPARATE deterministic head with the
    # SAME patch config; its raw output is treated as per-bin mask LOGITS.
    mask_head = None
    if mask:
        mask_head = SpectrogramOutputHead(
            n_channels=C, d_model=d_model, patch_f=pf, patch_t=pt,
            n_patches_f=npf, n_patches_t=npt,
        )
    # (D) optional discrete FSQ bottleneck between encoder tokens and decoder.
    fsq = None
    if bool(spec.get("fsq", 0)):
        fsqd = int(spec.get("fsqd", 0))
        levels = [int(spec.get("fsqL", 8))] * fsqd if fsqd > 0 else fsq_levels
        if levels:
            fsq = FSQBottleneck(d_model, levels)
    extras = []
    if fstem: extras.append("fStem")
    if istem: extras.append("invStem")
    label = (f"{spec['name']:<10} patch({pf},{pt}) {npf}fx{npt}t fPE{fpe} tPE{tpe}"
             + (" " + "+".join(extras) if extras else ""))
    if struct: label += " +A"
    if mask: label += " +B"
    if cond: label += " +C"
    if fsq is not None:
        _d = fsq.fsq.dim; _L = fsq.fsq.levels_list
        label += f" +FSQ(dim={_d},L={_L[0]},~{_d * math.log2(_L[0]):.0f}bits)"
    return tok, head, mask_head, fsq, label


def train(tok, head, mask_head, X, steps, lr, device, mu_f, sd_f, mode_k,
          lam_struct, lam_mask, struct, mask, cond, fsq=None, no_flow=False,
          log_every=500, tag=""):
    tok.train(); head.train()
    params = list(tok.parameters()) + list(head.parameters())
    if mask_head is not None:
        mask_head.train()
        params = params + list(mask_head.parameters())
    if fsq is not None:
        fsq.train()
        params = params + list(fsq.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    X = X.to(device)
    mu_f = mu_f.to(device); sd_f = sd_f.to(device)
    n = X.shape[0]
    # CRITICAL: set the per-(channel,freq) residual scale sigma_pb, exactly as
    # the production trainer does from per-bin stats. The flow models
    # (target-mu)/sigma_pb; with sigma_pb=1 (the default) a small residual makes
    # the velocity target ~0 -> the net learns nothing (loss stuck at ~1) and the
    # eval sample = mu + 1*noise = garbage. Setting sigma_pb to the data's per-bin
    # std makes the standardised residual unit-scale (the flow learns structure)
    # and the injected noise is scaled correctly (quiet bins stay quiet).
    with torch.no_grad():
        sig = X.std(dim=(0, 3)).clamp_min(0.05)       # (C, F) over samples & time
        head.set_sigma_pb(sig.to(device))
    print(f"  [{tag}] sigma_pb set: mean={sig.mean().item():.3f} "
          f"min={sig.min().item():.3f} max={sig.max().item():.3f}", flush=True)
    # hard target masks from the TRUE data are independent of the model, so the
    # struct-loss target and the mask-head BCE target can be precomputed once.
    with torch.no_grad():
        tgt_hard = mode_hard(X, mu_f, sd_f, mode_k)   # (N, C, F, T) in {0, 1}
    for s in range(steps):
        opt.zero_grad(set_to_none=True)
        tokens = tok._encode(X)                       # (N, n_tok, d_model)
        if fsq is not None:
            tokens, _ = fsq(tokens)                   # discrete FSQ bottleneck
        mu = head.mean_head(tokens)
        mae = (mu - X).abs().mean()
        if no_flow:
            flow = torch.zeros((), device=device)
            loss = mae
        else:
            flow = head.flow_loss(tokens, mu, X, mask=None)
            loss = mae + head.flow_lambda * flow
        Ls = torch.zeros((), device=device)
        Lm = torch.zeros((), device=device)
        if struct:
            # (A) push the deterministic mean's soft mode-mask toward the true
            # hard mode-mask -> rewards SHARP coherent ridges (not blurry mean).
            Ls = dice_loss(mode_soft(mu, mu_f, sd_f, mode_k), tgt_hard)
            loss = loss + lam_struct * Ls
        if mask and mask_head is not None:
            # (B) auxiliary binary mode-mask head: per-bin logits vs true mask.
            ml = mask_head(tokens)
            Lm = F.binary_cross_entropy_with_logits(ml, tgt_hard)
            loss = loss + lam_mask * Lm
        loss.backward()
        opt.step()
        if (s + 1) % log_every == 0 or s == 0:
            print(f"  [{tag}] step {s+1}/{steps}  mae={mae.item():.4f} "
                  f"flow={flow.item():.4f}  Ls={Ls.item():.4f} "
                  f"Lm={Lm.item():.4f}", flush=True)
    return tok, head, mask_head


# --------------------------------------------------------------------------- #
# Mode binarization (mirror production GT-fusion: smooth -> z-score -> thresh) #
# --------------------------------------------------------------------------- #
def _smooth_ft(x, ks=3):
    """Average-pool over (F, T). x: (N, C, F, T)."""
    return F.avg_pool2d(x, ks, stride=1, padding=ks // 2)


def _mode_z(x, mu_f, sd_f, ks=3):
    """Per-bin z-score of the smoothed spectrogram. mu_f, sd_f: (C, F)."""
    xs = _smooth_ft(x, ks)
    return (xs - mu_f[None, :, :, None]) / sd_f[None, :, :, None].clamp_min(1e-3)


# --- PRODUCTION binarization (exact mirror of eval_e2e_animation_tokamak.
# fuse_spectro_with_gt, the rule that successfully extracts modes on 200729):
# gaussian-smooth (sigma_f, sigma_t) -> per-FREQUENCY background mu/sd over TIME
# -> soft_mask = clip((smooth-mu)/(k*sd), 0, 1)^gamma. It is SELF-NORMALIZING
# (mu/sd from the input), hence INVARIANT to the per-channel standardization of
# the model space, so it gives the identical mask on normalized X as on log10.
_USE_PROD_BIN = False
_PROD_GAMMA = 2.0
_PROD_SMOOTH_F = 1.0       # gaussian sigma along freq (matches _MASK_SMOOTH_F)
_PROD_SMOOTH_T = 2.0       # gaussian sigma along time (matches _MASK_SMOOTH_T)
_PROD_HARD_CUT = 0.5       # soft_mask > cut -> "this is a mode" (binary)


def _gauss1d(sigma, device, dtype):
    r = max(1, int(round(3 * sigma)))
    xs = torch.arange(-r, r + 1, device=device, dtype=dtype)
    k = torch.exp(-(xs ** 2) / (2.0 * sigma * sigma))
    return (k / k.sum()), r


def _gauss_smooth(x, sf, st):
    """Separable gaussian blur over (F, T). x: (N, C, F, T)."""
    N, C, Fb, T = x.shape
    kf, rf = _gauss1d(sf, x.device, x.dtype)
    kt, rt = _gauss1d(st, x.device, x.dtype)
    xr = x.reshape(N * C, 1, Fb, T)
    xr = F.conv2d(xr, kf.view(1, 1, -1, 1), padding=(rf, 0))
    xr = F.conv2d(xr, kt.view(1, 1, 1, -1), padding=(0, rt))
    return xr.reshape(N, C, Fb, T)


def _mode_z_prod(x, k):
    """soft_mask ARGUMENT (smooth-mu)/(k*sd); per-freq mu/sd over TIME."""
    sm = _gauss_smooth(x, _PROD_SMOOTH_F, _PROD_SMOOTH_T)
    mu = sm.mean(dim=-1, keepdim=True)                 # per (n,c,freq), over T
    sd = sm.std(dim=-1, keepdim=True).clamp_min(1e-6)
    return (sm - mu) / (k * sd)


def mode_soft(x, mu_f, sd_f, k, alpha=4.0, ks=3):
    """Soft (differentiable) mode mask in [0, 1]."""
    if _USE_PROD_BIN:
        return _mode_z_prod(x, k).clamp(0.0, 1.0) ** _PROD_GAMMA
    return torch.sigmoid(alpha * (_mode_z(x, mu_f, sd_f, ks) - k))


def mode_hard(x, mu_f, sd_f, k, ks=3):
    """Hard mode mask as float."""
    if _USE_PROD_BIN:
        return (mode_soft(x, mu_f, sd_f, k) > _PROD_HARD_CUT).float()
    return (_mode_z(x, mu_f, sd_f, ks) > k).float()


def dice_loss(p, t, eps=1.0):
    """Soft-Dice LOSS (1 - dice). p, t broadcastable tensors in [0, 1]."""
    num = 2 * (p * t).sum() + eps
    den = p.sum() + t.sum() + eps
    return 1 - num / den


def dice_sim(p, t, eps=1.0):
    """Dice SIMILARITY = 2*|A∩B|/(|A|+|B|) in [0, 1]; = 1 - dice_loss."""
    num = 2 * (p * t).sum() + eps
    den = p.sum() + t.sum() + eps
    return float((num / den).item())


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #
def _psnr(recon, gt, drange):
    mse = ((recon - gt) ** 2).mean().item()
    return 99.0 if mse <= 1e-12 else 10.0 * math.log10((drange ** 2) / mse)


def _ssim(recon, gt, drange, win=7):
    """Windowed SSIM averaged over (C,F,T). recon/gt: (C,F,T) tensors."""
    x = recon.unsqueeze(0); y = gt.unsqueeze(0)       # (1,C,F,T)
    pad = win // 2
    k = (1.0 / (win * win))
    def blur(z):
        return F.avg_pool2d(z, win, stride=1, padding=pad)
    mx, my = blur(x), blur(y)
    mxx, myy, mxy = blur(x * x), blur(y * y), blur(x * y)
    vx, vy, cxy = mxx - mx * mx, myy - my * my, mxy - mx * my
    c1, c2 = (0.01 * drange) ** 2, (0.03 * drange) ** 2
    s = ((2 * mx * my + c1) * (2 * cxy + c2)) / (
        (mx * mx + my * my + c1) * (vx + vy + c2) + 1e-12)
    return s.mean().item()


def _lcr(recon, gt, mask):
    """line-contrast ratio: (recon[line]-recon[bg]) / (gt[line]-gt[bg]).

    mask may be (F,T) [synthetic: one mask broadcast over channels] or (C,F,T)
    [real: per-channel binarized GT]. mode_hard returns float, so cast to bool.
    """
    if mask.dim() < recon.dim():
        mask = mask.unsqueeze(0).expand_as(recon)     # (F,T) -> (C,F,T)
    m = mask.bool()
    bg = ~m
    gt_c = gt[m].mean().item() - gt[bg].mean().item()
    rc = recon[m].mean().item() - recon[bg].mean().item()
    return rc / gt_c if abs(gt_c) > 1e-6 else float("nan")


def _corr(recon, gt):
    a = recon.flatten().float(); b = gt.flatten().float()
    a = a - a.mean(); b = b - b.mean()
    d = (a.norm() * b.norm()).item()
    return (a @ b).item() / d if d > 1e-9 else 0.0


@torch.no_grad()
def evaluate(tok, head, mask_head, X, masks, types, device, mu_f, sd_f, mode_k,
             cond, fsq=None, no_flow=False, n_eval_samples=1, seed=0):
    tok.eval(); head.eval()
    if mask_head is not None:
        mask_head.eval()
    if fsq is not None:
        fsq.eval()
    X = X.to(device)
    mu_f = mu_f.to(device); sd_f = sd_f.to(device)
    tokens = tok._encode(X)
    if fsq is not None:
        tokens, _ = fsq(tokens)                       # discrete FSQ bottleneck
    mu = head.mean_head(tokens)
    torch.manual_seed(seed)
    # no_flow: report mu as the "sample" (the FSQ recon gate is on the mu
    # decode; skipping head.sample avoids the flow U-Net's slow-compiling convs)
    samp = mu.clone() if no_flow else head.sample(tokens, mu)

    # mask-head soft mask (B/C); used for the mask-head Dice and cond gating.
    mask_prob = None
    if mask_head is not None:
        mask_prob = torch.sigmoid(mask_head(tokens))  # (N, C, F, T) in [0, 1]

    # (C) mask-gated residual: keep the flow residual only where the mask head
    # predicts a mode, otherwise fall back to the (coherent-ish) mean. When
    # cond is on we report the gated draw in the "samp" column.
    if cond and mask_prob is not None:
        samp = mu + mask_prob * (samp - mu)

    # TRUE mode masks -> (N,C,F,T) float for Dice. Synthetic masks are (N,F,T)
    # (modes identical across channels) and get broadcast over C; real-data
    # masks are already per-channel (N,C,F,T) and are used as-is.
    C = X.shape[1]
    if masks.dim() == 3:
        masks_d = masks.to(device).unsqueeze(1).expand(-1, C, -1, -1).float()
    else:
        masks_d = masks.to(device).float()

    drange = (X.max() - X.min()).item()
    out = {}                                          # family -> {metric: (mu, sample)}
    for ti, fam in enumerate(FAMILIES):
        idx = (types == ti).nonzero(as_tuple=True)[0]
        rows = {"psnr": [], "ssim": [], "lcr": [], "corr": [], "mdice": []}
        for which, rec in (("mu", mu), ("samp", samp)):
            ps, ss, lc, co, md = [], [], [], [], []
            for i in idx:
                g = X[i].cpu(); r = rec[i].cpu(); mk = masks[i]
                ps.append(_psnr(r, g, drange)); ss.append(_ssim(r, g, drange))
                lc.append(_lcr(r, g, mk)); co.append(_corr(r, g))
                # mode-coherence Dice: predicted hard mode-mask vs TRUE mask.
                ph = mode_hard(rec[i:i + 1], mu_f, sd_f, mode_k)   # (1,C,F,T)
                md.append(dice_sim(ph, masks_d[i:i + 1]))
            rows["psnr"].append(float(np.mean(ps)))
            rows["ssim"].append(float(np.mean(ss)))
            rows["lcr"].append(float(np.nanmean(lc)))
            rows["corr"].append(float(np.mean(co)))
            rows["mdice"].append(float(np.mean(md)))
        out[fam] = rows

    # (validity) does the binarization itself recover the true modes? Dice of
    # the hard mask of the TRUE spectrogram vs the true mask -> same for all.
    out["_target_mdice"] = dice_sim(
        mode_hard(X, mu_f, sd_f, mode_k), masks_d)
    # mask-head Dice (B): thresholded predicted mask vs true mask.
    out["_mask_dice"] = (
        dice_sim((mask_prob > 0.5).float(), masks_d)
        if mask_prob is not None else float("nan"))
    return out, mu.cpu(), samp.cpu()


# --------------------------------------------------------------------------- #
# Figure                                                                       #
# --------------------------------------------------------------------------- #
def save_figure(X, types, recons, out_path):
    """recons: dict label -> (mu (N,C,F,T), samp). One representative row per
    family; columns GT | <variant sample> ... (channel 0, flow sample)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = list(recons.keys())
    cols = ["Ground truth"] + [lab.split()[0] for lab in labels]   # GT + sample/variant
    nrow, ncol = len(FAMILIES), len(cols)
    fig, ax = plt.subplots(nrow, ncol, figsize=(2.4 * ncol, 2.2 * nrow),
                           squeeze=False)
    for ri, fam in enumerate(FAMILIES):
        i = int((types == ri).nonzero(as_tuple=True)[0][0])
        panels = [X[i, 0]] + [recons[lab][1][i, 0] for lab in labels]  # [1]=sample
        vmax = float(X[i, 0].max())
        for ci, p in enumerate(panels):
            a = ax[ri, ci]
            a.imshow(np.asarray(p), aspect="auto", origin="lower",
                     vmin=0.0, vmax=vmax, cmap="magma")
            if ri == 0:
                a.set_title(cols[ci], fontsize=9)
            if ci == 0:
                a.set_ylabel(fam, fontsize=10)
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Thin mode-like pattern reconstruction: encode→decode round-trip "
                 "(equal 24-token budget)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=110)
    fig.savefig(out_path.replace(".png", ".pdf"))
    print(f"[figure] wrote {out_path}", flush=True)


# --------------------------------------------------------------------------- #
def main():
    global FAMILIES, _USE_PROD_BIN
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="eval_runs/spectro_thin_test")
    ap.add_argument("--n_per", type=int, default=8)
    ap.add_argument("--channels", type=int, default=2)
    ap.add_argument("--freq_bins", type=int, default=512)
    ap.add_argument("--time_frames", type=int, default=96)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--base_ch", type=int, default=48)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--flow_steps", type=int, default=12)
    ap.add_argument(
        "--variants",
        default="base:pf=64,pt=32,fpe=16,tpe=8;"
                "A:pf=64,pt=32,fpe=16,tpe=8,struct=1;"
                "B:pf=64,pt=32,fpe=16,tpe=8,mask=1;"
                "AB:pf=64,pt=32,fpe=16,tpe=8,struct=1,mask=1;"
                "ABC:pf=64,pt=32,fpe=16,tpe=8,struct=1,mask=1,cond=1",
        help="';'-separated variant specs name:pf=..,pt=..,fpe=..,tpe=..,"
             "fstem=0/1,istem=0/1,struct=0/1,mask=0/1,cond=0/1 (omitted keys = "
             "OLD baseline 512,4,0,...). struct=A (soft-binarized Dice loss on "
             "mu), mask=B (binary mode-mask head), cond=C (mask-gated residual).",
    )
    ap.add_argument("--fsq_levels", default="8,8,8,5,5,5",
                    help="FSQ per-dim levels (comma list) for variants with "
                         "fsq=1. Codebook size = product. e.g. 8,8,8,5,5,5=64000.")
    ap.add_argument("--no_flow", action="store_true",
                    help="skip the flow U-Net in train AND eval (mu-only "
                         "reconstruction). For the FSQ recon-fidelity gate: "
                         "the flow head's 512xT convs are what stall MIOpen "
                         "compilation, and 'do modes survive quantization' only "
                         "needs the encode->quantize->decode(mu) round trip.")
    ap.add_argument("--mode_k", type=float, default=2.0,
                    help="z-score threshold k for mode binarization.")
    ap.add_argument("--lam_struct", type=float, default=1.0,
                    help="weight of the (A) structural Dice loss.")
    ap.add_argument("--lam_mask", type=float, default=1.0,
                    help="weight of the (B) mask-head BCE loss.")
    ap.add_argument("--seed", type=int, default=0)
    # --- real-data benchmark (single shot, model-normalized spectrograms) ---
    ap.add_argument("--real_shot", type=int, default=0,
                    help="if >0, benchmark on this real shot's spectrograms "
                         "instead of synthetic planted modes (e.g. 200729).")
    ap.add_argument("--real_modality", default="co2",
                    help="spectro modality for --real_shot (ece, co2, bes).")
    ap.add_argument("--data_dir",
                    default="/lustre/orion/fus187/proj-shared/foundation_model")
    ap.add_argument(
        "--stats_path",
        default="/lustre/orion/fus187/proj-shared/foundation_model_meta/"
                "preprocessing_stats.pt")
    ap.add_argument("--n_real_windows", type=int, default=40,
                    help="number of (sub-sampled) windows from the real shot.")
    ap.add_argument("--real_channels", type=int, default=0,
                    help="cap channels for --real_shot (0 = all; ece has 40).")
    ap.add_argument("--real_shots", default="",
                    help="comma-list of shots to POOL into the overfit set "
                         "(overrides --real_shot; e.g. good high-mode shots).")
    # --- random lambda SCAN (overfit recipe-finder): one job, N A-variants at
    # random log-uniform struct-loss weights ---
    ap.add_argument("--lam_scan", type=int, default=0,
                    help="if >0, replace --variants with base + N A-variants at "
                         "RANDOM log-uniform lambda_struct values (the overfit "
                         "lambda scan).")
    ap.add_argument("--lam_range", default="0.3,30",
                    help="lo,hi (log-uniform) for --lam_scan.")
    ap.add_argument("--lam_seed", type=int, default=0,
                    help="RNG seed for the random lambda scan.")
    # --- #1 mode-survival / decorrelation analysis (model-free) ---
    ap.add_argument("--survival", action="store_true",
                    help="run the model-free mode-survival/decorrelation curve "
                         "(predictable-vs-stochastic test) instead of the recon "
                         "benchmark. Uses --data_dir/--stats_path/--out_dir.")
    ap.add_argument("--survival_modalities", default="ece,co2",
                    help="comma-list of spectro modalities for --survival.")
    ap.add_argument("--survival_shots", type=int, default=12,
                    help="number of shots to pool for --survival (incl --real_shot).")
    ap.add_argument("--survival_max_lag", type=int, default=400,
                    help="max time-gap lag in FRAMES (~0.51 ms/frame; 400≈205ms).")
    ap.add_argument("--survival_max_windows", type=int, default=0,
                    help="cap non-overlapping windows per shot (0 = whole shot).")
    ap.add_argument("--survival_freq_tols", default="0,8,24",
                    help="comma-list of FREQUENCY tolerances in bins for the "
                         "survival mask (max-pool ±tol over freq). 0 = strict "
                         "exact-pixel; >0 = band/drift-tolerant mode-presence "
                         "(the predictability the model can actually use). "
                         "~0.49 kHz/bin, so 8≈±4kHz, 24≈±12kHz.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    if args.survival:                       # #1 model-free predictability test
        run_survival(args, device)
        return
    print(f"[setup] device={device} d_model={args.d_model} base_ch={args.base_ch} "
          f"steps={args.steps} patterns={FAMILIES} n_per={args.n_per}", flush=True)

    if args.real_shot or args.real_shots:
        shots = ([int(s) for s in args.real_shots.split(",") if s.strip()]
                 if args.real_shots else [args.real_shot])
        Xs = []
        for sh in shots:
            Xi, _ = load_real_shot_spectro(
                sh, args.data_dir, args.stats_path, args.real_modality,
                args.n_real_windows, args.real_channels)
            Xs.append(Xi)
        X = torch.cat(Xs, dim=0)                       # pool windows across shots
        types = torch.zeros(X.shape[0], dtype=torch.long)
        # adopt the real tensor's geometry so the heads are built to match.
        args.channels, args.freq_bins, args.time_frames = (
            X.shape[1], X.shape[2], X.shape[3])
        FAMILIES = [f"real:{args.real_modality}"]      # single "family"
        masks = None                                   # filled in after mu_f/sd_f
        # Use the EXACT production GT-fusion binarization + per-modality k, so
        # "modes" here are the ones the overlay successfully extracts on 200729
        # (the shot-local z>2 rule found ZERO modes on real co2 -> mDice=1.0
        # artifact). Self-normalizing, so OK on the model-normalized X.
        _USE_PROD_BIN = True
        _k_by_mod = {"ece": 2.5, "co2": 2.0, "bes": 2.0}
        args.mode_k = _k_by_mod.get(args.real_modality, 2.0)
        print(f"[data] REAL shots {shots}/{args.real_modality}  "
              f"X={tuple(X.shape)}  prod_binarize=ON k={args.mode_k}", flush=True)
    else:
        X, masks, types = make_dataset(
            args.n_per, args.channels, args.freq_bins, args.time_frames,
            args.seed)
        print(f"[data] X={tuple(X.shape)}  ({len(FAMILIES)} families x "
              f"{args.n_per})", flush=True)

    # per-(channel, freq) background stats for the mode binarization, computed
    # ONCE from the training data and shared by every variant's train + eval.
    mu_f = X.mean(dim=(0, 3))                          # (C, F)
    sd_f = X.std(dim=(0, 3)).clamp_min(0.05)          # (C, F)
    if args.real_shot or args.real_shots:
        # no planted ground-truth on real data -> the "true" modes ARE the
        # binarized GT (per-channel). Validity ceiling is then trivially ~1.0;
        # the signal is how close pred's binarized modes get to GT's.
        masks = mode_hard(X, mu_f, sd_f, args.mode_k)  # (N, C, F, T)
    print(f"[binarize] mode_k={args.mode_k} lam_struct={args.lam_struct} "
          f"lam_mask={args.lam_mask}  mu_f mean={mu_f.mean().item():.3f} "
          f"sd_f mean={sd_f.mean().item():.3f}", flush=True)

    if args.lam_scan > 0:
        lo, hi = [float(x) for x in args.lam_range.split(",")]
        rng = np.random.default_rng(args.lam_seed)
        lams = sorted(float(x) for x in
                      np.exp(rng.uniform(np.log(lo), np.log(hi), args.lam_scan)))

        def _mk(name, **extra):
            d = dict(_VAR_DEFAULTS); d["name"] = name
            d.update({"pf": 64, "pt": 32, "fpe": 16, "tpe": 8}); d.update(extra)
            return d
        specs = [_mk("base")] + [_mk(f"A_lam{l:.2f}", struct=1, lam=l) for l in lams]
        print(f"[lam-scan] {args.lam_scan} random lambda in [{lo},{hi}]: "
              f"{[round(l, 3) for l in lams]}", flush=True)
    else:
        specs = parse_variants(args.variants)
    print(f"[variants] {[s['name'] for s in specs]}", flush=True)
    _fsq_levels = [int(x) for x in args.fsq_levels.split(",") if x.strip()]
    results, recons, order = {}, {}, []
    # Incremental results file: one row appended per variant the instant it is
    # evaluated, so a walltime cut never discards finished variants.
    _res_path = os.path.join(args.out_dir, "abc_results.txt")
    with open(_res_path, "w") as _fh:
        _fh.write("variant | mDice mu/samp | SSIM mu/samp | maskDice\n")
    for spec in specs:
        struct = bool(spec.get("struct", 0))
        mask = bool(spec.get("mask", 0))
        cond = bool(spec.get("cond", 0))
        tok, head, mask_head, fsq, label = build_variant(
            spec, args.channels, args.freq_bins, args.time_frames,
            args.d_model, args.base_ch, args.flow_steps,
            fsq_levels=_fsq_levels)
        tok.to(device); head.to(device)
        if mask_head is not None:
            mask_head.to(device)
        if fsq is not None:
            fsq.to(device)
        np_ = sum(p.numel() for p in tok.parameters()) + \
            sum(p.numel() for p in head.parameters())
        if mask_head is not None:
            np_ += sum(p.numel() for p in mask_head.parameters())
        if fsq is not None:
            np_ += sum(p.numel() for p in fsq.parameters())
        print(f"\n=== {label}  ({np_/1e6:.2f}M params) ===", flush=True)
        lam_v = spec.get("lam")
        lam_struct_v = float(lam_v) if lam_v is not None else args.lam_struct
        train(tok, head, mask_head, X, args.steps, args.lr, device,
              mu_f, sd_f, args.mode_k, lam_struct_v, args.lam_mask,
              struct, mask, cond, fsq=fsq, no_flow=args.no_flow,
              tag=spec["name"])
        res, mu, sp = evaluate(
            tok, head, mask_head, X, masks, types, device,
            mu_f, sd_f, args.mode_k, cond, fsq=fsq, no_flow=args.no_flow,
            seed=args.seed)
        results[label] = res; recons[label] = (mu, sp); order.append(label)
        # incremental per-variant row (survives a walltime cut)
        _mdmu = float(np.mean([res[f]["mdice"][0] for f in FAMILIES]))
        _mdsp = float(np.mean([res[f]["mdice"][1] for f in FAMILIES]))
        _ssmu = float(np.mean([res[f]["ssim"][0] for f in FAMILIES]))   # SSIM mu (fidelity)
        _ssp = float(np.mean([res[f]["ssim"][1] for f in FAMILIES]))
        _mh = res.get("_mask_dice")
        _ln = (f"{label:<30} | mDice mu/samp {_mdmu:.3f}/{_mdsp:.3f} | "
               f"SSIM mu/samp {_ssmu:.3f}/{_ssp:.3f} | maskDice "
               f"{('n/a' if _mh is None else format(_mh, '.3f'))}")
        print("  [variant-done] " + _ln, flush=True)
        with open(_res_path, "a") as _fh:
            _fh.write(_ln + "\n")

    def mm(lab, metric, which):    # mean over families; which 0=mu, 1=sample
        return float(np.mean([results[lab][f][metric][which] for f in FAMILIES]))

    # ---- objective table: mu (deterministic decode) AND flow sample ----
    print("\n" + "=" * 116)
    print("OBJECTIVE RECONSTRUCTION RESULTS  (mu = deterministic decode | samp = flow sample)")
    print("=" * 116)
    print(f"{'config':<48} | {'SSIM mu/samp':>15} | {'LCR mu/samp':>15} | "
          f"{'mDice mu/samp':>15} | {'PSNR samp':>9}")
    print("-" * 116)
    for lab in order:
        print(f"{lab:<48} | {mm(lab,'ssim',0):>6.3f}/{mm(lab,'ssim',1):<8.3f} | "
              f"{mm(lab,'lcr',0):>6.3f}/{mm(lab,'lcr',1):<8.3f} | "
              f"{mm(lab,'mdice',0):>6.3f}/{mm(lab,'mdice',1):<8.3f} | "
              f"{mm(lab,'psnr',1):>9.2f}")
    print("-" * 116)
    print("\nper-family flow-SAMPLE  ssim|mDice:")
    print(f"{'family':<14}" + "".join(f"{lab.split()[0]+' '+lab.split()[1]:>22}" for lab in order))
    for fam in FAMILIES:
        cells = "".join(
            f"{results[lab][fam]['ssim'][1]:>10.3f}|{results[lab][fam]['mdice'][1]:<11.3f}"
            for lab in order)
        print(f"{fam:<14}{cells}")
    print("-" * 116)
    # binarization validity + mask-head Dice (special non-family keys).
    tgt_md = results[order[0]]["_target_mdice"]    # same for all variants
    print(f"binarization validity: hard-mask(GT) vs true mask = Dice {tgt_md:.3f} "
          f"(mode_k={args.mode_k}; ~1 => the threshold recovers the true modes)")
    print(f"{'config':<48} | {'mask-head Dice (B)':>20}")
    for lab in order:
        md = results[lab]["_mask_dice"]
        cell = "n/a" if md != md else f"{md:.3f}"   # NaN -> n/a (no mask head)
        print(f"{lab:<48} | {cell:>20}")
    print("=" * 116)

    save_figure(X, types, recons, os.path.join(args.out_dir, "thin_pattern_recon.png"))

    # ---- verdict ---- (baseline = a variant named 'old' or 'base'; else first)
    base_labs = [l for l in order
                 if l.split()[0].lower().startswith(("old", "base"))]
    cand_labs = [l for l in order if l not in base_labs]
    pool = cand_labs or order
    best = max(pool, key=lambda l: mm(l, "ssim", 1))
    ss_n, lc_n = mm(best, "ssim", 1), mm(best, "lcr", 1)
    msg = f"[VERDICT] best = {best.split()[0]}: sample SSIM={ss_n:.3f} LCR={lc_n:.3f}"
    if base_labs:
        ss_o, lc_o = mm(base_labs[0], "ssim", 1), mm(base_labs[0], "lcr", 1)
        msg += (f"  |  baseline SSIM={ss_o:.3f} LCR={lc_o:.3f}  "
                f"(Δssim {ss_n-ss_o:+.3f})")
    print(msg, flush=True)
    q = ("HIGH-QUALITY" if (ss_n >= 0.80 and lc_n >= 0.70)
         else "GOOD" if ss_n >= 0.60 else "LOW")
    print(f"[VERDICT] reconstruction quality: {q} (target SSIM>=0.80, LCR>=0.70). "
          f"High = thin mode-like patterns RETAINED by the encoder/decoder.",
          flush=True)

    # ---- coherence verdict: the key question of the A/B/C experiment ----
    best_md = max(order, key=lambda l: mm(l, "mdice", 1))
    print(f"[VERDICT/coherence] best sample mode-Dice = {best_md.split()[0]} "
          f"({mm(best_md,'mdice',1):.3f}); validity ceiling = {tgt_md:.3f}.",
          flush=True)
    if base_labs:
        bl = base_labs[0]
        # A's effect on the DETERMINISTIC mean = the central question.
        a_labs = [l for l in order if "+A" in l and l not in base_labs]
        if a_labs:
            a0 = a_labs[0]
            d_mu = mm(a0, "mdice", 0) - mm(bl, "mdice", 0)
            verdict = ("YES" if d_mu > 0.02 else "NO")
            print(f"[VERDICT/coherence] structural loss (A) effect on the "
                  f"DETERMINISTIC mean mode-Dice: {mm(bl,'mdice',0):.3f} -> "
                  f"{mm(a0,'mdice',0):.3f} (Δ {d_mu:+.3f}) => makes mu coherent? "
                  f"{verdict}.", flush=True)


if __name__ == "__main__":
    main()
