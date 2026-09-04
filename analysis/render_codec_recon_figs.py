#!/usr/bin/env python
"""Render REAL GT-vs-reconstruction figures for IGNITE codecs (CPU, judge-by-eye).

Loads each codec's saved best-ckpt, rebuilds the codec from the cfg stored in the ckpt
(mirrors ignite spike._save_best_checkpoint: {"codec": state_dict, "cfg": <dataclass>, ...}),
picks a mode-rich REAL window from shot 200729 via the codec's OWN dataset class
(CodecPairDataset / SlowTSCodecPairDataset), runs encode->quantize->decode, and plots GT vs
reconstruction in the SAME space the codec operates in (log-power for spectro; standardized
profile for slow-TS). Both panels come from the codec's own input/output tensors, so there is
NO denorm/scale mismatch.

Output PNGs (overwritten in place):
    eval_runs/codec_recon_figs/spectro_ece_24v192.png
    eval_runs/codec_recon_figs/spectro_bes_24v192.png   (if quick)
    eval_runs/codec_recon_figs/spectro_mhr_24v192.png   (if quick)
    eval_runs/codec_recon_figs/slowts_ts_core_density_1v4.png

AUDIT mode (``--audit``) — the quantitative companion to the judge-by-eye figures. Streams
HELD-OUT windows (the last ``--eval_n_shots`` shots, i.e. the same disjoint split
``train_codec`` uses) through one or more spectro checkpoints and prints, per checkpoint:

    lattice / gt_lattice   gate.patch_lattice_metrics patch_lattice_ratio, recon and the
                           ground-truth CONTROL. 1.0 = no patch grid; the mhr baseline reads
                           61.6 against a GT control of 1.15.
    lat_frac               share of the recon's 2-D spectral energy sitting on the patch
                           lattice (GT ~0.03).
    seam_f / seam_t        locally-normalized boundary seam ratios (diagnostic).
    hf_ratio               gate.decode_fidelity sharpness = HF gradient energy recon/GT. This
                           is the number a "fix" that merely BLURS would destroy, so it is
                           always printed beside the lattice ratio.
    env_corr / peak_f1     the existing gate reconstruction metrics. Both are computed on the
                           TIME-COLLAPSED power envelope (spec.mean(axis=-1)), so neither can
                           see a missing mode track.
    spec_nrmse / corr2d    gate.full_spectro_metrics: RMSE/std and Pearson correlation over
                           the FULL (F, T) array per (window, channel) — nothing collapsed —
                           plus the same two restricted to the 0-``--band_bins`` mode band.
                           Printed in a second table with the trivial baselines of
                           gate.trivial_spectro_baselines (self / time-mean envelope /
                           per-(C,F) batch mean / per-window mean) in the SAME units, because
                           a reconstruction number means nothing without them.
    codes / util           distinct FSQ codes and effective codes over ALL audited tokens.

    python analysis/render_codec_recon_figs.py --audit \
        base=/path/ms2_s1/codec_best.pt ref2=/path/ref2_s1/codec_best.pt \
        --modality mhr --n_windows 720 --fig eval_runs/codec_recon_figs/mhr_lattice.png
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from tokamak_foundation_model.ignite import train_codec as tc
from tokamak_foundation_model.ignite import gate as gate_mod
from tokamak_foundation_model.ignite import spike as spike_mod
from tokamak_foundation_model.ignite.codec import SpectroCodec
from tokamak_foundation_model.ignite.slow_ts_codec import SlowTSCodec

REPO = Path("/lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub")
OUT = REPO / "eval_runs" / "codec_recon_figs"
OUT.mkdir(parents=True, exist_ok=True)
DATA_DIR = tc.DEFAULT_DATA_DIR
SHOT = "200729"
# local temp caches so we do NOT clobber the shared multi-shot lengths caches under
# foundation_model_meta (single-shot cache would poison the shared ones).
CACHE_DIR = OUT / "_cache"
CACHE_DIR.mkdir(exist_ok=True)

torch.manual_seed(0)


def _cache(modality: str) -> str:
    return str(CACHE_DIR / f"codec_{modality}_200729_lengths.pt")


# --------------------------------------------------------------------------------------- #
# SPECTRO
# --------------------------------------------------------------------------------------- #
def load_spectro_codec(ckpt_path: Path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    codec = SpectroCodec(cfg)
    codec.load_state_dict(ck["codec"])
    codec.eval()
    return codec, cfg, ck


def pick_mode_rich_spectro(modality: str, cfg, n_scan: int = 60):
    """Scan windows of shot 200729, return the highest-std (most structured) spec_a AND the
    index of the single most mode-rich channel (highest per-freq-envelope std) in that window,
    so the figure shows a REAL coherent-mode channel, not the mode-diluting 40-channel mean."""
    cfg_local = cfg  # cfg carries channels already; dataset does not mutate it
    ds = tc.CodecPairDataset(
        modality, [SHOT], cfg_local, data_dir=DATA_DIR,
        lengths_cache_path=_cache(modality),
    )
    n = len(ds)
    # evenly sample n_scan windows across the shot
    idxs = np.linspace(0, n - 1, min(n_scan, n)).astype(int)
    best = None
    best_std = -1.0
    for i in idxs:
        a, _b = ds[int(i)]
        s = float(a.std())
        if s > best_std:
            best_std, best = s, (int(i), a)
    idx, spec_a = best
    # most mode-rich channel = the one whose time-averaged per-freq power has the most spread
    # (a strong ridge above the broadband floor). Envelope = mean over time per (C,F).
    env = spec_a.mean(dim=2)               # (C, F)
    ch_std = env.std(dim=1)                # (C,)
    ch = int(torch.argmax(ch_std).item())
    return idx, spec_a, best_std, n, ch


@torch.no_grad()
def spectro_reconstruct(codec, spec_a):
    x = spec_a.unsqueeze(0)  # (1, C, F, T)
    out = codec(x)
    recon = out["recon"][0]  # (C, F, T)
    return recon


def spectro_env_corr(recon, target):
    """gate._envelope_correlation on this single window (B=1)."""
    r = recon.unsqueeze(0).numpy()
    t = target.unsqueeze(0).numpy()
    return gate_mod._envelope_correlation(r, t)


def _render_spectro_row(fig, axes_row, rel, label, modality):
    """One [GT | RECON | diff] row for a spectro codec, on the single most mode-rich channel."""
    codec, cfg, ck = load_spectro_codec(REPO / rel)
    idx, spec_a, std, n, ch = pick_mode_rich_spectro(modality, cfg)
    recon = spectro_reconstruct(codec, spec_a)
    ec = spectro_env_corr(recon, spec_a)          # env_corr over ALL channels (the gate metric)
    gt = spec_a[ch].numpy()                        # (F, T) single mode-rich channel
    rc = recon[ch].numpy()
    df = rc - gt
    # SHARED color scale for GT + RECON, robustly set from the GT distribution ([2,98] pct) so
    # (a) the two panels are directly comparable and (b) the scale is NOT pulled around by a
    # recon outlier. Ridges (modes) then read the SAME way in both panels.
    vmin = float(np.percentile(gt, 2))
    vmax = float(np.percentile(gt, 98))
    dmax = float(np.percentile(np.abs(df), 98)) or 1.0
    _score = ck.get("score")
    _score_s = f"{_score:.3f}" if isinstance(_score, (int, float)) else str(_score)
    titles = [
        f"{modality} GT | {label}\nwin {idx}/{n} ch{ch}, win-std={std:.2f}, n_tok={cfg.n_tok}",
        f"{modality} RECON | env_corr={ec:.3f}\nstep={ck.get('step')}, score={_score_s}",
        "RECON - GT (diff)",
    ]
    for c, (arr, title, cmap, vlo, vhi) in enumerate([
        (gt, titles[0], "magma", vmin, vmax),
        (rc, titles[1], "magma", vmin, vmax),
        (df, titles[2], "RdBu_r", -dmax, dmax),
    ]):
        ax = axes_row[c]
        im = ax.imshow(arr, aspect="auto", origin="lower", cmap=cmap,
                       vmin=vlo, vmax=vhi, extent=[0, arr.shape[1], 0, arr.shape[0]])
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("time frame")
        if c == 0:
            ax.set_ylabel("freq bin")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def render_spectro_ece():
    """ece: one figure, row per token-count (24 converged | 192 under-converged)."""
    rows = [
        ("eval_runs/ignite_d2_ece/codec_best.pt", "24 tok (CONVERGED)"),
        ("eval_runs/ignite_d5_ece/codec_best.pt", "192 tok (UNDER-converged)"),
    ]
    fig, axes = plt.subplots(len(rows), 3, figsize=(13, 4.2 * len(rows)))
    if len(rows) == 1:
        axes = axes[None, :]
    for r, (rel, label) in enumerate(rows):
        _render_spectro_row(fig, axes[r], rel, label, "ece")
    fig.suptitle("IGNITE ece spectro codec — GT vs reconstruction "
                 "(log-power, single mode-rich channel, shot 200729)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = OUT / "spectro_ece_24v192.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"WROTE {p}")


def render_spectro_generic(modality: str):
    """bes / mhr: d2 (24 tok) vs d5 (192 tok) in one figure."""
    rows = [
        (f"eval_runs/ignite_d2_{modality}/codec_best.pt", "24 tok (d2)"),
        (f"eval_runs/ignite_d5_{modality}/codec_best.pt", "192 tok (d5)"),
    ]
    rows = [(rel, lab) for rel, lab in rows if (REPO / rel).exists()]
    if not rows:
        print(f"SKIP {modality}: no ckpt found")
        return
    fig, axes = plt.subplots(len(rows), 3, figsize=(13, 4.2 * len(rows)))
    if len(rows) == 1:
        axes = axes[None, :]
    for r, (rel, label) in enumerate(rows):
        _render_spectro_row(fig, axes[r], rel, label, modality)
    fig.suptitle(f"IGNITE {modality} spectro codec — GT vs reconstruction "
                 f"(log-power, single mode-rich channel, shot 200729)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = OUT / f"spectro_{modality}_24v192.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"WROTE {p}")


# --------------------------------------------------------------------------------------- #
# SLOW-TS
# --------------------------------------------------------------------------------------- #
def load_slowts_codec(ckpt_path: Path):
    """Rebuild the slow-TS codec from the ckpt, RECONCILING the stored cfg with the actual
    saved weights (ground-truth-from-artifact). Some ckpts carry a stale n_zones/n_tok in the
    cfg dataclass while the trained weights encode a DIFFERENT token count — the weights are
    authoritative. We read the true token count from ``encoder.pos_emb.pos_pe`` (shape
    (n_pos_patch, d_model)) and, if it disagrees with the cfg, rebuild the cfg so the net
    matches the weights (n_zones = true tokens, patch_c = ceil(channels / n_zones))."""
    import dataclasses
    import math

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    sd = ck["codec"]
    true_pos = int(sd["encoder.pos_emb.pos_pe"].shape[0])  # true position tokens in the WEIGHTS
    stored_pos = int(getattr(cfg, "n_pos_patch", cfg.n_zones))
    if true_pos != stored_pos:
        # Reconcile: the weights are the ground truth. Rebuild cfg with n_zones == true tokens.
        patch_c = math.ceil(cfg.channels / true_pos)
        cfg = dataclasses.replace(cfg, n_zones=true_pos, patch_c=patch_c)
    codec = SlowTSCodec(cfg)
    codec.load_state_dict(ck["codec"])
    codec.eval()
    return codec, cfg, ck


def pick_present_slowts(signal: str, cfg, n_scan: int = 80):
    """Scan windows, return the one with the most present samples AND real spread (structure)."""
    ds = tc.SlowTSCodecPairDataset(
        signal, [SHOT], cfg, data_dir=DATA_DIR,
        lengths_cache_path=_cache(signal),
    )
    n = len(ds)
    idxs = np.linspace(0, n - 1, min(n_scan, n)).astype(int)
    best = None
    best_score = -1.0
    for i in idxs:
        sig, mask = ds[int(i)]
        pres = float(mask.mean())
        if pres < 0.5:
            continue
        # spread over present positions (structure, not flat)
        m = mask > 0.5
        vals = sig[m]
        spread = float(vals.std()) if vals.numel() > 1 else 0.0
        sc = pres * spread
        if sc > best_score:
            best_score = sc
            best = (int(i), sig, mask, pres, spread)
    if best is None:  # fallback: max presence
        for i in idxs:
            sig, mask = ds[int(i)]
            pres = float(mask.mean())
            if pres > best_score:
                best_score, best = pres, (int(i), sig, mask, pres, 0.0)
    return best, n


@torch.no_grad()
def slowts_reconstruct(codec, sig):
    x = sig.unsqueeze(0)  # (1, C, T)
    out = codec(x)
    return out["recon"][0]  # (C, T)


def slowts_env_corr(recon, target, mask):
    r = recon.unsqueeze(0).numpy()
    t = target.unsqueeze(0).numpy()
    m = mask.unsqueeze(0).numpy()
    d = gate_mod.slowts_decode_fidelity(r, t, m)
    return d.get("envelope_corr", float("nan"))


def render_slowts(signal: str = "ts_core_density", variants=None, out_name: str = None,
                  subtitle: str = None):
    """Slow-TS GT-vs-recon panels, one per (ckpt, label) variant, on the SAME window.

    Default = the historical ts_core_density d4-vs-d5 comparison. Pass ``variants`` as
    [(ckpt_rel_path, label), ...] to render any slow-TS signal (all variants must share the
    dataset geometry: same channels/time_steps). Line plot of the standardized profile
    (present channels) vs reconstruction, time-averaged over the 50 ms window.
    """
    if variants is None:
        variants = [
            ("eval_runs/ignite_d4_ts_core_density/codec_best.pt", "d4 (1-token)"),
            ("eval_runs/ignite_d5_ts_core_density/codec_best.pt", "d5 (4-zone)"),
        ]
        out_name = out_name or "slowts_ts_core_density_1v4.png"
        subtitle = subtitle or ("blue dashed = radial-zone token boundaries; d4 = whole "
                                "profile in 1 token, d5 = 4 radial zones")
    out_name = out_name or f"slowts_{signal}.png"
    subtitle = subtitle or "blue dashed = radial-zone token boundaries"
    # Pick ONE mode-rich window with the LAST variant's cfg, then evaluate ALL variants on the
    # SAME window (fair comparison; identical windowing given shared geometry).
    _c0, cfg0, _ck0 = load_slowts_codec(REPO / variants[-1][0])
    (idx0, _sig0, _m0, pres0, _sp0), n = pick_present_slowts(signal, cfg0)

    fig, axes = plt.subplots(1, len(variants), figsize=(7.0 * len(variants), 5.2), squeeze=False)
    axes = axes[0]
    for j, (rel, tag) in enumerate(variants):
        codec, cfg, ck = load_slowts_codec(REPO / rel)
        ds = tc.SlowTSCodecPairDataset(
            signal, [SHOT], cfg, data_dir=DATA_DIR,
            lengths_cache_path=_cache(signal),
        )
        sig, mask = ds[idx0]                      # SAME window index for both variants
        recon = slowts_reconstruct(codec, sig)
        C = cfg.channels
        # time-average over the 50 ms window (profile is quasi-static) for a clean profile plot,
        # weighting by the validity mask so missing samples don't drag the average.
        msk = (mask[:C] > 0.5).float()            # (C, T)
        wsum = msk.sum(1).clamp_min(1.0)
        gt = ((sig[:C] * msk).sum(1) / wsum).numpy()
        rc = ((recon[:C] * msk).sum(1) / wsum).numpy()
        present = msk.sum(1).numpy() > 0.5        # channel present in >=1 sample
        # honest scalar metrics on this window (masked, over present samples)
        m_all = mask[:C] > 0.5
        mae = float((sig[:C][m_all] - recon[:C][m_all]).abs().mean())
        pc = float(np.corrcoef(gt[present], rc[present])[0, 1]) if present.sum() > 2 else float("nan")
        x = np.arange(C)
        ax = axes[j]
        ax.plot(x[present], gt[present], "-o", color="k", ms=3, lw=1.4, label="GT (standardized)")
        ax.plot(x[present], rc[present], "-s", color="tab:red", ms=3, lw=1.4, label="reconstruction")
        if (~present).any():
            ax.scatter(x[~present], gt[~present], marker="x", color="gray", s=40, zorder=5,
                       label="missing (excluded)")
        for z in range(1, cfg.n_zones):
            b = z * cfg.patch_c
            if b < C:
                ax.axvline(b, color="tab:blue", ls="--", lw=0.8, alpha=0.6)
        score = ck.get("score")
        ax.set_title(
            f"{signal} {tag}: n_tok={cfg.n_zones}, patch_c={cfg.patch_c} "
            f"(padded_ch={cfg.padded_channels})\n"
            f"win {idx0}/{n}, present={pres0:.2f} | masked MAE={mae:.3f}, "
            f"profile-corr={pc:.3f}\nstep={ck.get('step')}"
            + (f", score={score:.3f}" if isinstance(score, (int, float)) else ""),
            fontsize=9)
        ax.set_xlabel("channel index (radial position)")
        ax.set_ylabel("standardized value")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)
    fig.suptitle(f"IGNITE slow-TS {signal} codec — GT vs reconstruction "
                 "(standardized profile, time-averaged over the 50 ms window, shot 200729)\n"
                 + subtitle, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = OUT / out_name
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"WROTE {p}")


# --------------------------------------------------------------------------------------- #
# VIDEO (tangtv per-divertor codecs)
# --------------------------------------------------------------------------------------- #
def load_video_codec(ckpt_path: Path):
    from tokamak_foundation_model.ignite.video_codec import VideoCodec
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    codec = VideoCodec(cfg)
    codec.load_state_dict(ck["codec"])
    codec.eval()
    return codec, cfg, ck


def pick_bright_clip(modality: str, cfg, n_scan: int = 40):
    """Scan clips of shot 200729, return the highest-std (most structured) one."""
    ds = tc.VideoCodecPairDataset(
        modality, [SHOT], cfg, data_dir=DATA_DIR,
        lengths_cache_path=_cache(modality),
    )
    n = len(ds)
    idxs = np.linspace(0, n - 1, min(n_scan, n)).astype(int)
    best, best_std = None, -1.0
    for i in idxs:
        clip, _mask = ds[int(i)]
        s = float(clip.std())
        if s > best_std:
            best_std, best = s, (int(i), clip)
    return best, n


def render_video(modality: str, variants, out_name: str, subtitle: str = ""):
    """Mid-frame [GT | recon-per-variant] image panels per camera channel, in the codec's
    OWN standardized space (``VideoCodec.standardize_input`` is a per-(B,C) static z-score,
    so the GT panel is identical for every variant — no denorm/scale mismatch). The
    checkerboard verdict figure: the linear-unpatchify baseline's 20x20 patch seams vs the
    conv-refinement decoder, on the SAME clip with the SAME color scale."""
    from tokamak_foundation_model.ignite.video_codec import VideoCodec
    _c0, cfg0, _ck0 = load_video_codec(REPO / variants[-1][0])
    (idx0, clip), n = pick_bright_clip(modality, cfg0)
    x = clip.unsqueeze(0)                                # (1, C, T, H, W) raw pixels
    gt_std = VideoCodec.standardize_input(x)[0]          # (C, T, H, W) standardized GT
    tmid = cfg0.frames // 2
    C = cfg0.channels
    recons = []
    for rel, tag in variants:
        codec, cfg, ck = load_video_codec(REPO / rel)
        with torch.no_grad():
            rc = codec(x)["recon"][0]                    # standardized space (matches gt_std)
        recons.append((tag, rc, ck))
    ncols = 1 + len(variants)
    fig, axes = plt.subplots(C, ncols, figsize=(4.8 * ncols, 2.3 * C + 1.8), squeeze=False)
    for c in range(C):
        g = gt_std[c, tmid].numpy()
        vmin, vmax = np.percentile(g, [1, 99])
        ax = axes[c][0]
        ax.imshow(g, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"GT (standardized), cam-ch {c}", fontsize=9)
        ax.set_axis_off()
        for j, (tag, rc, ck) in enumerate(recons):
            r = rc[c, tmid].numpy()
            pc = float(np.corrcoef(g.ravel(), r.ravel())[0, 1])
            ax = axes[c][j + 1]
            ax.imshow(r, cmap="viridis", vmin=vmin, vmax=vmax)
            score = ck.get("score")
            ax.set_title(
                f"{tag}\nstep={ck.get('step')}"
                + (f", score={score:.2f}" if isinstance(score, (int, float)) else "")
                + f" | frame-corr={pc:.3f}",
                fontsize=8)
            ax.set_axis_off()
    fig.suptitle(f"IGNITE {modality} codec — GT vs reconstruction "
                 f"(mid frame of clip {idx0}/{n}, shot {SHOT}, standardized space, "
                 f"shared color scale)\n{subtitle}", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    p = OUT / out_name
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"WROTE {p}")


# --------------------------------------------------------------------------------------- #
# AUDIT — quantitative patch-lattice / reconstruction report on HELD-OUT windows
# --------------------------------------------------------------------------------------- #
def _audit_stream(codec, cfg, modality: str, shots, n_windows: int, batch_size: int,
                  num_workers: int, cache_tag: str, notch: bool = False,
                  band_bins: int = gate_mod.DEFAULT_MODE_BAND_BINS, device: str = "cpu",
                  want_baseline: bool = True, detrended: bool = False):
    """Encode+decode ``n_windows`` held-out windows, accumulating metrics PER BATCH.

    Streaming (rather than stacking 720 x (6, 512, 96) float64 arrays and metering them once)
    keeps peak memory at one batch: the pooled form needed ~10 GB per FFT and thrashed. The
    per-batch mean is also exactly what ``spike.compute_gate`` reports, so audit numbers are
    directly comparable to the arms' ``gate_*.json``. Only the codes are kept in full (they are
    the utilization statistic) plus ONE batch of (gt, recon) for the figure.

    ``full_spec=True`` adds gate.full_spectro_metrics (spec_nrmse / spec_corr2d, full band and
    the 0-``band_bins`` mode band) and gate.trivial_spectro_baselines to every batch dict. NOTE
    the ``base_cfmean_*`` baseline's "global" per-(C, F) mean is taken WITHIN each batch (the
    streaming unit), so it is a ``batch_size``-window mean, not a dataset mean; the other three
    baselines (self / tmean / wcmean) are per-window and unaffected by the batching.
    """
    from torch.utils.data import DataLoader, Subset

    ds = tc.CodecPairDataset(
        modality, shots, cfg, data_dir=DATA_DIR,
        lengths_cache_path=str(CACHE_DIR / f"codec_{modality}_{cache_tag}_lengths.pt"),
    )
    n = len(ds)
    idx = np.linspace(0, n - 1, min(n_windows, n)).astype(int).tolist()
    dl = DataLoader(Subset(ds, idx), batch_size=batch_size, num_workers=num_workers,
                    shuffle=False)
    acc: dict = {}
    codes, n_used, keep = [], 0, None
    # ``device`` defaults to "cpu" (the prior behaviour, byte-identical). The paper-sized conv
    # decoder is ~39M params and is impractically slow on CPU for 720 windows, so --device cuda
    # moves BOTH the codec and each batch; every metric is still computed on numpy, on the host.
    codec = codec.to(device)
    with torch.no_grad():
        for bi, (spec_a, _spec_b) in enumerate(dl):
            spec_a = spec_a.to(device)
            out = codec(spec_a)
            rec = out["recon"].cpu().numpy()
            gt = spec_a.cpu().numpy()
            if notch:
                rec = gate_mod.remove_patch_lattice(rec, cfg.patch_f, cfg.patch_t)
            # mode_baseline only on the FIRST batch of the FIRST arm: the tmean bar depends
            # only on the target, so computing it per arm per batch doubled the metric cost.
            dm = gate_mod.decode_fidelity(rec, gt, patch_f=cfg.patch_f, patch_t=cfg.patch_t,
                                          full_spec=True, band_bins=band_bins,
                                          mode_baseline=want_baseline,
                                          detrended=detrended)
            for k, v in dm.items():
                acc.setdefault(k, []).append(float(v))
            codes.append(out["codes"].cpu().numpy())
            n_used += gt.shape[0]
            if bi == 0:
                keep = (gt.copy(), rec.copy())
    dec = {k: float(np.mean(v)) for k, v in acc.items()}
    return dec, np.concatenate(codes), keep, n, n_used


def run_audit(specs, modality: str, n_windows: int, eval_n_shots: int, n_eval_shots_used: int,
              batch_size: int, num_workers: int, fig_path=None, notch: bool = False,
              band_bins: int = gate_mod.DEFAULT_MODE_BAND_BINS, device: str = "cpu",
              detrended: bool = False):
    """Print the patch-lattice + reconstruction table for each ``label=ckpt`` spec.

    The shot split mirrors ``train_codec.main``: ``eval_shots = discover_shots()[-eval_n_shots:]``
    — genuinely held out from every arm's training set — of which the first
    ``n_eval_shots_used`` are streamed (each mhr shot yields ~220 windows x 192 tokens).
    """
    all_shots = spike_mod.discover_shots(DATA_DIR)
    eval_shots = all_shots[-eval_n_shots:][:n_eval_shots_used]
    print(f"[audit] modality={modality} held-out shots={eval_shots} "
          f"(last {eval_n_shots} of {len(all_shots)})")
    hdr = (f"{'arm':<14}{'step':>8}{'grid':>17}{'lattice':>9}{'gt_lat':>8}{'lat_frac':>9}"
           f"{'seam_f':>8}{'seam_t':>8}{'hf_ratio':>9}{'env_corr':>9}{'peak_f1':>8}"
           f"{'codes':>7}{'eff':>8}{'tokens':>9}")
    print(hdr)
    print("-" * len(hdr))
    rows, panels = [], []
    for spec in specs:
        label, _, path = spec.partition("=")
        if not path:
            label, path = Path(spec).parent.name, spec
        codec, cfg, ck = load_spectro_codec(Path(path))
        dec, codes, keep, n_avail, n_used = _audit_stream(
            codec, cfg, modality, eval_shots, n_windows, batch_size, num_workers,
            cache_tag=f"audit{n_eval_shots_used}", notch=notch, band_bins=band_bins,
            device=device, want_baseline=(len(rows) == 0), detrended=detrended)
        util = gate_mod.utilization(codes.reshape(1, *codes.shape), cfg=cfg)
        row = {
            "arm": label, "step": ck.get("step"), "ckpt": str(path),
            "patch_lattice_ratio": dec["patch_lattice_ratio"],
            "target_patch_lattice_ratio": dec["target_patch_lattice_ratio"],
            "lattice_energy_frac": dec["lattice_energy_frac"],
            "seam_ratio_freq": dec["seam_ratio_freq"],
            "seam_ratio_time": dec["seam_ratio_time"],
            "hf_ratio": dec["sharpness"], "envelope_corr": dec["envelope_corr"],
            "peak_f1": dec["peak_f1"],
            # FULL-spectrogram reconstruction (nothing collapsed) + the trivial baselines
            # in the same units, so the codec row is interpretable on its own.
            **{k: dec[k] for k in dec if k.startswith("spec_") or k.startswith("base_")},
            # mode-track ranking metrics (gate.mode_structure_metrics)
            **{k: dec[k] for k in dec
               if k in ("ms_ssim", "mode_track_f1", "mode_track_f1_detr",
                        "mode_track_precision", "mode_track_recall", "mode_track_median_df",
                        "mode_track_n_gt_peaks", "mode_track_n_gt_peaks_detr",
                        "ridge_traj_corr", "spectral_contrast_ratio", "spectral_contrast_gt")},
            "n_distinct_codes": util["n_distinct_codes"],
            "effective_codes": util.get("effective_codes", float("nan")),
            "n_tokens": util["n_observed_tokens"], "n_windows": n_used,
            "windows_available": n_avail,
            # STFT geometry of THIS arm — arms with different n_fft are metered against
            # DIFFERENT targets (same raw windows, different spectral representation), so the
            # table must say which grid each row was scored on.
            "stft_n_fft": int(getattr(cfg, "stft_n_fft", 1024)),
            "stft_hop": int(getattr(cfg, "stft_hop", 256)),
            "freq_bins": int(cfg.freq_bins), "time_frames": int(cfg.time_frames),
            "patch_f": int(cfg.patch_f), "patch_t": int(cfg.patch_t), "n_tok": int(cfg.n_tok),
        }
        rows.append(row)
        panels.append((label, keep[0], keep[1], cfg))
        _grid = f"{row['stft_n_fft']}/{row['freq_bins']}/{row['patch_f']}x{row['patch_t']}"
        print(f"{label:<14}{str(row['step']):>8}{_grid:>17}{row['patch_lattice_ratio']:>9.2f}"
              f"{row['target_patch_lattice_ratio']:>8.2f}{row['lattice_energy_frac']:>9.3f}"
              f"{row['seam_ratio_freq']:>8.3f}{row['seam_ratio_time']:>8.3f}"
              f"{row['hf_ratio']:>9.4f}{row['envelope_corr']:>9.4f}{row['peak_f1']:>8.4f}"
              f"{row['n_distinct_codes']:>7d}{row['effective_codes']:>8.1f}"
              f"{row['n_tokens']:>9d}", flush=True)
    _base = {k: v for k, v in rows[0].items() if k.startswith("base_")} if rows else {}
    for r in rows[1:]:
        r.update({k: v for k, v in _base.items() if k not in r})
    _print_full_spectro_table(rows, band_bins)
    if fig_path is not None:
        _render_audit_fig(panels, Path(fig_path), modality)
    return rows


def _print_full_spectro_table(rows, band_bins: int) -> None:
    """The FULL-SPECTROGRAM table: nothing collapsed, with the trivial baselines beneath it.

    The baselines depend only on the TARGET windows — identical for every arm up to the
    streaming batch grouping — so they are printed once, from the first arm's row, with the
    max spread across arms reported so that assumption is visible rather than assumed.
    """
    if not rows or "spec_corr2d" not in rows[0]:
        return
    print()
    # The band cut is expressed in STFT BINS, whose width depends on n_fft. Group the rows by
    # grid so the kHz label (and the trivial baselines, which are computed on each grid's own
    # target) are correct for every row rather than assumed to be the 512-bin ones.
    def _grid_key(r):
        return (r.get("stft_n_fft", 1024), r.get("freq_bins", 512))

    grids = []
    for r in rows:
        if _grid_key(r) not in grids:
            grids.append(_grid_key(r))
    print("FULL-SPECTROGRAM reconstruction (no axis collapsed)")
    print("  spec_nrmse = RMSE/std(target) per (window, channel): 0 = perfect, 1.0 = the "
          "window's own constant mean")
    if len(grids) > 1:
        print("  NOTE: arms span DIFFERENT STFT grids. Same raw windows, different spectral "
              "representation, so each group is scored against its OWN target — read each "
              "group against ITS OWN ~tmean / ~wcmean baselines, printed beneath it.")
    for nfft, fbins in grids:
        grp = [r for r in rows if _grid_key(r) == (nfft, fbins)]
        khz_per_bin = 250.0 / fbins
        k = min(band_bins, fbins)
        print()
        print(f"  --- grid n_fft={nfft}, freq_bins={fbins} ({khz_per_bin:.3f} kHz/bin, full "
              f"band 0-{fbins * khz_per_bin:.0f} kHz); band = bins 0-{k} = "
              f"0-{k * khz_per_bin:.1f} kHz ---")
        hdr = (f"{'arm':<14}{'spec_nrmse':>11}{'corr2d':>9}{'nrmse_band':>11}{'corr2d_band':>12}"
               f"{'valid_frac':>11}   {'env_corr(old)':>13}")
        print(hdr)
        print("-" * len(hdr))
        for r in grp:
            print(f"{r['arm']:<14}{r['spec_nrmse']:>11.4f}{r['spec_corr2d']:>9.4f}"
                  f"{r['spec_nrmse_band']:>11.4f}{r['spec_corr2d_band']:>12.4f}"
                  f"{r['spec_valid_frac']:>11.4f}   {r['envelope_corr']:>13.4f}", flush=True)
        print("-" * len(hdr))
        # ---- MODE-TRACK table: the RANKING metrics ------------------------------------ #
        if "ms_ssim" in grp[0]:
            print()
            print("  MODE-STRUCTURE metrics (the RANKING keys; spec_nrmse above is a FLOOR "
                  "only). Per TIME FRAME, full 0-250 kHz.")
            h2 = (f"{'arm':<14}{'ms_ssim':>9}{'mode_f1':>9}{'f1_detr':>9}{'ridge':>8}"
                  f"{'contrast':>10}{'hf_ratio':>10}{'lattice':>9}{'codes':>7}")
            print(h2)
            print("-" * len(h2))
            for r in grp:
                print(f"{r['arm']:<14}{r['ms_ssim']:>9.4f}{r['mode_track_f1']:>9.4f}"
                      f"{r.get('mode_track_f1_detr', float('nan')):>9.4f}"
                      f"{r['ridge_traj_corr']:>8.4f}"
                      f"{r['spectral_contrast_ratio']:>10.4f}{r['hf_ratio']:>10.4f}"
                      f"{r['patch_lattice_ratio']:>9.2f}{r['n_distinct_codes']:>7d}", flush=True)
            print("-" * len(h2))
            print(f"{'~tmean':<14}{grp[0]['base_tmean_ms_ssim']:>9.4f}"
                  f"{grp[0]['base_tmean_mode_track_f1']:>9.4f}"
                  f"{grp[0].get('base_tmean_mode_track_f1_detr', float('nan')):>9.4f}"
                  f"{grp[0]['base_tmean_ridge_traj_corr']:>8.4f}"
                  f"{grp[0]['base_tmean_spectral_contrast_ratio']:>10.4f}"
                  f"{'':>10}{'':>9}{'':>7}   the envelope bar (no temporal structure)",
                  flush=True)
        for base, what in (("self", "target vs itself (metric self-check)"),
                           ("tmean", "target's time-averaged envelope, broadcast over T"),
                           ("cfmean", "per-(chan, freq) mean over the batch"),
                           ("wcmean", "per-(window, chan) scalar mean")):
            keys = [f"base_{base}_spec_nrmse", f"base_{base}_spec_corr2d",
                    f"base_{base}_spec_nrmse_band", f"base_{base}_spec_corr2d_band"]
            if keys[0] not in grp[0]:
                continue
            spread = max(max(abs(r[key] - grp[0][key]) for r in grp) for key in keys)
            print(f"{'~' + base:<14}{grp[0][keys[0]]:>11.4f}{grp[0][keys[1]]:>9.4f}"
                  f"{grp[0][keys[2]]:>11.4f}{grp[0][keys[3]]:>12.4f}"
                  f"{'':>11}   (spread over arms {spread:.1e})  {what}", flush=True)


def _render_audit_fig(panels, out_path: Path, modality: str):
    """[GT | recon per arm] x [crop, log|2-D FFT|] — the visual companion to the audit table.

    The FFT panel is where the artifact is unmistakable: a patch lattice shows up as a regular
    grid of bright dots at multiples of (F/patch_f, T/patch_t) that the ground truth does not
    have."""
    _lab0, gt0, _r0, cfg0 = panels[0]
    # Pick the MOST STRUCTURED (window, channel) instead of always (0, 0): a flat window shows
    # nothing whether the codec works or not. Score by the std of the per-frequency envelope
    # over time -- high where mode tracks / bursts actually move.
    _env = gt0.std(axis=3).std(axis=2)                      # (B, C): variation across freq of
    w, ch = np.unravel_index(int(np.argmax(_env)), _env.shape)  # the per-freq temporal spread
    w, ch = int(w), int(ch)
    ncol = 1 + len(panels)
    fig, axes = plt.subplots(2, ncol, figsize=(4.6 * ncol, 8.4), squeeze=False)
    # Default to the FULL frequency axis. The old hard-coded 128-256 crop (62-125 kHz) sat
    # ENTIRELY ABOVE the 0-58.6 kHz mode band where the physics lives, so the deliverable
    # figure never showed the structure it was meant to judge.
    f0 = int(os.environ.get("FIG_F0", 0))
    f1 = int(os.environ.get("FIG_F1", gt0.shape[2]))

    def _fft(a):
        y = a - a.mean()
        return np.log10(np.abs(np.fft.fftshift(np.fft.fft2(y))) + 1e-6)

    gt = gt0[w, ch]
    vmin, vmax = np.percentile(gt, [2, 98])
    # Each arm's GT is the same raw window on that arm's own STFT grid, so show the GT that
    # belongs to the FIRST arm and let each recon panel carry its own geometry.
    imgs = [("GROUND TRUTH", gt)] + [(lab, r[w, ch]) for lab, _g, r, _c in panels]
    fmin, fmax = None, None
    # each panel carries its OWN cfg: arms can differ in n_fft (hence freq_bins) and patch size,
    # so the frequency crop and the patch guide-lines must be resolved per panel in kHz, not in
    # bins. FIG_F0 / FIG_F1 stay bin-indices INTO THE GT panel and are converted to kHz once.
    _khz0 = 250.0 / gt0.shape[2]
    kz0, kz1 = f0 * _khz0, f1 * _khz0
    cfgs = [cfg0] + [c for _l, _g, _r, c in panels]
    for j, (lab, arr) in enumerate(imgs):
        cfj = cfgs[j]
        _khz = 250.0 / arr.shape[0]
        a0, a1 = int(round(kz0 / _khz)), min(arr.shape[0], int(round(kz1 / _khz)))
        # PHYSICAL axes: frequency in kHz (so panels on different STFT grids are directly
        # comparable by eye) and time in STFT frames.
        axes[0][j].imshow(arr[a0:a1], aspect="auto", origin="lower", cmap="magma",
                          vmin=vmin, vmax=vmax, interpolation="nearest",
                          extent=[0, arr.shape[1], a0 * _khz, a1 * _khz])
        for k in range(1, max(1, (a1 - a0) // cfj.patch_f)):
            axes[0][j].axhline((a0 + k * cfj.patch_f) * _khz, color="c", lw=0.35, alpha=0.5)
        for k in range(1, cfj.time_frames // cfj.patch_t):
            axes[0][j].axvline(k * cfj.patch_t - 0.5, color="c", lw=0.4, alpha=0.7)
        axes[0][j].set_ylabel("frequency (kHz)")
        axes[0][j].set_xlabel("STFT time frame")
        axes[0][j].set_title(
            f"{lab}\n{kz0:.0f}-{kz1:.0f} kHz  (n_fft {getattr(cfj, 'stft_n_fft', 1024)}, "
            f"{arr.shape[0]} bins x {arr.shape[1]} frames; cyan = "
            f"{cfj.patch_f}x{cfj.patch_t} patches)", fontsize=9)
        sp = _fft(arr)
        if fmin is None:
            fmin, fmax = np.percentile(sp, [5, 99.5])
        axes[1][j].imshow(sp, aspect="auto", origin="lower", cmap="viridis",
                          vmin=fmin, vmax=fmax)
        axes[1][j].set_title(f"{lab} log|2-D FFT|", fontsize=9)
    fig.suptitle(f"IGNITE {modality} spectro codec vs GROUND TRUTH — most-structured "
                 f"held-out window (w={w}, ch={ch}); bottom row = log|2-D FFT|", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"WROTE {out_path}")


def run_trajectory(dirs, keys=("envelope_corr", "sharpness", "patch_lattice_ratio")):
    """Print each arm's GATE TRAJECTORY from its ``gate_*.json`` files.

    Codec training here does not converge — best-of-N picks a transient — so an arm is judged
    by its trajectory, not its best checkpoint. One row per eval step: distinct codes, the
    reconstruction metrics, and the patch-lattice ratio, plus a footer with the best-scoring
    step. ``n_distinct_codes = 1`` marks a collapsed eval.
    """
    import json as _json
    for d in dirs:
        d = Path(d)
        files = sorted(d.glob("gate_*.json"),
                       key=lambda f: int(f.stem.split("_")[1]))
        if not files:
            print(f"[trajectory] {d}: no gate_*.json"); continue
        print(f"\n=== {d.name} ({len(files)} evals) ===")
        print(f"{'step':>7}{'codes':>7}{'env_corr':>10}{'peak_f1':>9}{'hf_ratio':>10}"
              f"{'lattice':>10}{'gt_lat':>8}{'nrmse':>8}{'corr2d':>8}{'score':>9}{'best':>6}")
        best_step, best_score = None, float("-inf")
        for f in files:
            g = _json.loads(f.read_text())
            dec, ut = g.get("decode", {}), g.get("utilization", {})
            sc = g.get("score")
            sc_f = sc if isinstance(sc, (int, float)) else float("-inf")
            if sc_f > best_score:
                best_score, best_step = sc_f, g.get("step")
            print(f"{g.get('step', -1):>7}{ut.get('n_distinct_codes', -1):>7}"
                  f"{dec.get('envelope_corr', float('nan')):>10.4f}"
                  f"{dec.get('peak_f1', float('nan')):>9.4f}"
                  f"{dec.get('sharpness', float('nan')):>10.4f}"
                  f"{dec.get('patch_lattice_ratio', float('nan')):>10.2f}"
                  f"{dec.get('target_patch_lattice_ratio', float('nan')):>8.2f}"
                  # full-spectrogram recon; nan on gate JSONs written before 2026-09-02
                  f"{dec.get('spec_nrmse', float('nan')):>8.3f}"
                  f"{dec.get('spec_corr2d', float('nan')):>8.3f}"
                  f"{sc_f:>9.4f}{'  <--' if g.get('is_best') else '':>6}")
        print(f"  best-scoring step {best_step} (score {best_score:.4f})")


# ===================================================================================== #
# SLOW-TS AUDIT (2026-09-03) — the slow-TS analogue of run_audit, plus the two things the
# spectro side learned it needed: a FULL-array nRMSE reported beside its TRIVIAL BASELINES,
# and an out-of-sample LINEAR FLOOR so an unreachable target is not chased.
# ===================================================================================== #
SLOWTS_SIGNALS_ALL = ("cer_rot", "cer_ti", "mse", "ts_core_density", "ts_core_temp",
                      "ts_tangential_density", "ts_tangential_temp")

# MEASURED out-of-sample PCA rank-4 floors (slowts_linear_floor, 8 fit / 4 score held-out shots,
# same mask + same normalisation as every codec row). This is the MATCHED-RATE reference: the
# codec spends 4 tokens x log2(1000) = 39.9 bits, so k=4 CONTINUOUS components is already a
# richer code than it has. Printed on each full-shot panel so the panel is self-contained.
SLOWTS_K4_FLOOR = {
    "cer_rot": 0.4518, "cer_ti": 0.5307, "mse": 0.6308, "ts_core_density": 0.7816,
    "ts_core_temp": 0.6750, "ts_tangential_density": 0.6645, "ts_tangential_temp": 0.7321,
}


def _slowts_level_stats(rec_blk, gt_blk, mask_blk):
    """ACROSS-window level amplitude ratio + correlation from blocked (N, C, T) arrays.

    `slowts_std_ratio_t` is a WITHIN-window quantity (T=5 samples), so it mostly measures
    high-frequency wiggle and does NOT track what the eye judges on a full-shot trace: whether
    the reconstruction's per-window LEVEL moves with the target's. This computes, per profile
    position, the std over windows of the per-(window, position) masked mean, for recon and
    target, and returns (variance-weighted ratio, mean correlation). Ideal 1.0 / 1.0.
    """
    m = mask_blk > 0.5
    n = m.sum(-1)
    ok = n >= 1
    gl = np.where(ok, (gt_blk * m).sum(-1) / np.maximum(n, 1), np.nan)
    rl = np.where(ok, (rec_blk * m).sum(-1) / np.maximum(n, 1), np.nan)
    num = den = 0.0
    cors = []
    for c in range(gl.shape[1]):
        g, r = gl[:, c], rl[:, c]
        f = np.isfinite(g) & np.isfinite(r)
        if f.sum() < 8:
            continue
        gd, rd = g[f] - g[f].mean(), r[f] - r[f].mean()
        if gd.std() <= 1e-8:
            continue
        num += (rd * rd).sum(); den += (gd * gd).sum()
        if rd.std() > 1e-8:
            cors.append(float((gd * rd).mean() / (gd.std() * rd.std())))
    return (float(np.sqrt(num / den)) if den > 0 else float("nan"),
            float(np.mean(cors)) if cors else float("nan"))


def _slowts_dataset(signal, cfg, shots, cache_tag):
    """Held-out slow-TS window stream for ``signal`` under a CANONICAL evaluation cfg.

    ACTIVITY-STRATIFIED SAMPLING IS FORCED OFF here. ``cfg.min_activity`` / ``cfg.active_bias``
    are a TRAINING knob (bias the batch toward well-observed windows), but they live on the same
    cfg the checkpoint stores, so a dataset built from each arm's own cfg hands DIFFERENT arms
    DIFFERENT eval windows — the stratified arm gets the more-present, easier ones. Measured
    2026-09-03: on ts_tangential_temp the tmean baseline itself moved 0.3962 -> 0.3609 between
    two arms of the same audit, i.e. the "improvement" was partly a change of test set. Every
    arm is therefore streamed through the unstratified draw, which is what the prod checkpoints
    were graded on and is the only thing that makes two arms comparable.
    """
    import dataclasses
    eval_cfg = dataclasses.replace(cfg, min_activity=0.0, active_bias=0.0)
    return tc.SlowTSCodecPairDataset(
        signal, shots, eval_cfg, data_dir=DATA_DIR,
        lengths_cache_path=str(CACHE_DIR / f"codec_{signal}_{cache_tag}_lengths.pt"),
    )


def _slowts_cfg_for(signal: str):
    """A cfg with the loader's real channel count AND the production channel stats injected.

    The same object ``train_codec.main`` builds, so a dataset built from it yields exactly the
    standardized windows every arm trains and evaluates on (baselines + floor included).
    """
    cfg = tc.slowts_codec_cfg(signal, tc.modality_channels(signal))
    method, mean, std = tc.load_slowts_channel_stats(signal)
    cfg.preprocess_method, cfg.channel_mean, cfg.channel_std = method, mean, std
    return cfg


def _slowts_load_windows(signal, cfg, shots, n_windows, batch_size, num_workers, cache_tag):
    """Stream ``n_windows`` (signal, mask) windows, evenly spread over the shots' window range."""
    from torch.utils.data import DataLoader, Subset
    ds = _slowts_dataset(signal, cfg, shots, cache_tag)
    n = len(ds)
    idx = np.linspace(0, n - 1, min(n_windows, n)).astype(int).tolist()
    dl = DataLoader(Subset(ds, idx), batch_size=batch_size, num_workers=num_workers,
                    shuffle=False)
    return dl, n


def _slowts_audit_stream(codec, cfg, signal, shots, n_windows, batch_size, num_workers,
                         cache_tag):
    """Encode+decode held-out slow-TS windows; accumulate per-batch metrics + all codes."""
    dl, n_avail = _slowts_load_windows(signal, cfg, shots, n_windows, batch_size,
                                       num_workers, cache_tag)
    acc, codes, n_used, keep = {}, [], 0, None
    with torch.no_grad():
        for bi, (sig, mask) in enumerate(dl):
            out = codec(sig)
            rec = out["recon"].numpy()
            gt = sig.numpy()
            m = mask.numpy()
            dm = gate_mod.slowts_decode_fidelity(rec, gt, m)
            dm.update(gate_mod.full_slowts_metrics(rec, gt, mask=m))
            dm.update(gate_mod.trivial_slowts_baselines(gt, mask=m))
            for k, v in dm.items():
                acc.setdefault(k, []).append(float(v))
            codes.append(out["codes"].numpy())
            n_used += gt.shape[0]
            if bi == 0:
                keep = (gt.copy(), rec.copy(), m.copy())
    dec = {k: float(np.nanmean(v)) for k, v in acc.items()}
    return dec, np.concatenate(codes), keep, n_avail, n_used


def _slowts_forecast_margin(codec, cfg, signal, shots, n_seq, seq_len, cache_tag):
    """gate.forecastability on CONSECUTIVE held-out windows — the predictability objective.

    Mirrors ``_stream_slowts_eval_data``'s frame_seq construction (consecutive 50 ms windows
    are consecutive dataset indices within a shot) but over a much larger sample than the
    32-window gate: ``n_seq`` blocks of ``seq_len`` windows.
    """
    ds = _slowts_dataset(signal, cfg, shots, cache_tag)
    n = len(ds)
    n_seq = min(n_seq, n // seq_len)
    seqs = []
    for b in range(n_seq):
        seqs.append(torch.stack([ds[b * seq_len + k][0] for k in range(seq_len)], dim=0))
    frame_seq = torch.stack(seqs, dim=0)                       # (B, F, C, T)
    B, F = frame_seq.shape[:2]
    with torch.no_grad():
        flat = frame_seq.reshape(B * F, *frame_seq.shape[2:])
        _, codes_flat = codec.quantize(codec.encode(flat))
    codes_seq = codes_flat.reshape(B, F, cfg.n_tok, cfg.fsq_dim)
    fc = gate_mod.forecastability(codes_seq, transition_mask=None)
    pers = gate_mod.persistence(codes_seq[:, 0], codes_seq[:, 1])
    return fc, float(pers), codes_seq.numpy()


def slowts_linear_floor(signal, cfg, fit_shots, score_shots, ks=(1, 2, 4, 8, 16, 32),
                        n_fit=400, n_score=400, num_workers=4):
    """Out-of-sample LINEAR nRMSE floor per modality, in the SAME normalisation as the codec.

    A PCA basis of the flattened (C*T) window is fit on ``fit_shots`` (held out from training)
    and SCORED on ``score_shots`` (different, also held out). Each scoring window is projected
    onto the top-k components by MASKED least squares (valid samples only), reconstructed, and
    metered with :func:`gate.full_slowts_metrics` — the identical function, mask and units the
    codec rows use. The k-sweep is reported because the honest floor depends on the rank you
    allow: the codec's 4 FSQ tokens carry ~40 bits, so k=4-8 CONTINUOUS components is already a
    considerably richer code than the codec has.
    """
    def _stack(shots, n, tag):
        dl, _ = _slowts_load_windows(signal, cfg, shots, n, 32, num_workers, tag)
        xs, ms = [], []
        for sig, mask in dl:
            xs.append(sig.numpy()); ms.append(mask.numpy())
        return np.concatenate(xs).astype(np.float64), np.concatenate(ms).astype(np.float64)

    Xf, Mf = _stack(fit_shots, n_fit, "floorfit")
    Xs, Ms = _stack(score_shots, n_score, "floorscore")
    B, C, T = Xf.shape
    F = Xf.reshape(B, -1)
    mu = F.mean(axis=0)
    U, S, Vt = np.linalg.svd(F - mu, full_matrices=False)
    out = {}
    Bs = Xs.shape[0]
    Fs = Xs.reshape(Bs, -1)
    Msf = (Ms.reshape(Bs, -1) > 0.5).astype(np.float64)
    for k in ks:
        k = int(min(k, Vt.shape[0]))
        Bk = Vt[:k].T                                            # (D, k)
        rec = np.empty_like(Fs)
        for i in range(Bs):
            w = Msf[i]
            A = (Bk * w[:, None])
            G = A.T @ Bk + 1e-8 * np.eye(k)
            rhs = A.T @ (Fs[i] - mu)
            a = np.linalg.solve(G, rhs)
            rec[i] = mu + Bk @ a
        m = gate_mod.full_slowts_metrics(rec.reshape(Bs, C, T), Xs, mask=Ms)
        out[k] = (m["slowts_nrmse_pooled"], m["slowts_nrmse_med"],
                  m["slowts_nrmse"], m["slowts_corr"])
    base = gate_mod.trivial_slowts_baselines(Xs, mask=Ms)
    return out, base, Bs


def run_slowts_audit(specs, signal, n_windows, eval_n_shots, n_eval_shots_used,
                     batch_size, num_workers, fig_path=None, floor=True,
                     n_seq=64, seq_len=8):
    """Print the slow-TS reconstruction table for each ``label=ckpt`` spec, with baselines."""
    all_shots = spike_mod.discover_shots(DATA_DIR)
    eval_pool = all_shots[-eval_n_shots:]
    eval_shots = eval_pool[:n_eval_shots_used]
    print(f"[slowts-audit] signal={signal} held-out shots={eval_shots} "
          f"(last {eval_n_shots} of {len(all_shots)})", flush=True)
    rows, panels = [], []
    for spec in specs:
        label, _, path = spec.partition("=")
        if not path:
            label, path = Path(spec).parent.name, spec
        codec, cfg, ck = load_slowts_codec(Path(path))
        dec, codes, keep, n_avail, n_used = _slowts_audit_stream(
            codec, cfg, signal, eval_shots, n_windows, batch_size, num_workers,
            cache_tag=f"audit{n_eval_shots_used}")
        util = gate_mod.utilization(codes.reshape(1, *codes.shape), cfg=cfg)
        fc, pers, _cs = _slowts_forecast_margin(
            codec, cfg, signal, eval_shots, n_seq, seq_len,
            cache_tag=f"audit{n_eval_shots_used}")
        row = {
            "arm": label, "signal": signal, "step": ck.get("step"), "ckpt": str(path),
            "fsq_levels": list(cfg.fsq_levels), "codebook_size": int(cfg.codebook_size),
            "n_tok": int(cfg.n_tok),
            **{k: dec[k] for k in dec if k.startswith("slowts_") or k.startswith("base_")},
            "envelope_corr": dec["envelope_corr"], "peak_f1": dec["peak_f1"],
            "sharpness": dec["sharpness"],
            "n_distinct_codes": util["n_distinct_codes"],
            "frac_codes_used": util["frac_codes_used"],
            "frac_of_observable": util["frac_of_observable"],
            "min_dim_entropy": util["min_dim_entropy"],
            "effective_codes": util.get("effective_codes", float("nan")),
            # UTILISATION AS A RATE IN BITS. frac_codes_used is a coupon-collector quantity
            # (and the in-training gate's version is capped at 0.384 by its 384-token eval), so
            # the honest utilisation number is how many of the code's OWN bits it actually
            # spends: realised = n_tok * H_joint / ln2, cap = n_tok * log2(codebook_size).
            "code_entropy_nats": util.get("code_entropy_nats", float("nan")),
            "bits_realised": (int(cfg.n_tok) * util.get("code_entropy_nats", float("nan"))
                              / math.log(2.0)),
            "bits_cap": int(cfg.n_tok) * math.log2(int(cfg.codebook_size)),
            "n_tokens": util["n_observed_tokens"], "n_windows": n_used,
            "forecast_margin_tr": fc["margin_transition"],
            "forecast_margin_all": fc["margin_overall"],
            "n_transition": fc["n_transition"], "persistence": pers,
            "windows_available": n_avail,
        }
        rows.append(row)
        panels.append((label, keep, cfg))
    _print_slowts_table(rows)
    if floor:
        fit_shots, score_shots = eval_pool[4:12], eval_pool[:4]
        cfg0 = _slowts_cfg_for(signal)
        fl, base, nsc = slowts_linear_floor(signal, cfg0, fit_shots, score_shots,
                                            num_workers=num_workers)
        print(f"\nLINEAR FLOOR ({signal}) — PCA basis fit on {len(fit_shots)} held-out shots, "
              f"scored on {len(score_shots)} DIFFERENT held-out shots ({nsc} windows), "
              f"same mask + same normalisation")
        print(f"  {'k comps':>9}{'pooled':>10}{'med':>9}{'mean':>10}{'corr':>9}")
        for k, v in fl.items():
            print(f"  {k:>9d}{v[0]:>10.4f}{v[1]:>9.4f}{v[2]:>10.4f}{v[3]:>9.4f}")
        print(f"  {'~tmean':>9}{base['base_tmean_slowts_nrmse_pooled']:>10.4f}"
              f"{base['base_tmean_slowts_nrmse_med']:>9.4f}"
              f"{base['base_tmean_slowts_nrmse']:>10.4f}"
              f"{base['base_tmean_slowts_corr']:>9.4f}   (score-shot baseline)")
        for r in rows:
            r["floor_pooled"] = {str(k): v[0] for k, v in fl.items()}
            r["floor_mean"] = {str(k): v[2] for k, v in fl.items()}
    if fig_path is not None:
        _render_slowts_audit_fig(panels, Path(fig_path), signal)
    return rows


def _print_slowts_table(rows):
    if not rows:
        return
    print()
    print("FULL-WINDOW slow-TS reconstruction (no axis collapsed; VALID samples only)")
    print("  slowts_nrmse = RMSE/std(target) per WINDOW over the flattened (C, T) plane:")
    print("    0.0 = perfect | 1.0 = that window's own constant mean | >1.0 = worse than a constant")
    print("  std_ratio = std(recon)/std(target), IDEAL 1.0: << 1 = dynamic-range COLLAPSE, which")
    print("    nRMSE (a squared error, minimised by shrinking to the mean) REWARDS and corr")
    print("    (scale-invariant) cannot see. sr_t is the ratio over TIME only (per window,chan).")
    print("  bits = n_tok * H_joint / ln2 (realised) of n_tok * log2(codebook) (cap): the honest")
    print("    utilisation RATE. frac_codes_used is a coupon-collector number - do not threshold.")
    hdr = (f"{'arm':<22}{'pooled':>9}{'med':>8}{'mean':>9}{'corr':>8}{'present':>9}   "
           f"{'sr_p':>7}{'sr_t':>7}   {'env_corr':>9}{'sharp':>8}{'codes':>7}{'eff':>8}"
           f"{'used%':>7}{'bits':>7}{'cap':>7}{'bits%':>7}{'fc_tr':>9}{'tokens':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['arm']:<22}{r['slowts_nrmse_pooled']:>9.4f}"
              f"{r['slowts_nrmse_med']:>8.4f}{r['slowts_nrmse']:>9.4f}"
              f"{r['slowts_corr']:>8.4f}{r['slowts_present_frac']:>9.3f}   "
              f"{r['slowts_std_ratio_pooled']:>7.3f}{r['slowts_std_ratio_t']:>7.3f}   "
              f"{r['envelope_corr']:>9.4f}{r['sharpness']:>8.4f}"
              f"{r['n_distinct_codes']:>7d}{r['effective_codes']:>8.1f}"
              f"{100 * r['frac_codes_used']:>6.1f}%"
              f"{r['bits_realised']:>7.1f}{r['bits_cap']:>7.1f}"
              f"{100 * r['bits_realised'] / max(r['bits_cap'], 1e-9):>6.0f}%"
              f"{r['forecast_margin_tr']:>+9.4f}{r['n_tokens']:>9d}", flush=True)
    print("-" * len(hdr))
    for base, what in (("self", "target vs itself (metric self-check; MUST be 0.0000 / 1.0000)"),
                       ("tmean", "per-(window,position) time mean -> THE BAR TO BEAT"),
                       ("cmean", "per-position mean over the whole batch"),
                       ("wcmean", "per-window scalar mean (the 1.0000 anchor)")):
        kp = f"base_{base}_slowts_nrmse_pooled"
        km = f"base_{base}_slowts_nrmse_med"
        k0 = f"base_{base}_slowts_nrmse"
        k1 = f"base_{base}_slowts_corr"
        ks = f"base_{base}_slowts_std_ratio_pooled"
        kt = f"base_{base}_slowts_std_ratio_t"
        if k0 not in rows[0]:
            continue
        spread = max(max(abs(r[k] - rows[0][k]) for r in rows) for k in (kp, k1))
        print(f"{'~' + base:<22}{rows[0][kp]:>9.4f}{rows[0][km]:>8.4f}{rows[0][k0]:>9.4f}"
              f"{rows[0][k1]:>8.4f}{'':>9}   {rows[0].get(ks, float('nan')):>7.3f}"
              f"{rows[0].get(kt, float('nan')):>7.3f}   "
              f"(spread over arms {spread:.1e})  {what}", flush=True)


def _render_slowts_audit_fig(panels, out_path, signal):
    """[GT | recon] profile-vs-time heatmaps for one held-out batch, one row per arm."""
    n = len(panels)
    fig, axes = plt.subplots(n, 3, figsize=(13, 3.1 * n), squeeze=False)
    for i, (label, (gt, rec, m), cfg) in enumerate(panels):
        b = int(np.argmax(m.reshape(m.shape[0], -1).mean(axis=1)))
        g, r, mm = gt[b], rec[b], m[b]
        vmin, vmax = np.percentile(g[mm > 0.5], [1, 99]) if (mm > 0.5).any() else (-1, 1)
        for j, (arr, ttl) in enumerate(((g, "ground truth"), (r, "reconstruction"),
                                        (r - g, "recon - GT"))):
            ax = axes[i][j]
            im = ax.imshow(arr, aspect="auto", origin="lower", cmap="viridis",
                           vmin=(vmin if j < 2 else None), vmax=(vmax if j < 2 else None))
            ax.set_title(f"{label} — {ttl}", fontsize=9)
            ax.set_xlabel("time sample"); ax.set_ylabel("profile position")
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"slow-TS {signal}: one held-out 50 ms window (standardized units)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"WROTE {out_path}")


@torch.no_grad()
def render_slowts_full_shot(specs, signals, shot=SHOT, out_path=None, positions=1,
                            max_windows=400, title="", ckpt_map=None):
    """GT vs reconstruction over the FULL shot time window — the judge-by-eye deliverable.

    For each signal every consecutive 50 ms window of ``shot`` is encoded -> quantized ->
    decoded and the per-window (C, T) outputs are STITCHED back into a continuous (C, N*T)
    trace, so the x-axis is real time across the whole shot rather than 5 samples.

    LEFT column: line traces for the ``positions`` most STRUCTURED profile positions (ranked by
    validity-occupancy x temporal std, so a dead all-zero channel is never picked). GT solid,
    each arm dashed/dotted in the same colour. Missing samples are plotted as NaN gaps, so a
    break in the line is honest missingness rather than a prediction failure.

    RIGHT column: the whole (position x time) plane as an image — GT on top, the first arm's
    reconstruction below, separated by a blank row and drawn on ONE shared colour scale. This
    is what shows whether the profile EVOLUTION is reproduced, which four overlaid line traces
    cannot.
    """
    # ``ckpt_map`` ({signal: path}) overrides ``specs`` per signal — needed for the FINAL
    # deliverable, where each modality's CURRENT BEST checkpoint lives in a different run dir
    # under a different arm name (and is sometimes codec_last.pt, not codec_best.pt), so one
    # "{signal}" template cannot address them.
    ckpt_of = dict(sp.split("=", 1) for sp in specs) if specs else {}
    rows = len(signals)
    fig, axes = plt.subplots(rows, 2, figsize=(19, 2.7 * rows), squeeze=False,
                             gridspec_kw={"width_ratios": [1.35, 1.0]})
    for i, sig_name in enumerate(signals):
        axl, axr = axes[i][0], axes[i][1]
        traces = {}
        per_sig = ({"best": ckpt_map[sig_name]} if (ckpt_map and sig_name in ckpt_map)
                   else ckpt_of)
        for label, path in per_sig.items():
            pth = Path(path.format(signal=sig_name))
            if not pth.exists():
                print(f"  [full-shot] {sig_name}: MISSING {pth}")
                continue
            codec, cfg, _ck = load_slowts_codec(pth)
            ds = _slowts_dataset(sig_name, cfg, [shot], f"fullshot{shot}")
            n = min(len(ds), max_windows)
            # NEVER call ds[w] here. ``SlowTSCodecPairDataset._draw_valid_window`` RE-DRAWS a
            # RANDOM in-shot window whenever the requested one has no data (``_build_window``
            # returns None because ``valid_len < need``), so ds[w] is NOT the window at time w.
            # That is fine for training/audit (every returned window is a real one) but it is
            # FATAL here, because this function STITCHES ds[0..n-1] into a time axis.
            #
            # MEASURED 2026-09-03 (shot 204990, all 12 held-out shots identical): the cer_rot /
            # cer_ti records end at ~6.03 s and mse at ~5.93 s, while the file's window grid runs
            # to 12.92 s from the Thomson signals. So only 100 of 219 chunk indices (45.7 %) have
            # real cer/mse data and indices 100-218 were being filled with RANDOM 50 ms windows
            # from elsewhere in the shot — e.g. ds[159] (t=8.95 s, no data) returned the t=5.15 s
            # window. Concatenated, that is a scrambled series with a 5-sample (50 ms) block
            # period, which is EXACTLY the "sample-to-sample +-3.5 square wave" the cer/mse
            # panels showed past figure-time 5.0 s, and it is why the 3 non-Thomson signals
            # "failed" while the 4 Thomson signals (0 redraws) "worked".
            #
            # _build_window is the UN-redrawn accessor. A rejected index becomes an all-zero
            # window with an ALL-ZERO MASK, so it is plotted as a NaN GAP (the existing
            # np.where(mask > 0.5, ...) does that) and excluded from every metric below (
            # full_slowts_metrics needs >= _SLOWTS_MIN_VALID valid samples per window).
            ds[0]                                   # open this worker's HDF5 handle
            G, R, Mk = [], [], []
            n_gap = 0
            for w in range(n):
                item = ds._build_window(w)
                if item is None:
                    n_gap += 1
                    z = np.zeros((cfg.padded_channels, cfg.time_steps), dtype=np.float32)
                    G.append(z); R.append(z.copy()); Mk.append(z.copy())
                    continue
                sw, mw = item
                out = codec(sw.unsqueeze(0))
                G.append(sw.numpy()); R.append(out["recon"][0].numpy()); Mk.append(mw.numpy())
            print(f"  [full-shot] {sig_name} {label}: {n - n_gap}/{n} windows have real data "
                  f"({100 * (n - n_gap) / max(n, 1):.1f}%); {n_gap} plotted as GAPS "
                  f"(no redraw)", flush=True)
            traces[label] = (np.concatenate(G, axis=1), np.concatenate(R, axis=1),
                             np.concatenate(Mk, axis=1), cfg)
        if not traces:
            axl.set_title(f"{sig_name}: no checkpoint"); continue
        lbl0 = list(traces)[0]
        G0, R0, M0, cfg0 = traces[lbl0]
        C = cfg0.channels
        # per-arm nRMSE over the whole shot, in the gate's units, for the panel titles.
        # RE-BLOCK the stitched (C, N*T) trace back into the (N, C, T) 50 ms windows the gate
        # normalises over. Scoring the whole shot as ONE window would divide by the SHOT's std
        # instead of each window's, which is a different (and much more flattering) quantity.
        Tw = cfg0.time_steps

        def _blk(a2d):
            n = a2d.shape[1] // Tw
            return a2d[:C, : n * Tw].reshape(C, n, Tw).transpose(1, 0, 2)

        scores = {}
        for label, (G, R, Mk, _c) in traces.items():
            m = gate_mod.full_slowts_metrics(_blk(R), _blk(G), mask=_blk(Mk))
            scores[label] = m["slowts_nrmse_pooled"]
        base = gate_mod.trivial_slowts_baselines(_blk(G0), mask=_blk(M0))

        # rank positions by occupancy x temporal spread so a dead constant channel is skipped.
        # Rank by (occupancy x SMOOTHED-trend std). Ranking on the RAW std picks the
        # spike-dominated channels, whose traces are unreadable and whose reconstruction nobody
        # can judge by eye; the trend std picks the channels that carry the shot's evolution.
        occ = M0[:C].mean(axis=1)
        vals = np.where(M0[:C] > 0.5, G0[:C], np.nan)
        ker = np.ones(9) / 9.0
        with np.errstate(invalid="ignore"):
            filled = np.nan_to_num(vals, nan=np.nan)
            trend = np.stack([
                np.convolve(np.nan_to_num(filled[c], nan=np.nanmean(filled[c])
                                          if np.isfinite(filled[c]).any() else 0.0),
                            ker, mode="same")
                for c in range(C)])
            spread = np.nan_to_num(trend.std(axis=1))
        rank = np.where(occ >= 0.4, occ * spread, -1.0)
        pick = sorted(np.argsort(-rank)[:positions].tolist())
        t_ax = np.arange(G0.shape[1]) * (tc.CHUNK_S / cfg0.time_steps)
        colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
        for pi, pos in enumerate(pick):
            gt = np.where(M0[pos] > 0.5, G0[pos], np.nan)
            axl.plot(t_ax, gt, color="#222222", lw=2.0, alpha=0.9,
                     label=f"GT (pos {pos})")
            arm_c = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]
            for li, (label, (G, R, Mk, _c)) in enumerate(traces.items()):
                rr = np.where(Mk[pos] > 0.5, R[pos], np.nan)
                axl.plot(t_ax, rr, color=arm_c[li % 4], lw=1.2,
                         ls=("--" if li == 0 else "-"), alpha=0.85,
                         label=f"{label}")
        sc = "  ".join(f"{k} nRMSE {v:.3f}" for k, v in scores.items())
        cov = 100.0 * float((M0[:C].reshape(C, -1).max(axis=0) > 0.5).mean())
        lvr, lvc = _slowts_level_stats(_blk(R0), _blk(G0), _blk(M0))
        flr = SLOWTS_K4_FLOOR.get(sig_name)
        flr_s = f"k=4 floor {flr:.3f}" if flr is not None else "k=4 floor n/a"
        axl.set_title(f"{sig_name} — shot {shot}, {t_ax[-1]:.1f} s, {cov:.0f}% of the axis has "
                      f"DATA (gaps = none)   |   {sc}   |   {flr_s}, constant 1.000   |   "
                      f"amplitude {lvr:.2f}, level corr {lvc:.2f}", fontsize=9)
        axl.set_xlabel("time (s)"); axl.set_ylabel("standardized value")
        axl.legend(fontsize=7, ncol=3, loc="lower left")
        axl.grid(alpha=0.25)

        gap_h = max(1, C // 12)
        gap = np.full((gap_h, G0.shape[1]), np.nan)
        bands = [("GT", np.where(M0[:C] > 0.5, G0[:C], np.nan))]
        for label, (G, R, Mk, _c) in traces.items():
            bands.append((label, np.where(Mk[:C] > 0.5, R[:C], np.nan)))
        stack, ticks, names = [], [], []
        for bi, (nm, arr) in enumerate(bands):
            if bi:
                stack.append(gap)
            ticks.append(sum(a2.shape[0] for a2 in stack) + C / 2)
            names.append(nm if nm == "GT" else f"recon\n{nm}")
            stack.append(arr)
        img = np.vstack(stack)
        fin = img[np.isfinite(img)]
        vmin, vmax = (np.percentile(fin, [2, 98]) if fin.size else (-1, 1))
        im = axr.imshow(img, aspect="auto", origin="upper", cmap="viridis",
                        vmin=vmin, vmax=vmax,
                        extent=[t_ax[0], t_ax[-1], img.shape[0], 0])
        for bi in range(1, len(bands)):
            axr.axhline(bi * (C + gap_h) - gap_h / 2, color="w", lw=1.0)
        axr.set_yticks(ticks)
        axr.set_yticklabels(names, fontsize=7)
        axr.set_xlabel("time (s)")
        axr.set_title(f"{sig_name} — profile position x time (white = missing)", fontsize=9)
        fig.colorbar(im, ax=axr, fraction=0.04)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.99 if title else 1.0))
    out_path = Path(out_path or (OUT / "slowts_full_shot.png"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=105)
    plt.close(fig)
    print(f"WROTE {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit", nargs="+", default=None,
                    help="AUDIT mode: one or more 'label=/path/to/codec_best.pt' specs "
                         "(label optional; defaults to the checkpoint's parent dir name).")
    ap.add_argument("--modality", default="mhr", help="spectro modality for --audit")
    ap.add_argument("--n_windows", type=int, default=720,
                    help="held-out windows to stream (720 x 192 tok = 138k tokens for mhr)")
    ap.add_argument("--eval_n_shots", type=int, default=16,
                    help="must match the arms' --eval_n_shots so the split is the held-out one")
    ap.add_argument("--audit_shots", type=int, default=4,
                    help="how many of those held-out shots to stream")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--fig", default=None, help="write the GT-vs-arms lattice figure here")
    ap.add_argument("--trajectory", nargs="+", default=None,
                    help="print each arm dir's gate_*.json trajectory (codes / recon metrics / "
                         "patch-lattice ratio per eval step) and exit.")
    ap.add_argument("--notch", action="store_true",
                    help="apply gate.remove_patch_lattice (ideal lattice notch) to the "
                         "reconstruction before metering — the DIAGNOSTIC that separates real "
                         "high-frequency content from the artifact.")
    ap.add_argument("--band_bins", type=int, default=gate_mod.DEFAULT_MODE_BAND_BINS,
                    help="upper STFT bin (exclusive) of the mode-band variant of the "
                         "full-spectrogram metrics (default 120 = 0-58.6 kHz)")
    ap.add_argument("--detrended", action="store_true",
                    help="also compute mode_track_f1_detr (a second peak-picking pass; the "
                         "detrended target removes the static envelope, which otherwise scores "
                         "HIGHEST on the plain metric).")
    ap.add_argument("--device", default="cpu",
                    help="torch device for the audit forward pass (cpu | cuda). The paper-sized "
                         "conv decoder is far too slow on CPU for 720 windows.")
    ap.add_argument("--json", default=None, help="also dump the audit rows as JSON here")
    # --- SLOW-TS audit / floor / full-shot figure (2026-09-03) --------------------------
    ap.add_argument("--slowts_audit", nargs="+", default=None,
                    help="SLOW-TS AUDIT mode: 'label=/path/to/codec_best.pt' specs. Use with "
                         "--modality <slow-TS signal>. Prints slowts_nrmse + the trivial "
                         "baselines + utilization + forecast_margin_tr, and (unless "
                         "--no_floor) the out-of-sample linear nRMSE floor.")
    ap.add_argument("--no_floor", action="store_true",
                    help="skip the PCA linear-floor probe in --slowts_audit")
    ap.add_argument("--seq_len", type=int, default=8,
                    help="consecutive windows per forecastability sequence (--slowts_audit)")
    ap.add_argument("--n_seq", type=int, default=64,
                    help="number of consecutive-window sequences (--slowts_audit)")
    ap.add_argument("--full_shot", nargs="+", default=None,
                    help="FULL-SHOT figure mode: 'label=/dir/{signal}/codec_best.pt' specs "
                         "({signal} is substituted per modality). Stitches every 50 ms window "
                         "of --shot into one continuous GT-vs-recon trace.")
    ap.add_argument("--signals", nargs="+", default=list(SLOWTS_SIGNALS_ALL),
                    help="slow-TS signals for --full_shot (default: all 7)")
    ap.add_argument("--shot", default=SHOT, help="shot for --full_shot")
    ap.add_argument("--fig_title", default="", help="suptitle for --full_shot")
    ap.add_argument("--best_ckpt", nargs="+", default=None,
                    help="FINAL-FIGURE mode: per-signal 'signal=/abs/path/codec_X.pt' entries. "
                         "Overrides --full_shot's single {signal} template for those signals, so "
                         "each modality can be rendered from its OWN current-best arm (different "
                         "run dir, different arm name, sometimes codec_last.pt).")
    args = ap.parse_args()

    if args.trajectory:
        run_trajectory(args.trajectory)
        raise SystemExit(0)

    if args.full_shot or args.best_ckpt:
        _cmap = (dict(x.split("=", 1) for x in args.best_ckpt) if args.best_ckpt else None)
        render_slowts_full_shot(args.full_shot or [], args.signals, shot=args.shot,
                                out_path=args.fig, title=args.fig_title, ckpt_map=_cmap)
        raise SystemExit(0)

    if args.slowts_audit:
        _rows = run_slowts_audit(args.slowts_audit, args.modality, args.n_windows,
                                 args.eval_n_shots, args.audit_shots, args.batch_size,
                                 args.num_workers, args.fig, floor=not args.no_floor,
                                 n_seq=args.n_seq, seq_len=args.seq_len)
        if args.json:
            import json as _json
            Path(args.json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.json).write_text(_json.dumps(_rows, indent=2, default=str))
            print(f"WROTE {args.json}")
        raise SystemExit(0)

    if args.audit:
        _rows = run_audit(args.audit, args.modality, args.n_windows, args.eval_n_shots,
                          args.audit_shots, args.batch_size, args.num_workers, args.fig,
                          notch=args.notch, band_bins=args.band_bins, device=args.device,
                          detrended=args.detrended)
        if args.json:
            import json as _json
            Path(args.json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.json).write_text(_json.dumps(_rows, indent=2, default=str))
            print(f"WROTE {args.json}")
        raise SystemExit(0)

    print("=== SPECTRO ece (24 vs 192) ===")
    render_spectro_ece()
    for mod in ("bes", "mhr"):
        print(f"=== SPECTRO {mod} (24 vs 192) ===")
        try:
            render_spectro_generic(mod)
        except Exception as e:  # noqa: BLE001
            print(f"  {mod} FAILED: {e!r}")
    print("=== SLOW-TS ts_core_density (d4 vs d5) ===")
    try:
        render_slowts()
    except Exception as e:  # noqa: BLE001
        print(f"  slowts FAILED: {e!r}")
    # tangential Thomson: old unstratified v6 best (.bak, kept at the 2026-08-05 relaunch)
    # vs the stratified retrain's best — judge-by-eye verdict on the stratification fix.
    for _sig in ("ts_tangential_density", "ts_tangential_temp"):
        print(f"=== SLOW-TS {_sig} (unstratified vs stratified) ===")
        try:
            render_slowts(
                _sig,
                variants=[
                    (f"eval_runs/ignite_codec_{_sig}_v6/codec_best.pt.bak", "v6 unstratified (old best)"),
                    (f"eval_runs/ignite_codec_{_sig}_v6/codec_best.pt", "v6 + stratification (best)"),
                ],
                out_name=f"slowts_{_sig}_strat.png",
                subtitle="blue dashed = radial-zone token boundaries; left = pre-fix best, "
                         "right = present-fraction-stratified retrain best",
            )
        except Exception as e:  # noqa: BLE001
            print(f"  {_sig} FAILED: {e!r}")
    print("DONE")
