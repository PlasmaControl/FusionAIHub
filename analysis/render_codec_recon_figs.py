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
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from tokamak_foundation_model.ignite import train_codec as tc
from tokamak_foundation_model.ignite import gate as gate_mod
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


if __name__ == "__main__":
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
