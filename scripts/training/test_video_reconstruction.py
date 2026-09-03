#!/usr/bin/env python
"""Objective video encoder/decoder test for the tangtv camera modality.

WHY
---
The deterministic VideoOutputHead regresses the partly-stochastic future video
to its conditional mean → the half-moon's sharp corner and the speckle dots
blur away (the same mean-collapse the spectrogram head had). We established the
real tangtv frame is a **rounded half-moon bright band, with sharp corners, and
scattered dots**, and that the model's 120×360 input *retains* that structure —
so the loss is the DECODER, not the encoder.

This test compares video decoders at the FIXED budget (300 tokens @ d_model
1024 — the encoder patch (3,12,12) is not changed) on synthetic tangtv-like
clips, with MISSING channels/frames, and reports per-structure reconstruction:

  * deconv     : per-patch ConvTranspose3d (checkerboard baseline / OLD)
  * resize     : resize-conv decoder (option B, checkerboard-free, deterministic)
  * flow       : VideoFlowHead = resize-conv mean + flow-matching residual
                 (option A) + spatial (H,W) positional embedding (option D)
  * flow_nope  : flow head with the PE off (isolates option D)

PATTERNS (idealized but representative): per clip, a curved **half-moon** bright
band that drifts across the 3 frames, a **sharp angular corner**, and scattered
bright **dots** (speckle), on a quiet noisy background. A random subset of
(channel, frame) pairs is marked MISSING (0-filled to the encoder, excluded from
loss + metrics) to exercise missing-data handling.

METRICS (per decoder, on PRESENT channels; for the flow head, mu = deterministic
decode and samp = flow sample):
  * PSNR (dB), SSIM (windowed)            — overall fidelity
  * half-moon contrast ratio              — recon[band]-bg vs GT[band]-bg (~1 good)
  * corner edge-energy ratio              — sharpness kept at the corner (~1 good)
  * dot recall                            — fraction of speckle dots recovered

Run on 1 GPU (CPU fallback). Writes a metrics table + a GT|variant figure.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from tokamak_foundation_model.e2e.tokenizers.video import VideoTokenizer  # noqa: E402
from tokamak_foundation_model.e2e.output_heads import (  # noqa: E402
    VideoOutputHead, VideoFlowHead,
)

C_DEF, T_DEF, H_DEF, W_DEF = 7, 3, 120, 360
PATCH = (3, 12, 12)


# --------------------------------------------------------------------------- #
# Synthetic tangtv-like clips: half-moon band + sharp corner + dots           #
# --------------------------------------------------------------------------- #
def _halfmoon(H, W, cx, cy, r_in, r_out, a0, a1):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    th = np.arctan2(yy - cy, xx - cx)
    return ((rr >= r_in) & (rr <= r_out) & (th >= a0) & (th <= a1)).astype(np.float32)


def _corner(H, W, x0, y0, size):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    # sharp right-triangle wedge with two straight edges (sharp corner at x0,y0)
    return ((xx >= x0) & (yy >= y0) & ((xx - x0) + (yy - y0) <= size)).astype(np.float32)


def make_video_dataset(n_clips, C, T, H, W, missing_frac=0.15, seed=0):
    rng = np.random.default_rng(seed)
    X = np.full((n_clips, C, T, H, W), 0.10, np.float32)
    halfmoon = np.zeros((n_clips, T, H, W), bool)
    corner = np.zeros((n_clips, T, H, W), bool)
    dots = np.zeros((n_clips, C, T, H, W), bool)
    present = np.ones((n_clips, C, T), np.float32)
    for i in range(n_clips):
        cx = rng.uniform(0.35, 0.65) * W; cy = rng.uniform(0.0, 0.3) * H
        r_in = rng.uniform(0.35, 0.5) * H; r_out = r_in + rng.uniform(0.12, 0.22) * H
        a0 = rng.uniform(0.05, 0.25) * math.pi; a1 = a0 + rng.uniform(0.45, 0.7) * math.pi
        drift = rng.uniform(4, 12)                       # px/frame half-moon drift
        cs = rng.uniform(40, 80)                         # corner wedge size
        for t in range(T):
            hm = _halfmoon(H, W, cx + drift * t, cy, r_in, r_out, a0, a1)
            cn = _corner(H, W, int(0.04 * W), int(0.02 * H), cs)
            halfmoon[i, t] = hm > 0; corner[i, t] = cn > 0
            for c in range(C):
                amp = rng.uniform(0.75, 1.0) * (0.6 + 0.4 * (c / max(1, C - 1)))
                frame = X[i, c, t]
                frame[hm > 0] = amp
                frame[cn > 0] = max(frame.max(), amp * 0.95) if False else amp * 0.95
                # speckle dots
                nd = rng.integers(6, 16)
                ys = rng.integers(0, H, nd); xs = rng.integers(0, W, nd)
                frame[ys, xs] = 1.0; dots[i, c, t, ys, xs] = True
                X[i, c, t] = frame + rng.normal(0, 0.02, (H, W)).astype(np.float32)
        # missing channels/frames
        m = rng.random((C, T)) < missing_frac
        present[i][m] = 0.0
    X = np.clip(X, 0, 1.2)
    Xin = X.copy(); Xin[present[:, :, :, None, None].repeat(H, 3).repeat(W, 4) == 0] = 0.0
    return (torch.from_numpy(Xin).float(), torch.from_numpy(X).float(),
            torch.from_numpy(present).float(),
            torch.from_numpy(halfmoon), torch.from_numpy(corner), torch.from_numpy(dots))


# --------------------------------------------------------------------------- #
# Variants                                                                     #
# --------------------------------------------------------------------------- #
def build_variant(kind, C, T, H, W, d_model, base_ch, flow_steps, pe):
    tok = VideoTokenizer(n_channels=C, n_frames=T, patch_size=PATCH,
                         d_model=d_model, spatial_size=(H, W))
    if kind in ("deconv", "resize"):
        dec = VideoOutputHead(n_channels=C, n_frames=T, patch_size=PATCH,
                              d_model=d_model, spatial_size=(H, W),
                              decoder=("resize_conv" if kind == "resize" else "deconv"))
    elif kind == "flow":
        dec = VideoFlowHead(n_channels=C, n_frames=T, patch_size=PATCH, d_model=d_model,
                            spatial_size=(H, W), flow_base_ch=base_ch,
                            flow_sample_steps=flow_steps,
                            flow_h_pe_ch=pe, flow_w_pe_ch=pe)
    elif kind == "flow_nope":
        dec = VideoFlowHead(n_channels=C, n_frames=T, patch_size=PATCH, d_model=d_model,
                            spatial_size=(H, W), flow_base_ch=base_ch,
                            flow_sample_steps=flow_steps, flow_h_pe_ch=0, flow_w_pe_ch=0)
    elif kind == "flow_ssig":
        dec = VideoFlowHead(n_channels=C, n_frames=T, patch_size=PATCH, d_model=d_model,
                            spatial_size=(H, W), flow_base_ch=base_ch,
                            flow_sample_steps=flow_steps, flow_h_pe_ch=pe, flow_w_pe_ch=pe,
                            sigma_spatial=True)
    else:
        raise ValueError(kind)
    return tok, dec


def _perturb(toks, token_noise):
    # Simulate the imperfect tokens the 48-layer backbone hands the decoder
    # (clean tokenizer output is reconstructed near-perfectly by ANY decoder, so
    # it cannot distinguish them). Additive Gaussian, scaled by the token std.
    if token_noise <= 0:
        return toks
    return toks + token_noise * toks.detach().std() * torch.randn_like(toks)


def train(tok, dec, Xin, Xtgt, present, kind, steps, lr, device, tag="", token_noise=0.0):
    tok.train(); dec.train()
    B, C, T, H, W = Xin.shape
    opt = torch.optim.Adam(list(tok.parameters()) + list(dec.parameters()), lr=lr)
    Xin, Xtgt, present = Xin.to(device), Xtgt.to(device), present.to(device)
    tgt_dec = Xtgt.permute(0, 2, 1, 3, 4)                 # (B,T,C,H,W) to match decoder
    mask_tc = present.permute(0, 2, 1)                    # (B,T,C)
    mexp = mask_tc[:, :, :, None, None].expand(B, T, C, H, W)   # masked-MAE over present pixels
    is_flow = kind.startswith("flow")
    if is_flow:
        with torch.no_grad():
            r = tgt_dec.reshape(B, T * C, H, W)
            if dec.sigma_pb.shape[-1] == 1:               # per-folded-channel scalar
                sig = r.std(dim=(0, 2, 3), unbiased=False).clamp_min(0.05)
            else:                                          # spatial per-pixel (C·T,H,W)
                sig = r.std(dim=0, unbiased=False).clamp_min(0.05)
            dec.set_sigma_pb(sig.to(device))
    for s in range(steps):
        opt.zero_grad(set_to_none=True)
        toks = _perturb(tok(Xin), token_noise)             # tokenizer wants (B,C,T,H,W)
        if is_flow:
            mu = dec.mean_head(toks)
            mae = (((mu - tgt_dec).abs()) * mexp).sum() / mexp.sum().clamp_min(1.0)
            flow = dec.flow_loss(toks, mu, tgt_dec, mask=mask_tc)
            loss = mae + dec.flow_lambda * flow
        else:
            out = dec(toks)
            loss = (((out - tgt_dec).abs()) * mexp).sum() / mexp.sum().clamp_min(1.0)
            flow = torch.tensor(0.0)
        loss.backward(); opt.step()
        if (s + 1) % 500 == 0 or s == 0:
            print(f"  [{tag}] step {s+1}/{steps} loss={loss.item():.4f}"
                  + (f" flow={flow.item():.4f}" if is_flow else ""), flush=True)
    return tok, dec


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #
def _ssim(r, g, drange, win=7):
    x = r[None, None]; y = g[None, None]; pad = win // 2
    blur = lambda z: F.avg_pool2d(z, win, 1, pad)
    mx, my = blur(x), blur(y)
    vx, vy = blur(x * x) - mx * mx, blur(y * y) - my * my
    cxy = blur(x * y) - mx * my
    c1, c2 = (0.01 * drange) ** 2, (0.03 * drange) ** 2
    return float((((2 * mx * my + c1) * (2 * cxy + c2)) /
                  ((mx * mx + my * my + c1) * (vx + vy + c2) + 1e-12)).mean())


@torch.no_grad()
def evaluate(tok, dec, Xin, Xtgt, present, halfmoon, corner, dots, kind, device,
             seed=0, token_noise=0.0):
    tok.eval(); dec.eval()
    B, C, T, H, W = Xin.shape
    torch.manual_seed(seed)                                          # reproducible token perturbation
    toks = _perturb(tok(Xin.to(device)), token_noise)               # SAME noisy tokens for all decoders
    is_flow = kind.startswith("flow")
    mu = dec.mean_head(toks).cpu() if is_flow else dec(toks).cpu()   # (B,T,C,H,W)
    if is_flow:
        torch.manual_seed(seed + 7); samp = dec.sample(toks, dec.mean_head(toks)).cpu()
    else:
        samp = mu
    tgt = Xtgt.permute(0, 2, 1, 3, 4)                                # (B,T,C,H,W)
    pres = present.permute(0, 2, 1).bool()                          # (B,T,C)
    hm = halfmoon[:, :, None].expand(B, T, C, H, W)                 # (B,T,C,H,W)
    cn = corner[:, :, None].expand(B, T, C, H, W)
    dt = dots.permute(0, 2, 1, 3, 4)                                # (B,T,C,H,W)
    dr = float(tgt.max() - tgt.min())

    def _tv(z):                                          # total variation (H+W)
        return (z.diff(dim=0).abs().mean() + z.diff(dim=1).abs().mean()).item()

    def _seam(z, ph=PATCH[1], pw=PATCH[2]):
        # Gradient energy AT the patch-grid boundaries vs the interior. A
        # per-patch decoder (deconv) fed imperfect tokens produces blocks that
        # don't align -> boundary >> interior (checkerboard). GT / an
        # overlapping decoder -> ~1. >1.3 reads as visible grid.
        gh = z.diff(dim=0).abs(); gw = z.diff(dim=1).abs()
        brow = [i * ph - 1 for i in range(1, z.shape[0] // ph) if i * ph - 1 < gh.shape[0]]
        bcol = [j * pw - 1 for j in range(1, z.shape[1] // pw) if j * pw - 1 < gw.shape[1]]
        if not brow or not bcol:
            return float("nan")
        bound = 0.5 * (gh[brow, :].mean().item() + gw[:, bcol].mean().item())
        inter = 0.5 * (gh.mean().item() + gw.mean().item())
        return bound / inter if inter > 1e-8 else float("nan")

    def metrics(rec):
        ps, ss, hmc, cne, dre, tvr, grn, sem = [], [], [], [], [], [], [], []
        for b in range(B):
            for t in range(T):
                for c in range(C):
                    if not pres[b, t, c]:
                        continue
                    R = rec[b, t, c]; G = tgt[b, t, c]
                    mse = ((R - G) ** 2).mean().item()
                    ps.append(99.0 if mse <= 1e-9 else 10 * math.log10(dr * dr / mse))
                    ss.append(_ssim(R, G, dr))
                    sem.append(_seam(R))
                    hmask = hm[b, t, c]; bg = ~(hmask | cn[b, t, c] | dt[b, t, c])
                    gc = G[hmask].mean().item() - G[bg].mean().item()
                    rc = R[hmask].mean().item() - R[bg].mean().item()
                    if abs(gc) > 1e-6:
                        hmc.append(rc / gc)
                    # global sharpness preservation (TV ratio): blur -> <1, grain
                    # -> >1, ~1 = matched. Not fooled by stochastic pixel mismatch.
                    gtv = _tv(G)
                    if gtv > 1e-6:
                        tvr.append(_tv(R) / gtv)
                    # background grain: recon std vs GT std in the QUIET region.
                    # ~1 = clean (matches GT noise floor), >>1 = injected grain.
                    gstd = G[bg].std().item()
                    if gstd > 1e-6:
                        grn.append(R[bg].std().item() / gstd)
                    # corner sharpness: edge energy ratio in corner bbox
                    cmask = cn[b, t, c]
                    if cmask.any():
                        ge = _tv(G * cmask); re = _tv(R * cmask)
                        if ge > 1e-6:
                            cne.append(re / ge)
                    # dot recall: GT dot pixels recovered as locally-bright in R
                    dmask = dt[b, t, c]
                    if dmask.any():
                        thr = R.mean().item() + 2 * R.std().item()
                        dre.append(float((R[dmask] > thr).float().mean()))
        f = lambda a: float(np.mean(a)) if a else float("nan")
        return dict(psnr=f(ps), ssim=f(ss), halfmoon=f(hmc), corner=f(cne),
                    dot=f(dre), tvr=f(tvr), grain=f(grn), seam=f(sem))

    return metrics(mu.float()), metrics(samp.float()), mu, samp, tgt


_PRETTY = {"deconv": "deconv\n(OLD, checkerboard)", "resize": "resize-conv\n(B, deterministic)",
           "flow": "flow\n(A+D, scalar σ)", "flow_ssig": "flow_ssig\n(A+D, spatial σ)",
           "flow_nope": "flow\n(A, no PE)"}


def save_figure(tgt, recons, out_path, results=None):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    labels = list(recons.keys())
    cols = ["Ground truth"] + [_PRETTY.get(l, l) for l in labels]
    rows = min(3, tgt.shape[0])
    fig, ax = plt.subplots(rows, len(cols), figsize=(2.7 * len(cols), 2.1 * rows),
                           squeeze=False)
    for ri in range(rows):
        t, c = 1, 0                                  # mid-frame, channel 0
        panels = [tgt[ri, t, c]] + [recons[l][ri, t, c] for l in labels]
        vmax = float(tgt[ri, t, c].max())
        for ci, p in enumerate(panels):
            a = ax[ri, ci]
            a.imshow(np.asarray(p), aspect="auto", vmin=0, vmax=vmax, cmap="inferno")
            if ri == 0:
                a.set_title(cols[ci], fontsize=9)
            # annotate variant columns with the deciding metrics on the first row
            if ri == 0 and ci > 0 and results is not None:
                k = labels[ci - 1]
                m = results[k][1] if k.startswith("flow") else results[k][0]
                a.set_xlabel(f"½moon {m['halfmoon']:.2f}  seam {m['seam']:.2f}  "
                             f"grain {m['grain']:.2f}", fontsize=7.0)
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle("tangtv video reconstruction — GT vs decoders (300 tokens @ d1024, missing data)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=110); fig.savefig(out_path.replace(".png", ".pdf"))
    print(f"[figure] wrote {out_path}", flush=True)


def save_metric_chart(results, out_path):
    """Per-structure metric bars. Ideal = 1.0 line for ratios; mean-collapse
    shows as half-moon/TVr near 0, grain shows as a tall grain bar."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    keys = list(results)
    fields = [("halfmoon", "half-moon\ncontrast"), ("corner", "corner\nsharpness"),
              ("dot", "dot\nrecall"), ("tvr", "TV ratio\n(sharpness)"),
              ("grain", "bg grain\n(1=clean)"), ("seam", "patch seam\n(1=no grid)"),
              ("ssim", "SSIM")]
    fig, axes = plt.subplots(1, len(fields), figsize=(2.3 * len(fields), 3.4))
    colors = {"deconv": "#888", "resize": "#d62728", "flow": "#1f77b4",
              "flow_ssig": "#2ca02c", "flow_nope": "#9467bd"}
    for ax, (fk, title) in zip(axes, fields):
        vals = [(results[k][1] if k.startswith("flow") else results[k][0])[fk] for k in keys]
        ax.bar(range(len(keys)), vals, color=[colors.get(k, "#555") for k in keys])
        if fk in ("halfmoon", "corner", "dot", "tvr", "grain", "seam"):
            ax.axhline(1.0, ls="--", lw=0.8, color="k", alpha=0.6)
        ax.set_title(title, fontsize=9)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=7)
    fig.suptitle("Video decoder comparison — per-structure metrics "
                 "(ratios: 1.0 = GT-matched)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=110); fig.savefig(out_path.replace(".png", ".pdf"))
    print(f"[figure] wrote {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="eval_runs/video_test")
    ap.add_argument("--n_clips", type=int, default=16)
    ap.add_argument("--d_model", type=int, default=1024)
    ap.add_argument("--base_ch", type=int, default=48)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--flow_steps", type=int, default=8)
    ap.add_argument("--pe", type=int, default=16)
    ap.add_argument("--missing_frac", type=float, default=0.15)
    ap.add_argument("--variants", default="deconv,resize,flow")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--token_noise", type=float, default=0.0,
                    help="Gaussian token perturbation (× token std) at train+eval, "
                         "simulating the imperfect tokens the backbone hands the "
                         "decoder. 0 = clean autoencoder (any decoder reconstructs).")
    ap.add_argument("--token_noise_list", default="",
                    help="Comma-sep noise levels for a robustness sweep in ONE run "
                         "(e.g. 0,0.5,1.0); each writes to <out_dir>/noise<level>/. "
                         "Overrides --token_noise when set.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    C, T, H, W = C_DEF, T_DEF, H_DEF, W_DEF
    noise_levels = ([float(x) for x in args.token_noise_list.split(",") if x.strip()]
                    if args.token_noise_list.strip() else [args.token_noise])
    print(f"[setup] device={device} d_model={args.d_model} base_ch={args.base_ch} "
          f"steps={args.steps} n_clips={args.n_clips} missing={args.missing_frac} "
          f"noise_levels={noise_levels} variants={args.variants}", flush=True)
    # One dataset, shared across all noise levels (apples-to-apples).
    data = make_video_dataset(args.n_clips, C, T, H, W, args.missing_frac, args.seed)
    print(f"[data] X={tuple(data[0].shape)} present_frac={float(data[2].mean()):.3f} "
          f"(300 tokens, patch {PATCH})", flush=True)

    per_noise = {}
    for nz in noise_levels:
        od = (os.path.join(args.out_dir, f"noise{nz:g}") if len(noise_levels) > 1
              else args.out_dir)
        os.makedirs(od, exist_ok=True)
        print(f"\n{'#' * 92}\n#### TOKEN_NOISE = {nz}  ->  {od}\n{'#' * 92}", flush=True)
        per_noise[nz] = _run_one(nz, od, data, args, device, C, T, H, W)
    if len(noise_levels) > 1:
        save_robustness_grid(per_noise, noise_levels,
                             os.path.join(args.out_dir, "robustness_grid.png"))
    print("=== VIDEO TEST DONE ===", flush=True)


def _run_one(token_noise, out_dir, data, args, device, C, T, H, W):
    Xin, Xtgt, present, hm, cn, dt = data
    hdr = (f"{'variant':<12} | {'PSNR':>6} | {'SSIM':>6} | {'half-moon':>9} | "
           f"{'corner':>7} | {'dot':>5} | {'TVr':>5} | {'grain':>6} | {'seam':>5}")
    res_path = os.path.join(out_dir, "results.txt")
    # Incremental results file: each variant's row is appended the instant it is
    # evaluated, so a walltime clip never discards already-finished variants.
    with open(res_path, "w") as fh:
        fh.write(f"VIDEO RECON RESULTS (token_noise={token_noise}; "
                 "eval output: mu for deterministic, flow-sample for flow)\n")
        fh.write(hdr + "\n" + "-" * 92 + "\n")

    def _row(k, m):
        return (f"{k:<12} | {m['psnr']:>6.2f} | {m['ssim']:>6.3f} | {m['halfmoon']:>9.3f} | "
                f"{m['corner']:>7.3f} | {m['dot']:>5.3f} | {m['tvr']:>5.2f} | {m['grain']:>6.2f} | "
                f"{m['seam']:>5.2f}")

    results, recons = {}, {}
    for kind in [k.strip() for k in args.variants.split(",") if k.strip()]:
        tok, dec = build_variant(kind, C, T, H, W, args.d_model, args.base_ch,
                                 args.flow_steps, args.pe)
        tok.to(device); dec.to(device)
        npar = sum(p.numel() for p in tok.parameters()) + sum(p.numel() for p in dec.parameters())
        print(f"\n=== {kind}  ({npar/1e6:.2f}M params, noise={token_noise}) ===", flush=True)
        train(tok, dec, Xin, Xtgt, present, kind, args.steps, args.lr, device, tag=kind,
              token_noise=token_noise)
        m_mu, m_sp, mu, sp, tgt = evaluate(tok, dec, Xin, Xtgt, present, hm, cn, dt,
                                           kind, device, seed=args.seed,
                                           token_noise=token_noise)
        results[kind] = (m_mu, m_sp); recons[kind] = sp   # show the eval output (sample for flow)
        row = _row(kind, results[kind][1] if kind.startswith("flow") else results[kind][0])
        print("  [result] " + row, flush=True)            # immediate, per-variant
        with open(res_path, "a") as fh:
            fh.write(row + "\n")
        save_figure(tgt, recons, os.path.join(out_dir, "video_recon.png"),
                    results=results)                       # refresh fig each variant

    print("\n" + "=" * 92)
    print(f"VIDEO RECON RESULTS  token_noise={token_noise}  "
          "(eval output: mu for deterministic, flow-sample for flow)")
    print("=" * 92)
    print(hdr)
    print("-" * 92)
    for k in results:
        m = results[k][1] if k.startswith("flow") else results[k][0]
        print(_row(k, m))
    print("-" * 92)
    print("(half-moon/corner ~1.0 = contrast/sharpness preserved; dot-recall = speckle recovered;"
          " TVr ~1 = sharpness matched; grain ~1 = bg clean; seam ~1 = no patch grid)")
    save_figure(tgt, recons, os.path.join(out_dir, "video_recon.png"), results=results)
    save_metric_chart(results, os.path.join(out_dir, "video_metrics.png"))
    return dict(tgt=tgt, recons=recons, results=results)


def save_robustness_grid(per_noise, noise_levels, out_path):
    """Degradation-under-noise grid: rows = token-noise level, columns =
    GT + each decoder. Same clip/frame/channel everywhere. This is THE figure:
    you can read the checkerboard / blur / collapse appear as noise rises."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    labels = list(next(iter(per_noise.values()))["recons"].keys())
    cols = ["Ground truth"] + [_PRETTY.get(l, l) for l in labels]
    rows = len(noise_levels)
    fig, ax = plt.subplots(rows, len(cols), figsize=(2.7 * len(cols), 2.2 * rows),
                           squeeze=False)
    ri = 0
    for nz in noise_levels:
        d = per_noise.get(nz)
        if d is None:
            continue
        tgt = d["tgt"]; recons = d["recons"]; results = d["results"]
        t, c, clip = 1, 0, 0                              # fixed mid-frame/ch/clip
        vmax = float(tgt[clip, t, c].max())
        panels = [tgt[clip, t, c]] + [recons[l][clip, t, c] for l in labels]
        for ci, p in enumerate(panels):
            a = ax[ri, ci]
            a.imshow(np.asarray(p), aspect="auto", vmin=0, vmax=vmax, cmap="inferno")
            if ri == 0:
                a.set_title(cols[ci], fontsize=9)
            if ci == 0:
                a.set_ylabel(f"token_noise\n{nz:g}", fontsize=9)
            if ci > 0:
                k = labels[ci - 1]
                m = results[k][1] if k.startswith("flow") else results[k][0]
                a.set_xlabel(f"seam {m['seam']:.2f}  ½m {m['halfmoon']:.2f}",
                             fontsize=6.8)
            a.set_xticks([]); a.set_yticks([])
        ri += 1
    fig.suptitle("Video decoder robustness to imperfect (backbone-like) tokens "
                 "— degradation as token noise rises", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=110); fig.savefig(out_path.replace(".png", ".pdf"))
    print(f"[figure] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
