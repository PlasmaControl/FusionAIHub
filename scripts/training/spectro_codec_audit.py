"""Codec AUDIT — the two questions the reconstruction benchmark can't answer.

Given a FROZEN spectro FSQ codec, on a real shot:

(1) CODE HISTOGRAM (imbalance).  Encode many GT windows -> per-dim int codes.
    Report, per dim, the coverage of the single most-common level (peaked
    marginals => argmax collapses to background) and the per-dim entropy; and
    the coverage of the single most-common *token code-tuple* (background
    dominance). This is the quantitative version of "a handful of codes cover
    >95% of tokens => the categorical head will never commit to mode codes".

(2) FAITHFULNESS SPLICE TEST (causal code control).  Reconstruction proves the
    codec can REPRESENT a mode; it does NOT prove the codes CAUSALLY control the
    rendered mode (a GAN decoder can hallucinate texture from patch context).
    Test: take a mode-POSITIVE window and a mode-FREE window; splice the codes of
    the mode's frequency-patch row(s) from the positive grid into the free grid;
    decode. If the mode renders at the right frequency in the FOREIGN context,
    codes causally control mode content and the world model's job is well-posed.
    If not, code-prediction accuracy will not correlate with mode accuracy and
    the codec must be fixed BEFORE any world-model work.

Env: MODALITIES (csv), SHOT, CODEC_DIR, NWIN, MODE_K, OUT_DIR, BG_SIGMA.
Runs on 1 GPU (falls back to CPU). No world model involved — codec + data only.
"""
import os
import sys
from pathlib import Path

FMH = "/lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub"
for p in (f"{FMH}/src", f"{FMH}/scripts/training"):
    if p not in sys.path:
        sys.path.insert(0, p)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from poc_fsq_stageB import load_pairs, _hard
from spectro_bg import baseline_residual
from tokamak_foundation_model.e2e.quantizers.spectro_codec import load_frozen_codec

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODS = [m.strip() for m in os.environ.get("MODALITIES", "ece,co2,bes,mhr").split(",") if m.strip()]
SHOT = os.environ.get("SHOT", "200729")
CODEC_DIR = os.environ.get("CODEC_DIR", "/lustre/orion/fus187/proj-shared/models/fsq_resid_p8_all")
DATA = os.environ.get("DATA_DIR", "/lustre/orion/fus187/proj-shared/foundation_model")
STATS = os.environ.get("STATS_PATH",
                       "/lustre/orion/fus187/proj-shared/foundation_model_meta/preprocessing_stats.pt")
NWIN = int(os.environ.get("NWIN", "80"))
MODE_K = float(os.environ.get("MODE_K", "2.5"))
BG_SIGMA = float(os.environ.get("BG_SIGMA", "8.0"))
OUT = Path(os.environ.get("OUT_DIR", f"{FMH}/eval_runs/codec_audit"))
OUT.mkdir(parents=True, exist_ok=True)
FS, NFFT = 500_000.0, 1024


def prominence(prof, f, half=6):
    """Mode prominence at freq-bin f: value minus local-background median."""
    lo, hi = max(0, f - 3 * half), min(len(prof), f + 3 * half + 1)
    bg = np.median(np.concatenate([prof[lo:max(lo, f - half)], prof[min(hi, f + half + 1):hi]]))
    return float(prof[f] - bg)


def audit_modality(MOD):
    print(f"\n===================== AUDIT {MOD} (shot {SHOT}) =====================", flush=True)
    codec, cfg = load_frozen_codec(f"{CODEC_DIR}/spectro_codec_{MOD}.pt", map_location="cpu")
    codec = codec.to(dev)
    C, Fq, patch_f = int(cfg["C"]), int(cfg["Fq"]), int(cfg.get("patch_f", 8))
    bg = bool(cfg.get("bg_subtract", False))
    npf = Fq // patch_f
    dim = int(cfg["fsq_dim"])
    L = int(cfg["fsq_L"])

    X, _ = load_pairs(SHOT, DATA, STATS, C, NWIN, modality=MOD)          # (N,C,F,T)
    X = X.to(dev)
    if bg:
        B, R = baseline_residual(X, sigma=BG_SIGMA)
        enc_in = R.to(dev)                                              # codec sees residual
    else:
        B = torch.zeros_like(X); enc_in = X

    # ---------- (1) CODE HISTOGRAM ----------
    with torch.no_grad():
        codes = torch.cat([codec.encode_codes(enc_in[i:i + 16]) for i in range(0, enc_in.shape[0], 16)], 0)
    codes = codes.cpu().long()                                          # (N, n_tok, dim)
    Ntok = codes.shape[0] * codes.shape[1]
    flat = codes.reshape(-1, dim).numpy()                              # (N*n_tok, dim)
    # per-dim: coverage of the most-common level + entropy
    per_dim_top1, per_dim_ent = [], []
    for d in range(dim):
        counts = np.bincount(flat[:, d], minlength=L).astype(np.float64)
        p = counts / counts.sum()
        per_dim_top1.append(p.max())
        per_dim_ent.append(float(-(p[p > 0] * np.log2(p[p > 0])).sum()))
    # most-common token code-tuple coverage (background dominance)
    view = np.ascontiguousarray(flat).view([('', flat.dtype)] * dim).ravel()
    _, cnts = np.unique(view, return_counts=True)
    top_tuple_cov = cnts.max() / cnts.sum()
    top10_tuple_cov = np.sort(cnts)[::-1][:10].sum() / cnts.sum()
    n_unique = len(cnts)
    print(f"[hist] {MOD}: tokens={Ntok}  dim={dim} L={L}  unique_tuples={n_unique}", flush=True)
    print(f"[hist] {MOD}: per-dim mean top-1-level coverage={np.mean(per_dim_top1):.3f} "
          f"(max {np.max(per_dim_top1):.3f})  mean per-dim entropy={np.mean(per_dim_ent):.2f}/{np.log2(L):.2f} bits", flush=True)
    print(f"[hist] {MOD}: most-common code-TUPLE covers {100*top_tuple_cov:.1f}% of tokens; "
          f"top-10 tuples cover {100*top10_tuple_cov:.1f}%  ==> "
          f"{'SEVERE imbalance (argmax->background)' if top_tuple_cov>0.5 else ('notable imbalance' if top_tuple_cov>0.2 else 'not tuple-dominated')}", flush=True)

    # ---------- (2) FAITHFULNESS SPLICE TEST ----------
    Xn = X.cpu().numpy()
    hard = _hard(X, MODE_K).cpu().numpy()                              # (N,C,F,T) binary mode mask
    ch = int(hard.sum(axis=(0, 2, 3)).argmax())                       # strongest-mode channel
    win_mode = hard[:, ch].sum(axis=(1, 2))                           # per-window mode pixel count
    w_pos = int(win_mode.argmax())                                    # mode-positive window
    w_free = int(win_mode.argmin())                                   # mode-free window
    # mode frequency (bin) in the positive window on ch, via high-pass profile
    prof_pos = np.abs(Xn[w_pos, ch]).mean(1)
    hp = prof_pos - np.convolve(prof_pos, np.ones(9) / 9, mode="same")
    f_mode = int(np.argmax(hp[5:]) + 5)
    mode_patch = f_mode // patch_f
    freqs = np.arange(Fq) * FS / NFFT / 1e3

    with torch.no_grad():
        c_pos = codec.encode_codes(enc_in[w_pos:w_pos + 1]).cpu()     # (1,n_tok,dim)
        c_free = codec.encode_codes(enc_in[w_free:w_free + 1]).cpu()
        npt = c_pos.shape[1] // npf
        gp = c_pos.reshape(1, npf, npt, dim)
        gf = c_free.reshape(1, npf, npt, dim)
        spliced = gf.clone()
        spliced[:, mode_patch] = gp[:, mode_patch]                    # graft the mode's freq-patch row
        def dec(grid):
            r = codec.decode_codes(grid.reshape(1, npf * npt, dim).to(dev)).cpu()
            return (r + B[w_free:w_free + 1].cpu()) if bg else r       # recombine bg of the HOST (free) window
        r_pos = (codec.decode_codes(c_pos.to(dev)).cpu() + (B[w_pos:w_pos + 1].cpu() if bg else 0))
        r_free = dec(gf)
        r_spl = dec(spliced)
    # prominence at the mode freq on ch (residual space to isolate the mode)
    def hp_prof(arr4, w=0):
        p = np.abs(arr4[w, ch].numpy()).mean(1)
        return p - np.convolve(p, np.ones(9) / 9, mode="same")
    pr_pos = prominence(hp_prof(r_pos), f_mode)
    pr_free = prominence(hp_prof(r_free), f_mode)
    pr_spl = prominence(hp_prof(r_spl), f_mode)
    ratio = pr_spl / pr_pos if abs(pr_pos) > 1e-9 else float("nan")
    verdict = ("FAITHFUL: codes causally control the mode" if ratio > 0.5
               else ("PARTIAL" if ratio > 0.2 else "UNFAITHFUL: decoder ignores spliced codes (fix CODEC first)"))
    print(f"[splice] {MOD}: ch={ch} f_mode={freqs[f_mode]:.1f}kHz patch={mode_patch}  "
          f"prominence  pos={pr_pos:.3f}  free={pr_free:.3f}  spliced={pr_spl:.3f}  "
          f"spliced/pos={ratio:.2f}  ==> {verdict}", flush=True)

    # figure: mode+ recon | mode-free recon | free+spliced recon (ch), + profile overlay
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.4))
    fmax = min(Fq, int(80 / (FS / NFFT / 1e3)))
    for a, (ttl, arr) in zip(ax[:3], [
            (f"mode+ recon (w{w_pos})", r_pos), (f"mode-free recon (w{w_free})", r_free),
            (f"free + spliced mode-codes", r_spl)]):
        a.imshow(np.abs(arr[0, ch, :fmax]), origin="lower", aspect="auto",
                 extent=[0, arr.shape[-1], 0, freqs[fmax]])
        a.axhline(freqs[f_mode], color="cyan", lw=0.6, ls="--")
        a.set_title(ttl, fontsize=9); a.set_ylabel("kHz")
    ax[3].plot(freqs[:fmax], hp_prof(r_pos)[:fmax], label="mode+", color="k")
    ax[3].plot(freqs[:fmax], hp_prof(r_free)[:fmax], label="free", color="tab:green")
    ax[3].plot(freqs[:fmax], hp_prof(r_spl)[:fmax], label="free+spliced", color="tab:orange")
    ax[3].axvline(freqs[f_mode], color="cyan", lw=0.6, ls="--")
    ax[3].legend(fontsize=7); ax[3].set_title(f"HP profile @ ch{ch}  spliced/pos={ratio:.2f}", fontsize=9)
    fig.suptitle(f"Codec faithfulness splice — {MOD.upper()} {SHOT}  ({verdict})", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / f"splice_{MOD}.png", dpi=110); plt.close(fig)
    print(f"[splice] {MOD}: saved {OUT}/splice_{MOD}.png", flush=True)


for MOD in MODS:
    try:
        audit_modality(MOD)
    except Exception as e:
        import traceback
        print(f"[WARN] {MOD} audit failed: {e}", flush=True)
        traceback.print_exc()

print("\n[codec_audit] done", flush=True)
