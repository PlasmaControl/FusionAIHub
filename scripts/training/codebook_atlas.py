"""Codebook ATLAS figure for a frozen FSQ spectrogram codec (vocabulary view).

FSQ has no enumerable codebook (each of n_tok tokens is a `dim`-D vector, each
dim snapped to `L` levels -> L**dim possible codes). What IS meaningful: the
codes that actually OCCUR in real data cluster into a small vocabulary of
time-frequency motifs. This builds:

  (A) ATLAS  - ~K representative used-codes. Each tile is the codec's REAL
               reconstruction of the patch that produced that code (medoid token
               per cluster, cropped in-context on its window's dominant-mode
               channel). Each tile spans ~15.6 kHz x 8.2 ms.
  (B) USAGE  - (dim x L) utilization heatmap: perplexity + % dead cells.

Mode-reconstruction EVIDENCE lives in the curated high-mode figures
(eval_runs/codec_highmode/highmode_<mod>_<shot>.png), NOT here — this figure is
the vocabulary + utilization only. Parameterized by MODALITY (env).
"""
import math
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
from scipy.cluster.vq import kmeans2
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

import poc_fsq_stageB as poc
from poc_fsq_stageB import FSQAutoencoder, load_pairs, _hard

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MOD = os.environ.get("MODALITY", "ece")
CODEC = os.environ.get(
    "CODEC_PATH",
    f"/lustre/orion/fus187/proj-shared/models/fsq_spectro_codecs_tok96/spectro_codec_{MOD}.pt")
DATA = os.environ.get("DATA_DIR", "/lustre/orion/fus187/proj-shared/foundation_model")
STATS = os.environ.get(
    "STATS_PATH", "/lustre/orion/fus187/proj-shared/foundation_model_meta/preprocessing_stats.pt")
SHOTS = [s.strip() for s in os.environ.get("SHOTS", "200729").split(",") if s.strip()]
NWIN = int(os.environ.get("NWIN_PER_SHOT", "60"))
K = int(os.environ.get("K_CLUSTERS", "120"))
MODE_K = float(os.environ.get("MODE_K", "2.5"))           # _hard threshold (ECE 2.5)
OUT = Path(os.environ.get("OUT_DIR", f"{FMH}/eval_runs/codebook_atlas/{MOD}"))
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.RandomState(0)

# STFT calibration (eval_e2e_animation_tokamak.py:360): fs=500 kHz, n_fft=1024, hop=256.
FS, NFFT, HOP = 500_000.0, 1024, 256

# ---- load frozen codec ----
ck = torch.load(CODEC, map_location="cpu", weights_only=False)
cfg = ck["cfg"]
poc.PATCH_F = int(cfg["patch_f"]); poc.PATCH_T = int(cfg["patch_t"]); poc.D_MODEL = int(cfg["d_model"])
C, Fq, Tq, DIM, L = cfg["C"], cfg["Fq"], cfg["Tq"], cfg["fsq_dim"], cfg["fsq_L"]
PF, PT = int(cfg["patch_f"]), int(cfg["patch_t"])
NPF, NPT = Fq // PF, Tq // PT
ae = FSQAutoencoder(C, Fq, Tq, DIM, L, per_channel=cfg.get("per_channel", False)).to(dev)
ae.load_state_dict(ck["ae"]); ae.eval()
FREQ_KHZ = np.arange(Fq) * FS / NFFT / 1e3
TIME_MS = np.arange(Tq) * HOP / FS * 1e3
print(f"[atlas] {MOD}: C={C} F={Fq} T={Tq} n_tok={ae.n_tok} dim={DIM} L={L} "
      f"grid={NPF}x{NPT} patch~{FREQ_KHZ[PF]:.1f}kHz x {TIME_MS[PT]:.1f}ms", flush=True)

# ---- collect real windows ----
Xs = []
for sh in SHOTS:
    try:
        xi, _ = load_pairs(sh, DATA, STATS, C, NWIN, modality=MOD)
        if xi.numel():
            Xs.append(xi); print(f"[atlas] shot {sh}: {tuple(xi.shape)}", flush=True)
    except Exception as e:
        print(f"[atlas] shot {sh} FAILED: {str(e)[:100]}", flush=True)
assert Xs, "no data loaded"
X = torch.cat(Xs, 0)
N = X.shape[0]
print(f"[atlas] total windows {N}", flush=True)

# ---- encode -> codes, decode -> recon, GT mode mask (for the atlas display channel) ----
codes, REC, MG = [], [], []
with torch.no_grad():
    for i in range(0, N, 16):
        xb = X[i:i + 16].to(dev)
        cb = ae.encode_codes(xb)
        codes.append(cb.cpu()); REC.append(ae.decode_codes(cb).cpu())
        MG.append(_hard(xb, MODE_K).cpu())
codes = torch.cat(codes, 0)
REC = torch.cat(REC, 0).numpy()
ch_mode = torch.cat(MG, 0).numpy().sum(axis=(2, 3)).argmax(axis=1)     # per-window dominant-mode channel

# ---- cluster used codes; medoid token -> (window, token) ----
tok_flat = codes.reshape(-1, DIM).numpy().astype(np.int64)
NT = tok_flat.shape[0]
sub = rng.choice(NT, min(NT, 80000), replace=False)
data = tok_flat[sub].astype(np.float64)
cent, lab = kmeans2(data, K, minit="++", seed=0, missing="warn")
med_flat, med_cent = [], []
for c in range(K):
    m = np.where(lab == c)[0]
    if not len(m):
        continue
    d = ((data[m] - cent[c]) ** 2).sum(1)
    med_flat.append(int(sub[m[int(d.argmin())]])); med_cent.append(cent[c])
med_cent = np.array(med_cent); Kk = len(med_flat)
print(f"[atlas] non-empty clusters: {Kk}/{K}", flush=True)


def real_recon_patch(flat):
    w, tk = flat // ae.n_tok, flat % ae.n_tok
    pf, pt = tk // NPT, tk % NPT
    return REC[w, ch_mode[w], pf * PF:(pf + 1) * PF, pt * PT:(pt + 1) * PT]


patches = np.stack([real_recon_patch(f) for f in med_flat])

# ---- 2D layout: PCA of medoid codes -> Hungarian snap to a grid ----
Z = med_cent - med_cent.mean(0)
_, _, Vt = np.linalg.svd(Z, full_matrices=False)
xy = Z @ Vt[:2].T
xy = (xy - xy.min(0)) / (np.ptp(xy, 0) + 1e-9)
cols = int(math.ceil(math.sqrt(Kk))); rows = int(math.ceil(Kk / cols))
gx, gy = np.meshgrid(np.linspace(0, 1, cols), np.linspace(0, 1, rows))
grid = np.stack([gx.ravel(), gy.ravel()], 1)
ri, ci = linear_sum_assignment(cdist(xy, grid))
cell2clust = {int(c): int(r) for r, c in zip(ri, ci)}

# ---- utilization over ALL real tokens ----
usage = np.stack([np.bincount(tok_flat[:, d], minlength=L)[:L] for d in range(DIM)]).astype(float)
pnorm = usage / usage.sum(1, keepdims=True).clip(1e-9)
perpl = np.exp(-(pnorm * np.log(pnorm + 1e-12)).sum(1))
dead_frac = float((usage == 0).mean())

# ================= FIGURE (vocabulary + utilization) =================
pv = np.percentile(patches, [2, 98]); gp = 2
canvas = np.full((rows * (PF + gp), cols * (PT + gp)), np.nan)
for c in range(rows * cols):
    if c not in cell2clust:
        continue
    r, cc = divmod(c, cols)
    canvas[r * (PF + gp):r * (PF + gp) + PF, cc * (PT + gp):cc * (PT + gp) + PT] = patches[cell2clust[c]]

fig = plt.figure(figsize=(15, 12))
gs = fig.add_gridspec(2, 1, height_ratios=[3.1, 1.0], hspace=0.16)
axA = fig.add_subplot(gs[0])
axA.imshow(np.ma.masked_invalid(canvas), origin="lower", aspect="auto", cmap="magma",
           vmin=pv[0], vmax=pv[1])
axA.set_title(f"(A) Codebook atlas — {MOD.upper()}: {Kk} representative used-codes\n"
              f"each tile = codec recon of a real {PF}x{PT} patch "
              f"(~{FREQ_KHZ[PF]:.1f} kHz x {TIME_MS[PT]:.1f} ms); PCA layout, neighbours similar",
              fontsize=12)
axA.set_xticks([]); axA.set_yticks([])
axB = fig.add_subplot(gs[1])
im = axB.imshow(usage.T, origin="lower", aspect="auto", cmap="viridis")
axB.set_title(f"(B) Code utilization — mean perplexity {perpl.mean():.1f}/{L} levels, "
              f"{100*dead_frac:.0f}% dead cells", fontsize=11)
axB.set_xlabel(f"latent dim (0..{DIM-1})"); axB.set_ylabel(f"level (0..{L-1})")
fig.colorbar(im, ax=axB, fraction=0.02, label="count")
fig.suptitle(f"FSQ codec codebook — {MOD.upper()}  (n_tok={ae.n_tok}, dim={DIM}, L={L}; "
             f"code space {L}^{DIM}; fs={FS/1e3:.0f} kHz).  "
             f"Mode-reconstruction evidence: eval_runs/codec_highmode/",
             fontsize=12, y=0.995)
for extn in ("png", "pdf"):
    fig.savefig(OUT / f"codebook_atlas_{MOD}.{extn}", dpi=140, bbox_inches="tight")
print(f"[atlas] saved {OUT}/codebook_atlas_{MOD}.png (+pdf)  perplexity={perpl.mean():.2f} "
      f"dead={dead_frac:.3f}", flush=True)
