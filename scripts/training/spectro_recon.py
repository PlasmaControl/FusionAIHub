"""Real spectrogram reconstruction: GT vs the PRODUCTION FSQ codec, one real shot.

Loads a real shot's spectrogram, runs it through the frozen production codec
(encode -> FSQ -> decode), and renders GT | codec reconstruction | difference on
the strongest-mode channel, over a CONTIGUOUS time span (non-overlapping windows
stitched), with physical Time (ms) / Frequency (kHz) axes and reconstruction corr.
Env: MODALITY, SHOT, CODEC_PATH, NWIN, MODE_K, FREQ_MAX_KHZ, OUT_DIR.
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
import poc_fsq_stageB as poc
from poc_fsq_stageB import FSQAutoencoder, load_pairs, _hard

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MOD = os.environ.get("MODALITY", "ece")
SHOT = os.environ.get("SHOT", "200729")
CODEC = os.environ.get(
    "CODEC_PATH",
    f"/lustre/orion/fus187/proj-shared/models/fsq_spectro_codecs_tok96/spectro_codec_{MOD}.pt")
# CODEC_PATHS: comma list of codecs to compare (one recon row each, same channel/display).
# Any codec whose cfg has bg_subtract=True is run in RESIDUAL space at inference
# (S -> R -> decode -> recombine B + R_rec) via the local spectro_bg.py. Defaults to CODEC.
CODEC_PATHS = [p.strip() for p in os.environ.get("CODEC_PATHS", CODEC).split(",") if p.strip()]
BG_SIGMA = float(os.environ.get("BG_SIGMA", "8.0"))
DATA = os.environ.get("DATA_DIR", "/lustre/orion/fus187/proj-shared/foundation_model")
STATS = os.environ.get("STATS_PATH",
                       "/lustre/orion/fus187/proj-shared/foundation_model_meta/preprocessing_stats.pt")
NWIN = int(os.environ.get("NWIN", "80"))
MODE_K = float(os.environ.get("MODE_K", "2.5"))
FREQ_MAX_KHZ = float(os.environ.get("FREQ_MAX_KHZ", "0"))     # 0 = full band; else crop for the zoom
OUT = Path(os.environ.get("OUT_DIR", f"{FMH}/eval_runs/codec_recon_real/{MOD}"))
OUT.mkdir(parents=True, exist_ok=True)
FS, NFFT, HOP = 500_000.0, 1024, 256

# load data once — all compared codecs share C/geometry (ece C=40, patch 32/16)
C = int(torch.load(CODEC_PATHS[0], map_location="cpu", weights_only=False)["cfg"]["C"])
xi, _ = load_pairs(SHOT, DATA, STATS, C, NWIN, modality=MOD)
X = xi.to(dev); Xn = X.cpu().numpy()
Fq = X.shape[-2]
ch = int(_hard(X, MODE_K).cpu().numpy().sum(axis=(0, 2, 3)).argmax())   # strongest-mode channel (from GT)

S = 5  # step 0.01 s, chunk 0.05 s -> every 5th window is contiguous/non-overlapping
def stitch(A4):
    a = A4[::S, ch]; n, Ff, Tt = a.shape
    return a.transpose(1, 0, 2).reshape(Ff, n * Tt), n

def label_for(path):
    d = Path(path).parent.name
    return "production" if d.startswith("fsq_spectro_codecs") else d


def reconstruct(path):
    ck = torch.load(path, map_location="cpu", weights_only=False); c = ck["cfg"]
    poc.PATCH_F = int(c["patch_f"]); poc.PATCH_T = int(c["patch_t"]); poc.D_MODEL = int(c["d_model"])
    ae = FSQAutoencoder(c["C"], c["Fq"], c["Tq"], c["fsq_dim"], c["fsq_L"],
                        per_channel=c.get("per_channel", False)).to(dev)
    ae.load_state_dict(ck["ae"]); ae.eval()
    bg = bool(c.get("bg_subtract", False))
    with torch.no_grad():
        if bg:                                   # residual-space: S->R->decode->recombine B+R_rec
            from spectro_bg import baseline_residual
            B, R = baseline_residual(X, sigma=BG_SIGMA)
            Rd = R.to(dev)
            Rrec = torch.cat([ae(Rd[i:i + 16])[0] for i in range(0, Rd.shape[0], 16)], 0).cpu()
            Srec = (Rrec + B).numpy()
        else:
            Srec = torch.cat([ae(X[i:i + 16])[0] for i in range(0, X.shape[0], 16)], 0).cpu().numpy()
    return label_for(path) + (" +bg" if bg else ""), Srec, ae.n_tok

FREQ = np.arange(Fq) * FS / NFFT / 1e3
fmax_bin = Fq if FREQ_MAX_KHZ <= 0 else int(min(Fq, FREQ_MAX_KHZ / (FS / NFFT / 1e3)))
G, nseg = stitch(Xn); G = G[:fmax_bin]
recons, srecs = [], []
for path in CODEC_PATHS:
    tag, Srec, ntok = reconstruct(path)
    srecs.append((tag, Srec))
    Rst, _ = stitch(Srec); Rst = Rst[:fmax_bin]
    corr = float(np.corrcoef(G.ravel(), Rst.ravel())[0, 1])
    recons.append((f"{tag} (tok{ntok})  band-corr={corr:.3f}", Rst))
    print(f"[compare] {tag}: wholeband-corr={corr:.3f} tok={ntok}", flush=True)

time_ms = np.arange(G.shape[1]) * HOP / FS * 1e3
ext = (0, time_ms[-1], 0, FREQ[fmax_bin - 1]); vmn, vmx = np.percentile(G, [2, 99])
panels = [(f"GT — {MOD.upper()} {SHOT} ch{ch}", G)] + recons
fig, ax = plt.subplots(len(panels), 1, figsize=(14, 2.7 * len(panels)), sharex=True)
ax = np.atleast_1d(ax)
for a_, (t, d) in zip(ax, panels):
    im = a_.imshow(d, aspect="auto", origin="lower", cmap="magma", vmin=vmn, vmax=vmx, extent=ext)
    a_.set_title(t, fontsize=10); a_.set_ylabel("Freq (kHz)")
    fig.colorbar(im, ax=a_, fraction=0.012, pad=0.01)
ax[-1].set_xlabel("Time (ms)")
band = f"0-{int(FREQ_MAX_KHZ)}kHz" if FREQ_MAX_KHZ > 0 else "full"
fig.suptitle(f"Codec reconstruction comparison — {MOD.upper()} {SHOT} ch{ch} "
             f"({nseg} contiguous windows, {band}, fs=500kHz)", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.97))
tag = f"codec_compare_{MOD}_{SHOT}" + (f"_0-{int(FREQ_MAX_KHZ)}kHz" if FREQ_MAX_KHZ > 0 else "")
for e in ("png", "pdf"):
    fig.savefig(OUT / f"{tag}.{e}", dpi=130, bbox_inches="tight")
print(f"[compare] saved {OUT}/{tag}.png  ch={ch} nseg={nseg} codecs={len(recons)}", flush=True)

# ---- MODE-BAND METRIC: high-pass (freq) correlation = thin-structure/mode fidelity ----
# Aggregated over ALL channels x windows (not one hand-picked example). The high-pass of
# GT IS the mode content; residual should WIN here while losing on whole-band corr.
from spectro_bg import baseline_residual as _bg
def _hp(a4):
    _, Rr = _bg(torch.from_numpy(np.ascontiguousarray(a4)), sigma=BG_SIGMA)
    return Rr.numpy()[:, :, :fmax_bin, :]                       # crop to the 0-60 kHz mode band
HPg = _hp(Xn); Nn, Cc = HPg.shape[0], HPg.shape[1]
print(f"\n[metric] MODE-BAND high-pass-freq corr, 0-{int(FREQ_MAX_KHZ)}kHz, over {Cc} ch x {Nn} win", flush=True)
print(f"[metric] {'codec':<18}{'mode_pix':>10}{'mode_pix_top25':>16}{'mode_prof_top25':>17}", flush=True)
for tagm, Srec in srecs:
    HPr = _hp(Srec); pix, prof, en = [], [], []
    for w in range(Nn):
        for c in range(Cc):
            a = HPg[w, c].ravel()
            if a.std() < 1e-6:
                continue
            b = HPr[w, c].ravel()
            pix.append(float(np.corrcoef(a, b)[0, 1])); en.append(float((a ** 2).mean()))
            pg = np.abs(HPg[w, c]).mean(1); pr = np.abs(HPr[w, c]).mean(1)
            prof.append(float(np.corrcoef(pg, pr)[0, 1]) if pg.std() > 1e-6 else np.nan)
    pix, prof, en = np.array(pix), np.array(prof), np.array(en)
    hi = en >= np.percentile(en, 75)                            # top-25% mode-content (w,c)
    print(f"[metric] {tagm:<18}{np.nanmedian(pix):>10.3f}{np.nanmedian(pix[hi]):>16.3f}"
          f"{np.nanmedian(prof[hi]):>17.3f}", flush=True)
