"""Deeper codec diagnostics on EXISTING smoothed-codec rungs (no retraining).

For each codec dir (e.g. fsq_smooth_ece_s8, s16), reading smooth_frames from cfg:

(1) STABILITY exact vs +-1-level tolerance. enc(GT) vs enc(GT shifted 1 frame), per-dim
    int codes. exact = frac dims equal; tol1 = frac dims within +-1 level. If tol1 >> exact,
    the flips are boundary crossings to a NEIGHBOURING level (soft/tolerant target could
    recover them); if tol1 ~ exact, flips are large jumps.

(2) MARGIN histogram: distance of the pre-quantization bounded value from the nearest FSQ
    round boundary (half-integer), per dim = 0.5 - |bound(z) - round(bound(z))| in [0,0.5].
    Small margin => sits on a boundary => flips under a tiny perturbation. Stratified
    active vs quiescent windows. Reports median margin + frac(margin<0.1) + PDF hist.

(3) CODE ENTROPY per dim (bits, mean over dims) over all/active/quiescent windows — the
    baseline to watch for a future collapse tripwire.

Env: CODEC_DIRS (comma), MODALITIES(ece), SHOTS, NWIN_PER_SHOT, OUT_DIR.
"""
import json
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
from scipy.ndimage import gaussian_filter1d
import poc_fsq_stageB as poc
from poc_fsq_stageB import load_pairs
from spectro_bg import baseline_residual, smooth_time_mag
from tokamak_foundation_model.e2e.quantizers.spectro_codec import load_frozen_codec

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MOD = os.environ.get("MODALITIES", "ece").split(",")[0]
CODEC_DIRS = [d.strip() for d in os.environ.get(
    "CODEC_DIRS",
    "/lustre/orion/fus187/proj-shared/models/fsq_smooth_ece_s8,"
    "/lustre/orion/fus187/proj-shared/models/fsq_smooth_ece_s16").split(",") if d.strip()]
SHOTS = os.environ.get("SHOTS", "200729,190996,204811,191001").split(",")
DATA = os.environ.get("DATA_DIR", "/lustre/orion/fus187/proj-shared/foundation_model")
STATS = os.environ.get("STATS_PATH",
                       "/lustre/orion/fus187/proj-shared/foundation_model_meta/preprocessing_stats.pt")
NWIN_PER_SHOT = int(os.environ.get("NWIN_PER_SHOT", "400"))
OUT = Path(os.environ.get("OUT_DIR", f"{FMH}/analysis/mode_audit"))
OUT.mkdir(parents=True, exist_ok=True)
BG_SIGMA = 8.0
FS, NFFT = 500_000.0, 1024
DF = FS / NFFT / 1e3
MODE_LO, MODE_HI = int(round(5.0 / DF)), int(round(40.0 / DF))


def win_P(x):
    out = 0.0
    for c in range(x.shape[0]):
        prof = np.abs(x[c, MODE_LO:MODE_HI]).mean(1)
        out = max(out, float((prof - gaussian_filter1d(prof, 6.0)).max()))
    return out


def per_dim_entropy(codes_int, L):     # codes_int (M, dim) -> mean per-dim entropy (bits)
    M, dim = codes_int.shape
    ents = []
    for d in range(dim):
        c = np.bincount(codes_int[:, d], minlength=L).astype(np.float64)
        p = c / c.sum(); p = p[p > 0]
        ents.append(float(-(p * np.log2(p)).sum()))
    return float(np.mean(ents)), float(np.max(ents))


all_res = {}
for CD in CODEC_DIRS:
    tag = Path(CD).name
    print(f"\n===================== {tag} ({MOD}) =====================", flush=True)
    try:
        codec, cfg = load_frozen_codec(f"{CD}/spectro_codec_{MOD}.pt", map_location="cpu")
        codec = codec.to(dev)
        sf = int(cfg.get("smooth_frames", 0) or 0); bg = bool(cfg.get("bg_subtract", False))
        C = int(cfg["C"]); L = int(cfg["fsq_L"]); dim = int(cfg["fsq_dim"])
        poc.PATCH_F = int(cfg.get("patch_f", 8)); poc.PATCH_T = int(cfg.get("patch_t", 16))

        Xt = []
        for sh in SHOTS:
            if not (Path(DATA) / f"{sh}_processed.h5").exists():
                continue
            try:
                _, xt = load_pairs(sh, DATA, STATS, C, NWIN_PER_SHOT, modality=MOD); Xt.append(xt)
            except Exception as e:
                print(f"[warn] {sh}: {e}", flush=True)
        X = torch.cat(Xt)
        R = (baseline_residual(X, sigma=BG_SIGMA)[1] if bg else X).cpu()
        Rc = smooth_time_mag(R, sf) if sf > 1 else R
        Rs = smooth_time_mag(torch.roll(R, 1, dims=-1), sf) if sf > 1 else torch.roll(R, 1, dims=-1)
        Pw = np.array([win_P(R[w].numpy()) for w in range(R.shape[0])])
        act = Pw >= np.percentile(Pw, 75); qui = Pw <= np.percentile(Pw, 25)

        # encode (int codes) + pre-quant bounded values, batched
        FB = codec.fsq                                   # FSQBottleneck
        def encode_full(xb):
            with torch.no_grad():
                t = codec.enc._encode(xb.to(dev))        # (b,ntok,d_model)
                z = FB.proj_in(t)
                bv = FB.fsq.bound(z)                      # pre-round bounded values
                codes = FB.fsq.codes_to_int(FB.fsq.quantize(z))
                return codes.cpu(), bv.cpu()
        c0, bv0, c1 = [], [], []
        for i in range(0, R.shape[0], 64):
            a, b = encode_full(Rc[i:i + 64]); c0.append(a); bv0.append(b)
            a2, _ = encode_full(Rs[i:i + 64]); c1.append(a2)
        c0 = torch.cat(c0); bv0 = torch.cat(bv0); c1 = torch.cat(c1)   # (N,ntok,dim)

        # (1) stability exact vs +-1
        def stab(mask):
            e = (c0[mask] == c1[mask]).float().mean().item()
            t1 = ((c0[mask] - c1[mask]).abs() <= 1).float().mean().item()
            return e, t1
        se, st1 = stab(torch.tensor(act)); qe, qt1 = stab(torch.tensor(qui))
        # (2) margin
        marg = (0.5 - (bv0 - bv0.round()).abs()).numpy()             # (N,ntok,dim) in [0,0.5]
        ma = marg[act].ravel(); mq = marg[qui].ravel()
        # (3) entropy
        ent_all = per_dim_entropy(c0.reshape(-1, dim).numpy(), L)
        ent_act = per_dim_entropy(c0[act].reshape(-1, dim).numpy(), L)
        ent_qui = per_dim_entropy(c0[qui].reshape(-1, dim).numpy(), L)

        res = {"codec": tag, "smooth_frames": sf, "L": L, "dim": dim,
               "stability_active_exact": se, "stability_active_tol1": st1,
               "stability_quiescent_exact": qe, "stability_quiescent_tol1": qt1,
               "margin_active_median": float(np.median(ma)), "margin_active_frac_lt_0.1": float((ma < 0.1).mean()),
               "margin_quiescent_median": float(np.median(mq)), "margin_quiescent_frac_lt_0.1": float((mq < 0.1).mean()),
               "entropy_bits_mean_all": ent_all[0], "entropy_bits_max": ent_all[1],
               "entropy_bits_mean_active": ent_act[0], "entropy_bits_mean_quiescent": ent_qui[0],
               "max_entropy_bits(log2 L)": float(np.log2(L))}
        all_res[tag] = res
        json.dump(res, open(OUT / f"margin_{tag}.json", "w"), indent=2, default=lambda o: float(o))
        print(f"[margin] {tag} sf={sf}: STABILITY active exact={se:.3f} tol1={st1:.3f} "
              f"(quiescent exact={qe:.3f} tol1={qt1:.3f})", flush=True)
        print(f"[margin] {tag}: MARGIN active median={res['margin_active_median']:.3f} "
              f"frac<0.1={res['margin_active_frac_lt_0.1']:.3f} | quiescent median={res['margin_quiescent_median']:.3f} "
              f"frac<0.1={res['margin_quiescent_frac_lt_0.1']:.3f} (0.5=safe, 0=on boundary)", flush=True)
        print(f"[margin] {tag}: CODE ENTROPY mean/dim={ent_all[0]:.2f} bits (active={ent_act[0]:.2f} "
              f"quiescent={ent_qui[0]:.2f}) of max {np.log2(L):.2f} = collapse-tripwire baseline", flush=True)
        # margin histogram PDF
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(ma, bins=50, range=(0, 0.5), density=True, alpha=0.6, label="active", color="tab:red")
        ax.hist(mq, bins=50, range=(0, 0.5), density=True, alpha=0.6, label="quiescent", color="tab:blue")
        ax.axvline(0.1, color="k", ls=":", lw=0.8); ax.set_xlabel("margin to nearest FSQ boundary (0=flips easily, 0.5=safe)")
        ax.set_ylabel("density"); ax.set_title(f"{tag} (smooth={sf}) pre-quant margin"); ax.legend()
        fig.tight_layout(); fig.savefig(OUT / f"margin_{tag}.pdf"); plt.close(fig)
    except Exception as e:
        import traceback
        print(f"[WARN] {tag} failed: {e}", flush=True); traceback.print_exc()

json.dump(all_res, open(OUT / "margin_all.json", "w"), indent=2, default=lambda o: float(o))
print("\n[margin] done", flush=True)
