"""IGNITE mode-loss audit — Task 4: PERSISTENCE ORACLE (the code-predictability ceiling).

encode(GT window t) vs encode(GT window t+1): per-token/per-dim code agreement.
This is the codeacc a PERSISTENCE predictor (copy the input window's codes) achieves,
and — since the world model SEES window t at prediction time — a floor the model
should be able to reach by copying. Stratified mode-active vs quiescent, and
mode-patch vs background tokens within active windows.

Decision (vs the model's Task-3 mode-patch codeacc ~0.10):
  oracle mode-patch >> 0.10  -> codes ARE persistable; model underperforms => MODEL-SIDE.
  oracle mode-patch ~ 0.10   -> codes flip ~completely each window => TARGET-SIDE
                                 (exact FSQ codes unpredictable 1-step; wrong target).

load_pairs returns (X_in=t, X_tgt=t+1) already consecutive. Codec-only, no world model.
Env: MODALITIES, SHOTS_FILE, CODEC_DIR, NWIN_PER_SHOT, OUT_DIR.
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
from spectro_bg import baseline_residual
from tokamak_foundation_model.e2e.quantizers.spectro_codec import load_frozen_codec

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODS = [m.strip() for m in os.environ.get("MODALITIES", "ece,co2").split(",") if m.strip()]
CODEC_DIR = os.environ.get("CODEC_DIR", "/lustre/orion/fus187/proj-shared/models/fsq_resid_p8_all")
SHOTS_FILE = os.environ.get("SHOTS_FILE", "/lustre/orion/fus187/proj-shared/models/codec_shots.txt")
DATA = os.environ.get("DATA_DIR", "/lustre/orion/fus187/proj-shared/foundation_model")
STATS = os.environ.get("STATS_PATH",
                       "/lustre/orion/fus187/proj-shared/foundation_model_meta/preprocessing_stats.pt")
NWIN_PER_SHOT = int(os.environ.get("NWIN_PER_SHOT", "800"))
OUT = Path(os.environ.get("OUT_DIR", f"{FMH}/analysis/mode_audit"))
OUT.mkdir(parents=True, exist_ok=True)
BG_SIGMA = float(os.environ.get("BG_SIGMA", "8.0"))
FS, NFFT = 500_000.0, 1024
DF = FS / NFFT / 1e3
MODE_LO, MODE_HI = int(round(5.0 / DF)), int(round(40.0 / DF))
SHOTS = [s.strip() for s in Path(SHOTS_FILE).read_text().split() if s.strip()]


def band_peakP(x_ch):
    prof = np.abs(x_ch[MODE_LO:MODE_HI]).mean(1)
    pd = prof - gaussian_filter1d(prof, 6.0)
    return float(pd.max())


def win_P(x):                      # x (C,F,T) -> max over channels of band-peak prominence
    return max(band_peakP(x[c]) for c in range(x.shape[0]))


def mode_pixel_mask(x_ch, k=3.0):
    a = np.abs(x_ch); base = gaussian_filter1d(a, 6.0, axis=0); r = a - base
    m = np.zeros_like(a, bool); band = r[MODE_LO:MODE_HI]
    mad = np.median(np.abs(band - np.median(band))) * 1.4826 + 1e-9
    m[MODE_LO:MODE_HI] = band > k * mad
    return m


def resid(X, bg):
    if bg:
        _, R = baseline_residual(X, sigma=BG_SIGMA)
        return R.cpu()
    return X.cpu()


def enc(codec, x):
    with torch.no_grad():
        return codec.encode_codes(x.to(dev)).cpu()


all_res = {}
for mod in MODS:
    print(f"\n===================== ORACLE {mod} =====================", flush=True)
    try:
        codec, cfg = load_frozen_codec(f"{CODEC_DIR}/spectro_codec_{mod}.pt", map_location="cpu")
        codec = codec.to(dev)
        bg = bool(cfg.get("bg_subtract", False))
        patch_f = int(cfg.get("patch_f", 8)); patch_t = int(cfg.get("patch_t", 16))
        C = int(cfg["C"]); Fq = int(cfg["Fq"])
        poc.PATCH_F = patch_f; poc.PATCH_T = patch_t
        Xin, Xtg = [], []
        for sh in SHOTS:
            try:
                xi, xt = load_pairs(sh, DATA, STATS, C, NWIN_PER_SHOT, modality=mod)
                Xin.append(xi); Xtg.append(xt)
            except Exception as e:
                print(f"[warn] {mod} shot {sh}: {e}", flush=True)
        Xin = torch.cat(Xin); Xtg = torch.cat(Xtg)
        Ri = resid(Xin, bg); Rt = resid(Xtg, bg)
        ci = torch.cat([enc(codec, Ri[i:i + 64]) for i in range(0, Ri.shape[0], 64)], 0)   # codes(t)
        ct = torch.cat([enc(codec, Rt[i:i + 64]) for i in range(0, Rt.shape[0], 64)], 0)   # codes(t+1)
        N, ntok, dim = ci.shape
        npf = Fq // patch_f; npt = ntok // npf
        tok_match = (ci == ct).float().mean(-1).numpy()          # (N,ntok) per-token codeacc
        # stratify windows by t+1 prominence (quartiles)
        P = np.array([win_P((Rt[w]).numpy()) for w in range(N)])
        P75, P25 = np.percentile(P, 75), np.percentile(P, 25)
        act = np.where(P >= P75)[0]; qui = np.where(P <= P25)[0]
        # mode-patch vs background tokens within active windows
        mp_hits = mp_tok = bg_hits = bg_tok = 0.0
        for w in act:
            m = np.zeros((Fq, Rt.shape[-1]), bool)
            xt = Rt[w].numpy()
            for c in range(C):
                m |= mode_pixel_mask(xt[c])
            pm = m[:npf * patch_f].reshape(npf, patch_f, npt, patch_t).any((1, 3)).reshape(-1)
            mp_hits += tok_match[w][pm].sum(); mp_tok += int(pm.sum())
            bg_hits += tok_match[w][~pm].sum(); bg_tok += int((~pm).sum())
        r = {"task": 4, "modality": mod, "N_pairs": int(N), "dim": dim, "L": int(cfg["fsq_L"]),
             "random_floor": 1.0 / int(cfg["fsq_L"]),
             "oracle_all": float(tok_match.mean()),
             "oracle_active": float(tok_match[act].mean()) if len(act) else None,
             "oracle_quiescent": float(tok_match[qui].mean()) if len(qui) else None,
             "oracle_active_mode_patch": (mp_hits / mp_tok if mp_tok else None),
             "oracle_active_background": (bg_hits / bg_tok if bg_tok else None),
             "n_active": int(len(act)), "n_quiescent": int(len(qui)),
             "n_active_mode_tokens": int(mp_tok), "n_active_bg_tokens": int(bg_tok)}
        all_res[mod] = r
        json.dump(r, open(OUT / f"task4_oracle_{mod}.json", "w"), indent=2, default=lambda o: float(o))
        print(f"[oracle] {mod}: N={N} L={r['L']} random={r['random_floor']:.3f} | "
              f"all={r['oracle_all']:.3f} active={r['oracle_active']:.3f} quiescent={r['oracle_quiescent']:.3f} "
              f"| active mode-patch={r['oracle_active_mode_patch']} background={r['oracle_active_background']} "
              f"(tokens {int(mp_tok)}/{int(bg_tok)})", flush=True)
    except Exception as e:
        import traceback
        print(f"[WARN] {mod} oracle failed: {e}", flush=True); traceback.print_exc()

json.dump(all_res, open(OUT / "task4_oracle_all.json", "w"), indent=2, default=lambda o: float(o))
print("\n[oracle] done", flush=True)
