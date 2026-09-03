"""IGNITE mode-loss audit — Task 7: DECODED-output stability (picks the fix).

Code stability is low on mode windows (0.18-0.26): a 0.5 ms shift flips most FSQ
codes. Question that decides the recommendation: does decode(codes) also move, or is
the DECODED spectrogram stable despite code churn (codes redundant)?

  decode(encode(GT)) vs decode(encode(shift(GT))) on ACTIVE windows.
    decoded stability HIGH  -> codes redundant; CE-on-codes penalizes unpredictable
                               jitter => FIX = decoded/perceptual world-model loss.
    decoded stability LOW   -> codec mode-rendering itself unstable => FIX = codec.

Reference: also decode(encode(GT)) vs GT (recon corr) so we know the decode is sane.
Metric: mode-band pixel corr (band 5-40 kHz), on the strongest-mode channel, ACTIVE
(top-quartile prominence) windows. Env: MODALITIES, SHOTS, CODEC_DIR, NWIN_PER_SHOT.
"""
import json
import os
import sys
from pathlib import Path

FMH = "/lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub"
for p in (f"{FMH}/src", f"{FMH}/scripts/training"):
    if p not in sys.path:
        sys.path.insert(0, p)
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
SHOTS = os.environ.get("SHOTS", "200729,190996,204811,190900,190904,201585").split(",")
DATA = os.environ.get("DATA_DIR", "/lustre/orion/fus187/proj-shared/foundation_model")
STATS = os.environ.get("STATS_PATH",
                       "/lustre/orion/fus187/proj-shared/foundation_model_meta/preprocessing_stats.pt")
NWIN_PER_SHOT = int(os.environ.get("NWIN_PER_SHOT", "400"))
OUT = Path(os.environ.get("OUT_DIR", f"{FMH}/analysis/mode_audit"))
OUT.mkdir(parents=True, exist_ok=True)
BG_SIGMA = float(os.environ.get("BG_SIGMA", "8.0"))
FS, NFFT = 500_000.0, 1024
DF = FS / NFFT / 1e3
MODE_LO, MODE_HI = int(round(5.0 / DF)), int(round(40.0 / DF))


def peakP(x_ch):
    prof = np.abs(x_ch[MODE_LO:MODE_HI]).mean(1)
    return float((prof - gaussian_filter1d(prof, 6.0)).max())


def strong_ch(x):
    return int(np.argmax([peakP(x[c]) for c in range(x.shape[0])]))


def bandcorr(a_ch, b_ch):
    a = np.abs(a_ch[MODE_LO:MODE_HI]).ravel(); b = np.abs(b_ch[MODE_LO:MODE_HI]).ravel()
    if a.std() < 1e-9 or b.std() < 1e-9:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def enc(codec, x):
    with torch.no_grad():
        return codec.encode_codes(x.to(dev)).cpu()


def dec(codec, c):
    with torch.no_grad():
        return codec.decode_codes(c.to(dev)).cpu()


all_res = {}
for mod in MODS:
    print(f"\n===================== DECODED-STAB {mod} =====================", flush=True)
    try:
        codec, cfg = load_frozen_codec(f"{CODEC_DIR}/spectro_codec_{mod}.pt", map_location="cpu")
        codec = codec.to(dev); bg = bool(cfg.get("bg_subtract", False))
        C = int(cfg["C"]); poc.PATCH_F = int(cfg.get("patch_f", 8)); poc.PATCH_T = int(cfg.get("patch_t", 16))
        Xs = []
        for sh in SHOTS:
            if not (Path(DATA) / f"{sh}_processed.h5").exists():
                continue
            try:
                _, xt = load_pairs(sh, DATA, STATS, C, NWIN_PER_SHOT, modality=mod); Xs.append(xt)
            except Exception as e:
                print(f"[warn] {sh}: {e}", flush=True)
        X = torch.cat(Xs)
        R = (baseline_residual(X, sigma=BG_SIGMA)[1] if bg else X).cpu()
        Rs = torch.roll(R, shifts=1, dims=-1)
        # select ACTIVE windows (top-quartile prominence) + strong channel each
        P = np.array([max(peakP(R[w, c].numpy()) for c in range(C)) for w in range(R.shape[0])])
        act = np.where(P >= np.percentile(P, 75))[0]
        dec_stab, code_stab, recon = [], [], []
        for i in range(0, len(act), 64):
            idx = act[i:i + 64]
            r = R[idx]; rs = Rs[idx]
            c0 = enc(codec, r); c1 = enc(codec, rs)
            d0 = dec(codec, c0); d1 = dec(codec, c1)
            for j, w in enumerate(idx):
                ch = strong_ch(r[j].numpy())
                dec_stab.append(bandcorr(d0[j, ch].numpy(), d1[j, ch].numpy()))   # decode(GT) vs decode(shift)
                recon.append(bandcorr(r[j].numpy()[ch], d0[j, ch].numpy()))        # recon fidelity (ref)
                code_stab.append(float((c0[j] == c1[j]).float().mean()))           # code stability (ref)
        r = {"task": 7, "modality": mod, "n_active": int(len(act)),
             "decoded_stability_bandcorr": float(np.nanmedian(dec_stab)),
             "code_stability": float(np.nanmedian(code_stab)),
             "recon_bandcorr": float(np.nanmedian(recon)),
             "verdict": ("DECODE STABLE -> codes redundant -> fix=decoded/perceptual LOSS"
                         if np.nanmedian(dec_stab) > 0.8 else
                         ("DECODE MODERATE" if np.nanmedian(dec_stab) > 0.5 else
                          "DECODE UNSTABLE -> codec mode-rendering unstable -> fix=CODEC"))}
        all_res[mod] = r
        json.dump(r, open(OUT / f"task7_decstab_{mod}.json", "w"), indent=2)
        print(f"[decstab] {mod}: N_active={len(act)} | decoded-stability(bandcorr)={r['decoded_stability_bandcorr']:.3f} "
              f"| code-stability={r['code_stability']:.3f} | recon(bandcorr)={r['recon_bandcorr']:.3f} "
              f"==> {r['verdict']}", flush=True)
    except Exception as e:
        import traceback
        print(f"[WARN] {mod} failed: {e}", flush=True); traceback.print_exc()

json.dump(all_res, open(OUT / "task7_decstab_all.json", "w"), indent=2)
print("\n[decstab] done", flush=True)
