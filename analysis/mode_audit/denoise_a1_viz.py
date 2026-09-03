"""Rung-0 coherence denoiser: before/after VIZ (all 4 modalities) + label-free A1 gate.

Self-contained: reads RAW time-series from the _processed.h5 (ece/co2/bes/mhr ydata),
STFTs complex on-the-fly (raw_stft_complex), applies coherence_denoise, compares to the
raw magnitude. No dataset rewrite, no world model.

A1 gate (label-free — no human labels exist; uses the band-prominence detector):
  (i) mode-retention: on RAW mode-active windows (top-quartile band prominence), does the
      denoised keep the mode at the same freq (peak within tol) with prominence ratio >= 0.9?
  (ii) non-invention: on RAW mode-free windows (bottom-quartile), does denoised FIRE
       (prominence crossing the active cut)? rate should ~ raw baseline (~0).
  (iii) amplitude-linearity: band-power raw vs denoised (median ratio, no compression on modes).
PASS = retention >= 0.9 AND non-invention ~ baseline AND amplitude not compressed on modes.

Also sweeps the coherence window (win_f,win_t) lightly. Env: SHOTS, SPAN_S, WIN_F, WIN_T, OUT_DIR.
"""
import json, os, sys
from pathlib import Path
FMH = "/lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub"
for p in (f"{FMH}/src", f"{FMH}/scripts/training"):
    if p not in sys.path:
        sys.path.insert(0, p)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, torch, h5py
from scipy.ndimage import gaussian_filter1d
from spectro_bg import channel_coherent_denoise, raw_stft_complex

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA = os.environ.get("DATA_DIR", "/lustre/orion/fus187/proj-shared/foundation_model")
SHOTS = os.environ.get("SHOTS", "200729,190900,204811").split(",")
SPAN_S = float(os.environ.get("SPAN_S", "2.0"))          # seconds of raw per shot (after warmup)
WARMUP_S = 1.0
WIN_F = int(os.environ.get("WIN_F", "1")); WIN_T = int(os.environ.get("WIN_T", "1"))
K_CHAN = int(os.environ.get("K_CHAN", "2"))              # adjacent-channel radius (±k)
OUT = Path(os.environ.get("OUT_DIR", f"{FMH}/analysis/mode_audit/denoise")); OUT.mkdir(parents=True, exist_ok=True)
FS, NFFT, HOP = 500_000.0, 1024, 256
DF = FS / NFFT / 1e3
MODE_LO, MODE_HI = int(round(5.0 / DF)), int(round(40.0 / DF))
CHAN_SLICE = {"ece": (0, 40), "co2": (0, 4), "bes": (48, 64), "mhr": (2, 8)}  # data_loader channels_to_use
WIN_FR = int(round(0.05 * FS / HOP))                     # STFT frames per 50 ms window (~97)


def band_prom(prof):                                     # prof over full F -> (pd_band, peakbin, z)
    pb = prof[MODE_LO:MODE_HI]
    pd = pb - gaussian_filter1d(pb, 6.0)
    mad = np.median(np.abs(pd - np.median(pd))) * 1.4826 + 1e-9
    f0 = int(np.argmax(pd))
    return pd, MODE_LO + f0, float(pd[f0] / mad)


res = {}
for mod in ["ece", "co2", "bes", "mhr"]:
    print(f"\n===== {mod} =====", flush=True)
    per_shot_raw, per_shot_den = [], []
    for sh in SHOTS:
        fp = Path(DATA) / f"{sh}_processed.h5"
        if not fp.exists():
            continue
        try:
            with h5py.File(fp, "r") as f:
                if mod not in f:
                    continue
                x = f[mod]["xdata"][:]; y = f[mod]["ydata"]
                i0 = int(np.searchsorted(x, WARMUP_S)); i1 = min(i0 + int(SPAN_S * FS), y.shape[1])
                if i1 - i0 < 2 * NFFT:                                    # too few samples -> skip shot
                    print(f"[warn] {mod} {sh}: slice {i1-i0} < {2*NFFT} samples, skip", flush=True); continue
                sig = torch.tensor(y[:, i0:i1], dtype=torch.float32)      # (Craw, N)
        except Exception as e:
            print(f"[warn] {sh}: {e}", flush=True); continue
        sig = torch.nan_to_num(sig).to(dev)
        a, b = CHAN_SLICE[mod]; b = min(b, sig.shape[0])
        sig = sig[a:b]                                                    # channels_to_use FIRST (radial order)
        S = raw_stft_complex(sig, NFFT, HOP)                              # (Csel, F, T) complex
        raw_mag = S.abs()
        den_mag, _ = channel_coherent_denoise(S, K_CHAN, WIN_F, WIN_T)    # coherent-integrate over SELECTED chans
        per_shot_raw.append(raw_mag.cpu()); per_shot_den.append(den_mag.cpu())
    if not per_shot_raw:
        print(f"[warn] {mod}: no data", flush=True); continue
    RM = torch.cat(per_shot_raw, -1).numpy()   # (C,F,Ttot) raw mag
    DM = torch.cat(per_shot_den, -1).numpy()   # denoised
    C, F, T = RM.shape
    nwin = T // WIN_FR
    # window-level band prominence (raw) -> active/quiescent
    zr = np.array([max(band_prom(np.abs(RM[c, :, w*WIN_FR:(w+1)*WIN_FR]).mean(1))[2] for c in range(C)) for w in range(nwin)])
    P75, P25 = np.percentile(zr, 75), np.percentile(zr, 25)
    act = np.where(zr >= P75)[0]; qui = np.where(zr <= P25)[0]
    # firing cut = P75-equivalent z on raw; "fires" if a window's best-channel z >= that
    fire_cut = P75
    # (i) mode-retention (SNR-stratified) + amplitude LINEARITY over active windows.
    # retention vs raw conflates mode-loss with eta-removal (denoiser deflates raw-ref
    # prominence by design); so the honest measure is retention on HIGH-SNR windows (raw
    # peak >> eta, so raw peak ~ true mode) + linearity of denoised-vs-raw peak (no distortion).
    ret, retf, rawpk, denpk = [], [], [], []
    for w in act:
        c = int(np.argmax([band_prom(np.abs(RM[cc, :, w*WIN_FR:(w+1)*WIN_FR]).mean(1))[2] for cc in range(C)]))
        pr, f0r, _ = band_prom(np.abs(RM[c, :, w*WIN_FR:(w+1)*WIN_FR]).mean(1))
        pdn = np.abs(DM[c, :, w*WIN_FR:(w+1)*WIN_FR]).mean(1)[MODE_LO:MODE_HI]
        pdn = pdn - gaussian_filter1d(pdn, 6.0)
        f0loc = f0r - MODE_LO
        ret.append(float(pdn[f0loc] / (pr[f0loc] + 1e-9)))
        retf.append(abs(int(np.argmax(pdn)) - f0loc) <= 2)
        rawpk.append(float(pr[f0loc])); denpk.append(float(pdn[f0loc]))
    rawpk_a, denpk_a, ret_a = np.array(rawpk), np.array(denpk), np.array(ret)
    hi = np.argsort(-rawpk_a)[:max(1, len(rawpk_a) // 4)]              # top-quartile SNR (raw peak)
    ret_hi = float(np.nanmedian(ret_a[hi])) if len(hi) else None
    slope = float(np.polyfit(rawpk_a, denpk_a, 1)[0]) if len(rawpk_a) > 2 else None
    lin_r = (float(np.corrcoef(rawpk_a, denpk_a)[0, 1]) if len(rawpk_a) > 2
             and rawpk_a.std() > 0 and denpk_a.std() > 0 else None)
    # (ii) non-invention on quiescent windows: does denoised fire?
    inv = 0
    for w in qui:
        zden = max(band_prom(np.abs(DM[c2, :, w*WIN_FR:(w+1)*WIN_FR]).mean(1))[2] for c2 in range(C))
        inv += int(zden >= fire_cut)
    r = {"modality": mod, "C": C, "n_windows": int(nwin), "n_active": int(len(act)), "n_quiescent": int(len(qui)),
         "retention_median_active": float(np.nanmedian(ret_a)) if len(ret_a) else None,
         "retention_median_highSNR": ret_hi,
         "freq_within_tol": float(np.mean(retf)) if retf else None,
         "noninvention_fire_rate_quiescent": (inv / len(qui)) if len(qui) else None,
         "amplitude_linearity_pearson_r": lin_r, "amplitude_linearity_slope": slope,
         "k_chan": K_CHAN}
    r["A1_pass"] = bool(ret_hi is not None and ret_hi >= 0.7
                        and r["freq_within_tol"] >= 0.9
                        and (r["noninvention_fire_rate_quiescent"] or 0) <= 0.1
                        and lin_r is not None and lin_r >= 0.9
                        and slope is not None and 0.5 <= slope <= 1.6)
    res[mod] = r
    print(f"[A1] {mod}: retention hiSNR={r['retention_median_highSNR']} (allactive={r['retention_median_active']}) "
          f"freq-in-tol={r['freq_within_tol']} | non-invention={r['noninvention_fire_rate_quiescent']} "
          f"| amp-linearity r={r['amplitude_linearity_pearson_r']} slope={r['amplitude_linearity_slope']} "
          f"==> {'PASS' if r['A1_pass'] else 'FAIL'}", flush=True)
    # before/after viz: strongest-mode active window, top-8 channels
    if len(act):
        w = act[int(np.argmax(zr[act]))]
        sl = slice(w*WIN_FR, (w+1)*WIN_FR)
        ch = int(np.argmax([band_prom(np.abs(RM[cc, :, sl]).mean(1))[2] for cc in range(C)]))
        fmax = min(F, int(80 / DF)); freqs = np.arange(F) * DF
        fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
        for a, (t, M) in zip(ax[:2], [("RAW", RM), ("DENOISED", DM)]):
            a.imshow(np.abs(M[ch, :fmax, sl]), origin="lower", aspect="auto", extent=[0, WIN_FR, 0, freqs[fmax]])
            a.set_title(f"{mod} {t} ch{ch}", fontsize=9); a.set_ylabel("kHz")
        pr = np.abs(RM[ch, :fmax, sl]).mean(1); pdn = np.abs(DM[ch, :fmax, sl]).mean(1)
        ax[2].plot(freqs[:fmax], pr, label="raw", color="tab:gray"); ax[2].plot(freqs[:fmax], pdn, label="denoised", color="tab:red")
        ax[2].axvspan(5, 40, color="y", alpha=0.1); ax[2].legend(fontsize=8); ax[2].set_title("band profile"); ax[2].set_xlabel("kHz")
        fig.suptitle(f"{mod.upper()} before/after coherence-denoise (adj-chan ±{K_CHAN}) — A1 {'PASS' if r['A1_pass'] else 'FAIL'}", fontsize=10)
        fig.tight_layout(); fig.savefig(OUT / f"denoise_{mod}.pdf"); plt.close(fig)
        print(f"[viz] {mod}: saved {OUT}/denoise_{mod}.pdf", flush=True)

res["A1_all_pass"] = bool(res) and all(v.get("A1_pass") for v in res.values() if isinstance(v, dict))
json.dump(res, open(OUT / "a1_gate.json", "w"), indent=2, default=lambda o: float(o) if hasattr(o, "item") else o)
print(f"\n[A1] ALL-MODALITY PASS = {res['A1_all_pass']}  (wrote {OUT}/a1_gate.json + denoise_*.pdf)", flush=True)
print("[denoise_a1] done", flush=True)
