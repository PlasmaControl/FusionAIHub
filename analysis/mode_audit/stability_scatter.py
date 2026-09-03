"""IGNITE mode-loss audit — Task 5 (stability) + Task 6 (bimodality scatter).

Uses IN-subset shots (in the codec's 8-shot training set) and OUT-subset mode shots
(longmode 190900/190904/201585 — strong modes the codec never trained on).

TASK 5 — STABILITY: encode(GT window) vs encode(SAME window, trivially time-shifted).
  Perturbation = roll the (correctly-normalized) residual spectrogram by 1 STFT frame
  = a 256-sample (0.5 ms) pre-STFT time shift; STFT magnitude of a signal shifted by
  one hop is the spectrogram shifted by one frame (interior). codeacc between the two
  encodings, stratified IN-subset vs OUT-subset (and active vs quiescent).
    high everywhere            -> codes stable => low persistence-oracle = REAL signal
                                  change => target-side / intrinsic.
    high IN, low OUT           -> codec-OOD scatter (codec overfit its 8 shots; unstable
                                  on the all-shots world-model training distribution).
    low everywhere             -> intrinsic codec code-assignment jitter (noisy target).

TASK 6 — BIMODALITY SCATTER: per-window PERSISTENCE codeacc (codes(t) vs codes(t+1),
  the ceiling proxy) vs residual band-variance, colored by subset. Identifies what the
  training-time bimodal codeacc (~0.9 easy / ~0.1 hard) actually IS: quiescent vs active?
  in- vs out-of-subset? Cross-checks Tasks 1-2.

Env: MODALITIES, SHOTS_IN, SHOTS_OUT, CODEC_DIR, NWIN_PER_SHOT, OUT_DIR.
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
SHOTS_IN = os.environ.get("SHOTS_IN", "200729,190996,204811,191001").split(",")
SHOTS_OUT = os.environ.get("SHOTS_OUT", "190900,190904,201585").split(",")
DATA = os.environ.get("DATA_DIR", "/lustre/orion/fus187/proj-shared/foundation_model")
STATS = os.environ.get("STATS_PATH",
                       "/lustre/orion/fus187/proj-shared/foundation_model_meta/preprocessing_stats.pt")
NWIN_PER_SHOT = int(os.environ.get("NWIN_PER_SHOT", "500"))
OUT = Path(os.environ.get("OUT_DIR", f"{FMH}/analysis/mode_audit"))
OUT.mkdir(parents=True, exist_ok=True)
BG_SIGMA = float(os.environ.get("BG_SIGMA", "8.0"))
FS, NFFT = 500_000.0, 1024
DF = FS / NFFT / 1e3
MODE_LO, MODE_HI = int(round(5.0 / DF)), int(round(40.0 / DF))


def band_peakP(x_ch):
    prof = np.abs(x_ch[MODE_LO:MODE_HI]).mean(1)
    return float((prof - gaussian_filter1d(prof, 6.0)).max())


def win_P(x):
    return max(band_peakP(x[c]) for c in range(x.shape[0]))


def enc(codec, x):
    with torch.no_grad():
        return codec.encode_codes(x.to(dev)).cpu()


def load_subset(mod, cfg, shots, tag):
    poc.PATCH_F = int(cfg.get("patch_f", 8)); poc.PATCH_T = int(cfg.get("patch_t", 16))
    C = int(cfg["C"]); ins, tgs = [], []
    for sh in shots:
        if not (Path(DATA) / f"{sh}_processed.h5").exists():
            print(f"[skip] {mod} {tag} shot {sh}: no file", flush=True); continue
        try:
            xi, xt = load_pairs(sh, DATA, STATS, C, NWIN_PER_SHOT, modality=mod)
            ins.append(xi); tgs.append(xt)
        except Exception as e:
            print(f"[warn] {mod} {tag} {sh}: {e}", flush=True)
    if not ins:
        return None, None
    return torch.cat(ins), torch.cat(tgs)


all_res = {}
for mod in MODS:
    print(f"\n===================== STABILITY/SCATTER {mod} =====================", flush=True)
    try:
        codec, cfg = load_frozen_codec(f"{CODEC_DIR}/spectro_codec_{mod}.pt", map_location="cpu")
        codec = codec.to(dev)
        bg = bool(cfg.get("bg_subtract", False))
        Fq = int(cfg["Fq"]); C = int(cfg["C"]); L = int(cfg["fsq_L"])
        rows = []   # (subset, persist_acc, stab_acc, resid_var, P) per window
        for tag, shots in [("in", SHOTS_IN), ("out", SHOTS_OUT)]:
            Xin, Xtg = load_subset(mod, cfg, shots, tag)
            if Xin is None:
                print(f"[warn] {mod} {tag}: no windows", flush=True); continue
            Ri = (baseline_residual(Xin, sigma=BG_SIGMA)[1] if bg else Xin).cpu()
            Rt = (baseline_residual(Xtg, sigma=BG_SIGMA)[1] if bg else Xtg).cpu()
            Rs = torch.roll(Rt, shifts=1, dims=-1)                 # 1-frame (~0.5ms) time shift
            ci = torch.cat([enc(codec, Ri[i:i+64]) for i in range(0, Ri.shape[0], 64)], 0)
            ct = torch.cat([enc(codec, Rt[i:i+64]) for i in range(0, Rt.shape[0], 64)], 0)
            cs = torch.cat([enc(codec, Rs[i:i+64]) for i in range(0, Rs.shape[0], 64)], 0)
            persist = (ci == ct).float().mean(-1).mean(-1).numpy()   # per-window persistence codeacc
            stab = (ct == cs).float().mean(-1).mean(-1).numpy()      # per-window stability codeacc
            rv = np.array([float(np.var(np.abs(Rt[w, :, MODE_LO:MODE_HI].numpy()))) for w in range(Rt.shape[0])])
            P = np.array([win_P(Rt[w].numpy()) for w in range(Rt.shape[0])])
            for w in range(Rt.shape[0]):
                rows.append((tag, float(persist[w]), float(stab[w]), float(rv[w]), float(P[w])))
            print(f"[{mod}/{tag}] N={Rt.shape[0]} stability={stab.mean():.3f} persistence={persist.mean():.3f} "
                  f"resid_var(med)={np.median(rv):.3f}", flush=True)
        if not rows:
            print(f"[warn] {mod}: nothing", flush=True); continue
        import numpy as _np
        tags = _np.array([r[0] for r in rows]); pa = _np.array([r[1] for r in rows])
        sa = _np.array([r[2] for r in rows]); rv = _np.array([r[3] for r in rows]); PP = _np.array([r[4] for r in rows])
        inm = tags == "in"; outm = tags == "out"
        # active/quiescent by pooled prominence quartiles
        P75, P25 = _np.percentile(PP, 75), _np.percentile(PP, 25)
        act = PP >= P75; qui = PP <= P25
        def m(a, msk):
            return float(a[msk].mean()) if msk.sum() else None
        r = {"task": "5+6", "modality": mod, "L": L, "random_floor": 1.0 / L,
             "n_in": int(inm.sum()), "n_out": int(outm.sum()),
             "stability_in": m(sa, inm), "stability_out": m(sa, outm),
             "stability_in_active": m(sa, inm & act), "stability_out_active": m(sa, outm & act),
             "persistence_in": m(pa, inm), "persistence_out": m(pa, outm),
             "persistence_active": m(pa, act), "persistence_quiescent": m(pa, qui),
             "corr_persist_vs_residvar": float(_np.corrcoef(pa, rv)[0, 1]) if len(pa) > 2 else None}
        all_res[mod] = r
        json.dump(r, open(OUT / f"task56_{mod}.json", "w"), indent=2)
        print(f"[stab] {mod}: stability in={r['stability_in']} out={r['stability_out']} "
              f"(active in={r['stability_in_active']} out={r['stability_out_active']}) | "
              f"persistence in={r['persistence_in']} out={r['persistence_out']} "
              f"active={r['persistence_active']} quiescent={r['persistence_quiescent']} | "
              f"corr(persist,residvar)={r['corr_persist_vs_residvar']:.2f} | random={r['random_floor']:.3f}", flush=True)
        # scatter PDF: persistence codeacc vs residual variance, colored by subset
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for msk, c, lab in [(inm, "tab:blue", "in-subset"), (outm, "tab:red", "out-subset")]:
            ax[0].scatter(rv[msk], pa[msk], s=6, alpha=0.4, c=c, label=lab)
        ax[0].set_xlabel("residual band-variance"); ax[0].set_ylabel("persistence codeacc (t vs t+1)")
        ax[0].axhline(r["random_floor"], color="k", ls=":", lw=0.8, label="random"); ax[0].legend(fontsize=8)
        ax[0].set_title(f"{mod}: what is 'easy'? codeacc vs activity")
        for msk, c, lab in [(inm, "tab:blue", "in"), (outm, "tab:red", "out")]:
            ax[1].scatter(rv[msk], sa[msk], s=6, alpha=0.4, c=c, label=lab)
        ax[1].set_xlabel("residual band-variance"); ax[1].set_ylabel("stability codeacc (1-frame shift)")
        ax[1].axhline(r["random_floor"], color="k", ls=":", lw=0.8); ax[1].legend(fontsize=8)
        ax[1].set_title(f"{mod}: stability vs activity")
        fig.suptitle(f"Task 5/6 — {mod.upper()}  (stability + bimodality scatter)", fontsize=11)
        fig.tight_layout(); fig.savefig(OUT / f"task56_{mod}.pdf"); plt.close(fig)
        print(f"[stab] {mod}: saved {OUT}/task56_{mod}.pdf", flush=True)
    except Exception as e:
        import traceback
        print(f"[WARN] {mod} failed: {e}", flush=True); traceback.print_exc()

json.dump(all_res, open(OUT / "task56_all.json", "w"), indent=2)
print("\n[stab_scatter] done", flush=True)
