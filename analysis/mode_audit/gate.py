"""IGNITE mode-loss audit — CODEC ACCEPTANCE GATE (pre-registered, no world model).

Run on a candidate (smoothed) codec BEFORE any world-model training. Reads bg_subtract
and smooth_frames from the codec cfg and applies the SAME preprocessing the codec was
trained on (residual -> temporal smooth) everywhere, then reports PASS/FAIL:

  1. stability(active)      >= 0.90   -- codes survive a 0.5 ms (1-frame) shift = structure
  2. persistence(active)    >> 0.10   -- codes now carry a forecastable dynamics signal
                                         (gate: >= 0.40; "well clear of chance")
  3. persistence(quiescent) ~  0.99   -- easy background still trivially persisted (where present)
  4. capture(active)        >= 0.69   -- gt-codes decode still renders the mode (fidelity kept)
  5. inverse-splice pass    high      -- erasing mode-patch codes removes the mode (necessity)

"active"/"quiescent" are fixed from the RAW residual (pre-smooth) band prominence
(top/bottom quartile) so the window sets are identical across smoothing levels -> the
stability<->fidelity tradeoff is read on the same windows.

Env: CODEC_DIR, MODALITIES, SHOTS_IN, SHOTS_OUT, NWIN_PER_SHOT, OUT_DIR,
     GATE_STABILITY(0.90), GATE_PERSIST_ACTIVE(0.40), GATE_CAPTURE(0.69).
Writes analysis/mode_audit/gate_<mod>.json.
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
from spectro_bg import baseline_residual, smooth_time_mag
from tokamak_foundation_model.e2e.quantizers.spectro_codec import load_frozen_codec

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODS = [m.strip() for m in os.environ.get("MODALITIES", "ece").split(",") if m.strip()]
CODEC_DIR = os.environ["CODEC_DIR"]
SHOTS_IN = os.environ.get("SHOTS_IN", "200729,190996,204811,191001").split(",")
SHOTS_OUT = os.environ.get("SHOTS_OUT", "190900,190904,201585").split(",")
DATA = os.environ.get("DATA_DIR", "/lustre/orion/fus187/proj-shared/foundation_model")
STATS = os.environ.get("STATS_PATH",
                       "/lustre/orion/fus187/proj-shared/foundation_model_meta/preprocessing_stats.pt")
NWIN_PER_SHOT = int(os.environ.get("NWIN_PER_SHOT", "400"))
OUT = Path(os.environ.get("OUT_DIR", f"{FMH}/analysis/mode_audit"))
OUT.mkdir(parents=True, exist_ok=True)
G_STAB = float(os.environ.get("GATE_STABILITY", "0.90"))
G_PA = float(os.environ.get("GATE_PERSIST_ACTIVE", "0.40"))
G_CAP = float(os.environ.get("GATE_CAPTURE", "0.69"))
BG_SIGMA = float(os.environ.get("BG_SIGMA", "8.0"))
FS, NFFT = 500_000.0, 1024
DF = FS / NFFT / 1e3
MODE_LO, MODE_HI = int(round(5.0 / DF)), int(round(40.0 / DF))


def band_prom(x_ch):
    prof = np.abs(x_ch[MODE_LO:MODE_HI]).mean(1)
    pd = prof - gaussian_filter1d(prof, 6.0)
    return pd, MODE_LO + int(np.argmax(pd)), float(pd.max())


def win_P(x):
    return max(band_prom(x[c])[2] for c in range(x.shape[0]))


def strong_ch(x):
    return int(np.argmax([band_prom(x[c])[2] for c in range(x.shape[0])]))


def mode_pixel_mask(x_ch, k=3.0):
    a = np.abs(x_ch); base = gaussian_filter1d(a, 6.0, axis=0); r = a - base
    m = np.zeros_like(a, bool); band = r[MODE_LO:MODE_HI]
    mad = np.median(np.abs(band - np.median(band))) * 1.4826 + 1e-9
    m[MODE_LO:MODE_HI] = band > k * mad
    return m


def capture(ref_ch, pred_ch):
    gp = np.abs(ref_ch[MODE_LO:MODE_HI]).mean(1); pf = np.abs(pred_ch[MODE_LO:MODE_HI]).mean(1)
    gd = gp - gaussian_filter1d(gp, 6.0); pd = pf - gaussian_filter1d(pf, 6.0)
    f0 = int(np.argmax(gd))
    return float(pd[f0] / gd[f0]) if gd[f0] > 1e-6 else np.nan


def codec_space(R, cfg):
    """residual R (already computed) -> smoothed as the codec was trained."""
    sf = int(cfg.get("smooth_frames", 0) or 0)
    return smooth_time_mag(R, sf) if sf > 1 else R


def enc(codec, x):
    with torch.no_grad():
        return codec.encode_codes(x.to(dev)).cpu()


def dec(codec, c):
    with torch.no_grad():
        return codec.decode_codes(c.to(dev)).cpu()


for mod in MODS:
    print(f"\n===================== GATE {mod}  codec={CODEC_DIR} =====================", flush=True)
    try:
        codec, cfg = load_frozen_codec(f"{CODEC_DIR}/spectro_codec_{mod}.pt", map_location="cpu")
        codec = codec.to(dev)
        bg = bool(cfg.get("bg_subtract", False)); sf = int(cfg.get("smooth_frames", 0) or 0)
        C = int(cfg["C"]); Fq = int(cfg["Fq"]); patch_f = int(cfg.get("patch_f", 8)); patch_t = int(cfg.get("patch_t", 16))
        npf = Fq // patch_f
        poc.PATCH_F = patch_f; poc.PATCH_T = patch_t
        print(f"[gate] cfg bg_subtract={bg} smooth_frames={sf} patch=({patch_f},{patch_t})", flush=True)

        def load(shots):
            Xi, Xt = [], []
            for sh in shots:
                if not (Path(DATA) / f"{sh}_processed.h5").exists():
                    continue
                try:
                    xi, xt = load_pairs(sh, DATA, STATS, C, NWIN_PER_SHOT, modality=mod)
                    Xi.append(xi); Xt.append(xt)
                except Exception as e:
                    print(f"[warn] {sh}: {e}", flush=True)
            return (torch.cat(Xi), torch.cat(Xt)) if Xi else (None, None)

        res = {"modality": mod, "codec_dir": CODEC_DIR, "bg_subtract": bg, "smooth_frames": sf}
        # ---- stability (active), in + out subset ----
        for tag, shots in [("in", SHOTS_IN), ("out", SHOTS_OUT)]:
            _, Xt = load(shots)
            if Xt is None:
                continue
            R = baseline_residual(Xt, sigma=BG_SIGMA)[1].cpu() if bg else Xt.cpu()      # raw residual
            Pw = np.array([win_P(R[w].numpy()) for w in range(R.shape[0])])
            act = np.where(Pw >= np.percentile(Pw, 75))[0]
            Rs = torch.roll(R, shifts=1, dims=-1)                                        # 0.5 ms shift
            stab = []
            for i in range(0, len(act), 64):
                idx = act[i:i + 64]
                c0 = enc(codec, codec_space(R[idx], cfg)); c1 = enc(codec, codec_space(Rs[idx], cfg))
                stab.extend((c0 == c1).float().mean(-1).mean(-1).numpy().tolist())
            res[f"stability_active_{tag}"] = float(np.median(stab))
        # ---- persistence (active/quiescent) + capture (active) + inverse-splice, in-subset ----
        Xi, Xt = load(SHOTS_IN)
        Ri = baseline_residual(Xi, sigma=BG_SIGMA)[1].cpu() if bg else Xi.cpu()
        Rt = baseline_residual(Xt, sigma=BG_SIGMA)[1].cpu() if bg else Xt.cpu()
        Pw = np.array([win_P(Rt[w].numpy()) for w in range(Rt.shape[0])])
        P75, P25 = np.percentile(Pw, 75), np.percentile(Pw, 25)
        act = np.where(Pw >= P75)[0]; qui = np.where(Pw <= P25)[0]
        ci = torch.cat([enc(codec, codec_space(Ri[i:i+64], cfg)) for i in range(0, Ri.shape[0], 64)], 0)
        ct = torch.cat([enc(codec, codec_space(Rt[i:i+64], cfg)) for i in range(0, Rt.shape[0], 64)], 0)
        pers = (ci == ct).float().mean(-1).mean(-1).numpy()
        res["persistence_active"] = float(np.median(pers[act]))
        res["persistence_quiescent"] = float(np.median(pers[qui])) if len(qui) else None
        # capture on active windows: decode(encode(smoothed GT)) vs RAW residual mode
        caps = []
        for i in range(0, len(act), 64):
            idx = act[i:i + 64]
            d = dec(codec, enc(codec, codec_space(Rt[idx], cfg)))
            for j, w in enumerate(idx):
                ch = strong_ch(Rt[w].numpy())
                caps.append(capture(Rt[w, ch].numpy(), d[j, ch].numpy()))
        res["capture_active"] = float(np.nanmedian(caps))
        # inverse splice: erase mode-patch codes in active windows -> mode should vanish
        FIRE = float(P75)
        inv_ok = inv_n = 0
        for w in act[:20]:
            xt = Rt[w].numpy()
            m = np.zeros((Fq, Rt.shape[-1]), bool)
            for c in range(C):
                m |= mode_pixel_mask(xt[c])
            pm = m[:npf * patch_f].reshape(npf, patch_f, m.shape[1] // patch_t, patch_t).any((1, 3))
            mode_pf = np.where(pm.any(1))[0]
            if len(mode_pf) == 0:
                continue
            cP = enc(codec, codec_space(Rt[w:w+1], cfg)).reshape(1, npf, -1, ci.shape[-1])
            # background codes = a quiescent window's grid
            wq = qui[0] if len(qui) else act[-1]
            cB = enc(codec, codec_space(Rt[wq:wq+1], cfg)).reshape(1, npf, -1, ci.shape[-1])
            inv = cP.clone(); inv[:, mode_pf] = cB[:, mode_pf]
            r_inv = dec(codec, inv.reshape(1, -1, ci.shape[-1]))[0].numpy()
            inv_ok += int(win_P(r_inv) < FIRE); inv_n += 1
        res["inverse_splice_pass"] = (inv_ok / inv_n) if inv_n else None
        # ---- PASS/FAIL ----
        s_in = res.get("stability_active_in", 0.0)
        checks = {
            "stability>=%.2f" % G_STAB: s_in >= G_STAB,
            "persist_active>=%.2f" % G_PA: res["persistence_active"] >= G_PA,
            "capture>=%.2f" % G_CAP: res["capture_active"] >= G_CAP,
        }
        res["checks"] = checks
        res["PASS"] = all(checks.values())
        json.dump(res, open(OUT / f"gate_{mod}.json", "w"), indent=2, default=lambda o: float(o))
        print(f"[gate] {mod} smooth={sf}: stability(active) in={res.get('stability_active_in'):.3f} "
              f"out={res.get('stability_active_out')} | persistence active={res['persistence_active']:.3f} "
              f"quiescent={res['persistence_quiescent']} | capture(active)={res['capture_active']:.3f} "
              f"| inv-splice={res['inverse_splice_pass']}", flush=True)
        print(f"[gate] {mod} smooth={sf}: CHECKS {checks} ==> {'PASS' if res['PASS'] else 'FAIL'}", flush=True)
    except Exception as e:
        import traceback
        print(f"[WARN] {mod} gate failed: {e}", flush=True); traceback.print_exc()

print("\n[gate] done", flush=True)
