"""Factorization PRE-GATE: is the shift-stable mode DESCRIPTOR forecastable at 50 ms?

The factorization fallback predicts a low-dim descriptor (band-power profile + peak
freq/amplitude) instead of exact FSQ codes. Before building that head, test whether
its TARGET is even forecastable: on mode-active windows, does the descriptor at
window t predict window t+1 (persistence), well above a shuffled control? Physics
says mode frequency persists 0.85-0.99 over 400 ms, so this should clear the floor
by a wide margin -- confirming "all modalities predictable" at the statistics level.

No model, no codec training -- pure measurement on the codec-input spectrograms
(load_pairs gives consecutive windows Xi=t, Xt=t+1 with channels_to_use applied).

Reports per modality:  freq_persist vs freq_shuffled, bandpower_corr vs shuffled.
GREEN (build the head) if freq_persist-shuffled >= 0.2 AND bp_corr-shuffled >= 0.2.

Env: SHOTS, NWIN_PER_SHOT, OUT_DIR.
"""
import json
import os
import sys
from pathlib import Path

FMH = "/lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub"
for p in (f"{FMH}/src", f"{FMH}/scripts/training", f"{FMH}/analysis/mode_audit"):
    if p not in sys.path:
        sys.path.insert(0, p)
import numpy as np
import torch
import poc_fsq_stageB as poc
from poc_fsq_stageB import load_pairs
from dist_gate import band_prom, band_profile, strong_ch, win_P, TOL_BINS

DATA = os.environ.get("DATA_DIR", "/lustre/orion/fus187/proj-shared/foundation_model")
STATS = os.environ.get("STATS_PATH",
                       "/lustre/orion/fus187/proj-shared/foundation_model_meta/preprocessing_stats.pt")
SHOTS = os.environ.get("SHOTS", "200729,190996,204811,191001").split(",")
NWIN = int(os.environ.get("NWIN_PER_SHOT", "150"))
OUT = Path(os.environ.get("OUT_DIR", f"{FMH}/analysis/mode_audit/descriptor_pregate"))
OUT.mkdir(parents=True, exist_ok=True)
CH = {"ece": 40, "co2": 4, "bes": 16, "mhr": 6}   # channels_to_use counts (load_pairs applies the slice)
poc.PATCH_F = 8; poc.PATCH_T = 16
rng = np.random.RandomState(0)


def _peakfreq(x_ch):
    return band_prom(x_ch)[1]


results = {}
for mod, C in CH.items():
    Xi, Xt = [], []
    for sh in SHOTS:
        if not (Path(DATA) / f"{sh}_processed.h5").exists():
            continue
        try:
            xi, xt = load_pairs(sh, DATA, STATS, C, NWIN, modality=mod)
            Xi.append(xi); Xt.append(xt)
        except Exception as e:
            print(f"[warn] {mod} {sh}: {e}", flush=True)
    if not Xi:
        print(f"[warn] {mod}: no data", flush=True); continue
    Xi = torch.cat(Xi, 0).float(); Xt = torch.cat(Xt, 0).float()   # (N,C,F,T)
    N = Xi.shape[0]
    Pw = np.array([win_P(Xt[i].numpy()) for i in range(N)])
    act = np.where(Pw >= np.percentile(Pw, 75))[0]
    perm = rng.permutation(act)

    freq_ok, bp = [], []
    freq_sh, bp_sh = [], []
    for k, i in enumerate(act):
        ch = strong_ch(Xt[i].numpy())
        f_t = _peakfreq(Xt[i, ch].numpy())
        # persistence: input window t predicts target t+1
        f_i = _peakfreq(Xi[i, ch].numpy())
        freq_ok.append(abs(f_i - f_t) <= TOL_BINS)
        a = band_profile(Xi[i, ch].numpy()); b = band_profile(Xt[i, ch].numpy())
        if a.std() > 1e-9 and b.std() > 1e-9:
            bp.append(float(np.corrcoef(a, b)[0, 1]))
        # shuffled control: unrelated input window j
        j = perm[k]
        chj = strong_ch(Xi[j].numpy())
        f_j = _peakfreq(Xi[j, chj].numpy())
        freq_sh.append(abs(f_j - f_t) <= TOL_BINS)
        aj = band_profile(Xi[j, chj].numpy())
        if aj.std() > 1e-9 and b.std() > 1e-9:
            bp_sh.append(float(np.corrcoef(aj, b)[0, 1]))

    fp = float(np.mean(freq_ok)) if freq_ok else float("nan")
    fps = float(np.mean(freq_sh)) if freq_sh else float("nan")
    bpc = float(np.median(bp)) if bp else float("nan")
    bpcs = float(np.median(bp_sh)) if bp_sh else float("nan")
    green = bool((fp - fps) >= 0.2 and (bpc - bpcs) >= 0.2)
    r = {"modality": mod, "n_windows": N, "n_active": int(len(act)),
         "freq_persist": fp, "freq_shuffled": fps, "freq_margin": fp - fps,
         "bandpower_corr": bpc, "bandpower_shuffled": bpcs, "bandpower_margin": bpc - bpcs,
         "GREEN": green}
    results[mod] = r
    print(f"[pregate] {mod}: freq_persist={fp:.3f} (shuffled {fps:.3f}, margin {fp-fps:+.3f}) | "
          f"bandpower_corr={bpc:.3f} (shuffled {bpcs:.3f}, margin {bpc-bpcs:+.3f}) "
          f"==> {'GREEN (forecastable)' if green else 'not clear'}", flush=True)

json.dump(results, open(OUT / "descriptor_pregate.json", "w"), indent=2,
          default=lambda o: float(o) if hasattr(o, "item") else o)
print(f"\n[pregate] wrote {OUT}/descriptor_pregate.json", flush=True)
print("[pregate] done", flush=True)
