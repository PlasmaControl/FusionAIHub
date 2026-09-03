"""IGNITE DECISION TEST — persistence at 50 ms under exact + ±1-level (ordinal) metric,
on the frozen s16 smoothed codec. Encoder ONLY (no world model). Diagnostic only.

Decides the plan branch (pre-registered):
  persistence_tol1(active, mode-band) >= ~0.5   -> SKIP codec retrain; freeze s16,
                                                   head retrain with soft/ordinal CE.
  ~0.15-0.25                                     -> codec retrain w/ temporal-consistency
                                                   loss + noise injection (Step 1).
  0.30-0.50                                      -> AMBIGUOUS: report + STOP, no branch.

Pins:
  codes(t) = encode(s16-smoothed residual GT window at t), int FSQ levels (ntok,dim).
  50 ms persistence pair = window i vs window i+5 (step_size 0.01 s -> 5 steps = 50 ms) =
    the world model's prediction stride. (NOT the +1-STFT-frame stability shift.)
  exact = mean_(tok,dim)[c_t==c_{t+1}] ; tol1 = mean_(tok,dim)[|c_t-c_{t+1}|<=1].
  exact-chance = 1/L. tol1-chance = mean_dim sum_k p_d(k)(p_d(k-1)+p_d(k)+p_d(k+1)),
    p_d = empirical per-dim marginal over the cell (proper ordinal chance).
  shuffled control = agreement on RANDOM same-shot/same-stratum pairs (empirical floor).
Strata: active/quiescent (band-prominence quartiles) x in/out-of-codec-subset. Same shots
as the sweep. Extra cuts: mode-band tokens (5-40 kHz freq-patches), lag curve {1,2,4,8}
windows, per-dim tol1. Env: CODEC_DIR, SHOTS_IN, SHOTS_OUT, NWIN_PER_SHOT, OUT_DIR.
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
CODEC_DIR = os.environ.get("CODEC_DIR", "/lustre/orion/fus187/proj-shared/models/fsq_smooth_ece_s16")
MOD = "ece"
SHOTS_IN = os.environ.get("SHOTS_IN", "200729,190996,204811,191001").split(",")
SHOTS_OUT = os.environ.get("SHOTS_OUT", "190900,190904,201585").split(",")
DATA = os.environ.get("DATA_DIR", "/lustre/orion/fus187/proj-shared/foundation_model")
STATS = os.environ.get("STATS_PATH", "/lustre/orion/fus187/proj-shared/foundation_model_meta/preprocessing_stats.pt")
NWIN_PER_SHOT = int(os.environ.get("NWIN_PER_SHOT", "4000"))     # large -> stride 1 -> consecutive windows
OUT = Path(os.environ.get("OUT_DIR", f"{FMH}/analysis/mode_audit"))
OUT.mkdir(parents=True, exist_ok=True)
BG_SIGMA = 8.0
FS, NFFT = 500_000.0, 1024
DF = FS / NFFT / 1e3
MODE_LO, MODE_HI = int(round(5.0 / DF)), int(round(40.0 / DF))   # 5-40 kHz bins
LAG_STEP = 5                                                     # 5 * 10 ms = 50 ms = 1 window-unit
assert Path(STATS).exists(), f"stats file missing: {STATS}"
print(f"[ptol] STATS (s16-matched, sweep default) = {STATS}", flush=True)

codec, cfg = load_frozen_codec(f"{CODEC_DIR}/spectro_codec_{MOD}.pt", map_location="cpu")
codec = codec.to(dev)
sf = int(cfg.get("smooth_frames", 0) or 0); bg = bool(cfg.get("bg_subtract", False))
C = int(cfg["C"]); Fq = int(cfg["Fq"]); L = int(cfg["fsq_L"]); DIM = int(cfg["fsq_dim"])
PF = int(cfg.get("patch_f", 8)); PT = int(cfg.get("patch_t", 16))
NPF, NPT = Fq // PF, None
assert sf == 16, f"expected s16 smooth_frames=16, got {sf}"
poc.PATCH_F = PF; poc.PATCH_T = PT
print(f"[ptol] codec={CODEC_DIR} smooth_frames={sf} bg={bg} patch=({PF},{PT}) L={L} dim={DIM}", flush=True)


def win_P(x):                                    # (C,F,T) raw residual -> band-peak prominence
    return max(float((np.abs(x[c, MODE_LO:MODE_HI]).mean(1) -
                      gaussian_filter1d(np.abs(x[c, MODE_LO:MODE_HI]).mean(1), 6.0)).max())
               for c in range(x.shape[0]))


def enc_codes(xb):                               # (b,C,F,T) codec-space -> (b,ntok,dim) int16
    with torch.no_grad():
        return codec.encode_codes(xb.to(dev)).cpu().to(torch.int16)


# ---- per-shot: load consecutive windows, residual+smooth, encode ----
shot_codes, shot_P, shot_sub = [], [], []        # per-shot ordered codes / prominence / subset
for sub, shots in [("in", SHOTS_IN), ("out", SHOTS_OUT)]:
    for sh in shots:
        if not (Path(DATA) / f"{sh}_processed.h5").exists():
            print(f"[skip] {sh}", flush=True); continue
        try:
            X, _ = load_pairs(sh, DATA, STATS, C, NWIN_PER_SHOT, modality=MOD)   # ordered, stride 1
        except Exception as e:
            print(f"[warn] {sh}: {e}", flush=True); continue
        Rraw = (baseline_residual(X, sigma=BG_SIGMA)[1] if bg else X).cpu()       # raw residual (for detect)
        Rc = smooth_time_mag(Rraw, sf)                                            # codec-space (smoothed)
        codes = torch.cat([enc_codes(Rc[i:i + 64]) for i in range(0, Rc.shape[0], 64)], 0).numpy()
        P = np.array([win_P(Rraw[w].numpy()) for w in range(Rraw.shape[0])])
        shot_codes.append(codes); shot_P.append(P); shot_sub.append(sub)
        print(f"[ptol] {sh} ({sub}): {codes.shape[0]} consecutive windows encoded", flush=True)

allP = np.concatenate(shot_P)
P75, P25 = np.percentile(allP, 75), np.percentile(allP, 25)
NTOK = shot_codes[0].shape[1]; NPT = NTOK // NPF
mode_pf = [pf for pf in range(NPF) if not (pf * PF > MODE_HI or (pf + 1) * PF < MODE_LO)]
mode_tok = np.array([pf * NPT + pt for pf in mode_pf for pt in range(NPT)])
print(f"[ptol] ntok={NTOK} npf={NPF} npt={NPT} mode freq-patches={mode_pf[0]}..{mode_pf[-1]} "
      f"({len(mode_tok)} mode-band tokens); active>={P75:.3f} quiescent<={P25:.3f}", flush=True)


def tol1_chance(codes_MD):                        # (M,dim) -> mean-over-dim ordinal chance
    ch = []
    for d in range(DIM):
        c = np.bincount(codes_MD[:, d], minlength=L).astype(np.float64); p = c / c.sum()
        pm1 = np.concatenate([[0], p[:-1]]); pp1 = np.concatenate([p[1:], [0]])
        ch.append(float((p * (pm1 + p + pp1)).sum()))
    return float(np.mean(ch))


def cell_pairs(mask_fn, tok=None):
    """collect (c_i, c_{i+5}) pairs where TARGET (i+5) passes mask_fn(P_target, sub)."""
    A, B, tgt_codes = [], [], []
    for codes, P, sub in zip(shot_codes, shot_P, shot_sub):
        n = codes.shape[0]
        for i in range(n - LAG_STEP):
            if mask_fn(P[i + LAG_STEP], sub):
                ci = codes[i]; cj = codes[i + LAG_STEP]
                if tok is not None:
                    ci = ci[tok]; cj = cj[tok]
                A.append(ci); B.append(cj); tgt_codes.append(cj)
    if not A:
        return None
    return np.stack(A), np.stack(B)


def metrics(A, B):
    exact = float((A == B).mean()); tol1 = float((np.abs(A.astype(int) - B.astype(int)) <= 1).mean())
    flat = np.concatenate([A.reshape(-1, A.shape[-1]), B.reshape(-1, B.shape[-1])], 0)
    t1c = tol1_chance(flat)
    # shuffled control: permute B rows (breaks temporal pairing)
    perm = np.random.RandomState(0).permutation(len(B))
    sh_exact = float((A == B[perm]).mean()); sh_tol1 = float((np.abs(A.astype(int) - B[perm].astype(int)) <= 1).mean())
    return {"n_pairs": int(len(A)), "exact": round(exact, 4), "tol1": round(tol1, 4),
            "exact_chance": round(1.0 / L, 4), "tol1_chance_empirical": round(t1c, 4),
            "shuffled_exact": round(sh_exact, 4), "shuffled_tol1": round(sh_tol1, 4)}


res = {"codec": CODEC_DIR, "smooth_frames": sf, "stats": STATS, "lag_ms_per_unit": 50,
       "definition": "50ms pair = window i vs i+5 (step 0.01s); stratum by target(i+5) prominence"}
# ---- 2x2 table ----
res["strata"] = {}
for sname, pcond in [("active", lambda p: p >= P75), ("quiescent", lambda p: p <= P25)]:
    for subn in ["in", "out"]:
        AB = cell_pairs(lambda p, s, pc=pcond, sn=subn: pc(p) and s == sn)
        res["strata"][f"{sname}_{subn}"] = metrics(*AB) if AB else {"n_pairs": 0}
        c = res["strata"][f"{sname}_{subn}"]
        print(f"[ptol] {sname:9s} {subn}: n={c.get('n_pairs')} exact={c.get('exact')} tol1={c.get('tol1')} "
              f"| chance exact={c.get('exact_chance')} tol1={c.get('tol1_chance_empirical')} "
              f"| shuffled tol1={c.get('shuffled_tol1')}", flush=True)

# ---- HEADLINE: mode-band tokens, active (in+out) ----
ABm = cell_pairs(lambda p, s: p >= P75, tok=mode_tok)
res["mode_band_active"] = metrics(*ABm) if ABm else {"n_pairs": 0}
mb = res["mode_band_active"]
print(f"[ptol] *** MODE-BAND active: n={mb.get('n_pairs')} exact={mb.get('exact')} "
      f"tol1={mb.get('tol1')} (tol1-chance={mb.get('tol1_chance_empirical')}, "
      f"shuffled tol1={mb.get('shuffled_tol1')}) ***", flush=True)

# ---- lag curve (active, all tokens) ----
lag_units = [1, 2, 4, 8]
res["lag_curve_tol1_active"] = {}
for Lu in lag_units:
    step = LAG_STEP * Lu; A, B = [], []
    for codes, P, sub in zip(shot_codes, shot_P, shot_sub):
        n = codes.shape[0]
        for i in range(n - step):
            if P[i + step] >= P75:
                A.append(codes[i]); B.append(codes[i + step])
    if A:
        A = np.stack(A); B = np.stack(B)
        res["lag_curve_tol1_active"][f"{Lu*50}ms"] = round(float((np.abs(A.astype(int) - B.astype(int)) <= 1).mean()), 4)
print(f"[ptol] lag curve (active tol1): {res['lag_curve_tol1_active']}", flush=True)

# ---- per-dim tol1 (active) ----
Aa, Ba = cell_pairs(lambda p, s: p >= P75)
perdim = [float((np.abs(Aa[:, :, d].astype(int) - Ba[:, :, d].astype(int)) <= 1).mean()) for d in range(DIM)]
res["per_dim_tol1_active_sorted"] = sorted([round(v, 3) for v in perdim], reverse=True)

# ---- branch verdict (pre-registered) ----
h = mb.get("tol1")
if h is None:
    branch = "NO DATA"
elif h >= 0.5:
    branch = f"SKIP codec retrain -> freeze s16 + soft/ordinal-CE head retrain (tol1={h} >= 0.5)"
elif h <= 0.25:
    branch = f"CODEC RETRAIN (temporal-consistency + noise) proceeds (tol1={h} <= 0.25)"
elif 0.30 <= h <= 0.50:
    branch = f"AMBIGUOUS (tol1={h} in 0.30-0.50) -> report + STOP, pick no branch"
else:
    branch = f"tol1={h} in (0.25,0.30) gap -> lean codec-retrain; flag for judgement"
res["plan_branch"] = branch
print(f"[ptol] PLAN BRANCH: {branch}", flush=True)

json.dump(res, open(OUT / "persistence_tol_s16.json", "w"), indent=2)

# ---- PDF: lag curve + per-dim ----
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
lc = res["lag_curve_tol1_active"]
ax[0].plot([int(k[:-2]) for k in lc], list(lc.values()), marker="o")
ax[0].axhline(mb.get("tol1_chance_empirical", 0), color="k", ls=":", lw=0.8, label="tol1 chance")
ax[0].set_xlabel("lag (ms)"); ax[0].set_ylabel("tol1 persistence (active)"); ax[0].set_ylim(0, 1); ax[0].legend(fontsize=8)
ax[0].set_title("s16 tol1 persistence vs lag")
ax[1].plot(range(DIM), res["per_dim_tol1_active_sorted"], marker=".")
ax[1].set_xlabel("FSQ dim (sorted)"); ax[1].set_ylabel("tol1 persistence (active, 50 ms)"); ax[1].set_ylim(0, 1)
ax[1].set_title("per-dim tol1 persistence (sorted)")
fig.suptitle(f"s16 persistence-tol1 (50 ms) — mode-band active tol1={mb.get('tol1')}", fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "persistence_tol_s16.pdf"); plt.close(fig)
print(f"[ptol] wrote {OUT}/persistence_tol_s16.json + .pdf", flush=True)
print("\n[ptol] done", flush=True)
