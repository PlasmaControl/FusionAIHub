"""IGNITE mode-loss audit — Task 3: k1 teacher-forced render triad + codeacc split.

Reuses the REAL trainer path (forward_batch -> token_slices -> head.code_logits /
head.encode_target), so the codes/logits are exactly those the CE loss saw.
forward_batch already returns targets in RESIDUAL space for a bg_subtract codec,
so encode_target(targets) are the correct R-space GT codes.

For 20 strongest-mode ece windows (band-restricted 5-40 kHz detector on the GT
residual), render three ways through the SAME frozen decoder:
  (a) argmax codes   (b) independent multinomial sample (T=1)   (c) GT codes
Metrics per render: mode-capture, peak-match, profile-corr, tvr.
PLUS code accuracy split: mode-patch tokens vs background tokens (argmax vs GT).
The split is the tie-breaker between imbalance and an upstream representation loss.

Env: CKPT, SHOTS, N_MODE_WIN, OUT_DIR, Z_POS.  Writes task3_ece.json + PDF.
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
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter1d
from eval_e2e_animation_tokamak import load_model
from train_e2e_stage1 import build_datasets, forward_batch, _core
from tokamak_foundation_model.data.data_loader import collate_fn

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT = os.environ.get("CKPT", "/lustre/orion/fus187/proj-shared/models/e2e_step2_fsq_finer/e2e_stage1_latest.pt")
MOD = os.environ.get("MOD", "ece")
SHOTS = os.environ.get("SHOTS", "200729,190996,204811").split(",")
N_MODE_WIN = int(os.environ.get("N_MODE_WIN", "20"))
Z_POS = float(os.environ.get("Z_POS", "4.0"))
OUT = Path(os.environ.get("OUT_DIR", f"{FMH}/analysis/mode_audit"))
OUT.mkdir(parents=True, exist_ok=True)
FS, NFFT = 500_000.0, 1024
DF = FS / NFFT / 1e3
MODE_LO, MODE_HI = int(round(5.0 / DF)), int(round(40.0 / DF))


def band_prom(x_ch):
    prof = np.abs(x_ch[MODE_LO:MODE_HI]).mean(1)
    pd = prof - gaussian_filter1d(prof, 6.0)
    mad = np.median(np.abs(pd - np.median(pd))) * 1.4826 + 1e-9
    f0 = int(np.argmax(pd))
    return pd, MODE_LO + f0, float(pd[f0] / mad)


def mode_pixel_mask(x_ch, k=3.0):
    a = np.abs(x_ch); base = gaussian_filter1d(a, 6.0, axis=0); r = a - base
    m = np.zeros_like(a, bool); band = r[MODE_LO:MODE_HI]
    mad = np.median(np.abs(band - np.median(band))) * 1.4826 + 1e-9
    m[MODE_LO:MODE_HI] = band > k * mad
    return m


model, ckpt = load_model(Path(CKPT), dev); model.eval()
core = _core(model)
a = ckpt["args"]
dn = [d["name"] for d in ckpt["diagnostics"]]; an = [c["name"] for c in ckpt["actuators"]]
dd = Path(a["data_dir"]); stats = torch.load(a["stats_path"], weights_only=False)
sfiles = [dd / f"{s}_processed.h5" for s in SHOTS]; sfiles = [f for f in sfiles if f.exists()]
_, ds = build_datasets(dd, sfiles, sfiles, stats, a["chunk_duration_s"],
                       a.get("prediction_horizon_s", a["chunk_duration_s"]), a["step_size_s"],
                       a["warmup_s"], dn, an, Path(f"{FMH}/eval_runs/modecode_cache"),
                       history_windows=int(a.get("history_windows", 1)))
ld = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2, collate_fn=collate_fn)
head = core.diag_heads[MOD]
patch_f = int(head.codec.patch_f); patch_t = int(head.codec.patch_t)
print(f"[triad] ckpt={CKPT} mod={MOD} shots={[f.stem for f in sfiles]} patch=({patch_f},{patch_t})", flush=True)

GT, RA, RS, RG = [], [], [], []           # GT-R, argmax, sample, gt-codes renders
TGTC, ARGC = [], []                        # gt codes, argmax codes
with torch.no_grad():
    for batch in ld:
        preds, din, targets, masks, slices = forward_batch(model, batch, dev)
        if MOD not in targets:
            continue
        tgt = torch.nan_to_num(targets[MOD].float())
        tgt_codes = head.encode_target(tgt)                        # (B,ntok,dim) R-space
        logits = head.code_logits(slices[MOD])                     # (B,ntok,dim,L)
        arg_codes = logits.argmax(-1)
        smp_codes = head.sample_codes(logits, temperature=1.0)
        GT.append(tgt.cpu()); RG.append(head.decode(tgt_codes).float().cpu())
        RA.append(head.decode(arg_codes).float().cpu()); RS.append(head.decode(smp_codes).float().cpu())
        TGTC.append(tgt_codes.cpu()); ARGC.append(arg_codes.cpu())

g = torch.cat(GT).numpy(); ra = torch.cat(RA).numpy(); rs = torch.cat(RS).numpy(); rg = torch.cat(RG).numpy()
tgtc = torch.cat(TGTC); argc = torch.cat(ARGC)
N, C, F, T = g.shape
npf = F // patch_f; npt = tgtc.shape[1] // npf
# global strongest-mode channel + per-window z; pick top-N mode windows
zwin = np.array([max(band_prom(g[w, c])[2] for c in range(C)) for w in range(N)])
ch = int(np.argmax([sum((np.abs(g[w, c, MODE_LO:MODE_HI]).mean(1) -
                         gaussian_filter1d(np.abs(g[w, c, MODE_LO:MODE_HI]).mean(1), 6.0)).max()
                        for w in range(N)) for c in range(C)]))
sel = np.argsort(-zwin)[:N_MODE_WIN]
print(f"[triad] N={N} strong-ch={ch} sel={len(sel)} z(sel) p50={np.median(zwin[sel]):.1f}", flush=True)

tol = max(1, int(2.0 / DF))
def _prom(a4, w):
    p = np.abs(a4[w, ch, MODE_LO:MODE_HI]).mean(1); return p - gaussian_filter1d(p, 6.0)
def capture(pp, w):
    gd = _prom(g, w); pd = _prom(pp, w); f0 = int(np.argmax(gd))
    return float(pd[f0] / gd[f0]) if gd[f0] > 1e-6 else np.nan
def peakmatch(pp, w):
    return abs(int(np.argmax(_prom(g, w))) - int(np.argmax(_prom(pp, w)))) <= tol
def profcorr(pp, w):
    pa = np.abs(g[w, ch, MODE_LO:MODE_HI]).mean(1); pb = np.abs(pp[w, ch, MODE_LO:MODE_HI]).mean(1)
    return float(np.corrcoef(pa, pb)[0, 1]) if pa.std() > 1e-9 and pb.std() > 1e-9 else np.nan
def tvr(pp):
    return float(pp[sel][:, ch, :MODE_HI].var(-1).mean() / (g[sel][:, ch, :MODE_HI].var(-1).mean() + 1e-9))

def metrics(pp, name):
    return {"render": name,
            "mode_capture": float(np.nanmedian([capture(pp, w) for w in sel])),
            "peak_match": float(np.mean([peakmatch(pp, w) for w in sel])),
            "profile_corr": float(np.nanmedian([profcorr(pp, w) for w in sel])),
            "tvr": tvr(pp)}

# codeacc split: mode-patch tokens vs background tokens (argmax vs GT), over sel windows
mode_tok, bg_tok, mode_hit, bg_hit = 0, 0, 0, 0
for w in sel:
    m = np.zeros((F, T), bool)
    for c in range(C):
        m |= mode_pixel_mask(g[w, c])
    pm = m[:npf * patch_f].reshape(npf, patch_f, npt, patch_t).any((1, 3)).reshape(-1)  # (ntok,)
    hit = (argc[w] == tgtc[w]).float().mean(-1).numpy()   # per-token acc over dims
    mode_tok += int(pm.sum()); bg_tok += int((~pm).sum())
    mode_hit += float(hit[pm].sum()); bg_hit += float(hit[~pm].sum())
res = {"task": 3, "modality": MOD, "ckpt": CKPT, "n_windows": int(N), "n_sel_mode": int(len(sel)),
       "strong_channel": ch, "renders": [metrics(ra, "argmax"), metrics(rs, "sample"), metrics(rg, "gt_codes")],
       "codeacc_mode_patch": (mode_hit / mode_tok if mode_tok else None),
       "codeacc_background": (bg_hit / bg_tok if bg_tok else None),
       "n_mode_tokens": mode_tok, "n_bg_tokens": bg_tok}
# interpretation
am = res["renders"][0]; sm = res["renders"][1]; gm = res["renders"][2]
if gm["mode_capture"] < 0.4:
    interp = "CODEC problem (GT-code render already loses modes) — cross-check Task 2"
elif res["codeacc_mode_patch"] is not None and res["codeacc_mode_patch"] < 0.5 * (res["codeacc_background"] or 1):
    interp = "REPRESENTATION problem upstream (mode-patch codeacc << background)"
elif am["mode_capture"] < 0.3 and sm["tvr"] < 0.5:
    interp = "IMBALANCE/distribution (argmax deletes modes + sample FLAT)"
elif am["mode_capture"] < 0.3 and sm["tvr"] > 1.5:
    interp = "SAMPLING-structure problem (argmax deletes + sample SPECKLES)"
else:
    interp = "mixed/inconclusive — see per-render numbers"
res["interpretation"] = interp
json.dump(res, open(OUT / f"task3_{MOD}.json", "w"), indent=2)
print(f"[triad] renders: " + " | ".join(f"{r['render']} cap={r['mode_capture']:.2f} pk={r['peak_match']:.2f} "
      f"prof={r['profile_corr']:.2f} tvr={r['tvr']:.2f}" for r in res["renders"]), flush=True)
print(f"[triad] codeacc  mode-patch={res['codeacc_mode_patch']}  background={res['codeacc_background']}  "
      f"(tokens {mode_tok}/{bg_tok})", flush=True)
print(f"[triad] INTERPRETATION ==> {interp}", flush=True)

# PDF: 4 example mode windows, rows = GT | argmax | sample | gt-codes
freqs = np.arange(F) * DF; fmax = min(F, int(60 / DF))
ex = sel[:4]
fig, ax = plt.subplots(4, len(ex), figsize=(3.2 * len(ex), 10))
rows = [("GT", g), ("argmax", ra), ("sample", rs), ("gt-codes", rg)]
for j, w in enumerate(ex):
    for i, (t, arr) in enumerate(rows):
        A = ax[i, j] if len(ex) > 1 else ax[i]
        A.imshow(np.abs(arr[w, ch, :fmax]), origin="lower", aspect="auto", extent=[0, T, 0, freqs[fmax]])
        A.set_title(f"{t} w{w}" + (f" z={zwin[w]:.1f}" if i == 0 else ""), fontsize=8)
        if j == 0:
            A.set_ylabel("kHz")
fig.suptitle(f"Task 3 triad — {MOD.upper()} ch{ch}  ({interp})", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / f"task3_{MOD}.pdf"); plt.close(fig)
print(f"[triad] saved {OUT}/task3_{MOD}.pdf", flush=True)
print("\n[triad] done", flush=True)
