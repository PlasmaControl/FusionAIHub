"""VALIDATE the mode-track metrics against a known-correct 5-reference ordering.

`spec_nrmse` ranks the BLUR best (lr1e4d6 0.9010 with hf_ratio 0.0143 and no visible
structure, vs ms2_s1 1.0972 / 0.8258 which reproduces the rising 50->65 kHz mode track). Any
replacement ranking metric must therefore reproduce this ordering on real held-out windows:

    GT vs itself      perfect
    rank-192 oracle   WELL      (out-of-sample PCA; renders burst columns with real texture)
    ms2_s1            ABOVE the blur
    lr1e4d6           BADLY     (featureless; nrmse ranks it BEST)
    ~tmean            BADLY     (no tracks at all)

A candidate that puts lr1e4d6 above ms2_s1 has reproduced the nrmse failure and is REJECTED.
Everything is scored on the SAME windows, in the per-freq-z space the arms are audited in, and
accumulated per chunk so the arrays stay small.

Usage: _specport_metric_validation.py <n_windows>
"""
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, "src")
sys.path.insert(0, "analysis")
from render_codec_recon_figs import load_spectro_codec

from tokamak_foundation_model.ignite import gate as gate_mod
from tokamak_foundation_model.ignite import spike
from tokamak_foundation_model.ignite import train_codec as tc
from tokamak_foundation_model.ignite.config import SpectroCodecConfig

CACHE = "/lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/eval_runs/codec_recon_figs/_cache"
M = "/lustre/orion/fus187/proj-shared/models"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 240
NTR = 400
CH = 24
K = 192
dev = "cuda" if torch.cuda.is_available() else "cpu"
STATS = f"{M}/ignite_codecs_noinorm/stats/codec_mhr_perfreq_stats.pt"
ARMS = [("ms2_s1", f"{M}/ignite_codecs_mhr_v1k_seeds/ms2_s1/codec_best.pt"),
        ("lr1e4d6", f"{M}/ignite_codecs_mhr_lr/l_1e4_d6/codec_best.pt")]

allsh = spike.discover_shots(tc.DEFAULT_DATA_DIR)
_pool = allsh[:-16]
tr_shots = _pool[:: max(1, len(_pool) // 40)][:40]
te_shots = allsh[-16:][:4]


def base_cfg():
    cfg = SpectroCodecConfig(channels=tc.modality_channels("mhr"))
    st = torch.load(STATS, map_location="cpu", weights_only=False)
    cfg.logpow_freq_mean = torch.as_tensor(st["mean"], dtype=torch.float32).tolist()
    cfg.logpow_freq_std = torch.as_tensor(st["std"], dtype=torch.float32).tolist()
    cfg.logpow_standardize = True
    return cfg


def stream(cfg, shots, n, tag):
    ds = tc.CodecPairDataset("mhr", shots, cfg, data_dir=tc.DEFAULT_DATA_DIR,
                             lengths_cache_path=f"{CACHE}/codec_mhr_{tag}_lengths.pt")
    idx = np.linspace(0, len(ds) - 1, min(n, len(ds))).astype(int).tolist()
    dl = DataLoader(Subset(ds, idx), batch_size=8, num_workers=6, shuffle=False)
    return np.concatenate([a.numpy() for a, _b in dl], 0)


cfg = base_cfg()
Xte = stream(cfg, te_shots, N, "audit4")
print(f"held-out windows {Xte.shape} from {te_shots}", flush=True)

# ---- build every reference's reconstruction on exactly these windows -------------------
recons = {}
recons["GT_self"] = Xte
recons["tmean"] = np.ascontiguousarray(
    np.broadcast_to(Xte.mean(axis=-1, keepdims=True), Xte.shape))

Xtr = stream(cfg, tr_shots, NTR, "readout_tr")
A = torch.from_numpy(Xtr.reshape(Xtr.shape[0], -1)).to(dev, torch.float32)
mu = A.mean(0, keepdim=True)
A -= mu
ev, V = torch.linalg.eigh((A @ A.T).double().cpu())     # host eigh: ROCm path is unstable
ev, V = ev.flip(0).to(dev)[:K], V.flip(1).to(dev)[:, :K]
W = (A.T @ V.float()) / ev.float().sqrt()[None, :]
del A, V
out = []
for i in range(0, Xte.shape[0], CH):
    t = Xte[i:i + CH]
    b = torch.from_numpy(t.reshape(t.shape[0], -1)).to(dev, torch.float32) - mu
    out.append(((b @ W) @ W.T + mu).reshape(t.shape).cpu().numpy())
    del b
recons["oracle192"] = np.concatenate(out, 0)
del W, mu, Xtr, out
if dev == "cuda":
    torch.cuda.empty_cache()

for label, path in ARMS:
    codec, acfg, _ck = load_spectro_codec(path)
    codec = codec.to(dev).eval()
    out = []
    with torch.no_grad():
        for i in range(0, Xte.shape[0], CH):
            x = torch.from_numpy(Xte[i:i + CH]).to(dev)
            out.append(codec(x)["recon"].cpu().numpy())
    recons[label] = np.concatenate(out, 0)
    del codec
    if dev == "cuda":
        torch.cuda.empty_cache()

# ---- score every reference, accumulating per chunk ------------------------------------
ORDER = ["GT_self", "oracle192", "ms2_s1", "lr1e4d6", "tmean"]
rows = {}
for name in ORDER:
    R = recons[name]
    acc = []
    for i in range(0, Xte.shape[0], CH):
        t, r = Xte[i:i + CH], R[i:i + CH]
        m = gate_mod.full_spectro_metrics(r, t, band_bins=None)
        d = gate_mod.decode_fidelity(r, t)   # no patch args: skip the FFT lattice
        d["patch_lattice_ratio"] = float("nan")  # reported by the audit instead
        # detrended=True is REQUIRED here: mode_structure_metrics made it opt-in for
        # audit speed, and this harness indexes mode_track_f1_detr directly.
        s = gate_mod.mode_structure_metrics(r, t, band_bins=None, detrended=True)
        acc.append([m["spec_nrmse"], d["sharpness"], d["patch_lattice_ratio"],
                    s["mode_track_f1"], s["mode_track_f1_detr"], s["ridge_traj_corr"],
                    s["spectral_contrast_ratio"], s["ms_ssim"]])
    rows[name] = np.nanmean(np.array(acc), axis=0)
    print(f"  scored {name}", flush=True)

hdr = (f"{'reference':<12}{'ms_ssim':>9}{'mode_f1':>9}{'f1_detr':>9}{'ridge':>8}"
       f"{'contrast':>10}{'spec_nrmse':>12}{'hf_ratio':>10}{'lattice':>9}")
print()
print("=== 5-REFERENCE VALIDATION (720-window pool, per-freq-z, full 0-250 kHz) ===")
print(hdr)
print("-" * len(hdr))
for name in ORDER:
    v = rows[name]
    print(f"{name:<12}{v[7]:>9.4f}{v[3]:>9.4f}{v[4]:>9.4f}{v[5]:>8.4f}{v[6]:>10.4f}"
          f"{v[0]:>12.4f}{v[1]:>10.4f}{v[2]:>9.2f}")
print("-" * len(hdr))

print()
print("VERDICT — does each candidate put ms2_s1 (mode track visible) ABOVE lr1e4d6 (blur)?")
NAMES = {7: "ms_ssim", 3: "mode_track_f1", 4: "mode_track_f1_DETRENDED",
         5: "ridge_traj_corr", 6: "spectral_contrast_ratio",
         0: "spec_nrmse (incumbent)", 1: "hf_ratio"}
for i, nm in NAMES.items():
    m2, lr = rows["ms2_s1"][i], rows["lr1e4d6"][i]
    orc, tmn, slf = rows["oracle192"][i], rows["tmean"][i], rows["GT_self"][i]
    if i == 0:                      # lower is better
        ok = m2 < lr
    elif i in (6,):                 # ideal 1.0 -> closer to 1 is better
        ok = abs(m2 - 1.0) < abs(lr - 1.0)
    else:                           # higher is better
        ok = m2 > lr
    print(f"  {nm:<26} ms2_s1={m2:.4f}  lr1e4d6={lr:.4f}   "
          f"{'PASSES' if ok else 'REJECTED (reproduces the nrmse failure)'}"
          f"   [self={slf:.4f} oracle={orc:.4f} tmean={tmn:.4f}]")
print("METRIC VALIDATION DONE")
