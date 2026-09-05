"""OUT-OF-SAMPLE LINEAR PLATEAU vs STFT geometry.

The codec spends n_tok x log2(codebook) bits on a (C, F, T) window. This measures the
strongest LINEAR code with the same *dimension* budget and INFINITE precision: fit PCA on
training windows, keep k components, reconstruct HELD-OUT windows. It is an upper bound on
what any n_tok-dimensional linear code can reach, and therefore a reference for how much of
the codec's residual error is a capacity/bit-budget bound rather than a modelling failure.

Run for the 1024/512 grid (today) and the 512/256 grid (the Spectral-Codec port).
Linear algebra runs on the GPU; the METRIC is gate.full_spectro_metrics on host numpy, i.e.
exactly the number the audit prints.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")
from torch.utils.data import DataLoader, Subset

from tokamak_foundation_model.ignite import gate as gate_mod
from tokamak_foundation_model.ignite import spike
from tokamak_foundation_model.ignite import train_codec as tc
from tokamak_foundation_model.ignite.config import SpectroCodecConfig

# 2026-09-03: generalised from mhr-only to ANY spectro modality (`--modality`) so each one is
# judged against ITS OWN floor -- the floors are not remotely alike (co2 0.6303, mhr 0.7487,
# bes 0.7592, ece 0.9915), and they track values-per-token, so a shared reference would be
# meaningless. The two-geometry sweep is gone: the n_fft 512 port is DEAD on mhr (worse on both
# axes), so only the production 1024/256 grid is measured unless --stft_n_fft says otherwise.
import argparse

_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--modality", default="mhr", choices=list(tc.SPECTRO_MODALITIES))
_ap.add_argument("--n", type=int, default=720, help="windows streamed per split")
_ap.add_argument("--stats", default=None,
                 help="per-freq log-z stats (--logpow_stats_path). Default: the shared "
                      "ignite_codecs_noinorm/stats file for this modality if it exists; "
                      "'none' disables the per-freq z (must match how the arms train).")
_ap.add_argument("--stft_n_fft", type=int, default=1024)
_ap.add_argument("--freq_bins", type=int, default=512)
_ap.add_argument("--patch_f", type=int, default=16)
_ap.add_argument("--patch_t", type=int, default=16)
_ap.add_argument("--ks", default="16,48,96,192,384,700")
_ap.add_argument("--band_pool", type=int, default=0,
                 help="BAND-POWER representation: mean-pool the freq_bins into this many equal "
                      "bands before measuring the floor (0 = raw bins, the default). The 1.0 "
                      "anchor and ~tmean are recomputed ON THE POOLED ARRAY, so 'does the floor "
                      "drop below 1.0' stays well-posed -- it asks whether THIS representation "
                      "is linearly predictable at rank n_tok, which is the question a codec "
                      "inherits. Motivated for ece: its band profile is FLAT (std 0.887-0.943, "
                      "ac1 0.288-0.371 across all 16 bands), so 512 bins carry little "
                      "differentiated structure while consuming the whole token budget.")
_args = _ap.parse_args()

MOD = _args.modality
N = int(_args.n)
CACHE = "/lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/eval_runs/codec_recon_figs/_cache"


def stream(cfg, shots, n_windows, tag):
    ds = tc.CodecPairDataset(MOD, shots, cfg, data_dir=tc.DEFAULT_DATA_DIR,
                             lengths_cache_path=f"{CACHE}/codec_{MOD}_{tag}_lengths.pt")
    n = len(ds)
    idx = np.linspace(0, n - 1, min(n_windows, n)).astype(int).tolist()
    dl = DataLoader(Subset(ds, idx), batch_size=8, num_workers=8, shuffle=False)
    return np.concatenate([a.numpy() for a, _b in dl], 0)      # (N, C, F, T)


def band_pool(X, k):
    """(N, C, F, T) -> (N, C, k, T) by mean-pooling F into k equal bands. k<=0 is a no-op."""
    if k <= 0:
        return X
    N, C, F, T = X.shape
    if F % k:
        raise SystemExit(f"--band_pool {k} must divide freq_bins {F}")
    return X.reshape(N, C, k, F // k, T).mean(axis=3)


def _chunked_metrics(pred_fn, Xte, chunk=24):
    """Mean of gate.full_spectro_metrics over CHUNKS of windows.

    Identical convention to the audit (which accumulates a per-batch mean), and it keeps peak
    host memory at one chunk: the float64 metric on all 240 windows at once allocates ~10 GB of
    temporaries and got the process killed on the login node.
    """
    acc = []
    for i in range(0, Xte.shape[0], chunk):
        t = Xte[i:i + chunk]
        r = pred_fn(i, i + chunk)
        m = gate_mod.full_spectro_metrics(r, t, band_bins=None)
        acc.append((m["spec_nrmse"], m["spec_corr2d"]))
    a = np.array(acc).mean(0)
    return float(a[0]), float(a[1])


def baselines_chunked(Xte, chunk=24):
    out = {}
    for name in ("tmean", "wcmean"):
        def f(i, j, name=name):
            t = Xte[i:j]
            if name == "tmean":
                return np.broadcast_to(t.mean(axis=-1, keepdims=True), t.shape)
            return np.broadcast_to(t.mean(axis=(-2, -1), keepdims=True), t.shape)
        out[name] = _chunked_metrics(f, Xte, chunk)
    return out


def pca_plateau(Xtr, Xte, ks, chunk=24):
    """Fit PCA on Xtr, reconstruct Xte at each k; return ({k: (nrmse, corr2d)}, rank)."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    A = torch.from_numpy(Xtr.reshape(Xtr.shape[0], -1)).to(dev, torch.float32)
    mu = A.mean(0, keepdim=True)
    A -= mu
    # eigh on the HOST: the Gram matrix is only (N, N) so this is microseconds, and the
    # ROCm eigh path kills the process outright on this machine (silent SIGKILL, no
    # traceback). Every LARGE operation stays on the GPU.
    G = (A @ A.T).double().cpu()
    ev, V = torch.linalg.eigh(G)
    ev, V = ev.to(A.device), V.to(A.device)
    ev, V = ev.flip(0), V.flip(1)
    keep = ev > ev.max() * 1e-12
    ev, V = ev[keep], V[:, keep]
    W = (A.T @ V.float()) / ev.float().sqrt()[None, :]         # (D, r) orthonormal columns
    del A
    B = torch.from_numpy(Xte.reshape(Xte.shape[0], -1)).to(dev, torch.float32) - mu
    Z = B @ W
    del B
    res = {}
    for k in ks:
        kk = min(k, W.shape[1])

        def f(i, j, kk=kk):
            return ((Z[i:j, :kk] @ W[:, :kk].T + mu)
                    .reshape(-1, *Xte.shape[1:]).cpu().numpy())

        res[k] = _chunked_metrics(f, Xte, chunk)
        print(f"      k={k} done ({time.strftime('%H:%M:%S')})", flush=True)
    rank = W.shape[1]
    del W, Z, mu
    if dev == "cuda":
        torch.cuda.empty_cache()
    return res, rank


all_shots = spike.discover_shots(tc.DEFAULT_DATA_DIR)
# FIT POOL: shots spread across the WHOLE campaign range, not the first 4. Using
# all_shots[:4] made this a CROSS-CAMPAIGN transfer test rather than a capacity test --
# a basis fitted on 4 early shots does not transfer to the held-out late shots (the
# known cross-campaign failure), and the oracle then scored WORSE than the 1.0 anchor.
#
# PRESENCE FILTER (2026-09-03): shots whose HDF5 group for this modality is an empty (C, 1)
# stub yield ONLY the eps-floor constant last-resort window, which would put a constant plate
# in both the PCA fit and the held-out set and make the "floor" meaningless. bes is empty in
# 61.8% of shots and co2 in 48.2%, so this is not a corner case. Uses the same precomputed
# liveness cache the trainer's --spectro_presence reads; passthrough (with a warning) if it
# has not been built yet.
all_shots = tc.spectro_live_shots(MOD, all_shots, log_fn=print)
_pool = all_shots[:-16]
train_shots = _pool[:: max(1, len(_pool) // 40)][:40]
test_shots = all_shots[-16:][:4]
print(f"[{MOD}] train shots {train_shots}  test shots {test_shots}   N={N} windows each",
      flush=True)

_DEFAULT_STATS = ("/lustre/orion/fus187/proj-shared/models/ignite_codecs_noinorm/stats/"
                  f"codec_{MOD}_perfreq_stats.pt")
stats = _args.stats or (_DEFAULT_STATS if Path(_DEFAULT_STATS).exists() else "none")

KS = [int(k) for k in _args.ks.split(",") if k.strip()]
cfg = SpectroCodecConfig(channels=tc.modality_channels(MOD))
cfg.stft_n_fft, cfg.stft_hop = _args.stft_n_fft, 256
cfg.freq_bins, cfg.time_frames = _args.freq_bins, 96
cfg.patch_f, cfg.patch_t = _args.patch_f, _args.patch_t
# RAW per-channel standardization, exactly as the trainer applies it (co2 only today). Without
# it co2's log-power is 100% clipped at the ceiling and the floor is measured on a flat plate.
tc.apply_spectro_standardization(cfg, MOD, log_fn=print)
# PER-FREQ LOG-Z, exactly as the arms train (--logpow_stats_path). Without it the target is a
# DIFFERENT tensor and none of these numbers are comparable to the audit: ~tmean reads 0.787 in
# raw log-power space vs 0.878 in the per-freq-z space the mhr arms use.
if stats != "none":
    _st = torch.load(stats, map_location="cpu", weights_only=False)
    assert len(_st["mean"][0]) == cfg.freq_bins, \
        f"{stats} has F={len(_st['mean'][0])}, need {cfg.freq_bins}"
    cfg.logpow_freq_mean = torch.as_tensor(_st["mean"], dtype=torch.float32).tolist()
    cfg.logpow_freq_std = torch.as_tensor(_st["std"], dtype=torch.float32).tolist()
    cfg.logpow_standardize = True
    print(f"[{MOD}] per-freq log-z ON from {stats}", flush=True)
else:
    print(f"[{MOD}] per-freq log-z OFF (no stats file)", flush=True)

t0 = time.time()
Xtr = stream(cfg, train_shots, N, "plateau_tr")
Xtr = band_pool(Xtr, _args.band_pool)
Xte = stream(cfg, test_shots, N, "audit4")
Xte = band_pool(Xte, _args.band_pool)
vals = int(np.prod(Xte.shape[1:]))
bits = cfg.n_tok * np.log2(cfg.codebook_size)
print(f"\n=== {MOD}  n_fft {cfg.stft_n_fft} / {cfg.freq_bins} bins / patch "
      f"{cfg.patch_f}x{cfg.patch_t} ===", flush=True)
print(f"  window {Xte.shape[1:]} = {vals} values | n_tok {cfg.n_tok} -> {bits:.0f} bits "
      f"= {bits / vals:.4f} bits/value | {vals // cfg.n_tok} values/token | "
      f"stream {time.time() - t0:.0f}s", flush=True)
base = baselines_chunked(Xte)
print(f"  ~tmean  (perfect envelope, no temporal structure): nrmse "
      f"{base['tmean'][0]:.4f}  corr2d {base['tmean'][1]:.4f}", flush=True)
print(f"  ~wcmean (per-window constant, the 1.0 anchor)   : nrmse "
      f"{base['wcmean'][0]:.4f}  corr2d {base['wcmean'][1]:.4f}", flush=True)
res, rank = pca_plateau(Xtr, Xte, KS)
print(f"  out-of-sample PCA (rank available {rank}) -- k continuous dims, INFINITE precision:",
      flush=True)
for k in KS:
    mark = "   <-- n_tok = THE FLOOR" if k == cfg.n_tok else ""
    print(f"    k={k:>4}   nrmse {res[k][0]:.4f}   corr2d {res[k][1]:.4f}{mark}", flush=True)
print(f"PLATEAU DONE {MOD} floor(k={cfg.n_tok})="
      f"{res.get(cfg.n_tok, (float('nan'),))[0]:.4f}", flush=True)
