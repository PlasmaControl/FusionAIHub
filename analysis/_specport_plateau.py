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

import numpy as np
import torch

sys.path.insert(0, "src")
from torch.utils.data import DataLoader, Subset

from tokamak_foundation_model.ignite import gate as gate_mod
from tokamak_foundation_model.ignite import spike
from tokamak_foundation_model.ignite import train_codec as tc
from tokamak_foundation_model.ignite.config import SpectroCodecConfig

MOD = "mhr"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 720
CACHE = "/lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/eval_runs/codec_recon_figs/_cache"


def stream(cfg, shots, n_windows, tag):
    ds = tc.CodecPairDataset(MOD, shots, cfg, data_dir=tc.DEFAULT_DATA_DIR,
                             lengths_cache_path=f"{CACHE}/codec_{MOD}_{tag}_lengths.pt")
    n = len(ds)
    idx = np.linspace(0, n - 1, min(n_windows, n)).astype(int).tolist()
    dl = DataLoader(Subset(ds, idx), batch_size=8, num_workers=8, shuffle=False)
    return np.concatenate([a.numpy() for a, _b in dl], 0)      # (N, C, F, T)


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
_pool = all_shots[:-16]
train_shots = _pool[:: max(1, len(_pool) // 40)][:40]
test_shots = all_shots[-16:][:4]
print(f"train shots {train_shots}  test shots {test_shots}   N={N} windows each", flush=True)

STATS_512 = ("/lustre/orion/fus187/proj-shared/models/ignite_codecs_noinorm/stats/"
             "codec_mhr_perfreq_stats.pt")
STATS_256 = ("/lustre/orion/fus187/proj-shared/models/ignite_codecs_mhr_specport/stats/"
             "codec_mhr_perfreq_stats_nfft512.pt")

# k = number of continuous coefficients the oracle may spend per window. 192 is the
# codec's token budget; the larger values locate where a linear code would have to sit to
# match ~tmean, which is itself a C*F = 3072-number oracle (the exact per-(channel,freq)
# time-average), i.e. 16x the codec's dimension budget before any quantisation.
KS = [16, 48, 96, 192, 384, 700]
for name, (nfft, fbins, pf, pt, stats) in (
    ("TODAY  n_fft1024 / 512 bins / patch 16x16", (1024, 512, 16, 16, STATS_512)),
    ("PORT   n_fft 512 / 256 bins / patch  8x16", (512, 256, 8, 16, STATS_256)),
):
    cfg = SpectroCodecConfig(channels=tc.modality_channels(MOD))
    cfg.stft_n_fft, cfg.stft_hop = nfft, 256
    cfg.freq_bins, cfg.time_frames = fbins, 96
    cfg.patch_f, cfg.patch_t = pf, pt
    # PER-FREQ LOG-Z, exactly as every mhr arm trains (--logpow_stats_path). Without it the
    # target is a DIFFERENT tensor and none of these numbers are comparable to the audit:
    # ~tmean reads 0.787 in raw log-power space vs 0.878 in the per-freq-z space the arms use.
    _st = torch.load(stats, map_location="cpu", weights_only=False)
    assert len(_st["mean"][0]) == fbins, f"{stats} has F={len(_st['mean'][0])}, need {fbins}"
    cfg.logpow_freq_mean = torch.as_tensor(_st["mean"], dtype=torch.float32).tolist()
    cfg.logpow_freq_std = torch.as_tensor(_st["std"], dtype=torch.float32).tolist()
    cfg.logpow_standardize = True
    t0 = time.time()
    # 720 train windows set the available PCA rank; the metric is accumulated in CHUNKS so the
    # float64 host arrays never exceed one chunk (the un-chunked version was OOM-killed).
    Xtr = stream(cfg, train_shots, N, "plateau_tr")
    Xte = stream(cfg, test_shots, N, "audit4")
    vals = int(np.prod(Xte.shape[1:]))
    bits = cfg.n_tok * np.log2(cfg.codebook_size)
    print(f"\n=== {name} ===", flush=True)
    print(f"  window {Xte.shape[1:]} = {vals} values | n_tok {cfg.n_tok} -> {bits:.0f} bits "
          f"= {bits / vals:.4f} bits/value | stream {time.time() - t0:.0f}s", flush=True)
    base = baselines_chunked(Xte)
    print(f"  ~tmean  (perfect envelope, no temporal structure): nrmse "
          f"{base['tmean'][0]:.4f}  corr2d {base['tmean'][1]:.4f}", flush=True)
    print(f"  ~wcmean (per-window constant, the 1.0 anchor)   : nrmse "
          f"{base['wcmean'][0]:.4f}  corr2d {base['wcmean'][1]:.4f}", flush=True)
    res, rank = pca_plateau(Xtr, Xte, KS)
    print(f"  out-of-sample PCA (rank available {rank}) -- k continuous dims, INFINITE "
          f"precision:", flush=True)
    for k in KS:
        mark = "   <-- n_tok (the codec's dimension budget)" if k == cfg.n_tok else ""
        print(f"    k={k:>4}   nrmse {res[k][0]:.4f}   corr2d {res[k][1]:.4f}{mark}", flush=True)
    del Xtr, Xte
print("PLATEAU DONE")
