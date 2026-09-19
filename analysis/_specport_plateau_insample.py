"""Is the briefed "linear plateau 0.7487" an IN-SAMPLE number?

The brief anchors on an "out-of-sample linear plateau" of 0.7487, but the out-of-sample
rank-k PCA measured in analysis/_specport_plateau.py bottoms out at 0.8193 by k=700 on the
same 720 held-out windows and the same per-freq-z target the arms are audited in. This
script prints, side by side on ONE window pool:

    OUT-OF-SAMPLE  PCA basis fitted on TRAINING shots, applied to the held-out windows
    IN-SAMPLE      PCA basis fitted on the held-out windows themselves

so the discrepancy is attributable rather than argued about. An in-sample rank-k fit is not
a codec bound -- it has already seen the windows it is asked to reconstruct.
"""
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, "src")
from tokamak_foundation_model.ignite import gate as gate_mod
from tokamak_foundation_model.ignite import spike
from tokamak_foundation_model.ignite import train_codec as tc
from tokamak_foundation_model.ignite.config import SpectroCodecConfig

CACHE = "/lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/eval_runs/codec_recon_figs/_cache"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 720
dev = "cuda" if torch.cuda.is_available() else "cpu"
STATS = {512: ("/lustre/orion/fus187/proj-shared/models/ignite_codecs_noinorm/stats/"
               "codec_mhr_perfreq_stats.pt"),
         256: ("/lustre/orion/fus187/proj-shared/models/ignite_codecs_mhr_specport/stats/"
               "codec_mhr_perfreq_stats_nfft512.pt")}
allsh = spike.discover_shots(tc.DEFAULT_DATA_DIR)
_pool = allsh[:-16]
tr_shots = _pool[:: max(1, len(_pool) // 40)][:40]
te_shots = allsh[-16:][:4]
KS = [96, 192, 384, 700]


def make_cfg(nfft, fbins, pf):
    cfg = SpectroCodecConfig(channels=tc.modality_channels("mhr"))
    cfg.stft_n_fft, cfg.stft_hop = nfft, 256
    cfg.freq_bins, cfg.time_frames = fbins, 96
    cfg.patch_f, cfg.patch_t = pf, 16
    st = torch.load(STATS[fbins], map_location="cpu", weights_only=False)
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


def basis(X, kmax):
    A = torch.from_numpy(X.reshape(X.shape[0], -1)).to(dev, torch.float32)
    mu = A.mean(0, keepdim=True)
    A = A - mu
    ev, V = torch.linalg.eigh((A @ A.T).double().cpu())   # host eigh (ROCm path is unstable)
    ev, V = ev.flip(0).to(dev), V.flip(1).to(dev)
    keep = ev > ev.max() * 1e-12
    ev, V = ev[keep][:kmax], V[:, keep][:, :kmax]
    W = (A.T @ V.float()) / ev.float().sqrt()[None, :]
    del A, V
    return W, mu


def score(W, mu, Xte, k, chunk=24):
    acc = []
    for i in range(0, Xte.shape[0], chunk):
        t = Xte[i:i + chunk]
        b = torch.from_numpy(t.reshape(t.shape[0], -1)).to(dev, torch.float32) - mu
        r = ((b @ W[:, :k]) @ W[:, :k].T + mu).reshape(t.shape).cpu().numpy()
        m = gate_mod.full_spectro_metrics(r, t, band_bins=None)
        acc.append((m["spec_nrmse"], m["spec_corr2d"]))
        del b, r
    a = np.array(acc).mean(0)
    return float(a[0]), float(a[1])


for nfft, fbins, pf in ((1024, 512, 16), (512, 256, 8)):
    cfg = make_cfg(nfft, fbins, pf)
    t0 = time.time()
    Xte = stream(cfg, te_shots, N, "audit4")
    Xtr = stream(cfg, tr_shots, N, "readout_tr")
    print(f"\n=== n_fft {nfft} / {fbins} bins / patch {pf}x16 -- {Xte.shape[0]} held-out "
          f"windows ({time.time()-t0:.0f}s) ===", flush=True)
    base = None
    for i in range(0, Xte.shape[0], 24):
        t = Xte[i:i + 24]
        m = gate_mod.full_spectro_metrics(
            np.broadcast_to(t.mean(axis=-1, keepdims=True), t.shape), t, band_bins=None)
        base = (m["spec_nrmse"], m["spec_corr2d"]) if base is None else (
            base[0] + m["spec_nrmse"], base[1] + m["spec_corr2d"])
    nb = len(range(0, Xte.shape[0], 24))
    print(f"  ~tmean                                nrmse {base[0]/nb:.4f}  "
          f"corr2d {base[1]/nb:.4f}", flush=True)
    Wo, muo = basis(Xtr, max(KS))
    Wi, mui = basis(Xte, max(KS))
    print(f"  {'k':>5}  {'OUT-OF-SAMPLE (held-out basis unseen)':<40}"
          f"{'IN-SAMPLE (basis fitted ON these windows)'}", flush=True)
    for k in KS:
        if k > Wo.shape[1] or k > Wi.shape[1]:
            continue
        o = score(Wo, muo, Xte, k)
        s = score(Wi, mui, Xte, k)
        print(f"  {k:>5}  nrmse {o[0]:.4f}  corr2d {o[1]:.4f}            "
              f"nrmse {s[0]:.4f}  corr2d {s[1]:.4f}", flush=True)
    del Wo, Wi, muo, mui, Xtr, Xte
    if dev == "cuda":
        torch.cuda.empty_cache()
print("INSAMPLE COMPARISON DONE")
