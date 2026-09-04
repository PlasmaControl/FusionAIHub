"""What does the BEST POSSIBLE 192-dimensional code look like?

Renders, for one held-out mhr window over the full 0-250 kHz band:
    GROUND TRUTH | rank-192 out-of-sample PCA | time-mean envelope (~tmean)
on BOTH STFT grids. The PCA panel is an ORACLE: 192 continuous coefficients at infinite
precision, fitted on training shots and applied out of sample -- strictly more capacity than
the codec's 192 x ~10-bit tokens. If that panel is already a blur, no codec with this token
budget can be sharp, and the target has to change rather than the model.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, "src")
from tokamak_foundation_model.ignite import gate as gate_mod
from tokamak_foundation_model.ignite import spike
from tokamak_foundation_model.ignite import train_codec as tc
from tokamak_foundation_model.ignite.config import SpectroCodecConfig

CACHE = "/lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/eval_runs/codec_recon_figs/_cache"
OUT = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 480
K = 192
dev = "cuda" if torch.cuda.is_available() else "cpu"
allsh = spike.discover_shots(tc.DEFAULT_DATA_DIR)
# FIT POOL spread across the whole campaign range (see analysis/_specport_plateau.py):
# fitting on the first 4 shots measured cross-campaign transfer, not capacity.
_pool = allsh[:-16]
tr_shots = _pool[:: max(1, len(_pool) // 40)][:40]
te_shots = allsh[-16:][:4]


def stream(cfg, shots, tag):
    ds = tc.CodecPairDataset("mhr", shots, cfg, data_dir=tc.DEFAULT_DATA_DIR,
                             lengths_cache_path=f"{CACHE}/codec_mhr_{tag}_lengths.pt")
    idx = np.linspace(0, len(ds) - 1, min(N, len(ds))).astype(int).tolist()
    dl = DataLoader(Subset(ds, idx), batch_size=8, num_workers=8, shuffle=False)
    return np.concatenate([a.numpy() for a, _b in dl], 0)


STATS_512 = ("/lustre/orion/fus187/proj-shared/models/ignite_codecs_noinorm/stats/"
             "codec_mhr_perfreq_stats.pt")
STATS_256 = ("/lustre/orion/fus187/proj-shared/models/ignite_codecs_mhr_specport/stats/"
             "codec_mhr_perfreq_stats_nfft512.pt")
grids = [("n_fft 1024 -> 512 bins (TODAY)", 1024, 512, 16, STATS_512),
         ("n_fft 512 -> 256 bins (PORT)", 512, 256, 8, STATS_256)]
rows = []
for lab, nfft, fbins, pf, stats in grids:
    cfg = SpectroCodecConfig(channels=tc.modality_channels("mhr"))
    cfg.stft_n_fft, cfg.stft_hop = nfft, 256
    cfg.freq_bins, cfg.time_frames = fbins, 96
    cfg.patch_f, cfg.patch_t = pf, 16
    # per-freq log-z: the space every mhr arm is trained and audited in
    _st = torch.load(stats, map_location="cpu", weights_only=False)
    cfg.logpow_freq_mean = torch.as_tensor(_st["mean"], dtype=torch.float32).tolist()
    cfg.logpow_freq_std = torch.as_tensor(_st["std"], dtype=torch.float32).tolist()
    cfg.logpow_standardize = True
    Xtr, Xte = stream(cfg, tr_shots, "plateau_tr"), stream(cfg, te_shots, "audit4")
    A = torch.from_numpy(Xtr.reshape(Xtr.shape[0], -1)).to(dev, torch.float32)
    mu = A.mean(0, keepdim=True)
    A -= mu
    # eigh on the HOST: the Gram matrix is only (N, N) so this is microseconds, and the
    # ROCm eigh path kills the process outright on this machine (silent SIGKILL, no
    # traceback). Every LARGE operation stays on the GPU.
    G = (A @ A.T).double().cpu()
    ev, V = torch.linalg.eigh(G)
    ev, V = ev.to(A.device), V.to(A.device)
    ev, V = ev.flip(0)[:K], V.flip(1)[:, :K]
    W = (A.T @ V.float()) / ev.float().sqrt()[None, :]
    del A
    B = torch.from_numpy(Xte.reshape(Xte.shape[0], -1)).to(dev, torch.float32) - mu
    R = ((B @ W) @ W.T + mu).reshape(Xte.shape).cpu().numpy()
    del B, W, mu
    torch.cuda.empty_cache() if dev == "cuda" else None
    tmean = np.broadcast_to(Xte.mean(axis=-1, keepdims=True), Xte.shape)
    m_pca = gate_mod.full_spectro_metrics(R, Xte, band_bins=None)
    m_tm = gate_mod.full_spectro_metrics(tmean, Xte, band_bins=None)
    hf_pca = gate_mod.decode_fidelity(R, Xte)["sharpness"]
    hf_tm = gate_mod.decode_fidelity(np.ascontiguousarray(tmean), Xte)["sharpness"]
    rows.append((lab, fbins, Xte, R, np.array(tmean), m_pca, m_tm, hf_pca, hf_tm))
    print(f"{lab}: PCA-{K} nrmse {m_pca['spec_nrmse']:.4f} corr2d "
          f"{m_pca['spec_corr2d']:.4f} hf {hf_pca:.4f} | tmean nrmse "
          f"{m_tm['spec_nrmse']:.4f} corr2d {m_tm['spec_corr2d']:.4f} hf {hf_tm:.4f}", flush=True)

env = rows[0][2].std(axis=3).std(axis=2)
w, ch = np.unravel_index(int(np.argmax(env)), env.shape)
w, ch = int(w), int(ch)
print(f"most-structured window w={w} ch={ch}")

fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
for i, (lab, fbins, Xte, R, TM, m_pca, m_tm, hf_pca, hf_tm) in enumerate(rows):
    khz = 250.0 / fbins
    gt = Xte[w, ch]
    vmin, vmax = np.percentile(gt, [2, 98])
    panels = [
        (f"GROUND TRUTH\n{lab}", gt, ""),
        (f"ORACLE rank-{K} PCA (192 FLOATS, out of sample)",
         R[w, ch], f"nrmse {m_pca['spec_nrmse']:.3f}  corr2d {m_pca['spec_corr2d']:.3f}  "
                   f"hf {hf_pca:.3f}"),
        ("~tmean  (perfect envelope, zero temporal structure)",
         TM[w, ch], f"nrmse {m_tm['spec_nrmse']:.3f}  corr2d {m_tm['spec_corr2d']:.3f}  "
                    f"hf {hf_tm:.3f}"),
    ]
    for j, (t, a, sub) in enumerate(panels):
        axes[i][j].imshow(a, aspect="auto", origin="lower", cmap="magma", vmin=vmin, vmax=vmax,
                          interpolation="nearest", extent=[0, a.shape[1], 0, a.shape[0] * khz])
        axes[i][j].set_title(f"{t}\n{sub}", fontsize=9)
        axes[i][j].set_ylabel("frequency (kHz)")
        axes[i][j].set_xlabel("STFT time frame")
fig.suptitle(f"CAPACITY CEILING at 192 tokens -- mhr held-out window (w={w}, ch={ch}), full "
             f"0-250 kHz. Middle column has MORE capacity than any 192-token codec.", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT, dpi=130)
print(f"WROTE {OUT}")
print("CEILING FIG DONE")
