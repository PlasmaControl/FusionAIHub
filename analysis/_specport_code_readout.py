"""Is the ENCODER or the DECODER the limiter? Best LINEAR readout of the existing codes.

For a trained codec, take its FROZEN quantized latent on held-out windows and fit the
BEST POSSIBLE linear map from that latent to the target spectrogram (ridge regression in
closed form, fitted on training shots, evaluated out of sample). Compare against:

  * the codec's OWN decoder on the same windows  -- what the trained decoder achieves,
  * ~tmean                                       -- the envelope bar,
  * the rank-192 PCA oracle (analysis/_specport_plateau.py) -- the capacity ceiling.

Reading:
  linear-readout ~= codec decoder      -> the codes are the limit; a bigger decoder cannot help.
  linear-readout << codec decoder      -> the codes ALREADY carry more than the decoder emits,
                                          i.e. the DECODER is the limiter and the paper's
                                          55M-vs-10M decoder-heavy design is the right fix.

Ridge is solved in SAMPLE space (Gram trick): the latent is 192*d_model ~ 49k dims but there
are only ~N training windows, so the (N, N) system is the whole problem.

Usage: _specport_code_readout.py <n_train> <n_test> <label=ckpt> [label=ckpt ...]
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

CACHE = "/lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/eval_runs/codec_recon_figs/_cache"
NTR, NTE = int(sys.argv[1]), int(sys.argv[2])
SPECS = sys.argv[3:]
dev = "cuda" if torch.cuda.is_available() else "cpu"
allsh = spike.discover_shots(tc.DEFAULT_DATA_DIR)
_pool = allsh[:-16]
tr_shots = _pool[:: max(1, len(_pool) // 20)][:20]
te_shots = allsh[-16:][:4]
LAMBDAS = [1e-2, 1e0, 1e2, 1e4]


def encode_and_target(codec, cfg, shots, n, tag):
    """-> (Z (N, 192*d_model) float32 latent, X (N, C, F, T) float32 target)."""
    ds = tc.CodecPairDataset("mhr", shots, cfg, data_dir=tc.DEFAULT_DATA_DIR,
                             lengths_cache_path=f"{CACHE}/codec_mhr_{tag}_lengths.pt")
    idx = np.linspace(0, len(ds) - 1, min(n, len(ds))).astype(int).tolist()
    dl = DataLoader(Subset(ds, idx), batch_size=8, num_workers=6, shuffle=False)
    zs, xs = [], []
    with torch.no_grad():
        for a, _b in dl:
            a = a.to(dev)
            q, _c = codec.quantize(codec.encode(a))
            zs.append(q.reshape(q.shape[0], -1).cpu())
            xs.append(a.cpu())
    return torch.cat(zs).numpy(), torch.cat(xs).numpy()


def chunked(pred_fn, X, chunk=24):
    acc = []
    for i in range(0, X.shape[0], chunk):
        t = X[i:i + chunk]
        m = gate_mod.full_spectro_metrics(pred_fn(i, i + chunk), t, band_bins=None)
        d = gate_mod.decode_fidelity(pred_fn(i, i + chunk), t)
        acc.append((m["spec_nrmse"], m["spec_corr2d"], d["sharpness"]))
    return np.array(acc).mean(0)


print(f"train shots {len(tr_shots)} / test shots {te_shots}   N={NTR}/{NTE}", flush=True)
print(f"\n{'arm':<12}{'predictor':<26}{'spec_nrmse':>11}{'corr2d':>9}{'hf_ratio':>10}")
print("-" * 68)
for spec in SPECS:
    label, _, path = spec.partition("=")
    codec, cfg, _ck = load_spectro_codec(path)
    codec = codec.to(dev).eval()
    Ztr, Xtr = encode_and_target(codec, cfg, tr_shots, NTR, "readout_tr")
    Zte, Xte = encode_and_target(codec, cfg, te_shots, NTE, "audit4")

    # the codec's own decoder, out of sample
    with torch.no_grad():
        rec = np.concatenate([
            codec.decode(torch.from_numpy(Zte[i:i + 24]).to(dev)
                         .reshape(-1, cfg.n_tok, cfg.d_model)).cpu().numpy()
            for i in range(0, Zte.shape[0], 24)])
    a = chunked(lambda i, j: rec[i:j], Xte)
    print(f"{label:<12}{'own decoder':<26}{a[0]:>11.4f}{a[1]:>9.4f}{a[2]:>10.4f}", flush=True)

    # BEST LINEAR readout of the same latent (ridge, closed form in sample space)
    A = torch.from_numpy(Ztr).to(dev, torch.float32)
    zmu = A.mean(0, keepdim=True)
    A = A - zmu
    Y = torch.from_numpy(Xtr.reshape(Xtr.shape[0], -1)).to(dev, torch.float32)
    ymu = Y.mean(0, keepdim=True)
    Y = Y - ymu
    G = (A @ A.T).double().cpu()                      # host eigh: the ROCm path is unstable
    B = torch.from_numpy(Zte).to(dev, torch.float32) - zmu
    KT = (B @ A.T).double().cpu()                     # (nte, ntr)
    ev, V = torch.linalg.eigh(G)
    for lam in LAMBDAS:
        # (G + lam I)^-1 via the eigendecomposition, then predictions = KT @ alpha @ Y
        alpha = (V @ torch.diag(1.0 / (ev + lam)) @ V.T)
        W = (KT @ alpha).float().to(dev)              # (nte, ntr)
        pred = (W @ Y + ymu).reshape(Xte.shape).cpu().numpy()
        a = chunked(lambda i, j: pred[i:j], Xte)
        print(f"{label:<12}{'best linear readout l=' + f'{lam:g}':<26}"
              f"{a[0]:>11.4f}{a[1]:>9.4f}{a[2]:>10.4f}", flush=True)
        del W, pred
    del A, Y, B, G, KT
    if dev == "cuda":
        torch.cuda.empty_cache()
    print("-" * 68, flush=True)
print("CODE READOUT DONE")
