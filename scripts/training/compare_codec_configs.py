"""Compare FSQ codec configs per modality on HELD-OUT mode-shots and pick the best.

For each spectro modality and each config (dir suffix), loads the frozen codec,
reconstructs held-out mode-shots (ranked just OUTSIDE the codec's top-500 training
set — a true generalization test), and reports max-mode-channel reconstruction
correlation. Renders a per-modality panel (GT + each config's recon, GT-normed
0-60 kHz contrast) and prints a winner table.

Env:
  MODALITIES  (default "ece co2 bes mhr")
  CONFIGS     (default "top500:cap48:cap64:cap96:cap64hifi"; ':'-sep dir suffixes;
               "top500" is the baseline 24/8 codec dir fsq_codec_<mod>_top500)
  HELDOUT_RANKS  (default "500:508"  -> rank_<mod>.txt indices [500,508))
  OUT_DIR     (default eval_runs/codec_compare)
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import poc_fsq_stageB as poc
from tokamak_foundation_model.e2e.quantizers import load_frozen_codec

DATA = os.environ.get("EVAL_DATA_DIR", "/lustre/orion/fus187/proj-shared/foundation_model")
STATS = os.environ.get("EVAL_STATS", "/lustre/orion/fus187/proj-shared/foundation_model_meta/preprocessing_stats.pt")
SCAN = "eval_runs/spectro_mode_scan"
LOWF = 123


def _corr(a, b):
    a = a.ravel() - a.mean(); b = b.ravel() - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else 0.0


def _rank_shots(m):
    out = []
    for ln in open(f"{SCAN}/rank_{m}.txt"):
        if ln.startswith("#"):
            continue
        out.append(int(ln.split()[0]))
    return out


def main():
    poc.PATCH_F, poc.PATCH_T = 64, 32
    mods = os.environ.get("MODALITIES", "ece co2 bes mhr").split()
    configs = os.environ.get("CONFIGS", "top500:cap48:cap64:cap96:cap64hifi").split(":")
    r0, r1 = (int(x) for x in os.environ.get("HELDOUT_RANKS", "500:508").split(":"))
    out_dir = Path(os.environ.get("OUT_DIR", "eval_runs/codec_compare"))
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for mod in mods:
        k = poc._SPEC_STRUCT_K.get(mod, 2.0)
        hold = _rank_shots(mod)[r0:r1]
        # load held-out windows once; pick the strongest-mode (shot, channel)
        best = (-1, None, None, None)
        for sh in hold:
            try:
                _, X = poc.load_pairs(str(sh), DATA, STATS, 64, 60, 0.4, mod)
            except Exception:
                continue
            if X.shape[0] == 0:
                continue
            act = poc._hard(X, k)[:, :, :LOWF, :].sum(axis=(0, 2, 3))
            ch = int(act.argmax())
            if float(act[ch]) > best[0]:
                best = (float(act[ch]), sh, ch, X)
        _, sh, ch, X = best
        if X is None:
            print(f"[{mod}] no held-out mode data found", flush=True)
            continue
        Xn = X.numpy()
        recons = {}
        corrs = {}
        for cfg in configs:
            path = f"eval_runs/fsq_codec_{mod}_{cfg}/spectro_codec_{mod}.pt"
            if not os.path.exists(path):
                corrs[cfg] = float("nan"); continue
            codec, meta = load_frozen_codec(path)
            codec.eval()
            with torch.no_grad():
                rec = torch.cat([codec(X[i:i + 64])[0] for i in range(0, X.shape[0], 64)], 0)
            R = rec.numpy()
            recons[cfg] = R
            corrs[cfg] = _corr(Xn[:, ch], R[:, ch])
        winner = max((c for c in corrs if corrs[c] == corrs[c]), key=lambda c: corrs[c], default=None)
        summary[mod] = (sh, ch, corrs, winner)
        line = "  ".join(f"{c}={corrs[c]:.3f}" for c in configs if corrs[c] == corrs[c])
        print(f"[{mod}] shot {sh} ch{ch}: {line}  -> WINNER {winner} ({corrs.get(winner,0):.3f})", flush=True)

        # panel: GT + each config recon (GT-normed contrast, 0-60kHz)
        def stitch(A, cc, nw=30):
            s = max(1, A.shape[0] // nw); a = A[::s, cc]; n, F, T = a.shape
            return a.transpose(1, 0, 2).reshape(F, n * T)
        g = stitch(Xn, ch); m_ = g.mean(1, keepdims=True); sd = g.std(1, keepdims=True) + 1e-6
        gz = np.clip((g - m_) / sd, 0, 4)[:LOWF]
        panels = [("GT held-out", gz)] + [
            (f"{c} (corr {corrs[c]:.2f})", np.clip((stitch(recons[c], ch) - m_) / sd, 0, 4)[:LOWF])
            for c in configs if c in recons]
        fig, ax = plt.subplots(len(panels), 1, figsize=(13, 2.1 * len(panels)), sharex=True)
        for a_, (t, d) in zip(ax, panels):
            im = a_.imshow(d, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=4)
            a_.set_title(t, fontsize=10); a_.set_ylabel("freq")
        fig.suptitle(f"{mod.upper()} codec config compare — held-out shot {sh} ch{ch}", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        p = out_dir / f"compare_{mod}.png"
        fig.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig)
        print(f"[{mod}] panel -> {p}", flush=True)

    print("\n===== WINNER SUMMARY =====", flush=True)
    for mod, (sh, ch, corrs, winner) in summary.items():
        print(f"{mod:4s}: WINNER={winner:10s} corr={corrs.get(winner,float('nan')):.3f}  "
              f"(all: {', '.join(f'{c} {corrs[c]:.3f}' for c in configs if corrs[c]==corrs[c])})",
              flush=True)


if __name__ == "__main__":
    main()
