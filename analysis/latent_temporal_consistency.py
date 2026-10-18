#!/usr/bin/env python
"""WHERE does temporal consistency die: encoder, or quantiser?

The token flip rate (0.17-0.905 per frame across the 15 v4 modalities) counts DISCRETE code
changes. An FSQ code flips whenever the continuous latent crosses a quantisation boundary, which
can happen when the latent barely moves. So a high flip rate has two very different causes:
  (a) the ENCODER's representation is realisation-dominated  -> only a codec change helps
  (b) the representation is stable and FSQ SHATTERS it       -> predict the continuous latent
                                                                (option b of the v4 diagnosis),
                                                                reusing every existing codec
Memory already says raw spectrograms carry a slow trend at ac1 0.84-0.99 that the codes carry
<2% of. This measures the chain frame-to-frame, per modality, on the SHIPPED v4 codecs:
    raw input x  ->  encoder latent (codec.encode)  ->  quantised latent (fsq zq)  ->  code idx
reporting lag-1 Pearson correlation for the three continuous stages and flip rate for the code.
No codec is trained or modified.

    python analysis/latent_temporal_consistency.py --n_shots 12
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "src")
from tokamak_foundation_model.ignite import train_dynamics as td, spike
from tokamak_foundation_model.ignite.dynamics_config import FROZEN_MODALITIES

MAN = "/lustre/orion/fus187/proj-shared/models/ignite_prod_v4/frame_codes/_codec_manifest.json"


def lag1(v):
    """mean Pearson r between consecutive rows of v (T, D)."""
    if len(v) < 3: return float("nan")
    a, b = v[:-1], v[1:]
    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
    num = (a * b).sum(1); den = np.sqrt((a * a).sum(1) * (b * b).sum(1)) + 1e-12
    return float(np.nanmean(num / den))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_shots", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--only", default="", help="comma list of modalities (default all 15)")
    p.add_argument("--out", default="eval_runs/latent_temporal_consistency.json")
    a = p.parse_args()
    man = json.load(open(MAN)); t0 = float(man["t0_start"])
    fam = {m.name: m.family for m in FROZEN_MODALITIES}
    mods = [m for m in a.only.split(",") if m] or sorted(man["resolved"])
    shots = sorted(spike.discover_shots(spike.DEFAULT_DATA_DIR))
    rng = np.random.default_rng(a.seed)
    pool = [shots[i] for i in rng.permutation(len(shots))[-200:]]      # held-out-ish tail
    res = {}
    print(f"  {'modality':24s} {'n':>3s} {'raw ac1':>8s} {'enc ac1':>8s} {'zq ac1':>7s} "
          f"{'flip':>6s}  {'enc->zq loss':>12s}", flush=True)
    for m in mods:
        t_m = time.time()
        codec, cfg = td._load_codec(fam[m], Path(man["resolved"][m]))
        codec.eval()
        R, E, Q, F, used = [], [], [], [], 0
        for s in pool:
            if used >= a.n_shots: break
            try:
                ds = td._single_shot_dataset(m, fam[m], cfg, s, spike.DEFAULT_DATA_DIR, t0_start=t0)
            except Exception:
                continue
            if len(ds) < 20: continue
            xs, es, qs, ids = [], [], [], []
            with torch.no_grad():
                for t in range(len(ds)):
                    it = ds[t]; x = (it[0] if isinstance(it, tuple) else it).unsqueeze(0)
                    f = codec.encode(x); zq, idx = codec.quantizer.fsq(f)
                    xs.append(x.flatten().float().numpy()); es.append(f.flatten().float().numpy())
                    qs.append(zq.flatten().float().numpy()); ids.append(idx.flatten().numpy())
            ids = np.stack(ids)
            if (ids == ids[0]).all():          # ABSENT diagnostic -> constant null code. Skip:
                continue                       # it would read as perfectly "stable".
            R.append(lag1(np.stack(xs))); E.append(lag1(np.stack(es))); Q.append(lag1(np.stack(qs)))
            F.append(float((ids[1:] != ids[:-1]).mean())); used += 1
        if not used:
            print(f"  {m:24s}   0  (no live shots in the pool)", flush=True); continue
        r, e, q, f = map(lambda v: float(np.nanmean(v)), (R, E, Q, F))
        res[m] = {"n_shots": used, "raw_ac1": r, "enc_ac1": e, "zq_ac1": q, "flip_rate": f,
                  "family": fam[m]}
        print(f"  {m:24s} {used:3d} {r:8.3f} {e:8.3f} {q:7.3f} {f:6.3f}  {e-q:12.3f}"
              f"   ({time.time()-t_m:.0f}s)", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\n  -> {a.out}")
    print("  READ: enc ac1 HIGH but flip HIGH  => quantiser shatters a stable latent (fix: predict the latent)")
    print("        enc ac1 LOW                  => the encoder itself is realisation-dominated (codec axis)")


if __name__ == "__main__":
    main()
