#!/usr/bin/env python
"""gate.forecastability on the SHIPPED codes, straight from the v4 cache.

The predictability objective (gate §4.4): per-position first-order Markov probe vs a
PERSISTENCE baseline, margin stratified into transition / stable steps. Reads the cache
directly, so it measures the tokens the world model will actually consume -- no codec
re-run, no dataset redraw (the bug that contaminated forecast_margin_tr for cer/mse).
"""
import glob, json, sys
import numpy as np, torch
sys.path.insert(0, "src")
from tokamak_foundation_model.ignite import gate

D = "/lustre/orion/fus187/proj-shared/models/ignite_prod_v4/frame_codes"
N_SHOTS, SEQ = 40, 40
fs = sorted(glob.glob(f"{D}/*.pt"))
rng = np.random.default_rng(42)
use = [fs[i] for i in rng.permutation(len(fs))[-N_SHOTS:]]      # held-out tail
mods = list(torch.load(use[0], map_location="cpu", weights_only=False)["codes"].keys())

print(f"{'modality':24s} {'flip_rate':>10} {'m_trans':>10} {'m_over':>10} {'n_trans':>9} {'codes_seen':>11}")
rows = {}
for m in sorted(mods):
    seqs, flips, ntr, seen = [], 0, 0, set()
    for f in use:
        c = torch.load(f, map_location="cpu", weights_only=False)["codes"].get(m)
        if c is None:
            continue
        c = c.numpy()
        flips += int((c[1:] != c[:-1]).sum()); ntr += c[1:].size
        seen.update(np.unique(c).tolist())
        for b in range(len(c) // SEQ):
            seqs.append(c[b * SEQ:(b + 1) * SEQ])
    if not seqs:
        continue
    flat = torch.from_numpy(np.stack(seqs, 0).astype(np.int64))  # (B, F, n_tok) FLAT indices
    # gate.forecastability wants (B, F, n_tok, fsq_dim): PER-FSQ-DIGIT, not flat code ids.
    # Unpack the mixed radix used by the quantiser (levels [8,5,5,5] -> radix [1,8,40,200]).
    # This is almost certainly why m_trans/m_over came out `nan` in every spectro arm table:
    # the audit passed flat codes to an instrument that needs digits.
    L = [8, 5, 5, 5]
    radix = np.cumprod([1] + L[:-1])
    arr = torch.stack([(flat // int(radix[i])) % L[i] for i in range(len(L))], dim=-1)
    r = gate.forecastability(arr)
    fr = flips / max(ntr, 1)
    rows[m] = {"flip_rate": fr, **{k: float(v) for k, v in r.items()}, "codes_seen": len(seen)}
    print(f"{m:24s} {fr:10.3f} {r.get('margin_transition', float('nan')):10.4f} "
          f"{r.get('margin_overall', float('nan')):10.4f} "
          f"{int(r.get('n_transition', 0)):9d} {len(seen):11d}")
mt = [v['margin_transition'] for v in rows.values() if v['margin_transition'] == v['margin_transition']]
print(f"\nMEAN flip_rate {np.mean([v['flip_rate'] for v in rows.values()]):.3f}"
      f"   MEAN margin_transition {np.mean(mt):+.4f}   (positive = beats persistence)")
print(json.dumps(rows))
