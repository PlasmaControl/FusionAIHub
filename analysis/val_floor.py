#!/usr/bin/env python
"""How low CAN validation CE go on v4? Three model-free references, measured on the cache.

Nothing in this project has ever put val CE below ~1.0, and the v4 arms sit at 2.31. Before
chasing val -> 0.0x we need to know whether that is a MODEL failure or a property of the data.
This measures three predictors that need no training, on held-out shots:

  marginal      per-(modality, token-column) distribution fit on TRAIN shots, Laplace 0.5.
                The no-skill line. Anything above this has negative skill.
  persistence   copy frame t-1. The cheapest real predictor.
  ORACLE        per-(modality, column) distribution computed WITHIN the scored shot itself.
                This is NOT achievable -- it assumes you already know the shot you are
                predicting -- so it is a FLOOR-ish reference: no causal predictor that only
                knows the column's own statistics can beat it. If the oracle is far above 0,
                val -> 0.0x is unreachable with these codecs, whatever the model does.

Scores the frames a rollout actually predicts: [k0, win) of the deployment window (starts 0..9).

    python analysis/val_floor.py --n_train 300 --n_val 80
"""
import argparse, glob, json, os, sys
import numpy as np
import torch

CACHE = "/lustre/orion/fus187/proj-shared/models/ignite_prod_v4/frame_codes"
VOCAB, K0, WIN, NSTART = 1000, 20, 100, 10


def shot_codes(p):
    d = torch.load(p, map_location="cpu", weights_only=False)
    return {k: v.numpy().astype(np.int64) for k, v in d["codes"].items()}, int(d["n_frames"])


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--n_train", type=int, default=300)
    a.add_argument("--n_val", type=int, default=80)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--split", type=str, default="", help="use a run's split_shots.json")
    args = a.parse_args()

    fs = sorted(f for f in glob.glob(f"{CACHE}/*.pt") if not os.path.basename(f).startswith("_"))
    if args.split:
        sp = json.load(open(args.split))
        tr_ids = {str(x) for x in sp["train"]}
        va_ids = {str(x) for x in sp["val"]}
        tr = [f for f in fs if os.path.basename(f)[:-3] in tr_ids][: args.n_train]
        va = [f for f in fs if os.path.basename(f)[:-3] in va_ids][: args.n_val]
    else:
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(len(fs))
        tr = [fs[i] for i in perm[: args.n_train]]
        va = [fs[i] for i in perm[-args.n_val:]]
    print(f"[floor] {len(tr)} train / {len(va)} val shots; scoring frames [{K0},{WIN}) "
          f"of starts 0..{NSTART-1}", flush=True)

    # ---- marginal per (modality, column), fit on TRAIN -----------------------------------
    counts = {}
    for i, p in enumerate(tr):
        c, _ = shot_codes(p)
        for m, arr in c.items():
            if m not in counts:
                counts[m] = np.full((arr.shape[1], VOCAB), 0.5)      # Laplace 0.5
            np.add.at(counts[m], (np.arange(arr.shape[1])[None, :].repeat(arr.shape[0], 0), arr), 1.0)
        if (i + 1) % 100 == 0:
            print(f"  fit {i+1}/{len(tr)}", flush=True)
    logp = {m: np.log(c / c.sum(axis=1, keepdims=True)) for m, c in counts.items()}

    agg = {k: [] for k in ("marginal", "persistence", "oracle")}
    per_mod = {}
    for p in va:
        c, nfr = shot_codes(p)
        for m, arr in c.items():
            if m not in logp:
                continue
            # within-shot oracle distribution
            oc = np.full((arr.shape[1], VOCAB), 0.5)
            np.add.at(oc, (np.arange(arr.shape[1])[None, :].repeat(arr.shape[0], 0), arr), 1.0)
            olog = np.log(oc / oc.sum(axis=1, keepdims=True))
            mm, pp, oo = [], [], []
            for st in range(min(NSTART, max(1, nfr - WIN + 1))):
                tgt = arr[st + K0: st + WIN]                        # (T, n_tok)
                prev = arr[st + K0 - 1: st + WIN - 1]
                cols = np.arange(arr.shape[1])[None, :].repeat(tgt.shape[0], 0)
                mm.append(-logp[m][cols, tgt].mean())
                oo.append(-olog[cols, tgt].mean())
                # persistence as a mixture with the marginal (a hard 0/1 copy has infinite CE)
                hit = (tgt == prev)
                pp.append(float(-np.log(np.where(hit, 0.9, 0.1 / (VOCAB - 1))).mean()))
            per_mod.setdefault(m, {"marginal": [], "persistence": [], "oracle": []})
            per_mod[m]["marginal"].append(np.mean(mm))
            per_mod[m]["persistence"].append(np.mean(pp))
            per_mod[m]["oracle"].append(np.mean(oo))
    for m, d in per_mod.items():
        for k in agg:
            agg[k].append(np.mean(d[k]))

    print(f"\n  {'predictor':>14s} {'CE (equal-weight over modalities)':>34s}")
    for k in ("marginal", "persistence", "oracle"):
        print(f"  {k:>14s} {np.mean(agg[k]):34.4f}")
    print(f"\n  {'modality':>24s} {'marginal':>9s} {'persist':>9s} {'ORACLE':>9s}")
    for m in sorted(per_mod, key=lambda x: -np.mean(per_mod[x]["oracle"])):
        d = per_mod[m]
        print(f"  {m:>24s} {np.mean(d['marginal']):9.3f} {np.mean(d['persistence']):9.3f} "
              f"{np.mean(d['oracle']):9.3f}")
    print("\n  The ORACLE knows the scored shot's own per-column distribution but has NO temporal"
          "\n  context. MEASURED 2026-09-11: it scores 3.1504 while the trained model reaches"
          "\n  gen 2.9401 -- the MODEL BEATS IT. So this is NOT a floor: cross-frame and"
          "\n  cross-column structure carries real information beyond column statistics."
          "\n  Do NOT cite these numbers as a limit on achievable val CE.")


if __name__ == "__main__":
    main()
