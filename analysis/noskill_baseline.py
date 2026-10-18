#!/usr/bin/env python
"""MEASURE the no-skill CE baseline of a frame-code cache (per-column token marginal).

The baseline every "above/below no-skill" claim is read against. Method fixed 2026-08-24 and
reproduced here verbatim, because the number is NOT transferable between caches: it depends on
the vocabulary AND on the modality set, so a frame-width change invalidates it (prod_v2 at 1017
tokens measured 2.9055; ln 8 = 2.0794 inherited from band-power was 0.83 nats too low).

    per-(modality, column) token marginal fit on --fit_shots TRAIN shots with Laplace --alpha,
    scored on --score_shots HELD-OUT shots, averaged EQUALLY over modalities exactly as
    `training_loss` does.

MASK_ABSENT=0 ("all shots", the tracked condition) is the default; --present_only gives the
MASK_ABSENT=1 variant. Fit on TRAIN, score on VAL -- scoring a shot with its own within-shot
distribution is a different (oracle) quantity and is NOT a baseline.

    python analysis/noskill_baseline.py --cache_dir .../ignite_prod_v4/frame_codes
"""
import argparse, glob, json, math, os
from collections import defaultdict
import numpy as np, torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--fit_shots", type=int, default=200)
    p.add_argument("--score_shots", type=int, default=60)
    p.add_argument("--alpha", type=float, default=0.5, help="Laplace smoothing")
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--split_json", default=None,
                   help="a run's split_shots.json: fit on its train, score on its val")
    p.add_argument("--present_only", action="store_true",
                   help="MASK_ABSENT=1 variant: drop constant-code (absent) columns per shot")
    p.add_argument("--predictability", action="store_true",
                   help="ALSO report the two PREDICTABILITY numbers per modality: FLIP RATE "
                        "(fraction of the frame's tokens that change frame-to-frame) and the "
                        "model-free BIGRAM gain (marginal CE - bigram CE, nats). A bigram is "
                        "model-free, so no trained model can clear it: it is the one-step "
                        "information actually present in the token stream.")
    a = p.parse_args()

    shots = sorted(os.path.basename(f)[:-3] for f in glob.glob(f"{a.cache_dir}/*.pt"))
    if a.split_json:
        # FAITHFUL to the 2026-08-24 method: fit on the run's own TRAIN shots, score on its VAL
        # shots. A fresh permutation is NOT equivalent -- it reproduced v2 as 2.7398 vs the
        # recorded 2.9055 (off by 0.166 nats), which is how this path was found to be needed.
        sp = json.load(open(a.split_json))
        have = set(shots)
        fit = [s for s in sp["train"] if s in have][: a.fit_shots]
        score = [s for s in sp["val"] if s in have][: a.score_shots]
    else:
        rng = np.random.default_rng(a.split_seed)
        perm = rng.permutation(len(shots))
        fit = [shots[i] for i in perm[: a.fit_shots]]
        score = [shots[i] for i in perm[-a.score_shots :]]      # disjoint tail
    print(f"[noskill] {len(shots)} shots | fit {len(fit)} | score {len(score)} (disjoint) "
          f"| alpha {a.alpha} | {'present-only' if a.present_only else 'all shots'}")

    counts, vocabs = defaultdict(lambda: None), {}
    for s in fit:
        d = torch.load(f"{a.cache_dir}/{s}.pt", map_location="cpu", weights_only=False)
        for m, c in d["codes"].items():
            V = int(d["vocabs"][m]); vocabs[m] = V
            c = c.numpy()
            if counts[m] is None:
                counts[m] = np.zeros((c.shape[1], V), dtype=np.float64)
            for j in range(c.shape[1]):
                np.add.at(counts[m][j], c[:, j], 1.0)

    # log p per (modality, column) with Laplace alpha
    logp = {m: np.log((k + a.alpha) / (k.sum(1, keepdims=True) + a.alpha * vocabs[m]))
            for m, k in counts.items()}

    per_mod, n_tok = {}, {}
    for m in sorted(logp):
        tot, n = 0.0, 0
        for s in score:
            d = torch.load(f"{a.cache_dir}/{s}.pt", map_location="cpu", weights_only=False)
            if m not in d["codes"]:
                continue
            c = d["codes"][m].numpy()
            for j in range(c.shape[1]):
                col = c[:, j]
                if a.present_only and np.unique(col).size == 1:
                    continue                      # constant column == absent diagnostic
                tot += -logp[m][j][col].sum(); n += col.size
        if n:
            per_mod[m] = tot / n; n_tok[m] = counts[m].shape[0]

    print("\n  modality                n_tok      CE (nats)")
    for m, v in sorted(per_mod.items(), key=lambda kv: -kv[1]):
        print(f"    {m:24s} {n_tok[m]:5d}      {v:.4f}")
    mean = float(np.mean(list(per_mod.values())))
    print(f"\n  NO-SKILL BASELINE (equal-weight over {len(per_mod)} modalities) = {mean:.4f}")

    if a.predictability:
        # FLIP RATE + BIGRAM, both on the SCORE shots. The bigram is fit on the FIT shots
        # (per modality/column, prev->next with Laplace) so it is a held-out one-step predictor.
        big = {}
        for m in sorted(logp):
            V = vocabs[m]
            # sparse transition counts per column: dict[(col,prev)] -> Counter
            tr = defaultdict(lambda: defaultdict(float))
            for s in fit:
                d = torch.load(f"{a.cache_dir}/{s}.pt", map_location="cpu", weights_only=False)
                if m not in d["codes"]:
                    continue
                c = d["codes"][m].numpy()
                for j in range(c.shape[1]):
                    for pv, nx in zip(c[:-1, j], c[1:, j]):
                        tr[(j, int(pv))][int(nx)] += 1.0
            tot, n, flips, ntr = 0.0, 0, 0, 0
            for s in score:
                d = torch.load(f"{a.cache_dir}/{s}.pt", map_location="cpu", weights_only=False)
                if m not in d["codes"]:
                    continue
                c = d["codes"][m].numpy()
                flips += int((c[1:] != c[:-1]).sum()); ntr += c[1:].size
                for j in range(c.shape[1]):
                    for pv, nx in zip(c[:-1, j], c[1:, j]):
                        row = tr.get((j, int(pv)))
                        Z = (sum(row.values()) + a.alpha * V) if row else a.alpha * V
                        cnt = (row.get(int(nx), 0.0) if row else 0.0) + a.alpha
                        tot += -math.log(cnt / Z); n += 1
            if n:
                big[m] = {"bigram_ce": tot / n, "flip_rate": flips / max(ntr, 1)}
        print("\n  modality                flip_rate   marginal_CE   bigram_CE   gain(nats)")
        for m in sorted(big, key=lambda k: -big[k]["flip_rate"]):
            g = per_mod[m] - big[m]["bigram_ce"]
            print(f"    {m:24s} {big[m]['flip_rate']:7.3f}      {per_mod[m]:8.4f}    "
                  f"{big[m]['bigram_ce']:8.4f}    {g:+7.4f}")
        gains = [per_mod[m] - big[m]["bigram_ce"] for m in big]
        print(f"\n  MEAN BIGRAM GAIN over marginal = {float(np.mean(gains)):+.4f} nats"
              f"   MEAN FLIP RATE = {float(np.mean([big[m]['flip_rate'] for m in big])):.3f}")
        out_pred = {m: big[m] for m in big}
    else:
        out_pred = None
    print(f"  frame tokens = {sum(n_tok.values())}")
    out = {"cache_dir": a.cache_dir, "baseline": mean, "per_modality": per_mod,
           "n_tok": n_tok, "frame_tokens": sum(n_tok.values()),
           "present_only": a.present_only, "fit_shots": len(fit), "score_shots": len(score),
           "predictability": out_pred}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
