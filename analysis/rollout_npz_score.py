#!/usr/bin/env python
"""Score rollout archives (codes_<shot>_step<N>.npz) against persistence, in a space BOTH heads share.

eval_dynamics archives gt__<m> and pred__<m> (100, n_tok) codes per shot even when it times out
before writing eval_metrics.json. This scores them offline, for a discrete arm and a latent arm
alike, on the SAME shots:

  exact acc     P(pred code == gt code) over the predicted frames [K0, F)
                -- near-useless for spectro: at flip ~0.99 both model and persistence sit at ~0.005
  near-1 acc    P(every FSQ grid coordinate within +-1 cell of gt)  <- sees the NEAR-MISS that
                exact match cannot; this is the metric the latent head is designed to move
  coord RMSE    RMS distance in normalised [0,1] coordinate space (lower is better)
each vs FREEZE-LAST-SEED-FRAME persistence, per modality and as the equal-weight mean.

    python analysis/rollout_npz_score.py runs/msk600/rollout15_examples runs/lat600/rollout15_examples
"""
import sys, glob, os
import numpy as np
sys.path.insert(0, "src")
LEVELS = (8, 5, 5, 5)


def coords(idx):
    r, acc = [], 1
    for L in LEVELS:
        r.append(acc); acc *= L
    radix = np.array(r); L = np.array(LEVELS)
    d = (idx[..., None] // radix) % L
    return d, d / (L - 1.0)


def score_dir(d):
    out = {}
    for f in sorted(glob.glob(f"{d}/codes_*.npz")):
        z = np.load(f, allow_pickle=True); K0, F = int(z["K0"]), int(z["F"]); sh = str(z["shot"])
        for pk in [k for k in z.files if k.startswith("pred__")]:
            m = pk[6:]; gt, pr = z[f"gt__{m}"], z[pk]
            if pr.shape != gt.shape: continue
            tgt, prd = gt[K0:F], pr[K0:F]; pers = np.broadcast_to(gt[K0 - 1][None, :], tgt.shape)
            # STATIC SCREEN (default on): an absent/frozen diagnostic encodes to constant tokens over the
            # whole predicted window, handing persistence AND the model a free 1.0 (28% of the 12-shot
            # protocol). Those pairs say nothing about dynamics, so they are dropped from every mean.
            if not KEEP_STATIC and (tgt == tgt[0]).all():
                SCREENED.append((sh, m)); continue
            dg, cg = coords(tgt); dp, cp = coords(prd); ds, cs = coords(pers)
            e = out.setdefault(m, {k: [] for k in ("exact", "exact_p", "near1", "near1_p", "rmse", "rmse_p", "shots")})
            e["exact"].append((prd == tgt).mean());      e["exact_p"].append((pers == tgt).mean())
            e["near1"].append((np.abs(dp - dg) <= 1).all(-1).mean())
            e["near1_p"].append((np.abs(ds - dg) <= 1).all(-1).mean())
            e["rmse"].append(np.sqrt(((cp - cg) ** 2).mean())); e["rmse_p"].append(np.sqrt(((cs - cg) ** 2).mean()))
            # MOTION: near-1 vs persistence REWARDS FLATNESS (a frozen rollout scores ~persistence), so report
            # how much the rollout moves relative to the data and whether it moves WHEN the data moves.
            _, cg1 = coords(gt[K0 - 1:F]); _, cp1 = coords(pr[K0 - 1:F])
            dg = np.abs(np.diff(cg1, axis=0)).mean((1, 2)); dp = np.abs(np.diff(cp1, axis=0)).mean((1, 2))
            e.setdefault("mot_pred", []).append(float(dp.mean())); e.setdefault("mot_gt", []).append(float(dg.mean()))
            e.setdefault("timing", []).append(float(np.corrcoef(dg, dp)[0, 1]) if dg.std() > 0 and dp.std() > 0 else np.nan)
            e["shots"].append(sh)
    return out


SPLIT = set()   # --split 190729,190735,190736 : report PINNED (in train) vs HELD-OUT means separately
KEEP_STATIC = False   # --keep_static : do NOT drop (shot, modality) pairs whose GT is constant over the window
SCREENED = []


def main():
    global SPLIT, KEEP_STATIC
    args = sys.argv[1:]
    if "--split" in args:
        i = args.index("--split"); SPLIT = set(args[i + 1].split(",")); args = args[:i] + args[i + 2:]
    if "--keep_static" in args:
        KEEP_STATIC = True; args.remove("--keep_static")
    dirs = args or ["/lustre/orion/fus187/proj-shared/models/ignite_prod_v4/runs/mskfull/rollout_best"]
    for d in dirs:
        SCREENED.clear(); r = score_dir(d)
        if not r: print(f"\n{d}: no archives"); continue
        n = len(set(sum((v["shots"] for v in r.values()), [])))
        npairs = sum(len(v["shots"]) for v in r.values())
        print(f"\n=== {d.rstrip('/').split('/runs/')[-1]}   ({n} shots; {npairs} scored pairs, {len(SCREENED)} static pairs screened out) ===")
        print(f"  {'modality':22s} {'exact':>6s} {'pers':>6s} {'skill':>7s} | {'near1':>6s} {'pers':>6s} {'skill':>7s} | {'rmse':>6s} {'pers':>6s} | {'motion':>6s} {'timing':>6s}")
        agg = {k: [] for k in ("exact", "exact_p", "near1", "near1_p", "rmse", "rmse_p", "mot_pred", "mot_gt", "timing")}
        for m in sorted(r, key=lambda k: -(np.mean(r[k]["near1"]) - np.mean(r[k]["near1_p"]))):
            v = {k: float(np.nanmean(r[m][k])) for k in agg}; v["motion"] = v["mot_pred"] / max(v["mot_gt"], 1e-9)
            for k in agg: agg[k].append(v[k])
            print(f"  {m:22s} {v['exact']:6.3f} {v['exact_p']:6.3f} {v['exact']-v['exact_p']:+7.3f} | "
                  f"{v['near1']:6.3f} {v['near1_p']:6.3f} {v['near1']-v['near1_p']:+7.3f} | "
                  f"{v['rmse']:6.3f} {v['rmse_p']:6.3f} | {v['motion']:6.2f} {v['timing']:6.3f}")
        A = {k: float(np.nanmean(v)) for k, v in agg.items()}; A["motion"] = A["mot_pred"] / max(A["mot_gt"], 1e-9)
        print(f"  {'MEAN':22s} {A['exact']:6.3f} {A['exact_p']:6.3f} {A['exact']-A['exact_p']:+7.3f} | "
              f"{A['near1']:6.3f} {A['near1_p']:6.3f} {A['near1']-A['near1_p']:+7.3f} | {A['rmse']:6.3f} {A['rmse_p']:6.3f} | {A['motion']:6.2f} {A['timing']:6.3f}")
        if SPLIT:
            for grp, sel in (("PINNED", lambda x: x in SPLIT), ("HELD-OUT", lambda x: x not in SPLIT)):
                n1, n1p, mp, mg = [], [], [], []
                for m in r:
                    idx = [i for i, x in enumerate(r[m]["shots"]) if sel(x)]
                    if not idx: continue
                    n1.append(np.mean([r[m]["near1"][i] for i in idx])); n1p.append(np.mean([r[m]["near1_p"][i] for i in idx]))
                    mp.append(np.mean([r[m]["mot_pred"][i] for i in idx])); mg.append(np.mean([r[m]["mot_gt"][i] for i in idx]))
                if n1:
                    print(f"  {grp:9s} near1 {np.mean(n1):.3f} vs pers {np.mean(n1p):.3f} ({np.mean(n1)-np.mean(n1p):+.3f})"
                          f"  motion {np.mean(mp)/max(np.mean(mg),1e-9):.2f}  [{sum(1 for a, b in zip(n1, n1p) if a > b)}/{len(n1)} beat pers]")
        print(f"  beats persistence: exact {sum(1 for m in r if np.mean(r[m]['exact'])>np.mean(r[m]['exact_p']))}/{len(r)}"
              f"   near-1 {sum(1 for m in r if np.mean(r[m]['near1'])>np.mean(r[m]['near1_p']))}/{len(r)}"
              f"   rmse {sum(1 for m in r if np.mean(r[m]['rmse'])<np.mean(r[m]['rmse_p']))}/{len(r)}")


if __name__ == "__main__":
    main()
