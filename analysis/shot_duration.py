#!/usr/bin/env python
"""Per-shot plasma END time, and what share of the trained/validated window is actually live.

WHY. The v4 frame cache is a fixed 219 frames per shot (t = 1.00..11.90 s at 50 ms), taken from
the ACQUISITION grid, which runs to 12.05 s -- far beyond any discharge. So late frames are
post-shot. Token flip rate CANNOT detect this (post-shot noise flips codes exactly like live
turbulence: measured 0.647 early vs 0.636 late, a 1.6% decline). Neither can "% of peak"
thresholding, which trips on the baseline (it returned t_end from 0.13 s to 19.09 s).

METHOD. The raw grids START AT NEGATIVE TIME (-4.34 s), which gives a free pre-shot noise
reference per shot per signal. Baseline = median and MAD over t in [-4, -0.5]; plasma-on where
|y| > median + K*MAD. t_end = the last such sample. Cross-checked across signals that each
REQUIRE a plasma, so agreement between independent diagnostics validates the estimator rather
than assuming it: ts_core_density (Thomson), bolo (radiated power), neutron_rate (fusion), co2
(interferometry).

    python analysis/shot_duration.py --n_shots 200 --k 8
"""
import argparse, glob, sys
import numpy as np
sys.path.insert(0, "src")
from tokamak_foundation_model.ignite import spike

SIGNALS = ["ts_core_density", "bolo", "neutron_rate", "co2"]
T0, DT, NFR, WIN, K0 = 1.0, 0.05, 219, 100, 20


def t_end_for(f, key, k):
    """Last time |y| exceeds the PRE-SHOT noise floor by k MADs. None if unusable."""
    import h5py
    if key not in f:
        return None
    try:
        x = np.asarray(f[f"{key}/xdata"], dtype=np.float64)
        y = np.asarray(f[f"{key}/ydata"], dtype=np.float64)
    except Exception:
        return None
    if x.ndim != 1 or x.size < 50:
        return None
    if y.ndim == 2:
        if y.shape[1] != x.size:
            y = y.T
        if y.shape[1] != x.size:
            return None
        y = np.nanmedian(np.abs(y), axis=0)          # median over channels: robust to dead ones
    y = np.abs(np.asarray(y, dtype=np.float64)).ravel()
    if y.size != x.size or not np.isfinite(y).any():
        return None
    pre = (x > -4.0) & (x < -0.5)
    if pre.sum() < 20:
        return None                                   # no pre-shot reference -> refuse to guess
    b = y[pre][np.isfinite(y[pre])]
    if b.size < 20:
        return None
    med = np.median(b)
    mad = np.median(np.abs(b - med)) * 1.4826
    if not np.isfinite(mad) or mad <= 0:
        return None
    on = np.where((y > med + k * mad) & (x > 0.0) & (x < 12.5))[0]
    return float(x[on[-1]]) if on.size else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_shots", type=int, default=200)
    p.add_argument("--k", type=float, default=8.0, help="MADs above the pre-shot floor")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shots", type=str, default="",
                   help="path to a JSON list of shot ids (e.g. a run's split_shots.json 'val'), "
                        "or a comma-separated list. Overrides --n_shots sampling.")
    p.add_argument("--out", type=str, default="",
                   help="write {shot: t_end} JSON here (only shots with >=2 agreeing signals)")
    p.add_argument("--shard", type=str, default="0/1",
                   help="I/N -- process shots where idx %% N == I. 8751 shots at ~7.5 s each is "
                        "18 h serially; 64 shards is ~20 min. Merge the per-shard --out files.")
    a = p.parse_args()
    import h5py

    fs = sorted(glob.glob(f"{spike.DEFAULT_DATA_DIR}/*_processed.h5"))
    if a.shots == "ALL":
        use = fs
        print(f"  --shots ALL: {len(use)} shots on disk")
    elif a.shots:
        import json as _j
        if a.shots.endswith(".json"):
            obj = _j.load(open(a.shots))
            ids = obj if isinstance(obj, list) else (obj.get("val") or obj.get("shots") or [])
        else:
            ids = [x for x in a.shots.split(",") if x]
        ids = {str(x) for x in ids}
        use = [p for p in fs if p.split("/")[-1].replace("_processed.h5", "") in ids]
        print(f"  --shots: {len(use)} of {len(ids)} requested shots found on disk")
    else:
        rng = np.random.default_rng(a.seed)
        use = [fs[i] for i in rng.permutation(len(fs))[: a.n_shots]]
    if a.shard != "0/1":
        _i, _n = (int(x) for x in a.shard.split("/"))
        use = [p for k, p in enumerate(use) if k % _n == _i]
        print(f"  shard {_i}/{_n}: {len(use)} shots", flush=True)
    per = {s: [] for s in SIGNALS}
    cons = []
    per_shot = {}
    for pth in use:
        try:
            with h5py.File(pth, "r") as f:
                got = {s: t_end_for(f, s, a.k) for s in SIGNALS}
        except Exception:
            continue
        vals = [v for v in got.values() if v is not None]
        for s, v in got.items():
            if v is not None:
                per[s].append(v)
        if len(vals) >= 2:
            cons.append(np.median(vals))
            per_shot[pth.split("/")[-1].replace("_processed.h5", "")] = float(np.median(vals))

    if a.out:
        import json as _j
        _j.dump(per_shot, open(a.out, "w"), indent=1)
        print(f"  wrote {len(per_shot)} per-shot t_end -> {a.out}")
    print(f"  {len(use)} shots sampled, k={a.k} MADs above the pre-shot floor\n")
    print(f"  {'signal':>18s} {'n':>5s} {'p10':>7s} {'median':>7s} {'p90':>7s} {'max':>7s}")
    for s in SIGNALS:
        v = np.array(per[s])
        if v.size:
            print(f"  {s:>18s} {v.size:5d} {np.percentile(v,10):7.2f} {np.median(v):7.2f} "
                  f"{np.percentile(v,90):7.2f} {v.max():7.2f}")
        else:
            print(f"  {s:>18s} {0:5d}   (unusable)")
    c = np.array(cons)
    if not c.size:
        print("\n  NO shot had >=2 usable signals -- estimator failed, do not trust any number above.")
        return
    print(f"\n  CONSENSUS t_end (median over >=2 agreeing signals), n={c.size}:")
    print(f"    p10 {np.percentile(c,10):.2f}  median {np.median(c):.2f}  "
          f"p90 {np.percentile(c,90):.2f}  max {c.max():.2f} s")

    # what fraction of the DEPLOYMENT window's prediction target is live?
    print(f"\n  DEPLOYMENT PROTOCOL (starts 0..9): predicts frames {K0}..{9+WIN-1} "
          f"= t {T0+DT*K0:.2f}..{T0+DT*(9+WIN-1):.2f} s")
    for st in (0, 9):
        p0, p1 = st + K0, st + WIN - 1
        t0, t1 = T0 + DT * p0, T0 + DT * p1
        live = np.clip((np.minimum(c, t1) - t0) / (t1 - t0), 0, 1)
        print(f"    start {st:2d}  predict t {t0:.2f}-{t1:.2f}s   "
              f"LIVE fraction: median {np.median(live)*100:.0f}%  "
              f"p10 {np.percentile(live,10)*100:.0f}%  "
              f"shots fully live: {100*(c>=t1).mean():.0f}%")
    n_dead = (c < T0 + DT * K0)
    print(f"\n    shots whose plasma ends BEFORE the prediction target even starts "
          f"({T0+DT*K0:.2f} s): {100*n_dead.mean():.1f}%")


if __name__ == "__main__":
    main()
