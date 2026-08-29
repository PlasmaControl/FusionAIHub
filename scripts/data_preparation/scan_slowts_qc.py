"""Slow-TS data-quality scan over a shot list (ProcessPool). Per shot, in the
dataset-standardized space, reports:
  * ts_core_density / ts_core_temp MIN standardized value  -> TS drop-to-zero depth
    (a raw~0 dropout -> log10(1)=0 -> standardized ~ -18; real values ~ +/-2).
  * mse: whether the standardized signal has inf/nan + its finite max-abs
    -> locates the shot(s) that drove the MSE codec to NaN.

Output: eval_runs/slowts_qc/slowts_qc.pt + ranked text tables (TS deepest drops,
MSE worst shots). Run via scripts/slurm_frontier/scan_slowts_qc.sbatch.
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))

_STATS = None
_MODS = ("ts_core_density", "ts_core_temp", "mse")


def _init(stats_path):
    global _STATS
    torch.set_num_threads(1)
    _STATS = torch.load(stats_path, weights_only=False)


def _scan_one(args):
    shot, data_dir = args
    from tokamak_foundation_model.data.multi_file_dataset import TokamakMultiFileDataset
    path = os.path.join(data_dir, f"{shot}_processed.h5")
    out = {"ts_dens_min": None, "ts_temp_min": None,
           "mse_inf": 0, "mse_maxabs": 0.0, "n_win": 0}
    try:
        ds = TokamakMultiFileDataset(
            hdf5_paths=[path], chunk_duration_s=0.05, prediction_mode=True,
            prediction_horizon_s=0.05, step_size_s=0.01, warmup_s=1.0,
            preprocessing_stats=_STATS, input_signals=list(_MODS),
            target_signals=list(_MODS), lengths_cache_path=None)
        n = len(ds)
        if n == 0:
            return shot, out
        dmin = tmin = np.inf
        minf = 0
        mmax = 0.0
        nw = 0
        for i in range(n):
            inp = ds[i]["inputs"]
            d = inp.get("ts_core_density")
            if d is not None:
                a = torch.as_tensor(d).float()
                a = a[torch.isfinite(a)]
                if a.numel():
                    dmin = min(dmin, float(a.min()))
            t = inp.get("ts_core_temp")
            if t is not None:
                a = torch.as_tensor(t).float()
                a = a[torch.isfinite(a)]
                if a.numel():
                    tmin = min(tmin, float(a.min()))
            m = inp.get("mse")
            if m is not None:
                a = torch.as_tensor(m).float()
                minf += int((~torch.isfinite(a)).sum())
                af = a[torch.isfinite(a)]
                if af.numel():
                    mmax = max(mmax, float(af.abs().max()))
            nw += 1
        out.update(ts_dens_min=(None if dmin == np.inf else dmin),
                   ts_temp_min=(None if tmin == np.inf else tmin),
                   mse_inf=minf, mse_maxabs=mmax, n_win=nw)
    except Exception as e:
        return shot, {"__error__": str(e)}
    return shot, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/lustre/orion/fus187/proj-shared/foundation_model")
    ap.add_argument("--stats", default="/lustre/orion/fus187/proj-shared/foundation_model_meta/preprocessing_stats.pt")
    ap.add_argument("--shots_file", default="/lustre/orion/fus187/proj-shared/foundation_model_meta/shots_slowts_1000.txt")
    ap.add_argument("--out", default="eval_runs/slowts_qc")
    ap.add_argument("--workers", type=int, default=56)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    shots = [int(l.split()[0]) for l in open(args.shots_file)
             if l.strip() and not l.startswith("#")]
    print(f"[qc] {len(shots)} shots, workers={args.workers}", flush=True)
    tasks = [(s, args.data_dir) for s in shots]
    res = {}
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=(args.stats,)) as ex:
        for shot, r in ex.map(_scan_one, tasks, chunksize=4):
            res[shot] = r; done += 1
            if done % 200 == 0:
                print(f"[qc] {done}/{len(shots)}", flush=True)
    torch.save(res, os.path.join(args.out, "slowts_qc.pt"))
    ok = {s: r for s, r in res.items() if "__error__" not in r}
    # TS drop depth: shots with the deepest (most negative) standardized min
    ts = [(s, min(r["ts_dens_min"] if r["ts_dens_min"] is not None else 0,
                  r["ts_temp_min"] if r["ts_temp_min"] is not None else 0))
          for s, r in ok.items()
          if r["ts_dens_min"] is not None or r["ts_temp_min"] is not None]
    ts.sort(key=lambda x: x[1])
    with open(os.path.join(args.out, "ts_drop_depth.txt"), "w") as fh:
        fh.write("# shot  min_standardized (dens/temp) — deepest first (drop-to-zero ~ -18)\n")
        for s, m in ts:
            fh.write(f"{s}  {m:.2f}\n")
    for thr in (-6, -8, -10, -15):
        print(f"[qc] TS shots with min < {thr}: {sum(1 for _, m in ts if m < thr)}", flush=True)
    print("[qc] deepest 10 TS drops:", [(s, round(m, 1)) for s, m in ts[:10]], flush=True)
    # MSE bad shots: any inf, or extreme finite max-abs
    mse = [(s, r["mse_inf"], r["mse_maxabs"]) for s, r in ok.items()]
    bad = sorted([x for x in mse if x[1] > 0 or x[2] > 1e3], key=lambda x: -(x[1] + x[2]))
    with open(os.path.join(args.out, "mse_bad.txt"), "w") as fh:
        fh.write("# shot  n_inf  finite_maxabs  (bad = inf>0 or maxabs>1e3)\n")
        for s, ninf, mx in bad:
            fh.write(f"{s}  {ninf}  {mx:.3g}\n")
    print(f"[qc] MSE bad shots (inf or maxabs>1e3): {len(bad)}", flush=True)
    print("[qc] worst MSE:", [(s, ninf, round(mx, 1)) for s, ninf, mx in bad[:10]], flush=True)
    print("=== SLOWTS QC DONE ===", flush=True)


if __name__ == "__main__":
    main()
