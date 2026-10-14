#!/usr/bin/env python
"""How much is the reported val/gen CE flattered by POST-SHOT padding?

The v4 cache is a fixed 219 frames/shot taken from the ACQUISITION grid (t 1.00-11.90 s), but
the median discharge ends at 5.85 s and the longest at 8.09 s (analysis/shot_duration.py).
The deployment validation protocol (starts 0..9) predicts t 2.00-6.40 s, so for many shots part
of the scored target is dead time -- which is near-trivially predictable and therefore LOWERS CE.

This scores the SAME 256 windows the trainer uses, then splits them by how much of each window's
PREDICTION TARGET is live plasma. No model change: each window is scored on its own (batch 1),
so grouping happens afterwards.

    python analysis/live_frame_bias.py --ckpt .../dynamics_latest.pt --tend val_tend.json
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, "src")
from tokamak_foundation_model.ignite.eval_dynamics import load_model
from tokamak_foundation_model.ignite import train_dynamics as td

T0, DT = 1.0, 0.05


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", required=True, help="run's split_shots.json")
    p.add_argument("--tend", required=True, help="{shot: t_end} from shot_duration.py --out")
    p.add_argument("--cache_dir", default="/lustre/orion/fus187/proj-shared/models/ignite_prod_v4/frame_codes")
    p.add_argument("--n_windows", type=int, default=256)
    p.add_argument("--val_windows_per_shot", type=int, default=10)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    # FAIL FAST. `--device cuda` on a task with no GPU binding used to die 15 s later inside
    # torch.load with a deserialization error that reads like a checkpoint problem rather than a
    # SLURM one (job 5465162). Say which it is, immediately.
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "[bias] --device cuda but torch.cuda.is_available() is False. The srun task has no "
            "GPU bound -- on Frontier use `--gpus-per-task=1 --gpu-bind=closest`, or run with "
            "--device cpu (~83 s/window).")
    dev = torch.device(a.device)
    print(f"[bias] devices visible: {torch.cuda.device_count()}  "
          f"ROCR_VISIBLE_DEVICES={__import__('os').environ.get('ROCR_VISIBLE_DEVICES','<unset>')}",
          flush=True)
    model, cfg, step = load_model(Path(a.ckpt), dev)
    print(f"[bias] ckpt step {step}  d_model {cfg.d_model} depth {cfg.depth} "
          f"k0 {cfg.k0_seed} n_predict {cfg.n_predict} device {dev}", flush=True)

    val = json.load(open(a.split))["val"]
    tend = {str(k): float(v) for k, v in json.load(open(a.tend)).items()}
    ds = td.FrameCodeDataset(a.cache_dir, [str(s) for s in val], cfg, presence=None,
                             windows_per_shot=a.val_windows_per_shot, window_sample="first")
    n = min(a.n_windows, len(ds))
    sel = [int(i) for i in torch.linspace(0, len(ds) - 1, steps=n).tolist()]
    print(f"[bias] val pool {len(ds)} windows, scoring {n} (same linspace the trainer uses)",
          flush=True)

    WIN, K0 = cfg.max_frames, cfg.k0_seed
    rows = []
    g = torch.Generator(device=dev)
    with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=torch.bfloat16,
                                         enabled=(dev.type == "cuda"), cache_enabled=False):
        for j, i in enumerate(sel):
            si, st = ds.index[i]
            shot = ds.shot_ids[si]
            te = tend.get(shot)
            if te is None:
                continue                       # no reliable t_end -> excluded, never guessed
            # PREDICTION TARGET of this window: frames st+K0 .. st+WIN-1
            f0, f1 = st + K0, st + WIN - 1
            t0, t1 = T0 + DT * f0, T0 + DT * f1
            live = float(np.clip((min(te, t1) - t0) / (t1 - t0), 0.0, 1.0))
            item = ds[i]
            batch = td._collate_frames([item])
            codes, act = batch[0], batch[1]
            cd = {k: v.to(dev) for k, v in codes.items()}
            av = act.to(dev)
            g.manual_seed(1234 + j)
            ce_inf = float(model.training_loss(cd, av, generator=g, ss_frac=0.0,
                                               present=None, gen_mask_p=0.0))
            g.manual_seed(1234 + j)
            ce_gen = float(model.training_loss(cd, av, generator=g, ss_frac=0.0, present=None,
                                               mask_ratio=1.0, history_frames=K0,
                                               gen_mask_p=0.0))
            rows.append((shot, st, live, ce_inf, ce_gen))
            if (j + 1) % 25 == 0:
                print(f"  scored {j+1}/{n}", flush=True)

    if not rows:
        print("[bias] no windows had a usable t_end -- cannot measure. ABORT.")
        return
    L = np.array([r[2] for r in rows])
    I = np.array([r[3] for r in rows])
    G = np.array([r[4] for r in rows])
    print(f"\n[bias] {len(rows)} windows with a usable t_end "
          f"(of {n} scored; {n - len(rows)} shots had none)\n")
    print(f"  {'group':>22s} {'n':>5s} {'infill CE':>10s} {'gen CE':>10s}")
    groups = [("ALL (reported metric)", L >= -1),
              ("fully live (>=99%)", L >= 0.99),
              ("mostly live (>=75%)", L >= 0.75),
              ("partly dead (25-75%)", (L >= 0.25) & (L < 0.75)),
              ("mostly dead (<25%)", L < 0.25)]
    res = {}
    for lab, m in groups:
        if m.sum() == 0:
            print(f"  {lab:>22s} {0:5d} {'-':>10s} {'-':>10s}"); continue
        res[lab] = (I[m].mean(), G[m].mean())
        print(f"  {lab:>22s} {int(m.sum()):5d} {I[m].mean():10.4f} {G[m].mean():10.4f}")
    if "ALL (reported metric)" in res and "fully live (>=99%)" in res:
        ai, ag = res["ALL (reported metric)"]
        li, lg = res["fully live (>=99%)"]
        print(f"\n  PADDING BIAS (live-only minus reported):")
        print(f"    infill  {li - ai:+.4f} nats     gen  {lg - ag:+.4f} nats")
        print(f"    -> the reported metric is OPTIMISTIC by that much; live plasma is harder.")
    r = np.corrcoef(L, G)[0, 1]
    print(f"\n  corr(live fraction, gen CE) = {r:+.3f}  "
          f"({'more live -> harder, as expected' if r > 0 else 'NO padding effect detected'})")


if __name__ == "__main__":
    main()
