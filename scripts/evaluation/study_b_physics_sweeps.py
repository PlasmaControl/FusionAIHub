"""B4: physics-signature actuator sweeps — K-step rollouts under sustained
physically-scaled actuator trajectories, checked against textbook responses.

Unlike B1/B2 (one-step, generic ±perturbations), each case here is a
response an experimentalist would predict without a model:

  gas_flow x{0.5..2}  -> line/core density ramps UP with dose (fueling
                          integrates over the second-scale rollout)
  pin      x{0.5..2}  -> Te, Ti, rotation UP (NBI heat + co-torque);
                          D-alpha / ELM activity UP with heating power
  ech_power x{0,1,2}  -> core Te UP; ne slightly DOWN (DIII-D ECH
                          density pump-out — soft expectation)
  rmp      x{0,1,2}   -> ne DOWN (RMP pump-out), rotation DOWN (braking),
                          ELM activity DOWN/suppressed (soft)

Scaling acts on the PHYSICAL actuator value over the whole K-step horizon
(invert stats -> scale -> re-apply), then the standard stage-2 rollout
loop re-tokenizes the scaled actuators every 50 ms step. Readouts are
inverse-transformed to physical units per step; filterscopes additionally
get a peak-rate estimate (MAD-prominence peaks/s on the stitched trace) as
a soft predicted-ELM-rate proxy. Ground-truth trajectories (actual future
under actual actuation) are stored for reference; missing-data windows are
cleaned to z=0 upstream, so GT curves are decoration, not a baseline.

Output: ``study_b/b4_physics_sweeps.npz`` + ``b4_meta.json`` — rendered by
``study_b_figures.py``. Single GCD, ~10 min at defaults.

    python scripts/evaluation/study_b_physics_sweeps.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "src"))
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parents[2] / "scripts" / "training"))

from tokamak_foundation_model.data.data_loader import (  # noqa: E402
    collate_fn_prediction,
)
from tokamak_foundation_model.e2e.rollout import TokenSpaceRollout  # noqa: E402

from eval_e2e import rollout_forward_one_batch  # noqa: E402
from study_b_actuator_scan import (  # noqa: E402
    READOUT_SIGNALS,
    build_pool,
    perturb_multiplicative,
)
from tfm_eval.ckpt import DEFAULT_CKPT, build_model_from_ckpt, load_ckpt  # noqa: E402
from tfm_eval.data import (  # noqa: E402
    assert_val_matches_training_cache,
    load_stats,
    make_prediction_dataset,
    resolve_split,
    shot_window_ranges,
)
from tfm_eval.inverse_preprocess import InversePreprocessor  # noqa: E402

# (case name, actuator, scales, readouts, expectation notes for the figure)
CASES = [
    ("gas_puff", "gas_flow", [0.5, 1.0, 1.5, 2.0],
     ["ne_core", "ne_tang"],
     {"ne_core": "expect: ramps UP with dose (fueling)",
      "ne_tang": "expect: ramps UP with dose (fueling)"}),
    ("nbi_power", "pin", [0.5, 1.0, 1.5, 2.0],
     ["te_core", "ti", "rotation", "d_alpha"],
     {"te_core": "expect: UP with dose (NBI heating)",
      "ti": "expect: UP with dose (NBI heating)",
      "rotation": "expect: UP with dose (co-injection torque)",
      "d_alpha": "expect: UP with dose (ELM activity rises)"}),
    ("ech", "ech_power", [0.0, 1.0, 2.0],
     ["te_core", "ne_core"],
     {"te_core": "expect: UP with dose (ECH electron heating)",
      "ne_core": "expect: slightly DOWN (pump-out, soft)"}),
    ("rmp", "rmp", [0.0, 1.0, 2.0],
     ["ne_core", "rotation", "d_alpha"],
     {"ne_core": "expect: DOWN with dose (RMP pump-out)",
      "rotation": "expect: DOWN with dose (braking, soft)",
      "d_alpha": "expect: DOWN with dose (ELM mitigation, soft)"}),
]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--k-steps", type=int, default=20)
    ap.add_argument("--n-shots", type=int, default=16)
    ap.add_argument("--windows-per-shot", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--out-root",
        default=str(_HERE.parents[2] / "data" / "outputs" / "eval_suite"
                    / "e2e_stage1_best"),
    )
    return ap.parse_args()


def peak_rate_hz(trace, fs_hz=10000.0, k_mad=6.0, min_dist_s=1e-3):
    """MAD-prominence peak rate on a 1-D physical D-alpha trace."""
    from scipy.signal import find_peaks

    x = np.asarray(trace, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 100:
        return float("nan")
    mad = np.median(np.abs(x - np.median(x)))
    if mad <= 0:
        return 0.0
    peaks, _ = find_peaks(
        x, prominence=k_mad * mad, distance=max(1, int(min_dist_s * fs_hz))
    )
    return float(len(peaks) / (x.size / fs_hz))


def rollout_readouts(model, rollout, pool, device, K, chunk_s, bsz, inv,
                     readout_names):
    """Rollout the pool in minibatches; per-step physical readout means.

    Returns ``traj`` {readout: (K, n_win)}, ``elm_rate`` (n_win,) from the
    stitched predicted filterscopes trace, and the same pair for GT.
    """
    sig_by_readout = dict((r, s) for r, s in READOUT_SIGNALS)
    n = next(iter(pool["inputs"].values())).shape[0]
    traj = {r: [] for r in readout_names}
    gt_traj = {r: [] for r in readout_names}
    rates, gt_rates = [], []
    fs_needed = "d_alpha" in readout_names
    with torch.no_grad():
        for i in range(0, n, bsz):
            sl = {
                side: {k: v[i:i + bsz] if torch.is_tensor(v) else v
                       for k, v in pool[side].items()}
                for side in ("inputs", "targets")
            }
            preds_k, _, tgts_k, _ = rollout_forward_one_batch(
                model, rollout, sl, device, K, chunk_s
            )
            for r in readout_names:
                sig = sig_by_readout[r]
                pk = torch.stack([
                    inv.inverse(sig, preds_k[k][sig].float().cpu())
                    .flatten(1).mean(1)
                    for k in range(K)
                ])  # (K, b)
                tk = torch.stack([
                    inv.inverse(sig, tgts_k[k][sig].float().cpu())
                    .flatten(1).mean(1)
                    for k in range(K)
                ])
                traj[r].append(pk.numpy())
                gt_traj[r].append(tk.numpy())
            if fs_needed:
                fp = torch.cat([
                    inv.inverse("filterscopes",
                                preds_k[k]["filterscopes"].float().cpu())
                    for k in range(K)
                ], dim=-1).squeeze(2).mean(1)  # (b, K*T)
                ft = torch.cat([
                    inv.inverse("filterscopes",
                                tgts_k[k]["filterscopes"].float().cpu())
                    for k in range(K)
                ], dim=-1).squeeze(2).mean(1)
                rates += [peak_rate_hz(fp[b]) for b in range(fp.shape[0])]
                gt_rates += [peak_rate_hz(ft[b]) for b in range(ft.shape[0])]
            del preds_k, tgts_k
    traj = {r: np.concatenate(v, axis=1) for r, v in traj.items()}
    gt_traj = {r: np.concatenate(v, axis=1) for r, v in gt_traj.items()}
    return traj, gt_traj, np.array(rates), np.array(gt_rates)


def main() -> int:
    args = parse_args()
    t0 = time.time()
    out_dir = Path(args.out_root) / "study_b"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    K = args.k_steps

    ckpt = load_ckpt(args.checkpoint)
    ck_args = ckpt.get("args", {}) or {}
    model, diagnostics, actuators = build_model_from_ckpt(ckpt, device=args.device)
    act_names = [c.name for c in actuators]
    chunk_s = ck_args["chunk_duration_s"]
    print(f"[{time.time()-t0:6.1f}s] model ready; horizon {K * chunk_s:.2f}s")

    _, val_files = resolve_split(
        ck_args.get("data_dir", "/lustre/orion/fus187/proj-shared/foundation_model"),
        seed=ck_args.get("seed", 42),
        val_fraction=ck_args.get("val_fraction", 0.1),
    )
    assert_val_matches_training_cache(val_files)
    stats = load_stats(ck_args["stats_path"])
    stride_s, warmup_s = 0.25, ck_args.get("warmup_s", 0.0)
    ds = make_prediction_dataset(
        val_files[: args.n_shots * 3], stats, ck_args, diagnostics, actuators,
        step_size_s=stride_s,
        prediction_horizon_s=K * chunk_s,
        lengths_cache_path=out_dir / f"lengths_b4_K{K}.pt",
        max_open_files=64,
    )
    samples, shot_names = build_pool(
        ds, shot_window_ranges(ds), args.n_shots, args.windows_per_shot,
        stride_s, warmup_s, t0,
    )
    pool = collate_fn_prediction(samples)
    n_pool = len(samples)
    print(f"[{time.time()-t0:6.1f}s] pool: {n_pool} windows from "
          f"{len(shot_names)} shots")

    inv = InversePreprocessor(
        str(ck_args["stats_path"]),
        dataset_kwargs={"channels_to_use": {"tangtv": [4, 6]}},
    )
    rollout = TokenSpaceRollout(model, dt_s=chunk_s)
    all_readouts = sorted({r for _, _, _, rs, _ in CASES for r in rs})

    # Baseline pass (scale 1.0) shared by every case.
    base_traj, gt_traj, base_rate, gt_rate = rollout_readouts(
        model, rollout, pool, device, K, chunk_s, args.batch_size, inv,
        all_readouts,
    )
    print(f"[{time.time()-t0:7.1f}s] baseline rollout done")

    data = {}
    for r in all_readouts:
        data[f"GT|{r}"] = gt_traj[r].astype(np.float32)
        data[f"BASE|{r}"] = base_traj[r].astype(np.float32)
    data["GT|elm_rate"] = gt_rate.astype(np.float32)
    data["BASE|elm_rate"] = base_rate.astype(np.float32)

    meta_cases = []
    for case, aname, scales, readouts, expect in CASES:
        if aname not in act_names:
            print(f"skip case {case}: actuator {aname} not in checkpoint")
            continue
        for scale in scales:
            if scale == 1.0:
                for r in readouts:
                    data[f"{case}|1.0|{r}"] = data[f"BASE|{r}"]
                if "d_alpha" in readouts:
                    data[f"{case}|1.0|elm_rate"] = data["BASE|elm_rate"]
                continue
            zb = pool["targets"][aname].float()
            zs = perturb_multiplicative(zb, aname, scale, inv)
            pert = {"inputs": pool["inputs"],
                    "targets": dict(pool["targets"])}
            pert["targets"][aname] = zs
            traj, _, rate, _ = rollout_readouts(
                model, rollout, pert, device, K, chunk_s, args.batch_size,
                inv, readouts,
            )
            for r in readouts:
                data[f"{case}|{scale}|{r}"] = traj[r].astype(np.float32)
            if rate.size:
                data[f"{case}|{scale}|elm_rate"] = rate.astype(np.float32)
            print(f"[{time.time()-t0:7.1f}s] {case} x{scale} done",
                  flush=True)
        meta_cases.append({
            "case": case, "actuator": aname, "scales": scales,
            "readouts": readouts, "expect": expect,
        })

    np.savez(out_dir / "b4_physics_sweeps.npz", **data,
             dt_s=chunk_s, n_windows=n_pool)
    (out_dir / "b4_meta.json").write_text(json.dumps({
        "cases": meta_cases, "k_steps": K, "dt_s": chunk_s,
        "n_windows": n_pool, "shots": shot_names,
        "elm_rate_note": "MAD-prominence peaks/s on stitched channel-mean "
                         "filterscopes trace; soft proxy",
    }, indent=1))
    print(f"[{time.time()-t0:7.1f}s] DONE → {out_dir / 'b4_physics_sweeps.npz'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
