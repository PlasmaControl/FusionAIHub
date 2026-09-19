"""Day-1 data landmine probe (CPU-only, login-node safe).

Answers, over N val shots × W windows each, BEFORE any GPU sweep:

1. **Copy-baseline degeneracy** — for every diagnostic, is the emitted target
   window actually different data from the input window? Suspicion from
   dev-peter's eval plan: bes/co2 may emit (near-)identical input/target,
   which would make their copy-baseline MAE ≈ 0 and the model/copy ratio
   meaningless (A1 caveat decision).
2. **Missingness** — per-modality ``*_valid`` rates (bes/co2/tangtv are known
   to be absent in a large fraction of shots).
3. **Contract check** — emitted tensor shapes vs the checkpoint's
   ``DiagnosticConfig``/``ActuatorConfig`` (channel drift, freq bins, frames).

Writes ``data_probe.json`` + prints a verdict table. Windows are sampled at a
1 s stride from t = 1 s, so verdicts cover ramp-up through flat-top.

Usage::

    python scripts/evaluation/smoke_data_probe.py --n-shots 40
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "src"))
sys.path.insert(0, str(_HERE.parent))

import torch  # noqa: E402

from tfm_eval.ckpt import DEFAULT_CKPT, configs_from_ckpt, load_ckpt  # noqa: E402
from tfm_eval.data import (  # noqa: E402
    assert_val_matches_training_cache,
    load_stats,
    make_prediction_dataset,
    resolve_split,
    shot_window_ranges,
)

IDENTICAL_TOL = 1e-7


def time_axis(kind: str) -> int:
    # video tensors are (C, T, H, W); everything else ends in time.
    return 1 if kind == "video" else -1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--n-shots", type=int, default=40)
    ap.add_argument("--windows-per-shot", type=int, default=8)
    ap.add_argument("--stride-s", type=float, default=1.0)
    ap.add_argument(
        "--out-dir",
        default=str(_HERE.parents[2] / "data" / "outputs" / "eval_suite" / "smoke"),
    )
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    ckpt = load_ckpt(args.checkpoint)
    ck_args = ckpt.get("args", {}) or {}
    diagnostics, actuators = configs_from_ckpt(ckpt)
    diag_by_name = {c.name: c for c in diagnostics}
    act_by_name = {c.name: c for c in actuators}
    print(f"[{time.time()-t0:6.1f}s] ckpt configs: {len(diagnostics)} diagnostics, "
          f"{len(actuators)} actuators")

    train_files, val_files = resolve_split(
        ck_args.get("data_dir", "/lustre/orion/fus187/proj-shared/foundation_model"),
        seed=ck_args.get("seed", 42),
        val_fraction=ck_args.get("val_fraction", 0.1),
    )
    assert_val_matches_training_cache(val_files)
    print(f"[{time.time()-t0:6.1f}s] split ok: {len(train_files)} train / "
          f"{len(val_files)} val (matches training-era cache)")

    shots = val_files[: args.n_shots]
    stats = load_stats(ck_args["stats_path"])
    ds = make_prediction_dataset(
        shots,
        stats,
        ck_args,
        diagnostics,
        actuators,
        step_size_s=args.stride_s,
        lengths_cache_path=out_dir / "lengths_smoke.pt",
        max_open_files=64,
    )
    ranges = shot_window_ranges(ds)
    print(f"[{time.time()-t0:6.1f}s] dataset: {len(ds)} windows over "
          f"{len(ranges)}/{len(shots)} readable shots")

    # Per-modality accumulators.
    st = defaultdict(lambda: {
        "n": 0, "n_valid": 0, "n_both_valid": 0, "n_identical": 0,
        "n_empty_target": 0, "sum_absdiff": 0.0, "sum_valid_frac": 0.0,
        "shape_in": None, "shape_tgt": None, "shape_mismatch": [],
    })

    sampled = []
    for path, start, n in ranges:
        for li in range(min(args.windows_per_shot, n)):
            sampled.append((path.name, start + li))
    print(f"[{time.time()-t0:6.1f}s] probing {len(sampled)} windows "
          f"(≤{args.windows_per_shot}/shot, stride {args.stride_s}s from t=1s)")

    for w, (shot_name, gidx) in enumerate(sampled):
        sample = ds[gidx]
        inputs, targets = sample["inputs"], sample["targets"]

        for name, cfg in diag_by_name.items():
            s = st[name]
            s["n"] += 1
            x, y = inputs.get(name), targets.get(name)
            if x is None or y is None:
                continue
            valid = float(inputs.get(f"{name}_valid", 0))
            s["sum_valid_frac"] += float(valid > 0)
            if s["shape_in"] is None:
                s["shape_in"], s["shape_tgt"] = list(x.shape), list(y.shape)
                exp = {
                    "slow_ts": (cfg.n_channels, cfg.window_samples),
                    "fast_ts": (cfg.n_channels, cfg.window_samples),
                    "spectrogram": (cfg.n_channels, cfg.freq_bins, cfg.window_samples),
                    "video": (cfg.n_channels, cfg.window_samples, cfg.height, cfg.width),
                }[cfg.kind]
                if tuple(x.shape) != exp:
                    s["shape_mismatch"].append(f"input {tuple(x.shape)} != ckpt {exp}")
            if valid <= 0:
                continue
            s["n_valid"] += 1
            ax = time_axis(cfg.kind)
            t_min = min(x.shape[ax], y.shape[ax])
            if t_min == 0:
                s["n_empty_target"] += 1
                continue
            xs = x.narrow(ax, 0, t_min)
            ys = y.narrow(ax, 0, t_min)
            diff = (xs.float() - ys.float()).abs()
            s["n_both_valid"] += 1
            s["sum_absdiff"] += float(diff.mean())
            if float(diff.max()) < IDENTICAL_TOL:
                s["n_identical"] += 1

        for name, cfg in act_by_name.items():
            s = st[f"act:{name}"]
            s["n"] += 1
            y = targets.get(name)
            if y is None:
                continue
            valid = float(targets.get(f"{name}_valid", 0))
            s["sum_valid_frac"] += float(valid > 0)
            if s["shape_tgt"] is None:
                s["shape_tgt"] = list(y.shape)
                exp = (cfg.n_channels, cfg.window_samples)
                if tuple(y.shape) != exp:
                    s["shape_mismatch"].append(f"target {tuple(y.shape)} != ckpt {exp}")

        if (w + 1) % 40 == 0:
            print(f"[{time.time()-t0:6.1f}s]   {w+1}/{len(sampled)} windows")

    # ── Verdicts ──────────────────────────────────────────────────────────
    report = {
        "checkpoint": args.checkpoint,
        "n_shots": len(ranges),
        "n_windows": len(sampled),
        "stride_s": args.stride_s,
        "identical_tol": IDENTICAL_TOL,
        "note": (
            "absdiff = mean|target-input| cropped to common time length, "
            "z-space, invalid positions zeroed upstream; equals a local "
            "copy-baseline MAE. valid_rate is the fraction of windows with "
            "*_valid > 0."
        ),
        "modalities": {},
    }
    print(f"\n{'modality':>18s} {'valid%':>7s} {'ident%':>7s} {'emptyT%':>8s} "
          f"{'copyMAE':>9s} {'T_in':>5s} {'T_tgt':>5s}  verdict")
    for name in list(diag_by_name) + [f"act:{a}" for a in act_by_name]:
        s = st[name]
        n = max(s["n"], 1)
        valid_rate = s["sum_valid_frac"] / n
        nbv = s["n_both_valid"]
        ident = s["n_identical"] / nbv if nbv else float("nan")
        empty = s["n_empty_target"] / max(s["n_valid"], 1)
        copy_mae = s["sum_absdiff"] / nbv if nbv else float("nan")
        if name.startswith("act:"):
            verdict = "actuator (targets only)"
        elif nbv == 0:
            verdict = "NO VALID WINDOWS"
        elif ident >= 0.5 or copy_mae < 1e-4:
            verdict = "DEGENERATE: target==input"
        elif copy_mae < 3e-2:
            verdict = "NEAR-IDENTICAL (weak copy baseline)"
        else:
            verdict = "distinct"
        if s["shape_mismatch"]:
            verdict += f"  SHAPE MISMATCH: {s['shape_mismatch']}"
        kind = diag_by_name[name].kind if name in diag_by_name else "actuator"
        ax = time_axis(kind)
        t_in = s["shape_in"][ax] if s["shape_in"] else -1
        t_tgt = s["shape_tgt"][ax] if s["shape_tgt"] else -1
        print(f"{name:>18s} {100*valid_rate:6.1f}% "
              f"{100*ident if nbv else float('nan'):6.1f}% {100*empty:7.1f}% "
              f"{copy_mae:9.4f} {t_in:5d} {t_tgt:5d}  {verdict}")
        report["modalities"][name] = {
            "kind": kind, "valid_rate": valid_rate, "n_both_valid": nbv,
            "frac_identical": ident if nbv else None,
            "frac_empty_target": empty, "copy_mae_local": copy_mae if nbv else None,
            "shape_in": s["shape_in"], "shape_tgt": s["shape_tgt"],
            "shape_mismatch": s["shape_mismatch"], "verdict": verdict,
        }

    metrics = ckpt.get("metrics") or {}
    copy_keys = {k: v for k, v in metrics.items() if "copy" in str(k).lower()}
    if copy_keys:
        print("\nckpt['metrics'] copy-baseline entries (training-time, for cross-ref):")
        for k in sorted(copy_keys):
            print(f"    {k}: {copy_keys[k]}")
        report["ckpt_copy_metrics"] = {
            str(k): float(v) for k, v in copy_keys.items()
            if isinstance(v, (int, float))
        }

    out_json = out_dir / "data_probe.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(f"\n[{time.time()-t0:6.1f}s] wrote {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
