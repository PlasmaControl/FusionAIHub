"""Run autoregressive IGNITE rollouts on MAST validation shots and render physical-unit plots.

Usage::

    OMP_NUM_THREADS=8 scripts/mast/py.sh scripts/mast/port_rollout_plots.py \
        --ckpt <run>/dynamics_best.pt \
        --cache-dir <frame_codes_dir> \
        --out-dir <plots_dir> \
        --n-shots 4 --seed 0 --actuator-mode real
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import torch

from tokamak_foundation_model.ignite.mast import paths, registry
from tokamak_foundation_model.ignite.mast.port import framecache, layout, rollout as port_rollout


def _resolve_device(spec: str) -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _select_shots(cache_dir: Path, n_shots: int, seed: int) -> list[str]:
    splits_path = cache_dir / "splits.json"
    if not splits_path.exists():
        raise FileNotFoundError(f"splits.json not found in {cache_dir}; pass --shots")
    splits = json.loads(splits_path.read_text())
    cached = {p.stem for p in cache_dir.glob("*.pt") if not p.stem.startswith("_")}
    val_shots = [str(s) for s in splits["val"] if str(s) in cached]
    if not val_shots:
        raise ValueError(f"no cached validation shots in {splits_path}")
    rng = random.Random(int(seed))
    k = min(int(n_shots), len(val_shots))
    return sorted(rng.sample(val_shots, k))


def _mode_tag(actuator_mode: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", actuator_mode).strip("_") or "real"


def _median_finite(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return float("nan")
    return float(np.median(finite))


def summarize_rollout_metrics(
    per_shot: dict[str, dict],
) -> dict[str, dict[str, float | int]]:
    """Aggregate per-shot metrics into per-signal medians and coverage.

    Every shot where a signal had finite truth in the predicted window is kept,
    including all-NaN rollouts (which carry ``pred_coverage = 0.0`` and are never dropped).
    """
    grouped: dict[str, list[dict[str, float]]] = {}
    for shot_rec in per_shot.values():
        for name, m in shot_rec["signals"].items():
            # Keep every shot where the signal had truth-finite predicted points.
            if math.isfinite(m["pred_coverage"]):
                grouped.setdefault(name, []).append(m)

    summary: dict[str, dict[str, float | int]] = {}
    for name, rows in sorted(grouped.items()):
        summary[name] = {
            "median_rollout_nrmse": _median_finite([r["rollout_nrmse"] for r in rows]),
            "median_ceiling_nrmse": _median_finite([r["ceiling_nrmse"] for r in rows]),
            "median_pred_coverage": float(np.median([r["pred_coverage"] for r in rows])),
            "median_ceil_coverage": _median_finite([r["ceil_coverage"] for r in rows]),
            "median_pred_missing_frac": _median_finite([r["pred_missing_frac"] for r in rows]),
            "median_ceil_missing_frac": _median_finite([r["ceil_missing_frac"] for r in rows]),
            "n_shots": len(rows),
        }
    return summary


def run_rollout_plots(
    *,
    ckpt: Path,
    cache_dir: Path,
    out_dir: Path,
    cache_root: Path | None = None,
    assignment_path: Path | None = None,
    fit_dir: Path | None = None,
    shots: list[str] | None = None,
    n_shots: int = 4,
    seed: int = 0,
    k0: int = 0,
    horizon: int = 0,
    temperature: float = 1.0,
    actuator_mode: str = "real",
    device_spec: str = "auto",
) -> dict:
    """Roll out selected shots, decode to physical units, plot PNGs, write ``rollout_metrics.json``."""
    cache_dir = Path(cache_dir)
    out_dir = paths.prepare_write_root(Path(out_dir))
    manifest = layout.load_manifest(cache_dir)
    dt_frame = float(manifest.get("dt_frame", framecache.DT_FRAME))
    act_stats = framecache.ActuatorStats.from_manifest_dict(manifest["actuators"])
    signal_stds = manifest.get("signal_physical_std")

    cache_root = Path(cache_root or manifest["cache_v1_root"])
    assignment_path = Path(assignment_path or manifest["assignment_path"])
    fit_dir = Path(fit_dir or manifest["fit_dir"])
    stats_sha = str(manifest.get("stats_sha256") or "") or None

    device = _resolve_device(device_spec)
    model, cfg, step = port_rollout.load_mast_model(ckpt, cache_dir, device)

    signals = registry.all_signals()
    codecs, stats, _assignment = framecache.build_mast_codecs(
        signals, cache_root, fit_dir, assignment_path
    )

    shot_list = (
        [str(s) for s in shots]
        if shots is not None
        else _select_shots(cache_dir, n_shots, seed)
    )
    tag = _mode_tag(actuator_mode)
    per_shot: dict[str, dict] = {}

    for shot in shot_list:
        ro, entry, edited_act = port_rollout.rollout_mast_shot(
            model,
            cfg,
            cache_dir,
            shot,
            k0=k0,
            horizon=horizon,
            temperature=temperature,
            seed=seed,
            device=device,
            actuator_mode=actuator_mode,
        )
        t0 = float(entry.get("t0", manifest.get("t0_start", 0.0)))
        decoded, sig_metrics = port_rollout.decode_and_score_rollout(
            ro,
            cache_root / "shots" / f"{shot}.h5",
            codecs,
            stats,
            t0=t0,
            dt_frame=dt_frame,
            present=entry.get("present"),
            stats_sha256=stats_sha,
            signal_stds=signal_stds,
        )
        cf_nbi = (
            port_rollout.counterfactual_nbi_physical(edited_act, act_stats, stats)
            if actuator_mode != "real"
            else None
        )
        png_path = out_dir / f"rollout_{shot}_{tag}.png"
        port_rollout.plot_shot_rollout(
            shot,
            decoded,
            sig_metrics,
            ro.k0,
            ro.n_frames,
            t0,
            dt_frame,
            png_path,
            actuator_mode=actuator_mode,
            step=step,
            cf_nbi=cf_nbi,
        )
        per_shot[shot] = {
            "k0": int(ro.k0),
            "n_frames": int(ro.n_frames),
            "t0": t0,
            "figure": str(png_path),
            "signals": sig_metrics,
        }
        print(
            f"[port_rollout] shot {shot}: k0={ro.k0} F={ro.n_frames} -> {png_path}",
            flush=True,
        )

    summary = summarize_rollout_metrics(per_shot)
    record = {
        "ckpt": str(Path(ckpt).resolve()),
        "step": int(step),
        "cache_dir": str(cache_dir.resolve()),
        "actuator_mode": actuator_mode,
        "temperature": float(temperature),
        "seed": int(seed),
        "k0": int(k0 or cfg.k0_seed),
        "horizon": int(horizon),
        "shots": shot_list,
        "summary": summary,
        "per_shot": per_shot,
    }
    metrics_path = out_dir / "rollout_metrics.json"
    framecache.save_json_atomic(record, metrics_path)
    print(f"[port_rollout] metrics -> {metrics_path}", flush=True)
    return record


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("MAST IGNITE rollout plots in physical units")
    p.add_argument("--ckpt", required=True, help="Dynamics checkpoint (.pt)")
    p.add_argument("--cache-dir", required=True, help="MAST frame-code cache directory")
    p.add_argument("--out-dir", required=True, help="Output directory for PNGs and rollout_metrics.json")
    p.add_argument("--cache-root", default=None, help="MAST cache v1 directory (default: from manifest)")
    p.add_argument("--assignment", default=None, help="Assignment JSON (default: from manifest)")
    p.add_argument("--fit", default=None, help="Fit directory (default: from manifest)")
    p.add_argument("--shots", default="", help="Comma-separated shot IDs (overrides --n-shots)")
    p.add_argument("--n-shots", type=int, default=4, help="Number of val shots to sample")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for shot selection and rollout")
    p.add_argument("--k0", type=int, default=0, help="Seed frames (0 = checkpoint k0_seed)")
    p.add_argument("--horizon", type=int, default=0, help="Predicted frames (0 = to shot end / max_frames)")
    p.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    p.add_argument(
        "--actuator-mode",
        default="real",
        help="Actuator mode (real, zero, freeze, shuffle, donor:<shot>, group_zero:<name>, scale:..., ramp:..., set:...)",
    )
    p.add_argument("--device", default="auto", help="Torch device (auto, cpu, cuda:0, ...)")
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    shots = (
        [s.strip() for s in args.shots.split(",") if s.strip()]
        if args.shots.strip()
        else None
    )
    run_rollout_plots(
        ckpt=Path(args.ckpt),
        cache_dir=Path(args.cache_dir),
        out_dir=Path(args.out_dir),
        cache_root=Path(args.cache_root) if args.cache_root else None,
        assignment_path=Path(args.assignment) if args.assignment else None,
        fit_dir=Path(args.fit) if args.fit else None,
        shots=shots,
        n_shots=args.n_shots,
        seed=args.seed,
        k0=args.k0,
        horizon=args.horizon,
        temperature=args.temperature,
        actuator_mode=args.actuator_mode,
        device_spec=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
