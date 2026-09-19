"""GPU sweep: per-window latents + errors + physics scalars → per-shot npz.

The single extraction pass feeding Studies A1/A2/A4-A8 and C. For every
window (default 0.25 s stride — non-overlapping, so NN / probe statistics
aren't corrupted by the 80 %-overlap native training grid) it records:

  * pooled backbone latents, fp16: one vector per token_layout entry
    (12 diagnostics + 9 actuators) + diag_global + act_global
  * per-diagnostic model MAE and persistence("copy") MAE with mask weights
    (re-aggregatable exactly: Σ mae·w / Σ w)
  * physics scalars in physical units (ne/te/ti/rotation/d_alpha, actuator
    totals) for coloring and probe targets
  * validity flags per modality

One npz per shot (atomic rename, so the sweep is resumable per shot) plus a
per-shard csv of shot-level means. Shard with ``--shard/--n-shards`` for the
8-GCD sbatch layout.

Usage::

    # login-GPU smoke (2 shots)
    python scripts/evaluation/extract_latents.py --split val --max-shots 2
    # one shard of the full sweep (inside sbatch, ROCR_VISIBLE_DEVICES set)
    python scripts/evaluation/extract_latents.py --split val --shard 3 --n-shards 8
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "src"))
sys.path.insert(0, str(_HERE.parent))

from torch.utils.data import DataLoader  # noqa: E402

from tokamak_foundation_model.data.data_loader import (  # noqa: E402
    collate_fn_prediction,
)

from tfm_eval.ckpt import DEFAULT_CKPT, build_model_from_ckpt, load_ckpt  # noqa: E402
from tfm_eval.data import (  # noqa: E402
    assert_val_matches_training_cache,
    load_stats,
    make_prediction_dataset,
    matched_train_subset,
    resolve_split,
    shot_window_ranges,
    window_start_s,
)
from tfm_eval.inverse_preprocess import InversePreprocessor  # noqa: E402
from tfm_eval.latents import copy_prediction, forward_with_latents  # noqa: E402
from tfm_eval.metrics import per_sample_masked_mae  # noqa: E402
from tfm_eval.physics import window_physics_scalars  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--split", choices=("val", "train"), default="val")
    ap.add_argument("--stride-s", type=float, default=0.25)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--max-shots", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--force", action="store_true", help="recompute existing npz")
    ap.add_argument(
        "--out-root",
        default=None,
        help="default: data/outputs/eval_suite/<ckpt-stem>/",
    )
    return ap.parse_args()


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=_HERE.parents[2],
        ).stdout.strip()
    except Exception:
        return "unknown"


def main() -> int:
    args = parse_args()
    t0 = time.time()

    ckpt = load_ckpt(args.checkpoint)
    ck_args = ckpt.get("args", {}) or {}
    model, diagnostics, actuators = build_model_from_ckpt(ckpt, device=args.device)
    diag_names = [c.name for c in diagnostics]
    act_names = [c.name for c in actuators]
    print(f"[{time.time()-t0:6.1f}s] model on {args.device} "
          f"({sum(p.numel() for p in model.parameters())/1e6:.0f}M params)")

    out_root = Path(
        args.out_root
        or _HERE.parents[2] / "data" / "outputs" / "eval_suite"
        / Path(args.checkpoint).stem
    )
    lat_dir = out_root / "latents" / args.split
    lat_dir.mkdir(parents=True, exist_ok=True)

    train_files, val_files = resolve_split(
        ck_args.get("data_dir", "/lustre/orion/fus187/proj-shared/foundation_model"),
        seed=ck_args.get("seed", 42),
        val_fraction=ck_args.get("val_fraction", 0.1),
    )
    assert_val_matches_training_cache(val_files)
    files = (
        list(val_files) if args.split == "val"
        else matched_train_subset(train_files, len(val_files))
    )
    files = files[args.shard :: args.n_shards]
    if args.max_shots is not None:
        files = files[: args.max_shots]
    print(f"[{time.time()-t0:6.1f}s] split={args.split} shard "
          f"{args.shard}/{args.n_shards}: {len(files)} shots")

    stats = load_stats(ck_args["stats_path"])
    ds = make_prediction_dataset(
        files, stats, ck_args, diagnostics, actuators,
        step_size_s=args.stride_s,
        lengths_cache_path=(
            lat_dir / f"lengths_shard{args.shard}of{args.n_shards}.pt"
        ),
        max_open_files=128,
    )
    inv = InversePreprocessor(
        str(ck_args["stats_path"]),
        dataset_kwargs={"channels_to_use": {"tangtv": [4, 6]}},
    )

    warmup_s = ck_args.get("warmup_s", 0.0)
    ranges = shot_window_ranges(ds)

    # Resume: build the flat list of pending global indices (whole shots).
    pending: list = []          # (global_idx, shot_key, local_idx)
    shot_meta: dict = {}        # shot_key -> {n, path, npz}
    for path, start, n in ranges:
        shot_key = path.name.replace("_processed.h5", "")
        npz_path = lat_dir / f"{shot_key}.npz"
        if npz_path.exists() and not args.force:
            continue
        shot_meta[shot_key] = {"n": n, "path": path, "npz": npz_path}
        pending.extend((start + li, shot_key, li) for li in range(n))
    print(f"[{time.time()-t0:6.1f}s] {len(shot_meta)}/{len(ranges)} shots pending "
          f"({len(pending)} windows)")
    if not pending:
        print("nothing to do")
        return 0

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        sampler=[g for g, _, _ in pending],
        num_workers=args.num_workers,
        collate_fn=collate_fn_prediction,
        pin_memory=args.device.startswith("cuda"),
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    config = {
        "checkpoint": args.checkpoint, "split": args.split,
        "stride_s": args.stride_s, "warmup_s": warmup_s,
        "shard": args.shard, "n_shards": args.n_shards,
        "git_sha": git_sha(), "diag_names": diag_names,
        "act_names": act_names,
        "ckpt_step": int(ckpt.get("step", -1)),
    }
    (out_root / "config.json").write_text(json.dumps(config, indent=2))

    csv_path = lat_dir / f"per_shot_shard{args.shard}of{args.n_shards}.csv"
    csv_new = not csv_path.exists()
    csv_fh = open(csv_path, "a", newline="")
    csv_w = csv.writer(csv_fh)
    if csv_new:
        csv_w.writerow(
            ["shot", "n_windows"]
            + [f"mae_{n}" for n in diag_names]
            + [f"copy_mae_{n}" for n in diag_names]
            + [f"valid_{n}" for n in diag_names]
        )

    device = torch.device(args.device)
    buffers: dict = {}
    cursor = 0
    done_shots = 0
    pooled_names = None

    def flush(shot_key: str) -> None:
        nonlocal done_shots
        meta = shot_meta[shot_key]
        buf = buffers.pop(shot_key)
        order = np.argsort(buf["local_idx"])
        arrays = {
            k: np.asarray(v)[order]
            for k, v in buf.items()
        }
        arrays["t_start_s"] = np.array(
            [window_start_s(li, warmup_s, args.stride_s)
             for li in arrays["local_idx"]],
            dtype=np.float32,
        )
        arrays["pooled_names"] = np.array(pooled_names)
        arrays["diag_names"] = np.array(diag_names)
        arrays["act_names"] = np.array(act_names)
        arrays["physics_names"] = np.array(buf_physics_names)
        # tmp name must END in .npz — np.savez appends the suffix otherwise,
        # and the rename source would never exist.
        tmp = meta["npz"].with_name(meta["npz"].stem + ".tmp.npz")
        np.savez_compressed(tmp, **arrays)
        tmp.rename(meta["npz"])

        mae = arrays["mae"]; cmae = arrays["copy_mae"]; vw = arrays["mae_w"]
        cw = arrays["copy_mae_w"]

        def pool(m, w):
            out = []
            for j in range(m.shape[1]):
                ok = np.isfinite(m[:, j]) & (w[:, j] > 0)
                out.append(
                    float((m[ok, j] * w[ok, j]).sum() / w[ok, j].sum())
                    if ok.any() else float("nan")
                )
            return out

        csv_w.writerow(
            [shot_key, len(order)]
            + [f"{v:.6f}" for v in pool(mae, vw)]
            + [f"{v:.6f}" for v in pool(cmae, cw)]
            + [f"{arrays['diag_valid'][:, j].mean():.3f}"
               for j in range(len(diag_names))]
        )
        csv_fh.flush()
        done_shots += 1

    for batch in loader:
        bsz = next(iter(batch["inputs"].values())).shape[0]
        rows = pending[cursor : cursor + bsz]
        cursor += bsz

        physics = window_physics_scalars(batch, inv)
        buf_physics_names = sorted(physics)

        result = forward_with_latents(model, batch, device)

        mae = torch.full((bsz, len(diag_names)), float("nan"))
        mae_w = torch.zeros(bsz, len(diag_names))
        cmae = torch.full((bsz, len(diag_names)), float("nan"))
        cmae_w = torch.zeros(bsz, len(diag_names))
        dvalid = torch.zeros(bsz, len(diag_names), dtype=torch.uint8)
        for j, cfg in enumerate(diagnostics):
            name = cfg.name
            m, w = per_sample_masked_mae(
                result["predictions"][name], result["targets"][name],
                result["masks"][name],
            )
            mae[:, j], mae_w[:, j] = m.cpu(), w.cpu()
            cm, cw = per_sample_masked_mae(
                copy_prediction(cfg, result), result["targets"][name],
                result["masks"][name],
            )
            cmae[:, j], cmae_w[:, j] = cm.cpu(), cw.cpu()
            v = batch["inputs"].get(f"{name}_valid")
            if v is not None:
                dvalid[:, j] = (v > 0).to(torch.uint8)
        avalid = torch.zeros(bsz, len(act_names), dtype=torch.uint8)
        for j, name in enumerate(act_names):
            v = batch["targets"].get(f"{name}_valid")
            if v is not None:
                avalid[:, j] = (v > 0).to(torch.uint8)

        pooled = result["pooled"]
        if pooled_names is None:
            pooled_names = list(pooled)
        lat = torch.stack(
            [pooled[k] for k in pooled_names], dim=1
        ).cpu().to(torch.float16)  # (B, K, d_model)
        phys = torch.stack(
            [physics[k] for k in buf_physics_names], dim=1
        ).float()

        for i, (_, shot_key, li) in enumerate(rows):
            buf = buffers.setdefault(shot_key, {
                "local_idx": [], "pooled": [], "mae": [], "mae_w": [],
                "copy_mae": [], "copy_mae_w": [], "diag_valid": [],
                "act_valid": [], "physics": [],
            })
            buf["local_idx"].append(li)
            buf["pooled"].append(lat[i].numpy())
            buf["mae"].append(mae[i].numpy())
            buf["mae_w"].append(mae_w[i].numpy())
            buf["copy_mae"].append(cmae[i].numpy())
            buf["copy_mae_w"].append(cmae_w[i].numpy())
            buf["diag_valid"].append(dvalid[i].numpy())
            buf["act_valid"].append(avalid[i].numpy())
            buf["physics"].append(phys[i].numpy())
            if len(buf["local_idx"]) == shot_meta[shot_key]["n"]:
                flush(shot_key)

        if cursor % (args.batch_size * 25) < args.batch_size:
            print(f"[{time.time()-t0:7.1f}s] {cursor}/{len(pending)} windows, "
                  f"{done_shots}/{len(shot_meta)} shots flushed", flush=True)

    for shot_key in list(buffers):
        print(f"WARNING: flushing incomplete shot {shot_key} "
              f"({len(buffers[shot_key]['local_idx'])}/{shot_meta[shot_key]['n']})")
        flush(shot_key)
    csv_fh.close()
    print(f"[{time.time()-t0:7.1f}s] DONE: {done_shots} shots → {lat_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
