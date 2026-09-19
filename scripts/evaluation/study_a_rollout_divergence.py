"""A9: token-space rollout divergence — per-modality MAE(k) vs copy(k).

Rolls the world model forward K steps (default 80 × 50 ms = 4 s) from
flat-top starts, feeding predicted diagnostic tokens back while re-tokenizing
ground-truth future actuators (``TokenSpaceRollout`` semantics — the same
loop as stage-2 training). At each horizon k it accumulates the masked MAE
of (a) the model prediction and (b) the step-0 persistence baseline against
the ground-truth window at t + (k+1)·50 ms.

Output: ``study_a/a9_rollout_divergence.npz`` with (K, n_diag) weighted-MAE
arrays for model and copy + per-modality divergence horizons (first k where
model MAE exceeds the copy baseline).

Run inside an sbatch (single GCD, ~15 min at defaults):
    python scripts/evaluation/study_a_rollout_divergence.py
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

from torch.utils.data import DataLoader  # noqa: E402

from tokamak_foundation_model.data.data_loader import (  # noqa: E402
    collate_fn_prediction,
)
from tokamak_foundation_model.e2e.rollout import TokenSpaceRollout  # noqa: E402

from eval_e2e import rollout_forward_one_batch  # noqa: E402
from tfm_eval.ckpt import DEFAULT_CKPT, build_model_from_ckpt, load_ckpt  # noqa: E402
from tfm_eval.data import (  # noqa: E402
    assert_val_matches_training_cache,
    load_stats,
    make_prediction_dataset,
    resolve_split,
    shot_window_ranges,
)
from tfm_eval.metrics import per_sample_masked_mae  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--k-steps", type=int, default=80)
    ap.add_argument("--n-shots", type=int, default=64)
    ap.add_argument("--starts-per-shot", type=int, default=4)
    ap.add_argument("--start-stride-s", type=float, default=1.5)
    ap.add_argument("--first-start-s", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--out-root",
        default=str(_HERE.parents[2] / "data" / "outputs" / "eval_suite"
                    / "e2e_stage1_best"),
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    t0 = time.time()
    out_dir = Path(args.out_root) / "study_a"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = load_ckpt(args.checkpoint)
    ck_args = ckpt.get("args", {}) or {}
    model, diagnostics, actuators = build_model_from_ckpt(ckpt, device=args.device)
    diag_names = [c.name for c in diagnostics]
    cfg_by_name = {c.name: c for c in diagnostics}
    chunk_s = ck_args["chunk_duration_s"]
    K = args.k_steps
    print(f"[{time.time()-t0:6.1f}s] model ready; K={K} "
          f"({K*chunk_s:.2f}s horizon)")

    _, val_files = resolve_split(
        ck_args.get("data_dir", "/lustre/orion/fus187/proj-shared/foundation_model"),
        seed=ck_args.get("seed", 42),
        val_fraction=ck_args.get("val_fraction", 0.1),
    )
    assert_val_matches_training_cache(val_files)
    stats = load_stats(ck_args["stats_path"])
    # Flat-top start grid: t = first_start + i*start_stride, i < starts_per_shot.
    ck2 = dict(ck_args, warmup_s=args.first_start_s)
    ds = make_prediction_dataset(
        val_files[: args.n_shots], stats, ck2, diagnostics, actuators,
        step_size_s=args.start_stride_s,
        prediction_horizon_s=K * chunk_s,
        lengths_cache_path=out_dir / f"lengths_a9_K{K}.pt",
        max_open_files=80,
    )
    indices = []
    for _, start, n in shot_window_ranges(ds):
        indices.extend(range(start, start + min(args.starts_per_shot, n)))
    print(f"[{time.time()-t0:6.1f}s] {len(indices)} rollout starts from "
          f"{args.n_shots} shots")

    loader = DataLoader(
        ds, batch_size=args.batch_size, sampler=indices,
        num_workers=args.num_workers, collate_fn=collate_fn_prediction,
        pin_memory=args.device.startswith("cuda"),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    rollout = TokenSpaceRollout(model, dt_s=chunk_s)
    device = torch.device(args.device)
    D = len(diag_names)
    num = np.zeros((K, D)); den = np.zeros((K, D))
    cnum = np.zeros((K, D)); cden = np.zeros((K, D))
    n_done = 0

    with torch.no_grad():
        for batch in loader:
            preds_k, diag_init, tgts_k, masks_k = rollout_forward_one_batch(
                model, rollout, batch, device, K, chunk_s
            )
            # Step-0 persistence in the target's normalized frame
            # (video already z-scored; spectro inputs cropped to trunc_t).
            copy0 = {}
            for cfg in diagnostics:
                x = diag_init[cfg.name]
                if cfg.kind == "spectrogram":
                    x = x[..., : tgts_k[0][cfg.name].shape[-1]]
                copy0[cfg.name] = x
            for k in range(K):
                for j, name in enumerate(diag_names):
                    m, w = per_sample_masked_mae(
                        preds_k[k][name], tgts_k[k][name], masks_k[k][name]
                    )
                    ok = torch.isfinite(m) & (w > 0)
                    num[k, j] += float((m[ok] * w[ok]).sum())
                    den[k, j] += float(w[ok].sum())
                    cm, cw = per_sample_masked_mae(
                        copy0[name], tgts_k[k][name], masks_k[k][name]
                    )
                    cok = torch.isfinite(cm) & (cw > 0)
                    cnum[k, j] += float((cm[cok] * cw[cok]).sum())
                    cden[k, j] += float(cw[cok].sum())
            del preds_k, diag_init, tgts_k, masks_k, copy0
            n_done += next(iter(batch["inputs"].values())).shape[0]
            print(f"[{time.time()-t0:7.1f}s] {n_done}/{len(indices)} rollouts",
                  flush=True)

    mae_k = np.where(den > 0, num / np.maximum(den, 1), np.nan)
    copy_k = np.where(cden > 0, cnum / np.maximum(cden, 1), np.nan)
    horizon = {}
    for j, name in enumerate(diag_names):
        worse = np.where(mae_k[:, j] > copy_k[:, j])[0]
        horizon[name] = int(worse[0]) if worse.size else -1
    np.savez(
        out_dir / "a9_rollout_divergence.npz",
        mae_k=mae_k, copy_k=copy_k, den=den,
        diag_names=np.array(diag_names), dt_s=chunk_s,
        n_rollouts=n_done,
    )
    (out_dir / "a9_divergence_horizons.json").write_text(
        json.dumps(
            {"dt_s": chunk_s, "n_rollouts": n_done,
             "first_k_worse_than_copy": horizon},
            indent=1,
        )
    )
    print(f"[{time.time()-t0:7.1f}s] DONE — divergence horizons (k, -1=never): "
          f"{horizon}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
