"""B3-layerwise: where does actuator information attenuate in the backbone?

For a deterministic flat-top val pool, compares the baseline forward
against actuator perturbations — ALL actuators zeroed, then one at a time.
Zeroing acts on the *normalized* actuator tensor, so z=0 means "replace the
window-specific trajectory with the dataset-mean trajectory": it removes
the information content rather than switching hardware off, which is the
right probe for information propagation. Measures

  (a) per-backbone-layer mean L2 over the DIAGNOSTIC token slice, both
      absolute and relative to the baseline token norms at that layer
      (pre-norm transformers grow token norms with depth — the relative
      profile is the interpretable one),
  (b) the same over the ACTUATOR token slice (sanity: the perturbation must
      enter large at layer 0),
  (c) per-diagnostic head-output relative L2 (what reaches the outputs).

Output ``study_b/b3_layerwise.npz`` — rendered by ``study_b_figures.py``.
Single GCD, ~5 min at defaults (intermediates kept for one minibatch pair:
(n_layers+2) x B x 1178 x 1024 fp32 ≈ 2 GB at B=8).

    python scripts/evaluation/study_b_layerwise.py
"""

import argparse
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

from eval_e2e import (  # noqa: E402
    _clean_and_mask,
    _spectro_trunc_t,
    _video_standardize_per_bc,
)
from study_b_actuator_scan import build_pool  # noqa: E402
from tfm_eval.ckpt import DEFAULT_CKPT, build_model_from_ckpt, load_ckpt  # noqa: E402
from tfm_eval.data import (  # noqa: E402
    assert_val_matches_training_cache,
    load_stats,
    make_prediction_dataset,
    resolve_split,
    shot_window_ranges,
)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--n-shots", type=int, default=16)
    ap.add_argument("--windows-per-shot", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--out-root",
        default=str(_HERE.parents[2] / "data" / "outputs" / "eval_suite"
                    / "e2e_stage1_best"),
    )
    return ap.parse_args()


def prep_inputs(model, batch, device):
    """Trainer-faithful input prep (mirrors tfm_eval.latents)."""
    diag_inputs = {}
    for cfg in model.diagnostics:
        raw = batch["inputs"][cfg.name].to(device).float()
        cleaned, _ = _clean_and_mask(raw, None)
        if cfg.kind == "video":
            cleaned, _, _ = _video_standardize_per_bc(cleaned)
        elif cfg.kind == "spectrogram":
            cleaned = cleaned[..., : _spectro_trunc_t(cfg)]
        diag_inputs[cfg.name] = cleaned
        if cfg.kind in ("video", "spectrogram"):
            vk = f"{cfg.name}_valid"
            if vk in batch["inputs"]:
                diag_inputs[vk] = batch["inputs"][vk].to(device)
    act_inputs = {}
    for cfg in model.actuators:
        raw = batch["targets"][cfg.name].to(device).float()
        cleaned, _ = _clean_and_mask(raw, None)
        act_inputs[cfg.name] = cleaned
    return diag_inputs, act_inputs


@torch.no_grad()
def forward_intermediates(model, diag_inputs, act_inputs, device):
    bsz = next(iter(act_inputs.values())).shape[0]
    step = torch.zeros(bsz, dtype=torch.long, device=device)
    toff = torch.zeros(bsz, device=device)
    tokens = model.tokenize(diag_inputs, act_inputs)
    inter = model.backbone(tokens, step, toff, return_intermediates=True)
    heads = model.decode(inter[-1])
    return inter, heads


def main() -> int:
    args = parse_args()
    t0 = time.time()
    out_dir = Path(args.out_root) / "study_b"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    ckpt = load_ckpt(args.checkpoint)
    ck_args = ckpt.get("args", {}) or {}
    model, diagnostics, actuators = build_model_from_ckpt(ckpt, device=args.device)
    diag_names = [c.name for c in diagnostics]
    act_names = [c.name for c in actuators]
    rows = ["ALL"] + act_names
    diag_end = max(lo.slice_.stop for lo in model.token_layout
                   if lo.is_diagnostic)
    n_tok = model.token_layout[-1].slice_.stop
    print(f"[{time.time()-t0:6.1f}s] model ready; diag tokens [0,{diag_end}), "
          f"act tokens [{diag_end},{n_tok})")

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
        lengths_cache_path=out_dir / "lengths_layerwise.pt",
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

    L = None
    acc = None  # filled once L is known
    for i in range(0, n_pool, args.batch_size):
        sl = {
            side: {k: v[i:i + args.batch_size] if torch.is_tensor(v) else v
                   for k, v in pool[side].items()}
            for side in ("inputs", "targets")
        }
        diag_in, act_in = prep_inputs(model, sl, device)
        base_inter, base_heads = forward_intermediates(
            model, diag_in, act_in, device
        )
        if L is None:
            L = len(base_inter)
            acc = {
                "diag_num": np.zeros((len(rows), L)),
                "act_num": np.zeros((len(rows), L)),
                "diag_base": np.zeros(L),
                "act_base": np.zeros(L),
                "head_num": np.zeros((len(rows), len(diag_names))),
                "head_den": np.zeros(len(diag_names)),
                "n": 0,
            }
        bsz = next(iter(act_in.values())).shape[0]
        for lix, a in enumerate(base_inter):
            acc["diag_base"][lix] += float(a[:, :diag_end].norm(dim=-1).sum())
            acc["act_base"][lix] += float(a[:, diag_end:].norm(dim=-1).sum())
        for dj, name in enumerate(diag_names):
            acc["head_den"][dj] += float(
                base_heads[name].float().pow(2).sum()
            )

        for ri, row in enumerate(rows):
            pert = {
                n: (torch.zeros_like(v) if (row == "ALL" or n == row) else v)
                for n, v in act_in.items()
            }
            inter_p, heads_p = forward_intermediates(
                model, diag_in, pert, device
            )
            for lix, (a, b) in enumerate(zip(base_inter, inter_p)):
                d = (a - b).norm(dim=-1)  # (B, n_tok)
                acc["diag_num"][ri, lix] += float(d[:, :diag_end].sum())
                acc["act_num"][ri, lix] += float(d[:, diag_end:].sum())
            for dj, name in enumerate(diag_names):
                acc["head_num"][ri, dj] += float(
                    (base_heads[name].float() - heads_p[name].float())
                    .pow(2).sum()
                )
            del inter_p, heads_p
        acc["n"] += bsz
        del base_inter, base_heads
        print(f"[{time.time()-t0:7.1f}s] {min(i+args.batch_size, n_pool)}"
              f"/{n_pool} windows", flush=True)

    n_diag_tok = diag_end
    n_act_tok = n_tok - diag_end
    layer_abs = acc["diag_num"] / (acc["n"] * n_diag_tok)
    layer_rel = acc["diag_num"] / np.maximum(acc["diag_base"], 1e-12)
    act_layer_rel = acc["act_num"] / np.maximum(acc["act_base"], 1e-12)
    head_rel = np.sqrt(acc["head_num"] / np.maximum(acc["head_den"], 1e-12))

    np.savez(
        out_dir / "b3_layerwise.npz",
        layer_abs=layer_abs, layer_rel=layer_rel,
        act_layer_rel=act_layer_rel, head_rel=head_rel,
        row_names=np.array(rows), diag_names=np.array(diag_names),
        n_windows=acc["n"], n_diag_tokens=n_diag_tok,
        n_act_tokens=n_act_tok,
    )
    print(f"[{time.time()-t0:7.1f}s] DONE — final-layer diag-slice relative "
          "L2 per row:")
    for ri, row in enumerate(rows):
        print(f"  {row:18s} L{L-1}: {layer_rel[ri, -1]:.5f}  "
              f"head max: {head_rel[ri].max():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
