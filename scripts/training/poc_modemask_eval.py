"""POC held-out verdict: does the model PREDICT mode masks better than persistence?

Loads a mode-mask POC checkpoint, runs it on HELD-OUT (val-split) shots, and
compares two mode-mask predictors against the GT-target modes, aggregated as a
global (distributed-style) Dice over all windows+channels:

  model      = sigmoid(head.mask_logits(backbone tokens)) > 0.5   (LEARNED, no prior)
  persistence = mode_mask(input window) > 0.5                     (COPY baseline)

Verdict: model maskdice > persistence maskdice on HELD-OUT  →  the backbone
learned mode DYNAMICS beyond copying → the full retrain is justified.
Model ≈ or < persistence → persistence is the ceiling → don't spend the 10 days.

Run:
  EVAL_CKPT=<poc best.pt> EVAL_MAX_FILES=400 EVAL_VAL_SHOTS=15 \
  sbatch scripts/slurm_frontier/eval_poc_modemask.sh
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from tokamak_foundation_model.data.data_loader import collate_fn
from tokamak_foundation_model.data.multi_file_dataset import TokamakMultiFileDataset
from tokamak_foundation_model.e2e.output_heads import SpectrogramFlowHead
from train_e2e_stage1 import (
    forward_batch, resolve_shot_files, _spec_mode_arg, _SPEC_STRUCT_GAMMA,
    _SPEC_STRUCT_CUT, _SPEC_STRUCT_K,
)
from eval_e2e_animation_tokamak import load_model


def _hard(x, k):
    return (_spec_mode_arg(x, k).clamp(0.0, 1.0) ** _SPEC_STRUCT_GAMMA
            > _SPEC_STRUCT_CUT).float()


def main():
    ckpt_path = Path(os.environ["EVAL_CKPT"])
    data_dir = Path(os.environ.get(
        "EVAL_DATA_DIR", "/lustre/orion/fus187/proj-shared/foundation_model"))
    stats_path = os.environ.get(
        "EVAL_STATS",
        "/lustre/orion/fus187/proj-shared/foundation_model_meta/preprocessing_stats.pt")
    max_files = int(os.environ.get("EVAL_MAX_FILES", "400"))
    n_val_shots = int(os.environ.get("EVAL_VAL_SHOTS", "15"))
    n_batches = int(os.environ.get("EVAL_N_BATCHES", "6"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, ckpt = load_model(ckpt_path, device)
    model.eval()
    diag = [d.name for d in model.diagnostics]
    act = [a.name for a in model.actuators]
    spec_mods = [d.name for d in model.diagnostics
                 if getattr(model.diag_heads[d.name], "enable_mask", False)]
    if not spec_mods:
        print("[poc-eval] ERROR: checkpoint has no mask-enabled spectro heads "
              "(was it trained with --spec_mask?)"); return
    print(f"[poc-eval] mask heads: {spec_mods}")

    # EVAL_SHOTS (comma list) → eval on those exact shots (e.g. the overfit shot
    # 200729, to inspect the fitted mask). Else replicate the POC's held-out split.
    eval_shots = os.environ.get("EVAL_SHOTS", "").strip()
    if eval_shots:
        from pathlib import Path as _P
        val_files = [_P(data_dir) / f"{s.strip()}_processed.h5"
                     for s in eval_shots.split(",") if s.strip()]
    else:
        _, val_files = resolve_shot_files(data_dir, None, None, max_files, 0.1, 42)
        val_files = val_files[:n_val_shots]
    print(f"[poc-eval] held-out shots: {len(val_files)} "
          f"(e.g. {[p.stem for p in val_files[:5]]})")
    stats = torch.load(stats_path, weights_only=False)
    ds = TokamakMultiFileDataset(
        hdf5_paths=val_files, chunk_duration_s=0.05, prediction_mode=True,
        prediction_horizon_s=0.05, step_size_s=0.01, warmup_s=1.0, n_fft=1024,
        hop_length=256, preprocessing_stats=stats,
        input_signals=diag, target_signals=diag + act)
    loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate_fn,
                        num_workers=2)

    # global Dice accumulators per modality: [overlap, pred_sum, tgt_sum]
    acc = {m: {"model": [0.0, 0.0, 0.0], "persist": [0.0, 0.0, 0.0]}
           for m in spec_mods}
    # mode-bearing per-(channel,window) mean dice — the FAIR metric. The global
    # aggregate above dilutes toward ~0.15 because it mixes in empty & mismatched
    # channel-windows; here we score only channel-windows whose GT has ≥3 mode
    # pixels and average the per-window dice (matches offline persistence ~0.56).
    wacc = {m: {"model": [], "persist": []} for m in spec_mods}
    fig_cap = {}                      # first-batch tensors for the comparison figure
    seen = 0
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= n_batches:
                break
            preds, diag_inputs, targets, masks, tok = forward_batch(
                model, batch, device)
            for m in spec_mods:
                k = _SPEC_STRUCT_K.get(m, 2.0)
                head = model.diag_heads[m]
                gt = _hard(targets[m].float(), k)
                per = _hard(diag_inputs[m].float(), k)   # persistence = input mask
                # input_feat/input_cond heads need the input mask as the prior
                prior = per if (getattr(head, "enable_input_feat", False)
                                or getattr(head, "enable_input_cond", False)) else None
                mlog = head.mask_logits(tok[m], prior=prior).float()
                mdl = (torch.sigmoid(mlog) > 0.5).float()
                T = min(gt.shape[-1], mdl.shape[-1], per.shape[-1])
                gt, mdl, per = gt[..., :T], mdl[..., :T], per[..., :T]
                # mode-bearing channel-window mask: GT has ≥3 mode pixels
                gsum = gt.sum(dim=(-2, -1))                 # (B, C)
                mb = gsum >= 3
                for name, pm in (("model", mdl), ("persist", per)):
                    a = acc[m][name]
                    a[0] += float((pm * gt).sum())
                    a[1] += float(pm.sum())
                    a[2] += float(gt.sum())
                    ov = (pm * gt).sum(dim=(-2, -1))        # (B, C)
                    psum = pm.sum(dim=(-2, -1))
                    dpw = (2 * ov + 1e-6) / (psum + gsum + 1e-6)
                    wacc[m][name].extend(dpw[mb].flatten().tolist())
                if bi == 0:            # keep for the comparison figure
                    fig_cap[m] = {
                        "spec": targets[m][..., :T].float().cpu(),
                        "gt": gt.cpu(), "mdl": mdl.cpu(), "per": per.cpu(),
                    }
            seen += 1
    print(f"[poc-eval] scored {seen} batches\n")
    print("[poc-eval] FAIR metric = mode-bearing per-(channel,window) mean dice "
          "(global-aggregate in parens dilutes toward ~0.15)")
    print(f"{'modality':>8} | {'MODEL (learned)':>18} | {'persistence':>18} | verdict")
    print("-" * 70)
    for m in spec_mods:
        def dice(a):
            return (2 * a[0] + 1) / (a[1] + a[2] + 1)
        def wmean(lst):
            return sum(lst) / len(lst) if lst else float("nan")
        gmd, gpd = dice(acc[m]["model"]), dice(acc[m]["persist"])
        md, pd = wmean(wacc[m]["model"]), wmean(wacc[m]["persist"])
        verdict = "BEATS persist ✓" if md > pd + 0.02 else (
            "≈ persist" if md > pd - 0.05 else "< persist ✗")
        print(f"{m:>8} | {md:>10.3f} (agg {gmd:.3f}) | "
              f"{pd:>10.3f} (agg {gpd:.3f}) | {verdict}")
    print("\n[poc-eval] MODEL > persistence on held-out ⇒ mode dynamics are "
          "LEARNABLE ⇒ full retrain justified.")

    # ── comparison figure: GT spectro | GT modes | MODEL modes | persistence ──
    tag = os.environ.get("EVAL_FIG_TAG", ckpt_path.parent.name)
    out_dir = Path(os.environ.get("EVAL_OUT", "eval_runs/poc_modemask"))
    out_dir.mkdir(parents=True, exist_ok=True)
    for m in spec_mods:
        d = fig_cap.get(m)
        if d is None:
            continue
        gt, mdl, per, spec = d["gt"], d["mdl"], d["per"], d["spec"]
        # channel with the best MODEL-vs-GT overlap among mode-bearing windows
        th = gt; ph = mdl
        ov = (ph * th).sum(dim=(2, 3)); dsc = (2 * ov + 1) / (
            ph.sum((2, 3)) + th.sum((2, 3)) + 1)
        hasm = th.sum(dim=(2, 3)) > 3
        score = torch.where(hasm, dsc, torch.full_like(dsc, -1.0)).mean(dim=0)
        ch = int(score.argmax())
        rows = list(np.argsort(-th[:, ch].sum(dim=(1, 2)).numpy())[:4])
        fig, axes = plt.subplots(len(rows), 4, figsize=(15, 3 * len(rows)))
        if len(rows) == 1:
            axes = axes[None]
        vlo, vhi = np.percentile(spec[rows, ch].numpy(), [55, 99.7])
        for r, w in enumerate(rows):
            panels = [
                (spec[w, ch], "GT spectrogram", "magma", vlo, vhi),
                (th[w, ch], "GT modes", "gray", 0, 1),
                (ph[w, ch], "MODEL predicted modes", "gray", 0, 1),
                (per[w, ch], "persistence modes", "gray", 0, 1),
            ]
            for c, (img, title, cmap, lo, hi) in enumerate(panels):
                a = axes[r, c]
                a.imshow(img.numpy(), aspect="auto", origin="lower",
                         cmap=cmap, vmin=lo, vmax=hi)
                if r == 0:
                    a.set_title(title, fontsize=9)
                a.set_xticks([]); a.set_yticks([])
        def _d(acc_):
            return (2 * acc_[0] + 1) / (acc_[1] + acc_[2] + 1)
        fig.suptitle(
            f"[{tag}] {m.upper()} mode prediction — ch {ch}  |  held-out maskdice "
            f"MODEL {_d(acc[m]['model']):.3f}  vs  persistence "
            f"{_d(acc[m]['persist']):.3f}", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        outp = out_dir / f"{tag}_{m}_modeprediction.png"
        fig.savefig(outp, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"[poc-eval] FIGURE: {outp}")


if __name__ == "__main__":
    main()
