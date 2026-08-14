"""Phase-0 validation: persistence-conditioned spectrogram forecast render.

Shows the proposed spectro fix END-TO-END on the real production model, WITHOUT
any architecture change or training: the model's forecast envelope μ (mean of the
generative spectro head) is fused with the PERSISTENCE mask computed from the
OBSERVED INPUT window (production binarization) — i.e. propagate the observed
modes forward, fill the rest with the forecast envelope. No ground truth is used
(input = observed past), so this is a genuine single-step forecast.

For each of a few windows of one shot it plots  GT-target | μ (flat) |
persistence-forecast (μ + input modes) | persistence mask,  and prints the
per-window maskdice (persistence vs GT-target modes) — the number that should
match the ~0.64 ECE ceiling.

Run via SLURM (needs a GPU for the d1024 backbone):
  EVAL_CKPT=<best.pt> EVAL_SHOT=200729 EVAL_MODALITY=ece \
  sbatch scripts/slurm_frontier/eval_phase0_persistence.sh
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
from train_e2e_stage1 import (
    forward_batch, _spec_mode_arg, _SPEC_STRUCT_GAMMA, _SPEC_STRUCT_CUT,
    _SPEC_STRUCT_K,
)

os.environ["EVAL_RENDER_MEAN"] = "1"          # spectro head returns μ (mean)
from eval_e2e_animation_tokamak import load_model  # noqa: E402


def _mode_soft(x, k):
    """Production soft mode mask (B,C,F,T) in [0,1]."""
    return _spec_mode_arg(x, k).clamp(0.0, 1.0) ** _SPEC_STRUCT_GAMMA


def main():
    ckpt_path = Path(os.environ["EVAL_CKPT"])
    shot = int(os.environ.get("EVAL_SHOT", "200729"))
    modality = os.environ.get("EVAL_MODALITY", "ece")
    data_dir = os.environ.get(
        "EVAL_DATA_DIR", "/lustre/orion/fus187/proj-shared/foundation_model"
    )
    stats_path = os.environ.get(
        "EVAL_STATS",
        "/lustre/orion/fus187/proj-shared/foundation_model_meta/preprocessing_stats.pt",
    )
    out_dir = Path(os.environ.get(
        "EVAL_OUT", "eval_runs/phase0_persistence"
    ))
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k = _SPEC_STRUCT_K.get(modality, 2.0)

    print(f"[phase0] loading model {ckpt_path}")
    model, ckpt = load_model(ckpt_path, device)
    model.eval()
    diag_names = [d.name for d in model.diagnostics]
    act_names = [a.name for a in model.actuators]
    assert modality in diag_names, f"{modality} not in {diag_names}"
    print(f"[phase0] diagnostics={diag_names} actuators={act_names}")

    stats = torch.load(stats_path, weights_only=False)
    ds = TokamakMultiFileDataset(
        hdf5_paths=[os.path.join(data_dir, f"{shot}_processed.h5")],
        chunk_duration_s=0.05, prediction_mode=True, prediction_horizon_s=0.05,
        step_size_s=0.01, warmup_s=1.0, n_fft=1024, hop_length=256,
        preprocessing_stats=stats,
        input_signals=diag_names, target_signals=diag_names + act_names,
    )
    # sample windows spread across the shot
    n = len(ds)
    idxs = list(range(0, n, max(1, n // 12)))[:12]
    loader = DataLoader([ds[i] for i in idxs], batch_size=len(idxs),
                        collate_fn=collate_fn)
    batch = next(iter(loader))

    with torch.no_grad():
        preds, diag_inputs, targets, masks, _ = forward_batch(model, batch, device)
    mu = preds[modality].float()                 # (B,C,F,T) forecast envelope
    xin = diag_inputs[modality].float()          # observed input window
    tgt = targets[modality].float()              # GT target window
    # align time length
    T = min(mu.shape[-1], xin.shape[-1], tgt.shape[-1])
    mu, xin, tgt = mu[..., :T], xin[..., :T], tgt[..., :T]

    pmask = _mode_soft(xin, k)                    # persistence mask (from INPUT)
    fused = mu * (1.0 - pmask) + xin * pmask      # μ background + observed modes
    tgt_soft = _mode_soft(tgt, k)

    # per-(window,channel) maskdice, then pick the channel by PERSISTENCE (not
    # density — high density ≠ coherent modes). "Forecastable" channel = the one
    # whose input modes best predict its output modes, among channels that
    # actually carry modes in ≥2 of the sampled windows.
    ph, th = (pmask > 0.5).float(), (tgt_soft > 0.5).float()
    ov = (ph * th).sum(dim=(2, 3))                       # (B,C)
    dice_bc = (2 * ov + 1) / (ph.sum((2, 3)) + th.sum((2, 3)) + 1)  # (B,C)
    has_mode = th.sum(dim=(2, 3)) > 3                     # (B,C) window carries modes
    n_mode_win = has_mode.sum(dim=0)                      # (C,)
    ch_score = torch.where(
        has_mode, dice_bc, torch.full_like(dice_bc, float("nan"))
    ).nanmean(dim=0)                                      # mean dice over mode windows
    ch_score = torch.where(n_mode_win >= 2, ch_score,
                           torch.full_like(ch_score, -1.0))
    ch = int(ch_score.argmax())
    mdice = dice_bc[:, ch].cpu().numpy()
    top = torch.topk(ch_score.clamp_min(-1), min(5, ch_score.numel()))
    print(f"[phase0] channel-persistence top-5 (ch:score): "
          f"{[(int(i), round(float(v), 3)) for v, i in zip(*top)]}")
    print(f"[phase0] plotting channel {ch} (n_mode_windows={int(n_mode_win[ch])}); "
          f"per-window maskdice {np.round(mdice, 3).tolist()}")

    # save tensors so re-plots don't need another model run
    torch.save({"mu": mu.cpu(), "xin": xin.cpu(), "tgt": tgt.cpu(),
                "pmask": pmask.cpu(), "idxs": idxs, "ch": ch, "k": k},
               out_dir / f"{shot}_{modality}_tensors.pt")

    # show mode-bearing windows first (skip the trivial empty ones)
    order = list(np.argsort(-th[:, ch].sum(dim=(1, 2)).cpu().numpy()))
    rows = order[: min(4, len(order))]
    fig, axes = plt.subplots(len(rows), 4, figsize=(15, 3 * len(rows)))
    if len(rows) == 1:
        axes = axes[None]
    # stretch the color scale to the mode range (log-mag is mostly low +
    # sparse bright modes → a full min/max scale renders ~black)
    tsel = tgt[rows, ch].cpu().numpy()
    vlo, vhi = np.percentile(tsel, [55, 99.7])
    for r, w in enumerate(rows):
        panels = [
            (tgt[w, ch], f"GT target (w{idxs[w]})", "magma", vlo, vhi),
            (fused[w, ch], f"persistence forecast  mDice={mdice[w]:.2f}", "magma", vlo, vhi),
            (th[w, ch], "GT modes (mask)", "gray", 0, 1),
            (ph[w, ch], "forecast modes (from input)", "gray", 0, 1),
        ]
        for c, (img, title, cmap, lo, hi) in enumerate(panels):
            a = axes[r, c]
            a.imshow(img.cpu().numpy(), aspect="auto", origin="lower",
                     cmap=cmap, vmin=lo, vmax=hi)
            if r == 0:
                a.set_title(title, fontsize=9)
            a.set_xticks([]); a.set_yticks([])
    mode_win = mdice[[w for w in rows if int(n_mode_win[ch]) and th[w, ch].sum() > 3]]
    mean_str = f"{mode_win.mean():.3f}" if len(mode_win) else "n/a"
    fig.suptitle(
        f"Persistence-conditioned {modality.upper()} forecast — shot {shot}, "
        f"channel {ch} (mode-window maskdice {mean_str}, NO GT used)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_png = out_dir / f"{shot}_{modality}_persistence_forecast.png"
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    print(f"[phase0] wrote {out_png}")


if __name__ == "__main__":
    main()
