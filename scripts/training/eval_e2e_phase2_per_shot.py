"""Stage-1 evaluation — Phase 2.1: per-shot summary plots.

Consumes the CSV.gz tables from Phase 1 and the aggregate-scatter plots
from Phase 2.0, plus a checkpoint, and produces a 2×2 summary plot for
every (shot, modality) pair listed in ``top_bottom_shots.csv.gz``.

Per-shot 2×2 grid (plan §5):
    TL: per-window MAE time series for this shot       (from CSV)
    TR: GT-vs-pred plot of the BEST window of this shot (from re-inference)
    BL: GT-vs-pred plot of the WORST window of this shot (from re-inference)
    BR: histogram of per-window MAE for this shot       (from CSV)

Per-modality rendering of the TR/BL panels:
    slow_ts:     line plot, ~4 highest-variance channels (overlaid GT/pred)
    fast_ts:     8 channels in a 2×4 small-multiples grid
    spectrogram: GT/pred/|diff| stacked heatmaps for one representative channel
    video:       middle frame, GT vs pred vs |diff|

Quality bar (§5):
    - GT solid black, prediction dashed tab:blue.
    - Honest axes (physical units in labels).
    - Self-documenting titles (shot_id, modality, split, MAE value,
      window_idx).
    - |GT − pred| panel where practical (spectrogram and video).
    - No rainbow colormaps.

Single-GPU execution (rank-0 style): re-inference is cheap because the
selected shots are few (~10 per modality × 12 modalities ≈ 120 shots after
dedup), and each shot has ~1000 windows that fit comfortably at
batch_size=128. No DDP for this phase.

Run::

    pixi run python scripts/training/eval_e2e_stage1_phase2_per_shot.py \\
        --output_dir eval_runs/stage1_phase1_e2e_stage1_best_4609988 \\
        --checkpoint /lustre/orion/fus187/proj-shared/models/e2e_stage1/e2e_stage1_best.pt \\
        --data_dir   /lustre/orion/fus187/proj-shared/foundation_model \\
        --stats_path /lustre/orion/fus187/proj-shared/foundation_model_meta/preprocessing_stats.pt

Plots land in ``<output_dir>/plots/<split>/<modality>/<shot_id>_summary.png``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from tokamak_foundation_model.data.data_loader import collate_fn
from tokamak_foundation_model.data.multi_file_dataset import (
    TokamakMultiFileDataset,
)
from tokamak_foundation_model.e2e.lora import apply_lora_to_backbone
from tokamak_foundation_model.e2e.model import (
    ActuatorConfig,
    DiagnosticConfig,
    E2EFoundationModel,
)

# Re-use Phase-0 audit-approved helpers from the legacy eval script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_e2e import (  # type: ignore[import]  # noqa: E402
    _clean_and_mask,
    _ts_mask,
    _video_loss_gate,
    _video_standardize_per_bc,
    copy_baseline_for_modality,
    detect_stage_K,
    forward_one_batch,
    load_checkpoint_with_refine_tolerance,
    make_rollout_if_needed,
    rollout_forward_one_batch,
)
from eval_e2e_phase1 import (  # type: ignore[import]  # noqa: E402
    _align_shapes,
    parse_shot_id,
)

logger = logging.getLogger("eval_stage1_phase2_per_shot")


# ─────────────────────────────────────────────────────────────────────
# Style conventions (§5 quality bar — applied globally)
# ─────────────────────────────────────────────────────────────────────

_GT_COLOR = "black"
_GT_LW = 1.4
_PRED_COLOR = "tab:blue"
_PRED_LS = "--"
_PRED_LW = 1.2
_DIFF_CMAP = "magma"
_HEAT_CMAP = "viridis"


# ─────────────────────────────────────────────────────────────────────
# Per-shot re-inference (rank-0 single-GPU)
# ─────────────────────────────────────────────────────────────────────


def _build_dataset_for_shot(
    file_path: Path,
    diag_names: List[str],
    act_names: List[str],
    args: argparse.Namespace,
    stats: dict,
    K: int,
) -> TokamakMultiFileDataset:
    """One-file dataset emitting every 50 ms window of a single shot,
    with prediction horizon spanning K rollout steps."""
    return TokamakMultiFileDataset(
        [file_path],
        chunk_duration_s=args.chunk_duration_s,
        prediction_mode=True,
        prediction_horizon_s=K * args.chunk_duration_s,
        step_size_s=args.step_size_s,
        warmup_s=args.warmup_s,
        preprocessing_stats=stats,
        input_signals=diag_names,
        target_signals=diag_names + act_names,
        lengths_cache_path=None,    # short shot, cache not worth the I/O
    )


@torch.no_grad()
def collect_best_worst_windows_for_shot(
    model: E2EFoundationModel,
    file_path: Path,
    device: torch.device,
    args: argparse.Namespace,
    stats: dict,
    K: int,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Re-run K-step rollout inference on every window of a single shot,
    return the best- and worst-MAE window's final-step (k=K) tensors per
    modality.

    The "best/worst" ranking uses the k=K (final rollout step) MAE,
    which is the most demanding view of the model. For Stage 1 (K=1)
    this collapses to single-step prediction MAE, byte-identical to
    the pre-unification behaviour.

    Returns
    -------
    dict
        ``{modality: {'best_pred','best_target','best_window_idx','best_mae',
                       'worst_pred','worst_target','worst_window_idx','worst_mae',
                       'kind'}}``.
        Tensors are CPU-resident, shape ``(1, *modality_shape)``.
    """
    diag_names = [c.name for c in model.diagnostics]
    act_names = [c.name for c in model.actuators]
    rollout = make_rollout_if_needed(model, K, args.chunk_duration_s)

    ds = _build_dataset_for_shot(file_path, diag_names, act_names, args, stats, K)
    if len(ds) == 0:
        logger.warning(f"shot {file_path.name}: empty dataset")
        return {}

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=False,
    )

    # State per modality:
    #   - running best+worst (mae, window_idx, pred, target)
    #   - fallback: first window seen, used for plotting when no window
    #     has any GT data (so the modality still produces a pred-only
    #     summary instead of being silently skipped).
    state: Dict[str, Dict[str, object]] = {
        cfg.name: {
            "kind": cfg.kind,
            "best_mae": float("inf"),
            "worst_mae": float("-inf"),
            "best_window_idx": -1, "worst_window_idx": -1,
            "best_pred": None, "best_target": None,
            "worst_pred": None, "worst_target": None,
            "fallback_pred": None, "fallback_target": None,
            "fallback_window_idx": -1,
            "has_gt": False,
        }
        for cfg in model.diagnostics
    }

    global_window_idx = 0
    for batch in loader:
        predictions_per_k, diag_initial, targets_per_k, masks_per_k = (
            rollout_forward_one_batch(
                model, rollout, batch, device, K, args.chunk_duration_s
            )
        )
        # Render at the final rollout step (k=K-1, 0-indexed).
        predictions = predictions_per_k[K - 1]
        targets = targets_per_k[K - 1]
        masks = masks_per_k[K - 1]
        diag_inputs = diag_initial
        bs = next(iter(diag_inputs.values())).shape[0]

        for cfg in model.diagnostics:
            n = cfg.name
            pred = predictions[n]
            tgt = targets[n]
            mask = masks[n]
            # Align shapes (spectrogram trunc_t=96 vs raw target=98).
            if mask is None:
                pad_mask = torch.ones_like(tgt)
                pred_a, tgt_a, pad_mask_a = _align_shapes(pred, tgt, pad_mask)
                mask_a = None
            else:
                pred_a, tgt_a, mask_a = _align_shapes(pred, tgt, mask)

            cleaned_pred, mask_p = _clean_and_mask(pred_a, None)
            cleaned_tgt, mask_t = _clean_and_mask(tgt_a, mask_a)
            joint = mask_p * mask_t
            flat = list(range(1, pred_a.ndim))
            denom = joint.sum(dim=flat).clamp_min(1.0)
            per_sample_mae = (
                (cleaned_pred - cleaned_tgt).abs() * joint
            ).sum(dim=flat) / denom

            for j in range(bs):
                w = global_window_idx + j
                s = state[n]
                # Always seed a fallback from the first window of this
                # modality, so a shot with no GT for this modality still
                # gets one representative window for the pred-only plot.
                # The fallback target is the (possibly NaN) raw target —
                # caller's renderer is NaN-aware and will blank the GT
                # panel when there's nothing valid in it.
                if s["fallback_pred"] is None:
                    s["fallback_pred"] = pred_a[j:j+1].detach().cpu()
                    s["fallback_target"] = tgt_a[j:j+1].detach().cpu()
                    s["fallback_window_idx"] = w
                # Best/worst tracking requires at least some GT support.
                if joint[j].sum().item() < 1.0:
                    continue
                s["has_gt"] = True
                m = float(per_sample_mae[j].item())
                if m < s["best_mae"]:
                    s["best_mae"] = m
                    s["best_window_idx"] = w
                    s["best_pred"] = cleaned_pred[j:j+1].detach().cpu()
                    s["best_target"] = cleaned_tgt[j:j+1].detach().cpu()
                if m > s["worst_mae"]:
                    s["worst_mae"] = m
                    s["worst_window_idx"] = w
                    s["worst_pred"] = cleaned_pred[j:j+1].detach().cpu()
                    s["worst_target"] = cleaned_tgt[j:j+1].detach().cpu()
        global_window_idx += bs

    # Post-processing: modalities with no GT-bearing windows still need
    # something to render. Promote the fallback to both best and worst
    # slots so the plot driver can treat them uniformly.
    for n, s in state.items():
        if not s["has_gt"] and s["fallback_pred"] is not None:
            s["best_pred"] = s["fallback_pred"]
            s["best_target"] = s["fallback_target"]
            s["best_window_idx"] = s["fallback_window_idx"]
            s["best_mae"] = float("nan")
            s["worst_pred"] = s["fallback_pred"]
            s["worst_target"] = s["fallback_target"]
            s["worst_window_idx"] = s["fallback_window_idx"]
            s["worst_mae"] = float("nan")

    return state


# ─────────────────────────────────────────────────────────────────────
# Per-modality window-render helpers (TR / BL panels)
# ─────────────────────────────────────────────────────────────────────


def _pick_top_variance_channels(target: torch.Tensor, k: int) -> List[int]:
    """For slow_ts panels: pick the k highest-variance channels.

    NaN-aware: falls back to ``np.nanvar`` so a target with missing GT
    on some channels still picks the most-informative channels among
    those with valid data. Channels with all-NaN values get treated as
    zero-variance and only chosen if nothing else is available."""
    # target: (1, n_ch, samples)
    t = target[0].cpu().numpy()
    n_ch = t.shape[0]
    if n_ch <= k:
        return list(range(n_ch))
    var = np.nanvar(t, axis=tuple(range(1, t.ndim)))
    var = np.where(np.isnan(var), 0.0, var)
    # Ignore channels with zero variance (would yield uninformative panels).
    nz = np.nonzero(var)[0]
    if len(nz) == 0:
        return list(range(min(k, n_ch)))
    order = np.argsort(-var[nz])
    return nz[order[:k]].tolist()


def _render_ts_window(
    ax: plt.Axes,
    pred: torch.Tensor,
    target: torch.Tensor,
    kind: str,
    n_channels_to_show: int,
    chunk_duration_s: float,
) -> None:
    """Line plot for slow_ts / fast_ts: GT solid + pred dashed for
    top-variance channels.

    The legend is intentionally minimal (2 entries, GT vs model) since
    enumerating ~4–8 channels per panel would clutter the figure. Channels
    share the GT/model color convention; the panel as a whole is a
    "channel ensemble" view, not a per-channel comparison."""
    # pred / target shape: (1, C, T_samples)
    p = pred[0].cpu().numpy()
    t = target[0].cpu().numpy()
    n_ch, t_samples = p.shape
    # NaN-aware: when GT has no valid samples for this channel we still
    # plot the prediction line. matplotlib already skips NaN gaps in a
    # line plot, so simply passing the array through is enough.
    has_any_gt = bool(np.isfinite(t).any())
    if has_any_gt:
        channels = _pick_top_variance_channels(target, n_channels_to_show)
    else:
        # Pick by prediction variance instead — no GT to score against.
        pred_var = p.var(axis=tuple(range(1, p.ndim)))
        nz = np.nonzero(pred_var)[0]
        if len(nz) >= n_channels_to_show:
            order = np.argsort(-pred_var[nz])
            channels = nz[order[:n_channels_to_show]].tolist()
        else:
            channels = list(range(min(n_channels_to_show, n_ch)))
    # Time axis in milliseconds (within the 50 ms window).
    time_ms = np.linspace(0, chunk_duration_s * 1000.0, t_samples, endpoint=False)
    for i, c in enumerate(channels):
        # Only attach legend labels to the first channel so the legend
        # has 2 entries (GT, model) not 2N.
        gt_kw = {"label": "GT"} if i == 0 and has_any_gt else {}
        pr_kw = {"label": "model"} if i == 0 else {}
        if has_any_gt:
            ax.plot(time_ms, t[c], color=_GT_COLOR, linewidth=_GT_LW, alpha=0.85,
                    **gt_kw)
        ax.plot(time_ms, p[c], color=_PRED_COLOR, linestyle=_PRED_LS,
                linewidth=_PRED_LW, alpha=0.85, **pr_kw)
    ax.set_xlabel("time within window (ms)", fontsize=8)
    ax.set_ylabel("standardised signal", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    legend_title = (
        f"{len(channels)} top-variance channels" if has_any_gt
        else f"{len(channels)} channels (no GT — pred only)"
    )
    ax.legend(loc="upper right", fontsize=7, framealpha=0.85,
              title=legend_title, title_fontsize=7)


def _render_spectrogram_window(
    ax: plt.Axes,
    pred: torch.Tensor,
    target: torch.Tensor,
    n_channels_to_show: int,
) -> None:
    """Two-row imshow for spectrogram modalities: GT (top) + pred (bottom),
    averaged across the representative channel subset. Shared colorbar so
    intensity is comparable across rows."""
    # pred / target shape: (1, C, freq, time)
    p_t = target[0].cpu().numpy()
    p_p = pred[0].cpu().numpy()
    has_any_gt = bool(np.isfinite(p_t).any())
    if has_any_gt:
        channels = _pick_top_variance_channels(target, n_channels_to_show)
    else:
        # No GT — pick by prediction variance.
        pred_var = p_p.var(axis=tuple(range(1, p_p.ndim)))
        nz = np.nonzero(pred_var)[0]
        if len(nz) >= n_channels_to_show:
            order = np.argsort(-pred_var[nz])
            channels = nz[order[:n_channels_to_show]].tolist()
        else:
            channels = list(range(min(n_channels_to_show, p_p.shape[0])))
    if not channels:
        ax.set_title("no plottable channels", fontsize=8)
        return
    p_t_m = p_t[channels].mean(axis=0)   # (freq, time) — NaN if no GT
    p_p_m = p_p[channels].mean(axis=0)
    # NaN-aware vmin/vmax: when GT is missing, anchor to prediction
    # range so the model panel renders meaningfully; the GT imshow then
    # gets a NaN array, which matplotlib draws as blank (bg-coloured)
    # via the default cmap.set_bad behaviour.
    if has_any_gt:
        vmin = float(min(np.nanmin(p_t_m), np.nanmin(p_p_m)))
        vmax = float(max(np.nanmax(p_t_m), np.nanmax(p_p_m)))
    else:
        vmin = float(np.nanmin(p_p_m))
        vmax = float(np.nanmax(p_p_m))
    # Use a divider to stack the two heatmaps in this single axes' bbox
    # and attach a single shared colorbar so the reader knows the
    # intensity scale is the same for both rows.
    div = make_axes_locatable(ax)
    ax_pred = div.append_axes("bottom", size="100%", pad=0.05, sharex=ax)
    cax = div.append_axes("right", size="3%", pad=0.05)
    im_gt = ax.imshow(p_t_m, aspect="auto", origin="lower",
                      cmap=_HEAT_CMAP, vmin=vmin, vmax=vmax)
    ax_pred.imshow(p_p_m, aspect="auto", origin="lower",
                   cmap=_HEAT_CMAP, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im_gt, cax=cax)
    cbar.set_label("spectral intensity (standardised)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    ax.set_ylabel("freq bin", fontsize=8)
    ax_pred.set_ylabel("freq bin", fontsize=8)
    ax_pred.set_xlabel("time frame", fontsize=8)
    ax.set_xticks([])
    ax.tick_params(labelsize=7)
    ax_pred.tick_params(labelsize=7)
    ax.text(0.01, 0.96, "GT", transform=ax.transAxes, fontsize=8,
            color="white", va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.7))
    ax_pred.text(0.01, 0.96, "model", transform=ax_pred.transAxes, fontsize=8,
                 color="white", va="top",
                 bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.7))


def _render_video_window(
    ax: plt.Axes,
    pred: torch.Tensor,
    target: torch.Tensor,
) -> None:
    """For video modalities: show middle frame of GT, pred, |diff| as a
    horizontal triptych. The host ``ax`` is replaced by a 1×3 sub-gridspec
    inside its bounding box so the three panels share the cell properly
    even when ``ax`` lives in a constrained outer gridspec — append_axes
    siblings would otherwise overflow the parent cell and end up overlapping
    other subplots (only the host's tiny GT thumbnail stayed visible)."""
    # pred / target shape: (1, C, T_frames, H, W). Show middle T frame,
    # collapse the DISPLAYED channels by mean.
    # Copy out of the source tensor so the channel-1 flip below doesn't
    # mutate caller-owned memory.
    p = pred[0].cpu().numpy().copy()
    t = target[0].cpu().numpy().copy()
    n_model_ch = t.shape[0]
    if n_model_ch >= 5:
        # NEW 7-channel model (model ch i == raw ch i): display only the
        # two divertor views — model ch2 (lower = LODIV_240RM1:PERP) and
        # ch4 (upper = UPDIV_0RP1:PERP). NO flip. Other raw channels are
        # mostly-NaN metadata and must not pollute the cross-channel mean.
        disp_chs = [2, 4]
    else:
        # OLD (<= 2 channel) model — unchanged. Channel 1 of tangtv is
        # rotated 180° vs channel 0 (not just a horizontal mirror), so flip
        # BOTH H and W before the cross-channel mean so the average doesn't
        # cancel structure. Matches the Phase 3.1 mp4 fix — see
        # project-tangtv-channel1-flip memory.
        if n_model_ch > 1:
            t[1] = t[1, :, ::-1, ::-1]
            p[1] = p[1, :, ::-1, ::-1]
        disp_chs = list(range(n_model_ch))
    t_idx = p.shape[1] // 2
    gt = t[disp_chs, t_idx].mean(axis=0)
    pr = p[disp_chs, t_idx].mean(axis=0)
    diff = np.abs(gt - pr)

    # Take over the host axes' bounding box with a 1×3 sub-gridspec.
    fig = ax.figure
    bbox = ax.get_subplotspec()
    ax.set_visible(False)
    sub_gs = bbox.subgridspec(1, 3, wspace=0.05)
    ax_gt = fig.add_subplot(sub_gs[0, 0])
    ax_pred = fig.add_subplot(sub_gs[0, 1], sharey=ax_gt)
    ax_diff = fig.add_subplot(sub_gs[0, 2], sharey=ax_gt)

    # Anchor colormap to GT when GT is present (so model outliers don't
    # blow out the range, matches the spectrogram fix). When GT is all-
    # NaN, anchor to prediction range; the GT and diff panels render
    # blank because NaN propagates through imshow's cmap.
    has_any_gt = bool(np.isfinite(gt).any())
    if has_any_gt:
        vmin = float(np.nanmin(gt))
        vmax = float(np.nanmax(gt))
    else:
        vmin = float(np.nanmin(pr))
        vmax = float(np.nanmax(pr))
    ax_gt.imshow(gt, cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")
    ax_pred.imshow(pr, cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")
    im_diff = ax_diff.imshow(diff, cmap=_DIFF_CMAP, aspect="equal")
    for sub_ax, label in [(ax_gt, "GT"), (ax_pred, "model"), (ax_diff, "|GT − model|")]:
        sub_ax.set_xticks([])
        sub_ax.set_yticks([])
        sub_ax.text(0.02, 0.96, label, transform=sub_ax.transAxes,
                    fontsize=8, color="white", va="top",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.7))
    # Colorbar attached to the diff panel via axes_grid1 (stays inside the cell).
    div = make_axes_locatable(ax_diff)
    cax = div.append_axes("bottom", size="6%", pad=0.05)
    cbar = plt.colorbar(im_diff, cax=cax, orientation="horizontal")
    cbar.set_label("|GT − model|", fontsize=7)
    cbar.ax.tick_params(labelsize=6)


def render_window_panel(
    ax: plt.Axes,
    pred: torch.Tensor,
    target: torch.Tensor,
    kind: str,
    chunk_duration_s: float,
    n_ts_channels: int = 4,
    n_spectro_channels: int = 4,
) -> None:
    """Dispatch to the right per-modality renderer for the TR/BL panels."""
    if pred is None or target is None:
        ax.text(0.5, 0.5, "no valid window found",
                transform=ax.transAxes, ha="center", va="center", fontsize=9)
        return
    if kind in ("slow_ts", "fast_ts"):
        # slow_ts has many channels (e.g., MSE has 69) — limit to 4.
        # fast_ts has 8 channels — show all.
        k = 8 if kind == "fast_ts" else n_ts_channels
        _render_ts_window(ax, pred, target, kind, k, chunk_duration_s)
    elif kind == "spectrogram":
        _render_spectrogram_window(ax, pred, target, n_spectro_channels)
    elif kind == "video":
        _render_video_window(ax, pred, target)
    else:
        ax.text(0.5, 0.5, f"unknown modality kind: {kind}",
                transform=ax.transAxes, ha="center", va="center")


# ─────────────────────────────────────────────────────────────────────
# Per-shot 2×2 summary plot
# ─────────────────────────────────────────────────────────────────────


def plot_per_shot_summary(
    per_window_subset: pd.DataFrame,
    shot_id: int,
    modality: str,
    kind: str,
    split: str,
    shot_state: Optional[Dict[str, object]],
    out_path: Path,
    chunk_duration_s: float,
) -> None:
    """Render the 2×2 summary plot for one (shot, modality) pair.

    Layout:
        TL: MAE-vs-window time series (from CSV)
        TR: best-MAE window GT/pred (from re-inference)
        BL: worst-MAE window GT/pred (from re-inference)
        BR: MAE histogram (from CSV)
    """
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.22)
    ax_tl = fig.add_subplot(gs[0, 0])
    ax_tr = fig.add_subplot(gs[0, 1])
    ax_bl = fig.add_subplot(gs[1, 0])
    ax_br = fig.add_subplot(gs[1, 1])

    # ── TL: MAE-vs-window time series ────────────────────────────────
    has_pw_data = not per_window_subset.empty
    if has_pw_data:
        pw = per_window_subset.sort_values("window_idx")
        t_s = pw["window_t_s"].to_numpy()
        mae = pw["mae"].to_numpy()
        copy_mae = pw["copy_mae"].to_numpy()
        ax_tl.plot(t_s, mae, color=_PRED_COLOR, linewidth=1.0,
                   label="model")
        ax_tl.plot(t_s, copy_mae, color=_GT_COLOR, linewidth=1.0,
                   alpha=0.6, label="copy baseline")
        ax_tl.set_xlabel("window-start time within shot (s)", fontsize=9)
        ax_tl.set_ylabel("MAE per window", fontsize=9)
        ax_tl.set_title("TL — per-window MAE across this shot", fontsize=10)
        ax_tl.legend(fontsize=8, loc="best")
        ax_tl.grid(True, alpha=0.3, linewidth=0.5)
        ax_tl.tick_params(labelsize=7)
    else:
        ax_tl.text(0.5, 0.5, "no per-window data (no valid GT)",
                   transform=ax_tl.transAxes, ha="center", va="center",
                   fontsize=10)
        ax_tl.set_title("TL — per-window MAE across this shot", fontsize=10)

    # ── TR + BL: best / worst window GT vs pred ──────────────────────
    # has_gt=False means the modality has no GT for this shot; the
    # fallback (representative) window was promoted into the best/worst
    # slots. Title reflects that — no MAE to report.
    if shot_state is None:
        ax_tr.text(0.5, 0.5, "no re-inference data (--checkpoint not provided)",
                   transform=ax_tr.transAxes, ha="center", va="center", fontsize=9)
        ax_bl.text(0.5, 0.5, "no re-inference data (--checkpoint not provided)",
                   transform=ax_bl.transAxes, ha="center", va="center", fontsize=9)
    else:
        has_gt = bool(shot_state.get("has_gt"))
        render_window_panel(
            ax_tr,
            pred=shot_state.get("best_pred"),
            target=shot_state.get("best_target"),
            kind=kind, chunk_duration_s=chunk_duration_s,
        )
        if has_gt:
            ax_tr.set_title(
                f"TR — best window: idx={shot_state.get('best_window_idx')}, "
                f"MAE={shot_state.get('best_mae'):.4f}",
                fontsize=10,
            )
        else:
            ax_tr.set_title(
                f"TR — representative window (no GT): "
                f"idx={shot_state.get('best_window_idx')}",
                fontsize=10,
            )
        render_window_panel(
            ax_bl,
            pred=shot_state.get("worst_pred"),
            target=shot_state.get("worst_target"),
            kind=kind, chunk_duration_s=chunk_duration_s,
        )
        if has_gt:
            ax_bl.set_title(
                f"BL — worst window: idx={shot_state.get('worst_window_idx')}, "
                f"MAE={shot_state.get('worst_mae'):.4f}",
                fontsize=10,
            )
        else:
            ax_bl.set_title(
                f"BL — representative window (no GT): "
                f"idx={shot_state.get('worst_window_idx')}",
                fontsize=10,
            )

    # ── BR: MAE histogram ────────────────────────────────────────────
    if has_pw_data:
        finite_mae = mae[np.isfinite(mae)]
        if finite_mae.size > 0:
            ax_br.hist(finite_mae, bins=40, color=_PRED_COLOR, alpha=0.7,
                       label=f"model (n={finite_mae.size})")
            finite_copy = copy_mae[np.isfinite(copy_mae)]
            if finite_copy.size > 0:
                ax_br.hist(finite_copy, bins=40, color=_GT_COLOR, alpha=0.4,
                           label=f"copy (n={finite_copy.size})")
            ax_br.set_xlabel("per-window MAE", fontsize=9)
            ax_br.set_ylabel("window count", fontsize=9)
            ax_br.set_title("BR — per-window MAE distribution", fontsize=10)
            ax_br.legend(fontsize=8, loc="best")
            ax_br.tick_params(labelsize=7)
        else:
            ax_br.set_title("BR — no valid windows", fontsize=10)
    else:
        ax_br.text(0.5, 0.5, "no per-window data",
                   transform=ax_br.transAxes, ha="center", va="center",
                   fontsize=10)
        ax_br.set_title("BR — per-window MAE distribution", fontsize=10)

    # Figure-wide title — self-documenting per §5.
    if has_pw_data:
        mae_mean = float(pw["mae"].mean())
        copy_mae_mean = float(pw["copy_mae"].mean())
        ratio = mae_mean / copy_mae_mean if copy_mae_mean > 0 else float("nan")
        suptitle = (
            f"shot {shot_id} — {modality} ({kind}) — split: {split}   |   "
            f"n_windows={len(pw)}  mae_mean={mae_mean:.4f}  "
            f"copy_mae_mean={copy_mae_mean:.4f}  ratio={ratio:.3f}"
        )
    else:
        suptitle = (
            f"shot {shot_id} — {modality} ({kind}) — split: {split}   |   "
            f"no valid GT for this shot"
        )
    fig.suptitle(suptitle, fontsize=11, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────


def _coverage_aware_shot_order(
    sel_by_shot: Dict[int, List[Tuple[str, str, str]]],
    cap: int = 0,
) -> List[int]:
    """Order shots so a small ``cap`` still produces a representative
    sample across modality kinds.

    Without this, ``sorted(sel_by_shot.keys())[:cap]`` slices by the
    lowest shot-ids and can leave whole modality kinds unrepresented
    (the symptom that originally surfaced: cap=3 → only slow_ts shots).

    Algorithm:
      1. Greedy set-cover by **kind** (slow_ts / fast_ts /
         spectrogram / video). Each round picks the shot that
         covers the most still-uncovered kinds. This guarantees
         that cap ≥ 4 includes at least one shot per kind (if
         available in the selection at all).
      2. Then prefer shots with the most top/bottom selections
         (i.e., shots that are flagged across many modalities
         — they make a single 'shot summary' figure carry the
         most modality-breadth per re-inference pass).
      3. Tie-break on numerical shot_id so the order is
         deterministic.

    If ``cap`` is 0 or larger than ``len(sel_by_shot)``, the full
    coverage-aware ordering is returned (no truncation).
    """
    if not sel_by_shot:
        return []

    # Precompute (kinds_set, selection_count) per shot.
    info = {
        s: (frozenset(k for _, _, k in sels), len(sels))
        for s, sels in sel_by_shot.items()
    }

    selected: List[int] = []
    remaining: set = set(info.keys())
    covered_kinds: set = set()

    # Phase A — set-cover by kind.
    all_kinds: set = set().union(*(ks for ks, _ in info.values()))
    while remaining and covered_kinds != all_kinds:
        def score(s: int) -> Tuple[int, int, int]:
            ks, cnt = info[s]
            # First: cover as many uncovered kinds as possible.
            # Second: prefer shots with more total selections.
            # Third: deterministic — prefer smaller shot_id (negate).
            return (
                len(ks - covered_kinds),
                cnt,
                -s,
            )
        nxt = max(remaining, key=score)
        if not (info[nxt][0] - covered_kinds):
            break        # no shot left contributes a new kind
        selected.append(nxt)
        remaining.discard(nxt)
        covered_kinds |= info[nxt][0]

    # Phase B — fill the remainder by selection count, then shot_id.
    leftover = sorted(
        remaining,
        key=lambda s: (-info[s][1], s),
    )
    selected.extend(leftover)

    if cap and cap > 0:
        return selected[:cap]
    return selected


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_dir", type=Path, required=True,
                   help="Existing eval output (Phase 1).")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data_dir", type=Path, required=True)
    p.add_argument("--stats_path", type=Path, required=True)
    p.add_argument("--plots_subdir", type=str, default="plots")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--chunk_duration_s", type=float, default=0.05)
    p.add_argument("--step_size_s", type=float, default=0.01)
    p.add_argument("--warmup_s", type=float, default=1.0)
    p.add_argument(
        "--max_shots_to_plot", type=int, default=0,
        help="Cap unique shots to plot. 0 = all selected by Phase 1's "
             "top/bottom-N. Small int for smokes.",
    )
    p.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--K", type=int, default=0,
        help="Rollout horizon. 0 (default) autodetects from checkpoint "
             "(K=1 for Stage 1, K=K_max for Stage 2). Plots render at k=K.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    device = torch.device(args.device)
    plots_root = args.output_dir / args.plots_subdir

    # ── Load CSV tables produced by Phase 1 ──────────────────────────
    # top_bottom_shots.csv.gz is no longer consulted as a filter: Phase 2
    # now iterates EVERY shot × EVERY model.diagnostics modality so the
    # eval is exhaustive. The cap-and-coverage path historically used
    # top_bottom_shots was hiding modalities whose top/bottom shots
    # didn't overlap with the picked-shot pool — bes and co2 were the
    # symptom that surfaced this. Only per_window_metrics.csv.gz is
    # required now.
    pw_path = args.output_dir / "per_window_metrics.csv.gz"
    if not pw_path.exists():
        raise SystemExit(f"required input not found: {pw_path}")
    per_window = pd.read_csv(pw_path, compression="gzip")
    logger.info(f"Loaded {len(per_window):,} per-window rows")

    # ── Load model from checkpoint ───────────────────────────────────
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    diagnostics = [DiagnosticConfig(**d) for d in ckpt["diagnostics"]]
    actuators = [ActuatorConfig(**a) for a in ckpt["actuators"]]
    ck_args = ckpt["args"]
    model = E2EFoundationModel(
        diagnostics=diagnostics, actuators=actuators,
        d_model=ck_args["d_model"], n_heads=ck_args["n_heads"],
        n_layers=ck_args["n_layers"], dropout=0.0,
    )
    state_dict = ckpt["model_state_dict"]
    if any(".lora_" in k for k in state_dict):
        rank_l = int(ck_args.get("lora_rank", 16))
        alpha_l = float(ck_args.get("lora_alpha", 16.0))
        apply_lora_to_backbone(model.backbone, rank=rank_l, alpha=alpha_l)
        logger.info(f"LoRA detected: rank={rank_l} alpha={alpha_l}")
    load_checkpoint_with_refine_tolerance(model, state_dict)
    model.eval().to(device)
    stats = torch.load(args.stats_path, weights_only=False)

    K = args.K if args.K > 0 else detect_stage_K(ckpt)
    logger.info(
        f"Eval horizon K={K} ({'autodetected' if args.K == 0 else 'override'})"
    )

    # ── Build (shot_id → split) map: every shot in per_window_metrics ─
    # Each (split, shot_id) is unique (deterministic train/val split), so
    # one row per shot is enough to recover the split.
    shot_split: Dict[int, str] = (
        per_window.drop_duplicates("shot_id")[["shot_id", "split"]]
                  .set_index("shot_id")["split"].to_dict()
    )
    all_shots = sorted(shot_split.keys())
    if args.max_shots_to_plot and args.max_shots_to_plot > 0:
        all_shots = all_shots[: args.max_shots_to_plot]
    logger.info(
        f"Plotting per-shot summaries for {len(all_shots)} shots "
        f"× {len(model.diagnostics)} modalities = "
        f"{len(all_shots) * len(model.diagnostics)} target plots"
    )

    # ── Per-shot loop: re-infer, then plot ALL model.diagnostics. ────
    # No top_bottom selection — every (shot, modality) gets a plot.
    # Modalities with no GT for this shot still render the prediction
    # (NaN-aware path in the per-modality renderers).
    diag_iter = [(c.name, c.kind) for c in model.diagnostics]
    for i, shot_id in enumerate(all_shots, start=1):
        file_path = args.data_dir / f"{shot_id}_processed.h5"
        if not file_path.exists():
            logger.warning(f"shot {shot_id}: file missing at {file_path}")
            continue
        split = shot_split[shot_id]
        logger.info(f"({i}/{len(all_shots)}) shot {shot_id} ({split}): re-inference …")
        shot_states = collect_best_worst_windows_for_shot(
            model=model, file_path=file_path, device=device,
            args=args, stats=stats, K=K,
        )
        for modality, kind in diag_iter:
            pw_sub = per_window.query(
                "split == @split and modality == @modality and shot_id == @shot_id"
            )
            # pw_sub may be empty for a (shot, modality) where Phase 1 had
            # no valid joint-mask windows. plot_per_shot_summary handles
            # empty by blanking the TL/BR panels and only rendering the
            # representative window (TR/BL) from re-inference.
            out_path = (
                plots_root / split / modality / f"{shot_id}_summary.png"
            )
            plot_per_shot_summary(
                per_window_subset=pw_sub,
                shot_id=shot_id, modality=modality, kind=kind, split=split,
                shot_state=shot_states.get(modality),
                out_path=out_path,
                chunk_duration_s=args.chunk_duration_s,
            )
            logger.info(f"  → {out_path.relative_to(args.output_dir)}")

    logger.info("Phase 2.1 complete.")


if __name__ == "__main__":
    main()
