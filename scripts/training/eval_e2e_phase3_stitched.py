"""Stage-1 evaluation — Phase 3.0: stitched-window plots.

The paper-grade centrepiece (plan §2-Q3 / §5). For every selected
(shot, modality) pair from ``top_bottom_shots.csv.gz``, produce
**3 stitched-window plots** at 25 / 50 / 75 % of the shot's length,
each spanning 80 consecutive 50 ms windows (~4 s of shot wall-time).

Per-modality stitched layout (plan §5):

  slow_ts:     overlaid line plot, GT solid + model dashed, ~4
               highest-variance channels per shot. x-axis = seconds
               since shot start (derived from window_idx × 0.05 s,
               monotonic in time within a shot — see §10 Q1).
  fast_ts:     same layout but 8 channels (filterscopes) split into
               an 4×2 small-multiples grid so each channel is
               legible.
  spectrogram: 3-row stacked heatmap per channel
               (GT / model / |GT − model|), shared frequency axis.
               One PNG per channel in the representative subset
               (per §10 Q2; default 4 channels for ECE/BES, 4 for
               CO2). Filename includes the channel index.

Video (tangtv) is intentionally OUT OF SCOPE for this script — handled
by the sibling Phase 3.1 video / mp4 generator.

Single-GPU re-inference, same pattern as Phase 2.1: each shot's
dataset is iterated once, three 80-window segments' worth of
prediction tensors are stashed in memory, plots are produced, memory
is freed before the next shot.

Run::

    pixi run python scripts/training/eval_e2e_stage1_phase3_stitched.py \\
        --output_dir eval_runs/stage1_phase1_e2e_stage1_best_4609988 \\
        --checkpoint /lustre/orion/fus187/proj-shared/models/e2e_stage1/e2e_stage1_best.pt \\
        --data_dir   /lustre/orion/fus187/proj-shared/foundation_model \\
        --stats_path /lustre/orion/fus187/proj-shared/foundation_model_meta/preprocessing_stats.pt

Plots land in
``<output_dir>/plots/<split>/<modality>/<shot_id>_stitched_<seg>[_ch<n>].png``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
from matplotlib.lines import Line2D
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

# Re-use Phase 0 / Phase 1 / Phase 2.1 helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_e2e import (  # type: ignore[import]  # noqa: E402
    _clean_and_mask,
    _ts_mask,
    _video_loss_gate,
    _video_standardize_per_bc,
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
from eval_e2e_phase2_per_shot import (  # type: ignore[import]  # noqa: E402
    _coverage_aware_shot_order,
    _pick_top_variance_channels,
)

logger = logging.getLogger("eval_stage1_phase3_stitched")


# ─────────────────────────────────────────────────────────────────────
# Style conventions (§5 quality bar — mirrors Phase 2.1).
# ─────────────────────────────────────────────────────────────────────

_GT_COLOR = "black"
_GT_LW = 1.2
_PRED_COLOR = "tab:blue"
_PRED_LS = "--"
_PRED_LW = 1.0
_DIFF_CMAP = "magma"
_HEAT_CMAP = "viridis"

_CHUNK_DURATION_S = 0.05      # 50 ms; verified at §1 of the plan.
_STEP_SIZE_S = 0.01           # data loader spacing — windows step every
                              # 10 ms, so consecutive windows overlap by
                              # 80 % of their content. Stitched plots
                              # MUST subsample by stride = chunk/step
                              # to get non-overlapping predictions.
_STITCH_STRIDE = int(round(_CHUNK_DURATION_S / _STEP_SIZE_S))   # = 5
_DEFAULT_SEG_WINDOWS = 80     # plan §5: ~80 stride-stepped windows ≈ 4 s.
                              # Raw segment span = 80 × stride = 400 windows.
_SEG_FRACTIONS = (0.0, 0.33, 0.66)    # plan §10 Q7 (revised 2026-05-18):
                              # segment 0 now starts at the beginning of
                              # usable shot data instead of 25 % in, so
                              # early-shot dynamics (current ramp,
                              # breakdown, early L-mode) appear in the
                              # stitched view.
_WARMUP_S = 1.0               # default dataset warmup_s; matches the
                              # CLI default. Used to convert window_idx
                              # → absolute time-since-shot-start in plot
                              # labels (target at window i starts at
                              # t = warmup_s + i × step_size_s + chunk_duration_s).

# Channel-subset defaults per spectrogram modality (plan §5 / §10 Q2).
_SPECTRO_CHANNEL_BUDGET = {
    "ece": 4,
    "bes": 4,
    "co2": 4,    # CO2 has only 4 channels — all of them.
}


# ─────────────────────────────────────────────────────────────────────
# Segment selection
# ─────────────────────────────────────────────────────────────────────


def compute_segment_ranges(
    n_windows: int,
    seg_windows: int = _DEFAULT_SEG_WINDOWS,
    fractions: Tuple[float, ...] = _SEG_FRACTIONS,
    stride: int = _STITCH_STRIDE,
) -> List[Tuple[int, int, int, int]]:
    """Return ``(seg_idx, start, end, stride)`` ranges within the shot.

    Each segment uses ``seg_windows`` **subsampled** windows spaced
    ``stride`` apart so consecutive predictions are non-overlapping.
    Raw window range covered is ``start … start + seg_windows × stride``.

    If a segment's raw range would run past the end of the shot, the
    segment is clipped (fewer subsampled windows). If even the first
    subsampled window doesn't fit, the segment is dropped (loud
    warning).
    """
    out: List[Tuple[int, int, int, int]] = []
    for seg_idx, frac in enumerate(fractions):
        start = int(frac * n_windows)
        raw_end_wanted = start + seg_windows * stride
        end = min(raw_end_wanted, n_windows)
        # How many subsampled windows actually fit?
        n_subsampled = max(0, (end - start + stride - 1) // stride)
        if n_subsampled < 2:
            logger.warning(
                f"Skipping segment {seg_idx} (start={start}, stride={stride}, "
                f"only {n_subsampled} subsampled windows fit before "
                f"n_windows={n_windows})"
            )
            continue
        # Clip ``end`` to last subsampled window + 1 so the loop's
        # range(start, end, stride) yields exactly n_subsampled entries.
        end = start + n_subsampled * stride
        out.append((seg_idx, start, end, stride))
    return out


# ─────────────────────────────────────────────────────────────────────
# Per-shot re-inference for stitched segments
# ─────────────────────────────────────────────────────────────────────


@torch.no_grad()
def collect_stitched_segments_for_shot(
    model: E2EFoundationModel,
    file_path: Path,
    device: torch.device,
    args: argparse.Namespace,
    stats: dict,
    K: int,
) -> Dict[int, Dict[str, Dict[str, torch.Tensor]]]:
    """Re-infer one shot with K-step rollout, stash final-step (k=K)
    predictions for each of the 3 stitched segments.

    For Stage 1 (K=1) this is byte-identical to the pre-unification
    behaviour. For Stage 2 (K>1) the stored prediction at each window
    is the final-step rollout output (model predicting K*chunk_duration_s
    into the future).

    Returns
    -------
    dict
        ``{seg_idx: {modality_name: {"pred": (T_seg, ...), "target": (T_seg, ...),
                                     "window_idx_range": (start, end),
                                     "kind": kind}}}``
        Tensors are on CPU, time-axis first (concatenated across the
        segment's windows). Spectrogram tensors are pre-sliced to the
        representative channel subset so storage stays bounded.
    """
    diag_names = [c.name for c in model.diagnostics]
    act_names = [c.name for c in model.actuators]
    rollout = make_rollout_if_needed(model, K, args.chunk_duration_s)

    ds = TokamakMultiFileDataset(
        [file_path],
        chunk_duration_s=args.chunk_duration_s,
        prediction_mode=True,
        prediction_horizon_s=K * args.chunk_duration_s,
        step_size_s=args.step_size_s,
        warmup_s=args.warmup_s,
        preprocessing_stats=stats,
        input_signals=diag_names,
        target_signals=diag_names + act_names,
        lengths_cache_path=None,
    )
    n_windows = len(ds)
    if n_windows == 0:
        logger.warning(f"shot {file_path.name}: empty dataset")
        return {}
    seg_ranges = compute_segment_ranges(n_windows)
    if not seg_ranges:
        return {}
    # Quick lookup: window_idx → (seg_idx, position_within_segment).
    # Only stride-stepped windows are mapped — others are inferred but
    # discarded.
    window_to_seg: Dict[int, Tuple[int, int]] = {}
    for seg_idx, start, end, stride in seg_ranges:
        for pos, w in enumerate(range(start, end, stride)):
            window_to_seg[w] = (seg_idx, pos)

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers,
        drop_last=False, pin_memory=False,
    )

    # Build storage. For each (seg_idx, modality_name) we collect lists
    # indexed by position-within-segment.
    storage: Dict[int, Dict[str, Dict[str, List[torch.Tensor]]]] = {
        seg_idx: {n: {"pred": [None] * ((end - start) // stride),
                      "target": [None] * ((end - start) // stride),
                      "kind": next(c.kind for c in model.diagnostics if c.name == n),
                      "window_idx_range": (start, end, stride),
                      "channels_used": None}
                  for n in diag_names}
        for seg_idx, start, end, stride in seg_ranges
    }

    # Pre-pick spectrogram channel subsets at first encounter so all
    # three segments use the same channels per modality (consistent
    # comparison across segments for a given shot).
    chan_locked: Dict[str, List[int]] = {}

    global_idx = 0
    for batch in loader:
        predictions_per_k, diag_initial, targets_per_k, masks_per_k = (
            rollout_forward_one_batch(
                model, rollout, batch, device, K, args.chunk_duration_s
            )
        )
        predictions = predictions_per_k[K - 1]
        targets = targets_per_k[K - 1]
        masks = masks_per_k[K - 1]
        diag_inputs = diag_initial
        bs = next(iter(diag_inputs.values())).shape[0]
        for j in range(bs):
            w = global_idx + j
            if w not in window_to_seg:
                continue
            seg_idx, pos = window_to_seg[w]
            for cfg in model.diagnostics:
                n = cfg.name
                if cfg.kind == "video":
                    # Phase 3.1 handles video — skip storing here.
                    continue
                pred = predictions[n][j:j+1]
                tgt = targets[n][j:j+1]
                # Spectrograms: align trunc_t (96) vs raw target (98).
                if cfg.kind == "spectrogram":
                    pred, tgt = _align_shapes(pred, tgt)
                    # Lock channel subset at first time we see this modality.
                    if n not in chan_locked:
                        k = _SPECTRO_CHANNEL_BUDGET.get(n, 4)
                        chan_locked[n] = _pick_top_variance_channels(tgt, k)
                    chs = chan_locked[n]
                    pred = pred[:, chs]   # (1, k, freq, time)
                    tgt = tgt[:, chs]
                storage[seg_idx][n]["pred"][pos] = pred.detach().cpu()
                storage[seg_idx][n]["target"][pos] = tgt.detach().cpu()
                if storage[seg_idx][n]["channels_used"] is None and n in chan_locked:
                    storage[seg_idx][n]["channels_used"] = chan_locked[n]
        global_idx += bs

    # Stack per-(seg, modality) into (T_seg, ...) tensors. Drop slots
    # that didn't get filled (shouldn't happen unless shot is shorter
    # than the segment range, which we already filtered).
    out: Dict[int, Dict[str, Dict[str, torch.Tensor]]] = {}
    for seg_idx, mods in storage.items():
        out[seg_idx] = {}
        for n, blob in mods.items():
            # Skip modalities with no data (video, or fully-skipped).
            preds = [t for t in blob["pred"] if t is not None]
            tgts = [t for t in blob["target"] if t is not None]
            if not preds:
                continue
            # Each tensor: (1, C, ...). Concatenate along time axis.
            # We want (T_seg, C, ...), so squeeze the leading 1 and stack.
            pred_stack = torch.cat([p[0:1] for p in preds], dim=0)
            tgt_stack = torch.cat([t[0:1] for t in tgts], dim=0)
            out[seg_idx][n] = {
                "pred": pred_stack,
                "target": tgt_stack,
                "kind": blob["kind"],
                "window_idx_range": blob["window_idx_range"],
                "channels_used": blob["channels_used"],
            }
    return out


# ─────────────────────────────────────────────────────────────────────
# Per-modality stitched renderers
# ─────────────────────────────────────────────────────────────────────


def _stitched_time_axis(window_idx_range: Tuple[int, int]) -> np.ndarray:
    """Return seconds-since-shot-start for each window-START of the segment.

    Plan §10 Q1: chunks are strictly monotonic in time within a shot,
    so t_s = window_idx × chunk_duration_s. The returned array has one
    entry per WINDOW (T_seg long), suitable for line plots that show
    one value per window (e.g., per-window aggregated stats). For
    raw-sample line plots that need within-window time, the renderer
    expands the window axis by ``n_samples`` and computes the per-sample
    timestamps internally.
    """
    start, end = window_idx_range
    return np.arange(start, end) * _CHUNK_DURATION_S


def _render_ts_stitched(
    pred_stack: torch.Tensor,
    target_stack: torch.Tensor,
    kind: str,
    window_idx_range: Tuple[int, int, int],
    out_path: Path,
    shot_id: int, modality: str, split: str, seg_idx: int,
    n_channels_to_show: int,
) -> None:
    """Stitched line plot for slow_ts / fast_ts.

    Concatenates the per-window prediction samples into a single long
    non-overlapping time series. Each stored window's prediction spans
    ``chunk_duration_s`` (50 ms); consecutive stored windows are
    ``stride × step_size_s`` apart in raw window index — by design this
    is exactly ``chunk_duration_s`` so neighbouring windows' predictions
    are contiguous, not overlapping. GT solid + model dashed.
    """
    # pred_stack / target_stack shape: (T_seg, C, n_samples)
    p = pred_stack.numpy()
    t = target_stack.numpy()
    t_seg, n_ch, n_samples = p.shape

    # Flatten the (T_seg, n_samples) axes into one long time series.
    p_flat = p.transpose(1, 0, 2).reshape(n_ch, t_seg * n_samples)
    t_flat = t.transpose(1, 0, 2).reshape(n_ch, t_seg * n_samples)

    # Time axis in absolute seconds since shot t=0 (NOT post-warmup
    # time). Dataset semantics: window i's prediction target spans
    # [t_pred_start, t_pred_start + chunk_duration_s] where
    #   t_pred_start = warmup_s + i × step_size_s + chunk_duration_s
    # i.e. the dataset skips warmup_s of leading shot data, and window
    # 0's input is at t ∈ [warmup_s, warmup_s + 50 ms], its prediction
    # target at [warmup_s + 50 ms, warmup_s + 100 ms]. The stored
    # windows are spaced ``stride × step_size_s = chunk_duration_s``
    # apart, so each window's n_samples cover its own 50 ms
    # non-overlapping slice. Total span = t_seg × chunk_duration_s.
    start_w, end_w, stride = window_idx_range
    t_window_start_s = (
        _WARMUP_S + _CHUNK_DURATION_S
        + (start_w + np.arange(t_seg) * stride) * _STEP_SIZE_S
    )
    dt_per_sample = _CHUNK_DURATION_S / n_samples
    time_s = np.empty(t_seg * n_samples)
    for wi in range(t_seg):
        time_s[wi * n_samples:(wi + 1) * n_samples] = (
            t_window_start_s[wi] + np.arange(n_samples) * dt_per_sample
        )

    # NaN-aware channel ranking: matplotlib draws NaN as line gaps, so
    # pred-only plotting needs no special handling per-line; we just
    # skip the GT line when GT is entirely missing for this segment.
    has_any_gt = bool(np.isfinite(t_flat).any())
    if has_any_gt:
        channels = _pick_top_variance_channels(target_stack[:1], n_channels_to_show)
    else:
        # Pick by prediction variance instead.
        pred_var = p_flat.var(axis=1)
        nz = np.nonzero(pred_var)[0]
        if len(nz) >= n_channels_to_show:
            order = np.argsort(-pred_var[nz])
            channels = nz[order[:n_channels_to_show]].tolist()
        else:
            channels = list(range(min(n_channels_to_show, n_ch)))

    if kind == "fast_ts":
        # 8 channels → 4×2 small-multiples grid.
        n_cols = 2
        n_rows = (len(channels) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(13, 1.6 * n_rows + 0.5),
            sharex=True, sharey=False, squeeze=False,
        )
        for i, c in enumerate(channels):
            ax = axes[i // n_cols][i % n_cols]
            gt_lbl = "GT" if i == 0 and has_any_gt else None
            pr_lbl = "model" if i == 0 else None
            if has_any_gt:
                ax.plot(time_s, t_flat[c], color=_GT_COLOR, linewidth=_GT_LW,
                        alpha=0.85, label=gt_lbl)
            ax.plot(time_s, p_flat[c], color=_PRED_COLOR,
                    linestyle=_PRED_LS, linewidth=_PRED_LW, alpha=0.85,
                    label=pr_lbl)
            ax.set_ylabel(f"ch {c}", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3, linewidth=0.5)
        # Hide unused subplots if odd channel count.
        for i in range(len(channels), n_rows * n_cols):
            axes[i // n_cols][i % n_cols].set_visible(False)
        axes[-1][0].set_xlabel("time since shot start (s)", fontsize=9)
        if n_cols > 1:
            axes[-1][1].set_xlabel("time since shot start (s)", fontsize=9)
        # One legend at the top.
        axes[0][0].legend(loc="upper right", fontsize=8, framealpha=0.85)
    else:
        # slow_ts: all chosen channels in one panel. Each channel gets its
        # own color (tab10) and GT/model share the color but differ in
        # linestyle (solid/dashed). Legend has two parts: channel→color
        # mapping, plus a style key showing "solid=GT, dashed=model".
        fig, ax = plt.subplots(figsize=(13, 4))
        ch_colors = plt.get_cmap("tab10").colors
        channel_proxies = []
        for i, c in enumerate(channels):
            color = ch_colors[i % len(ch_colors)]
            if has_any_gt:
                ax.plot(time_s, t_flat[c], color=color, linewidth=_GT_LW,
                        alpha=0.9)
            ax.plot(time_s, p_flat[c], color=color, linestyle=_PRED_LS,
                    linewidth=_PRED_LW, alpha=0.9)
            channel_proxies.append(
                Line2D([0], [0], color=color, linewidth=_GT_LW, label=f"ch {c}")
            )
        style_proxies = [
            Line2D([0], [0], color="black", linewidth=_GT_LW, label="GT"),
            Line2D([0], [0], color="black", linestyle=_PRED_LS,
                   linewidth=_PRED_LW, label="model"),
        ]
        ax.set_xlabel("time since shot start (s)", fontsize=9)
        ax.set_ylabel("standardized signal", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ch_legend = ax.legend(
            handles=channel_proxies, loc="upper right", fontsize=8,
            framealpha=0.85,
            title=f"{len(channels)} top-variance channels",
            title_fontsize=7,
        )
        ax.add_artist(ch_legend)
        ax.legend(handles=style_proxies, loc="upper left", fontsize=8,
                  framealpha=0.85)

    span_s = t_seg * _CHUNK_DURATION_S
    fig.suptitle(
        f"shot {shot_id} — {modality} ({kind}) — split: {split}   |   "
        f"segment {seg_idx} ({t_seg} stride-{stride} windows from raw "
        f"{start_w}–{end_w}, "
        f"{span_s:.2f} s of non-overlapping prediction)",
        fontsize=11, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _render_spectrogram_stitched(
    pred_stack: torch.Tensor,
    target_stack: torch.Tensor,
    window_idx_range: Tuple[int, int, int],
    channels_used: List[int],
    out_dir: Path,
    shot_id: int, modality: str, split: str, seg_idx: int,
) -> List[Path]:
    """Stitched spectrogram heatmap, **one PNG per channel** in the
    representative subset (plan §10 Q2). Three rows: GT, model, |diff|.
    Time axis spans the full segment (~4 s) by concatenating
    non-overlapping per-window predictions.

    Returns the list of paths written.
    """
    # pred_stack / target_stack shape: (T_seg, n_subset_channels, freq, time_per_window)
    p = pred_stack.numpy()
    t = target_stack.numpy()
    t_seg, k, n_freq, n_time = p.shape
    # Stitch along the per-window time axis.
    p_stitched = p.transpose(1, 2, 0, 3).reshape(k, n_freq, t_seg * n_time)
    t_stitched = t.transpose(1, 2, 0, 3).reshape(k, n_freq, t_seg * n_time)
    diff = np.abs(t_stitched - p_stitched)

    # Time axis in absolute seconds since shot t=0 (matches _render_ts_stitched).
    # First prediction window starts at warmup_s + chunk_duration_s; stitched
    # windows are spaced chunk_duration_s apart.
    start_w, end_w, stride = window_idx_range
    t_start_s = _WARMUP_S + _CHUNK_DURATION_S + start_w * _STEP_SIZE_S
    t_end_s = t_start_s + t_seg * _CHUNK_DURATION_S

    paths = []
    for kk, ch in enumerate(channels_used):
        fig, axes = plt.subplots(3, 1, figsize=(13, 6.5), sharex=True)
        # NaN-aware vmin/vmax. When GT is present we still anchor to it
        # so model outliers don't compress the GT color range. When
        # GT is missing for this channel/shot we fall back to the
        # prediction range so the model panel renders meaningfully;
        # the GT and diff panels then contain NaN and matplotlib draws
        # them blank.
        has_gt = bool(np.isfinite(t_stitched[kk]).any())
        if has_gt:
            vmin = float(np.nanmin(t_stitched[kk]))
            vmax = float(np.nanmax(t_stitched[kk]))
        else:
            vmin = float(np.nanmin(p_stitched[kk]))
            vmax = float(np.nanmax(p_stitched[kk]))

        im0 = axes[0].imshow(
            t_stitched[kk], aspect="auto", origin="lower", cmap=_HEAT_CMAP,
            vmin=vmin, vmax=vmax, extent=(t_start_s, t_end_s, 0, n_freq),
        )
        im1 = axes[1].imshow(
            p_stitched[kk], aspect="auto", origin="lower", cmap=_HEAT_CMAP,
            vmin=vmin, vmax=vmax, extent=(t_start_s, t_end_s, 0, n_freq),
        )
        im2 = axes[2].imshow(
            diff[kk], aspect="auto", origin="lower", cmap=_DIFF_CMAP,
            extent=(t_start_s, t_end_s, 0, n_freq),
        )
        for ax_, label in zip(axes, ["GT", "model", "|GT − model|"]):
            ax_.set_ylabel(f"freq bin (ch {ch})", fontsize=8)
            ax_.text(0.005, 0.95, label, transform=ax_.transAxes, fontsize=9,
                     color="white", va="top",
                     bbox=dict(boxstyle="round,pad=0.25", fc="black", alpha=0.7))
            ax_.tick_params(labelsize=7)
        axes[-1].set_xlabel("time since shot start (s)", fontsize=9)

        # Per-row colorbar slots via axes_grid1 so all three data axes
        # end up with identical physical width — fig.colorbar(ax=...) was
        # shrinking the GT/model rows and the diff row by different
        # amounts, leaving the bottom panel's x-axis misaligned with the
        # top two. The middle row's slot is created invisible so its
        # data axis matches widths but no duplicate cbar is drawn (the
        # GT colorbar applies to both GT and model since they share vmin/vmax).
        d0 = make_axes_locatable(axes[0])
        cax0 = d0.append_axes("right", size="1.5%", pad=0.08)
        fig.colorbar(im0, cax=cax0, label="spectral intensity (standardized)")
        d1 = make_axes_locatable(axes[1])
        cax1 = d1.append_axes("right", size="1.5%", pad=0.08)
        cax1.set_visible(False)
        d2 = make_axes_locatable(axes[2])
        cax2 = d2.append_axes("right", size="1.5%", pad=0.08)
        fig.colorbar(im2, cax=cax2, label="|GT − model|")

        span_s = t_seg * _CHUNK_DURATION_S
        fig.suptitle(
            f"shot {shot_id} — {modality} ch {ch} — split: {split}   |   "
            f"segment {seg_idx} ({t_seg} stride-{stride} windows from "
            f"raw {start_w}–{end_w}, {span_s:.2f} s)",
            fontsize=11, y=0.99,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.965))

        out_path = out_dir / f"{shot_id}_stitched_{seg_idx}_ch{ch}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        paths.append(out_path)
    return paths


# ─────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_dir", type=Path, required=True)
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
        help="Cap unique shots to plot. 0 = all top/bottom-selected. "
             "Coverage-aware ordering (set-cover by kind first).",
    )
    p.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--K", type=int, default=0,
        help="Rollout horizon. 0 (default) autodetects from checkpoint "
             "(K=1 for Stage 1, K=K_max for Stage 2). Stitched plots "
             "render the final-step (k=K) prediction.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    device = torch.device(args.device)
    plots_root = args.output_dir / args.plots_subdir

    # top_bottom_shots.csv.gz is no longer a filter — Phase 3 now
    # iterates EVERY shot × EVERY non-video model.diagnostics modality.
    # per_window_metrics.csv.gz provides the canonical shot/split list.
    pw_path = args.output_dir / "per_window_metrics.csv.gz"
    if not pw_path.exists():
        raise SystemExit(f"required input not found: {pw_path}")
    per_window = pd.read_csv(pw_path, compression="gzip")
    logger.info(f"Loaded {len(per_window):,} per-window rows")

    # Load model.
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

    # Every shot in the val set, with its split derived from per_window.
    shot_split: Dict[int, str] = (
        per_window.drop_duplicates("shot_id")[["shot_id", "split"]]
                  .set_index("shot_id")["split"].to_dict()
    )
    all_shots = sorted(shot_split.keys())
    if args.max_shots_to_plot and args.max_shots_to_plot > 0:
        all_shots = all_shots[: args.max_shots_to_plot]
    # Every non-video diagnostic — Phase 3.1 handles the video kind.
    diag_iter = [
        (c.name, c.kind) for c in model.diagnostics if c.kind != "video"
    ]
    logger.info(
        f"Plotting stitched segments for {len(all_shots)} shots "
        f"× {len(diag_iter)} non-video modalities"
    )

    for i, shot_id in enumerate(all_shots, start=1):
        file_path = args.data_dir / f"{shot_id}_processed.h5"
        if not file_path.exists():
            logger.warning(f"shot {shot_id}: file missing at {file_path}")
            continue
        split = shot_split[shot_id]
        logger.info(f"({i}/{len(all_shots)}) shot {shot_id} ({split}): re-inference …")
        segments = collect_stitched_segments_for_shot(
            model=model, file_path=file_path, device=device,
            args=args, stats=stats, K=K,
        )

        # All non-video diagnostics get a stitched plot, even if no GT
        # exists for them on this shot (renderers are NaN-aware).
        for modality, kind in diag_iter:
            out_dir = plots_root / split / modality
            for seg_idx, blob_by_mod in segments.items():
                if modality not in blob_by_mod:
                    continue
                blob = blob_by_mod[modality]
                pred_stack = blob["pred"]
                tgt_stack = blob["target"]
                win_range = blob["window_idx_range"]

                if kind in ("slow_ts", "fast_ts"):
                    n_show = 8 if kind == "fast_ts" else 4
                    out_path = (
                        out_dir / f"{shot_id}_stitched_{seg_idx}.png"
                    )
                    _render_ts_stitched(
                        pred_stack=pred_stack, target_stack=tgt_stack,
                        kind=kind, window_idx_range=win_range,
                        out_path=out_path,
                        shot_id=shot_id, modality=modality,
                        split=split, seg_idx=seg_idx,
                        n_channels_to_show=n_show,
                    )
                    logger.info(
                        f"  → {out_path.relative_to(args.output_dir)}"
                    )
                elif kind == "spectrogram":
                    chs = blob["channels_used"] or []
                    paths = _render_spectrogram_stitched(
                        pred_stack=pred_stack, target_stack=tgt_stack,
                        window_idx_range=win_range,
                        channels_used=chs, out_dir=out_dir,
                        shot_id=shot_id, modality=modality,
                        split=split, seg_idx=seg_idx,
                    )
                    for p in paths:
                        logger.info(
                            f"  → {p.relative_to(args.output_dir)}"
                        )

    logger.info("Phase 3.0 (stitched plots for TS + spectrogram) complete.")


if __name__ == "__main__":
    main()
