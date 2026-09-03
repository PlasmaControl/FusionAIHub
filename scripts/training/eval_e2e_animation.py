"""Single-shot animated movie: tangtv video on top + growing time traces.

Layout
------
  Top (gridspec_top):  2 channel rows × 3 cols (GT / Pred / |GT−Pred|)
                       of tangtv frames. The current rollout window's
                       last frame is shown at each animation step.
                       Channel 1 is rotated 180° vs channel 0 (per
                       project-tangtv-channel1-flip memory).
  Bottom (gridspec_bot): 4 rows × 4 cols growing-time-trace panels.
                       Default mapping mirrors the baseline:
                         row 0 → ts_core_temp (≈ "tste")
                         row 1 → ts_core_density (≈ "tsne")
                         row 2 → ece (spectrogram — placeholder until
                                      rolling-heatmap rendering is added)
                         row 3 → co2 (spectrogram — placeholder)
                       Trace rows accumulate samples as the cursor
                       advances; cursor x-position is shared with the
                       video frame above so both panels stay in lockstep.

Both top and bottom share the same animation timeline: one frame per
rollout window, advanced by ``--stride``.

Use
---
    pixi run python scripts/training/eval_e2e_animation.py \\
        --checkpoint /lustre/orion/fus187/proj-shared/models/e2e_stage1_d1024_48L/e2e_stage1_best.pt \\
        --data_dir   /lustre/orion/fus187/proj-shared/foundation_model \\
        --stats_path /lustre/orion/fus187/proj-shared/foundation_model_meta/preprocessing_stats.pt \\
        --shot_id    193159 \\
        --output_dir eval_runs/animations \\
        --fps 20 --stride 1
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

# Frontier compute nodes do not ship a system ffmpeg; point matplotlib's
# FFMpegWriter at the binary bundled with the imageio-ffmpeg pip package
# (already a dependency of our Phase 3.1 video renderer). Without this,
# matplotlib falls back to PillowWriter and emits an enormous GIF.
try:
    from imageio_ffmpeg import get_ffmpeg_exe as _get_ffmpeg_exe
    plt.rcParams["animation.ffmpeg_path"] = _get_ffmpeg_exe()
except Exception:
    pass

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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_e2e import (  # type: ignore[import]  # noqa: E402
    detect_stage_K,
    load_checkpoint_with_refine_tolerance,
    make_rollout_if_needed,
    rollout_forward_one_batch,
)

logger = logging.getLogger("eval_e2e_animation")


# ─────────────────────────────────────────────────────────────────────
# Style + defaults
# ─────────────────────────────────────────────────────────────────────

_WARMUP_S = 1.0
_CHUNK_DURATION_S = 0.05
_STEP_SIZE_S = 0.01

_GT_COLOR = "black"
_PRED_COLOR = "#e41a1c"   # crisp red, sharper than tab:red on projectors
_PRED_LS = "--"
_HEAT_CMAP = "gray"
_DIFF_CMAP = "magma"

# Presentation-grade rcParams. Applied per-call in build_animation so the
# script doesn't pollute a parent process's matplotlib state.
_PRESENTATION_RC = {
    "font.size": 12,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "regular",
    "axes.linewidth": 1.0,
    "axes.grid": True,
    "axes.grid.axis": "both",
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": 1.8,
    "legend.fontsize": 10,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#cccccc",
    "figure.titlesize": 15,
    "figure.titleweight": "bold",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
}

# Default row mapping: (modality_name, label, kind, grid_position).
# grid_position = (row, col, rowspan, colspan) inside the 3×2 trace
# sub-grid. Column-wise layout:
#   Col 0:  T_e   →  n_e    →  ECE (spectro placeholder)
#   Col 1:  T_i   →  v_tor  →  CO2 (spectro placeholder)
# Top two rows hold the four slow_ts trace panels; bottom row holds
# the two spectrogram placeholder tiles awaiting the rolling-heatmap
# renderer.
_DEFAULT_ROWS: List[Tuple[str, str, str, Tuple[int, int, int, int]]] = [
    ("ts_core_temp",    "Electron Temperature",  "slow_ts",    (0, 0, 1, 1)),
    ("cer_ti",          "Ion Temperature",       "slow_ts",    (0, 1, 1, 1)),
    ("ts_core_density", "Electron Density",      "slow_ts",    (1, 0, 1, 1)),
    ("cer_rot",         "Plasma Rotation",       "slow_ts",    (1, 1, 1, 1)),
    ("ece",             "ECE",                   "spectrogram",(2, 0, 1, 1)),
    ("co2",             r"CO$_2$",               "spectrogram",(2, 1, 1, 1)),
]
_TRACE_GRID_SHAPE = (3, 2)        # rows × cols of the trace sub-grid


def _denormalize_slow_ts(
    arr: np.ndarray, modality: str, stats: dict,
) -> np.ndarray:
    """Inverse of the data loader's ``log_standardize`` for slow_ts /
    fast_ts modalities.

    The forward transform (data_loader.py: ``log_standardize``) is::

        x_clipped = clip(x_raw, min=-0.99)
        x_log     = log10(x_clipped + 1)
        x_norm    = (x_log - log_mean) / log_std.clamp(1e-3)

    The inverse is::

        x_raw = 10 ** (x_norm * log_std + log_mean) - 1

    The clip is a saturating op that we don't try to invert; for any
    plausible plasma signal it never fires.

    arr shape: ``(n_windows, n_channels, n_samples)`` for slow_ts/fast_ts.
    Mean/std are broadcast over the (n_windows, n_samples) axes.
    Returns physical units (m⁻³ for density, eV for temperature, rad/s
    for rotation, etc., depending on modality).

    If stats are missing for the modality, returns the input unchanged
    so the caller falls back to plotting in normalized units.
    """
    if modality not in stats or "log" not in stats[modality]:
        return arr
    mean = np.asarray(stats[modality]["log"]["mean"], dtype=arr.dtype)
    std = np.asarray(stats[modality]["log"]["std"], dtype=arr.dtype)
    # Broadcast: arr is (n_windows, n_ch, n_samples), mean/std are (n_ch,).
    mean_b = mean[None, :, None]
    std_b = std[None, :, None]
    out = np.power(10.0, arr * std_b + mean_b) - 1.0
    # Optional per-modality post-scale (e.g., eV → keV for temperatures).
    scale = _PHYS_SCALE.get(modality, 1.0)
    if scale != 1.0:
        out = out * scale
    return out


# Per-spectrogram-modality Nyquist frequency (kHz) for y-axis extent.
# Project default: 500 kHz sample stream with n_fft=1024 → Nyquist
# = 250 kHz, 512 kept bins. ECE/CO2/BES all use this STFT config in
# the project's data preprocessing. If a future modality uses a
# different sample rate, add an entry here.
_SPECTRO_MAX_FREQ_KHZ: Dict[str, float] = {
    "ece": 250.0,
    "co2": 250.0,
    "bes": 250.0,
}


# Physical channel names for the tangtv video — DIII-D's two tangential
# views (upper and lower divertor).
_VIDEO_CH_NAMES: Dict[int, str] = {
    0: "Upper Divertor",
    1: "Lower Divertor",
}


def _video_display_channels(n_model_channels: int) -> List[tuple]:
    """Return the tangtv model channels to DISPLAY as ``(model_channel,
    label)`` pairs, given how many channels the model predicts.

    NEW 7-channel model (model ch i == raw ch i): lower divertor
    (model ch2 = LODIV_240RM1:PERP) then upper divertor (model ch4 =
    UPDIV_0RP1:PERP). No 180° flip on either (see the flip gate in the
    video-data block, which is restricted to the old 2-channel path).

    OLD (<= 2 channel) model: the first ``_VIDEO_N_CHANNELS_DISPLAY``
    model channels labelled via ``_VIDEO_CH_NAMES`` — i.e. ch0 "Upper
    Divertor", ch1 "Lower Divertor" — exactly as before (backward-compat,
    including the ch1 flip).
    """
    if n_model_channels >= 5:
        return [(2, "Lower Divertor"), (4, "Upper Divertor")]
    n = min(_VIDEO_N_CHANNELS_DISPLAY, n_model_channels)
    return [(c, _VIDEO_CH_NAMES.get(c, f"ch {c}")) for c in range(n)]


# Slow-TS panels that should display the *same* set of channels as a
# source panel. Te+ne share Thomson Scattering chord indices; Ti+vtor
# share CER chord indices. The "linked" panel (key) reuses the channel
# selection picked by its source (value), so the two panels above each
# other in a column are spatially co-located.
_CHANNEL_LINK_SOURCE: Dict[str, str] = {
    "ts_core_density": "ts_core_temp",
    "cer_rot":         "cer_ti",
}


# Physical-unit labels for the y-axis of each modality. Temperature
# modalities are displayed in keV; the raw stats are in eV, so the
# corresponding scale factor lives in `_PHYS_SCALE` below.
_PHYS_UNITS: Dict[str, str] = {
    "ts_core_density":      r"$n_e$ (m$^{-3}$)",
    "ts_core_temp":         r"$T_e$ (keV)",
    "ts_tangential_density":r"$n_e$ (m$^{-3}$)",
    "ts_tangential_temp":   r"$T_e$ (keV)",
    "cer_ti":               r"$T_i$ (keV)",
    "cer_rot":              r"$v_{tor}$ (km/s)",
    "mse":                  r"MSE (signed)",
    "filterscopes":         r"intensity (a.u.)",
}

# Optional post-denormalize scale (multiplicative). Temperatures get
# /1000 to convert eV → keV; everything else is identity (1.0).
_PHYS_SCALE: Dict[str, float] = {
    "ts_core_temp":         1e-3,
    "ts_tangential_temp":   1e-3,
    "cer_ti":               1e-3,
}

# Video constants
_VIDEO_MODALITY = "tangtv"
_VIDEO_N_CHANNELS_DISPLAY = 2   # show channels 0 + 1
_VIDEO_N_COLS = 3               # GT / Pred / |diff|


# ─────────────────────────────────────────────────────────────────────
# Inference: gather full-shot predictions per modality
# ─────────────────────────────────────────────────────────────────────


@torch.no_grad()
def collect_shot_predictions(
    model: E2EFoundationModel,
    file_path: Path,
    device: torch.device,
    args: argparse.Namespace,
    stats: dict,
    K: int,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Re-infer every window of one shot. Returns per-modality stacks of
    the k=K-1 (final-step) predictions + targets.

    Shapes:
      * slow_ts / fast_ts: pred/target ``(n_windows, n_channels, n_samples_per_window)``
      * spectrogram:       pred/target ``(n_windows, n_channels, freq_bins, trunc_t)``
      * video:             pred/target ``(n_windows, n_channels, n_frames, H, W)``
    All tensors are CPU.
    """
    diag_names = [c.name for c in model.diagnostics]
    act_names = [c.name for c in model.actuators]
    rollout = make_rollout_if_needed(model, K, args.chunk_duration_s)

    # IMPORTANT: use step_size_s == chunk_duration_s so consecutive
    # windows are non-overlapping. The animation's time-axis math
    # assumes a stitched non-overlapping timeline. If we used the
    # default args.step_size_s (10 ms), n_windows would be ~5× too
    # many and the time axis would blow out by 5× (e.g., 30 s instead
    # of the real ~6 s shot duration). Phase 3 stitched solves the
    # same problem by skipping 4/5 windows at iteration time; we just
    # configure the dataset coarser to begin with.
    ds = TokamakMultiFileDataset(
        [file_path],
        chunk_duration_s=args.chunk_duration_s,
        prediction_mode=True,
        prediction_horizon_s=K * args.chunk_duration_s,
        step_size_s=args.chunk_duration_s,
        warmup_s=args.warmup_s,
        preprocessing_stats=stats,
        input_signals=diag_names,
        target_signals=diag_names + act_names,
        lengths_cache_path=None,
    )
    n_windows = len(ds)
    if n_windows == 0:
        raise SystemExit(f"shot {file_path.name}: empty dataset")
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers,
        drop_last=False, pin_memory=False,
    )

    pred_lists: Dict[str, List[torch.Tensor]] = {n: [] for n in diag_names}
    tgt_lists: Dict[str, List[torch.Tensor]] = {n: [] for n in diag_names}
    for batch in loader:
        predictions_per_k, _, targets_per_k, _ = rollout_forward_one_batch(
            model, rollout, batch, device, K, args.chunk_duration_s
        )
        # Always take the 1-step-ahead prediction (rollout index 0) so
        # the rendered frame aligns with the time-axis helper, which
        # assumes a single-chunk lookahead per window. predictions_per_k
        # has length K (=1 for Stage 1, =K_max for Stage 2). Taking
        # index K-1 (the K-step-ahead chunk) was the original code and
        # shifted the plot left by (K-1)*chunk_duration_s — verified
        # against job 4759613 (Stage 2 delta, K=10 → 0.45 s shift).
        pred = predictions_per_k[0]
        tgt = targets_per_k[0]
        for n in diag_names:
            pred_lists[n].append(pred[n].detach().cpu())
            tgt_lists[n].append(tgt[n].detach().cpu())
    out: Dict[str, Dict[str, torch.Tensor]] = {}
    for n in diag_names:
        if not pred_lists[n]:
            continue
        out[n] = {
            "pred": torch.cat(pred_lists[n], dim=0),
            "target": torch.cat(tgt_lists[n], dim=0),
        }
    logger.info(f"Collected predictions: {n_windows} windows × {len(diag_names)} modalities")
    return out


# ─────────────────────────────────────────────────────────────────────
# Time-axis helpers
# ─────────────────────────────────────────────────────────────────────


def _ts_time_axis_ms(n_windows: int, n_samples_per_window: int) -> np.ndarray:
    """Per-sample time axis (ms) for a stitched TS prediction.

    Window w of the prediction targets t ∈ [warmup + (w+1)*chunk,
    warmup + (w+2)*chunk]. Windows are spaced by chunk_duration_s
    (non-overlapping) when stitched, so per-sample dt = chunk/n_samples.
    """
    t0_s = _WARMUP_S + _CHUNK_DURATION_S
    dt_s = _CHUNK_DURATION_S / n_samples_per_window
    return (t0_s + np.arange(n_windows * n_samples_per_window) * dt_s) * 1000.0


def _window_end_time_ms(w: int) -> float:
    """Time (ms) at the end of rollout window ``w``."""
    return (_WARMUP_S + (w + 1) * _CHUNK_DURATION_S + _CHUNK_DURATION_S) * 1000.0


# ─────────────────────────────────────────────────────────────────────
# Animation builder
# ─────────────────────────────────────────────────────────────────────


def build_animation(
    blobs: Dict[str, Dict[str, torch.Tensor]],
    row_spec: List[Tuple[str, str, str, Tuple[int, int, int, int]]],
    stats: dict,
    out_path: Path,
    *,
    shot_id: int,
    fps: int,
    stride: int,
    dpi: int,
    t_start_s: float = 1.0,
    t_end_s: float = 4.5,
    video_smooth_sigma: float = 0.0,
    mode: str = "both",
) -> None:
    """Build the combined video + 4×4 traces animation and save as mp4."""
    plt.rcParams.update(_PRESENTATION_RC)
    # ── Establish animation length from any TS modality with data ───
    ts_blob = next(
        (blobs[name] for name, _, kind, _ in row_spec
         if kind in ("slow_ts", "fast_ts") and name in blobs),
        None,
    )
    if ts_blob is None:
        raise SystemExit(
            "Need at least one slow_ts / fast_ts row for animation timing."
        )
    pred_ts = ts_blob["pred"].numpy()   # (n_windows, C, n_samples)
    n_windows_all, _, n_samples = pred_ts.shape

    # ── Time-range filter: keep only windows whose end-time falls in
    #    [t_start_s, t_end_s]. Reduces clutter and animation length for
    #    presentation use. Defaults give a clean 1 s slice (1-2 s).
    t_end_per_window_s = (
        _WARMUP_S + _CHUNK_DURATION_S + np.arange(1, n_windows_all + 1) * _CHUNK_DURATION_S
    )
    in_range = (t_end_per_window_s >= t_start_s) & (t_end_per_window_s <= t_end_s)
    if not in_range.any():
        raise SystemExit(
            f"No rollout window's end-time falls in [{t_start_s}, {t_end_s}] s. "
            f"Shot end-time range was [{t_end_per_window_s[0]:.2f}, "
            f"{t_end_per_window_s[-1]:.2f}] s."
        )
    w_lo = int(np.argmax(in_range))            # first True index
    w_hi = int(len(in_range) - np.argmax(in_range[::-1]))  # one past last True
    n_windows = w_hi - w_lo
    logger.info(
        f"Time-range filter: kept windows [{w_lo}, {w_hi}) of "
        f"{n_windows_all} total → t ∈ [{t_end_per_window_s[w_lo]:.2f}, "
        f"{t_end_per_window_s[w_hi - 1]:.2f}] s"
    )
    n_anim_frames = (n_windows + stride - 1) // stride

    # All TS time-axis arrays will reference samples in [w_lo*n_samples,
    # w_hi*n_samples] of the full per-sample axis. Precompute once.
    full_t_axis = _ts_time_axis_ms(n_windows_all, n_samples)
    sample_lo = w_lo * n_samples
    sample_hi = w_hi * n_samples
    t_ms_per_sample = full_t_axis[sample_lo:sample_hi]
    t_total_samples = len(t_ms_per_sample)

    # ── Video data ──────────────────────────────────────────────────
    has_video = _VIDEO_MODALITY in blobs
    if has_video:
        # Slice to the chosen time range FIRST.
        vp = blobs[_VIDEO_MODALITY]["pred"].numpy()[w_lo:w_hi]
        vt = blobs[_VIDEO_MODALITY]["target"].numpy()[w_lo:w_hi]
        # Suppress patch-boundary discontinuities in the prediction
        # (Stage 2 K-step rollout amplifies token noise → visible 12×12
        # grid). Spatial-only Gaussian; GT untouched. shape: (n_w, C, T, H, W).
        if video_smooth_sigma > 0:
            from scipy.ndimage import gaussian_filter
            logger.info(
                f"Smoothing video predictions with σ={video_smooth_sigma}"
                " px on (H, W) only"
            )
            vp = gaussian_filter(
                vp, sigma=(0, 0, 0, video_smooth_sigma, video_smooth_sigma),
                mode="reflect",
            )
        # DISPLAY channel selection — old 2-channel models keep model ch0/1
        # ("Upper"/"Lower"); new 7-channel models show model ch2 (lower) +
        # ch4 (upper). video_display = [(model_channel, label), ...]; the
        # render/update index by display row but pull data from model ch.
        is_seven_ch_video = vp.shape[1] >= 5
        video_display = _video_display_channels(vp.shape[1])
        video_model_chs = [mc for mc, _ in video_display]
        video_labels = [lbl for _, lbl in video_display]
        # Channel-1 180° rotation (per project-tangtv-channel1-flip) —
        # OLD 2-channel path ONLY. The 7-channel path applies no flip.
        if (not is_seven_ch_video) and vp.shape[1] > 1:
            vp[:, 1] = vp[:, 1, :, ::-1, ::-1]
            vt[:, 1] = vt[:, 1, :, ::-1, ::-1]
        n_video_channels = len(video_display)   # number of DISPLAY rows
        # Per-display-row intensity range across the kept window range
        # (pulled from the corresponding model channel).
        v_vmin = np.full(n_video_channels, +np.inf, dtype=np.float64)
        v_vmax = np.full(n_video_channels, -np.inf, dtype=np.float64)
        v_dmax = np.zeros(n_video_channels, dtype=np.float64)
        for c, mc in enumerate(video_model_chs):
            tc = vt[:, mc]
            pc = vp[:, mc]
            if np.isfinite(tc).any():
                v_vmin[c] = float(np.nanmin(tc))
                v_vmax[c] = float(np.nanmax(tc))
            else:
                v_vmin[c] = float(np.nanmin(pc))
                v_vmax[c] = float(np.nanmax(pc))
            v_dmax[c] = float(np.nanmax(np.abs(tc - pc))) if np.isfinite(tc).any() else 1.0
    else:
        n_video_channels = 0

    # ── Figure + gridspec ──────────────────────────────────────────
    # Landscape presentation layout (broader to accommodate the 1-4.5 s
    # time range without crowding):
    #   - Top: video band (n_video_channels × 3 = GT/Pred/|diff|).
    #   - Bottom: 2×2 trace sub-grid (`_TRACE_GRID_SHAPE`). Modalities
    #     placed via the per-row (row, col, rowspan, colspan) tuple in
    #     `row_spec` so a single panel can span both columns.
    trace_rows_total, trace_cols_total = _TRACE_GRID_SHAPE
    fig_w = 18.0
    video_h = 4.2 if n_video_channels >= 2 else 2.1
    trace_h = 2.0 * trace_rows_total
    fig_h = video_h + trace_h + 0.7
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs_root = fig.add_gridspec(
        2, 1,
        height_ratios=[video_h, trace_h],
        hspace=0.12,
        top=0.93, bottom=0.07, left=0.06, right=0.99,
    )
    # Video sub-grid. Number of columns depends on mode:
    #   both  → 3 columns: GT | Predicted | |GT − Predicted|
    #   gt    → 1 column:  GT only
    #   pred  → 1 column:  Predicted only
    if mode == "gt":
        active_cols = [0]
    elif mode == "pred":
        active_cols = [1]
    else:
        active_cols = [0, 1, 2]
    n_video_cols_eff = len(active_cols)
    col_titles_all = ["Ground truth", "Predicted", "|GT − Predicted|"]
    if has_video:
        gs_video = gs_root[0].subgridspec(
            n_video_channels, n_video_cols_eff, hspace=0.28, wspace=0.04,
        )
        video_axes: List[List[plt.Axes]] = []
        video_ims: List[List[matplotlib.image.AxesImage]] = []
        col_titles = [col_titles_all[i] for i in active_cols]
        H, W = vp.shape[3], vp.shape[4]
        for c in range(n_video_channels):
            row_axes = []
            row_ims = []
            for col_idx, col in enumerate(active_cols):
                ax = fig.add_subplot(gs_video[c, col_idx])
                ch_name = video_labels[c]
                # Two-tier title stack — main = "Ground truth" / etc.
                # (row 0 only, lifted via pad so it doesn't overlap the
                # subtitle); subtitle = divertor name (italic, small,
                # gray) sitting just above each panel. Row spacing
                # (`hspace`) leaves room for the subtitle without
                # touching the panel above it.
                if c == 0:
                    ax.set_title(col_titles[col_idx], pad=18)
                ax.text(
                    0.5, 1.02, ch_name,
                    transform=ax.transAxes,
                    ha="center", va="bottom",
                    fontsize=9, fontstyle="italic", color="#444444",
                )
                cmap = _HEAT_CMAP if col < 2 else _DIFF_CMAP
                vmin = 0.0 if col == 2 else v_vmin[c]
                vmax = v_dmax[c] if col == 2 else v_vmax[c]
                im = ax.imshow(
                    np.zeros((H, W)), cmap=cmap, vmin=vmin, vmax=vmax,
                    aspect="equal", interpolation="nearest",
                )
                ax.set_xticks([])
                ax.set_yticks([])
                row_axes.append(ax)
                row_ims.append(im)
            video_axes.append(row_axes)
            video_ims.append(row_ims)
    else:
        video_axes = []
        video_ims = []

    # Trace sub-grid — 2×2, panels placed per the (row,col,rowspan,
    # colspan) tuple in row_spec so a third panel can span both columns.
    # Row 2 (spectrograms) gets a 1.35× height boost — each spectro
    # cell is then internally split into ax_gt + ax_pr, so the boost
    # is needed to keep the sub-panels readable.
    gs_traces = gs_root[1].subgridspec(
        trace_rows_total, trace_cols_total,
        hspace=0.40, wspace=0.20,
        height_ratios=[1.0, 1.0, 1.35],
    )
    ch_colors = plt.get_cmap("tab10").colors
    lines_gt: List[List[plt.Line2D]] = []
    lines_pred: List[List[plt.Line2D]] = []
    cursors: List[List[plt.Line2D]] = []
    # Per-spectrogram panel state for the rolling-heatmap reveal.
    spectro_panels: List[Dict[str, object]] = []

    # Per-row TS time-trace setup. Panels in the same column share
    # x-axes via `col_anchor_ax` so the time cursor stays aligned
    # across rows and we only need one xlabel/tick-label set per
    # column (applied post-loop to the bottom panel).
    col_anchor_ax: Dict[int, plt.Axes] = {}
    col_panels: Dict[int, List[Tuple[int, plt.Axes]]] = {}
    # Channels picked per panel — looked up by linked panels (see
    # _CHANNEL_LINK_SOURCE) so Te+ne and Ti+vtor share chord indices.
    panel_channels: Dict[str, List[int]] = {}
    for r, (name, label, kind, gridpos) in enumerate(row_spec):
        gr, gc, grs, gcs = gridpos
        chs: List[int] = []   # unused with auto top-variance selection
        row_gt: List[plt.Line2D] = []
        row_pred: List[plt.Line2D] = []
        row_cursor: List[plt.Line2D] = []
        if kind in ("slow_ts", "fast_ts") and name in blobs:
            pred_norm_full = blobs[name]["pred"].numpy()
            target_norm_full = blobs[name]["target"].numpy()
            n_w_all, n_ch, n_s = pred_norm_full.shape

            # Channel selection: linked panels (e.g. ts_core_density →
            # ts_core_temp) reuse their source's channels so Te+ne and
            # Ti+vtor display matching chord indices. Otherwise pick the
            # top-N highest-variance channels, then sort ascending so the
            # plot order matches channel index.
            link_src = _CHANNEL_LINK_SOURCE.get(name)
            if link_src and link_src in panel_channels:
                channels = list(panel_channels[link_src])
            else:
                # Rank channels by top-variance on the NORMALIZED data
                # FIRST — float32 variance on denormalized n_e (~1e19)
                # overflows when squared. Variance ordering is invariant
                # under the affine + log transform anyway, so we get the
                # same ranking either way without the overflow.
                tgt_norm_stitched = target_norm_full.transpose(1, 0, 2).reshape(
                    n_ch, n_w_all * n_s
                )
                n_top = 8 if kind == "fast_ts" else 3
                var = np.nanvar(tgt_norm_stitched, axis=1)
                var = np.where(np.isnan(var), 0.0, var)
                nz = np.nonzero(var)[0]
                if len(nz) >= n_top:
                    order = np.argsort(-var[nz])
                    channels = nz[order[:n_top]].tolist()
                else:
                    channels = list(range(min(n_top, n_ch)))
                # Sort ascending so plotted signals are in channel-index
                # order (legend reads ch_low → ch_high).
                channels = sorted(channels)
                panel_channels[name] = channels

            # Now denormalize for plotting (physical units).
            pred_full = _denormalize_slow_ts(pred_norm_full, name, stats)
            target_full = _denormalize_slow_ts(target_norm_full, name, stats)
            # Apply the time-range window slice.
            pred = pred_full[w_lo:w_hi]
            target = target_full[w_lo:w_hi]
            n_w = pred.shape[0]
            pred_stitched = pred.transpose(1, 0, 2).reshape(n_ch, n_w * n_s)
            tgt_stitched = target.transpose(1, 0, 2).reshape(n_ch, n_w * n_s)
            t_ms = full_t_axis[w_lo * n_s : w_hi * n_s] \
                if n_s == n_samples else \
                _ts_time_axis_ms(n_w_all, n_s)[w_lo * n_s : w_hi * n_s]

            # Place panel at (gr, gc) spanning (grs, gcs). First panel
            # per column becomes the anchor — subsequent rows in the
            # column inherit its x-axis via sharex.
            sharex_anchor = col_anchor_ax.get(gc)
            ax = fig.add_subplot(
                gs_traces[gr : gr + grs, gc : gc + gcs],
                sharex=sharex_anchor,
            )
            if sharex_anchor is None:
                col_anchor_ax[gc] = ax
            col_panels.setdefault(gc, []).append((gr, ax))
            # Per-panel ylim across all displayed channels (NaN-aware).
            chan_data = np.concatenate(
                [pred_stitched[c][np.isfinite(pred_stitched[c])]
                 for c in channels]
                + [tgt_stitched[c][np.isfinite(tgt_stitched[c])]
                   for c in channels]
            ) if channels else np.array([0.0])
            if chan_data.size > 0:
                lo, hi = float(chan_data.min()), float(chan_data.max())
                pad = 0.1 * (hi - lo) + 1e-8
                ax.set_ylim(lo - pad, hi + pad)
            ax.set_xlim(t_ms[0], t_ms[-1])
            ax.set_title(label)
            ax.set_ylabel(_PHYS_UNITS.get(name, ""))

            # Plot all channels overlaid: GT solid + Pred dashed, sharing
            # a tab10 color per channel. In gt/pred mode, the unused
            # set of lines is created with alpha=0 (still in lists so
            # update() doesn't index out of range, just invisible).
            gt_alpha = 0.95 if mode != "pred" else 0.0
            pred_alpha = 0.95 if mode != "gt" else 0.0
            show_pred_line_in_legend = mode != "gt"
            show_gt_line_in_legend = mode != "pred"
            channel_handles = []
            for i, c in enumerate(channels):
                color = ch_colors[i % len(ch_colors)]
                lg, = ax.plot([], [], color=color, lw=1.6, alpha=gt_alpha)
                lp, = ax.plot([], [], color=color, ls=_PRED_LS, lw=1.6,
                              alpha=pred_alpha)
                lg.set_array_data_local = (t_ms, tgt_stitched[c])
                lp.set_array_data_local = (t_ms, pred_stitched[c])
                row_gt.append(lg)
                row_pred.append(lp)
                channel_handles.append(
                    plt.Line2D([0], [0], color=color, lw=2.0, label=f"ch {c}")
                )
            cu = ax.axvline(t_ms[0], color="#333333", lw=1.2, ls=":")
            row_cursor.append(cu)

            # Style legend reflects current mode (single line in gt/pred
            # mode, both in 'both' mode).
            if r == 0:
                style_handles = []
                if show_gt_line_in_legend:
                    style_handles.append(
                        plt.Line2D([0], [0], color="black", lw=2.0, label="GT")
                    )
                if show_pred_line_in_legend:
                    style_handles.append(
                        plt.Line2D([0], [0], color="black", lw=2.0,
                                   ls=_PRED_LS, label="model")
                    )
                ch_leg = ax.legend(handles=channel_handles, loc="upper right",
                                    ncol=min(4, len(channel_handles)),
                                    framealpha=0.85)
                ax.add_artist(ch_leg)
                ax.legend(handles=style_handles, loc="upper left",
                          framealpha=0.85)
            else:
                ax.legend(handles=channel_handles, loc="upper right",
                          ncol=min(4, len(channel_handles)),
                          framealpha=0.85)
        elif kind == "spectrogram" and name in blobs:
            # Rolling spectrogram heatmap. Stitch consecutive windows'
            # spectrograms along the time axis into one (F, n_w * T_w)
            # heatmap per panel; split into a GT (top) and Pred (bottom)
            # sub-axes pair inside the cell. The animation update()
            # progressively reveals columns up to the current cursor
            # by overwriting them; unrevealed columns remain NaN and
            # render as the cmap's bad-colour (default transparent →
            # axes facecolor).
            pred_full = blobs[name]["pred"].numpy()       # (n_w, C, F, T)
            target_full = blobs[name]["target"].numpy()
            n_w_all, n_ch_s, n_freq, n_t_s = pred_full.shape
            # Pick the SINGLE highest-variance channel rather than
            # averaging across all channels. Selection runs on the
            # DENORMALIZED (raw log-magnitude) data — after per-channel
            # log_standardize, every channel has var ≈ 1 by construction,
            # so picking on the normalized tensor was effectively random
            # (verified on shot 200729: top-variance channel had only
            # 8 % of its energy in the top-3 freq bins). Denormalizing
            # recovers the raw spectral-energy scale, so channels with
            # actual mode activity stand out.
            if name in stats and "log" in stats[name]:
                _lmean = np.asarray(
                    stats[name]["log"]["mean"], dtype=np.float32
                )[:n_ch_s]
                _lstd = np.clip(
                    np.asarray(stats[name]["log"]["std"], dtype=np.float32),
                    1e-3, None,
                )[:n_ch_s]
                _mean_b = _lmean[None, :, None, None]   # broadcast over (n_w,C,F,T)
                _std_b = _lstd[None, :, None, None]
                # Undo (val - mean)/std → log10(|STFT|+1); also the
                # un-log version for channel-selection variance (raw
                # spectral energy). Display uses log-magnitude so the
                # wide dynamic range stays readable.
                tgt_logmag = target_full * _std_b + _mean_b
                pred_logmag = pred_full * _std_b + _mean_b
                tgt_denorm = np.power(10.0, tgt_logmag) - 1.0
            else:
                tgt_logmag = target_full
                pred_logmag = pred_full
                tgt_denorm = target_full
            tgt_per_ch = tgt_denorm.transpose(1, 0, 2, 3).reshape(n_ch_s, -1)
            var_ch = np.nanvar(tgt_per_ch, axis=1)
            var_ch = np.where(np.isfinite(var_ch), var_ch, -np.inf)
            best_ch = int(np.argmax(var_ch))
            # Use UN-STANDARDIZED log-magnitude for display.
            pred_arr = pred_logmag[w_lo:w_hi, best_ch]      # (n_w_local, F, T)
            target_arr = tgt_logmag[w_lo:w_hi, best_ch]
            n_w_local = pred_arr.shape[0]
            pred_stitched = pred_arr.transpose(1, 0, 2).reshape(
                n_freq, n_w_local * n_t_s
            )
            tgt_stitched = target_arr.transpose(1, 0, 2).reshape(
                n_freq, n_w_local * n_t_s
            )
            # Time axis in ms covering the kept window range.
            spectro_t0_ms = (
                _WARMUP_S + _CHUNK_DURATION_S + w_lo * _CHUNK_DURATION_S
            ) * 1000.0
            spectro_t_end_ms = (
                spectro_t0_ms + n_w_local * _CHUNK_DURATION_S * 1000.0
            )
            # Anchor colour to GT (NaN-safe); fall back to pred range
            # if GT is entirely absent.
            if np.isfinite(tgt_stitched).any():
                vmin = float(np.nanmin(tgt_stitched))
                vmax = float(np.nanmax(tgt_stitched))
            else:
                vmin = float(np.nanmin(pred_stitched))
                vmax = float(np.nanmax(pred_stitched))

            # Sub-gridspec: GT on top, Pred below; share the x-axis so
            # the time cursor reaches both. tight hspace keeps the cell
            # compact. ax_gt also shares x with the column's anchor (if
            # already set by an earlier TS panel above) so the whole
            # column's time axis stays locked together.
            cell_gs = gs_traces[gr:gr + grs, gc:gc + gcs].subgridspec(
                2, 1, hspace=0.06,
            )
            sharex_anchor = col_anchor_ax.get(gc)
            ax_gt = fig.add_subplot(cell_gs[0], sharex=sharex_anchor)
            ax_pr = fig.add_subplot(cell_gs[1], sharex=ax_gt)
            if sharex_anchor is None:
                col_anchor_ax[gc] = ax_gt
            # In single-side modes hide the irrelevant sub-panel. We
            # still create the imshow object (update() addresses both)
            # but it never renders.
            if mode == "gt":
                ax_pr.set_visible(False)
            elif mode == "pred":
                ax_gt.set_visible(False)
            # The bottommost panel in the column (keeps xlabel + tick
            # labels post-loop) is ax_pr in 'both' / 'pred' modes; in
            # 'gt' mode, ax_pr is hidden so we put the xlabel on ax_gt.
            xlabel_ax = ax_gt if mode == "gt" else ax_pr
            col_panels.setdefault(gc, []).append((gr, xlabel_ax))
            # Empty NaN buffers — update() will fill columns up to the
            # cursor on each frame.
            gt_buf0 = np.full(tgt_stitched.shape, np.nan, dtype=np.float32)
            pr_buf0 = np.full(pred_stitched.shape, np.nan, dtype=np.float32)
            # Convert the freq-bin axis to kHz via the modality's Nyquist
            # frequency. Each kept bin spans (max_freq_khz / n_freq) kHz.
            max_freq_khz = _SPECTRO_MAX_FREQ_KHZ.get(name, 250.0)
            im_gt = ax_gt.imshow(
                gt_buf0, cmap="viridis", vmin=vmin, vmax=vmax,
                aspect="auto", origin="lower",
                extent=(spectro_t0_ms, spectro_t_end_ms, 0, max_freq_khz),
            )
            im_pr = ax_pr.imshow(
                pr_buf0, cmap="viridis", vmin=vmin, vmax=vmax,
                aspect="auto", origin="lower",
                extent=(spectro_t0_ms, spectro_t_end_ms, 0, max_freq_khz),
            )
            ax_gt.set_title(label)
            # Joint y-label centered between ax_gt + ax_pr, placed via
            # the cell's SubplotSpec bbox so it doesn't collide with
            # either sub-panel's tick labels. Single label spans both
            # rows = no duplication, much cleaner read. The 0.026
            # offset (= ~28 pt at the 18" figure width) mirrors the
            # default labelpad spacing used by the TS panels above:
            # leaves room for the widest tick label ("200") plus a few
            # points of breathing space before the ylabel.
            cell_bbox = gs_traces[gr:gr + grs, gc:gc + gcs].get_position(fig)
            fig.text(
                cell_bbox.x0 - 0.026,
                0.5 * (cell_bbox.y0 + cell_bbox.y1),
                "Frequency (kHz)",
                rotation=90, ha="center", va="center",
                fontsize=11,
            )
            ax_gt.tick_params(labelbottom=False)
            # Sparse y-ticks every 100 kHz (0/100/200 for 250-kHz
            # Nyquist) — each spectro sub-panel is only ~1" tall once
            # the trace grid is divided 3-ways, so 6 ticks would
            # overlap regardless of font size.
            _y_ticks_khz = np.arange(0.0, max_freq_khz + 1e-3, 100.0)
            for _ax in (ax_gt, ax_pr):
                _ax.set_yticks(_y_ticks_khz)
                _ax.tick_params(axis="y", labelsize=8, pad=2)
            # Inline corner badges — white text on black bbox.
            _badge_bbox = dict(boxstyle="round,pad=0.2", fc="black",
                               alpha=0.75)
            # In single-side modes, only one badge is meaningful.
            if mode != "pred":
                ax_gt.text(0.02, 0.92, "GT", transform=ax_gt.transAxes,
                           fontsize=10, color="white", va="top", ha="left",
                           bbox=_badge_bbox)
            if mode != "gt":
                ax_pr.text(0.02, 0.92, "model", transform=ax_pr.transAxes,
                           fontsize=10, color="white", va="top", ha="left",
                           bbox=_badge_bbox)
            spectro_panels.append({
                "im_gt": im_gt, "im_pr": im_pr,
                "tgt": tgt_stitched.astype(np.float32),
                "pred": pred_stitched.astype(np.float32),
                "n_t": n_t_s,
                "n_freq": n_freq,
            })
        else:
            # Unknown kind or modality absent from blobs — keep a small
            # placeholder so the grid stays consistent. Don't share the
            # column anchor (placeholders have no real time axis) and
            # don't register in col_panels so the bottom-panel x-label
            # logic keeps targeting a real-data panel.
            ax = fig.add_subplot(
                gs_traces[gr : gr + grs, gc : gc + gcs]
            )
            ax.text(
                0.5, 0.5,
                f"{label} — no data",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=11, color="#888888", style="italic",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#dddddd")
        lines_gt.append(row_gt)
        lines_pred.append(row_pred)
        cursors.append(row_cursor)

    # Shared-x axis cleanup: keep xlabel + tick labels only on the
    # bottom-most panel of each column. All other panels in the column
    # hide their tick labels (sharex already keeps their range in
    # lock-step) and drop the xlabel.
    for gc, panels in col_panels.items():
        panels.sort(key=lambda x: x[0])
        for i, (gr, ax) in enumerate(panels):
            is_bottom = (i == len(panels) - 1)
            if is_bottom:
                ax.set_xlabel("Time (ms)")
                ax.tick_params(labelbottom=True)
            else:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)

    title_obj = fig.suptitle("")

    # ── Animation update ────────────────────────────────────────────
    def update(frame_idx: int):
        # w_local indexes into the sliced [w_lo, w_hi) range; w_global
        # is the original window index (used only for the wall-clock
        # cursor time displayed on traces).
        w_local = min(frame_idx * stride, n_windows - 1)
        w_global = w_lo + w_local
        t_cur_ms = _window_end_time_ms(w_global)
        n_samples_revealed = min((w_local + 1) * n_samples, t_total_samples)

        artists: List = [title_obj]

        # Time traces — variable-length per row (4 channels for slow_ts,
        # 8 for fast_ts, 0 for spectro placeholder).
        for r in range(len(row_spec)):
            row_lg = lines_gt[r]
            row_lp = lines_pred[r]
            row_cu = cursors[r]
            for lg, lp in zip(row_lg, row_lp):
                t_arr, gt_arr = lg.set_array_data_local
                _,      pr_arr = lp.set_array_data_local
                lg.set_data(t_arr[:n_samples_revealed], gt_arr[:n_samples_revealed])
                lp.set_data(t_arr[:n_samples_revealed], pr_arr[:n_samples_revealed])
                artists.append(lg)
                artists.append(lp)
            for cu in row_cu:
                cu.set_xdata([t_cur_ms, t_cur_ms])
                artists.append(cu)

        # Spectrogram rolling heatmaps — reveal columns up to the cursor.
        # We rebuild a NaN buffer each frame (cheap relative to model
        # inference and the matplotlib draw itself) and copy the
        # revealed slab from the precomputed stitched arrays.
        for sp in spectro_panels:
            n_t = int(sp["n_t"])
            tgt = sp["tgt"]
            pred = sp["pred"]
            n_cols = min((w_local + 1) * n_t, tgt.shape[1])
            gt_buf = np.full(tgt.shape, np.nan, dtype=np.float32)
            pr_buf = np.full(pred.shape, np.nan, dtype=np.float32)
            gt_buf[:, :n_cols] = tgt[:, :n_cols]
            pr_buf[:, :n_cols] = pred[:, :n_cols]
            sp["im_gt"].set_data(gt_buf)
            sp["im_pr"].set_data(pr_buf)
            artists.append(sp["im_gt"])
            artists.append(sp["im_pr"])

        # Video frames at sliced window w_local, last frame of that window.
        if has_video:
            fi = vp.shape[2] - 1
            for c, mc in enumerate(video_model_chs):
                gt_im = vt[w_local, mc, fi]
                pr_im = vp[w_local, mc, fi]
                diff_im = np.abs(gt_im - pr_im)
                col_imgs = {0: gt_im, 1: pr_im, 2: diff_im}
                for col_idx, col in enumerate(active_cols):
                    video_ims[c][col_idx].set_data(col_imgs[col])
                artists.extend(video_ims[c])

        title_obj.set_text(
            f"shot {shot_id}   •   t = {t_cur_ms / 1000:.3f} s   •   "
            f"window {w_local + 1}/{n_windows}"
        )
        return artists

    def init():
        return update(0)

    ani = animation.FuncAnimation(
        fig, update, frames=n_anim_frames,
        init_func=init, blit=True, interval=1000 / fps,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        writer = animation.FFMpegWriter(fps=fps, bitrate=2400)
        ani.save(str(out_path), writer=writer, dpi=dpi)
        logger.info(f"saved {out_path}  ({n_anim_frames} frames @ {fps} fps)")
    except Exception as e:
        gif_path = out_path.with_suffix(".gif")
        logger.warning(
            f"ffmpeg writer failed ({e}); falling back to GIF → {gif_path}"
        )
        ani.save(str(gif_path), writer="pillow", fps=fps, dpi=dpi)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data_dir", type=Path, required=True)
    p.add_argument("--stats_path", type=Path, required=True)
    p.add_argument("--shot_id", type=int, required=True)
    p.add_argument(
        "--output_dir", type=Path, default=Path("eval_runs/animations"),
        help="Where the resulting <shot_id>_animation.mp4 lands.",
    )
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--chunk_duration_s", type=float, default=0.05)
    p.add_argument("--step_size_s", type=float, default=0.01)
    p.add_argument("--warmup_s", type=float, default=1.0)
    p.add_argument(
        "--K", type=int, default=0,
        help="Rollout horizon. 0 (default) autodetects from checkpoint.",
    )
    p.add_argument(
        "--fps", type=int, default=8,
        help="Playback frame-rate. Default 8 ≈ 8 windows/sec wall = 6× "
             "slowed down vs real shot time (50 ms / window → 125 ms / "
             "frame). Lower for slower motion, higher for faster.",
    )
    p.add_argument(
        "--stride", type=int, default=1,
        help="Animation steps per window. Default 1 = one frame per "
             "rollout window (50 ms per frame at chunk=0.05s).",
    )
    p.add_argument("--dpi", type=int, default=140)
    p.add_argument(
        "--t_start_s", type=float, default=1.0,
        help="Time-range start (seconds since shot t=0). Animation only "
             "covers windows whose end-time falls in [t_start_s, t_end_s].",
    )
    p.add_argument(
        "--t_end_s", type=float, default=4.5,
        help="Time-range end (seconds since shot t=0). Default 4.5 s — "
             "covers the active phase of most shots without dragging "
             "into the long flat tail.",
    )
    p.add_argument(
        "--mode", choices=("both", "gt", "pred"), default="both",
        help="Animation content. 'both' (default) shows GT and model side "
             "by side. 'gt' shows only ground truth (TS: only GT lines; "
             "spectro: only GT sub-panel; video: only GT column). 'pred' "
             "shows only model predictions. Useful for presentation slides "
             "where the comparison panel is distracting.",
    )
    p.add_argument(
        "--video_smooth_sigma", type=float, default=1.5,
        help="Inference-time Gaussian smoothing sigma (in pixels) applied "
        "to the PREDICTED video over the (H, W) spatial dims. Mitigates "
        "the per-patch reconstruction discontinuity at the 12×12 pixel "
        "grid (visible especially with Stage 2 models, where K-step "
        "rollout amplifies token noise → patch-boundary checkerboard). "
        "Default 1.5 ≈ 1/8 of a 12-pixel patch — blends boundaries "
        "without losing plasma features. 0 disables. GT is never "
        "smoothed so the visual comparison stays honest.",
    )
    p.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    # Per-row modality override only. Channel selection is auto-picked
    # from top variance (matches Phase 3 stitched style); no per-row
    # channel CLI args needed.
    for row_idx, (mod, _, _, _) in enumerate(_DEFAULT_ROWS):
        p.add_argument(
            f"--row{row_idx}_modality", type=str, default=mod,
            help=f"Modality name for trace row {row_idx} (default {mod}).",
        )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    device = torch.device(args.device)

    # ── Load checkpoint ─────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    diagnostics = [DiagnosticConfig(**d) for d in ckpt["diagnostics"]]
    actuators = [ActuatorConfig(**a) for a in ckpt["actuators"]]
    ck_args = ckpt["args"]
    # `video_seam_refine=True` keeps the eval-side architecture aligned
    # with Stage 2 checkpoints saved after the 2026-06-08 refine_block
    # addition. For older Stage 1 checkpoints without the refine_block
    # keys, load_checkpoint_with_refine_tolerance permits them missing
    # and the zero-init residual produces bit-identical output.
    model = E2EFoundationModel(
        diagnostics=diagnostics, actuators=actuators,
        d_model=ck_args["d_model"], n_heads=ck_args["n_heads"],
        n_layers=ck_args["n_layers"], dropout=0.0,
        video_seam_refine=True,
        spectro_seam_refine=True,
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

    # ── Re-infer the single shot ────────────────────────────────────
    file_path = args.data_dir / f"{args.shot_id}_processed.h5"
    if not file_path.exists():
        raise SystemExit(f"shot file not found: {file_path}")
    blobs = collect_shot_predictions(
        model=model, file_path=file_path, device=device,
        args=args, stats=stats, K=K,
    )

    # ── Resolve per-row modality into the spec, preserving the
    #    default grid position (row, col, rowspan, colspan).
    diag_lookup = {c.name: c for c in diagnostics}
    row_spec: List[Tuple[str, str, str, Tuple[int, int, int, int]]] = []
    for row_idx, (_default_mod, label_default, _, gridpos) in enumerate(_DEFAULT_ROWS):
        mod_name = getattr(args, f"row{row_idx}_modality")
        if mod_name in diag_lookup:
            kind = diag_lookup[mod_name].kind
            label = label_default if mod_name == _default_mod else mod_name
        else:
            kind = "spectrogram"  # fallback if unknown
            label = mod_name
        row_spec.append((mod_name, label, kind, gridpos))

    # Suffix the filename with the mode so three side-by-side runs
    # (both / gt / pred) don't clobber each other.
    _mode_suffix = "" if args.mode == "both" else f"_{args.mode}"
    out_path = args.output_dir / f"{args.shot_id}_animation{_mode_suffix}.mp4"
    build_animation(
        blobs=blobs, row_spec=row_spec, stats=stats,
        out_path=out_path,
        shot_id=args.shot_id,
        fps=args.fps, stride=args.stride, dpi=args.dpi,
        t_start_s=args.t_start_s, t_end_s=args.t_end_s,
        video_smooth_sigma=args.video_smooth_sigma,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
