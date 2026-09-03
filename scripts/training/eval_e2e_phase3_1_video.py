"""Stage-1 evaluation — Phase 3.1: video stitched grid + mp4.

Companion to ``eval_e2e_stage1_phase3_stitched.py`` (which handles
TS / spectrogram). For each top/bottom-N selected shot that has video
modalities, produce two deliverables per shot:

  1. **5×6 grid PNG** per stitched segment — up to 30 (GT, model)
     frame pairs taken every ``_STITCHED_FRAME_STRIDE``-th frame
     (currently 10) from the segment, GT on top of each cell, model
     below, time-of-frame in each cell title. One PNG per
     (shot, segment). The grid PNG visualises a single channel
     (``_VIDEO_CHANNEL``).
     Filename: ``<shot_id>_stitched_<seg>_grid.png``.
  2. **MP4 per shot** — one continuous video over the full shot
     (every window, no segment subsampling or separators) at native
     60 fps (tangtv has 3 frames per 50 ms window). Each frame is a
     **2×3 grid**: rows = channels, cols = GT / model / |GT − model|.
     Filename: ``<shot_id>.mp4``.

Plan §10 Q8 decisions baked in:
  - One mp4 per shot covering the whole shot end-to-end.
  - Native 60 fps.
  - 2×3 layout per frame (rows = channels, cols = GT/model/|diff|).
  - libx264 codec via the imageio-ffmpeg bundled binary
    (no OS-level ffmpeg dependency).

Per-channel intensity ranges are computed independently across the
whole shot so each channel keeps its native contrast (channels can
have very different scales). The diff column uses magma on a
per-channel max so faint errors stay visible.

Run::

    pixi run python scripts/training/eval_e2e_stage1_phase3_1_video.py \\
        --output_dir eval_runs/stage1_phase1_e2e_stage1_best_4609988 \\
        --checkpoint /lustre/orion/fus187/proj-shared/models/e2e_stage1/e2e_stage1_best.pt \\
        --data_dir   /lustre/orion/fus187/proj-shared/foundation_model \\
        --stats_path /lustre/orion/fus187/proj-shared/foundation_model_meta/preprocessing_stats.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
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

# Sibling-import Phase 0 / Phase 1 / Phase 2.1 / Phase 3.0 helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_e2e import (  # type: ignore[import]  # noqa: E402
    _video_standardize_per_bc,
    detect_stage_K,
    forward_one_batch,
    load_checkpoint_with_refine_tolerance,
    make_rollout_if_needed,
    rollout_forward_one_batch,
)
from eval_e2e_phase2_per_shot import (  # type: ignore[import]  # noqa: E402
    _coverage_aware_shot_order,
)
from eval_e2e_phase3_stitched import (  # type: ignore[import]  # noqa: E402
    _CHUNK_DURATION_S,
    _STEP_SIZE_S,
    _STITCH_STRIDE,
    _DEFAULT_SEG_WINDOWS,
    _SEG_FRACTIONS,
    _WARMUP_S,
    compute_segment_ranges,
)

logger = logging.getLogger("eval_stage1_phase3_1_video")


# ─────────────────────────────────────────────────────────────────────
# Style + encoder config
# ─────────────────────────────────────────────────────────────────────

_VIDEO_CHANNEL = 0          # channel used by the 5×6 grid PNG only;
                            # the mp4 renders all channels in a 2×3 grid.
_GRID_ROWS = 5
_GRID_COLS = 6
_STITCHED_FRAME_STRIDE = 10 # grid takes frames 0, 10, 20, ... from the
                            # segment (drop unused cells if fewer than
                            # _GRID_ROWS × _GRID_COLS frames remain).
_MP4_FPS = 60               # tangtv has 3 frames per 50 ms window =>
                            # native = 1 / (0.05 / 3) = 60 fps.
_MP4_CODEC = "libx264"
_FRAMES_PER_WINDOW = 3      # tangtv-specific; matches multimodal.py.


def _video_display_rows(n_model_channels: int):
    """Rows to render for a tangtv video, as ``(model_channel, label,
    flip)`` tuples.

    NEW 7-channel model (model ch i == raw ch i): show model ch2 (lower
    divertor = LODIV_240RM1:PERP) + ch4 (upper divertor = UPDIV_0RP1:PERP),
    NO flip on either.

    OLD (<= 2 channel) model: render every model channel as before —
    "channel 0", "channel 1", ... — with the channel-1 180° flip kept for
    backward compatibility.
    """
    if n_model_channels >= 5:
        return [(2, "Lower Divertor", False), (4, "Upper Divertor", False)]
    return [(c, f"channel {c}", c == 1) for c in range(n_model_channels)]


# ─────────────────────────────────────────────────────────────────────
# Per-shot video re-inference
# ─────────────────────────────────────────────────────────────────────


@torch.no_grad()
def collect_full_video_for_shot(
    model: E2EFoundationModel,
    file_path: Path,
    device: torch.device,
    args: argparse.Namespace,
    stats: dict,
    K: int,
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], int]:
    """Re-infer one shot with K-step rollout and stash the **final-step
    (k=K)** video predictions for every window. The full sequence
    drives the mp4; the segment-grid renderer slices its 3 sub-ranges
    from the same tensor so we only pay one inference pass per shot.

    For Stage 1 (K=1) this is byte-identical to the pre-unification
    behaviour. For Stage 2 (K>1) every frame in the mp4 is the model's
    K-step rollout output at that window.

    Returns
    -------
    (blobs, n_windows)
        ``blobs[modality_name] = {'pred','target'}`` — both tensors are
        CPU, shape ``(n_windows, n_channels, n_frames, H, W)``.
    """
    diag_names = [c.name for c in model.diagnostics]
    act_names = [c.name for c in model.actuators]
    video_cfgs = [c for c in model.diagnostics if c.kind == "video"]
    if not video_cfgs:
        return {}, 0
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
        return {}, 0

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers,
        drop_last=False, pin_memory=False,
    )

    storage: Dict[str, Dict[str, list]] = {
        c.name: {"pred": [None] * n_windows,
                 "target": [None] * n_windows}
        for c in video_cfgs
    }

    global_idx = 0
    for batch in loader:
        predictions_per_k, diag_initial, targets_per_k, masks_per_k = (
            rollout_forward_one_batch(
                model, rollout, batch, device, K, args.chunk_duration_s
            )
        )
        predictions = predictions_per_k[K - 1]
        targets = targets_per_k[K - 1]
        diag_inputs = diag_initial
        bs = next(iter(diag_inputs.values())).shape[0]
        for j in range(bs):
            w = global_idx + j
            for cfg in video_cfgs:
                n = cfg.name
                pred = predictions[n][j:j+1]   # (1, C, T_frames, H, W)
                tgt = targets[n][j:j+1]
                storage[n]["pred"][w] = pred.detach().cpu()
                storage[n]["target"][w] = tgt.detach().cpu()
        global_idx += bs

    out: Dict[str, Dict[str, torch.Tensor]] = {}
    for n, blob in storage.items():
        preds = [t for t in blob["pred"] if t is not None]
        tgts = [t for t in blob["target"] if t is not None]
        if not preds:
            continue
        out[n] = {
            "pred": torch.cat(preds, dim=0),    # (T_full, C, T_frames, H, W)
            "target": torch.cat(tgts, dim=0),
        }
    return out, n_windows


# ─────────────────────────────────────────────────────────────────────
# Frame normalisation + RGB conversion (for mp4 + grid)
# ─────────────────────────────────────────────────────────────────────


def _normalize_to_uint8(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Map ``arr`` into [0, 255] uint8 using the global GT/model range so
    GT and model are visually comparable across the mp4."""
    span = max(vmax - vmin, 1e-6)
    scaled = np.clip((arr - vmin) / span, 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def _gray_to_rgb(u8: np.ndarray) -> np.ndarray:
    """(H, W) uint8 → (H, W, 3) uint8 (grayscale replicated to RGB)."""
    return np.stack([u8, u8, u8], axis=-1)


def _diff_to_rgb_magma(diff: np.ndarray, vmax: float) -> np.ndarray:
    """(H, W) float → (H, W, 3) uint8 via magma colormap, normalized to
    [0, vmax] for cross-frame consistency."""
    span = max(vmax, 1e-6)
    scaled = np.clip(diff / span, 0.0, 1.0)
    rgba = cm.get_cmap("magma")(scaled)  # (H, W, 4) in [0, 1]
    return (rgba[..., :3] * 255.0).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────
# Static 5×6 grid PNG per (shot, segment)
# ─────────────────────────────────────────────────────────────────────


def _render_video_grid(
    pred_stack: torch.Tensor,
    target_stack: torch.Tensor,
    window_idx_range: Tuple[int, int, int],
    out_path: Path,
    shot_id: int, modality: str, split: str, seg_idx: int,
) -> None:
    """5×6 grid of (GT, model) frame pairs from this segment.

    Takes every ``_STITCHED_FRAME_STRIDE``-th frame from the segment's
    ``T_seg × n_frames`` total frames (capped at ``_GRID_ROWS ×
    _GRID_COLS`` cells; trailing cells are blanked if fewer frames
    remain). Renders only one channel (the multi-channel view lives in the
    mp4): the upper-divertor view — model ch0 for old 2-channel models
    (= ``_VIDEO_CHANNEL``), model ch4 for new 7-channel models.
    """
    # Old 2-ch model: ch0 (= _VIDEO_CHANNEL, upper divertor) — unchanged.
    # 7-ch model: ch4 (upper divertor; ch0 is mostly-NaN metadata).
    grid_ch = 4 if pred_stack.shape[1] >= 5 else _VIDEO_CHANNEL
    p = pred_stack[:, grid_ch].numpy()          # (T_seg, n_frames, H, W)
    t = target_stack[:, grid_ch].numpy()
    t_seg, n_frames, H, W = p.shape
    total_frames = t_seg * n_frames
    if total_frames == 0:
        return

    # Take every _STITCHED_FRAME_STRIDE-th frame, capped at the grid size.
    n_cells = _GRID_ROWS * _GRID_COLS
    indices = np.arange(0, total_frames, _STITCHED_FRAME_STRIDE)[:n_cells]

    # Global intensity range across this segment for consistent display.
    # NaN-safe so a missing GT (all-NaN target) doesn't break the range —
    # NaN values in the GT half of each cell propagate through imshow as
    # blank (bg-coloured) pixels, which is exactly what we want when no
    # ground truth is available.
    arrs = [p, t] if np.isfinite(t).any() else [p]
    vmin = float(min(np.nanmin(a) for a in arrs))
    vmax = float(max(np.nanmax(a) for a in arrs))
    if not (np.isfinite(vmin) and np.isfinite(vmax)):
        return

    start_w, _end_w, stride = window_idx_range
    dt_frame_s = _CHUNK_DURATION_S / n_frames

    fig, axes = plt.subplots(_GRID_ROWS, _GRID_COLS,
                             figsize=(_GRID_COLS * 2.4, _GRID_ROWS * 2.4))
    for cell_idx in range(_GRID_ROWS * _GRID_COLS):
        ax = axes[cell_idx // _GRID_COLS][cell_idx % _GRID_COLS]
        if cell_idx >= len(indices):
            ax.axis("off")
            continue
        frame_idx = indices[cell_idx]
        wi = frame_idx // n_frames
        fi = frame_idx % n_frames

        window_global = start_w + wi * stride
        t_s = (
            _WARMUP_S + _CHUNK_DURATION_S
            + window_global * _STEP_SIZE_S
            + fi * dt_frame_s
        )

        gt = t[wi, fi]
        pr = p[wi, fi]
        # Stack GT (top) above model (bottom).
        combined = np.vstack([gt, pr])
        ax.imshow(combined, cmap="gray", vmin=vmin, vmax=vmax,
                  aspect="auto", interpolation="nearest")
        # Divider between GT and model.
        ax.axhline(H - 0.5, color="tab:red", linewidth=0.8)
        ax.set_title(f"t = {t_s:.2f} s", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    span_s = t_seg * _CHUNK_DURATION_S
    t0_label = _WARMUP_S + _CHUNK_DURATION_S + start_w * _STEP_SIZE_S
    fig.suptitle(
        f"shot {shot_id} — {modality} (video, ch {grid_ch}) — "
        f"split: {split}   |   segment {seg_idx} (every "
        f"{_STITCHED_FRAME_STRIDE}-th frame, {len(indices)} pairs from "
        f"t = {t0_label:.2f}–{t0_label + span_s:.2f} s; "
        f"GT above, model below in each cell)",
        fontsize=10, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# MP4 per shot — concatenated 3-panel (GT | model | |diff|) at 60 fps
# ─────────────────────────────────────────────────────────────────────


def _render_video_mp4(
    pred: torch.Tensor,
    target: torch.Tensor,
    modality: str,
    out_path: Path,
    shot_id: int, split: str,
) -> None:
    """One continuous mp4 covering the full shot — every window, no
    segment subsampling or separators.

    Frame layout per timestep — **n_channels × 3 labeled grid** rendered
    via matplotlib so every panel carries proper annotations:

      ╭────────────────┬──────────────┬──────────────┬───────────────╮
      │                │ Ground truth │  Predicted   │ |GT − Predicted| │
      ├────────────────┼──────────────┼──────────────┼───────────────┤
      │ channel 0      │   <gt_0>     │   <pred_0>   │    <diff_0>   │
      ├────────────────┼──────────────┼──────────────┼───────────────┤
      │ channel 1      │   <gt_1>     │   <pred_1>   │    <diff_1>   │
      ╰────────────────┴──────────────┴──────────────┴───────────────╯
      suptitle: "shot <id> (<split>)  •  t = <X.XXX> s"

    Per-channel intensity ranges (and per-channel diff max) are computed
    across the full shot so colour mapping stays consistent throughout.
    GT + model are gray; |diff| is magma. Native 60 fps.
    """
    if pred.numel() == 0:
        return
    _t, n_model_ch, _nf, H, W = pred.shape

    # DISPLAY rows — old 2-channel models render every model channel
    # ("channel 0", "channel 1", ...); new 7-channel models render the two
    # divertor views (model ch2 lower + ch4 upper). Rows index by display
    # position; data is pulled from the row's model channel.
    display_rows = _video_display_rows(n_model_ch)
    n_ch = len(display_rows)           # number of DISPLAY rows
    row_model_chs = [mc for mc, _, _ in display_rows]
    row_labels = [lbl for _, lbl, _ in display_rows]
    row_flips = [fl for _, _, fl in display_rows]
    if n_ch == 0:
        return

    # Per-display-row intensity scale + diff max across the whole shot
    # (pulled from the matching model channel).
    # NaN-safe: when GT for a channel is entirely missing (all-NaN), we
    # fall back to the prediction range and leave the diff colour-bar at
    # a sentinel. NaN values propagate through set_data so the GT and
    # |diff| panels render blank (bg-coloured) automatically.
    p_all = pred.numpy()
    t_all = target.numpy()
    g_min = np.full(n_ch, +np.inf, dtype=np.float64)
    g_max = np.full(n_ch, -np.inf, dtype=np.float64)
    d_max = np.zeros(n_ch, dtype=np.float64)
    for c, mc in enumerate(row_model_chs):
        p_c = p_all[:, mc]
        t_c = t_all[:, mc]
        if np.isfinite(t_c).any():
            g_min[c] = float(min(np.nanmin(p_c), np.nanmin(t_c)))
            g_max[c] = float(max(np.nanmax(p_c), np.nanmax(t_c)))
            d_max[c] = float(np.nanmax(np.abs(t_c - p_c)))
        else:
            g_min[c] = float(np.nanmin(p_c))
            g_max[c] = float(np.nanmax(p_c))
            d_max[c] = 1.0   # diff panel stays all-NaN, colour-bar unused.
    if not np.isfinite(g_min).all():
        return

    # ── Build the matplotlib figure once; update imshow data per frame ──
    col_titles = ["Ground truth", "Predicted", "|GT − Predicted|"]
    # figsize chosen so each panel ends up close to native 120×360 (3:1
    # wide aspect): 3 cols × ~3.6 in + label margin ≈ 12 in wide;
    # n_ch rows × 1.2 in + title margin per row.
    fig, axes = plt.subplots(
        n_ch, 3,
        figsize=(12, 1.4 * n_ch + 1.0),
        constrained_layout=True,
    )
    if n_ch == 1:                          # axes is 1D when n_ch == 1
        axes = np.array([axes])
    ims: List[List] = [[None, None, None] for _ in range(n_ch)]
    for c in range(n_ch):
        for col in range(3):
            ax = axes[c, col]
            if c == 0:
                ax.set_title(col_titles[col], fontsize=10)
            if col == 0:
                ax.set_ylabel(row_labels[c], fontsize=10)
            cmap = "gray" if col < 2 else "magma"
            vmin = 0.0 if col == 2 else g_min[c]
            vmax = d_max[c] if col == 2 else g_max[c]
            ims[c][col] = ax.imshow(
                np.zeros((H, W)), cmap=cmap, vmin=vmin, vmax=vmax,
                aspect="equal", interpolation="nearest",
            )
            ax.set_xticks([])
            ax.set_yticks([])
    suptitle = fig.suptitle("", fontsize=11)
    fig.canvas.draw()                      # finalize layout before grabbing size

    def _grab_rgb() -> np.ndarray:
        """Render current figure state to an (H, W, 3) uint8 array."""
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        return buf.copy()

    frames: List[np.ndarray] = []
    t_full, _ch, n_frames, _h, _w = p_all.shape
    dt_frame_s = _CHUNK_DURATION_S / n_frames

    for wi in range(t_full):
        for fi in range(n_frames):
            t_s = (
                _WARMUP_S + _CHUNK_DURATION_S
                + wi * _STEP_SIZE_S
                + fi * dt_frame_s
            )
            for c, mc in enumerate(row_model_chs):
                gt = t_all[wi, mc, fi]
                pr = p_all[wi, mc, fi]
                if row_flips[c]:
                    # OLD-model channel 1 is rotated 180° vs channel 0 (not
                    # just horizontally mirrored), so flip BOTH axes — H and
                    # W — before display. See project-tangtv-channel1-flip.
                    # 7-channel models set no flip.
                    gt = gt[::-1, ::-1]
                    pr = pr[::-1, ::-1]
                ims[c][0].set_data(gt)
                ims[c][1].set_data(pr)
                ims[c][2].set_data(np.abs(gt - pr))
            suptitle.set_text(
                f"shot {shot_id} ({split}) • {modality}  "
                f"•  t = {t_s:.3f} s"
            )
            frames.append(_grab_rgb())

    plt.close(fig)
    if not frames:
        return

    # libx264 needs frame dims divisible by 2 — pad to even if needed.
    fh, fw, _ = frames[0].shape
    new_h, new_w = fh + (fh % 2), fw + (fw % 2)
    if (new_h, new_w) != (fh, fw):
        padded = []
        for f in frames:
            f = np.pad(
                f,
                ((0, new_h - f.shape[0]), (0, new_w - f.shape[1]), (0, 0)),
                mode="constant",
            )
            padded.append(f)
        frames = padded

    out_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(
        out_path,
        np.stack(frames, axis=0),
        fps=_MP4_FPS,
        codec=_MP4_CODEC,
        macro_block_size=1,
    )


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
        help="Cap unique shots. 0 = all top/bottom-selected. "
             "Coverage-aware ordering still applied.",
    )
    p.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--skip_mp4", action="store_true",
        help="Produce only the static 5×6 grid PNGs; skip mp4 encoding "
             "(useful for very-fast smoke runs).",
    )
    p.add_argument(
        "--K", type=int, default=0,
        help="Rollout horizon. 0 (default) autodetects from checkpoint "
             "(K=1 for Stage 1, K=K_max for Stage 2). Frames render the "
             "k=K (final rollout) prediction.",
    )
    p.add_argument(
        "--only_shots", type=int, nargs="+", default=None,
        help="Restrict processing to these shot IDs only. Overrides the "
             "default 'every shot' iteration. Useful for quick targeted "
             "re-renders (e.g. verify a fix on a single shot).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    device = torch.device(args.device)
    plots_root = args.output_dir / args.plots_subdir

    # top_bottom_shots.csv.gz is no longer a filter. Phase 3.1 iterates
    # every shot in the val set and emits video output (grid + mp4) for
    # every model.diagnostics modality with kind="video". per_window
    # provides the canonical shot/split list.
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

    # Every shot in the val set + every video diagnostic in the model.
    video_diags = [c.name for c in model.diagnostics if c.kind == "video"]
    if not video_diags:
        logger.warning("No video diagnostics in this checkpoint; nothing to plot.")
        return
    shot_split: Dict[int, str] = (
        per_window.drop_duplicates("shot_id")[["shot_id", "split"]]
                  .set_index("shot_id")["split"].to_dict()
    )
    all_shots = sorted(shot_split.keys())
    if args.only_shots:
        only_set = set(args.only_shots)
        all_shots = [s for s in all_shots if s in only_set]
        logger.info(f"--only_shots filter: {sorted(only_set)} → {len(all_shots)} matched")
    if args.max_shots_to_plot and args.max_shots_to_plot > 0:
        all_shots = all_shots[: args.max_shots_to_plot]
    logger.info(
        f"Phase 3.1 video — plotting {len(all_shots)} shots × "
        f"{len(video_diags)} video modalities"
        + (f" (mp4 disabled via --skip_mp4)" if args.skip_mp4 else "")
    )

    for i, shot_id in enumerate(all_shots, start=1):
        file_path = args.data_dir / f"{shot_id}_processed.h5"
        if not file_path.exists():
            logger.warning(f"shot {shot_id}: file missing at {file_path}")
            continue
        split = shot_split[shot_id]
        logger.info(f"({i}/{len(all_shots)}) shot {shot_id} ({split}): video re-inference …")
        full_blobs, n_windows = collect_full_video_for_shot(
            model=model, file_path=file_path, device=device,
            args=args, stats=stats, K=K,
        )
        if not full_blobs:
            continue
        seg_ranges = compute_segment_ranges(n_windows)

        # All video diagnostics for this shot — NaN-aware renderers
        # handle missing GT gracefully.
        for modality in video_diags:
            if modality not in full_blobs:
                continue
            out_dir = plots_root / split / modality
            pred_full = full_blobs[modality]["pred"]
            target_full = full_blobs[modality]["target"]

            # 5×6 grids — one per segment, sliced from the full-shot tensor.
            for seg_idx, start, end, stride in seg_ranges:
                pred_seg = pred_full[start:end:stride]
                target_seg = target_full[start:end:stride]
                if pred_seg.shape[0] == 0:
                    continue
                grid_path = out_dir / f"{shot_id}_stitched_{seg_idx}_grid.png"
                _render_video_grid(
                    pred_stack=pred_seg,
                    target_stack=target_seg,
                    window_idx_range=(start, end, stride),
                    out_path=grid_path,
                    shot_id=shot_id, modality=modality,
                    split=split, seg_idx=seg_idx,
                )
                logger.info(f"  → {grid_path.relative_to(args.output_dir)}")

            # MP4 — one continuous video over the whole shot.
            if not args.skip_mp4:
                mp4_path = out_dir / f"{shot_id}.mp4"
                _render_video_mp4(
                    pred=pred_full, target=target_full,
                    modality=modality, out_path=mp4_path,
                    shot_id=shot_id, split=split,
                )
                logger.info(f"  → {mp4_path.relative_to(args.output_dir)}")

    logger.info("Phase 3.1 (video grid + mp4) complete.")


if __name__ == "__main__":
    main()
