"""Tokamak-themed animation layout — step-by-step build.

Step 1: static 16:9 figure framework. Two tokamak PNGs placed in the
middle two columns (digital twin = pred side on the left; reactor =
GT side on the right). Outer two columns reserved (empty) for the
spectrogram panels. No cams, no traces yet.

Content alignment:
  * Both PNGs have asymmetric padding (content flush against the top
    of the bbox, blank rows at the bottom). Auto-detect the content
    bbox via alpha (twin: RGBA) / luminance+chroma (reactor: RGB).
  * TWIN: keep displayed size unchanged; shift via imshow `extent`
    so the visible vessel is vertically centered in the axes.
  * REACTOR: crop to its content bbox, then size its column so the
    rendered content height equals the twin's rendered content
    height. Anchor=C centers it vertically in the panel.

Output: eval_runs/animations/_tokamak_layout_step1.png
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import torch

# Inference helpers live in the sibling legacy animation script so
# both renderers share exactly the same forward-pass + window-stitching
# logic. Path-insert so the module is importable when this script
# is invoked from outside scripts/training/.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_e2e_animation import (  # type: ignore[import]  # noqa: E402
    _CHUNK_DURATION_S,
    _WARMUP_S,
    _denormalize_slow_ts,
    _ts_time_axis_ms,
    collect_shot_predictions,
)
from eval_e2e import (  # type: ignore[import]  # noqa: E402
    detect_stage_K,
    load_checkpoint_with_refine_tolerance,
)
from tokamak_foundation_model.e2e.lora import apply_lora_to_backbone  # noqa: E402
from tokamak_foundation_model.e2e.model import (  # noqa: E402
    ActuatorConfig,
    DiagnosticConfig,
    E2EFoundationModel,
)

# imageio-ffmpeg ships its own ffmpeg binary; matplotlib's default
# search for a system ffmpeg fails on Frontier compute nodes.
try:
    from imageio_ffmpeg import get_ffmpeg_exe as _get_ffmpeg_exe
    plt.rcParams["animation.ffmpeg_path"] = _get_ffmpeg_exe()
except Exception:
    pass

# Seaborn "talk" context — presentation-grade font sizing. Reproduced
# from seaborn/rcmod.py (font_scale=1.3 over its `base_context`)
# rather than importing seaborn, which isn't in the pixi env. These
# rcParams put the whole figure in presentation-readable proportions
# without forcing a new dependency.
_SEABORN_TALK_RC = {
    "font.size":         15.6,
    "axes.labelsize":    15.6,
    "axes.titlesize":    15.6,
    "xtick.labelsize":   14.3,
    "ytick.labelsize":   14.3,
    "legend.fontsize":   14.3,
    "legend.title_fontsize": 15.6,
    "axes.linewidth":    1.625,
    "grid.linewidth":    1.3,
    "lines.linewidth":   2.275,
    "lines.markersize":  9.1,
    "patch.linewidth":   1.3,
    "xtick.major.width": 1.625,
    "ytick.major.width": 1.625,
    "xtick.minor.width": 1.3,
    "ytick.minor.width": 1.3,
    "xtick.major.size":  7.8,
    "ytick.major.size":  7.8,
    "xtick.minor.size":  5.2,
    "ytick.minor.size":  5.2,
}
plt.rcParams.update(_SEABORN_TALK_RC)

# Nature-style rcParams for the static --comparison_figure render. Applied
# ONLY inside a `with plt.rc_context(_FIGURE_RC)` block (see
# _render_comparison_figure) so it never perturbs the presentation
# animation, which keeps the _SEABORN_TALK_RC sizing above.
_FIGURE_RC = {
    "pdf.fonttype":      42,    # embed TrueType, not Type 3 (journal-safe)
    "ps.fonttype":       42,
    "svg.fonttype":      "none",
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size":         8.0,
    "axes.labelsize":    8.0,
    "axes.titlesize":    8.0,
    "xtick.labelsize":   7.0,
    "ytick.labelsize":   7.0,
    "legend.fontsize":   7.0,
    "axes.linewidth":    0.6,
    "lines.linewidth":   1.0,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "legend.frameon":    False,
    "figure.dpi":        150,
    "savefig.dpi":       600,
}


# New PNGs (2026 × 1350, aspect 1.5 = 3:2 portrait) — designed as
# the LEFT and RIGHT halves of a complete tokamak cross-section, so
# they sit flush against each other in the figure with no middle
# gutter. LEFT half = fusion-reactor render (= GT side); RIGHT half
# = digital-twin render (= predictions side).
_PNG_REACTOR = Path("eval_runs/animations/tokamak_left_half_ai.png")   # LEFT, GT
_PNG_TWIN = Path("eval_runs/animations/tokamak_right_half.png")        # RIGHT, pred
# Crop a fraction of each PNG's OUTER side (left edge of reactor,
# right edge of twin) — focuses each half on the inner plasma /
# central-column region instead of the outer vessel walls, and the
# tokamak columns shrink horizontally so the spec columns get
# usable width.
_TOKAMAK_OUTER_CROP_FRAC = 0.20

# Default shot for --shot_id. GT (traces/spectro/video) is loaded from the
# requested shot's own processed H5 (args.data_dir/{shot_id}_processed.h5) —
# the SAME file inference uses — so any shot renders correctly, not just this
# one. (No more hardcoded _SAMPLE_SHOT_FILE.)
_SAMPLE_SHOT = 200729
# Preprocessing stats — log_mean / log_std per channel. Used for the
# same log_standardize transform the dataset applies, so channel-
# ranking variance is computed on normalised data exactly like
# eval_e2e_animation.py does it.
_STATS_PATH = Path(
    "/lustre/orion/fus187/proj-shared/foundation_model_meta/"
    "preprocessing_stats.pt"
)
# Time index into tangtv/ydata (354 frames per shot). Pick something
# in the bright-plasma phase; ch0 peaks around t=150.
_SAMPLE_FRAME_IDX = 150

_FIG_W = 16.0
_FIG_H = 9.0

# Per-side cam transformation. Tune these to align the cam frame
# with the visible upper divertor in each PNG. Six numbers per side:
#   rotation_deg   — CCW rotation applied to the cam image
#   flip_h         — bool, horizontal flip applied AFTER rotation
#   x0, y0         — inset bottom-left corner in axes fraction
#                    (matches inset_axes(): origin = bottom-left)
#   w, h           — inset width / height in axes fraction
# The cam image fills the inset with aspect="equal", so picking
# w / h close to the cam's 3:1 aspect minimises padding around it.
# scale_h / scale_w: non-uniform scaling applied to the cam image
# after rotation+flip+tilt (Photoshop reference: 272% H × 125.8% W).
# Bumping h_frac + lowering y0 keeps the cam centred in the
# upper-divertor area while accommodating the now-much-taller image
# (new image aspect H/W = 0.72, was 0.33).
_CAM_TRANSFORM_REACTOR = {
    "rotation_deg":  0.0,
    "flip_h":        True,
    "tilt_deg":    -32.0,   # depth tilt: positive raises the FRONT
                            # (bottom) edge and foreshortens it
    "scale_h":      2.72,
    "scale_w":      1.258,
    "x0":  0.025,
    "y0":  0.58,
    "w":   0.95,
    "h":   0.37,
    # Elliptical mask (in post-transform normalised image coords):
    # smoothly fades the cam frame to zero outside the ellipse so
    # the rectangular outline doesn't show on the tokamak photo.
    "mask_center_x":    0.50,
    "mask_center_y":    0.50,
    "mask_semi_axis_x": 0.50,
    "mask_semi_axis_y": 0.50,
    "mask_edge_soft":   0.15,
}
_CAM_TRANSFORM_TWIN = {
    "rotation_deg":  0.0,
    "flip_h":        False,
    "tilt_deg":    -32.0,
    "scale_h":      2.72,
    "scale_w":      1.258,
    "x0":  0.025,
    "y0":  0.58,
    "w":   0.95,
    "h":   0.37,
    "mask_center_x":    0.50,
    "mask_center_y":    0.50,
    "mask_semi_axis_x": 0.50,
    "mask_semi_axis_y": 0.50,
    "mask_edge_soft":   0.15,
}

# Time-range slice for the animation (s). Matches the legacy
# animation's default [1.0, 4.5] s window — covers the active
# phase of a typical shot.
_T_START_S = 1.0
_T_END_S = 4.5
# Backward-compat raw-channel reselection for OLD video checkpoints. tangtv has
# 7 raw channels; models trained before 2026-06-22 used 2 of them (raw [4,6] —
# raw 0/1/2/3/5 are largely-NaN metadata). The dataset now defaults to all 7,
# so when evaluating an N-channel checkpoint we reselect the matching legacy
# channels. {movie_name: {model_n_channels: [raw_idx, ...]}}.
_LEGACY_VIDEO_CHANNELS = {"tangtv": {2: [4, 6]}}


def tangtv_display_views(n_model_channels: int):
    """Return the tangtv views to DISPLAY, given how many channels the
    reconstructed model predicts for tangtv.

    Each entry is ``(model_channel, gt_raw_channel, label)``:
      * ``model_channel`` indexes the model's prediction block.
      * ``gt_raw_channel`` indexes raw ``tangtv/ydata`` for the GT load.

    NEW 7-channel model — model ch i == raw ch i — so we show the
    lower divertor (raw/model ch2 = LODIV_240RM1:PERP) and the upper
    divertor (raw/model ch4 = UPDIV_0RP1:PERP). No flip on either.

    OLD (<= 2 channel) model — trained on legacy raw [4,6] (model ch0
    = raw ch4 upper PERP, model ch1 = raw ch6 upper PAR). Backward-compat
    path: a SINGLE upper-divertor view (model ch0 / GT raw ch4), exactly
    as before (the channel-1 flip lives in the legacy renderers).
    """
    if n_model_channels >= 5:
        return [(2, 2, "Lower Divertor"), (4, 4, "Upper Divertor")]
    return [(0, 4, "Upper Divertor")]
# Ground-truth lead-in: show GT from 50 ms before the prediction starts
# (a dashed line at _T_START_S marks where prediction begins).
_GT_LEAD_S = 0.95
# Animation timing — 50 ms per frame (matches eval_e2e_animation's
# _CHUNK_DURATION_S) and 4 fps playback (matches the legacy script's
# default fps).
_DT_FRAME_S = 0.05
_FPS = 4


def content_rows(img: np.ndarray) -> tuple[int, int]:
    """First and last pixel rows that contain visible content.

    Uses alpha for RGBA PNGs; uses luminance + chroma for RGB PNGs
    (treats near-white pixels with no colour as background).
    """
    if img.shape[2] == 4:
        mask = img[..., 3] > 0.05
    else:
        lum = img[..., :3].mean(axis=2)
        chroma = img[..., :3].std(axis=2)
        mask = (lum < 0.95) | (chroma > 0.05)
    rows = mask.any(axis=1)
    top = int(np.argmax(rows))
    bot = int(rows.shape[0] - np.argmax(rows[::-1]) - 1)
    return top, bot


def content_cols(img: np.ndarray) -> tuple[int, int]:
    if img.shape[2] == 4:
        mask = img[..., 3] > 0.05
    else:
        lum = img[..., :3].mean(axis=2)
        chroma = img[..., :3].std(axis=2)
        mask = (lum < 0.95) | (chroma > 0.05)
    cols = mask.any(axis=0)
    left = int(np.argmax(cols))
    right = int(cols.shape[0] - np.argmax(cols[::-1]) - 1)
    return left, right


def detect_divertor_y(
    png: np.ndarray,
    region: str,
    content_top: int = 0,
    content_bot: int | None = None,
    upper_target_frac: float = 0.18,
    lower_target_frac: float = 0.80,
) -> int:
    """Auto-locate the y-pixel coord of the upper or lower divertor in
    a tokamak PNG via peak detection on the horizontal-edge profile,
    snapped to a structural peak closest to a prior-knowledge target
    fraction.

    Image-processing side: ``scipy.signal.find_peaks`` over a
    light-Gaussian-smoothed (σ=3) Sobel row-sum profile gives the
    y-coords of every salient horizontal structure in the PNG
    (vessel walls, divertor tiles, wireframe details, plasma
    boundaries). Without prior knowledge it's ambiguous which of
    these IS the divertor.

    Prior knowledge: in a DIII-D tokamak cross-section, the upper
    divertor sits ~18 % from the top of the visible content and the
    lower divertor ~80 % from the top. We select the structural
    peak whose y-coord is closest to the target fraction. This
    pairs the image's true edge structure with anatomical priors
    so the result is robust to peak-strength noise (avoids snapping
    to the wall outline) while still adapting to the actual PNG.

    Args:
        png: H×W×{3,4} float image, values in [0, 1].
        region: ``"upper"`` or ``"lower"``.
        content_top / content_bot: y-pixel bounds of visible PNG
                content (defaults to full image).
        upper_target_frac, lower_target_frac: target y-fraction
                (within content region) for the respective
                divertor's expected location.

    Returns:
        Pixel y coord (origin top) of the closest structural peak.
    """
    from scipy.signal import find_peaks
    if content_bot is None:
        content_bot = png.shape[0] - 1
    if png.shape[2] == 4:
        gray = png[..., :3].mean(axis=2) * png[..., 3]
    else:
        gray = png[..., :3].mean(axis=2)
    edges = np.abs(ndi.sobel(gray, axis=0))
    row_strength = edges.sum(axis=1)
    smoothed = ndi.gaussian_filter1d(row_strength, sigma=3.0)
    peaks, _ = find_peaks(
        smoothed[content_top : content_bot + 1],
        prominence=smoothed.max() * 0.03,
        distance=20,
    )
    peaks_abs = peaks + content_top
    content_h = content_bot - content_top + 1
    if region == "upper":
        target_y = content_top + int(upper_target_frac * content_h)
    elif region == "lower":
        target_y = content_top + int(lower_target_frac * content_h)
    else:
        raise ValueError(f"region must be 'upper' or 'lower', got {region}")
    if len(peaks_abs) == 0:
        return target_y
    return int(peaks_abs[np.argmin(np.abs(peaks_abs - target_y))])


_TRACE_GROUPS = {
    "Te": "ts_core_temp",
    "ne": "ts_core_density",
    "Ti": "cer_ti",
}
# Spectrogram modalities. STFT params match data_loader's STFT config:
# n_fft=1024, hop_length=256, fs=500 kHz → Nyquist=250 kHz, 513 freq
# bins (we use 512 for symmetry with the dataset's drop-DC convention).
_SPECTRO_GROUPS = {
    "ECE": "ece",
    "CO2": "co2",
}
_SPECTRO_LABELS = {
    "ECE": "ECE",
    "CO2": r"CO$_2$",
}
_STFT_N_FFT = 1024
_STFT_HOP = 256
_STFT_FS = 500_000
# Soft-mask GT-fusion parameters (spec mean-collapse visualization
# workaround — see fuse_spectro_with_gt + the note in main()). Per-modality
# k_threshold: ECE bumped above CO2 because ECE carries more broadband
# background that a lower cutoff lets through as visual noise.
_MASK_K_BY_MOD = {"ECE": 2.5, "CO2": 2.0}
_MASK_GAMMA = 2.0
_MASK_SMOOTH_F = 1.0      # Gaussian σ along freq axis (bins)
_MASK_SMOOTH_T = 2.0      # Gaussian σ along time axis (bins)
# Y-axis labels — matches eval_e2e_animation.py:_PHYS_UNITS so the
# two renderers display the same physical units.
_TRACE_LABELS = {
    "Te": r"$T_e$ (keV)",
    "ne": r"$n_e$ (m$^{-3}$)",
    "Ti": r"$T_i$ (keV)",
}
# Raw → display scale factors. Matches eval_e2e_animation.py:
# _PHYS_SCALE: temperatures get eV → keV (×1e-3); density stays in
# m^-3.
_TRACE_SCALES = {
    "Te": 1e-3,
    "ne": 1.0,
    "Ti": 1e-3,
}


def load_sample_traces(shot_file) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load raw Te / ne / Ti from ``shot_file`` (the SAME processed H5 the
    model runs inference on). Returns {short_name: (xdata_s, ydata_ch_time)}.
    """
    traces = {}
    with h5py.File(shot_file, "r") as f:
        for short, group in _TRACE_GROUPS.items():
            x = f[f"{group}/xdata"][:]
            y = f[f"{group}/ydata"][:]
            traces[short] = (x, y)
    return traces


def log_standardize(
    y: np.ndarray, log_mean: np.ndarray, log_std: np.ndarray,
) -> np.ndarray:
    """Same transform as data_loader.log_standardize. Channel-axis is
    axis 0.
    """
    y_c = np.maximum(y, -0.99)
    y_log = np.log10(y_c + 1.0)
    return (y_log - log_mean[:, None]) / np.maximum(log_std[:, None], 1e-3)


def pick_top_channels(y_norm: np.ndarray, n: int = 3) -> list[int]:
    """Indices of the n highest-variance channels of LOG-STANDARDIZED
    data. Matches eval_e2e_animation.py:589 — variance is computed
    on normalised values (mean ≈ 0, std ≈ 1 per channel), so the
    raw 1e19-scale of n_e never enters the squared sum.
    """
    var = np.nanvar(y_norm, axis=1)
    var = np.where(np.isfinite(var), var, -np.inf)
    nz = np.nonzero(var > -np.inf)[0]
    if len(nz) >= n:
        order = nz[np.argsort(-var[nz])[:n]]
    else:
        order = np.arange(min(n, y_norm.shape[0]))
    return sorted(int(i) for i in order)


def load_and_spectrogram(
    group: str,
    t_start_s: float,
    t_end_s: float,
    shot_file,
    n_fft: int = _STFT_N_FFT,
    hop: int = _STFT_HOP,
    fs: int = _STFT_FS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Load the raw time-series for a spectro modality, pick the
    highest-variance channel, and compute log10(|STFT| + 1).

    Returns ``(freqs_khz, times_ms, log_mag, best_ch)``. Slices the
    raw signal to the chosen time window before reading from H5 so
    we don't pull the entire ~3 M-sample channel into RAM.
    """
    from scipy.signal import spectrogram
    with h5py.File(shot_file, "r") as f:
        x = f[f"{group}/xdata"][:]
        in_range = np.where((x >= t_start_s) & (x <= t_end_s))[0]
        if in_range.size == 0:
            raise SystemExit(f"{group}: no samples in [{t_start_s}, {t_end_s}] s")
        i_lo, i_hi = int(in_range[0]), int(in_range[-1]) + 1
        y_slice = f[f"{group}/ydata"][:, i_lo:i_hi]
    # Channel pick: highest-variance on the raw time-series slice
    # (no log_standardize available for spectro modalities since
    # their stats are over STFT magnitude, not the raw signal).
    var = np.nanvar(y_slice.astype(np.float64), axis=1)
    var = np.where(np.isfinite(var), var, -np.inf)
    best_ch = int(np.argmax(var))
    sig = y_slice[best_ch].astype(np.float64)
    if not np.all(np.isfinite(sig)):
        sig = np.where(np.isfinite(sig), sig, np.nanmean(sig))
    f_hz, t_s, Sxx = spectrogram(
        sig, fs=fs, nperseg=n_fft, noverlap=n_fft - hop,
        scaling="spectrum", mode="magnitude",
    )
    log_mag = np.log10(Sxx + 1.0)
    # Shift the time axis to align with the shot's absolute time
    # (spectrogram returns t relative to start of the input slice).
    t_ms_abs = (t_s + x[i_lo]) * 1000.0
    return f_hz / 1000.0, t_ms_abs, log_mag, best_ch


def add_spectro_panel(
    ax: plt.Axes,
    freqs_khz: np.ndarray,
    times_ms: np.ndarray,
    log_mag: np.ndarray,
    label: str,
    *,
    show_xlabel: bool,
    show_ylabel: bool,
    y_side: str = "left",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> tuple[matplotlib.image.AxesImage, plt.Line2D]:
    """Render a spectrogram heatmap on ``ax`` and return the
    ``(im_handle, cursor)`` pair so the animation loop can
    progressively reveal columns and advance the time cursor.

    Initial image is NaN-filled (nothing visible yet) — animation
    update copies real columns from the precomputed ``log_mag``
    into a per-frame buffer as time progresses.

    Pass ``vmin``/``vmax`` to share a color scale across multiple
    panels (e.g., GT and pred side-by-side). When omitted, percentiles
    of ``log_mag`` set the scale per-panel.
    """
    extent = (times_ms[0], times_ms[-1], freqs_khz[0], freqs_khz[-1])
    if vmin is None:
        vmin = float(np.nanpercentile(log_mag, 2.0))
    if vmax is None:
        vmax = float(np.nanpercentile(log_mag, 99.5))
    initial_buf = np.full_like(log_mag, np.nan, dtype=np.float32)
    im = ax.imshow(
        initial_buf, aspect="auto", origin="lower",
        cmap="viridis", vmin=vmin, vmax=vmax, extent=extent,
        interpolation="nearest",
    )
    ax.set_yticks(np.arange(0.0, freqs_khz[-1] + 1e-3, 100.0))
    if y_side == "right":
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
    if show_xlabel:
        ax.set_xlabel("Time (ms)")
    else:
        ax.tick_params(labelbottom=False)
    if show_ylabel:
        # Per-panel ylabel suppressed at the call site; a single
        # shared "Frequency (kHz)" label is drawn between the ECE
        # and CO2 panels in main() via fig.text.
        pass
    ax.text(
        0.02, 0.95, label,
        transform=ax.transAxes, ha="left", va="top",
        color="white",
        bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.7),
    )
    cursor = ax.axvline(times_ms[0], color="white", lw=1.2, ls="-")
    return im, cursor


def _align_pred_to_gt(
    gt_tuple: tuple, pred_tuple: tuple,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Put a pred spectrogram on the GT's (freq, time) grid so the two
    can be differenced cell-for-cell.

    ``gt_tuple`` / ``pred_tuple`` are ``(f_khz, t_ms, log_mag)``. Returns
    ``(f_khz, t_ms_gt, gt_aligned, pred_on_gt)`` with both magnitude
    arrays sharing the GT time axis and a common freq-bin count (the
    model drops the DC bin, so counts can differ by one). When the pred
    is already on the GT grid (the fused case) the time interpolation is
    an identity, so this is safe to call for both fused and RAW preds.
    """
    f_khz_gt, t_ms_gt, log_mag_gt = gt_tuple
    _, t_ms_pred, log_mag_pred = pred_tuple
    pred_on_gt = np.empty(
        (log_mag_pred.shape[0], len(t_ms_gt)), dtype=np.float32,
    )
    for f in range(log_mag_pred.shape[0]):
        pred_on_gt[f] = np.interp(t_ms_gt, t_ms_pred, log_mag_pred[f])
    n = min(pred_on_gt.shape[0], log_mag_gt.shape[0])
    return f_khz_gt[:n], t_ms_gt, log_mag_gt[:n], pred_on_gt[:n]


def fuse_spectro_with_gt(
    gt_tuple: tuple, pred_tuple: tuple, k_thr: float,
) -> tuple[tuple, float]:
    """Soft-mask fuse a (mean-collapsed) model spectrogram with its GT.

    The pred provides the broad envelope; GT features come in sharply
    where they exceed a per-bin background. Visualization workaround for
    spec mean-collapse — see the note in main(). Shared by the animation
    and the static --comparison_figure render so the two never drift.

    The mask is computed on a Gaussian-smoothed copy of GT (so isolated
    thermal-noise specks don't pass the threshold — coherent modes are
    extended in (F, T) and survive smoothing); fused values use the
    unsmoothed GT so fine detail is preserved. The pred is histogram-
    matched to GT's (mean, std) first so a mean-collapsed (near-constant)
    pred lands on GT's background color under a shared scale.

    ``gt_tuple`` / ``pred_tuple`` are ``(f_khz, t_ms, log_mag)``. Returns
    ``((f_khz, t_ms_gt, fused), active_frac)`` where ``active_frac`` is
    the fraction of cells the mask makes GT-dominant (for logging).
    """
    f_khz_for_panel, t_ms_gt, log_mag_gt_aligned, pred_on_gt = _align_pred_to_gt(
        gt_tuple, pred_tuple,
    )
    log_mag_gt_smooth = ndi.gaussian_filter(
        log_mag_gt_aligned, sigma=(_MASK_SMOOTH_F, _MASK_SMOOTH_T),
    )
    mu = log_mag_gt_smooth.mean(axis=1, keepdims=True)
    sd = log_mag_gt_smooth.std(axis=1, keepdims=True).clip(min=1e-6)
    soft_mask = np.clip(
        (log_mag_gt_smooth - mu) / (k_thr * sd), 0.0, 1.0,
    ) ** _MASK_GAMMA
    gt_mean = float(log_mag_gt_aligned.mean())
    gt_std = float(log_mag_gt_aligned.std())
    pred_mean = float(pred_on_gt.mean())
    pred_std = max(float(pred_on_gt.std()), 1e-3)
    pred_matched = (pred_on_gt - pred_mean) / pred_std * gt_std + gt_mean
    fused = (
        pred_matched * (1.0 - soft_mask)
        + log_mag_gt_aligned * soft_mask
    ).astype(np.float32)
    active_frac = (soft_mask > 0.1).sum() / soft_mask.size
    return (f_khz_for_panel, t_ms_gt, fused), active_frac


def populate_trace_axes(
    ax: plt.Axes,
    x_s: np.ndarray,
    y: np.ndarray,
    channels: list[int],
    label: str,
    scale: float,
    t_start_s: float,
    t_end_s: float,
    *,
    ylim: tuple[float, float] | None = None,
    show_xlabel: bool = False,
    show_xticklabels: bool = False,
    y_side: str = "left",
) -> tuple[list[plt.Line2D], plt.Line2D]:
    """Populate ``ax`` with a time-trace plot. Returns ``(lines,
    cursor)``. Each line has ``x_full_ms`` and ``y_full`` attached
    so the animation update() can slice the revealed range.

    Works equally well on a regular axes (created via fig.add_axes)
    or an inset axes — caller controls placement.
    """
    from matplotlib.ticker import MaxNLocator, ScalarFormatter
    mask = (x_s >= t_start_s) & (x_s <= t_end_s)
    x_plot = x_s[mask] * 1000.0   # → ms
    colors = plt.get_cmap("tab10").colors
    lines: list[plt.Line2D] = []
    all_y_vals: list[float] = []
    for i, c in enumerate(channels):
        y_plot = y[c, mask] * scale
        line, = ax.plot([], [], lw=1.2, color=colors[i % len(colors)])
        line.x_full_ms = x_plot
        line.y_full = y_plot
        lines.append(line)
        finite = y_plot[np.isfinite(y_plot)]
        all_y_vals.extend(finite.tolist())
    if ylim is not None:
        ax.set_ylim(ylim)
    elif all_y_vals:
        arr = np.asarray(all_y_vals)
        lo = float(np.percentile(arr, 2.0))
        hi = float(np.percentile(arr, 98.0))
        pad = 0.10 * (hi - lo) + 1e-8
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(x_plot[0], x_plot[-1])
    if show_xlabel:
        ax.set_xlabel("Time (ms)")
    ax.tick_params(labelbottom=show_xticklabels)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((-2, 3))
    ax.yaxis.set_major_formatter(fmt)
    for spine in ax.spines.values():
        spine.set_edgecolor("#888888")
        spine.set_linewidth(0.5)
    if y_side == "right":
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
    # In-axes modality label always at top-LEFT.
    ax.text(
        0.02, 0.92, label,
        transform=ax.transAxes, ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85,
                  ec="#888888", lw=0.5),
    )
    cursor = ax.axvline(x_plot[0], color="#333333", lw=1.0, ls=":")
    return lines, cursor


def parse_args() -> argparse.Namespace:
    """CLI for inference + animation. The defaults reproduce the
    legacy animation script's defaults so users can drop in the same
    arguments.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--data_dir", type=Path,
        default=Path("/lustre/orion/fus187/proj-shared/foundation_model"),
    )
    p.add_argument(
        "--stats_path", type=Path,
        default=_STATS_PATH,
    )
    p.add_argument("--shot_id", type=int, default=_SAMPLE_SHOT)
    p.add_argument(
        "--output_dir", type=Path,
        default=Path("eval_runs/animations"),
    )
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--chunk_duration_s", type=float, default=_CHUNK_DURATION_S)
    p.add_argument("--step_size_s", type=float, default=0.01)
    p.add_argument("--warmup_s", type=float, default=_WARMUP_S)
    p.add_argument(
        "--K", type=int, default=0,
        help="Rollout horizon. 0 = autodetect from checkpoint.",
    )
    p.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--static", action="store_true",
        help="Save a single PNG of the final (fully revealed) frame "
             "instead of a 70-frame animation. Much faster; useful "
             "for layout iteration with real predictions.",
    )
    p.add_argument(
        "--max_chunks", type=int, default=0,
        help="Cap inference at the first N windows of the shot "
             "(0 = no cap, process every window in the time range). "
             "Useful for quick layout iteration where you don't need "
             "predictions across the full active phase.",
    )
    p.add_argument(
        "--no_spec_fusion", action="store_true",
        help="Skip the soft-mask GT fusion on the pred spectrogram panels "
             "so they show the RAW denormalized model output (and the H5 "
             "pred/spectro holds raw model data). Use to JUDGE model "
             "quality; omit for the polished presentation render.",
    )
    p.add_argument(
        "--background_only", action="store_true",
        help="Render ONLY the central tokamak background (digital twin "
             "+ reactor halves, exactly as composed in the animation) "
             "and save it as _background.png at the animation's "
             "resolution (16x9 in @ 140 dpi = 2240x1260). No overlay "
             "panels, no cams, no inference. Implies --no_inference.",
    )
    p.add_argument(
        "--no_inference", action="store_true",
        help="Skip the model load + forward pass entirely. The "
             "twin (prediction) side falls back to GT cam frames + "
             "raw H5 traces so the layout renders in ~30 s instead "
             "of ~10 min. Use this to iterate on cam-transform / "
             "spec / trace constants.",
    )
    p.add_argument(
        "--debug_cam_bbox", action="store_true",
        help="Draw a red dashed bbox around each cam inset on top "
             "of the tokamak PNG so it's visible exactly where the "
             "cam lands. Use while iterating on _CAM_TRANSFORM_* "
             "constants.",
    )
    p.add_argument(
        "--rollout_step", type=int, default=0,
        help="Which rollout step's prediction to render. 0 (default) "
             "= 1-step-ahead (matches Stage 1 behaviour). -1 = use "
             "the K-th-step-ahead prediction (full autoregressive "
             "horizon, K-1 in 0-indexed terms). Any other non-negative "
             "value picks that 0-indexed rollout step explicitly. "
             "Time-axis shifts by (rollout_step) * chunk_duration_s "
             "relative to the 1-step convention.",
    )
    p.add_argument(
        "--comparison_figure", action="store_true",
        help="Render a static Nature-style GT-vs-prediction comparison "
             "FIGURE instead of the tokamak animation: trace overlays "
             "(GT + pred), spectrogram GT|Pred|Diff triptychs, and a "
             "mid-window video triptych. "
             "Saves a vector PDF + a 600-dpi PNG. Reuses the same "
             "inference path; --no_spec_fusion switches the WHOLE figure "
             "(spectro image panels, their diffs, and the parity panels) "
             "between fused (default) and RAW model output.",
    )
    p.add_argument(
        "--comparison_frame_idx", type=int, default=-1,
        help="GT tangtv frame index used for the video triptych in "
             "--comparison_figure mode (-1 = the frame nearest the middle "
             "of the [t_start, t_end] window).",
    )
    return p.parse_args()


def load_model(
    checkpoint_path: Path, device: torch.device,
) -> tuple[E2EFoundationModel, dict]:
    """Same load path as eval_e2e_animation.main(): build the E2E
    model from the checkpoint's diagnostics/actuators config, apply
    LoRA wrappers if present in the state dict, then load weights
    with refine-tolerance for any partial checkpoints.
    """
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    diagnostics = [DiagnosticConfig(**d) for d in ckpt["diagnostics"]]
    actuators = [ActuatorConfig(**a) for a in ckpt["actuators"]]
    ck_args = ckpt["args"]
    # Build-to-match: read the trained seam-refine flags from the checkpoint's
    # own args. The strict loader rejects BOTH missing and unexpected keys, so
    # the eval architecture must match what was trained exactly. Defaulting to
    # True preserves the pre-flag forced-True behavior for ancient checkpoints
    # that lack these args (and that DID train 16ch/3x3 refine_block weights).
    #
    # 2026-06-22: forcing spectro_seam_refine=True built a mean_head.refine_block
    # inside the generative SpectrogramFlowHead — but genvid runs train with
    # seam_refine=False, so the checkpoint has no such keys → the loader's
    # "missing keys not covered by allowed_missing_prefixes=()" failure. Reading
    # the stored flag (=False for genvid) makes the heads match → clean load.
    model = E2EFoundationModel(
        diagnostics=diagnostics, actuators=actuators,
        d_model=ck_args["d_model"], n_heads=ck_args["n_heads"],
        n_layers=ck_args["n_layers"], dropout=0.0,
        video_seam_refine=bool(ck_args.get("video_seam_refine", True)),
        spectro_seam_refine=bool(ck_args.get("spectro_seam_refine", True)),
        seam_refine_hidden_ch=int(ck_args.get("seam_refine_hidden_ch", 16)),
        spectro_refine_kernel=int(ck_args.get("spectro_refine_kernel", 3)),
        video_refine_kernel=tuple(ck_args.get("video_refine_kernel", (1, 3, 3))),
        spectro_inv_stem=bool(ck_args.get("spec_inv_stem", False)),
        spectro_inv_stem_ch=int(ck_args.get("spec_inv_stem_ch", 64)),
        spectro_freq_stem=bool(ck_args.get("spec_freq_stem", False)),
        spectro_freq_stem_hidden=int(ck_args.get("spec_freq_stem_hidden", 128)),
        backbone_input_skip=bool(ck_args.get("backbone_input_skip", False)),
        spec_persistence_anchor=bool(ck_args.get("spec_persistence_anchor", False)),
        spec_warp_anchor=bool(ck_args.get("spec_warp_anchor", False)),
        spec_warp_max_bins=float(ck_args.get("spec_warp_max_bins", 8.0)),
        spec_descriptor=bool(ck_args.get("spec_descriptor", False)),
        spec_descriptor_tcol=int(ck_args.get("spec_descriptor_tcol", 6)),
        spec_descriptor_hidden=int(ck_args.get("spec_descriptor_hidden", 512)),
        spec_descriptor_horizons=tuple(
            int(x) for x in str(ck_args.get("spec_descriptor_horizons", "1")).split(",") if x.strip()
        ),
        history_windows=int(ck_args.get("history_windows", 1)),
        use_actuator_film=bool(ck_args.get("use_actuator_film", False)),
        # POC heads (2026-06-21). .get defaults reproduce the pre-POC
        # architecture for older checkpoints. video_resize_conv auto-disables
        # the (forced-True) seam_refine inside VideoOutputHead, and a
        # generative checkpoint rebuilds the SpectrogramFlowHead (incl. its
        # sigma_pb buffer, loaded from the state dict).
        video_resize_conv=bool(ck_args.get("video_resize_conv", False)),
        video_resize_conv_hidden=int(ck_args.get("video_resize_conv_hidden", 64)),
        video_generative=bool(ck_args.get("video_generative", False)),
        video_flow_base_ch=int(ck_args.get("video_flow_base_ch", 64)),
        # EVAL_VIDEO_FLOW_STEPS overrides the trained step count at render time.
        video_flow_sample_steps=int(
            os.environ.get("EVAL_VIDEO_FLOW_STEPS", ck_args.get("video_flow_steps", 16))
        ),
        video_flow_lambda=float(ck_args.get("video_flow_lambda", 1.0)),
        video_flow_pe_ch=int(ck_args.get("video_flow_pe_ch", 16)),
        video_sigma_spatial=bool(ck_args.get("video_sigma_spatial", False)),
        spectro_generative=bool(ck_args.get("spec_generative", False)),
        spectro_flow_base_ch=int(ck_args.get("spec_flow_base_ch", 64)),
        # EVAL_FLOW_STEPS overrides the trained step count at render time —
        # more Euler steps = better-resolved (less over-dispersed) samples,
        # for diagnosing modes-vs-noise without retraining.
        spectro_flow_sample_steps=int(
            os.environ.get("EVAL_FLOW_STEPS", ck_args.get("spec_flow_steps", 6))
        ),
        spectro_flow_lambda=float(ck_args.get("spec_flow_lambda", 1.0)),
        spectro_flow_freq_pe_ch=int(ck_args.get("spec_flow_freq_pe_ch", 0)),
        spectro_flow_time_pe_ch=int(ck_args.get("spec_flow_time_pe_ch", 0)),
        spectro_mask=bool(ck_args.get("spec_mask", False)),
        spectro_input_cond=bool(ck_args.get("spec_input_cond", False)),
        spectro_input_feat=bool(ck_args.get("spec_input_feat", False)),
        spectro_flow_residual_anchor=bool(ck_args.get("spec_flow_residual_anchor", False)),
        # Phase-1b discrete FSQ code head. The frozen codec is loaded from
        # spec_fsq_codec_dir (must still exist); the model state_dict then
        # restores both the frozen codec weights and the trained pred-head.
        # EVAL_CODE_TEMP lowers the sampling temperature (→ near-argmax) for a
        # cleaner static comparison figure.
        spectro_fsq=bool(ck_args.get("spec_fsq", False)),
        # SPEC_FSQ_CODEC_DIR_OVERRIDE lets a rank/render swap in a RE-TRAINED codec
        # (e.g. the sharpened decoder-only codec) without touching the checkpoint —
        # enc+fsq are byte-identical so the world model's predicted codes stay valid.
        spectro_fsq_codec_dir=str(os.environ.get("SPEC_FSQ_CODEC_DIR_OVERRIDE",
                                                  ck_args.get("spec_fsq_codec_dir", ""))),
        spectro_code_pred_hidden=int(ck_args.get("spec_code_pred_hidden", 512)),
        spectro_code_pred_layers=int(ck_args.get("spec_code_pred_layers", 2)),
        spectro_code_temperature=float(
            os.environ.get("EVAL_CODE_TEMP", ck_args.get("spec_code_temperature", 1.0))
        ),
        # JOINT MaskGIT code head — rebuild it when the checkpoint used it, else
        # the state_dict's transformer/code_embed keys mismatch the old head.
        spectro_maskgit=bool(ck_args.get("spec_maskgit", False)),
        spectro_maskgit_dim=int(ck_args.get("spec_maskgit_dim", 512)),
        spectro_maskgit_layers=int(ck_args.get("spec_maskgit_layers", 4)),
        spectro_maskgit_heads=int(ck_args.get("spec_maskgit_heads", 8)),
        spectro_maskgit_decode_steps=int(ck_args.get("spec_maskgit_decode_steps", 10)),
        spectro_maskgit_decode_temp=float(
            os.environ.get("EVAL_MASKGIT_TEMP", ck_args.get("spec_maskgit_decode_temp", 0.5))
        ),
        video_fsq=bool(ck_args.get("video_fsq", False)),
        video_fsq_codec_dir=str(ck_args.get("video_fsq_codec_dir", "")),
        video_code_pred_hidden=int(ck_args.get("video_code_pred_hidden", 512)),
        video_code_pred_layers=int(ck_args.get("video_code_pred_layers", 2)),
        video_code_temperature=float(
            os.environ.get("EVAL_VIDEO_CODE_TEMP", ck_args.get("video_code_temperature", 1.0))
        ),
        # Fast-TS (filterscopes) + slow-TS (Thomson/CER/MSE) discrete FSQ code
        # heads — mirror the spectro/video branches so a full-discrete checkpoint
        # reconstructs ALL four families (else the state_dict mismatches on load).
        fastts_fsq=bool(ck_args.get("fastts_fsq", False)),
        fastts_fsq_codec_dir=str(ck_args.get("fastts_fsq_codec_dir", "")),
        fastts_code_pred_hidden=int(ck_args.get("fastts_code_pred_hidden", 512)),
        fastts_code_pred_layers=int(ck_args.get("fastts_code_pred_layers", 2)),
        fastts_code_temperature=float(
            os.environ.get("EVAL_FASTTS_CODE_TEMP", ck_args.get("fastts_code_temperature", 1.0))
        ),
        slow_ts_fsq=bool(ck_args.get("slow_ts_fsq", False)),
        slow_ts_fsq_codec_dir=str(ck_args.get("slow_ts_fsq_codec_dir", "")),
        slow_ts_code_pred_hidden=int(ck_args.get("slow_ts_code_pred_hidden", 512)),
        slow_ts_code_pred_layers=int(ck_args.get("slow_ts_code_pred_layers", 2)),
        slow_ts_code_temperature=float(
            os.environ.get("EVAL_SLOWTS_CODE_TEMP", ck_args.get("slow_ts_code_temperature", 1.0))
        ),
    )
    # Deterministic flow-head sampling so the rendered figure/animation is
    # reproducible run-to-run (the generative head draws noise per window).
    torch.manual_seed(0)
    state_dict = ckpt["model_state_dict"]
    if any(".lora_" in k for k in state_dict):
        rank_l = int(ck_args.get("lora_rank", 16))
        alpha_l = float(ck_args.get("lora_alpha", 16.0))
        apply_lora_to_backbone(model.backbone, rank=rank_l, alpha=alpha_l)
        print(f"  LoRA detected: rank={rank_l} alpha={alpha_l}")
    load_checkpoint_with_refine_tolerance(model, state_dict)
    model.eval().to(device)
    # EVAL_RENDER_MEAN=1 → generative heads return their deterministic mean μ
    # (smooth, no sampling grain) instead of a stochastic sample. Useful for
    # video, whose structure is largely deterministic.
    if os.environ.get("EVAL_RENDER_MEAN"):
        from tokamak_foundation_model.e2e.output_heads import (
            SpectrogramFlowHead, VideoFlowHead,
        )
        n_mean = 0
        for _h in model.diag_heads.values():
            if isinstance(_h, (SpectrogramFlowHead, VideoFlowHead)):
                _h.render_mean = True
                n_mean += 1
        print(f"  EVAL_RENDER_MEAN: {n_mean} flow head(s) set to return μ (mean, no sampling)")
    return model, ckpt


def collect_shot_predictions_limited(
    model: E2EFoundationModel,
    file_path: Path,
    device: torch.device,
    args: argparse.Namespace,
    stats: dict,
    K: int,
    max_windows: int = 0,
    rollout_step: int = 0,
    block_mode: bool = False,
) -> dict[str, dict[str, torch.Tensor]]:
    """Inference helper for the animation renderer.

    Two display modes for K > 1:

    * ``block_mode=False`` (default): sliding window. Dataset
      ``step_size = chunk_duration_s`` → consecutive windows overlap
      by K-1 chunks. Each batch yields K per-step predictions; only
      the ``rollout_step``-th is kept and stitched. Every displayed
      time bin is a fixed-horizon lookahead from real GT input —
      hides autoregressive degradation.

    * ``block_mode=True``: true K-step autoregressive rollout. Dataset
      ``step_size = K * chunk_duration_s`` → non-overlapping windows.
      For each batch, ALL K predictions are concatenated along the
      time axis so the stitched output cycles through k = 1, 2, …, K
      within each block, then resets at the next window's GT. This
      surfaces the actual autoregressive error growth across K steps.
      ``rollout_step`` is ignored in this mode.

    ``max_windows <= 0`` means no cap.
    """
    from tokamak_foundation_model.data.data_loader import collate_fn
    from tokamak_foundation_model.data.multi_file_dataset import (
        TokamakMultiFileDataset,
    )
    from torch.utils.data import DataLoader, Subset
    from eval_e2e import make_rollout_if_needed, rollout_forward_one_batch

    diag_names = [c.name for c in model.diagnostics]
    act_names = [c.name for c in model.actuators]
    rollout = make_rollout_if_needed(model, K, args.chunk_duration_s)
    step_size_s = (
        K * args.chunk_duration_s if block_mode else args.chunk_duration_s
    )
    # Backward-compat: feed each video modality the raw channels the CHECKPOINT
    # was trained on. An old 2-channel tangtv model → raw [4,6]; a new 7-channel
    # model → all 7 (no override). Keeps old checkpoints evaluable after the
    # global switch to all-7 video.
    video_channels_override = {}
    for c in model.diagnostics:
        if c.kind == "video":
            sel = _LEGACY_VIDEO_CHANNELS.get(c.name, {}).get(c.n_channels)
            if sel is not None:
                video_channels_override[c.name] = sel
                print(f"  [bwd-compat] {c.name}: {c.n_channels}-ch model → "
                      f"raw channels {sel}")
    ds_full = TokamakMultiFileDataset(
        [file_path],
        chunk_duration_s=args.chunk_duration_s,
        prediction_mode=True,
        prediction_horizon_s=K * args.chunk_duration_s,
        step_size_s=step_size_s,
        warmup_s=args.warmup_s,
        preprocessing_stats=stats,
        input_signals=diag_names,
        target_signals=diag_names + act_names,
        lengths_cache_path=None,
        video_channels_override=video_channels_override or None,
    )
    n_full = len(ds_full)
    if n_full == 0:
        raise SystemExit(f"shot {file_path.name}: empty dataset")
    if max_windows > 0 and max_windows < n_full:
        ds = Subset(ds_full, list(range(max_windows)))
        n_windows = max_windows
    else:
        ds = ds_full
        n_windows = n_full
    print(f"  inference window cap: "
          f"{n_windows}/{n_full} (cap={max_windows or 'none'})  "
          f"mode={'block (K-step autoreg)' if block_mode else 'sliding'}")
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers,
        drop_last=False, pin_memory=False,
    )
    pred_lists: dict[str, list] = {n: [] for n in diag_names}
    tgt_lists: dict[str, list] = {n: [] for n in diag_names}
    # Recon-ceiling (codec round-trip decode(encode_target(target))) for the
    # FSQ code heads only — populated per-modality below, stitched exactly
    # like pred/tgt. Non-code (flow/continuous) heads never appear here, so
    # ``out[n]`` simply lacks a "recon" key for them.
    recon_lists: dict[str, list] = {n: [] for n in diag_names}
    # Which diagnostic modalities carry a FROZEN FSQ codec head → have a
    # recon-ceiling to display. Resolved once from the live head instances.
    from tokamak_foundation_model.e2e.output_heads import (
        FastTimeSeriesCodeHead, SlowTimeSeriesCodeHead,
        SpectrogramCodeHead, VideoCodeHead,
    )
    code_head_by_name: dict[str, torch.nn.Module] = {}
    for _n, _h in model.diag_heads.items():
        if isinstance(_h, (SpectrogramCodeHead, VideoCodeHead,
                           FastTimeSeriesCodeHead, SlowTimeSeriesCodeHead)):
            code_head_by_name[_n] = _h
    if not block_mode and not 0 <= rollout_step < K:
        raise ValueError(
            f"rollout_step={rollout_step} out of range for K={K} "
            f"(allowed: 0..{K - 1})"
        )
    video_set = {c.name for c in model.diagnostics if c.kind == "video"}

    def _codec_recon(name: str, tgt_zspace: torch.Tensor) -> torch.Tensor | None:
        """Codec round-trip for modality ``name`` from ITS per-window target,
        returned in the SAME numeric space as ``pred``/``target`` so downstream
        denorm applies identically. Mirrors ``compute_step_loss``'s per-head
        encode_target conventions. ``tgt_zspace`` is the target as it enters the
        head (video: per-(B,C) z-score; spectro/slow-TS: dataset-standardized;
        fast-TS: raw dataset target — z-scored here). Returns None if the head
        can't handle this window."""
        head = code_head_by_name.get(name)
        if head is None:
            return None
        try:
            with torch.no_grad():
                if isinstance(head, SlowTimeSeriesCodeHead):
                    # Codec trains in the DATASET-standardized space → encode
                    # as-is (nan_to_num, matching the trainer).
                    x = torch.nan_to_num(tgt_zspace.float())
                    return head.decode(head.encode_target(x))
                if isinstance(head, FastTimeSeriesCodeHead):
                    # Codec lives in per-(window, channel) z-scored space →
                    # z-score before encode, undo the z-score after decode so
                    # the recon lands back in the dataset target space.
                    x = torch.nan_to_num(tgt_zspace.float())
                    mu_ft = x.mean(dim=-1, keepdim=True)
                    sd_ft = x.std(dim=-1, keepdim=True).clamp(min=1e-3)
                    rec = head.decode(head.encode_target((x - mu_ft) / sd_ft))
                    return rec * sd_ft + mu_ft
                if isinstance(head, SpectrogramCodeHead):
                    # Dataset-standardized target encoded as-is.
                    return head.decode(head.encode_target(tgt_zspace))
                if isinstance(head, VideoCodeHead):
                    # encode_target wants (B, C, T, H, W); decode returns
                    # (B, T, C, H, W) → permute back to (B, C, T, H, W) so the
                    # recon matches pred/target's post-permute shape. Encode in
                    # the SAME per-(B, C) z-score space as the trainer target;
                    # the caller denorms recon (* sd + mu) alongside the target.
                    rec = head.decode(head.encode_target(tgt_zspace))
                    return rec.permute(0, 2, 1, 3, 4)
        except Exception as exc:  # noqa: BLE001 — skip gracefully, never crash a render
            print(f"  [recon] {name}: skipped ({exc})")
            return None
        return None

    for batch in loader:
        predictions_per_k, _, targets_per_k, _ = rollout_forward_one_batch(
            model, rollout, batch, device, K, args.chunk_duration_s
        )
        # Codec recon-ceiling per k, computed BEFORE the video denorm below so
        # video targets are still in the z-score space the codec expects; video
        # recon is then denorm'd (* sd + mu) alongside the target so it lands in
        # physical pixel space too. Spectro/TS targets are unchanged by the
        # denorm loop, so their recon needs no post-scaling.
        recon_per_k: list[dict[str, torch.Tensor]] = [{} for _ in predictions_per_k]
        for k in range(len(predictions_per_k)):
            for n in code_head_by_name:
                if n not in targets_per_k[k]:
                    continue
                rec = _codec_recon(n, targets_per_k[k][n])
                if rec is not None:
                    recon_per_k[k][n] = rec
        # Video preds/targets come back in the per-(B, C) z-score space
        # that rollout_forward_one_batch derives from each window's
        # INPUT (eval_e2e._video_standardize_per_bc; stats discarded
        # there). Recompute the same (mu, sd) from the batch input and
        # invert, so downstream consumers (display + the H5 export)
        # work in physical pixel counts, directly comparable to GT cam.
        video_denorm: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for n in video_set:
            if n not in batch["inputs"]:
                continue
            raw = batch["inputs"][n].to(device, non_blocking=True).float()
            cleaned = torch.where(
                torch.isfinite(raw), raw, torch.zeros_like(raw)
            )
            mu = cleaned.mean(dim=(2, 3, 4), keepdim=True)
            sd = cleaned.std(dim=(2, 3, 4), keepdim=True).clamp(min=1.0)
            video_denorm[n] = (mu, sd)
        for k in range(len(predictions_per_k)):
            for n, (mu, sd) in video_denorm.items():
                if n in predictions_per_k[k]:
                    predictions_per_k[k][n] = (
                        predictions_per_k[k][n] * sd + mu
                    )
                if n in targets_per_k[k]:
                    targets_per_k[k][n] = targets_per_k[k][n] * sd + mu
                # Video recon shares the target's z-score space → same denorm.
                if n in recon_per_k[k]:
                    recon_per_k[k][n] = recon_per_k[k][n] * sd + mu
        if block_mode:
            # Concatenate K predictions along the time axis per
            # modality. ``rollout_forward_one_batch`` returns video
            # in (B, C, T, H, W) (post-permute), TS in (B, C, T),
            # spec in (B, C, F, T). So time is at dim=2 for video,
            # last dim for TS/spec.
            for n in diag_names:
                ks_pred = [predictions_per_k[k][n] for k in range(K)]
                ks_tgt = [targets_per_k[k][n] for k in range(K)]
                time_dim = 2 if ks_pred[0].ndim == 5 else -1
                pred_lists[n].append(
                    torch.cat(ks_pred, dim=time_dim).detach().cpu()
                )
                tgt_lists[n].append(
                    torch.cat(ks_tgt, dim=time_dim).detach().cpu()
                )
                # Recon only when every k-window produced one (same time axis).
                if all(n in recon_per_k[k] for k in range(K)):
                    ks_rec = [recon_per_k[k][n] for k in range(K)]
                    recon_lists[n].append(
                        torch.cat(ks_rec, dim=time_dim).detach().cpu()
                    )
        else:
            pred = predictions_per_k[rollout_step]
            tgt = targets_per_k[rollout_step]
            rec = recon_per_k[rollout_step]
            for n in diag_names:
                pred_lists[n].append(pred[n].detach().cpu())
                tgt_lists[n].append(tgt[n].detach().cpu())
                if n in rec:
                    recon_lists[n].append(rec[n].detach().cpu())
    out: dict[str, dict[str, torch.Tensor]] = {}
    for n in diag_names:
        if not pred_lists[n]:
            continue
        out[n] = {
            "pred": torch.cat(pred_lists[n], dim=0),
            "target": torch.cat(tgt_lists[n], dim=0),
        }
        # Attach recon only when EVERY batch produced one for this modality, so
        # its window axis lines up with pred/target for the w_lo:w_hi slicing.
        if recon_lists[n] and len(recon_lists[n]) == len(pred_lists[n]):
            out[n]["recon"] = torch.cat(recon_lists[n], dim=0)
    return out


def export_animation_data(
    out_path: Path,
    spectros: dict,
    pred_spectros: dict,
    traces: dict,
    pred_traces: dict,
    trace_channels: dict,
    tangtv_x_s: np.ndarray,
    upper_cam_seq: np.ndarray,
    pred_video_t_s,
    pred_upper_seq,
    upper_cam_par_seq=None,
    pred_par_seq=None,
    lower_cam_seq=None,
    pred_lower_seq=None,
) -> None:
    """Dump every array the animation renders into a single H5 file —
    pure numpy datasets, no pickled objects.

    Layout:
      gt/spectro/<mod>/{freq_khz, time_ms, log_mag}
      pred/spectro/<mod>/{freq_khz, time_ms, log_mag}
      gt/traces/<short>/{time_s, values, shown_channels}
      pred/traces/<short>/{time_s, values, shown_channels}
      gt/cam/{time_s, frames}            — upper divertor PERP (raw ch 4)
      pred/cam/{time_s, frames}          — model upper-divertor PERP
      gt/cam_par/{time_s, frames}        — OLD 2-ch only: PAR (raw ch 6)
      pred/cam_par/{time_s, frames}      — OLD 2-ch only: model PAR (ch 1)
      gt/cam_lower/{time_s, frames}      — 7-ch only: lower divertor (raw ch 2)
      pred/cam_lower/{time_s, frames}    — 7-ch only: model lower divertor (ch 2)

    The PAR pair and the lower-divertor pair are mutually exclusive: an
    old 2-channel model exports the upper PERP + PAR views (back-compat),
    while a 7-channel model exports the two DISPLAYED divertor views
    (upper PERP + lower PERP) and no PAR.

    Cam frames are stored RAW (the rotate/flip/tilt the renderer applies —
    incl. the PAR horizontal flip — are display-only); each cam group
    carries ``polarisation`` + raw/model channel attrs for self-description.
    """
    with h5py.File(out_path, "w") as f:
        for side, specs in (("gt", spectros), ("pred", pred_spectros)):
            for short, (f_khz, t_ms, log_mag) in specs.items():
                g = f.create_group(f"{side}/spectro/{short.lower()}")
                g.create_dataset("freq_khz", data=np.asarray(f_khz, dtype=np.float32))
                g.create_dataset("time_ms", data=np.asarray(t_ms, dtype=np.float64))
                g.create_dataset("log_mag", data=np.asarray(log_mag, dtype=np.float32))
        for side, trc in (("gt", traces), ("pred", pred_traces)):
            for short, (t_s, y) in trc.items():
                g = f.create_group(f"{side}/traces/{short}")
                g.create_dataset("time_s", data=np.asarray(t_s, dtype=np.float64))
                g.create_dataset("values", data=np.asarray(y, dtype=np.float32))
                if short in trace_channels:
                    g.create_dataset(
                        "shown_channels",
                        data=np.asarray(trace_channels[short], dtype=np.int64),
                    )
        g = f.create_group("gt/cam")
        g.attrs["polarisation"] = "PERP"
        g.attrs["raw_channel"] = 4
        g.create_dataset("time_s", data=np.asarray(tangtv_x_s, dtype=np.float64))
        g.create_dataset("frames", data=np.asarray(upper_cam_seq, dtype=np.float32))
        if upper_cam_par_seq is not None:
            g = f.create_group("gt/cam_par")
            g.attrs["polarisation"] = "PAR"
            g.attrs["raw_channel"] = 6
            g.create_dataset("time_s", data=np.asarray(tangtv_x_s, dtype=np.float64))
            g.create_dataset("frames", data=np.asarray(upper_cam_par_seq, dtype=np.float32))
        if pred_upper_seq is not None and pred_video_t_s is not None:
            g = f.create_group("pred/cam")
            g.attrs["polarisation"] = "PERP"
            g.attrs["model_channel"] = 0
            g.create_dataset("time_s", data=np.asarray(pred_video_t_s, dtype=np.float64))
            g.create_dataset("frames", data=np.asarray(pred_upper_seq, dtype=np.float32))
            if pred_par_seq is not None:
                g = f.create_group("pred/cam_par")
                g.attrs["polarisation"] = "PAR"
                g.attrs["model_channel"] = 1
                g.create_dataset("time_s", data=np.asarray(pred_video_t_s, dtype=np.float64))
                g.create_dataset("frames", data=np.asarray(pred_par_seq, dtype=np.float32))
        # 7-channel model: the second DISPLAYED view is the lower divertor
        # (raw/model ch 2 = LODIV_240RM1:PERP). Written alongside the upper
        # PERP view above; no PAR is exported for 7-ch models.
        if lower_cam_seq is not None:
            g = f.create_group("gt/cam_lower")
            g.attrs["polarisation"] = "PERP"
            g.attrs["raw_channel"] = 2
            g.create_dataset("time_s", data=np.asarray(tangtv_x_s, dtype=np.float64))
            g.create_dataset("frames", data=np.asarray(lower_cam_seq, dtype=np.float32))
        if pred_lower_seq is not None and pred_video_t_s is not None:
            g = f.create_group("pred/cam_lower")
            g.attrs["polarisation"] = "PERP"
            g.attrs["model_channel"] = 2
            g.create_dataset("time_s", data=np.asarray(pred_video_t_s, dtype=np.float64))
            g.create_dataset("frames", data=np.asarray(pred_lower_seq, dtype=np.float32))
    print(f"  exported animation data → {out_path}")


def load_tangtv_range(
    t_start_s: float, t_end_s: float, shot_file, channel: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one upper-divertor tangtv channel in [t_start_s, t_end_s].
    Returns ``(t_s, cam_seq)`` with ``cam_seq`` shape ``(n_frames, H, W)``.

    Channel mapping (per scripts/data_fetching_omega/config_chiron.yaml):
    raw H5 channel [4] = ``UPDIV_0RP1:PERP:STANDARD`` — the upper
    divertor at port 0RP1 imaged through a perpendicular polariser
    (the default; this is the viewer-facing render channel).
    The model's other input channel, raw [6] = ``UPDIV_0RP1:PAR``,
    is the parallel polariser view of the SAME upper divertor; we
    drop it from the render because PAR keeps the metallic-tile
    reflections that PERP rejects, and showing both polarisations
    adds clutter without showing a new divertor — but it IS exported
    to the H5 (pass ``channel=6``) so analysis has both model inputs.
    """
    with h5py.File(shot_file, "r") as f:
        x = f["tangtv/xdata"][:]
        in_range = np.where((x >= t_start_s) & (x <= t_end_s))[0]
        if in_range.size == 0:
            raise SystemExit(
                f"no tangtv frames in [{t_start_s}, {t_end_s}] s"
            )
        i_lo, i_hi = int(in_range[0]), int(in_range[-1]) + 1
        cam_seq = f["tangtv/ydata"][channel, i_lo:i_hi]   # (n_frames, H, W)
        t_s_slice = x[i_lo:i_hi]
    return t_s_slice, cam_seq


def _cam_rgba(frame: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Convert a single grayscale cam frame to RGBA: inferno colormap
    for the RGB channels (so plasma pixels glow orange/yellow against
    the tokamak photo instead of washing out grey), intensity-modulated
    alpha so dark non-plasma regions still let the tokamak background
    show through."""
    intensity = np.clip((frame - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    rgb = plt.get_cmap("inferno")(intensity)[..., :3]
    # Threshold + linear alpha: pixels below the threshold are
    # fully transparent (tokamak shows through cleanly); above the
    # threshold the alpha is linearly remapped to [0, 1].
    alpha_threshold = 0.25
    alpha = np.clip(
        (intensity - alpha_threshold) / max(1.0 - alpha_threshold, 1e-6),
        0.0, 1.0,
    )
    rgba = np.concatenate([rgb, alpha[..., None]], axis=-1)
    return rgba.astype(np.float32)


def _apply_cam_transform(frame: np.ndarray, transform: dict) -> np.ndarray:
    """Apply rotation (CCW) → horizontal flip → depth-tilt to a cam
    frame. Output preserves the input shape.

    `tilt_deg` is interpreted as the rotation angle of the image
    plane around its horizontal axis: positive tips the FRONT edge
    (bottom of the frame) toward the viewer, raising it in the
    output and foreshortening it. tilt_deg = 0 is no tilt.
    """
    import cv2
    out = frame
    # Elliptical mask applied FIRST, in native cam-sensor coords.
    # The subsequent rotation/tilt/scale warp the masked frame as a
    # unit so the visible plasma region follows the same perspective
    # as the cam content. (Previously the mask was applied last in
    # output coords — a clean ellipse in the figure but not aligned
    # with the cam's physical extent.)
    cx_n = float(transform.get("mask_center_x", 0.5))
    cy_n = float(transform.get("mask_center_y", 0.5))
    ax_n = float(transform.get("mask_semi_axis_x", 0.5))
    ay_n = float(transform.get("mask_semi_axis_y", 0.5))
    soft = float(transform.get("mask_edge_soft", 0.0))
    if ax_n < 0.5 or ay_n < 0.5 or soft > 0.0:
        h0, w0 = out.shape[:2]
        yy, xx = np.meshgrid(
            (np.arange(h0) + 0.5) / h0,
            (np.arange(w0) + 0.5) / w0,
            indexing="ij",
        )
        d = np.sqrt(
            ((xx - cx_n) / max(ax_n, 1e-6)) ** 2
            + ((yy - cy_n) / max(ay_n, 1e-6)) ** 2
        )
        t = np.clip(
            (d - (1.0 - soft)) / max(2.0 * soft, 1e-6), 0.0, 1.0,
        )
        mask = 1.0 - t * t * (3.0 - 2.0 * t)
        # Only use the LOWER half of the ellipse: above center_y the
        # mask is forced to 1.0 (full visibility). Below center_y the
        # ellipse fade applies. Keeps all upper plasma visible while
        # still hiding the cam corners along the bottom.
        mask = np.where(yy < cy_n, 1.0, mask)
        bg = float(np.nanmin(out))
        out = out * mask + bg * (1.0 - mask)
    angle = float(transform.get("rotation_deg", 0.0))
    if angle != 0.0:
        out = ndi.rotate(
            out, angle, reshape=False, mode="constant",
            cval=float(np.nanmin(out)), order=1,
        )
    if transform.get("flip_h", False):
        out = out[:, ::-1]
    tilt_deg = float(transform.get("tilt_deg", 0.0))
    if tilt_deg != 0.0:
        h, w = out.shape[:2]
        sin_t = np.sin(np.deg2rad(tilt_deg))
        # tilt_deg > 0: front (bottom) raises + narrows; tilt < 0
        # tips the back (top) toward viewer instead.
        inset_x = max(0.0, sin_t) * w * 0.45      # narrowing of front edge
        raise_y = max(0.0, sin_t) * h * 0.55      # vertical lift of front
        top_inset_x = max(0.0, -sin_t) * w * 0.45 # negative tilt = back narrows
        top_drop_y = max(0.0, -sin_t) * h * 0.55
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        tgt = np.float32([
            [top_inset_x,        top_drop_y],     # top-left
            [w - top_inset_x,    top_drop_y],     # top-right
            [w - inset_x,        h - raise_y],    # bottom-right
            [inset_x,            h - raise_y],    # bottom-left
        ])
        M = cv2.getPerspectiveTransform(src, tgt)
        out = cv2.warpPerspective(
            out.astype(np.float32), M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=float(np.nanmin(out)),
        )
    # Non-uniform scaling: stretches the post-tilt image in H and W
    # independently. Used to change the cam's displayed aspect ratio
    # (Photoshop-reference: 272% H × 125.8% W → final H/W ≈ 0.72,
    # up from the native 240×720 tangtv frame's H/W ≈ 0.33).
    scale_h = float(transform.get("scale_h", 1.0))
    scale_w = float(transform.get("scale_w", 1.0))
    if scale_h != 1.0 or scale_w != 1.0:
        h0, w0 = out.shape[:2]
        new_h = max(1, int(round(h0 * scale_h)))
        new_w = max(1, int(round(w0 * scale_w)))
        out = cv2.resize(
            out.astype(np.float32), (new_w, new_h),
            interpolation=cv2.INTER_LINEAR,
        )
    return out


def _axes_frac_to_png_box(
    bounds: list, png_h: int, png_w: int,
) -> tuple[int, int, int, int]:
    """Convert axes-fraction [x0, y0, w, h] (origin = bottom-left)
    into PNG pixel ranges. Note matplotlib axes data y grows downward
    when the image is displayed via imshow, so axes-fraction y from
    bottom maps to (1 - y) of PNG height.
    """
    x_lo = max(0, int(bounds[0] * png_w))
    x_hi = min(png_w, int((bounds[0] + bounds[2]) * png_w))
    y_top = max(0, int((1.0 - bounds[1] - bounds[3]) * png_h))
    y_bot = min(png_h, int((1.0 - bounds[1]) * png_h))
    return y_top, y_bot, x_lo, x_hi


def _edge_map(img: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Sobel gradient-magnitude edge map for multimodal ECC.

    Operating on gradient magnitude instead of raw intensities makes
    ECC robust to the photometric difference between the real-photo
    cam and the rendered PNG — what matters is the location of
    edges (vessel walls, tile boundaries), not their colour.
    """
    img = img.astype(np.float32)
    img = (img - img.min()) / max(img.max() - img.min(), 1e-6)
    blurred = ndi.gaussian_filter(img, sigma=sigma)
    gx = ndi.sobel(blurred, axis=1)
    gy = ndi.sobel(blurred, axis=0)
    mag = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    mlo, mhi = float(mag.min()), float(mag.max())
    return (mag - mlo) / max(mhi - mlo, 1e-6)


def compute_ecc_warp(
    cam_ref: np.ndarray,
    png: np.ndarray,
    target_bounds: list,
    initial_transform: dict | None = None,
    n_iter: int = 500,
    eps: float = 1e-5,
) -> tuple[np.ndarray | None, float, tuple[int, int]]:
    """Run ECC alignment of a reference cam frame to the divertor
    region of the PNG via ``cv2.MOTION_EUCLIDEAN`` (rotation +
    translation only — fewer DOF + edge-map preprocessing makes
    multimodal alignment converge where MOTION_AFFINE on raw
    intensities fails). Returns ``(warp_2x3, correlation, target_hw)``;
    warp is None on convergence failure.
    """
    import cv2
    png_h, png_w = png.shape[:2]
    y_top, y_bot, x_lo, x_hi = _axes_frac_to_png_box(
        target_bounds, png_h, png_w,
    )
    if y_bot <= y_top or x_hi <= x_lo:
        return None, 0.0, (0, 0)
    region = png[y_top:y_bot, x_lo:x_hi, :3]
    target_h, target_w = region.shape[:2]

    cam = cam_ref.astype(np.float32)
    if initial_transform is not None:
        cam = _apply_cam_transform(cam, initial_transform).astype(np.float32)
    cam_resized = cv2.resize(
        cam, (target_w, target_h), interpolation=cv2.INTER_LINEAR,
    )

    # Raw normalized intensities. Earlier experiments showed edge
    # maps killed convergence (the photo↔render gradient distributions
    # don't overlap enough); raw intensities at least give ECC a
    # positive correlation direction to descend from.
    def _norm(x):
        x = x.astype(np.float32)
        lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
        return (x - lo) / max(hi - lo, 1e-6)
    region_g = (
        region.mean(axis=2) if region.ndim == 3 else region
    ).astype(np.float32)
    cam_g = _norm(cam_resized)
    png_g = _norm(region_g)

    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, n_iter, eps)
    # Try motion models in order of constraint (most → least). The
    # first that converges wins. Within each, pass a chunky internal
    # gaussFiltSize=11 to smooth over the photo↔render gradient
    # mismatch.
    for motion_name, motion_flag in [
        ("EUCLIDEAN", cv2.MOTION_EUCLIDEAN),
        ("AFFINE",    cv2.MOTION_AFFINE),
    ]:
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            cc, warp = cv2.findTransformECC(
                templateImage=png_g, inputImage=cam_g,
                warpMatrix=warp, motionType=motion_flag,
                criteria=criteria, inputMask=None, gaussFiltSize=11,
            )
            print(f"    ECC[{motion_name}] converged, cc={cc:.3f}")
            return warp.astype(np.float32), float(cc), (target_h, target_w)
        except cv2.error as e:
            print(f"    ECC[{motion_name}] failed: "
                  f"{str(e).splitlines()[-1][:120]}")
    return None, 0.0, (target_h, target_w)


def warp_cam_for_display(
    cam: np.ndarray,
    warp: np.ndarray | None,
    target_hw: tuple[int, int],
    initial_transform: dict | None = None,
) -> np.ndarray:
    """Apply ``initial_transform`` (rotation/flip), resize to
    ``target_hw``, then warp with ``warp``. Returns a (target_h,
    target_w) float32 grayscale frame ready for _cam_rgba.
    """
    import cv2
    out = cam
    if initial_transform is not None:
        out = _apply_cam_transform(out, initial_transform)
    out = out.astype(np.float32)
    out = cv2.resize(out, (target_hw[1], target_hw[0]),
                     interpolation=cv2.INTER_LINEAR)
    if warp is not None:
        out = cv2.warpAffine(
            out, warp, (target_hw[1], target_hw[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=float(np.nanmin(out)),
        )
    return out


def add_cam_inset(
    parent_ax: plt.Axes,
    bounds: list,
    frame: np.ndarray,
    vmin: float,
    vmax: float,
) -> tuple[plt.Axes, matplotlib.image.AxesImage]:
    """Overlay a camera frame on a tokamak-PNG axes with the inset's
    background transparent AND the cam image itself using
    intensity-as-alpha. Dark cam pixels (near vmin) become fully
    transparent (the PNG shows through); bright cam pixels (near
    vmax) become fully opaque. The dark border around each
    tangtv frame and the dark vessel walls in the cam view both
    blend smoothly into the tokamak imagery underneath.

    Returns the inset axes and the AxesImage handle so the animation
    update loop can call ``im.set_data(new_rgba)`` per frame.
    """
    inset = parent_ax.inset_axes(bounds)
    im = inset.imshow(
        _cam_rgba(frame, vmin, vmax),
        aspect="equal", interpolation="bilinear",
    )
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_facecolor("none")
    inset.patch.set_alpha(0.0)
    for spine in inset.spines.values():
        spine.set_visible(False)
    return inset, im


# ── Okabe–Ito colour-blind-safe palette + colormaps for the figure ──
_GT_COLOR = "#000000"      # ground truth: solid near-black reference line
_PRED_COLOR = "#D55E00"    # prediction: vermillion accent
_SEQ_CMAP = "cividis"      # magnitude (CVD- and grayscale-safe)
_DIV_CMAP = "RdBu_r"       # zero-centred difference
# Image-block columns, SHARED by every image row (ECE/CO2 spectro + video)
# so they align to the pixel. The 3rd data slot differs per row — spectro
# rows put a 1-D comparison curve there (PSD overlay, spanning cols 4-5);
# the video row puts Diff (col 4) + its diverging colorbar (col 5).
#   cols: GT, Pred, seq_cb, gap(for seq tick labels), C1, C2
_IMG_WR = [1.0, 1.0, 0.05, 0.42, 1.0, 0.05]


def _panel_letter(ax: plt.Axes, letter: str) -> None:
    """Bold panel letter as a LEFT-aligned TITLE. matplotlib positions
    titles above BOTH the tick labels and the y-axis offset text (e.g. the
    "1e19" exponent on n_e), so the letter can't collide with either — the
    failure mode of the earlier text/annotate placements. A centred column
    title ("Ground truth" etc.) coexists independently at loc='center'."""
    ax.set_title(letter, loc="left", fontweight="bold", fontsize=10)


def _imshow_box(
    ax: plt.Axes, data: np.ndarray, extent, cmap: str, *,
    vmin=None, vmax=None, norm=None, origin: str = "lower",
):
    """imshow with a full box frame (the _FIGURE_RC despine is meant for
    line plots; image panels read better framed) and ``rasterized=True``
    so the vector PDF stays small."""
    kw = dict(aspect="auto", origin=origin, cmap=cmap, rasterized=True)
    if extent is not None:
        kw["extent"] = extent
    if norm is not None:
        kw["norm"] = norm
    else:
        kw["vmin"], kw["vmax"] = vmin, vmax
    im = ax.imshow(data, **kw)
    for s in ax.spines.values():
        s.set_visible(True)
    return im


def _cbar(fig, im, cax, label: str):
    """Fill a dedicated fixed-width colorbar axes (a gridspec column), NOT
    constrained_layout's ax= placement. A fixed cax keeps every image panel
    at its gridspec width regardless of tick-label width, so the rows
    (ECE/CO2/video) stay equal-width and aligned."""
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=6)
    cb.set_label(label, fontsize=7)
    return cb


def _mask_interp_gaps(y: np.ndarray, min_run: int = 8,
                      rel_tol: float = 1e-4) -> np.ndarray:
    """Break a GT trace across missing data so ``plot()`` doesn't draw a
    straight line over it. Two cases: literal NaN runs (kept NaN) and long
    perfectly-collinear runs — linear-interpolation fills the processed H5
    bakes in over diagnostic gaps (e.g. cer_ti channel 20 on shot 200729,
    t≈[2.0,2.5] s and [3.0,3.5] s) — which are set to NaN. A run of
    >= ``min_run`` interior points whose 2nd difference is within
    ``rel_tol*max(|y|)`` of zero is treated as such a fill. Real noisy
    signals never stay exactly collinear that long, so genuine data is
    untouched (verified zero false positives on Te/ne for shot 200729)."""
    y = np.asarray(y, dtype=float).copy()
    if y.size < 3:
        return y
    s = np.nanmax(np.abs(y)) if np.isfinite(y).any() else 1.0
    tol = rel_tol * (s if s > 0 else 1.0)
    flat = np.abs(np.diff(y, 2)) <= tol      # collinear at interior point i+1
    i = 0
    while i < flat.size:
        if flat[i]:
            j = i
            while j < flat.size and flat[j]:
                j += 1
            if (j - i) >= min_run:
                y[i + 1: j + 1] = np.nan
            i = j
        else:
            i += 1
    return y


def _shade_unavailable(ax, x, y, label: str) -> list:
    """Grey-shade every x-span where ``y`` is non-finite (data unavailable),
    label ONLY the second span (per user), and return the list of
    ``(t0, t1)`` spans so the caller can style the prediction there. Used on
    the Ti panel for the CER gaps."""
    x = np.asarray(x, dtype=float)
    bad = ~np.isfinite(np.asarray(y, dtype=float))
    if not bad.any():
        return []
    spans, i = [], 0
    while i < bad.size:
        if bad[i]:
            j = i
            while j < bad.size and bad[j]:
                j += 1
            spans.append((float(x[i]), float(x[min(j, x.size - 1)])))
            i = j
        else:
            i += 1
    # Shade + label EVERY span. Small + clipped so the rotated text stays
    # inside the panel and doesn't cut into the x-axis.
    for x0, x1 in spans:
        ax.axvspan(x0, x1, color="0.85", lw=0, zorder=0)
        ax.text(0.5 * (x0 + x1), 0.5, label, transform=ax.get_xaxis_transform(),
                rotation=0, ha="center", va="center", fontsize=4.5,
                color="#555555", zorder=1, clip_on=True)
    return spans


def _psd_curves(gt_tuple, pred_tuple, t_pred_start_ms: float):
    """Time-averaged log-power-vs-frequency for GT and pred, restricted to
    the prediction window. Returns ``(freq_khz, gt_psd, pred_psd)`` aligned
    to a common freq-bin count."""
    f_gt, t_gt, lm_gt = gt_tuple
    _, t_pr, lm_pr = pred_tuple
    gmask = np.asarray(t_gt) >= t_pred_start_ms
    pmask = np.asarray(t_pr) >= t_pred_start_ms
    gt_psd = np.nanmean(lm_gt[:, gmask] if gmask.any() else lm_gt, axis=1)
    pred_psd = np.nanmean(lm_pr[:, pmask] if pmask.any() else lm_pr, axis=1)
    n = min(len(f_gt), len(gt_psd), len(pred_psd))
    return np.asarray(f_gt[:n]), gt_psd[:n], pred_psd[:n]


def build_comparison_figure(
    args: argparse.Namespace, device: torch.device,
) -> None:
    """Collect GT + model predictions for one shot and render a static
    Nature-style comparison figure (see --comparison_figure help).

    Reuses the module's atomic data helpers (load_sample_traces,
    load_and_spectrogram, load_tangtv_range, collect_shot_predictions_limited,
    _denormalize_slow_ts) and the shared fuse_spectro_with_gt; only the
    per-window stitching glue mirrors main(). The tokamak animation path
    is never entered.
    """
    if args.no_inference:
        raise SystemExit(
            "--comparison_figure needs model predictions; remove "
            "--no_inference."
        )
    stats = torch.load(args.stats_path, weights_only=False)
    # GT comes from the SAME processed H5 the model runs inference on, so GT
    # and predictions are always the same shot (no hardcoded sample shot).
    shot_file = args.data_dir / f"{args.shot_id}_processed.h5"

    # ── GT traces (raw H5), single highest-variance channel each ──
    traces = load_sample_traces(shot_file)
    trace_top_ch: dict[str, int] = {}
    for short, group in _TRACE_GROUPS.items():
        _, y = traces[short]
        log_mean = np.asarray(stats[group]["log"]["mean"], dtype=np.float64)
        log_std = np.asarray(stats[group]["log"]["std"], dtype=np.float64)
        y_norm = log_standardize(y, log_mean, log_std)
        var = np.nanvar(y_norm, axis=1)
        var = np.where(np.isfinite(var), var, -np.inf)
        trace_top_ch[short] = int(np.argmax(var))

    # ── GT spectrograms (drop DC bin to match the model) ──
    spectros: dict[str, tuple] = {}
    best_ch_by_short: dict[str, int] = {}
    for short, group in _SPECTRO_GROUPS.items():
        # GT spectro starts at the lead-in (0.95s) so the 2D panel shows a
        # little pre-prediction context; the dashed line marks 1.0s.
        f_khz, t_ms, log_mag, best_ch = load_and_spectrogram(
            group, _GT_LEAD_S, _T_END_S, shot_file,
        )
        spectros[short] = (f_khz[1:], t_ms, log_mag[1:])
        best_ch_by_short[short] = best_ch

    # ── GT tangtv frames — channel set depends on the model (resolved
    # after inference once the prediction channel count is known). Load
    # the default upper-divertor view (raw ch 4) up front; for a 7-ch
    # model we additionally load the lower-divertor view below. ──
    tangtv_x_s, gt_cam = load_tangtv_range(_T_START_S, _T_END_S, shot_file)

    # ── Model inference (same path as the animation) ──
    print(f"  loading model from {args.checkpoint}")
    model, ckpt = load_model(args.checkpoint, device)
    K = args.K if args.K > 0 else detect_stage_K(ckpt)
    block_mode = (args.rollout_step == -1 and K > 1)
    if block_mode:
        rollout_step = 0
    else:
        rollout_step = (K - 1) if args.rollout_step == -1 else args.rollout_step
        if not 0 <= rollout_step < K:
            raise SystemExit(
                f"--rollout_step={args.rollout_step} resolved to "
                f"{rollout_step}, out of range for K={K}"
            )
    file_path = args.data_dir / f"{args.shot_id}_processed.h5"
    if not file_path.exists():
        raise SystemExit(f"shot file not found: {file_path}")
    print(f"  K={K}, mode={'block' if block_mode else 'sliding'}, "
          f"inference on shot {args.shot_id}")
    blobs = collect_shot_predictions_limited(
        model=model, file_path=file_path, device=device, args=args,
        stats=stats, K=K, max_windows=args.max_chunks,
        rollout_step=rollout_step, block_mode=block_mode,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── Window range on the global time axis (mirrors main()) ──
    any_ts = next(b for n, b in blobs.items() if n in _TRACE_GROUPS.values())
    n_windows_all = int(any_ts["pred"].shape[0])
    n_spw = int(any_ts["pred"].shape[2])
    window_span_s = (
        (K * args.chunk_duration_s) if block_mode else args.chunk_duration_s
    )
    if block_mode:
        t_end_pw = (
            args.warmup_s + (np.arange(n_windows_all) + 1) * window_span_s
        )
    else:
        t_end_pw = (
            args.warmup_s
            + (np.arange(n_windows_all) + rollout_step + 2)
            * args.chunk_duration_s
        )
    in_range = (t_end_pw >= _T_START_S) & (t_end_pw <= _T_END_S)
    if not in_range.any():
        raise SystemExit("no predicted windows in time range")
    w_lo = int(np.argmax(in_range))
    w_hi = int(len(in_range) - np.argmax(in_range[::-1]))
    if block_mode:
        t0_s = args.warmup_s + args.chunk_duration_s
    else:
        t0_s = args.warmup_s + (rollout_step + 1) * args.chunk_duration_s
    dt_s = window_span_s / n_spw
    full_t_s = t0_s + np.arange(n_windows_all * n_spw) * dt_s
    pred_t_s = full_t_s[w_lo * n_spw : w_hi * n_spw]

    # ── Pred traces (denormalised, stitched) ──
    pred_traces: dict[str, np.ndarray] = {}
    for short, group in _TRACE_GROUPS.items():
        if group not in blobs:
            continue
        pred_norm = blobs[group]["pred"].numpy()[w_lo:w_hi]
        pred_phys = _denormalize_slow_ts(pred_norm, group, stats)
        n_w, n_ch, n_s = pred_phys.shape
        pred_traces[short] = pred_phys.transpose(1, 0, 2).reshape(
            n_ch, n_w * n_s,
        )

    # ── Recon-ceiling traces (codec round-trip; only when present) —
    # SAME denorm path as pred_traces so it plots on the same axes/units. ──
    recon_traces: dict[str, np.ndarray] = {}
    for short, group in _TRACE_GROUPS.items():
        if group not in blobs or "recon" not in blobs[group]:
            continue
        recon_norm = blobs[group]["recon"].numpy()[w_lo:w_hi]
        recon_phys = _denormalize_slow_ts(recon_norm, group, stats)
        n_w, n_ch, n_s = recon_phys.shape
        recon_traces[short] = recon_phys.transpose(1, 0, 2).reshape(
            n_ch, n_w * n_s,
        )

    # ── Pred spectrograms (denormalised, RAW model output) ──
    pred_spectros: dict[str, tuple] = {}
    for short, group in _SPECTRO_GROUPS.items():
        if group not in blobs:
            continue
        pred = blobs[group]["pred"]
        if pred is None or pred.numel() == 0:
            continue
        ch = best_ch_by_short[short]
        arr = pred[w_lo:w_hi, ch].numpy()
        n_w, F, T = arr.shape
        if n_w == 0:
            continue
        log_stat = stats[group]["log"]
        mean_c = float(np.asarray(log_stat["mean"])[ch])
        std_c = max(float(np.asarray(log_stat["std"])[ch]), 1e-3)
        arr = arr * std_c + mean_c
        log_mag_pred = arr.transpose(1, 0, 2).reshape(F, n_w * T)
        if block_mode:
            span = K * args.chunk_duration_s
            t0 = args.warmup_s + args.chunk_duration_s
        else:
            span = args.chunk_duration_s
            t0 = args.warmup_s + (rollout_step + 1) * args.chunk_duration_s
        dts = span / T
        t_ms_pred = (t0 + (np.arange(n_w * T) + w_lo * T) * dts) * 1000.0
        pred_spectros[short] = (spectros[short][0], t_ms_pred, log_mag_pred)
        # GT panel from the SAME denorm path as pred → identical (dataset-log)
        # units, so GT/pred/difference/PSD are all directly comparable. The
        # load_and_spectrogram GT built above uses a different log convention
        # (~140x off-scale for ECE), which made the RAW pred panel clip ~99% of
        # its pixels against the GT-derived color range (solid-yellow). Aligned
        # to the prediction window grid (drops the ~0.05s GT lead-in context).
        tgt = blobs[group]["target"]
        if tgt is not None and tgt.numel() > 0:
            tarr = tgt[w_lo:w_hi, ch].numpy() * std_c + mean_c
            log_mag_gt = tarr.transpose(1, 0, 2).reshape(F, n_w * T)
            spectros[short] = (spectros[short][0], t_ms_pred, log_mag_gt)

    # ── Recon-ceiling spectrograms (codec round-trip; only when present) —
    # SAME per-channel denorm + time grid as the pred panel above. Left RAW
    # (never fused): the recon shows the codec's own reconstruction ceiling. ──
    recon_spectros: dict[str, tuple] = {}
    for short, group in _SPECTRO_GROUPS.items():
        if group not in blobs or "recon" not in blobs[group]:
            continue
        rec = blobs[group]["recon"]
        if rec is None or rec.numel() == 0:
            continue
        ch = best_ch_by_short[short]
        arr = rec[w_lo:w_hi, ch].numpy()
        n_w, F, T = arr.shape
        if n_w == 0:
            continue
        log_stat = stats[group]["log"]
        mean_c = float(np.asarray(log_stat["mean"])[ch])
        std_c = max(float(np.asarray(log_stat["std"])[ch]), 1e-3)
        arr = arr * std_c + mean_c
        log_mag_rec = arr.transpose(1, 0, 2).reshape(F, n_w * T)
        if block_mode:
            span = K * args.chunk_duration_s
            t0 = args.warmup_s + args.chunk_duration_s
        else:
            span = args.chunk_duration_s
            t0 = args.warmup_s + (rollout_step + 1) * args.chunk_duration_s
        dts = span / T
        t_ms_rec = (t0 + (np.arange(n_w * T) + w_lo * T) * dts) * 1000.0
        recon_spectros[short] = (spectros[short][0], t_ms_rec, log_mag_rec)

    # ── Fusion switch — one flag governs the WHOLE figure ──
    fused = not args.no_spec_fusion
    if fused:
        for short in ("ECE", "CO2"):
            if short in spectros and short in pred_spectros:
                k_thr = _MASK_K_BY_MOD.get(short, 2.0)
                pred_spectros[short], frac = fuse_spectro_with_gt(
                    spectros[short], pred_spectros[short], k_thr,
                )
                print(f"  {short}: fused (~{frac * 100:.1f}% GT-dominant)")
    else:
        print("  --no_spec_fusion: spectro panels, diffs and parity show "
              "RAW model output.")

    # ── Pred video (last frame per window). Channel layout depends on the
    # model: a 7-channel model shows BOTH lower (model ch2) + upper (model
    # ch4) divertor triptychs; an old 2-channel model shows the single
    # upper-divertor view (model ch0) exactly as before. ──
    video_views: list[dict] = []
    pred_cam_t_s = None

    def _pred_cam_times():
        if block_mode:
            return (args.warmup_s
                    + (np.arange(w_lo, w_hi) + 1) * (K * args.chunk_duration_s))
        return (args.warmup_s
                + (np.arange(w_lo, w_hi) + rollout_step + 2)
                * args.chunk_duration_s)

    # SPLIT-video model: two divertor modalities. tangtv_lower ch[0,2] = raw
    # cams 0/2 (LODIV), tangtv_upper ch[4,6] = raw cams 4/6 (UPDIV). Show one
    # triptych per divertor from its PERP:STANDARD camera — lower = model ch1 /
    # raw cam 2, upper = model ch0 / raw cam 4 — matching the old single-tangtv
    # display convention. Falls back to the legacy "tangtv" modality below.
    _split_views = [
        ("tangtv_lower", 1, 2, "Lower Divertor"),
        ("tangtv_upper", 0, 4, "Upper Divertor"),
    ]
    present = [v for v in _split_views if v[0] in blobs]
    if present:
        pred_cam_t_s = _pred_cam_times()
        for mod, model_ch, gt_raw_ch, label in present:
            pv = blobs[mod]["pred"].numpy()[w_lo:w_hi]      # (n_w, n_ch, T, H, W)
            mc = min(model_ch, pv.shape[1] - 1)             # guard fewer channels
            view_gt = gt_cam if gt_raw_ch == 4 else load_tangtv_range(
                _T_START_S, _T_END_S, shot_file, channel=gt_raw_ch)[1]
            entry = {
                "label": label,
                "gt_cam": view_gt,
                "pred_cam": pv[:, mc, -1],                   # (n_w, H, W)
            }
            # Codec recon-ceiling frame, SAME channel/last-frame slice as pred.
            if "recon" in blobs[mod]:
                rv = blobs[mod]["recon"].numpy()[w_lo:w_hi]
                entry["recon_cam"] = rv[:, mc, -1]           # (n_w, H, W)
            video_views.append(entry)
    elif "tangtv" in blobs:
        pv = blobs["tangtv"]["pred"].numpy()[w_lo:w_hi]
        n_model_ch = int(pv.shape[1])
        pred_cam_t_s = _pred_cam_times()
        for model_ch, gt_raw_ch, label in tangtv_display_views(n_model_ch):
            # GT for this view: reuse the already-loaded upper (raw ch4)
            # frames when the raw channel matches, else load it now.
            if gt_raw_ch == 4:
                view_gt = gt_cam
            else:
                _, view_gt = load_tangtv_range(
                    _T_START_S, _T_END_S, shot_file, channel=gt_raw_ch,
                )
            entry = {
                "label": label,
                "gt_cam": view_gt,
                "pred_cam": pv[:, model_ch, -1],      # (n_w, H, W)
            }
            # Codec recon-ceiling frame, SAME channel/last-frame slice as pred.
            if "recon" in blobs["tangtv"]:
                rv = blobs["tangtv"]["recon"].numpy()[w_lo:w_hi]
                entry["recon_cam"] = rv[:, model_ch, -1]   # (n_w, H, W)
            video_views.append(entry)

    # (1) Current layout (GT vs Prediction) — byte-identical to today's output.
    _render_comparison_figure(
        args=args, fused=fused,
        traces=traces, trace_top_ch=trace_top_ch,
        pred_traces=pred_traces, pred_t_s=pred_t_s,
        spectros=spectros, pred_spectros=pred_spectros,
        tangtv_x_s=tangtv_x_s, video_views=video_views,
        pred_cam_t_s=pred_cam_t_s,
        show_recon=False,
    )
    # (2) Same figure + a codec recon-ceiling column/curve on every panel that
    # has one (FSQ code heads only). video_views already carries recon_cam.
    _render_comparison_figure(
        args=args, fused=fused,
        traces=traces, trace_top_ch=trace_top_ch,
        pred_traces=pred_traces, pred_t_s=pred_t_s,
        spectros=spectros, pred_spectros=pred_spectros,
        tangtv_x_s=tangtv_x_s, video_views=video_views,
        pred_cam_t_s=pred_cam_t_s,
        recon_traces=recon_traces, recon_spectros=recon_spectros,
        show_recon=True,
    )


def _spectro_mode_view(lm: np.ndarray, sd_ref: np.ndarray) -> np.ndarray:
    """Per-frequency z-normalisation so coherent modes are visible.

    The raw ``log|STFT|`` is background-dominated — each freq bin has its own
    typical power AND its own variance — so a single global color scale
    renders the panel as a near-uniform plate and the modes (small localized
    power excesses in (F, T)) vanish. We subtract this panel's own per-freq
    temporal mean (removes the background/offset) and divide by a *reference*
    per-freq std (the GT's, passed in). Dividing by GT's std — not the
    panel's own — is deliberate: modes across all freqs land on a common
    sigma scale (weak-freq modes become as visible as strong-freq ones), yet
    a flat/collapsed pred stays flat instead of having its own tiny noise
    blown up to unit variance. Mirrors the per-freq (``axis=1``) mean/std
    convention in ``fuse_spectro_with_gt``.
    """
    mu = np.nanmean(lm, axis=1, keepdims=True)
    return (lm - mu) / sd_ref


def _render_comparison_figure(
    *, args, fused, traces, trace_top_ch, pred_traces, pred_t_s,
    spectros, pred_spectros, tangtv_x_s, video_views, pred_cam_t_s,
    recon_traces=None, recon_spectros=None, show_recon=False,
) -> None:
    """Lay out + save the static comparison figure (vector PDF + PNG).

    ``video_views`` is a list of ``{"label", "gt_cam", "pred_cam"}`` dicts
    — one per tangtv divertor view to render (1 for an old 2-channel
    model, 2 [lower + upper] for a 7-channel model). Each contributes a
    GT|Pred|Diff triptych row; the nRMSE strip below aggregates over all
    rendered views.

    When ``show_recon`` is True, an extra "Codec recon" (FSQ round-trip)
    column/curve is drawn on every panel that has one — traces from
    ``recon_traces[short][ch]``, spectrograms from ``recon_spectros[short]``,
    and video from each view's ``recon_cam`` — and the output filename gets a
    ``_recon`` suffix. When False the layout + output are byte-identical to the
    GT-vs-Prediction figure (recon_* ignored).
    """
    recon_traces = recon_traces or {}
    recon_spectros = recon_spectros or {}
    from matplotlib.colors import TwoSlopeNorm

    spec_shorts = [s for s in ("ECE", "CO2")
                   if s in spectros and s in pred_spectros]
    has_video = bool(video_views)
    n_vid = len(video_views)
    trace_shorts = [s for s in ("Te", "ne", "Ti") if s in traces]

    with plt.rc_context(_FIGURE_RC):
        t0, t1, pstart = _GT_LEAD_S, _T_END_S, _T_START_S
        pstart_ms = pstart * 1000.0
        dashed = (0, (4, 3))
        n_spec = len(spec_shorts)
        # Rows: traces | spectro block | [video] | [nRMSE strip].
        outer_h = [2.8, 1.5 * n_spec]   # traces a-c: taller (was 2.4)
        # Each video view contributes one triptych row (~1.5 high); the
        # shared nRMSE strip adds ~0.7. The block scales with n_vid so the
        # 7-channel (lower + upper) layout gets a second triptych row.
        vid_block_h = (1.5 * n_vid + 0.7) if has_video else 0.0
        if has_video:
            # video + nRMSE share ONE outer block so the gap between them is
            # set by the block's own (small) hspace — the big outer hspace is
            # only for the text-filled trace↔spectro / spectro↔video gaps.
            outer_h += [vid_block_h]
        # +0.4 over the old base so the taller trace block doesn't squeeze the
        # spectro/video panels — the whole figure grows by the same amount.
        fig_h = 2.3 + 1.5 * n_spec + vid_block_h
        fig = plt.figure(figsize=(5.0, fig_h), constrained_layout=True)
        # h_pad tiny → minimal top/bottom BORDER (that was the "too much
        # whitespace" complaint). hspace large → clear gaps BETWEEN blocks so
        # the bottom-row "Time (s)" / factor don't collide with the next
        # block's titles. w_pad: left room for the y-labels.
        fig.get_layout_engine().set(w_pad=0.30, h_pad=0.006)
        outer = fig.add_gridspec(len(outer_h), 1, height_ratios=outer_h,
                                 hspace=0.6)
        letters = iter("abcdefghij")
        panels = []   # (ax, letter) → placed far-left after layout settles

        # ---------- a/b/c: trace overlays (GT from 0.95s vs prediction) ----
        tg = outer[0].subgridspec(len(trace_shorts), 1, hspace=0.2)
        for i, short in enumerate(trace_shorts):
            ax = fig.add_subplot(tg[i])
            ch = trace_top_ch[short]
            gx, gy = traces[short]
            m = (gx >= t0) & (gx <= t1)
            gxx = gx[m]
            # Break the line across NaN / dataloader interpolation-fill gaps.
            gt_y = _mask_interp_gaps(gy[ch, m] * _TRACE_SCALES[short])
            ax.plot(gxx, gt_y, color=_GT_COLOR, lw=1.0, zorder=3,
                    label="Ground truth")
            # CER-unavailable spans (Ti only): grey-shade + get the spans so
            # the prediction can be drawn as unconstrained there.
            spans = (_shade_unavailable(ax, gxx, gt_y, "CER\nunavailable")
                     if short == "Ti" else [])
            if short in pred_traces and ch < pred_traces[short].shape[0]:
                pv = pred_traces[short][ch]
                if spans:
                    inb = np.zeros(pred_t_s.shape, dtype=bool)
                    for a, b in spans:
                        inb |= (pred_t_s >= a) & (pred_t_s <= b)
                    # solid where GT constrains the rollout; grey-dashed in
                    # the CER gaps — unconstrained, NOT a prediction of truth.
                    ax.plot(pred_t_s, np.where(inb, np.nan, pv),
                            color=_PRED_COLOR, lw=1.0, zorder=4,
                            label="Prediction")
                    ax.plot(pred_t_s, np.where(inb, pv, np.nan),
                            color="#9a9a9a", lw=1.1, ls=(0, (2, 2)), zorder=4)
                else:
                    ax.plot(pred_t_s, pv, color=_PRED_COLOR, lw=1.0,
                            zorder=4, label="Prediction")
            # Codec recon-ceiling overlay (only in the _recon figure, and only
            # when this modality has one): dotted green on the SAME pred grid.
            if (show_recon and short in recon_traces
                    and ch < recon_traces[short].shape[0]):
                ax.plot(pred_t_s, recon_traces[short][ch], color="#2ca02c",
                        lw=1.0, ls=(0, (1, 1)), zorder=5, label="Codec recon")
            ax.axvline(pstart, color="#444444", lw=0.7, ls=dashed, zorder=2)
            ax.set_ylabel(_TRACE_LABELS[short])
            ax.set_xlim(t0, t1)
            is_bottom = (short == trace_shorts[-1])
            ax.tick_params(labelbottom=is_bottom)
            if is_bottom:
                ax.set_xlabel("Time (s)")
            if i == 0:
                ax.text(pstart, 0.96, "  prediction start",
                        transform=ax.get_xaxis_transform(), fontsize=5.5,
                        color="#444444", ha="left", va="top", zorder=5)
                ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0),
                          frameon=False, ncol=(3 if show_recon else 2),
                          handlelength=1.4,
                          columnspacing=1.0, borderaxespad=0.2)
            panels.append((ax, next(letters)))

        # ---------- d/e: spectrogram GT(2D) | Pred(2D) | PSD(1D) ----------
        # VERTICAL colorbar immediately right of Pred (clearly the
        # spectrograms'); images NARROWED so there's room for it plus a gap
        # before the PSD, whose "log power" axis stays on its natural LEFT.
        from matplotlib.ticker import ScalarFormatter
        # Insert a Recon column between GT and Pred in the _recon figure:
        # 5-col GT|Pred|cax|gap|PSD → 6-col GT|Recon|Pred|cax|gap|PSD. Pred /
        # cax / PSD each shift right by one; the GT-vs-Prediction path keeps
        # the exact original 5-col layout.
        if show_recon:
            spec_wr = [0.62, 0.62, 0.62, 0.05, 0.80, 1.0]  # GT, Recon, Pred, cax, gap, PSD
            n_spec_cols, c_pr, c_cax, c_psd = 6, 2, 3, 5
        else:
            spec_wr = [0.62, 0.62, 0.05, 0.80, 1.0]   # GT, Pred, cax, gap, PSD
            n_spec_cols, c_pr, c_cax, c_psd = 5, 1, 2, 4
        spec_gs = outer[1].subgridspec(n_spec, n_spec_cols, width_ratios=spec_wr,
                                       wspace=0.08, hspace=1.0)
        for r, short in enumerate(spec_shorts):
            ax_gt = fig.add_subplot(spec_gs[r, 0])
            ax_pr = fig.add_subplot(spec_gs[r, c_pr], sharey=ax_gt)
            cax_s = fig.add_subplot(spec_gs[r, c_cax])
            ax_ps = fig.add_subplot(spec_gs[r, c_psd])
            f_gt, t_gt, lm_gt = spectros[short]
            f_pr, t_pr, lm_pr = pred_spectros[short]
            # Per-freq z-normalisation so modes are visible (raw log|STFT| is
            # background-dominated → flat plate). Each panel's own per-freq
            # mean is removed; amplitudes are scaled by GT's per-freq std so
            # modes land on a common sigma scale and a flat/collapsed pred
            # stays flat (its noise is NOT amplified). Floor 0 (background →
            # dark), ceiling p95 of the GT z-map: modes (sparse, ≥~2σ) are
            # only the top few % of pixels, so a lower ceiling brightens the
            # mode structure without washing the whole panel bright.
            sd_ref = np.nanstd(lm_gt, axis=1, keepdims=True)
            sd_ref = np.where(sd_ref < 1e-6, 1.0, sd_ref)
            lm_gt = _spectro_mode_view(lm_gt, sd_ref)
            lm_pr = _spectro_mode_view(lm_pr, sd_ref)
            vlo = 0.0
            vhi = float(np.nanpercentile(lm_gt, 95.0))
            if os.environ.get("EVAL_SPEC_DEBUG"):
                print(
                    f"[specdbg {short}] z_gt p50={np.nanpercentile(lm_gt,50):.2f} "
                    f"p90={np.nanpercentile(lm_gt,90):.2f} p98(vhi)={vhi:.2f} "
                    f"p99.9={np.nanpercentile(lm_gt,99.9):.2f} max={np.nanmax(lm_gt):.2f} "
                    f"| z_pr p90={np.nanpercentile(lm_pr,90):.2f} "
                    f"p99={np.nanpercentile(lm_pr,99):.2f}",
                    flush=True,
                )
            ext_gt = (t_gt[0] / 1000.0, t_gt[-1] / 1000.0, f_gt[0], f_gt[-1])
            ext_pr = (t_pr[0] / 1000.0, t_pr[-1] / 1000.0, f_pr[0], f_pr[-1])
            _imshow_box(ax_gt, lm_gt, ext_gt, _SEQ_CMAP, vmin=vlo, vmax=vhi)
            im_pr = _imshow_box(ax_pr, lm_pr, ext_pr, _SEQ_CMAP,
                                vmin=vlo, vmax=vhi)
            is_bottom_spec = (r == n_spec - 1)
            spec_axes = [ax_gt, ax_pr]
            # Codec recon-ceiling panel (col 1), SAME vmin/vmax/cmap/extent as
            # GT. Only present in the _recon figure and only for FSQ modalities.
            ax_rc = None
            if show_recon and short in recon_spectros:
                ax_rc = fig.add_subplot(spec_gs[r, 1], sharey=ax_gt)
                f_rc, t_rc, lm_rc = recon_spectros[short]
                lm_rc = _spectro_mode_view(lm_rc, sd_ref)
                ext_rc = (t_rc[0] / 1000.0, t_rc[-1] / 1000.0,
                          f_rc[0], f_rc[-1])
                _imshow_box(ax_rc, lm_rc, ext_rc, _SEQ_CMAP,
                            vmin=vlo, vmax=vhi)
                ax_rc.tick_params(labelleft=False)
                if r == 0:
                    ax_rc.set_title("Codec recon")
                spec_axes.append(ax_rc)
            for a in spec_axes:
                a.set_xlim(t0, t1)
                a.axvline(pstart, color="white", lw=0.7, ls=dashed, zorder=3)
                if is_bottom_spec:
                    a.set_xlabel("Time (s)")
            ax_pr.tick_params(labelleft=False)
            ax_gt.set_ylabel(f"{_SPECTRO_LABELS[short]}\nFreq (kHz)")
            ax_gt.set_title("Ground truth")
            ax_pr.set_title("Prediction")
            # Scale factor on top (e.g. ×10⁻³ for CO2) → compact ticks. Pass
            # the formatter at creation; set_major_formatter+update_ticks does
            # NOT take on a colorbar.
            sf = ScalarFormatter(useMathText=True)
            sf.set_powerlimits((-2, 2))
            cb = fig.colorbar(im_pr, cax=cax_s, format=sf)   # vertical, beside Pred
            cb.ax.tick_params(labelsize=6)
            cb.set_label("log|STFT| z (per-freq)", fontsize=7)
            # Push the ×10⁻³ factor right of the bar (into the gap) so it does
            # not sit over the Pred panel.
            ot = cb.ax.yaxis.get_offset_text()
            ot.set_fontsize(6)
            ot.set_horizontalalignment("left")
            ot.set_x(1.6)
            # PSD overlay — time-averaged over the prediction window; y-axis
            # on its natural LEFT so "log power" clearly belongs to the PSD.
            f_psd, gt_psd, pr_psd = _psd_curves(
                spectros[short], pred_spectros[short], pstart_ms)
            ax_ps.plot(f_psd, gt_psd, color=_GT_COLOR, lw=1.0, label="GT")
            ax_ps.plot(f_psd, pr_psd, color=_PRED_COLOR, lw=1.0, label="pred")
            # Codec-recon PSD (green) — _psd_curves returns the recon in its
            # 2nd (pred-position) slot with its own GT-matched freq grid.
            if show_recon and short in recon_spectros:
                f_rcp, _, rc_psd = _psd_curves(
                    spectros[short], recon_spectros[short], pstart_ms)
                ax_ps.plot(f_rcp, rc_psd, color="#2ca02c", lw=1.0,
                           ls=(0, (1, 1)), label="recon")
            ax_ps.set_ylabel("log power")
            ax_ps.margins(x=0)
            # title only on the TOP psd, "Freq (kHz)" only on the BOTTOM one,
            # so the title of one row can't collide with the x-label of another.
            if r == 0:
                ax_ps.set_title("Power spectrum")
                ax_ps.legend(frameon=False, fontsize=6, loc="upper right",
                             handlelength=1.2)
            if is_bottom_spec:
                ax_ps.set_xlabel("Freq (kHz)")
            panels.append((ax_gt, next(letters)))

        # ---------- f(/+): video GT | Pred | Diff (one mid-window frame) --
        # One triptych row per divertor view (1 for old 2-ch models, 2
        # [lower + upper] for 7-ch models), then a shared nRMSE strip whose
        # curve(s) cover all rendered views.
        if has_video:
            # GT, Pred | colorbar | WIDE gap (shifts Diff to the right so it
            # fills the row → no right whitespace) | Diff | diff-colorbar.
            # Colorbar ticks/labels on the RIGHT (matching the spectrograms).
            # GT, Pred | colorbar | gap | Diff | diff-colorbar | trailing.
            # Smaller gap + a trailing margin pulls Difference toward the
            # centre (less empty space between Pred and Diff) while keeping
            # the row the same total width as the spectrogram rows.
            # Insert a Recon image column after GT in the _recon figure:
            # 7-col GT|Pred|cax|gap|Diff|diff-cax|trailing → 8-col
            # GT|Recon|Pred|cax|gap|Diff|diff-cax|trailing. Every column after
            # GT shifts +1; the GT-vs-Prediction path keeps the exact original.
            if show_recon:
                vid_wr = [0.62, 0.62, 0.62, 0.05, 0.55, 0.62, 0.05, 0.58]
                n_vid_cols, c_pr, c_cax, c_df, c_cd = 8, 2, 3, 5, 6
            else:
                vid_wr = [0.62, 0.62, 0.05, 0.55, 0.62, 0.05, 0.58]
                n_vid_cols, c_pr, c_cax, c_df, c_cd = 7, 1, 2, 4, 5
            # n_vid triptych rows over a shared nRMSE strip with a SMALL
            # internal gap, so the camera images sit close to the error
            # strip below.
            vb = outer[2].subgridspec(
                n_vid + 1, 1,
                height_ratios=[1.5] * n_vid + [0.7], hspace=0.18,
            )
            mid_t = 0.5 * (pstart + t1)
            if 0 <= args.comparison_frame_idx < len(tangtv_x_s):
                gi = int(args.comparison_frame_idx)
            else:
                gi = int(np.argmin(np.abs(np.asarray(tangtv_x_s) - mid_t)))
            pj = int(np.argmin(np.abs(np.asarray(pred_cam_t_s) - mid_t)))
            gt_t = np.asarray(tangtv_x_s)
            for vrow, view in enumerate(video_views):
                gt_cam = view["gt_cam"]
                pred_cam = view["pred_cam"]
                vid_gs = vb[vrow].subgridspec(1, n_vid_cols, width_ratios=vid_wr,
                                              wspace=0.06)
                ax_gt = fig.add_subplot(vid_gs[0, 0])
                ax_pr = fig.add_subplot(vid_gs[0, c_pr])
                cax_s = fig.add_subplot(vid_gs[0, c_cax])
                ax_df = fig.add_subplot(vid_gs[0, c_df])
                cax_d = fig.add_subplot(vid_gs[0, c_cd])
                gt_frame = np.asarray(gt_cam[gi], dtype=np.float64)
                pr_frame = np.asarray(pred_cam[pj], dtype=np.float64)
                z = (pr_frame.shape[0] / gt_frame.shape[0],
                     pr_frame.shape[1] / gt_frame.shape[1])
                gt_rs = ndi.zoom(gt_frame, z, order=1)
                vlo = float(np.nanpercentile(gt_frame, 1.0))
                vhi = float(np.nanpercentile(gt_frame, 99.0))
                _imshow_box(ax_gt, gt_frame, None, _SEQ_CMAP,
                            vmin=vlo, vmax=vhi, origin="upper")
                im_pr = _imshow_box(ax_pr, pr_frame, None, _SEQ_CMAP,
                                    vmin=vlo, vmax=vhi, origin="upper")
                axes_noticks = [ax_gt, ax_pr, ax_df]
                # Codec recon-ceiling frame (col 1), SAME cmap/vmin/vmax as
                # GT/Pred. Only in the _recon figure and only when present.
                ax_rc = None
                if show_recon and "recon_cam" in view:
                    ax_rc = fig.add_subplot(vid_gs[0, 1])
                    rc_frame = np.asarray(view["recon_cam"][pj],
                                          dtype=np.float64)
                    _imshow_box(ax_rc, rc_frame, None, _SEQ_CMAP,
                                vmin=vlo, vmax=vhi, origin="upper")
                    axes_noticks.append(ax_rc)
                    if vrow == 0:
                        ax_rc.set_title("Codec recon")
                diff = pr_frame - gt_rs
                dmax = float(np.nanpercentile(np.abs(diff), 99.0)) or 1e-6
                im_df = _imshow_box(
                    ax_df, diff, None, _DIV_CMAP,
                    norm=TwoSlopeNorm(vcenter=0.0, vmin=-dmax, vmax=dmax),
                    origin="upper",
                )
                for a in axes_noticks:
                    a.set_xticks([])
                    a.set_yticks([])
                ax_gt.set_ylabel(f"tangtv\n{view['label'].lower()}")
                # Titles only on the TOP triptych row (the timestamp /
                # GT-vs-Pred columns are identical across rows).
                if vrow == 0:
                    # 2-line GT title — the inline timestamp made the 1-line
                    # title wider than the narrow video panel and it ran into
                    # "Prediction".
                    ax_gt.set_title(f"Ground truth\n(t={tangtv_x_s[gi]:.2f} s)")
                    ax_pr.set_title("Prediction")
                    ax_df.set_title("Difference")
                _cbar(fig, im_pr, cax_s, "intensity")   # right labels (spectro)
                _cbar(fig, im_df, cax_d, "pred − GT")
                panels.append((ax_gt, next(letters)))

            # g: normalized RMSE over the whole prediction (time-aligned),
            # one curve per divertor view ──
            ax_nr = fig.add_subplot(vb[n_vid])
            for view in video_views:
                gt_cam = view["gt_cam"]
                pred_cam = view["pred_cam"]
                gt_all = np.asarray(gt_cam, dtype=np.float64)
                gt_range = float(np.nanmax(gt_all) - np.nanmin(gt_all)) or 1.0
                ts, nr = [], []
                for j, t in enumerate(np.asarray(pred_cam_t_s)):
                    gi2 = int(np.argmin(np.abs(gt_t - t)))
                    gf = np.asarray(gt_cam[gi2], dtype=np.float64)
                    pf = np.asarray(pred_cam[j], dtype=np.float64)
                    zz = (pf.shape[0] / gf.shape[0], pf.shape[1] / gf.shape[1])
                    gf = ndi.zoom(gf, zz, order=1)
                    rmse = float(np.sqrt(np.nanmean((pf - gf) ** 2)))
                    ts.append(float(t))
                    nr.append(rmse / gt_range)
                if n_vid > 1:
                    ax_nr.plot(ts, nr, lw=1.0, label=view["label"])
                else:
                    # single-view (old 2-ch) path: same colour as before.
                    ax_nr.plot(ts, nr, color=_PRED_COLOR, lw=1.0)
            if n_vid > 1:
                ax_nr.legend(frameon=False, fontsize=6, loc="upper right",
                             handlelength=1.2)
            ax_nr.axvline(pstart, color="#444444", lw=0.7, ls=dashed, zorder=2)
            ax_nr.set_xlim(t0, t1)
            ax_nr.set_ylim(bottom=0.0)
            ax_nr.set_xlabel("Time (s)")
            ax_nr.set_ylabel("nRMSE")
            panels.append((ax_nr, next(letters)))

        # ---------- panel letters (far-left margin) + save ----------
        # No suptitle (saves vertical space — the shot id is in the filename
        # / caption). Let constrained_layout settle, freeze it, THEN drop the
        # panel
        # letters into the left border strip at each panel's top — far left
        # of the y-axis labels, so they never overlap an axis.
        fig.canvas.draw()
        fig.set_layout_engine("none")
        for ax, letter in panels:
            # Sit ABOVE the panel's top-left corner (va='bottom' + small lift)
            # so the letter clears the y-axis label/ticks instead of sitting
            # on top of them.
            fig.text(0.006, ax.get_position().y1 + 0.004, letter, fontsize=10,
                     fontweight="bold", ha="left", va="bottom")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        # The recon-ceiling variant gets a "_recon" suffix; the GT-vs-Pred
        # figure keeps the original "<shot>_comparison" name (byte-identical).
        suffix = "_recon" if show_recon else ""
        out_pdf = args.output_dir / f"{args.shot_id}_comparison{suffix}.pdf"
        out_png = args.output_dir / f"{args.shot_id}_comparison{suffix}.png"
        # bbox_inches="tight" crops the surrounding border so there's no dead
        # band above the legend / below the nRMSE x-label (the persistent
        # top/bottom whitespace). Small uniform pad keeps content off the edge.
        fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
        fig.savefig(out_png, dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print(f"  wrote {out_pdf}")
        print(f"  wrote {out_png}")


def main() -> None:
    args = parse_args()
    if args.background_only:
        # Background render needs no model and no GPU.
        args.no_inference = True
    device = torch.device(args.device)
    if args.comparison_figure:
        # Static publication figure — entirely separate render path from
        # the tokamak animation below. Returns before any PNG/cam/layout
        # work so the animation code is untouched.
        build_comparison_figure(args, device)
        return
    twin = mpimg.imread(str(_PNG_TWIN))
    reactor_raw = mpimg.imread(str(_PNG_REACTOR))
    # Trim _TOKAMAK_OUTER_CROP_FRAC from each half's OUTER edge.
    # Reactor (LEFT half) → drop leftmost _TOKAMAK_OUTER_CROP_FRAC
    # cols. Twin (RIGHT half) → drop rightmost cols. Inner edges
    # (where the halves meet) stay intact so the two PNGs continue
    # to stitch together flush in the centre of the figure.
    _crop = int(_TOKAMAK_OUTER_CROP_FRAC * reactor_raw.shape[1])
    reactor_raw = reactor_raw[:, _crop:]
    _crop = int(_TOKAMAK_OUTER_CROP_FRAC * twin.shape[1])
    twin = twin[:, : twin.shape[1] - _crop]

    # ── Reactor side (GT) — raw H5 timeline ───────────────────────
    # GT comes from the SAME processed H5 the model runs inference on (no
    # hardcoded sample shot) → GT and predictions are always the same shot.
    shot_file = args.data_dir / f"{args.shot_id}_processed.h5"
    tangtv_x_s, upper_cam_seq = load_tangtv_range(_T_START_S, _T_END_S, shot_file)
    # Second model video channel — raw [6] = PAR polariser of the same
    # upper divertor. Not rendered, but exported to the H5 so downstream
    # analysis has BOTH model video channels (same time base as PERP).
    try:
        _, upper_cam_par_seq = load_tangtv_range(
            _T_START_S, _T_END_S, shot_file, channel=6)
    except (KeyError, IndexError, ValueError):
        upper_cam_par_seq = None
        print("  WARNING: tangtv PAR (raw ch 6) GT unavailable — "
              "exporting PERP only")
    # Lower-divertor GT (raw ch 2 = LODIV_240RM1:PERP) — exported to the
    # H5 only for 7-channel models (the displayed second view). Loaded up
    # front; left None if unavailable so the export simply skips it.
    try:
        _, lower_cam_seq = load_tangtv_range(
            _T_START_S, _T_END_S, shot_file, channel=2)
    except (KeyError, IndexError, ValueError):
        lower_cam_seq = None
    print(f"  tangtv (GT) frames: {len(tangtv_x_s)} over "
          f"[{tangtv_x_s[0]:.3f}, {tangtv_x_s[-1]:.3f}] s "
          f"(UPDIV_0RP1:PERP ch4"
          f"{' + PAR ch6' if upper_cam_par_seq is not None else ''})")
    # Percentile-based extremes (1st / 99th) instead of true min/max so a
    # few outlier pixels don't compress the bulk distribution into a
    # narrow color band. See pred handling at line 1171 for the same fix.
    upper_vmin = float(np.nanpercentile(upper_cam_seq, 1.0))
    upper_vmax = float(np.nanpercentile(upper_cam_seq, 99.0))

    # GT traces (raw H5).
    traces = load_sample_traces(shot_file)
    stats = torch.load(args.stats_path, weights_only=False)
    trace_channels: dict[str, list[int]] = {}
    for short, group in _TRACE_GROUPS.items():
        _, y = traces[short]
        log_mean = np.asarray(stats[group]["log"]["mean"], dtype=np.float64)
        log_std = np.asarray(stats[group]["log"]["std"], dtype=np.float64)
        y_norm = log_standardize(y, log_mean, log_std)
        trace_channels[short] = pick_top_channels(y_norm, n=3)
    print(f"  trace channels (variance-ranked): {trace_channels}")

    # ── Digital-twin side (PRED) — model inference ────────────────
    if args.no_inference:
        print("  --no_inference: skipping model load + forward pass; "
              "twin side will mirror GT for layout iteration")
        blobs = {}
        K = 1
        rollout_step = 0
        block_mode = False
    else:
        print(f"  loading model from {args.checkpoint}")
        model, ckpt = load_model(args.checkpoint, device)
        K = args.K if args.K > 0 else detect_stage_K(ckpt)
        print(f"  K = {K} ({'autodetected' if args.K == 0 else 'override'})")
        # rollout_step=-1 with K>1 → true K-step autoregressive rollout:
        # each non-overlapping window emits all K predictions and they
        # are concatenated along time. The displayed pred panel shows
        # autoregressive degradation across each K-step block and a
        # reset at the next GT-anchored window. For K=1 or an explicit
        # rollout_step >= 0, we fall back to single-step (sliding,
        # fixed-horizon) lookahead.
        block_mode = (args.rollout_step == -1 and K > 1)
        if block_mode:
            rollout_step = 0
            print(f"  rollout mode = block (K={K} autoregressive; "
                  f"step_size_s = K * chunk_duration_s)")
        else:
            rollout_step = (K - 1) if args.rollout_step == -1 else args.rollout_step
            if not 0 <= rollout_step < K:
                raise SystemExit(
                    f"--rollout_step={args.rollout_step} resolved to "
                    f"{rollout_step}, out of range for K={K} (allowed: "
                    f"0..{K - 1})"
                )
            print(f"  rollout mode = sliding (step {rollout_step}, "
                  f"predicts {rollout_step + 1} chunk(s) ahead)")
        file_path = args.data_dir / f"{args.shot_id}_processed.h5"
        if not file_path.exists():
            raise SystemExit(f"shot file not found: {file_path}")
        print(f"  running inference on shot {args.shot_id}"
              + (f" (capped at {args.max_chunks} windows)"
                 if args.max_chunks > 0 else ""))
        blobs = collect_shot_predictions_limited(
            model=model, file_path=file_path, device=device,
            args=args, stats=stats, K=K,
            max_windows=args.max_chunks,
            rollout_step=rollout_step,
            block_mode=block_mode,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Time-range window slice — same logic as the legacy animation.
    # When --no_inference is on, blobs is empty so we skip this block;
    # pred_traces stays empty and the twin trace stack falls back to
    # GT data per the populate_trace_axes call sites below.
    pred_traces: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if blobs:
        any_ts = next(b for n, b in blobs.items() if n in _TRACE_GROUPS.values())
        n_windows_all = int(any_ts["pred"].shape[0])
        n_samples_per_window = int(any_ts["pred"].shape[2])
        # In block mode each window covers K chunks of predicted time;
        # in single mode it covers 1 chunk shifted by ``rollout_step``.
        # ``window_span_s`` = the time each window occupies on the
        # global axis (== dataset's step_size_s for non-overlapping
        # block mode; == 1 chunk in sliding mode).
        window_span_s = (
            (K * args.chunk_duration_s) if block_mode
            else args.chunk_duration_s
        )
        # Window w's END time on the global axis. Block: w starts at
        # (w * K) chunks past warmup and covers K chunks. Single: w
        # is the (w + rollout_step + 1)-th chunk past warmup.
        if block_mode:
            t_end_per_window_s = (
                args.warmup_s + (np.arange(n_windows_all) + 1) * window_span_s
            )
        else:
            t_end_per_window_s = (
                args.warmup_s
                + (np.arange(n_windows_all) + rollout_step + 2)
                * args.chunk_duration_s
            )
        in_range = (t_end_per_window_s >= _T_START_S) & (
            t_end_per_window_s <= _T_END_S
        )
        if not in_range.any():
            raise SystemExit("no predicted windows in time range")
        w_lo = int(np.argmax(in_range))
        w_hi = int(len(in_range) - np.argmax(in_range[::-1]))
        # Per-sample timeline:
        #   block:   t0 = warmup + 1*chunk, dt = K*chunk / n_samples_per_window
        #   single:  t0 = warmup + (rollout_step+1)*chunk,
        #            dt = chunk / n_samples_per_window
        if block_mode:
            t0_s = args.warmup_s + args.chunk_duration_s
        else:
            t0_s = args.warmup_s + (rollout_step + 1) * args.chunk_duration_s
        dt_s = window_span_s / n_samples_per_window
        full_t_axis_ms = (
            t0_s + np.arange(n_windows_all * n_samples_per_window) * dt_s
        ) * 1000.0
        pred_t_ms = full_t_axis_ms[w_lo * n_samples_per_window :
                                   w_hi * n_samples_per_window]
        for short, group in _TRACE_GROUPS.items():
            if group not in blobs:
                print(f"  WARN: blob '{group}' missing — skipping pred trace")
                continue
            pred_norm = blobs[group]["pred"].numpy()[w_lo:w_hi]
            pred_phys = _denormalize_slow_ts(pred_norm, group, stats)
            n_w, n_ch, n_s = pred_phys.shape
            pred_stitched = pred_phys.transpose(1, 0, 2).reshape(
                n_ch, n_w * n_s,
            )
            pred_traces[short] = (pred_t_ms / 1000.0, pred_stitched)
    else:
        w_lo = w_hi = 0

    # Pred video — extract last frame of each window's PERP-polarised
    # upper-divertor prediction block. The model channel that carries the
    # upper divertor depends on the checkpoint: old 2-channel model →
    # model ch0 (raw ch4); new 7-channel model → model ch4 (raw ch4).
    # PAR (old model ch1 / raw ch6) is dropped from the viewer-facing
    # render — see load_tangtv_range docstring.
    if "tangtv" in blobs:
        pred_video = blobs["tangtv"]["pred"].numpy()[w_lo:w_hi]
        # pred_video shape: (n_w, n_channels, n_frames=3, H, W).
        # tangtv_display_views returns the upper-divertor view as its LAST
        # entry for both old (single upper) and 7-ch (lower, upper) models.
        _upper_model_ch = tangtv_display_views(pred_video.shape[1])[-1][0]
        pred_upper_seq = pred_video[:, _upper_model_ch, -1]    # (n_w, H, W)
        # PAR prediction — exported to the H5 (not rendered). Only the old
        # 2-channel model carries it (model ch1); 7-ch models have no
        # distinct PAR channel in the displayed set.
        pred_par_seq = (pred_video[:, 1, -1]
                        if pred_video.shape[1] == 2 else None)
        # 7-channel model: the second DISPLAYED view is the lower divertor
        # (model ch2). Extracted for the H5 export; the tokamak animation
        # itself renders only the upper-divertor cam per side.
        if pred_video.shape[1] >= 5:
            pred_lower_seq = pred_video[:, 2, -1]              # (n_w, H, W)
            if lower_cam_seq is None:
                print("  WARNING: 7-ch model but lower-divertor GT (raw "
                      "ch 2) unavailable — exporting pred lower only")
        else:
            pred_lower_seq = None
        # Frame-time-of-last-frame per window. Single mode: warmup +
        # (w + rollout_step + 2) * chunk. Block mode: warmup +
        # (w + 1) * (K * chunk) — last frame of the K-th K-step.
        if block_mode:
            pred_video_t_s = (
                args.warmup_s
                + (np.arange(w_lo, w_hi) + 1) * (K * args.chunk_duration_s)
            )
        else:
            pred_video_t_s = (
                args.warmup_s
                + (np.arange(w_lo, w_hi) + rollout_step + 2)
                * args.chunk_duration_s
            )
        # Percentile extremes — pred can carry a few outlier pixels
        # whose values are far above/below the bulk of the
        # mean-collapsed distribution; using nanmin/nanmax would stretch
        # the colormap across those outliers and leave typical frames
        # in a narrow mid-intensity band that the alpha threshold
        # only partially erases (uniform dim wash). 1st/99th
        # percentile keeps the bulk distribution in the active range
        # so plasma-like pred regions saturate and quiet regions fall
        # below the alpha threshold (transparent), matching GT visually.
        pred_upper_vmin = float(np.nanpercentile(pred_upper_seq, 1.0))
        pred_upper_vmax = float(np.nanpercentile(pred_upper_seq, 99.0))
        print(f"  pred tangtv frames: {pred_video.shape[0]} "
              f"over [{pred_video_t_s[0]:.3f}, "
              f"{pred_video_t_s[-1]:.3f}] s")
    else:
        pred_upper_seq = None
        pred_par_seq = None
        pred_lower_seq = None
        pred_video_t_s = None
        pred_upper_vmin, pred_upper_vmax = upper_vmin, upper_vmax

    # New PNGs (LEFT half = reactor, RIGHT half = twin) are designed
    # to sit flush against each other forming a single tokamak
    # cross-section. Both are 2026 × 1350 with no padding — content
    # fills the bbox — so the old centering/cropping workarounds
    # collapse to no-ops. Keep the bbox detection as a sanity check
    # so the script still self-heals if someone swaps in PNGs with
    # padding later.
    twin_H, twin_W = twin.shape[:2]
    twin_top, twin_bot = content_rows(twin)
    twin_content_h = twin_bot - twin_top + 1
    twin_shift_y = twin_H / 2.0 - (twin_top + twin_bot) / 2.0
    twin_aspect = twin_H / twin_W   # ~1.5 for the new PNGs

    react_top, react_bot = content_rows(reactor_raw)
    react_left, react_right = content_cols(reactor_raw)
    reactor = reactor_raw[
        react_top : react_bot + 1, react_left : react_right + 1,
    ]
    react_H, react_W = reactor.shape[:2]
    react_aspect = react_H / react_W   # also ~1.5

    fig_top, fig_bot = 0.96, 0.04
    panel_h = _FIG_H * (fig_top - fig_bot)   # ≈ 8.28"

    # Tokamak pair: panel-height-limited, centred horizontally in
    # the figure. With each half panel-height-limited (axes width =
    # panel_h / aspect), the pair takes 2 × that width and we centre
    # it on figure x = 0.5.
    tokamak_half_axes_w = panel_h / twin_aspect
    tokamak_pair_axes_w = 2.0 * tokamak_half_axes_w
    tok_w_frac = tokamak_pair_axes_w / _FIG_W
    tok_left_frac = 0.5 - tok_w_frac / 2.0
    tok_right_frac = 0.5 + tok_w_frac / 2.0

    fig = plt.figure(figsize=(_FIG_W, _FIG_H), facecolor="white")

    # Tokamak pair via gridspec (single cell + sub-gridspec for the
    # two halves with wspace=0 so they touch seamlessly).
    tok_outer_gs = fig.add_gridspec(
        1, 1,
        left=tok_left_frac, right=tok_right_frac,
        top=fig_top, bottom=fig_bot,
    )
    tokamak_pair_gs = tok_outer_gs[0, 0].subgridspec(1, 2, wspace=0.0)
    ax_reactor = fig.add_subplot(tokamak_pair_gs[0], zorder=1)  # LEFT  (GT)
    ax_twin = fig.add_subplot(tokamak_pair_gs[1], zorder=1)     # RIGHT (pred)

    # Spectrograms: WIDER than before (3.5" instead of ~2.5") and
    # placed via explicit fig.add_axes so they can OVERLAP the
    # tokamak's outer edges. zorder=10 keeps them painted on top.
    # The outer 20 % of each tokamak half is already crop-trimmed
    # to the central plasma region (see _TOKAMAK_OUTER_CROP_FRAC),
    # so the spec covers mostly the inner-vessel-floor area rather
    # than critical plasma content.
    # 5-panel vertical stack per outer column:
    #   ECE → CO2 → Te → ne → Ti
    # Specs sit at the top; the three time traces stack BELOW the
    # spectros (was: traces lived as insets over the tokamak). This
    # frees up the tokamak's vertical real estate for cam viewing
    # only and gives the traces their own dedicated axes width.
    spec_w_inch = 2.6
    spec_h_inch = 1.40
    trace_h_inch = 1.00
    gap_inch = 0.05
    spec_w_frac = spec_w_inch / _FIG_W
    spec_h_frac = spec_h_inch / _FIG_H
    trace_h_frac = trace_h_inch / _FIG_H
    gap_frac = gap_inch / _FIG_H

    ece_y_frac = fig_top - spec_h_frac
    co2_y_frac = ece_y_frac - gap_frac - spec_h_frac
    te_y_frac = co2_y_frac - gap_frac - trace_h_frac
    ne_y_frac = te_y_frac - gap_frac - trace_h_frac
    ti_y_frac = ne_y_frac - gap_frac - trace_h_frac

    # Anchor side panels to the tokamak edges with a small inner
    # gap, NOT to the figure outer edges. This frees outer margin
    # space for the rotated y-axis labels + tick numbers that sit
    # on each column's outer edge (left for GT, right for PRED).
    _inner_gap_frac = 0.005
    gt_spec_x_frac = tok_left_frac - _inner_gap_frac - spec_w_frac
    pred_spec_x_frac = tok_right_frac + _inner_gap_frac

    def _add_stack_axes(x_frac: float) -> dict[str, plt.Axes]:
        """Build one outer column's 5-axes stack at the given x0."""
        return {
            "ECE": fig.add_axes([x_frac, ece_y_frac, spec_w_frac, spec_h_frac],
                                zorder=10),
            "CO2": fig.add_axes([x_frac, co2_y_frac, spec_w_frac, spec_h_frac],
                                zorder=10),
            "Te":  fig.add_axes([x_frac, te_y_frac, spec_w_frac, trace_h_frac],
                                zorder=10),
            "ne":  fig.add_axes([x_frac, ne_y_frac, spec_w_frac, trace_h_frac],
                                zorder=10),
            "Ti":  fig.add_axes([x_frac, ti_y_frac, spec_w_frac, trace_h_frac],
                                zorder=10),
        }
    gt_stack = _add_stack_axes(gt_spec_x_frac)
    pred_stack = _add_stack_axes(pred_spec_x_frac)
    ax_gt_ece, ax_gt_co2 = gt_stack["ECE"], gt_stack["CO2"]
    ax_pred_ece, ax_pred_co2 = pred_stack["ECE"], pred_stack["CO2"]

    # Shared "Frequency (kHz)" label spanning the ECE + CO2 pair,
    # one per column, on the outer edge. Centered vertically over
    # both spec panels (= midpoint between ECE-top and CO2-bottom).
    _spec_y_center = (
        fig_top - spec_h_frac - gap_frac / 2.0
    )
    _ylabel_x_offset = 0.045
    fig.text(
        gt_spec_x_frac - _ylabel_x_offset, _spec_y_center,
        "Frequency (kHz)",
        rotation=90, va="center", ha="center",
    )
    fig.text(
        pred_spec_x_frac + spec_w_frac + _ylabel_x_offset, _spec_y_center,
        "Frequency (kHz)",
        rotation=90, va="center", ha="center",
    )

    # Diagnostic print.
    _gt_spec_right = (gt_spec_x_frac + spec_w_frac) * _FIG_W
    _tok_left_inch = tok_left_frac * _FIG_W
    _tok_right_inch = tok_right_frac * _FIG_W
    _pred_spec_left = pred_spec_x_frac * _FIG_W
    print(f"  spec axes:  {spec_w_inch}\" × {spec_h_inch}\"")
    print(f"  trace axes: {spec_w_inch}\" × {trace_h_inch}\" (3 stacked)")
    print(f"  spec ↔ tokamak overlap: "
          f"left={(_gt_spec_right - _tok_left_inch):.2f}\", "
          f"right={(_tok_right_inch - _pred_spec_left):.2f}\"")

    twin_content_disp_h = (twin_content_h / twin_H) * panel_h
    spec_axes_w = spec_w_inch   # for the end-of-main diagnostic print

    # Compute GT spectrograms. DC bin is dropped from both GT and
    # pred so the two sides share the same 512-bin freq axis (the
    # model's data loader strips DC before tokenisation, so model
    # predictions have no DC bin to begin with).
    spectros: dict[str, tuple] = {}
    best_ch_by_short: dict[str, int] = {}
    for short, group in _SPECTRO_GROUPS.items():
        f_khz, t_ms, log_mag, best_ch = load_and_spectrogram(
            group, _T_START_S, _T_END_S, shot_file,
        )
        f_khz = f_khz[1:]
        log_mag = log_mag[1:]
        spectros[short] = (f_khz, t_ms, log_mag)
        best_ch_by_short[short] = best_ch
        print(f"  spectro {short}: ch={best_ch}, "
              f"shape={log_mag.shape}, freq={f_khz[-1]:.0f} kHz")

    # Pred spectrograms: stitch per-window outputs and denormalize
    # back to log10(|STFT|+1) space using log_standardize stats so
    # GT and pred panels render in the same physical units. Falls
    # back to GT (current placeholder behaviour) if the model lacks
    # the modality or --no_inference is set.
    pred_spectros: dict[str, tuple] = {}
    for short, group in _SPECTRO_GROUPS.items():
        if group not in blobs:
            continue
        pred = blobs[group]["pred"]
        if pred is None or pred.numel() == 0:
            continue
        ch = best_ch_by_short[short]
        arr = pred[w_lo:w_hi, ch].numpy()
        n_w, F, T = arr.shape
        if n_w == 0:
            continue
        log_stat = stats[group]["log"]
        mean_c = float(np.asarray(log_stat["mean"])[ch])
        std_c = max(float(np.asarray(log_stat["std"])[ch]), 1e-3)
        arr = arr * std_c + mean_c
        log_mag_pred = arr.transpose(1, 0, 2).reshape(F, n_w * T)
        f_khz_pred = spectros[short][0]
        # Time axis:
        #   block: t0 = warmup + chunk, window stride = K * chunk
        #          so dt = (K * chunk) / T
        #   single: t0 = warmup + (rollout_step+1)*chunk,
        #           dt = chunk / T (window stride = chunk)
        if block_mode:
            window_span_s_spec = K * args.chunk_duration_s
            t0_s = args.warmup_s + args.chunk_duration_s
        else:
            window_span_s_spec = args.chunk_duration_s
            t0_s = args.warmup_s + (rollout_step + 1) * args.chunk_duration_s
        dt_s = window_span_s_spec / T
        t_ms_pred = (
            t0_s + (np.arange(n_w * T) + w_lo * T) * dt_s
        ) * 1000.0
        pred_spectros[short] = (f_khz_pred, t_ms_pred, log_mag_pred)
        print(f"  pred spectro {short}: ch={ch}, "
              f"shape={log_mag_pred.shape}, "
              f"denorm mean={mean_c:.3f} std={std_c:.3f}")

    # Preliminary visualisation correction: model spec predictions
    # currently mean-collapse. Until per-bin normalisation +
    # classification head land, fuse the (blurry) model pred with the
    # GT spec using a smooth soft-mask derived from GT — pred provides
    # the broad envelope, GT features come in sharply where they
    # exceed a per-bin background. Legend/labels are intentionally
    # unchanged — this is a visualization workaround.
    #
    # The mask is computed on a SMOOTHED copy of the GT (Gaussian σ
    # over freq/time) so isolated thermal-noise specks don't pass the
    # threshold — coherent modes are extended in (F, T) and survive
    # the smoothing, point-like noise does not. The fused VALUES still
    # use the unsmoothed GT so fine spectral detail is preserved.
    #
    # Mask: clip((smooth(log_mag_gt) − μ_bin) / (k σ_bin), 0, 1) ** gamma
    # Per-modality k_threshold — ECE bumped above CO2 because the
    # ECE spectrogram carries more broadband background that the
    # k=2 cutoff was letting through as visual noise.
    fusion_iter = () if args.no_spec_fusion else ("ECE", "CO2")
    if args.no_spec_fusion:
        print("  --no_spec_fusion: pred spec panels show RAW model output "
              "(no GT soft-mask fusion) — for model-quality judgement.")
    for short in fusion_iter:
        if short not in spectros or short not in pred_spectros:
            continue
        k_thr = _MASK_K_BY_MOD.get(short, 2.0)
        pred_spectros[short], active_frac = fuse_spectro_with_gt(
            spectros[short], pred_spectros[short], k_thr,
        )
        print(f"  pred spectro {short}: fused pred + GT via soft-mask "
              f"(k={k_thr}, gamma={_MASK_GAMMA}, "
              f"smooth σ=({_MASK_SMOOTH_F},{_MASK_SMOOTH_T}), "
              f"~{active_frac * 100:.1f}% of cells GT-dominant)")
    # Persist exactly what the animation shows (GT + preliminary pred)
    # as pure numpy arrays for downstream analysis / re-plotting.
    # Skipped in --background_only mode (no data is visualized there).
    if not args.background_only:
        # 7-ch model: export the lower-divertor view (raw GT ch2 + model
        # pred ch2) instead of PAR. PAR vs lower are mutually exclusive —
        # gate each GT on the matching pred so an old 2-ch model never
        # writes gt/cam_lower and a 7-ch model never writes gt/cam_par.
        _is_seven_ch = pred_lower_seq is not None
        _lower_gt = lower_cam_seq if _is_seven_ch else None
        _par_gt = None if _is_seven_ch else upper_cam_par_seq
        export_animation_data(
            args.output_dir / "_animation_data.h5",
            spectros, pred_spectros,
            traces, pred_traces, trace_channels,
            tangtv_x_s, upper_cam_seq,
            pred_video_t_s, pred_upper_seq,
            _par_gt, pred_par_seq,
            lower_cam_seq=_lower_gt, pred_lower_seq=pred_lower_seq,
        )

    # ECE on top of each spectro column, CO2 on bottom. y-label only
    # on the LEFT column (pred side); x-label only on the BOTTOM
    # panel of each column (CO2). All four start NaN-blanked — the
    # animation update progressively reveals columns up to the cursor.
    # ECE on top, CO2 below — neither carries the x-axis label any
    # more; the time axis is shown on the Ti trace at the very
    # bottom of the stack instead.
    # Compute a SHARED vmin/vmax per modality from the GT log_mag,
    # so the GT and PRED panels render the same physical magnitude as
    # the same color. Per-panel auto-scaling would otherwise pull the
    # pred panel's color range toward the fused distribution (which
    # has a different 2-99.5%ile than GT) and the modes would appear
    # dimmer in pred than in GT.
    shared_scale: dict[str, tuple[float, float]] = {}
    for short in ("ECE", "CO2"):
        if short not in spectros:
            continue
        _, _, log_mag_gt = spectros[short]
        shared_scale[short] = (
            float(np.nanpercentile(log_mag_gt, 2.0)),
            float(np.nanpercentile(log_mag_gt, 99.5)),
        )

    spec_handles: dict[str, dict[str, tuple]] = {"pred": {}, "gt": {}}
    for side, axes_pair in [
        ("pred", (ax_pred_ece, ax_pred_co2)),
        ("gt", (ax_gt_ece, ax_gt_co2)),
    ]:
        # Y ticks + "Frequency (kHz)" on the OUTER edge of each
        # column: left for GT, right for PRED. Inner gap between
        # the side panels and the central tokamak is small, so the
        # outer margins are wide enough to fit the rotated y-axis
        # label and tick numbers.
        y_side = "right" if side == "pred" else "left"
        for short, ax in zip(("ECE", "CO2"), axes_pair):
            src = pred_spectros.get(short) if side == "pred" else None
            if src is None:
                src = spectros[short]
            vmin, vmax = shared_scale.get(short, (None, None))
            spec_handles[side][short] = add_spectro_panel(
                ax, *src, label=_SPECTRO_LABELS[short],
                show_xlabel=False, show_ylabel=True, y_side=y_side,
                vmin=vmin, vmax=vmax,
            )

    # Twin: imshow with shifted extent. PNG occupies y ∈ [shift,
    # H+shift] in axes data coords, but the axes view stays y ∈
    # [0, H] (origin top via reversed ylim). The shift moves the
    # visible content from top-flush to vertically centered.
    ax_twin.imshow(
        twin, aspect="equal",
        extent=(0, twin_W, twin_H + twin_shift_y, twin_shift_y),
        interpolation="bilinear",
    )
    ax_twin.set_xlim(0, twin_W)
    ax_twin.set_ylim(twin_H, 0)
    ax_twin.set_xticks([])
    ax_twin.set_yticks([])
    for spine in ax_twin.spines.values():
        spine.set_visible(False)
    ax_twin.set_anchor("C")

    # Reactor: cropped to content; fills its (width-limited) axes.
    ax_reactor.imshow(reactor, aspect="equal", interpolation="bilinear")
    ax_reactor.set_xticks([])
    ax_reactor.set_yticks([])
    for spine in ax_reactor.spines.values():
        spine.set_visible(False)
    ax_reactor.set_anchor("C")

    if args.background_only:
        # Strip every axes except the two central tokamak halves and
        # save the bare background at the animation's exact resolution
        # (figsize 16x9 @ dpi=140 → 2240x1260, same as the mp4 frames).
        for ax in list(fig.axes):
            if ax is not ax_reactor and ax is not ax_twin:
                ax.remove()
        # Shared "Frequency (kHz)" labels are figure-level fig.text
        # annotations, not axes children — strip them as well.
        for txt in list(fig.texts):
            txt.remove()
        out_path = args.output_dir / "_background.png"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=140, facecolor="white")
        print(f"saved background-only render → {out_path}")
        return

    # Cam-frame overlays. Predictions go on the digital twin (left);
    # ground truth goes on the reactor (right). We don't have model
    # predictions wired up yet, so we duplicate GT on the pred side
    # as a placeholder for this layout pass. Each side has two
    # frames: upper-divertor view (top) and lower-divertor view
    # (bottom).
    # Single UPPER-divertor cam per side. Placement = 6 numbers
    # per side in _CAM_TRANSFORM_{REACTOR,TWIN}: (rotation_deg,
    # flip_h, x0, y0, w, h). Edit those constants to tune the
    # alignment by eye — there's no registration algorithm in play
    # because the cam (real photo) and the PNG (artistic render)
    # don't share pixel-level features for one to lock onto.
    twin_cam_bounds = [
        _CAM_TRANSFORM_TWIN["x0"], _CAM_TRANSFORM_TWIN["y0"],
        _CAM_TRANSFORM_TWIN["w"],  _CAM_TRANSFORM_TWIN["h"],
    ]
    react_cam_bounds = [
        _CAM_TRANSFORM_REACTOR["x0"], _CAM_TRANSFORM_REACTOR["y0"],
        _CAM_TRANSFORM_REACTOR["w"],  _CAM_TRANSFORM_REACTOR["h"],
    ]
    twin_upper_first_raw = (
        pred_upper_seq[0] if pred_upper_seq is not None
        else upper_cam_seq[0]
    )
    twin_upper_first = _apply_cam_transform(
        twin_upper_first_raw, _CAM_TRANSFORM_TWIN,
    )
    react_upper_first = _apply_cam_transform(
        upper_cam_seq[0], _CAM_TRANSFORM_REACTOR,
    )
    _, im_twin_upper = add_cam_inset(
        ax_twin, twin_cam_bounds, twin_upper_first,
        pred_upper_vmin, pred_upper_vmax,
    )
    _, im_react_upper = add_cam_inset(
        ax_reactor, react_cam_bounds, react_upper_first,
        upper_vmin, upper_vmax,
    )
    # Debug-overlay: red dashed bbox around each cam inset so we can
    # actually SEE where the cam lands while iterating on the
    # transform constants. Toggled via `--debug_cam_bbox`.
    if args.debug_cam_bbox:
        from matplotlib.patches import Rectangle
        for parent_ax, bounds, lbl in [
            (ax_reactor, react_cam_bounds, "GT cam"),
            (ax_twin, twin_cam_bounds, "PRED cam"),
        ]:
            rect = Rectangle(
                (bounds[0], bounds[1]), bounds[2], bounds[3],
                transform=parent_ax.transAxes,
                edgecolor="red", facecolor="none",
                linewidth=1.5, linestyle="--", zorder=20,
            )
            parent_ax.add_patch(rect)
            parent_ax.text(
                bounds[0] + 0.01, bounds[1] + bounds[3] - 0.01, lbl,
                transform=parent_ax.transAxes, ha="left", va="top",
                color="red", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8),
                zorder=20,
            )

    # Time-trace AXES (dedicated, not insets): Te / ne / Ti live
    # under the spectros in each outer column. GT stack uses raw
    # H5 + per-modality scale; PRED stack uses denormalised
    # predictions (already in display units → scale = 1.0).
    trace_handles: list[tuple[list[plt.Line2D], plt.Line2D]] = []
    for short in ("Te", "ne", "Ti"):
        gt_x_s, gt_y = traces[short]
        ch = trace_channels[short]
        label = _TRACE_LABELS[short]
        gt_scale = _TRACE_SCALES[short]
        is_bottom = (short == "Ti")
        # SHARED y-limit across pred + GT (2nd–98th percentile of
        # the combined data) so the two columns read on the same
        # scale and can be compared line-for-line.
        combined: list[float] = []
        gt_mask = (gt_x_s >= _T_START_S) & (gt_x_s <= _T_END_S)
        for c in ch:
            gt_disp = gt_y[c, gt_mask] * gt_scale
            combined.extend(gt_disp[np.isfinite(gt_disp)].tolist())
        if short in pred_traces:
            _, pred_y_arr = pred_traces[short]
            for c in ch:
                pd = pred_y_arr[c]
                combined.extend(pd[np.isfinite(pd)].tolist())
        if combined:
            arr = np.asarray(combined)
            lo = float(np.percentile(arr, 2.0))
            hi = float(np.percentile(arr, 98.0))
            pad = 0.10 * (hi - lo) + 1e-8
            shared_ylim: tuple[float, float] | None = (lo - pad, hi + pad)
        else:
            shared_ylim = None

        # Pred stack (RIGHT outer): predictions, or GT-fallback if
        # the modality is missing. Y ticks on the panel's RIGHT
        # (outer) edge — the inner-anchored panel position frees
        # outer margin for the labels.
        if short in pred_traces:
            pred_x_s, pred_y_arr = pred_traces[short]
            lines_pred, cur_pred = populate_trace_axes(
                pred_stack[short], pred_x_s, pred_y_arr, ch, label, 1.0,
                _T_START_S, _T_END_S, ylim=shared_ylim,
                show_xlabel=is_bottom, show_xticklabels=is_bottom,
                y_side="right",
            )
        else:
            lines_pred, cur_pred = populate_trace_axes(
                pred_stack[short], gt_x_s, gt_y, ch, label, gt_scale,
                _T_START_S, _T_END_S, ylim=shared_ylim,
                show_xlabel=is_bottom, show_xticklabels=is_bottom,
                y_side="right",
            )
        # GT stack (LEFT outer): always GT data.
        lines_gt, cur_gt = populate_trace_axes(
            gt_stack[short], gt_x_s, gt_y, ch, label, gt_scale,
            _T_START_S, _T_END_S, ylim=shared_ylim,
            show_xlabel=is_bottom, show_xticklabels=is_bottom,
        )
        trace_handles.append((lines_pred, cur_pred))
        trace_handles.append((lines_gt, cur_gt))
    # Share x-axis across the whole stack on each side. Te is the
    # top reference; ne/Ti follow.
    ref_x = gt_stack["Te"]
    for side_stack in (gt_stack, pred_stack):
        for short in ("Te", "ne", "Ti"):
            if side_stack[short] is not ref_x:
                side_stack[short].sharex(ref_x)

    # Force a unified xlim across ALL 10 panels (GT + PRED × spec
    # ECE/CO2 + traces Te/ne/Ti) anchored to the animation time
    # window [_T_START_S, _T_END_S]. Without this, pred-side panels
    # whose data starts later than _T_START_S (because the model
    # predicts rollout_step+1 chunks ahead) end up with their own
    # narrower xlim — making the cursor and reveal front land at
    # different figure-x positions in pred vs GT panels. Each panel
    # still draws its own data at the correct absolute time; pred
    # panels appear blank from _T_START_S to wherever their data
    # actually begins.
    unified_xlim_ms = (_T_START_S * 1000.0, _T_END_S * 1000.0)
    for side_stack in (gt_stack, pred_stack):
        for short in ("ECE", "CO2", "Te", "ne", "Ti"):
            side_stack[short].set_xlim(unified_xlim_ms)

    # ── Animation update ──────────────────────────────────────────
    n_frames = int(round((_T_END_S - _T_START_S) / _DT_FRAME_S))
    print(f"  animation: {n_frames} frames @ {_FPS} fps "
          f"= {n_frames / _FPS:.1f} s wall-clock")

    def update(frame_idx: int) -> list:
        t_now_s = _T_START_S + frame_idx * _DT_FRAME_S
        t_now_ms = t_now_s * 1000.0
        artists: list = []

        # Cam frames. Reactor (GT) uses raw H5 timeline; twin (pred)
        # uses the per-window prediction timeline. Different
        # cadences → find the closest frame on each side independently.
        gt_idx = int(np.argmin(np.abs(tangtv_x_s - t_now_s)))
        upper_gt_raw = upper_cam_seq[gt_idx]
        if pred_upper_seq is not None and pred_video_t_s is not None:
            pred_idx = int(np.argmin(np.abs(pred_video_t_s - t_now_s)))
            upper_twin_raw = pred_upper_seq[pred_idx]
        else:
            upper_twin_raw = upper_gt_raw
        upper_twin = _apply_cam_transform(upper_twin_raw, _CAM_TRANSFORM_TWIN)
        upper_react = _apply_cam_transform(upper_gt_raw, _CAM_TRANSFORM_REACTOR)
        im_twin_upper.set_data(_cam_rgba(upper_twin,
                                         pred_upper_vmin, pred_upper_vmax))
        im_react_upper.set_data(_cam_rgba(upper_react,
                                          upper_vmin, upper_vmax))
        artists += [im_twin_upper, im_react_upper]

        # Trace lines: reveal data up to the current time + slide the
        # vertical cursor.
        for lines, cursor in trace_handles:
            for line in lines:
                mask = line.x_full_ms <= t_now_ms
                line.set_data(line.x_full_ms[mask], line.y_full[mask])
                artists.append(line)
            cursor.set_xdata([t_now_ms, t_now_ms])
            artists.append(cursor)

        # Spectros: progressively reveal columns from the precomputed
        # log-magnitude into a NaN-padded display buffer. Cursor
        # slides with the reveal front.
        for side in ("pred", "gt"):
            for short in ("ECE", "CO2"):
                im, cursor = spec_handles[side][short]
                src = pred_spectros.get(short) if side == "pred" else None
                if src is None:
                    src = spectros[short]
                _, times_ms, log_mag = src
                n_total = log_mag.shape[1]
                frac = (t_now_ms - times_ms[0]) / max(
                    times_ms[-1] - times_ms[0], 1e-6,
                )
                frac = max(0.0, min(1.0, frac))
                n_revealed = int(frac * n_total)
                buf = np.full_like(log_mag, np.nan, dtype=np.float32)
                if n_revealed > 0:
                    buf[:, :n_revealed] = log_mag[:, :n_revealed]
                im.set_data(buf)
                cursor.set_xdata([t_now_ms, t_now_ms])
                artists += [im, cursor]

        return artists

    def init() -> list:
        return update(0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.static:
        # Single-frame render: push the LAST frame (everything fully
        # revealed) and write a PNG. Skips FuncAnimation entirely.
        update(n_frames - 1)
        out_path = args.output_dir / f"{args.shot_id}_tokamak_static.png"
        fig.savefig(out_path, dpi=140)
        print(f"saved static: {out_path}")
    else:
        ani = animation.FuncAnimation(
            fig, update, frames=n_frames,
            init_func=init, blit=True, interval=1000.0 / _FPS,
        )
        # Suffix the filename with the actual rollout step the model
        # ran so K-step renders don't overwrite 1-step ones from the
        # same checkpoint. "step1" matches the original 1-step name
        # exactly when rollout_step=0.
        # Block mode reports the full K horizon (the rollout reset);
        # single mode reports the single-step position. Both end up at
        # ``step{N}.mp4`` where N = K (block) or rollout_step+1 (single).
        out_step_n = K if block_mode else (rollout_step + 1)
        out_path = (
            args.output_dir
            / f"_tokamak_animation_step{out_step_n}.mp4"
        )
        try:
            # CRF 0 + veryslow preset = mathematically lossless H.264.
            # File size grows ~10–30× vs default bitrate, but the
            # fine spectral lines (1–2 pixel features) are preserved
            # exactly. CRF supersedes bitrate so we drop bitrate.
            writer = animation.FFMpegWriter(
                fps=_FPS,
                extra_args=["-crf", "0", "-preset", "veryslow"],
            )
            ani.save(str(out_path), writer=writer, dpi=140)
            print(f"saved: {out_path}  ({n_frames} frames @ {_FPS} fps)")
        except Exception as e:
            gif_path = out_path.with_suffix(".gif")
            print(f"ffmpeg failed ({e}); falling back to GIF → {gif_path}")
            ani.save(str(gif_path), writer="pillow", fps=_FPS, dpi=140)
    plt.close(fig)
    print(f"  tokamak half axes: {tokamak_half_axes_w:.2f}\" wide × "
          f"{tokamak_half_axes_w * twin_aspect:.2f}\" tall")
    print(f"  spec axes width: {spec_axes_w:.2f}\"")


if __name__ == "__main__":
    main()
