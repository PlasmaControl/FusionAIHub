"""One-off experimental plot.

Apply Stage 1 best.pt to shot 200729 with per-(channel, freq_bin)
log-magnitude normalisation for spectrograms, where the per-bin stats
are computed from THIS SHOT only (not from the global preprocessing
stats). All other modalities use the existing channel-wise stats.

Stage 1 was trained with channel-wise input normalisation, so feeding
per-bin normalised inputs is off-distribution — this is the experiment
we want to see. The resulting spec predictions are denormalised
back to log10(|STFT|+1) space using the same per-bin stats and rendered
side-by-side with the GT spectrogram for ECE, CO2, BES.

Output: ``eval_runs/animations/200729_per_bin_stage1.png``
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "training"))

from tokamak_foundation_model.data.data_loader import collate_fn  # noqa: E402
from tokamak_foundation_model.data.multi_file_dataset import (  # noqa: E402
    TokamakMultiFileDataset,
)
from eval_e2e import (  # noqa: E402
    make_rollout_if_needed,
    rollout_forward_one_batch,
)
from eval_e2e_animation_tokamak import load_model  # noqa: E402


SPEC_NAMES = ("ece", "co2", "bes")
# Per-modality channel slice that the data_loader applies on top of
# the raw HDF5 channel axis. Must match
# ``SignalConfig.channels_to_use`` in ``data_loader.py`` for these
# three signals — duplicated here only so this one-off script can
# operate on the same channel subset the model was trained on.
#   ece: slice(0, 40)    — skip last 8 channels
#   co2: None            — all 4 channels
#   bes: slice(48, 64)   — only 2 poloidal rows (indices 48-63)
SPEC_CHANNEL_SLICES: dict[str, slice | None] = {
    "ece": slice(0, 40),
    "co2": None,
    "bes": slice(48, 64),
}
DEFAULT_SHOT = "/lustre/orion/fus187/proj-shared/foundation_model/200729_processed.h5"
DEFAULT_CKPT = (
    "/lustre/orion/fus187/proj-shared/models/e2e_stage1_d1024_48L/"
    "e2e_stage1_best.pt"
)
DEFAULT_STATS = (
    "/lustre/orion/fus187/proj-shared/foundation_model_meta/"
    "preprocessing_stats.pt"
)
DEFAULT_OUT = "eval_runs/animations/200729_per_bin_stage1.png"


def compute_local_per_bin_stats(
    shot_path: Path, n_fft: int = 1024, hop_length: int = 256,
) -> dict[str, dict[str, np.ndarray]]:
    """Per-(C, F) mean/std of log10(|STFT|+1) from one shot.

    Matches the data_loader STFT exactly: same n_fft, hop, Hann window,
    center=True (default), DC bin dropped — so the resulting stats live
    in the same space the channel-wise stats live in.
    """
    window = torch.hann_window(n_fft)
    out: dict[str, dict[str, np.ndarray]] = {}
    with h5py.File(shot_path, "r") as f:
        for name in SPEC_NAMES:
            y = torch.from_numpy(f[name]["ydata"][:]).float()
            if y.ndim == 1:
                y = y.unsqueeze(0)
            # Match the data_loader's channel subset for this modality.
            sl = SPEC_CHANNEL_SLICES.get(name)
            if sl is not None:
                y = y[sl]
            # Plasma diagnostics typically have NaN samples in
            # pre-shot / post-shot regions. torch.stft propagates NaN
            # across all freq bins of the affected frames; the
            # resulting per-bin mean/std would be NaN everywhere.
            # Replace with 0 so those frames contribute a "silent"
            # ~0 magnitude after log10(|·|+1) — the stats are then
            # well-defined and dominated by the active phase.
            n_nan = int(torch.isnan(y).sum())
            if n_nan:
                y = torch.nan_to_num(y, nan=0.0)
            spec = torch.stft(
                y, n_fft=n_fft, hop_length=hop_length,
                window=window, return_complex=True,
            )
            mag = torch.abs(spec)[:, 1:, :]            # (C, F=n_fft/2, T)
            log_mag = torch.log10(mag + 1.0)
            mean = log_mag.mean(dim=2).numpy()         # (C, F)
            std = log_mag.std(dim=2).clamp(min=1e-3).numpy()
            out[name] = {"mean": mean, "std": std}
            print(f"  local per-bin stats {name}: shape={mean.shape}  "
                  f"mean∈[{mean.min():.3g},{mean.max():.3g}]  "
                  f"std∈[{std.min():.3g},{std.max():.3g}]  "
                  f"(nan_samples={n_nan})", flush=True)
    return out


def renorm_spec_tensor(
    spec_channel_norm: torch.Tensor,
    mean_c: torch.Tensor, std_c: torch.Tensor,
    mean_pb: torch.Tensor, std_pb: torch.Tensor,
) -> torch.Tensor:
    """Undo channel-wise log-standardize, redo per-bin.

    Parameters
    ----------
    spec_channel_norm : (B, C, F, T) in channel-wise log-standardize space.
    mean_c, std_c     : (C,) channel-wise stats (clamped at 1e-3 on std).
    mean_pb, std_pb   : (C, F) per-bin stats (clamped at 1e-3 on std).
    """
    B, C, F, T = spec_channel_norm.shape
    mean_c = mean_c.view(1, C, 1, 1)
    std_c = std_c.clamp(min=1e-3).view(1, C, 1, 1)
    mean_pb = mean_pb.view(1, C, F, 1)
    std_pb = std_pb.clamp(min=1e-3).view(1, C, F, 1)
    log_mag = spec_channel_norm * std_c + mean_c
    return (log_mag - mean_pb) / std_pb


def denorm_pred_per_bin(
    pred_per_bin: torch.Tensor,
    mean_pb: torch.Tensor, std_pb: torch.Tensor,
) -> torch.Tensor:
    """Denormalise (B,C,F,T) per-bin → log10(|STFT|+1) space."""
    _, C, F, _ = pred_per_bin.shape
    mean_pb = mean_pb.view(1, C, F, 1)
    std_pb = std_pb.clamp(min=1e-3).view(1, C, F, 1)
    return pred_per_bin * std_pb + mean_pb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shot", default=DEFAULT_SHOT, type=Path)
    p.add_argument("--checkpoint", default=DEFAULT_CKPT, type=Path)
    p.add_argument("--stats", default=DEFAULT_STATS, type=Path)
    p.add_argument("--output", default=DEFAULT_OUT, type=Path)
    p.add_argument("--chunk_duration_s", default=0.05, type=float)
    p.add_argument("--warmup_s", default=1.0, type=float)
    p.add_argument("--batch_size", default=8, type=int)
    p.add_argument("--num_workers", default=2, type=int)
    p.add_argument("--max_windows", default=0, type=int,
                   help="Cap inference windows for fast iteration (0=all).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  shot={args.shot.name}  ckpt={args.checkpoint.name}",
          flush=True)

    # Pre-load CPU/H5 work BEFORE load_model so the ROCm runtime has
    # a few seconds to fully initialise between the first
    # ``torch.cuda.is_available()`` probe (above) and the heavy
    # ``model.eval().to(device)`` transfer inside ``load_model``. The
    # animation script does this implicitly (PNG reads, traces, stats
    # load) before its own load_model; this script previously called
    # load_model immediately after the device probe and hung on a
    # HIP IPC primitive (wchan=ipclow, job 4798078).

    # 1. Global stats (channel-wise for everything; we'll override spec)
    print("Loading global preprocessing stats...", flush=True)
    stats = torch.load(args.stats, weights_only=False)
    print(f"  stats loaded ({len(stats)} modalities)", flush=True)

    # 2. Local per-bin stats from THIS shot (CPU H5 + STFT work,
    #    keeps GPU subsystem warming up while we read raw signals).
    print("Computing per-bin stats from shot...", flush=True)
    local = compute_local_per_bin_stats(args.shot)

    # Channel-wise stats as tensors (kept on CPU for now; moved to
    # GPU after the model is on GPU). Apply the same NaN→0/1
    # sanitization as the data_loader.
    chan_cpu: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name in SPEC_NAMES:
        entry = stats[name]["log"]
        m = torch.as_tensor(np.array(entry["mean"], dtype=np.float64))
        s = torch.as_tensor(np.array(entry["std"], dtype=np.float64))
        m[torch.isnan(m)] = 0.0
        s[torch.isnan(s)] = 1.0
        sl = SPEC_CHANNEL_SLICES.get(name)
        if sl is not None:
            m = m[sl]
            s = s[sl]
        chan_cpu[name] = (m.float(), s.float())

    # 3. Model — heavy GPU transfer; runs AFTER the warm-up above.
    print(f"Loading model from {args.checkpoint.name}...", flush=True)
    model, ckpt = load_model(args.checkpoint, device)
    diag_names = [c.name for c in model.diagnostics]
    act_names = [c.name for c in model.actuators]
    K = 1                                              # Stage 1
    rollout = make_rollout_if_needed(model, K, args.chunk_duration_s)
    print(f"  model loaded (K={K})", flush=True)

    # 4. Move stats tensors to GPU now that GPU is initialised.
    chan_t: dict[str, tuple[torch.Tensor, torch.Tensor]] = {
        name: (m.to(device), s.to(device)) for name, (m, s) in chan_cpu.items()
    }
    local_t = {
        name: {
            "mean": torch.from_numpy(local[name]["mean"]).float().to(device),
            "std": torch.from_numpy(local[name]["std"]).float().to(device),
        }
        for name in SPEC_NAMES
    }

    # 5. Dataset (single shot)
    ds_full = TokamakMultiFileDataset(
        [args.shot],
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
    n_full = len(ds_full)
    if args.max_windows > 0 and args.max_windows < n_full:
        from torch.utils.data import Subset
        ds = Subset(ds_full, list(range(args.max_windows)))
    else:
        ds = ds_full
    print(f"  windows: {len(ds)}/{n_full}")
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers,
        drop_last=False, pin_memory=False,
    )

    # 6. Inference loop with per-bin spec normalization
    pred_lists: dict[str, list[torch.Tensor]] = {n: [] for n in SPEC_NAMES}
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            # Re-normalize spec inputs + targets in-place: undo
            # channel-wise (which the dataset already applied), redo
            # per-bin (with this shot's local stats).
            for name in SPEC_NAMES:
                if name not in batch["inputs"]:
                    continue
                mc, sc = chan_t[name]
                mp = local_t[name]["mean"]
                sp = local_t[name]["std"]
                batch["inputs"][name] = renorm_spec_tensor(
                    batch["inputs"][name].to(device), mc, sc, mp, sp,
                ).cpu()
                if name in batch["targets"]:
                    batch["targets"][name] = renorm_spec_tensor(
                        batch["targets"][name].to(device), mc, sc, mp, sp,
                    ).cpu()

            predictions_per_k, _, _, _ = rollout_forward_one_batch(
                model, rollout, batch, device, K, args.chunk_duration_s,
            )
            pred = predictions_per_k[0]
            for name in SPEC_NAMES:
                if name in pred:
                    pred_lists[name].append(pred[name].detach().cpu())
            n_batches += 1
            if n_batches % 10 == 0:
                print(f"  batch {n_batches}")
    print(f"  done: {n_batches} batches")

    # 7. Stitch pred chunks and denormalize per-bin back to log space
    pred_log: dict[str, np.ndarray] = {}
    pred_t_ms: dict[str, np.ndarray] = {}
    for name in SPEC_NAMES:
        if not pred_lists[name]:
            print(f"  WARN: no pred collected for {name}; skipping")
            continue
        # (N, C, F, T) → (C, F, N*T) by concatenating along time axis
        stacked = torch.cat(pred_lists[name], dim=0)        # (N, C, F, T)
        N, C, F, T = stacked.shape
        # Denormalize using local per-bin stats
        mp = torch.from_numpy(local[name]["mean"]).float()
        sp = torch.from_numpy(local[name]["std"]).float().clamp(min=1e-3)
        log_per_bin = stacked * sp.view(1, C, F, 1) + mp.view(1, C, F, 1)
        # Reorder to (C, F, N*T)
        log_per_bin = log_per_bin.permute(1, 2, 0, 3).reshape(C, F, N * T)
        pred_log[name] = log_per_bin.numpy()
        # Time axis: each window starts at warmup + i*chunk_duration and
        # the rollout step (K=1) produces T frames covering one chunk.
        t_window_start = args.warmup_s + np.arange(N) * args.chunk_duration_s
        # T frames per chunk → linearly spaced inside the chunk
        per_chunk = np.linspace(0, args.chunk_duration_s, T, endpoint=False)
        pred_t_ms[name] = (t_window_start[:, None] + per_chunk[None, :]).ravel() * 1000.0
        print(f"  {name}: pred log_mag shape {pred_log[name].shape}")

    # 8. Full-shot GT spectrogram for comparison
    gt_log: dict[str, np.ndarray] = {}
    gt_t_ms: dict[str, np.ndarray] = {}
    gt_f_khz: dict[str, np.ndarray] = {}
    n_fft, hop = 1024, 256
    window = torch.hann_window(n_fft)
    with h5py.File(args.shot, "r") as f:
        for name in SPEC_NAMES:
            if name not in pred_log:
                continue
            xdata = f[name]["xdata"][:]
            ydata = torch.from_numpy(f[name]["ydata"][:]).float()
            if ydata.ndim == 1:
                ydata = ydata.unsqueeze(0)
            sl = SPEC_CHANNEL_SLICES.get(name)
            if sl is not None:
                ydata = ydata[sl]
            if torch.isnan(ydata).any():
                ydata = torch.nan_to_num(ydata, nan=0.0)
            spec = torch.stft(
                ydata, n_fft=n_fft, hop_length=hop,
                window=window, return_complex=True,
            )
            mag = torch.abs(spec)[:, 1:, :]
            gt_log[name] = torch.log10(mag.clamp(min=-0.99) + 1.0).numpy()
            n_frames = gt_log[name].shape[2]
            t0_s = float(xdata[0])
            dt_s = float(xdata[1] - xdata[0])
            gt_t_ms[name] = (t0_s + np.arange(n_frames) * hop * dt_s) * 1000.0
            fs = 1.0 / dt_s
            freqs = np.fft.rfftfreq(n_fft, d=1 / fs)[1:]
            gt_f_khz[name] = freqs / 1000.0

    # 9. Plot: 3 rows (one per modality) × 2 cols (GT, pred)
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    n_rows = sum(1 for n in SPEC_NAMES if n in pred_log)
    if n_rows == 0:
        raise SystemExit("No predictions collected; nothing to plot.")
    fig, axes = plt.subplots(
        n_rows, 2, figsize=(16, 3.5 * n_rows),
        sharex="row", sharey="row", constrained_layout=True,
    )
    if n_rows == 1:
        axes = axes[None, :]

    row = 0
    for name in SPEC_NAMES:
        if name not in pred_log:
            continue
        gt = gt_log[name]                                  # (C, F, T_gt)
        pr = pred_log[name]                                # (C, F, T_pr)
        # Pick highest-variance channel (over time, summed over freq)
        per_ch_var = gt.var(axis=2).sum(axis=1)
        c = int(np.argmax(per_ch_var))
        # Shared color scale: percentile of GT
        vmin = float(np.percentile(gt[c], 1))
        vmax = float(np.percentile(gt[c], 99))
        ax_gt, ax_pr = axes[row]
        ax_gt.imshow(
            gt[c], origin="lower", aspect="auto", cmap="viridis",
            vmin=vmin, vmax=vmax,
            extent=[gt_t_ms[name][0], gt_t_ms[name][-1],
                    gt_f_khz[name][0], gt_f_khz[name][-1]],
        )
        ax_gt.set_title(f"{name.upper()} ch{c} — GT")
        ax_gt.set_ylabel("Frequency (kHz)")
        # For pred, the freq axis is the same (n_fft/2 bins, DC dropped)
        ax_pr.imshow(
            pr[c], origin="lower", aspect="auto", cmap="viridis",
            vmin=vmin, vmax=vmax,
            extent=[pred_t_ms[name][0], pred_t_ms[name][-1],
                    gt_f_khz[name][0], gt_f_khz[name][-1]],
        )
        ax_pr.set_title(
            f"{name.upper()} ch{c} — Stage 1 pred (per-bin normalised input)"
        )
        # Clip both panels to the shot's spec-active extent
        # (0 - 6300 ms). The dataset's window count is driven by the
        # longest-spanning modality (slow signals run past spec
        # data), so pred is computed over zero-padded post-shot
        # windows whose output is meaningless — hide that region.
        # 6300 ms matches ECE/BES spec data end (~6.14 - 6.39 s).
        ax_gt.set_xlim(0.0, 6300.0)
        ax_pr.set_xlim(0.0, 6300.0)
        if row == n_rows - 1:
            ax_gt.set_xlabel("Time (ms)")
            ax_pr.set_xlabel("Time (ms)")
        row += 1

    fig.suptitle(
        f"Shot {args.shot.stem.split('_')[0]} — Stage 1 with per-bin spec "
        f"normalisation (local stats from this shot only)",
        fontsize=12,
    )
    fig.savefig(output, dpi=140, bbox_inches="tight")
    print(f"saved {output}")


if __name__ == "__main__":
    main()
