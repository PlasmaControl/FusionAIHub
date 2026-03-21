"""Visualize CO2 FSQ-VAE (SpecAugment run) reconstruction vs. original.

Shows original | reconstructed | error for each channel, plus a codebook
utilisation bar and train/val loss curves from the checkpoint tracker.

Usage:
    pixi run python scripts/training/visualize_co2_fsq_vae_specaugment.py [--best] [--sample N]
"""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from tokamak_foundation_model.data.data_loader import TokamakH5Dataset
from tokamak_foundation_model.models.model_factory import build_model

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = Path("runs/co2_fsq_vae_specaugment/co2_spectrogram_fsq_vae")
DATA_DIR       = Path("/scratch/gpfs/EKOLEMEN/foundation_model")
STATS_PATH     = Path("data/preprocessing_stats.pt")
SIGNAL         = "co2"
N_FFT          = 256
HOP_LENGTH     = 128
FSQ_LEVELS     = [8, 5, 5, 5, 5]
PATCH_H        = 4
PATCH_W        = 16
D_MODEL        = 256
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--best",   action="store_true", help="Load best-val checkpoint instead of latest")
parser.add_argument("--sample", type=int, default=10, help="Dataset sample index to visualise")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load checkpoint ───────────────────────────────────────────────────────────
ckpt_name = "checkpoint_best.pth" if args.best else "checkpoint.pth"
ckpt_path = CHECKPOINT_DIR / ckpt_name
print(f"Loading checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

# ── Dataset ───────────────────────────────────────────────────────────────────
hdf5_files = sorted(DATA_DIR.glob("*_processed.h5"))
hdf5_files = [f for f in hdf5_files
              if 200000 <= int(f.stem.split("_")[0]) <= 200500]
stats = torch.load(STATS_PATH, weights_only=False)
dataset = TokamakH5Dataset(
    hdf5_path=str(hdf5_files[0]),
    preprocessing_stats=stats,
    input_signals=[SIGNAL],
    target_signals=[SIGNAL],
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    prediction_mode=False,
)
sample = dataset[args.sample][SIGNAL]   # (C, F, T)
n_channels = sample.shape[0]

# ── Model ─────────────────────────────────────────────────────────────────────
model = build_model(
    "spectrogram_fsq_vae",
    d_model=D_MODEL,
    n_tokens=0,
    n_channels=n_channels,
    patch_h=PATCH_H,
    patch_w=PATCH_W,
    fsq_levels=FSQ_LEVELS,
    per_channel_patch=True,
)
model.load_state_dict(ckpt["model_state_dict"])
model.to(device).eval()

n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,}")

# ── Inference ─────────────────────────────────────────────────────────────────
with torch.no_grad():
    x = sample.unsqueeze(0).to(device)   # (1, C, F, T)
    out = model(x)
    # eval mode returns (1,C,F,T); training mode returns ((1,C,F,T), indices)
    recon = out[0] if isinstance(out, tuple) else out
    reconstructed = recon[0].cpu()  # strip batch dim → (C, F, T)

original = sample.cpu()

n_nan_orig  = torch.isnan(original).sum().item()
n_nan_recon = torch.isnan(reconstructed).sum().item()
if n_nan_orig > 0:
    print(f"WARNING: {n_nan_orig} NaN values in original data")
if n_nan_recon > 0:
    print(f"WARNING: {n_nan_recon} NaN values in reconstructed output")
print(f"Original  range: [{original.nan_to_num().min():.4f}, {original.nan_to_num().max():.4f}]")
print(f"Reconstructed range: [{reconstructed.nan_to_num().min():.4f}, {reconstructed.nan_to_num().max():.4f}]")

# ── Extract tracker state ──────────────────────────────────────────────────────
tracker = ckpt.get("tracker_state_dict", {})
history = tracker.get("history", {})
train_losses = history.get("train", {}).get("loss", [])
val_losses   = history.get("val",   {}).get("loss", [])
codebook_util = history.get("train", {}).get("codebook_utilization", [])
epoch = ckpt.get("epoch", len(train_losses) - 1)

# ── Figure layout: loss curves + per-channel reconstruction ───────────────────
has_curves = len(train_losses) > 0
n_curve_rows = 1 if has_curves else 0
n_rows = n_channels + n_curve_rows
fig = plt.figure(figsize=(15, 3 * n_rows))
gs = fig.add_gridspec(n_rows, 3, hspace=0.35, wspace=0.08)

# ── Loss curves (top row) ──────────────────────────────────────────────────────
if has_curves:
    ax_loss = fig.add_subplot(gs[0, :2])
    epochs_x = np.arange(1, len(train_losses) + 1)
    ax_loss.plot(epochs_x, train_losses, label="Train L1", linewidth=1.5)
    if val_losses:
        val_x = np.linspace(1, len(train_losses), len(val_losses))
        ax_loss.plot(val_x, val_losses, label="Val L1", linewidth=1.5)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("L1 Loss")
    ax_loss.set_title("Training curves")
    ax_loss.legend(fontsize=9)
    ax_loss.grid(alpha=0.3)

    if codebook_util:
        ax_util = fig.add_subplot(gs[0, 2])
        util_x = np.arange(1, len(codebook_util) + 1)
        ax_util.plot(util_x, codebook_util, color="tab:green", linewidth=1.5)
        ax_util.set_ylim(0, 1)
        ax_util.set_xlabel("Epoch")
        ax_util.set_title(f"Codebook utilisation (last: {codebook_util[-1]:.2%})")
        ax_util.grid(alpha=0.3)

# ── Per-channel reconstruction rows ───────────────────────────────────────────
for ch in range(n_channels):
    row = ch + n_curve_rows
    orig_ch  = np.nan_to_num(original[ch].numpy())
    recon_ch = np.nan_to_num(reconstructed[ch].numpy())
    err_ch   = recon_ch - orig_ch
    err_abs  = np.abs(err_ch).max()

    # Per-channel bounds so each channel uses its own dynamic range
    ch_vmin = min(orig_ch.min(), recon_ch.min())
    ch_vmax = max(orig_ch.max(), recon_ch.max())
    if ch_vmin == ch_vmax:
        ch_vmax = ch_vmin + 1.0  # avoid degenerate colormap

    ax_o = fig.add_subplot(gs[row, 0])
    ax_r = fig.add_subplot(gs[row, 1])
    ax_e = fig.add_subplot(gs[row, 2])

    kw = dict(origin="lower", aspect="auto", interpolation="nearest")
    ax_o.imshow(orig_ch,  cmap="viridis", vmin=ch_vmin, vmax=ch_vmax, **kw)
    ax_r.imshow(recon_ch, cmap="viridis", vmin=ch_vmin, vmax=ch_vmax, **kw)
    ax_e.imshow(err_ch,   cmap="bwr",     vmin=-err_abs, vmax=err_abs, **kw)

    for ax in (ax_o, ax_r, ax_e):
        ax.set_xticks([])
        ax.set_yticks([])

    ch_mae = np.abs(err_ch).mean()
    ax_o.set_ylabel(f"Ch {ch}\nMAE={ch_mae:.4f}", fontsize=8)

    if row == n_curve_rows:   # label columns on first reconstruction row
        ax_o.set_title("Original",      fontsize=10)
        ax_r.set_title("Reconstructed", fontsize=10)
        ax_e.set_title("Error (R − O)", fontsize=10)

# ── Suptitle ──────────────────────────────────────────────────────────────────
ckpt_tag = "best-val" if args.best else "latest"
title_parts = [f"CO2 FSQ-VAE (SpecAugment) — epoch {epoch + 1} [{ckpt_tag}]"]
title_parts.append(f"patch {PATCH_H}×{PATCH_W}  per_channel_patch=True")
if train_losses:
    title_parts.append(f"train L1={train_losses[-1]:.4f}")
if val_losses:
    title_parts.append(f"val L1={val_losses[-1]:.4f}")
fig.suptitle("  |  ".join(title_parts), fontsize=11, y=1.01)

# ── Save ──────────────────────────────────────────────────────────────────────
out_dir = CHECKPOINT_DIR / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"reconstruction_{ckpt_tag}_sample{args.sample}.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved → {out_path}")
