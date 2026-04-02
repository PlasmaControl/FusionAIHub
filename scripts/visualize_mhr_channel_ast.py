"""Visualize MHR Channel-AST (no FSQ) reconstruction for a single shot.

Produces two figures:
  1. Original / Reconstructed / Error per channel
  2. Training + validation loss curves

Usage:
    module load pixi && pixi run python scripts/visualize_mhr_channel_ast.py
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

# ---------------------------------------------------------------------------
# Mock optional dependencies needed to unpickle the checkpoint's tracker state.
# ---------------------------------------------------------------------------
_tb = types.ModuleType("tensorboard")
sys.modules["tensorboard"] = _tb
_tw = types.ModuleType("torch.utils.tensorboard")
_tw.SummaryWriter = type("SummaryWriter", (), {})  # type: ignore[attr-defined]
sys.modules["torch.utils.tensorboard"] = _tw
_wb = types.ModuleType("wandb")
_wb.init = lambda **kw: None  # type: ignore[attr-defined]
_wb.log = lambda *a, **kw: None  # type: ignore[attr-defined]
sys.modules["wandb"] = _wb

import matplotlib.pyplot as plt
import torch

from tokamak_foundation_model.data.data_loader import TokamakH5Dataset
from tokamak_foundation_model.models.modality.spectrogram_channel_ast import (
    SpectrogramChannelASTAutoEncoder,
)

# == Config ================================================================
CHECKPOINT = Path(
    "runs/mhr_channel_ast_nofsq_fw16"
    "/mhr_spectrogram_channel_ast_fsq/checkpoint.pth"
)
DATA_DIR = Path("/scratch/gpfs/EKOLEMEN/foundation_model")
STATS_PATH = Path("/scratch/gpfs/kb0246/faith/data/preprocessing_stats.pt")
SIGNAL = "mhr"
N_FFT = 256
HOP_LENGTH = 128
SHOT = 201423
SAMPLE_IDX = 4
OUT_DIR = CHECKPOINT.parent / "plots"
# ==========================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- Load checkpoint -------------------------------------------------------
ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)

# -- Locate shot file ------------------------------------------------------
shot_file = DATA_DIR / f"{SHOT}_processed.h5"
if not shot_file.exists():
    print(f"WARNING: Shot {SHOT} not found at {shot_file}")
    candidates = sorted(
        DATA_DIR.glob("*_processed.h5"),
        key=lambda p: int(p.stem.split("_")[0]),
    )
    if not candidates:
        raise FileNotFoundError(f"No shot files found in {DATA_DIR}")
    shot_file = candidates[-1]
    SHOT = int(shot_file.stem.split("_")[0])
    print(f"         Using shot {SHOT} ({shot_file.name})")

# -- Dataset ---------------------------------------------------------------
stats = torch.load(STATS_PATH, weights_only=False)
dataset = TokamakH5Dataset(
    hdf5_path=str(shot_file),
    preprocessing_stats=stats,
    input_signals=[SIGNAL],
    target_signals=[SIGNAL],
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    prediction_mode=False,
)
sample = dataset[SAMPLE_IDX][SIGNAL]  # (C, F, T)
n_channels = sample.shape[0]
freq_bins = sample.shape[1]

print(f"Shot {SHOT}, chunk {SAMPLE_IDX}: shape {tuple(sample.shape)} "
      f"(C={n_channels}, F={freq_bins}, T={sample.shape[2]})")

# -- Build model -----------------------------------------------------------
model = SpectrogramChannelASTAutoEncoder(
    n_channels=n_channels,
    d_model=256,
    freq_bins=freq_bins,
    frame_width=16,
    n_enc_layers=4,
    n_dec_layers=4,
    n_heads=4,
    time_conv_kernel=7,
)
model.load_state_dict(ckpt["model_state_dict"])
model.to(device).eval()

# -- Inference -------------------------------------------------------------
with torch.no_grad():
    x = sample.unsqueeze(0).to(device)
    reconstructed = model(x)[0].cpu().squeeze(0)

original = sample.cpu()

epoch = ckpt.get("epoch", "?")
history = ckpt["tracker_state_dict"]["history"]
train_losses = history["train"]["loss"]
val_losses = history.get("validate", {}).get("loss", [])
final_train = train_losses[-1]
final_val = val_losses[-1] if val_losses else None

OUT_DIR.mkdir(parents=True, exist_ok=True)

# -- Figure 1: Reconstruction ---------------------------------------------
n_cols = 3
fig, axes = plt.subplots(n_channels, n_cols,
                          figsize=(n_cols * 5, n_channels * 3))
if n_channels == 1:
    axes = axes[None, :]

vmin = original.min().item()
vmax = original.max().item()

for ch in range(n_channels):
    orig_ch = original[ch].numpy()
    recon_ch = reconstructed[ch].numpy()
    err_ch = recon_ch - orig_ch

    axes[ch, 0].imshow(orig_ch, cmap="viridis", origin="lower",
                       aspect="auto", vmin=vmin, vmax=vmax)
    axes[ch, 1].imshow(recon_ch, cmap="viridis", origin="lower",
                       aspect="auto", vmin=vmin, vmax=vmax)
    emax = max(abs(err_ch).max(), 1e-8)
    axes[ch, 2].imshow(err_ch, cmap="bwr", origin="lower",
                       aspect="auto", vmin=-emax, vmax=emax)

    for ax in axes[ch]:
        ax.set_xticks([])
        ax.set_yticks([])
    axes[ch, 0].set_ylabel(f"Ch {ch}", fontsize=9)

axes[0, 0].set_title("Original", fontsize=11)
axes[0, 1].set_title("Reconstructed", fontsize=11)
axes[0, 2].set_title("Error (R - O)", fontsize=11)

subtitle = f"epoch {epoch + 1}, train L1={final_train:.4f}"
if final_val is not None:
    subtitle += f", val L1={final_val:.4f}"
fig.suptitle(
    f"MHR Channel-AST (shot {SHOT}, chunk {SAMPLE_IDX}) — {subtitle}",
    fontsize=11,
)
fig.tight_layout()
out = OUT_DIR / "reconstruction.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved -> {out}")
plt.close(fig)

# -- Figure 2: Loss curves ------------------------------------------------
fig2, ax_loss = plt.subplots(figsize=(9, 4))
ax_loss.plot(range(1, len(train_losses) + 1), train_losses,
             color="tab:blue", label="Train L1")
if val_losses:
    ax_loss.plot(range(1, len(val_losses) + 1), val_losses,
                 color="tab:orange", label="Val L1")
ax_loss.set_xlabel("Epoch")
ax_loss.set_ylabel("L1 Loss")
ax_loss.grid(True, alpha=0.3)
ax_loss.legend()
ax_loss.set_title("MHR Channel-AST (no FSQ, fw=16) — Loss Curves")
fig2.tight_layout()
out = OUT_DIR / "loss_curve.png"
fig2.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved -> {out}")
plt.close(fig2)
