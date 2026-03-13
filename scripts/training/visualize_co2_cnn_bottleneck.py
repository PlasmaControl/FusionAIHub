"""Visualize CO2 CNN with bottleneck projection (bottleneck_dim=8).

Usage:
    pixi run python scripts/training/visualize_co2_cnn_bottleneck.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from tokamak_foundation_model.data.data_loader import TokamakH5Dataset
from tokamak_foundation_model.models.model_factory import build_model

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT  = Path("runs/co2_cnn_bn8/co2_spectrogram_cnn/checkpoint.pth")
DATA_DIR    = Path("/scratch/gpfs/EKOLEMEN/foundation_model")
STATS_PATH  = Path("data/preprocessing_stats.pt")
SIGNAL      = "co2"
N_FFT       = 256
HOP_LENGTH  = 128
CNN_DIMS    = [64, 128]
BOTTLENECK  = 8
SAMPLE_IDX  = 10          # matches other viz scripts
SHOT_MIN    = 200000
SHOT_MAX    = 200500
OUT_DIR     = CHECKPOINT.parent / "plots"
# ─────────────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load checkpoint ──────────────────────────────────────────────────────────
ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)

# ── Dataset (default log_standardize preprocessing) ──────────────────────────
hdf5_files = sorted(DATA_DIR.glob("*_processed.h5"))
hdf5_files = [f for f in hdf5_files
              if SHOT_MIN <= int(f.stem.split("_")[0]) <= SHOT_MAX]

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
sample = dataset[SAMPLE_IDX][SIGNAL]   # (C, F, T)
n_channels = sample.shape[0]

# ── Build model (must match training config) ─────────────────────────────────
model = build_model("spectrogram_cnn", d_model=256, n_tokens=0,
                    n_channels=n_channels, dims=CNN_DIMS,
                    bottleneck_dim=BOTTLENECK)
model.load_state_dict(ckpt["model_state_dict"])
model.to(device).eval()

# ── Inference ────────────────────────────────────────────────────────────────
with torch.no_grad():
    x = sample.unsqueeze(0).to(device)        # (1, C, F, T)
    reconstructed = model(x).cpu().squeeze(0)  # (C, F, T)

original = sample.cpu()  # (C, F, T)

# ── Figure 1: Original / Reconstructed / Error ──────────────────────────────
n_cols = 3
fig, axes = plt.subplots(n_channels, n_cols,
                          figsize=(n_cols * 5, n_channels * 3))

vmin = original.min().item()
vmax = original.max().item()

for ch in range(n_channels):
    orig_ch  = original[ch].numpy()
    recon_ch = reconstructed[ch].numpy()
    err_ch   = recon_ch - orig_ch

    axes[ch, 0].imshow(orig_ch,  cmap="viridis", origin="lower",
                       aspect="auto", vmin=vmin, vmax=vmax)
    axes[ch, 1].imshow(recon_ch, cmap="viridis", origin="lower",
                       aspect="auto", vmin=vmin, vmax=vmax)
    emax = max(abs(err_ch).max(), 1e-8)
    axes[ch, 2].imshow(err_ch,   cmap="bwr",     origin="lower",
                       aspect="auto", vmin=-emax, vmax=emax)

    for ax in axes[ch]:
        ax.set_xticks([])
        ax.set_yticks([])

    axes[ch, 0].set_ylabel(f"Ch {ch}", fontsize=9)

axes[0, 0].set_title("Original (log_standardize)", fontsize=11)
axes[0, 1].set_title("Reconstructed",              fontsize=11)
axes[0, 2].set_title("Error (R − O)",              fontsize=11)

epoch = ckpt.get("epoch", "?")
losses = ckpt["tracker_state_dict"]["history"]["train"]["loss"]
final_loss = losses[-1]
C_in, F_in, T_in = sample.shape
spatial = (F_in // 4) * (((T_in + 3) // 4))
compression = C_in * F_in * T_in / (BOTTLENECK * spatial)
fig.suptitle(
    f"CO2 CNN bottleneck_dim={BOTTLENECK} ({compression:.0f}× compression) "
    f"— epoch {epoch + 1}, train L1={final_loss:.4f}",
    fontsize=12,
)
fig.tight_layout()

OUT_DIR.mkdir(parents=True, exist_ok=True)
out_recon = OUT_DIR / "reconstruction.png"
fig.savefig(out_recon, dpi=150, bbox_inches="tight")
print(f"Saved → {out_recon}")
plt.close(fig)

# ── Figure 2: Training loss curve ───────────────────────────────────────────
val_losses = ckpt["tracker_state_dict"]["history"].get("validate", {}).get("loss", [])

fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.plot(range(1, len(losses) + 1), losses, label="Train")
if val_losses:
    ax2.plot(range(1, len(val_losses) + 1), val_losses, label="Val")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("L1 Loss")
ax2.set_title(f"Training Loss Curve (bottleneck_dim={BOTTLENECK})")
ax2.legend()
ax2.grid(True, alpha=0.3)
fig2.tight_layout()

out_loss = OUT_DIR / "loss_curve.png"
fig2.savefig(out_loss, dpi=150, bbox_inches="tight")
print(f"Saved → {out_loss}")
plt.close(fig2)
