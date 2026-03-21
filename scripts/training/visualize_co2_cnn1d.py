"""Visualize CO2 1D ConvNeXt autoencoder reconstruction.

Produces three figures:
  1. Original / Reconstructed / Error per channel
  2. Training + validation loss curves
  3. Bottleneck token heatmap (first 16 of bottleneck_dim channels)

Usage:
    pixi run python scripts/training/visualize_co2_cnn1d.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from tokamak_foundation_model.data.data_loader import TokamakH5Dataset
from tokamak_foundation_model.models.model_factory import build_model

# -- Config ------------------------------------------------------------------
CHECKPOINT = Path("runs/co2_cnn1d_v2/co2_spectrogram_cnn1d/checkpoint.pth")
DATA_DIR   = Path("/scratch/gpfs/EKOLEMEN/foundation_model")
STATS_PATH = Path("data/preprocessing_stats.pt")
SIGNAL     = "co2"
N_FFT      = 256
HOP_LENGTH = 128
DIM             = 256
DEPTH           = 6
STEM_DIMS       = [64, 128]
FRAME_WIDTH     = 2
BOTTLENECK_DIM  = 32
D_MODEL         = 256
SAMPLE_IDX = 10
OUT_DIR    = CHECKPOINT.parent / "plots"
# ----------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- Load checkpoint ---------------------------------------------------------
ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)

# -- Dataset -----------------------------------------------------------------
hdf5_files = [
    f for f in sorted(DATA_DIR.glob("*_processed.h5"))
    if 200000 <= int(f.stem.split("_")[0]) <= 200500
]
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
freq_bins = sample.shape[1]

# -- Build model -------------------------------------------------------------
model = build_model(
    "spectrogram_cnn1d", d_model=D_MODEL, n_tokens=0,
    n_channels=n_channels,
    freq_bins=freq_bins,
    frame_width=FRAME_WIDTH,
    dim=DIM, depth=DEPTH,
    stem_dims=STEM_DIMS,
    bottleneck_dim=BOTTLENECK_DIM,
)
model.load_state_dict(ckpt["model_state_dict"])
model.to(device).eval()

# -- Inference ---------------------------------------------------------------
with torch.no_grad():
    x = sample.unsqueeze(0).to(device)           # (1, C, F, T)
    reconstructed = model(x).cpu().squeeze(0)     # (C, F, T)

    # Get bottleneck tokens for visualization
    z_tokens = model.encoder(x).cpu().squeeze(0)  # (T', bottleneck_dim)

original = sample.cpu()

epoch = ckpt.get("epoch", "?")
train_losses = ckpt["tracker_state_dict"]["history"]["train"]["loss"]
val_losses   = ckpt["tracker_state_dict"]["history"].get("validate", {}).get("loss", [])
final_loss   = train_losses[-1]

OUT_DIR.mkdir(parents=True, exist_ok=True)

# -- Figure 1: Reconstruction -----------------------------------------------
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

axes[0, 0].set_title("Original",      fontsize=11)
axes[0, 1].set_title("Reconstructed", fontsize=11)
axes[0, 2].set_title("Error (R - O)", fontsize=11)

n_tokens = z_tokens.shape[0]
C_in, F_in, T_in = sample.shape
compression = C_in * F_in * T_in / (n_tokens * BOTTLENECK_DIM)
fig.suptitle(
    f"CO2 CNN1d dim={DIM} bn{BOTTLENECK_DIM} fw={FRAME_WIDTH} "
    f"({n_tokens} tokens x d={BOTTLENECK_DIM}, {compression:.0f}x compression) "
    f"-- epoch {epoch + 1}, train L1={final_loss:.4f}",
    fontsize=11,
)
fig.tight_layout()
out = OUT_DIR / "reconstruction.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved -> {out}")
plt.close(fig)

# -- Figure 2: Loss curves --------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.plot(range(1, len(train_losses) + 1), train_losses, label="Train")
if val_losses:
    ax2.plot(range(1, len(val_losses) + 1), val_losses, label="Val")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("L1 Loss")
ax2.set_title(f"CO2 CNN1d dim={DIM} bn{BOTTLENECK_DIM} -- Loss Curves")
ax2.legend()
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
out = OUT_DIR / "loss_curve.png"
fig2.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved -> {out}")
plt.close(fig2)

# -- Figure 3: Bottleneck tokens (first 16 channels) ------------------------
n_show = min(16, BOTTLENECK_DIM)
fig3, axes3 = plt.subplots(n_show, 1, figsize=(12, n_show * 0.6), sharex=True)
for i in range(n_show):
    axes3[i].plot(z_tokens[:, i].numpy(), linewidth=0.5)
    axes3[i].set_ylabel(f"d{i}", fontsize=7, rotation=0, labelpad=20)
    axes3[i].set_yticks([])
    if i < n_show - 1:
        axes3[i].set_xticks([])
axes3[-1].set_xlabel("Time position")
fig3.suptitle(
    f"CNN1d bn{BOTTLENECK_DIM} -- Bottleneck tokens "
    f"({n_tokens} positions, showing 16/{BOTTLENECK_DIM} dims)",
    fontsize=11,
)
fig3.tight_layout()
out = OUT_DIR / "bottleneck_tokens.png"
fig3.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved -> {out}")
plt.close(fig3)
