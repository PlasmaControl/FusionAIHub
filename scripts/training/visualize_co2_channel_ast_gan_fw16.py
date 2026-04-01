"""Visualize CO2 Channel-AST-GAN (fw=16, d_model=512) reconstruction.

Produces two figures:
  1. Original / Reconstructed / Error per channel
  2. Training loss curves (G total, recon, D)

Usage:
    pixi run python scripts/training/visualize_co2_channel_ast_gan_fw16.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import math

from tokamak_foundation_model.data.data_loader import TokamakH5Dataset
from tokamak_foundation_model.models.model_factory import build_model

# -- Config -------------------------------------------------------------------
CHECKPOINT = Path("runs/co2_channel_ast_gan_fw16/co2_spectrogram_channel_ast_gan/checkpoint.pth")
DATA_DIR   = Path("/scratch/gpfs/EKOLEMEN/foundation_model")
STATS_PATH = Path("data/preprocessing_stats.pt")
SIGNAL     = "co2"
N_FFT      = 256
HOP_LENGTH = 128
FRAME_WIDTH = 16
SAMPLE_IDX = 10
OUT_DIR    = CHECKPOINT.parent / "plots"
# -----------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- Load checkpoint ----------------------------------------------------------
ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)

# -- Dataset ------------------------------------------------------------------
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

# -- Build model --------------------------------------------------------------
model = build_model(
    "spectrogram_channel_ast_gan",
    d_model=512, n_tokens=0, n_channels=n_channels,
    freq_bins=freq_bins, frame_width=FRAME_WIDTH,
    time_conv_kernel=7,
)
model.load_state_dict(ckpt["model_state_dict"])
model.to(device).eval()

# -- Inference ----------------------------------------------------------------
with torch.no_grad():
    x = sample.unsqueeze(0).to(device)
    reconstructed = model(x).cpu().squeeze(0)

original = sample.cpu()

epoch = ckpt.get("epoch", "?")
history = ckpt["tracker_state_dict"]["history"]
train_losses = history["train"]["loss"]
final_loss = train_losses[-1]

n_frames = math.ceil(sample.shape[2] / FRAME_WIDTH)
n_tokens_total = n_channels * n_frames

OUT_DIR.mkdir(parents=True, exist_ok=True)

# -- Figure 1: Reconstruction ------------------------------------------------
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

fig.suptitle(
    f"CO2 Channel-AST-GAN (fw={FRAME_WIDTH}, d=512, {n_tokens_total} tokens) "
    f"-- epoch {epoch + 1}, G loss={final_loss:.4f}",
    fontsize=11,
)
fig.tight_layout()
out = OUT_DIR / "reconstruction.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved -> {out}")
plt.close(fig)

# -- Figure 2: Loss curves ---------------------------------------------------
fig2, (ax_g, ax_d) = plt.subplots(1, 2, figsize=(14, 4))

# G losses
ax_g.plot(range(1, len(train_losses) + 1), train_losses,
          color="tab:blue", label="G total")
recon_losses = history["train"].get("recon_loss", [])
if recon_losses:
    ax_g.plot(range(1, len(recon_losses) + 1), recon_losses,
              color="tab:green", label="Recon (L1)")
g_adv_losses = history["train"].get("g_adv", [])
if g_adv_losses:
    ax_g.plot(range(1, len(g_adv_losses) + 1), g_adv_losses,
              color="tab:red", label="G adversarial", alpha=0.7)
val_losses = history.get("validate", {}).get("loss", [])
if val_losses:
    ax_g.plot(range(1, len(val_losses) + 1), val_losses,
              color="tab:orange", label="Val L1")
ax_g.set_xlabel("Epoch")
ax_g.set_ylabel("Loss")
ax_g.set_title("Generator Losses")
ax_g.grid(True, alpha=0.3)
ax_g.legend()

# D losses
d_losses = history["train"].get("d_loss", [])
if d_losses:
    ax_d.plot(range(1, len(d_losses) + 1), d_losses,
              color="tab:purple", label="D total")
r1_losses = history["train"].get("r1", [])
if r1_losses:
    ax_d.plot(range(1, len(r1_losses) + 1), r1_losses,
              color="tab:brown", label="R1", alpha=0.7)
r2_losses = history["train"].get("r2", [])
if r2_losses:
    ax_d.plot(range(1, len(r2_losses) + 1), r2_losses,
              color="tab:pink", label="R2", alpha=0.7)
ax_d.set_xlabel("Epoch")
ax_d.set_ylabel("Loss")
ax_d.set_title("Discriminator Losses")
ax_d.grid(True, alpha=0.3)
ax_d.legend()

fig2.suptitle(
    f"CO2 Channel-AST-GAN (fw={FRAME_WIDTH}, d=512) -- Loss Curves",
    fontsize=11,
)
fig2.tight_layout()
out = OUT_DIR / "loss_curve.png"
fig2.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved -> {out}")
plt.close(fig2)
