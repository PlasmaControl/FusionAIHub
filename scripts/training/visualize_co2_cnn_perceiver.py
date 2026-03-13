"""Visualize CO2 CNN-Perceiver reconstruction (continuous bottleneck).

Produces three figures:
  1. Original / Reconstructed / Error per channel
  2. Training + validation loss curves
  3. Latent token heatmap — (N, d_model) from the encoder, one sample

Usage:
    pixi run python scripts/training/visualize_co2_cnn_perceiver.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from tokamak_foundation_model.data.data_loader import TokamakH5Dataset
from tokamak_foundation_model.models.model_factory import build_model

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT = Path("runs/co2_cnn_perceiver/co2_spectrogram_cnn_perceiver/checkpoint.pth")
DATA_DIR   = Path("/scratch/gpfs/EKOLEMEN/foundation_model")
STATS_PATH = Path("data/preprocessing_stats.pt")
SIGNAL     = "co2"
N_FFT      = 256
HOP_LENGTH = 128
CNN_DIMS   = [64, 128]
D_MODEL    = 256
N_TOKENS   = 16
N_HEADS    = 4
N_SELF_LAYERS     = 2
N_DEC_SELF_LAYERS = 2
SAMPLE_IDX = 10
OUT_DIR    = CHECKPOINT.parent / "plots"
# ─────────────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load checkpoint ───────────────────────────────────────────────────────────
ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)

# ── Dataset ───────────────────────────────────────────────────────────────────
hdf5_files = sorted(DATA_DIR.glob("*_processed.h5"))
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

# ── Build model ───────────────────────────────────────────────────────────────
model = build_model(
    "spectrogram_cnn_perceiver", d_model=D_MODEL, n_tokens=N_TOKENS,
    n_channels=n_channels,
    dims=CNN_DIMS, n_heads=N_HEADS,
    n_self_layers=N_SELF_LAYERS, n_dec_self_layers=N_DEC_SELF_LAYERS,
)
model.load_state_dict(ckpt["model_state_dict"])
model.to(device).eval()

# ── Inference ─────────────────────────────────────────────────────────────────
with torch.no_grad():
    x = sample.unsqueeze(0).to(device)           # (1, C, F, T)
    reconstructed = model(x).cpu().squeeze(0)    # (C, F, T)
    latent = model.encoder(x).cpu().squeeze(0)   # (N, d_model)

original = sample.cpu()

epoch = ckpt.get("epoch", "?")
train_losses = ckpt["tracker_state_dict"]["history"]["train"]["loss"]
val_losses   = ckpt["tracker_state_dict"]["history"].get("validate", {}).get("loss", [])
final_loss   = train_losses[-1]

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Figure 1: Reconstruction ──────────────────────────────────────────────────
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
axes[0, 2].set_title("Error (R − O)", fontsize=11)

C_in, F_in, T_in = sample.shape
compression = C_in * F_in * T_in / (N_TOKENS * D_MODEL)
fig.suptitle(
    f"CO2 CNN-Perceiver (continuous, N={N_TOKENS} tokens, {compression:.0f}× compression) "
    f"— epoch {epoch + 1}, train L1={final_loss:.4f}",
    fontsize=11,
)
fig.tight_layout()
out = OUT_DIR / "reconstruction.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
plt.close(fig)

# ── Figure 2: Loss curves ─────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.plot(range(1, len(train_losses) + 1), train_losses, label="Train")
if val_losses:
    ax2.plot(range(1, len(val_losses) + 1), val_losses, label="Val")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("L1 Loss")
ax2.set_title(f"CO2 CNN-Perceiver — Loss Curves (N={N_TOKENS} tokens)")
ax2.legend()
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
out = OUT_DIR / "loss_curve.png"
fig2.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
plt.close(fig2)

# ── Figure 3: Latent token heatmap ───────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(12, 3))
im = ax3.imshow(latent.numpy(), cmap="RdBu_r", origin="upper", aspect="auto")
ax3.set_xlabel("d_model dimension")
ax3.set_ylabel("Token index")
ax3.set_title(f"CO2 CNN-Perceiver — Latent tokens (N={N_TOKENS} × d={D_MODEL})")
plt.colorbar(im, ax=ax3, shrink=0.8)
fig3.tight_layout()
out = OUT_DIR / "latent_tokens.png"
fig3.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
plt.close(fig3)
