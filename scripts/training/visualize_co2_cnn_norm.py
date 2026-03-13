"""Visualize CO2 CNN+Normalizer reconstruction vs. original for all channels.

Also plots the learned normalizer parameters (gamma, per-freq mean/std) and
the training loss curve.

Usage:
    pixi run python scripts/training/visualize_co2_cnn_norm.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from tokamak_foundation_model.data.data_loader import TokamakH5Dataset
from tokamak_foundation_model.models.model_factory import build_model
from tokamak_foundation_model.models.modality.spectrogram_normalizer import (
    NormalizedSpectrogramAutoEncoder,
)

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT  = Path("runs/co2_cnn_norm/co2_spectrogram_cnn/checkpoint.pth")
DATA_DIR    = Path("/scratch/gpfs/EKOLEMEN/foundation_model")
STATS_PATH  = Path("data/preprocessing_stats.pt")
SIGNAL      = "co2"
N_FFT       = 256
HOP_LENGTH  = 128
CNN_DIMS    = [64, 128]
SMOOTH_K    = 7
SAMPLE_IDX  = 10          # which dataset sample to visualise (matches FSQ viz)
SHOT_MIN    = 200000
SHOT_MAX    = 200500
OUT_DIR     = CHECKPOINT.parent / "plots"
# ─────────────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load checkpoint ──────────────────────────────────────────────────────────
ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)

# ── Dataset (log preprocessing, matching training) ───────────────────────────
for cfg in TokamakH5Dataset.SIGNAL_CONFIGS:
    if cfg.name == SIGNAL:
        cfg.preprocess.method = "log"
        break

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
inner = build_model("spectrogram_cnn", d_model=256, n_tokens=0,
                    n_channels=n_channels, dims=CNN_DIMS)
n_freq = N_FFT // 2
model = NormalizedSpectrogramAutoEncoder(inner, n_channels, n_freq,
                                         smooth_kernel_size=SMOOTH_K)
model.load_state_dict(ckpt["model_state_dict"])
model.to(device).eval()

# ── Inference ────────────────────────────────────────────────────────────────
with torch.no_grad():
    x = sample.unsqueeze(0).to(device)           # (1, C, F, T)
    x_norm = model.normalizer.normalize(x)        # (1, C, F, T)
    reconstructed = model(x).cpu().squeeze(0)     # (C, F, T)

original = sample.cpu()              # (C, F, T)
normalized = x_norm.cpu().squeeze(0) # (C, F, T)

# ── Figure 1: Original / Normalized / Reconstructed / Error ──────────────────
n_cols = 4
fig, axes = plt.subplots(n_channels, n_cols,
                          figsize=(n_cols * 5, n_channels * 3))

vmin = original.min().item()
vmax = original.max().item()

for ch in range(n_channels):
    orig_ch  = original[ch].numpy()
    norm_ch  = normalized[ch].numpy()
    recon_ch = reconstructed[ch].numpy()
    err_ch   = recon_ch - orig_ch

    axes[ch, 0].imshow(orig_ch,  cmap="viridis", origin="lower",
                       aspect="auto", vmin=vmin, vmax=vmax)
    axes[ch, 1].imshow(norm_ch,  cmap="viridis", origin="lower",
                       aspect="auto")
    axes[ch, 2].imshow(recon_ch, cmap="viridis", origin="lower",
                       aspect="auto", vmin=vmin, vmax=vmax)
    emax = max(abs(err_ch).max(), 1e-8)
    axes[ch, 3].imshow(err_ch,   cmap="bwr",     origin="lower",
                       aspect="auto", vmin=-emax, vmax=emax)

    for ax in axes[ch]:
        ax.set_xticks([])
        ax.set_yticks([])

    axes[ch, 0].set_ylabel(f"Ch {ch}", fontsize=9)

axes[0, 0].set_title("Original (log)",   fontsize=11)
axes[0, 1].set_title("After Normalizer", fontsize=11)
axes[0, 2].set_title("Reconstructed",    fontsize=11)
axes[0, 3].set_title("Error (R − O)",    fontsize=11)

epoch = ckpt.get("epoch", "?")
losses = ckpt["tracker_state_dict"]["history"]["train"]["loss"]
final_loss = losses[-1]
fig.suptitle(f"CO2 CNN+Norm — epoch {epoch + 1}, train L1={final_loss:.4f}",
             fontsize=12)
fig.tight_layout()

OUT_DIR.mkdir(parents=True, exist_ok=True)
out_recon = OUT_DIR / "reconstruction.png"
fig.savefig(out_recon, dpi=150, bbox_inches="tight")
print(f"Saved → {out_recon}")
plt.close(fig)

# ── Figure 2: Learned normalizer parameters ─────────────────────────────────
norm = model.normalizer
gamma = norm.gamma.item()
smoothed_mean, smoothed_std = norm._smooth_params()
smoothed_mean = smoothed_mean.detach().cpu().squeeze()   # (C, F)
smoothed_std  = smoothed_std.detach().cpu().squeeze()    # (C, F)

fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))

# Per-channel smoothed mean
for ch in range(n_channels):
    axes2[0].plot(smoothed_mean[ch].numpy(), label=f"Ch {ch}")
axes2[0].set_title("Learned Smoothed Mean (per freq bin)")
axes2[0].set_xlabel("Frequency bin")
axes2[0].legend(fontsize=8)

# Per-channel smoothed std
for ch in range(n_channels):
    axes2[1].plot(smoothed_std[ch].numpy(), label=f"Ch {ch}")
axes2[1].set_title("Learned Smoothed Std (per freq bin)")
axes2[1].set_xlabel("Frequency bin")
axes2[1].legend(fontsize=8)

# Smoothing kernel
kernel = norm.smooth_conv.weight.detach().cpu().squeeze()  # (C, K) or (K,)
if kernel.ndim == 2:
    kernel = kernel[0]  # all channels same init, show one
axes2[2].bar(range(len(kernel)), kernel.numpy())
axes2[2].set_title(f"Smoothing Kernel (gamma={gamma:.3f})")
axes2[2].set_xlabel("Kernel tap")

fig2.suptitle("Learned Normalizer Parameters", fontsize=12)
fig2.tight_layout()

out_norm = OUT_DIR / "normalizer_params.png"
fig2.savefig(out_norm, dpi=150, bbox_inches="tight")
print(f"Saved → {out_norm}")
plt.close(fig2)

# ── Figure 3: Training loss curve ───────────────────────────────────────────
val_losses = ckpt["tracker_state_dict"]["history"].get("validate", {}).get("loss", [])

fig3, ax3 = plt.subplots(figsize=(8, 4))
ax3.plot(range(1, len(losses) + 1), losses, label="Train")
if val_losses:
    ax3.plot(range(1, len(val_losses) + 1), val_losses, label="Val")
ax3.set_xlabel("Epoch")
ax3.set_ylabel("L1 Loss")
ax3.set_title("Training Loss Curve")
ax3.legend()
ax3.grid(True, alpha=0.3)
fig3.tight_layout()

out_loss = OUT_DIR / "loss_curve.png"
fig3.savefig(out_loss, dpi=150, bbox_inches="tight")
print(f"Saved → {out_loss}")
plt.close(fig3)
