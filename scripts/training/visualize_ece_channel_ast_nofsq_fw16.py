"""Visualize ECE Channel-AST (no FSQ, fw=16) reconstruction.

Usage:
    pixi run python scripts/training/visualize_ece_channel_ast_nofsq_fw16.py
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from tokamak_foundation_model.data.data_loader import TokamakH5Dataset
from tokamak_foundation_model.models.model_factory import build_model

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT = Path("runs/ece_channel_ast_nofsq_fw16_v2/ece_spectrogram_channel_ast_fsq/checkpoint.pth")
DATA_DIR   = Path("/scratch/gpfs/EKOLEMEN/foundation_model")
STATS_PATH = Path("data/preprocessing_stats.pt")
SIGNAL     = "ece"
N_FFT      = 256
HOP_LENGTH = 128
SAMPLE_IDX = 10
FW         = 16
OUT_DIR    = CHECKPOINT.parent / "plots"
# ─────────────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)

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

model = build_model("spectrogram_channel_ast_fsq", d_model=256, n_tokens=0,
                    n_channels=n_channels, freq_bins=freq_bins, frame_width=FW,
                    fsq_levels=[], time_conv_kernel=7)
model.load_state_dict(ckpt["model_state_dict"])
model.to(device).eval()

with torch.no_grad():
    x = sample.unsqueeze(0).to(device)
    reconstructed = model(x).cpu().squeeze(0)

original = sample.cpu()

epoch = ckpt.get("epoch", "?")
history = ckpt["tracker_state_dict"]["history"]
train_losses = history["train"]["loss"]
val_losses   = history.get("validate", {}).get("loss", [])
final_loss   = train_losses[-1]

n_frames = math.ceil(sample.shape[2] / FW)
n_tokens_total = n_channels * n_frames

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Figure 1: Reconstruction (show subset of channels) ───────────────────────
# ECE has 48 channels — show 8 evenly spaced for readability
show_channels = list(range(0, n_channels, max(1, n_channels // 8)))[:8]
n_rows = len(show_channels)
n_cols = 3
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 2.5))
vmin = original.min().item()
vmax = original.max().item()

for row, ch in enumerate(show_channels):
    orig_ch  = original[ch].numpy()
    recon_ch = reconstructed[ch].numpy()
    err_ch   = recon_ch - orig_ch

    axes[row, 0].imshow(orig_ch,  cmap="viridis", origin="lower",
                        aspect="auto", vmin=vmin, vmax=vmax)
    axes[row, 1].imshow(recon_ch, cmap="viridis", origin="lower",
                        aspect="auto", vmin=vmin, vmax=vmax)
    emax = max(abs(err_ch).max(), 1e-8)
    axes[row, 2].imshow(err_ch,   cmap="bwr",     origin="lower",
                        aspect="auto", vmin=-emax, vmax=emax)

    for ax in axes[row]:
        ax.set_xticks([])
        ax.set_yticks([])
    axes[row, 0].set_ylabel(f"Ch {ch}", fontsize=9)

axes[0, 0].set_title("Original",      fontsize=11)
axes[0, 1].set_title("Reconstructed", fontsize=11)
axes[0, 2].set_title("Error (R − O)", fontsize=11)

fig.suptitle(
    f"ECE Channel-AST (no FSQ, fw={FW}, C={n_channels}, "
    f"{n_tokens_total} tokens) — epoch {epoch + 1}, train L1={final_loss:.4f}",
    fontsize=10,
)
fig.tight_layout()
out = OUT_DIR / "reconstruction.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
plt.close(fig)

# ── Figure 2: Loss curves ───────────────────────────────────────────────────
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
ax_loss.set_title(
    f"ECE Channel-AST (no FSQ, fw={FW}, C={n_channels}, {n_tokens_total} tokens)"
)
fig2.tight_layout()
out = OUT_DIR / "loss_curve.png"
fig2.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
plt.close(fig2)
