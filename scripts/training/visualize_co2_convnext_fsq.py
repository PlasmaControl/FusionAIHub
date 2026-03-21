"""Visualize CO2 ConvNeXt-FSQ reconstruction vs. original for all channels.

Usage:
    pixi run python scripts/training/visualize_co2_convnext_fsq.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from tokamak_foundation_model.data.data_loader import TokamakH5Dataset
from tokamak_foundation_model.models.model_factory import build_model

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT  = Path("runs/co2_convnext_fsq/co2_spectrogram_convnext_fsq/checkpoint.pth")
DATA_DIR    = Path("/scratch/gpfs/EKOLEMEN/foundation_model")
STATS_PATH  = Path("data/preprocessing_stats.pt")
SIGNAL      = "co2"
N_FFT       = 256
HOP_LENGTH  = 128
SAMPLE_IDX  = 10        # which dataset sample to visualise
OUT_PATH    = Path("runs/co2_convnext_fsq/co2_spectrogram_convnext_fsq/plots/reconstruction_final.png")
# ─────────────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load checkpoint ───────────────────────────────────────────────────────────
ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)

# ── Build model (match training config) ──────────────────────────────────────
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
sample = dataset[SAMPLE_IDX][SIGNAL]   # (C, F, T)
n_channels = sample.shape[0]

model = build_model(
    "spectrogram_convnext_fsq", d_model=256, n_tokens=0,
    n_channels=n_channels,
    dims=[64, 128, 256], depths=[2, 2, 6],
    stem_stride=4, fsq_levels=[8, 5, 5, 5, 5],
)
model.load_state_dict(ckpt["model_state_dict"])
model.to(device).eval()

# ── Inference ─────────────────────────────────────────────────────────────────
with torch.no_grad():
    x = sample.unsqueeze(0).to(device)   # (1, C, F, T)
    reconstructed = model(x)[0].cpu()    # (C, F, T)

original = sample.cpu()   # (C, F, T)

# ── Plot ──────────────────────────────────────────────────────────────────────
n_cols = 3   # original | reconstructed | error
fig, axes = plt.subplots(n_channels, n_cols,
                          figsize=(n_cols * 5, n_channels * 3))

vmin = original.min().item()
vmax = original.max().item()

for ch in range(n_channels):
    orig_ch  = original[ch].numpy()
    recon_ch = reconstructed[ch].numpy()
    err_ch   = (recon_ch - orig_ch)

    axes[ch, 0].imshow(orig_ch,  cmap="viridis", origin="lower",
                       aspect="auto", vmin=vmin, vmax=vmax)
    axes[ch, 1].imshow(recon_ch, cmap="viridis", origin="lower",
                       aspect="auto", vmin=vmin, vmax=vmax)
    axes[ch, 2].imshow(err_ch,   cmap="bwr",     origin="lower",
                       aspect="auto",
                       vmin=-abs(err_ch).max(), vmax=abs(err_ch).max())

    for ax in axes[ch]:
        ax.set_xticks([])
        ax.set_yticks([])

    axes[ch, 0].set_ylabel(f"Ch {ch}", fontsize=9)

axes[0, 0].set_title("Original",       fontsize=11)
axes[0, 1].set_title("Reconstructed",  fontsize=11)
axes[0, 2].set_title("Error (R − O)",  fontsize=11)

epoch = ckpt.get("epoch", "?")
train_losses = ckpt["tracker_state_dict"]["history"]["train"]["loss"]
final_loss = train_losses[-1]
fig.suptitle(f"CO2 ConvNeXt-FSQ — epoch {epoch + 1}, train L1={final_loss:.4f}", fontsize=12)
fig.tight_layout()

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT_PATH}")
