"""Visualize CO2 Channel-AST-FSQ reconstruction (shots 200000–200500).

Produces two figures:
  1. Original / Reconstructed / Error per channel
  2. Training + validation loss curves with codebook utilization

Usage:
    pixi run python scripts/training/visualize_co2_channel_ast_fsq.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from tokamak_foundation_model.data.data_loader import TokamakH5Dataset
from tokamak_foundation_model.models.model_factory import build_model

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT = Path("runs/co2_channel_ast_fsq/co2_spectrogram_channel_ast_fsq/checkpoint.pth")
DATA_DIR   = Path("/scratch/gpfs/EKOLEMEN/foundation_model")
STATS_PATH = Path("data/preprocessing_stats.pt")
SIGNAL     = "co2"
N_FFT      = 256
HOP_LENGTH = 128
SAMPLE_IDX = 10
OUT_DIR    = CHECKPOINT.parent / "plots"
# ─────────────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load checkpoint ───────────────────────────────────────────────────────────
ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)

# ── Dataset (matched shot range) ─────────────────────────────────────────────
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

# ── Build model ───────────────────────────────────────────────────────────────
model = build_model("spectrogram_channel_ast_fsq", d_model=256, n_tokens=0,
                    n_channels=n_channels, freq_bins=freq_bins, frame_width=2,
                    fsq_levels=[8,8,5,5,5,5,5], time_conv_kernel=7)
model.load_state_dict(ckpt["model_state_dict"])
model.to(device).eval()

# ── Inference ─────────────────────────────────────────────────────────────────
with torch.no_grad():
    x = sample.unsqueeze(0).to(device)           # (1, C, F, T)
    reconstructed = model(x).cpu().squeeze(0)    # (C, F, T)

original = sample.cpu()

epoch = ckpt.get("epoch", "?")
history = ckpt["tracker_state_dict"]["history"]
train_losses = history["train"]["loss"]
val_losses   = history.get("validate", {}).get("loss", [])
util_history = history["train"].get("codebook_utilization", [])
final_loss   = train_losses[-1]
n_codes      = model.fsq.n_codes if hasattr(model, "fsq") else 0

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

final_util = util_history[-1] if util_history else float("nan")
fig.suptitle(
    f"CO2 Channel-AST-FSQ frame_width=2 ({n_codes} codes) "
    f"— epoch {epoch + 1}, train L1={final_loss:.4f}, "
    f"codebook util={final_util:.1%}",
    fontsize=11,
)
fig.tight_layout()
out = OUT_DIR / "reconstruction.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
plt.close(fig)

# ── Figure 2: Loss + utilisation curves ──────────────────────────────────────
fig2, ax_loss = plt.subplots(figsize=(9, 4))
ax_loss.plot(range(1, len(train_losses) + 1), train_losses,
             color="tab:blue", label="Train L1")
if val_losses:
    ax_loss.plot(range(1, len(val_losses) + 1), val_losses,
                 color="tab:orange", label="Val L1")
ax_loss.set_xlabel("Epoch")
ax_loss.set_ylabel("L1 Loss", color="tab:blue")
ax_loss.tick_params(axis="y", labelcolor="tab:blue")
ax_loss.grid(True, alpha=0.3)

if util_history:
    ax_util = ax_loss.twinx()
    ax_util.plot(range(1, len(util_history) + 1), util_history,
                 color="tab:green", linestyle="--", alpha=0.8,
                 label="Codebook util")
    ax_util.set_ylabel("Codebook utilisation", color="tab:green")
    ax_util.tick_params(axis="y", labelcolor="tab:green")
    ax_util.set_ylim(0, 1)
    lines1, labels1 = ax_loss.get_legend_handles_labels()
    lines2, labels2 = ax_util.get_legend_handles_labels()
    ax_loss.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
else:
    ax_loss.legend()

ax_loss.set_title("CO2 Channel-AST-FSQ — Loss & Codebook Utilisation")
fig2.tight_layout()
out = OUT_DIR / "loss_curve.png"
fig2.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
plt.close(fig2)
