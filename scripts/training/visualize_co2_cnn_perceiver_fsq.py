"""Visualize CO2 CNN-Perceiver-FSQ reconstruction and codebook usage.

Produces three figures:
  1. Original / Reconstructed / Error per channel
  2. Training + validation loss curves, with codebook utilization on a twin axis
  3. Codebook index heatmap — token indices across a batch of samples

Usage:
    pixi run python scripts/training/visualize_co2_cnn_perceiver_fsq.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from tokamak_foundation_model.data.data_loader import TokamakH5Dataset, collate_fn
from tokamak_foundation_model.data.utils import worker_init_fn
from tokamak_foundation_model.models.model_factory import build_model

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT = Path("runs/co2_cnn_perceiver_fsq/co2_spectrogram_cnn_perceiver/checkpoint.pth")
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
FSQ_LEVELS = [8, 5, 5, 5, 5]
SAMPLE_IDX = 10
N_INDEX_SAMPLES = 32   # samples to use for codebook index heatmap
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
    fsq_levels=FSQ_LEVELS,
)
model.load_state_dict(ckpt["model_state_dict"])
model.to(device).eval()

# ── Inference (single sample) ─────────────────────────────────────────────────
with torch.no_grad():
    x = sample.unsqueeze(0).to(device)           # (1, C, F, T)
    reconstructed = model(x).cpu().squeeze(0)    # (C, F, T)

original = sample.cpu()

epoch = ckpt.get("epoch", "?")
history = ckpt["tracker_state_dict"]["history"]
train_losses  = history["train"]["loss"]
val_losses    = history.get("validate", {}).get("loss", [])
util_history  = history["train"].get("codebook_utilization", [])
final_loss    = train_losses[-1]
n_codes       = model.fsq.n_codes

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
    f"CO2 CNN-Perceiver-FSQ (N={N_TOKENS} tokens, {n_codes} codes) "
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

ax_loss.set_title(f"CO2 CNN-Perceiver-FSQ — Loss & Codebook Utilisation (N={N_TOKENS})")
fig2.tight_layout()
out = OUT_DIR / "loss_curve.png"
fig2.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
plt.close(fig2)

# ── Figure 3: Codebook index heatmap across a batch ───────────────────────────
loader = DataLoader(
    dataset,
    batch_size=N_INDEX_SAMPLES,
    shuffle=False,
    collate_fn=collate_fn,
    worker_init_fn=worker_init_fn,
    num_workers=0,
)
batch = next(iter(loader))
x_batch = batch[SIGNAL].to(device)   # (B, C, F, T)

# Switch to training mode temporarily so the model returns (recon, indices)
model.train()
with torch.no_grad():
    _, indices = model(x_batch)       # indices: (B, N)
model.eval()

indices_np = indices.cpu().numpy()    # (B, N)

fig3, ax3 = plt.subplots(figsize=(max(6, N_TOKENS * 0.5), max(4, N_INDEX_SAMPLES * 0.25)))
im = ax3.imshow(indices_np, cmap="tab20", origin="upper", aspect="auto",
                vmin=0, vmax=n_codes - 1)
ax3.set_xlabel("Token index")
ax3.set_ylabel("Sample")
ax3.set_title(
    f"CO2 CNN-Perceiver-FSQ — Codebook indices "
    f"(N={N_TOKENS} tokens, {n_codes} codes total)"
)
plt.colorbar(im, ax=ax3, shrink=0.8, label="Code index")
fig3.tight_layout()
out = OUT_DIR / "codebook_indices.png"
fig3.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
plt.close(fig3)
