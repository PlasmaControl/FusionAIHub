"""Compare CO2 Channel-AST fw=8 vs fw=16 reconstruction on shot 205010."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from tokamak_foundation_model.data.data_loader import TokamakH5Dataset
from tokamak_foundation_model.models.model_factory import build_model

# ── Config ────────────────────────────────────────────────────────────────────
SHOT       = 205010
SHOT_FILE  = Path(f"/scratch/gpfs/EKOLEMEN/foundation_model/{SHOT}_processed.h5")
STATS_PATH = Path("data/preprocessing_stats.pt")
N_FFT      = 256
HOP_LENGTH = 128
CHUNK_IDX  = 0
OUT_DIR    = Path(f"docs/shot_{SHOT}_spectrograms")

VARIANTS = {
    "fw=16": {
        "ckpt": Path("runs/co2_channel_ast_nofsq_fw16/co2_spectrogram_channel_ast_fsq/checkpoint.pth"),
        "frame_width": 16,
    },
    "fw=8": {
        "ckpt": Path("runs/co2_channel_ast_nofsq_fw8/co2_spectrogram_channel_ast_fsq/checkpoint.pth"),
        "frame_width": 8,
    },
}
# ─────────────────────────────────────────────────────────────────────────────

_eps = 1e-10


def clip_and_normalize(X, *, clip_quantiles=(0.005, 0.999), clip_frequency=(2, None)):
    X = X.copy()
    X[:, np.sum(np.abs(X), axis=0) == 0.0] = np.nan
    xmin, xmax = np.nanquantile(X[clip_frequency[0]:clip_frequency[1], :], clip_quantiles)
    if xmin == xmax:
        return np.full_like(X, np.nan)
    return np.clip((X - xmin) / (xmax - xmin), 0.0, 1.0)


def gamma_correction(X, *, gamma=2.0, alpha=0.0):
    return np.clip((X - alpha) ** gamma, None, 1.0)


def cleanup_spectrogram(X):
    X = clip_and_normalize(X)
    X = gamma_correction(X, gamma=1.2)
    return X


# ─────────────────────────────────────────────────────────────────────────────

OUT_DIR.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stats = torch.load(STATS_PATH, weights_only=False)

# Load data
dataset = TokamakH5Dataset(
    hdf5_path=str(SHOT_FILE),
    preprocessing_stats=stats,
    input_signals=["co2"],
    target_signals=["co2"],
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    prediction_mode=False,
)
sample = dataset[CHUNK_IDX]["co2"]  # (C, F, T)
n_channels, freq_bins, time_bins = sample.shape
print(f"CO2 shape: ({n_channels}, {freq_bins}, {time_bins})")

# Run both models
recons = {}
meta = {}
for name, cfg in VARIANTS.items():
    ckpt = torch.load(cfg["ckpt"], map_location=device, weights_only=False)
    model = build_model(
        "spectrogram_channel_ast_fsq", d_model=256, n_tokens=0,
        n_channels=n_channels, freq_bins=freq_bins, frame_width=cfg["frame_width"],
        fsq_levels=[], time_conv_kernel=7,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    with torch.no_grad():
        x = sample.unsqueeze(0).to(device)
        recons[name] = model(x).cpu().squeeze(0)

    epoch = ckpt.get("epoch", "?")
    history = ckpt["tracker_state_dict"]["history"]
    train_loss = history["train"]["loss"][-1]
    val_losses = history.get("validate", {}).get("loss", [])
    val_loss = val_losses[-1] if val_losses else None

    import math
    n_frames = math.ceil(time_bins / cfg["frame_width"])
    n_tokens = n_channels * n_frames
    compression = (freq_bins * cfg["frame_width"]) / 256

    meta[name] = {
        "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
        "n_tokens": n_tokens, "compression": compression,
    }
    print(f"  {name}: epoch {epoch+1}, train L1={train_loss:.4f}, "
          f"val L1={val_loss:.4f}, " if val_loss else "val L1=N/A, "
          f"{n_tokens} tokens, {compression:.1f}:1 compression")

original = sample.cpu()

# Plot: 3 columns (original, fw=16, fw=8) x n_channels rows
n_cols = 3
fig, axes = plt.subplots(n_channels, n_cols, figsize=(n_cols * 5, n_channels * 3))

col_labels = ["Original", "fw=16 recon", "fw=8 recon"]
col_data = [original, recons["fw=16"], recons["fw=8"]]

for ch in range(n_channels):
    for col, (label, data) in enumerate(zip(col_labels, col_data)):
        ax = axes[ch, col]
        ch_data = data[ch].numpy()
        ax.imshow(ch_data, cmap="inferno", origin="lower", aspect="auto")
        ax.set_yticks([0, freq_bins // 2, freq_bins - 1])
        if ch < n_channels - 1:
            ax.set_xticks([])
        if col == 0:
            ax.set_ylabel(f"Ch {ch}", fontsize=9)

    # Add title on first row
    if ch == 0:
        for col, label in enumerate(col_labels):
            if col == 0:
                axes[0, col].set_title(label, fontsize=11)
            else:
                name = list(VARIANTS.keys())[col - 1]
                m = meta[name]
                loss_str = f"val L1={m['val_loss']:.4f}" if m['val_loss'] else f"train L1={m['train_loss']:.4f}"
                axes[0, col].set_title(
                    f"{label}\n{m['n_tokens']} tokens, {m['compression']:.0f}:1 | {loss_str}",
                    fontsize=10,
                )

axes[-1, 1].set_xlabel("Time bin")

fig.suptitle(
    f"Shot {SHOT} — CO2 Channel-AST: fw=16 vs fw=8 (no FSQ, d_model=256)",
    fontsize=12, y=1.01,
)
fig.tight_layout()
out = OUT_DIR / "co2_fw8_vs_fw16.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved -> {out}")
plt.close(fig)
