"""Visualize original vs reconstructed spectrograms for MHR, ECE, CO2.

For each modality, shows original (top row per channel) and reconstruction
(bottom row per channel) side by side using the fw=16 Channel-AST models.

Usage:
    pixi run python scripts/training/visualize_shot_205010_reconstruction.py
"""

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

MODELS = {
    "mhr": Path("runs/mhr_channel_ast_nofsq_fw16/mhr_spectrogram_channel_ast_fsq/checkpoint.pth"),
    "ece": Path("runs/ece_channel_ast_nofsq_fw16/ece_spectrogram_channel_ast_fsq/checkpoint.pth"),
    "co2": Path("runs/co2_channel_ast_nofsq_fw16/co2_spectrogram_channel_ast_fsq/checkpoint.pth"),
}
# ─────────────────────────────────────────────────────────────────────────────

_eps = 1e-10


def clip_and_normalize(
    X: np.ndarray,
    *,
    clip_quantiles: tuple[float, float] = (0.005, 0.999),
    clip_frequency: tuple[int | None, int | None] = (2, None),
) -> np.ndarray:
    X = X.copy()
    X[:, np.sum(np.abs(X), axis=0) == 0.0] = np.nan
    xmin, xmax = np.nanquantile(
        X[clip_frequency[0]:clip_frequency[1], :], clip_quantiles
    )
    if xmin == xmax:
        return np.full_like(X, np.nan)
    return np.clip((X - xmin) / (xmax - xmin), 0.0, 1.0)


def gamma_correction(
    X: np.ndarray, *, gamma: float = 2.0, alpha: float = 0.0
) -> np.ndarray:
    return np.clip((X - alpha) ** gamma, None, 1.0)


def cleanup_spectrogram(X: np.ndarray) -> np.ndarray:
    X = clip_and_normalize(X)
    X = gamma_correction(X, gamma=1.2)
    return X


# ─────────────────────────────────────────────────────────────────────────────

OUT_DIR.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stats = torch.load(STATS_PATH, weights_only=False)

for signal, ckpt_path in MODELS.items():
    print(f"\n── {signal.upper()} ──")

    # Load data
    dataset = TokamakH5Dataset(
        hdf5_path=str(SHOT_FILE),
        preprocessing_stats=stats,
        input_signals=[signal],
        target_signals=[signal],
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        prediction_mode=False,
    )
    sample = dataset[CHUNK_IDX][signal]  # (C, F, T)
    n_channels, freq_bins, time_bins = sample.shape
    print(f"  Shape: ({n_channels}, {freq_bins}, {time_bins})")

    # Load model
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(
        "spectrogram_channel_ast_fsq", d_model=256, n_tokens=0,
        n_channels=n_channels, freq_bins=freq_bins, frame_width=16,
        fsq_levels=[], time_conv_kernel=7,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    epoch = ckpt.get("epoch", "?")
    history = ckpt["tracker_state_dict"]["history"]
    train_loss = history["train"]["loss"][-1]
    val_losses = history.get("validate", {}).get("loss", [])
    val_loss = val_losses[-1] if val_losses else None

    # Reconstruct
    with torch.no_grad():
        x = sample.unsqueeze(0).to(device)
        recon = model(x).cpu().squeeze(0)

    original = sample.cpu()

    # Channel selection for ECE
    if n_channels > 12:
        show_channels = list(range(0, n_channels, max(1, n_channels // 10)))[:10]
        subtitle = f"(showing {len(show_channels)} of {n_channels} channels)"
    else:
        show_channels = list(range(n_channels))
        subtitle = f"({n_channels} channels)"

    # Plot: 2 rows per channel (original + reconstruction)
    n_show = len(show_channels)
    fig, axes = plt.subplots(
        n_show * 2, 1,
        figsize=(14, n_show * 3.2),
        squeeze=False,
    )

    for i, ch in enumerate(show_channels):
        ax_orig = axes[i * 2, 0]
        ax_recon = axes[i * 2 + 1, 0]

        orig_clean = cleanup_spectrogram(original[ch].numpy())
        recon_clean = cleanup_spectrogram(recon[ch].numpy())

        # Use same vmax for original and reconstruction
        vmax = max(
            np.nanmax(orig_clean) if not np.all(np.isnan(orig_clean)) else 1,
            np.nanmax(recon_clean) if not np.all(np.isnan(recon_clean)) else 1,
        )

        ax_orig.imshow(
            orig_clean, cmap="inferno", origin="lower",
            aspect="auto", vmin=0, vmax=vmax,
        )
        ax_recon.imshow(
            recon_clean, cmap="inferno", origin="lower",
            aspect="auto", vmin=0, vmax=vmax,
        )

        ax_orig.set_ylabel(f"Ch {ch}\norig", fontsize=8)
        ax_recon.set_ylabel(f"Ch {ch}\nrecon", fontsize=8)

        for ax in [ax_orig, ax_recon]:
            ax.set_yticks([0, freq_bins // 2, freq_bins - 1])
            ax.set_xticks([])

    axes[-1, 0].set_xlabel("Time bin")

    loss_str = f"train L1={train_loss:.4f}"
    if val_loss is not None:
        loss_str += f", val L1={val_loss:.4f}"

    axes[0, 0].set_title(
        f"Shot {SHOT} — {signal.upper()} Channel-AST (fw=16, no FSQ) {subtitle}\n"
        f"epoch {epoch + 1} | {loss_str} | "
        f"shape=({n_channels}, {freq_bins}, {time_bins})",
        fontsize=11,
    )

    fig.tight_layout()
    out = OUT_DIR / f"{signal}_reconstruction.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved -> {out}")
    plt.close(fig)

print("\nDone!")
