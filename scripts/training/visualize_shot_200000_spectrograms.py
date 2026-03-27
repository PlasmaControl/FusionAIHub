"""Visualize spectrogram samples for MHR, ECE, CO2 with cleanup pipeline.

Applies per-channel: quantile clipping + gamma correction + stationary gate.

Usage:
    pixi run python scripts/training/visualize_shot_200000_spectrograms.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import torch

from tokamak_foundation_model.data.data_loader import TokamakH5Dataset

# ── Config ────────────────────────────────────────────────────────────────────
SHOT       = 205010
SHOT_FILE  = Path(f"/scratch/gpfs/EKOLEMEN/foundation_model/{SHOT}_processed.h5")
STATS_PATH = Path("data/preprocessing_stats.pt")
N_FFT      = 256
HOP_LENGTH = 128
OUT_DIR    = Path(f"docs/shot_{SHOT}_spectrograms")
SIGNALS    = ["mhr", "ece", "co2"]
# ─────────────────────────────────────────────────────────────────────────────

_eps = 1e-10


def clip_and_normalize(
    X: np.ndarray,
    *,
    clip_quantiles: tuple[float, float] = (0.005, 0.999),
    clip_frequency: tuple[int | None, int | None] = (2, None),
) -> np.ndarray:
    """Quantile clip and normalize to [0, 1]. Input is already log-transformed."""
    X = X.copy()
    X[:, np.sum(np.abs(X), axis=0) == 0.0] = np.nan

    xmin, xmax = np.nanquantile(
        X[clip_frequency[0]:clip_frequency[1], :], clip_quantiles
    )
    if xmin == xmax:
        return np.full_like(X, np.nan)

    X_clipped = np.clip((X - xmin) / (xmax - xmin), 0.0, 1.0)
    return X_clipped


def gamma_correction(
    X: np.ndarray, *, gamma: float = 2.0, alpha: float = 0.0
) -> np.ndarray:
    return np.clip((X - alpha) ** gamma, None, 1.0)


def stationary_gate(
    X: np.ndarray,
    *,
    threshold: float = 1.5,
    gate_smooth: float = 2.0,
    gate_factor: float = 0.5,
) -> np.ndarray:
    freq_mean = np.nanmean(X, axis=1)
    freq_std = np.nanstd(X, axis=1)
    X_gate = X > (freq_mean + threshold * freq_std)[:, np.newaxis]
    X_gate = X_gate.astype(np.float32)
    X_gate = scipy.ndimage.gaussian_filter(X_gate, sigma=gate_smooth)
    return X * (X_gate * gate_factor + (1.0 - gate_factor))


def cleanup_spectrogram(X: np.ndarray) -> np.ndarray:
    """Apply full cleanup pipeline to a single (F, T) spectrogram."""
    X = clip_and_normalize(X)
    X = gamma_correction(X, gamma=1.2)
    return X


# ─────────────────────────────────────────────────────────────────────────────

OUT_DIR.mkdir(parents=True, exist_ok=True)
stats = torch.load(STATS_PATH, weights_only=False)

for signal in SIGNALS:
    print(f"\n── {signal.upper()} ──")
    dataset = TokamakH5Dataset(
        hdf5_path=str(SHOT_FILE),
        preprocessing_stats=stats,
        input_signals=[signal],
        target_signals=[signal],
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        prediction_mode=False,
    )
    n_chunks = len(dataset)
    print(f"  Chunks available: {n_chunks}")

    for chunk_idx in range(min(n_chunks, 2)):
        sample = dataset[chunk_idx][signal]  # (C, F, T)
        n_channels, freq_bins, time_bins = sample.shape
        print(f"  Chunk {chunk_idx} shape: ({n_channels}, {freq_bins}, {time_bins})")

        # For ECE (many channels), show evenly spaced subset
        if n_channels > 12:
            show_channels = list(range(0, n_channels, max(1, n_channels // 10)))[:10]
            subtitle = f"(showing {len(show_channels)} of {n_channels} channels)"
        else:
            show_channels = list(range(n_channels))
            subtitle = f"({n_channels} channels)"

        n_rows = len(show_channels)
        fig, axes = plt.subplots(n_rows, 1, figsize=(14, n_rows * 2), squeeze=False)

        for row, ch in enumerate(show_channels):
            ax = axes[row, 0]
            ch_data = cleanup_spectrogram(sample[ch].numpy())
            im = ax.imshow(
                ch_data,
                cmap="inferno",
                origin="lower",
                aspect="auto",
                vmin=0,
                vmax=np.nanmax(ch_data) if not np.all(np.isnan(ch_data)) else 1,
            )
            ax.set_ylabel(f"Ch {ch}", fontsize=9)
            ax.set_yticks([0, freq_bins // 2, freq_bins - 1])
            if row < n_rows - 1:
                ax.set_xticks([])

        axes[-1, 0].set_xlabel("Time bin")
        axes[0, 0].set_title(
            f"Shot {SHOT} — {signal.upper()} spectrograms {subtitle}\n"
            f"STFT: n_fft={N_FFT}, hop={HOP_LENGTH} | chunk {chunk_idx} | "
            f"shape=({n_channels}, {freq_bins}, {time_bins}) | "
            f"clip + gamma(1.2)",
            fontsize=11,
        )

        fig.tight_layout()
        out = OUT_DIR / f"{signal}_chunk{chunk_idx}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {out}")
        plt.close(fig)

print("\nDone!")
