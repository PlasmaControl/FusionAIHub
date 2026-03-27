"""Diagnose data loading for MHR, ECE, CO2 on shot 205010.

Checks raw HDF5 data vs what TokamakH5Dataset produces.
"""

from pathlib import Path

import h5py
import numpy as np
import torch

from tokamak_foundation_model.data.data_loader import TokamakH5Dataset

SHOT_FILE  = Path("/scratch/gpfs/EKOLEMEN/foundation_model/205010_processed.h5")
STATS_PATH = Path("data/preprocessing_stats.pt")
N_FFT      = 256
HOP_LENGTH = 128
SIGNALS    = ["mhr", "ece", "co2"]

stats = torch.load(STATS_PATH, weights_only=False)

# ── 1. Raw HDF5 inspection ──────────────────────────────────────────────────
print("=" * 60)
print("RAW HDF5 INSPECTION")
print("=" * 60)
with h5py.File(SHOT_FILE, "r") as f:
    print(f"\nTop-level keys: {list(f.keys())}")
    for sig in SIGNALS:
        if sig in f:
            grp = f[sig]
            print(f"\n── {sig.upper()} ──")
            print(f"  Keys: {list(grp.keys())}")
            if "ydata" in grp:
                ydata = grp["ydata"]
                print(f"  ydata shape: {ydata.shape}, dtype: {ydata.dtype}")
                data = ydata[:]
                print(f"  ydata range: [{data.min():.6f}, {data.max():.6f}]")
                print(f"  ydata mean: {data.mean():.6f}, std: {data.std():.6f}")
                # Check per-channel stats
                for ch in range(min(data.shape[0], 10)):
                    ch_data = data[ch]
                    nonzero = np.count_nonzero(ch_data)
                    print(f"    Ch {ch}: range=[{ch_data.min():.4f}, {ch_data.max():.4f}], "
                          f"mean={ch_data.mean():.4f}, std={ch_data.std():.4f}, "
                          f"nonzero={nonzero}/{ch_data.size} ({100*nonzero/ch_data.size:.1f}%)")
                if data.shape[0] > 10:
                    print(f"    ... ({data.shape[0]} channels total, showing first 10)")
            if "xdata" in grp:
                xdata = grp["xdata"][:]
                print(f"  xdata shape: {xdata.shape}, range: [{xdata.min():.2f}, {xdata.max():.2f}] ms")
        else:
            print(f"\n── {sig.upper()} — NOT FOUND in HDF5 ──")

# ── 2. DataLoader output inspection ─────────────────────────────────────────
print("\n" + "=" * 60)
print("DATALOADER OUTPUT INSPECTION")
print("=" * 60)
for sig in SIGNALS:
    print(f"\n── {sig.upper()} ──")
    dataset = TokamakH5Dataset(
        hdf5_path=str(SHOT_FILE),
        preprocessing_stats=stats,
        input_signals=[sig],
        target_signals=[sig],
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        prediction_mode=False,
    )
    print(f"  Number of chunks: {len(dataset)}")

    for chunk_idx in range(min(len(dataset), 2)):
        sample = dataset[chunk_idx][sig]  # (C, F, T)
        C, F, T = sample.shape
        print(f"\n  Chunk {chunk_idx}: shape=({C}, {F}, {T})")
        print(f"    Global range: [{sample.min():.4f}, {sample.max():.4f}]")
        print(f"    Global mean: {sample.mean():.4f}, std: {sample.std():.4f}")
        print(f"    NaN count: {torch.isnan(sample).sum().item()}")
        print(f"    Inf count: {torch.isinf(sample).sum().item()}")
        print(f"    All-zero frames (time): {(sample.abs().sum(dim=(0,1)) == 0).sum().item()}/{T}")

        for ch in range(min(C, 10)):
            ch_data = sample[ch]
            all_zero_freq = (ch_data.abs().sum(dim=1) == 0).sum().item()
            all_zero_time = (ch_data.abs().sum(dim=0) == 0).sum().item()
            print(f"    Ch {ch}: range=[{ch_data.min():.4f}, {ch_data.max():.4f}], "
                  f"mean={ch_data.mean():.4f}, std={ch_data.std():.4f}, "
                  f"zero_time_cols={all_zero_time}/{T}, zero_freq_rows={all_zero_freq}/{F}")
        if C > 10:
            # Summarize remaining channels
            for ch in range(10, C):
                ch_data = sample[ch]
                if ch_data.abs().max() < 1e-6:
                    print(f"    Ch {ch}: DEAD (all near-zero)")
            print(f"    ... ({C} channels total)")
