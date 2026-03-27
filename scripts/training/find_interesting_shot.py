"""Scan shots and rank by combined spectral interestingness across MHR+ECE+CO2."""

from pathlib import Path
import numpy as np
import torch

from tokamak_foundation_model.data.data_loader import TokamakH5Dataset

DATA_DIR   = Path("/scratch/gpfs/EKOLEMEN/foundation_model")
STATS_PATH = Path("data/preprocessing_stats.pt")
N_FFT      = 256
HOP_LENGTH = 128
SIGNALS    = ["mhr", "ece", "co2"]

stats = torch.load(STATS_PATH, weights_only=False)
all_files = sorted(DATA_DIR.glob("*_processed.h5"))
print(f"Total shots: {len(all_files)}")

# Sample ~50 shots spread across the range
indices = np.linspace(0, len(all_files) - 1, 50, dtype=int)
candidates = [all_files[i] for i in indices]

results = []
for f in candidates:
    shot = int(f.stem.split("_")[0])
    scores = {}
    all_ok = True
    for sig in SIGNALS:
        try:
            ds = TokamakH5Dataset(
                hdf5_path=str(f),
                preprocessing_stats=stats,
                input_signals=[sig],
                target_signals=[sig],
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                prediction_mode=False,
            )
            if len(ds) == 0:
                all_ok = False
                break
            sample = ds[0][sig]
            var = sample.var(dim=-1).mean().item()
            # Also check the signal isn't dead (very low overall magnitude)
            mag = sample.abs().mean().item()
            if mag < 0.01:
                all_ok = False
                break
            scores[sig] = var
        except Exception:
            all_ok = False
            break

    if all_ok and len(scores) == 3:
        # Combined score: geometric mean so all signals must contribute
        combined = (scores["mhr"] * scores["ece"] * scores["co2"]) ** (1/3)
        results.append((shot, combined, scores, str(f)))

results.sort(key=lambda x: x[1], reverse=True)

print(f"\nTop 10 shots (geometric mean of MHR/ECE/CO2 variance):")
print(f"{'Shot':>8}  {'Combined':>9}  {'MHR':>8}  {'ECE':>8}  {'CO2':>8}")
print("-" * 50)
for shot, combined, scores, path in results[:10]:
    print(f"{shot:>8}  {combined:>9.4f}  {scores['mhr']:>8.4f}  {scores['ece']:>8.4f}  {scores['co2']:>8.4f}")
