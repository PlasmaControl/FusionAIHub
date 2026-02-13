from pathlib import Path
from tokamak_foundation_model.data.data_loader import (
    TokamakH5Dataset, compute_preprocessing_stats)

hdf5_files = sorted(
    Path(
        "C:/Users/admin/PycharmProjects/nstx/foundation_model_notes/tokamak_package/"
    ).glob("*_processed.h5")
)
all_input_signals = [
    "mhr", "ece", "co2",              # spectrograms
    "gas", "ech", "pin", "tin",        # actuators
    "d_alpha", "mse", "ts_core_density",  # diagnostics
    "bolo", "irtv", "tangtv",          # videos
    "text",                            # metadata
]

datasets = [
    TokamakH5Dataset(
        hdf5_path=str(f),
        input_signals=all_input_signals,
        target_signals=all_input_signals,
    ) for f in hdf5_files]
stats = compute_preprocessing_stats(datasets, 'preprocessing_stats.pt')
