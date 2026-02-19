from pathlib import Path
from tokamak_foundation_model.data.data_loader import (
    TokamakH5Dataset, compute_preprocessing_stats)

def main():
    hdf5_files = sorted(
        Path("/scratch/gpfs/EKOLEMEN/foundation_model/"
             ).glob("[0-9]*_processed.h5")
    )

    # hdf5_files = sorted(
    #     Path("/scratch/gpfs/EKOLEMEN/foundation_model").glob("*_processed.h5")
    # )

    all_input_signals = [
        "mhr", "ece", "co2", "bes",  # spectrograms
        "gas", "ech", "pin", "tin",  # actuators
        "d_alpha", "mse", "ts_core_density",  # diagnostics
        "bolo", "irtv", "tangtv",  # videos
        # "text",  # metadata
    ]

    datasets = [
        TokamakH5Dataset(
            hdf5_path=str(f),
            input_signals=all_input_signals,
            target_signals=all_input_signals,
        ) for f in hdf5_files]

    stats = compute_preprocessing_stats(datasets, 'preprocessing_stats.pt')


if __name__ == "__main__":
    # python scripts/data_preparation/make_processing_stats.py
    main()
