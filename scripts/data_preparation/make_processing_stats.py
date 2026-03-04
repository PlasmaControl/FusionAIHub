from pathlib import Path
from tokamak_foundation_model.data.data_loader import (
    TokamakH5Dataset, compute_preprocessing_stats)

def main():
    hdf5_files = sorted(
        Path("/scratch/gpfs/EKOLEMEN/foundation_model/").glob("*_processed.h5")
    )

    all_input_signals = [
        # STFT spectrograms
        "mhr", "ece", "co2",
        # actuators / gas / heating
        "ech", "pin", "tin", "gas_flow", "gas_raw", "ich",
        # diagnostics
        "filterscopes", "vib", "mse", "ts_core_density", "ts_core_temp",
        "ts_tangential_density", "ts_tangential_temp", "cer_ti", "cer_rot",
        "sxr", "neutron_rate", "bolo_raw", "mirnov", "langmuir", "i_coil",
        "bes",
        # cameras
        "irtv", "tangtv",
        # "text",  # metadata
    ]

    datasets = [
        TokamakH5Dataset(
            hdf5_path=str(f),
            input_signals=all_input_signals,
            target_signals=all_input_signals,
            max_duration_s=10.,
        ) for f in hdf5_files]

    compute_preprocessing_stats(datasets, 'preprocessing_stats.pt')


if __name__ == "__main__":
    # python scripts/data_preparation/make_processing_stats.py
    main()
