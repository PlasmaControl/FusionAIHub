from pathlib import Path
from tokamak_foundation_model.data.multi_file_dataset import TokamakMultiFileDataset
from tokamak_foundation_model.data.preprocess_data import compute_preprocessing_stats


def main():
    hdf5_files = sorted(
        Path("/scratch/gpfs/EKOLEMEN/foundation_model/").glob("20000*_processed.h5")
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

    dataset = TokamakMultiFileDataset(
        hdf5_paths=hdf5_files,
        input_signals=all_input_signals,
        target_signals=all_input_signals,
        lengths_cache_path="dataset_lengths.pt",
        max_open_files=8,
        max_duration_s=10.,
    )

    compute_preprocessing_stats(dataset, 'preprocessing_stats_tmp.pt')


if __name__ == "__main__":
    # python scripts/data_preparation/make_processing_stats.py
    main()
