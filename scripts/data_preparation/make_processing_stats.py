from pathlib import Path
from tokamak_foundation_model.data.preprocess_data import compute_preprocessing_stats


def main():
    hdf5_files = sorted(
        Path("/scratch/gpfs/EKOLEMEN/foundation_model/").glob("*_processed.h5")
    )

    all_signals = [
        # STFT spectrograms
        "mhr", "ece", "co2",
        # actuators / gas / heating
        "ech_power", "ech_tor_angle", "ech_pol_angle", "ech_polarization",
        "pin", "beam_voltage", "tin", "gas_flow", "gas_raw", "ich", "rmp",
        # diagnostics
        "filterscopes", "vib", "mse", "ts_core_density", "ts_core_temp",
        "ts_tangential_density", "ts_tangential_temp", "cer_ti", "cer_rot",
        "sxr", "neutron_rate", "bolo_raw", "mirnov", "langmuir", "i_coil",
        "bes",
        # cameras
        "irtv", "tangtv",
    ]

    stft_signals = {"mhr", "ece", "co2", "mirnov", "langmuir", "bes"}

    compute_preprocessing_stats(
        hdf5_paths=hdf5_files,
        signal_names=all_signals,
        output_path="preprocessing_stats.pt",
        stft_signals=stft_signals,
        num_workers=7,
    )


if __name__ == "__main__":
    main()