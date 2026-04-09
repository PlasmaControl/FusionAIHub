from pathlib import Path
import torch
from torch.utils.data import ConcatDataset

from tokamak_foundation_model.data.data_loader import TokamakH5Dataset


def worker_init_fn(worker_id):
    """Each worker needs to open its own file handle."""
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        # Force re-open file for this worker
        if hasattr(dataset, 'datasets'):  # ConcatDataset
            for ds in dataset.datasets:
                ds.h5_file = None
                ds._open_hdf5()
        else:
            dataset.h5_file = None
            dataset._open_hdf5()


def data_loading_demo():
    print("Initializing and demonstrating custom DataLoader with updated TokamakH5Dataset")
    # Use glob to find all generated HDF5 files
    hdf5_files = sorted(
        Path("C:/Users/admin/PycharmProjects/nstx/foundation_model_notes/"
             "tokamak_package/").glob("*_processed.h5")
    )
    stats = torch.load(
        "C:/Users/admin/PycharmProjects/nstx/foundation_model_notes/"
        "tokamak_package/preprocessing_stats.pt",
        weights_only=False
    )
    all_input_signals = [
        "mhr",
        "ece",
        "co2",
        "mirnov",
        "langmuir",  # spectrograms
        "i_coil",  # fast time series
        "gas_flow",
        "gas_raw",
        "ech_power",
        "ech_tor_angle",
        "ech_pol_angle",
        "ech_polarization",
        "pin",
        "beam_voltage",
        "tin",
        "ich",
        "rmp",  # actuators
        "d_alpha",
        "mse",
        "ts_core_density",  # diagnostics
        "bolo",
        "irtv",
        "tangtv",  # videos
        "text",  # metadata
    ]

    datasets_processed = [TokamakH5Dataset(hdf5_path=str(f), preprocessing_stats=stats,
                                           input_signals=all_input_signals,
                                           target_signals=all_input_signals,
                                           prediction_mode=False) for f in hdf5_files]

    concatenated_dataset = ConcatDataset(datasets_processed)


    # Get and print the first batch from DataLoader to verify functionality
    for k in range(len(concatenated_dataset)):
        concatenated_dataset.__getitem__(k)

if __name__ == "__main__":
    data_loading_demo()
