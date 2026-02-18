from pathlib import Path
from tqdm.auto import tqdm
from torch.utils.data import ConcatDataset
from tokamak_foundation_model.data.data_loader import (
    TokamakH5Dataset)

# hdf5_files = sorted(
#     Path(
#         "/scratch/gpfs/EKOLEMEN/foundation_model"
#     ).glob("*_processed.h5")
# )

hdf5_files = sorted(
    Path(
        "/scratch/gpfs/EKOLEMEN/big_d3d_data/dummy_foundation_model_data"
    ).glob("*_processed.h5")
)

all_input_signals = [
    "mhr", "ece", "co2", "bes",              # spectrograms
    "gas", "ech", "pin", "tin",        # actuators
    "d_alpha", "mse", "ts_core_density",  # diagnostics
    "bolo", "irtv", "tangtv",          # videos
    # "text",                            # metadata
]

datasets = [
    TokamakH5Dataset(
        hdf5_path=str(f),
        input_signals=all_input_signals,
        target_signals=all_input_signals,
    ) for f in hdf5_files[:1]]

datasets = ConcatDataset(datasets)

for i in tqdm(range(len(datasets))):
    print(datasets[i]['inputs'].keys())
    break