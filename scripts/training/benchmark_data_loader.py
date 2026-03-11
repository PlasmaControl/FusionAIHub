from pathlib import Path
import torch
from torch.utils.data import ConcatDataset, DataLoader

from tokamak_foundation_model.data.data_loader import TokamakH5Dataset, collate_fn
import time


def main():
    hdf5_files = sorted(
        Path("/scratch/gpfs/EKOLEMEN/foundation_model/").glob("[0-9]*_processed.h5")
    )
    preprocessing_stats = torch.load("/scratch/gpfs/ps9551/FusionAIHub/scripts/slurm/preprocessing_stats.pt", weights_only=False)

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
            preprocessing_stats=preprocessing_stats,
        ) for f in hdf5_files]
    combined = ConcatDataset(datasets)
    dataloader = DataLoader(combined, batch_size=32, collate_fn=collate_fn,
                            num_workers=32)

    for epoch in range(10):
        epoch_start = time.time()
        for batch in dataloader:
            continue
        print(f"Epoch {epoch} / 10 took {time.time() - epoch_start} s.")

    exit(0)


if __name__ == "__main__":
    main()
