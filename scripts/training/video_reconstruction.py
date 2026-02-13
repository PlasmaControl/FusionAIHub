from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader

from tokamak_foundation_model.data.data_loader import TokamakH5Dataset, collate_fn
from tokamak_foundation_model.models.modality.fast_time_series_baseline import (
    TimeSeriesEncoder, TimeSeriesDecoder)
from tokamak_foundation_model.trainer.trainer import UnimodalTrainer


class DummyModel(torch.nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.encoder = TimeSeriesEncoder(
            kernel_size=11, n_channels=8, input_length=5000, d_model=512,
            n_output_tokens=100)
        self.decoder = TimeSeriesDecoder(
            kernel_size=11, n_channels=8, input_length=5000, d_model=512,
            n_input_tokens=100)

    def forward(self, x):
        x_encoded = self.encoder(x)
        return self.decoder(x_encoded)


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


model = DummyModel()


hdf5_files = sorted(
    Path("C:/Users/admin/PycharmProjects/FusionAIHub/scripts/").glob("*_processed.h5")
)
stats = torch.load(
    Path("C:/Users/admin/PycharmProjects/FusionAIHub/scripts/preprocessing_stats.pt")
)

datasets_processed = [
    TokamakH5Dataset(
        hdf5_path=str(f),
        preprocessing_stats=stats,
        input_signals=["pin", ],
        target_signals=["pin", ],
        prediction_mode=False,
    )
    for f in hdf5_files
]

concatenated_dataset = ConcatDataset(datasets_processed)

dataloader = DataLoader(
    concatenated_dataset,
    batch_size=8,
    shuffle=False,
    collate_fn=collate_fn,
    worker_init_fn=worker_init_fn
    )

optimizer = optim.AdamW(model.parameters(), lr=0.005)
loss_fn = nn.MSELoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
trainer = UnimodalTrainer(model, optimizer, loss_fn, device=device, epochs=50)
trainer.train(dataloader, val_dataloader=dataloader, modality_key="pin")
