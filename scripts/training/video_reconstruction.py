from pathlib import Path
import sys
repo_root = Path().resolve().parents[1]
sys.path.append(str(repo_root / "src"))
print(repo_root)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader

from tokamak_foundation_model.data.data_loader import TokamakH5Dataset, collate_fn
from tokamak_foundation_model.models.modality.video_baseline import (
    VideoBaselineEncoder, VideoBaselineDecoder, VideoBaselineAutoEncoder)
from tokamak_foundation_model.trainer.trainer import UnimodalTrainer


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


model = VideoBaselineAutoEncoder(n_tokens=32)


hdf5_files = sorted(
    Path("/scratch/gpfs/EKOLEMEN/big_d3d_data/dummy_foundation_model_data/").glob("*_processed.h5")
)
# stats = torch.load(
#     Path("C:/Users/admin/PycharmProjects/FusionAIHub/scripts/preprocessing_stats.pt")
# )

datasets_processed = [
    TokamakH5Dataset(
        hdf5_path=str(f),
        # preprocessing_stats=stats,
        input_signals=["tangtv", ],
        target_signals=["tangtv", ],
        prediction_mode=False,
    )
    for f in hdf5_files
]

concatenated_dataset = ConcatDataset(datasets_processed)

dataloader = DataLoader(
    concatenated_dataset,
    batch_size=2,
    shuffle=False,
    collate_fn=collate_fn,
    worker_init_fn=worker_init_fn
    )
x = next(iter(dataloader))
x = x.to(device)
print("x     :", tuple(x.shape))
optimizer = optim.AdamW(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
trainer = UnimodalTrainer(model, optimizer, loss_fn, device=device, epochs=10)
trainer.train(dataloader, modality_key="bolo")
