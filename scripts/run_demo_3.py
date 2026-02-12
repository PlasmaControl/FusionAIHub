import numpy as np
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from dataclasses import dataclass

from tokamak_foundation_model.data.data_loader import (
    TokamakH5Dataset, collate_fn_prediction, compute_preprocessing_stats)
from tokamak_foundation_model.models.dummy_model_2 import (
    Prediction4FusionModel, DictMSELoss, DEFAULT_MODALITY_CONFIGS)
from tokamak_foundation_model.trainer.trainer import Trainer
from tokamak_foundation_model.models.modality import PROCESSOR_REGISTRY


def worker_init_fn(worker_id):
    """Each worker needs to open its own file handle."""
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        if hasattr(dataset, 'datasets'):  # ConcatDataset
            for ds in dataset.datasets:
                ds.h5_file = None
                ds._open_hdf5()
        else:
            dataset.h5_file = None
            dataset._open_hdf5()


# --- 1. Load data ---
print("--- 1. Loading data ---")
hdf5_files = sorted(
    Path("/scratch/gpfs/EKOLEMEN/big_d3d_data/dummy_foundation_model_data").glob("*_processed.h5")
)
stats = torch.load('data/preprocessing_stats.pt')

all_input_signals = [
    "mhr", "ece", "co2",
    "gas", "ech", "pin", "tin",
    "d_alpha", "mse", "ts_core_density",
    "bolo", "irtv", "tangtv",
    "text",
]


datasets_processed = [
    TokamakH5Dataset(
        hdf5_path=str(f),
        preprocessing_stats=stats,
        input_signals=all_input_signals,
    ) for f in hdf5_files]

concatenated_dataset = ConcatDataset(datasets_processed)
print(f"ConcatDataset with {len(concatenated_dataset)} samples.")

dataloader = DataLoader(
    concatenated_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=collate_fn_prediction,
    worker_init_fn=worker_init_fn,
)


# --- 2. Infer target configs from a sample batch ---
print("\n--- 2. Inferring target configs from sample batch ---")
batch = next(iter(dataloader))
print("Input keys:", list(batch['inputs'].keys()))
print("Target keys:", list(batch['targets'].keys()))

target_configs = {}
for name, tensor in batch['targets'].items():
    # targets are (B, C, T)
    n_channels = tensor.shape[1]
    n_frames = tensor.shape[-1]
    target_configs[name] = (n_channels, n_frames)
    print(f"  target '{name}': channels={n_channels}, frames={n_frames}")


# --- 3. Build model ---
print("\n--- 3. Building Prediction4FusionModel ---")

@dataclass
class ModalityConfig:
    name: str
    processor_type: str
    group: str | None = None
    embed_dim: int = 64

encoder_modalities = [
    ModalityConfig("mhr", "spectrogram"),
    ModalityConfig("ece", "spectrogram"),
    ModalityConfig("co2", "spectrogram"),
    ModalityConfig("gas", "timeseries", group="actuators"),
    ModalityConfig("ech", "timeseries", group="actuators"),
    ModalityConfig("pin", "timeseries", group="actuators"),
    ModalityConfig("tin", "timeseries", group="actuators"),
    ModalityConfig("d_alpha", "fast_timeseries"),
    ModalityConfig("mse", "timeseries", group="diagnostics"),
    ModalityConfig("ts_core_density", "timeseries", group="diagnostics"),
    ModalityConfig("tangtv", "video"),
]

decoder_modalities = [
    ModalityConfig("d_alpha", "fast_timeseries"),
    ModalityConfig("mse", "timeseries", group="diagnostics"),
    ModalityConfig("ts_core_density", "timeseries", group="diagnostics"),
]

encoder_embeddings = {}
decoder_embeddings = {}
global_embeddings = {}

model = Prediction4FusionModel(
    modality_configs=DEFAULT_MODALITY_CONFIGS,
    feature_dim=64,
    num_heads=4,
    target_configs=target_configs,
)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# --- 4. Forward pass test ---
print("\n--- 4. Forward pass test (no_grad) ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

# Move batch to device
inputs_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
              for k, v in batch['inputs'].items()}

model.eval()
with torch.no_grad():
    output = model(inputs_dev)

print("Prediction shapes:")
for k, v in output.items():
    target_shape = batch['targets'][k].shape
    print(f"  {k}: pred={v.shape}, target={target_shape}")

# --- 5. Training test (1 epoch) ---
print("\n--- 5. Training test (1 epoch) ---")
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = DictMSELoss()

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    device=device,
    epochs=1,
    batch_size=1,
    checkpoint_path="dummy_trainer_checkpoint.pth",
)

model.train()
train_loss = trainer.train(dataloader)
print(f"Training complete. Final metrics: {train_loss}")

print("\nDemo complete!")
