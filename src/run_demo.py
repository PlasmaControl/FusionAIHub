import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset

from tokamak_foundation_model.data.data_loader import (
    TokamakH5Dataset, collate_fn, collate_fn_prediction, compute_preprocessing_stats)
from tokamak_foundation_model.models.dummy_model import MultiModalTokamakModel, MultiModalPredictionModel
from tokamak_foundation_model.trainer.trainer import Trainer


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

print("Initializing and demonstrating custom DataLoader with updated TokamakH5Dataset")
# Use glob to find all generated HDF5 files
hdf5_files = sorted(Path(".").glob("??????_processed.h5"))

# Create TokamakH5Dataset instances for each HDF5 file
# datasets = [TokamakH5Dataset(hdf5_path=str(f)) for f in hdf5_files]
# stats = compute_preprocessing_stats(datasets, 'preprocessing_stats.pt')
stats = torch.load('preprocessing_stats.pt')
datasets_processed = [
    TokamakH5Dataset(hdf5_path=str(f), preprocessing_stats=stats) for f in hdf5_files]

# Concatenate the datasets
concatenated_dataset = ConcatDataset(datasets_processed)

print(f"Initialized ConcatDataset with {len(concatenated_dataset)} samples.")

# Initialize DataLoader
dataloader = DataLoader(concatenated_dataset, batch_size=2, shuffle=False,
                        collate_fn=collate_fn_prediction, worker_init_fn=worker_init_fn)

# Get and print the first batch from DataLoader to verify functionality
try:
    for batch in dataloader:
        inputs = batch['inputs']
        targets = batch['targets']
    print("Last batch from DataLoader:")
    print("Input data:")
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  Modality: {key}, Type: {type(value)}, Shape: {value.shape}")
        elif isinstance(value, list) and all(isinstance(i, str) for i in value):
            print(f"  Modality: {key}, Type: {type(value)}, Number of text samples: {len(value)}")
        else:
            print(f"  Modality: {key}, Type: {type(value)}")
    print("Target data:")
    for key, value in targets.items():
        if isinstance(value, torch.Tensor):
            print(f"  Modality: {key}, Type: {type(value)}, Shape: {value.shape}")
        elif isinstance(value, list) and all(isinstance(i, str) for i in value):
            print(f"  Modality: {key}, Type: {type(value)}, Number of text samples: {len(value)}")
        else:
            print(f"  Modality: {key}, Type: {type(value)}")

    print("DataLoader functionality verified successfully.")
except Exception as e:
    print(f"Error during DataLoader iteration: {e}")

# --- 3. Initialize and Demonstrate Dummy PyTorch Model with text input ---
print("\n--- 3. Initializing and demonstrating Dummy PyTorch Model with text input ---")
model = MultiModalPredictionModel()
print("MultiModalPredictionModel structure:")
print(model)

model.eval()
with torch.no_grad():
    # The batch now includes 'text' data
    output = model(batch)
print(f"Model output type: {type(output)}, shape: {output.shape}")

# --- 4. Initialize and Demonstrate Extensible PyTorch Trainer ---
print("\n--- 4. Initializing and demonstrating Extensible PyTorch Trainer ---")
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()  # Dummy loss for regression
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    device=device,
    epochs=10, # Only 1 epoch for demonstration
    batch_size=2,
    checkpoint_path="dummy_trainer_checkpoint.pth"
)
print("Trainer class initialized.")

print("Running dummy training epoch...")
# Ensure the model is in training mode before calling _train_epoch
model.train()
train_metrics = trainer.train(dataloader) # Corrected method call
print(f"  Finished dummy training epoch. Metrics: {train_metrics}")

print("Running dummy validation epoch...")
# Ensure the model is in evaluation mode before calling _validate_epoch
model.eval()
val_metrics = trainer._validate_epoch(dataloader) # Corrected method call
print(f"  Finished dummy validation epoch. Metrics: {val_metrics}")

print("\nDemonstration complete!")
