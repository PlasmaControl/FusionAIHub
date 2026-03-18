from pathlib import Path
import sys
repo_root = Path().resolve().parents[1]
sys.path.append(str(repo_root / "src"))
print(repo_root)

import argparse
import logging
import random
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader
from torchinfo import summary

from tokamak_foundation_model.data.data_loader import TokamakH5Dataset, collate_fn
from tokamak_foundation_model.data.utils import worker_init_fn
from tokamak_foundation_model.trainer.trainer import UnimodalTrainer
from tokamak_foundation_model.utils import DefaultDrawer
from tokamak_foundation_model.models.loss import SparseVideoWeightedMSE
from tokamak_foundation_model.utils.distributed import DistributedManager

from tokamak_foundation_model.models.model_factory import (
    build_model, MODEL_REGISTRY, SIGNAL_MODEL_DEFAULTS)


# TODO: Add ddp support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_weights(m):
    if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
        # Kaiming normal is great for Leaky ReLU
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def build_dataloader(hdf5_files: list, signal: str, batch_size: int,
                     num_workers: int, shuffle: bool) -> DataLoader:

    datasets = [
        TokamakH5Dataset(
            hdf5_path=str(f),
            input_signals=[signal],
            target_signals=[signal],
            prediction_mode=False,
        )
        for f in hdf5_files
    ]
    dataset = ConcatDataset(datasets)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
        shuffle=shuffle,
    )
    return dataloader
def main():
    parser = argparse.ArgumentParser(description="Train a video autoencoder (template-aligned)")

    # Data / signal
    parser.add_argument("--signal", type=str, default="tangtv",
                        help="Key/name of the video signal inside each HDF5 file")
    parser.add_argument("--data_dir", type=str,
                        default=None, # /scra
                        help="Path to HDF5 data directory")
    parser.add_argument("--file_glob", type=str, default="*_processed.h5",
                        help="Glob pattern for HDF5 files inside data_dir")
    parser.add_argument("--shuffle", action="store_true", default=True,
                        help="Shuffle training dataset")
    parser.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY.keys()),
        default="video",
        help="Model type (default: auto-selected from signal)"
    )
    parser.add_argument(
        "--stats_path",
        type=str,
        default="/scratch/gpfs/ps9551/FusionAIHub/scripts/slurm/preprocessing_stats.pt",
        help="Path to preprocessing stats file"
    )
    parser.add_argument(
        "--n_fft", type=int, default=1024, help="FFT size",
    )
    parser.add_argument(
        "--hop_length", type=int, default=256, help="Hop length for STFT.",
    )
    # Video chunking / target geometry
    parser.add_argument("--clip_seconds", type=float, default=0.5,
                        help="Clip duration in seconds (0.5s -> 25 frames at 50fps)")
    parser.add_argument("--target_fps", type=float, default=50.0,
                        help="Target FPS (used to compute clip length)")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Spatial size (H=W=image_size)")
    parser.add_argument("--n_channels", type=int, default=1,
                        help="Number of channels")

    # Latent / model
    parser.add_argument("--n_tokens", type=int, default=128,
                        help="Latent tokens N (latent is N x 512)")
    parser.add_argument("--d_model", type=int, default=512,
                        help="Token dimension (keep 512 to match the design)")

    # Optimization
    parser.add_argument("--batch_size", type=int, default=16) # 32 is also good
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10) # 500 epochs?
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--min_lr", type=float, default=0.0,
                        help="Minimum LR at end of cosine decay")
    # Logging / checkpoints
    parser.add_argument("--checkpoint_dir", type=str, default="runs",
                        help="Directory for checkpoints")
    parser.add_argument("--num_plots", type=int, default=0,
                        help="Number of reconstruction plots per epoch (0 to disable)")
    parser.add_argument("--log_interval", type=int, default=1,
                        help="Log/plot every N epochs")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume training from checkpoint if it exists")

    args = parser.parse_args()

    ### Paths ###
    signal_name = args.signal
    model_name = args.model or SIGNAL_MODEL_DEFAULTS[signal_name]
    data_dir = Path(args.data_dir)
    statistics_path = Path(args.stats_path)
    datenow = datetime.now().strftime('%Y%m%d%H%M')
    checkpoint_fld = f"{signal_name}-{model_name}-lr{args.lr}-ntk{args.n_tokens}-ep{args.epochs}-{datenow}"
    logger.info(f"checkpoint folder: {checkpoint_fld}")
    checkpoint_path = (
            Path(args.checkpoint_dir) / f"{checkpoint_fld}" / "checkpoint.pth"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Signal: {signal_name}, Model: {model_name}")

    ### Dataset Setup ###
    with open('/scratch/gpfs/aj17/runs/tangtv_flist.txt', 'r') as file:
        hdf5_files = [line.strip() for line in file]
    # hdf5_files = sorted(hdf5_files)

    # hdf5_files = sorted(data_dir.glob("*_processed.h5"))

    random.seed(42)
    n = len(hdf5_files)
    n_val = int(.1 * n)
    n_test = int(.1 * n)
    train_paths = hdf5_files[n_val + n_test:]
    val_paths   = hdf5_files[:n_val]
    test_paths  = hdf5_files[n_val:n_val + n_test]
    logger.info(f"Train shots: {len(train_paths)}, Valid shots: {len(val_paths)}, Test shots: {len(test_paths)}")       

    stats = torch.load(statistics_path, weights_only=False)

    shared_kwargs = dict(
        preprocessing_stats=stats,
        input_signals=[signal_name],
        target_signals=[signal_name],
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        prediction_mode=False,
    )


    train_dataset = build_dataloader(
        hdf5_files=train_paths,
        signal=signal_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=args.shuffle,
    )

    validation_dataset = build_dataloader(
        hdf5_files=val_paths,
        signal=signal_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=args.shuffle,
    )

    test_dataset = build_dataloader(
        hdf5_files=test_paths,
        signal=signal_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=args.shuffle,
    )

    # Not sure if this is elegant
    sample_data = next(iter(train_dataset))[signal_name]
    n_channels = sample_data.shape[1]
 
    logger.info(f"Sample data shape: {sample_data.shape}, n_channels: {n_channels}")       

    ### Model Setup ###
    model = build_model(model_name, d_model=args.d_model, n_tokens=args.n_tokens,
                        n_channels=n_channels).to(device)
    model.apply(init_weights)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    summary(
        model,
        input_size=(1,args.n_channels, 25, 128, 128),  # batch=1
        col_names=("input_size", "output_size", "num_params"),
        )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    # loss_fn = nn.MSELoss()
    loss_fn = SparseVideoWeightedMSE(l1l2='l2')

    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr
    )

    ### Training ###
    drawer = DefaultDrawer(plot_channel=0)
    trainer = UnimodalTrainer(
        epochs=args.epochs,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        checkpoint_path=checkpoint_path,
        drawer=drawer,
        log_interval=args.log_interval,
    )

    if args.resume and checkpoint_path.exists():
        logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
        trainer.load_checkpoint(checkpoint_path=checkpoint_path)

    trainer.fit(
        train_dataset,
        validation_dataset,
        modality_key=signal_name)


if __name__ == "__main__":
    main()