from pathlib import Path
import argparse
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader

from tokamak_foundation_model.data.data_loader import TokamakH5Dataset, collate_fn
from tokamak_foundation_model.data.utils import worker_init_fn
from tokamak_foundation_model.trainer.trainer import UnimodalTrainer
from tokamak_foundation_model.models.model_factory import (
    build_model, MODEL_REGISTRY, SIGNAL_MODEL_DEFAULTS)

from tokamak_foundation_model.utils import DefaultDrawer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    ### Settings ###
    parser = argparse.ArgumentParser(description="Train a unimodal autoencoder")
    parser.add_argument(
        "--signal", choices=list(SIGNAL_MODEL_DEFAULTS.keys()),
        default="mse",
        help="Signal name to train on"
    )
    parser.add_argument(
        "--n_fft", type=int, default=1024, help="FFT size",
    )
    parser.add_argument(
        "--hop_length", type=int, default=256, help="Hop length for STFT.",
    )
    parser.add_argument(
        "--model", choices=list(MODEL_REGISTRY.keys()), default="profile",
        help="Model type (default: auto-selected from signal)"
    )
    parser.add_argument(
        "--data_dir", type=str,
        default="C:/Users/admin/PycharmProjects/FusionAIHub/scripts/",
        help="Path to HDF5 data directory"
    )
    parser.add_argument(
        "--stats_path", type=str,
        default="C:/Users/admin/PycharmProjects/FusionAIHub/scripts/preprocessing_stats.pt",
        help="Path to preprocessing stats file"
    )
    parser.add_argument(
        "--d_model", type=int, default=512, help="Model dimension"
    )
    parser.add_argument(
        "--n_tokens", type=int, default=140,
        help="Number of latent tokens (default: use model default)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=2,
        help="Batch size (for spectrograms, each sample's C channels are processed "
             "independently, so effective batch = batch_size * C)"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loader workers"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=5e-3, help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="AdamW weight decay"
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=5,
        help="LR warmup epochs (0 to disable scheduler)"
    )
    parser.add_argument(
        "--min_lr", type=float, default=0.0, help="Minimum LR at end of cosine decay"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="runs", help="Directory for checkpoints"
    )
    parser.add_argument(
        "--num_plots", type=int, default=4,
        help="Number of reconstruction plots per epoch"
    )
    parser.add_argument(
        "--log_interval", type=int, default=1, help="Plot every N epochs"
    )
    parser.add_argument(
        "--resume", action="store_true", default=False,
        help="Resume training from checkpoint"
    )
    args = parser.parse_args()

    ### Paths ###
    signal_name = args.signal
    model_name = args.model or SIGNAL_MODEL_DEFAULTS[signal_name]
    data_dir = Path(args.data_dir)
    statistics_path = Path(args.stats_path)
    checkpoint_path = (
            Path(args.checkpoint_dir) / f"{signal_name}_{model_name}" / "checkpoint.pth"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Signal: {signal_name}, Model: {model_name}")

    ### Dataset Setup ###
    hdf5_files = sorted(data_dir.glob("*_processed.h5"))
    stats = torch.load(statistics_path)

    datasets_processed = [
        TokamakH5Dataset(
            hdf5_path=str(f),
            preprocessing_stats=stats,
            input_signals=[signal_name],
            target_signals=[signal_name],
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            prediction_mode=False,
        )
        for f in hdf5_files
    ]

    concatenated_dataset = ConcatDataset(datasets_processed)

    # Not sure if this is elegant
    sample_data = next(iter(concatenated_dataset))[signal_name]
    logger.info(f"Sample data shape: {sample_data.shape}")
    n_spatial_points = sample_data.shape[0]
    n_time_points = sample_data.shape[1]
    logger.info(f"n_spatial_points: {n_spatial_points}, n_time_points: {n_time_points}")
    
    ### Model Setup ###
    model = build_model(model_name, d_model=args.d_model, n_tokens=args.n_tokens,
                        n_channels=1, n_spatial_points=n_spatial_points,
                        n_time_points=n_time_points, kernel_size=3)

    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
    )

    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr
    )

    loss_fn = nn.L1Loss()

    dataloader = DataLoader(
        concatenated_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
        shuffle=True,
    )

    ### Training ###
    drawer = DefaultDrawer(num_plots=args.num_plots)
    trainer = UnimodalTrainer(
        epochs=args.epochs,
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        loss_fn=loss_fn,
        device=device,
        drawer=drawer,
        log_interval=args.log_interval,
    )

    if args.resume and checkpoint_path.exists():
        logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
        trainer.load_checkpoint(checkpoint_path=checkpoint_path)

    trainer.train(dataloader, modality_key=signal_name)


if __name__ == "__main__":
    main()
