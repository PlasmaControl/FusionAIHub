from pathlib import Path
import argparse
import logging

import random
import torch
import torch.optim as optim

from tokamak_foundation_model.data.multi_file_dataset import (
    TokamakMultiFileDataset, make_dataloader)
from tokamak_foundation_model.trainer.trainer import UnimodalTrainer
from tokamak_foundation_model.models.model_factory import (
    build_model, MODEL_REGISTRY, SIGNAL_MODEL_DEFAULTS)

from tokamak_foundation_model.models.loss import MaskedMSELoss
from tokamak_foundation_model.utils import DefaultDrawer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    ### Settings ###
    parser = argparse.ArgumentParser(
        description="Train a unimodal autoencoder"
    )
    parser.add_argument(
        "--signal", choices=list(SIGNAL_MODEL_DEFAULTS.keys()),
        default="filterscopes",
        help="Signal name to train on"
    )
    parser.add_argument(
        "--n_fft", type=int, default=1024, help="FFT size",
    )
    parser.add_argument(
        "--hop_length", type=int, default=256, help="Hop length for STFT.",
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY.keys()),
        default="fast_time_series",
        help="Model type (default: auto-selected from signal)"
    )
    parser.add_argument(
        "--data_dir", type=str,
        default="/scratch/gpfs/EKOLEMEN/foundation_model/",
        help="Path to HDF5 data directory"
    )
    parser.add_argument(
        "--stats_path",
        type=str,
        default="/scratch/gpfs/ps9551/FusionAIHub/scripts/slurm/preprocessing_stats.pt",
        help="Path to preprocessing stats file"
    )
    parser.add_argument(
        "--d_model", type=int, default=512, help="Model dimension"
    )
    parser.add_argument(
        "--n_tokens", type=int, default=16,
        help="Number of latent tokens (default: 16)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size (for spectrograms, each sample's C channels are "
             "processed independently, so effective batch = batch_size * C)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loader workers"
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=4,
        help="Batches to prefetch per worker"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=5e-3, help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="AdamW weight decay"
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=5,
        help="LR warmup epochs (0 to disable scheduler)"
    )
    parser.add_argument(
        "--min_lr", type=float, default=0.0,
        help="Minimum LR at end of cosine decay"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str,
        default="/scratch/gpfs/ps9551/FusionAIHub/scripts/slurm/runs",
        help="Directory for checkpoints"
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
    random.seed(42)
    n = len(hdf5_files)
    n_val = int(0.1 * n)
    n_test = int(0.1 * n)

    train_paths = hdf5_files[n_val + n_test:]
    val_paths   = hdf5_files[:n_val]
    test_paths  = hdf5_files[n_val:n_val + n_test]

    stats = torch.load(statistics_path, weights_only=False)

    shared_kwargs = dict(
        preprocessing_stats=stats,
        input_signals=[signal_name],
        target_signals=[signal_name],
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        prediction_mode=False,
    )

    train_dataset = TokamakMultiFileDataset(
        train_paths,
        lengths_cache_path="lengths_train.pt",
        **shared_kwargs
    )
    validation_dataset = TokamakMultiFileDataset(
        val_paths,
        lengths_cache_path="lengths_validation.pt",
        **shared_kwargs
    )
    test_dataset = TokamakMultiFileDataset(
        test_paths,
        lengths_cache_path="lengths_test.pt",
        **shared_kwargs
    )

    # Infer spatial and temporal dimensions from first sample
    sample_data = next(iter(train_dataset))[signal_name]
    n_channels = sample_data.shape[0]
    logger.info(f"Sample data shape: {sample_data.shape}, "
                f"n_channels: {n_channels}"
                )

    ### Model Setup ###
    model = build_model(
        model_name,
        d_model=args.d_model,
        n_tokens=args.n_tokens,
        n_channels=n_channels,
        kernel_size=3
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    if args.warmup_epochs > 0:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, end_factor=1.0,
            total_iters=args.warmup_epochs,
        )
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs - args.warmup_epochs,
            eta_min=args.min_lr,
        )
        lr_scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[args.warmup_epochs],
        )
    else:
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.min_lr,
        )

    loss_fn = MaskedMSELoss()

    train_dataloader = make_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
    )

    validation_dataloader = make_dataloader(
        validation_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
    )

    ### Training ###
    drawer = DefaultDrawer()
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
        train_dataloader,
        validation_dataloader,
        modality_key=signal_name,
    )


if __name__ == "__main__":
    main()
