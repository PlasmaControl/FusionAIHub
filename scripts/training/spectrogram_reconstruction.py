from pathlib import Path
import argparse
import logging
import random

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
        default="co2",
        help="Signal name to train on"
    )
    parser.add_argument(
        "--n_fft", type=int, default=1024, help="FFT size",
    )
    parser.add_argument(
        "--hop_length", type=int, default=256, help="Hop length for STFT.",
    )
    parser.add_argument(
        "--model", choices=list(MODEL_REGISTRY.keys()), default=None,
        help="Model type (default: auto-selected from signal)"
    )
    parser.add_argument(
        "--data_dir", type=str,
        default="/scratch/gpfs/EKOLEMEN/foundation_model",
        help="Path to HDF5 data directory"
    )
    parser.add_argument(
        "--stats_path", type=str,
        default="data/preprocessing_stats.pt",
        help="Path to preprocessing stats file"
    )
    parser.add_argument(
        "--d_model", type=int, default=512, help="Model dimension"
    )
    parser.add_argument(
        "--n_tokens", type=int, default=0,
        help="Number of latent tokens (default: use model default)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=2,
        help="Batch size"
    )
    parser.add_argument(
        "--num_workers", type=int, default=1, help="Number of data loader workers"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=5e-3, help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-3, help="AdamW weight decay"
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=5,
        help="LR warmup epochs (cosine scheduler only)"
    )
    parser.add_argument(
        "--scheduler", type=str, default="cosine",
        choices=["cosine", "none"],
        help="LR scheduler: 'cosine' (warmup + cosine decay) or 'none' (flat LR)"
    )
    parser.add_argument(
        "--min_lr", type=float, default=0.0, help="Minimum LR at end of cosine decay"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="runs", help="Directory for checkpoints"
    )
    parser.add_argument(
        "--log_interval", type=int, default=1, help="Plot every N epochs"
    )
    parser.add_argument(
        "--resume", action="store_true", default=False,
        help="Resume training from checkpoint"
    )
    parser.add_argument(
        "--shot_min", type=int, default=None,
        help="Inclusive lower bound on shot number (filters HDF5 files by name)"
    )
    parser.add_argument(
        "--shot_max", type=int, default=None,
        help="Inclusive upper bound on shot number (filters HDF5 files by name)"
    )
    parser.add_argument(
        "--val_split", type=float, default=0.1,
        help="Fraction of shots to hold out for validation (split by shot)"
    )
    parser.add_argument(
        "--grad_clip", type=float, default=1.0,
        help="Max gradient norm for clipping (0 = disabled)"
    )
    parser.add_argument(
        "--preprocessing", type=str, default=None,
        choices=["log_standardize", "log", "standardize", "normalize", "none"],
        help="Override preprocessing method for the signal (default: use signal's built-in)"
    )
    # Channel-AST specific
    parser.add_argument(
        "--frame_width", type=int, default=2,
        help="Time steps per frame token (spectrogram_channel_ast)"
    )
    parser.add_argument(
        "--time_conv_kernel", type=int, default=7,
        help="Temporal ConvNeXt kernel size (spectrogram_channel_ast)"
    )
    parser.add_argument(
        "--n_heads", type=int, default=4,
        help="Attention heads (spectrogram_channel_ast)"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1,
        help="Dropout rate (spectrogram_channel_ast)"
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

    if args.shot_min is not None or args.shot_max is not None:
        lo = args.shot_min if args.shot_min is not None else 0
        hi = args.shot_max if args.shot_max is not None else float("inf")

        def _shot_num(p: Path):
            try:
                return int(p.stem.split("_")[0])
            except ValueError:
                return None

        hdf5_files = [f for f in hdf5_files if (n := _shot_num(f)) is not None and lo <= n <= hi]
        logger.info(f"Shot filter [{lo}, {hi}]: {len(hdf5_files)} files retained")

    logger.info(f"Found {len(hdf5_files)} shot files")

    # Override preprocessing method if requested
    if args.preprocessing:
        for cfg in TokamakH5Dataset.SIGNAL_CONFIGS:
            if cfg.name == signal_name:
                cfg.preprocess.method = args.preprocessing
                logger.info(f"Preprocessing override: {signal_name} -> {args.preprocessing}")
                break

    stats = torch.load(statistics_path, weights_only=False)

    # Shuffle shot list before splitting so val is a random draw
    random.seed(42)
    random.shuffle(hdf5_files)

    n_val = max(1, int(len(hdf5_files) * args.val_split))
    train_files = hdf5_files[:-n_val]
    val_files = hdf5_files[-n_val:]
    logger.info(f"Train shots: {len(train_files)}, Val shots: {len(val_files)}")

    def _make_datasets(files):
        return [
            TokamakH5Dataset(
                hdf5_path=str(f),
                preprocessing_stats=stats,
                input_signals=[signal_name],
                target_signals=[signal_name],
                n_fft=args.n_fft,
                hop_length=args.hop_length,
                prediction_mode=False,
            )
            for f in files
        ]

    train_dataset = ConcatDataset(_make_datasets(train_files))
    val_dataset = ConcatDataset(_make_datasets(val_files))

    sample_data = next(iter(train_dataset))[signal_name]
    n_channels = sample_data.shape[0]
    logger.info(f"Sample data shape: {sample_data.shape}, n_channels: {n_channels}")

    ### Model Setup ###
    extra_kwargs = {}
    if model_name == "spectrogram_channel_ast":
        extra_kwargs["freq_bins"] = sample_data.shape[1]
        extra_kwargs["frame_width"] = args.frame_width
        extra_kwargs["n_heads"] = args.n_heads
        extra_kwargs["dropout"] = args.dropout
        extra_kwargs["time_conv_kernel"] = args.time_conv_kernel

    model = build_model(
        model_name, args.d_model, args.n_tokens, n_channels, **extra_kwargs
    )
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    if args.scheduler == "none":
        lr_scheduler = None
    elif args.warmup_epochs > 0:
        warmup = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, total_iters=args.warmup_epochs
        )
        cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=args.min_lr
        )
        lr_scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_epochs]
        )
    else:
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_lr
        )

    loss_fn = nn.L1Loss()

    dataloader_kwargs = dict(
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2,
        pin_memory=False,
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **dataloader_kwargs,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **dataloader_kwargs,
    )

    ### Training ###
    drawer = DefaultDrawer()
    trainer = UnimodalTrainer(
        epochs=args.epochs,
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        loss_fn=loss_fn,
        drawer=drawer,
        log_interval=args.log_interval,
        grad_clip=args.grad_clip,
    )

    if args.resume and checkpoint_path.exists():
        logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
        trainer.load_checkpoint(checkpoint_path=checkpoint_path)

    trainer.fit(dataloader, val_dataloader=val_dataloader, modality_key=signal_name)


if __name__ == "__main__":
    main()
