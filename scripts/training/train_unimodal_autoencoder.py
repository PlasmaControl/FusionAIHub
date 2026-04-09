from pathlib import Path
import argparse
import json
import logging

import torch
import torch.nn as nn
from torchvision.transforms import GaussianBlur

import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader

from torch.utils.data.distributed import DistributedSampler

from tokamak_foundation_model.data.data_loader import TokamakH5Dataset, collate_fn
from tokamak_foundation_model.data.utils import worker_init_fn
from tokamak_foundation_model.trainer.trainer import UnimodalTrainer
from tokamak_foundation_model.models.model_factory import (
    build_model, MODEL_REGISTRY, SIGNAL_MODEL_DEFAULTS)
from tokamak_foundation_model.utils.distributed import DistributedManager
from tokamak_foundation_model.utils import DefaultDrawer, NullDrawer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():

    ### Settings ###
    parser = argparse.ArgumentParser(description="Train a unimodal autoencoder")
    parser.add_argument(
        "--signal", required=True, choices=list(SIGNAL_MODEL_DEFAULTS.keys()),
        help="Signal name to train on"
    )
    parser.add_argument(
        "--n_fft", type=int, default=1024, help="FFT size",
    )
    parser.add_argument(
        "--hop_length", type=int, default=256, help="Hop length for STFT.",
    )
    parser.add_argument(
        "--chunk_duration_s", type=float, default=0.5,
        help="Duration of each data chunk in seconds",
    )
    parser.add_argument(
        "--model", choices=list(MODEL_REGISTRY.keys()), default=None,
        help="Model type (default: auto-selected from signal)"
    )
    parser.add_argument(
        "--data_dir", type=str,
        default="/scratch/gpfs/EKOLEMEN/big_d3d_data/dummy_foundation_model_data",
        help="Path to HDF5 data directory"
    )
    parser.add_argument(
        "--stats_path", type=str, default="data/preprocessing_stats.pt",
        help="Path to preprocessing stats file"
    )
    parser.add_argument(
        "--d_model", type=int, default=64, help="Model dimension"
    )
    parser.add_argument(
        "--n_tokens", type=int, default=None,
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
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="AdamW weight decay"
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=5,
        help="LR warmup epochs (0 to disable warmup)"
    )
    parser.add_argument(
        "--min_lr", type=float, default=0.0, help="Minimum LR at end of cosine decay"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="runs", help="Run directory for checkpoints (used directly, e.g. runs/ece_spectrogram)"
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
    parser.add_argument(
        "--model_kwargs", type=str, default="{}",
        help="JSON string of extra model constructor kwargs (e.g., '{\"n_layers\": 7}')"
    )
    parser.add_argument(
        "--plot_channel", type=int, default=None,
        help="Channel index to visualize in reconstruction plots (default: middle channel)"
    )
    parser.add_argument(
        "--plot_indices", type=int, nargs="+", default=None,
        help="Dataset indices to visualize (default: first num_plots samples)"
    )
    parser.add_argument(
        "--val_split", type=float, default=0.0,
        help="Fraction of data for validation (0.0 = no validation)"
    )
    parser.add_argument(
        "--use_wandb", action="store_true", default=False,
        help="Enable wandb offline logging"
    )
    parser.add_argument(
        "--use_metrics", action="store_true", default=False,
        help="Enable PSNR/SSIM metric tracking"
    )
    parser.add_argument(
        "--patience", type=int, default=0,
        help="Early stopping patience (0 = disabled)"
    )
    parser.add_argument(
        "--use_gated_target", action="store_true", default=False,
        help="Train against spectral-gated target instead of raw target"
    )
    args = parser.parse_args()

    ### Distributed Setup ###
    dm = DistributedManager()

    log_level = logging.INFO if dm.is_main else logging.WARNING
    logging.basicConfig(level=log_level)

    ### Paths ###
    signal_name = args.signal
    model_name = args.model or SIGNAL_MODEL_DEFAULTS[signal_name]
    data_dir = Path(args.data_dir)
    statistics_path = Path(args.stats_path)
    checkpoint_path = Path(args.checkpoint_dir) / "checkpoint.pth"
    if dm.is_main:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    dm.barrier()

    logger.info(f"Signal: {signal_name}, Model: {model_name}")

    ### Dataset Setup ###
    hdf5_files = sorted(data_dir.glob("*.h5"))
    logger.info(f"Found {len(hdf5_files)} Shots")

    stats = torch.load(statistics_path, weights_only=False)

    ### Train/Val Split (file-level) ###
    val_dataset = None
    if args.val_split > 0:
        rng = torch.Generator().manual_seed(42)
        n_val_files = max(1, int(len(hdf5_files) * args.val_split))
        perm = torch.randperm(len(hdf5_files), generator=rng)
        val_indices = perm[:n_val_files].tolist()
        train_indices = perm[n_val_files:].tolist()
        train_files = [hdf5_files[i] for i in train_indices]
        val_files = [hdf5_files[i] for i in val_indices]
    else:
        train_files = hdf5_files
        val_files = []

    def make_dataset(files):
        datasets = []
        for f in files:
            try:
                ds = TokamakH5Dataset(
                    hdf5_path=str(f),
                    preprocessing_stats=stats,
                    input_signals=[signal_name],
                    target_signals=[signal_name],
                    chunk_duration_s=args.chunk_duration_s,
                    n_fft=args.n_fft,
                    hop_length=args.hop_length,
                    prediction_mode=False,
                )
                datasets.append(ds)
            except OSError:
                logger.warning(f"Skipping corrupt file: {f}")
        return ConcatDataset(datasets)

    train_dataset = make_dataset(train_files)
    if val_files:
        val_dataset = make_dataset(val_files)

    logger.info(f"Train dataset length: {len(train_dataset)}")
    if val_dataset is not None:
        logger.info(f"Val dataset length: {len(val_dataset)}")
    logger.info(f"Train/Val file split: {len(train_files)}/{len(val_files)}")

    sample_data = next(iter(train_dataset))[signal_name]
    n_channels = sample_data.shape[0]
    logger.info(f"Sample data shape: {sample_data.shape}, n_channels: {n_channels}")

    ### Model Setup ###
    model_kwargs = json.loads(args.model_kwargs)
    model = build_model(model_name, n_channels=n_channels, d_model=args.d_model,
                        n_tokens=args.n_tokens, **model_kwargs).to(dm.device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    loss_fn = nn.L1Loss()

    if args.warmup_epochs > 0:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=args.min_lr
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_lr
        )

    train_sampler = None
    if dm.distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=dm.world_size,
            rank=dm.rank,
            shuffle=True,
        )

    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
    )

    ### Validation DataLoader ###
    val_dataloader = None
    val_sampler = None
    if val_dataset is not None:
        if dm.distributed:
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=dm.world_size,
                rank=dm.rank,
                shuffle=False,
            )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            worker_init_fn=worker_init_fn,
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
            pin_memory=True,
            shuffle=False,
            sampler=val_sampler,
        )

    ### Metrics ###
    metrics = None
    if args.use_metrics:
        from tokamak_foundation_model.utils.metrics import PSNR, SSIM
        metrics = [PSNR(), SSIM()]

    ### wandb ###
    if args.use_wandb and dm.is_main:
        import wandb
        wandb.init(mode="offline", project="faith-unimodal", config=vars(args))

    ### Training ###
    if dm.is_main:
        drawer = DefaultDrawer(plot_channel=args.plot_channel)
    else:
        drawer = NullDrawer()

    trainer = UnimodalTrainer(
        epochs=args.epochs,
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        drawer=drawer,
        scheduler=scheduler,
        log_interval=args.log_interval,
        distributed_manager=dm,
        metrics=metrics,
    )

    if args.resume and checkpoint_path.exists():
        logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
        trainer.load_checkpoint(checkpoint_path=checkpoint_path)

    trainer.fit(
        train_dataloader=dataloader,
        val_dataloader=val_dataloader,
        modality_key=signal_name,
        train_sampler=train_sampler,
    )

    ### Save config.json alongside checkpoint ###
    if dm.is_main:
        config = {
            "signal_name": signal_name,
            "model_type": model_name,
            "d_model": args.d_model,
            "n_tokens": args.n_tokens,
            "n_channels": n_channels,
            "model_kwargs": model_kwargs,
        }
        config_path = checkpoint_path.parent / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved config to {config_path}")


if __name__ == "__main__":
    main()
