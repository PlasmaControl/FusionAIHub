from pathlib import Path
import argparse
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader

from tokamak_foundation_model.data.data_loader import TokamakH5Dataset, collate_fn
from tokamak_foundation_model.trainer.trainer import UnimodalTrainer

from tokamak_foundation_model.utils import DefaultDrawer
from tokamak_foundation_model.models.modality import (
    SlowTimeSeriesAutoEncoder,
    FastTimeSeriesAutoEncoder,
    SpatialProfileAutoEncoder,
    SpectrogramAutoEncoder,
    VideoAutoEncoder,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

### Signal-to-model default mapping ###

SIGNAL_MODEL_DEFAULTS = {
    "mhr": "spectrogram", 
    "ece": "spectrogram", 
    "co2": "spectrogram",
    "d_alpha": "fast_time_series", 
    "gas": "fast_time_series", 
    "ech": "fast_time_series",
    "pin": "fast_time_series", 
    "tin": "fast_time_series",
    "mse": "profile", 
    "ts_core_density": "profile",
    "bolo": "video", 
    "irtv": "video", 
    "tangtv": "video",
}

MODEL_REGISTRY = {
    "fast_time_series": FastTimeSeriesAutoEncoder,
    "slow_time_series": SlowTimeSeriesAutoEncoder,
    "profile": SpatialProfileAutoEncoder,
    "spectrogram": SpectrogramAutoEncoder,
    "video": VideoAutoEncoder,
}


def build_model(model_name, n_channels, d_model, n_tokens):
    """Build the appropriate autoencoder.

    All autoencoders share the same interface: (n_channels, d_model, n_tokens).
    """
    cls = MODEL_REGISTRY[model_name]
    kwargs = dict(n_channels=n_channels, d_model=d_model)
    if n_tokens is not None:
        kwargs["n_tokens"] = n_tokens
    return cls(**kwargs)


# TODO: Move to data loader
def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        if hasattr(dataset, 'datasets'):
            for ds in dataset.datasets:
                ds.h5_file = None
                ds._open_hdf5()
        else:
            dataset.h5_file = None
            dataset._open_hdf5()


def main():

    ### Settings ###
    parser = argparse.ArgumentParser(description="Train a unimodal autoencoder")
    parser.add_argument("--signal", required=True, choices=list(SIGNAL_MODEL_DEFAULTS.keys()),
                        help="Signal name to train on")
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), default=None,
                        help="Model type (default: auto-selected from signal)")
    parser.add_argument("--data_dir", type=str,
                        default="/scratch/gpfs/EKOLEMEN/big_d3d_data/dummy_foundation_model_data",
                        help="Path to HDF5 data directory")
    parser.add_argument("--stats_path", type=str, default="data/preprocessing_stats.pt",
                        help="Path to preprocessing stats file")
    parser.add_argument("--d_model", type=int, default=64, help="Model dimension")
    parser.add_argument("--n_tokens", type=int, default=None,
                        help="Number of latent tokens (default: use model default)")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size (for spectrograms, each sample's C channels are "
                             "processed independently, so effective batch = batch_size * C)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.05,
                        help="AdamW weight decay (FCMAE default: 0.05)")
    parser.add_argument("--warmup_epochs", type=int, default=5,
                        help="LR warmup epochs (0 to disable scheduler)")
    parser.add_argument("--min_lr", type=float, default=0.0,
                        help="Minimum LR at end of cosine decay")
    parser.add_argument("--checkpoint_dir", type=str, default="runs",
                        help="Directory for checkpoints")
    parser.add_argument("--num_plots", type=int, default=4,
                        help="Number of reconstruction plots per epoch")
    parser.add_argument("--log_interval", type=int, default=1,
                        help="Plot every N epochs")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume training from checkpoint")
    args = parser.parse_args()

    ### Paths ###
    signal_name = args.signal
    model_name = args.model or SIGNAL_MODEL_DEFAULTS[signal_name]
    data_dir = Path(args.data_dir)
    statistics_path = Path(args.stats_path)
    checkpoint_path = Path(args.checkpoint_dir) / f"{signal_name}_{model_name}" / "checkpoint.pth"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Signal: {signal_name}, Model: {model_name}")

    ### Dataset Setup ###
    hdf5_files = sorted(data_dir.glob("*.h5"))
    stats = torch.load(statistics_path)

    datasets_processed = [
        TokamakH5Dataset(
            hdf5_path=str(f),
            preprocessing_stats=stats,
            input_signals=[signal_name],
            target_signals=[signal_name],
            prediction_mode=False,
        )
        for f in hdf5_files
    ]

    concatenated_dataset = ConcatDataset(datasets_processed)

    sample_data = next(iter(concatenated_dataset))[signal_name]
    n_channels = sample_data.shape[0]
    logger.info(f"Sample data shape: {sample_data.shape}, n_channels: {n_channels}")

    ### Model Setup ###
    model = build_model(model_name, n_channels, args.d_model, args.n_tokens).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    # FCMAE-style optimizer: AdamW with betas=(0.9, 0.95)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.L1Loss()

    dataloader = DataLoader(
        concatenated_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
        num_workers=args.num_workers,
        persistent_workers=True,
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
