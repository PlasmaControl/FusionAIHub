from pathlib import Path
import sys
repo_root = Path().resolve().parents[0]
sys.path.append(str(repo_root / "src"))
print(repo_root)

import argparse
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader

from tokamak_foundation_model.data.data_loader import TokamakH5Dataset, collate_fn
from tokamak_foundation_model.data.utils import worker_init_fn
from tokamak_foundation_model.trainer.trainer import UnimodalTrainer
from tokamak_foundation_model.utils import DefaultDrawer

from tokamak_foundation_model.models.modality import video_baseline

# TODO: Add ddp support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_dataloader(data_dir: Path, file_glob: str, signal: str, batch_size: int,
                     num_workers: int, shuffle: bool) -> DataLoader:
    hdf5_files = sorted(data_dir.glob(file_glob))
    if len(hdf5_files) == 0:
        raise FileNotFoundError(f"No HDF5 files matched: {data_dir}/{file_glob}")

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
    parser.add_argument("--signal", type=str, default="bolo",
                        help="Key/name of the video signal inside each HDF5 file")
    parser.add_argument("--data_dir", type=str,
                        default="/scratch/gpfs/EKOLEMEN/big_d3d_data/dummy_foundation_model_data/",
                        help="Path to HDF5 data directory")
    parser.add_argument("--file_glob", type=str, default="*_processed.h5",
                        help="Glob pattern for HDF5 files inside data_dir")
    parser.add_argument("--shuffle", action="store_true", default=True,
                        help="Shuffle training dataset")

    # Video chunking / target geometry
    parser.add_argument("--clip_seconds", type=float, default=0.5,
                        help="Clip duration in seconds (0.5s -> 25 frames at 50fps)")
    parser.add_argument("--target_fps", type=float, default=50.0,
                        help="Target FPS (used to compute clip length)")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Spatial size (H=W=image_size)")

    # Latent / model
    parser.add_argument("--n_tokens", type=int, default=32,
                        help="Latent tokens N (latent is N x 512)")
    parser.add_argument("--token_dim", type=int, default=512,
                        help="Token dimension (keep 512 to match the design)")

    # Optimization
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.05)
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

    signal_name = args.signal
    model_name = "video_baseline"

    # Compute clip length from clip_seconds and target_fps
    t_clip = int(round(args.clip_seconds * args.target_fps))
    if t_clip <= 0:
        raise ValueError("clip_seconds * target_fps must be > 0")

    data_dir = Path(args.data_dir)
    checkpoint_path = Path(args.checkpoint_dir) / f"{signal_name}_{model_name}" / "checkpoint.pth"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Signal: {signal_name}, Model: {model_name}")
    logger.info(f"Target clip: T={t_clip}, H=W={args.image_size}, latent: N={args.n_tokens} x {args.token_dim}")

    # Dataset
    dataloader = build_dataloader(
        data_dir=data_dir,
        file_glob=args.file_glob,
        signal=signal_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=args.shuffle,
    )

    # Model
    model = video_baseline.VideoBaselineAutoEncoder(
        n_tokens=args.n_tokens,
        target_size=(t_clip, args.image_size, args.image_size),
        token_dim=args.token_dim,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.MSELoss()

    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr
    )
    drawer = DefaultDrawer(num_plots=args.num_plots) if args.num_plots and args.num_plots > 0 else None

    trainer = UnimodalTrainer(
        epochs=args.epochs,
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        drawer=drawer,
        lr_scheduler=lr_scheduler,
        log_interval=args.log_interval,
    )

    if args.resume and checkpoint_path.exists():
        logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
        trainer.load_checkpoint(checkpoint_path=checkpoint_path)

    trainer.train(dataloader, modality_key=signal_name)


if __name__ == "__main__":
    main()