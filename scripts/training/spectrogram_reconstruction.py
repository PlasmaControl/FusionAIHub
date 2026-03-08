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


class FSQUnimodalTrainer(UnimodalTrainer):
    """UnimodalTrainer for SpectrogramFSQVAEAutoEncoder.

    Uses plain L1 loss over all patches — consistent with the val metric so
    train and val losses are directly comparable.
    Also logs codebook utilisation as ``unique_idx / fsq.n_codes``.
    """

    def _train_step(self, batch: dict):
        data = batch[self.modality_key].to(self.dm.device)
        self.optimizer.zero_grad()
        output = self.model(data)
        if isinstance(output, tuple):
            reconstructed, indices = output
            loss = self.loss_fn(reconstructed, data)

            # Codebook utilisation metric
            n_codes = self.model.fsq.n_codes
            utilisation = indices.unique().numel() / n_codes
        else:
            loss = self.loss_fn(output, data)
            utilisation = None

        loss.backward()
        self.optimizer.step()
        metrics = {"loss": loss}
        if utilisation is not None:
            metrics["codebook_utilization"] = torch.tensor(utilisation)
        return metrics


class MAEUnimodalTrainer(UnimodalTrainer):
    """UnimodalTrainer variant that computes loss only on masked patches.

    When the model returns (reconstructed, mask) — i.e. during training with
    SpectrogramMAEAutoEncoder — the loss is restricted to masked patches and
    weighted by the inverse of each patch's standard deviation.  This prevents
    the easy low-energy background from dominating the gradient signal and
    forces the model to learn structure in the high-SNR frequency bands.

    Validation inherits the base _validate_step which computes full-image loss
    (the model returns only reconstructed in eval mode), giving a consistent
    reconstruction metric.
    """

    def _train_step(self, batch: dict):
        data = batch[self.modality_key].to(self.dm.device)
        self.optimizer.zero_grad()
        output = self.model(data)
        if isinstance(output, tuple):
            reconstructed, mask = output  # mask: (B, 1, F_orig, T_orig)
            B, C, F_orig, T_orig = data.shape
            ph, pw = self.model.patch_h, self.model.patch_w

            # Pad to patch-aligned dims (mirrors model forward)
            pad_f = (ph - F_orig % ph) % ph
            pad_t = (pw - T_orig % pw) % pw
            data_pad = torch.nn.functional.pad(data, (0, pad_t, 0, pad_f))
            n_h = data_pad.shape[2] // ph
            n_w = data_pad.shape[3] // pw

            # Patchify data → (B, N, C*ph*pw)
            data_patches = (
                data_pad.reshape(B, C, n_h, ph, n_w, pw)
                .permute(0, 2, 4, 1, 3, 5)
                .reshape(B, n_h * n_w, C * ph * pw)
            )
            patch_std = data_patches.std(dim=-1).clamp(min=1e-6)  # (B, N)

            # Patch-level mask: pad pixel mask then subsample at patch stride
            # Valid because each ph×pw block is uniformly True or False
            mask_pad = torch.nn.functional.pad(mask.float(), (0, pad_t, 0, pad_f))
            mask_patch = mask_pad[:, 0, ::ph, ::pw].bool().reshape(B, n_h * n_w)

            # Patchify reconstruction (cropped to F_orig,T_orig by model; re-pad)
            recon_pad = torch.nn.functional.pad(reconstructed, (0, pad_t, 0, pad_f))
            recon_patches = (
                recon_pad.reshape(B, C, n_h, ph, n_w, pw)
                .permute(0, 2, 4, 1, 3, 5)
                .reshape(B, n_h * n_w, C * ph * pw)
            )

            # Per-patch L1 weighted by inverse patch std; normalize weights so their
            # mean over masked patches = 1, keeping loss on the same scale as plain L1
            patch_l1 = (recon_patches - data_patches).abs().mean(dim=-1)  # (B, N)
            weights = 1.0 / patch_std  # (B, N)
            masked_weights = weights[mask_patch]
            masked_weights = masked_weights / masked_weights.mean()
            loss = (patch_l1[mask_patch] * masked_weights).mean()
        else:
            loss = self.loss_fn(output, data)
        loss.backward()
        self.optimizer.step()
        return {"loss": loss}


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
        "--model", choices=list(MODEL_REGISTRY.keys()), default="actuator",
        help="Model type (default: auto-selected from signal)"
    )
    parser.add_argument(
        "--data_dir", type=str,
        default="/scratch/gpfs/EKOLEMEN/big_d3d_data/dummy_foundation_model_data",
        help="Path to HDF5 data directory"
    )
    parser.add_argument(
        "--stats_path", type=str,
        default="/scratch/gpfs/EKOLEMEN/big_d3d_data/dummy_foundation_model_data/preprocessing_stats.pt",
        help="Path to preprocessing stats file"
    )
    parser.add_argument(
        "--mask_ratio", type=float, default=0.75,
        help="Fraction of patches to mask (spectrogram_mae only)"
    )

    parser.add_argument(
        "--fsq_levels", type=int, nargs="+", default=[8, 5, 5, 5, 5],
        help="FSQ quantization levels per dimension (spectrogram_fsq_vae only)"
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
        help="Fraction of shots to hold out for validation (split by shot, default 0.1)"
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
    stats = torch.load(statistics_path, weights_only=False)

    # Split at shot level to avoid train/val leakage
    n_val = max(1, int(len(hdf5_files) * args.val_split))
    train_files = hdf5_files[:-n_val]
    val_files   = hdf5_files[-n_val:]
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
    val_dataset   = ConcatDataset(_make_datasets(val_files))

    sample_data = next(iter(train_dataset))[signal_name]
    n_channels = sample_data.shape[0]
    logger.info(f"Sample data shape: {sample_data.shape}, n_channels: {n_channels}")

    ### Model Setup ###
    extra_kwargs = {}
    if model_name == "spectrogram_mae":
        extra_kwargs["mask_ratio"] = args.mask_ratio
    if model_name == "spectrogram_fsq_vae":
        extra_kwargs["fsq_levels"] = args.fsq_levels

    model = build_model(
        model_name, args.d_model, args.n_tokens, n_channels, **extra_kwargs
    ).to(device)

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
    # loss_fn = nn.MSELoss()

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
    if model_name == "spectrogram_mae":
        TrainerClass = MAEUnimodalTrainer
    elif model_name == "spectrogram_fsq_vae":
        TrainerClass = FSQUnimodalTrainer
    else:
        TrainerClass = UnimodalTrainer
    trainer = TrainerClass(
        epochs=args.epochs,
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        loss_fn=loss_fn,
        drawer=drawer,
        log_interval=args.log_interval,
    )

    if args.resume and checkpoint_path.exists():
        logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
        trainer.load_checkpoint(checkpoint_path=checkpoint_path)

    trainer.fit(dataloader, val_dataloader=val_dataloader, modality_key=signal_name)


if __name__ == "__main__":
    main()
