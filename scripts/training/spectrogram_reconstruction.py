from pathlib import Path
import argparse
import logging
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader

from tokamak_foundation_model.data.data_loader import TokamakH5Dataset, collate_fn, SignalConfig
from tokamak_foundation_model.data.utils import worker_init_fn
from tokamak_foundation_model.trainer.trainer import UnimodalTrainer
from tokamak_foundation_model.models.model_factory import (
    build_model, MODEL_REGISTRY, SIGNAL_MODEL_DEFAULTS)
from tokamak_foundation_model.models.modality.spectrogram_normalizer import (
    NormalizedSpectrogramAutoEncoder,
)

from tokamak_foundation_model.utils import DefaultDrawer


class SpecAugment(nn.Module):
    """Time and frequency masking (Park et al. 2019 / AST).

    Applies n_freq_masks random frequency-band masks and n_time_masks random
    time-frame masks to the input spectrogram. Each mask width is sampled
    uniformly in [0, mask_param]. Masked regions are zeroed out.
    """

    def __init__(
        self,
        freq_mask_param: int,
        time_mask_param: int,
        n_freq_masks: int = 2,
        n_time_masks: int = 2,
    ):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, F, T)
        B, C, F, T = x.shape
        out = x.clone()
        for _ in range(self.n_freq_masks):
            if self.freq_mask_param > 0:
                f = torch.randint(0, self.freq_mask_param + 1, (B,))
                f0 = torch.randint(0, max(1, F - self.freq_mask_param), (B,))
                for b in range(B):
                    out[b, :, f0[b]: f0[b] + f[b], :] = 0.0
        for _ in range(self.n_time_masks):
            if self.time_mask_param > 0:
                t = torch.randint(0, self.time_mask_param + 1, (B,))
                t0 = torch.randint(0, max(1, T - self.time_mask_param), (B,))
                for b in range(B):
                    out[b, :, :, t0[b]: t0[b] + t[b]] = 0.0
        return out


class FSQUnimodalTrainer(UnimodalTrainer):
    """UnimodalTrainer for SpectrogramFSQVAEAutoEncoder.

    Uses plain L1 loss over all patches — consistent with the val metric so
    train and val losses are directly comparable.
    Also logs codebook utilisation as ``unique_idx / fsq.n_codes``.

    Optional extras:
    - specaugment: SpecAugment instance applied to inputs during training
    - loss_weighting: 'none' (plain L1) or 'variance' (weight by per-pixel
      batch variance to upweight high-variance / fine-detail regions)
    - grad_clip: max gradient norm (0 = disabled)
    """

    def __init__(self, *args, specaugment=None, loss_weighting="none", grad_clip=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.specaugment = specaugment
        self.loss_weighting = loss_weighting
        self.grad_clip = grad_clip

    def _train_step(self, batch: dict):
        data = batch[self.modality_key].to(self.dm.device)
        self.optimizer.zero_grad()

        model_input = self.specaugment(data) if self.specaugment is not None else data
        output = self.model(model_input)

        if isinstance(output, tuple):
            reconstructed, indices = output
            if self.loss_weighting == "variance":
                w = data.var(dim=0, keepdim=True).clamp(min=1e-6)
                w = w / w.mean()
                loss = (w * (reconstructed - data).abs()).mean()
            else:
                loss = self.loss_fn(reconstructed, data)

            # Codebook utilisation metric
            n_codes = self.model.fsq.n_codes
            utilisation = indices.unique().numel() / n_codes
        else:
            loss = self.loss_fn(output, data)
            utilisation = None

        loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        metrics = {"loss": loss}
        if utilisation is not None:
            metrics["codebook_utilization"] = torch.tensor(utilisation)
        return metrics

    def _log_train(self, epoch: int):
        train_loss = self.tracker.metrics["train"]["mean"]["loss"]()
        util = self.tracker.metrics["train"]["mean"].get("codebook_utilization")
        util_str = f", Codebook Util: {util():.3f}" if util is not None else ""
        logger.info(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}{util_str}")


class MAEUnimodalTrainer(UnimodalTrainer):
    """UnimodalTrainer variant that computes loss only on masked patches.

    When the model returns (reconstructed, mask) — i.e. during training with
    SpectrogramMAEAutoEncoder — the loss is restricted to masked pixels.
    Validation inherits the base _validate_step which computes full-image loss
    (the model returns only reconstructed in eval mode), giving a consistent
    reconstruction metric.
    """

    def _train_step(self, batch: dict):
        data = batch[self.modality_key].to(self.dm.device)
        self.optimizer.zero_grad()
        output = self.model(data)
        if isinstance(output, tuple):
            reconstructed, mask = output
            # Expand mask (B,1,F,T) → (B,C,F,T) to index both tensors
            mask_expanded = mask.expand_as(reconstructed)
            loss = self.loss_fn(
                reconstructed[mask_expanded],
                data[mask_expanded],
            )
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
        "--patch_h", type=int, default=16,
        help="Patch height in pixels (spectrogram_mae / spectrogram_fsq_vae)"
    )
    parser.add_argument(
        "--patch_w", type=int, default=16,
        help="Patch width in pixels (spectrogram_mae / spectrogram_fsq_vae)"
    )
    parser.add_argument(
        "--per_channel_patch", action="store_true", default=False,
        help="Use per-channel patch embed/unembed instead of flat projection "
             "(spectrogram_fsq_vae only)"
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
    parser.add_argument(
        "--freq_mask_param", type=int, default=0,
        help="SpecAugment: max frequency bins to mask (0 = disabled)"
    )
    parser.add_argument(
        "--time_mask_param", type=int, default=0,
        help="SpecAugment: max time frames to mask (0 = disabled)"
    )
    parser.add_argument(
        "--n_freq_masks", type=int, default=2,
        help="SpecAugment: number of frequency masks"
    )
    parser.add_argument(
        "--n_time_masks", type=int, default=2,
        help="SpecAugment: number of time masks"
    )
    parser.add_argument(
        "--loss_weighting", type=str, default="none", choices=["none", "variance"],
        help="Loss weighting: 'none' (plain L1) or 'variance' (upweight high-variance pixels)"
    )
    parser.add_argument(
        "--grad_clip", type=float, default=0.0,
        help="Max gradient norm for clipping (0 = disabled)"
    )
    parser.add_argument(
        "--convnext_dims", type=int, nargs="+", default=None,
        help="ConvNeXt channel dims per stage (spectrogram_convnext_fsq only)"
    )
    parser.add_argument(
        "--convnext_depths", type=int, nargs="+", default=None,
        help="ConvNeXt blocks per stage (spectrogram_convnext_fsq only)"
    )
    parser.add_argument(
        "--stem_stride", type=int, default=4,
        help="Stem conv stride (spectrogram_convnext_fsq only)"
    )
    parser.add_argument(
        "--cnn_dims", type=int, nargs="+", default=None,
        help="Channel dims per stage (spectrogram_cnn only, default [64, 128])"
    )
    parser.add_argument(
        "--bottleneck_dim", type=int, default=None,
        help="Bottleneck channel dim for 1x1 projection (spectrogram_cnn only, "
             "default: no projection)"
    )
    parser.add_argument(
        "--normalize", action="store_true", default=False,
        help="Wrap model with learned SpectrogramNormalizer"
    )
    parser.add_argument(
        "--preprocessing", type=str, default=None,
        choices=["log_standardize", "log", "standardize", "normalize", "none"],
        help="Override preprocessing method for the signal (default: use signal's built-in)"
    )
    parser.add_argument(
        "--smooth_kernel_size", type=int, default=7,
        help="Normalizer smoothing conv kernel size (--normalize only)"
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

    # Shuffle shot list before splitting so val is a random draw, not the
    # last N shots by shot number (which would be a different campaign/config).
    random.seed(42)
    random.shuffle(hdf5_files)

    # Split at shot level to avoid train/val leakage
    n_val = max(1, int(len(hdf5_files) * args.val_split))
    train_files = hdf5_files[:-n_val]
    val_files   = hdf5_files[-n_val:]
    logger.info(f"Train shots: {len(train_files)}, Val shots: {len(val_files)}")

    # Override preprocessing method if requested (mutates class-level config
    # before TokamakH5Dataset deep-copies it at __init__ time)
    if args.preprocessing:
        for cfg in TokamakH5Dataset.SIGNAL_CONFIGS:
            if cfg.name == signal_name:
                cfg.preprocess.method = args.preprocessing
                logger.info(f"Preprocessing override: {signal_name} → {args.preprocessing}")
                break

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
    if model_name in ("spectrogram_mae", "spectrogram_fsq_vae"):
        extra_kwargs["patch_h"] = args.patch_h
        extra_kwargs["patch_w"] = args.patch_w
    if model_name == "spectrogram_mae":
        extra_kwargs["mask_ratio"] = args.mask_ratio
    if model_name == "spectrogram_fsq_vae":
        extra_kwargs["fsq_levels"] = args.fsq_levels
        extra_kwargs["per_channel_patch"] = args.per_channel_patch
    if model_name == "spectrogram_convnext_fsq":
        extra_kwargs["fsq_levels"] = args.fsq_levels
        extra_kwargs["stem_stride"] = args.stem_stride
        if args.convnext_dims is not None:
            extra_kwargs["dims"] = args.convnext_dims
        if args.convnext_depths is not None:
            extra_kwargs["depths"] = args.convnext_depths
    if model_name == "spectrogram_cnn":
        if args.cnn_dims is not None:
            extra_kwargs["dims"] = args.cnn_dims
        if args.bottleneck_dim is not None:
            extra_kwargs["bottleneck_dim"] = args.bottleneck_dim

    model = build_model(
        model_name, args.d_model, args.n_tokens, n_channels, **extra_kwargs
    )

    if args.normalize:
        n_freq = args.n_fft // 2
        model = NormalizedSpectrogramAutoEncoder(
            model, n_channels, n_freq,
            smooth_kernel_size=args.smooth_kernel_size,
        )
        logger.info(f"Normalizer enabled: n_freq={n_freq}, kernel_size={args.smooth_kernel_size}")

    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
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
    trainer_kwargs = dict(
        epochs=args.epochs,
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        loss_fn=loss_fn,
        drawer=drawer,
        log_interval=args.log_interval,
    )
    if model_name == "spectrogram_mae":
        TrainerClass = MAEUnimodalTrainer
    elif model_name in ("spectrogram_fsq_vae", "spectrogram_convnext_fsq"):
        TrainerClass = FSQUnimodalTrainer
        specaugment = None
        if args.freq_mask_param > 0 or args.time_mask_param > 0:
            specaugment = SpecAugment(
                freq_mask_param=args.freq_mask_param,
                time_mask_param=args.time_mask_param,
                n_freq_masks=args.n_freq_masks,
                n_time_masks=args.n_time_masks,
            )
            logger.info(
                f"SpecAugment enabled: freq_mask={args.freq_mask_param}, "
                f"time_mask={args.time_mask_param}, "
                f"n_freq={args.n_freq_masks}, n_time={args.n_time_masks}"
            )
        trainer_kwargs["specaugment"] = specaugment
        trainer_kwargs["loss_weighting"] = args.loss_weighting
        trainer_kwargs["grad_clip"] = args.grad_clip
    else:
        TrainerClass = UnimodalTrainer
    trainer = TrainerClass(**trainer_kwargs)

    if args.resume and checkpoint_path.exists():
        logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
        trainer.load_checkpoint(checkpoint_path=checkpoint_path)

    trainer.fit(dataloader, val_dataloader=val_dataloader, modality_key=signal_name)


if __name__ == "__main__":
    main()
