from pathlib import Path
import argparse
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader

from tokamak_foundation_model.data.data_loader import TokamakH5Dataset, collate_fn
from tokamak_foundation_model.data.utils import worker_init_fn
from tokamak_foundation_model.trainer.trainer import MultimodalTrainer
from tokamak_foundation_model.models.model_factory import SIGNAL_MODEL_DEFAULTS
from tokamak_foundation_model.models.latent_feature_space.baseline_fusion_transformer \
    import BaselineFusionTransformer  # , BaselineForecastingDecoder
from tokamak_foundation_model.utils import DefaultDrawer


# Signals that are input-only (not predicted at output)
INPUT_ONLY_SIGNALS = [key for key, value in SIGNAL_MODEL_DEFAULTS.items() if value ==
                      "actuator"]  # Only diagnostic signals are currently predicted

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_frozen_encoder(checkpoint_path: Path, device: torch.device) -> nn.Module:
    """
    Load pre-trained autoencoder from checkpoint and extract frozen encoder.

    Parameters
    ----------
    checkpoint_path : Path
        Path to the autoencoder checkpoint
    device : torch.device
        Device to load the model on

    Returns
    -------
    nn.Module
        Frozen encoder extracted from the autoencoder
    """
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    logger.info(
        f"Loaded checkpoint from {checkpoint_path}: "
        f"epoch {checkpoint['epoch']}, loss {checkpoint['loss']:.4f}"
    )
    model = checkpoint["model"]
    encoder = model.encoder

    # Freeze all encoder parameters
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()

    return encoder


def main():

    ### Settings ###
    parser = argparse.ArgumentParser(
        description="Train multimodal fusion transformer with forecasting decoders"
    )
    parser.add_argument(
        "--signals", required=False, nargs="+",
        default=['d_alpha', 'mse', 'pin', 'tin', 'ts_core_density', 'irtv'],
        choices=list(SIGNAL_MODEL_DEFAULTS.keys()),
        help="List of input signal names"
    )
    parser.add_argument(
        "--n_fft", type=int, default=1024, help="FFT size"
    )
    parser.add_argument(
        "--hop_length", type=int, default=512, help="STFT hop length"
    )
    parser.add_argument(
        "--data_dir", type=str,
        default="C:/Users/admin/PycharmProjects/FusionAIHub/scripts/",
        help="Path to HDF5 data directory"
    )
    parser.add_argument(
        "--stats_path", type=str, default="preprocessing_stats.pt",
        help="Path to preprocessing stats file"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="runs",
        help="Directory containing pre-trained autoencoder checkpoints "
             "and saving fusion model checkpoints"
    )
    parser.add_argument(
        "--d_model", type=int, default=64, help="Model dimension"
    )
    parser.add_argument(
        "--n_heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument(
        "--n_layers", type=int, default=6, help="Number of transformer layers"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout rate"
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Batch size"
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
        help="LR warmup epochs (0 to disable scheduler)"
    )
    parser.add_argument(
        "--min_lr", type=float, default=0.0,
        help="Minimum LR at end of cosine decay"
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
    checkpoint_dir = Path(args.checkpoint_dir)
    data_dir = Path(args.data_dir)
    statistics_path = Path(args.stats_path)
    fusion_checkpoint_path = checkpoint_dir / "fusion" / "checkpoint.pth"
    fusion_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    ### Resolve input and output signals ###
    input_signals = args.signals
    output_signals = [s for s in input_signals if s not in INPUT_ONLY_SIGNALS]

    logger.info(f"Input signals:  {input_signals}")
    logger.info(f"Output signals: {output_signals}")

    ### Dataset Setup ###
    hdf5_files = sorted(data_dir.glob("*_processed.h5"))
    stats = torch.load(statistics_path)

    datasets_processed = [
        TokamakH5Dataset(
            hdf5_path=str(f),
            preprocessing_stats=stats,
            input_signals=input_signals,
            target_signals=output_signals,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            prediction_mode=True,
        )
        for f in hdf5_files
    ]

    concatenated_dataset = ConcatDataset(datasets_processed)

    ### Load frozen encoders ###
    encoders = {}
    for signal_name in input_signals:
        model_name = SIGNAL_MODEL_DEFAULTS[signal_name]
        ckpt_path = checkpoint_dir / f"{signal_name}_{model_name}" / "checkpoint_best.pth"

        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Pre-trained checkpoint not found for signal '{signal_name}' "
                f"at {ckpt_path}. Run unimodal pre-training first."
            )

        encoders[signal_name] = load_frozen_encoder(ckpt_path, device)
        logger.info(f"Loaded frozen encoder for: {signal_name}")

    ### Infer token counts and output shapes from sample data ###
    data = next(iter(concatenated_dataset))

    # Total tokens across all modalities (for transformer max_tokens)
    total_tokens = 0
    modality_token_counts = {}
    for signal_name, encoder in encoders.items():
        with torch.no_grad():
            sample = data["inputs"][signal_name].unsqueeze(0).to(device)
            tokens = encoder(sample)
            modality_token_counts[signal_name] = tokens.shape[1]
            total_tokens += tokens.shape[1]
            logger.info(
                f"Signal '{signal_name}': {tokens.shape[1]} tokens, "
                f"shape {tokens.shape}"
            )

    # Output shapes for forecasting decoders
    output_shapes = {}
    for signal_name in output_signals:
        output_shapes[signal_name] = tuple(data["targets"][signal_name].shape)
        logger.info(f"Output '{signal_name}': shape {output_shapes[signal_name]}")

    ### Model Setup ###
    fusion_transformer = BaselineFusionTransformer(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        n_modalities=len(input_signals),
        max_tokens=total_tokens,
    ).to(device)

    """
    forecasting_decoders = nn.ModuleDict({
        signal_name: BaselineForecastingDecoder(
            output_shape=output_shapes[signal_name],
            d_model=args.d_model,
        ).to(device)
        for signal_name in output_signals
    })
    """

    n_params_transformer = sum(
        p.numel() for p in fusion_transformer.parameters()
    )
    """
    n_params_decoders = sum(
        p.numel() for p in forecasting_decoders.parameters()
    )
    """
    logger.info(f"Fusion transformer parameters: {n_params_transformer:,}")
    """
    logger.info(f"Forecasting decoder parameters: {n_params_decoders:,}")
    """
    # Only optimize transformer and forecasting decoders (encoders are frozen)
    optimizer = optim.AdamW(
        list(fusion_transformer.parameters()),  # + list(forecasting_decoders.parameters())
        lr=args.lr,
        weight_decay=args.weight_decay,
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
    trainer = MultimodalTrainer(
        epochs=args.epochs,
        checkpoint_path=fusion_checkpoint_path,
        encoders=encoders,
        fusion_transformer=fusion_transformer,
        forecasting_decoders=forecasting_decoders,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        drawer=drawer,
        log_interval=args.log_interval,
    )

    if args.resume and fusion_checkpoint_path.exists():
        logger.info(f"Resuming training from checkpoint: {fusion_checkpoint_path}")
        trainer.load_checkpoint(checkpoint_path=fusion_checkpoint_path)

    trainer.train(dataloader)


if __name__ == "__main__":
    main()