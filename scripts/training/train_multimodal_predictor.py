"""Stage 3: Train multimodal prediction model.

Uses pre-computed token HDF5 files from Stage 2 and frozen decoders
from Stage 1 to train a fusion transformer that predicts future tokens.

Usage:
    python scripts/training/train_multimodal_predictor.py \
        --token_dir data/tokens \
        --checkpoint_dir runs \
        --signals ece co2 pin tin ech mse ts_core_density filterscopes
"""

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from tokamak_foundation_model.data.token_dataset import make_token_dataloader
from tokamak_foundation_model.models.model_factory import (
    build_model,
    SIGNAL_MODEL_DEFAULTS,
)
from tokamak_foundation_model.models.multimodal_prediction import (
    build_prediction_model,
)
from tokamak_foundation_model.models.prediction_loss import CombinedPredictionLoss
from tokamak_foundation_model.trainer.trainer import Stage3Trainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Actuator signals are input-only (not predicted)
INPUT_ONLY_MODEL_TYPES = {"actuator"}


def load_frozen_decoder(checkpoint_dir: Path, signal_name: str):
    """Load a frozen decoder from a Stage 1 checkpoint."""
    config_path = checkpoint_dir / signal_name / "config.json"
    ckpt_path = checkpoint_dir / signal_name / "checkpoint.pth"

    with open(config_path) as f:
        config = json.load(f)

    model_kwargs = config.get("model_kwargs", {})
    autoencoder = build_model(
        config["model_type"],
        n_channels=config["n_channels"],
        d_model=config["d_model"],
        n_tokens=config.get("n_tokens"),
        **model_kwargs,
    )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        autoencoder.load_state_dict(ckpt["model_state_dict"])
    else:
        autoencoder.load_state_dict(ckpt)

    decoder = autoencoder.decoder
    for p in decoder.parameters():
        p.requires_grad = False
    decoder.to(device)

    logger.info(f"Loaded frozen decoder for '{signal_name}'")
    return decoder


def main():
    parser = argparse.ArgumentParser(
        description="Train multimodal prediction model (Stage 3)"
    )
    parser.add_argument(
        "--token_dir",
        type=str,
        default="data/tokens",
        help="Directory containing token HDF5 files from Stage 2",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="runs",
        help="Directory containing Stage 1 checkpoints",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/multimodal",
        help="Output directory for Stage 3 checkpoints",
    )
    parser.add_argument(
        "--signals",
        nargs="+",
        default=None,
        help="Signals to use (default: auto-detect from token files)",
    )
    parser.add_argument("--d_model", type=int, default=64, help="Model dimension")
    parser.add_argument(
        "--n_heads", type=int, default=8, help="Transformer attention heads"
    )
    parser.add_argument(
        "--n_layers", type=int, default=6, help="Transformer layers"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="DataLoader workers"
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="AdamW weight decay"
    )
    parser.add_argument(
        "--token_weight", type=float, default=1.0, help="Token-space loss weight"
    )
    parser.add_argument(
        "--obs_weight", type=float, default=1.0, help="Observation-space loss weight"
    )
    parser.add_argument(
        "--prediction_horizon",
        type=int,
        default=1,
        help="Number of chunks ahead to predict",
    )
    parser.add_argument(
        "--val_split", type=float, default=0.2, help="Validation split fraction"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume from checkpoint",
    )
    args = parser.parse_args()

    token_dir = Path(args.token_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fusion_ckpt_path = output_dir / "checkpoint.pth"

    # Determine signals
    if args.signals:
        input_signals = args.signals
    else:
        # Auto-detect from checkpoint dirs
        input_signals = [
            s
            for s in SIGNAL_MODEL_DEFAULTS
            if (checkpoint_dir / s / "config.json").exists()
        ]
    logger.info(f"Input signals: {input_signals}")

    # Target signals = non-actuator inputs
    target_signals = [
        s
        for s in input_signals
        if SIGNAL_MODEL_DEFAULTS.get(s) not in INPUT_ONLY_MODEL_TYPES
    ]
    logger.info(f"Target signals: {target_signals}")

    # Create dataloaders
    train_dl, val_dl = make_token_dataloader(
        token_dir=token_dir,
        input_signals=input_signals,
        target_signals=target_signals,
        prediction_horizon_chunks=args.prediction_horizon,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
    )

    # Infer token counts from first sample
    sample = next(iter(train_dl))
    token_counts = {}
    for sig in input_signals:
        if sig in sample["inputs"]:
            token_counts[sig] = sample["inputs"][sig].shape[1]
            logger.info(
                f"  {sig}: {token_counts[sig]} tokens, "
                f"shape {tuple(sample['inputs'][sig].shape)}"
            )

    # Load frozen decoders for target signals
    frozen_decoders = {}
    for sig in target_signals:
        try:
            frozen_decoders[sig] = load_frozen_decoder(checkpoint_dir, sig)
        except FileNotFoundError as e:
            logger.warning(f"No decoder for {sig}: {e}")

    # Build model
    model = build_prediction_model(
        input_signals=[s for s in input_signals if s in token_counts],
        target_signals=[s for s in target_signals if s in frozen_decoders],
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        token_counts=token_counts,
        frozen_decoders=frozen_decoders,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {n_params:,}")

    # Loss and optimizer
    loss_fn = CombinedPredictionLoss(
        frozen_decoders=frozen_decoders,
        token_weight=args.token_weight,
        obs_weight=args.obs_weight,
    )

    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # Train
    trainer = Stage3Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        epochs=args.epochs,
        checkpoint_path=fusion_ckpt_path,
        scheduler=scheduler,
    )

    if args.resume and fusion_ckpt_path.exists():
        trainer.load_checkpoint()

    trainer.train(train_dl, val_dl)


if __name__ == "__main__":
    main()
