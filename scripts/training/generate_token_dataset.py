"""Stage 2: Generate token dataset from trained unimodal autoencoders.

Runs each trained autoencoder in inference mode on raw HDF5 shot files,
saving encoder output tokens and preprocessed observations to per-shot
HDF5 files for Stage 3 multimodal prediction training.

Usage:
    # Single shot test
    python scripts/training/generate_token_dataset.py --max_shots 1

    # Process 100 shots
    python scripts/training/generate_token_dataset.py --max_shots 100

    # Process a range (for SLURM array jobs)
    python scripts/training/generate_token_dataset.py --shot_start 0 --shot_end 1000

    # All shots
    python scripts/training/generate_token_dataset.py
"""

import argparse
import json
import logging
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from tokamak_foundation_model.data.data_loader import TokamakH5Dataset, collate_fn
from tokamak_foundation_model.models.model_factory import (
    build_model,
    SIGNAL_MODEL_DEFAULTS,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_autoencoder(checkpoint_dir: Path, signal_name: str):
    """Load a trained autoencoder from checkpoint.

    Tries config.json first; falls back to inferring from signal defaults.
    Tries checkpoint_best.pth first; falls back to checkpoint.pth.
    """
    signal_dir = checkpoint_dir / signal_name
    config_path = signal_dir / "config.json"

    # Load or infer config
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        model_type = config["model_type"]
        d_model = config["d_model"]
        n_tokens = config.get("n_tokens")
        n_channels = config["n_channels"]
        model_kwargs = config.get("model_kwargs", {})
    else:
        # Infer from defaults — need to determine n_channels from data
        model_type = SIGNAL_MODEL_DEFAULTS[signal_name]
        d_model = 64
        n_tokens = None
        n_channels = None  # caller must set this
        model_kwargs = {}

    return model_type, d_model, n_tokens, n_channels, model_kwargs


def build_and_load(
    signal_name: str,
    model_type: str,
    d_model: int,
    n_tokens,
    n_channels: int,
    model_kwargs: dict,
    checkpoint_dir: Path,
):
    """Build model and load weights from checkpoint."""
    signal_dir = checkpoint_dir / signal_name

    # Profile models override n_channels
    kwargs = dict(model_kwargs)
    if model_type == "profile":
        kwargs.setdefault("n_spatial_points", n_channels)
        build_channels = 1
    else:
        build_channels = n_channels

    model = build_model(
        model_type,
        n_channels=build_channels,
        d_model=d_model,
        n_tokens=n_tokens,
        **kwargs,
    )

    # Try best checkpoint first, then regular
    best_path = signal_dir / "checkpoint_best.pth"
    regular_path = signal_dir / "checkpoint.pth"

    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
        logger.info(f"Loaded {signal_name} from {best_path}")
    elif regular_path.exists():
        ckpt = torch.load(regular_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
        logger.info(f"Loaded {signal_name} from {regular_path}")
    else:
        raise FileNotFoundError(f"No checkpoint found in {signal_dir}")

    model.to(device)
    model.eval()
    return model


def process_shot(
    hdf5_path: Path,
    encoders: dict,
    stats: dict,
    output_path: Path,
    chunk_duration_s: float,
    n_fft: int,
    hop_length: int,
    batch_size: int,
):
    """Process a single shot file, encoding all available signals."""
    with h5py.File(output_path, "w") as out_f:
        signals_written = 0

        for signal_name, model in encoders.items():
            try:
                ds = TokamakH5Dataset(
                    hdf5_path=str(hdf5_path),
                    preprocessing_stats=stats,
                    input_signals=[signal_name],
                    target_signals=[signal_name],
                    chunk_duration_s=chunk_duration_s,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    prediction_mode=False,
                )
            except Exception:
                continue

            if len(ds) == 0:
                continue

            loader = DataLoader(
                ds,
                batch_size=batch_size,
                collate_fn=collate_fn,
                num_workers=0,
                shuffle=False,
            )

            all_tokens = []
            all_observations = []

            with torch.no_grad():
                for batch in loader:
                    data = batch[signal_name].to(device)
                    _, tokens = model(data)
                    all_tokens.append(tokens.cpu().numpy())
                    all_observations.append(data.cpu().numpy())

            tokens_arr = np.concatenate(all_tokens, axis=0)
            obs_arr = np.concatenate(all_observations, axis=0)

            grp = out_f.create_group(signal_name)
            grp.create_dataset(
                "tokens", data=tokens_arr, dtype="float32",
                compression="gzip", compression_opts=4,
            )
            grp.create_dataset(
                "observations", data=obs_arr, dtype="float32",
                compression="gzip", compression_opts=4,
            )
            signals_written += 1

    return signals_written


def main():
    parser = argparse.ArgumentParser(
        description="Generate token dataset from trained autoencoders (Stage 2)"
    )
    parser.add_argument(
        "--signals",
        nargs="+",
        default=None,
        help="Signals to encode (default: auto-detect from checkpoints)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="runs",
        help="Directory containing Stage 1 checkpoints",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/scratch/gpfs/EKOLEMEN/foundation_model",
        help="Directory containing raw HDF5 shot files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/tokens",
        help="Output directory for token HDF5 files",
    )
    parser.add_argument(
        "--stats_path",
        type=str,
        default="data/preprocessing_stats.pt",
        help="Path to preprocessing stats file",
    )
    parser.add_argument(
        "--chunk_duration_s",
        type=float,
        default=0.2,
        help="Duration of each data chunk in seconds",
    )
    parser.add_argument("--n_fft", type=int, default=256, help="FFT size")
    parser.add_argument(
        "--hop_length", type=int, default=256, help="Hop length for STFT"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Inference batch size"
    )
    parser.add_argument(
        "--max_shots",
        type=int,
        default=None,
        help="Maximum number of shots to process (for testing)",
    )
    parser.add_argument(
        "--shot_start",
        type=int,
        default=None,
        help="Start index for shot range (for SLURM array)",
    )
    parser.add_argument(
        "--shot_end",
        type=int,
        default=None,
        help="End index for shot range (for SLURM array)",
    )
    parser.add_argument(
        "--no_skip_existing",
        action="store_true",
        default=False,
        help="Reprocess shots that already have token files",
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = torch.load(args.stats_path, weights_only=False)

    # Discover available signals (those with checkpoints)
    if args.signals:
        signal_names = args.signals
    else:
        signal_names = []
        for sig in SIGNAL_MODEL_DEFAULTS:
            sig_dir = checkpoint_dir / sig
            if (sig_dir / "checkpoint.pth").exists() or (
                sig_dir / "checkpoint_best.pth"
            ).exists():
                signal_names.append(sig)
    logger.info(f"Signals to encode: {signal_names}")

    if not signal_names:
        logger.error("No signals with checkpoints found!")
        return

    # Probe data shapes from first available shot
    hdf5_files = sorted(data_dir.glob("*.h5"))
    logger.info(f"Found {len(hdf5_files)} shot files")

    probe_path = hdf5_files[0]
    signal_n_channels = {}
    for sig in signal_names:
        try:
            ds = TokamakH5Dataset(
                hdf5_path=str(probe_path),
                preprocessing_stats=stats,
                input_signals=[sig],
                target_signals=[sig],
                chunk_duration_s=args.chunk_duration_s,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
            )
            sample = ds[0]
            signal_n_channels[sig] = sample[sig].shape[0]
        except Exception as e:
            logger.warning(f"Cannot probe {sig}: {e}")

    # Load all autoencoders
    encoders = {}
    for sig in signal_names:
        if sig not in signal_n_channels:
            continue
        try:
            model_type, d_model, n_tokens, n_channels, model_kwargs = (
                load_autoencoder(checkpoint_dir, sig)
            )
            # Use probed n_channels if config didn't have it
            if n_channels is None:
                n_channels = signal_n_channels[sig]

            model = build_and_load(
                sig, model_type, d_model, n_tokens, n_channels,
                model_kwargs, checkpoint_dir,
            )
            encoders[sig] = model
            logger.info(
                f"  {sig}: {type(model).__name__}, "
                f"n_channels={n_channels}, d_model={d_model}"
            )
        except Exception as e:
            logger.warning(f"Failed to load {sig}: {e}")

    logger.info(f"Loaded {len(encoders)} encoders: {list(encoders.keys())}")

    # Select shot range
    if args.shot_start is not None and args.shot_end is not None:
        hdf5_files = hdf5_files[args.shot_start : args.shot_end]
    if args.max_shots is not None:
        hdf5_files = hdf5_files[: args.max_shots]

    logger.info(f"Processing {len(hdf5_files)} shots")
    skip_existing = not args.no_skip_existing

    # Process shots
    processed = 0
    skipped = 0
    failed = 0

    for i, hdf5_path in enumerate(hdf5_files):
        shot_id = hdf5_path.stem
        output_path = output_dir / f"{shot_id}_tokens.h5"

        if skip_existing and output_path.exists():
            skipped += 1
            continue

        try:
            n_signals = process_shot(
                hdf5_path=hdf5_path,
                encoders=encoders,
                stats=stats,
                output_path=output_path,
                chunk_duration_s=args.chunk_duration_s,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
                batch_size=args.batch_size,
            )
            processed += 1
            if (i + 1) % 10 == 0 or i == 0:
                logger.info(
                    f"[{i + 1}/{len(hdf5_files)}] {shot_id}: "
                    f"{n_signals} signals encoded"
                )
        except Exception as e:
            logger.warning(f"Failed to process {shot_id}: {e}")
            # Remove partial file
            if output_path.exists():
                output_path.unlink()
            failed += 1

    logger.info(
        f"Done. Processed: {processed}, Skipped: {skipped}, Failed: {failed}"
    )


if __name__ == "__main__":
    main()
