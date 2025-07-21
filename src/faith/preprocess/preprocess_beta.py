#!/usr/bin/env python3
"""
Main script for fusion dataset preparation.

This script orchestrates the complete dataset preparation pipeline using
modular components and YAML configuration.
"""

import logging
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from .pipelines import pipeline_v0_stable as pipeline
from .util import ParallelMapper, index_dataset

# Set up logger for this module
logger = logging.getLogger(__name__)


def list_shots(
    cfg: dict,
) -> list[int]:
    """
    List all shot numbers in the raw data directory.

    Args:
        cfg: Configuration dictionary loaded from YAML
    """
    # Collect and sort all shot numbers
    raw_data_dir = Path(cfg["raw_data_dir"])
    logger.info(f"Collecting shots from {raw_data_dir}...")
    all_shots = [int(p.stem) for p in raw_data_dir.iterdir() if p.suffix == ".h5"]
    all_shots.sort()

    # Apply shot selection and randomization if configured
    if cfg.get("randomize_shots", False):
        np.random.seed(cfg["random_seed"])
        all_shots = np.random.permutation(all_shots)

    # Set to -1 to use all shots, or just don't include as argument
    # However, keep argument to stay consistent with other scripts
    if cfg.get("num_shots") is not None:
        all_shots = all_shots[: cfg["num_shots"]]

    logger.info(f"Found {len(all_shots)} shots")
    return all_shots


def log_config(
    cfg: dict,
) -> None:
    """
    Log the configuration.
    """
    logger.info("Configuration:")
    logger.info(f"Target sampling frequency: {cfg['fs_khz']} kHz")
    logger.info("Signals configured:")
    for signal in cfg["signal"].items():
        signal_name = signal[0]
        signal_abbr = signal[1]["abbr"]
        should_transform = signal[1].get("make_stft", False)
        logger.info(f"  - {signal_name} ({signal_abbr}): transform={should_transform}")
    logger.info("=" * 40)


def prepare_dataset(
    cfg: dict,
) -> None:
    """
    Prepare the complete fusion dataset using the modular pipeline.

    This function orchestrates the complete dataset preparation workflow:
    1. Collects shot numbers from raw data directory
    2. Processes shots in parallel using the modular pipeline
    3. Splits processed data into train/validation sets
    4. Creates dataset indices for both sets

    Args:
        cfg: Configuration dictionary loaded from YAML

    # TODO: Add option to check if the dataset is mid-way through processing,
    # and if so, continue from there.
    """
    # Log the configuration
    log_config(cfg)

    # List the shots
    shots = list_shots(cfg)

    # Create the output directory
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Process shots using the appropriate function
    if cfg.get("debug", False):
        output_dir = Path("data") / "debug"
        output_dir.mkdir(parents=True, exist_ok=True)
        pipeline(cfg["debug_shot"], cfg, output_dir)  # For debugging
    else:
        mapper = ParallelMapper()
        mapper(pipeline, shots, cfg=cfg, out_dir=out_dir)

    # Index the datasets
    index_dataset(out_dir)


def preprocess(
    cfg: DictConfig,
) -> None:
    """
    Main entry point for dataset preparation using Hydra configuration.

    Args:
        cfg: Hydra DictConfig object containing configuration
    """
    # Set up logging based on Hydra's log level
    log_level = getattr(logging, cfg.get("log_level", "INFO").upper())
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Prepare dataset using existing function
    prepare_dataset(cfg)
