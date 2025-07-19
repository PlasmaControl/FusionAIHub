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


def prepare_dataset(cfg: dict) -> None:
    """
    Prepare the complete fusion dataset using the modular pipeline.
    
    This function orchestrates the complete dataset preparation workflow:
    1. Collects shot numbers from raw data directory
    2. Processes shots in parallel using the modular pipeline
    3. Splits processed data into train/validation sets
    4. Creates dataset indices for both sets
    
    Args:
        cfg: Configuration dictionary loaded from YAML
    """
    raw_data_dir = Path(cfg["raw_data_dir"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Target sampling frequency: {cfg['fs_khz']} kHz")
    logger.info("Signals configured:")
    for signal in cfg['signal'].items():
        signal_name = signal[0]
        signal_abbr = signal[1]['abbr']
        should_transform = signal[1].get("make_stft", False)
        logger.info(f"  - {signal_name} ({signal_abbr}): transform={should_transform}")
    logger.info("=" * 40)

    # Collect and sort all shot numbers
    logger.info(f"Collecting shots from {raw_data_dir}...")
    all_shots = [
        int(p.stem)
        for p in raw_data_dir.iterdir()
        if p.suffix == ".h5"
    ]
    all_shots.sort()

    # Apply shot selection and randomization if configured
    if cfg.get("randomize_shots", False):
        np.random.seed(cfg["random_seed"])
        all_shots = np.random.permutation(all_shots)

    # Set to -1 to use all shots, or just don't include as argument
    # However, keep argument to stay consistent with other scripts
    if cfg.get("num_shots") is not None:
        all_shots = all_shots[:cfg["num_shots"]]

    logger.info(f"Processing {len(all_shots)} shots into cache...")

    # Clean up existing cache directory if it exists
    if output_dir.exists():
        import shutil
        logger.info(f"Removing existing cache directory: {output_dir}")
        shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Process shots using the appropriate function
    if cfg.get("debug", False):
        output_dir = Path("data") / "debug"
        output_dir.mkdir(parents=True, exist_ok=True)
        pipeline(170000, cfg, output_dir) # For debugging
    else:
        mapper = ParallelMapper()
        mapper(pipeline, all_shots, cfg=cfg, out_dir=output_dir)

    # Move cached files into train/test split
    logger.info("Indexing dataset...")
    all_files = list(output_dir.glob("*.joblib"))
    all_files.sort()

    if len(all_files) == 0:
        logger.warning("Warning: No processed files found. Dataset preparation incomplete.")
        return

    # Index the datasets
    index_dataset(output_dir)


def preprocess(cfg: DictConfig) -> None:
    """
    Main entry point for dataset preparation using Hydra configuration.
    
    Args:
        cfg: Hydra DictConfig object containing configuration
    """
    # Set up logging based on Hydra's log level
    log_level = getattr(logging, cfg.get('log_level', 'INFO').upper())
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Convert DictConfig to regular dict for compatibility with existing code
    cfg_dict = dict(cfg)

    logger.info("Starting dataset preparation with Hydra configuration")
    logger.info(f"Configuration: {cfg_dict}")

    # Prepare dataset using existing function
    prepare_dataset(cfg_dict)
