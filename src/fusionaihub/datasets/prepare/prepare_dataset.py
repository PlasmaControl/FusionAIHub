#!/usr/bin/env python3
"""
Main script for fusion dataset preparation.

This script orchestrates the complete dataset preparation pipeline using
modular components and YAML configuration.
"""

import yaml
import numpy as np
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Optional
from ...util.parmap import ParallelMapper

from .shot_processing import process_shot_stft
from .dataset_utils import index_dataset

# Set up logger for this module
logger = logging.getLogger(__name__)


def load_config(config_path: Optional[str] = None) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file.
                    If None, uses default.yaml in config directory.
                    
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = str(Path(__file__).parent / "config" / "default.yaml")
    
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    return cfg


def should_use_stft_processing(cfg: dict) -> bool:
    """
    Determine whether to use STFT processing based on configuration.
    
    Args:
        cfg: Configuration dictionary
        
    Returns:
        True if any signals should be transformed with STFT, False otherwise
    """
    for signal in cfg["signal"]:
        if signal.get("make_stft", False):
            return True
    return False


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
    cache_dir = Path(cfg["output_dir"]) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Target sampling frequency: {cfg['fs_khz']} kHz")
    logger.info("Signals configured:")
    for signal in cfg["signal"]:
        signal_name = signal["name"]
        signal_abbr = signal["abbr"]
        should_transform = signal.get("make_stft", False)
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
    
    if cfg.get("num_shots") is not None:
        all_shots = all_shots[:cfg["num_shots"]]
    
    logger.info(f"Processing {len(all_shots)} shots into cache...")
    
    # Clean up existing cache directory if it exists
    # if cache_dir.exists():
    #     import shutil
    #     logger.info(f"Removing existing cache directory: {cache_dir}")
    #     shutil.rmtree(cache_dir)
    #     cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Process shots using the appropriate function
    # process_shot_stft(170000, cfg, cache_dir) # For debugging
    # mapper = ParallelMapper()
    # mapper(process_shot_stft, all_shots, cfg=cfg, out_dir=cache_dir)
    
    # Move cached files into train/test split
    logger.info("Splitting dataset into train and valid sets...")
    all_files = list(cache_dir.glob("*.joblib"))
    all_files.sort()
    
    if len(all_files) == 0:
        logger.warning("Warning: No processed files found. Dataset preparation incomplete.")
        return
    
    # Handle edge case where there are too few files for train-test split
    if len(all_files) == 1:
        logger.warning("Warning: Only 1 file found. Placing in train directory.")
        train_files = all_files
        valid_files = []
    else:
        train_files, valid_files = train_test_split(
            all_files, 
            test_size=cfg.get("train_test_split", 0.2), 
            random_state=cfg["random_seed"]
        )

    # Create train and validation directories
    train_dir = Path(cfg["output_dir"]) / "train"
    valid_dir = Path(cfg["output_dir"]) / "valid"
    train_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)

    # Move files to appropriate directories
    for f in train_files:
        f.rename(train_dir / f.name)
    for f in valid_files:
        f.rename(valid_dir / f.name)
    
    # Index the datasets
    index_dataset(train_dir)
    index_dataset(valid_dir)
        
    # Clean up cache directory
    for f in cache_dir.glob("*"): 
        f.unlink()
    cache_dir.rmdir()

    logger.info("Dataset preparation complete.")
    logger.info(f"Training samples: {len(train_files)}")
    logger.info(f"Validation samples: {len(valid_files)}")


def main():
    """Main entry point for the dataset preparation script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare fusion dataset")
    parser.add_argument(
        "--config", 
        type=str, 
        default=None,
        help="Path to configuration YAML file (default: config/default.yaml)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Load configuration
    cfg = load_config(args.config)
    
    # Prepare dataset
    prepare_dataset(cfg)


if __name__ == "__main__":
    main() 