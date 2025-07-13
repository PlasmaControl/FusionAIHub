"""
Sample processing utilities for fusion dataset preparation.

This module contains functions for splitting signals into time windows,
applying transformations to samples, and saving processed data.
"""

import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path
from typing import Dict

# Set up logger for this module
logger = logging.getLogger(__name__)


def split_samples(
    df: pd.DataFrame,
    shot_number: int,
    window_ms: int | None = None,
    hop_ms: int | None = None,
    fs_khz: float | None = None,
) -> list[dict[str, pd.DataFrame]]:
    """
    Split signal data into overlapping time windows.
    
    Args:
        df: Input DataFrame with signal data
        window_ms: Window size in milliseconds
        hop_ms: Hop size in milliseconds  
        fs_khz: Sampling frequency in kHz
        
    Returns:
        List of DataFrame samples
    """

    if window_ms is None or hop_ms is None or fs_khz is None:
        return [{f"{shot_number}_0": df}]
    
    else:
        # Create sample indicies
        num_samples = int((window_ms) * fs_khz)
        hop_samples = int((hop_ms) * fs_khz)

        # Separate samples
        samples = []
        start_window_idx = 0
        for start_index in range(0, len(df) - num_samples + 1, hop_samples):
            end_index = start_index + num_samples
            sample = df.iloc[start_index:end_index]
            if len(sample) == num_samples:
                samples.append({
                    f"{shot_number}_{start_window_idx}": sample,
                })
                start_window_idx += 1

        return samples


def remove_empty_samples(
    samples: list[dict[str, pd.DataFrame]],
) -> list[dict[str, pd.DataFrame]]:
    """
    Remove empty samples from a list of samples.
    
    Args:
        samples: List of sample DataFrames
    """
    samples = [
        {
            key: value.drop(columns=[col for col in value.columns if col.endswith('_state')])
            for key, value in sample.items()
            if np.any(value.loc[:, value.columns.str.endswith('_state')].to_numpy())
        }
        for sample in samples
    ]

    return samples


def save_sample(
    sample: Dict,
    directory: Path,
    id_val: str,
) -> None:
    """
    Save processed samples to disk using joblib compression.
    
    Args:
        samples: List of sample dictionaries to save
        directory: Output directory
        shot: Shot number for filename generation
    """
    logger.info(f"Saving sample to {directory / f'{id_val}.joblib'}")
    joblib.dump(sample, directory / f"{id_val}.joblib") 