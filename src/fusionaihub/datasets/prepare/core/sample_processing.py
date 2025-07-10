"""
Sample processing utilities for fusion dataset preparation.

This module contains functions for splitting signals into time windows,
applying transformations to samples, and saving processed data.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import List, Dict
from .signal_processing import resample_nearest, transform_individual_sample


def split(df: pd.DataFrame, window_ms: int, hop_ms: int, fs_khz: float) -> List[pd.DataFrame]:
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
    # Create sample indicies
    num_samples = int((window_ms) * fs_khz)
    hop_samples = int((hop_ms) * fs_khz)
    
    # Separate samples
    samples = []
    for start in range(0, len(df) - num_samples + 1, hop_samples):
        end = start + num_samples
        sample = df.iloc[start:end]
        if len(sample) == num_samples:
            samples.append(sample)
            
    return samples


def transform_samples(
    samples: List[pd.DataFrame],
    directory: Path,
    signal_config: List[tuple],
    shot: int,
) -> List[Dict]:
    """
    Transform and process samples based on signal configuration.
    
    Applies transformations (like STFT) to specified signals and resamples
    non-transformed signals to match the dimensions of transformed ones.
    
    Args:
        samples: List of sample DataFrames
        directory: Output directory
        signal_config: List of (signal_name, abbreviation, should_transform) tuples
        shot: Shot number for logging
        
    Returns:
        List of dictionaries containing processed sample data
    """
    directory.mkdir(parents=True, exist_ok=True)
    print(f"Processing {len(samples)} samples for shot {shot}")
    samples_dict = []
    
    # Create mapping from signal abbreviation to whether it should be transformed
    transform_map = {}
    for signal_name, signal_abbr, should_transform in signal_config:
        transform_map[signal_abbr] = should_transform
    
    for i, sample in enumerate(samples):
        
        # Remove columns ending with '_state'
        sample_to_save = sample.loc[:, ~sample.columns.str.endswith('_state')]
        
        # Only save if not fully off (i.e., at least one True in any state col)
        state_cols = [col for col in sample.columns if col.endswith('_state')]
        if np.any(sample[state_cols].to_numpy()):
            
            # First pass: apply transformations and collect results
            sample_dict = {}
            transformed_sample = None
            original_time_length = len(sample_to_save)
            
            for col in sample_to_save.columns:
                # Convert each column to float32 numpy array
                col_array = sample_to_save[col].values.astype(np.float32)
                
                # Determine if this column should be transformed based on signal abbreviation
                should_transform = False
                for signal_abbr in transform_map.keys():
                    if col.startswith(signal_abbr):
                        should_transform = transform_map[signal_abbr]
                        break
                
                if should_transform:
                    transformed_array = transform_individual_sample(col_array)
                    sample_dict[col] = transformed_array
                    # Store an example transformed sample to get target dimensions
                    if transformed_sample is None:
                        transformed_sample = transformed_array
                        print(f"Reference transformed sample shape: {transformed_array.shape}")
                else:
                    # Store original array for now, will resample later
                    sample_dict[col] = col_array
            
            # Second pass: resample non-transformed samples to match transformed dimensions
            if transformed_sample is not None:
                target_width = transformed_sample.shape[-1]  # Last dimension is time
                # Calculate target sample frequency based on transformed sample
                target_fs = target_width / original_time_length
                print(f"Target frequency ratio: {target_fs:.4f} (target width: {target_width}, original length: {original_time_length})")
                
                for col in sample_dict.keys():
                    # Check if this column was transformed
                    should_transform = False
                    for signal_abbr in transform_map.keys():
                        if col.startswith(signal_abbr):
                            should_transform = transform_map[signal_abbr]
                            break
                    
                    if not should_transform:
                        # Resample non-transformed data to match target width
                        original_array = sample_dict[col]
                        resampled_array = resample_nearest(original_array, target_width)
                        
                        # Crop end if needed to ensure exact match
                        if len(resampled_array) > target_width:
                            resampled_array = resampled_array[:target_width]
                        elif len(resampled_array) < target_width:
                            # Pad with zeros if too short (shouldn't happen with resample_nearest)
                            pad_width = target_width - len(resampled_array)
                            resampled_array = np.pad(resampled_array, (0, pad_width), mode='constant')
                        
                        sample_dict[col] = resampled_array.astype(np.float32)
                        print(f"Resampled {col} from {len(original_array)} to {len(resampled_array)}")
            
            samples_dict.append(sample_dict)
            print(f"Sample {i} processed with {len(sample_dict)} signals")

    return samples_dict


def save_samples(samples: List[Dict], directory: Path, shot: int) -> None:
    """
    Save processed samples to disk using joblib compression.
    
    Args:
        samples: List of sample dictionaries to save
        directory: Output directory
        shot: Shot number for filename generation
    """
    directory.mkdir(parents=True, exist_ok=True)
    print(f"Saving {len(samples)} samples to {directory}")
    for i, sample in enumerate(samples):
        # Save using joblib
        joblib.dump(sample, directory / f"{shot}_{i}.pkl", compress=True) 