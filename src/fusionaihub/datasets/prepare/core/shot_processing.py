"""
Shot processing utilities for fusion dataset preparation.

This module contains the main shot processing logic that orchestrates
data extraction, alignment, transformation, and saving for individual shots.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict

from .data_extraction import extract, running_time, align
from .sample_processing import transform_samples, save_samples
from .dataset_utils import create_missing_signal_dataframes


def process_shot_stft(shot: int, cfg: Dict, out_dir: Path) -> None:
    """
    Process a single shot through the complete data preparation pipeline accounting for STFT transformations.
    
    This function orchestrates the complete processing workflow for a shot:
    1. Determines plasma running time
    2. Extracts and aligns all configured signals to be transformed
    3. Handles missing signals by creating placeholder dataframes
    4. Combines all signals into a unified dataframe
    5. Transforms and saves the processed samples using STFT transformations.
    6. Downsamples shots not transformed to the same length as the transformed shots.
    
    Args:
        shot: Shot number to process
        cfg: Configuration dictionary
        out_dir: Output directory for processed files
    """
    try:
        dfs = []
        start_time = None
        end_time = None
        
        try:
            start_time, end_time = running_time(
                directory=Path(cfg["raw_data_dir"]),
                shot=shot,
                ip_threshold=cfg["ip_threshold"]
            )
            reference_len = int((end_time - start_time) * cfg["fs_khz"])
            print(f"Running time for shot {shot}: {start_time} to {end_time}")  
        except Exception as e:
            print(f"Error: Could not determine running time for shot {shot}: {e}")
            return
        
        # Process each signal and track which ones succeeded
        processed_signals = set()
        
        for signal in cfg["signal"]:
            signal_name, signal_abbr, is_transformed = signal
            df = None
            
            if is_transformed:
                try:
                    # Try to extract and process the signal
                    df = extract(shot=shot, directory=Path(cfg["raw_data_dir"]), signal=signal_name)
                    df.columns = [f"{signal_abbr}{col}" if col != "time" else col for col in df.columns]
                    df = align(df, start_time, end_time, cfg["fs_khz"])
                    processed_signals.add(signal_abbr)
                    dfs.append(df)
                    print(f"Successfully processed signal {signal_name} for shot {shot}")
                except Exception as e:
                    print(f"Error processing signal {signal_name} for shot {shot}: {e}")
        
        if not dfs:
            print(f"Error: No dataframes created for shot {shot}")
            return
            
        df = pd.concat(dfs, axis=1, join='inner')

        # num_samples = len(df)
        # new_index = np.linspace(start_time, end_time, num_samples)
        # df.index = new_index
        # df.index = pd.to_timedelta(df.index, unit='ms')
        
        samples = [df]  # no splitting for this dataset
        print(f"Shot {shot} has {len(samples)} samples after splitting.")
        samples_dict_list = transform_samples(samples, out_dir, cfg["signal"], shot)

        # Get the first (and only) sample dictionary since we don't split
        sample_dict = samples_dict_list[0]
        
        # Get a sample from sample_dict to determine STFT dimensions
        first_key = next(iter(sample_dict.keys()))
        stft_width = sample_dict[first_key].shape[-1]
        print(f"Using {first_key} as reference for STFT dimensions: {stft_width}")
        stft_fs = stft_width / (end_time - start_time)
        
        for signal in cfg["signal"]:
            signal_name, signal_abbr, is_transformed = signal
            if not is_transformed:
                try:
                    df = extract(shot=shot, directory=Path(cfg["raw_data_dir"]), signal=signal_name)
                    df.columns = [f"{signal_abbr}{col}" if col != "time" else col for col in df.columns]
                    df = align(df, start_time, end_time, stft_fs)
                    # Ensure signal matches STFT width by truncating if necessary
                    if len(df) > stft_width:
                        df = df.iloc[:stft_width]
                        print(f"Truncated {signal_abbr} from {len(df)} to {stft_width} samples to match STFT width")
                    elif len(df) < stft_width:
                        print(f"Warning: {signal_abbr} has {len(df)} samples, less than STFT width {stft_width}")
                        pad_width = stft_width - len(df)
                        df = np.pad(df, (0, pad_width), mode='constant', constant_values=0)
                        print(f"Padded {signal_abbr} from {len(df)} to {stft_width} samples with zeros")
                    print(f"Successfully processed non-transformed signal {signal_name} for shot {shot}")
                except Exception as e:
                    print(f"Error processing non-transformed signal {signal_name} for shot {shot}: {e}")

        save_samples([sample_dict], out_dir, shot)
        print(f"Processed shot {shot} successfully with {len(cfg['signal'])} signals.")

    except Exception as e:
        print(f"Error processing shot {shot}: {e}")
        return

    return