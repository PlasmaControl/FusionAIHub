"""
Shot processing utilities for fusion dataset preparation.

This module contains the main shot processing logic that orchestrates
data extraction, alignment, transformation, and saving for individual shots.
"""

# https://youtu.be/dQw4w9WgXcQ?si=0000000000000000
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

from ..extract.data_extraction import (
    extract_signal, 
    extract_running_time, 
    align_signal,
)
from ..transform.sample_processing import (
    split_samples,
    remove_empty_samples,
    save_sample,
)
from ..transform.signal_processing import (
    identity_transform,
    stft_transform,
    resample_transform,
    resample_linear_transform,
)

# Set up logger for this module
logger = logging.getLogger(__name__)


def pipeline(
    shot_number: int,
    cfg: Dict,
    out_dir: Path,
) -> None:
    """
    Process a single shot through the complete data preparation pipeline accounting transformations.
    
    This function orchestrates the complete processing workflow for a shot:
    1. Determines plasma running time
    2. Extracts and aligns all configured signals to be transformed
    3. Handles missing signals by creating placeholder dataframes
    4. Combines all signals into a unified dataframe
    5. Splits into samples
    6. Transforms and saves the processed samples using STFT transformations.
    
    Args:
        shot: Shot number to process
        cfg: Configuration dictionary
        out_dir: Output directory for processed files
    """

    # Extract running time
    # TODO: Change to call this running_time from ip_threshold
    try:
        # TODO: shot defined as plasma current > 0.5 MA, or 0.5s
        # TODO: George Sips, (find reference slide and cite it)
        start_time, end_time = extract_running_time(
            shot_number=shot_number,
            directory=Path(cfg["raw_data_dir"]),
            ip_threshold=cfg["ip_threshold"],
            start_time=cfg["start_time"],
            end_time=cfg["end_time"],
        )
        logger.info(f"Running time for shot {shot_number}: {start_time} to {end_time}")  
    except Exception as e:
        logger.error(f"Error: Could not determine running time for shot {shot_number}: {e}")
        return
    
    # Extract all signals
    try:
        dfs = []
        missing_signals = []
        for signal in cfg['signal'].items():
            try:
                df = extract_signal(
                    shot_number=shot_number,
                    directory=Path(cfg["raw_data_dir"]),
                    signal=signal[0], 
                    start_time=start_time,
                    end_time=end_time,
                )
                df.columns = [
                    f"{signal[1]['abbr']}_{col}" if col != "time" else col
                    for col in range(len(df.columns))
                ]
                df = align_signal(
                    df=df,
                    start_time=start_time,
                    end_time=end_time,
                    fs=cfg["fs_khz"],
                )
                dfs.append(df)
            except Exception as e:
                for channel in range(int(signal[1]['expected_channels'])):
                    missing_signals.append((signal[1]['abbr'], channel))
    except Exception as e:
        logger.error(f"Error: Could not extract signals for shot {shot_number}: {e}")
        raise e
    
    # Create main aligned dataframe (important since interpolated signals 
    # could have alignment off)
    try:
        # TODO: if df is fixed to same length, then join without inner
        df = pd.concat(dfs, axis=1, join='inner')
    except Exception as e:
        logger.error(f"Error: Could not concatenate dataframes for shot {shot_number}: {e}")
        raise e
    
    # Add missing signals
    if len(missing_signals) > 0:
        try:
            for signal_abbr, channel in missing_signals:
                df[f"{signal_abbr}_{channel}"] = np.nan
                df[f"{signal_abbr}_{channel}_state"] = False
        except Exception as e:
            logger.error(f"Error: Could not add missing signals for shot {shot_number}: {e}")
            raise e

    # Add time column
    df['time_ms'] = np.linspace(start_time, end_time, len(df))

    # Split into samples
    # TODO: rename this to slice_windows
    try:
        samples = split_samples(
            df=df,
            shot_number=shot_number,
            window_ms=cfg["window_ms"],
            hop_ms=cfg["hop_ms"],
            fs_khz=cfg["fs_khz"],
        )
    except Exception as e:
        logger.error(f"Error: Could not split samples for shot {shot_number}: {e}")
        raise e

    # Remove empty samples
    # TODO: Add warning if samples change even if no windows and using ip criterion
    try:
        samples = remove_empty_samples(samples)
    except Exception as e:
        logger.error(f"Error: Could not remove empty samples for shot {shot_number}: {e}")
        raise e

    # If no transform function is provided, save the samples as is
    try:
        if not cfg["do_stft"]:
            for sample in samples:
                transformed_samples = {}
                for key, value in sample.items():
                    for signal in cfg['signal'].items():
                        abbr = signal[1]['abbr']
                        cols = [col for col in value.columns if abbr in col]
                        transformed_samples[abbr] = identity_transform(
                            x=value[cols].to_numpy().T)
                    transformed_samples['time_ms'] = identity_transform(
                        x=np.array([value['time_ms'].to_numpy().T]))
                    save_sample(transformed_samples, out_dir, key)
            return
    except Exception as e:
        logger.error(f"Error: Could not save samples for shot {shot_number}: {e}")
        raise e

    # Get the first transformed sample to determine STFT dimensions
    try:
        first_arr = np.array([list(samples[0].values())[0].iloc[:, 0].values])
        transform_shape = stft_transform(x=first_arr).shape
        logger.info(f"Using {first_arr.shape} as reference for STFT dimensions: {transform_shape}")
    except Exception as e:
        logger.error(f"Error: Could not get first transformed sample for shot {shot_number}: {e}")
        raise e

    # Transform and save samples
    try:
        for sample in samples:
            transformed_samples = {}
            for key, value in sample.items():
                for signal in cfg['signal'].items():
                    abbr = signal[1]['abbr']
                    cols = [col for col in value.columns if abbr in col]
                    if signal[1]['make_stft']:
                        transformed_samples[abbr] = stft_transform(
                            x=value[cols].to_numpy().T,
                            n_fft=cfg["stft"]["n_fft"],
                            hop_length=cfg["stft"]["hop_length"],
                        )
                    else:
                        transformed_samples[abbr] = resample_transform(
                            x=value[cols].to_numpy().T,
                            ref_shape=transform_shape,
                        )
                    transformed_samples['time_ms'] = resample_linear_transform(
                        x=np.array([value['time_ms'].to_numpy().T]),
                        ref_shape=transform_shape,
                    )
                save_sample(transformed_samples, out_dir, key)
    except Exception as e:
        logger.error(f"Error: Could not transform samples for shot {shot_number}: {e}")
        raise e

    return