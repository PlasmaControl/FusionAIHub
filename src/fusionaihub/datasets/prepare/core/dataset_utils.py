"""
Dataset utilities for fusion dataset preparation.

This module contains utility functions for handling missing signals,
creating placeholder dataframes, and indexing dataset files.
"""

import pandas as pd
from pathlib import Path
from typing import Set, List, Dict


def create_missing_signal_dataframes(
    cfg: Dict,
    processed_signals: Set,
    reference_df: pd.DataFrame
) -> List[pd.DataFrame]:
    """
    Create fully off dataframes for missing signals using reference dataframe structure.
    
    When some signals are missing from a shot, this function creates placeholder
    dataframes with the same structure but with all data set to 0 and all states
    set to False.
    
    Args:
        cfg: Configuration dictionary containing signal definitions
        processed_signals: Set of signal abbreviations that were successfully processed
        reference_df: Reference DataFrame to use for structure
        
    Returns:
        List of DataFrames for missing signals
    """
    missing_dfs = []
    
    for signal in cfg["signal"]:
        signal_name, signal_abbr, do_transform = signal
        if signal_abbr not in processed_signals:
            print(f"Creating fully off dataframe for missing signal {signal_name}")
            
            # Create off dataframe by copying structure and zeroing values
            off_df = reference_df.copy()
            
            # Get columns that belong to the reference signal (to replace with new signal columns)
            ref_signal_cols = [col for col in off_df.columns if not col.endswith('_state')]
            
            # Create new column names for the missing signal
            new_cols = {}
            new_state_cols = {}
            
            for i, col in enumerate(ref_signal_cols):
                new_col_name = f"{signal_abbr}col{i}"
                new_cols[col] = new_col_name
                new_state_cols[f"{col}_state"] = f"{new_col_name}_state"
            
            # Rename columns to match the missing signal
            off_df = off_df.rename(columns={**new_cols, **new_state_cols})
            
            # Set all data columns to 0 and all state columns to False
            for col in off_df.columns:
                if col.endswith('_state'):
                    off_df[col] = False
                else:
                    off_df[col] = 0.0
            
            missing_dfs.append(off_df)
    
    return missing_dfs


def index_dataset(out_dir: Path) -> None:
    """
    Create an index file listing all dataset files in the directory.
    
    Scans the output directory for .pkl files and creates an index.pkl
    file containing the list of all dataset files.
    
    Args:
        out_dir: Directory to index
    """
    files = list(out_dir.glob("*.pkl"))
    df_files = pd.DataFrame({'files': [str(file) for file in files]})
    df_files.to_pickle(out_dir / "index.pkl")

    print(f"Indexed {len(files)} files.") 