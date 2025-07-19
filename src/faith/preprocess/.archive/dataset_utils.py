"""
Dataset utilities for fusion dataset preparation.

This module contains utility functions for handling missing signals,
creating placeholder dataframes, and indexing dataset files.
"""

import logging
from pathlib import Path

import pandas as pd

# Set up logger for this module
logger = logging.getLogger(__name__)


def index_dataset(out_dir: Path) -> None:
    """
    Create an index file listing all dataset files in the directory.

    Scans the output directory for .joblib files and creates an index.pkl
    file containing the list of all dataset files.

    Args:
        out_dir: Directory to index
    """
    files = list(out_dir.glob("*.joblib"))
    df_files = pd.DataFrame({"files": [str(file) for file in files]})
    df_files.to_csv(out_dir / "index.csv", index=False)

    logger.info(f"Indexed {len(files)} files.")
