from .data import get_file_paths
from .data.datasets.file_based import JoblibDataset as Dataset

# Package metadata
__version__ = "0.1.0"
__author__ = "Peter Steiner"
__email__ = "peter.steiner@princeton.edu"

# Public API - only these should be imported by users
__all__ = [
    # Metadata
    "__version__",
    "__author__",
    "__email__",
]
