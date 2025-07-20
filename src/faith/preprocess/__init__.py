# Package metadata
__version__ = "0.1.0"
__author__ = "Nathaniel Chen"
__email__ = "nathaniel@princeton.edu"

# Public API - only these should be imported by users
__all__ = [
    # Metadata
    "__version__",
    "__author__",
    "__email__",
    # Functions
    "preprocess",
]

from .preprocess import preprocess
