"""Hyperparameter tuning module using Ray Tune."""

import warnings

try:
    from .ray_tuner import (
        RayTuner,
        RayTuneReportCallback,
        cleanup_ray,
        create_basic_search_space,
        suggest_scheduler_config,
    )
    from .search_spaces import (
        CustomSearchSpace,
        SearchSpaces,
        get_search_space,
    )

    __all__ = [
        "RayTuner",
        "RayTuneReportCallback",
        "SearchSpaces",
        "CustomSearchSpace",
        "get_search_space",
        "create_basic_search_space",
        "suggest_scheduler_config",
        "cleanup_ray",
    ]

except ImportError:
    warnings.warn(
        "Ray Tune not available. Hyperparameter tuning "
        "functionality disabled. "
        "Install with: pip install ray[tune] optuna hyperopt"
    )

    __all__ = []
