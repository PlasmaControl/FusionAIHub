"""Configuration management for autoencoder models.

This module provides utilities for creating, saving, loading, and managing
configurations for autoencoder models. It includes preset configurations
for common use cases and tools for custom configuration management.
"""

import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import yaml

from .autoencoder import BlockBasedAutoencoder
from .mae import MaskedAutoencoder, MaskGenerator


@dataclass
class ModelConfig:
    """Configuration dataclass for autoencoder models.

    This dataclass provides a structured way to define and manage
    autoencoder configurations with validation and serialization support.

    Parameters
    ----------
    input_channels : int
        Number of input channels.
    block_configs : list of dict
        Configuration for each encoder block.
    bottleneck_channels : int, optional
        Number of channels in bottleneck.
    hidden_dim : int, optional
        Target frequency dimension after adaptive pooling.
    kernel_size : int or tuple of int, default=3
        Default kernel size for convolutions.
    bias : bool, default=True
        Whether to use bias in convolutions.
    upsampling_mode : str, default='nearest'
        Upsampling mode for decoder.
    use_batch_norm : bool, default=True
        Whether to use batch normalization.
    activation : str, default='relu'
        Default activation function.
    init_method : str, default='kaiming'
        Weight initialization method.
    model_type : str, default='autoencoder'
        Type of model ('autoencoder' or 'mae').
    mae_config : dict, optional
        Configuration for MAE-specific parameters.
    metadata : dict, optional
        Additional metadata (description, version, etc.).

    Examples
    --------
    >>> config = ModelConfig(
    ...     input_channels=80,
    ...     block_configs=[
    ...         {'out_channels': 128, 'pool_size': (1, 2)},
    ...         {'out_channels': 256, 'pool_size': (1, 4)},
    ...     ],
    ...     hidden_dim=16
    ... )
    >>> model = create_model_from_config(config)
    """

    input_channels: int
    block_configs: list[dict[str, Any]]
    bottleneck_channels: Optional[int] = None
    hidden_dim: Optional[int] = None
    kernel_size: Union[int, tuple[int, int]] = 3
    bias: bool = True
    upsampling_mode: str = "nearest"
    use_batch_norm: bool = True
    activation: str = "relu"
    init_method: str = "kaiming"
    model_type: str = "autoencoder"
    mae_config: Optional[dict[str, Any]] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()

    def _validate(self):
        """Validate configuration parameters."""
        if self.input_channels <= 0:
            raise ValueError(
                f"input_channels must be positive, got {self.input_channels}"
            )

        if not self.block_configs:
            raise ValueError("block_configs cannot be empty")

        for i, config in enumerate(self.block_configs):
            if "out_channels" not in config:
                raise ValueError(
                    f"Block {i} missing required 'out_channels' key"
                )
            if config["out_channels"] <= 0:
                raise ValueError(f"Block {i} out_channels must be positive")

        if (
            self.bottleneck_channels is not None
            and self.bottleneck_channels <= 0
        ):
            raise ValueError(
                "bottleneck_channels must be positive if specified"
            )

        if self.hidden_dim is not None and self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive if specified")

        valid_model_types = ["autoencoder", "mae"]
        if self.model_type not in valid_model_types:
            raise ValueError(f"model_type must be one of {valid_model_types}")

        valid_activations = ["relu", "leaky_relu", "gelu", "swish", "mish"]
        if self.activation not in valid_activations:
            raise ValueError(f"activation must be one of {valid_activations}")

        valid_init_methods = ["kaiming", "xavier", "default"]
        if self.init_method not in valid_init_methods:
            raise ValueError(
                f"init_method must be one of {valid_init_methods}"
            )

        valid_upsampling_modes = ["nearest", "bilinear", "bicubic", "area"]
        if self.upsampling_mode not in valid_upsampling_modes:
            raise ValueError(
                f"upsampling_mode must be one of {valid_upsampling_modes}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "ModelConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)

    def save(self, filepath: Union[str, Path], format: str = "yaml") -> None:
        """Save configuration to file.

        Parameters
        ----------
        filepath : str or Path
            Path to save configuration.
        format : str, default='yaml'
            File format ('yaml' or 'json').
        """
        filepath = Path(filepath)
        config_dict = self.to_dict()

        if format.lower() == "yaml":
            with open(filepath, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        elif format.lower() == "json":
            with open(filepath, "w") as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(
                f"Unsupported format: {format}. Use 'yaml' or 'json'."
            )

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "ModelConfig":
        """Load configuration from file.

        Parameters
        ----------
        filepath : str or Path
            Path to configuration file.

        Returns
        -------
        ModelConfig
            Loaded configuration.
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {filepath}"
            )

        suffix = filepath.suffix.lower()

        with open(filepath) as f:
            if suffix in [".yaml", ".yml"]:
                config_dict = yaml.safe_load(f)
            elif suffix == ".json":
                config_dict = json.load(f)
            else:
                raise ValueError(
                    f"Unsupported file format: {suffix}. "
                    f"Use .yaml, .yml, or .json"
                )

        return cls.from_dict(config_dict)

    def copy(self) -> "ModelConfig":
        """Create a deep copy of the configuration."""
        return ModelConfig.from_dict(deepcopy(self.to_dict()))

    def update(self, **kwargs) -> "ModelConfig":
        """Create a new configuration with updated parameters.

        Parameters
        ----------
        **kwargs
            Parameters to update.

        Returns
        -------
        ModelConfig
            New configuration with updated parameters.
        """
        config_dict = self.to_dict()
        config_dict.update(kwargs)
        return self.from_dict(config_dict)


# Preset configurations for common use cases
PRESET_CONFIGS = {
    "default": ModelConfig(
        input_channels=80,  # Will be overridden by user
        block_configs=[
            {"out_channels": 128, "pool_size": (1, 2)},
            {"out_channels": 256, "pool_size": (1, 2)},
            {"out_channels": 256, "pool_size": (1, 2)},
            {"out_channels": 128, "pool_size": (1, 2)},
            {"out_channels": 64, "pool_size": (1, 2)},
        ],
        metadata={
            "description": "Default balanced configuration",
            "use_case": "General purpose autoencoder",
        },
    ),
    "light": ModelConfig(
        input_channels=80,
        block_configs=[
            {"out_channels": 64, "pool_size": (1, 2)},
            {"out_channels": 128, "pool_size": (1, 2)},
            {"out_channels": 64, "pool_size": (1, 2)},
        ],
        metadata={
            "description": "Lightweight configuration for fast training",
            "use_case": "Resource-constrained environments",
        },
    ),
    "heavy": ModelConfig(
        input_channels=80,
        block_configs=[
            {"out_channels": 128, "pool_size": (1, 2)},
            {"out_channels": 256, "pool_size": (1, 2)},
            {"out_channels": 512, "pool_size": (1, 2)},
            {"out_channels": 512, "pool_size": (1, 2)},
            {"out_channels": 256, "pool_size": (1, 2)},
            {"out_channels": 128, "pool_size": (1, 2)},
        ],
        metadata={
            "description": "Heavy configuration for maximum capacity",
            "use_case": "Large datasets, high-quality reconstruction",
        },
    ),
    "asymmetric": ModelConfig(
        input_channels=80,
        block_configs=[
            {"out_channels": 64, "pool_size": (1, 4)},
            {"out_channels": 128, "pool_size": (1, 2)},
            {"out_channels": 256, "pool_size": (1, 2)},
        ],
        metadata={
            "description": "Asymmetric pooling for different compression "
            "ratios",
            "use_case": "Audio with varying frequency resolution needs",
        },
    ),
    "variable_dropout": ModelConfig(
        input_channels=80,
        block_configs=[
            {"out_channels": 128, "pool_size": (1, 2), "dropout": 0.1},
            {"out_channels": 256, "pool_size": (1, 2), "dropout": 0.2},
            {"out_channels": 256, "pool_size": (1, 2), "dropout": 0.3},
            {"out_channels": 128, "pool_size": (1, 2), "dropout": 0.4},
        ],
        metadata={
            "description": "Progressive dropout for regularization",
            "use_case": "Preventing overfitting in deep networks",
        },
    ),
    "mae_default": ModelConfig(
        input_channels=80,
        block_configs=[
            {"out_channels": 128, "pool_size": (1, 2)},
            {"out_channels": 256, "pool_size": (1, 2)},
            {"out_channels": 128, "pool_size": (1, 2)},
        ],
        model_type="mae",
        mae_config={
            "mask_ratio": 0.75,
            "patch_size": (8, 8),
            "min_mask_size": 1,
            "max_mask_size": None,
            "mask_token_value": 0.0,
        },
        metadata={
            "description": "Default configuration for Masked Autoencoder",
            "use_case": "Self-supervised pre-training",
        },
    ),
    "mae_aggressive": ModelConfig(
        input_channels=80,
        block_configs=[
            {"out_channels": 64, "pool_size": (1, 2)},
            {"out_channels": 128, "pool_size": (1, 4)},
            {"out_channels": 256, "pool_size": (1, 2)},
        ],
        model_type="mae",
        mae_config={
            "mask_ratio": 0.85,
            "patch_size": (4, 4),
            "min_mask_size": 2,
            "max_mask_size": 16,
            "mask_token_value": 0.0,
        },
        metadata={
            "description": "Aggressive masking for challenging pre-training",
            "use_case": "Learning robust representations",
        },
    ),
}


def get_preset_config(name: str) -> ModelConfig:
    """Get a preset configuration by name.

    Parameters
    ----------
    name : str
        Name of the preset configuration.

    Returns
    -------
    ModelConfig
        Copy of the preset configuration.

    Raises
    ------
    KeyError
        If preset name is not found.
    """
    if name not in PRESET_CONFIGS:
        available = list(PRESET_CONFIGS.keys())
        raise KeyError(
            f"Unknown preset: {name}. Available presets: {available}"
        )

    return PRESET_CONFIGS[name].copy()


def list_preset_configs() -> list[str]:
    """List available preset configuration names.

    Returns
    -------
    list of str
        Available preset names.
    """
    return list(PRESET_CONFIGS.keys())


def create_block_autoencoder(
    config_name: str = "default",
    input_channels: Optional[int] = None,
    **kwargs,
) -> BlockBasedAutoencoder:
    """Create autoencoder with predefined or custom configuration.

    Parameters
    ----------
    config_name : str, default='default'
        Name of the preset configuration.
    input_channels : int, optional
        Override input channels in preset.
    **kwargs
        Additional parameters to override in preset.

    Returns
    -------
    BlockBasedAutoencoder
        Configured autoencoder instance.

    Examples
    --------
    >>> # Use preset with custom input channels
    >>> autoencoder = create_block_autoencoder('light', input_channels=80)

    >>> # Override multiple parameters
    >>> autoencoder = create_block_autoencoder(
    ...     'default',
    ...     input_channels=80,
    ...     hidden_dim=16,
    ...     activation='gelu'
    ... )
    """
    # Get base configuration
    config = get_preset_config(config_name)

    # Override input channels if provided
    if input_channels is not None:
        config = config.update(input_channels=input_channels)

    # Override any additional parameters
    if kwargs:
        config = config.update(**kwargs)

    # Create model based on type
    if config.model_type == "autoencoder":
        return create_autoencoder_from_config(config)
    elif config.model_type == "mae":
        return create_mae_from_config(config)
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")


def create_autoencoder_from_config(
    config: ModelConfig,
) -> BlockBasedAutoencoder:
    """Create BlockBasedAutoencoder from configuration.

    Parameters
    ----------
    config : ModelConfig
        Model configuration.

    Returns
    -------
    BlockBasedAutoencoder
        Configured autoencoder.
    """
    # Extract autoencoder parameters
    autoencoder_params = {
        "input_channels": config.input_channels,
        "block_configs": config.block_configs,
        "bottleneck_channels": config.bottleneck_channels,
        "hidden_dim": config.hidden_dim,
        "kernel_size": config.kernel_size,
        "bias": config.bias,
        "upsampling_mode": config.upsampling_mode,
        "use_batch_norm": config.use_batch_norm,
        "activation": config.activation,
        "init_method": config.init_method,
    }

    return BlockBasedAutoencoder(**autoencoder_params)


def create_mae_from_config(config: ModelConfig) -> MaskedAutoencoder:
    """Create MaskedAutoencoder from configuration.

    Parameters
    ----------
    config : ModelConfig
        Model configuration with MAE parameters.

    Returns
    -------
    MaskedAutoencoder
        Configured MAE model.
    """
    # Create base autoencoder
    autoencoder = create_autoencoder_from_config(config)

    # Create mask generator
    mae_config = config.mae_config or {}
    mask_generator = MaskGenerator(
        mask_ratio=mae_config.get("mask_ratio", 0.75),
        patch_size=mae_config.get("patch_size", (8, 8)),
        min_mask_size=mae_config.get("min_mask_size", 1),
        max_mask_size=mae_config.get("max_mask_size", None),
    )

    # Create MAE
    mae = MaskedAutoencoder(
        autoencoder=autoencoder,
        mask_generator=mask_generator,
        mask_token_value=mae_config.get("mask_token_value", 0.0),
    )

    return mae


def save_model_config(
    model: Union[BlockBasedAutoencoder, MaskedAutoencoder],
    filepath: Union[str, Path],
    format: str = "yaml",
    include_metadata: bool = True,
) -> None:
    """Save model configuration to file.

    Parameters
    ----------
    model : BlockBasedAutoencoder or MaskedAutoencoder
        Model to save configuration for.
    filepath : str or Path
        Path to save configuration.
    format : str, default='yaml'
        File format ('yaml' or 'json').
    include_metadata : bool, default=True
        Whether to include metadata in saved config.
    """
    if isinstance(model, MaskedAutoencoder):
        config_dict = model.get_config()
        model_type = "mae"

        # Restructure for ModelConfig format
        autoencoder_config = config_dict["autoencoder_config"]
        mae_params = {
            k: v for k, v in config_dict.items() if k != "autoencoder_config"
        }

        config = ModelConfig(
            model_type=model_type, mae_config=mae_params, **autoencoder_config
        )

    elif isinstance(model, BlockBasedAutoencoder):
        config_dict = model.get_config()
        config = ModelConfig(model_type="autoencoder", **config_dict)

    else:
        raise TypeError(f"Unsupported model type: {type(model)}")

    # Add metadata if requested
    if include_metadata:
        import datetime

        config.metadata.update(
            {
                "saved_at": datetime.datetime.now().isoformat(),
                "model_class": model.__class__.__name__,
                "parameter_count": getattr(model, "parameter_count", None),
            }
        )

    config.save(filepath, format)


def load_model_config(filepath: Union[str, Path]) -> ModelConfig:
    """Load model configuration from file.

    Parameters
    ----------
    filepath : str or Path
        Path to configuration file.

    Returns
    -------
    ModelConfig
        Loaded configuration.
    """
    return ModelConfig.load(filepath)


def create_model_from_config_file(
    filepath: Union[str, Path],
) -> Union[BlockBasedAutoencoder, MaskedAutoencoder]:
    """Create model from configuration file.

    Parameters
    ----------
    filepath : str or Path
        Path to configuration file.

    Returns
    -------
    BlockBasedAutoencoder or MaskedAutoencoder
        Model created from configuration.
    """
    config = load_model_config(filepath)

    if config.model_type == "autoencoder":
        return create_autoencoder_from_config(config)
    elif config.model_type == "mae":
        return create_mae_from_config(config)
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")
