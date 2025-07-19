"""Utility functions for autoencoder models.

This module provides convenience functions, model analysis tools, validation
utilities, and helper functions for working with autoencoder models.
"""

from typing import Any, Optional, Union

import torch

from . import PRESET_CONFIGS, create_block_autoencoder
from .autoencoder import BlockBasedAutoencoder
from .mae import MASK_TYPES, MaskedAutoencoder


def create_mae_model(
    config_name: str = "mae_default",
    input_channels: Optional[int] = None,
    mask_ratio: float = 0.75,
    mask_type: str = "random",
    **kwargs,
) -> MaskedAutoencoder:
    """Create a masked autoencoder with sensible defaults.

    This convenience function creates a complete MAE model with a single
    function call, using preset configurations and sensible defaults.

    Parameters
    ----------
    config_name : str, default='mae_default'
        Preset configuration name for the base autoencoder.
        Use 'mae_default' or 'mae_aggressive' for MAE-specific presets,
        or any autoencoder preset which will be converted to MAE.
    input_channels : int, optional
        Number of input channels. If None, must be specified in kwargs.
    mask_ratio : float, default=0.75
        Ratio of input to mask (0.0 to 1.0).
    mask_type : str, default='random'
        Default masking strategy to use during training.
    **kwargs
        Additional arguments passed to create_block_autoencoder.

    Returns
    -------
    MaskedAutoencoder
        Configured MAE model ready for training.

    Examples
    --------
    >>> # Simple MAE creation
    >>> mae = create_mae_model('mae_default', input_channels=80)
    >>> x = torch.randn(1, 80, 100, 128)
    >>> reconstructed, mask, masked_input = mae(x)

    >>> # Custom configuration
    >>> mae = create_mae_model(
    ...     'light',
    ...     input_channels=80,
    ...     mask_ratio=0.8,
    ...     hidden_dim=16
    ... )
    """
    # Handle input_channels
    if input_channels is not None:
        kwargs["input_channels"] = input_channels
    elif "input_channels" not in kwargs:
        raise ValueError(
            "input_channels must be specified either as parameter or in kwargs"
        )

    # Override MAE config if using non-MAE preset
    if not config_name.startswith("mae_"):
        kwargs.setdefault("model_type", "mae")
        kwargs.setdefault(
            "mae_config",
            {
                "mask_ratio": mask_ratio,
                "patch_size": (8, 8),
                "min_mask_size": 1,
                "max_mask_size": None,
                "mask_token_value": 0.0,
            },
        )

    # Create the model
    model = create_block_autoencoder(config_name, **kwargs)

    if not isinstance(model, MaskedAutoencoder):
        raise RuntimeError(
            f"Expected MaskedAutoencoder, got {type(model).__name__}"
        )

    return model


def get_model_info() -> dict[str, Any]:
    """Get information about available models and configurations.

    Returns comprehensive information about the models package including
    available configurations, model types, masking strategies, and versions.

    Returns
    -------
    dict
        Dictionary containing model information including:
        - available_configs: List of preset configuration names
        - model_types: List of available model types
        - mask_types: List of available masking strategies
        - preset_descriptions: Descriptions of each preset
        - version: Package version (if available)

    Examples
    --------
    >>> info = get_model_info()
    >>> print(f"Available configs: {info['available_configs']}")
    >>> print(f"MAE presets: {info['mae_presets']}")
    >>> for name, desc in info['preset_descriptions'].items():
    ...     print(f"{name}: {desc}")
    """
    # Get preset information
    preset_info = {}
    mae_presets = []
    autoencoder_presets = []

    for name, config in PRESET_CONFIGS.items():
        preset_info[name] = {
            "description": config.metadata.get(
                "description", "No description"
            ),
            "use_case": config.metadata.get("use_case", "General"),
            "model_type": config.model_type,
            "num_blocks": len(config.block_configs),
            "has_mae_config": config.mae_config is not None,
        }

        if config.model_type == "mae":
            mae_presets.append(name)
        else:
            autoencoder_presets.append(name)

    # Package version
    try:
        from src import __version__

        version = __version__
    except ImportError:
        version = "unknown"

    return {
        "available_configs": list(PRESET_CONFIGS.keys()),
        "autoencoder_presets": autoencoder_presets,
        "mae_presets": mae_presets,
        "model_types": ["BlockBasedAutoencoder", "MaskedAutoencoder"],
        "mask_types": MASK_TYPES,
        "preset_descriptions": {
            name: info["description"] for name, info in preset_info.items()
        },
        "preset_details": preset_info,
        "version": version,
        "description": "Block-based autoencoders for audio and spectral data",
    }


def validate_input_shape(shape: tuple[int, ...]) -> bool:
    """Validate input tensor shape for autoencoder models.

    Checks if the input shape is compatible with autoencoder models,
    which expect 4D tensors (batch, channels, height, width).

    Parameters
    ----------
    shape : tuple of int
        Input tensor shape to validate.

    Returns
    -------
    bool
        True if shape is valid, False otherwise.

    Examples
    --------
    >>> validate_input_shape((32, 80, 100, 128))  # Valid
    True
    >>> validate_input_shape((80, 100, 128))  # Missing batch dimension
    False
    >>> validate_input_shape((32, 80, 100, 128, 1))  # Too many dimensions
    False
    """
    if len(shape) != 4:
        return False

    batch, channels, height, width = shape
    return all(dim > 0 for dim in shape)


def get_memory_estimate(
    model: Union[BlockBasedAutoencoder, MaskedAutoencoder],
    input_shape: tuple[int, ...],
    batch_size: int = 1,
    dtype: torch.dtype = torch.float32,
) -> dict[str, float]:
    """Estimate memory usage for a model with given input shape.

    Provides estimates for parameter memory, activation memory, and total
    memory usage. Useful for planning training on resource-constrained systems.

    Parameters
    ----------
    model : BlockBasedAutoencoder or MaskedAutoencoder
        The autoencoder model to analyze.
    input_shape : tuple of int
        Input tensor shape (channels, height, width)
        or (batch, channels, height, width).
    batch_size : int, default=1
        Batch size for estimation (ignored if input_shape includes batch).
    dtype : torch.dtype, default=torch.float32
        Data type for memory calculation.

    Returns
    -------
    dict
        Memory estimates in MB including:
        - parameters_mb: Model parameter memory
        - activations_mb: Estimated activation memory
        - total_mb: Total estimated memory
        - encoder_mb: Encoder-specific memory
        - decoder_mb: Decoder-specific memory

    Examples
    --------
    >>> model = create_mae_model('light', input_channels=80)
    >>> memory = get_memory_estimate(model, (80, 100, 128), batch_size=32)
    >>> print(f"Total memory: {memory['total_mb']:.1f} MB")
    >>> print(f"Parameters: {memory['parameters_mb']:.1f} MB")
    """

    # Handle different input shapes
    if len(input_shape) == 3:
        full_shape = (batch_size,) + input_shape
    elif len(input_shape) == 4:
        full_shape = input_shape
    else:
        raise ValueError(
            f"Input shape must be 3D or 4D, got {len(input_shape)}D"
        )

    # Validate shape
    if not validate_input_shape(full_shape):
        raise ValueError(f"Invalid input shape: {full_shape}")

    # Get base autoencoder for analysis
    if isinstance(model, MaskedAutoencoder):
        base_model = model.autoencoder
    else:
        base_model = model

    # Bytes per element based on dtype
    dtype_bytes = {
        torch.float32: 4,
        torch.float16: 2,
        torch.float64: 8,
        torch.int32: 4,
        torch.int64: 8,
    }.get(dtype, 4)

    # Parameter memory
    total_params = sum(p.numel() for p in base_model.parameters())
    param_memory_mb = (total_params * dtype_bytes) / (1024 * 1024)

    # Encoder memory
    encoder_params = sum(p.numel() for p in base_model.encoder.parameters())
    encoder_param_mb = (encoder_params * dtype_bytes) / (1024 * 1024)

    # Decoder memory
    decoder_params = sum(p.numel() for p in base_model.decoder.parameters())
    decoder_param_mb = (decoder_params * dtype_bytes) / (1024 * 1024)

    # Estimate activation memory (rough approximation)
    # This is a simplified estimation - actual memory depends on implementation
    # details

    # Input activation memory
    input_elements = 1
    for dim in full_shape:
        input_elements *= dim
    input_memory_mb = (input_elements * dtype_bytes) / (1024 * 1024)

    # Estimate latent representation size
    try:
        latent_shape = base_model.get_latent_shape(full_shape)
        latent_elements = 1
        for dim in latent_shape:
            latent_elements *= dim
        latent_memory_mb = (latent_elements * dtype_bytes) / (1024 * 1024)
    except:
        # Fallback estimation
        latent_memory_mb = input_memory_mb * 0.1  # Assume 10x compression

    # Rough estimate of intermediate activations (2-3x input + latent)
    activation_memory_mb = input_memory_mb * 2.5 + latent_memory_mb * 1.5

    # Total memory (parameters + activations + gradients during training)
    # Gradients roughly equal parameter memory during training
    total_memory_mb = param_memory_mb * 2 + activation_memory_mb

    return {
        "parameters_mb": param_memory_mb,
        "encoder_params_mb": encoder_param_mb,
        "decoder_params_mb": decoder_param_mb,
        "activations_mb": activation_memory_mb,
        "input_mb": input_memory_mb,
        "latent_mb": latent_memory_mb,
        "total_mb": total_memory_mb,
        "total_params": total_params,
        "dtype": str(dtype),
        "batch_size": full_shape[0],
    }


def analyze_model_architecture(
    model: Union[BlockBasedAutoencoder, MaskedAutoencoder],
    input_shape: tuple[int, ...] = (1, 80, 100, 128),
) -> dict[str, Any]:
    """Analyze model architecture and provide detailed information.

    Parameters
    ----------
    model : BlockBasedAutoencoder or MaskedAutoencoder
        Model to analyze.
    input_shape : tuple of int, default=(1, 80, 100, 128)
        Input shape for analysis.

    Returns
    -------
    dict
        Detailed architecture analysis.
    """
    # Get base autoencoder
    if isinstance(model, MaskedAutoencoder):
        base_model = model.autoencoder
        model_type = "MaskedAutoencoder"
        has_masking = True
        mask_info = {
            "mask_ratio": model.mask_generator.mask_ratio,
            "patch_size": model.mask_generator.patch_size,
            "available_mask_types": MASK_TYPES,
        }
    else:
        base_model = model
        model_type = "BlockBasedAutoencoder"
        has_masking = False
        mask_info = None

    # Basic model info
    total_params = sum(p.numel() for p in base_model.parameters())
    trainable_params = sum(
        p.numel() for p in base_model.parameters() if p.requires_grad
    )

    # Encoder analysis
    encoder_blocks = []
    for i, block in enumerate(base_model.encoder.blocks):
        encoder_blocks.append(
            {
                "block_id": i,
                "in_channels": block.in_channels,
                "out_channels": block.out_channels,
                "pool_size": block.pool_size,
                "dropout": block.dropout_prob,
                "parameters": sum(p.numel() for p in block.parameters()),
            }
        )

    # Decoder analysis
    decoder_blocks = []
    for i, block in enumerate(base_model.decoder.blocks):
        decoder_blocks.append(
            {
                "block_id": i,
                "in_channels": block.in_channels,
                "out_channels": block.out_channels,
                "upsample_factor": block.upsample_factor,
                "dropout": block.dropout_prob,
                "parameters": sum(p.numel() for p in block.parameters()),
            }
        )

    # Shape analysis
    try:
        latent_shape = base_model.get_latent_shape(input_shape)
        output_shape = base_model.get_output_shape(input_shape)
        compression_ratio = (input_shape[2] * input_shape[3]) / (
            latent_shape[2] * latent_shape[3]
        )
    except Exception:
        latent_shape = None
        output_shape = None
        compression_ratio = None

    analysis = {
        "model_type": model_type,
        "has_masking": has_masking,
        "mask_info": mask_info,
        "parameters": {
            "total": total_params,
            "trainable": trainable_params,
            "encoder": sum(p.numel() for p in base_model.encoder.parameters()),
            "decoder": sum(p.numel() for p in base_model.decoder.parameters()),
        },
        "architecture": {
            "input_channels": base_model.input_channels,
            "bottleneck_channels": base_model.encoder.bottleneck_channels,
            "hidden_dim": base_model.encoder.hidden_dim,
            "num_encoder_blocks": len(base_model.encoder.blocks),
            "num_decoder_blocks": len(base_model.decoder.blocks),
            "upsampling_mode": base_model.decoder.upsampling_mode,
        },
        "encoder_blocks": encoder_blocks,
        "decoder_blocks": decoder_blocks,
        "shapes": {
            "input": input_shape,
            "latent": latent_shape,
            "output": output_shape,
            "compression_ratio": compression_ratio,
        },
        "config": base_model.get_config()
        if hasattr(base_model, "get_config")
        else None,
    }

    return analysis


def compare_models(
    models: dict[str, Union[BlockBasedAutoencoder, MaskedAutoencoder]],
    input_shape: tuple[int, ...] = (1, 80, 100, 128),
) -> dict[str, Any]:
    """Compare multiple models side by side.

    Parameters
    ----------
    models : dict
        Dictionary of model_name -> model pairs.
    input_shape : tuple of int
        Input shape for comparison.

    Returns
    -------
    dict
        Comparison results.
    """
    comparison = {"input_shape": input_shape, "models": {}, "summary": {}}

    # Analyze each model
    for name, model in models.items():
        try:
            analysis = analyze_model_architecture(model, input_shape)
            memory = get_memory_estimate(model, input_shape)

            comparison["models"][name] = {
                "analysis": analysis,
                "memory": memory,
                "error": None,
            }
        except Exception as e:
            comparison["models"][name] = {
                "analysis": None,
                "memory": None,
                "error": str(e),
            }

    # Create summary comparison
    successful_models = {
        name: data
        for name, data in comparison["models"].items()
        if data["error"] is None
    }

    if successful_models:
        comparison["summary"] = {
            "parameter_counts": {
                name: data["analysis"]["parameters"]["total"]
                for name, data in successful_models.items()
            },
            "memory_usage": {
                name: data["memory"]["total_mb"]
                for name, data in successful_models.items()
            },
            "compression_ratios": {
                name: data["analysis"]["shapes"]["compression_ratio"]
                for name, data in successful_models.items()
                if data["analysis"]["shapes"]["compression_ratio"] is not None
            },
        }

    return comparison


def print_model_summary(
    model: Union[BlockBasedAutoencoder, MaskedAutoencoder],
    input_shape: tuple[int, ...] = (1, 80, 100, 128),
) -> None:
    """Print a formatted summary of model architecture.

    Parameters
    ----------
    model : BlockBasedAutoencoder or MaskedAutoencoder
        Model to summarize.
    input_shape : tuple of int
        Input shape for analysis.
    """
    analysis = analyze_model_architecture(model, input_shape)
    memory = get_memory_estimate(model, input_shape)

    print(f"=== {analysis['model_type']} Summary ===")
    print(f"Total Parameters: {analysis['parameters']['total']:,}")
    print(f"Trainable Parameters: {analysis['parameters']['trainable']:,}")
    print(f"Memory Usage: {memory['total_mb']:.1f} MB")

    if analysis["has_masking"]:
        print(f"Mask Ratio: {analysis['mask_info']['mask_ratio']:.2f}")
        print(f"Patch Size: {analysis['mask_info']['patch_size']}")

    print("\nArchitecture:")
    print(f"  Input Channels: {analysis['architecture']['input_channels']}")
    print(
        f"  Bottleneck Channels: "
        f"{analysis['architecture']['bottleneck_channels']}"
    )
    print(
        f"  Encoder Blocks: {analysis['architecture']['num_encoder_blocks']}"
    )
    print(
        f"  Decoder Blocks: {analysis['architecture']['num_decoder_blocks']}"
    )

    if analysis["shapes"]["compression_ratio"]:
        print(
            f"  Compression Ratio: "
            f"{analysis['shapes']['compression_ratio']:.1f}x"
        )

    print("\nShapes:")
    print(f"  Input: {analysis['shapes']['input']}")
    print(f"  Latent: {analysis['shapes']['latent']}")
    print(f"  Output: {analysis['shapes']['output']}")


# Example usage and testing
if __name__ == "__main__":
    # Test utility functions
    print("Testing model utilities...")

    # Test model creation
    mae = create_mae_model("mae_default", input_channels=80)
    autoencoder = create_block_autoencoder("light", input_channels=80)

    print(f"Created MAE: {type(mae).__name__}")
    print(f"Created Autoencoder: {type(autoencoder).__name__}")

    # Test model info
    info = get_model_info()
    print(f"\nAvailable configs: {info['available_configs']}")
    print(f"MAE presets: {info['mae_presets']}")

    # Test validation
    valid_shape = (32, 80, 100, 128)
    invalid_shape = (80, 100, 128)
    print("\nShape validation:")
    print(f"  {valid_shape}: {validate_input_shape(valid_shape)}")
    print(f"  {invalid_shape}: {validate_input_shape(invalid_shape)}")

    # Test memory estimation
    memory = get_memory_estimate(mae, (80, 100, 128), batch_size=32)
    print("\nMemory estimate for MAE:")
    print(f"  Total: {memory['total_mb']:.1f} MB")
    print(f"  Parameters: {memory['parameters_mb']:.1f} MB")
    print(f"  Activations: {memory['activations_mb']:.1f} MB")

    # Test architecture analysis
    print("\nModel summary:")
    print_model_summary(mae, (1, 80, 100, 128))

    # Test model comparison
    models = {"mae": mae, "autoencoder": autoencoder}
    comparison = compare_models(models)
    print("\nModel comparison:")
    for name, params in comparison["summary"]["parameter_counts"].items():
        memory = comparison["summary"]["memory_usage"][name]
        print(f"  {name}: {params:,} params, {memory:.1f} MB")

    print("Utility tests completed successfully!")
