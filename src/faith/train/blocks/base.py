"""Base classes and utilities for neural network blocks.

This module provides abstract base classes, common utilities, and shared
functionality that can be inherited by specific block implementations.
It ensures consistency across different block types and provides common
patterns for initialization, forward passes, and configuration.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Union, Any, Optional
import math


class BaseConvBlock(nn.Module, ABC):
    """Abstract base class for all convolutional-based neural network blocks.

    This class defines the common interface that all blocks should implement,
    ensuring consistency across different block types in the autoencoder
    architecture.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple of int, default=3
        Kernel size for convolutions.
    bias : bool, default=True
        Whether to use bias in convolution layers.

    Attributes
    ----------
    in_channels : int
        Stored input channel count.
    out_channels : int
        Stored output channel count.
    kernel_size : int or tuple
        Stored kernel size.
    bias : bool
        Stored bias setting.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, tuple[int, int]] = 3,
            bias: bool = True,
            **kwargs
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError(
                f"in_channels must be positive, got {in_channels}")
        if out_channels <= 0:
            raise ValueError(
                f"out_channels must be positive, got {out_channels}")

        if not isinstance(in_channels, int):
            raise TypeError(
                f"in_channels must be an int, got {type(in_channels)}")

        if not isinstance(out_channels, int):
            raise TypeError(
                f"out_channels must be an int, got {type(out_channels)}")

        if isinstance(kernel_size, int) and kernel_size <= 0:
            raise ValueError(
                f"kernel_size must be positive, got {kernel_size}")
        if isinstance(kernel_size, tuple) and any(k <= 0 for k in kernel_size):
            raise ValueError(
                f"kernel_size must be positive, got {kernel_size}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = self._normalize_kernel_size(kernel_size)
        self.bias = bias

    @staticmethod
    def _normalize_kernel_size(
            kernel_size: Union[int, tuple[int, int]]
    ) -> tuple[int, int]:
        """Normalize kernel size to tuple format."""
        if isinstance(kernel_size, int):
            return (kernel_size, kernel_size)
        return kernel_size

    @staticmethod
    def _calculate_padding(
            kernel_size: Union[int, tuple[int, int]],
            padding: Union[int, tuple[int, int], str]
    ) -> tuple[int, ...]:
        """Calculate padding based on kernel size and padding specification."""
        if padding == 'auto':
            if isinstance(kernel_size, int):
                return kernel_size // 2, kernel_size // 2
            else:
                return tuple(k // 2 for k in kernel_size)
        elif isinstance(padding, int):
            return padding, padding
        else:
            return padding

    @abstractmethod
    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        pass

    def get_config(self) -> dict[str, Any]:
        """
        Get configuration dictionary for this block.

        Returns
        -------
        dict
            Configuration dictionary containing all parameters needed
            to reconstruct this block.
        """
        return {
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'bias': self.bias,
        }

    @property
    def parameter_count(self) -> int:
        """Get total number of trainable parameters in this block."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"in_channels={self.in_channels}, "
                f"out_channels={self.out_channels}, "
                f"kernel_size={self.kernel_size}, "
                f"bias={self.bias})")


class SequentialBlock(BaseConvBlock):
    """Base class for blocks that apply operations sequentially.

    This class provides common functionality for blocks that consist of
    multiple sequential operations (like EncoderBlock and DecoderBlock).

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    operations : list of nn.Module
        List of operations to apply sequentially.
    **kwargs
        Additional arguments passed to BaseBlock.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            operations: Optional[list[nn.Module]] = None,
            **kwargs
    ) -> None:
        super().__init__(in_channels, out_channels, **kwargs)

        if operations is None:
            operations = []

        self.operations = nn.Sequential(*operations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through sequential operations."""
        return self.operations(x)

    def add_operation(self, operation: nn.Module) -> None:
        """Add an operation to the sequential block."""
        self.operations.add_module(str(len(self.operations)), operation)


class ConfigurableBlock(BaseConvBlock):
    """
    Base class for blocks with extensive configuration options.

    This class provides utilities for blocks that need to handle complex
    configuration dictionaries and parameter validation.
    """

    def __init__(self, **kwargs) -> None:
        # Extract base parameters
        in_channels = kwargs.pop('in_channels')
        out_channels = kwargs.pop('out_channels')
        kernel_size = kwargs.pop('kernel_size', 3)
        bias = kwargs.pop('bias', True)

        super().__init__(in_channels, out_channels, kernel_size, bias)

        # Store additional configuration
        self._config = kwargs

    def get_config(self) -> dict[str, Any]:
        """Get full configuration including additional parameters."""
        config = super().get_config()
        config.update(self._config)
        return config

    @classmethod
    def from_config(
            cls,
            config: dict[str, Any]
    ) -> 'ConfigurableBlock':
        """Create block instance from configuration dictionary."""
        return cls(**config)


class WeightInitializer:
    """Utilities for weight initialization in blocks."""

    @staticmethod
    def xavier_uniform_(module: nn.Module) -> None:
        """Apply Xavier uniform initialization to conv and linear layers."""
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @staticmethod
    def kaiming_normal_(module: nn.Module) -> None:
        """Apply Kaiming normal initialization to conv and linear layers."""
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out',
                                    nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @staticmethod
    def init_batch_norm_(module: nn.Module) -> None:
        """Initialize batch normalization layers."""
        if isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


class BlockUtils:
    """Utility functions for working with blocks."""

    @staticmethod
    def calculate_output_shape(
            input_shape: tuple[int, ...],
            kernel_size: Union[int, tuple[int, int]],
            stride: Union[int, tuple[int, int]] = 1,
            padding: Union[int, tuple[int, int]] = 0,
            dilation: Union[int, tuple[int, int]] = 1
    ) -> tuple[int, ...]:
        """Calculate output shape after convolution operation."""
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        batch_size, channels = input_shape[:2]
        height, width = input_shape[2:]

        out_height = math.floor(
            (height + 2 * padding[0]
             - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1
        )
        out_width = math.floor(
            (width + 2 * padding[1]
             - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1
        )

        return batch_size, channels, out_height, out_width

    @staticmethod
    def count_parameters(
            block: nn.Module,
            trainable_only: bool = True
    ) -> int:
        """Count parameters in a block."""
        if trainable_only:
            return sum(
                p.numel() for p in block.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in block.parameters())

    @staticmethod
    def get_memory_usage(
            block: nn.Module,
            input_shape: tuple[int, ...]
    ) -> dict[str, float]:
        """Estimate memory usage of a block."""
        # This is a simplified estimation
        # 4 bytes per float32
        param_memory = BlockUtils.count_parameters(block) * 4

        # Estimation of activation memory
        output_elements = math.prod(input_shape)
        # 4 bytes per float32
        activation_memory = output_elements * 4

        return {
            'parameters_mb': param_memory / (1024 * 1024),
            'activations_mb': activation_memory / (1024 * 1024),
            'total_mb': (param_memory + activation_memory) / (1024 * 1024)
        }
