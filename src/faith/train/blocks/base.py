"""Base classes and utilities for neural network blocks.

This module provides abstract base classes, common utilities, and shared
functionality that can be inherited by specific block implementations.
It ensures consistency across different block types and provides common
patterns for initialization, forward passes, and configuration.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import torch
import torch.nn as nn


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
        kernel_size: int | tuple[int, int] = 3,
        bias: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}")

        if not isinstance(in_channels, int):
            raise TypeError(f"in_channels must be an int, got {type(in_channels)}")

        if not isinstance(out_channels, int):
            raise TypeError(f"out_channels must be an int, got {type(out_channels)}")

        if isinstance(kernel_size, int) and kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")
        if isinstance(kernel_size, tuple) and any(k <= 0 for k in kernel_size):
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = self._normalize_kernel_size(kernel_size)
        self.bias = bias

    @staticmethod
    def _normalize_kernel_size(
        kernel_size: int | tuple[int, int],
    ) -> tuple[int, int]:
        """Normalize kernel size to tuple format."""
        if isinstance(kernel_size, int):
            return kernel_size, kernel_size
        return kernel_size

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "kernel_size": self.kernel_size,
            "bias": self.bias,
        }

    @property
    def parameter_count(self) -> int:
        """Get total number of trainable parameters in this block."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, "
            f"bias={self.bias})"
        )


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
        operations: list[nn.Module] | None = None,
        **kwargs: Any,
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
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
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
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
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
            (height + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1)
            / stride[0]
            + 1
        )
        out_width = math.floor(
            (width + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1)
            / stride[1]
            + 1
        )

        return batch_size, channels, out_height, out_width

    @staticmethod
    def count_parameters(block: nn.Module, trainable_only: bool = True) -> int:
        """Count parameters in a block."""
        if trainable_only:
            return sum(p.numel() for p in block.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in block.parameters())

    @staticmethod
    def get_memory_usage(
        block: nn.Module, input_shape: tuple[int, ...]
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
            "parameters_mb": param_memory / (1024 * 1024),
            "activations_mb": activation_memory / (1024 * 1024),
            "total_mb": (param_memory + activation_memory) / (1024 * 1024),
        }


class _Identity(nn.Module):
    """Identity block that returns the input tensor unchanged."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Identity function that returns the input tensor unchanged."""
        return x


def _calculate_padding(
    kernel_size: int | tuple[int, int],
    padding: int | tuple[int, int] | str = "auto",
) -> int | tuple[int, int]:
    """Calculate padding based on kernel size and padding specification."""
    if isinstance(kernel_size, int) and kernel_size % 2 == 0:
        raise ValueError(f"Kernel size must be odd, got {kernel_size} (even).")
    if isinstance(kernel_size, tuple) and any(k % 2 == 0 for k in kernel_size):
        raise ValueError(f"Kernel size must be odd, got {kernel_size} (even).")
    if padding == "auto":
        if isinstance(kernel_size, int):
            return kernel_size // 2
        else:
            return tuple(k // 2 for k in kernel_size)
    return padding


def _create_activation(activation_name: str = "relu") -> nn.Module:
    """Create activation function based on name."""
    activations = {
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "relu": nn.ReLU(inplace=True),
        "leaky_relu": nn.LeakyReLU(0.1, inplace=True),
        "gelu": nn.GELU(),
        "swish": nn.SiLU(),  # SiLU is the same as Swish
        "mish": nn.Mish(),
    }
    return activations[activation_name]


# Kouroche's implementation.
class _ResidualBlock(ABC, nn.Module):
    """
    Abstract base class for residual blocks in neural networks.
    """

    _conv_type: ClassVar[
        type[nn.Conv1d]
        | type[nn.Conv2d]
        | type[nn.ConvTranspose1d]
        | type[nn.ConvTranspose2d]
    ]
    _norm_type: ClassVar[type[nn.BatchNorm1d] | type[nn.BatchNorm2d] | None]

    conv_1: nn.Sequential
    conv_2: nn.Sequential
    mixing: nn.Sequential | None

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        bias: bool = True,
        activation_name: str = "relu",
        weight_init_method: str = "kaiming",
    ) -> None:
        super().__init__()

        if isinstance(kernel_size, int) and kernel_size % 2 == 0:
            raise ValueError(f"Kernel size must be odd, got {kernel_size} (even).")
        if isinstance(kernel_size, tuple) and any(k % 2 == 0 for k in kernel_size):
            raise ValueError(f"Kernel size must be odd, got {kernel_size} (even).")
        if isinstance(kernel_size, int) and kernel_size <= 0:
            raise ValueError(f"Kernel size must be positive, got {kernel_size}.")
        if isinstance(kernel_size, tuple) and any(k <= 0 for k in kernel_size):
            raise ValueError(f"Kernel size must be positive, got {kernel_size}.")

        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}.")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}.")

        if not isinstance(in_channels, int):
            raise TypeError(f"in_channels must be an int, got {type(in_channels)}.")
        if not isinstance(out_channels, int):
            raise TypeError(f"out_channels must be an int, got {type(out_channels)}.")

        if isinstance(stride, int) and stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}.")
        if isinstance(stride, tuple) and any(s <= 0 for s in stride):
            raise ValueError(f"stride must be positive, got {stride}.")
        if not isinstance(stride, int) and not isinstance(stride, tuple):
            raise TypeError(f"stride must be an int or tuple, got {type(stride)}.")

        self.padding = _calculate_padding(kernel_size, padding="auto")
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.out_channels = out_channels

        self.conv_1 = nn.Sequential(
            self._conv_type(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=self.padding,
                bias=bias,
            ),
            (
                self._norm_type(out_channels)
                if self._norm_type is not None
                else _Identity()
            ),
            _create_activation(activation_name),
        )
        self.conv_2 = nn.Sequential(
            self._conv_type(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=self.padding,
                bias=bias,
            ),
            (
                self._norm_type(out_channels)
                if self._norm_type is not None
                else _Identity()
            ),
        )

        if in_channels != out_channels or stride != 1:
            self.mixing = nn.Sequential(
                self._conv_type(
                    in_channels, out_channels, kernel_size=1, stride=stride
                ),
                (
                    self._norm_type(out_channels)
                    if self._norm_type is not None
                    else _Identity()
                ),
            )
        else:
            self.mixing = None

        self.final_activation = _create_activation(activation_name)
        self.init_method = weight_init_method
        self._initialize_weights()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the residual block.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor of shape (batch, channels, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor after applying the residual block.
        """
        residual = input

        output = self.conv_1(input)
        output = self.conv_2(output)

        if self.mixing:
            residual = self.mixing(residual)

        output = output + residual
        output = self.final_activation(output)
        return output

    def _initialize_weights(self) -> None:
        """Initialize weights according to the specified method."""
        if self.init_method == "kaiming":
            self.apply(WeightInitializer.kaiming_normal_)
        elif self.init_method == "xavier":
            self.apply(WeightInitializer.xavier_uniform_)
        elif self.init_method == "default":
            pass  # Use PyTorch's default initialization

        # Always properly initialize batch norm
        if self._norm_type is not None:
            self.apply(WeightInitializer.init_batch_norm_)

    def __repr__(self) -> str:
        """String representation of the ResidualBlock."""
        return (
            f"ResidualBlock("
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, "
            f"stride={self.stride}, "
            f"bias={self.bias}, "
            f"activation_name='{self.activation_name}', "
            f"weight_init_method={self.init_method})"
        )

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """
        Calculate output shape given input shape.

        Parameters
        ----------
        input_shape : tuple
            Input tensor shape (batch, channels, height, width).

        Returns
        -------
        tuple
            Output tensor shape.
        """
        from src.faith.train.blocks import BlockUtils

        # Account for stride in the first convolution
        temp_shape = BlockUtils.calculate_output_shape(
            input_shape,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )

        # Update channels
        batch_size, _, height, width = temp_shape
        return batch_size, self.out_channels, height, width
