"""Residual block implementation derived from base classes.

This module implements the ResidualBlock class that inherits from BaseBlock,
following the established patterns and interfaces defined in the base module.
"""

from typing import Any, Union

import torch
import torch.nn as nn

from .base import BaseConvBlock, WeightInitializer


class ResidualBlock(BaseConvBlock):
    """Residual convolutional block with batch normalization and ReLU.

    This block implements a standard residual connection with two convolutional
    layers, batch normalization, and ReLU activation. It includes an optional
    projection layer for dimension matching in the skip connection.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple of int, default=3
        Size of the convolving kernel.
    stride : int or tuple of int, default=1
        Stride of the convolution.
    bias : bool, default=True
        If True, adds a learnable bias to the output.
    use_batch_norm : bool, default=True
        Whether to use batch normalization layers.
    activation : str, default='relu'
        Activation function to use ('relu', 'leaky_relu', 'gelu', etc.).
    init_method : str, default='kaiming'
        Weight initialization method ('kaiming', 'xavier', 'default').

    Attributes
    ----------
    conv1 : torch.nn.Conv2d
        First convolutional layer.
    batch_norm_1 : torch.nn.BatchNorm2d or None
        First batch normalization layer.
    activation_fn : torch.nn.Module
        Activation function.
    conv2 : torch.nn.Conv2d
        Second convolutional layer.
    batch_norm_2 : torch.nn.BatchNorm2d or None
        Second batch normalization layer.
    skip_conv : torch.nn.Conv2d or None
        Optional 1x1 convolution for dimension matching.
    stride : tuple of int
        Stored stride values.
    padding : tuple of int
        Stored padding values.

    Examples
    --------
    >>> block = ResidualBlock(64, 128)
    >>> x = torch.randn(1, 64, 32, 32)
    >>> out = block(x)
    >>> print(out.shape)
    torch.Size([1, 128, 32, 32])

    >>> # Custom configuration
    >>> block = ResidualBlock(64, 128, stride=2, activation='gelu')
    >>> config = block.get_config()
    >>> new_block = ResidualBlock.from_config(config)
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, tuple[int, int]] = 3,
            stride: Union[int, tuple[int, int]] = 1,
            bias: bool = True,
            use_batch_norm: bool = True,
            activation: str = 'relu',
            init_method: str = 'kaiming'
    ) -> None:
        """Initialize ResidualBlock.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        kernel_size : int or tuple of int, default=3
            Size of the convolving kernel.
        stride : int or tuple of int, default=1
            Stride of the convolution.
        bias : bool, default=True
            Whether to use bias in convolutions.
        use_batch_norm : bool, default=True
            Whether to use batch normalization.
        activation : str, default='relu'
            Activation function name.
        init_method : str, default='kaiming'
            Weight initialization method.
        """
        # Initialize base class
        super().__init__(in_channels, out_channels, kernel_size, bias)

        if isinstance(stride, int) and stride < 1:
            raise ValueError(f"Stride must be a positive integer or tuple, "
                             f"got {stride}")
        if isinstance(stride, tuple) and any(s < 1 for s in stride):
            raise ValueError(f"Stride must be a positive integer or tuple, "
                             f"got {stride}")
        if (isinstance(stride, float) or isinstance(stride, tuple)
                and any(isinstance(s, float) for s in stride)):
            raise TypeError(f"Stride must be an integer or tuple, "
                            f"got float {stride}")

        # Normalize stride and padding
        self.stride = self._normalize_stride(stride)
        self.padding = self._calculate_padding(self.kernel_size, "auto")
        self.use_batch_norm = use_batch_norm
        self.activation_name = activation
        self.init_method = init_method

        # Validate parameters
        self._validate_parameters()

        # Build the block layers
        self._build_layers()

        # Initialize weights
        self._initialize_weights()

    def _normalize_stride(self,
                          stride: Union[int, tuple[int, int]]
                          ) -> tuple[int, int]:
        """Normalize stride to tuple format."""
        if isinstance(stride, int):
            return (stride, stride)
        return stride

    def _validate_parameters(self) -> None:
        """Validate input parameters."""
        valid_activations = {'tanh', 'sigmoid', 'relu', 'leaky_relu', 'gelu',
                             'swish', 'mish'}
        if self.activation_name not in valid_activations:
            raise ValueError(f"activation must be one of {valid_activations}, "
                             f"got {self.activation_name}")

        valid_init_methods = {'kaiming', 'xavier', 'default'}
        if self.init_method not in valid_init_methods:
            raise ValueError(f"init_method must be one of {valid_init_methods}"
                             f", got {self.init_method}")

    def _build_layers(self) -> None:
        """Build the convolutional layers and other components."""
        # First convolutional layer
        self.conv1 = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=self.bias and not self.use_batch_norm
            # No bias if using batch norm
        )

        # First batch normalization (optional)
        if self.use_batch_norm:
            self.batch_norm_1 = nn.BatchNorm2d(self.out_channels)
        else:
            self.batch_norm_1 = None

        # Activation function
        self.activation_fn = self._create_activation()

        # Second convolutional layer (always stride=1 to maintain dimensions)
        self.conv2 = nn.Conv2d(
            self.out_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding,
            bias=self.bias and not self.use_batch_norm
        )

        # Second batch normalization (optional)
        if self.use_batch_norm:
            self.batch_norm_2 = nn.BatchNorm2d(self.out_channels)
        else:
            self.batch_norm_2 = None

        # Projection for skip connection if dimensions don't match
        if self.in_channels != self.out_channels or self.stride != (1, 1):
            self.skip_conv = nn.Conv2d(
                self.in_channels,
                self.out_channels,
                kernel_size=1,
                stride=self.stride,
                padding=0,
                bias=self.bias and not self.use_batch_norm
            )
            if self.use_batch_norm:
                self.skip_batch_norm = nn.BatchNorm2d(self.out_channels)
            else:
                self.skip_batch_norm = None
        else:
            self.skip_conv = None
            self.skip_batch_norm = None

    def _create_activation(self) -> nn.Module:
        """Create activation function based on name."""
        activations = {
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'relu': nn.ReLU(inplace=True),
            'leaky_relu': nn.LeakyReLU(0.1, inplace=True),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),  # SiLU is the same as Swish
            'mish': nn.Mish(),
        }
        return activations[self.activation_name]

    def _initialize_weights(self) -> None:
        """Initialize weights according to the specified method."""
        if self.init_method == 'kaiming':
            self.apply(WeightInitializer.kaiming_normal_)
        elif self.init_method == 'xavier':
            self.apply(WeightInitializer.xavier_uniform_)
        elif self.init_method == 'default':
            pass  # Use PyTorch's default initialization

        # Always properly initialize batch norm
        if self.use_batch_norm:
            self.apply(WeightInitializer.init_batch_norm_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the residual block.

        TODO: What does ResNet do for initializing the residual connections?

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch_size, in_channels, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor with shape
            (batch_size, out_channels, height', width') where height'
            and 'width' depend on stride.
        """
        # Store input for residual connection
        residual = x

        # First conv block
        out = self.conv1(x)
        if self.batch_norm_1 is not None:
            out = self.batch_norm_1(out)
        out = self.activation_fn(out)

        # Second conv block
        out = self.conv2(out)
        if self.batch_norm_2 is not None:
            out = self.batch_norm_2(out)

        # Apply skip connection with optional projection
        if self.skip_conv is not None:
            residual = self.skip_conv(residual)
            if self.skip_batch_norm is not None:
                residual = self.skip_batch_norm(residual)

        # Add residual connection
        out += residual

        # Final activation
        out = self.activation_fn(out)

        return out

    def get_config(self) -> dict[str, Any]:
        """Get configuration dictionary for this block.

        Returns
        -------
        dict
            Configuration dictionary containing all parameters needed
            to reconstruct this block.
        """
        config = super().get_config()
        config.update({
            'stride': self.stride,
            'padding': self.padding,
            'use_batch_norm': self.use_batch_norm,
            'activation': self.activation_name,
            'init_method': self.init_method,
        })
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> 'ResidualBlock':
        """Create ResidualBlock instance from configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration dictionary.

        Returns
        -------
        ResidualBlock
            New ResidualBlock instance.
        """
        return cls(**config)

    def __repr__(self) -> str:
        """String representation of the ResidualBlock."""
        return (f"ResidualBlock("
                f"in_channels={self.in_channels}, "
                f"out_channels={self.out_channels}, "
                f"kernel_size={self.kernel_size}, "
                f"stride={self.stride}, "
                f"padding={self.padding}, "
                f"bias={self.bias}, "
                f"use_batch_norm={self.use_batch_norm}, "
                f"activation='{self.activation_name}')")

    @property
    def has_skip_connection(self) -> bool:
        """Check if this block has a skip connection projection."""
        return self.skip_conv is not None

    def get_output_shape(
            self,
            input_shape: tuple[int, ...]
    ) -> tuple[int, ...]:
        """Calculate output shape given input shape.

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
            padding=self.padding
        )

        # Update channels
        batch_size, _, height, width = temp_shape
        return batch_size, self.out_channels, height, width
