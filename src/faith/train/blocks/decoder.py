"""Decoder block implementations derived from base classes.

This module implements the DecoderBlock and BlockBasedDecoder classes that
inherit from the base classes, following established patterns and interfaces.
The decoder creates a symmetric reconstruction path to the encoder.
"""

from typing import Any, Union

import torch
import torch.nn as nn

from .base import SequentialBlock
from .encoder import BlockBasedEncoder
from .residual import ResidualBlock


class DecoderBlock(SequentialBlock):
    """
    Single decoder block: ConvTranspose2d + Dropout + ResidualBlock.

    This block represents the fundamental building unit of the decoder,
    combining spatial upsampling through ConvTranspose2d, regularization
    through Dropout, and feature refinement through ResidualBlock.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels from the ResidualBlock.
    upsample_factor : tuple of int, default=(1, 2)
        Scale factor for upsampling. Format: (height_factor, width_factor).
    kernel_size : int or tuple of int, default=3
        Kernel size for convolutions in ResidualBlock.
    stride : int or tuple of int, default=1
        Stride for convolutions in ResidualBlock. The DecoderBlock uses
        stride=1 and relies on ConvTranspose2d for upsampling.
    dropout : float, default=0.3
        Dropout probability. Must be between 0.0 and 1.0.
    bias : bool, default=True
        Whether to use bias in convolution layers.
    use_batch_norm : bool, default=True
        Whether to use batch normalization in ResidualBlock.
    activation : str, default='relu'
        Activation function for ResidualBlock.
    residual_init_method : str, default='kaiming'
        Weight initialization method for ResidualBlock.

    Attributes
    ----------
    conv_transpose : nn.ConvTranspose2d
        Transposed convolution layer for spatial upsampling.
    dropout_layer : nn.Dropout
        Dropout layer for regularization.
    residual_block : ResidualBlock
        The residual convolutional block for feature refinement.
    upsample_factor : tuple of int
        Stored upsampling factor for encoder symmetry.
    dropout : float
        Stored dropout probability.

    Examples
    --------
    >>> block = DecoderBlock(in_channels=128, out_channels=64)
    >>> x = torch.randn(1, 128, 16, 8)
    >>> out = block(x)
    >>> print(out.shape)
    torch.Size([1, 64, 16, 16])

    >>> # Custom configuration
    >>> block = DecoderBlock(
    ...     in_channels=128, out_channels=64,
    ...     upsample_factor=(2, 2), dropout=0.5, activation='gelu'
    ... )
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upsample_factor: tuple[int, int] = (1, 2),
        kernel_size: Union[int, tuple[int, int]] = 3,
        stride: Union[int, tuple[int, int]] = 1,
        dropout: float = 0.3,
        bias: bool = True,
        use_batch_norm: bool = True,
        activation: str = "relu",
        residual_init_method: str = "kaiming",
    ) -> None:
        """Initialize DecoderBlock."""

        # Validate parameters
        if not 0.0 <= dropout <= 1.0:
            raise ValueError(
                f"Dropout must be between 0.0 and 1.0, got {dropout}"
            )

        if len(upsample_factor) != 2:
            raise ValueError(f"upsample_factor must be a tuple of length 2, "
                             f"got {upsample_factor}")

        # Store configuration
        self.upsample_factor = upsample_factor
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.activation_name = activation
        self.residual_init_method = residual_init_method

        # Build the sequential operations
        operations = self._build_operations(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            bias,
            use_batch_norm,
            activation,
            residual_init_method,
        )

        # Initialize SequentialBlock with operations
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            operations=operations,
            kernel_size=kernel_size,
            bias=bias,
        )

        # Store individual components for introspection
        self.conv_transpose = self.operations[0]
        self.dropout_layer = self.operations[1]
        self.residual_block = self.operations[2]

    def _build_operations(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple[int, int]],
        stride: Union[int, tuple[int, int]],
        bias: bool,
        use_batch_norm: bool,
        activation: str,
        init_method: str,
    ) -> list[nn.Module]:
        """Build the list of operations for this decoder block."""

        operations = []

        # 1. ConvTranspose2d for upsampling
        # Calculate kernel size and padding based on upsample factor
        # For stride=1 in any dimension, we need to handle it carefully

        # Determine kernel size and padding for each dimension
        kernel_h = 3 if self.upsample_factor[0] == 1 else 4
        kernel_w = 3 if self.upsample_factor[1] == 1 else 4

        conv_transpose_layer = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels,  # Keep same channels for upsampling
            kernel_size=(kernel_h, kernel_w),
            stride=self.upsample_factor,
            padding=(1, 1),  # Use padding=1 for both cases
            bias=bias,
        )
        operations.append(conv_transpose_layer)

        # 2. Dropout for regularization
        dropout_layer = nn.Dropout(p=self.dropout)
        operations.append(dropout_layer)

        # 3. ResidualBlock for feature refinement
        residual_block = ResidualBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
            use_batch_norm=use_batch_norm,
            activation=activation,
            init_method=init_method,
        )
        operations.append(residual_block)

        return operations

    def get_config(self) -> dict[str, Any]:
        """Get configuration dictionary for this block."""
        config = super().get_config()
        config.update(
            {
                "upsample_factor": self.upsample_factor,
                "dropout": self.dropout,
                "use_batch_norm": self.use_batch_norm,
                "activation": self.activation_name,
                "residual_init_method": self.residual_init_method,
                "stride": getattr(self.residual_block, "stride", 1),
            }
        )
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DecoderBlock":
        """Create DecoderBlock instance from configuration dictionary."""
        return cls(**config)

    def get_output_shape(
        self, input_shape: tuple[int, ...]
    ) -> tuple[int, ...]:
        """Calculate output shape given input shape."""

        # Get shape after ConvTranspose2d upsampling
        batch_size, channels, height, width = input_shape

        # Calculate upsampled dimensions using ConvTranspose2d formula
        # output_size = (input_size - 1) * stride - 2 * padding + kernel_size

        # Use appropriate kernel for each dimension based on upsample factor
        kernel_h = 3 if self.upsample_factor[0] == 1 else 4
        kernel_w = 3 if self.upsample_factor[1] == 1 else 4
        padding = 1  # Always use padding=1

        upsampled_height = (
            (height - 1) * self.upsample_factor[0] - 2 * padding + kernel_h
        )
        upsampled_width = (
            (width - 1) * self.upsample_factor[1] - 2 * padding + kernel_w
        )

        # Shape after ConvTranspose2d (channels remain the same)
        conv_transpose_output_shape = (
            batch_size,
            channels,
            upsampled_height,
            upsampled_width,
        )

        # Apply ResidualBlock (changes channels to out_channels)
        residual_output_shape = self.residual_block.get_output_shape(
            conv_transpose_output_shape
        )

        return residual_output_shape

    def __repr__(self) -> str:
        """String representation of the DecoderBlock."""
        return (
            f"DecoderBlock("
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"upsample_factor={self.upsample_factor}, "
            f"dropout={self.dropout}, "
            f"activation='{self.activation_name}')"
        )


class BlockBasedDecoder(SequentialBlock):
    """Decoder architecture built from a sequence of DecoderBlocks.

    This decoder provides a flexible architecture where each decoding stage
    consists of a DecoderBlock with configurable parameters. The blocks are
    automatically chained together with matching input/output channels.

    Parameters
    ----------
    in_channels : int
        Number of input channels (typically from encoder bottleneck).
    block_configs : list of dict
        Configuration for each decoder block. Each dict should contain:
        - 'out_channels' (int): Output channels for the block (required)
        - 'upsample_factor' (tuple, optional): Upsampling factor,
           default (1, 2)
        - 'dropout' (float, optional): Dropout probability, default 0.3
        - 'kernel_size' (int/tuple, optional): Conv kernel size, default 3
        - 'activation' (str, optional): Activation function, default 'relu'
        - 'use_batch_norm' (bool, optional): Use batch norm, default True
        - 'bias' (bool, optional): Use bias in convolutions, default True

    Attributes
    ----------
    blocks : list of DecoderBlock
        List of DecoderBlock modules (accessed via self.operations).
    block_configs : list of dict
        Stored block configurations.

    Examples
    --------
    >>> # Simple 3-block decoder (reverse of encoder)
    >>> configs = [
    ...     {'out_channels': 128, 'upsample_factor': (2, 2)},
    ...     {'out_channels': 64, 'upsample_factor': (1, 2)},
    ...     {'out_channels': 3}  # Final output channels
    ... ]
    >>> decoder = BlockBasedDecoder(in_channels=256, block_configs=configs)
    >>> z = torch.randn(1, 256, 8, 4)
    >>> output = decoder(z)

    >>> # Create decoder that mirrors an encoder
    >>> encoder_configs = [
    ...     {'out_channels': 64},
    ...     {'out_channels': 128, 'pool_size': (2, 2)},
    ...     {'out_channels': 256, 'pool_size': (1, 2)}
    ... ]
    >>> decoder_configs = BlockBasedDecoder.reverse_encoder_configs(
    ...     encoder_configs, final_out_channels=3
    ... )
    >>> decoder = BlockBasedDecoder(in_channels=256,
    ...     block_configs=decoder_configs)
    """

    def __init__(
        self,
        in_channels: int,
        block_configs: list[dict[str, Any]],
        kernel_size: Union[int, tuple[int, int]] = 3,
        bias: bool = True,
    ) -> None:
        """Initialize BlockBasedDecoder."""

        # Validate inputs
        if not block_configs:
            raise ValueError("block_configs cannot be empty")

        if in_channels <= 0:
            raise ValueError(
                f"in_channels must be positive, got {in_channels}"
            )

        # Validate that all configs have out_channels
        for i, config in enumerate(block_configs):
            if "out_channels" not in config:
                raise ValueError(
                    f"Block {i} missing required 'out_channels' key"
                )
            if config["out_channels"] <= 0:
                raise ValueError(f"out_channels must be positive, "
                                 f"got {config['out_channels']} in block {i}")

        self.block_configs = block_configs

        # Build decoder blocks
        operations = self._build_decoder_blocks(in_channels, kernel_size, bias)

        # Get final output channels from last block
        final_out_channels = block_configs[-1]["out_channels"]

        # Initialize SequentialBlock with operations
        super().__init__(
            in_channels=in_channels,
            out_channels=final_out_channels,
            operations=operations,
            kernel_size=kernel_size,
            bias=bias,
        )

    def _build_decoder_blocks(
            self,
            in_channels: int,
            default_kernel_size: Union[int, tuple[int, int]],
            default_bias: bool,
    ) -> list[nn.Module]:
        """
        Build the sequence of decoder blocks with automatic channel chaining.
        """
        blocks = []
        current_channels = in_channels

        for i, config in enumerate(self.block_configs):
            # Prepare block configuration with defaults
            block_config = {
                "in_channels": current_channels,
                "out_channels": config["out_channels"],
                "upsample_factor": config.get("upsample_factor", (1, 2)),
                "kernel_size": config.get("kernel_size", default_kernel_size),
                "stride": config.get("stride", 1),
                "dropout": config.get("dropout", 0.3),
                "bias": config.get("bias", default_bias),
                "use_batch_norm": config.get("use_batch_norm", True),
                "activation": config.get("activation", "relu"),
                "residual_init_method": config.get(
                    "residual_init_method", "kaiming"
                ),
            }

            # Create decoder block
            block = DecoderBlock(**block_config)
            blocks.append(block)

            # Update current channels for next block
            current_channels = config["out_channels"]

        return blocks

    def get_feature_maps(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Get intermediate feature maps from each decoder block.

        Useful for visualization, debugging, and skip connections.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        list of torch.Tensor
            Feature maps after each decoder block.
        """
        feature_maps = []

        for block in self.operations:
            x = block(x)
            feature_maps.append(x.clone())

        return feature_maps

    def get_channel_progression(self) -> list[int]:
        """Get the channel count progression through the decoder.

        Returns
        -------
        list of int
            Channel counts: [in_channels, block1_out, block2_out, ...]
        """
        channels = [self.in_channels]
        for config in self.block_configs:
            channels.append(config["out_channels"])
        return channels

    def get_config(self) -> dict[str, Any]:
        """Get configuration dictionary for this decoder."""
        config = super().get_config()
        config.update(
            {
                "block_configs": self.block_configs,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BlockBasedDecoder":
        """Create BlockBasedDecoder instance from configuration dictionary."""
        return cls(**config)

    @classmethod
    def reverse_encoder_configs(
            cls,
            encoder_configs: list[dict[str, Any]],
            final_out_channels: int
    ) -> list[dict[str, Any]]:
        """Create decoder configs that reverse an encoder's configuration.

        This method helps create symmetric encoder-decoder architectures by
        automatically generating decoder configs that mirror the encoder.

        Parameters
        ----------
        encoder_configs : list of dict
            Encoder block configurations with 'out_channels' and optional
            'pool_size'.
        final_out_channels : int
            Number of output channels for the final decoder block.

        Returns
        -------
        list of dict
            Decoder block configurations with reversed channel progression
            and mirrored upsampling factors.

        Examples
        --------
        >>> encoder_configs = [
        ...     {'out_channels': 64, 'pool_size': (1, 2)},
        ...     {'out_channels': 128, 'pool_size': (2, 2)},
        ...     {'out_channels': 256, 'pool_size': (1, 2)}
        ... ]
        >>> decoder_configs = BlockBasedDecoder.reverse_encoder_configs(
        ...     encoder_configs, final_out_channels=3
        ... )
        >>> # Result: [
        >>> #     {'out_channels': 128, 'upsample_factor': (1, 2)},
        >>> #     {'out_channels': 64, 'upsample_factor': (2, 2)},
        >>> #     {'out_channels': 3, 'upsample_factor': (1, 2)}
        >>> # ]
        """
        if not encoder_configs:
            raise ValueError("encoder_configs cannot be empty")

        # Get channel progression from encoder
        encoder_channels = []
        for config in encoder_configs:
            encoder_channels.append(config["out_channels"])

        # Create reversed decoder configs
        decoder_configs = []

        # Reverse the encoder configs
        for i, encoder_config in enumerate(reversed(encoder_configs)):
            # Determine output channels for this decoder block
            if i == len(encoder_configs) - 1:
                # Last decoder block outputs final channels
                out_channels = final_out_channels
            else:
                # Use the input channels from the corresponding encoder stage
                # For encoder: input -> block1 -> block2 -> block3
                # For decoder: block3_out -> block2_in -> block1_in -> input
                corresponding_encoder_idx = len(encoder_configs) - 2 - i
                if corresponding_encoder_idx == 0:
                    # This would be the original input channels to the encoder
                    # We'll use the final_out_channels as a reasonable guess
                    out_channels = final_out_channels
                else:
                    out_channels = encoder_configs[
                        corresponding_encoder_idx - 1
                    ]["out_channels"]

            # Create decoder config
            decoder_config = {
                "out_channels": out_channels,
                "upsample_factor": encoder_config.get("pool_size", (1, 2)),
            }

            # Copy other relevant parameters
            for key in [
                "dropout",
                "kernel_size",
                "activation",
                "use_batch_norm",
                "bias",
            ]:
                if key in encoder_config:
                    decoder_config[key] = encoder_config[key]

            decoder_configs.append(decoder_config)

        return decoder_configs

    @classmethod
    def from_encoder(
            cls,
            encoder: "BlockBasedEncoder",
            final_out_channels: int,
            **kwargs
    ) -> "BlockBasedDecoder":
        """Create decoder that mirrors a BlockBasedEncoder.

        Parameters
        ----------
        encoder : BlockBasedEncoder
            Encoder to mirror.
        final_out_channels : int
            Number of output channels for the decoder.
        **kwargs
            Additional arguments for decoder configuration.

        Returns
        -------
        BlockBasedDecoder
            Configured decoder instance that mirrors the encoder.

        Examples
        --------
        >>> encoder = BlockBasedEncoder(in_channels=3, block_configs=[...])
        >>> decoder = BlockBasedDecoder.from_encoder(encoder,
         ...     final_out_channels=3)
        """
        # Create reversed configs from encoder
        decoder_configs = cls.reverse_encoder_configs(
            encoder.block_configs, final_out_channels
        )

        return cls(
            in_channels=encoder.out_channels,
            block_configs=decoder_configs,
            **kwargs,
        )

    @property
    def blocks(self) -> list[nn.Module]:
        """Access to decoder blocks for compatibility."""
        return list(self.operations)

    def __repr__(self) -> str:
        """String representation of the BlockBasedDecoder."""
        channel_progression = ' → '.join(
            map(str, self.get_channel_progression()))
        return (f"BlockBasedDecoder("
                f"blocks={len(self.operations)}, "
                f"channels={channel_progression})")
