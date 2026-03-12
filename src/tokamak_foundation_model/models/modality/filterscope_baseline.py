import math
import torch.nn as nn
import torch
from .base import ModalityEncoder, ModalityDecoder, ModalityAutoEncoder


class StridedResBlockTranspose1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        # Pre-norm on branch input only; shortcut carries raw amplitude unchanged
        self.norm = nn.InstanceNorm1d(in_channels, affine=True)
        self.net = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=kernel_size//2,
                               output_padding=stride - 1),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size,
                      stride=1, padding=kernel_size//2),  # refine without expanding
        )

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=1,
                                               stride=stride, output_padding=stride - 1)
        else:
            self.shortcut = nn.Identity()

        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(self.net(self.norm(x)) + self.shortcut(x))


class StridedResBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        # Pre-norm on branch input only; shortcut carries raw amplitude unchanged
        self.norm = nn.InstanceNorm1d(in_channels, affine=True)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=kernel_size//2),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size,
                      stride=1, padding=kernel_size//2),  # stride only on first conv
        )

        # Shortcut must match output shape whenever channels or stride differ
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.shortcut = nn.Identity()

        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(self.net(self.norm(x)) + self.shortcut(x))


class FilterscopeBaselineEncoder(ModalityEncoder):
    """
    Encodes fast time-series diagnostics using strided 1D convolutions.

    Parameters
    ----------
    n_channels : int, optional
        Number of input channels (e.g., 6 for filterscopes), by default 6
    input_length : int, optional
        Length of input time series (e.g., 5000 for 500ms @ 10kHz), by default 5000
    d_model : int, optional
        Model dimension for transformer, by default 512
    n_tokens : int, optional
        Number of temporal tokens to output, by default 100
    n_conv_layers : int, optional
        Number of convolutional layers, by default 4
    kernel_size : int, optional
        Kernel size for convolutions, by default 15

    Attributes
    ----------
    stride : int
        Calculated stride for convolutions based on desired compression ratio
    channels : list of int
        Channel sizes at each layer, dynamically computed
    conv_layers : nn.ModuleList
        List of 1D convolutional layers
    compress_conv : nn.Conv1d
        Learned strided convolution that compresses to approximately n_tokens
    adaptive_pool : nn.AdaptiveAvgPool1d
        Adaptive pooling layer to ensure exact output token count
    """

    def __init__(
            self,
            n_channels: int,
            d_model: int = 512,
            n_tokens: int = 100,
            input_length: int = 5000,
            n_conv_layers: int = 4,
            kernel_size: int = 7,
            n_transformer_layers: int = 2,
            n_heads: int = 8,
    ):
        super().__init__(n_channels, d_model, n_tokens)
        self.d_model = d_model
        self.n_conv_layers = n_conv_layers

        # Calculate stride from input_length and n_tokens.
        # Use floor so the conv layers slightly over-compress, then the learned
        # compress_conv + AdaptiveAvgPool1d reduce to exactly n_tokens.
        total_reduction = input_length / n_tokens
        self.stride = int(math.floor(total_reduction ** (1 / n_conv_layers)))
        self.stride = max(2, min(self.stride, 5))

        # Dynamically build channel progression:
        # start at 64, double each layer, cap at d_model
        intermediate = [
            min(64 * (2 ** i), d_model) for i in range(n_conv_layers - 1)]
        self.channels = [n_channels] + intermediate + [d_model]

        # Build conv layers
        self.conv_layers = nn.ModuleList([
            StridedResBlock1d(
                in_channels=self.channels[i],
                out_channels=self.channels[i + 1],
                kernel_size=kernel_size,
                stride=self.stride
            )
            for i in range(n_conv_layers)
        ])

        # Learned compression: strided Conv1d does the bulk of the reduction
        # (differentiable, learns what to preserve from both peaks and background),
        # AdaptiveAvgPool1d handles the exact token count as a small safety net.
        approx_after_convs = math.ceil(input_length / (self.stride ** n_conv_layers))
        compress_stride = max(1, approx_after_convs // n_tokens)
        self.compress_conv = nn.Conv1d(
            d_model, d_model, kernel_size=3, stride=compress_stride, padding=1
        )
        self.adaptive_pool = nn.AdaptiveAvgPool1d(n_tokens)

        # Learnable positional embeddings so the transformer knows token order
        self.pos_embedding = nn.Embedding(n_tokens, d_model)

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=2 * d_model,
            dropout=0.1,
            batch_first=True,
            norm_first=True,  # pre-norm, consistent with residual blocks
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=n_transformer_layers)

    def forward(self, x):
        """
        Encode time-series into tokens.

        Parameters
        ----------
        x : torch.Tensor
            Input time-series of shape [batch, n_channels, input_length]

        Returns
        -------
        torch.Tensor
            Encoded tokens of shape [batch, n_output_tokens, d_model]
        """
        for conv in self.conv_layers:
            x = conv(x)                                   # [B, d_model, T']

        x = self.compress_conv(x)                         # [B, d_model, ~n_tokens]
        x = self.adaptive_pool(x).transpose(1, 2)         # [B, n_tokens, d_model]

        positions = torch.arange(x.shape[1], device=x.device)
        x = x + self.pos_embedding(positions)             # inject temporal order
        x = self.transformer(x)                           # [B, n_tokens, d_model]

        return x


class FilterscopeBaselineDecoder(ModalityDecoder):
    """
    Mirrors FilterscopeBaselineEncoder for pre-training via masked autoencoding.
    Reconstructs the original input time-series from encoder tokens.

    Parameters
    ----------
    n_channels : int, optional
        Number of output channels (e.g., 6 for filterscopes), by default 6
    input_length : int, optional
        Length of original input to reconstruct (e.g., 5000 for 500ms @ 10kHz),
        by default 5000
    d_model : int, optional
        Model dimension from encoder, by default 512
    n_tokens : int, optional
        Number of input tokens from encoder, by default 100
    n_deconv_layers : int, optional
        Number of deconvolutional layers (should match encoder), by default 4
    kernel_size : int, optional
        Kernel size for transposed convolutions, by default 15

    Attributes
    ----------
    stride : int
        Calculated stride for transposed convolutions
    channels : list of int
        Channel sizes at each layer, dynamically computed (reversed from encoder)
    deconv_layers : nn.ModuleList
        List of 1D transposed convolutional layers
    adaptive_pool : nn.AdaptiveMaxPool1d
        Adaptive pooling layer to ensure exact output length
    """

    def __init__(
            self,
            n_channels: int = 6,
            input_length: int = 5000,
            d_model: int = 512,
            n_tokens: int = 100,
            n_deconv_layers: int = 4,
            kernel_size: int = 7,
    ):
        super().__init__(n_channels, n_tokens)
        self.d_model = d_model
        self.n_deconv_layers = n_deconv_layers

        # Mirror encoder stride calculation
        total_expansion = input_length / n_tokens
        self.stride = int(math.floor(total_expansion ** (1 / n_deconv_layers)))
        self.stride = max(2, min(self.stride, 5))

        # Mirror encoder channel progression (reversed)
        intermediate = [
            min(64 * (2 ** i), d_model) for i in range(n_deconv_layers - 1)]
        self.channels = [d_model] + list(reversed(intermediate)) + [n_channels]

        # Build deconv layers
        self.deconv_layers = nn.ModuleList([
            StridedResBlockTranspose1d(
                in_channels=self.channels[i],
                out_channels=self.channels[i + 1],
                kernel_size=kernel_size,
                stride=self.stride,
            )
            for i in range(n_deconv_layers)
        ])

        self.output_proj = nn.Conv1d(n_channels, n_channels, kernel_size=1)

        self.adaptive_pool = nn.AdaptiveAvgPool1d(input_length)

    def forward(self, z, output_shape=None):
        """
        Decode tokens back to original time-series (pre-training only).

        Parameters
        ----------
        z : torch.Tensor
            Input tokens of shape [batch, n_input_tokens, d_model]

        Returns
        -------
        torch.Tensor
            Reconstructed time-series of shape [batch, n_channels, input_length]
        """
        z = z.transpose(1, 2)                    # [B, d_model, n_input_tokens]

        for deconv in self.deconv_layers:
            z = deconv(z)

        z = self.adaptive_pool(z)                # [B, n_channels, input_length]
        z = self.output_proj(z)

        return z


class FilterscopeBaselineAutoEncoder(ModalityAutoEncoder):
    """Combines TimeSeriesEncoder and TimeSeriesDecoder into an autoencoder model."""

    def __init__(
            self,
            n_channels: int = 6,
            input_length: int = 5000,
            d_model: int = 512,
            n_tokens: int = 100,
            n_layers: int = 4,
            kernel_size: int = 7,
            n_transformer_layers: int = 2,
            n_heads: int = 8,
    ):
        super().__init__(n_channels, d_model, n_tokens)
        self.encoder = FilterscopeBaselineEncoder(
            n_channels=n_channels,
            input_length=input_length,
            d_model=d_model,
            n_tokens=n_tokens,
            n_conv_layers=n_layers,
            kernel_size=kernel_size,
            n_transformer_layers=n_transformer_layers,
            n_heads=n_heads,
        )
        self.decoder = FilterscopeBaselineDecoder(
            n_channels=n_channels,
            input_length=input_length,
            d_model=d_model,
            n_tokens=n_tokens,
            n_deconv_layers=n_layers,
            kernel_size=kernel_size,
        )

    def forward(self, x):
        """
        Forward pass through the autoencoder.

        Parameters
        ----------
        x : torch.Tensor
            Input time-series of shape [batch, n_channels, input_length]

        Returns
        -------
        torch.Tensor
            Reconstructed time-series of shape [batch, n_channels, input_length]
        """
        tokens = self.encoder(x)
        recon = self.decoder(tokens)
        return recon
