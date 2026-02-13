import math
import torch.nn as nn
import torch
from .base import ModalityEncoder, ModalityDecoder
import numpy as np


def create_timeseries_test_signal(
    batch_size: int = 4,
    n_channels: int = 6,
    length: int = 5000,
    sampling_rate: int = 10000
):
    """
    Create deterministic test signal for time-series encoder/decoder.

    Parameters
    ----------
    batch_size : int, optional
        Number of samples in batch, by default 4
    n_channels : int, optional
        Number of channels, by default 6
    length : int, optional
        Length of time series, by default 5000
    sampling_rate : int, optional
        Sampling rate in Hz, by default 10000

    Returns
    -------
    torch.Tensor
        Test signal of shape [batch_size, n_channels, length]

    Notes
    -----
    Test patterns per batch (applied to all channels):
    - Batch 0: Single impulse at center
    - Batch 1: Impulse train every 500 samples
    - Batch 2: 100 Hz sine wave
    - Batch 3: Linear chirp from 100 to 1000 Hz
    """
    t = np.linspace(0, length / sampling_rate, length)
    signal = np.zeros((batch_size, n_channels, length))

    if batch_size > 0:
        signal[0, :, length // 2] = 1.0

    if batch_size > 1:
        signal[1, :, ::500] = 1.0

    if batch_size > 2:
        signal[2, :, :] = np.sin(2 * np.pi * 100 * t)

    if batch_size > 3:
        f0, f1 = 100, 1000
        chirp_rate = (f1 - f0) / (length / sampling_rate)
        phase = 2 * np.pi * (f0 * t + 0.5 * chirp_rate * t ** 2)
        signal[3, :, :] = np.sin(phase)

    return torch.from_numpy(signal).float()


class TimeSeriesEncoder(nn.Module):
    """
    Encodes kHz time-series diagnostics using strided 1D convolutions.

    Parameters
    ----------
    n_channels : int, optional
        Number of input channels (e.g., 6 for filterscopes), by default 6
    input_length : int, optional
        Length of input time series (e.g., 5000 for 500ms @ 10kHz), by default 5000
    d_model : int, optional
        Model dimension for transformer, by default 512
    n_output_tokens : int, optional
        Number of temporal tokens to output, by default 100
    n_conv_layers : int, optional
        Number of convolutional layers, by default 4
    kernel_size : int, optional
        Kernel size for convolutions, by default 15
    verbose : bool, optional
        If True, print debug information during initialization, by default False

    Attributes
    ----------
    stride : int
        Calculated stride for convolutions based on desired compression ratio
    channels : list of int
        Channel sizes at each layer, dynamically computed
    conv_layers : nn.ModuleList
        List of 1D convolutional layers
    adaptive_pool : nn.AdaptiveAvgPool1d
        Adaptive pooling layer to ensure exact output token count
    """

    def __init__(
            self,
            n_channels: int = 6,
            input_length: int = 5000,
            d_model: int = 512,
            n_output_tokens: int = 100,
            n_conv_layers: int = 4,
            kernel_size: int = 3,
            verbose: bool = False
    ):
        super().__init__()

        self.n_channels = n_channels
        self.input_length = input_length
        self.d_model = d_model
        self.n_output_tokens = n_output_tokens
        self.n_conv_layers = n_conv_layers
        self.verbose = verbose

        # Calculate stride from input_length and n_output_tokens
        # stride = (input_length / n_output_tokens)^(1 / n_conv_layers)
        total_reduction = input_length / n_output_tokens
        self.stride = int(math.ceil(total_reduction ** (1 / n_conv_layers)))
        self.stride = max(2, min(self.stride, 5))

        # Dynamically build channel progression:
        # start at 64, double each layer, cap at d_model
        intermediate = [min(64 * (2 ** i), d_model) for i in range(n_conv_layers - 1)]
        self.channels = [n_channels] + intermediate + [d_model]

        # Build conv layers
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(
                in_channels=self.channels[i],
                out_channels=self.channels[i + 1],
                kernel_size=kernel_size,
                stride=self.stride,
                padding=kernel_size // 2
            )
            for i in range(n_conv_layers)
        ])

        self.norms = nn.ModuleList([
            nn.InstanceNorm1d(self.channels[i + 1]) for i in range(n_conv_layers)
        ])

        self.adaptive_pool = nn.AdaptiveAvgPool1d(n_output_tokens)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(d_model)

        if self.verbose:
            print(f"TimeSeriesEncoder:")
            print(f"  Stride:   {self.stride}")
            print(f"  Channels: {self.channels}")
            print(f"  Theoretical length before pool: "
                  f"{input_length / (self.stride ** n_conv_layers):.1f}")

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
        for conv, norm in zip(self.conv_layers, self.norms):
            x = conv(x)         # [B, channels[i+1], T']
            x = norm(x)
            x = self.activation(x)

        x = self.adaptive_pool(x)                # [B, d_model, n_output_tokens]
        x = x.transpose(1, 2)                    # [B, n_output_tokens, d_model]

        return x


class TimeSeriesDecoder(nn.Module):
    """
    Mirrors TimeSeriesEncoder for pre-training via masked autoencoding.
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
    n_input_tokens : int, optional
        Number of input tokens from encoder, by default 100
    n_deconv_layers : int, optional
        Number of deconvolutional layers (should match encoder), by default 4
    kernel_size : int, optional
        Kernel size for transposed convolutions, by default 15
    verbose : bool, optional
        If True, print debug information during initialization, by default False

    Attributes
    ----------
    stride : int
        Calculated stride for transposed convolutions
    channels : list of int
        Channel sizes at each layer, dynamically computed (reversed from encoder)
    deconv_layers : nn.ModuleList
        List of 1D transposed convolutional layers
    adaptive_pool : nn.AdaptiveAvgPool1d
        Adaptive pooling layer to ensure exact output length
    """

    def __init__(
            self,
            n_channels: int = 6,
            input_length: int = 5000,
            d_model: int = 512,
            n_input_tokens: int = 100,
            n_deconv_layers: int = 4,
            kernel_size: int = 3,
            verbose: bool = False
    ):
        super().__init__()

        self.n_channels = n_channels
        self.input_length = input_length
        self.d_model = d_model
        self.n_input_tokens = n_input_tokens
        self.n_deconv_layers = n_deconv_layers
        self.verbose = verbose

        # Mirror encoder stride calculation
        total_expansion = input_length / n_input_tokens
        self.stride = int(math.ceil(total_expansion ** (1 / n_deconv_layers)))
        self.stride = max(2, min(self.stride, 5))

        # Mirror encoder channel progression (reversed)
        intermediate = [min(64 * (2 ** i), d_model) for i in range(n_deconv_layers - 1)]
        self.channels = [d_model] + list(reversed(intermediate)) + [n_channels]

        # Build deconv layers
        self.deconv_layers = nn.ModuleList([
            nn.ConvTranspose1d(
                in_channels=self.channels[i],
                out_channels=self.channels[i + 1],
                kernel_size=kernel_size,
                stride=self.stride,
                padding=kernel_size // 2,
                output_padding=self.stride - 1
            )
            for i in range(n_deconv_layers)
        ])

        self.adaptive_pool = nn.AdaptiveAvgPool1d(input_length)
        self.activation = nn.GELU()

        if self.verbose:
            print(f"TimeSeriesDecoder:")
            print(f"  Stride:   {self.stride}")
            print(f"  Channels: {self.channels}")
            print(f"  Theoretical length before pool: "
                  f"{n_input_tokens * (self.stride ** n_deconv_layers):.1f}")

    def forward(self, x):
        """
        Decode tokens back to original time-series (pre-training only).

        Parameters
        ----------
        x : torch.Tensor
            Input tokens of shape [batch, n_input_tokens, d_model]

        Returns
        -------
        torch.Tensor
            Reconstructed time-series of shape [batch, n_channels, input_length]
        """
        x = x.transpose(1, 2)                    # [B, d_model, n_input_tokens]

        for i, deconv in enumerate(self.deconv_layers):
            x = deconv(x)
            if i < len(self.deconv_layers) - 1:
                x = self.activation(x)

        x = self.adaptive_pool(x)                # [B, n_channels, input_length]

        return x

class TimeSeriesAutoencoder(nn.Module):
    """Combines TimeSeriesEncoder and TimeSeriesDecoder into an autoencoder model."""

    def __init__(
            self,
            n_channels: int = 6,
            input_length: int = 5000,
            d_model: int = 512,
            n_tokens: int = 100,
            n_layers: int = 4,
            kernel_size: int = 3,
            verbose: bool = False
    ):
        super().__init__()
        self.encoder = TimeSeriesEncoder(
            n_channels=n_channels,
            input_length=input_length,
            d_model=d_model,
            n_output_tokens=n_tokens,
            n_conv_layers=n_layers,
            kernel_size=kernel_size,
            verbose=verbose
        )
        self.decoder = TimeSeriesDecoder(
            n_channels=n_channels,
            input_length=input_length,
            d_model=d_model,
            n_input_tokens=n_tokens,
            n_deconv_layers=n_layers,
            kernel_size=kernel_size,
            verbose=verbose
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


class FastTimeSeriesEncoder(ModalityEncoder):

    def __init__(self, in_channels, out_features=64, hidden_dim=128):
        super().__init__(in_channels, out_features)
        self.conv_layers = nn.Sequential(
            # Layer 1: (B, C, T) -> (B, 64, T//5)
            nn.Conv1d(in_channels, 64, kernel_size=10, stride=5, padding=2),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            # Layer 2: -> (B, 128, T//15)
            nn.Conv1d(64, hidden_dim, kernel_size=5, stride=3, padding=1),
            nn.GroupNorm(16, hidden_dim),
            nn.GELU(),
            # Layer 3: -> (B, 256, T//30)
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, hidden_dim * 2),
            nn.GELU(),
            # Layer 4: -> (B, 256, T//60)
            nn.Conv1d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, hidden_dim * 2),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim * 2, out_features),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.proj(self.pool(self.conv_layers(x)))


class FastTimeSeriesDecoder(ModalityDecoder):

    def __init__(self, in_features=64, out_channels=1, target_length=5000, hidden_dim=128):
        super().__init__(in_features, out_channels)
        self.target_length = target_length
        self.hidden_dim = hidden_dim
        self.proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim * 2),
            nn.ReLU(),
            nn.Unflatten(1, (hidden_dim * 2, 1)),
        )
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose1d(
                hidden_dim * 2,
                hidden_dim * 2,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.GELU(),
            nn.ConvTranspose1d(
                hidden_dim * 2,
                hidden_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.GELU(),
            nn.ConvTranspose1d(
                hidden_dim, 64, kernel_size=5, stride=3, padding=1, output_padding=2
            ),
            nn.GELU(),
            nn.ConvTranspose1d(
                64, out_channels, kernel_size=10, stride=5, padding=2, output_padding=4
            ),
        )
        self.resample = nn.AdaptiveAvgPool1d(target_length)

    def forward(self, z):
        return self.resample(self.deconv_layers(self.proj(z)))


if __name__ == "__main__":
    print("=" * 60)
    print("TimeSeriesEncoder / TimeSeriesDecoder")
    print("=" * 60)
    ts_enc = TimeSeriesEncoder(n_channels=6, input_length=5000,
                               d_model=512, n_output_tokens=100, verbose=True)
    ts_dec = TimeSeriesDecoder(n_channels=6, input_length=5000,
                               d_model=512, n_input_tokens=100, verbose=True)

    x_ts = create_timeseries_test_signal()
    tokens_ts = ts_enc(x_ts)
    recon_ts = ts_dec(tokens_ts)
    print(f"Input:  {x_ts.shape}")       # [4, 6, 5000]
    print(f"Tokens: {tokens_ts.shape}")  # [4, 100, 512]
    print(f"Recon:  {recon_ts.shape}")   # [4, 6, 5000]
