import math
import torch.nn as nn
import torch
import torch.nn.functional as F
from .base import ModalityEncoder, ModalityDecoder, ModalityAutoEncoder
import numpy as np


class FastTimeSeriesBaselineEncoder(ModalityEncoder):
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
    n_output_tokens : int, optional
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
            kernel_size: int = 3,
    ):
        super().__init__(n_channels, d_model, n_tokens)
        self.d_model = d_model
        self.n_conv_layers = n_conv_layers

        # Calculate stride from input_length and n_tokens
        # stride = (input_length / n_tokens)^(1 / n_conv_layers)
        total_reduction = input_length / n_tokens
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

        self.adaptive_pool = nn.AdaptiveAvgPool1d(n_tokens)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(d_model)

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


class FastTimeSeriesBaselineDecoder(ModalityDecoder):
    """
    Mirrors FastTimeSeriesEncoder for pre-training via masked autoencoding.
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
            n_tokens: int = 100,
            n_deconv_layers: int = 4,
            kernel_size: int = 3,
    ):
        super().__init__(n_channels, n_tokens)
        self.d_model = d_model
        self.n_deconv_layers = n_deconv_layers

        # Mirror encoder stride calculation
        total_expansion = input_length / n_tokens
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

        for i, deconv in enumerate(self.deconv_layers):
            z = deconv(z)
            if i < len(self.deconv_layers) - 1:
                z = self.activation(z)

        if output_shape is not None:
            z = F.adaptive_avg_pool1d(z, output_shape)
        else:
            z = self.adaptive_pool(z)            # [B, n_channels, input_length]

        return z


class FastTimeSeriesBaselineAutoEncoder(ModalityAutoEncoder):
    """Combines TimeSeriesEncoder and TimeSeriesDecoder into an autoencoder model."""

    def __init__(
            self,
            n_channels: int = 6,
            input_length: int = 5000,
            d_model: int = 512,
            n_tokens: int = 100,
            n_layers: int = 4,
            kernel_size: int = 3,
    ):
        super().__init__(n_channels, d_model, n_tokens)
        self.encoder = FastTimeSeriesBaselineEncoder(
            n_channels=n_channels,
            input_length=input_length,
            d_model=d_model,
            n_tokens=n_tokens,
            n_conv_layers=n_layers,
            kernel_size=kernel_size,
        )
        self.decoder = FastTimeSeriesBaselineDecoder(
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
        tuple[torch.Tensor, torch.Tensor]
            ``(reconstructed, tokens)`` where reconstructed has the same
            shape as the input and tokens is ``(B, n_tokens, d_model)``.
        """
        output_length = x.shape[-1]
        tokens = self.encoder(x)
        recon = self.decoder(tokens, output_shape=output_length)
        return recon, tokens

def create_fast_timeseries_test_signal(
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


if __name__ == "__main__":
    # python -m tokamak_foundation_model.models.modality.fast_time_series_baseline

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("FastTimeSeriesBaselineEncoder / FastTimeSeriesBaselineDecoder")
    print("=" * 60)
    ts_enc = FastTimeSeriesBaselineEncoder(
        n_channels=6,
        out_features=512,
        hidden_dim=128,
    )
    ts_dec = FastTimeSeriesBaselineDecoder(
        in_features=512,
        out_channels=6,
        target_length=5000,
        hidden_dim=128,
    )

    x_ts = create_fast_timeseries_test_signal()
    tokens_ts = ts_enc(x_ts)
    recon_ts = ts_dec(tokens_ts)
    print(f"Input:  {x_ts.shape}")       # [4, 6, 5000]
    print(f"Tokens: {tokens_ts.shape}")  # [4, 100, 512]
    print(f"Recon:  {recon_ts.shape}")   # [4, 6, 5000]
