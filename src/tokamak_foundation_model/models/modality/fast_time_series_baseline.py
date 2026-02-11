import math
import torch.nn as nn
import torch
from base import ModalityEncoder, ModalityDecoder
import numpy as np


def create_test_signal(batch_size=4, n_channels=6, length=5000, sampling_rate=10000):
    """
    Create a deterministic test signal with different test patterns per batch.

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
    Different test patterns per batch:
    - Batch 0: Single impulse at center (all channels)
    - Batch 1: Impulse train every 500 samples (all channels)
    - Batch 2: 100 Hz sine wave (all channels)
    - Batch 3: Linear chirp from 100 to 1000 Hz (all channels)
    """
    t = np.linspace(0, length / sampling_rate, length)
    signal = np.zeros((batch_size, n_channels, length))

    # Batch 0: Single impulse at center
    if batch_size > 0:
        signal[0, :, length // 2] = 1.0

    # Batch 1: Impulse train every 500 samples
    if batch_size > 1:
        signal[1, :, ::500] = 1.0

    # Batch 2: 100 Hz sine wave
    if batch_size > 2:
        for ch in range(n_channels):
            signal[2, ch, :] = np.sin(2 * np.pi * 100 * t)

    # Batch 3: Chirp from 100 to 1000 Hz
    if batch_size > 3:
        f0, f1 = 100, 1000
        chirp_rate = (f1 - f0) / (length / sampling_rate)
        phase = 2 * np.pi * (f0 * t + 0.5 * chirp_rate * t**2)
        for ch in range(n_channels):
            signal[3, ch, :] = np.sin(phase)

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
    verbose : bool, optional
        If True, print debug information during initialization, by default False

    Attributes
    ----------
    stride : int
        Calculated stride for convolutions based on desired compression ratio
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
            verbose: bool = False,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.input_length = input_length
        self.d_model = d_model
        self.n_output_tokens = n_output_tokens
        self.n_conv_layers = n_conv_layers
        self.verbose = verbose

        # Calculate the stride needed to get from input_length to n_output_tokens
        # We want: input_length / (stride^n_conv_layers) ≈ n_output_tokens
        # So: stride = (input_length / n_output_tokens)^(1 / n_conv_layers)
        total_reduction = input_length / n_output_tokens
        self.stride = int(math.ceil(total_reduction ** (1 / n_conv_layers)))

        # Clamp stride to reasonable values (typically 2-5)
        self.stride = max(2, min(self.stride, 5))

        if self.verbose:
            print(f"Encoder calculated stride: {self.stride} "
                  f"for {n_conv_layers} layers")
            print(
                f"Theoretical output before adaptive pool: "
                f"{input_length / (self.stride**n_conv_layers):.1f}")

        # Define channel progression
        channels = [n_channels, 64, 128, 256, d_model]

        # Build conv layers dynamically
        self.conv_layers = nn.ModuleList()
        for i in range(n_conv_layers):
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=15,
                    stride=self.stride,
                    padding=7,
                )
            )

        # Adaptive pooling to get exact output length
        self.adaptive_pool = nn.AdaptiveAvgPool1d(n_output_tokens)

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
        # Apply conv layers
        for conv in self.conv_layers:
            x = self.activation(conv(x))

        # Adaptive pooling to exact output length
        x = self.adaptive_pool(x)  # [B, d_model, n_output_tokens]

        # Transpose to [B, T, C] for transformer
        x = x.transpose(1, 2)  # [B, n_output_tokens, d_model]

        # Layer norm
        x = self.norm(x)

        return x


class TimeSeriesDecoder(nn.Module):
    """
    Decodes from transformer output back to kHz time-series.

    Parameters
    ----------
    n_channels : int, optional
        Number of output channels (e.g., 6 for filterscopes), by default 6
    output_length : int, optional
        Length of output time series (e.g., 500 for 50ms @ 10kHz), by default 500
    d_model : int, optional
        Model dimension from transformer, by default 512
    n_input_tokens : int, optional
        Number of input tokens from transformer, by default 100
    n_deconv_layers : int, optional
        Number of deconvolutional layers, by default 4
    verbose : bool, optional
        If True, print debug information during initialization, by default False

    Attributes
    ----------
    stride : int
        Calculated stride for transposed convolutions based on desired expansion ratio
    deconv_layers : nn.ModuleList
        List of 1D transposed convolutional layers
    adaptive_pool : nn.AdaptiveAvgPool1d
        Adaptive pooling layer to ensure exact output length
    """

    def __init__(
            self,
            n_channels: int = 6,
            output_length: int = 500,
            d_model: int = 512,
            n_input_tokens: int = 100,
            n_deconv_layers: int = 4,
            verbose: bool = False,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.output_length = output_length
        self.d_model = d_model
        self.n_deconv_layers = n_deconv_layers
        self.verbose = verbose

        # Calculate the stride needed for upsampling
        # We want: n_input_tokens * (stride^n_deconv_layers) ≈ output_length
        # So: stride = (output_length / n_input_tokens)^(1 / n_deconv_layers)
        total_upsampling = output_length / n_input_tokens
        self.stride = int(math.ceil(total_upsampling ** (1 / n_deconv_layers)))

        # Clamp stride to reasonable values
        self.stride = max(2, min(self.stride, 5))

        if self.verbose:
            print(f"Decoder calculated stride: {self.stride} "
                  f"for {n_deconv_layers} layers")
            print(f"Theoretical output before adaptive pool: "
                  f"{n_input_tokens * (self.stride**n_deconv_layers):.1f}")

        # Define channel progression (reverse of encoder)
        channels = [d_model, 256, 128, 64, n_channels]

        # Build deconv layers dynamically
        self.deconv_layers = nn.ModuleList()
        for i in range(n_deconv_layers):
            self.deconv_layers.append(
                nn.ConvTranspose1d(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=15,
                    stride=self.stride,
                    padding=7,
                    output_padding=self.stride - 1,
                )
            )

        # Adaptive pooling to get exact output length
        self.adaptive_pool = nn.AdaptiveAvgPool1d(output_length)

        self.activation = nn.GELU()

    def forward(self, x):
        """
        Decode tokens back to time-series.

        Parameters
        ----------
        x : torch.Tensor
            Input tokens of shape [batch, n_input_tokens, d_model]

        Returns
        -------
        torch.Tensor
            Decoded time-series of shape [batch, n_channels, output_length]
        """
        # Transpose to [B, C, T]
        x = x.transpose(1, 2)  # [B, d_model, n_input_tokens]

        # Apply deconv layers (except last one without activation)
        for i, deconv in enumerate(self.deconv_layers):
            x = deconv(x)
            if i < len(self.deconv_layers) - 1:  # No activation on final layer
                x = self.activation(x)

        # Ensure exact output length
        x = self.adaptive_pool(x)  # [B, n_channels, output_length]

        return x


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
    print("Testing TimeSeriesEncoder and TimeSeriesDecoder")
    print("=" * 60)

    # Create encoder with verbose=True
    encoder = TimeSeriesEncoder(
        n_channels=6,
        input_length=5000,
        d_model=512,
        n_output_tokens=100,
        n_conv_layers=4,
        verbose=True,
    )

    # Create decoder with verbose=True
    decoder = TimeSeriesDecoder(
        n_channels=6,
        output_length=500,
        d_model=512,
        n_input_tokens=100,
        n_deconv_layers=4,
        verbose=True,
    )

    # Create deterministic test signal
    print("Generating deterministic test signal...")
    x = create_test_signal(batch_size=4, n_channels=6, length=5000, sampling_rate=10000)
    print(f"Input shape: {x.shape}")
    print(f"Input statistics - Mean: {x.mean():.4f}, Std: {x.std():.4f}")
    print(
        f"Channel 0 (100 Hz sine) - Min: {x[0, 0].min():.4f}, Max: {x[0, 0].max():.4f}"
    )

    # Encode
    print("Encoding...")
    tokens = encoder(x)
    print(f"Encoded shape: {tokens.shape}")
    print(f"Token statistics - Mean: {tokens.mean():.4f}, Std: {tokens.std():.4f}")

    # Decode
    print("Decoding...")
    output = decoder(tokens)
    print(f"Decoded shape: {output.shape}")
    print(f"Output statistics - Mean: {output.mean():.4f}, Std: {output.std():.4f}")

    print("" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
