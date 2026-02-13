import math
import torch.nn as nn
import torch
import torch.nn.functional as F
from .base import ModalityEncoder, ModalityDecoder
import numpy as np


class FastTimeSeriesBaselineEncoder(ModalityEncoder):

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


class FastTimeSeriesBaselineDecoder(ModalityDecoder):

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

    def forward(self, z, output_shape=None):
        return self.resample(self.deconv_layers(self.proj(z)))


class FastTimeSeriesBaselineAutoEncoder(nn.Module):

    def __init__(self, n_channels, d_model=64, n_tokens=None):
        super().__init__()
        self.encoder = FastTimeSeriesBaselineEncoder(in_channels=n_channels, out_features=d_model)
        self.decoder = FastTimeSeriesBaselineDecoder(in_features=d_model, out_channels=n_channels, target_length=5000)

    def forward(self, x):
        target_length = x.shape[-1]
        z = self.encoder(x)
        out = self.decoder(z)
        if out.shape[-1] != target_length:
            out = F.adaptive_avg_pool1d(out, target_length)
        return out

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
        in_channels=6,
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
