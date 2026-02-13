import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .base import ModalityEncoder, ModalityDecoder, ModalityAutoEncoder


class SpatialProfileBaselineEncoder(ModalityEncoder):
    def __init__(self,
        n_channels: int,
        d_model: int = 64,
        n_tokens: int = 0,
        n_spatial_points: int = 50,
        n_time_points: int = 50,
        kernel_size: int = 5,
    ):
        super().__init__(n_channels, d_model, n_tokens)

        self.n_spatial_points = n_spatial_points
        self.n_time_points = n_time_points
        self.d_model = d_model
        self.n_tokens = n_tokens

        self.adaptive_pool = nn.AdaptiveAvgPool1d(n_tokens)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(d_model)

        # Spatial MLP: encodes each time step's spatial profile
        self.spatial_encoder = nn.Sequential(
            nn.Linear(n_spatial_points, 128),
            self.activation,
            nn.Linear(128, 256),
            self.activation,
            nn.Linear(256, d_model)
        )

        # Temporal conv: compresses time dimension
        self.temporal_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=kernel_size // 2,
            padding=kernel_size // 2
        )

    def forward(self, x):
        B, S, T = x.shape

        # Encode spatial structure at each time step independently
        x = x.transpose(1, 2)                # [B, n_time, S]
        x = x.reshape(B * T, S)                  # [B*T, S]
        x = self.spatial_encoder(x)               # [B*T, d_model]
        x = x.reshape(B, T, self.d_model)         # [B, T, d_model]

        # Encode temporal evolution
        x = x.transpose(1, 2)                    # [B, d_model, T]
        x = self.activation(self.temporal_conv(x))  # [B, d_model, T']
        x = self.adaptive_pool(x)                # [B, d_model, n_output_tokens]

        x = x.transpose(1, 2)                    # [B, n_output_tokens, d_model]
        x = self.norm(x)

        return x


class SpatialProfileBaselineDecoder(ModalityDecoder):

    def __init__(self,
        n_channels: int,
        d_model: int = 64,
        n_tokens: int = 0,
        n_spatial_points: int = 50,
        n_time_points: int = 50,
        kernel_size: int = 5,
    ):
        super().__init__(n_channels, d_model)

        self.n_spatial_points = n_spatial_points
        self.n_time_points = n_time_points
        self.d_model = d_model
        self.n_tokens = n_tokens

        self.activation = nn.GELU()
        self.adaptive_pool = nn.AdaptiveAvgPool1d(n_tokens)

        # Mirror temporal conv
        self.temporal_deconv = nn.ConvTranspose1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=kernel_size // 2,
            padding=kernel_size // 2,
            output_padding=max(0, (kernel_size // 2) - 1)
        )

        # Mirror spatial MLP (reversed)
        self.spatial_decoder = nn.Sequential(
            nn.Linear(d_model, 256),
            self.activation,
            nn.Linear(256, 128),
            self.activation,
            nn.Linear(128, n_spatial_points)
        )

    def forward(self, z, output_shape=None):
        B, D, T = z.shape

        # Upsample temporal dimension
        z = z.transpose(1, 2)              # [B, d_model, n_input_tokens]
        z = self.activation(self.temporal_deconv(z))  # [B, d_model, T']
        z = self.adaptive_pool(z)                     # [B, d_model, n_time]

        # Decode spatial structure at each time step independently
        z = z.transpose(1, 2)                         # [B, n_time, d_model]
        T = z.shape[1]
        z = z.reshape(B * T, self.d_model)            # [B*T, d_model]
        z = self.spatial_decoder(z)                   # [B*n_time, n_spatial]
        z = z.reshape(B, T, self.n_spatial_points)    # [B, n_time, n_spatial]
        z = z.transpose(1, 2)                         # [B, n_spatial, n_time]

        return z


class SpatialProfileBaselineAutoEncoder(ModalityAutoEncoder):

    def __init__(self, 
        n_channels: int, 
        d_model: int = 64, 
        n_tokens: int = 0,
    ):
        super().__init__(n_channels, d_model, n_tokens)
        self.encoder = SpatialProfileBaselineEncoder(n_channels, d_model, n_tokens)
        self.decoder = SpatialProfileBaselineDecoder(n_channels, d_model, n_tokens)

    def forward(self, x):
        n_time = x.shape[-1]
        z = self.encoder(x)
        out = self.decoder(z)
        if out.shape[-1] != n_time:
            out = F.adaptive_avg_pool1d(out, n_time)
        return out

def create_spatial_profile_test_signal(
    batch_size=4, 
    n_spatial_points=50, 
    n_time_points=50,
):
    signal = np.zeros((batch_size, n_spatial_points, n_time_points))

    # Spatial coordinate (normalized 0 to 1)
    x_spatial = np.linspace(0, 1, n_spatial_points)

    # Temporal coordinate (normalized 0 to 1)
    t_temporal = np.linspace(0, 1, n_time_points)

    # Batch 0: Constant profile (all ones)
    if batch_size > 0:
        signal[0, :, :] = 1.0

    # Batch 1: Linear spatial gradient (0 to 1), constant in time
    if batch_size > 1:
        for t in range(n_time_points):
            signal[1, :, t] = x_spatial

    # Batch 2: Spatial step function (0 before midpoint, 1 after)
    if batch_size > 2:
        midpoint = n_spatial_points // 2
        signal[2, midpoint:, :] = 1.0

    # Batch 3: Traveling pulse
    if batch_size > 3:
        for t_idx, t in enumerate(t_temporal):
            # Sine wave that appears to move from left to right
            signal[3, 10+t_idx:20+t_idx, t_idx] = 1
            if 20+t_idx >= n_spatial_points:
                break
    return torch.from_numpy(signal).float()

if __name__ == "__main__":
    print("=" * 60)
    print("SpatialProfileEncoder / SpatialProfileDecoder")
    print("=" * 60)
    sp_enc = SpatialProfileBaselineEncoder(
        n_channels=50, 
        n_time_points=50,
        d_model=64, 
        n_tokens=10, 
        kernel_size=3,
    )
    sp_dec = SpatialProfileBaselineDecoder(
        n_channels=50, 
        d_model=64, 
        n_tokens=10, 
        kernel_size=3,
    )
    x_sp = create_spatial_profile_test_signal()
    tokens_sp = sp_enc(x_sp)
    recon_sp = sp_dec(tokens_sp)
    print(f"Input:  {x_sp.shape}")       # [4, 50, 50]
    print(f"Tokens: {tokens_sp.shape}")  # [4, 10, 512]
    print(f"Recon:  {recon_sp.shape}")   # [4, 50, 50]
