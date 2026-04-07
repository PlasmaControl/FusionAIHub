import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .base import (
    ModalityEncoder, ModalityDecoder, ModalityAutoEncoder,
    StridedResBlock1d, StridedResBlockTranspose1d,
)


class SpatialProfileBaselineEncoder(ModalityEncoder):
    def __init__(self,
        n_channels: int,
        d_model: int = 64,
        n_tokens: int = 4,
        n_spatial_points: int = 50,
        n_time_points: int = 50,
        kernel_size: int = 5,
        n_transformer_layers: int = 4,
        n_heads: int = 8,
    ):
        super().__init__(n_channels, d_model, n_tokens)

        self.n_spatial_points = n_spatial_points
        self.n_time_points = n_time_points
        self.d_model = d_model
        self.n_tokens = n_tokens

        self.adaptive_pool = nn.AdaptiveMaxPool1d(n_tokens)
        self.activation = nn.SELU()

        # Spatial MLP: encodes each time step's spatial profile
        self.spatial_encoder = nn.Sequential(
            nn.Linear(n_spatial_points, 128),
            self.activation,
            nn.AlphaDropout(0.2),
            nn.Linear(128, 256),
            self.activation,
            nn.AlphaDropout(0.2),
            nn.Linear(256, 512),
            self.activation,
            nn.AlphaDropout(0.2),
            nn.Linear(512, d_model),
        )

        # Temporal residual block: compresses time dimension
        self.temporal_conv = StridedResBlock1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=max(1, kernel_size // 2),
        )

        # Transformer encoder: learns to pack information into n_tokens
        self.pos_embedding = nn.Embedding(n_tokens, d_model)
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=2 * d_model,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            transformer_layer, num_layers=n_transformer_layers)

        # LeCun normal init for SELU self-normalisation
        for module in self.spatial_encoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='linear')
                nn.init.zeros_(module.bias)

    def forward(self, x):
        B, S, T = x.shape

        # Encode spatial structure at each time step independently
        x = x.transpose(1, 2)                    # [B, n_time, S]
        x = x.reshape(B * T, S)                  # [B*T, S]
        x = self.spatial_encoder(x)               # [B*T, d_model]
        x = x.reshape(B, T, self.d_model)         # [B, T, d_model]

        # Encode temporal evolution
        x = x.transpose(1, 2)                    # [B, d_model, T]
        x = self.temporal_conv(x)                # [B, d_model, T']
        x = self.adaptive_pool(x)                # [B, d_model, n_tokens]

        x = x.transpose(1, 2)                    # [B, n_tokens, d_model]

        # Transformer mixing across tokens
        positions = torch.arange(x.shape[1], device=x.device)
        x = x + self.pos_embedding(positions)
        x = self.transformer(x)                  # [B, n_tokens, d_model]

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

        self.activation = nn.SELU()
        self.adaptive_pool = nn.AdaptiveAvgPool1d(n_time_points)

        # Mirror temporal residual block
        self.temporal_deconv = StridedResBlockTranspose1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=max(1, kernel_size // 2),
        )

        # Mirror spatial MLP (reversed)
        self.spatial_decoder = nn.Sequential(
            nn.Linear(d_model, 512),
            self.activation,
            nn.Linear(512, 256),
            self.activation,
            nn.Linear(256, 128),
            self.activation,
            nn.Linear(128, n_spatial_points),
        )

    def forward(self, x, output_shape=None):
        B = x.shape[0]

        # Upsample temporal dimension
        x = x.transpose(1, 2)                        # [B, d_model, n_input_tokens]
        x = self.temporal_deconv(x)                  # [B, d_model, T']
        x = self.adaptive_pool(x)                    # [B, d_model, n_time]

        # Decode spatial structure at each time step independently
        x = x.transpose(1, 2)                        # [B, n_time, d_model]
        T = x.shape[1]
        x = x.reshape(B * T, self.d_model)           # [B*T, d_model]
        x = self.spatial_decoder(x)                  # [B*n_time, n_spatial]
        x = x.reshape(B, T, self.n_spatial_points)   # [B, n_time, n_spatial]
        x = x.transpose(1, 2)                        # [B, n_spatial, n_time]

        return x


class SpatialProfileBaselineAutoEncoder(ModalityAutoEncoder):

    def __init__(
            self,
            n_channels: int,
            d_model: int = 64,
            n_tokens: int = 4,
            n_spatial_points: int = 50,
            n_time_points: int = 50,
            kernel_size: int = 3,
            n_transformer_layers: int = 4,
            n_heads: int = 8,
    ):
        super().__init__(n_channels, d_model, n_tokens)

        self.encoder = SpatialProfileBaselineEncoder(
            n_channels, d_model, n_tokens,
            n_spatial_points, n_time_points,
            kernel_size, n_transformer_layers, n_heads,
        )
        self.decoder = SpatialProfileBaselineDecoder(
            n_channels, d_model, n_tokens,
            n_spatial_points, n_time_points,
            kernel_size,
        )

    def forward(self, x):
        """
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(reconstructed, tokens)`` where reconstructed has the same
            shape as the input and tokens is ``(B, n_tokens, d_model)``.
        """
        n_time = x.shape[-1]
        tokens = self.encoder(x)
        out = self.decoder(tokens)
        if out.shape[-1] != n_time:
            out = F.adaptive_avg_pool1d(out, n_time)
        return out, tokens


def create_spatial_profile_test_signal(
    batch_size=4,
    n_spatial_points=50,
    n_time_points=50,
):
    signal = np.zeros((batch_size, n_spatial_points, n_time_points))
    x_spatial = np.linspace(0, 1, n_spatial_points)
    t_temporal = np.linspace(0, 1, n_time_points)

    if batch_size > 0:
        signal[0, :, :] = 1.0
    if batch_size > 1:
        for t in range(n_time_points):
            signal[1, :, t] = x_spatial
    if batch_size > 2:
        midpoint = n_spatial_points // 2
        signal[2, midpoint:, :] = 1.0
    if batch_size > 3:
        for t_idx, t in enumerate(t_temporal):
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
    print(f"Input:  {x_sp.shape}")
    print(f"Tokens: {tokens_sp.shape}")
    print(f"Recon:  {recon_sp.shape}")
