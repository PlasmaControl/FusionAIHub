import torch
import torch.nn as nn
import numpy as np


def create_spatial_profile_test_signal(
    batch_size=4, n_spatial_points=50, n_time_points=50
):
    """
    Create deterministic test signal for spatial profiles with simple patterns.

    Parameters
    ----------
    batch_size : int, optional
        Number of samples in batch, by default 4
    n_spatial_points : int, optional
        Number of spatial measurement points, by default 50
    n_time_points : int, optional
        Number of temporal samples, by default 50

    Returns
    -------
    torch.Tensor
        Test signal of shape [batch_size, n_spatial_points, n_time_points]

    Notes
    -----
    Different test patterns per batch for easy debugging:
    - Batch 0: Constant profile (all ones) - tests DC preservation
    - Batch 1: Linear spatial gradient (0 to 1) - tests spatial interpolation
    - Batch 2: Step function in space (0 before midpoint, 1 after) - tests spatial edges
    - Batch 3: Traveling pulse of width 20

    All patterns are deterministic and mathematically simple for verification.
    """
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


class SpatialProfileEncoder(nn.Module):
    """
    Encodes spatio-temporal profiles (e.g., Thomson scattering, CER, MSE)
    using a spatial MLP followed by temporal 1D convolutions.

    Parameters
    ----------
    n_spatial_points : int, optional
        Number of spatial measurement points, by default 50
    n_time_points : int, optional
        Number of temporal samples (e.g., 50 for 500ms @ 100Hz), by default 50
    d_model : int, optional
        Model dimension for transformer, by default 512
    n_output_tokens : int, optional
        Number of output tokens, by default 10
    kernel_size : int
        Kernel size for temporal convolution
    verbose : bool, optional
        If True, print debug information during initialization, by default False

    Attributes
    ----------
    spatial_encoder : nn.Sequential
        MLP that encodes each spatial profile independently
    temporal_conv : nn.Conv1d
        Compresses temporal dimension
    adaptive_pool : nn.AdaptiveAvgPool1d
        Ensures exact output token count
    """

    def __init__(
            self,
            n_spatial_points: int = 50,
            n_time_points: int = 50,
            d_model: int = 512,
            n_output_tokens: int = 10,
            kernel_size: int = 5,
            verbose: bool = False,
    ):
        super().__init__()

        self.n_spatial_points = n_spatial_points
        self.n_time_points = n_time_points
        self.d_model = d_model
        self.n_output_tokens = n_output_tokens
        self.verbose = verbose

        self.adaptive_pool = nn.AdaptiveAvgPool1d(n_output_tokens)
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

        if self.verbose:
            print(f"SpatialProfileEncoder:")
            print(f"  Spatial points: {n_spatial_points}")
            print(f"  Time points:    {n_time_points}")
            print(f"  Output tokens:  {n_output_tokens}")

    def forward(self, x):
        """
        Encode spatio-temporal profile into tokens.

        Parameters
        ----------
        x : torch.Tensor
            Input profiles of shape [batch, n_spatial_points, n_time_points]

        Returns
        -------
        torch.Tensor
            Encoded tokens of shape [batch, n_output_tokens, d_model]
        """
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


class SpatialProfileDecoder(nn.Module):
    """
    Mirrors SpatialProfileEncoder for pre-training via masked autoencoding.
    Reconstructs the original spatio-temporal profile from encoder tokens.

    Parameters
    ----------
    n_spatial_points : int, optional
        Number of spatial measurement points, by default 50
    n_time_points : int, optional
        Number of temporal samples to reconstruct, by default 50
    d_model : int, optional
        Model dimension from encoder, by default 512
    n_input_tokens : int, optional
        Number of input tokens from encoder, by default 10
    kernel_size : int
        Kernel size for temporal convolution
    verbose : bool, optional
        If True, print debug information during initialization, by default False

    Attributes
    ----------
    temporal_deconv : nn.ConvTranspose1d
        Mirrors temporal_conv in encoder
    spatial_decoder : nn.Sequential
        Mirrors spatial_encoder MLP (reversed)
    adaptive_pool : nn.AdaptiveAvgPool1d
        Ensures exact output time points
    """

    def __init__(
            self,
            n_spatial_points: int = 50,
            n_time_points: int = 50,
            d_model: int = 512,
            n_input_tokens: int = 10,
            kernel_size: int = 5,
            verbose: bool = False
    ):
        super().__init__()

        self.n_spatial_points = n_spatial_points
        self.n_time_points = n_time_points
        self.d_model = d_model
        self.n_input_tokens = n_input_tokens
        self.verbose = verbose

        self.activation = nn.GELU()
        self.adaptive_pool = nn.AdaptiveAvgPool1d(n_time_points)

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

        if self.verbose:
            print(f"SpatialProfileDecoder:")
            print(f"  Spatial points: {n_spatial_points}")
            print(f"  Time points:    {n_time_points}")
            print(f"  Input tokens:   {n_input_tokens}")

    def forward(self, x):
        """
        Decode tokens back to original spatio-temporal profile (pre-training only).

        Parameters
        ----------
        x : torch.Tensor
            Input tokens of shape [batch, n_input_tokens, d_model]

        Returns
        -------
        torch.Tensor
            Reconstructed profiles of shape [batch, n_spatial_points, n_time_points]
        """
        B = x.shape[0]

        # Upsample temporal dimension
        x = x.transpose(1, 2)              # [B, d_model, n_input_tokens]
        x = self.activation(self.temporal_deconv(x))  # [B, d_model, T']
        x = self.adaptive_pool(x)                     # [B, d_model, n_time]

        # Decode spatial structure at each time step independently
        x = x.transpose(1, 2)                         # [B, n_time, d_model]
        T = x.shape[1]
        x = x.reshape(B * T, self.d_model)            # [B*T, d_model]
        x = self.spatial_decoder(x)                   # [B*n_time, n_spatial]
        x = x.reshape(B, T, self.n_spatial_points)    # [B, n_time, n_spatial]
        x = x.transpose(1, 2)                         # [B, n_spatial, n_time]

        return x


if __name__ == "__main__":
    print("=" * 60)
    print("SpatialProfileEncoder / SpatialProfileDecoder")
    print("=" * 60)
    sp_enc = SpatialProfileEncoder(n_spatial_points=50, n_time_points=50,
                                   d_model=512, n_output_tokens=10, kernel_size=3,
                                   verbose=True)
    sp_dec = SpatialProfileDecoder(n_spatial_points=50, n_time_points=50,
                                   d_model=512, n_input_tokens=10, kernel_size=3,
                                   verbose=True)
    x_sp = create_spatial_profile_test_signal()
    tokens_sp = sp_enc(x_sp)
    recon_sp = sp_dec(tokens_sp)
    print(f"Input:  {x_sp.shape}")       # [4, 50, 50]
    print(f"Tokens: {tokens_sp.shape}")  # [4, 10, 512]
    print(f"Recon:  {recon_sp.shape}")   # [4, 50, 50]