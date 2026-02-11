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
    Encodes spatio-temporal profiles (e.g., Thomson scattering electron density).

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
    verbose : bool, optional
        If True, print debug information during initialization, by default False

    Attributes
    ----------
    spatial_encoder : nn.Module
        Encodes spatial structure at each time point
    temporal_encoder : nn.Module
        Encodes temporal evolution of spatial features
    """

    def __init__(
        self,
        n_spatial_points: int = 50,
        n_time_points: int = 50,
        d_model: int = 512,
        n_output_tokens: int = 10,
        verbose: bool = False,
    ):
        super().__init__()

        self.n_spatial_points = n_spatial_points
        self.n_time_points = n_time_points
        self.d_model = d_model
        self.n_output_tokens = n_output_tokens
        self.verbose = verbose

        self.activation = nn.GELU()

        # Spatial encoder: Process spatial profile at each time step
        # Input: [B*T, n_spatial_points] → Output: [B*T, d_spatial]
        self.spatial_encoder = nn.Sequential(
            nn.Linear(n_spatial_points, 128),
            self.activation,
            nn.Linear(128, 256),
            self.activation,
            nn.Linear(256, d_model),
        )

        # Temporal encoder: Process evolution of spatial features
        # Input: [B, T, d_model] → Output: [B, n_output_tokens, d_model]
        self.temporal_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=5,
            stride=2,
            padding=2,
        )

        self.adaptive_pool = nn.AdaptiveAvgPool1d(n_output_tokens)

        self.norm = nn.LayerNorm(d_model)

        if self.verbose:
            print(f"SpatialProfileEncoder initialized:")
            print(f"  Spatial points: {n_spatial_points}")
            print(f"  Time points: {n_time_points}")
            print(f"  Output tokens: {n_output_tokens}")

    def forward(self, x):
        """
        Encode spatio-temporal profiles into tokens.

        Parameters
        ----------
        x : torch.Tensor
            Input profiles of shape [batch, n_spatial_points, n_time_points]
            Each time slice is a spatial profile

        Returns
        -------
        torch.Tensor
            Encoded tokens of shape [batch, n_output_tokens, d_model]
        """
        B, S, T = x.shape

        # Reshape to process each time step independently
        # [B, S, T] → [B, T, S] → [B*T, S]
        x = x.transpose(1, 2)  # [B, T, S]
        x = x.reshape(B * T, S)  # [B*T, S]

        # Encode spatial structure
        x = self.spatial_encoder(x)  # [B*T, d_model]

        # Reshape back to separate batch and time
        x = x.reshape(B, T, self.d_model)  # [B, T, d_model]

        # Transpose for temporal convolution
        x = x.transpose(1, 2)  # [B, d_model, T]

        # Encode temporal evolution
        x = self.activation(self.temporal_conv(x))  # [B, d_model, T']

        # Pool to exact number of tokens
        x = self.adaptive_pool(x)  # [B, d_model, n_output_tokens]

        # Transpose back
        x = x.transpose(1, 2)  # [B, n_output_tokens, d_model]

        # Layer norm
        x = self.norm(x)

        return x


class SpatialProfileDecoder(nn.Module):
    """
    Decodes from transformer output back to spatio-temporal profiles.

    Parameters
    ----------
    n_spatial_points : int, optional
        Number of spatial measurement points, by default 50
    n_time_points : int, optional
        Number of temporal samples to output (e.g., 5 for 50ms @ 100Hz), by default 5
    d_model : int, optional
        Model dimension from transformer, by default 512
    n_input_tokens : int, optional
        Number of input tokens from transformer, by default 10
    verbose : bool, optional
        If True, print debug information during initialization, by default False
    """

    def __init__(
        self,
        n_spatial_points: int = 50,
        n_time_points: int = 5,
        d_model: int = 512,
        n_input_tokens: int = 10,
        verbose: bool = False,
    ):
        super().__init__()

        self.n_spatial_points = n_spatial_points
        self.n_time_points = n_time_points
        self.d_model = d_model
        self.n_input_tokens = n_input_tokens
        self.verbose = verbose

        # Temporal decoder: Upsample from tokens to time steps
        self.temporal_deconv = nn.ConvTranspose1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1,
        )

        self.adaptive_pool = nn.AdaptiveAvgPool1d(n_time_points)

        self.activation = nn.GELU()

        # Spatial decoder: Reconstruct spatial profile from features
        self.spatial_decoder = nn.Sequential(
            nn.Linear(d_model, 256),
            self.activation,
            nn.Linear(256, 128),
            self.activation,
            nn.Linear(128, n_spatial_points),
        )

        if self.verbose:
            print(f"SpatialProfileDecoder initialized:")
            print(f"  Spatial points: {n_spatial_points}")
            print(f"  Time points: {n_time_points}")
            print(f"  Input tokens: {n_input_tokens}")

    def forward(self, x):
        """
        Decode tokens back to spatio-temporal profiles.

        Parameters
        ----------
        x : torch.Tensor
            Input tokens of shape [batch, n_input_tokens, d_model]

        Returns
        -------
        torch.Tensor
            Decoded profiles of shape [batch, n_spatial_points, n_time_points]
        """
        B = x.shape[0]

        # Transpose for temporal processing
        x = x.transpose(1, 2)  # [B, d_model, n_input_tokens]

        # Upsample temporally
        x = self.activation(self.temporal_deconv(x))  # [B, d_model, T']

        # Pool to exact time points
        x = self.adaptive_pool(x)  # [B, d_model, n_time_points]

        # Transpose back
        x = x.transpose(1, 2)  # [B, n_time_points, d_model]

        # Reshape to process each time step
        T = x.shape[1]
        x = x.reshape(B * T, self.d_model)  # [B*T, d_model]

        # Decode spatial structure
        x = self.spatial_decoder(x)  # [B*T, n_spatial_points]

        # Reshape back
        x = x.reshape(B, T, self.n_spatial_points)  # [B, T, S]

        # Transpose to [B, S, T]
        x = x.transpose(1, 2)  # [B, S, T]

        return x


if __name__ == "__main__":
    print("=" * 60)
    print("Testing TimeSeriesEncoder and TimeSeriesDecoder")
    print("=" * 60)

    # Create encoder with verbose=True
    encoder = SpatialProfileEncoder(
        n_spatial_points=50,
        n_time_points=50,
        d_model=512,
        n_output_tokens=10,
        verbose=True,
    )

    # Create decoder with verbose=True
    decoder = SpatialProfileDecoder(
        n_spatial_points=50,
        n_time_points=5,
        d_model=512,
        verbose=True,
    )

    # Create deterministic test signal
    print("Generating deterministic test signal...")
    x = create_spatial_profile_test_signal(
        batch_size=4, n_spatial_points=50, n_time_points=50
    )
    print(f"Input shape: {x.shape}")
    print(f"Input statistics - Mean: {x.mean():.4f}, Std: {x.std():.4f}")

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

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
