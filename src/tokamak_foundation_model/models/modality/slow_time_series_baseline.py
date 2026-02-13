import torch
import torch.nn as nn
import torch.nn.functional as F


class SlowTimeSeriesEncoder(nn.Module):
    """
    Encodes low-rate time-series diagnostics using 1D convolutions (no stride).

    Parameters
    ----------
    n_channels : int
        Number of input channels
    d_model : int
        Model dimension
    n_output_tokens : int
        Number of temporal tokens to output
    """

    def __init__(self, n_channels: int, d_model: int = 64,
                 n_tokens: int = 0):
        super().__init__()

        self.n_channels = n_channels
        self.d_model = d_model
        self.n_tokens = n_tokens
        self.n_conv_layers = 3
        self.kernel_size = 7

        # Build channel progression: n_channels -> intermediates -> d_model
        intermediate = [min(32 * (2 ** i), d_model) for i in range(self.n_conv_layers - 1)]
        channels = [n_channels] + intermediate + [d_model]

        self.conv_layers = nn.ModuleList([
            nn.Conv1d(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
            )
            for i in range(self.n_conv_layers)
        ])

        if n_tokens > 0:
            self.adaptive_pool = nn.AdaptiveAvgPool1d(n_tokens)

        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input of shape [batch, n_channels, input_length]

        Returns
        -------
        torch.Tensor
            Encoded tokens of shape [batch, n_tokens, d_model]
        """
        for conv in self.conv_layers:
            x = self.activation(conv(x))

        if self.n_tokens > 0:
            x = self.adaptive_pool(x)                 # [B, d_model, n_tokens]

        x = x.transpose(1, 2)                     # [B, n_tokens, d_model]
        x = self.norm(x)

        return x


class SlowTimeSeriesDecoder(nn.Module):
    """
    Mirrors SlowTimeSeriesEncoder for pre-training via autoencoding.

    Parameters
    ----------
    n_channels : int
        Number of output channels
    d_model : int
        Model dimension from encoder
    n_output_tokens : int
        Number of input tokens from encoder
    """

    def __init__(self, n_channels: int, d_model: int = 64,
                 n_tokens: int = 0):
        super().__init__()

        self.n_channels = n_channels
        self.d_model = d_model
        self.n_tokens = n_tokens
        self.n_deconv_layers = 3
        self.kernel_size = 7

        # Mirror encoder channel progression (reversed)
        intermediate = [min(32 * (2 ** i), d_model) for i in range(self.n_deconv_layers - 1)]
        channels = [d_model] + list(reversed(intermediate)) + [n_channels]

        self.deconv_layers = nn.ModuleList([
            nn.ConvTranspose1d(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
            )
            for i in range(self.n_deconv_layers)
        ])

        self.activation = nn.GELU()

    def forward(self, x, output_length=None):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tokens of shape [batch, n_tokens, d_model]
        output_length : int, optional
            Target output length. If provided, adaptive pool to this size.

        Returns
        -------
        torch.Tensor
            Reconstructed time-series of shape [batch, n_channels, output_length]
        """
        x = x.transpose(1, 2)                     # [B, d_model, n_tokens]

        for i, deconv in enumerate(self.deconv_layers):
            x = deconv(x)
            if i < len(self.deconv_layers) - 1:
                x = self.activation(x)

        if output_length is not None:
            x = F.adaptive_avg_pool1d(x, output_length)

        return x


class SlowTimeSeriesAutoEncoder(nn.Module):

    def __init__(self, n_channels: int, d_model: int = 64,
                 n_tokens: int = 0):
        super().__init__()
        self.encoder = SlowTimeSeriesEncoder(n_channels, d_model, n_tokens)
        self.decoder = SlowTimeSeriesDecoder(n_channels, d_model, n_tokens)

    def forward(self, x):
        output_length = x.shape[-1]
        return self.decoder(self.encoder(x), output_length=output_length)


if __name__ == "__main__":
    # python -m tokamak_foundation_model.models.modality.slow_time_series_baseline
    B, C, T = 4, 6, 100
    d_model = 64

    n_tokens = 10

    encoder = SlowTimeSeriesEncoder(C, d_model, n_tokens=n_tokens)
    decoder = SlowTimeSeriesDecoder(C, d_model, n_tokens=n_tokens)

    x = torch.randn(B, C, T)
    z = encoder(x)
    y = decoder(z, output_length=T)

    print(f"Input:   {x.shape}")
    print(f"Encoded: {z.shape}")
    print(f"Decoded: {y.shape}")

    autoencoder = SlowTimeSeriesAutoEncoder(C, d_model, n_tokens=n_tokens)
    y = autoencoder(x)
    
    print(f"Autoencoder Input:  {x.shape}, Output: {y.shape}")
