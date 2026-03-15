"""Spectrogram baseline modality autoencoder implementation.

Conv2D + LSTM architecture with batch-folded channels.
Ported from dev/train_mhr_conv2d_1channel_lstm_better.ipynb.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from .base import ModalityEncoder, ModalityDecoder, ModalityAutoEncoder


class ResidualBlock(nn.Module):
    """Residual block with two Conv2d layers and optional GroupNorm."""

    DEFAULT_GROUPS = 32

    def __init__(self, in_channels, out_channels=None, use_groupnorm=False):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels

        if use_groupnorm:
            norm_layer = lambda c: nn.GroupNorm(
                num_groups=min(self.DEFAULT_GROUPS, c), num_channels=c
            )
        else:
            norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = norm_layer(out_channels)
        self.activation = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = norm_layer(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                norm_layer(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.activation(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += residual
        return self.activation(out)


class LSTMBlock(nn.Module):
    """Bidirectional LSTM over the time dimension of a 4D feature map."""

    def __init__(self, channels, freq_dim, hidden_dim=128, num_layers=1):
        super().__init__()
        self.channels = channels
        input_dim = channels * freq_dim
        self.freq_dim = freq_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, input_dim),
            nn.GELU(),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x):
        B, C, F, T = x.shape
        residual = x

        x_seq = rearrange(x, "b c f t -> b t (c f)")
        lstm_out, _ = self.lstm(x_seq)
        proj_out = self.proj(lstm_out)
        x_back = rearrange(proj_out, "b t (c f) -> b c f t", c=C, f=F)
        x_back = self.conv(x_back)

        return self.norm(x_back + residual)


class SpectrogramBaselineEncoder(ModalityEncoder):
    """Conv2D + LSTM encoder for multichannel spectrograms.

    Uses batch-folded channels: processes each channel independently
    via ``(B, C, F, T) -> (B*C, 1, F, T)``, then recombines.

    Parameters
    ----------
    n_channels : int
        Number of spectrogram channels (e.g. 8 for MHR, 48 for ECE).
    d_model : int
        Token embedding dimension for fusion.
    n_tokens : int
        Number of output tokens for the fusion transformer.
    hidden_dim : int
        Hidden dimension for Conv2D layers.
    latent_dim : int
        Number of latent channels per input channel.
    freq_dim : int
        Frequency dimension after downsampling (for LSTM).
    lstm_hidden : int
        LSTM hidden size.
    lstm_layers : int
        Number of LSTM layers.
    """

    def __init__(
        self,
        n_channels: int,
        d_model: int = 64,
        n_tokens: int = 20,
        hidden_dim: int = 64,
        latent_dim: int = 2,
        freq_dim: int = 16,
        lstm_hidden: int = 32,
        lstm_layers: int = 1,
        **kwargs,
    ):
        super().__init__(n_channels, d_model, n_tokens)
        self.n_channels = n_channels
        self.d_model = d_model
        self.n_tokens = n_tokens
        self.latent_dim = latent_dim
        self.freq_dim = freq_dim

        # Per-channel encoder (batch-folded, in_channels=1)
        layers = []
        c = 1
        dims = [hidden_dim, hidden_dim, hidden_dim]
        for d in dims:
            layers.append(ResidualBlock(c, d))
            layers.append(
                nn.Conv2d(d, d, kernel_size=3, stride=(2, 2), padding=1, bias=False)
            )
            c = d
        self.net = nn.Sequential(*layers)
        self.to_latent = nn.Conv2d(dims[-1], latent_dim, 1)

        self.lstm_block = LSTMBlock(
            channels=latent_dim,
            freq_dim=freq_dim,
            hidden_dim=lstm_hidden,
            num_layers=lstm_layers,
        )

        # Token projection: (B, C*latent_dim*F'*T') -> (B, n_tokens, d_model)
        # We use adaptive pooling + linear to get exact n_tokens
        self.token_pool = nn.AdaptiveAvgPool1d(n_tokens)
        self.token_proj = nn.Linear(n_channels * latent_dim, d_model)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            ``(B, C, F, T)`` spectrogram input.

        Returns
        -------
        torch.Tensor
            ``(B, n_tokens, d_model)`` token sequence.
        """
        B, C, F, T = x.shape

        # Batch-fold channels
        x_flat = rearrange(x, "b c f t -> (b c) 1 f t")
        z = self.to_latent(self.net(x_flat))  # (B*C, latent_dim, F', T')
        z = self.lstm_block(z)

        # Recombine channels
        _, ld, Fp, Tp = z.shape
        z = rearrange(z, "(b c) ld fp tp -> b (c ld) (fp tp)", b=B, c=C)
        # z: (B, C*latent_dim, F'*T')

        # Pool to n_tokens along spatial-temporal dim
        z = self.token_pool(z)  # (B, C*latent_dim, n_tokens)
        z = z.transpose(1, 2)   # (B, n_tokens, C*latent_dim)

        # Project to d_model
        tokens = self.token_proj(z)  # (B, n_tokens, d_model)
        return tokens


class SpectrogramBaselineDecoder(ModalityDecoder):
    """Conv2D + LSTM decoder for multichannel spectrograms.

    Reverses the encoder: unprojection -> reshape -> per-channel decode.

    Parameters
    ----------
    n_channels : int
        Number of output spectrogram channels.
    d_model : int
        Token embedding dimension.
    n_tokens : int
        Number of input tokens.
    hidden_dim : int
        Hidden dimension for Conv2D layers.
    latent_dim : int
        Number of latent channels per input channel.
    freq_dim : int
        Frequency dimension of bottleneck.
    lstm_hidden : int
        LSTM hidden size.
    lstm_layers : int
        Number of LSTM layers.
    """

    def __init__(
        self,
        n_channels: int,
        d_model: int = 64,
        n_tokens: int = 20,
        hidden_dim: int = 64,
        latent_dim: int = 2,
        freq_dim: int = 16,
        lstm_hidden: int = 32,
        lstm_layers: int = 1,
        **kwargs,
    ):
        super().__init__(n_channels, d_model)
        self.n_channels = n_channels
        self.d_model = d_model
        self.n_tokens = n_tokens
        self.latent_dim = latent_dim
        self.freq_dim = freq_dim

        # Unproject: (B, n_tokens, d_model) -> (B, n_tokens, C*latent_dim)
        self.token_unproj = nn.Linear(d_model, n_channels * latent_dim)

        # We need to know the time dimension of the bottleneck
        # It's computed from n_tokens: total spatial = freq_dim * time_dim
        # We set time_dim = n_tokens (adaptive pool will handle mismatch)
        self.bottleneck_time = n_tokens

        self.lstm_block = LSTMBlock(
            channels=latent_dim,
            freq_dim=freq_dim,
            hidden_dim=lstm_hidden,
            num_layers=lstm_layers,
        )

        self.from_latent = nn.Conv2d(latent_dim, hidden_dim, 1)

        dims = [hidden_dim, hidden_dim, hidden_dim]
        layers = []
        c = dims[0]
        for d in dims[1:]:
            layers.append(ResidualBlock(c, d))
            layers.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=(2, 2), mode="nearest"),
                    nn.Conv2d(d, d, kernel_size=3, padding=1, bias=False),
                )
            )
            c = d

        # Final upsample layer outputs 1 channel (batch-folded)
        layers.append(ResidualBlock(c, 1))
        layers.append(
            nn.Sequential(
                nn.Upsample(scale_factor=(2, 2), mode="nearest"),
                nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False),
            )
        )
        self.net = nn.Sequential(*layers)
        self.head = nn.Conv2d(1, 1, 1)

    def forward(self, tokens, output_shape=None):
        """
        Parameters
        ----------
        tokens : torch.Tensor
            ``(B, n_tokens, d_model)`` token sequence.
        output_shape : tuple, optional
            Target ``(F, T)`` for the output spectrogram.

        Returns
        -------
        torch.Tensor
            ``(B, C, F, T)`` reconstructed spectrogram.
        """
        B = tokens.shape[0]

        # Unproject tokens
        z = self.token_unproj(tokens)  # (B, n_tokens, C*latent_dim)
        z = z.transpose(1, 2)  # (B, C*latent_dim, n_tokens)

        # Reshape to (B*C, latent_dim, freq_dim, time_dim)
        # Adaptive pool n_tokens -> freq_dim * bottleneck_time
        target_spatial = self.freq_dim * self.bottleneck_time
        z = F.adaptive_avg_pool1d(z, target_spatial)  # (B, C*latent_dim, F'*T')
        z = rearrange(
            z,
            "b (c ld) (fp tp) -> (b c) ld fp tp",
            c=self.n_channels,
            ld=self.latent_dim,
            fp=self.freq_dim,
            tp=self.bottleneck_time,
        )

        z = self.lstm_block(z)
        y = self.from_latent(z)
        y = self.net(y)
        y = self.head(y)

        # Un-batch-fold channels
        y = rearrange(y, "(b c) 1 f t -> b c f t", b=B, c=self.n_channels)

        # Interpolate to match original shape if needed
        if output_shape is not None and y.shape[2:] != torch.Size(output_shape):
            y = F.interpolate(y, size=output_shape, mode="bilinear", align_corners=False)

        return y


class SpectrogramBaselineAutoEncoder(ModalityAutoEncoder):
    """Multichannel Conv2D + LSTM spectrogram autoencoder.

    Processes each channel independently via batch folding, then
    projects the latent space to a token sequence for fusion.

    Parameters
    ----------
    n_channels : int
        Number of spectrogram channels.
    d_model : int
        Token embedding dimension.
    n_tokens : int
        Number of output tokens.
    hidden_dim : int
        Hidden dimension for conv layers.
    latent_dim : int
        Latent channels per input channel.
    freq_dim : int
        Bottleneck frequency dimension.
    lstm_hidden : int
        LSTM hidden size.
    lstm_layers : int
        Number of LSTM layers.
    """

    def __init__(
        self,
        n_channels: int,
        d_model: int = 64,
        n_tokens: int = 20,
        hidden_dim: int = 64,
        latent_dim: int = 2,
        freq_dim: int = 16,
        lstm_hidden: int = 32,
        lstm_layers: int = 1,
        **kwargs,
    ):
        super().__init__(n_channels, d_model, n_tokens)

        self.encoder = SpectrogramBaselineEncoder(
            n_channels=n_channels,
            d_model=d_model,
            n_tokens=n_tokens,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            freq_dim=freq_dim,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
        )
        self.decoder = SpectrogramBaselineDecoder(
            n_channels=n_channels,
            d_model=d_model,
            n_tokens=n_tokens,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            freq_dim=freq_dim,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            ``(B, C, F, T)`` spectrogram.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(reconstructed, tokens)`` where reconstructed has the same
            shape as the input and tokens is ``(B, n_tokens, d_model)``.
        """
        B, C, F, T = x.shape
        tokens = self.encoder(x)  # (B, n_tokens, d_model)
        reconstructed = self.decoder(tokens, output_shape=(F, T))
        return reconstructed, tokens


if __name__ == "__main__":
    # python -m tokamak_foundation_model.models.modality.spectrogram_baseline

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _run_test(label, n_channels, freq, time, **kwargs):
        print(f"=== {label} (n_channels={n_channels}) ===")
        ae = SpectrogramBaselineAutoEncoder(n_channels, **kwargs).to(device)
        n_params = sum(p.numel() for p in ae.parameters())
        print(f"  Parameters: {n_params:,}")

        x = torch.randn(2, n_channels, freq, time, device=device)
        with torch.inference_mode():
            recon, tokens = ae(x)

        print(f"  Input:   {tuple(x.shape)} ({x.numel():,} values)")
        print(f"  Tokens:  {tuple(tokens.shape)} ({tokens.numel():,} values)")
        print(f"  Recon:   {tuple(recon.shape)}")
        print(f"  Compression: {x.numel() / tokens.numel():.1f}:1")
        assert recon.shape == x.shape, f"Shape mismatch: {recon.shape} vs {x.shape}"
        print()

    _run_test("MHR", n_channels=8, freq=128, time=391, d_model=64, n_tokens=20)
    _run_test("CO2", n_channels=4, freq=128, time=256, d_model=64, n_tokens=20)
    _run_test("ECE", n_channels=40, freq=128, time=100, d_model=64, n_tokens=20)
