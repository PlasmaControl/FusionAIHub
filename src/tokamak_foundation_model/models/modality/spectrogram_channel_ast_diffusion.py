"""Channel-Attention AST + Diffusion decoder autoencoder for tokamak spectrograms.

Replaces the deterministic decoder of Channel-AST with a **diffusion-based
denoiser** conditioned on encoder latent ``z`` and noise timestep ``t``.
The diffusion objective directly addresses the high-frequency smearing problem
inherent to L1/L2 pixel-wise losses by modelling the full conditional
distribution ``p(x|z)`` and generating sharp samples.

Architecture
------------
Encoder (reuse ``_ChannelASTEncoder``):
  Per-channel frame embed: (B, C, N, F*fw) → Linear → (B, C, N, d_model)
  + channel_pos_embed + time_pos_embed
  n_enc_layers × ChannelTimeBlock
  LayerNorm regularisation (DiTo-style, no KL, no FSQ)

Diffusion Decoder:
  Frame embed noisy x_t: (B, C, N, F*fw) → Linear → (B, C, N, d_model)
  + channel_pos_embed + time_pos_embed
  n_dec_layers × AdaLN-Zero ChannelTimeBlock (timestep-conditioned)
  Layer-wise additive z conditioning
  Frame unembed: Linear(d_model → F*fw)

Noise schedule — Rectified flow:
  x_t = (1-t)*x + t*ε,  t ∈ (0, 1)
  Model uses **x-prediction**: directly predicts clean x from noisy x_t
  (JiT / BackToBasics shows x-prediction outperforms v/ε-prediction for
  high-dimensional patches)

Timestep sampling — Logit-normal (JiT):
  logit(t) ~ N(μ, σ²), default μ=-0.8, σ=0.8

Return contract
---------------
Training : (reconstruction_onestep, diffusion_loss)
           reconstruction_onestep is the one-step x̂ (for metrics/logging)
           diffusion_loss is a scalar MSE(x̂, x) for backprop
Eval     : reconstructed — shape (B, C, F, T) via multi-step Euler ODE

References
----------
- DiTo: Diffusion Autoencoders are Scalable Image Tokenizers (arXiv:2501.18593)
- JiT / BackToBasics: Just Image Transformers (Tschannen et al.)
- PixelDiT: Pixel Diffusion Transformers
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tokamak_foundation_model.models.modality.base import ModalityAutoEncoder
from tokamak_foundation_model.models.modality.spectrogram_channel_ast_fsq import (
    _ChannelASTEncoder,
    _ChannelTimeBlock,
)


# ---------------------------------------------------------------------------
# Sinusoidal timestep embedding
# ---------------------------------------------------------------------------

class _TimestepEmbedding(nn.Module):
    """Sinusoidal positional encoding of scalar t → MLP → d_model.

    Parameters
    ----------
    d_model : int
        Output dimension.
    max_period : float
        Controls the minimum frequency of the sinusoidal embedding.
    """

    def __init__(self, d_model: int, max_period: float = 10000.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.SiLU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, t: Tensor) -> Tensor:
        """(B,) → (B, d_model)."""
        half = self.d_model // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=t.device, dtype=t.dtype)
            / half
        )
        args = t[:, None] * freqs[None, :]  # (B, half)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, d_model)
        if self.d_model % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


# ---------------------------------------------------------------------------
# AdaLN-Zero wrapper around ChannelTimeBlock
# ---------------------------------------------------------------------------

class _AdaLNChannelTimeBlock(nn.Module):
    """ChannelTimeBlock with AdaLN-Zero timestep modulation.

    Wraps an existing ``_ChannelTimeBlock`` without modifying it.  The
    timestep embedding produces (scale, shift, gate) via a linear projection
    with zero-initialised weights on the gate — so the block initially
    acts as identity for the timestep conditioning, ensuring stable training.

    Parameters
    ----------
    d_model : int
        Hidden dimension.
    n_heads : int
        Attention heads for channel attention.
    dropout : float
        Dropout rate.
    time_conv_kernel : int
        Kernel size for temporal ConvNeXt block.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        time_conv_kernel: int,
    ) -> None:
        super().__init__()
        self.block = _ChannelTimeBlock(d_model, n_heads, dropout, time_conv_kernel)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 3 * d_model),
        )
        # Zero-initialise the gate portion for training stability
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        """(B, C, N, D), (B, D) → (B, C, N, D)."""
        params = self.adaLN(t_emb)  # (B, 3*D)
        scale, shift, gate = params.chunk(3, dim=-1)  # each (B, D)
        # Broadcast to (B, 1, 1, D)
        scale = scale[:, None, None, :]
        shift = shift[:, None, None, :]
        gate = gate[:, None, None, :]
        # Pre-modulate → run block → gated residual
        x_mod = x * (1.0 + scale) + shift
        out = self.block(x_mod)
        return x + gate * (out - x)


# ---------------------------------------------------------------------------
# Diffusion decoder
# ---------------------------------------------------------------------------

class _DiffusionDecoder(nn.Module):
    """Denoiser network: predicts clean x from noisy x_t, conditioned on z and t.

    Parameters
    ----------
    freq_bins : int
        Frequency dimension (F).
    frame_width : int
        Number of time steps per frame token.
    d_model : int
        Hidden dimension.
    n_heads : int
        Attention heads for channel attention.
    n_layers : int
        Number of AdaLN-Zero ChannelTimeBlocks.
    dropout : float
        Dropout rate.
    max_channels : int
        Channel positional embedding table capacity.
    max_time_frames : int
        Time positional embedding table capacity.
    time_conv_kernel : int
        Kernel size for temporal ConvNeXt blocks.
    """

    def __init__(
        self,
        freq_bins: int,
        frame_width: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        max_channels: int,
        max_time_frames: int,
        time_conv_kernel: int,
    ) -> None:
        super().__init__()
        self.freq_bins = freq_bins
        self.frame_width = frame_width

        # Frame embedding for noisy input
        self.frame_proj = nn.Linear(freq_bins * frame_width, d_model)

        # Positional embeddings (separate from encoder)
        self.channel_pos_embed = nn.Parameter(
            torch.zeros(1, max_channels, 1, d_model)
        )
        self.time_pos_embed = nn.Parameter(
            torch.zeros(1, 1, max_time_frames, d_model)
        )
        nn.init.trunc_normal_(self.channel_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.time_pos_embed, std=0.02)

        # Timestep embedding
        self.t_embed = _TimestepEmbedding(d_model)

        # AdaLN-Zero ChannelTimeBlocks
        self.blocks = nn.ModuleList([
            _AdaLNChannelTimeBlock(d_model, n_heads, dropout, time_conv_kernel)
            for _ in range(n_layers)
        ])

        # Per-layer z conditioning scale (initialised small)
        self.z_scale = nn.ParameterList([
            nn.Parameter(torch.full((), 0.1))
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Frame unembed
        self.frame_unembed = nn.Linear(d_model, freq_bins * frame_width)

    def forward(
        self,
        x_t: Tensor,
        z: Tensor,
        t: Tensor,
        n_channels: int,
        n_frames: int,
    ) -> Tensor:
        """Predict clean x from noisy x_t.

        Parameters
        ----------
        x_t : (B, C, F, T_padded)
            Noisy spectrogram.
        z : (B, C, N, d_model)
            Encoder latent (conditioning signal).
        t : (B,)
            Noise timestep in [0, 1].
        n_channels : int
            Number of channels C.
        n_frames : int
            Number of time frames N.

        Returns
        -------
        x_hat : (B, C, F, T_padded)
            Predicted clean spectrogram.
        """
        B = x_t.shape[0]
        Fr = self.freq_bins
        fw = self.frame_width

        # Frame-embed noisy input: (B, C, F, T) → (B, C, N, F*fw) → (B, C, N, D)
        frames = (
            x_t.reshape(B, n_channels, Fr, n_frames, fw)
            .permute(0, 1, 3, 2, 4)       # (B, C, N, F, fw)
            .reshape(B, n_channels, n_frames, Fr * fw)
        )
        tokens = self.frame_proj(frames)   # (B, C, N, d_model)

        # Add positional embeddings
        tokens = (
            tokens
            + self.channel_pos_embed[:, :n_channels]
            + self.time_pos_embed[:, :, :n_frames]
        )

        # Timestep embedding
        t_emb = self.t_embed(t)  # (B, d_model)

        # AdaLN-Zero blocks with layer-wise z injection
        for i, block in enumerate(self.blocks):
            tokens = tokens + self.z_scale[i] * z
            tokens = block(tokens, t_emb)

        tokens = self.norm(tokens)

        # Frame unembed: (B, C, N, d_model) → (B, C, N, F*fw) → (B, C, F, T_padded)
        pixels = self.frame_unembed(tokens)  # (B, C, N, F*fw)
        T_padded = n_frames * fw
        x_hat = (
            pixels
            .reshape(B, n_channels, n_frames, Fr, fw)
            .permute(0, 1, 3, 2, 4)           # (B, C, F, N, fw)
            .reshape(B, n_channels, Fr, T_padded)
        )
        return x_hat


# ---------------------------------------------------------------------------
# Full Channel-AST-Diffusion autoencoder
# ---------------------------------------------------------------------------

class SpectrogramChannelASTDiffusionAutoEncoder(ModalityAutoEncoder):
    """Channel-Attention AST encoder + Diffusion decoder autoencoder.

    The encoder produces latent tokens z (conditioned for the fusion
    transformer); the diffusion decoder reconstructs sharp spectrograms
    by iterative denoising conditioned on z and timestep t.

    Parameters
    ----------
    n_channels : int
        Number of spectrogram channels.
    d_model : int
        Hidden dimension.
    n_tokens : int
        Unused; kept for interface compatibility with ModalityAutoEncoder.
    freq_bins : int
        Frequency dimension of the input spectrogram.
    frame_width : int
        Number of time steps per frame token (default 16).
    n_enc_layers, n_dec_layers : int
        Depth for encoder and decoder (default 4 each).
    n_heads : int
        Attention heads (default 4).
    dropout : float
        Dropout rate (default 0.1).
    max_channels : int
        Channel positional embedding table capacity (default 64).
    max_time_frames : int
        Time positional embedding table capacity (default 2048).
    time_conv_kernel : int
        Kernel size for temporal ConvNeXt blocks (default 7).
    logit_normal_mu : float
        Mean of the logit-normal timestep distribution (default -0.8).
    logit_normal_sigma : float
        Std of the logit-normal timestep distribution (default 0.8).
    eval_steps : int
        Number of Euler ODE steps during eval (default 20).
    """

    def __init__(
        self,
        n_channels: int,
        d_model: int = 256,
        n_tokens: int = 0,
        *,
        freq_bins: int = 512,
        frame_width: int = 16,
        n_enc_layers: int = 4,
        n_dec_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
        max_channels: int = 64,
        max_time_frames: int = 2048,
        time_conv_kernel: int = 7,
        logit_normal_mu: float = -0.8,
        logit_normal_sigma: float = 0.8,
        eval_steps: int = 20,
    ) -> None:
        super().__init__(n_channels, d_model, n_tokens)
        self.n_channels = n_channels
        self.freq_bins = freq_bins
        self.frame_width = frame_width
        self.logit_normal_mu = logit_normal_mu
        self.logit_normal_sigma = logit_normal_sigma
        self.eval_steps = eval_steps

        # Encoder (reuse Channel-AST encoder)
        self.encoder = _ChannelASTEncoder(
            freq_bins=freq_bins,
            frame_width=frame_width,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_enc_layers,
            dropout=dropout,
            max_channels=max_channels,
            max_time_frames=max_time_frames,
            time_conv_kernel=time_conv_kernel,
        )

        # DiTo-style latent regularisation (LayerNorm, no KL)
        self.latent_norm = nn.LayerNorm(d_model)

        # Diffusion decoder
        self.decoder = _DiffusionDecoder(
            freq_bins=freq_bins,
            frame_width=frame_width,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_dec_layers,
            dropout=dropout,
            max_channels=max_channels,
            max_time_frames=max_time_frames,
            time_conv_kernel=time_conv_kernel,
        )

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _encode(self, x: Tensor) -> tuple[Tensor, int, int, int]:
        """Encode spectrogram to latent z.

        Returns (z, C, n_frames, T_orig) where z is (B, C, N, d_model).
        Uses encoder components directly (same pattern as Channel-AST-FSQ)
        to preserve the 4D shape before flattening.
        """
        B, C, Fr, T_orig = x.shape
        fw = self.frame_width

        # Pad T to multiple of frame_width
        pad_t = (fw - T_orig % fw) % fw
        if pad_t > 0:
            x = F.pad(x, (0, pad_t))
        T_padded = T_orig + pad_t
        n_frames = T_padded // fw

        # Per-channel frame embed
        frames = (
            x.reshape(B, C, Fr, n_frames, fw)
            .permute(0, 1, 3, 2, 4)       # (B, C, N, F, fw)
            .reshape(B, C, n_frames, Fr * fw)
        )
        tokens = self.encoder.frame_proj(frames)  # (B, C, N, d_model)

        # Add positional embeddings
        tokens = (
            tokens
            + self.encoder.channel_pos_embed[:, :C]
            + self.encoder.time_pos_embed[:, :, :n_frames]
        )

        # Encoder blocks
        for block in self.encoder.blocks:
            tokens = block(tokens)
        z = self.encoder.norm(tokens)  # (B, C, N, d_model)

        # DiTo-style latent regularisation
        z = self.latent_norm(z)

        return z, C, n_frames, T_orig

    # ------------------------------------------------------------------
    # Diffusion training step
    # ------------------------------------------------------------------

    def _sample_timesteps(self, batch_size: int, device: torch.device) -> Tensor:
        """Sample timesteps from logit-normal distribution."""
        logit_t = (
            self.logit_normal_mu
            + self.logit_normal_sigma * torch.randn(batch_size, device=device)
        )
        t = torch.sigmoid(logit_t)  # (B,) in (0, 1)
        return t

    def _diffusion_loss(
        self, x: Tensor, z: Tensor, C: int, n_frames: int
    ) -> tuple[Tensor, Tensor]:
        """Compute flow matching loss with x-prediction.

        Returns (loss, x_hat) where x_hat is the one-step prediction.
        """
        B = x.shape[0]
        device = x.device

        # Sample timesteps and noise
        t = self._sample_timesteps(B, device)
        eps = torch.randn_like(x)

        # Rectified flow interpolation: x_t = (1-t)*x + t*ε
        t_expand = t[:, None, None, None]  # (B, 1, 1, 1)
        x_t = (1.0 - t_expand) * x + t_expand * eps

        # Predict clean x
        x_hat = self.decoder(x_t, z, t, C, n_frames)

        # x-prediction loss: MSE(x̂, x)
        loss = F.mse_loss(x_hat, x)

        return loss, x_hat

    # ------------------------------------------------------------------
    # Multi-step Euler ODE sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _sample(
        self,
        z: Tensor,
        shape: tuple[int, ...],
        n_steps: int,
        C: int,
        n_frames: int,
    ) -> Tensor:
        """Generate reconstruction via Euler ODE solver.

        Starts from pure noise x_T and iteratively denoises to x_0
        using the rectified flow ODE with x-prediction.

        For x-prediction with rectified flow x_t = (1-t)x + tε:
            velocity v(x_t, t) = ε - x = (x_t - x_hat) / t
            Euler step: x_{t-dt} = x_t - dt * v
        """
        device = z.device
        x_curr = torch.randn(shape, device=device)

        # Time steps from 1.0 → 0.0
        timesteps = torch.linspace(1.0, 0.0, n_steps + 1, device=device)

        for i in range(n_steps):
            t_val = timesteps[i]
            dt = timesteps[i] - timesteps[i + 1]  # positive
            t_batch = t_val.expand(shape[0])  # (B,)

            # Predict clean x
            x_hat = self.decoder(x_curr, z, t_batch, C, n_frames)

            # Compute velocity: v = (x_curr - x_hat) / t
            v = (x_curr - x_hat) / t_val.clamp(min=1e-5)

            # Euler step: x_{t-dt} = x_t - dt * v
            x_curr = x_curr - dt * v

        # Final prediction at t ≈ 0
        return x_curr

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor] | Tensor:
        """Forward pass.

        Training : returns (x_hat_onestep, diffusion_loss)
        Eval     : returns reconstructed (B, C, F, T) via multi-step ODE
        """
        B, C, Fr, T_orig = x.shape
        fw = self.frame_width

        # Pad T to multiple of frame_width
        pad_t = (fw - T_orig % fw) % fw
        if pad_t > 0:
            x_padded = F.pad(x, (0, pad_t))
        else:
            x_padded = x
        T_padded = T_orig + pad_t
        n_frames = T_padded // fw

        # Encode
        z, _, _, _ = self._encode(x)

        if self.training:
            # Diffusion training: compute loss on padded input
            loss, x_hat = self._diffusion_loss(x_padded, z, C, n_frames)
            # Crop one-step estimate to original T
            x_hat = x_hat[:, :, :, :T_orig]
            return x_hat, loss
        else:
            # Multi-step Euler sampling
            x_hat = self._sample(
                z,
                shape=(B, C, Fr, T_padded),
                n_steps=self.eval_steps,
                C=C,
                n_frames=n_frames,
            )
            return x_hat[:, :, :, :T_orig]
