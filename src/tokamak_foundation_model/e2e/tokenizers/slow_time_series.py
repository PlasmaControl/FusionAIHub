"""Slow time-series tokenizer (100 Hz diagnostics).

One token per channel for Thomson (core/tangential density, temperature), CER
(Ti, rotation), and MSE. See ``ResearchPlan.MD`` §3.3 and §5.1.
"""

import torch
import torch.nn as nn


class SlowTimeSeriesTokenizer(nn.Module):
    """Tokenize a 50 ms window of a slow time series, one token per channel.

    Parameters
    ----------
    n_channels
        Number of channels in the modality.
    window_samples
        Samples per channel in one 50 ms window (``5`` at 100 Hz).
    d_model
        Token embedding dimension.

    Notes
    -----
    A single ``Linear(window_samples, d_model)`` is shared across channels.
    Per-channel structure is carried by a learned positional embedding of
    shape ``(n_channels, d_model)``; a learned modality embedding of shape
    ``(d_model,)`` identifies which modality each token belongs to once
    concatenated in the backbone. Both embeddings are initialised with
    ``std=0.02`` so the raw-signal projection dominates the output at init
    (required for §5.1 impulse tests).
    """

    def __init__(self, n_channels: int, window_samples: int, d_model: int) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.window_samples = window_samples
        self.d_model = d_model
        self.proj = nn.Linear(window_samples, d_model)
        self.channel_pos = nn.Parameter(torch.empty(n_channels, d_model))
        self.modality_embed = nn.Parameter(torch.empty(d_model))
        nn.init.normal_(self.channel_pos, std=0.02)
        nn.init.normal_(self.modality_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Tokenize a batch.

        Parameters
        ----------
        x
            Raw signal of shape ``(batch, n_channels, window_samples)``.

        Returns
        -------
        torch.Tensor
            Tokens of shape ``(batch, n_channels, d_model)``.
        """
        tokens = self.proj(x)
        tokens = tokens + self.channel_pos
        tokens = tokens + self.modality_embed
        return tokens