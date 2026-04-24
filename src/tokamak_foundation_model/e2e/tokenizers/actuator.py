"""Actuator tokenizer (one actuator group per instance).

Conv1d channel-mixing patching produces a small number of tokens (typically
three per group) covering one 50 ms window. The backbone cross-attends to the
concatenated stack of actuator tokens at each rollout step
(``ResearchPlan.MD`` §3.1 principle 6, §3.6, §5.5).
"""

import torch
import torch.nn as nn


class ActuatorTokenizer(nn.Module):
    """Tokenize one actuator group (e.g. NBI, ECH, gas, RMP) for one window.

    Parameters
    ----------
    n_channels
        Number of channels in the actuator group.
    window_samples
        Samples per channel in one 50 ms window. Must be divisible by
        ``n_tokens``.
    d_model
        Token embedding dimension.
    n_tokens
        Number of tokens to emit per window (``3`` by default, per
        ``ResearchPlan.MD`` §3.3).

    Notes
    -----
    Channel mixing via ``Conv1d(in=n_channels, out=d_model, k=s=patch_size)``.
    Per-patch and per-group structure is carried by learned embeddings
    initialised with ``std=0.02``; no LayerNorm is applied after this
    concatenation. §5.5 explicitly forbids LayerNorm on concatenated actuator
    tokens because it dilutes the data-dependent signal relative to the
    learned embeddings.
    """

    def __init__(
        self,
        n_channels: int,
        window_samples: int,
        d_model: int,
        n_tokens: int = 3,
    ) -> None:
        super().__init__()
        if window_samples % n_tokens != 0:
            raise ValueError(
                f"window_samples ({window_samples}) must be a multiple of "
                f"n_tokens ({n_tokens})"
            )
        self.n_channels = n_channels
        self.window_samples = window_samples
        self.d_model = d_model
        self.n_tokens = n_tokens
        patch_size = window_samples // n_tokens

        self.conv = nn.Conv1d(
            in_channels=n_channels,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.patch_pos = nn.Parameter(torch.empty(n_tokens, d_model))
        self.modality_embed = nn.Parameter(torch.empty(d_model))
        nn.init.normal_(self.patch_pos, std=0.02)
        nn.init.normal_(self.modality_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Tokenize one batch of actuator commands.

        Parameters
        ----------
        x
            Actuator signal of shape ``(batch, n_channels, window_samples)``.

        Returns
        -------
        torch.Tensor
            Tokens of shape ``(batch, n_tokens, d_model)``.
        """
        tokens = self.conv(x).transpose(1, 2)  # (B, n_tokens, d_model)
        tokens = tokens + self.patch_pos
        tokens = tokens + self.modality_embed
        return tokens
