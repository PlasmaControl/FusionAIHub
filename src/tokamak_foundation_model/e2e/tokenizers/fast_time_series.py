"""Fast time-series tokenizer (10 kHz diagnostics, e.g. filterscopes).

Each channel is patched independently with a shared Conv1d of kernel and
stride equal to ``patch_size`` (50 by default), yielding
``n_channels * (window_samples // patch_size)`` tokens per 50 ms window.
See ``ResearchPlan.MD`` §3.3 and §5.2.
"""

import torch
import torch.nn as nn


class FastTimeSeriesTokenizer(nn.Module):
    """Conv1d-patched tokenizer for fast per-channel time series.

    Parameters
    ----------
    n_channels
        Number of diagnostic channels (``8`` for filterscopes).
    window_samples
        Samples per channel in one 50 ms window (``500`` at 10 kHz).
    d_model
        Token embedding dimension.
    patch_size
        Kernel and stride of the Conv1d patching (``50`` by default, producing
        10 tokens per channel at 10 kHz). Must divide ``window_samples``.

    Notes
    -----
    The Conv1d is shared across channels: channels are reshaped into the batch
    axis so each channel receives the same patching filter. Per-channel and
    per-patch structure is carried by learned embeddings of shape
    ``(n_channels, d_model)`` and ``(n_patches, d_model)`` respectively, plus
    a learned modality embedding of shape ``(d_model,)``. All embeddings are
    initialised with ``std=0.02`` so the signal projection dominates at init.

    Token ordering is channel-major:
    ``(c=0, p=0), (c=0, p=1), ..., (c=0, p=P-1), (c=1, p=0), ...``.
    """

    def __init__(
        self,
        n_channels: int,
        window_samples: int,
        d_model: int,
        patch_size: int = 50,
    ) -> None:
        super().__init__()
        if window_samples % patch_size != 0:
            raise ValueError(
                f"window_samples ({window_samples}) must be a multiple of "
                f"patch_size ({patch_size})"
            )
        self.n_channels = n_channels
        self.window_samples = window_samples
        self.d_model = d_model
        self.patch_size = patch_size
        self.n_patches = window_samples // patch_size

        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.channel_pos = nn.Parameter(torch.empty(n_channels, d_model))
        self.patch_pos = nn.Parameter(torch.empty(self.n_patches, d_model))
        self.modality_embed = nn.Parameter(torch.empty(d_model))
        nn.init.normal_(self.channel_pos, std=0.02)
        nn.init.normal_(self.patch_pos, std=0.02)
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
            Tokens of shape ``(batch, n_channels * n_patches, d_model)`` in
            channel-major order.
        """
        batch = x.shape[0]
        x_flat = x.reshape(batch * self.n_channels, 1, self.window_samples)
        patches = self.conv(x_flat)  # (B*C, d_model, n_patches)
        patches = patches.transpose(1, 2)  # (B*C, n_patches, d_model)
        patches = patches.reshape(
            batch, self.n_channels, self.n_patches, self.d_model
        )
        patches = patches + self.patch_pos
        patches = patches + self.channel_pos.unsqueeze(1)
        patches = patches + self.modality_embed
        return patches.reshape(
            batch, self.n_channels * self.n_patches, self.d_model
        )
