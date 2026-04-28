"""Per-modality output heads.

Each head is an approximate inverse of its sibling tokenizer. They fire only
to compute the training loss against ground-truth raw signals — during
autoregressive rollout the backbone's token output is fed directly to the
next step, bypassing the heads (``ResearchPlan.MD`` §3.5, §3.6, §5.7).
"""

import torch
import torch.nn as nn


class SlowTimeSeriesHead(nn.Module):
    """Linear head reconstructing a slow time-series modality.

    Parameters
    ----------
    d_model
        Token embedding dimension.
    n_channels
        Number of diagnostic channels.
    window_samples
        Samples per channel in one 50 ms window (``5`` at 100 Hz).

    Notes
    -----
    Approximate inverse of :class:`SlowTimeSeriesTokenizer`: a single shared
    ``Linear(d_model, window_samples)`` unprojects each per-channel token back
    to raw signal samples.
    """

    def __init__(
        self, d_model: int, n_channels: int, window_samples: int
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_channels = n_channels
        self.window_samples = window_samples
        self.proj = nn.Linear(d_model, window_samples)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Reconstruct raw signal.

        Parameters
        ----------
        tokens
            ``(batch, n_channels, d_model)`` — per-channel tokens from the
            backbone for this modality.

        Returns
        -------
        torch.Tensor
            ``(batch, n_channels, window_samples)`` raw-signal reconstruction.
        """
        return self.proj(tokens)


class FastTimeSeriesHead(nn.Module):
    """ConvTranspose1d head reconstructing a fast time-series modality.

    Parameters
    ----------
    d_model
        Token embedding dimension.
    n_channels
        Number of diagnostic channels.
    window_samples
        Samples per channel in one 50 ms window (``500`` at 10 kHz).
    patch_size
        Patch length matching the sibling tokenizer (``50`` by default). Must
        divide ``window_samples``.

    Notes
    -----
    Approximate inverse of :class:`FastTimeSeriesTokenizer`. Channels are
    reshaped into the batch axis so a single shared
    ``ConvTranspose1d(in=d_model, out=1, k=s=patch_size)`` unpacks each
    per-channel patch sequence back to raw samples.
    """

    def __init__(
        self,
        d_model: int,
        n_channels: int,
        window_samples: int,
        patch_size: int = 50,
    ) -> None:
        super().__init__()
        if window_samples % patch_size != 0:
            raise ValueError(
                f"window_samples ({window_samples}) must be a multiple of "
                f"patch_size ({patch_size})"
            )
        self.d_model = d_model
        self.n_channels = n_channels
        self.window_samples = window_samples
        self.patch_size = patch_size
        self.n_patches = window_samples // patch_size

        self.deconv = nn.ConvTranspose1d(
            in_channels=d_model,
            out_channels=1,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Reconstruct raw signal.

        Parameters
        ----------
        tokens
            ``(batch, n_channels * n_patches, d_model)`` in channel-major
            order (matching :class:`FastTimeSeriesTokenizer`).

        Returns
        -------
        torch.Tensor
            ``(batch, n_channels, window_samples)`` raw-signal reconstruction.
        """
        batch = tokens.shape[0]
        t = tokens.reshape(batch, self.n_channels, self.n_patches, self.d_model)
        t = t.reshape(batch * self.n_channels, self.n_patches, self.d_model)
        t = t.transpose(1, 2)  # (B*C, d_model, n_patches)
        out = self.deconv(t)  # (B*C, 1, window_samples)
        return out.reshape(batch, self.n_channels, self.window_samples)