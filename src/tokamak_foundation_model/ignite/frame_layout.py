"""Frame tokenizer: frozen per-modality codes to token embeddings and back to logits.

Each token of a frame is the sum of four learned embeddings: its code (in the
modality's own vocabulary), its modality, its position within the modality and its
frame index. Codes are passed as a dict ``{name: LongTensor (B, F, n_tok)}`` in the
canonical ``FROZEN_MODALITIES`` order, so each modality keeps its own vocabulary.
See ``docs/IGNITE_DESIGN.md`` section 5.1.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .dynamics_config import DynamicsConfig


def fsq_radix(levels) -> torch.Tensor:
    """Return the mixed-radix multipliers of FSQ levels.

    Parameters
    ----------
    levels : sequence of int
        FSQ levels, e.g. ``(8, 5, 5, 5)``.

    Returns
    -------
    radix : torch.LongTensor of shape (len(levels),)
        Multipliers, e.g. ``[1, 8, 40, 200]``.
    """
    r, acc = [], 1
    for n in levels:
        r.append(acc)
        acc *= int(n)
    return torch.tensor(r, dtype=torch.long)


def code_to_coords(idx: torch.Tensor, levels) -> torch.Tensor:
    """Unpack flat FSQ codes into grid coordinates normalised to ``[0, 1]``.

    Parameters
    ----------
    idx : torch.LongTensor of shape (...)
        Codes in ``[0, prod(levels))``.
    levels : sequence of int
        FSQ levels.

    Returns
    -------
    coords : torch.FloatTensor of shape (..., len(levels))
        Digit ``i`` divided by ``levels[i] - 1``. Exact inverse of
        :func:`coords_to_code` on the grid.
    """
    radix = fsq_radix(levels).to(idx.device)
    n = torch.tensor(list(levels), dtype=torch.long, device=idx.device)
    digits = (idx.unsqueeze(-1) // radix) % n
    return digits.float() / (n.float() - 1.0).clamp_min(1.0)


def coords_to_code(coords: torch.Tensor, levels) -> torch.Tensor:
    """Round normalised grid coordinates to the nearest flat FSQ code.

    Parameters
    ----------
    coords : torch.Tensor of shape (..., len(levels))
        Coordinates in ``[0, 1]``; values outside are clamped onto the grid.
    levels : sequence of int
        FSQ levels.

    Returns
    -------
    idx : torch.LongTensor of shape (...)
        Flat codes.
    """
    radix = fsq_radix(levels).to(coords.device)
    n = torch.tensor(list(levels), dtype=torch.long, device=coords.device)
    digits = torch.round(coords.float() * (n.float() - 1.0)).long()
    digits = torch.minimum(torch.clamp_min(digits, 0), n - 1)
    return (digits * radix).sum(-1)


class FrameTokenizer(nn.Module):
    """Embed per-modality codes and project hidden states back to per-modality heads.

    Parameters
    ----------
    cfg : DynamicsConfig
        Model configuration. ``cfg.latent_head`` selects a Gaussian head over the
        FSQ coordinate instead of a classifier; ``cfg.lag_embed_k`` adds
        own-column history embeddings.

    Notes
    -----
    The input vocabulary has one extra row, ``codebook_size``, used as the
    MaskGIT ``[MASK]`` id. The heads keep the attribute name ``heads`` for both
    head types, so loaders tell them apart by the weight shape.
    """

    def __init__(self, cfg: DynamicsConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.code_embed = nn.ModuleDict(
            {m.name: nn.Embedding(m.codebook_size + 1, d) for m in cfg.modalities}
        )
        self.latent_head = bool(cfg.latent_head)
        self.fsq_levels = tuple(int(x) for x in cfg.fsq_levels)
        self.heads = nn.ModuleDict(
            {
                m.name: nn.Linear(
                    d, 2 * len(self.fsq_levels) if self.latent_head else m.codebook_size
                )
                for m in cfg.modalities
            }
        )
        self.mask_ids = {m.name: m.codebook_size for m in cfg.modalities}
        self.pos_embed = nn.ModuleDict(
            {m.name: nn.Embedding(m.n_tok, d) for m in cfg.modalities}
        )
        self.modality_embed = nn.Embedding(cfg.n_modalities, d)
        self.frame_embed = nn.Embedding(cfg.max_frames, d)
        self._mod_index = {m.name: i for i, m in enumerate(cfg.modalities)}
        # Lag-k embeddings let a column see its own last k codes directly, which a
        # per-column count table exploits and shared attention dilutes.
        self.lag_k = int(cfg.lag_embed_k or 0)
        if self.lag_k > 0:
            self.lag_embed = nn.ModuleDict(
                {
                    m.name: nn.ModuleList(
                        [
                            nn.Embedding(m.codebook_size + 1, d)
                            for _ in range(self.lag_k)
                        ]
                    )
                    for m in cfg.modalities
                }
            )

    def _check_tokens(self, h: torch.Tensor, where: str) -> None:
        if h.shape[2] != self.cfg.tokens_per_frame:
            raise ValueError(
                f"{where}: expected {self.cfg.tokens_per_frame} tokens, "
                f"got {h.shape[2]}"
            )

    def embed(
        self, codes: Dict[str, torch.Tensor], frame_offset: int = 0
    ) -> torch.Tensor:
        """Embed a window of codes.

        Parameters
        ----------
        codes : dict of str to torch.Tensor
            Codes of every modality, a LongTensor of shape (B, F, n_tok) whose
            masked positions hold the ``[MASK]`` id, or already embedded codes of
            shape (B, F, n_tok, d_model) (see :meth:`code_vectors`), which a
            rollout trained end to end feeds back.
        frame_offset : int, default=0
            Index of the first frame, for windows that do not start at frame 0.

        Returns
        -------
        x : torch.Tensor of shape (B, F, tokens_per_frame, d_model)
            Token embeddings in canonical modality order.
        """
        missing = {m.name for m in self.cfg.modalities} - set(codes)
        if missing:
            raise ValueError(
                f"FrameTokenizer.embed: codes missing modalities {sorted(missing)}"
            )
        reference = codes[self.cfg.modalities[0].name]
        B, F = reference.shape[:2]
        if frame_offset + F > self.cfg.max_frames:
            raise ValueError(
                f"frame_offset+F={frame_offset + F} exceeds "
                f"max_frames={self.cfg.max_frames}"
            )
        device = reference.device
        frame_index = torch.arange(frame_offset, frame_offset + F, device=device)
        frame_embedding = self.frame_embed(frame_index)
        embeddings = []
        for m in self.cfg.modalities:
            modality_codes = codes[m.name]
            if modality_codes.shape[:2] != (B, F) or modality_codes.shape[2] != m.n_tok:
                raise ValueError(
                    f"{m.name}: expected (B={B}, F={F}, {m.n_tok}); "
                    f"got {tuple(modality_codes.shape)}"
                )
            embedded = modality_codes.is_floating_point()
            if embedded and self.lag_k > 0:
                raise ValueError("embedded codes do not support lag_embed_k > 0")
            positions = torch.arange(m.n_tok, device=device)
            code_part = (
                modality_codes if embedded else self.code_embed[m.name](modality_codes)
            )
            embedding = (
                code_part
                + self.modality_embed.weight[self._mod_index[m.name]].view(1, 1, 1, -1)
                + self.pos_embed[m.name](positions).view(1, 1, m.n_tok, -1)
                + frame_embedding.view(1, F, 1, -1)
            )
            for lag_index in range(self.lag_k):
                # Token n at frame f sees its own column at frame f - lag_index - 1
                # of the masked input; frames without that history get [MASK].
                lag = modality_codes.new_full(
                    modality_codes.shape, self.mask_ids[m.name]
                )
                if F > lag_index + 1:
                    lag[:, lag_index + 1 :] = modality_codes[:, : F - (lag_index + 1)]
                embedding = embedding + self.lag_embed[m.name][lag_index](lag)
            embeddings.append(embedding)
        return torch.cat(embeddings, dim=2)

    def code_vectors(self, name: str, codes: torch.Tensor) -> torch.Tensor:
        """Embed the codes of one modality, shape (..., d_model)."""
        return self.code_embed[name](codes)

    def soft_code_vectors(self, name: str, weights: torch.Tensor) -> torch.Tensor:
        """Mix the code embeddings of one modality by ``weights`` (..., codebook).

        One-hot weights give exactly :meth:`code_vectors`; the ``[MASK]`` row is
        never mixed in.
        """
        table = self.code_embed[name].weight[: weights.shape[-1]]
        return weights.to(table.dtype) @ table

    def logits(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Apply the per-modality heads to every position.

        Parameters
        ----------
        h : torch.Tensor of shape (B, F, tokens_per_frame, d_model)
            Backbone output.

        Returns
        -------
        out : dict of str to torch.Tensor of shape (B, F, n_tok, n_out)
            Head outputs, ``n_out`` being the codebook size or ``2 * len(levels)``.
        """
        self._check_tokens(h, "logits")
        return {
            m.name: self.heads[m.name](h[:, :, s:e, :])
            for (s, e), m in zip(self.cfg.modality_token_slices(), self.cfg.modalities)
        }

    def masked_logits(
        self, h: torch.Tensor, mask: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Apply the per-modality heads to the masked positions only.

        Gathering before projecting is what keeps training memory bounded: full
        logits over a 100-frame window are hundreds of GB per sample.

        Parameters
        ----------
        h : torch.Tensor of shape (B, F, tokens_per_frame, d_model)
            Backbone output.
        mask : dict of str to torch.BoolTensor of shape (B, F, n_tok)
            Positions to project.

        Returns
        -------
        out : dict of str to torch.Tensor of shape (n_masked, n_out)
            Head outputs at the masked positions, in row-major order.
        """
        self._check_tokens(h, "masked_logits")
        return {
            m.name: self.heads[m.name](h[:, :, s:e, :][mask[m.name]])
            for (s, e), m in zip(self.cfg.modality_token_slices(), self.cfg.modalities)
        }
