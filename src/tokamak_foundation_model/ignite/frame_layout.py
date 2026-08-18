"""Phase-B frame tokenizer: frozen per-modality codes <-> embedded token sequence + logits.

Maps the frozen Phase-A codes of a rollout window into the ST-transformer's token embeddings
and back to per-modality categorical logits. Each of the 1012 tokens/frame carries four learned
signals (all summed): the code embedding (per-modality vocab), a modality-type embedding, a
within-modality position embedding, and a frame-index (temporal) embedding. See
docs/IGNITE_DESIGN.md §5.1.

Codes are laid out per frame in the canonical FROZEN_MODALITIES order; ``codes`` is therefore a
dict {modality_name: LongTensor (B, F, n_tok_m)} rather than one ragged tensor, so each modality
keeps its own vocab. embed() returns a single (B, F, tokens_per_frame, d_model) tensor for the
backbone; logits() splits the backbone output back per modality.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .dynamics_config import DynamicsConfig


class FrameTokenizer(nn.Module):
    def __init__(self, cfg: DynamicsConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        # per-modality code-embedding tables + output heads (own vocab, own semantics).
        # Input vocab is codebook_size + 1: the extra row (id == codebook_size) is the MaskGIT
        # [MASK] token. Output heads stay codebook_size — the model predicts REAL codes, never MASK.
        self.code_embed = nn.ModuleDict(
            {m.name: nn.Embedding(m.codebook_size + 1, d) for m in cfg.modalities}
        )
        self.heads = nn.ModuleDict(
            {m.name: nn.Linear(d, m.codebook_size) for m in cfg.modalities}
        )
        self.mask_ids = {m.name: m.codebook_size for m in cfg.modalities}
        # per-modality within-frame position (freq/time-patch for spectro, zone for slow-TS, ...)
        self.pos_embed = nn.ModuleDict(
            {m.name: nn.Embedding(m.n_tok, d) for m in cfg.modalities}
        )
        # modality-type (shared across a modality's tokens) + temporal frame-index
        self.modality_embed = nn.Embedding(cfg.n_modalities, d)
        self.frame_embed = nn.Embedding(cfg.max_frames, d)
        self._mod_index = {m.name: i for i, m in enumerate(cfg.modalities)}

    def _check(self, codes: Dict[str, torch.Tensor]) -> None:
        names = {m.name for m in self.cfg.modalities}
        missing = names - set(codes)
        if missing:
            raise ValueError(f"FrameTokenizer.embed: codes missing modalities {sorted(missing)}")

    def embed(self, codes: Dict[str, torch.Tensor], frame_offset: int = 0) -> torch.Tensor:
        """codes[name]: (B, F, n_tok_m) Long -> (B, F, tokens_per_frame, d_model).

        ``frame_offset`` shifts the temporal frame-index embedding (for rollout windows that do
        not start at frame 0). Tokens are concatenated in canonical FROZEN_MODALITIES order.
        """
        self._check(codes)
        ref = codes[self.cfg.modalities[0].name]
        B, F, _ = ref.shape
        if frame_offset + F > self.cfg.max_frames:
            raise ValueError(
                f"frame_offset+F={frame_offset + F} exceeds max_frames={self.cfg.max_frames}"
            )
        dev = ref.device
        frame_idx = torch.arange(frame_offset, frame_offset + F, device=dev)
        frame_e = self.frame_embed(frame_idx)                      # (F, d)
        parts = []
        for m in self.cfg.modalities:
            c = codes[m.name]                                      # (B, F, n_tok_m)
            if c.shape[:2] != (B, F) or c.shape[2] != m.n_tok:
                raise ValueError(f"{m.name}: expected (B={B}, F={F}, {m.n_tok}); got {tuple(c.shape)}")
            pos = torch.arange(m.n_tok, device=dev)
            e = (
                self.code_embed[m.name](c)                          # (B, F, n_tok_m, d)
                + self.modality_embed.weight[self._mod_index[m.name]].view(1, 1, 1, -1)
                + self.pos_embed[m.name](pos).view(1, 1, m.n_tok, -1)
                + frame_e.view(1, F, 1, -1)
            )
            parts.append(e)
        return torch.cat(parts, dim=2)                              # (B, F, tokens_per_frame, d)

    def logits(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Backbone output (B, F, tokens_per_frame, d) -> {name: (B, F, n_tok_m, codebook_m)}."""
        if h.shape[2] != self.cfg.tokens_per_frame:
            raise ValueError(
                f"logits: expected {self.cfg.tokens_per_frame} tokens, got {h.shape[2]}"
            )
        out: Dict[str, torch.Tensor] = {}
        for (s, e), m in zip(self.cfg.modality_token_slices(), self.cfg.modalities):
            out[m.name] = self.heads[m.name](h[:, :, s:e, :])
        return out

    def masked_logits(self, h: torch.Tensor,
                      mask: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Project ONLY the masked positions to vocab -> {name: (n_masked_m, codebook_m)}.

        Memory-critical for training: the full (B, F, tokens_per_frame, vocab) logits are ~B*400 GB
        at F=100 / vocab=1000, but MaskGIT only supervises masked positions (~50%). Gathering the
        masked hidden states FIRST then projecting shrinks that to ~B*200 MB. ``mask[name]`` is the
        per-modality (B, F, n_tok_m) bool from :meth:`MaskGITDynamics._random_mask`.
        """
        if h.shape[2] != self.cfg.tokens_per_frame:
            raise ValueError(
                f"masked_logits: expected {self.cfg.tokens_per_frame} tokens, got {h.shape[2]}"
            )
        out: Dict[str, torch.Tensor] = {}
        for (s, e), m in zip(self.cfg.modality_token_slices(), self.cfg.modalities):
            mk = mask[m.name]                                   # (B, F, n_tok_m) bool
            h_masked = h[:, :, s:e, :][mk]                      # (n_masked_m, d) gathered
            out[m.name] = self.heads[m.name](h_masked)          # (n_masked_m, codebook_m)
        return out

    def logits_last(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Project ONLY the last frame to vocab -> {name: (B, n_tok_m, codebook_m)}.

        Rollout generates one frame at a time and reads only ``logits[:, -1]``, but
        :meth:`logits` projects every frame: ~15 GB of transient fp32 per decode step at
        the production layout (1593 tokens, four 64k vocabs), x10 steps x80 frames.
        Slicing the hidden states first drops that to ~150 MB. Numerically identical.
        """
        if h.shape[2] != self.cfg.tokens_per_frame:
            raise ValueError(
                f"logits_last: expected {self.cfg.tokens_per_frame} tokens, got {h.shape[2]}"
            )
        hl = h[:, -1]                                            # (B, tokens_per_frame, d)
        out: Dict[str, torch.Tensor] = {}
        for (s, e), m in zip(self.cfg.modality_token_slices(), self.cfg.modalities):
            out[m.name] = self.heads[m.name](hl[:, s:e, :])
        return out
