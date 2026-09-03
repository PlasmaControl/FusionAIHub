"""Adversarial FSQ codec for SLOW time-series (Thomson / CER / MSE profiles) —
per-modality analog of the fast-TS codec. One token per channel over a 5-sample
window: SlowTimeSeriesTokenizer -> FSQ -> SlowTimeSeriesHead, n_tok = C. Mirrors the
POC ``SlowTSFSQAutoencoder`` submodule names (enc/fsq/dec) so the frozen codec loads.

Trained in the DATASET-standardized space directly (no extra per-window z-score — the
5-sample window is too short to z-score), so ``encode_codes`` takes the dataset target
as-is (the CE branch does NOT re-normalize, unlike fast-TS/video)."""
from typing import Tuple

import torch
import torch.nn as nn

from ..tokenizers.slow_time_series import SlowTimeSeriesTokenizer
from ..output_heads import SlowTimeSeriesHead
from .fsq import FSQBottleneck


class SlowTSFSQCodec(nn.Module):
    """SlowTimeSeriesTokenizer (enc) -> FSQ -> SlowTimeSeriesHead (dec). n_tok = C."""

    def __init__(self, C: int, window_samples: int, fsq_dim: int, fsq_L: int, d_model: int = 256):
        super().__init__()
        self.C, self.window_samples, self.d_model = C, window_samples, d_model
        self.enc = SlowTimeSeriesTokenizer(n_channels=C, window_samples=window_samples, d_model=d_model)
        self.n_tok = C
        self.fsq = FSQBottleneck(d_model, [fsq_L] * fsq_dim)
        self.dec = SlowTimeSeriesHead(d_model=d_model, n_channels=C, window_samples=window_samples)
        self.dim, self.levels = fsq_dim, fsq_L

    def encode_codes(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, WIN) dataset-standardized -> per-dim int codes (B, n_tok, dim)."""
        _, codes = self.fsq(self.enc(x))
        return codes

    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """per-dim int codes (B, n_tok, dim) -> (B, C, WIN). Frozen decoder."""
        return self.dec(self.fsq.codes_to_tokens(codes))


def load_frozen_slowts_codec(path: str, map_location="cpu") -> Tuple[SlowTSFSQCodec, dict]:
    """Load a slow-TS codec ``.pt`` ({"ae": state_dict, "cfg": {...}}) -> FROZEN codec.
    cfg keys: C, WIN, fsq_dim, fsq_L, d_model."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    cfg = dict(ckpt["cfg"])
    codec = SlowTSFSQCodec(
        C=cfg["C"], window_samples=cfg["WIN"], fsq_dim=cfg["fsq_dim"], fsq_L=cfg["fsq_L"],
        d_model=cfg.get("d_model", 256))
    codec.load_state_dict(ckpt["ae"])
    codec.eval()
    for p in codec.parameters():
        p.requires_grad_(False)
    return codec, cfg
