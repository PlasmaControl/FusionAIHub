"""FSQ codec for FAST time-series (filterscopes) — 1-D analog of SpectroFSQCodec /
VideoFSQCodec. Mirrors the POC ``FastTSFSQAutoencoder`` submodule names
(``enc``/``fsq``/``dec``) so a frozen POC/final checkpoint loads directly.

``n_tok = C * (window_samples // patch)`` in channel-major order, which matches the
backbone ``FastTimeSeriesTokenizer`` output — so the ``FastTimeSeriesCodeHead`` code
logits and the codec codes are the same length. Frozen at inference; renders sharp
ELM spikes from any plausible codes (the VQ property validated in the POC)."""
from typing import Tuple

import torch
import torch.nn as nn

from ..tokenizers.fast_time_series import FastTimeSeriesTokenizer
from ..output_heads import FastTimeSeriesHead
from .fsq import FSQBottleneck


class FastTSFSQCodec(nn.Module):
    """FastTimeSeriesTokenizer (enc) -> FSQ bottleneck -> FastTimeSeriesHead (dec)."""

    def __init__(self, C: int, window_samples: int, fsq_dim: int, fsq_L: int,
                 patch: int = 50, d_model: int = 256):
        super().__init__()
        self.C, self.window_samples, self.patch, self.d_model = C, window_samples, patch, d_model
        self.enc = FastTimeSeriesTokenizer(
            n_channels=C, window_samples=window_samples, d_model=d_model, patch_size=patch)
        self.n_tok = C * (window_samples // patch)
        self.fsq = FSQBottleneck(d_model, [fsq_L] * fsq_dim)
        self.dec = FastTimeSeriesHead(
            d_model=d_model, n_channels=C, window_samples=window_samples, patch_size=patch)
        self.dim, self.levels = fsq_dim, fsq_L

    def encode_codes(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, WIN) -> per-dim int codes (B, n_tok, dim). Frozen: CE targets.

        ``x`` must be in the codec's training space = per-(window, channel) z-scored
        filterscopes (the POC ``load_fastts_windows`` normalization)."""
        _, codes = self.fsq(self.enc(x))
        return codes

    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """per-dim int codes (B, n_tok, dim) -> (B, C, WIN). Frozen decoder."""
        return self.dec(self.fsq.codes_to_tokens(codes))


def load_frozen_fastts_codec(path: str, map_location="cpu") -> Tuple[FastTSFSQCodec, dict]:
    """Load a fast-TS codec ``.pt`` ({"ae": state_dict, "cfg": {...}}) -> a FROZEN
    ``FastTSFSQCodec`` + its cfg. cfg keys: C, WIN, patch, fsq_dim, fsq_L, d_model."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    cfg = dict(ckpt["cfg"])
    codec = FastTSFSQCodec(
        C=cfg["C"], window_samples=cfg["WIN"], fsq_dim=cfg["fsq_dim"], fsq_L=cfg["fsq_L"],
        patch=cfg.get("patch", 50), d_model=cfg.get("d_model", 256))
    codec.load_state_dict(ckpt["ae"])
    codec.eval()
    for p in codec.parameters():
        p.requires_grad_(False)
    return codec, cfg
