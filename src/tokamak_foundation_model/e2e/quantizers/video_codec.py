"""Frozen adversarial-FSQ VIDEO codec — the e2e-facing promotion of the POC's
``VideoFSQAutoencoder`` (``scripts/training/poc_fsq_video.py``). Video analog of
``spectro_codec.SpectroFSQCodec``.

    encoder (VideoTokenizer tube-patch, 3x12x12 -> 300 tokens)
      -> FSQBottleneck (discrete codes; no learned codebook -> no collapse)
      -> decoder (VideoOutputHead, resize-conv -> no checkerboard)
trained adversarially (3D PatchGAN + hinge + FM + R1) then FROZEN. Phase-1b video
path: the backbone predicts the codes (VideoCodeHead, class-weighted CE) and they
decode through this frozen decoder to render sharp frames.

Submodule names ``enc``/``fsq``/``dec`` match the POC ``VideoFSQAutoencoder`` so a
codec ``.pt`` saved by poc_fsq_video.py loads here unchanged. n_frames/spatial are
ctor args (POC used module globals) so the loader is self-describing from cfg.
"""
from typing import Tuple

import torch
import torch.nn as nn

from ..output_heads import VideoOutputHead
from ..tokenizers.video import VideoTokenizer
from .fsq import FSQBottleneck


class VideoFSQCodec(nn.Module):
    """VideoTokenizer -> FSQ bottleneck -> VideoOutputHead(resize-conv). Frozen in
    Phase 1b. ``encode_codes``/``decode_codes`` are the e2e interface; the decode
    returns the VideoOutputHead-native ``(B, T, C, H, W)`` (the trainer permutes it
    like the continuous video head)."""

    def __init__(self, C: int, fsq_dim: int, fsq_L: int, n_frames: int = 3,
                 height: int = 120, width: int = 360,
                 patch: Tuple[int, int, int] = (3, 12, 12),
                 d_model: int = 256, decoder: str = "resize_conv"):
        super().__init__()
        self.C = C
        self.enc = VideoTokenizer(n_channels=C, n_frames=n_frames, patch_size=patch,
                                  d_model=d_model, spatial_size=(height, width))
        self.n_tok = self.enc.n_tokens
        self.fsq = FSQBottleneck(d_model, [fsq_L] * fsq_dim)
        self.dec = VideoOutputHead(n_channels=C, n_frames=n_frames, patch_size=patch,
                                   d_model=d_model, spatial_size=(height, width),
                                   decoder=decoder)
        self.dim, self.levels = fsq_dim, fsq_L

    @torch.no_grad()
    def encode_codes(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T, H, W) -> per-dim int codes (B, n_tok, dim). Frozen; CE targets."""
        return self.fsq(self.enc._encode(x))[1]

    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """per-dim int codes (B, n_tok, dim) -> video (B, T, C, H, W). Frozen decoder."""
        return self.dec(self.fsq.codes_to_tokens(codes))


def load_frozen_video_codec(path: str, map_location="cpu") -> Tuple[VideoFSQCodec, dict]:
    """Load a poc_fsq_video ``.pt`` -> FROZEN ``VideoFSQCodec`` + cfg. cfg holds
    {C, fsq_dim, fsq_L, patch, d_model, decoder}. Returns eval-mode, requires_grad
    False on all params."""
    ck = torch.load(path, map_location=map_location, weights_only=False)
    cfg = dict(ck["cfg"])
    patch = tuple(cfg.get("patch", (3, 12, 12)))
    codec = VideoFSQCodec(
        C=cfg["C"], fsq_dim=cfg["fsq_dim"], fsq_L=cfg["fsq_L"],
        patch=patch, d_model=cfg.get("d_model", 256),
        decoder=cfg.get("decoder", "resize_conv"))
    codec.load_state_dict(ck["ae"])
    codec.eval()
    for p in codec.parameters():
        p.requires_grad_(False)
    return codec, cfg
