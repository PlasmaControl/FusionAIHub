"""Frozen adversarial-FSQ spectrogram codec — the e2e-facing promotion of the
POC's ``FSQAutoencoder`` (``scripts/training/poc_fsq_stageB.py``) that Phase 1a
pre-trains and freezes.

The codec is a discrete autoencoder:
    encoder (SpectrogramTokenizer patch, 64x32 -> 24 tokens)
      -> FSQBottleneck (dim 24, L 8; no learned codebook -> no collapse)
      -> decoder (SpectrogramOutputHead)
trained with the validated VQ-GAN recipe (see ``SpectroDiscriminator`` + the
adversarial loop in ``train_fsq_codec.py``). The adversarial decoder is what
renders SHARP modes from codes instead of the mean-collapsed MAE envelope.

Phase 1b (the main run) loads this **frozen** and uses it two ways:
  * ``encode_codes(target)``  -> per-dim integer codes = class-weighted-CE targets
  * ``decode_codes(codes)``   -> spectrogram = viz / rollout output
while the backbone learns to PREDICT the codes (see ``SpectrogramCodeHead`` in
``output_heads.py``). ``SpectroFSQCodec`` is a structural mirror of the POC
``FSQAutoencoder`` (identical submodule names ``enc``/``fsq``/``dec``) so a
codec ``.pt`` saved by the POC or ``train_fsq_codec.py`` loads here unchanged.
Patch/d_model are ctor args here (POC used module globals) so the loader is
self-describing from the saved ``cfg``.
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..output_heads import SpectrogramOutputHead
from ..tokenizers.spectrogram import SpectrogramTokenizer
from .fsq import FSQBottleneck


class SpectroFSQCodec(nn.Module):
    """encoder (patch tokenizer) -> FSQ bottleneck -> decoder. Mirror of the POC
    ``FSQAutoencoder`` with patch_f/patch_t/d_model as ctor args.

    per_channel=False (production): all C channels folded into ONE 24-token
    budget (the extreme bottleneck that matches the production spectro token
    reservation). per_channel=True: a shared single-channel codec gives each
    channel its own 24 tokens (capacity test; not used in production).
    """

    def __init__(self, C: int, F_: int, T_: int, fsq_dim: int, fsq_L: int,
                 patch_f: int = 64, patch_t: int = 32, d_model: int = 256,
                 per_channel: bool = False):
        super().__init__()
        self.C, self.per_channel = C, per_channel
        self.patch_f, self.patch_t, self.d_model = patch_f, patch_t, d_model
        enc_ch = 1 if per_channel else C
        self.enc = SpectrogramTokenizer(
            n_channels=enc_ch, d_model=d_model, patch_f=patch_f, patch_t=patch_t,
            freq_bins=F_, time_frames=T_, enable_freq_stem=True)
        npf, npt = F_ // patch_f, T_ // patch_t
        self.n_tok_per = npf * npt                       # 24 tokens per channel-group
        self.n_tok = self.n_tok_per * (C if per_channel else 1)
        self.fsq = FSQBottleneck(d_model, [fsq_L] * fsq_dim)
        self.dec = SpectrogramOutputHead(
            n_channels=enc_ch, d_model=d_model, patch_f=patch_f, patch_t=patch_t,
            n_patches_f=npf, n_patches_t=npt)
        self.dim, self.levels = fsq_dim, fsq_L

    def _fold(self, x: torch.Tensor) -> torch.Tensor:     # (B,C,F,T) -> (B*C,1,F,T)
        return x.reshape(x.shape[0] * self.C, 1, *x.shape[2:]) if self.per_channel else x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        tq, codes = self.fsq(self.enc._encode(self._fold(x)))
        rec = self.dec(tq)
        if self.per_channel:
            rec = rec.reshape(B, self.C, *rec.shape[2:])
            codes = codes.reshape(B, self.n_tok, -1)
        return rec, codes

    @torch.no_grad()
    def encode_codes(self, x: torch.Tensor) -> torch.Tensor:
        """(B,C,F,T) -> per-dim int codes (B, n_tok, dim). Frozen: CE targets."""
        B = x.shape[0]
        _, codes = self.fsq(self.enc._encode(self._fold(x)))
        return codes.reshape(B, self.n_tok, -1) if self.per_channel else codes

    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """per-dim int codes (B, n_tok, dim) -> spectrogram (B, C, F, T). Frozen
        decoder; renders sharp modes from any plausible codes."""
        if self.per_channel:
            B = codes.shape[0]
            codes = codes.reshape(B * self.C, self.n_tok_per, -1)
            rec = self.dec(self.fsq.codes_to_tokens(codes))
            return rec.reshape(B, self.C, *rec.shape[2:])
        return self.dec(self.fsq.codes_to_tokens(codes))


class SpectroDiscriminator(nn.Module):
    """PatchGAN discriminator on spectrograms (real vs FSQ-reconstructed) — the
    VQ-GAN / audio-codec ingredient that forces the decoder to render SHARP modes
    instead of the blurry MAE mean (which no amount of code prediction can fix).
    Returns (patch_logits, [features]) for hinge + feature-matching losses.
    Used only in Phase 1a (codec pre-training); not part of the frozen codec.
    """

    def __init__(self, C: int, base: int = 64):
        super().__init__()

        def blk(i, o, s):
            return nn.Sequential(nn.Conv2d(i, o, 4, s, 1),
                                 nn.GroupNorm(min(8, o), o),
                                 nn.LeakyReLU(0.2, inplace=True))
        self.b1 = blk(C, base, 2)
        self.b2 = blk(base, base * 2, 2)
        self.b3 = blk(base * 2, base * 4, 2)
        self.out = nn.Conv2d(base * 4, 1, 3, 1, 1)

    def forward(self, x: torch.Tensor):
        f1 = self.b1(x); f2 = self.b2(f1); f3 = self.b3(f2)
        return self.out(f3), [f1, f2, f3]


def load_frozen_codec(path: str, map_location="cpu") -> Tuple[SpectroFSQCodec, dict]:
    """Load a Phase-1a codec ``.pt`` -> a FROZEN ``SpectroFSQCodec`` + its cfg.

    The checkpoint holds ``{"ae": state_dict, "cfg": {...}}`` written by
    ``train_fsq_codec.py`` / the POC. ``d_model`` defaults to 256 when absent
    (older POC ckpts predate the field). The returned codec is in eval mode with
    all params requires_grad_(False)."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    cfg = dict(ckpt["cfg"])
    codec = SpectroFSQCodec(
        C=cfg["C"], F_=cfg["Fq"], T_=cfg["Tq"], fsq_dim=cfg["fsq_dim"],
        fsq_L=cfg["fsq_L"], patch_f=cfg.get("patch_f", 64),
        patch_t=cfg.get("patch_t", 32), d_model=cfg.get("d_model", 256),
        per_channel=cfg.get("per_channel", False))
    codec.load_state_dict(ckpt["ae"])
    codec.eval()
    for p in codec.parameters():
        p.requires_grad_(False)
    return codec, cfg
