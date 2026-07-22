"""TDD spec for ignite.nets.SpectroEncoder / SpectroDecoder.

Small synthetic CPU-only tensors. Contract (config.py):

    SpectroEncoder(cfg)(x: (B,C,F,T)) -> feats (B, n_tok, d_model)
    SpectroDecoder(cfg)(quant: (B,n_tok,d_model)) -> recon (B,C,F,T)

    n_tok = (freq_bins // patch_f) * (time_frames // patch_t)
"""
from __future__ import annotations

import torch

from tokamak_foundation_model.ignite.config import SpectroCodecConfig
from tokamak_foundation_model.ignite.nets import SpectroDecoder, SpectroEncoder


def _small_cfg() -> SpectroCodecConfig:
    # tiny transformer + small grid, but keep patch divisibility.
    return SpectroCodecConfig(
        channels=1,
        freq_bins=16,
        time_frames=8,
        patch_f=8,
        patch_t=4,
        d_model=32,
        enc_depth=2,
        dec_depth=2,
        heads=4,
    )


def test_encoder_shape() -> None:
    cfg = _small_cfg()
    enc = SpectroEncoder(cfg)
    B = 3
    x = torch.randn(B, cfg.channels, cfg.freq_bins, cfg.time_frames)
    feats = enc(x)
    assert feats.shape == (B, cfg.n_tok, cfg.d_model)


def test_decoder_shape() -> None:
    cfg = _small_cfg()
    dec = SpectroDecoder(cfg)
    B = 3
    quant = torch.randn(B, cfg.n_tok, cfg.d_model)
    recon = dec(quant)
    assert recon.shape == (B, cfg.channels, cfg.freq_bins, cfg.time_frames)


def test_multichannel_roundtrip_shape() -> None:
    cfg = _small_cfg()
    cfg.channels = 2
    enc = SpectroEncoder(cfg)
    dec = SpectroDecoder(cfg)
    B = 2
    x = torch.randn(B, cfg.channels, cfg.freq_bins, cfg.time_frames)
    recon = dec(enc(x))
    assert recon.shape == x.shape


def test_encoder_decoder_roundtrip() -> None:
    cfg = _small_cfg()
    enc = SpectroEncoder(cfg)
    dec = SpectroDecoder(cfg)
    B = 2
    x = torch.randn(B, cfg.channels, cfg.freq_bins, cfg.time_frames)
    feats = enc(x)
    recon = dec(feats)
    assert recon.shape == x.shape
    assert torch.isfinite(recon).all()


def test_gradient_flows_end_to_end() -> None:
    cfg = _small_cfg()
    enc = SpectroEncoder(cfg)
    dec = SpectroDecoder(cfg)
    x = torch.randn(2, cfg.channels, cfg.freq_bins, cfg.time_frames, requires_grad=True)
    recon = dec(enc(x))
    recon.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0


def test_n_tok_matches_config() -> None:
    cfg = _small_cfg()
    # sanity: the contract's n_tok derivation is what the nets produce.
    assert cfg.n_tok == (cfg.freq_bins // cfg.patch_f) * (cfg.time_frames // cfg.patch_t)
