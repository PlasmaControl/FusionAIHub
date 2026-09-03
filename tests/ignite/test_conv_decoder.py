"""TDD spec for nets.SpectroConvDecoder — the HiFi-GAN/VQGAN-style 2D conv upsampling
decoder, a gated drop-in replacement for the linear SpectroDecoder.

Contract (config.py):

    SpectroConvDecoder(cfg)(quant: (B, n_tok, d_model)) -> recon (B, C, freq_bins, time_frames)
    .last_layer -> the final conv's .weight Parameter (VQGAN adaptive-adv anchor)

    n_tok = (freq_bins // patch_f) * (time_frames // patch_t), freq-outer / time-inner order.

Small synthetic CPU-only tensors.
"""
from __future__ import annotations

import pytest
import torch

from tokamak_foundation_model.ignite.codec import SpectroCodec
from tokamak_foundation_model.ignite.config import SpectroCodecConfig
from tokamak_foundation_model.ignite.discriminator import FreqAwarePatchGAN
from tokamak_foundation_model.ignite.nets import SpectroConvDecoder, SpectroDecoder


def _cfg(patch_f: int, patch_t: int, *, freq_bins: int = 128, time_frames: int = 32,
         channels: int = 1, decoder: str = "conv") -> SpectroCodecConfig:
    """Small-but-realistic spectro cfg with the requested (patch_f, patch_t)."""
    return SpectroCodecConfig(
        channels=channels,
        freq_bins=freq_bins,
        time_frames=time_frames,
        patch_f=patch_f,
        patch_t=patch_t,
        d_model=32,
        enc_depth=1,
        dec_depth=1,
        heads=2,
        fsq_levels=[4, 4, 3],
        decoder=decoder,
        conv_dec_base_ch=64,
        conv_dec_res_blocks=1,
    )


# --------------------------------------------------------------------------- #
# SpectroConvDecoder forward-shape contract (asymmetric + symmetric patches)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("patch_f,patch_t", [(64, 32), (16, 16)])
def test_conv_decoder_shape_and_finite(patch_f, patch_t):
    cfg = _cfg(patch_f, patch_t)
    dec = SpectroConvDecoder(cfg)
    B = 2
    quant = torch.randn(B, cfg.n_tok, cfg.d_model)
    recon = dec(quant)
    assert recon.shape == (B, cfg.channels, cfg.freq_bins, cfg.time_frames)
    assert torch.isfinite(recon).all()


@pytest.mark.parametrize("patch_f,patch_t", [(64, 32), (16, 16)])
def test_conv_decoder_multichannel_shape(patch_f, patch_t):
    cfg = _cfg(patch_f, patch_t, channels=2)
    dec = SpectroConvDecoder(cfg)
    quant = torch.randn(3, cfg.n_tok, cfg.d_model)
    recon = dec(quant)
    assert recon.shape == (3, 2, cfg.freq_bins, cfg.time_frames)


def test_conv_decoder_gradient_flows_to_last_layer():
    cfg = _cfg(64, 32)
    dec = SpectroConvDecoder(cfg)
    quant = torch.randn(2, cfg.n_tok, cfg.d_model)
    recon = dec(quant)
    recon.sum().backward()
    assert dec.last_layer.grad is not None
    assert torch.isfinite(dec.last_layer.grad).all()
    assert dec.last_layer.grad.abs().sum() > 0.0


def test_conv_decoder_last_layer_is_final_conv_weight():
    """last_layer must be the FINAL conv's .weight Parameter (the VQGAN adaptive-adv anchor)."""
    cfg = _cfg(64, 32, channels=2)
    dec = SpectroConvDecoder(cfg)
    assert dec.last_layer is dec.to_out.weight
    assert isinstance(dec.last_layer, torch.nn.Parameter)
    # final conv maps its input channels -> cfg.channels (it is the last conv before output).
    assert dec.to_out.out_channels == cfg.channels


def test_conv_decoder_gradient_flows_to_input():
    cfg = _cfg(16, 16)
    dec = SpectroConvDecoder(cfg)
    quant = torch.randn(2, cfg.n_tok, cfg.d_model, requires_grad=True)
    dec(quant).sum().backward()
    assert quant.grad is not None and quant.grad.abs().sum() > 0.0


# --------------------------------------------------------------------------- #
# SpectroCodec with decoder="conv": full forward + backprop
# --------------------------------------------------------------------------- #
def test_codec_conv_forward_shapes():
    cfg = _cfg(64, 32)
    codec = SpectroCodec(cfg)
    assert isinstance(codec.decoder, SpectroConvDecoder)
    B = 2
    x = torch.randn(B, cfg.channels, cfg.freq_bins, cfg.time_frames)
    out = codec(x)
    assert set(out) == {"recon", "feats", "quant", "codes"}
    assert out["recon"].shape == (B, cfg.channels, cfg.freq_bins, cfg.time_frames)
    assert out["feats"].shape == (B, cfg.n_tok, cfg.d_model)
    assert out["quant"].shape == (B, cfg.n_tok, cfg.d_model)
    assert out["codes"].shape == (B, cfg.n_tok, cfg.fsq_dim)
    assert out["codes"].dtype == torch.long
    assert torch.isfinite(out["recon"]).all()


def test_codec_conv_generator_losses_backprop():
    """generator_losses + adaptive-adv weight (which reads decoder.last_layer) work unchanged
    with the conv decoder; total backprops to codec params."""
    torch.manual_seed(0)
    cfg = _cfg(64, 32)
    codec = SpectroCodec(cfg)
    disc = FreqAwarePatchGAN(cfg)
    x = torch.randn(2, cfg.channels, cfg.freq_bins, cfg.time_frames)
    x_shift = x + 0.01 * torch.randn_like(x)
    out = codec.generator_losses(x, x_shift, disc, cfg)
    for k in ("total", "adversarial", "pixel", "consistency", "entropy"):
        assert k in out and out[k].dim() == 0 and torch.isfinite(out[k]).all()
    out["total"].backward()
    grads = [p.grad for p in codec.parameters() if p.grad is not None]
    assert len(grads) > 0 and all(torch.isfinite(g).all() for g in grads)
    # the adaptive-adv anchor (last_layer) received gradient.
    assert codec.decoder.last_layer.grad is not None


# --------------------------------------------------------------------------- #
# decoder="linear" default path is UNCHANGED (still builds SpectroDecoder)
# --------------------------------------------------------------------------- #
def test_default_decoder_is_linear():
    cfg = SpectroCodecConfig()
    assert cfg.decoder == "linear"


def test_codec_default_builds_linear_decoder():
    cfg = _cfg(64, 32, decoder="linear")
    codec = SpectroCodec(cfg)
    assert isinstance(codec.decoder, SpectroDecoder)
    assert not isinstance(codec.decoder, SpectroConvDecoder)


def test_codec_default_no_decoder_field_still_linear():
    """A cfg constructed with no explicit decoder arg builds the linear SpectroDecoder."""
    cfg = SpectroCodecConfig(
        channels=1, freq_bins=64, time_frames=32, patch_f=32, patch_t=16,
        d_model=32, enc_depth=1, dec_depth=1, heads=2,
    )
    codec = SpectroCodec(cfg)
    assert isinstance(codec.decoder, SpectroDecoder)
