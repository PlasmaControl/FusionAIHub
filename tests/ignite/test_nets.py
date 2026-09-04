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


def test_spectro_default_is_192_tokens_and_pe_sized_32x6() -> None:
    """DESIGNED per-frame budget: the production spectro default is 192 tokens/modality.

    patch_f=16 -> 512/16 = 32 freq-patches; patch_t=16 -> 96/16 = 6 time-patches; n_tok = 192.
    The freq/time positional-embedding tables must be config-driven (sized 32 / 6), NOT the old
    8 / 3.
    """
    cfg = SpectroCodecConfig()  # real production default
    assert cfg.patch_f == 16 and cfg.patch_t == 16
    assert cfg.n_freq_patch == 32 and cfg.n_time_patch == 6
    assert cfg.n_tok == 192
    # PE tables span the new grid (config-driven, not hard-coded 8/3).
    enc = SpectroEncoder(cfg)
    dec = SpectroDecoder(cfg)
    assert tuple(enc.pos_emb.freq_pe.shape) == (32, cfg.d_model)
    assert tuple(enc.pos_emb.time_pe.shape) == (6, cfg.d_model)
    assert tuple(dec.pos_emb.freq_pe.shape) == (32, cfg.d_model)
    assert tuple(dec.pos_emb.time_pe.shape) == (6, cfg.d_model)
    # encoder round-trips at the real default grid -> 192 tokens.
    x = torch.randn(1, cfg.channels, cfg.freq_bins, cfg.time_frames)
    feats = enc(x)
    assert feats.shape == (1, 192, cfg.d_model)
    recon = dec(feats)
    assert recon.shape == x.shape and torch.isfinite(recon).all()


# ------------------------------------------------------------------------------------- #
# decoder conv REFINEMENT head (patch-lattice / checkerboard fix)
# ------------------------------------------------------------------------------------- #
def test_refine_depth_default_is_off_and_state_dict_unchanged() -> None:
    """refine_depth defaults to 0: NOTHING is constructed, so the state_dict is byte-identical
    to a pre-refine checkpoint's and every production / live-arm codec loads unchanged."""
    cfg = _small_cfg()
    assert cfg.refine_depth == 0
    dec = SpectroDecoder(cfg)
    assert dec.refine is None
    assert not any(k.startswith("refine") for k in dec.state_dict())
    # last_layer keeps pointing at to_pixels (the adaptive-adv weight's balance point).
    assert dec.last_layer is dec.to_pixels.weight


def test_refine_head_is_a_bit_identical_no_op_when_off() -> None:
    """Same weights + refine_depth 0 -> bit-identical output to the pre-change decoder path."""
    torch.manual_seed(0)
    cfg = _small_cfg()
    dec = SpectroDecoder(cfg).eval()
    quant = torch.randn(2, cfg.n_tok, cfg.d_model)
    with torch.no_grad():
        out = dec(quant)
        # reference: the unpatchify path with the refine head explicitly bypassed.
        dec.refine = None
        ref = dec(quant)
    assert torch.equal(out, ref)


def test_refine_head_starts_as_exact_identity() -> None:
    """The final conv is zero-init, so a freshly built refine head is an EXACT no-op: an arm
    with the head ON begins from the same function as the baseline arm."""
    torch.manual_seed(0)
    cfg = _small_cfg()
    dec_off = SpectroDecoder(cfg).eval()
    cfg_on = _small_cfg()
    cfg_on.refine_depth = 3
    cfg_on.refine_hidden = 8
    torch.manual_seed(0)
    dec_on = SpectroDecoder(cfg_on).eval()
    dec_on.load_state_dict(dec_off.state_dict(), strict=False)
    assert dec_on.refine is not None
    quant = torch.randn(2, cfg.n_tok, cfg.d_model)
    with torch.no_grad():
        assert torch.equal(dec_off(quant), dec_on(quant))
    # ...and the adaptive-adv balance point moves to the head's last conv.
    assert dec_on.last_layer is dec_on.refine[-1].weight


def test_refine_head_dilations_and_gradients() -> None:
    """Dilated head: receptive field 2^(D+1)-1, output shape preserved, gradients flow."""
    cfg = _small_cfg()
    cfg.refine_depth = 3
    cfg.refine_hidden = 8
    cfg.refine_dilated = True
    dec = SpectroDecoder(cfg)
    convs = [m for m in dec.refine if isinstance(m, torch.nn.Conv2d)]
    assert [c.dilation[0] for c in convs] == [1, 2, 4]
    quant = torch.randn(1, cfg.n_tok, cfg.d_model)
    out = dec(quant)
    assert out.shape == (1, cfg.channels, cfg.freq_bins, cfg.time_frames)
    # zero-init last conv means a zero grad at init would hide a wiring bug -> perturb first.
    with torch.no_grad():
        dec.refine[-1].weight.normal_(std=0.01)
    dec(quant).sum().backward()
    assert dec.refine[0].weight.grad is not None
    assert torch.isfinite(dec.refine[0].weight.grad).all()


def test_decoder_noise_default_off_and_zero_init_identity() -> None:
    """decoder_noise: no parameter when off (byte-identical state_dict); zero-init when on, so
    a noise-enabled decoder is an EXACT identity to the noiseless one at step 0 — yet the
    scale still receives a gradient, so it can grow if the objective rewards it."""
    torch.manual_seed(0)
    cfg = _small_cfg()
    assert cfg.decoder_noise is False
    dec_off = SpectroDecoder(cfg).eval()
    assert dec_off.noise_scale is None
    assert not any("noise" in k for k in dec_off.state_dict())

    cfg_on = _small_cfg()
    cfg_on.decoder_noise = True
    torch.manual_seed(0)
    dec_on = SpectroDecoder(cfg_on).eval()
    dec_on.load_state_dict(dec_off.state_dict(), strict=False)
    quant = torch.randn(2, cfg.n_tok, cfg.d_model)
    with torch.no_grad():
        assert torch.equal(dec_off(quant), dec_on(quant))
    assert tuple(dec_on.noise_scale.shape) == (cfg.channels,)

    with torch.no_grad():
        dec_on.noise_scale.fill_(0.1)
    dec_on(quant).sum().backward()
    assert dec_on.noise_scale.grad is not None
    assert torch.isfinite(dec_on.noise_scale.grad).all()
    # ...and with a non-zero scale the decoder is genuinely stochastic.
    with torch.no_grad():
        assert not torch.equal(dec_on(quant), dec_on(quant))
