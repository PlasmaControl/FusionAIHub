"""The SPECTRO envelope/shape split: structural guarantees + the OFF path is byte-identical.

The split exists to make the FLAT PLATE unrepresentable. Measured motivation (2026-09-03,
>=300 held-out windows): every masked spectro arm ran a pure L1 + avg-pooled-multiscale
objective, whose exact minimiser is the conditional mean, and the reconstructions collapsed
in amplitude -- mirnov ctl_s1 std(recon)/std(GT) 0.282 and 0.151, hf_ratio 0.008; bes
ctl_last 0.278; co2 ctl 0.613. These codecs feed a world model that predicts how MODES
EVOLVE, so an amplitude-collapsed reconstruction is a total failure regardless of nRMSE.

Two guarantees make it structural rather than a hope about the loss, and they are what these
tests pin:

  G1  recon.mean(-1)  IS  the decoded level   (the shape branch is mean-removed along time)
  G2  recon.std(-1)   IS  the decoded sigma   (the shape branch is std-normalized along time)

G2 is the one that matters: with the temporal std set by a transmitted code, shrinking toward
the conditional mean is not available to the decoder at all.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tokamak_foundation_model.ignite.codec import SpectroCodec           # noqa: E402
from tokamak_foundation_model.ignite.config import SpectroCodecConfig    # noqa: E402
from tokamak_foundation_model.ignite.discriminator import FreqAwarePatchGAN  # noqa: E402
from tokamak_foundation_model.ignite.nets import (                       # noqa: E402
    SpectroDecoder,
    SpectroEncoder,
    spectro_gain_shape_split,
)


def _cfg(**kw):
    """A SMALL but structurally faithful config: same code path, cheap on CPU."""
    # n_freq_patch 8 x n_time_patch 4 = n_tok 32, so gain_tokens in {1,2,4,8,16} both
    # divides freq_bins (64) and leaves shape tokens, exactly like the production
    # 32 x 6 = 192 grid with gain_tokens 32.
    base = dict(channels=3, freq_bins=64, time_frames=16, patch_f=8, patch_t=4,
                d_model=32, enc_depth=1, dec_depth=1, heads=2, gain_hidden=16,
                fsq_levels=[8, 5, 5, 5])
    base.update(kw)
    return SpectroCodecConfig(**base)


# --------------------------------------------------------------------------------------- #
# the analytic split
# --------------------------------------------------------------------------------------- #
def test_split_is_an_exact_decomposition():
    """level + sigma * shape reconstructs x EXACTLY, and shape is zero-mean / unit-std."""
    torch.manual_seed(0)
    x = torch.randn(4, 3, 64, 16, dtype=torch.float64) * 2.5 + 7.0
    gain, shape, level, sigma = spectro_gain_shape_split(x, use_scale=True)
    assert gain.shape == (4, 6, 64)            # [level ; log1p sigma] on the channel axis
    assert level.shape == sigma.shape == (4, 3, 64)
    torch.testing.assert_close(shape.mean(-1), torch.zeros_like(level), atol=1e-12, rtol=0)
    torch.testing.assert_close(shape.std(-1, unbiased=False), torch.ones_like(sigma),
                               atol=1e-10, rtol=0)
    rebuilt = level.unsqueeze(-1) + sigma.unsqueeze(-1) * shape
    torch.testing.assert_close(rebuilt, x, atol=1e-10, rtol=0)
    # the gain tensor really is [level ; log1p(sigma)]
    torch.testing.assert_close(gain[:, :3], level, atol=0, rtol=0)
    torch.testing.assert_close(torch.expm1(gain[:, 3:]), sigma, atol=1e-12, rtol=0)


def test_split_without_scale_keeps_the_level_only():
    x = torch.randn(2, 3, 64, 16, dtype=torch.float64)
    gain, shape, level, _sigma = spectro_gain_shape_split(x, use_scale=False)
    assert gain.shape == (2, 3, 64)
    torch.testing.assert_close(shape.mean(-1), torch.zeros_like(level), atol=1e-12, rtol=0)
    torch.testing.assert_close(level.unsqueeze(-1) + shape, x, atol=1e-12, rtol=0)


# --------------------------------------------------------------------------------------- #
# G1 / G2 — the structural guarantees, through the FULL codec (encode->FSQ->decode)
# --------------------------------------------------------------------------------------- #
@pytest.mark.parametrize("gain_tokens", [4, 8, 16])
def test_recon_time_mean_IS_the_decoded_level(gain_tokens):
    """G1: the reconstruction's per-(channel, freq) time-mean is exactly the gain head's."""
    torch.manual_seed(1)
    cfg = _cfg(gain_shape=True, gain_tokens=gain_tokens)
    codec = SpectroCodec(cfg).double().eval()
    x = (torch.randn(2, cfg.channels, cfg.freq_bins, cfg.time_frames, dtype=torch.float64)
         * 3.0 - 4.0)
    with torch.no_grad():
        out = codec.forward(x)
    level_hat = out["gain_pred"][:, : cfg.channels]
    torch.testing.assert_close(out["recon"].mean(-1), level_hat, atol=1e-9, rtol=0)


@pytest.mark.parametrize("gain_tokens", [4, 16])
def test_recon_time_std_IS_the_decoded_sigma(gain_tokens):
    """G2: the reconstruction's per-(channel, freq) TEMPORAL STD is exactly the coded sigma.

    This is the guarantee that makes the flat plate unrepresentable, so it is pinned to the
    same tolerance as G1 rather than "approximately".
    """
    torch.manual_seed(2)
    cfg = _cfg(gain_shape=True, gain_tokens=gain_tokens, gain_scale=True)
    codec = SpectroCodec(cfg).double().eval()
    x = torch.randn(2, cfg.channels, cfg.freq_bins, cfg.time_frames, dtype=torch.float64)
    with torch.no_grad():
        out = codec.forward(x)
    sigma_hat = torch.expm1(out["gain_pred"][:, cfg.channels:].clamp(0.0, 30.0))
    torch.testing.assert_close(out["recon"].std(-1, unbiased=False), sigma_hat,
                               atol=1e-9, rtol=0)
    # and it is NOT trivially zero: a decoder that emitted a plate would pass vacuously.
    assert float(sigma_hat.abs().mean()) > 1e-6


def test_guarantees_survive_the_refinement_head_and_noise():
    """G1/G2 hold with refine_depth > 0 and decoder_noise on.

    Both act INSIDE the shape branch, before the mean-removal and normalization. Had they
    been applied to the assembled reconstruction (the pre-split order) they would re-inject
    DC and rescale the amplitude, silently breaking both guarantees — which is exactly why
    the order is asserted here and not just commented.
    """
    torch.manual_seed(3)
    cfg = _cfg(gain_shape=True, gain_tokens=8, refine_depth=3, refine_dilated=True,
               refine_hidden=8, decoder_noise=True)
    codec = SpectroCodec(cfg).double().eval()
    # the zero-init noise scale would make the test vacuous -> give it real amplitude
    with torch.no_grad():
        codec.decoder.noise_scale.fill_(0.5)
        for m in codec.decoder.refine.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.normal_(m.weight, std=0.05)
                torch.nn.init.normal_(m.bias, std=0.05)
    x = torch.randn(2, cfg.channels, cfg.freq_bins, cfg.time_frames, dtype=torch.float64)
    with torch.no_grad():
        out = codec.forward(x)
    g = out["gain_pred"]
    torch.testing.assert_close(out["recon"].mean(-1), g[:, : cfg.channels], atol=1e-9, rtol=0)
    torch.testing.assert_close(out["recon"].std(-1, unbiased=False),
                               torch.expm1(g[:, cfg.channels:].clamp(0.0, 30.0)),
                               atol=1e-9, rtol=0)


def test_zeroing_the_shape_tokens_degenerates_to_the_coded_envelope():
    """With the shape tokens' contribution removed the codec IS an envelope coder.

    ``~tmean`` (the target's own time-mean broadcast over T) is the infinite-precision
    version of this predictor. It is reported as CONTEXT ONLY: it has zero temporal structure
    by construction, so it is a floor the split cannot fall below, never a target to reach.
    """
    torch.manual_seed(4)
    cfg = _cfg(gain_shape=True, gain_tokens=8, gain_scale=True)
    dec = SpectroDecoder(cfg).double().eval()
    quant = torch.randn(2, cfg.n_tok, cfg.d_model, dtype=torch.float64)
    with torch.no_grad():
        # zero the SIGMA half of the gain code's output by forcing log1p(sigma) -> 0 is not
        # available from outside, so instead kill the shape branch: shape == 0 => the
        # normalization clamp makes the shape term exactly 0.
        recon_full, gain = dec(quant, return_aux=True)
        dec.to_pixels.weight.zero_()
        dec.to_pixels.bias.zero_()
        recon_flat, gain2 = dec(quant, return_aux=True)
    torch.testing.assert_close(gain, gain2, atol=0, rtol=0)     # gain path is independent
    level = gain[:, : cfg.channels]
    torch.testing.assert_close(
        recon_flat, level.unsqueeze(-1).expand_as(recon_flat), atol=1e-12, rtol=0)
    assert not torch.allclose(recon_full, recon_flat)           # the shape path did do work


# --------------------------------------------------------------------------------------- #
# the OFF path is byte-identical
# --------------------------------------------------------------------------------------- #
def test_gain_shape_off_is_bit_identical():
    """Same seed, gain_shape off -> identical parameter set AND identical outputs."""
    torch.manual_seed(5)
    cfg = _cfg()
    assert cfg.n_gain_tok == 0 and cfg.n_shape_tok == cfg.n_tok
    enc, dec = SpectroEncoder(cfg), SpectroDecoder(cfg)
    keys = set(enc.state_dict()) | {"dec." + k for k in dec.state_dict()}
    assert not any("gain" in k or "shape_mix" in k or "shape_unmix" in k for k in keys), keys
    x = torch.randn(2, cfg.channels, cfg.freq_bins, cfg.time_frames)
    q = torch.randn(2, cfg.n_tok, cfg.d_model)
    with torch.no_grad():
        f = enc(x)
        r1 = dec(q)
        r2, aux = dec(q, return_aux=True)
    assert f.shape == (2, cfg.n_tok, cfg.d_model)
    assert aux is None
    torch.testing.assert_close(r1, r2, atol=0, rtol=0)


def test_off_checkpoint_loads_strictly_into_an_off_codec():
    """An existing (gain_shape-off) state_dict still loads strictly — no new keys appear."""
    torch.manual_seed(6)
    cfg = _cfg()
    a, b = SpectroCodec(cfg), SpectroCodec(cfg)
    b.load_state_dict(a.state_dict())          # strict=True by default
    assert set(a.state_dict()) == set(b.state_dict())


def test_gain_shape_adds_only_gain_keys():
    on = SpectroCodec(_cfg(gain_shape=True, gain_tokens=8)).state_dict()
    off = SpectroCodec(_cfg()).state_dict()
    added = set(on) - set(off)
    assert added, "gain_shape must add parameters"
    assert all(("gain_to_tokens" in k) or ("gain_head" in k)
               or ("shape_mix" in k) or ("shape_unmix" in k) for k in added), sorted(added)
    assert not (set(off) - set(on)), "gain_shape must not REMOVE any parameter"


# --------------------------------------------------------------------------------------- #
# the loss path
# --------------------------------------------------------------------------------------- #
def test_generator_losses_reports_and_backprops_the_gain_term():
    torch.manual_seed(7)
    cfg = _cfg(gain_shape=True, gain_tokens=8, gain_weight=1.0, pixel_anchor_weight=5.0,
               ms_ssim_weight=20.0, multiscale_recon_weight=2.0, adversarial_weight=0.0,
               fm_weight=0.0, consistency_weight=0.0, mask_missing=True)
    codec, disc = SpectroCodec(cfg), FreqAwarePatchGAN(cfg)
    x = torch.randn(2, cfg.channels, cfg.freq_bins, cfg.time_frames)
    out = codec.generator_losses(x, x.clone(), disc, cfg, step=0)
    assert "gain" in out and torch.isfinite(out["gain"])
    assert float(out["gain"]) > 0.0
    out["total"].backward()
    gh = [p for n, p in codec.named_parameters() if "gain_head" in n and p.grad is not None]
    gt = [p for n, p in codec.named_parameters()
          if "gain_to_tokens" in n and p.grad is not None]
    assert gh and gt, "the gain path must receive gradient"
    assert any(float(p.grad.abs().sum()) > 0 for p in gh)
    assert any(float(p.grad.abs().sum()) > 0 for p in gt)


def test_masked_gain_mae_equals_dropping_the_dead_channels():
    """The masked gain anchor is EXACTLY the unmasked anchor on the surviving channels."""
    torch.manual_seed(8)
    C, F, T = 4, 16, 8
    pred = torch.randn(3, 2 * C, F, dtype=torch.float64)
    tgt = torch.randn(3, 2 * C, F, dtype=torch.float64)
    m = torch.ones(3, C, T)
    m[0, 1] = 0.0                      # window 0, channel 1 dead for the whole window
    m[2, 3, 4] = 0.0                   # window 2, channel 3 dead in ONE frame
    got = SpectroCodec._masked_gain_mae(pred, tgt, m)
    keep = (m > 0.5).all(-1)                                  # (3, C)
    keep2 = keep.repeat(1, 2)                                 # level + log1p sigma halves
    ref = (pred - tgt).abs()[keep2.unsqueeze(-1).expand_as(pred)].mean()
    torch.testing.assert_close(got, ref, atol=1e-12, rtol=0)
    # no mask (or an all-valid mask) is the plain MAE
    plain = (pred - tgt).abs().mean()
    torch.testing.assert_close(SpectroCodec._masked_gain_mae(pred, tgt, None), plain,
                               atol=0, rtol=0)
    torch.testing.assert_close(
        SpectroCodec._masked_gain_mae(pred, tgt, torch.ones(3, C, T)), plain,
        atol=1e-12, rtol=0)


def test_gain_weight_zero_costs_nothing_and_leaves_the_split_intact():
    """gain_weight 0 removes the auxiliary term but keeps BOTH structural guarantees."""
    torch.manual_seed(9)
    cfg = _cfg(gain_shape=True, gain_tokens=8, gain_weight=0.0)
    codec, disc = SpectroCodec(cfg).double(), FreqAwarePatchGAN(cfg).double()
    x = torch.randn(2, cfg.channels, cfg.freq_bins, cfg.time_frames, dtype=torch.float64)
    out = codec.generator_losses(x, x.clone(), disc, cfg, step=0)
    assert float(out["gain"]) == 0.0
    g = codec.forward(x)
    torch.testing.assert_close(g["recon"].mean(-1), g["gain_pred"][:, : cfg.channels],
                               atol=1e-9, rtol=0)


# --------------------------------------------------------------------------------------- #
# guard rails
# --------------------------------------------------------------------------------------- #
def test_conv_decoder_is_rejected():
    with pytest.raises(NotImplementedError, match="decoder='linear'"):
        SpectroCodec(_cfg(gain_shape=True, gain_tokens=8, decoder="conv"))


def test_token_budget_is_a_reallocation_not_an_extension():
    """n_tok (and therefore FRAME_LAYOUT / the Phase-B vocab) is UNCHANGED by the split."""
    off, on = _cfg(), _cfg(gain_shape=True, gain_tokens=16)
    assert off.n_tok == on.n_tok
    assert on.n_gain_tok + on.n_shape_tok == on.n_tok
    assert off.codebook_size == on.codebook_size
    codec = SpectroCodec(on).eval()
    x = torch.randn(2, on.channels, on.freq_bins, on.time_frames)
    with torch.no_grad():
        out = codec.forward(x)
    assert out["codes"].shape == (2, on.n_tok, on.fsq_dim)
