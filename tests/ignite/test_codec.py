"""CPU smoke tests for the composed Phase-A codec + the gate spike.

Contract (docs/IGNITE_DESIGN.md §4, and the component signatures):

    codec.SpectroCodec(cfg)
        .encode(x: (B,C,F,T)) -> feats (B, n_tok, d_model)
        .forward(x)           -> dict(recon (B,C,F,T), feats, quant, codes)
        .generator_losses(x, x_shift, disc, cfg) -> dict(total, adversarial, pixel,
                                                          consistency, recon, codes)

    spike.synthetic_batches(cfg, n_batches, batch_size, device) -> [(spec_a, spec_b), ...]
    spike.run_spike(cfg, batches, steps, device, eval_every) -> gate dict

Small synthetic tensors only, CPU. No real HDF5. No SLURM / GPU.
"""
from __future__ import annotations

import torch

from tokamak_foundation_model.ignite import spike
from tokamak_foundation_model.ignite.codec import SpectroCodec
from tokamak_foundation_model.ignite.config import SpectroCodecConfig
from tokamak_foundation_model.ignite.discriminator import FreqAwarePatchGAN


def _small_cfg() -> SpectroCodecConfig:
    """Tiny transformer + small patch grid, but full-STFT-grid freq/time so the
    data-glue crop/pad path exercises realistic sizes."""
    return SpectroCodecConfig(
        channels=1,
        freq_bins=64,
        time_frames=32,
        patch_f=32,
        patch_t=16,
        d_model=32,
        enc_depth=1,
        dec_depth=1,
        heads=2,
        fsq_levels=[4, 4, 3],
    )


# --------------------------------------------------------------------------- #
# SpectroCodec forward shapes
# --------------------------------------------------------------------------- #
def test_codec_forward_shapes():
    cfg = _small_cfg()
    codec = SpectroCodec(cfg)
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


def test_codec_encode_matches_forward_feats():
    cfg = _small_cfg()
    codec = SpectroCodec(cfg)
    codec.eval()  # no dropout/randomness between the two calls
    x = torch.randn(2, cfg.channels, cfg.freq_bins, cfg.time_frames)
    feats = codec.encode(x)
    out = codec(x)
    assert torch.allclose(feats, out["feats"], atol=1e-5)


def test_codec_codes_in_range():
    cfg = _small_cfg()
    codec = SpectroCodec(cfg)
    x = torch.randn(4, cfg.channels, cfg.freq_bins, cfg.time_frames) * 5.0
    codes = codec(x)["codes"]
    levels = torch.tensor(cfg.fsq_levels)
    assert (codes >= 0).all()
    assert (codes < levels).all()
    assert codec.codebook_size == cfg.codebook_size


# --------------------------------------------------------------------------- #
# generator_losses: one step reduces the total generator loss
# --------------------------------------------------------------------------- #
def test_generator_losses_keys_and_finite():
    cfg = _small_cfg()
    codec = SpectroCodec(cfg)
    disc = FreqAwarePatchGAN(cfg)
    x = torch.randn(2, cfg.channels, cfg.freq_bins, cfg.time_frames)
    x_shift = x + 0.01 * torch.randn_like(x)
    out = codec.generator_losses(x, x_shift, disc, cfg)
    for k in ("total", "adversarial", "pixel", "consistency"):
        assert k in out and out[k].dim() == 0 and torch.isfinite(out[k]).all()
    assert out["recon"].shape == x.shape
    assert out["codes"].shape == (2, cfg.n_tok, cfg.fsq_dim)


def test_generator_step_reduces_total_loss():
    """One Adam step on the generator loss should lower it on the SAME batch."""
    torch.manual_seed(0)
    cfg = _small_cfg()
    codec = SpectroCodec(cfg)
    disc = FreqAwarePatchGAN(cfg)
    opt = torch.optim.Adam(codec.parameters(), lr=1e-2)

    batches = spike.synthetic_batches(cfg, n_batches=1, batch_size=2, seed=1)
    x, x_shift = batches[0]

    codec.train()
    loss_before = codec.generator_losses(x, x_shift, disc, cfg)["total"]
    opt.zero_grad(set_to_none=True)
    loss_before.backward()
    # gradient actually reaches codec params
    grads = [p.grad for p in codec.parameters() if p.grad is not None]
    assert len(grads) > 0 and all(torch.isfinite(g).all() for g in grads)
    opt.step()

    loss_after = codec.generator_losses(x, x_shift, disc, cfg)["total"]
    assert loss_after.item() < loss_before.item(), (
        loss_before.item(), loss_after.item()
    )


# --------------------------------------------------------------------------- #
# gate functions run on codec outputs and return finite values
# --------------------------------------------------------------------------- #
def test_gate_runs_on_codec_outputs():
    cfg = _small_cfg()
    codec = SpectroCodec(cfg)
    eval_pairs = spike.synthetic_batches(cfg, n_batches=1, batch_size=2, seed=5)
    frame_seq = spike._consecutive_frame_sequence(cfg, batch_size=2, n_frames=4, seed=6)

    g = spike.compute_gate(codec, eval_pairs, frame_seq, cfg)
    assert 0.0 <= g["stability"] <= 1.0
    assert 0.0 <= g["persistence"] <= 1.0
    assert isinstance(g["pass_stability"], bool)
    assert isinstance(g["pass_persistence"], bool)

    fc = g["forecastability"]
    for key in ("margin_transition", "margin_stable", "margin_overall", "beats_persistence"):
        assert key in fc
    assert isinstance(fc["beats_persistence"], bool)

    dec = g["decode"]
    for key in ("envelope_corr", "peak_f1", "sharpness"):
        assert key in dec
        assert isinstance(dec[key], float)
    # correlation / f1 are bounded; sharpness is a finite ratio
    assert -1.0 <= dec["envelope_corr"] <= 1.0
    assert 0.0 <= dec["peak_f1"] <= 1.0
    import math as _math
    assert _math.isfinite(dec["sharpness"])


# --------------------------------------------------------------------------- #
# synthetic_batches shape contract
# --------------------------------------------------------------------------- #
def test_synthetic_batches_shapes():
    cfg = _small_cfg()
    batches = spike.synthetic_batches(cfg, n_batches=3, batch_size=2, seed=0)
    assert len(batches) == 3
    for spec_a, spec_b in batches:
        assert spec_a.shape == (2, cfg.channels, cfg.freq_bins, cfg.time_frames)
        assert spec_b.shape == (2, cfg.channels, cfg.freq_bins, cfg.time_frames)
        assert torch.isfinite(spec_a).all() and torch.isfinite(spec_b).all()
        # δ-shift pair shares modes but is not identical
        assert not torch.allclose(spec_a, spec_b, atol=1e-4)


# NOTE: real_ece_batches is now fully implemented (loads real ece HDF5). Its behaviour
# is verified against real data in scratch/verify_real_ece_path.py, not a hermetic CPU
# unit test — so the former "must remain a stub" assertion has been removed.


# --------------------------------------------------------------------------- #
# run_spike end-to-end (~3 steps) on synthetic data
# --------------------------------------------------------------------------- #
def test_run_spike_returns_gate_dict():
    cfg = _small_cfg()
    batches = spike.synthetic_batches(cfg, n_batches=3, batch_size=2, seed=2)
    g = spike.run_spike(
        cfg, batches, steps=3, device="cpu", eval_every=1, log_fn=None, seed=3
    )
    for key in ("stability", "persistence", "forecastability", "decode",
                "pass_stability", "pass_persistence", "steps"):
        assert key in g, f"missing gate key '{key}'"
    assert g["steps"] == 3
    assert 0.0 <= g["stability"] <= 1.0
    assert 0.0 <= g["persistence"] <= 1.0
    import math as _math
    assert _math.isfinite(g["forecastability"]["margin_transition"])
    assert _math.isfinite(g["decode"]["envelope_corr"])
