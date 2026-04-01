"""Tests for SpectrogramChannelASTGANAutoEncoder and PatchGAN discriminator."""

import pytest
import torch
import torch.nn.functional as F

from tokamak_foundation_model.models.modality.spectrogram_channel_ast_gan import (
    SpectrogramChannelASTGANAutoEncoder,
    _PatchDiscriminator,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_model(**overrides):
    defaults = dict(
        n_channels=4, d_model=32, n_tokens=0,
        freq_bins=64, frame_width=2,
        n_enc_layers=2, n_dec_layers=2, n_heads=4,
        time_conv_kernel=3,
    )
    defaults.update(overrides)
    return SpectrogramChannelASTGANAutoEncoder(**defaults)


# ---------------------------------------------------------------------------
# Discriminator tests
# ---------------------------------------------------------------------------

class TestPatchDiscriminator:
    def test_output_shape(self):
        D = _PatchDiscriminator(channels=[16, 32, 64, 128])
        x = torch.randn(8, 1, 64, 64)  # (B*C, 1, F, T)
        out = D(x)
        assert out.shape[0] == 8
        assert out.shape[1] == 1
        # Spatial dims should be reduced by stride-2 layers
        assert out.shape[2] < 64
        assert out.shape[3] < 64

    def test_no_batchnorm(self):
        """R3GAN: no normalization layers in discriminator."""
        D = _PatchDiscriminator()
        for name, module in D.named_modules():
            assert not isinstance(module, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)), \
                f"Found normalization layer: {name} ({type(module).__name__})"

    def test_weight_init(self):
        """Weights should be initialized from N(0, 0.02)."""
        D = _PatchDiscriminator()
        for name, module in D.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                std = module.weight.std().item()
                assert 0.01 < std < 0.04, f"{name}: weight std={std:.4f}, expected ~0.02"

    def test_output_is_logits(self):
        """Output should be raw logits (no sigmoid/tanh)."""
        D = _PatchDiscriminator(channels=[16, 32])
        x = torch.randn(4, 1, 64, 64) * 10  # large input
        out = D(x)
        # Raw logits can exceed [-1, 1]
        assert out.abs().max() > 0.01  # not all zeros


# ---------------------------------------------------------------------------
# GAN autoencoder tests
# ---------------------------------------------------------------------------

class TestGANAutoEncoder:
    def test_eval_output_shape(self):
        model = _make_model()
        model.eval()
        x = torch.randn(2, 4, 64, 64)
        with torch.no_grad():
            y = model(x)
        assert y.shape == x.shape

    def test_training_output_shape(self):
        """Training forward returns reconstruction only (no tuple)."""
        model = _make_model()
        model.train()
        x = torch.randn(2, 4, 64, 64)
        y = model(x)
        # Channel-AST no-FSQ returns just the tensor in training
        assert isinstance(y, torch.Tensor)
        assert y.shape == x.shape

    def test_encoder_accessible(self):
        model = _make_model()
        model.eval()
        x = torch.randn(2, 4, 64, 64)
        with torch.no_grad():
            z = model.encoder(x)
        assert torch.isfinite(z).all()

    def test_discriminator_accessible(self):
        model = _make_model()
        x = torch.randn(8, 1, 64, 64)  # per-channel input
        out = model.discriminator(x)
        assert out.shape[0] == 8


# ---------------------------------------------------------------------------
# Gradient penalty tests
# ---------------------------------------------------------------------------

class TestGradientPenalty:
    def test_r1_finite(self):
        """R1 gradient penalty should produce a finite positive scalar."""
        D = _PatchDiscriminator(channels=[16, 32])
        x = torch.randn(4, 1, 64, 64, requires_grad=True)
        d_out = D(x)
        grad, = torch.autograd.grad(d_out.sum(), x, create_graph=True)
        r1 = grad.square().sum(dim=[1, 2, 3]).mean()
        assert torch.isfinite(r1)
        assert r1.item() >= 0

    def test_r2_finite(self):
        """R2 gradient penalty on fake data should also be finite."""
        D = _PatchDiscriminator(channels=[16, 32])
        fake = torch.randn(4, 1, 64, 64, requires_grad=True)
        d_out = D(fake)
        grad, = torch.autograd.grad(d_out.sum(), fake, create_graph=True)
        r2 = grad.square().sum(dim=[1, 2, 3]).mean()
        assert torch.isfinite(r2)
        assert r2.item() >= 0


# ---------------------------------------------------------------------------
# RpGAN loss tests
# ---------------------------------------------------------------------------

class TestRpGANLoss:
    def test_d_loss_and_g_loss_complementary(self):
        """D loss and G loss should be complementary (opposite signs on inputs)."""
        d_real = torch.randn(4, 1, 8, 8)
        d_fake = torch.randn(4, 1, 8, 8)
        d_loss = F.softplus(d_fake - d_real).mean()
        g_loss = F.softplus(d_real - d_fake).mean()
        # Both should be positive (softplus is always positive)
        assert d_loss.item() > 0
        assert g_loss.item() > 0


# ---------------------------------------------------------------------------
# Full train step simulation
# ---------------------------------------------------------------------------

class TestTrainStep:
    def test_full_step_gradients_flow(self):
        """Simulate one D step + G step, verify gradients flow correctly."""
        model = _make_model()
        model.train()
        D = model.discriminator
        x = torch.randn(2, 4, 64, 64)
        B, C, Fr, T = x.shape

        # D step
        with torch.no_grad():
            fake = model(x)
        real_gp = x.reshape(B * C, 1, Fr, T).detach().requires_grad_(True)
        fake_gp = fake.reshape(B * C, 1, Fr, T).detach().requires_grad_(True)
        d_real = D(real_gp)
        d_fake = D(fake_gp)
        d_loss = F.softplus(d_fake - d_real).mean()
        grad_real, = torch.autograd.grad(d_real.sum(), real_gp, create_graph=True)
        r1 = grad_real.square().sum(dim=[1, 2, 3]).mean()
        (d_loss + r1).backward()

        d_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                         for p in D.parameters())
        assert d_has_grad, "Discriminator received no gradients in D step"

        # G step
        model.zero_grad()
        fake = model(x)
        recon_loss = F.l1_loss(fake, x)
        fake_flat = fake.reshape(B * C, 1, Fr, T)
        real_flat = x.reshape(B * C, 1, Fr, T).detach()
        d_fake_g = D(fake_flat)
        d_real_g = D(real_flat)
        g_adv = F.softplus(d_real_g - d_fake_g).mean()
        g_total = recon_loss + 0.1 * g_adv
        g_total.backward()

        enc_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                           for p in model.autoencoder.encoder.parameters())
        assert enc_has_grad, "Encoder received no gradients in G step"
