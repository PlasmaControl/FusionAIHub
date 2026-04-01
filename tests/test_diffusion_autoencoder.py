"""Tests for SpectrogramChannelASTDiffusionAutoEncoder."""

import pytest
import torch

from tokamak_foundation_model.models.modality.spectrogram_channel_ast_diffusion import (
    SpectrogramChannelASTDiffusionAutoEncoder,
    _TimestepEmbedding,
    _AdaLNChannelTimeBlock,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_model(**overrides):
    defaults = dict(
        n_channels=4,
        d_model=32,
        n_tokens=0,
        freq_bins=64,
        frame_width=2,
        n_enc_layers=2,
        n_dec_layers=2,
        n_heads=4,
        time_conv_kernel=3,
        eval_steps=2,
    )
    defaults.update(overrides)
    return SpectrogramChannelASTDiffusionAutoEncoder(**defaults)


# ---------------------------------------------------------------------------
# Component tests
# ---------------------------------------------------------------------------

class TestTimestepEmbedding:
    def test_output_shape(self):
        emb = _TimestepEmbedding(d_model=32)
        t = torch.rand(8)
        out = emb(t)
        assert out.shape == (8, 32)

    def test_different_timesteps_produce_different_embeddings(self):
        emb = _TimestepEmbedding(d_model=32)
        t1 = torch.tensor([0.1])
        t2 = torch.tensor([0.9])
        out1 = emb(t1)
        out2 = emb(t2)
        assert not torch.allclose(out1, out2)


class TestAdaLNChannelTimeBlock:
    def test_output_shape(self):
        block = _AdaLNChannelTimeBlock(d_model=32, n_heads=4, dropout=0.0, time_conv_kernel=3)
        x = torch.randn(2, 4, 8, 32)  # (B, C, N, D)
        t_emb = torch.randn(2, 32)    # (B, D)
        out = block(x, t_emb)
        assert out.shape == x.shape

    def test_zero_init_gate_gives_identity(self):
        """At init, gate=0 so output should equal input (identity)."""
        block = _AdaLNChannelTimeBlock(d_model=32, n_heads=4, dropout=0.0, time_conv_kernel=3)
        x = torch.randn(1, 2, 4, 32)
        t_emb = torch.zeros(1, 32)
        with torch.no_grad():
            out = block(x, t_emb)
        # Gate is zero-initialized → output = x + 0 * (block(x) - x) = x
        assert torch.allclose(out, x, atol=1e-5)


# ---------------------------------------------------------------------------
# Full model tests
# ---------------------------------------------------------------------------

class TestDiffusionAutoEncoder:
    def test_eval_output_shape(self):
        model = _make_model()
        model.eval()
        x = torch.randn(2, 4, 64, 64)
        with torch.no_grad():
            y = model(x)
        assert y.shape == x.shape

    def test_eval_output_shape_fw8(self):
        model = _make_model(frame_width=8)
        model.eval()
        x = torch.randn(2, 4, 64, 64)
        with torch.no_grad():
            y = model(x)
        assert y.shape == x.shape

    def test_eval_output_shape_fw16(self):
        model = _make_model(frame_width=16)
        model.eval()
        x = torch.randn(2, 4, 64, 64)
        with torch.no_grad():
            y = model(x)
        assert y.shape == x.shape

    def test_eval_output_shape_odd_time(self):
        """T not divisible by frame_width should still work (padded internally)."""
        model = _make_model(frame_width=8)
        model.eval()
        x = torch.randn(2, 4, 64, 50)  # 50 not divisible by 8
        with torch.no_grad():
            y = model(x)
        assert y.shape == x.shape

    def test_training_returns_tuple(self):
        model = _make_model()
        model.train()
        x = torch.randn(2, 4, 64, 64)
        output = model(x)
        assert isinstance(output, tuple)
        assert len(output) == 2
        x_hat, loss = output
        assert x_hat.shape == x.shape
        assert loss.dim() == 0  # scalar

    def test_training_loss_is_finite(self):
        model = _make_model()
        model.train()
        x = torch.randn(2, 4, 64, 64)
        _, loss = model(x)
        assert torch.isfinite(loss).all()

    def test_training_loss_requires_grad(self):
        model = _make_model()
        model.train()
        x = torch.randn(2, 4, 64, 64)
        _, loss = model(x)
        assert loss.requires_grad

    def test_encoder_output_is_finite(self):
        model = _make_model()
        model.eval()
        x = torch.randn(2, 4, 64, 64)
        with torch.no_grad():
            z = model.encoder(x)
        assert torch.isfinite(z).all()

    def test_encoder_output_shape(self):
        """Encoder returns (B, C*N, d_model)."""
        model = _make_model(frame_width=2)
        model.eval()
        x = torch.randn(2, 4, 64, 64)
        with torch.no_grad():
            z = model.encoder(x)
        # N = T / fw = 64 / 2 = 32, C = 4 → C*N = 128
        assert z.shape == (2, 4 * 32, 32)

    def test_backward_pass(self):
        """Verify gradients flow through the full model."""
        model = _make_model()
        model.train()
        x = torch.randn(2, 4, 64, 64)
        _, loss = model(x)
        loss.backward()
        # Check that encoder and decoder both received gradients
        enc_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.encoder.parameters())
        dec_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.decoder.parameters())
        assert enc_grad, "Encoder received no gradients"
        assert dec_grad, "Decoder received no gradients"

    def test_single_channel(self):
        model = _make_model(n_channels=1)
        model.eval()
        x = torch.randn(2, 1, 64, 64)
        with torch.no_grad():
            y = model(x)
        assert y.shape == x.shape

    def test_batch_size_one(self):
        model = _make_model()
        model.eval()
        x = torch.randn(1, 4, 64, 64)
        with torch.no_grad():
            y = model(x)
        assert y.shape == x.shape
