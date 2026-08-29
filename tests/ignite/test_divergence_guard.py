"""Tests for the DDP-safe DIVERGENCE GUARD (stabilization FIX 2).

Diagnosis (two real crashes): a collapsing co2 / video codec drove the adaptive adversarial
weight + generator loss to Inf/NaN on ONE rank; that rank corrupted its params while the
other ranks stepped cleanly -> ranks desynced -> the next NCCL collective hit the watchdog
-> SIGTERM (exit 143). The guard makes the SKIP decision identical on every rank:

    * local bad-flag = 1.0 if this rank's loss is non-finite else 0.0
    * DDP (world_size>1): all_reduce(MAX) the flag -> ALL ranks skip if ANY rank is bad
    * non-DDP (world_size==1): local flag only
    * only optimizer.step() is skipped; backward() (+ grad all-reduce) still runs uniformly.

These tests cover the SINGLE-PROCESS path (real skip + counter) and the SYMMETRY of the
collective decision (mocked dist.all_reduce), without any GPU / real process group.
"""
from __future__ import annotations

import math
from unittest import mock

import torch

from tokamak_foundation_model.ignite import spike
from tokamak_foundation_model.ignite import train_codec as tc
from tokamak_foundation_model.ignite.codec import SpectroCodec
from tokamak_foundation_model.ignite.config import SpectroCodecConfig
from tokamak_foundation_model.ignite.discriminator import FreqAwarePatchGAN


def _small_cfg(**overrides) -> SpectroCodecConfig:
    base = dict(
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
    base.update(overrides)
    return SpectroCodecConfig(**base)


def _pair(cfg):
    x = torch.randn(2, cfg.channels, cfg.freq_bins, cfg.time_frames)
    return x, x + 0.05 * torch.randn_like(x)


# --------------------------------------------------------------------------- #
# is_step_diverged: local (non-distributed) decision
# --------------------------------------------------------------------------- #
def test_local_finite_loss_is_not_diverged():
    assert spike.is_step_diverged(torch.tensor(1.234)) is False
    assert spike.is_step_diverged(torch.tensor(0.0)) is False


def test_local_nonfinite_loss_is_diverged():
    assert spike.is_step_diverged(torch.tensor(float("nan"))) is True
    assert spike.is_step_diverged(torch.tensor(float("inf"))) is True
    assert spike.is_step_diverged(torch.tensor(float("-inf"))) is True


def test_local_any_nonfinite_among_several_is_diverged():
    good = torch.tensor(0.5)
    bad = torch.tensor(float("inf"))
    assert spike.is_step_diverged(good, bad) is True
    assert spike.is_step_diverged(good, good) is False


# --------------------------------------------------------------------------- #
# skip counter
# --------------------------------------------------------------------------- #
def test_skip_counter_reset_and_increment():
    spike.reset_skipped_steps()
    assert spike.skipped_steps() == 0
    spike.note_skipped_step()
    spike.note_skipped_step()
    assert spike.skipped_steps() == 2
    spike.reset_skipped_steps()
    assert spike.skipped_steps() == 0


# --------------------------------------------------------------------------- #
# integration: single-process codec_train_step SKIPS on a non-finite total
# --------------------------------------------------------------------------- #
def _snapshot(module):
    return [p.detach().clone() for p in module.parameters()]


def _params_unchanged(module, snap):
    return all(torch.equal(p.detach(), s) for p, s in zip(module.parameters(), snap))


def test_train_step_skips_optimizer_on_nonfinite_total():
    """A non-finite generator total must leave codec params UNCHANGED (opt_g.step skipped)."""
    torch.manual_seed(0)
    cfg = _small_cfg()
    codec = SpectroCodec(cfg)
    disc = FreqAwarePatchGAN(cfg)
    opt_g = torch.optim.Adam(codec.parameters(), lr=1e-1)
    opt_d = torch.optim.Adam(disc.parameters(), lr=1e-1)
    x, x_shift = _pair(cfg)

    # Force the generator total non-finite, but keep it differentiable w.r.t. params so the
    # backward (and, under DDP, its grad all-reduce) still runs uniformly before the skip.
    real_gen = codec.generator_losses

    def _poisoned(*a, **k):
        out = real_gen(*a, **k)
        out["total"] = out["total"] * float("inf")  # differentiable -> non-finite
        return out

    spike.reset_skipped_steps()
    codec_snap = _snapshot(codec)
    with mock.patch.object(codec, "generator_losses", side_effect=_poisoned):
        spike.codec_train_step(codec, disc, opt_g, opt_d, x, x_shift, cfg, step=0)

    assert _params_unchanged(codec, codec_snap), "codec params must be unchanged on skip"
    assert spike.skipped_steps() >= 1, "the skip counter must have fired"


def test_train_step_takes_step_on_finite_total():
    """The healthy path is unchanged: a finite total DOES update the codec params."""
    torch.manual_seed(1)
    cfg = _small_cfg()
    codec = SpectroCodec(cfg)
    disc = FreqAwarePatchGAN(cfg)
    opt_g = torch.optim.Adam(codec.parameters(), lr=1e-1)
    opt_d = torch.optim.Adam(disc.parameters(), lr=1e-1)
    x, x_shift = _pair(cfg)

    spike.reset_skipped_steps()
    codec_snap = _snapshot(codec)
    spike.codec_train_step(codec, disc, opt_g, opt_d, x, x_shift, cfg, step=0)

    assert not _params_unchanged(codec, codec_snap), "codec params must update on a finite step"
    assert spike.skipped_steps() == 0


# --------------------------------------------------------------------------- #
# COLLECTIVE SYMMETRY: mocked dist so the decision is the SAME on every rank
# --------------------------------------------------------------------------- #
class _FakeReduceOp:
    MAX = "MAX"


def _mock_dist(world_size: int, reduced_value: float):
    """Patch spike.dist to look distributed with world_size and an all_reduce that writes
    ``reduced_value`` into the flag tensor (simulating MAX across ranks)."""
    m = mock.MagicMock()
    m.is_available.return_value = True
    m.is_initialized.return_value = True
    m.get_world_size.return_value = world_size
    m.ReduceOp = _FakeReduceOp

    def _all_reduce(tensor, op=None):
        assert op == _FakeReduceOp.MAX, "guard must use ReduceOp.MAX for a symmetric decision"
        tensor.fill_(reduced_value)  # MAX across ranks -> same value on every rank

    m.all_reduce.side_effect = _all_reduce
    return m


def test_collective_all_ranks_skip_when_any_rank_bad():
    """If ANY rank is non-finite, the MAX-reduced flag is 1 on EVERY rank -> all skip.

    Simulated per-rank: this rank's loss is FINITE (local flag 0), but a peer rank was bad,
    so all_reduce(MAX) yields 1 -> this (locally-clean) rank still returns True."""
    finite_here = torch.tensor(0.5)
    with mock.patch.object(spike, "dist", _mock_dist(world_size=4, reduced_value=1.0)):
        assert spike.is_step_diverged(finite_here) is True


def test_collective_no_rank_skips_when_all_clean():
    """All ranks finite -> MAX-reduced flag is 0 on every rank -> nobody skips."""
    finite_here = torch.tensor(0.5)
    with mock.patch.object(spike, "dist", _mock_dist(world_size=4, reduced_value=0.0)):
        assert spike.is_step_diverged(finite_here) is False


def test_collective_bad_local_rank_participates_in_all_reduce():
    """A locally-bad rank must still enter the collective (all_reduce called), so the collective
    stays symmetric — every rank calls all_reduce exactly once regardless of local state."""
    bad_here = torch.tensor(float("nan"))
    fake = _mock_dist(world_size=4, reduced_value=1.0)
    with mock.patch.object(spike, "dist", fake):
        assert spike.is_step_diverged(bad_here) is True
    assert fake.all_reduce.call_count == 1


def test_single_process_never_calls_all_reduce():
    """world_size==1: purely local decision, no collective (all_reduce not called)."""
    fake = _mock_dist(world_size=1, reduced_value=0.0)
    with mock.patch.object(spike, "dist", fake):
        assert spike.is_step_diverged(torch.tensor(float("inf"))) is True
        assert spike.is_step_diverged(torch.tensor(1.0)) is False
    assert fake.all_reduce.call_count == 0


def test_not_initialized_is_local_only():
    """dist available but not initialized -> local decision, no collective."""
    m = mock.MagicMock()
    m.is_available.return_value = True
    m.is_initialized.return_value = False
    m.get_world_size.return_value = 8
    with mock.patch.object(spike, "dist", m):
        assert spike.is_step_diverged(torch.tensor(float("nan"))) is True
        assert spike.is_step_diverged(torch.tensor(1.0)) is False
    m.all_reduce.assert_not_called()


# --------------------------------------------------------------------------- #
# the DDP train-steps route through spike's shared guard helpers
# --------------------------------------------------------------------------- #
def test_ddp_train_steps_use_shared_guard():
    """The production DDP steps must call the SAME guard helpers (single source of truth)."""
    import inspect

    for fn in (
        tc._ddp_codec_train_step,
        tc._ddp_video_train_step,
        tc._ddp_slowts_train_step,
    ):
        src = inspect.getsource(fn)
        assert "spike.is_step_diverged" in src, fn.__name__
        assert "spike.note_skipped_step" in src, fn.__name__

    from tokamak_foundation_model.ignite import fastts_train as ft

    src = inspect.getsource(ft._ddp_fastts_train_step)
    assert "spike.is_step_diverged" in src
    assert "spike.note_skipped_step" in src
