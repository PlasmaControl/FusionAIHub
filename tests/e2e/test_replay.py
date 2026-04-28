"""Unit tests for :class:`TrajectoryPool` + :class:`ReplayBuffer`.

Synthetic trajectories only — no real dataset access so these are fast (<5 s)
and fully deterministic.
"""

from typing import Dict, List

import pytest
import torch

from tokamak_foundation_model.e2e.replay import (
    BufferBatch,
    PoolTrajectory,
    ReplayBuffer,
    TrajectoryPool,
)

DIAG = ("slow_a", "slow_b", "fast_c")
ACT = ("act_a",)
SAMPLE_RATES: Dict[str, float] = {
    "slow_a": 100.0,
    "slow_b": 100.0,
    "fast_c": 10_000.0,
    "act_a": 10_000.0,
}
CHUNK_S = 0.05
K_MAX = 5
N_DIAG_TOKENS = 4  # arbitrary token count for the fake tokenizer
D_MODEL = 8


def _synth_trajectory(seed: int) -> PoolTrajectory:
    g = torch.Generator().manual_seed(seed)
    diag: Dict[str, torch.Tensor] = {}
    diag_mask: Dict[str, torch.Tensor | None] = {}
    for name in DIAG:
        per = round(CHUNK_S * SAMPLE_RATES[name])
        total = (K_MAX + 1) * per
        channels = 6 if name != "fast_c" else 3
        diag[name] = torch.randn(channels, total, generator=g)
        # Give fast_c a mask to exercise that path
        if name == "fast_c":
            diag_mask[name] = torch.ones_like(diag[name])
        else:
            diag_mask[name] = None
    act: Dict[str, torch.Tensor] = {}
    for name in ACT:
        per = round(CHUNK_S * SAMPLE_RATES[name])
        total = K_MAX * per
        act[name] = torch.randn(4, total, generator=g)
    return PoolTrajectory(diag=diag, diag_mask=diag_mask, act=act, time_offset_s=0.0)


def _fake_tokenize(diag_input: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Stub tokeniser: returns a ``(1, N_DIAG_TOKENS, D_MODEL)`` tensor
    whose contents depend on the input so different inputs give different
    tokens."""
    pieces = []
    for name in sorted(diag_input):
        x = diag_input[name]  # (1, C, T)
        # Mean across (C, T) → scalar; broadcast into a token shape
        pieces.append(x.mean().reshape(1, 1, 1).expand(1, 1, D_MODEL))
    stacked = torch.cat(pieces, dim=1)
    # Pad or truncate to N_DIAG_TOKENS tokens
    if stacked.shape[1] < N_DIAG_TOKENS:
        pad = torch.zeros(1, N_DIAG_TOKENS - stacked.shape[1], D_MODEL)
        stacked = torch.cat([stacked, pad], dim=1)
    return stacked[:, :N_DIAG_TOKENS]


@pytest.fixture
def pool() -> TrajectoryPool:
    return TrajectoryPool(
        trajectories=[_synth_trajectory(i) for i in range(8)],
        K_max=K_MAX,
    )


@pytest.fixture
def buffer(pool: TrajectoryPool) -> ReplayBuffer:
    buf = ReplayBuffer(
        pool=pool,
        size=16,
        K_max=K_MAX,
        diagnostic_names=DIAG,
        actuator_names=ACT,
        sample_rates_hz=SAMPLE_RATES,
        chunk_duration_s=CHUNK_S,
        tokenize_initial_fn=_fake_tokenize,
        device=torch.device("cpu"),
        seed=0,
    )
    buf.initialize()
    return buf


def test_initialize_fills_buffer(buffer: ReplayBuffer) -> None:
    assert len(buffer.entries) == buffer.size
    assert all(e.rollout_step == 0 for e in buffer.entries)
    for e in buffer.entries:
        assert e.state_tokens.shape == (N_DIAG_TOKENS, D_MODEL)
        assert 0 <= e.pool_idx < len(buffer.pool)


def test_sample_shapes_and_step_indices(buffer: ReplayBuffer) -> None:
    batch_size = 4
    k_steps = 3
    batch: BufferBatch = buffer.sample(batch_size, k_steps=k_steps)

    assert batch.state_tokens.shape == (batch_size, N_DIAG_TOKENS, D_MODEL)
    assert batch.rollout_step.shape == (batch_size,)
    assert len(batch.gt_per_step) == k_steps
    assert len(batch.act_per_step) == k_steps
    assert len(batch.mask_per_step) == k_steps

    for k in range(k_steps):
        for name in DIAG:
            per = round(CHUNK_S * SAMPLE_RATES[name])
            # channels are fixed by _synth_trajectory
            expected_c = 6 if name != "fast_c" else 3
            assert batch.gt_per_step[k][name].shape == (batch_size, expected_c, per)
            if name == "fast_c":
                assert batch.mask_per_step[k][name] is not None
                assert batch.mask_per_step[k][name].shape == (batch_size, expected_c, per)
            else:
                assert batch.mask_per_step[k][name] is None
        for name in ACT:
            per = round(CHUNK_S * SAMPLE_RATES[name])
            assert batch.act_per_step[k][name].shape == (batch_size, 4, per)


def test_sample_respects_eligibility(buffer: ReplayBuffer) -> None:
    """An entry at rollout_step = K_max - 1 cannot supply 2 future steps.
    Setting all entries to K_max-1 and requesting k_steps=2 must trigger the
    refresh path, which repopulates fresh entries at step 0.
    """
    for e in buffer.entries:
        e.rollout_step = K_MAX - 1
    batch = buffer.sample(batch_size=4, k_steps=2)
    # After refresh, all sampled entries are at rollout_step=0 (fresh).
    assert (batch.rollout_step == 0).all()


def test_update_advances_and_detaches(buffer: ReplayBuffer) -> None:
    batch = buffer.sample(batch_size=4, k_steps=2)
    # Make new tokens that require grad; update() must detach them before
    # storing.
    new_tokens = torch.randn(
        4, N_DIAG_TOKENS, D_MODEL, requires_grad=True
    )
    buffer.update(batch.entries, new_tokens, advance_by=2)
    for entry in batch.entries:
        assert not entry.state_tokens.requires_grad
        # rollout_step was 0 in fresh fixture → now 2 (< K_max=5), still alive
        assert entry.rollout_step == 2


def test_update_evicts_at_K_max(buffer: ReplayBuffer) -> None:
    """Entries whose advance would hit K_max are evicted + refilled."""
    # Force entries to step K_max - 1, then advance by 1.
    for e in buffer.entries:
        e.rollout_step = K_MAX - 1

    entries_to_update = buffer.entries[:4]
    new_tokens = torch.randn(4, N_DIAG_TOKENS, D_MODEL)
    buffer.update(entries_to_update, new_tokens, advance_by=1)
    # Buffer size preserved.
    assert len(buffer.entries) == buffer.size
    # The 4 entries we updated should be gone — replaced by fresh
    # rollout_step=0 entries. The original `entries_to_update` objects are
    # still references, but they're no longer in the buffer.
    for e in entries_to_update:
        assert e not in buffer.entries


def test_periodic_refresh_preserves_size(buffer: ReplayBuffer) -> None:
    original = {id(e) for e in buffer.entries}
    buffer.periodic_refresh(fraction=0.5)
    assert len(buffer.entries) == buffer.size
    new_ids = {id(e) for e in buffer.entries}
    # At least some old entries replaced.
    assert len(original & new_ids) < buffer.size


def test_act_window_indexing_matches_rollout_step(buffer: ReplayBuffer) -> None:
    """Actuator for pushforward step k of a buffer entry at rollout_step=r
    must come from act[r + k] (i.e. actuator driving the transition to
    window r + k + 1). Verify by constructing a trajectory with synthetic
    integer markers in each window and checking the sampled slices.
    """
    # Build a deterministic trajectory where act_a[0, :, window_idx * per]
    # encodes the window_idx in the first sample of each channel.
    per = round(CHUNK_S * SAMPLE_RATES["act_a"])
    n_channels = 4
    marker = torch.zeros(n_channels, K_MAX * per)
    for w in range(K_MAX):
        # Fill window ``w`` with the value ``w + 1`` (so act[0] = value 1,
        # act[1] = value 2, etc — matches "actuator driving step w+1").
        marker[:, w * per : (w + 1) * per] = float(w + 1)
    traj = _synth_trajectory(99)
    traj.act["act_a"] = marker
    buffer.pool.replace(0, traj)

    # Force the first entry to use pool_idx=0 at rollout_step=2.
    buffer.entries[0].pool_idx = 0
    buffer.entries[0].rollout_step = 2

    # Hand-pick only that entry into a batch of 1.
    target = buffer.entries[0]
    # Manually construct a minimal batch.
    class _OneShotBuf:
        def __init__(self, parent: ReplayBuffer, e):
            self.p = parent
            self.e = e

        def sample_one(self, k_steps: int) -> BufferBatch:
            self.p.entries = [self.e]
            return self.p.sample(1, k_steps)

    batch = _OneShotBuf(buffer, target).sample_one(k_steps=2)
    # First pushforward step should use act[rollout_step + 0] = act[2] → value 3
    assert batch.act_per_step[0]["act_a"].unique().tolist() == [3.0]
    # Second step should use act[rollout_step + 1] = act[3] → value 4
    assert batch.act_per_step[1]["act_a"].unique().tolist() == [4.0]
