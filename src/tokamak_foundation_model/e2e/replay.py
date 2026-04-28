"""Lightweight replay buffer for Stage 3 long-rollout training.

Design (``ResearchPlan.MD`` §4.3, with a memory-budget-aware simplification):

  - :class:`TrajectoryPool` preloads a small number of ``(K_max + 1)``-window
    trajectories from the dataset (~200 of them, ~4 GB total host RAM).
    Each trajectory carries diagnostic signals, diagnostic masks, and
    actuator signals spanning ``(K_max + 1) · 50 ms``.
  - :class:`ReplayBuffer` holds up to ``buffer_size`` entries; each entry is
    just ``(pool_idx, rollout_step, state_tokens)``. Ground-truth and
    actuator context for the next step is looked up lazily from the pool —
    that's the lightweight part. Buffer entries advance by ``k_steps``
    rollout steps at a time (matching the pushforward curriculum) and are
    evicted once ``rollout_step >= K_max`` or refresh is triggered.

The plan's 50k-entry version keeps entire trajectories per entry (~40 GB).
This lightweight version keeps only one copy per trajectory (shared by many
buffer entries at different rollout depths) and a small ``state_tokens``
tensor per entry. The behavioural property the plan cares about — training
on *model-generated* states — is preserved since ``state_tokens`` is always
the model's own drifted token output.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from .model import E2EFoundationModel


def _samples_per_step(sample_rate_hz: float, chunk_duration_s: float) -> int:
    return round(chunk_duration_s * sample_rate_hz)


@dataclass
class PoolTrajectory:
    """One ``(K_max + 1)``-window sample held in memory.

    Attributes
    ----------
    diag
        ``name → (C, (K_max + 1) * samples_per_step[name])`` tensors.
    diag_mask
        ``name → same shape as diag[name]`` or ``None`` for modalities with
        no mask. Float 0/1 values.
    act
        ``name → (C, K_max * samples_per_step[name])`` tensors covering the
        actuator trajectory for rollout steps 1..K_max.
    time_offset_s
        Absolute time at which window 0 of this trajectory begins (used only
        for step-conditioning ``time_offset_s``).
    """

    diag: Dict[str, torch.Tensor]
    diag_mask: Dict[str, Optional[torch.Tensor]]
    act: Dict[str, torch.Tensor]
    time_offset_s: float


class TrajectoryPool:
    """Pool of :class:`PoolTrajectory` held in CPU memory.

    Refills on demand by drawing new ``(K_max + 1)``-window chunks from a
    provided generator function.
    """

    def __init__(
        self,
        trajectories: List[PoolTrajectory],
        K_max: int,
    ) -> None:
        self.trajectories = trajectories
        self.K_max = K_max

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> PoolTrajectory:
        return self.trajectories[idx]

    def replace(self, idx: int, traj: PoolTrajectory) -> None:
        self.trajectories[idx] = traj


def build_pool_from_dataset(
    dataset,
    size: int,
    K_max: int,
    diagnostic_names: Sequence[str],
    actuator_names: Sequence[str],
    sample_rates_hz: Dict[str, float],
    chunk_duration_s: float,
    collate_fn: Callable,
    seed: int = 0,
) -> TrajectoryPool:
    """Pre-load ``size`` trajectories from the dataset.

    The dataset is expected to be configured with ``prediction_mode=True``
    and ``prediction_horizon_s = K_max * chunk_duration_s``. Its
    ``__getitem__`` then returns one sample containing input (step 0) and
    target (steps 1..K_max) halves for every requested signal.

    Each trajectory is constructed by concatenating the input and target
    halves along the time axis — so ``pool[i].diag[name]`` has length
    ``(K_max + 1) * samples_per_step(name)``.
    """
    rng = random.Random(seed)
    ds_indices = rng.sample(range(len(dataset)), k=min(size, len(dataset)))
    trajectories: List[PoolTrajectory] = []
    for i, idx in enumerate(ds_indices):
        sample = dataset[idx]
        batch = collate_fn([sample])
        diag: Dict[str, torch.Tensor] = {}
        diag_mask: Dict[str, Optional[torch.Tensor]] = {}
        for name in diagnostic_names:
            input_half = batch["inputs"][name][0].float()  # drop batch dim
            target_half = batch["targets"][name][0].float()
            diag[name] = torch.cat([input_half, target_half], dim=-1).contiguous()
            mask_key = f"{name}_mask"
            if mask_key in batch["targets"]:
                mask_input = batch["inputs"][mask_key][0].float()
                mask_target = batch["targets"][mask_key][0].float()
                diag_mask[name] = torch.cat(
                    [mask_input, mask_target], dim=-1
                ).contiguous()
            else:
                diag_mask[name] = None
        act: Dict[str, torch.Tensor] = {}
        for name in actuator_names:
            # Actuators only live in the target half.
            act[name] = batch["targets"][name][0].float().contiguous()
        trajectories.append(
            PoolTrajectory(
                diag=diag,
                diag_mask=diag_mask,
                act=act,
                time_offset_s=0.0,
            )
        )
    return TrajectoryPool(trajectories, K_max=K_max)


@dataclass(eq=False)
class BufferEntry:
    """One replay-buffer entry.

    ``state_tokens`` is the current (possibly drifted) diagnostic-token
    state, detached from the graph. ``pool_idx`` references the trajectory
    providing ground-truth / actuator context; ``rollout_step`` tracks how
    far along that trajectory the entry has advanced (0 = ground-truth
    start).

    ``eq=False`` so ``__eq__`` falls back to identity. The dataclass default
    would try element-wise tensor comparison on ``state_tokens`` and raise
    from ``list.remove`` / ``in`` in :class:`ReplayBuffer`.
    """

    state_tokens: torch.Tensor
    pool_idx: int
    rollout_step: int


@dataclass
class BufferBatch:
    """What ``ReplayBuffer.sample`` returns for one training step.

    All fields are batched along dim 0 of size ``B``.

    Attributes
    ----------
    state_tokens
        ``(B, n_diag_tokens, d_model)`` — starting token state per entry.
    rollout_step
        ``(B,)`` long tensor; the step index of ``state_tokens`` within its
        trajectory. The ``k``-th push-forward step targets
        ``rollout_step + k + 1``.
    act_per_step
        Length ``k_steps``; entry ``j`` is a dict mapping actuator name →
        tensor of shape ``(B, C, samples_per_step)`` covering rollout step
        ``rollout_step + j + 1``.
    gt_per_step
        Same structure, diagnostic ground truth at the same steps.
    mask_per_step
        Same structure, diagnostic masks (float, 0/1). ``None`` for entries
        of modalities without a mask (stored as a ``None`` value in the
        dict).
    entries
        The :class:`BufferEntry` objects selected, in the same order as the
        batched tensors — needed so ``ReplayBuffer.update`` can advance them.
    """

    state_tokens: torch.Tensor
    rollout_step: torch.Tensor
    act_per_step: List[Dict[str, torch.Tensor]]
    gt_per_step: List[Dict[str, torch.Tensor]]
    mask_per_step: List[Dict[str, Optional[torch.Tensor]]]
    entries: List[BufferEntry]


class ReplayBuffer:
    """Fixed-size replay buffer of :class:`BufferEntry` backed by a :class:`TrajectoryPool`.

    Parameters
    ----------
    pool
        Trajectory pool providing ground-truth context.
    size
        Number of entries held. Typical: 10000.
    K_max
        Maximum rollout step after which an entry is evicted.
    diagnostic_names, actuator_names, sample_rates_hz, chunk_duration_s
        Windowing metadata needed to slice pool trajectories per rollout
        step.
    tokenize_initial_fn
        Callable ``diag_input → state_tokens`` used to produce the initial
        state tokens when a fresh entry is added. Typically
        ``lambda d: model.tokenize(d, act_zero)[:, :n_diag]`` but the
        buffer is agnostic — provide any function that turns a diag input
        dict into a ``(n_diag_tokens, d_model)`` tensor.
    device
        Device onto which batched tensors are moved when ``sample`` is
        called. Entry ``state_tokens`` stays wherever the update puts it.
    seed
        RNG seed for deterministic sampling.
    """

    def __init__(
        self,
        pool: TrajectoryPool,
        size: int,
        K_max: int,
        diagnostic_names: Sequence[str],
        actuator_names: Sequence[str],
        sample_rates_hz: Dict[str, float],
        chunk_duration_s: float,
        tokenize_initial_fn: Callable[[Dict[str, torch.Tensor]], torch.Tensor],
        device: torch.device,
        seed: int = 0,
    ) -> None:
        self.pool = pool
        self.size = size
        self.K_max = K_max
        self.diagnostic_names = list(diagnostic_names)
        self.actuator_names = list(actuator_names)
        self.sample_rates_hz = dict(sample_rates_hz)
        self.chunk_duration_s = chunk_duration_s
        self.tokenize_initial_fn = tokenize_initial_fn
        self.device = device
        self.rng = random.Random(seed)
        self.entries: List[BufferEntry] = []

    # ── Life-cycle ────────────────────────────────────────────────────

    def initialize(self) -> None:
        """Populate the buffer with ``size`` fresh (rollout_step=0) entries."""
        for _ in range(self.size):
            self.entries.append(self._fresh_entry())

    def _fresh_entry(self) -> BufferEntry:
        pool_idx = self.rng.randrange(len(self.pool))
        # Initial state tokens from the tokenizer acting on window 0.
        traj = self.pool[pool_idx]
        diag_window = {
            name: self._window(traj.diag[name], name, 0).unsqueeze(0)
            for name in self.diagnostic_names
        }
        with torch.no_grad():
            state = self.tokenize_initial_fn(diag_window)
        return BufferEntry(
            state_tokens=state.squeeze(0).detach().cpu(),
            pool_idx=pool_idx,
            rollout_step=0,
        )

    def periodic_refresh(self, fraction: float) -> None:
        """Evict ``fraction`` of entries (uniformly at random) and refill."""
        n_evict = int(fraction * len(self.entries))
        if n_evict <= 0:
            return
        evict_idxs = self.rng.sample(range(len(self.entries)), n_evict)
        for i in sorted(evict_idxs, reverse=True):
            del self.entries[i]
        for _ in range(n_evict):
            self.entries.append(self._fresh_entry())

    # ── Sampling + update ─────────────────────────────────────────────

    def _window(
        self, tensor: torch.Tensor, name: str, window_index: int
    ) -> torch.Tensor:
        """Slice the ``window_index``-th 50 ms window from a pool tensor."""
        per = _samples_per_step(
            self.sample_rates_hz[name], self.chunk_duration_s
        )
        start = window_index * per
        return tensor[..., start : start + per]

    def sample(self, batch_size: int, k_steps: int) -> BufferBatch:
        """Return a batch of entries + their next ``k_steps`` of context.

        Only entries whose ``rollout_step + k_steps <= K_max`` are eligible
        (we need enough future context to cover the pushforward chain). If
        fewer than ``batch_size`` are eligible, we refresh and resample.
        """

        def _eligible() -> List[BufferEntry]:
            return [e for e in self.entries if e.rollout_step + k_steps <= self.K_max]

        eligible = _eligible()
        if len(eligible) < batch_size:
            self.periodic_refresh(fraction=1.0)
            eligible = _eligible()
        selected = self.rng.sample(eligible, batch_size)

        state_tokens = torch.stack([e.state_tokens for e in selected]).to(self.device)
        rollout_step = torch.tensor(
            [e.rollout_step for e in selected],
            dtype=torch.long,
            device=self.device,
        )
        gt_per_step: List[Dict[str, torch.Tensor]] = []
        mask_per_step: List[Dict[str, Optional[torch.Tensor]]] = []
        act_per_step: List[Dict[str, torch.Tensor]] = []
        for k in range(k_steps):
            gt_k: Dict[str, torch.Tensor] = {}
            mk_k: Dict[str, Optional[torch.Tensor]] = {}
            act_k: Dict[str, torch.Tensor] = {}
            for name in self.diagnostic_names:
                slices = []
                mask_slices: List[Optional[torch.Tensor]] = []
                for e in selected:
                    traj = self.pool[e.pool_idx]
                    window_idx = e.rollout_step + k + 1
                    slices.append(self._window(traj.diag[name], name, window_idx))
                    full_mask = traj.diag_mask[name]
                    if full_mask is not None:
                        mask_slices.append(
                            self._window(full_mask, name, window_idx)
                        )
                    else:
                        mask_slices.append(None)
                gt_k[name] = torch.stack(slices).to(self.device)
                if all(m is None for m in mask_slices):
                    mk_k[name] = None
                else:
                    # A modality either has a mask consistently across the
                    # pool or not — mixed case shouldn't arise. If it does,
                    # fall back to all-ones where None.
                    filled = [
                        m if m is not None else torch.ones_like(slices[j])
                        for j, m in enumerate(mask_slices)
                    ]
                    mk_k[name] = torch.stack(filled).to(self.device)
            for name in self.actuator_names:
                slices = []
                for e in selected:
                    traj = self.pool[e.pool_idx]
                    # Actuator arrays cover steps 1..K_max — i.e. index 0
                    # of act[name] is the step-1 window. For a buffer entry
                    # at rollout_step=r, the k-th pushforward step wants the
                    # actuator for window (r + k + 1) — stored at act index
                    # (r + k).
                    act_window_idx = e.rollout_step + k
                    slices.append(self._window(traj.act[name], name, act_window_idx))
                act_k[name] = torch.stack(slices).to(self.device)
            gt_per_step.append(gt_k)
            mask_per_step.append(mk_k)
            act_per_step.append(act_k)

        return BufferBatch(
            state_tokens=state_tokens,
            rollout_step=rollout_step,
            act_per_step=act_per_step,
            gt_per_step=gt_per_step,
            mask_per_step=mask_per_step,
            entries=selected,
        )

    def update(
        self,
        entries: List[BufferEntry],
        new_state_tokens: torch.Tensor,
        advance_by: int,
    ) -> None:
        """Write the model's new predictions back and advance rollout step.

        ``new_state_tokens`` has shape ``(B, n_diag_tokens, d_model)`` and is
        detached + moved to CPU before storage. Entries whose advanced
        rollout step exceeds ``K_max`` are evicted and replaced with a fresh
        ground-truth-initialised entry so the buffer size stays constant.
        """
        detached = new_state_tokens.detach().cpu()
        for i, entry in enumerate(entries):
            entry.state_tokens = detached[i].clone()
            entry.rollout_step += advance_by
            if entry.rollout_step >= self.K_max:
                # Evict + replace.
                try:
                    self.entries.remove(entry)
                except ValueError:
                    pass  # already removed — shouldn't happen but be defensive
                self.entries.append(self._fresh_entry())
