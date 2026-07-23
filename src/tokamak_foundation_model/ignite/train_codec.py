"""IGNITE Phase-A **streaming, DDP-capable** production codec trainer.

The gate-spike (:mod:`ignite.spike`) pre-builds a FIXED batch pool via
``spike.real_ece_batches``, which opens shots SEQUENTIALLY (~19 min for 40 shots) and
caps around 40 shots. That does not scale to the thousands of shots we want a production
codec to see.

This module replaces the fixed pool with the **production data pipeline** — it reuses
:class:`~tokamak_foundation_model.data.multi_file_dataset.TokamakMultiFileDataset`, the
exact same class the main foundation-model trainer streams 7878 shots through under DDP.
That class already implements everything the old hand-rolled ``StreamingSpectroPairs``
re-invented: a global-idx → ``(file_idx, chunk_idx)`` map via cumulative-length binary
search, a per-worker LRU HDF5 file-handle cache (with correct ``__getstate__`` /
``__setstate__`` pickling into DataLoader workers), a length-cache sidecar, and a
``make_dataloader`` factory.

Design
------
* **Data**: :class:`CodecPairDataset` is a THIN subclass of ``TokamakMultiFileDataset``.
  It reuses the parent's idx-mapping + LRU file handles + length cache + num_workers
  machinery UNCHANGED and overrides ONLY the per-item transform hook
  (:meth:`_getitem_standard`) to return the codec's ``(spec_a, spec_b)`` δ-shift log-power
  pair for ONE spectro modality (ece / co2 / bes / mhr) instead of the processed
  multi-modal training dict. The parent's ``__getitem__`` still performs the binary-search
  index map and sets ``self.h5_file = self._get_file_handle(file_idx)`` before dispatching
  to our ``_getitem_standard(chunk_idx)`` — so the file-handle safety under
  ``num_workers > 0`` is inherited verbatim (each worker owns its own object copy + LRU).
* **Training**: reuses :func:`ignite.spike.codec_train_step` (the SINGLE per-step
  generator/discriminator alternation shared with the spike) so the training math is
  identical. The codec is DDP-wrapped through a thin :class:`_GenLossAdapter` whose
  ``forward`` dispatches to ``generator_losses`` — this makes DDP's autograd reducer hook
  the generator loss (the discriminator has its own optimizer / graph and is wrapped
  separately).
* **Gate + best-ckpt**: periodically streams a held-out eval set (disjoint shots) and
  computes the §4.4 gate via :func:`ignite.spike.compute_gate`, selecting the best
  checkpoint by :func:`ignite.spike.gate_score` — exactly as the spike does. Rank-0 only
  logs + writes checkpoints.

Reuse boundary (§7): only ``torch`` + sibling ``ignite`` modules + the read-only data
pipeline (``data.data_loader`` / ``data.multi_file_dataset``). No FAITH *model* code.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from . import data, spike
from .codec import SpectroCodec
from .config import CHUNK_S, SpectroCodecConfig
from .discriminator import FreqAwarePatchGAN

# spectro modalities the codec can train on. All are apply_stft=True + target_fs=500e3
# (== STFT_FS) in TokamakH5Dataset.SIGNAL_CONFIGS, so their raw windows STFT to the same
# grid data.log_power_stft expects. Channel counts (after channels_to_use) are read from
# the loader's SignalConfig at runtime — NOT hard-coded here.
SPECTRO_MODALITIES = ("ece", "co2", "bes", "mhr")

DEFAULT_DATA_DIR = spike.DEFAULT_DATA_DIR


def _import_multifile():
    """Lazily import the production multi-file dataset + samplers.

    Imported lazily so that ``import ignite.train_codec`` for module-level helpers
    (``modality_channels`` / the CLI parser) does not eagerly pull in the heavier data
    stack; the classes are needed only when a dataset/loader is actually constructed.
    """
    from tokamak_foundation_model.data.multi_file_dataset import (
        DistributedTwoLevelSampler,
        TokamakMultiFileDataset,
        TwoLevelSampler,
    )

    return TokamakMultiFileDataset, TwoLevelSampler, DistributedTwoLevelSampler


# Bind the production dataset + samplers at import time (cheap; the module is already a
# dependency of the trainer). CodecPairDataset subclasses TokamakMultiFileDataset directly.
(
    TokamakMultiFileDataset,
    TwoLevelSampler,
    DistributedTwoLevelSampler,
) = _import_multifile()


# ------------------------------------------------------------------------------------- #
# modality channel-count helper (read-only against the loader config)
# ------------------------------------------------------------------------------------- #
def modality_channels(modality: str) -> int:
    """Channels actually loaded for ``modality`` after ``channels_to_use``.

    Reads ``TokamakH5Dataset.SIGNAL_CONFIGS`` (class-level, read-only). Mirrors
    ``spike._num_ece_channels`` but for any spectro modality.
    """
    from tokamak_foundation_model.data.data_loader import TokamakH5Dataset

    cfg_sig = next(
        (c for c in TokamakH5Dataset.SIGNAL_CONFIGS if c.name == modality), None
    )
    if cfg_sig is None:
        raise ValueError(f"unknown modality {modality!r}; not in SIGNAL_CONFIGS")
    if not cfg_sig.apply_stft:
        raise ValueError(f"modality {modality!r} is not a spectro (apply_stft) signal")
    if cfg_sig.channels_to_use is not None:
        return len(range(*cfg_sig.channels_to_use.indices(cfg_sig.num_channels)))
    return cfg_sig.num_channels


# ------------------------------------------------------------------------------------- #
# δ-shift-pair dataset — a THIN subclass of the production TokamakMultiFileDataset
# ------------------------------------------------------------------------------------- #
def _shot_paths(shots: Sequence[Union[str, int]], data_dir: Union[str, Path]) -> List[Path]:
    """Resolve ``shots`` to existing ``{shot}_processed.h5`` paths under ``data_dir``.

    Missing files are dropped (the parent dataset would give them length 0 anyway); the
    order of the surviving shots is preserved so ranks derive a deterministic file list.
    """
    d = Path(data_dir)
    paths: List[Path] = []
    for s in shots:
        p = d / f"{s}_processed.h5"
        if p.exists():
            paths.append(p)
    return paths


class CodecPairDataset(TokamakMultiFileDataset):
    """``(spec_a, spec_b)`` δ-shift log-power pairs for ONE spectro modality.

    A **thin** subclass of :class:`~tokamak_foundation_model.data.multi_file_dataset.\
TokamakMultiFileDataset`. It re-uses the parent's production streaming machinery WHOLESALE
    — the global-idx → ``(file_idx, chunk_idx)`` binary-search map, the per-worker LRU
    HDF5 file-handle cache (with the correct ``__getstate__``/``__setstate__`` pickling into
    DataLoader workers), the length-cache sidecar, and the ``num_workers`` plumbing — and
    overrides ONLY the per-item transform hook :meth:`_getitem_standard`.

    Why ``_getitem_standard`` is the right hook
    -------------------------------------------
    The parent's :meth:`TokamakMultiFileDataset.__getitem__` does, for a global ``idx``:
    (1) a binary-search of the cumulative-length array to get ``pos``; (2) ``file_idx =
    self._valid_indices[pos]`` and ``chunk_idx = idx - cumulative_lengths[pos]``; (3)
    ``self.h5_file = self._get_file_handle(file_idx)`` (per-worker LRU handle); then (4)
    ``return self._getitem_standard(chunk_idx)`` — our override.

    So by overriding *only* ``_getitem_standard`` we inherit the binary-search index map,
    the LRU file-handle acquisition, and the num_workers file-handle safety **verbatim** —
    we do NOT re-implement any of it. When our override runs, ``self.h5_file`` is already the
    correct open handle for this item's shot (owned by this worker's own object copy), and
    ``chunk_idx`` is the within-shot window index.

    Windowing reuse
    ---------------
    The parent's :meth:`TokamakH5Dataset._getitem_standard` computes the window start as
    ``t_start = warmup_s + chunk_idx * step_size_s``. We reuse that EXACT formula (see
    :meth:`_getitem_standard` below). We construct the dataset with
    ``chunk_duration_s = CHUNK_S + span_tail_s`` (the FULL δ-extended span) so the parent's
    length computation (``floor((duration - chunk_duration_s) / step_size_s) + 1``) already
    guarantees the extended window ``[t_start, t_start + CHUNK_S + δ_max]`` fits inside the
    shot's real data for every ``chunk_idx`` — no per-item overrun guard needed for the map.
    ``step_size_s = CHUNK_S`` keeps successive windows 50 ms apart (non-overlapping cores).

    δ-pair construction
    -------------------
    ``_getitem_standard`` loads the RAW extended window for the modality's ``SignalConfig``
    over ``[t_start, t_start + span_s]`` via the parent's
    :meth:`TokamakH5Dataset._load_signal_raw` (already resampled to ``target_fs == STFT_FS``,
    zero-/NaN-padded), then builds the pair with :func:`ignite.data.shift_pair_windows`
    (raw δ-shift + re-STFT to log-power). Degenerate windows (all-NaN / near-flat / past the
    real data end) are guarded exactly as the spike's ``_load_raw_span`` does, and replaced by
    a re-draw from a nearby valid chunk so every item is a real, finite pair.

    Parameters
    ----------
    modality : str
        One of :data:`SPECTRO_MODALITIES`.
    shots : sequence of shot ids (str/int)
        Shots to stream windows from (typically thousands). Resolved to
        ``{shot}_processed.h5`` under ``data_dir``; missing files are dropped.
    cfg : SpectroCodecConfig
        Provides ``freq_bins`` / ``time_frames`` / ``consistency_delta_ms``. ``cfg`` is NOT
        mutated — the caller sets ``cfg.channels`` (use :func:`modality_channels`).
    data_dir : path
        Directory of ``{shot}_processed.h5`` files.
    delta_ms_range : (lo, hi) or None
        δ draw range in ms; defaults to ``cfg.consistency_delta_ms``.
    t0_start : float
        Earliest window start (s), forwarded to the parent as ``warmup_s`` (default 1.0,
        skips ramp-up per the windowing convention).
    min_std : float
        Minimum raw std for a window to count as non-degenerate.
    seed : int
        Base RNG seed; combined with the item ``idx`` so the δ draw / re-draw is
        deterministic per item (reproducible across workers and resumes).
    lengths_cache_path : path or None
        Forwarded to the parent's length-cache sidecar (skip re-scanning shot lengths).
    max_open_files : int
        Forwarded to the parent's per-worker LRU file-handle cap.
    max_tries : int
        Max nearby-chunk re-draws before falling back to the last finite pair for a bad item.
    """

    def __init__(
        self,
        modality: str,
        shots: Sequence[Union[str, int]],
        cfg: SpectroCodecConfig,
        data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
        *,
        delta_ms_range: Optional[Tuple[float, float]] = None,
        t0_start: float = 1.0,
        min_std: float = 1e-3,
        seed: int = 0,
        lengths_cache_path: Optional[Union[str, Path]] = None,
        max_open_files: int = 512,
        max_tries: int = 8,
    ) -> None:
        if modality not in SPECTRO_MODALITIES:
            raise ValueError(f"modality {modality!r} not in {SPECTRO_MODALITIES}")

        self.modality = modality
        self.codec_cfg = cfg
        self.delta_ms_range = delta_ms_range or tuple(cfg.consistency_delta_ms)
        self.min_std = float(min_std)
        self.pair_seed = int(seed)
        self.max_tries = int(max_tries)

        # δ-extended span the STFT pair needs: A covers [t0, t0+CHUNK_S]; B is shifted by up
        # to δ_max, ending at t0 + CHUNK_S + δ_max. Make the parent reserve room for the WHOLE
        # span by using it as chunk_duration_s, while step_size_s stays at CHUNK_S so windows
        # advance 50 ms at a time (non-overlapping cores).
        self.span_tail_s = float(self.delta_ms_range[1]) / 1000.0
        span_s = CHUNK_S + self.span_tail_s

        paths = _shot_paths(shots, data_dir)
        if not paths:
            raise ValueError(
                f"CodecPairDataset: no {{shot}}_processed.h5 files found under {data_dir} "
                f"for the {len(list(shots))} requested shots."
            )

        # Reuse the parent's full production machinery: idx-map + LRU handles + length cache.
        # chunk_duration_s = full δ-span so windows fit; step_size_s = CHUNK_S; warmup_s =
        # t0_start (the parent's _getitem_standard uses warmup_s + idx*step_size_s as t_start).
        super().__init__(
            hdf5_paths=paths,
            chunk_duration_s=span_s,
            step_size_s=CHUNK_S,
            warmup_s=float(t0_start),
            input_signals=[modality],
            target_signals=[modality],
            prediction_mode=False,
            lengths_cache_path=lengths_cache_path,
            max_open_files=max_open_files,
        )
        # The parent's __getitem__ dispatches on self.signal_configs; keep only ours so
        # _load_signal_raw / duration reflect the codec modality (harmless for span math).
        self._cfg_sig = next(c for c in self.signal_configs if c.name == modality)

    # -- the ONLY overridden hook: per-item δ-pair transform -------------------------- #
    def _getitem_standard(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        """Return one ``(spec_a, spec_b)`` δ-shift log-power pair for window ``idx``.

        Called by the parent's ``__getitem__`` AFTER it has (a) binary-search-mapped the
        global index to ``(file_idx, chunk_idx)`` and (b) set ``self.h5_file`` to this shot's
        per-worker LRU handle. Here ``idx`` is the within-shot ``chunk_idx``.

        Reuses the parent's windowing formula (``t_start = warmup_s + idx * step_size_s``)
        and the parent's :meth:`_load_signal_raw` to pull the RAW δ-extended window, then
        builds the pair via :func:`ignite.data.shift_pair_windows`.
        """
        pair = self._build_pair(idx)
        if pair is not None:
            return pair
        # Degenerate window (padded tail / all-flat / NaN): re-draw from nearby chunks so we
        # never return garbage. Bounded to this shot's chunk range.
        n_chunks = self._chunks_in_current_shot(idx)
        gen = torch.Generator().manual_seed(self.pair_seed + 100_003 * int(idx) + 7)
        for _ in range(self.max_tries):
            alt = int(torch.randint(0, max(1, n_chunks), (1,), generator=gen).item())
            pair = self._build_pair(alt)
            if pair is not None:
                return pair
        # Last resort: an eps-floor "silent" pair (finite, non-NaN). Extremely rare — only if
        # a whole shot is degenerate; the batch/DDP average still moves.
        C = self._num_channels()
        floor = float(math.log10(1e-10))
        z = torch.full((C, self.codec_cfg.freq_bins, self.codec_cfg.time_frames), floor)
        return z, z.clone()

    # -- helpers (all reuse parent state; no re-implementation of the index map) ------- #
    def _build_pair(self, chunk_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Build the δ-pair for a within-shot ``chunk_idx``; None if the window is degenerate.

        ``t_start`` reuses the parent's EXACT formula. The extended span is loaded via the
        parent's ``_load_signal_raw`` (already at ``target_fs == STFT_FS``); guards mirror
        :func:`ignite.spike._load_raw_span`.
        """
        step = getattr(self, "step_size_s", self.chunk_duration_s)
        warmup = getattr(self, "warmup_s", 0.0)
        t_start = warmup + chunk_idx * step          # parent windowing, reused verbatim
        span_s = CHUNK_S + self.span_tail_s
        t_end = t_start + span_s

        raw, valid_len, _nan = self._load_signal_raw(
            self.h5_file, self._cfg_sig, t_start, t_end
        )
        need = round(span_s * self._cfg_sig.target_fs)
        if valid_len < need:
            return None                              # runs into padded / missing tail
        if not torch.isfinite(raw).all():
            return None
        if float(raw.std()) < self.min_std:
            return None                              # all-zero / flat window

        lo, hi = self.delta_ms_range
        gen = torch.Generator().manual_seed(self.pair_seed + 1_000_003 * int(chunk_idx) + 1)
        d = float(lo + (hi - lo) * float(torch.rand((), generator=gen)))
        # `raw` already starts at t_start -> rebase t0 to 0.0 for the pair slicer.
        spec_a, spec_b = data.shift_pair_windows(
            raw, t0=0.0, cfg=self.codec_cfg, delta_ms=d
        )
        return spec_a, spec_b

    def _chunks_in_current_shot(self, chunk_idx: int) -> int:
        """Number of chunks in the shot currently pinned on ``self.h5_file`` (best-effort).

        Used only to bound the re-draw range. Falls back to ``chunk_idx + 1`` if the shot
        length can't be recovered (never smaller than the requested chunk).
        """
        try:
            xd = None
            for key_path in self._cfg_sig.hdf5_keys:
                grp = self.h5_file
                ok = True
                for part in key_path.split("/"):
                    if part in grp:
                        grp = grp[part]
                    else:
                        ok = False
                        break
                if ok and "xdata" in grp:
                    xd = grp["xdata"]
                    break
            if xd is None or xd.shape[0] < 2:
                return chunk_idx + 1
            real_end = float(xd[-1])
            step = getattr(self, "step_size_s", self.chunk_duration_s)
            warmup = getattr(self, "warmup_s", 0.0)
            span_s = CHUNK_S + self.span_tail_s
            n = int(math.floor((real_end - warmup - span_s) / step)) + 1
            return max(1, n)
        except Exception:
            return chunk_idx + 1

    def _num_channels(self) -> int:
        cfg_sig = self._cfg_sig
        if cfg_sig.channels_to_use is not None:
            return len(range(*cfg_sig.channels_to_use.indices(cfg_sig.num_channels)))
        return cfg_sig.num_channels


def _pair_collate(
    batch: List[Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stack a list of ``(spec_a, spec_b)`` pairs into ``(B, C, F, T)`` batched tensors."""
    spec_a = torch.stack([a for a, _ in batch], dim=0)
    spec_b = torch.stack([b for _, b in batch], dim=0)
    return spec_a, spec_b


def make_codec_loader(
    dataset: CodecPairDataset,
    *,
    batch_size: int,
    num_workers: int,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    """Build a δ-pair DataLoader over a :class:`CodecPairDataset`, reusing the parent's
    file-locality sampler.

    Sampler choice mirrors the main foundation-model trainer:
      * DDP (``world_size > 1``): :class:`DistributedTwoLevelSampler` — file-level sharding
        with sequential intra-file iteration, which keeps the parent's per-worker LRU
        file-handle cache warm (the DDP analogue of ``DistributedSampler`` tuned for this
        dataset; see its docstring).
      * single-process: :class:`TwoLevelSampler` — the same locality-preserving sampler
        ``make_dataloader`` uses.

    The collate stacks ``(spec_a, spec_b)`` items into ``(B, C, F, T)`` batches (the codec's
    δ-pair contract), rather than the parent's multi-modal ``collate_fn``.
    """
    if world_size > 1:
        sampler: torch.utils.data.Sampler = DistributedTwoLevelSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, seed=seed,
        )
    else:
        sampler = TwoLevelSampler(dataset, shuffle=shuffle)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=_pair_collate,
        pin_memory=False,           # pairs are small; pinning adds little and copies RAM
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True,             # equal per-rank step count under DDP
    )


def _epoch_cycler(loader: DataLoader):
    """Yield batches from ``loader`` forever, re-iterating (a new epoch) when exhausted.

    The production dataset/loader is a FINITE map-style epoch (one pass over the shot
    windows), while the trainer runs a fixed ``steps`` budget that may span many epochs. This
    turns the finite loader into an endless batch stream. ``set_epoch`` is called on the
    sampler each epoch (mirrors the main trainer) so DDP shuffling advances across epochs.
    """
    sampler = getattr(loader, "sampler", None)
    epoch = 0
    while True:
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


# ------------------------------------------------------------------------------------- #
# DDP generator-loss adapter (routes generator_losses through the DDP wrapper)
# ------------------------------------------------------------------------------------- #
class _GenLossAdapter(torch.nn.Module):
    """Thin wrapper whose ``forward`` dispatches to ``SpectroCodec.generator_losses``.

    DDP's autograd reducer only hooks the parameters used in the module's ``forward``. The
    codec's generator step is ``codec.generator_losses(...)``, not ``codec.forward(x)``, so
    to make DDP correctly all-reduce the generator gradients we wrap the codec in this
    adapter and DDP-wrap the ADAPTER. The training step then calls the DDP-wrapped adapter,
    whose ``forward`` runs ``generator_losses`` — so every codec parameter touched there is
    seen by the reducer.

    ``self.codec`` is a registered submodule, so ``adapter.codec`` is the underlying
    :class:`SpectroCodec` (used for the detached discriminator recon + gate eval + EMA +
    state_dict, which must NOT go through the DDP graph).
    """

    def __init__(self, codec: SpectroCodec) -> None:
        super().__init__()
        self.codec = codec

    def forward(
        self,
        spec_a: torch.Tensor,
        spec_b: torch.Tensor,
        disc: torch.nn.Module,
        cfg: SpectroCodecConfig,
        step: int,
    ) -> Dict[str, torch.Tensor]:
        return self.codec.generator_losses(spec_a, spec_b, disc, cfg, step=step)


def _ddp_codec_train_step(
    gen_module: torch.nn.Module,        # DDP-wrapped _GenLossAdapter (or bare adapter)
    codec: SpectroCodec,                # the underlying raw codec
    disc: torch.nn.Module,              # DDP-wrapped discriminator (or bare)
    disc_raw: FreqAwarePatchGAN,        # the underlying raw discriminator
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    spec_a: torch.Tensor,
    spec_b: torch.Tensor,
    cfg: SpectroCodecConfig,
    step: int,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """DDP-aware analogue of :func:`spike.codec_train_step`.

    Same math as :func:`spike.codec_train_step` but routes the GENERATOR forward through the
    DDP-wrapped adapter (so the codec-gradient all-reduce is hooked) and the DISCRIMINATOR
    forward through its DDP wrapper. The discriminator inside ``generator_losses`` is called
    with ``disc_raw`` (no grad on D during the G step, matching the spike). ``codec.forward``
    for the detached D-step recon uses the raw codec (no DDP graph needed — it's under
    ``no_grad``).
    """
    from .losses import discriminator_loss

    codec.train()
    disc_raw.train()

    # ---- generator (codec) step: forward through the DDP-wrapped adapter ----
    opt_g.zero_grad(set_to_none=True)
    g_terms = gen_module(spec_a, spec_b, disc_raw, cfg, step)
    g_terms["total"].backward()
    opt_g.step()

    # ---- discriminator step (fresh detached recon; forward through DDP-wrapped disc) ----
    opt_d.zero_grad(set_to_none=True)
    with torch.no_grad():
        recon = codec.forward(spec_a)["recon"]
    d_loss = discriminator_loss(disc, spec_a, recon, cfg)
    d_loss.backward()
    opt_d.step()

    return g_terms, d_loss


# ------------------------------------------------------------------------------------- #
# held-out streamed eval set (disjoint shots) for the gate
# ------------------------------------------------------------------------------------- #
def _stream_eval_data(
    modality: str,
    eval_shots: Sequence[Union[str, int]],
    cfg: SpectroCodecConfig,
    *,
    data_dir: Union[str, Path],
    eval_batches: int,
    eval_batch_size: int,
    eval_frames: int,
    device: torch.device,
    seed: int,
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    """Materialize a small held-out gate set by streaming ``eval_shots`` on THIS rank only.

    Returns ``(eval_pairs, frame_seq)`` matching :func:`spike.compute_gate`'s contract:
    ``eval_pairs`` = list of ``(spec_a, spec_b)`` batches; ``frame_seq`` =
    ``(B, n_frames, C, F, T)`` consecutive-frame sequence. The eval set is small + fixed
    (materialized once) so gate scores are comparable across evals. Runs single-process here
    (world_size=1) so every rank has the SAME held-out data — the gate is computed on rank 0
    only, but building it uniformly keeps the code rank-agnostic.
    """
    # δ-pairs via the SAME production dataset (single-process, deterministic draw). Iterate
    # its map-style windows in order and materialize the first ``eval_batches*eval_batch_size``.
    ds = CodecPairDataset(modality, eval_shots, cfg, data_dir=data_dir, seed=seed)
    n_pairs = eval_batches * eval_batch_size
    n_avail = len(ds)
    if n_avail < n_pairs:
        raise RuntimeError(
            f"_stream_eval_data: only {n_avail} eval windows across {len(list(eval_shots))} "
            f"eval shots; need {n_pairs} (eval_batches={eval_batches} * "
            f"eval_batch_size={eval_batch_size}). Add more eval shots."
        )
    flat_pairs = [ds[i] for i in range(n_pairs)]
    eval_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for bi in range(eval_batches):
        chunk = flat_pairs[bi * eval_batch_size : (bi + 1) * eval_batch_size]
        a = torch.stack([p[0] for p in chunk], dim=0).to(device)
        b = torch.stack([p[1] for p in chunk], dim=0).to(device)
        eval_pairs.append((a, b))

    # consecutive-frame sequence for persistence / forecastability, streamed from eval shots.
    frame_seq = _stream_frame_sequence(
        modality, eval_shots, cfg,
        data_dir=data_dir, batch_size=max(2, eval_batch_size // 2),
        n_frames=eval_frames, device=device, seed=seed + 999,
    )
    return eval_pairs, frame_seq


def _stream_frame_sequence(
    modality: str,
    shots: Sequence[Union[str, int]],
    cfg: SpectroCodecConfig,
    *,
    data_dir: Union[str, Path],
    batch_size: int,
    n_frames: int,
    device: torch.device,
    seed: int,
    t0_start: float = 1.0,
    min_std: float = 1e-3,
) -> torch.Tensor:
    """Consecutive-frame ece/co2/... sequence for the gate (mirrors spike.real_frame_sequence).

    Scans ``shots`` for blocks of ``n_frames`` consecutive valid 50 ms windows; STFTs each to
    log-power. Reuses the read-only loader access + ``data.log_power_stft``. Single-process.
    """
    seq_span_s = n_frames * CHUNK_S
    gen = torch.Generator().manual_seed(seed)
    shot_list = [str(s) for s in shots]

    seqs: List[torch.Tensor] = []
    for shot in shot_list:
        if len(seqs) >= batch_size:
            break
        path = Path(data_dir) / f"{shot}_processed.h5"
        if not path.exists():
            continue
        # open via the generic per-modality path (spike._open_ece_dataset is ece-only).
        from tokamak_foundation_model.data.data_loader import TokamakH5Dataset

        ds = TokamakH5Dataset(path, input_signals=[modality], target_signals=[modality])
        ds._open_hdf5()
        cfg_sig = next(c for c in ds.signal_configs if c.name == modality)
        try:
            end_time = min(ds.duration, spike._real_end_s(ds, cfg_sig))
            t0 = t0_start
            while t0 + seq_span_s <= end_time and len(seqs) < batch_size:
                frames: List[torch.Tensor] = []
                ok = True
                for fi in range(n_frames):
                    raw = spike._load_raw_span(
                        ds, cfg_sig, t0 + fi * CHUNK_S, CHUNK_S, min_std=min_std
                    )
                    if raw is None:
                        ok = False
                        break
                    frames.append(data.log_power_stft(raw.unsqueeze(0), cfg)[0])
                if ok:
                    seqs.append(torch.stack(frames, dim=0))
                t0 += seq_span_s
        finally:
            spike._close_dataset(ds)

    if len(seqs) < batch_size:
        raise RuntimeError(
            f"_stream_frame_sequence: only {len(seqs)} valid {n_frames}-frame sequences "
            f"across {len(shot_list)} eval shots; need {batch_size}. Add more eval shots."
        )
    return torch.stack(seqs, dim=0).to(device)


# ------------------------------------------------------------------------------------- #
# DDP init (env-based; matches utils.distributed / _srun_rank_wrapper.sh contract)
# ------------------------------------------------------------------------------------- #
class _DDPState:
    """Minimal DDP context read from the env vars set by ``_srun_rank_wrapper.sh``.

    ``RANK`` / ``LOCAL_RANK`` / ``WORLD_SIZE`` (torchrun / srun), NCCL backend, one GPU per
    rank (Frontier ``--gpus-per-task=1`` masks each rank to a single visible GCD → device
    index 0). Single-process when ``WORLD_SIZE`` is unset or 1 (the CPU smoke path).
    """

    def __init__(self, backend: str = "nccl") -> None:
        ws = int(os.environ.get("WORLD_SIZE", "1"))
        if ws > 1:
            self.rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.world_size = ws
            visible = torch.cuda.device_count()
            self.device_index = self.local_rank if visible > 1 else 0
            self.distributed = True
            if not dist.is_initialized():
                dist.init_process_group(backend, rank=self.rank, world_size=self.world_size)
            if torch.cuda.is_available():
                torch.cuda.set_device(self.device_index)
        else:
            self.rank, self.local_rank, self.world_size = 0, 0, 1
            self.device_index = 0
            self.distributed = False

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda", self.device_index)
        return torch.device("cpu")

    def wrap(self, module: torch.nn.Module) -> torch.nn.Module:
        if not self.distributed:
            return module
        from torch.nn.parallel import DistributedDataParallel

        return DistributedDataParallel(
            module,
            device_ids=[self.device_index] if torch.cuda.is_available() else None,
            find_unused_parameters=False,
        )

    def barrier(self) -> None:
        if self.distributed:
            dist.barrier()

    def shutdown(self) -> None:
        if self.distributed and dist.is_initialized():
            dist.destroy_process_group()


# ------------------------------------------------------------------------------------- #
# the streaming trainer
# ------------------------------------------------------------------------------------- #
def train_codec(
    cfg: SpectroCodecConfig,
    modality: str,
    train_shots: Sequence[Union[str, int]],
    eval_shots: Sequence[Union[str, int]],
    *,
    steps: int,
    eval_every: int,
    batch_size: int,
    num_workers: int,
    data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
    lr: float = 1e-3,
    disc_lr: Optional[float] = None,
    ema: bool = False,
    ema_decay: float = 0.999,
    eval_batches: int = 8,
    eval_batch_size: int = 4,
    eval_frames: int = 6,
    out_dir: Optional[Union[str, Path]] = None,
    seed: int = 0,
    ddp: Optional[_DDPState] = None,
    resume_state: Optional[Dict[str, object]] = None,
    lengths_cache_path: Optional[Union[str, Path]] = None,
    log_fn=print,
) -> Dict[str, object]:
    """Streaming, DDP-capable codec training loop (reuses spike's per-step + gate logic).

    Wires the streaming DataLoader → :func:`_ddp_codec_train_step` (== spike's per-step math,
    DDP-hooked) → periodic gate on a held-out streamed eval set → best-ckpt by
    :func:`spike.gate_score`. Rank-0 only logs + writes ``codec_last.pt`` / ``codec_best.pt``
    / (optional) ``codec_ema.pt`` / ``gate_<step>.json``.

    Returns the final gate dict (with ``steps`` / ``global_step`` / ``best_score`` /
    ``best_step`` keys), like :func:`spike.run_spike`.
    """
    if steps < 1:
        raise ValueError("train_codec: steps must be >= 1")
    ddp = ddp if ddp is not None else _DDPState()
    device = ddp.device
    torch.manual_seed(seed + ddp.rank)

    out_path = Path(out_dir) if out_dir is not None else None
    if out_path is not None and ddp.is_main:
        out_path.mkdir(parents=True, exist_ok=True)

    # --- models ---
    codec = SpectroCodec(cfg).to(device)
    disc_raw = FreqAwarePatchGAN(cfg).to(device)

    start_step = 0
    if resume_state is not None:
        codec.load_state_dict(resume_state["codec"])
        if resume_state.get("disc") is not None:
            disc_raw.load_state_dict(resume_state["disc"])
        start_step = int(resume_state.get("step", 0))

    opt_g = torch.optim.Adam(codec.parameters(), lr=lr)
    opt_d = torch.optim.Adam(
        disc_raw.parameters(), lr=disc_lr if disc_lr is not None else lr
    )
    if resume_state is not None:
        if resume_state.get("opt_g") is not None:
            opt_g.load_state_dict(resume_state["opt_g"])
        if resume_state.get("opt_d") is not None:
            opt_d.load_state_dict(resume_state["opt_d"])

    gen_module = ddp.wrap(_GenLossAdapter(codec))
    disc = ddp.wrap(disc_raw)

    ema_shadow = spike._EMA(codec, ema_decay) if ema else None

    # --- held-out gate set (same on every rank; gate computed on rank 0) ---
    eval_pairs, frame_seq = _stream_eval_data(
        modality, eval_shots, cfg,
        data_dir=data_dir, eval_batches=eval_batches, eval_batch_size=eval_batch_size,
        eval_frames=eval_frames, device=device, seed=seed + 777,
    )

    # --- train loader over the production TokamakMultiFileDataset subclass ---
    train_ds = CodecPairDataset(
        modality, train_shots, cfg,
        data_dir=data_dir, seed=seed, lengths_cache_path=lengths_cache_path,
    )
    loader = make_codec_loader(
        train_ds,
        batch_size=batch_size, num_workers=num_workers,
        rank=ddp.rank, world_size=ddp.world_size, seed=seed, shuffle=True,
    )
    # The loader is a FINITE map-style epoch (unlike the old infinite IterableDataset). Wrap
    # it so the trainer's fixed ``steps`` budget cycles through epochs transparently.
    stream = _epoch_cycler(loader)

    best_score = float("-inf")
    best_step: Optional[int] = None
    final_gate: Dict[str, object] = {}

    for local_step in range(steps):
        step = start_step + local_step
        spec_a, spec_b = next(stream)
        spec_a = spec_a.to(device, non_blocking=True)
        spec_b = spec_b.to(device, non_blocking=True)

        g_terms, d_loss = _ddp_codec_train_step(
            gen_module, codec, disc, disc_raw, opt_g, opt_d,
            spec_a, spec_b, cfg, step=step,
        )
        if ema_shadow is not None:
            ema_shadow.update(codec)

        is_last = local_step == steps - 1
        if (step % eval_every == 0) or is_last:
            g = spike.compute_gate(codec, eval_pairs, frame_seq, cfg)
            g["g_total"] = float(g_terms["total"].detach())
            g["d_loss"] = float(d_loss.detach())
            g["adaptive_weight"] = float(g_terms["adaptive_weight"])
            g["step"] = step
            score = spike.gate_score(g, recon_floor=cfg.gate_recon_floor)
            g["score"] = score
            is_best = score > best_score
            g["is_best"] = bool(is_best)
            if is_best:
                best_score = score
                best_step = step

            if ddp.is_main:
                if log_fn is not None:
                    log_fn(
                        spike._fmt_gate(step, g)
                        + f" g_total={g['g_total']:+.4f} d_loss={g['d_loss']:.4f}"
                        + f" adv_lam={g['adaptive_weight']:.4g} score={score:+.4f}"
                    )
                if out_path is not None:
                    spike._write_gate_json(out_path, step, g)
                    spike._save_checkpoint(out_path, step, codec, opt_g, opt_d, disc_raw, cfg)
                    if is_best:
                        spike._save_best_checkpoint(
                            out_path, step, score, codec, disc_raw, cfg, g
                        )
                        if log_fn is not None:
                            log_fn(f"[train_codec] new BEST score={score:+.4f} @ step {step}")
                    if ema_shadow is not None:
                        spike._save_ema_checkpoint(out_path, step, ema_shadow.state_dict(), cfg)
            final_gate = g
        ddp.barrier()

    final_gate["steps"] = steps
    final_gate["global_step"] = start_step + steps
    final_gate["best_score"] = best_score
    final_gate["best_step"] = best_step
    return final_gate


# ------------------------------------------------------------------------------------- #
# CLI
# ------------------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tokamak_foundation_model.ignite.train_codec",
        description="IGNITE Phase-A STREAMING, DDP codec trainer (scales past the spike's "
                    "~40-shot pre-built pool to thousands of shots on the fly).",
    )
    p.add_argument("--modality", type=str, default="ece", choices=list(SPECTRO_MODALITIES),
                   help="Spectro modality to train the codec for.")
    p.add_argument("--shots", type=str, default=None,
                   help="Explicit comma-separated train shot list (overrides --n_shots).")
    p.add_argument("--n_shots", type=int, default=1000,
                   help="Number of shots to auto-discover for the train stream.")
    p.add_argument("--eval_n_shots", type=int, default=16,
                   help="Held-out shots for the gate eval set (disjoint from train).")
    p.add_argument("--steps", type=int, default=20000, help="Training/gate steps.")
    p.add_argument("--eval_every", type=int, default=1000, help="Gate/checkpoint cadence.")
    p.add_argument("--batch_size", type=int, default=8, help="Pairs per batch per rank.")
    p.add_argument("--num_workers", type=int, default=6,
                   help="DataLoader parallel workers per rank (streaming).")
    p.add_argument("--lr", type=float, default=1e-3, help="Codec (generator) Adam LR.")
    p.add_argument("--disc_lr", type=float, default=None, help="Discriminator LR (default lr).")
    p.add_argument("--ema", action="store_true", help="Maintain + write an EMA codec shadow.")
    p.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay in [0,1).")
    p.add_argument("--eval_batches", type=int, default=8, help="Held-out gate δ-pair batches.")
    p.add_argument("--eval_batch_size", type=int, default=4, help="Pairs per eval batch.")
    p.add_argument("--eval_frames", type=int, default=6,
                   help="Consecutive frames per persistence/forecastability sequence.")
    p.add_argument("--out_dir", type=str, required=True,
                   help="Output dir for gate_<step>.json / codec_{last,best,ema}.pt / summary.")
    p.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR,
                   help="Directory of {shot}_processed.h5 files.")
    p.add_argument("--lengths_cache_dir", type=str, default=None,
                   help="Directory for the per-file chunk-length sidecar cache "
                        "(codec_<modality>_lengths.pt); reuses the parent dataset's cache to "
                        "skip re-scanning shot lengths at startup. None disables caching.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", type=str, default=None,
                   help="Optional codec_last.pt to warm-start codec/disc/opt/step from.")
    p.add_argument("--entropy_weight", type=float, default=None,
                   help="Override cfg.entropy_weight (anti-collapse strength).")
    p.add_argument("--consistency_weight", type=float, default=None,
                   help="Override cfg.consistency_weight (shift-invariance strength).")
    return p


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, object]:
    args = build_arg_parser().parse_args(argv)

    ddp = _DDPState()
    device = ddp.device

    # cfg with the modality's real channel count baked in (all ranks agree).
    channels = modality_channels(args.modality)
    cfg = SpectroCodecConfig(channels=channels)
    if args.entropy_weight is not None:
        cfg.entropy_weight = float(args.entropy_weight)
    if args.consistency_weight is not None:
        cfg.consistency_weight = float(args.consistency_weight)

    # resolve the train + disjoint eval shot lists (rank 0 discovers; the list is
    # deterministic from data_dir + sort so every rank derives the same split).
    if args.shots is not None:
        all_shots = [s.strip() for s in args.shots.split(",") if s.strip()]
    else:
        all_shots = spike.discover_shots(args.data_dir)
    # eval = last eval_n_shots; train = the rest, capped to n_shots.
    eval_shots = all_shots[-args.eval_n_shots:]
    train_pool = all_shots[: -args.eval_n_shots] if args.eval_n_shots > 0 else all_shots
    train_shots = train_pool[: args.n_shots]
    if not train_shots:
        raise RuntimeError(
            f"no train shots (discovered {len(all_shots)}, eval_n_shots={args.eval_n_shots})"
        )

    resume_state = None
    if args.resume is not None:
        resume_state = torch.load(args.resume, map_location=device, weights_only=False)

    if ddp.is_main:
        print(
            f"[train_codec] modality={args.modality} channels={channels} "
            f"train_shots={len(train_shots)} eval_shots={len(eval_shots)} "
            f"world_size={ddp.world_size} batch_size={args.batch_size} "
            f"num_workers={args.num_workers} steps={args.steps} device={device}",
            flush=True,
        )

    t0 = time.time()
    final_gate = train_codec(
        cfg,
        args.modality,
        train_shots,
        eval_shots,
        steps=args.steps,
        eval_every=args.eval_every,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir,
        lr=args.lr,
        disc_lr=args.disc_lr,
        ema=args.ema,
        ema_decay=args.ema_decay,
        eval_batches=args.eval_batches,
        eval_batch_size=args.eval_batch_size,
        eval_frames=args.eval_frames,
        out_dir=args.out_dir,
        seed=args.seed,
        ddp=ddp,
        resume_state=resume_state,
        lengths_cache_path=args.lengths_cache_dir and (
            Path(args.lengths_cache_dir) / f"codec_{args.modality}_lengths.pt"
        ),
    )
    final_gate["wall_s"] = time.time() - t0

    if ddp.is_main and args.out_dir is not None:
        summary = dict(final_gate)
        summary["config"] = {
            "modality": args.modality,
            "channels": channels,
            "n_train_shots": len(train_shots),
            "n_eval_shots": len(eval_shots),
            "steps": args.steps,
            "eval_every": args.eval_every,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "world_size": ddp.world_size,
            "lr": args.lr,
            "disc_lr": args.disc_lr,
            "ema": args.ema,
            "seed": args.seed,
            "resumed_from": args.resume,
        }
        with open(Path(args.out_dir) / "summary.json", "w") as fh:
            json.dump(spike._jsonable(summary), fh, indent=2, sort_keys=True)
        print(
            f"[train_codec] DONE steps={final_gate.get('steps')} "
            f"global_step={final_gate.get('global_step')} "
            f"best_score={final_gate.get('best_score')} "
            f"best_step={final_gate.get('best_step')} wall_s={final_gate['wall_s']:.1f}",
            flush=True,
        )

    ddp.barrier()
    ddp.shutdown()
    return final_gate


if __name__ == "__main__":
    main()
