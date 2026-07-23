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

from . import data, gate, spike
from .codec import SpectroCodec
from .config import (
    CHUNK_S,
    SLOWTS_FS,
    SLOWTS_SIGNALS as _SLOWTS_SIGNALS,
    SlowTSCodecConfig,
    SpectroCodecConfig,
    VideoCodecConfig,
    slowts_patch_for,
)
from .discriminator import FreqAwarePatchGAN
from .slow_ts_codec import SlowTSCodec
from .video_codec import VideoCodec
from .video_discriminator import FramePatchGAN

# spectro modalities the codec can train on. All are apply_stft=True + target_fs=500e3
# (== STFT_FS) in TokamakH5Dataset.SIGNAL_CONFIGS, so their raw windows STFT to the same
# grid data.log_power_stft expects. Channel counts (after channels_to_use) are read from
# the loader's SignalConfig at runtime — NOT hard-coded here.
SPECTRO_MODALITIES = ("ece", "co2", "bes", "mhr")

# video (tangtv) modalities — the two divertor codecs. Each is a MovieConfig in
# TokamakH5Dataset.MOVIE_CONFIGS: tangtv_lower keeps raw camera channels [0, 2]
# (LODIV PAR-int + LODIV PERP); tangtv_upper keeps [4, 6] (UPDIV0 PERP + UPDIV0 PAR).
# Both are 2-channel after channels_to_use (read at runtime, NOT hard-coded here).
VIDEO_MODALITIES = ("tangtv_lower", "tangtv_upper")

# slow-TS modalities — the 7 smooth kinetic-profile signals (Thomson / CER / MSE). Each is a
# non-STFT SignalConfig in TokamakH5Dataset.SIGNAL_CONFIGS at target_fs=100 Hz. Thomson uses
# zero_is_missing; CER/MSE use an explicit NaN mask (see config.SLOWTS_* + data_loader). One
# codec per signal, each frozen independently (§4.3 "lightest touch").
SLOWTS_MODALITIES = tuple(_SLOWTS_SIGNALS)

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
def _channels_after_selection(num_channels: int, channels_to_use) -> int:
    """Channel count after applying a ``channels_to_use`` slice / index-list / None."""
    if channels_to_use is None:
        return num_channels
    if isinstance(channels_to_use, slice):
        return len(range(*channels_to_use.indices(num_channels)))
    return len(list(channels_to_use))


def modality_channels(modality: str) -> int:
    """Channels actually loaded for ``modality`` after ``channels_to_use``.

    Reads ``TokamakH5Dataset.SIGNAL_CONFIGS`` (spectro) or ``MOVIE_CONFIGS`` (video),
    class-level + read-only. Mirrors ``spike._num_ece_channels`` but for any codec modality.
    """
    from tokamak_foundation_model.data.data_loader import TokamakH5Dataset

    if modality in VIDEO_MODALITIES:
        mv = next(
            (m for m in TokamakH5Dataset.MOVIE_CONFIGS if m.name == modality), None
        )
        if mv is None:
            raise ValueError(f"unknown video modality {modality!r}; not in MOVIE_CONFIGS")
        return mv.channels

    cfg_sig = next(
        (c for c in TokamakH5Dataset.SIGNAL_CONFIGS if c.name == modality), None
    )
    if cfg_sig is None:
        raise ValueError(f"unknown modality {modality!r}; not in SIGNAL_CONFIGS")
    # slow-TS signals (Thomson / CER / MSE) are non-STFT raw signals; spectro signals are
    # STFT. Anything else that is non-STFT is not a codec modality we support here.
    if not cfg_sig.apply_stft and modality not in SLOWTS_MODALITIES:
        raise ValueError(
            f"modality {modality!r} is not a spectro (apply_stft) or slow-TS signal"
        )
    return _channels_after_selection(cfg_sig.num_channels, cfg_sig.channels_to_use)


def video_divertor(modality: str) -> str:
    """Map a video modality name to its divertor selector ('lower' / 'upper')."""
    if modality == "tangtv_lower":
        return "lower"
    if modality == "tangtv_upper":
        return "upper"
    raise ValueError(f"{modality!r} is not a video modality {VIDEO_MODALITIES}")


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


# ===================================================================================== #
# VIDEO (tangtv) codec — dataset + loader + gate, mirroring the spectro machinery above.
# ===================================================================================== #
class VideoCodecPairDataset(TokamakMultiFileDataset):
    """tangtv frame windows ``(frames (C,T,H,W), frame_mask (T,))`` for ONE divertor.

    A **thin** subclass of :class:`~tokamak_foundation_model.data.multi_file_dataset.\
TokamakMultiFileDataset`, EXACTLY as :class:`CodecPairDataset` is for spectro. It reuses the
    parent's production streaming machinery WHOLESALE — the global-idx → ``(file_idx,
    chunk_idx)`` binary-search map, the per-worker LRU HDF5 file-handle cache (with the
    ``__getstate__``/``__setstate__`` pickling into DataLoader workers), the length-cache
    sidecar, and the ``num_workers`` plumbing — and overrides ONLY the per-item transform hook
    :meth:`_getitem_standard`. It does **not** touch ``data_loader.py`` /
    ``multi_file_dataset.py``.

    Difference from :class:`CodecPairDataset` (spectro)
    ---------------------------------------------------
    The video codec has **no δ-shift consistency pair** (docs/IGNITE_DESIGN.md §4.3; video
    has no STFT-phase realization nuisance). So the transform returns the video **frames**
    plus a **per-frame validity mask** — NOT a pair. The parent's ``__getitem__`` still does
    the binary-search index map and sets ``self.h5_file`` before dispatching to our
    ``_getitem_standard(chunk_idx)``; here ``chunk_idx`` is the within-shot window index and
    ``self.h5_file`` is this worker's LRU handle. We reuse the parent's windowing formula
    (``t_start = warmup_s + chunk_idx * step_size_s``) and the parent's
    :meth:`TokamakH5Dataset._load_movie_raw` (already resampled to ``target_fps`` and
    zero-filled where the camera was off / NaN) — the video analogue of the spectro dataset's
    ``_load_signal_raw`` reuse.

    Parameters
    ----------
    modality : str
        One of :data:`VIDEO_MODALITIES` (``tangtv_lower`` / ``tangtv_upper``).
    shots, cfg, data_dir, seed, lengths_cache_path, max_open_files, max_tries, t0_start,
    min_std : as :class:`CodecPairDataset` (``cfg`` here is a :class:`VideoCodecConfig`).
        ``cfg`` is NOT mutated. ``min_std`` guards degenerate (flat / dead-camera) windows.
    """

    def __init__(
        self,
        modality: str,
        shots: Sequence[Union[str, int]],
        cfg: VideoCodecConfig,
        data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
        *,
        t0_start: float = 1.0,
        min_std: float = 1e-6,
        seed: int = 0,
        lengths_cache_path: Optional[Union[str, Path]] = None,
        max_open_files: int = 512,
        max_tries: int = 8,
    ) -> None:
        if modality not in VIDEO_MODALITIES:
            raise ValueError(f"modality {modality!r} not in {VIDEO_MODALITIES}")

        self.modality = modality
        self.codec_cfg = cfg
        self.min_std = float(min_std)
        self.pair_seed = int(seed)
        self.max_tries = int(max_tries)

        paths = _shot_paths(shots, data_dir)
        if not paths:
            raise ValueError(
                f"VideoCodecPairDataset: no {{shot}}_processed.h5 files found under "
                f"{data_dir} for the {len(list(shots))} requested shots."
            )

        # A video window is a single CHUNK_S (50 ms) clip: chunk_duration_s = step_size_s =
        # CHUNK_S (non-overlapping 50 ms windows), warmup_s = t0_start. The parent's length
        # formula floor((duration - chunk_duration_s)/step_size_s)+1 guarantees each window
        # fits inside the shot's real data. We request the movie by NAME so the parent loads
        # THIS divertor's channel set only.
        super().__init__(
            hdf5_paths=paths,
            chunk_duration_s=CHUNK_S,
            step_size_s=CHUNK_S,
            warmup_s=float(t0_start),
            input_signals=[modality],
            target_signals=[modality],
            prediction_mode=False,
            lengths_cache_path=lengths_cache_path,
            max_open_files=max_open_files,
        )
        self._cfg_movie = next(c for c in self.movie_configs if c.name == modality)

    # -- the ONLY overridden hook: per-item frame-window transform -------------------- #
    def _getitem_standard(self, idx: int):  # type: ignore[override]
        """Return one ``(frames (C,T,H,W), frame_mask (T,))`` for window ``idx``.

        Called by the parent's ``__getitem__`` AFTER it mapped the global index to
        ``(file_idx, chunk_idx)`` and set ``self.h5_file``. ``idx`` here is the within-shot
        ``chunk_idx``. Reuses the parent's windowing formula + :meth:`_load_movie_raw`.
        """
        clip = self._build_clip(idx)
        if clip is not None:
            return clip
        # Degenerate window (dead camera / all-flat): re-draw from nearby chunks.
        n_chunks = max(1, self._cumulative_lengths_span())
        gen = torch.Generator().manual_seed(self.pair_seed + 100_003 * int(idx) + 7)
        for _ in range(self.max_tries):
            alt = int(torch.randint(0, n_chunks, (1,), generator=gen).item())
            clip = self._build_clip(alt)
            if clip is not None:
                return clip
        # Last resort: a finite all-zero clip (rare; whole shot degenerate) + all-invalid mask.
        cfg = self.codec_cfg
        z = torch.zeros((cfg.channels, cfg.frames, cfg.height, cfg.width))
        m = torch.zeros(cfg.frames, dtype=torch.float32)
        return z, m

    # -- helpers (reuse parent state; no re-implementation of the index map) ---------- #
    def _build_clip(self, chunk_idx: int):
        """Build ``(frames, frame_mask)`` for a within-shot ``chunk_idx``; None if degenerate.

        ``t_start`` reuses the parent's EXACT windowing formula; the frames come from the
        parent's ``_load_movie_raw`` (already resampled to ``target_fps``, NaN→0 filled).
        A per-CHANNEL availability mask is returned by ``_load_movie_raw``; we fold it into a
        per-FRAME validity mask (a frame is valid iff ANY channel is live and the frame is
        finite + non-flat).
        """
        cfg = self.codec_cfg
        step = getattr(self, "step_size_s", self.chunk_duration_s)
        warmup = getattr(self, "warmup_s", 0.0)
        t_start = warmup + chunk_idx * step          # parent windowing, reused verbatim
        t_end = t_start + CHUNK_S

        frames, channel_valid = self._load_movie_raw(
            self.h5_file, self._cfg_movie, t_start, t_end
        )
        # frames: (C, T, H, W) at target_fps. Crop/pad T to cfg.frames defensively (a 50 ms
        # window at target_fps should already be cfg.frames; guard rounding drift).
        frames = self._fit_frames(frames)
        if not torch.isfinite(frames).all():
            return None
        if float(frames.std()) < self.min_std:
            return None                              # dead camera / flat window
        # per-frame validity: any live channel AND finite. channel_valid is (C,) bool; a frame
        # is valid if at least one channel is live (the loader zero-fills off cameras).
        any_live = bool(channel_valid.any().item())
        if not any_live:
            return None
        frame_mask = torch.ones(cfg.frames, dtype=torch.float32)
        return frames, frame_mask

    def _fit_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Crop/pad the loaded ``(C, T, H, W)`` to exactly ``(cfg.channels, cfg.frames, H, W)``.

        ``_load_movie_raw`` already resamples H×W to its ``MovieConfig`` target (120×360 for
        tangtv, == the production ``VideoCodecConfig`` H/W), and a 50 ms window at target_fps
        is already ``cfg.frames``. This guard makes the codec ``cfg`` authoritative for the
        WHOLE ``(T, H, W)`` shape — the time axis can drift a frame from rounding, and H/W is
        trilinearly resized only if it differs (a no-op in production; lets tests use a smaller
        grid). Time is right-cropped / right-padded (repeat last frame); H/W is resized via
        ``F.interpolate`` (the same trilinear-family resample the loader itself uses).
        """
        cfg = self.codec_cfg
        C, T, H, W = frames.shape
        # spatial fit (usually a no-op: loader already at cfg.height × cfg.width).
        if (H, W) != (cfg.height, cfg.width):
            frames = torch.nn.functional.interpolate(
                frames.unsqueeze(0), size=(T, cfg.height, cfg.width),
                mode="trilinear", align_corners=False,
            ).squeeze(0)
            C, T = frames.shape[0], frames.shape[1]
        # time fit.
        if T > cfg.frames:
            frames = frames[:, : cfg.frames]
        elif T < cfg.frames:
            if T == 0:
                return torch.zeros((cfg.channels, cfg.frames, cfg.height, cfg.width))
            pad = frames[:, -1:].expand(C, cfg.frames - T, cfg.height, cfg.width)
            frames = torch.cat([frames, pad], dim=1)
        return frames

    def _cumulative_lengths_span(self) -> int:
        """Best-effort chunk count for the re-draw bound (total dataset length; safe upper bound)."""
        try:
            return int(self._cumulative_lengths[-1])
        except Exception:
            return 1


def _video_collate(batch):
    """Stack ``(frames, frame_mask)`` items into ``((B,C,T,H,W), (B,T))`` batched tensors."""
    frames = torch.stack([f for f, _ in batch], dim=0)
    masks = torch.stack([m for _, m in batch], dim=0)
    return frames, masks


def make_video_loader(
    dataset: VideoCodecPairDataset,
    *,
    batch_size: int,
    num_workers: int,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    """Build a frame-window DataLoader over a :class:`VideoCodecPairDataset`.

    Identical sampler/plumbing choice as :func:`make_codec_loader` (the file-locality
    ``DistributedTwoLevelSampler`` under DDP, ``TwoLevelSampler`` single-process), only the
    collate differs (``(frames, mask)`` instead of ``(spec_a, spec_b)``).
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
        collate_fn=_video_collate,
        pin_memory=False,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True,
    )


# ===================================================================================== #
# SLOW-TS (Thomson / CER / MSE) codec — dataset + loader + gate, mirroring the machinery
# above. The "lightest touch" family (§4.3): masked reconstruction + entropy, NO δ-pair,
# NO discriminator.
# ===================================================================================== #
def slowts_codec_cfg(signal: str, channels: int) -> SlowTSCodecConfig:
    """Build a :class:`SlowTSCodecConfig` for ``signal`` with the loader's real channel count.

    ``channels`` is the number of profile positions actually loaded (from
    :func:`modality_channels`). The patch sizes come from :func:`slowts_patch_for` (one token
    per window by default). Kept as a helper so the trainer + CLI + tests build the cfg the
    same way.
    """
    patch_c, patch_t = slowts_patch_for(channels)
    return SlowTSCodecConfig(
        signal=signal, channels=channels, patch_c=patch_c, patch_t=patch_t
    )


class SlowTSCodecPairDataset(TokamakMultiFileDataset):
    """slow-TS windows ``(signal (C,T), mask (C,T))`` for ONE signal (Thomson / CER / MSE).

    A **thin** subclass of :class:`~tokamak_foundation_model.data.multi_file_dataset.\
TokamakMultiFileDataset`, EXACTLY as :class:`CodecPairDataset` (spectro) and
    :class:`VideoCodecPairDataset` (video) are. It reuses the parent's production streaming
    machinery WHOLESALE — the global-idx → ``(file_idx, chunk_idx)`` binary-search map, the
    per-worker LRU HDF5 file-handle cache (with the ``__getstate__``/``__setstate__`` pickling
    into DataLoader workers), the length-cache sidecar, and the ``num_workers`` plumbing — and
    overrides ONLY the per-item transform hook :meth:`_getitem_standard`. It does **not** touch
    ``data_loader.py`` / ``multi_file_dataset.py``.

    Difference from the spectro / video datasets
    ---------------------------------------------
    The slow-TS codec has NO δ-shift consistency pair (§4.3; no realization nuisance) and NO
    adversarial term. So the transform returns the slow-TS **window** ``(C, T)`` plus a
    per-sample **validity mask** ``(C, T)`` — NOT a pair. The parent's ``__getitem__`` still
    does the binary-search index map and sets ``self.h5_file`` before dispatching to our
    ``_getitem_standard(chunk_idx)``; here ``chunk_idx`` is the within-shot window index and
    ``self.h5_file`` is this worker's LRU handle. We reuse the parent's windowing formula
    (``t_start = warmup_s + chunk_idx * step_size_s``) and the parent's
    :meth:`TokamakH5Dataset._load_signal_raw` (already resampled to ``target_fs`` == 100 Hz,
    NaN→0 filled, with the raw NaN mask returned) — the slow-TS analogue of the other datasets'
    ``_load_signal_raw`` / ``_load_movie_raw`` reuse.

    Missingness / mask contract
    ---------------------------
    ``_load_signal_raw`` returns ``(raw (C,T), valid_len, nan_mask (C,T))`` where ``nan_mask``
    is 1.0 where the raw HDF5 value was NaN. We derive the per-sample VALIDITY mask (1 = real)
    EXACTLY as ``data_loader.TokamakH5Dataset._process_signal`` would for this signal:
      * ``zero_is_missing`` signals (Thomson): valid = (raw != 0) AND (nan_mask == 0).
      * otherwise (CER / MSE): valid = (nan_mask == 0).
    This mirrors the loader's own ``element_mask = data != 0.0`` (zero_is_missing) vs the NaN
    mask, so the codec's masked reconstruction never trains toward the zero/NaN-fill of a
    missing sample.

    Parameters
    ----------
    signal : str
        One of :data:`SLOWTS_MODALITIES`.
    shots, cfg, data_dir, seed, lengths_cache_path, max_open_files, max_tries, t0_start :
        as :class:`CodecPairDataset` (``cfg`` here is a :class:`SlowTSCodecConfig`). ``cfg`` is
        NOT mutated.
    """

    def __init__(
        self,
        signal: str,
        shots: Sequence[Union[str, int]],
        cfg: SlowTSCodecConfig,
        data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
        *,
        t0_start: float = 1.0,
        seed: int = 0,
        lengths_cache_path: Optional[Union[str, Path]] = None,
        max_open_files: int = 512,
        max_tries: int = 8,
    ) -> None:
        if signal not in SLOWTS_MODALITIES:
            raise ValueError(f"signal {signal!r} not in {SLOWTS_MODALITIES}")

        self.modality = signal
        self.codec_cfg = cfg
        self.item_seed = int(seed)
        self.max_tries = int(max_tries)

        paths = _shot_paths(shots, data_dir)
        if not paths:
            raise ValueError(
                f"SlowTSCodecPairDataset: no {{shot}}_processed.h5 files found under "
                f"{data_dir} for the {len(list(shots))} requested shots."
            )

        # A slow-TS window is a single CHUNK_S (50 ms) clip: chunk_duration_s = step_size_s =
        # CHUNK_S (non-overlapping 50 ms windows), warmup_s = t0_start. The parent's length
        # formula floor((duration - chunk_duration_s)/step_size_s)+1 guarantees each window
        # fits inside the shot's real data. We request the signal by NAME so the parent's
        # config reflects this modality.
        super().__init__(
            hdf5_paths=paths,
            chunk_duration_s=CHUNK_S,
            step_size_s=CHUNK_S,
            warmup_s=float(t0_start),
            input_signals=[signal],
            target_signals=[signal],
            prediction_mode=False,
            lengths_cache_path=lengths_cache_path,
            max_open_files=max_open_files,
        )
        self._cfg_sig = next(c for c in self.signal_configs if c.name == signal)

    # -- the ONLY overridden hook: per-item window + validity-mask transform ------------ #
    def _getitem_standard(self, idx: int):  # type: ignore[override]
        """Return one ``(signal (C,T), mask (C,T))`` for window ``idx``.

        Called by the parent's ``__getitem__`` AFTER it mapped the global index to
        ``(file_idx, chunk_idx)`` and set ``self.h5_file``. ``idx`` here is the within-shot
        ``chunk_idx``. Reuses the parent's windowing formula + :meth:`_load_signal_raw`.
        """
        item = self._build_window(idx)
        if item is not None:
            return item
        # Degenerate window (all-missing / padded tail): re-draw from nearby chunks.
        n_chunks = max(1, self._cumulative_lengths_span())
        gen = torch.Generator().manual_seed(self.item_seed + 100_003 * int(idx) + 7)
        for _ in range(self.max_tries):
            alt = int(torch.randint(0, n_chunks, (1,), generator=gen).item())
            item = self._build_window(alt)
            if item is not None:
                return item
        # Last resort: a finite all-zero window + all-INVALID mask (rare; whole shot missing).
        cfg = self.codec_cfg
        z = torch.zeros((cfg.channels, cfg.time_steps))
        m = torch.zeros((cfg.channels, cfg.time_steps), dtype=torch.float32)
        return z, m

    # -- helpers (reuse parent state; no re-implementation of the index map) ------------ #
    def _build_window(self, chunk_idx: int):
        """Build ``(signal (C,T), mask (C,T))`` for a within-shot ``chunk_idx``; None if bad.

        ``t_start`` reuses the parent's EXACT windowing formula; the raw window + NaN mask come
        from the parent's ``_load_signal_raw`` (already resampled to ``target_fs`` == 100 Hz).
        The validity mask mirrors ``_process_signal``'s missingness policy for this signal
        (see the class docstring). A window with NO valid samples at all is rejected (None) so
        the codec never trains on a fully-missing window.
        """
        cfg = self.codec_cfg
        step = getattr(self, "step_size_s", self.chunk_duration_s)
        warmup = getattr(self, "warmup_s", 0.0)
        t_start = warmup + chunk_idx * step          # parent windowing, reused verbatim
        t_end = t_start + CHUNK_S

        raw, valid_len, nan_mask = self._load_signal_raw(
            self.h5_file, self._cfg_sig, t_start, t_end
        )
        need = round(CHUNK_S * self._cfg_sig.target_fs)
        if valid_len < need:
            return None                              # runs into padded / missing tail
        signal, mask = self._fit_window(raw, nan_mask)
        if not torch.isfinite(signal).all():
            return None
        if float(mask.sum()) <= 0.0:
            return None                              # whole window missing -> reject
        return signal, mask

    def _fit_window(self, raw: torch.Tensor, nan_mask: torch.Tensor):
        """Crop/pad ``raw`` to ``(cfg.channels, cfg.time_steps)`` + build the validity mask.

        ``_load_signal_raw`` returns ``(C, T)`` at ``target_fs`` (T == cfg.time_steps for a
        50 ms window at 100 Hz); this guard makes the codec ``cfg`` authoritative for the whole
        ``(C, T)`` shape (T can drift a sample from rounding). The validity mask is built from
        the loader's missingness policy for this signal (see the class docstring).
        """
        cfg = self.codec_cfg
        C, T = raw.shape
        # time fit (usually a no-op).
        if T > cfg.time_steps:
            raw = raw[:, : cfg.time_steps]
            nan_mask = nan_mask[:, : cfg.time_steps]
        elif T < cfg.time_steps:
            if T == 0:
                z = torch.zeros((cfg.channels, cfg.time_steps))
                return z, torch.zeros_like(z)
            pad_r = raw[:, -1:].expand(C, cfg.time_steps - T)
            pad_m = nan_mask[:, -1:].expand(C, cfg.time_steps - T)
            raw = torch.cat([raw, pad_r], dim=1)
            nan_mask = torch.cat([nan_mask, pad_m], dim=1)
        # channel fit (defensive; loader already yields cfg.channels positions).
        if C != cfg.channels:
            raw = raw[: cfg.channels]
            nan_mask = nan_mask[: cfg.channels]
            if raw.shape[0] < cfg.channels:
                pad = torch.zeros((cfg.channels - raw.shape[0], cfg.time_steps))
                raw = torch.cat([raw, pad], dim=0)
                nan_mask = torch.cat([nan_mask, torch.ones_like(pad)], dim=0)

        # validity mask: 1 = real sample. Mirror _process_signal's missingness policy.
        valid = nan_mask < 0.5                       # NaN mask 1.0 == NaN -> invalid
        if cfg.zero_is_missing:                      # Thomson: a 0 is a missing sample.
            valid = valid & (raw != 0.0)
        return raw, valid.to(torch.float32)

    def _cumulative_lengths_span(self) -> int:
        """Best-effort chunk count for the re-draw bound (total dataset length; safe bound)."""
        try:
            return int(self._cumulative_lengths[-1])
        except Exception:
            return 1


def _slowts_collate(batch):
    """Stack ``(signal, mask)`` items into ``((B,C,T), (B,C,T))`` batched tensors."""
    sig = torch.stack([s for s, _ in batch], dim=0)
    masks = torch.stack([m for _, m in batch], dim=0)
    return sig, masks


def make_slowts_loader(
    dataset: SlowTSCodecPairDataset,
    *,
    batch_size: int,
    num_workers: int,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    """Build a slow-TS window DataLoader over a :class:`SlowTSCodecPairDataset`.

    Identical sampler/plumbing choice as :func:`make_codec_loader` / :func:`make_video_loader`
    (the file-locality ``DistributedTwoLevelSampler`` under DDP, ``TwoLevelSampler``
    single-process), only the collate differs (``(signal, mask)``).
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
        collate_fn=_slowts_collate,
        pin_memory=False,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True,
    )


# ------------------------------------------------------------------------------------- #
# slow-TS stability nuisance + gate (mirrors spike.compute_gate for slow-TS)
# ------------------------------------------------------------------------------------- #
def slowts_nuisance(x: torch.Tensor, *, jitter: float = 0.02, seed: int = 0) -> torch.Tensor:
    """A realization-level nuisance transform of a slow-TS window for the STABILITY gate.

    Slow-TS has no STFT-phase pair (like video), so the statistics-first "stability" property —
    codes unchanged under a realization-only perturbation — is probed with a small,
    structure-preserving augmentation: a per-window multiplicative amplitude jitter (a few %).
    The profile SHAPE (which positions carry more/less, its smooth time trend) is preserved;
    only the measurement-scale realization moves — so a statistics-first codec should map ``x``
    and ``slowts_nuisance(x)`` to (nearly) the same codes. ``x`` is ``(B, C, T)``.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    scale = 1.0 + jitter * (2.0 * torch.rand((), generator=gen).item() - 1.0)
    return x * scale


@torch.no_grad()
def slowts_compute_gate(
    codec: SlowTSCodec,
    eval_windows: List[torch.Tensor],
    eval_masks: List[torch.Tensor],
    frame_seq: torch.Tensor,
    cfg: SlowTSCodecConfig,
) -> Dict[str, object]:
    """slow-TS analogue of :func:`spike.compute_gate` (§4.4 gate on codes + masked recon).

    Parameters
    ----------
    codec : SlowTSCodec
    eval_windows : list of (B, C, T)
        Held-out windows; used for **stability** (codes of window vs :func:`slowts_nuisance`)
        and **slow-TS decode-fidelity** (masked recon of window vs window).
    eval_masks : list of (B, C, T)
        The matching per-sample validity masks (aligned with ``eval_windows``), used to mask
        the decode-fidelity profile statistic.
    frame_seq : (B, n_windows, C, T)
        Consecutive world-model windows; used for **persistence** (window t vs t+1) and
        **forecastability** (whole sequence of codes). Each "frame" is one 50 ms slow-TS window.
    cfg : SlowTSCodecConfig

    Returns the same key set as :func:`spike.compute_gate`, so :func:`spike.gate_score` and the
    trainer plumbing consume it unchanged.
    """
    was_training = codec.training
    codec.eval()

    stab_vals: List[float] = []
    dec_corr: List[float] = []
    dec_f1: List[float] = []
    dec_sharp: List[float] = []
    for wi, x in enumerate(eval_windows):
        out = codec.forward(x)
        nuis = slowts_nuisance(x, seed=wi)
        _, codes_n = codec.quantize(codec.encode(nuis))
        stab_vals.append(gate.stability(out["codes"], codes_n))
        m = eval_masks[wi] if wi < len(eval_masks) else None
        dm = gate.slowts_decode_fidelity(out["recon"], x, mask=m)
        dec_corr.append(dm["envelope_corr"])
        dec_f1.append(dm["peak_f1"])
        dec_sharp.append(dm["sharpness"])

    stability_val = float(sum(stab_vals) / len(stab_vals))
    decode = {
        "envelope_corr": float(sum(dec_corr) / len(dec_corr)),
        "peak_f1": float(sum(dec_f1) / len(dec_f1)),
        "sharpness": float(sum(dec_sharp) / len(dec_sharp)),
    }

    B, n_win = frame_seq.shape[0], frame_seq.shape[1]
    flat = frame_seq.reshape(B * n_win, *frame_seq.shape[2:])   # (B*n_win, C, T)
    _, codes_flat = codec.quantize(codec.encode(flat))
    codes_seq = codes_flat.reshape(B, n_win, cfg.n_tok, cfg.fsq_dim)

    persistence_val = gate.persistence(codes_seq[:, 0], codes_seq[:, 1])
    forecast = gate.forecastability(codes_seq, transition_mask=None)
    util = gate.utilization(codes_seq, cfg=cfg)

    if was_training:
        codec.train()

    return {
        "stability": stability_val,
        "persistence": float(persistence_val),
        "forecastability": forecast,
        "decode": decode,
        "utilization": util,
        "pass_stability": bool(stability_val >= cfg.gate_stability),
        "pass_persistence": bool(persistence_val >= cfg.gate_persistence),
        "pass_utilization": bool(not util["collapsed"]),
    }


# ------------------------------------------------------------------------------------- #
# slow-TS DDP generator-loss adapter + per-step (no δ-pair, no discriminator)
# ------------------------------------------------------------------------------------- #
class _SlowTSGenLossAdapter(torch.nn.Module):
    """Thin wrapper whose ``forward`` dispatches to ``SlowTSCodec.generator_losses``.

    Same DDP purpose as :class:`_GenLossAdapter` / :class:`_VideoGenLossAdapter`: DDP's reducer
    only hooks params used in ``forward``; the codec's step is ``codec.generator_losses(...)``,
    so we DDP-wrap THIS adapter. ``self.codec`` is the underlying :class:`SlowTSCodec` (used for
    gate + EMA + state_dict). The slow-TS signature has NO ``x_shift`` and NO ``disc`` (masked
    reconstruction + entropy only).
    """

    def __init__(self, codec: SlowTSCodec) -> None:
        super().__init__()
        self.codec = codec

    def forward(
        self,
        x: torch.Tensor,
        cfg: SlowTSCodecConfig,
        step: int,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.codec.generator_losses(x, cfg, step=step, mask=mask)


def slowts_codec_train_step(
    codec: SlowTSCodec,
    opt_g: torch.optim.Optimizer,
    x: torch.Tensor,
    mask: Optional[torch.Tensor],
    cfg: SlowTSCodecConfig,
    step: int,
) -> Dict[str, torch.Tensor]:
    """ONE codec training step on a slow-TS window (single-GPU / bare path).

    There is NO GAN alternation (no discriminator): a single generator (codec) step on the
    masked-reconstruction + entropy loss. Shared by the DDP trainer's ``world_size==1`` path
    and the tests.
    """
    codec.train()
    opt_g.zero_grad(set_to_none=True)
    g_terms = codec.generator_losses(x, cfg, step=step, mask=mask)
    g_terms["total"].backward()
    opt_g.step()
    return g_terms


def _ddp_slowts_train_step(
    gen_module: torch.nn.Module,        # DDP-wrapped _SlowTSGenLossAdapter (or bare)
    codec: SlowTSCodec,                 # underlying raw codec (unused here; kept for symmetry)
    opt_g: torch.optim.Optimizer,
    x: torch.Tensor,
    mask: Optional[torch.Tensor],
    cfg: SlowTSCodecConfig,
    step: int,
) -> Dict[str, torch.Tensor]:
    """DDP-aware analogue of :func:`slowts_codec_train_step` (routes the G forward through DDP)."""
    opt_g.zero_grad(set_to_none=True)
    g_terms = gen_module(x, cfg, step, mask)
    g_terms["total"].backward()
    opt_g.step()
    return g_terms


# ------------------------------------------------------------------------------------- #
# held-out streamed SLOW-TS eval set (windows + masks + consecutive-window sequence)
# ------------------------------------------------------------------------------------- #
def _stream_slowts_eval_data(
    signal: str,
    eval_shots: Sequence[Union[str, int]],
    cfg: SlowTSCodecConfig,
    *,
    data_dir: Union[str, Path],
    eval_batches: int,
    eval_batch_size: int,
    eval_frames: int,
    device: torch.device,
    seed: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
    """Materialize a small held-out SLOW-TS gate set (mirrors :func:`_stream_video_eval_data`).

    Returns ``(eval_windows, eval_masks, frame_seq)`` for :func:`slowts_compute_gate`:
    ``eval_windows`` / ``eval_masks`` = lists of ``(B, C, T)`` window / validity-mask batches
    (for stability + slow-TS decode-fidelity); ``frame_seq`` = ``(B, n_windows, C, T)``
    consecutive 50 ms windows (for persistence + forecastability). Streamed on THIS rank
    (world_size=1) so every rank has the same held-out data; the gate is computed on rank 0
    only. ``eval_frames`` = the number of consecutive world-model WINDOWS in the sequence.
    """
    ds = SlowTSCodecPairDataset(signal, eval_shots, cfg, data_dir=data_dir, seed=seed)
    n_win = eval_batches * eval_batch_size
    if len(ds) < n_win:
        raise RuntimeError(
            f"_stream_slowts_eval_data: only {len(ds)} eval windows across "
            f"{len(list(eval_shots))} eval shots; need {n_win} "
            f"(eval_batches={eval_batches} * eval_batch_size={eval_batch_size})."
        )
    flat = [ds[i] for i in range(n_win)]
    eval_windows: List[torch.Tensor] = []
    eval_masks: List[torch.Tensor] = []
    for bi in range(eval_batches):
        chunk = flat[bi * eval_batch_size : (bi + 1) * eval_batch_size]
        eval_windows.append(torch.stack([c[0] for c in chunk], dim=0).to(device))
        eval_masks.append(torch.stack([c[1] for c in chunk], dim=0).to(device))

    bsz = max(2, eval_batch_size // 2)
    n_windows = eval_frames
    n_avail = len(ds)
    seqs: List[torch.Tensor] = []
    b = 0
    while len(seqs) < bsz and (b + 1) * n_windows <= n_avail:
        block = [ds[b * n_windows + k][0] for k in range(n_windows)]
        seqs.append(torch.stack(block, dim=0))  # (n_windows, C, T)
        b += 1
    if len(seqs) < bsz:
        raise RuntimeError(
            f"_stream_slowts_eval_data: only {len(seqs)} consecutive {n_windows}-window "
            f"sequences across eval shots; need {bsz}. Add more eval shots."
        )
    frame_seq = torch.stack(seqs, dim=0).to(device)  # (bsz, n_windows, C, T)
    return eval_windows, eval_masks, frame_seq


# ------------------------------------------------------------------------------------- #
# video stability nuisance + gate (mirrors spike.compute_gate for video)
# ------------------------------------------------------------------------------------- #
def video_nuisance(clip: torch.Tensor, *, shift: int = 1, bright: float = 0.02,
                   seed: int = 0) -> torch.Tensor:
    """A realization-level nuisance transform of a video clip for the STABILITY gate.

    Video has no STFT-phase pair (unlike spectro), so the statistics-first "stability"
    property — codes unchanged under a realization-only perturbation — is probed with a
    small, structure-preserving augmentation: a 1-pixel spatial roll (sub-structure jitter,
    like the sub-window δ-shift for spectro) + a few-% brightness jitter. The plasma
    *structure* (where the light sits, its intensity envelope) is preserved; only the
    pixel-level realization moves — so a statistics-first codec should map ``clip`` and
    ``video_nuisance(clip)`` to (nearly) the same codes.

    ``clip`` is ``(B, C, T, H, W)``; returns the same shape.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    # 1-pixel spatial roll in H and W (structure-preserving realization jitter).
    out = torch.roll(clip, shifts=(shift, shift), dims=(-2, -1))
    # small multiplicative brightness jitter (per-clip scalar).
    jitter = 1.0 + bright * (2.0 * torch.rand((), generator=gen).item() - 1.0)
    return out * jitter


@torch.no_grad()
def video_compute_gate(
    codec: VideoCodec,
    eval_clips: List[torch.Tensor],
    frame_seq: torch.Tensor,
    cfg: VideoCodecConfig,
) -> Dict[str, object]:
    """Video analogue of :func:`spike.compute_gate` (§4.4 gate on frame codes + recon).

    Parameters
    ----------
    codec : VideoCodec
    eval_clips : list of (B, C, T, H, W)
        Held-out clips; used for **stability** (codes of clip vs :func:`video_nuisance`) and
        **video decode-fidelity** (recon of clip vs clip).
    frame_seq : (B, n_windows, C, T, H, W)
        Consecutive world-model windows; used for **persistence** (window t vs t+1) and
        **forecastability** (whole sequence of codes). Each "frame" of the world model is one
        50 ms video window (the same stepping unit as spectro).
    cfg : VideoCodecConfig

    Returns the same key set as :func:`spike.compute_gate`, so :func:`spike.gate_score` and
    the trainer plumbing consume it unchanged: ``stability`` / ``persistence`` /
    ``forecastability`` / ``decode`` / ``utilization`` + the ``pass_*`` booleans.
    """
    was_training = codec.training
    codec.eval()

    stab_vals: List[float] = []
    dec_corr: List[float] = []
    dec_f1: List[float] = []
    dec_sharp: List[float] = []
    for clip in eval_clips:
        out = codec.forward(clip)
        nuis = video_nuisance(clip, seed=len(stab_vals))
        _, codes_n = codec.quantize(codec.encode(nuis))
        stab_vals.append(gate.stability(out["codes"], codes_n))
        dm = gate.video_decode_fidelity(out["recon"], clip)
        dec_corr.append(dm["envelope_corr"])
        dec_f1.append(dm["peak_f1"])
        dec_sharp.append(dm["sharpness"])

    stability_val = float(sum(stab_vals) / len(stab_vals))
    decode = {
        "envelope_corr": float(sum(dec_corr) / len(dec_corr)),
        "peak_f1": float(sum(dec_f1) / len(dec_f1)),
        "sharpness": float(sum(dec_sharp) / len(dec_sharp)),
    }

    B, n_win = frame_seq.shape[0], frame_seq.shape[1]
    flat = frame_seq.reshape(B * n_win, *frame_seq.shape[2:])   # (B*n_win, C, T, H, W)
    _, codes_flat = codec.quantize(codec.encode(flat))
    codes_seq = codes_flat.reshape(B, n_win, cfg.n_tok, cfg.fsq_dim)

    persistence_val = gate.persistence(codes_seq[:, 0], codes_seq[:, 1])
    forecast = gate.forecastability(codes_seq, transition_mask=None)
    util = gate.utilization(codes_seq, cfg=cfg)

    if was_training:
        codec.train()

    return {
        "stability": stability_val,
        "persistence": float(persistence_val),
        "forecastability": forecast,
        "decode": decode,
        "utilization": util,
        "pass_stability": bool(stability_val >= cfg.gate_stability),
        "pass_persistence": bool(persistence_val >= cfg.gate_persistence),
        "pass_utilization": bool(not util["collapsed"]),
    }


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
# video DDP generator-loss adapter + per-step (mirrors the spectro pair above, no x_shift)
# ------------------------------------------------------------------------------------- #
class _VideoGenLossAdapter(torch.nn.Module):
    """Thin wrapper whose ``forward`` dispatches to ``VideoCodec.generator_losses``.

    Same DDP purpose as :class:`_GenLossAdapter`: DDP's reducer only hooks params used in
    ``forward``; the codec's generator step is ``codec.generator_losses(...)``, so we DDP-wrap
    THIS adapter. ``self.codec`` is the underlying :class:`VideoCodec` (used for the detached
    D-recon + gate + EMA + state_dict). The video signature has NO ``x_shift`` (no δ-pair).
    """

    def __init__(self, codec: VideoCodec) -> None:
        super().__init__()
        self.codec = codec

    def forward(
        self,
        x: torch.Tensor,
        disc: torch.nn.Module,
        cfg: VideoCodecConfig,
        step: int,
        frame_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.codec.generator_losses(x, disc, cfg, step=step, frame_mask=frame_mask)


def video_codec_train_step(
    codec: VideoCodec,
    disc: FramePatchGAN,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    frames: torch.Tensor,
    frame_mask: Optional[torch.Tensor],
    cfg: VideoCodecConfig,
    step: int,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """ONE generator+discriminator alternation step on a video clip (single-GPU / bare path).

    The video analogue of :func:`spike.codec_train_step` — same alternation (generator step
    on the codec's ``generator_losses``, then a fresh DETACHED-recon discriminator step) but
    with no δ-shift pair. Shared by the DDP trainer's ``world_size==1`` path and the tests.
    """
    codec.train()

    opt_g.zero_grad(set_to_none=True)
    g_terms = codec.generator_losses(frames, disc, cfg, step=step, frame_mask=frame_mask)
    g_terms["total"].backward()
    opt_g.step()

    opt_d.zero_grad(set_to_none=True)
    with torch.no_grad():
        recon = codec.forward(frames)["recon"]
    d_loss = _video_discriminator_loss(disc, frames, recon, cfg)
    d_loss.backward()
    opt_d.step()

    return g_terms, d_loss


def _ddp_video_train_step(
    gen_module: torch.nn.Module,        # DDP-wrapped _VideoGenLossAdapter (or bare)
    codec: VideoCodec,                  # underlying raw codec
    disc: torch.nn.Module,              # DDP-wrapped discriminator (or bare)
    disc_raw: FramePatchGAN,            # underlying raw discriminator
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    frames: torch.Tensor,
    frame_mask: Optional[torch.Tensor],
    cfg: VideoCodecConfig,
    step: int,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """DDP-aware analogue of :func:`video_codec_train_step` (routes the G forward through DDP)."""
    codec.train()
    disc_raw.train()

    opt_g.zero_grad(set_to_none=True)
    g_terms = gen_module(frames, disc_raw, cfg, step, frame_mask)
    g_terms["total"].backward()
    opt_g.step()

    opt_d.zero_grad(set_to_none=True)
    with torch.no_grad():
        recon = codec.forward(frames)["recon"]
    d_loss = _video_discriminator_loss(disc, frames, recon, cfg)
    d_loss.backward()
    opt_d.step()

    return g_terms, d_loss


def _video_discriminator_loss(
    disc: torch.nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    cfg: VideoCodecConfig,
) -> torch.Tensor:
    """Hinge GAN discriminator loss for the frame PatchGAN (reuses the shared hinge helpers).

    Identical math to :func:`losses.discriminator_loss` (``E[relu(1-D(real))] +
    E[relu(1+D(fake))]`` over the per-frame score maps); factored here only because
    ``losses.discriminator_loss`` is typed to ``SpectroCodecConfig`` (it reads no cfg field,
    but keeping a video-typed entry point is clearer and avoids any future cfg-field coupling).
    """
    from .losses import _as_score_list, _hinge_fake, _hinge_real

    real_maps = _as_score_list(disc(real))
    fake_maps = _as_score_list(disc(fake.detach() if fake.requires_grad else fake))
    return _hinge_real(real_maps) + _hinge_fake(fake_maps)


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
# held-out streamed VIDEO eval set (clips + consecutive-window sequence)
# ------------------------------------------------------------------------------------- #
def _stream_video_eval_data(
    modality: str,
    eval_shots: Sequence[Union[str, int]],
    cfg: VideoCodecConfig,
    *,
    data_dir: Union[str, Path],
    eval_batches: int,
    eval_batch_size: int,
    eval_frames: int,
    device: torch.device,
    seed: int,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Materialize a small held-out VIDEO gate set (mirrors :func:`_stream_eval_data`).

    Returns ``(eval_clips, frame_seq)`` for :func:`video_compute_gate`: ``eval_clips`` = list
    of ``(B, C, T, H, W)`` clip batches (for stability + video decode-fidelity); ``frame_seq``
    = ``(B, n_windows, C, T, H, W)`` consecutive 50 ms windows (for persistence +
    forecastability). Streamed on THIS rank (world_size=1) so every rank has the same held-out
    data; the gate is computed on rank 0 only. ``eval_frames`` here is the number of
    consecutive world-model WINDOWS in the sequence (each window is one 50 ms video clip).
    """
    ds = VideoCodecPairDataset(modality, eval_shots, cfg, data_dir=data_dir, seed=seed)
    n_clips = eval_batches * eval_batch_size
    if len(ds) < n_clips:
        raise RuntimeError(
            f"_stream_video_eval_data: only {len(ds)} eval windows across "
            f"{len(list(eval_shots))} eval shots; need {n_clips} "
            f"(eval_batches={eval_batches} * eval_batch_size={eval_batch_size})."
        )
    flat = [ds[i][0] for i in range(n_clips)]  # frames only (mask unused in the gate)
    eval_clips: List[torch.Tensor] = []
    for bi in range(eval_batches):
        chunk = flat[bi * eval_batch_size : (bi + 1) * eval_batch_size]
        eval_clips.append(torch.stack(chunk, dim=0).to(device))

    # consecutive-window sequence: for each sample, ``eval_frames`` back-to-back 50 ms clips.
    bsz = max(2, eval_batch_size // 2)
    n_windows = eval_frames
    n_avail = len(ds)
    seqs: List[torch.Tensor] = []
    # stride blocks of n_windows consecutive dataset indices (dataset windows are already the
    # non-overlapping 50 ms cores, in shot order per the parent's cumulative-length layout).
    b = 0
    while len(seqs) < bsz and (b + 1) * n_windows <= n_avail:
        block = [ds[b * n_windows + k][0] for k in range(n_windows)]
        seqs.append(torch.stack(block, dim=0))  # (n_windows, C, T, H, W)
        b += 1
    if len(seqs) < bsz:
        raise RuntimeError(
            f"_stream_video_eval_data: only {len(seqs)} consecutive {n_windows}-window "
            f"sequences across eval shots; need {bsz}. Add more eval shots."
        )
    frame_seq = torch.stack(seqs, dim=0).to(device)  # (bsz, n_windows, C, T, H, W)
    return eval_clips, frame_seq


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
# the streaming VIDEO trainer (mirrors train_codec; reuses the SAME DDP + best-ckpt + gate)
# ------------------------------------------------------------------------------------- #
def train_video_codec(
    cfg: VideoCodecConfig,
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
    ddp: Optional["_DDPState"] = None,
    resume_state: Optional[Dict[str, object]] = None,
    lengths_cache_path: Optional[Union[str, Path]] = None,
    log_fn=print,
) -> Dict[str, object]:
    """Streaming, DDP-capable VIDEO codec training loop.

    The video analogue of :func:`train_codec` — it reuses the SAME infrastructure (the
    ``_DDPState`` init/wrap, ``spike._EMA``, ``spike.gate_score``, ``spike._fmt_gate``,
    ``spike._write_gate_json`` / ``spike._save_checkpoint`` / ``spike._save_best_checkpoint``
    / ``spike._save_ema_checkpoint`` best-ckpt+EMA+gate path) and differs only where video
    differs: a :class:`VideoCodec` + :class:`FramePatchGAN`, the ``(frames, frame_mask)``
    loader, the no-δ-pair :func:`_ddp_video_train_step`, and :func:`video_compute_gate`.
    Rank-0 only logs + writes ``codec_last.pt`` / ``codec_best.pt`` / (optional)
    ``codec_ema.pt`` / ``gate_<step>.json`` (identical filenames + score contract as spectro).
    """
    if steps < 1:
        raise ValueError("train_video_codec: steps must be >= 1")
    ddp = ddp if ddp is not None else _DDPState()
    device = ddp.device
    torch.manual_seed(seed + ddp.rank)

    out_path = Path(out_dir) if out_dir is not None else None
    if out_path is not None and ddp.is_main:
        out_path.mkdir(parents=True, exist_ok=True)

    codec = VideoCodec(cfg).to(device)
    disc_raw = FramePatchGAN(cfg).to(device)

    start_step = 0
    if resume_state is not None:
        codec.load_state_dict(resume_state["codec"])
        if resume_state.get("disc") is not None:
            disc_raw.load_state_dict(resume_state["disc"])
        start_step = int(resume_state.get("step", 0))

    opt_g = torch.optim.Adam(codec.parameters(), lr=lr)
    opt_d = torch.optim.Adam(disc_raw.parameters(), lr=disc_lr if disc_lr is not None else lr)
    if resume_state is not None:
        if resume_state.get("opt_g") is not None:
            opt_g.load_state_dict(resume_state["opt_g"])
        if resume_state.get("opt_d") is not None:
            opt_d.load_state_dict(resume_state["opt_d"])

    gen_module = ddp.wrap(_VideoGenLossAdapter(codec))
    disc = ddp.wrap(disc_raw)

    ema_shadow = spike._EMA(codec, ema_decay) if ema else None

    eval_clips, frame_seq = _stream_video_eval_data(
        modality, eval_shots, cfg,
        data_dir=data_dir, eval_batches=eval_batches, eval_batch_size=eval_batch_size,
        eval_frames=eval_frames, device=device, seed=seed + 777,
    )

    train_ds = VideoCodecPairDataset(
        modality, train_shots, cfg,
        data_dir=data_dir, seed=seed, lengths_cache_path=lengths_cache_path,
    )
    loader = make_video_loader(
        train_ds, batch_size=batch_size, num_workers=num_workers,
        rank=ddp.rank, world_size=ddp.world_size, seed=seed, shuffle=True,
    )
    stream = _epoch_cycler(loader)

    best_score = float("-inf")
    best_step: Optional[int] = None
    final_gate: Dict[str, object] = {}

    for local_step in range(steps):
        step = start_step + local_step
        frames, frame_mask = next(stream)
        frames = frames.to(device, non_blocking=True)
        frame_mask = frame_mask.to(device, non_blocking=True)

        g_terms, d_loss = _ddp_video_train_step(
            gen_module, codec, disc, disc_raw, opt_g, opt_d,
            frames, frame_mask, cfg, step=step,
        )
        if ema_shadow is not None:
            ema_shadow.update(codec)

        is_last = local_step == steps - 1
        if (step % eval_every == 0) or is_last:
            g = video_compute_gate(codec, eval_clips, frame_seq, cfg)
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
                            log_fn(f"[train_video_codec] new BEST score={score:+.4f} @ step {step}")
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
# the streaming SLOW-TS trainer (mirrors train_codec / train_video_codec; SAME DDP +
# best-ckpt + gate infra; no discriminator, no δ-pair)
# ------------------------------------------------------------------------------------- #
def train_slowts_codec(
    cfg: SlowTSCodecConfig,
    signal: str,
    train_shots: Sequence[Union[str, int]],
    eval_shots: Sequence[Union[str, int]],
    *,
    steps: int,
    eval_every: int,
    batch_size: int,
    num_workers: int,
    data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
    lr: float = 1e-3,
    disc_lr: Optional[float] = None,   # accepted + ignored (no discriminator); uniform signature
    ema: bool = False,
    ema_decay: float = 0.999,
    eval_batches: int = 8,
    eval_batch_size: int = 4,
    eval_frames: int = 6,
    out_dir: Optional[Union[str, Path]] = None,
    seed: int = 0,
    ddp: Optional["_DDPState"] = None,
    resume_state: Optional[Dict[str, object]] = None,
    lengths_cache_path: Optional[Union[str, Path]] = None,
    log_fn=print,
) -> Dict[str, object]:
    """Streaming, DDP-capable SLOW-TS codec training loop.

    The slow-TS analogue of :func:`train_codec` / :func:`train_video_codec` — it reuses the
    SAME infrastructure (the ``_DDPState`` init/wrap, ``spike._EMA``, ``spike.gate_score``,
    ``spike._fmt_gate``, ``spike._write_gate_json`` / ``spike._save_checkpoint`` /
    ``spike._save_best_checkpoint`` / ``spike._save_ema_checkpoint`` best-ckpt+EMA+gate path)
    and differs only where slow-TS differs: a :class:`SlowTSCodec` with **no discriminator**,
    the ``(signal, mask)`` loader, the no-δ-pair / no-GAN :func:`_ddp_slowts_train_step`, and
    :func:`slowts_compute_gate`. Rank-0 only logs + writes ``codec_last.pt`` /
    ``codec_best.pt`` / (optional) ``codec_ema.pt`` / ``gate_<step>.json`` (identical filenames
    + score contract as the other codecs).

    NOTE: the ``_save_*`` helpers persist a discriminator; the slow-TS codec has none, so a
    tiny zero-parameter :class:`_NullDisc` stand-in is passed (its ``state_dict`` is empty),
    keeping the checkpoint schema uniform without a real GAN.
    """
    if steps < 1:
        raise ValueError("train_slowts_codec: steps must be >= 1")
    ddp = ddp if ddp is not None else _DDPState()
    device = ddp.device
    torch.manual_seed(seed + ddp.rank)

    out_path = Path(out_dir) if out_dir is not None else None
    if out_path is not None and ddp.is_main:
        out_path.mkdir(parents=True, exist_ok=True)

    codec = SlowTSCodec(cfg).to(device)
    disc_raw = _NullDisc().to(device)  # no discriminator; empty-state stand-in for _save_*

    start_step = 0
    if resume_state is not None:
        codec.load_state_dict(resume_state["codec"])
        start_step = int(resume_state.get("step", 0))

    opt_g = torch.optim.Adam(codec.parameters(), lr=lr)
    if resume_state is not None and resume_state.get("opt_g") is not None:
        opt_g.load_state_dict(resume_state["opt_g"])
    # dummy optimizer over the null disc's (empty) params so _save_checkpoint has an opt_d.
    opt_d = torch.optim.Adam([torch.zeros(1, requires_grad=True)], lr=lr)

    gen_module = ddp.wrap(_SlowTSGenLossAdapter(codec))

    ema_shadow = spike._EMA(codec, ema_decay) if ema else None

    eval_windows, eval_masks, frame_seq = _stream_slowts_eval_data(
        signal, eval_shots, cfg,
        data_dir=data_dir, eval_batches=eval_batches, eval_batch_size=eval_batch_size,
        eval_frames=eval_frames, device=device, seed=seed + 777,
    )

    train_ds = SlowTSCodecPairDataset(
        signal, train_shots, cfg,
        data_dir=data_dir, seed=seed, lengths_cache_path=lengths_cache_path,
    )
    loader = make_slowts_loader(
        train_ds, batch_size=batch_size, num_workers=num_workers,
        rank=ddp.rank, world_size=ddp.world_size, seed=seed, shuffle=True,
    )
    stream = _epoch_cycler(loader)

    best_score = float("-inf")
    best_step: Optional[int] = None
    final_gate: Dict[str, object] = {}

    for local_step in range(steps):
        step = start_step + local_step
        x, mask = next(stream)
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        g_terms = _ddp_slowts_train_step(
            gen_module, codec, opt_g, x, mask, cfg, step=step,
        )
        if ema_shadow is not None:
            ema_shadow.update(codec)

        is_last = local_step == steps - 1
        if (step % eval_every == 0) or is_last:
            g = slowts_compute_gate(codec, eval_windows, eval_masks, frame_seq, cfg)
            g["g_total"] = float(g_terms["total"].detach())
            g["d_loss"] = 0.0  # no discriminator
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
                            log_fn(f"[train_slowts_codec] new BEST score={score:+.4f} @ step {step}")
                    if ema_shadow is not None:
                        spike._save_ema_checkpoint(out_path, step, ema_shadow.state_dict(), cfg)
            final_gate = g
        ddp.barrier()

    final_gate["steps"] = steps
    final_gate["global_step"] = start_step + steps
    final_gate["best_score"] = best_score
    final_gate["best_step"] = best_step
    return final_gate


class _NullDisc(torch.nn.Module):
    """A zero-parameter discriminator stand-in for the slow-TS codec (which has no GAN).

    The shared ``spike._save_checkpoint`` / ``_save_best_checkpoint`` helpers persist a
    discriminator ``state_dict`` to keep the checkpoint schema uniform across codecs. The
    slow-TS codec is a reconstruction + entropy codec with no discriminator (§4.3), so we
    pass this empty module — ``state_dict()`` is ``{}`` — instead of a real one. It is never
    called in a forward pass.
    """

    def state_dict(self, *args, **kwargs):  # noqa: D401 (empty state)
        return {}


# ------------------------------------------------------------------------------------- #
# CLI
# ------------------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tokamak_foundation_model.ignite.train_codec",
        description="IGNITE Phase-A STREAMING, DDP codec trainer (scales past the spike's "
                    "~40-shot pre-built pool to thousands of shots on the fly).",
    )
    p.add_argument("--modality", type=str, default="ece",
                   choices=list(SPECTRO_MODALITIES) + list(VIDEO_MODALITIES)
                           + list(SLOWTS_MODALITIES),
                   help="Codec modality: a spectro signal (ece/co2/bes/mhr), a tangtv "
                        "divertor video (tangtv_lower/tangtv_upper), or a slow-TS signal "
                        "(ts_core_density/ts_core_temp/ts_tangential_density/"
                        "ts_tangential_temp/cer_ti/cer_rot/mse).")
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

    is_video = args.modality in VIDEO_MODALITIES
    is_slowts = args.modality in SLOWTS_MODALITIES

    # cfg with the modality's real channel count baked in (all ranks agree).
    channels = modality_channels(args.modality)
    if is_video:
        cfg = VideoCodecConfig(channels=channels, divertor=video_divertor(args.modality))
    elif is_slowts:
        cfg = slowts_codec_cfg(args.modality, channels)
    else:
        cfg = SpectroCodecConfig(channels=channels)
    if args.entropy_weight is not None:
        cfg.entropy_weight = float(args.entropy_weight)
    if args.consistency_weight is not None:
        if is_video or is_slowts:
            raise SystemExit(
                "--consistency_weight is not valid for a video / slow-TS modality "
                "(those codecs have no shift-consistency term; see IGNITE_DESIGN §4.3)."
            )
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

    if is_video:
        trainer = train_video_codec
    elif is_slowts:
        trainer = train_slowts_codec
    else:
        trainer = train_codec
    t0 = time.time()
    final_gate = trainer(
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
