"""IGNITE Phase-A **fast-TS (filterscopes) codec** — streaming, DDP-capable trainer.

The fast-TS analogue of :mod:`ignite.train_codec` (spectro / video). It REUSES the shared
IGNITE infrastructure wholesale and only supplies the fast-TS-specific pieces:

* **Data** — :class:`FastTSCodecPairDataset` is a THIN subclass of
  :class:`~tokamak_foundation_model.data.multi_file_dataset.TokamakMultiFileDataset` (exactly
  like :class:`ignite.train_codec.CodecPairDataset`): it reuses the parent's global-idx →
  ``(file_idx, chunk_idx)`` binary-search map, the per-worker LRU HDF5 file-handle cache (with
  ``__getstate__`` / ``__setstate__`` pickling into DataLoader workers), the length-cache
  sidecar and the ``num_workers`` plumbing UNCHANGED, and overrides ONLY the per-item transform
  hook :meth:`_getitem_standard` to return the codec's ``(env_a, env_b)`` δ-shift ELM-envelope
  pair for the ``filterscopes`` modality. It does **not** touch ``data_loader.py`` /
  ``multi_file_dataset.py``.
* **Trainer** — reuses the SAME per-step generator/discriminator alternation MATH as the
  spectro codec: a δ-pair ``generator_losses`` step (through a DDP-wrapped adapter so DDP's
  reducer hooks the generator gradients) then a fresh detached-recon discriminator step. The
  ``_DDPState`` init/wrap, ``spike._EMA``, ``spike.gate_score`` / ``spike._fmt_gate`` /
  ``spike._save_{checkpoint,best_checkpoint,ema_checkpoint}`` / ``spike._write_gate_json``
  best-ckpt+EMA+gate path, and the file-locality samplers + ``_epoch_cycler`` are all imported
  from :mod:`ignite.train_codec` / :mod:`ignite.spike` — nothing is re-implemented.
* **Gate** — :func:`fastts_compute_gate` mirrors :func:`spike.compute_gate` but computes
  **stability** on the δ-shift envelope pair, **envelope decode-fidelity** via
  :func:`gate.fastts_decode_fidelity`, and **persistence** / **forecastability** / **utilization**
  on a consecutive-window envelope-code sequence — returning the SAME key set so
  :func:`spike.gate_score` (recon-floor -inf disqualification) consumes it unchanged.

WHY fast-TS keeps the δ-pair (unlike video): the ELM spike TIMING is a realization nuisance
(docs/IGNITE_DESIGN.md §4.3), the fast-TS analogue of the spectrogram's STFT phase — so the
codec, like spectro, trains on a δ-shifted pair and enforces shift-consistency.

Reuse boundary (§7): only ``torch`` + sibling ``ignite`` modules + the read-only data pipeline
(via the reused ``TokamakMultiFileDataset``). No FAITH *model* code.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader

from . import data, gate, spike
from .config import CHUNK_S, FASTTS_FS, FastTSCodecConfig
from .fastts_codec import FastTSCodec
from .fastts_discriminator import Env1DPatchGAN
from .losses import _as_score_list, _hinge_fake, _hinge_real
from .train_codec import (
    DistributedTwoLevelSampler,
    TokamakMultiFileDataset,
    TwoLevelSampler,
    _DDPState,
    _channels_after_selection,
    _epoch_cycler,
    _shot_paths,
)

DEFAULT_DATA_DIR = spike.DEFAULT_DATA_DIR

# The single fast-TS modality (a non-STFT SignalConfig in TokamakH5Dataset.SIGNAL_CONFIGS,
# target_fs=10 kHz, channels_to_use=slice(0,8), apply_stft=False). Named exactly as the loader.
FASTTS_MODALITY = "filterscopes"


# ------------------------------------------------------------------------------------- #
# modality channel-count helper (read-only against the loader config)
# ------------------------------------------------------------------------------------- #
def fastts_channels(modality: str = FASTTS_MODALITY) -> int:
    """Channels actually loaded for the fast-TS ``modality`` after ``channels_to_use``.

    Reads ``TokamakH5Dataset.SIGNAL_CONFIGS`` class-level + read-only (mirrors
    ``train_codec.modality_channels`` but for the non-STFT filterscopes signal). Returns 8 for
    ``filterscopes`` (``channels_to_use=slice(0, 8)``).
    """
    from tokamak_foundation_model.data.data_loader import TokamakH5Dataset

    if modality != FASTTS_MODALITY:
        raise ValueError(
            f"{modality!r} is not the fast-TS modality (only {FASTTS_MODALITY!r})"
        )
    cfg_sig = next(
        (c for c in TokamakH5Dataset.SIGNAL_CONFIGS if c.name == modality), None
    )
    if cfg_sig is None:
        raise ValueError(f"fast-TS modality {modality!r} not in SIGNAL_CONFIGS")
    return _channels_after_selection(cfg_sig.num_channels, cfg_sig.channels_to_use)


# ------------------------------------------------------------------------------------- #
# δ-shift-pair dataset — a THIN subclass of the production TokamakMultiFileDataset
# ------------------------------------------------------------------------------------- #
class FastTSCodecPairDataset(TokamakMultiFileDataset):
    """``(env_a, env_b)`` δ-shift ELM-envelope pairs for the ``filterscopes`` modality.

    A **thin** subclass of :class:`~tokamak_foundation_model.data.multi_file_dataset.\
TokamakMultiFileDataset`, structurally identical to :class:`ignite.train_codec.CodecPairDataset`
    (spectro) — it re-uses the parent's production streaming machinery WHOLESALE (the
    global-idx → ``(file_idx, chunk_idx)`` binary-search map, the per-worker LRU HDF5
    file-handle cache with correct pickling, the length-cache sidecar, the ``num_workers``
    plumbing) and overrides ONLY the per-item transform hook :meth:`_getitem_standard`.

    Windowing reuse (identical to CodecPairDataset)
    ----------------------------------------------
    The parent's ``__getitem__`` maps a global ``idx`` to ``(file_idx, chunk_idx)`` via the
    parent's binary-search over ``_cumulative_lengths``, sets ``self.h5_file`` to the
    per-worker LRU handle, then dispatches to our ``_getitem_standard(chunk_idx)`` — so we
    inherit the index map + file-handle safety verbatim (we re-implement none of it).
    ``t_start = warmup_s + chunk_idx * step_size_s`` is the
    parent's exact windowing formula. We construct the parent with ``chunk_duration_s =
    CHUNK_S + δ_max`` (the full δ-extended span) so the parent's length computation guarantees
    the shifted window ``[t_start, t_start + CHUNK_S + δ_max]`` fits inside the shot; ``step_size_s
    = CHUNK_S`` keeps successive windows 50 ms apart.

    δ-pair construction
    -------------------
    ``_getitem_standard`` loads the RAW δ-extended filterscope window via the parent's
    :meth:`TokamakH5Dataset._load_signal_raw` (already resampled to ``target_fs == FASTTS_FS``,
    zero-/NaN-padded), then builds the pair with :func:`ignite.data.fastts_shift_pair_windows`
    (raw δ-shift + re-envelope). Degenerate windows (all-NaN / near-flat / past the real data
    end) are re-drawn from nearby chunks, exactly as CodecPairDataset does.
    """

    def __init__(
        self,
        shots: Sequence[Union[str, int]],
        cfg: FastTSCodecConfig,
        data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
        *,
        modality: str = FASTTS_MODALITY,
        delta_ms_range: Optional[Tuple[float, float]] = None,
        t0_start: float = 1.0,
        min_std: float = 1e-6,
        seed: int = 0,
        lengths_cache_path: Optional[Union[str, Path]] = None,
        max_open_files: int = 512,
        max_tries: int = 8,
    ) -> None:
        if modality != FASTTS_MODALITY:
            raise ValueError(f"modality {modality!r} != {FASTTS_MODALITY!r}")

        self.modality = modality
        self.codec_cfg = cfg
        self.delta_ms_range = delta_ms_range or tuple(cfg.consistency_delta_ms)
        self.min_std = float(min_std)
        self.pair_seed = int(seed)
        self.max_tries = int(max_tries)

        # δ-extended span: A covers [t0, t0+CHUNK_S]; B is shifted by up to δ_max. Reserve room
        # for the WHOLE span via chunk_duration_s; step_size_s stays CHUNK_S (50 ms windows).
        self.span_tail_s = float(self.delta_ms_range[1]) / 1000.0
        span_s = CHUNK_S + self.span_tail_s

        paths = _shot_paths(shots, data_dir)
        if not paths:
            raise ValueError(
                f"FastTSCodecPairDataset: no {{shot}}_processed.h5 files found under {data_dir} "
                f"for the {len(list(shots))} requested shots."
            )

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
        self._cfg_sig = next(c for c in self.signal_configs if c.name == modality)

    # -- the ONLY overridden hook: per-item δ-pair transform -------------------------- #
    def _getitem_standard(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        """Return one ``(env_a, env_b)`` δ-shift ELM-envelope pair for window ``idx``.

        Called by the parent's ``__getitem__`` AFTER it has (a) binary-search-mapped the global
        index to ``(file_idx, chunk_idx)`` and (b) set ``self.h5_file`` to this shot's per-worker
        LRU handle. Here ``idx`` is the within-shot ``chunk_idx``.
        """
        pair = self._build_pair(idx)
        if pair is not None:
            return pair
        n_chunks = self._chunks_in_current_shot(idx)
        gen = torch.Generator().manual_seed(self.pair_seed + 100_003 * int(idx) + 7)
        for _ in range(self.max_tries):
            alt = int(torch.randint(0, max(1, n_chunks), (1,), generator=gen).item())
            pair = self._build_pair(alt)
            if pair is not None:
                return pair
        # Last resort: a finite silence-floor envelope pair (rare; whole shot degenerate).
        C = self._num_channels()
        z = torch.zeros((C, self.codec_cfg.env_bins))
        return z, z.clone()

    # -- helpers (all reuse parent state; no re-implementation of the index map) ------- #
    def _build_pair(self, chunk_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Build the δ-pair for a within-shot ``chunk_idx``; None if the window is degenerate."""
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
        env_a, env_b = data.fastts_shift_pair_windows(
            raw, t0=0.0, cfg=self.codec_cfg, delta_ms=d
        )
        return env_a, env_b

    def _chunks_in_current_shot(self, chunk_idx: int) -> int:
        """Number of chunks in the shot on ``self.h5_file`` (best-effort; bounds the re-draw)."""
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
    """Stack ``(env_a, env_b)`` pairs into ``(B, C, E)`` batched tensors."""
    env_a = torch.stack([a for a, _ in batch], dim=0)
    env_b = torch.stack([b for _, b in batch], dim=0)
    return env_a, env_b


def make_fastts_loader(
    dataset: FastTSCodecPairDataset,
    *,
    batch_size: int,
    num_workers: int,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    """Build a δ-pair DataLoader over a :class:`FastTSCodecPairDataset`.

    Identical sampler/plumbing choice as :func:`ignite.train_codec.make_codec_loader` (the
    file-locality ``DistributedTwoLevelSampler`` under DDP, ``TwoLevelSampler`` single-process);
    only the collate differs (``(env_a, env_b)`` envelope pairs).
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
        pin_memory=False,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True,
    )


# ------------------------------------------------------------------------------------- #
# per-step generator/discriminator alternation (mirrors spike.codec_train_step, envelope δ-pair)
# ------------------------------------------------------------------------------------- #
def _fastts_discriminator_loss(
    disc: torch.nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
) -> torch.Tensor:
    """Hinge GAN discriminator loss for the 1-D envelope PatchGAN (shared hinge helpers).

    Identical math to :func:`losses.discriminator_loss` (``E[relu(1-D(real))] +
    E[relu(1+D(fake))]`` over the per-scale score maps); a fast-TS-typed entry point (avoids
    any future cfg-field coupling), mirroring ``train_codec._video_discriminator_loss``.
    """
    real_maps = _as_score_list(disc(real))
    fake_maps = _as_score_list(disc(fake.detach() if fake.requires_grad else fake))
    return _hinge_real(real_maps) + _hinge_fake(fake_maps)


def fastts_codec_train_step(
    codec: FastTSCodec,
    disc: Env1DPatchGAN,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    env_a: torch.Tensor,
    env_b: torch.Tensor,
    cfg: FastTSCodecConfig,
    step: int,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """ONE generator+discriminator alternation step on a δ-shift envelope pair (bare path).

    The fast-TS analogue of :func:`spike.codec_train_step` — same alternation (generator step on
    the codec's δ-pair ``generator_losses``, then a fresh DETACHED-recon discriminator step).
    Shared by the DDP trainer's ``world_size==1`` path and the tests.
    """
    codec.train()

    opt_g.zero_grad(set_to_none=True)
    g_terms = codec.generator_losses(env_a, env_b, disc, cfg, step=step)
    g_terms["total"].backward()
    opt_g.step()

    opt_d.zero_grad(set_to_none=True)
    with torch.no_grad():
        recon = codec.forward(env_a)["recon"]
    d_loss = _fastts_discriminator_loss(disc, env_a, recon)
    d_loss.backward()
    opt_d.step()

    return g_terms, d_loss


class _FastTSGenLossAdapter(torch.nn.Module):
    """Thin wrapper whose ``forward`` dispatches to ``FastTSCodec.generator_losses``.

    Same DDP purpose as ``train_codec._GenLossAdapter``: DDP's reducer only hooks params used in
    ``forward``; the codec's generator step is ``codec.generator_losses(...)``, so we DDP-wrap
    THIS adapter. ``self.codec`` is the underlying :class:`FastTSCodec` (used for the detached
    D-recon + gate + EMA + state_dict). The fast-TS signature keeps the δ-pair ``x_shift``.
    """

    def __init__(self, codec: FastTSCodec) -> None:
        super().__init__()
        self.codec = codec

    def forward(
        self,
        env_a: torch.Tensor,
        env_b: torch.Tensor,
        disc: torch.nn.Module,
        cfg: FastTSCodecConfig,
        step: int,
    ) -> Dict[str, torch.Tensor]:
        return self.codec.generator_losses(env_a, env_b, disc, cfg, step=step)


def _ddp_fastts_train_step(
    gen_module: torch.nn.Module,        # DDP-wrapped _FastTSGenLossAdapter (or bare)
    codec: FastTSCodec,                 # underlying raw codec
    disc: torch.nn.Module,              # DDP-wrapped discriminator (or bare)
    disc_raw: Env1DPatchGAN,            # underlying raw discriminator
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    env_a: torch.Tensor,
    env_b: torch.Tensor,
    cfg: FastTSCodecConfig,
    step: int,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """DDP-aware analogue of :func:`fastts_codec_train_step` (routes the G forward through DDP)."""
    codec.train()
    disc_raw.train()

    opt_g.zero_grad(set_to_none=True)
    g_terms = gen_module(env_a, env_b, disc_raw, cfg, step)
    g_terms["total"].backward()
    opt_g.step()

    opt_d.zero_grad(set_to_none=True)
    with torch.no_grad():
        recon = codec.forward(env_a)["recon"]
    d_loss = _fastts_discriminator_loss(disc, env_a, recon)
    d_loss.backward()
    opt_d.step()

    return g_terms, d_loss


# ------------------------------------------------------------------------------------- #
# fast-TS gate (mirrors spike.compute_gate for the envelope)
# ------------------------------------------------------------------------------------- #
@torch.no_grad()
def fastts_compute_gate(
    codec: FastTSCodec,
    eval_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    frame_seq: torch.Tensor,
    cfg: FastTSCodecConfig,
) -> Dict[str, object]:
    """Fast-TS analogue of :func:`spike.compute_gate` (§4.4 gate on envelope codes + recon).

    Parameters
    ----------
    codec : FastTSCodec
    eval_pairs : list of (env_a, env_b)
        Held-out δ-shift envelope pairs; used for **stability** (codes of env_a vs env_b) and
        **envelope decode-fidelity** (recon of env_a vs env_a).
    frame_seq : (B, n_frames, C, E)
        Consecutive world-model windows' envelopes; used for **persistence** (window t vs t+1)
        and **forecastability** (whole sequence of codes).
    cfg : FastTSCodecConfig

    Returns the same key set as :func:`spike.compute_gate` so :func:`spike.gate_score` and the
    trainer plumbing consume it unchanged: ``stability`` / ``persistence`` /
    ``forecastability`` / ``decode`` / ``utilization`` + the ``pass_*`` booleans.
    """
    was_training = codec.training
    codec.eval()

    stab_vals: List[float] = []
    dec_corr: List[float] = []
    dec_f1: List[float] = []
    dec_sharp: List[float] = []
    for env_a, env_b in eval_pairs:
        out_a = codec.forward(env_a)
        _, codes_b = codec.quantize(codec.encode(env_b))
        stab_vals.append(gate.stability(out_a["codes"], codes_b))
        dm = gate.fastts_decode_fidelity(out_a["recon"], env_a)
        dec_corr.append(dm["envelope_corr"])
        dec_f1.append(dm["peak_f1"])
        dec_sharp.append(dm["sharpness"])

    stability_val = float(sum(stab_vals) / len(stab_vals))
    decode = {
        "envelope_corr": float(sum(dec_corr) / len(dec_corr)),
        "peak_f1": float(sum(dec_f1) / len(dec_f1)),
        "sharpness": float(sum(dec_sharp) / len(dec_sharp)),
    }

    B, n_frames = frame_seq.shape[0], frame_seq.shape[1]
    flat = frame_seq.reshape(B * n_frames, *frame_seq.shape[2:])   # (B*Fr, C, E)
    _, codes_flat = codec.quantize(codec.encode(flat))
    codes_seq = codes_flat.reshape(B, n_frames, cfg.n_tok, cfg.fsq_dim)

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
# held-out streamed eval set (disjoint shots) for the gate
# ------------------------------------------------------------------------------------- #
def _stream_fastts_eval_data(
    eval_shots: Sequence[Union[str, int]],
    cfg: FastTSCodecConfig,
    *,
    data_dir: Union[str, Path],
    eval_batches: int,
    eval_batch_size: int,
    eval_frames: int,
    device: torch.device,
    seed: int,
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    """Materialize a small held-out gate set by streaming ``eval_shots`` on THIS rank only.

    Returns ``(eval_pairs, frame_seq)`` matching :func:`fastts_compute_gate`'s contract:
    ``eval_pairs`` = list of ``(env_a, env_b)`` batches; ``frame_seq`` = ``(B, n_frames, C, E)``
    consecutive-window envelope sequence. Reuses :class:`FastTSCodecPairDataset` for both (the
    δ-pair gives env_a; consecutive dataset indices give the frame sequence).
    """
    ds = FastTSCodecPairDataset(eval_shots, cfg, data_dir=data_dir, seed=seed)
    n_pairs = eval_batches * eval_batch_size
    n_avail = len(ds)
    if n_avail < n_pairs:
        raise RuntimeError(
            f"_stream_fastts_eval_data: only {n_avail} eval windows across "
            f"{len(list(eval_shots))} eval shots; need {n_pairs} (eval_batches={eval_batches} * "
            f"eval_batch_size={eval_batch_size}). Add more eval shots."
        )
    flat_pairs = [ds[i] for i in range(n_pairs)]
    eval_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for bi in range(eval_batches):
        chunk = flat_pairs[bi * eval_batch_size : (bi + 1) * eval_batch_size]
        a = torch.stack([p[0] for p in chunk], dim=0).to(device)
        b = torch.stack([p[1] for p in chunk], dim=0).to(device)
        eval_pairs.append((a, b))

    # consecutive-window envelope sequence for persistence / forecastability: env_a of blocks
    # of `eval_frames` consecutive dataset windows (already the non-overlapping 50 ms cores).
    bsz = max(2, eval_batch_size // 2)
    n_windows = eval_frames
    seqs: List[torch.Tensor] = []
    b = 0
    while len(seqs) < bsz and (b + 1) * n_windows <= n_avail:
        block = [ds[b * n_windows + k][0] for k in range(n_windows)]  # env_a per window
        seqs.append(torch.stack(block, dim=0))  # (n_windows, C, E)
        b += 1
    if len(seqs) < bsz:
        raise RuntimeError(
            f"_stream_fastts_eval_data: only {len(seqs)} consecutive {n_windows}-window "
            f"sequences across eval shots; need {bsz}. Add more eval shots."
        )
    frame_seq = torch.stack(seqs, dim=0).to(device)  # (bsz, n_windows, C, E)
    return eval_pairs, frame_seq


# ------------------------------------------------------------------------------------- #
# the streaming trainer (mirrors train_codec; reuses the SAME DDP + best-ckpt + gate path)
# ------------------------------------------------------------------------------------- #
def train_fastts_codec(
    cfg: FastTSCodecConfig,
    train_shots: Sequence[Union[str, int]],
    eval_shots: Sequence[Union[str, int]],
    *,
    steps: int,
    eval_every: int,
    batch_size: int,
    num_workers: int,
    modality: str = FASTTS_MODALITY,
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
    """Streaming, DDP-capable fast-TS codec training loop.

    The fast-TS analogue of :func:`ignite.train_codec.train_codec` — it reuses the SAME
    infrastructure (``_DDPState`` init/wrap, ``spike._EMA``, ``spike.gate_score`` /
    ``spike._fmt_gate`` / ``spike._write_gate_json`` / ``spike._save_{checkpoint,best_checkpoint,
    ema_checkpoint}`` best-ckpt+EMA+gate path) and differs only where fast-TS differs: a
    :class:`FastTSCodec` + :class:`Env1DPatchGAN`, the ``(env_a, env_b)`` envelope δ-pair loader,
    the :func:`_ddp_fastts_train_step`, and :func:`fastts_compute_gate`. Rank-0 only logs +
    writes ``codec_last.pt`` / ``codec_best.pt`` / (optional) ``codec_ema.pt`` /
    ``gate_<step>.json`` (identical filenames + score contract as spectro/video).
    """
    if steps < 1:
        raise ValueError("train_fastts_codec: steps must be >= 1")
    ddp = ddp if ddp is not None else _DDPState()
    device = ddp.device
    torch.manual_seed(seed + ddp.rank)

    out_path = Path(out_dir) if out_dir is not None else None
    if out_path is not None and ddp.is_main:
        out_path.mkdir(parents=True, exist_ok=True)

    codec = FastTSCodec(cfg).to(device)
    disc_raw = Env1DPatchGAN(cfg).to(device)

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

    gen_module = ddp.wrap(_FastTSGenLossAdapter(codec))
    disc = ddp.wrap(disc_raw)

    ema_shadow = spike._EMA(codec, ema_decay) if ema else None

    eval_pairs, frame_seq = _stream_fastts_eval_data(
        eval_shots, cfg,
        data_dir=data_dir, eval_batches=eval_batches, eval_batch_size=eval_batch_size,
        eval_frames=eval_frames, device=device, seed=seed + 777,
    )

    train_ds = FastTSCodecPairDataset(
        train_shots, cfg,
        data_dir=data_dir, modality=modality, seed=seed,
        lengths_cache_path=lengths_cache_path,
    )
    loader = make_fastts_loader(
        train_ds, batch_size=batch_size, num_workers=num_workers,
        rank=ddp.rank, world_size=ddp.world_size, seed=seed, shuffle=True,
    )
    stream = _epoch_cycler(loader)

    best_score = float("-inf")
    best_step: Optional[int] = None
    final_gate: Dict[str, object] = {}

    for local_step in range(steps):
        step = start_step + local_step
        env_a, env_b = next(stream)
        env_a = env_a.to(device, non_blocking=True)
        env_b = env_b.to(device, non_blocking=True)

        g_terms, d_loss = _ddp_fastts_train_step(
            gen_module, codec, disc, disc_raw, opt_g, opt_d,
            env_a, env_b, cfg, step=step,
        )
        if ema_shadow is not None:
            ema_shadow.update(codec)

        is_last = local_step == steps - 1
        if (step % eval_every == 0) or is_last:
            g = fastts_compute_gate(codec, eval_pairs, frame_seq, cfg)
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
                            log_fn(f"[train_fastts_codec] new BEST score={score:+.4f} @ step {step}")
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
        prog="python -m tokamak_foundation_model.ignite.fastts_train",
        description="IGNITE Phase-A STREAMING, DDP fast-TS (filterscopes) ELM-envelope codec "
                    "trainer (statistics-first; encodes the ELM ACTIVITY ENVELOPE, not spikes).",
    )
    p.add_argument("--modality", type=str, default=FASTTS_MODALITY,
                   choices=[FASTTS_MODALITY],
                   help="Fast-TS codec modality (only 'filterscopes').")
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
                   help="Consecutive windows per persistence/forecastability sequence.")
    p.add_argument("--out_dir", type=str, required=True,
                   help="Output dir for gate_<step>.json / codec_{last,best,ema}.pt / summary.")
    p.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR,
                   help="Directory of {shot}_processed.h5 files.")
    p.add_argument("--lengths_cache_dir", type=str, default=None,
                   help="Directory for the per-file chunk-length sidecar cache "
                        "(codec_filterscopes_lengths.pt); reuses the parent dataset's cache.")
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

    channels = fastts_channels(args.modality)
    cfg = FastTSCodecConfig(channels=channels)
    if args.entropy_weight is not None:
        cfg.entropy_weight = float(args.entropy_weight)
    if args.consistency_weight is not None:
        cfg.consistency_weight = float(args.consistency_weight)

    if args.shots is not None:
        all_shots = [s.strip() for s in args.shots.split(",") if s.strip()]
    else:
        all_shots = spike.discover_shots(args.data_dir)
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
            f"[fastts_train] modality={args.modality} channels={channels} "
            f"env_bins={cfg.env_bins} n_tok={cfg.n_tok} "
            f"train_shots={len(train_shots)} eval_shots={len(eval_shots)} "
            f"world_size={ddp.world_size} batch_size={args.batch_size} "
            f"num_workers={args.num_workers} steps={args.steps} device={device}",
            flush=True,
        )

    t0 = time.time()
    final_gate = train_fastts_codec(
        cfg,
        train_shots,
        eval_shots,
        steps=args.steps,
        eval_every=args.eval_every,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        modality=args.modality,
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
            "env_bins": cfg.env_bins,
            "n_tok": cfg.n_tok,
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
            f"[fastts_train] DONE steps={final_gate.get('steps')} "
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
