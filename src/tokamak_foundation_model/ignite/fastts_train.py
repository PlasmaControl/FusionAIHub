"""IGNITE Phase-A fast-TS (filterscopes) codec: streaming, DDP-capable trainer.

The fast-TS counterpart of :mod:`ignite.train_codec` (spectrogram and video). It reuses
the shared IGNITE infrastructure and adds only the fast-TS pieces. ``cfg.target``
selects envelope or raw mode (see ``config.FastTSCodecConfig``); :func:`main` builds the
default envelope config.

* Data: :class:`FastTSCodecPairDataset` is a thin subclass of
  :class:`~tokamak_foundation_model.data.multi_file_dataset.TokamakMultiFileDataset`,
  like :class:`ignite.train_codec.CodecPairDataset`. It keeps the parent's global index
  -> ``(file_idx, chunk_idx)`` map, per-worker LRU HDF5 handle cache (with pickling into
  DataLoader workers), length-cache sidecar and ``num_workers`` plumbing, and overrides
  only :meth:`_getitem_standard` to return a ``(win_a, win_b)`` δ-shift pair for
  ``filterscopes``. ``data_loader.py`` and ``multi_file_dataset.py`` are not modified.
* Trainer: the same generator/discriminator alternation as the spectrogram codec, a
  ``generator_losses`` step through a DDP-wrapped adapter (so DDP's reducer sees the
  generator gradients), then a discriminator step on the detached reconstruction.
  ``_DDPState``, ``spike._EMA``, ``spike.gate_score`` / ``spike._fmt_gate`` /
  ``spike._save_{checkpoint,best_checkpoint,ema_checkpoint}`` / ``spike._write_gate_json``
  and the file-locality samplers with ``_epoch_cycler`` come from
  :mod:`ignite.train_codec` and :mod:`ignite.spike`.
* Gate: :func:`fastts_compute_gate` mirrors :func:`spike.compute_gate`, with stability
  on the δ-shift pair, decode fidelity from :func:`gate.fastts_decode_fidelity` and
  :func:`gate.full_fastts_metrics` (nRMSE and trivial baselines), and persistence,
  forecastability and utilization on a sequence of consecutive windows. It returns the
  same keys, so :func:`spike.gate_score` (with its reconstruction-floor
  disqualification) reads it unchanged.

In raw mode the δ-pair is only reported. A sample-wise codec has to carry spike timing,
so ``consistency_weight`` is 0 there, and the pair only lets the gate report how far the
codes move under a shift (``spike.gate_score`` does not read ``stability``).

Reuse boundary (§7): ``torch``, sibling ``ignite`` modules and the read-only data
pipeline (through ``TokamakMultiFileDataset``). No FAITH model code.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
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
    _stratified_draw,
)

DEFAULT_DATA_DIR = spike.DEFAULT_DATA_DIR

# Per-channel raw mean/std the data_loader uses to standardize filterscopes (the file
# every eval/train script passes as --stats_path). The codec standardizes its raw input
# with the filterscopes 'raw' entry.
DEFAULT_STATS_PATH = "/lustre/orion/fus187/proj-shared/foundation_model_meta/preprocessing_stats.pt"

# The fast-TS modality, named as in TokamakH5Dataset.SIGNAL_CONFIGS (10 kHz,
# channels_to_use=slice(0,8), no STFT).
FASTTS_MODALITY = "filterscopes"


def load_fastts_channel_stats(
    stats_path: Union[str, Path] = DEFAULT_STATS_PATH,
    modality: str = FASTTS_MODALITY,
) -> Tuple[List[float], List[float]]:
    """Per-channel raw mean/std for ``modality``, sliced by its ``channels_to_use``.

    Reads the ``preprocessing_stats.pt`` the data_loader uses and returns the
    per-channel ``mean`` / ``std`` from the ``raw`` sub-dict (filterscopes uses
    method="standardize", which the loader maps to ``raw``), sliced so the length
    matches the codec's ``C`` channels. NaNs become mean 0 / std 1, as in
    ``data_loader._update_preprocessing_stats``.

    Returns ``(mean_list, std_list)``, each of length ``C``. Raises if the modality or
    the stats are missing, so a wrong ``--stats_path`` fails instead of silently
    skipping the standardization.
    """
    from tokamak_foundation_model.data.data_loader import TokamakH5Dataset

    cfg_sig = next(
        (c for c in TokamakH5Dataset.SIGNAL_CONFIGS if c.name == modality), None
    )
    if cfg_sig is None:
        raise ValueError(f"modality {modality!r} not in SIGNAL_CONFIGS")

    stats = torch.load(str(stats_path), map_location="cpu", weights_only=False)
    if modality not in stats:
        raise KeyError(f"{modality!r} not in preprocessing_stats at {stats_path}")
    entry = stats[modality]
    # method="standardize" means the loader uses the 'raw' sub-dict, not 'log'.
    sub = entry.get("raw", entry) if ("raw" in entry or "log" in entry) else entry
    if "mean" not in sub or "std" not in sub:
        raise KeyError(f"{modality!r} raw stats missing mean/std at {stats_path}")

    mean = np.asarray(sub["mean"], dtype=np.float64)
    std = np.asarray(sub["std"], dtype=np.float64)
    mean[np.isnan(mean)] = 0.0   # as in _update_preprocessing_stats
    std[np.isnan(std)] = 1.0
    ch = cfg_sig.channels_to_use
    if ch is not None:
        mean = mean[ch]
        std = std[ch]
    return mean.astype(np.float32).tolist(), std.astype(np.float32).tolist()


def fastts_channels(modality: str = FASTTS_MODALITY) -> int:
    """Channels actually loaded for the fast-TS ``modality`` after ``channels_to_use``.

    Reads ``TokamakH5Dataset.SIGNAL_CONFIGS`` (like ``train_codec.modality_channels``,
    for the non-STFT filterscopes signal). Returns 8 for ``filterscopes``
    (``channels_to_use=slice(0, 8)``).
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


class FastTSCodecPairDataset(TokamakMultiFileDataset):
    """``(win_a, win_b)`` δ-shift pairs for the ``filterscopes`` modality.

    A thin subclass of :class:`~tokamak_foundation_model.data.multi_file_dataset.\
TokamakMultiFileDataset`, built like :class:`ignite.train_codec.CodecPairDataset`. It
    keeps the parent's streaming machinery (global index -> ``(file_idx, chunk_idx)``
    map, per-worker LRU HDF5 handle cache, length-cache sidecar, ``num_workers``) and
    overrides only :meth:`_getitem_standard`.

    Windowing
    ---------
    The parent's ``__getitem__`` maps a global ``idx`` to ``(file_idx, chunk_idx)`` by
    binary search over ``_cumulative_lengths``, sets ``self.h5_file`` to the per-worker
    handle, then calls ``_getitem_standard(chunk_idx)``. Windows start at ``t_start =
    warmup_s + chunk_idx * step_size_s``, the parent's formula. The parent is built with
    ``chunk_duration_s = CHUNK_S + δ_max`` so its length computation guarantees the
    shifted window ``[t_start, t_start + CHUNK_S + δ_max]`` fits in the shot, and with
    ``step_size_s = CHUNK_S`` so windows are 50 ms apart.

    Pair construction
    -----------------
    ``_getitem_standard`` loads the δ-extended window with the parent's
    :meth:`TokamakH5Dataset._load_signal_raw` (resampled to ``target_fs == FASTTS_FS``,
    zero/NaN-padded) and builds the pair with
    :func:`ignite.data.fastts_shift_pair_windows` (δ-shift plus per-channel
    standardization). Degenerate windows (all NaN, near flat or past the end of the
    data) are redrawn from nearby chunks, as in CodecPairDataset.
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
        # Activity-stratified sampling (0 = off).
        self.min_activity = float(getattr(cfg, "min_activity", 0.0))
        self.active_bias = float(getattr(cfg, "active_bias", 0.0))

        # Window A covers [t0, t0+CHUNK_S] and B is shifted by up to δ_max, so reserve
        # the whole span through chunk_duration_s; step_size_s stays CHUNK_S.
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

    # The one overridden hook: the per-item δ-pair transform.
    def _getitem_standard(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        """Return one ``(win_a, win_b)`` δ-shift pair for window ``idx``.

        Called by the parent's ``__getitem__`` after it has mapped the global index to
        ``(file_idx, chunk_idx)`` and set ``self.h5_file`` to this shot's handle;
        ``idx`` is the within-shot ``chunk_idx``.

        With ``cfg.active_bias > 0``, :func:`ignite.train_codec._stratified_draw` biases
        the draw toward windows whose std is ``>= cfg.min_activity``. On 4 held-out
        shots the top std quartile is by far the most reconstructable (0-1 kHz lowpass
        oracle nRMSE 0.40 vs 0.68-0.73 below), so this favours windows with structure.
        With ``active_bias == 0`` (default) it is the plain build with degenerate
        redraws.
        """
        return _stratified_draw(
            idx, self._draw_valid_pair, lambda p: float(p[0].std()),
            lambda: self._chunks_in_current_shot(idx),
            min_activity=self.min_activity, active_bias=self.active_bias,
            max_tries=self.max_tries, seed=self.pair_seed,
        )

    def _draw_valid_pair(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build the δ-pair for window ``idx``, redrawing degenerate windows."""
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
        # Last resort, when the whole shot is degenerate: a silent pair.
        C = self._num_channels()
        cc = self.codec_cfg
        z = torch.zeros((C, cc.window if cc.is_raw else cc.env_bins))
        return z, z.clone()

    # Helpers (they use the parent's state)
    def _build_pair(self, chunk_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Build the δ-pair for a within-shot ``chunk_idx``; None if the window is degenerate."""
        step = getattr(self, "step_size_s", self.chunk_duration_s)
        warmup = getattr(self, "warmup_s", 0.0)
        t_start = warmup + chunk_idx * step          # the parent's windowing
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
        # `raw` already starts at t_start, so the pair slicer gets t0 = 0.0.
        win_a, win_b = data.fastts_shift_pair_windows(
            raw, t0=0.0, cfg=self.codec_cfg, delta_ms=d
        )
        return win_a, win_b

    def _chunks_in_current_shot(self, chunk_idx: int) -> int:
        """Chunks in the shot on ``self.h5_file`` (best effort; bounds the redraw)."""
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
    """Stack ``(win_a, win_b)`` pairs into batched ``(B, C, W)`` tensors."""
    win_a = torch.stack([a for a, _ in batch], dim=0)
    win_b = torch.stack([b for _, b in batch], dim=0)
    return win_a, win_b


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

    Same samplers as :func:`ignite.train_codec.make_codec_loader` (file-locality
    ``DistributedTwoLevelSampler`` under DDP, ``TwoLevelSampler`` in a single process);
    only the collate differs.
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


# Generator/discriminator alternation (as in spike.codec_train_step)
def _fastts_discriminator_loss(
    disc: torch.nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
) -> torch.Tensor:
    """Hinge discriminator loss for the 1-D PatchGAN.

    Same math as :func:`losses.discriminator_loss` (``E[relu(1-D(real))] +
    E[relu(1+D(fake))]`` over the per-scale score maps), but without a config argument,
    like ``train_codec._video_discriminator_loss``.
    """
    real_maps = _as_score_list(disc(real))
    fake_maps = _as_score_list(disc(fake.detach() if fake.requires_grad else fake))
    return _hinge_real(real_maps) + _hinge_fake(fake_maps)


def fastts_codec_train_step(
    codec: FastTSCodec,
    disc: Env1DPatchGAN,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    win_a: torch.Tensor,
    win_b: torch.Tensor,
    cfg: FastTSCodecConfig,
    step: int,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """One generator and discriminator step on a δ-shift pair, without DDP.

    The fast-TS version of :func:`spike.codec_train_step`: a generator step on
    ``generator_losses``, then a discriminator step on a freshly computed, detached
    reconstruction. Used by the tests.
    """
    codec.train()

    opt_g.zero_grad(set_to_none=True)
    g_terms = codec.generator_losses(win_a, win_b, disc, cfg, step=step)
    g_terms["total"].backward()
    # Skip the update if the loss is non-finite.
    if spike.is_step_diverged(g_terms["total"]):
        spike.note_skipped_step()
    else:
        opt_g.step()

    opt_d.zero_grad(set_to_none=True)
    with torch.no_grad():
        recon = codec.forward(win_a)["recon"]
    d_loss = _fastts_discriminator_loss(disc, win_a, recon)
    d_loss.backward()
    if spike.is_step_diverged(d_loss):
        spike.note_skipped_step()
    else:
        opt_d.step()

    return g_terms, d_loss


class _FastTSGenLossAdapter(torch.nn.Module):
    """Thin wrapper whose ``forward`` dispatches to ``FastTSCodec.generator_losses``.

    As ``train_codec._GenLossAdapter``: DDP's reducer only hooks parameters used in
    ``forward``, and the generator step is ``codec.generator_losses(...)``, so this
    adapter is what gets DDP-wrapped. ``self.codec`` is the underlying
    :class:`FastTSCodec` (used for the discriminator's reconstruction, the gate, EMA and
    state_dict). The signature keeps the shifted window ``win_b``.
    """

    def __init__(self, codec: FastTSCodec) -> None:
        super().__init__()
        self.codec = codec

    def forward(
        self,
        win_a: torch.Tensor,
        win_b: torch.Tensor,
        disc: torch.nn.Module,
        cfg: FastTSCodecConfig,
        step: int,
    ) -> Dict[str, torch.Tensor]:
        return self.codec.generator_losses(win_a, win_b, disc, cfg, step=step)


def _ddp_fastts_train_step(
    gen_module: torch.nn.Module,        # DDP-wrapped _FastTSGenLossAdapter (or bare)
    codec: FastTSCodec,                 # underlying codec
    disc: torch.nn.Module,              # DDP-wrapped discriminator (or bare)
    disc_raw: Env1DPatchGAN,            # underlying discriminator
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    win_a: torch.Tensor,
    win_b: torch.Tensor,
    cfg: FastTSCodecConfig,
    step: int,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """DDP version of :func:`fastts_codec_train_step` (generator runs through DDP)."""
    codec.train()
    disc_raw.train()

    opt_g.zero_grad(set_to_none=True)
    g_terms = gen_module(win_a, win_b, disc_raw, cfg, step)
    g_terms["total"].backward()
    # Every rank runs backward, then all ranks skip opt_g.step() if any rank's loss is
    # non-finite.
    if spike.is_step_diverged(g_terms["total"]):
        spike.note_skipped_step()
    else:
        opt_g.step()

    opt_d.zero_grad(set_to_none=True)
    with torch.no_grad():
        recon = codec.forward(win_a)["recon"]
    d_loss = _fastts_discriminator_loss(disc, win_a, recon)
    d_loss.backward()
    if spike.is_step_diverged(d_loss):
        spike.note_skipped_step()
    else:
        opt_d.step()

    return g_terms, d_loss


# Gate (as in spike.compute_gate)
@torch.no_grad()
def fastts_compute_gate(
    codec: FastTSCodec,
    eval_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    frame_seq: torch.Tensor,
    cfg: FastTSCodecConfig,
) -> Dict[str, object]:
    """Fast-TS version of :func:`spike.compute_gate`: reconstruction and code metrics.

    Parameters
    ----------
    codec : FastTSCodec
    eval_pairs : list of (win_a, win_b)
        Held-out δ-shift window pairs, for stability (codes of win_a vs win_b; reported
        only, ``spike.gate_score`` does not read it) and decode fidelity (recon of win_a
        vs win_a).
    frame_seq : (B, n_frames, C, W)
        Consecutive world-model windows, for persistence (window t vs t+1) and
        forecastability (the whole code sequence).
    cfg : FastTSCodecConfig

    Returns the keys of :func:`spike.compute_gate` (``stability``, ``persistence``,
    ``forecastability``, ``decode``, ``utilization`` and the ``pass_*`` booleans), so
    :func:`spike.gate_score` and the trainer read it unchanged. ``decode`` also holds:

      * ``fastts_nrmse``: RMSE / std(target) per (window, channel) over samples,
        averaged. 0.0 is perfect; 1.0 is predicting that pair's own mean.
      * ``fastts_corr`` / ``fastts_global_nrmse`` / ``fastts_valid_frac``.
      * Envelope mode: :func:`gate.fastts_envelope_metrics` (see the comment below).
      * Raw mode: ``fastts_ssim_cs`` / ``fastts_ssim_contrast`` /
        ``fastts_ssim_structure`` / ``fastts_ms_ssim`` / ``fastts_std_ratio`` from
        :func:`gate.fastts_ssim_metrics`. nRMSE is a floor, not a ranking key: its
        minimizer is the conditional mean, so it rewards shrinking the amplitude, and
        ``fastts_ssim_contrast`` and ``fastts_std_ratio`` (1.0 = right dynamic range)
        detect that. Also ``ref_*`` from :func:`gate.fastts_structural_references`
        (self, +noise, smoothed, 0.4x-shrunk and flat references scored on the same
        batch, which re-validates the metric each time).
      * ``base_*``: :func:`gate.trivial_fastts_baselines` on the same batch. Sanity
        checks: ``base_self_fastts_nrmse`` must be 0.0 and ``base_wcmean_fastts_nrmse``
        exactly 1.0. ``base_tmean_fastts_nrmse`` is the baseline to beat; for a 1-D
        window it is the same predictor as ``wcmean`` and also reads 1.0.
        ``base_binmean100_*`` is the envelope resolution (10 ms piecewise constant).

    ``decode["envelope_corr"]``, which ``spike.gate_score`` reads, is the per-(window,
    channel) Pearson correlation over the last axis (``gate.fastts_decode_fidelity``),
    i.e. over samples in raw mode. That is much smaller than an envelope correlation,
    hence the raw-mode ``gate_recon_floor`` of 0.02.
    """
    was_training = codec.training
    codec.eval()

    stab_vals: List[float] = []
    dec_corr: List[float] = []
    dec_f1: List[float] = []
    dec_sharp: List[float] = []
    recon_all: List[torch.Tensor] = []
    target_all: List[torch.Tensor] = []
    for win_a, win_b in eval_pairs:
        out_a = codec.forward(win_a)
        _, codes_b = codec.quantize(codec.encode(win_b))
        stab_vals.append(gate.stability(out_a["codes"], codes_b))
        dm = gate.fastts_decode_fidelity(out_a["recon"], win_a)
        dec_corr.append(dm["envelope_corr"])
        dec_f1.append(dm["peak_f1"])
        dec_sharp.append(dm["sharpness"])
        recon_all.append(out_a["recon"].detach().float().cpu())
        target_all.append(win_a.detach().float().cpu())

    stability_val = float(sum(stab_vals) / len(stab_vals))
    decode = {
        "envelope_corr": float(sum(dec_corr) / len(dec_corr)),
        "peak_f1": float(sum(dec_f1) / len(dec_f1)),
        "sharpness": float(sum(dec_sharp) / len(dec_sharp)),
    }
    # nRMSE and the trivial baselines over the whole eval set at once (not a mean of
    # per-batch means), so the baselines are scored on exactly the codec's target array.
    r_cat = torch.cat(recon_all, dim=0).numpy()
    t_cat = torch.cat(target_all, dim=0).numpy()
    decode.update(gate.full_fastts_metrics(r_cat, t_cat))
    decode.update(gate.trivial_fastts_baselines(t_cat))
    # The SSIM block is raw-mode only: a 17-sample (1.7 ms) window means nothing on a
    # 5-bin envelope.
    if not cfg.is_raw:
        # Envelope mode: add the pooled, global-std metrics
        # (gate.fastts_envelope_metrics), which are what this codec is judged on.
        # fastts_nrmse normalizes each window by its own std, and since 96% of the
        # envelope variance is the between-window DC level, that makes the level free.
        # env_nrmse normalizes by each channel's global std, so a constant scores 1.0
        # and the level counts. env_base_ratematched is the trivial window-mean encoder
        # at this codec's bit budget (0.19, against 0.53 for the plain codec), the
        # number to beat; env_std_ratio and env_burst_keep check amplitude, which nRMSE
        # cannot. spike.gate_score reads none of these keys.
        decode.update(gate.fastts_envelope_metrics(
            r_cat, t_cat, bits=cfg.n_tok * math.log2(float(cfg.codebook_size))))
        return _finish_fastts_gate(codec, was_training, decode, frame_seq, cfg,
                                   stability=stability_val)
    # Structural metrics and their references (see the docstring). The ref_* rows are
    # built from the target alone, so they re-validate the metric on this batch.
    _sw = int(getattr(cfg, "ssim_win", 17))
    decode.update(gate.fastts_ssim_metrics(r_cat, t_cat, win=_sw))
    decode.update(gate.fastts_structural_references(t_cat, win=_sw))

    return _finish_fastts_gate(codec, was_training, decode, frame_seq, cfg,
                               stability=stability_val)


@torch.no_grad()
def _finish_fastts_gate(codec, was_training, decode, frame_seq, cfg, stability: float = 0.0):
    """Code-sequence half of the gate (persistence / forecastability / utilization).

    Split out so the envelope path can return before the raw-only SSIM block without
    duplicating this.
    """
    B, n_frames = frame_seq.shape[0], frame_seq.shape[1]
    flat = frame_seq.reshape(B * n_frames, *frame_seq.shape[2:])   # (B*Fr, C, E|W)
    _, codes_flat = codec.quantize(codec.encode(flat))
    codes_seq = codes_flat.reshape(B, n_frames, cfg.n_tok, cfg.fsq_dim)

    persistence_val = gate.persistence(codes_seq[:, 0], codes_seq[:, 1])
    forecast = gate.forecastability(codes_seq, transition_mask=None)
    util = gate.utilization(codes_seq, cfg=cfg)

    if was_training:
        codec.train()

    return {
        "stability": stability,
        "persistence": float(persistence_val),
        "forecastability": forecast,
        "decode": decode,
        "utilization": util,
        "pass_stability": bool(stability >= cfg.gate_stability),
        "pass_persistence": bool(persistence_val >= cfg.gate_persistence),
        "pass_utilization": bool(not util["collapsed"]),
    }


# Held-out eval set for the gate (shots disjoint from training)
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
    """Build a small held-out gate set by streaming ``eval_shots`` on this rank only.

    Returns ``(eval_pairs, frame_seq)`` as :func:`fastts_compute_gate` expects:
    ``eval_pairs`` is a list of ``(win_a, win_b)`` batches and ``frame_seq`` a
    ``(B, n_frames, C, W)`` sequence of consecutive windows. Both come from
    :class:`FastTSCodecPairDataset` (win_a of consecutive dataset indices gives the
    sequence).
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

    # Sequences for persistence and forecastability: win_a of `eval_frames` consecutive
    # dataset windows (non-overlapping 50 ms windows).
    bsz = max(2, eval_batch_size // 2)
    n_windows = eval_frames
    seqs: List[torch.Tensor] = []
    b = 0
    while len(seqs) < bsz and (b + 1) * n_windows <= n_avail:
        block = [ds[b * n_windows + k][0] for k in range(n_windows)]  # win_a per window
        seqs.append(torch.stack(block, dim=0))  # (n_windows, C, W)
        b += 1
    if len(seqs) < bsz:
        raise RuntimeError(
            f"_stream_fastts_eval_data: only {len(seqs)} consecutive {n_windows}-window "
            f"sequences across eval shots; need {bsz}. Add more eval shots."
        )
    frame_seq = torch.stack(seqs, dim=0).to(device)  # (bsz, n_windows, C, W)
    return eval_pairs, frame_seq


# Streaming trainer (as train_codec, with the same DDP, best-checkpoint and gate path)
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

    The fast-TS version of :func:`ignite.train_codec.train_codec`, with the same
    infrastructure (``_DDPState``, ``spike._EMA``, ``spike.gate_score`` /
    ``spike._fmt_gate`` / ``spike._write_gate_json`` /
    ``spike._save_{checkpoint,best_checkpoint,ema_checkpoint}``). Only the codec and
    discriminator (:class:`FastTSCodec`, :class:`Env1DPatchGAN`), the δ-pair loader,
    :func:`_ddp_fastts_train_step` and :func:`fastts_compute_gate` differ. Rank 0 logs
    and writes ``codec_last.pt``, ``codec_best.pt``, optionally ``codec_ema.pt``, and
    ``gate_<step>.json``, with the same file names and score as the other codecs.
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

    best_score = spike.resume_best_score(out_path)  # best score across resumed legs
    best_step: Optional[int] = None
    final_gate: Dict[str, object] = {}
    spike.reset_skipped_steps()  # count of skipped (non-finite) steps in this run

    for local_step in range(steps):
        step = start_step + local_step
        win_a, win_b = next(stream)
        win_a = win_a.to(device, non_blocking=True)
        win_b = win_b.to(device, non_blocking=True)

        g_terms, d_loss = _ddp_fastts_train_step(
            gen_module, codec, disc, disc_raw, opt_g, opt_d,
            win_a, win_b, cfg, step=step,
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
            score = spike.gate_score(g, recon_floor=cfg.gate_recon_floor, hard_min_codes=getattr(cfg, "gate_hard_min_codes", 8))
            g["score"] = score
            is_best = score > best_score
            g["is_best"] = bool(is_best)
            if is_best:
                best_score = score
                best_step = step

            g["skipped_steps"] = spike.skipped_steps()
            if ddp.is_main:
                if log_fn is not None:
                    log_fn(
                        spike._fmt_gate(step, g)
                        + f" g_total={g['g_total']:+.4f} d_loss={g['d_loss']:.4f}"
                        + f" adv_lam={g['adaptive_weight']:.4g} score={score:+.4f}"
                        + f" skipped={g['skipped_steps']}"
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


# CLI
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tokamak_foundation_model.ignite.fastts_train",
        description="IGNITE Phase-A streaming DDP trainer for the fast-TS "
                    "(filterscopes) codec. The default config encodes the ELM "
                    "activity envelope, not spikes.",
    )
    p.add_argument("--modality", type=str, default=FASTTS_MODALITY,
                   choices=[FASTTS_MODALITY],
                   help="Fast-TS codec modality (only 'filterscopes').")
    p.add_argument("--shots", type=str, default=None,
                   help="Explicit comma-separated train shot list (overrides --n_shots).")
    p.add_argument("--n_shots", type=int, default=1000,
                   help="Number of shots to auto-discover for the train stream.")
    p.add_argument("--eval_n_shots", type=int, default=16,
                   help="Held-out shots for the gate eval set (not in training).")
    p.add_argument("--steps", type=int, default=20000, help="Training/gate steps.")
    p.add_argument("--eval_every", type=int, default=1000, help="Gate/checkpoint cadence.")
    p.add_argument("--batch_size", type=int, default=8, help="Pairs per batch per rank.")
    p.add_argument("--num_workers", type=int, default=6,
                   help="DataLoader parallel workers per rank (streaming).")
    p.add_argument("--lr", type=float, default=1e-3, help="Codec (generator) Adam LR.")
    p.add_argument("--disc_lr", type=float, default=None, help="Discriminator LR (default lr).")
    p.add_argument("--ema", action="store_true", help="Keep and save an EMA codec.")
    p.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay in [0,1).")
    p.add_argument("--eval_batches", type=int, default=8, help="Held-out gate δ-pair batches.")
    p.add_argument("--eval_batch_size", type=int, default=4, help="Pairs per eval batch.")
    p.add_argument("--eval_frames", type=int, default=6,
                   help="Consecutive windows per persistence/forecastability sequence.")
    p.add_argument("--out_dir", type=str, required=True,
                   help="Output dir for gate_<step>.json / codec_{last,best,ema}.pt / summary.")
    p.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR,
                   help="Directory of {shot}_processed.h5 files.")
    p.add_argument("--stats_path", type=str, default=DEFAULT_STATS_PATH,
                   help="preprocessing_stats.pt with the per-channel raw filterscopes "
                        "mean/std used to standardize the input (the file the "
                        "data_loader and eval scripts use). '' disables "
                        "standardization (raw ~1e15 input; debugging only).")
    p.add_argument("--lengths_cache_dir", type=str, default=None,
                   help="Directory for the per-file chunk-length sidecar cache "
                        "(codec_filterscopes_lengths.pt), shared with the parent "
                        "dataset.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", type=str, default=None,
                   help="Optional codec_last.pt to resume codec, discriminator, "
                        "optimizers and step from.")
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
    # Per-modality activity-bias overrides, shared with train_codec.main so filterscopes
    # gets the same settings from either entry point.
    from .train_codec import apply_activity_overrides
    apply_activity_overrides(cfg, args.modality, log_fn=(print if ddp.is_main else None))

    # Inject the per-channel raw mean/std so the input is standardized to ~O(1), as the
    # data_loader does, before detrend/rectify/RMS/log1p. Unstandardized ~1e15 input
    # pins log1p at its ceiling and collapses the codec to one code. --stats_path=''
    # disables it (debugging only).
    if args.stats_path:
        mean, std = load_fastts_channel_stats(args.stats_path, args.modality)
        cfg.channel_mean = mean
        cfg.channel_std = std
        if ddp.is_main:
            print(
                f"[fastts_train] raw standardization ON: per-channel raw stats from "
                f"{args.stats_path} (C={len(mean)}, std range "
                f"[{min(std):.3e}, {max(std):.3e}])",
                flush=True,
            )
    elif ddp.is_main:
        print("[fastts_train] WARNING: --stats_path='' -> raw standardization OFF "
              "(unstandardized ~1e15 input). Debugging only.", flush=True)

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
            f"window={cfg.window} patch_w={cfg.patch_w} n_tok={cfg.n_tok} "
            f"bits/value={cfg.bits_per_value:.4f} "
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
            "window": cfg.window,
            "patch_w": cfg.patch_w,
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
