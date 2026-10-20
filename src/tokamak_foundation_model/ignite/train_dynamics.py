"""Trainer, frame-code cache builder and smoke test of the IGNITE dynamics model.

The frozen codecs encode every 50 ms window of each modality to flat FSQ codes,
which :func:`precompute_frame_codes` caches per shot together with the frame-mean
actuators. :func:`train` then trains :class:`~.maskgit.MaskGITDynamics` on windows of
that cache, with scheduled sampling, optional loss masks and periodic validation.

Environment
-----------
IGNITE_ACT_GLOBAL
    Directory of dataset-wide normalised actuators (built by
    ``models/ignite_bandpower/build_actglobal.py``). When set, they replace the
    per-shot z-scored actuators of the cache, whose whole-shot statistics leak the
    future. A shot missing from it is an error. Read once at import.
IGNITE_DETECT_ANOMALY
    Enable ``torch.autograd.set_detect_anomaly`` during training.
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import json
import math
import os
import random
import signal
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from . import spike
from . import train_codec as tc
from .dynamics_config import FROZEN_MODALITIES, DynamicsConfig, ModalitySpec
from .maskgit import MaskGITDynamics

# Fallback checkpoints of the frozen codec set: name -> (family, repo-relative path).
FROZEN_CODEC_CKPTS: Dict[str, Tuple[str, str]] = {
    "ece": ("spectro", "eval_runs/ignite_d5_ece/codec_best.pt"),
    "bes": ("spectro", "eval_runs/ignite_d5_bes/codec_best.pt"),
    "mhr": ("spectro", "eval_runs/ignite_d5_mhr/codec_best.pt"),
    "co2": ("spectro", "eval_runs/ignite_co2_rawstd192/codec_best.pt"),
    "tangtv_lower": ("video", "eval_runs/ignite_d4_tangtv_lower/codec_best.pt"),
    "tangtv_upper": ("video", "eval_runs/ignite_d4_tangtv_upper/codec_best.pt"),
    "ts_core_density": ("slowts", "eval_runs/ignite_d5_ts_core_density/codec_best.pt"),
    "ts_core_temp": ("slowts", "eval_runs/ignite_d5_ts_core_temp/codec_best.pt"),
    "ts_tangential_density": (
        "slowts",
        "eval_runs/ignite_d5_ts_tangential_density/codec_best.pt",
    ),
    "ts_tangential_temp": (
        "slowts",
        "eval_runs/ignite_d5_ts_tangential_temp/codec_best.pt",
    ),
    "cer_ti": ("slowts", "eval_runs/ignite_d5_cer_ti/codec_best.pt"),
    "cer_rot": ("slowts", "eval_runs/ignite_d5_cer_rot/codec_best.pt"),
    "mse": ("slowts", "eval_runs/ignite_mse_ent5/codec_best.pt"),
    "filterscopes": ("fastts", "eval_runs/ignite_fastts_filterscopes/codec_best.pt"),
}

# Reserved-slot modalities: without a resolved codec they are cached as a constant
# code, so the frame keeps its width and a real codec can be swapped in later.
PLACEHOLDER_MODALITIES: Tuple[str, ...] = ("filterscopes",)
_PLACEHOLDER_CODE = 0

_ACT_GLOBAL_DIR = os.environ.get("IGNITE_ACT_GLOBAL", "").strip()

# (H5 group, channels) of the actuator signals, in cache column order. Changing it
# changes the actuator width; the trainer reads the width from the cache instead.
_ACT_SPEC = (
    ("ech_power", 12),
    ("pinj", 8),
    ("beam_voltage", 8),
    ("tinj", 8),
    ("gas_flow", 11),
    ("gas_raw", 11),
    ("rmp", 12),
    ("i_coil", 18),
)

_FRAME_S = 0.05
_DEFAULT_SAMPLE_RATE = 10_000.0
_MIN_CACHED_FRAMES = 8
_LOG_EVERY = 50
_PROFILED_FIRST_STEPS = 5
# Seeds of the per-epoch window draw and of the fixed validation masks.
_WINDOW_SEED_BASE = 0x16117E * 1000003
_VALIDATION_SEED = 1234


class FrozenCodec(NamedTuple):
    """A loaded frozen codec with its configuration and family."""

    codec: torch.nn.Module
    cfg: object
    family: str


def _silent(*args, **kwargs) -> None:
    """Logger that discards everything, for non-zero ranks."""


def _load_codec(family: str, path: Path):
    """Load a frozen codec checkpoint in eval mode with gradients disabled.

    Parameters
    ----------
    family : {"spectro", "video", "slowts", "fastts"}
        Codec family.
    path : Path
        Checkpoint path.

    Returns
    -------
    codec : torch.nn.Module
        The codec.
    cfg : object
        Its configuration.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    codec_cfg = checkpoint["cfg"]
    if family == "spectro":
        from .codec import SpectroCodec as codec_class
    elif family == "video":
        from .video_codec import VideoCodec as codec_class
    elif family == "slowts":
        from .slow_ts_codec import SlowTSCodec as codec_class
    elif family == "fastts":
        from .fastts_codec import FastTSCodec as codec_class
    else:
        raise ValueError(f"unknown codec family {family!r}")
    codec = codec_class(codec_cfg)
    codec.load_state_dict(checkpoint["codec"])
    codec.eval()
    for parameter in codec.parameters():
        parameter.requires_grad_(False)
    return codec, codec_cfg


def resolve_codec_path(name: str, repo: Path, tmpl: str = None):
    """Return the repo-relative checkpoint of a modality's codec, or None.

    Parameters
    ----------
    name : str
        Modality name.
    repo : Path
        Repository root the paths are relative to.
    tmpl : str, default=None
        Template such as ``"eval_runs/ignite_codec_{m}_v6/codec_best.pt"``, tried
        before :data:`FROZEN_CODEC_CKPTS`. A placeholder modality resolves only
        through it.

    Returns
    -------
    rel : str or None
        Path of an existing checkpoint.
    """
    if tmpl:
        templated = tmpl.format(m=name)
        if (repo / templated).exists():
            return templated
    if name in PLACEHOLDER_MODALITIES or name not in FROZEN_CODEC_CKPTS:
        return None
    fallback = FROZEN_CODEC_CKPTS[name][1]
    return fallback if (repo / fallback).exists() else None


def load_frozen_codecs(
    names: List[str], repo: Path = None, tmpl: str = None
) -> Dict[str, FrozenCodec]:
    """Load the codecs of ``names`` whose checkpoint resolves.

    Parameters
    ----------
    names : list of str
        Modalities to load; unresolved ones are skipped.
    repo : Path, default=None
        Repository root; defaults to the working directory.
    tmpl : str, default=None
        Checkpoint template, see :func:`resolve_codec_path`.

    Returns
    -------
    codecs : dict of str to FrozenCodec
        The loaded codecs, which unpack as ``(codec, cfg, family)``.
    """
    repo = repo or Path.cwd()
    layout_family = {m.name: m.family for m in FROZEN_MODALITIES}
    codecs = {}
    for name in names:
        checkpoint = resolve_codec_path(name, repo, tmpl)
        if checkpoint is None:
            continue
        # The layout carries the family; the manifest has no row for late additions.
        family = layout_family.get(name) or FROZEN_CODEC_CKPTS[name][0]
        codecs[name] = FrozenCodec(*_load_codec(family, repo / checkpoint), family)
    return codecs


@torch.no_grad()
def encode_flat(codec, x: torch.Tensor) -> torch.Tensor:
    """Encode a batch of modality windows to flat FSQ codes.

    Parameters
    ----------
    codec : torch.nn.Module
        Frozen codec, on any device.
    x : torch.Tensor of shape (B, ...)
        Modality windows.

    Returns
    -------
    codes : torch.LongTensor of shape (B, n_tok)
        Codes in ``[0, codebook_size)``, on the CPU.
    """
    device = next(codec.parameters()).device
    features = codec.encode(x.to(device))
    return codec.quantizer.fsq(features)[1].long().cpu()


def _single_shot_dataset(
    name: str, fam: str, cfg, shot: str, data_dir, t0_start: float = 1.0
):
    """Return the codec dataset of one shot; item ``t`` is the window of frame ``t``.

    ``t0_start`` is the shot time of frame 0 (1.0 skips the ramp-up second).
    """
    common = dict(data_dir=data_dir, lengths_cache_path=None, t0_start=t0_start)
    if fam == "spectro":
        return tc.CodecPairDataset(name, [shot], cfg, **common)
    if fam == "video":
        return tc.VideoCodecPairDataset(name, [shot], cfg, **common)
    if fam == "fastts":
        from .fastts_train import FastTSCodecPairDataset

        return FastTSCodecPairDataset([shot], cfg, **common)
    return tc.SlowTSCodecPairDataset(name, [shot], cfg, **common)


def _window(item) -> torch.Tensor:
    """The signal window of a codec-dataset item, which may carry a mask too."""
    return item[0] if isinstance(item, tuple) else item


def _encode_frames(dataset, codec, n_frames: int) -> torch.Tensor:
    """Encode the first ``n_frames`` windows of a dataset, shape (n_frames, n_tok)."""
    frames = [
        encode_flat(codec, _window(dataset[t]).unsqueeze(0))[0] for t in range(n_frames)
    ]
    return torch.stack(frames, 0)


@torch.no_grad()
def build_frame_codes(
    shot: str, codecs: Dict, n_frames: int, data_dir
) -> Dict[str, torch.Tensor]:
    """Encode the first ``n_frames`` frames of a shot, ``{name: (n_frames, n_tok)}``."""
    codes = {}
    for name, (codec, codec_cfg, family) in codecs.items():
        dataset = _single_shot_dataset(name, family, codec_cfg, shot, data_dir)
        if len(dataset) < n_frames:
            raise RuntimeError(
                f"{name}@{shot}: only {len(dataset)} frames < {n_frames}"
            )
        codes[name] = _encode_frames(dataset, codec, n_frames)
    return codes


def _frame_bounds(times, n_samples, n_frames, t0_start, legacy):
    """Return the ``[start, stop)`` sample range of every frame window of a record."""
    frame_starts = [float(t0_start) + t * _FRAME_S for t in range(n_frames)]
    if legacy:
        rate = (
            1.0 / np.median(np.diff(times)) if times.size > 1 else _DEFAULT_SAMPLE_RATE
        )
        return [(int(t * rate), int((t + _FRAME_S) * rate)) for t in frame_starts]
    span = float(times[-1] - times[0]) if times.size > 1 else 0.0
    rate = (times.size - 1) / span if span > 0 else _DEFAULT_SAMPLE_RATE
    record_start = float(times[0]) if times.size else 0.0

    def sample_index(t):
        return min(max(int(round((t - record_start) * rate)), 0), n_samples)

    return [(sample_index(t), sample_index(t + _FRAME_S)) for t in frame_starts]


def _raw_actuators(
    shot: str, n_frames: int, data_dir, t0_start: float, legacy: bool = False
) -> np.ndarray:
    """Average every actuator channel over each frame window, shape (n_frames, C).

    Missing signals are zero-filled, so the width is always the one of
    :data:`_ACT_SPEC`. Windows are located in shot time like the diagnostics:
    sample ``round((t - x[0]) * fs)`` with ``fs`` from the record span, clamped
    into the record. ``legacy=True`` reproduces the pre-2026-08-11 misaligned
    indexing ``int(t * fs)`` with ``fs = 1 / median(diff(x))``, which is only
    used to verify old caches before patching them.
    """
    import h5py

    columns = []
    with h5py.File(Path(data_dir) / f"{shot}_processed.h5", "r") as h5:
        for key, n_channels in _ACT_SPEC:
            if not (
                key in h5 and "ydata" in h5[key] and h5[key]["ydata"].shape[-1] > 1
            ):
                columns.append(np.zeros((n_frames, n_channels)))
                continue
            values = np.asarray(h5[key]["ydata"], dtype=np.float64)
            default_times = np.arange(values.shape[-1]) / 1e4
            times = np.asarray(h5[key].get("xdata", default_times), dtype=np.float64)
            bounds = _frame_bounds(times, values.shape[-1], n_frames, t0_start, legacy)
            frame_means = []
            for start, stop in bounds:
                segment = values[:, start:stop]
                frame_means.append(
                    np.nan_to_num(segment).mean(axis=1)
                    if segment.size
                    else np.zeros(values.shape[0])
                )
            column = np.stack(frame_means, 0)
            if column.shape[1] != n_channels:
                fixed_width = np.zeros((n_frames, n_channels))
                kept = min(n_channels, column.shape[1])
                fixed_width[:, :kept] = column[:, :n_channels]
                column = fixed_width
            columns.append(column)
    return np.concatenate(columns, axis=1)


def actuator_frames(
    shot: str,
    n_frames: int,
    data_dir,
    t0_start: float = 1.0,
    return_stats: bool = False,
):
    """Return the frame-mean actuators of a shot, z-scored per shot and channel.

    Parameters
    ----------
    shot : str
        Shot number.
    n_frames : int
        Number of frames from ``t0_start``.
    data_dir : path-like
        Directory of the ``{shot}_processed.h5`` files.
    t0_start : float, default=1.0
        Shot time of frame 0; must match the code windows.
    return_stats : bool, default=False
        Also return the per-channel mean and standard deviation, so that
        ``raw = z * (sd + 1e-6) + mu``.

    Returns
    -------
    act : torch.FloatTensor of shape (n_frames, n_channels)
        Normalised actuators.
    mu, sd : ndarray of shape (n_channels,)
        Only if ``return_stats``.
    """
    raw = _raw_actuators(shot, n_frames, data_dir, t0_start)
    mean, std = raw.mean(0, keepdims=True), raw.std(0, keepdims=True)
    normalised = torch.tensor((raw - mean) / (std + 1e-6), dtype=torch.float32)
    if return_stats:
        return normalised, mean[0].copy(), std[0].copy()
    return normalised


def _legacy_actuator_frames(shot: str, n_frames: int, data_dir, t0_start: float = 1.0):
    """Return the pre-2026-08-11 misaligned actuators, to verify an old cache entry.

    Wrong on purpose; used only by :func:`patch_cache_actuators`.
    """
    raw = _raw_actuators(shot, n_frames, data_dir, t0_start, legacy=True)
    normalised = (raw - raw.mean(0, keepdims=True)) / (raw.std(0, keepdims=True) + 1e-6)
    return torch.tensor(normalised, dtype=torch.float32)


def build_batch(shots: List[str], codecs: Dict, n_frames: int, data_dir):
    """Encode a batch of shots, ``({name: (B, F, n_tok)}, actuators (B, F, C))``."""
    per_shot = [build_frame_codes(s, codecs, n_frames, data_dir) for s in shots]
    actuators = [actuator_frames(s, n_frames, data_dir) for s in shots]
    codes = {
        name: torch.stack([shot_codes[name] for shot_codes in per_shot], 0)
        for name in per_shot[0]
    }
    return codes, torch.stack(actuators, 0)


class _ShotTimeout(Exception):
    """A shot exceeded its encode budget."""


def _install_shot_timeout(budget_s: int, log) -> bool:
    """Make ``SIGALRM`` raise :class:`_ShotTimeout`; False outside the main thread."""

    def on_alarm(_signum, _frame):
        raise _ShotTimeout(f"shot exceeded {budget_s}s encode budget")

    try:
        signal.signal(signal.SIGALRM, on_alarm)
    except ValueError:
        log("[precompute] NOTE: not in main thread — per-shot timeout guard disabled")
        return False
    return True


def _encode_shot(shot, codecs, data_dir, placeholder_specs, t0_start):
    """Encode one shot over the frames every codec covers; None if too short."""
    datasets = {}
    n_frames = None
    for name, (codec, codec_cfg, family) in codecs.items():
        dataset = _single_shot_dataset(
            name, family, codec_cfg, shot, data_dir, t0_start=t0_start
        )
        datasets[name] = (dataset, codec)
        n_frames = len(dataset) if n_frames is None else min(n_frames, len(dataset))
    if not n_frames or n_frames < _MIN_CACHED_FRAMES:
        return None
    codes = {
        name: _encode_frames(dataset, codec, n_frames).to(torch.int32)
        for name, (dataset, codec) in datasets.items()
    }
    for name, n_tok in placeholder_specs.items():
        codes[name] = torch.full(
            (n_frames, n_tok), _PLACEHOLDER_CODE, dtype=torch.int32
        )
    vocabs = {
        name: int(codec.quantizer.fsq.codebook_size)
        for name, (_dataset, codec) in datasets.items()
    }
    actuators = actuator_frames(shot, n_frames, data_dir, t0_start=t0_start)
    return {
        "codes": codes,
        "actuators": actuators.to(torch.float16),
        "n_frames": n_frames,
        "vocabs": vocabs,
    }


def _save_atomically(obj, path: Path) -> None:
    """``torch.save`` to a temporary file, then rename it over ``path``."""
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


@torch.no_grad()
def precompute_frame_codes(
    shots: List[str],
    codecs: Dict,
    out_dir,
    data_dir,
    placeholder_specs: Dict[str, int] = None,
    shot_timeout_s: int = 900,
    t0_start: float = 1.0,
    log=print,
) -> int:
    """Encode every frame of each shot once and cache it.

    Writes ``{shot}.pt`` holding ``codes`` (``{name: int32 (F, n_tok)}``),
    ``actuators`` (float16 (F, C)), ``n_frames`` and ``vocabs``. Shots already
    cached are skipped, as are shots with fewer than 8 frames or whose encoding
    raises or exceeds ``shot_timeout_s``.

    Parameters
    ----------
    shots : list of str
        Shots to encode.
    codecs : dict
        Output of :func:`load_frozen_codecs`.
    out_dir : path-like
        Cache directory.
    data_dir : path-like
        Directory of the H5 files.
    placeholder_specs : dict of str to int, default=None
        Reserved-slot modalities and their token counts, cached as a constant.
    shot_timeout_s : int, default=900
        Per-shot wall-time budget, enforced with ``SIGALRM`` in the main thread.
    t0_start : float, default=1.0
        Shot time of frame 0.
    log : callable, default=print
        Logger.

    Returns
    -------
    n_ok : int
        Number of shots cached, including those already present.
    """
    cache = Path(out_dir)
    cache.mkdir(parents=True, exist_ok=True)
    placeholder_specs = placeholder_specs or {}
    can_alarm = _install_shot_timeout(shot_timeout_s, log)
    n_cached = 0
    for index, shot in enumerate(shots):
        path = cache / f"{shot}.pt"
        if path.exists():
            n_cached += 1
            continue
        if can_alarm:
            signal.alarm(shot_timeout_s)
        try:
            entry = _encode_shot(shot, codecs, data_dir, placeholder_specs, t0_start)
            if entry is None:
                continue
            _save_atomically(entry, path)
            n_cached += 1
        except Exception as e:
            log(f"[precompute] skip {shot}: {type(e).__name__}: {e}")
        finally:
            if can_alarm:
                signal.alarm(0)
        if (index + 1) % 50 == 0:
            log(f"[precompute] {index + 1}/{len(shots)} shots, {n_cached} cached")
    return n_cached


def split_shots(
    cache_dir,
    val_n: int,
    split_seed: int = 0,
    train_cap: int = 0,
    test_n: int = 0,
    pin_val=(),
    pin_train=(),
    pin_both=(),
):
    """Split the cached shots into train, validation and test.

    Parameters
    ----------
    cache_dir : path-like
        Frame-code cache.
    val_n : int
        Validation size, pinned shots included.
    split_seed : int, default=0
        0 keeps the sorted order (validation is the sorted tail); otherwise the
        shots are shuffled with this seed first.
    train_cap : int, default=0
        Keep only the first ``train_cap`` training shots (0 keeps all).
    test_n : int, default=0
        Size of the held-out test partition.
    pin_val : sequence of str, default=()
        Shots forced into validation.
    pin_train : sequence of str, default=()
        Shots forced into training, placed first so ``train_cap`` keeps them.
        ``pin_val`` wins over it.
    pin_both : sequence of str, default=()
        Shots placed in both training and validation, so that train and
        validation losses can be compared on the same shots. They do not count
        towards ``val_n`` and override both other pins. Validation metrics over
        them are not held out.

    Returns
    -------
    train, val, test : list of str
        The partitions.
    """
    shots = sorted(p.stem for p in Path(cache_dir).glob("*.pt"))
    if split_seed:
        random.Random(split_seed).shuffle(shots)
    cached = set(shots)

    def pinned(names):
        return [s for s in dict.fromkeys(names) if s in cached]

    def without(partition, removed):
        return [s for s in partition if s not in removed]

    val_pins = pinned(pin_val)
    rest = without(shots, set(val_pins))
    # Explicit guards: xs[:-0] is [], not xs.
    test = rest[-test_n:] if test_n else []
    pool = rest[:-test_n] if test_n else rest
    n_drawn = max(0, val_n - len(val_pins))
    val = val_pins + (pool[-n_drawn:] if n_drawn else [])
    train = pool[:-n_drawn] if n_drawn else pool
    train_pins = [s for s in pinned(pin_train) if s not in set(val)]
    if train_pins:
        moved = set(train_pins)
        val, test = without(val, moved), without(test, moved)
        train = train_pins + without(train, moved)
    shared = pinned(pin_both)
    if shared:
        moved = set(shared)
        train = shared + without(train, moved)
        val = shared + without(val, moved)
        test = without(test, moved)
    if train_cap:
        train = train[:train_cap]
    return train, val, test


def _log_overlap(train, val, log) -> None:
    """Log the shots present in both training and validation."""
    in_val = set(val)
    overlap = [s for s in train if s in in_val]
    if overlap:
        log(
            f"[dynamics] NOTE {len(overlap)} shot(s) in BOTH train and val: "
            f"{','.join(overlap[:8])}{'...' if len(overlap) > 8 else ''} "
            f"-- val CE on these is NOT held out"
        )


def resolve_split(
    cache_dir,
    out_dir,
    val_n: int,
    split_seed: int = 0,
    train_cap: int = 0,
    test_n: int = 0,
    pin_val=(),
    pin_train=(),
    pin_both=(),
    write: bool = True,
    log=print,
):
    """Return the run's split, computed once and then frozen to ``out_dir``.

    A seeded shuffle is only stable for a fixed pool, and the cache may grow
    between chained legs, so the first leg persists the uncapped partitions to
    ``split_shots.json`` and later legs reload them, intersected with what is
    cached. See :func:`split_shots` for the parameters.

    Parameters
    ----------
    write : bool, default=True
        Persist the split. Must be True on exactly one rank; the others compute
        the identical split in memory.
    log : callable, default=print
        Logger.

    Returns
    -------
    train, val, test : list of str
        The partitions, with ``train_cap`` applied.
    """
    path = Path(out_dir) / "split_shots.json"
    if path.exists():
        frozen = json.loads(path.read_text())
        cached = {p.stem for p in Path(cache_dir).glob("*.pt")}
        train, val, test = (
            [s for s in frozen[part] if s in cached]
            for part in ("train", "val", "test")
        )
        n_missing = len(frozen["train"]) - len(train)
        log(
            f"[dynamics] split RELOADED from {path} (train {len(train)}/val {len(val)}/"
            f"test {len(test)}"
            + (f"; {n_missing} train shots not yet cached" if n_missing else "")
            + ")"
        )
    else:
        train, val, test = split_shots(
            cache_dir,
            val_n,
            split_seed=split_seed,
            train_cap=0,
            test_n=test_n,
            pin_val=pin_val,
            pin_train=pin_train,
            pin_both=pin_both,
        )
        sizes = f"train {len(train)}/val {len(val)}/test {len(test)}"
        if write:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            record = {
                "split_seed": split_seed,
                "val_n": val_n,
                "test_n": test_n,
                "pin_val": list(pin_val),
                "pin_train": list(pin_train),
                "pin_both": list(pin_both),
                "cache_dir": str(cache_dir),
                "train": train,
                "val": val,
                "test": test,
            }
            tmp = path.with_suffix(f".json.tmp.{os.getpid()}")
            tmp.write_text(json.dumps(record, indent=1))
            tmp.replace(path)
            log(
                f"[dynamics] split FROZEN -> {path} ({sizes}, seed={split_seed}, "
                f"pinned={list(pin_val)})"
            )
        else:
            log(f"[dynamics] split computed in memory ({sizes}); rank 0 persists it")
    if train_cap:
        train = train[:train_cap]
    _log_overlap(train, val, log)
    return train, val, test


class _EnvRank:
    """Shard identity from environment variables, for the per-shot cache passes.

    No process group is created: ranks finish at very different times, and an
    NCCL group then fails at teardown either way (barrier timeouts or broken
    communicators).
    """

    def __init__(self):
        self.rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device_index = local_rank if torch.cuda.device_count() > 1 else 0
        self.device = torch.device(
            f"cuda:{device_index}" if torch.cuda.is_available() else "cpu"
        )
        self.is_main = self.rank == 0
        if torch.cuda.is_available():
            torch.cuda.set_device(self.device)


def _patch_one_entry(shot, path, data_dir, t0_start, verify, log):
    """Return ``(status, entry, old_actuators)`` for one cache entry."""
    entry = torch.load(path, map_location="cpu", weights_only=False)
    n_frames = int(entry["n_frames"])
    stored = entry["actuators"]
    if verify:
        legacy = _legacy_actuator_frames(shot, n_frames, data_dir, t0_start=t0_start)
        tolerance = 1e-2 + 1e-3 * float(stored.float().abs().max())
        deviation = float((legacy - stored.float()).abs().max())
        if deviation > tolerance:
            log(
                f"[patch_act] MISMATCH {shot}: stored actuators are not the legacy "
                f"t0={t0_start} output (max dev {deviation:.4g} > tol "
                f"{tolerance:.4g}) — SKIPPED"
            )
            return "mismatch", None, None
    aligned = actuator_frames(shot, n_frames, data_dir, t0_start=t0_start)
    aligned = aligned.to(stored.dtype)
    if aligned.shape != stored.shape:
        log(
            f"[patch_act] SHAPE {shot}: {tuple(stored.shape)} -> "
            f"{tuple(aligned.shape)} — SKIPPED"
        )
        return "mismatch", None, None
    entry["actuators"] = aligned
    return "patched", entry, stored.clone()


def patch_cache_actuators(
    shots: List[str],
    cache_dir,
    data_dir,
    t0_start: float,
    backup_path=None,
    verify: bool = True,
    dry_run: bool = False,
    log=print,
) -> Dict[str, int]:
    """Rewrite only the ``actuators`` of existing cache entries with the aligned ones.

    Parameters
    ----------
    shots : list of str
        Shots to patch.
    cache_dir, data_dir : path-like
        Cache and H5 directories.
    t0_start : float
        Shot time of frame 0 of the cache.
    backup_path : path-like, default=None
        Where to save the replaced actuators, so the pass can be undone.
    verify : bool, default=True
        Patch an entry only if its stored actuators match the legacy output to
        float16 resolution; otherwise count it as a mismatch and skip it.
    dry_run : bool, default=False
        Verify and count without writing.
    log : callable, default=print
        Logger.

    Returns
    -------
    counts : dict of str to int
        Counts of ``patched``, ``mismatch``, ``missing`` and ``error`` shots.
    """
    counts = {"patched": 0, "mismatch": 0, "missing": 0, "error": 0}
    backup = {}
    for index, shot in enumerate(shots):
        path = Path(cache_dir) / f"{shot}.pt"
        if not path.exists():
            counts["missing"] += 1
            continue
        try:
            status, entry, old = _patch_one_entry(
                shot, path, data_dir, t0_start, verify, log
            )
            if status == "patched":
                backup[shot] = old
                if not dry_run:
                    _save_atomically(entry, path)
            counts[status] += 1
        except Exception as e:  # noqa: BLE001 (per-shot isolation)
            log(f"[patch_act] skip {shot}: {type(e).__name__}: {e}")
            counts["error"] += 1
        if (index + 1) % 200 == 0:
            log(
                f"[patch_act] {index + 1}/{len(shots)} shots, "
                f"{counts['patched']} patched"
            )
    if backup_path is not None and backup and not dry_run:
        Path(backup_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actuators": backup,
                "t0_start": t0_start,
                "note": "pre-2026-08-11 misaligned actuators; restore to undo the "
                "time-base fix",
            },
            backup_path,
        )
    return counts


def _manifest_path(cache_dir) -> Path:
    return Path(cache_dir) / "_codec_manifest.json"


def _manifest_t0(cache_dir) -> float:
    """Return ``t0_start`` from a cache's ``_codec_manifest.json`` (default 1.0)."""
    manifest = _manifest_path(cache_dir)
    if not manifest.exists():
        return 1.0
    return float(json.loads(manifest.read_text()).get("t0_start", 1.0))


def run_patch_actuators(cache_dir, data_dir, dry_run: bool = False, log=print) -> int:
    """Shard :func:`patch_cache_actuators` over ranks; return the shots patched.

    ``t0_start`` comes from the cache manifest, never from a guess.
    """
    shard_id = _EnvRank()
    cache = Path(cache_dir)
    manifest = _manifest_path(cache)
    if not manifest.exists():
        raise SystemExit(
            f"--patch_actuators needs {manifest} to recover the cache's t0_start"
        )
    t0_start = float(json.loads(manifest.read_text())["t0_start"])
    shots = sorted(p.stem for p in cache.glob("[0-9]*.pt"))
    shard = shots[shard_id.rank :: shard_id.world_size]
    if shard_id.is_main:
        log(
            f"[patch_act] {len(shots)} cached shots / {shard_id.world_size} ranks "
            f"(~{len(shard)}/rank); t0_start={t0_start} cache={cache} "
            f"dry_run={dry_run}",
            flush=True,
        )
    backup = cache / "_actuators_prefix_backup" / f"rank{shard_id.rank:03d}.pt"
    counts = patch_cache_actuators(
        shard, cache, data_dir, t0_start, backup_path=backup, dry_run=dry_run, log=log
    )
    log(f"[patch_act rank {shard_id.rank}] {counts} (backup -> {backup})")
    return counts["patched"]


def run_precompute(
    cache_dir,
    data_dir,
    max_shots: int = 0,
    codec_tmpl: str = None,
    shot_sample: int = 0,
    shot_seed: int = 0,
    t0_start: float = 1.0,
    shot_timeout_s: int = 900,
    log=print,
) -> int:
    """Build the frame-code cache, sharding the shots over ranks.

    Modalities whose codec does not resolve are cached as a constant if they are
    reserved slots (:data:`PLACEHOLDER_MODALITIES`) and excluded from the frame
    otherwise. Rank 0 writes ``_codec_manifest.json``. Ranks exit independently;
    a final barrier would hit the NCCL watchdog on uneven shards.

    Parameters
    ----------
    cache_dir, data_dir : path-like
        Cache and H5 directories.
    max_shots : int, default=0
        Keep only the first shots (0 = all).
    codec_tmpl : str, default=None
        Checkpoint template, see :func:`resolve_codec_path`.
    shot_sample : int, default=0
        Seeded random sample of this many shots; overrides ``max_shots``.
    shot_seed : int, default=0
        Seed of ``shot_sample``.
    t0_start : float, default=1.0
        Shot time of frame 0.
    shot_timeout_s : int, default=900
        Per-shot encode budget.
    log : callable, default=print
        Logger.

    Returns
    -------
    n : int
        Shots cached on this rank's shard.
    """
    shard_id = _EnvRank()
    repo = Path.cwd()
    resolved = {
        m.name: resolve_codec_path(m.name, repo, codec_tmpl) for m in FROZEN_MODALITIES
    }
    unresolved = [m for m in FROZEN_MODALITIES if resolved[m.name] is None]
    real = [name for name, path in resolved.items() if path is not None]
    placeholder_specs = {
        m.name: m.n_tok for m in unresolved if m.name in PLACEHOLDER_MODALITIES
    }
    excluded = [m.name for m in unresolved if m.name not in PLACEHOLDER_MODALITIES]
    codecs = load_frozen_codecs(real, repo=repo, tmpl=codec_tmpl)
    if shard_id.is_main:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        frame_tokens = sum(
            int(c.cfg.n_tok) for c in codecs.values() if hasattr(c.cfg, "n_tok")
        ) + sum(placeholder_specs.values())
        manifest = {
            "resolved": resolved,
            "codec_tmpl": codec_tmpl,
            "t0_start": t0_start,
            "placeholders": sorted(placeholder_specs),
            "excluded": excluded,
            "frame_tokens": frame_tokens,
        }
        with open(_manifest_path(cache_dir), "w") as f:
            json.dump(manifest, f, indent=2)
    for frozen in codecs.values():
        frozen.codec.to(shard_id.device)
    all_shots = sorted(spike.discover_shots(data_dir))
    if shot_sample:
        n_sampled = min(shot_sample, len(all_shots))
        all_shots = sorted(random.Random(shot_seed).sample(all_shots, n_sampled))
    elif max_shots:
        all_shots = all_shots[:max_shots]
    shard = all_shots[shard_id.rank :: shard_id.world_size]
    if shard_id.is_main:
        log(
            f"[precompute] {len(all_shots)} shots / {shard_id.world_size} ranks "
            f"(~{len(shard)}/rank); real={len(real)} "
            f"placeholders={list(placeholder_specs)} cache={cache_dir}",
            flush=True,
        )
        log(f"[precompute] REAL modalities ({len(real)}): {sorted(real)}", flush=True)
        if excluded:
            log(
                f"[precompute] EXCLUDED from the frame (no codec resolved, not a "
                f"reserved slot): {excluded} — confirm this is intended before "
                f"training on this cache",
                flush=True,
            )
    n_cached = precompute_frame_codes(
        shard,
        codecs,
        cache_dir,
        data_dir,
        placeholder_specs=placeholder_specs,
        t0_start=t0_start,
        shot_timeout_s=shot_timeout_s,
        log=log,
    )
    log(
        f"[precompute rank {shard_id.rank}] cached {n_cached}/{len(shard)} shots "
        f"on this shard"
    )
    return n_cached


def build_presence(cache_dir, out_path=None, log=print) -> Dict[str, Dict[str, bool]]:
    """Find which diagnostics each cached shot actually recorded.

    A diagnostic that did not record encodes to the same constant frame in every
    shot. A shot's modality is absent if all its frames equal that modality's
    null pattern, the most common pattern among constant shots; a diagnostic that
    is merely flat with another value counts as present.

    Parameters
    ----------
    cache_dir : path-like
        Frame-code cache.
    out_path : path-like, default=None
        If given, write ``{"cache_dir": ..., "presence": ...}`` there as JSON.
    log : callable, default=print
        Logger.

    Returns
    -------
    presence : dict of str to dict of str to bool
        ``{shot: {modality: True if present}}``.
    """
    cache = Path(cache_dir)
    stems = sorted(p.stem for p in cache.glob("*.pt") if not p.stem.startswith("_"))
    constant_patterns: Dict[str, Counter] = {}
    per_shot: Dict[str, Dict[str, tuple]] = {}
    for shot in stems:
        try:
            codes = torch.load(cache / f"{shot}.pt", map_location="cpu")["codes"]
        except Exception as e:
            log(f"[presence] skip {shot}: {type(e).__name__}: {e}")
            continue
        shot_record = {}
        for modality, frames in codes.items():
            is_constant = bool((frames == frames[0:1]).all())
            pattern = hash(frames[0].numpy().tobytes())
            shot_record[modality] = (is_constant, pattern)
            if is_constant:
                constant_patterns.setdefault(modality, Counter())[pattern] += 1
        per_shot[shot] = shot_record
    null_pattern = {
        modality: counts.most_common(1)[0][0]
        for modality, counts in constant_patterns.items()
        if counts
    }
    presence = {
        shot: {
            modality: not (is_constant and null_pattern.get(modality) == pattern)
            for modality, (is_constant, pattern) in shot_record.items()
        }
        for shot, shot_record in per_shot.items()
    }
    if per_shot:
        _log_presence(presence, constant_patterns, log)
    if out_path:
        tmp = Path(str(out_path) + f".tmp.{os.getpid()}")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(
            json.dumps({"cache_dir": str(cache), "presence": presence}, indent=0)
        )
        tmp.replace(Path(out_path))
        log(f"[presence] -> {out_path}")
    return presence


def _log_presence(presence, constant_patterns, log) -> None:
    modalities = sorted(next(iter(presence.values())))
    n_shots = len(presence)
    log(f"[presence] {n_shots} shots x {len(modalities)} modalities")
    for modality in modalities:
        n_absent = sum(1 for shot in presence if not presence[shot][modality])
        n_patterns = len(constant_patterns.get(modality, {})) or 0
        log(
            f"[presence]   {modality:<22} absent in {n_absent:5d}/{n_shots} "
            f"({100 * n_absent / max(1, n_shots):4.1f}%)  "
            f"distinct frozen patterns={n_patterns}"
        )
    n_absent_total = sum(
        1 for shot in presence for m in modalities if not presence[shot][m]
    )
    mean_absent = n_absent_total / max(1, n_shots * len(modalities))
    log(
        f"[presence] MEAN per-modality absence = {100 * mean_absent:.1f}% "
        f"(= the share of loss TERMS that would score constants)"
    )


def _n_live_frames(n_frames: int, t_end, t0_start: float, frame_s: float) -> int:
    """Return the number of leading frames before the plasma end ``t_end``.

    ``None`` means not measured: every frame counts as live. The ``1e-9`` keeps a
    frame that lands exactly on ``t_end`` despite binary rounding.
    """
    if t_end is None:
        return int(n_frames)
    last_live = math.floor((float(t_end) - t0_start) / frame_s + 1e-9)
    return int(max(0, min(int(n_frames), last_live + 1)))


class WindowSample(enum.StrEnum):
    """Which windows each shot contributes per epoch, see :class:`FrameCodeDataset`."""

    RANDOM = "random"
    FIRST = "first"
    LIVE = "live"


def _narrowest_int(codes: torch.Tensor, vocab: int) -> torch.Tensor:
    """Cast codes to the smallest integer type holding ``vocab`` values."""
    if vocab <= torch.iinfo(torch.uint8).max + 1:
        return codes.to(torch.uint8)
    if vocab <= torch.iinfo(torch.int16).max + 1:
        return codes.to(torch.int16)
    return codes


def _global_actuators(shot, cached: torch.Tensor) -> torch.Tensor:
    """Dataset-wide normalised actuators of a shot, truncated to the cache width."""
    side_entry = Path(_ACT_GLOBAL_DIR) / f"{shot}.pt"
    if not side_entry.exists():
        raise FileNotFoundError(
            f"IGNITE_ACT_GLOBAL={_ACT_GLOBAL_DIR} has no entry for shot {shot}. "
            f"Refusing to fall back to the per-shot z-scored actuators, which "
            f"would silently mix two normalisations within one run."
        )
    width = cached.shape[-1]
    loaded = torch.load(side_entry, map_location="cpu", weights_only=False)
    return loaded["actuators_global"][:, :width].to(cached.dtype)


class FrameCodeDataset(torch.utils.data.Dataset):
    """Windows of ``cfg.max_frames`` frames from the frame-code cache.

    All selected shots are loaded into memory once, so an item is a slice. Codes
    are narrowed to the smallest integer type holding the vocabulary, since every
    rank holds the whole split.

    Parameters
    ----------
    cache_dir : path-like
        Frame-code cache.
    shots : list of str
        Shots to load; shots not cached or not covering the layout are skipped.
    cfg : DynamicsConfig
        Model configuration (layout and window length).
    presence : dict, default=None
        Output of :func:`build_presence`; None marks everything present.
    stride : int, default=1
        Frames between consecutive window starts.
    windows_per_shot : int, default=0
        Windows each shot contributes per epoch (0 = every start).
    window_sample : {"random", "first", "live"}, default="random"
        Which starts, if ``windows_per_shot`` is set. ``"first"`` takes starts
        ``0..n-1``, the region a rollout seeded at the shot start covers.
        ``"random"`` redraws ``n`` of all starts each epoch. ``"live"`` redraws
        them from the starts whose window is at least half before the plasma
        end, plus starts ``0..n-1``; it is meant for training only.
    live_end : dict of str to float, default=None
        Plasma end time of each shot; later frames get zero loss weight. Shots
        not listed count as live throughout.
    window_jitter : int, default=0
        Augmentation for ``"first"``: each epoch draws the ``n`` starts of a shot
        from ``0..n-1+window_jitter`` instead of taking ``0..n-1``. Ignored by the
        other modes, which redraw their starts anyway.
    t0_start : float, default=1.0
        Shot time of frame 0 of the cache.
    frame_s : float, default=0.05
        Frame duration in seconds.

    Notes
    -----
    With ``IGNITE_ACT_GLOBAL`` set, actuators come from that side cache,
    truncated to the cache's actuator width.
    """

    def __init__(
        self,
        cache_dir,
        shots,
        cfg: DynamicsConfig,
        presence=None,
        stride: int = 1,
        windows_per_shot: int = 0,
        window_sample: str = "random",
        live_end=None,
        window_jitter: int = 0,
        t0_start: float = 1.0,
        frame_s: float = 0.05,
    ):
        self.cfg = cfg
        self.window_jitter = max(int(window_jitter), 0)
        self.win = cfg.max_frames
        self.stride = max(int(stride), 1)
        self.windows_per_shot = max(int(windows_per_shot), 0)
        self.window_sample = WindowSample(str(window_sample).lower())
        self.live_end = dict(live_end or {})
        self.t0_start, self.frame_s = float(t0_start), float(frame_s)
        self.shots = []
        self.shot_ids = []
        self.n_live = []
        self.index = []
        self._starts = []
        self._live_starts = []
        self.presence = []
        presence = presence or {}
        for shot in shots:
            entry = self._load_entry(Path(cache_dir) / f"{shot}.pt")
            if entry is None:
                continue
            shot_index = len(self.shots)
            self.shots.append(entry)
            self.shot_ids.append(str(shot))
            t_end = self.live_end.get(str(shot))
            n_live = _n_live_frames(
                entry["n_frames"], t_end, self.t0_start, self.frame_s
            )
            self.n_live.append(n_live)
            recorded = presence.get(str(shot), {})
            self.presence.append(
                torch.tensor(
                    [1.0 if recorded.get(m.name, True) else 0.0 for m in cfg.modalities]
                )
            )
            starts = list(range(0, entry["n_frames"] - self.win + 1, self.stride))
            self._starts.append(starts)
            deployment_starts = starts[: max(self.windows_per_shot, 1)]
            live_starts = (
                [s for s in starts if s + self.win / 2.0 <= n_live]
                if t_end is not None
                else []
            )
            self._live_starts.append(sorted(set(live_starts) | set(deployment_starts)))
            if not self.windows_per_shot:
                self.index.extend((shot_index, start) for start in starts)
        if self.windows_per_shot:
            self.set_epoch(0)

    def _load_entry(self, path: Path):
        """Load a cache entry restricted to the layout, or None if unusable."""
        if not path.exists():
            return None
        entry = torch.load(path, map_location="cpu", weights_only=False)
        modalities = self.cfg.modalities
        if not {m.name for m in modalities}.issubset(entry["codes"]):
            return None
        entry["codes"] = {
            m.name: _narrowest_int(entry["codes"][m.name], m.codebook_size)
            for m in modalities
        }
        if _ACT_GLOBAL_DIR:
            entry["actuators"] = _global_actuators(path.stem, entry["actuators"])
        return entry

    def set_epoch(self, epoch: int) -> None:
        """Redraw the windows of every shot for an epoch.

        The draw is seeded on ``epoch`` alone, so every rank sees the same index
        and the sampler's sharding stays consistent. Call it before creating the
        loader iterator, so that worker processes fork with the new index.
        """
        if not self.windows_per_shot:
            return
        rng = random.Random(_WINDOW_SEED_BASE + int(epoch))
        if self.window_sample is WindowSample.FIRST:
            n, pool = self.windows_per_shot, self.windows_per_shot + self.window_jitter
            self.index = [
                (shot_index, start)
                for shot_index, starts in enumerate(self._starts)
                for start in (
                    sorted(rng.sample(starts[:pool], min(n, len(starts[:pool]))))
                    if self.window_jitter
                    else starts[:n]
                )
            ]
            return
        pools = (
            self._live_starts
            if self.window_sample is WindowSample.LIVE
            else self._starts
        )
        index = []
        for shot_index, starts in enumerate(pools):
            n_drawn = min(self.windows_per_shot, len(starts))
            index.extend((shot_index, s) for s in sorted(rng.sample(starts, n_drawn)))
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        """Return ``(codes, actuators, presence, live)`` of window ``i``."""
        shot_index, start = self.index[i]
        entry = self.shots[shot_index]
        stop = start + self.win
        codes = {
            m.name: entry["codes"][m.name][start:stop].long()
            for m in self.cfg.modalities
        }
        actuators = entry["actuators"][start:stop].float()
        live = (torch.arange(start, stop) < self.n_live[shot_index]).float()
        return codes, actuators, self.presence[shot_index], live


def _collate_frames(batch):
    codes = {name: torch.stack([b[0][name] for b in batch], 0) for name in batch[0][0]}
    actuators = torch.stack([b[1] for b in batch], 0)
    present = torch.stack([b[2] for b in batch], 0)
    live = torch.stack([b[3] for b in batch], 0)
    return codes, actuators, present, live


def smoke(
    n_shots: int = 2,
    n_steps: int = 30,
    depth: int = 4,
    d_model: int = 128,
    seed: int = 0,
):
    """Train a small model briefly on real frames and roll it out, on the CPU.

    Returns
    -------
    losses : list of float
        Training losses.
    ok : bool
        Whether every committed rollout code is inside its vocabulary.
    """
    from ..data.multi_file_dataset import filter_signal_present_files

    torch.manual_seed(seed)
    data_dir = spike.DEFAULT_DATA_DIR
    repo = Path(__file__).resolve().parents[3]
    available = tuple(
        m for m in FROZEN_MODALITIES if resolve_codec_path(m.name, repo) is not None
    )
    skipped = [m.name for m in FROZEN_MODALITIES if m not in available]
    if skipped:
        print(f"[smoke] NOTE skipping (no ckpt yet): {skipped}", flush=True)
    cfg = DynamicsConfig(
        modalities=available, d_model=d_model, depth=depth, k0_seed=4, n_predict=3
    )
    names = [m.name for m in cfg.modalities]
    print(f"[smoke] loading {len(names)} frozen codecs ...", flush=True)
    codecs = load_frozen_codecs(names, repo=repo)
    n_frames = cfg.max_frames
    meta = Path("/lustre/orion/fus187/proj-shared/foundation_model_meta")
    co2_files = filter_signal_present_files(
        tc._shot_paths(spike.discover_shots(data_dir), data_dir),
        "co2",
        cache_path=meta / "codec_co2_present.pt",
    )
    shots = [p.name.split("_")[0] for p in co2_files][:n_shots]
    print(
        f"[smoke] building real frame codes for shots {shots} ({n_frames} frames) ...",
        flush=True,
    )
    codes, actuators = build_batch(shots, codecs, n_frames, data_dir)
    print(
        f"[smoke] frame codes ok: co2 {tuple(codes['co2'].shape)} "
        f"mse {tuple(codes['mse'].shape)} actuators {tuple(actuators.shape)}",
        flush=True,
    )

    model = MaskGITDynamics(cfg).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    generator = torch.Generator().manual_seed(seed)
    losses = []
    for step in range(n_steps):
        optimizer.zero_grad()
        loss = model.training_loss(
            codes, actuators, generator=generator, ss_frac=model.ss_fraction(step)
        )
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    print(
        f"[smoke] MaskGIT loss: start {losses[0]:.3f} -> end {losses[-1]:.3f} "
        f"(min {min(losses):.3f}) over {n_steps} steps",
        flush=True,
    )

    model.eval()
    seed_codes = {name: c[:, : cfg.k0_seed] for name, c in codes.items()}
    trajectory = model.rollout(
        seed_codes, actuators, n_predict=cfg.n_predict, generator=generator
    )
    vocab = {m.name: m.codebook_size for m in cfg.modalities}
    ok = all(
        int(c[:, cfg.k0_seed :].min()) >= 0
        and int(c[:, cfg.k0_seed :].max()) < vocab[name]
        for name, c in trajectory.items()
    )
    print(
        f"[smoke] rollout -> {tuple(trajectory['co2'].shape)} per modality; "
        f"all committed codes valid: {ok}",
        flush=True,
    )
    print(
        f"[smoke] RESULT: loss decreased={losses[-1] < losses[0]}  rollout_valid={ok}",
        flush=True,
    )
    return losses, ok


def _render_loss_curve(hist_path, png_path) -> None:
    """Plot the train and validation losses of ``loss_history.jsonl`` to a PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train_steps, train_losses, val_steps, val_losses = [], [], [], []
    with open(hist_path) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if "val_loss" in record:
                val_steps.append(record["step"])
                val_losses.append(record["val_loss"])
            elif "loss" in record:
                train_steps.append(record["step"])
                train_losses.append(record["loss"])
    if not train_steps:
        return
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(
        train_steps,
        train_losses,
        color="C0",
        alpha=0.3,
        lw=0.8,
        label="train masked CE (per-log)",
    )
    smoothed, ema = [], None
    for value in train_losses:
        ema = value if ema is None else 0.98 * ema + 0.02 * value
        smoothed.append(ema)
    ax.plot(train_steps, smoothed, color="C0", lw=1.6, label="train (EMA)")
    if val_steps:
        ax.plot(
            val_steps, val_losses, "o-", color="C1", ms=3, lw=1.2, label="val masked CE"
        )
    # The latent head's Gaussian NLL can be negative, which a log axis cannot show.
    if min(train_losses) > 0 and (not val_losses or min(val_losses) > 0):
        ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("masked-token CE  (latent head: Gaussian NLL, may be < 0)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(png_path, dpi=110)
    plt.close(fig)


def cache_modality_specs(cache_dir):
    """Derive the frame layout from a cache.

    Token counts come from the cached code shapes and vocabularies from the
    ``vocabs`` entry, falling back to :data:`FROZEN_MODALITIES` for old caches.

    Returns
    -------
    specs : tuple of ModalitySpec
        Layout in canonical order, restricted to the cached modalities.
    """
    files = sorted(Path(cache_dir).glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"no cached shots under {cache_dir}")
    entry = torch.load(files[0], map_location="cpu", weights_only=False)
    vocabs = entry.get("vocabs", {})
    return tuple(
        ModalitySpec(
            m.name,
            m.family,
            int(entry["codes"][m.name].shape[1]),
            int(vocabs.get(m.name, m.codebook_size)),
        )
        for m in FROZEN_MODALITIES
        if m.name in entry["codes"]
    )


def _mean_one(raw: Dict[str, float], names: List[str]) -> Dict[str, float]:
    """Scale weights to mean 1, so the loss scale and LR behaviour are unchanged."""
    scale = len(names) / sum(raw.values())
    return {name: raw[name] * scale for name in names}


def resolve_mod_weights(spec: str, names: List[str]) -> Dict[str, float]:
    """Parse explicit per-modality loss weights.

    Parameters
    ----------
    spec : str
        Path to a JSON file, or an inline JSON object ``{modality: weight}``.
        Prefer the path: ``sbatch --export`` splits values on commas.
    names : list of str
        Modalities of the layout. Missing ones weigh 1; keys starting with
        ``_`` are metadata and ignored; unknown names are an error.

    Returns
    -------
    weights : dict of str to float
        Non-negative weights scaled to mean 1.
    """
    text = str(spec).strip()
    if not text:
        raise ValueError("resolve_mod_weights: empty spec")
    # Anything that looks like JSON is parsed as JSON, so a typo is a JSON error.
    if text[0] not in "{[":
        path = Path(text)
        if not path.exists():
            raise FileNotFoundError(f"--mod_weights_json: no such file {path}")
        text = path.read_text()
    given = json.loads(text)
    if not isinstance(given, dict) or not given:
        raise ValueError(
            "--mod_weights_json must be a non-empty JSON object {modality: weight}"
        )
    given = {k: v for k, v in given.items() if not str(k).startswith("_")}
    if not given:
        raise ValueError("--mod_weights_json: object contains only metadata keys")
    unknown = sorted(set(given) - set(names))
    if unknown:
        raise ValueError(
            f"--mod_weights_json has unknown modalities {unknown}; "
            f"this cache/config has {names}"
        )
    invalid = {
        k: v
        for k, v in given.items()
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0
    }
    if invalid:
        raise ValueError(
            f"--mod_weights_json weights must be non-negative numbers; got {invalid}"
        )
    raw = {name: float(given.get(name, 1.0)) for name in names}
    if sum(raw.values()) <= 0:
        raise ValueError("--mod_weights_json: weights sum to zero")
    return _mean_one(raw, names)


_SAME_AS_DEFAULT = object()


def _option(
    default,
    description: str,
    *,
    cli: bool = True,
    cli_default=_SAME_AS_DEFAULT,
    cli_type=None,
    choices=None,
    from_cli=None,
):
    """A :class:`TrainConfig` field with the metadata its command-line flag needs.

    Parameters
    ----------
    default : object
        Default of the field.
    description : str
        Help text of the flag.
    cli : bool, default=True
        Whether the field has a flag.
    cli_default : object, optional
        Default of the flag, if it differs from the field's.
    cli_type : callable, default=None
        Type of the flag; inferred from the default when it is an int, float or
        str. A bool default becomes a ``store_true`` flag.
    choices : sequence, default=None
        Allowed values of the flag.
    from_cli : callable, default=None
        Converts the parsed value to the field's type.
    """
    metadata = dict(
        help=description,
        cli=cli,
        cli_default=default if cli_default is _SAME_AS_DEFAULT else cli_default,
        cli_type=cli_type,
        choices=choices,
        from_cli=from_cli,
    )
    return dataclasses.field(default=default, metadata=metadata)


def _comma_separated(text: str) -> tuple:
    return tuple(item.strip() for item in text.split(",") if item.strip())


@dataclasses.dataclass
class TrainConfig:
    """Options of :func:`train`; ``None`` keeps the :class:`DynamicsConfig` default.

    Every field with a flag is exposed by :func:`build_arg_parser`, so the help
    texts below are the command-line documentation.
    """

    # Run length and optimisation.
    steps: int = _option(200_000, "optimizer steps")
    batch_size: int = _option(8, "micro-batch per rank")
    lr: float = _option(3e-4, "peak learning rate")
    warmup_steps: int = _option(0, "linear LR warmup steps")
    min_lr_ratio: float = _option(0.01, "cosine floor as a fraction of the peak LR")
    lr_decay_from: int = _option(
        0,
        "step at which the cosine starts at full LR (0 = end of warmup); set to "
        "the resume step of a warm-started fork",
    )
    beta2: float = _option(0.999, "AdamW beta2")
    weight_decay: float = _option(0.01, "AdamW weight decay")
    wd_exclude_norms: bool = _option(
        False,
        "1 = no weight decay on norms and biases; changes the param groups, so "
        "toggling it mid-chain resets the Adam moments",
        cli_default=0,
        cli_type=int,
        from_cli=bool,
    )
    grad_clip: float = _option(
        0.0,
        "clip the all-reduced gradient norm; 0 = off. Typical norms are ~0.05, so "
        "1.0 never fires",
    )
    accum_steps: int = _option(
        1, "gradient-accumulation micro-steps per optimizer step"
    )
    num_workers: int = _option(4, "data-loader workers")
    ckpt_every: int = _option(
        1000,
        "checkpoint and validation cadence in steps; keep it below the steps of one "
        "chained job, or a resume restarts from step 0",
        cli_default=200,
    )
    patience: int = _option(
        0, "stop after this many evaluations without improvement; 0 = never"
    )
    # Model.
    depth: int = _option(24, "transformer blocks")
    d_model: int = _option(1024, "model width")
    n_heads: Optional[int] = _option(None, "attention heads", cli_type=int)
    k0_seed: Optional[int] = _option(None, "seed frames", cli_type=int)
    n_predict: Optional[int] = _option(None, "predicted frames", cli_type=int)
    dropout: float = _option(0.0, "attention and FFN dropout")
    grad_ckpt: Optional[int] = _option(
        None,
        "1 = gradient checkpointing (needed once d_model*depth >= 4096)",
        cli_type=int,
    )
    act_cross_attn: Optional[int] = _option(
        None, "1 = per-token cross-attention over the actuator channels", cli_type=int
    )
    latent_head: Optional[int] = _option(
        None,
        "1 = Gaussian head over the FSQ grid coordinate; its loss is a Gaussian NLL, "
        "not comparable to token CE",
        cli_type=int,
    )
    lag_embed_k: Optional[int] = _option(
        None,
        "own-column history embeddings per token; changes the parameter set",
        cli_type=int,
    )
    # Objective.
    ss_final_frac: Optional[float] = _option(
        None, "final scheduled-sampling fraction; 0 disables it", cli_type=float
    )
    ss_ramp_steps: Optional[int] = _option(
        None,
        "steps of the scheduled-sampling ramp; set it to the run length",
        cli_type=int,
    )
    gen_mask_p: Optional[float] = _option(
        None,
        "fraction of samples masked like a rollout; 0 = cosine prior only",
        cli_type=float,
    )
    gen_horizon_alpha: Optional[float] = _option(
        None,
        "generation-mode horizon exponent, p(h) ~ h**-alpha; 0 = uniform split",
        cli_type=float,
    )
    gen_horizon_max: Optional[int] = _option(
        None, "generation-mode horizon uniform in [1, N]", cli_type=int
    )
    label_smoothing: float = _option(
        0.0, "cross-entropy label smoothing, training only"
    )
    block_forcing: bool = _option(
        False,
        "1 = block teacher forcing: every frame is predicted from a fully visible "
        "history of earlier frames, and scheduled sampling fills that history "
        "with the model's own one-step forecasts. Roughly doubles the step cost",
        cli_default=0,
        cli_type=int,
        from_cli=bool,
    )
    act_drop_p: float = _option(
        0.0,
        "probability of training a sample without any actuator term, the "
        "unconditional branch classifier-free guidance needs (eval --guidance)",
    )
    forecast_depth: int = _option(
        1,
        "with --block_forcing 1: levels of the model's own forecasts in the history "
        "(pushforward multi-step training); each extra level costs one no-grad "
        "two-stream pass",
    )
    seed_drop_p: float = _option(
        0.0,
        "augmentation: probability of hiding a whole modality from the input in the "
        "seed frames (first k0); hidden tokens are scored, so the model learns to "
        "predict a modality missing from the seed",
    )
    seed_column_drop_p: float = _option(
        0.0,
        "augmentation: probability of hiding a random fraction of a modality's token "
        "positions in every seed frame; hidden tokens are scored",
    )
    actuator_noise: float = _option(
        0.0,
        "augmentation: std of Gaussian noise added to the normalised actuators, on "
        "channels that vary within the window; training only",
    )
    mask_absent: bool = _option(
        False,
        "exclude absent diagnostics from the loss (they stay in the input); builds "
        "or loads <cache>/_presence.json",
    )
    presence_path: Optional[str] = _option(
        None, "presence map (default <cache_dir>/_presence.json)"
    )
    balance_presence: bool = _option(
        False,
        "weight each modality by 1 / (fraction of train shots recording it); "
        "requires --mask_absent",
    )
    mod_weight_pow: float = _option(
        0.0, "per-modality loss weight n_tok**a; 0 = equal weighting"
    )
    mod_weights_json: str = _option(
        "",
        "explicit loss weights {modality: weight} as a file path (preferred) or "
        "inline JSON; overrides --mod_weight_pow and --balance_presence",
    )
    live_end_json: str = _option(
        "",
        "JSON {shot: plasma end [s]}; zero-weights post-shot frames in train and val "
        "loss. Redefines val/gen CE",
    )
    # Data split.
    val_frac: float = _option(0.05, "validation fraction if val_n is 0", cli=False)
    val_n: int = _option(0, "validation size in shots; 0 = val_frac of the cache")
    test_n: int = _option(0, "held-out test size in shots (overrides --test_frac)")
    test_frac: float = _option(0.0, "held-out test size as a fraction of the cache")
    split_seed: int = _option(0, "seed of a random split; 0 = sorted split")
    train_cap: int = _option(0, "train on the first N train shots")
    pin_val: tuple = _option(
        (),
        "comma-separated shots forced into val",
        cli_default="",
        cli_type=str,
        from_cli=_comma_separated,
    )
    pin_train: tuple = _option(
        (),
        "comma-separated shots forced into train, first in order",
        cli_default="",
        cli_type=str,
        from_cli=_comma_separated,
    )
    pin_both: tuple = _option(
        (),
        "comma-separated shots in both train and val (not held out)",
        cli_default="",
        cli_type=str,
        from_cli=_comma_separated,
    )
    # Windows.
    window_stride: int = _option(1, "frames between training-window starts")
    windows_per_shot: int = _option(0, "training windows per shot and epoch; 0 = all")
    window_sample: str = _option(
        "random",
        "which --windows_per_shot windows a shot contributes: 'first' = starts "
        "0..n-1, 'random' = redrawn from all starts each epoch, 'live' = redrawn "
        "from starts at least half before the plasma end (train only)",
        choices=["random", "first", "live"],
    )
    window_jitter: int = _option(
        0,
        "augmentation for --window_sample first: draw each shot's starts from "
        "0..n-1+J each epoch instead of 0..n-1",
    )
    val_window_sample: str = _option(
        "",
        "window sampling of validation; '' = same as --window_sample",
        choices=["", "random", "first"],
    )
    val_windows: int = _option(32, "fixed validation windows")
    val_windows_per_shot: int = _option(
        0, "windows per validation shot; 0 = all starts. Redefines val CE"
    )
    best_metric: str = _option(
        "masked",
        "metric ranking dynamics_best.pt: 'masked' (cosine-prior infilling) or "
        "'gen' (whole frame unknown, real history: the rollout condition)",
        choices=("masked", "gen"),
    )

    @classmethod
    def from_cli(cls, args: argparse.Namespace) -> TrainConfig:
        """Build the config from parsed command-line arguments."""
        values = {}
        for field in dataclasses.fields(cls):
            if not field.metadata["cli"]:
                continue
            value = getattr(args, field.name)
            convert = field.metadata["from_cli"]
            values[field.name] = convert(value) if convert else value
        return cls(**values)


class BestMetric(enum.StrEnum):
    """Validation metric that ranks ``dynamics_best.pt``."""

    MASKED = "masked"
    GEN = "gen"


class ValScore(NamedTuple):
    """Equal-weight validation loss and its per-modality breakdown."""

    loss: float
    per_modality: Dict[str, float]


class _PhaseTimer:
    """Wall time per training phase, accumulated on profiled steps only."""

    PHASES = ("data", "h2d", "fb", "sync", "gn", "opt")

    def __init__(self, device):
        self.cuda = device.type == "cuda"
        self.reset()
        self.mark()

    def reset(self):
        self.seconds = dict.fromkeys(self.PHASES, 0.0)

    def mark(self):
        self._last = time.perf_counter()

    def lap(self, phase, sync=False):
        if sync and self.cuda:
            torch.cuda.synchronize()
        now = time.perf_counter()
        self.seconds[phase] += now - self._last
        self._last = now

    def total(self) -> float:
        return sum(self.seconds.values())

    def summary(self) -> str:
        if self.total() <= 0:
            return ""
        s = self.seconds
        return (
            f" t={self.total():.1f}s(data {s['data']:.1f} h2d {s['h2d']:.2f} "
            f"fb {s['fb']:.1f} sync {s['sync']:.2f} gn {s['gn']:.2f} "
            f"opt {s['opt']:.2f})"
        )


def _append_jsonl(path, record) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def _build_model_config(cache_dir, config: TrainConfig, is_main, log) -> DynamicsConfig:
    """The model configuration: layout and actuator width come from the cache."""
    model_kwargs = dict(
        modalities=cache_modality_specs(cache_dir),
        d_model=config.d_model,
        depth=config.depth,
    )
    probe = next(Path(cache_dir).glob("*.pt"), None)
    if probe is not None:
        cached = torch.load(probe, map_location="cpu", weights_only=False)
        cache_width = int(cached["actuators"].shape[-1])
        model_kwargs["actuator_dim"] = cache_width
        spec_width = sum(n for _key, n in _ACT_SPEC)
        if is_main and cache_width != spec_width:
            log(
                f"[dynamics] cache actuator width {cache_width} != current _ACT_SPEC "
                f"{spec_width} — using the CACHE width (rebuild the cache "
                f"to pick up new actuator channels)"
            )
    for name in ("n_heads", "k0_seed", "n_predict"):
        if getattr(config, name):
            model_kwargs[name] = getattr(config, name)
    model_cfg = DynamicsConfig(**model_kwargs)
    if config.dropout:
        model_cfg.dropout = float(config.dropout)
    if config.ss_final_frac is not None:
        model_cfg.ss_ramp_final_frac = float(config.ss_final_frac)
    if config.ss_ramp_steps is not None:
        model_cfg.ss_ramp_steps = int(config.ss_ramp_steps)
    if config.gen_mask_p is not None:
        model_cfg.gen_mask_p = float(config.gen_mask_p)
    if config.gen_horizon_alpha is not None:
        model_cfg.gen_horizon_alpha = float(config.gen_horizon_alpha)
    if config.gen_horizon_max is not None:
        model_cfg.gen_horizon_max = int(config.gen_horizon_max)
    if config.act_cross_attn is not None:
        model_cfg.act_cross_attn = bool(config.act_cross_attn)
    if config.grad_ckpt is not None:
        model_cfg.grad_checkpointing = bool(int(config.grad_ckpt))
    if config.latent_head is not None:
        model_cfg.latent_head = bool(int(config.latent_head))
        if model_cfg.latent_head and is_main:
            log(
                "[dynamics] LATENT HEAD ON: Gaussian regression of the FSQ grid "
                f"coordinate (levels {tuple(model_cfg.fsq_levels)}) replaces the "
                "1000-way classifier. Train/val CE are Gaussian NLL in coordinate "
                "space -- NOT comparable to token-CE arms."
            )
    if config.lag_embed_k is not None:
        model_cfg.lag_embed_k = int(config.lag_embed_k)
    return model_cfg


def _loss_weights(model_cfg, config: TrainConfig, train_shots, presence, is_main, log):
    """Per-modality training-loss weights, or None for equal weighting.

    Later sources override earlier ones: ``n_tok**mod_weight_pow``, then the
    inverse presence fraction over the training shots, then explicit weights.
    Validation is always scored with equal weights.
    """
    names = [m.name for m in model_cfg.modalities]
    weights = None
    power = config.mod_weight_pow
    if power and abs(float(power)) > 1e-9:
        raw = {m.name: float(m.n_tok) ** float(power) for m in model_cfg.modalities}
        weights = _mean_one(raw, names)
        if is_main:
            log(
                f"[dynamics] token-weighted loss (n_tok**{float(power):g}): "
                + "  ".join(f"{n} x{weights[n]:.3f}" for n in names)
            )
    if presence is not None and config.balance_presence:
        train_set = {str(s) for s in train_shots}
        recorded_fraction = {}
        for name in names:
            n_recorded = sum(
                1
                for shot, flags in presence.items()
                if shot in train_set and flags.get(name, False)
            )
            recorded_fraction[name] = max(n_recorded / max(len(train_set), 1), 1e-3)
        weights = _mean_one({n: 1.0 / recorded_fraction[n] for n in names}, names)
        if is_main:
            log(
                "[dynamics] presence-balanced weights: "
                + "  ".join(
                    f"{n} {100 * recorded_fraction[n]:.1f}%->x{weights[n]:.2f}"
                    for n in names
                )
            )
    if str(config.mod_weights_json).strip():
        weights = resolve_mod_weights(config.mod_weights_json, names)
        if is_main:
            log(
                "[dynamics] EXPLICIT mod_weights (normalised to mean 1): "
                + "  ".join(f"{n} x{weights[n]:.3f}" for n in names)
            )
            log(
                "[dynamics]   NOTE val/gen CE are scored EQUAL-WEIGHT regardless, so "
                "they stay comparable to every historical arm; judge this from the "
                "per-modality readout."
            )
    return weights


def _load_live_end(path_text: str):
    """Read ``{shot: plasma end time}``; None when no file is given."""
    if not str(path_text).strip():
        return None
    path = Path(str(path_text).strip())
    if not path.exists():
        raise FileNotFoundError(f"--live_end_json: no such file {path}")
    live_end = {str(k): float(v) for k, v in json.loads(path.read_text()).items()}
    if not live_end:
        raise ValueError(f"--live_end_json: {path} is empty")
    return live_end


class Trainer:
    """Train the dynamics model on a frame-code cache, resuming from ``out_dir``.

    Gradients are averaged across ranks by an explicit all-reduce: DDP's hooks
    break the gradient-checkpoint recompute. Each optimizer step accumulates
    ``accum_steps`` micro-batches under bf16 autocast, and validation runs every
    ``ckpt_every`` steps on fixed windows and masks.

    Parameters
    ----------
    cache_dir : path-like
        Frame-code cache.
    out_dir : path-like
        Run directory: checkpoints, loss history, loss curve and frozen split.
    config : TrainConfig
        Options of the run.
    log : callable, default=print
        Logger.
    """

    def __init__(self, cache_dir, out_dir, config: TrainConfig, log=print):
        self.cache_dir = cache_dir
        self.out_dir = out_dir
        self.config = config
        self.log = log
        self.ddp = tc._DDPState()
        self.device = self.ddp.device
        self.is_main = self.ddp.is_main

    def fit(self) -> int:
        """Run (or resume) training; return the last optimizer step."""
        self._setup_model_config()
        self._setup_split()
        self._setup_loss_masks()
        self._setup_train_data()
        self._setup_model()
        self._setup_optimizer()
        self._setup_history()
        self._setup_validation()
        self._restore_best()
        return self._train_loop()

    # Set-up, in the order the steps must run.

    def _setup_model_config(self):
        self.model_cfg = _build_model_config(
            self.cache_dir, self.config, self.is_main, self.log
        )

    def _setup_split(self):
        config = self.config
        self.n_cached = len(list(Path(self.cache_dir).glob("*.pt")))
        n_val = config.val_n or max(1, int(self.n_cached * config.val_frac))
        if config.test_n:
            n_test = config.test_n
        elif config.test_frac:
            n_test = max(1, int(self.n_cached * config.test_frac))
        else:
            n_test = 0
        self.train_shots, self.val_shots, self.test_shots = resolve_split(
            self.cache_dir,
            self.out_dir,
            n_val,
            split_seed=config.split_seed,
            train_cap=config.train_cap,
            test_n=n_test,
            pin_val=config.pin_val,
            pin_train=config.pin_train,
            pin_both=config.pin_both,
            write=self.is_main,
            log=self.log if self.is_main else _silent,
        )

    def _setup_loss_masks(self):
        config = self.config
        self.live_end = _load_live_end(config.live_end_json)
        self.use_live = self.live_end is not None
        # The frame-to-time map must come from the cache, or wrong frames are masked.
        self.t0_start = _manifest_t0(self.cache_dir)
        self.use_presence = bool(config.mask_absent)
        self.presence = self._load_presence() if self.use_presence else None
        self.mod_weights = _loss_weights(
            self.model_cfg,
            config,
            self.train_shots,
            self.presence,
            self.is_main,
            self.log,
        )

    def _load_presence(self):
        if self.config.presence_path:
            path = Path(self.config.presence_path)
        else:
            path = Path(self.cache_dir) / "_presence.json"
        if not path.exists():
            if self.is_main:
                self.log(
                    f"[dynamics] no presence map at {path} — building it once "
                    f"(single pass)"
                )
                build_presence(self.cache_dir, out_path=path, log=self.log)
            if self.ddp.world_size > 1:
                dist.barrier()
        presence = json.loads(path.read_text())["presence"]
        if self.is_main:
            self.log(
                f"[dynamics] MASK_ABSENT on: presence map {path} "
                f"({len(presence)} shots)"
            )
        return presence

    def _dataset(self, shots, **window_options) -> FrameCodeDataset:
        return FrameCodeDataset(
            self.cache_dir,
            shots,
            self.model_cfg,
            presence=self.presence,
            live_end=self.live_end,
            t0_start=self.t0_start,
            **window_options,
        )

    def _setup_train_data(self):
        config = self.config
        self.train_ds = self._dataset(
            self.train_shots,
            stride=config.window_stride,
            windows_per_shot=config.windows_per_shot,
            window_sample=config.window_sample,
            window_jitter=config.window_jitter,
        )
        if self.use_live and self.is_main:
            dataset = self.train_ds
            n_measured = sum(1 for shot in dataset.shot_ids if shot in self.live_end)
            live_fractions = [
                n_live / max(entry["n_frames"], 1)
                for n_live, entry in zip(dataset.n_live, dataset.shots)
            ]
            mean_live = sum(live_fractions) / max(len(live_fractions), 1)
            self.log(
                f"[dynamics] POST-SHOT MASKING ON (t0_start={self.t0_start}): "
                f"{n_measured}/{len(dataset.shot_ids)} train shots have a measured "
                f"t_end; mean live fraction {mean_live:.3f}. **THIS REDEFINES val/gen "
                f"CE** -- numbers are NOT comparable to arms trained without it."
            )
        if self.is_main:
            self._log_run_setup()
        world_size = self.ddp.world_size
        self.sampler = (
            DistributedSampler(
                self.train_ds, num_replicas=world_size, rank=self.ddp.rank, shuffle=True
            )
            if world_size > 1
            else None
        )
        self.loader = DataLoader(
            self.train_ds,
            batch_size=config.batch_size,
            sampler=self.sampler,
            shuffle=self.sampler is None,
            num_workers=config.num_workers,
            collate_fn=_collate_frames,
            drop_last=True,
        )

    def _log_run_setup(self):
        config, model_cfg, dataset, log = (
            self.config,
            self.model_cfg,
            self.train_ds,
            self.log,
        )
        log(
            f"[dynamics] cache={self.cache_dir} shots={self.n_cached} "
            f"split_seed={config.split_seed} (train {len(self.train_shots)}/val "
            f"{len(self.val_shots)}/test {len(self.test_shots)} HELD OUT) "
            f"windows={len(dataset)} d_model={config.d_model} depth={config.depth} "
            f"heads={model_cfg.n_heads} frame_tokens={model_cfg.tokens_per_frame} "
            f"win={model_cfg.max_frames}"
        )
        n = config.windows_per_shot
        if n:
            per_val = f"val_windows_per_shot={config.val_windows_per_shot}"
            if config.window_sample == WindowSample.FIRST:
                log(
                    f"[dynamics] window offsets FIRST {n} ONLY (deployment region): "
                    f"starts 0..{n - 1} = frames "
                    f"0..{n - 1 + model_cfg.max_frames - 1}; "
                    f"{per_val}"
                )
            elif config.window_sample == WindowSample.LIVE:
                pool_sizes = [len(pool) for pool in dataset._live_starts]
                n_wide = sum(1 for size in pool_sizes if size > n)
                log(
                    f"[dynamics] window offsets LIVE-RESAMPLED EACH EPOCH: {n}/shot "
                    f"drawn from the live-plasma starts (>=50% of the window before "
                    f"t_end) + starts 0..{n - 1}; pool mean "
                    f"{sum(pool_sizes) / max(len(pool_sizes), 1):.1f} (min "
                    f"{min(pool_sizes) if pool_sizes else 0}, max "
                    f"{max(pool_sizes) if pool_sizes else 0}); {n_wide}/"
                    f"{len(pool_sizes)} shots have a pool wider than the deployment "
                    f"region; {per_val}"
                )
            else:
                log(
                    f"[dynamics] window offsets RESAMPLED EACH EPOCH: {n}/shot drawn "
                    f"from all {len(dataset._starts[0])} start offsets, seeded on "
                    f"epoch; {per_val}"
                )
        log(
            f"[dynamics] objective: gen_mask_p={model_cfg.gen_mask_p} "
            f"label_smoothing={config.label_smoothing} "
            f"gen_horizon_alpha={model_cfg.gen_horizon_alpha} "
            f"gen_horizon_max={model_cfg.gen_horizon_max} "
            f"ss_final_frac={model_cfg.ss_ramp_final_frac} "
            f"ss_ramp_steps={model_cfg.ss_ramp_steps} "
            f"dropout={model_cfg.dropout} mask_absent={int(self.use_presence)} "
            f"block_forcing={int(config.block_forcing)} "
            f"forecast_depth={config.forecast_depth} act_drop_p={config.act_drop_p} "
            f"(val is ALWAYS scored at gen_mask_p=0 so arms stay comparable)"
        )
        jitter_note = (
            ""
            if config.window_sample == WindowSample.FIRST or not config.window_jitter
            else " (IGNORED: only --window_sample first uses it)"
        )
        log(
            f"[dynamics] augmentation (train only): seed_drop_p={config.seed_drop_p} "
            f"seed_column_drop_p={config.seed_column_drop_p} "
            f"actuator_noise={config.actuator_noise} "
            f"window_jitter={config.window_jitter}{jitter_note}"
        )
        world_size, accum = self.ddp.world_size, config.accum_steps
        log(
            f"[dynamics] batch: micro={config.batch_size} x ranks={world_size} "
            f"x accum={accum} -> EFFECTIVE {config.batch_size * world_size * accum}"
            + (f"  ({accum} fwd/bwd per optimizer step)" if accum > 1 else "")
        )
        # The allocator config as the rank process sees it, not the launching shell.
        log(
            f"[dynamics] RANK-SEEN PYTORCH_ALLOC_CONF="
            f"{os.environ.get('PYTORCH_ALLOC_CONF', '<UNSET>')} "
            f"(HIP={os.environ.get('PYTORCH_HIP_ALLOC_CONF', '<unset>')})"
        )

    def _setup_model(self):
        self.model = MaskGITDynamics(self.model_cfg).to(self.device)
        self.latest_path = Path(self.out_dir) / "dynamics_latest.pt"
        self.best_path = Path(self.out_dir) / "dynamics_best.pt"
        self.resumed = None
        self.start_step = 0
        if self.latest_path.exists():
            self.resumed = torch.load(
                self.latest_path, map_location=self.device, weights_only=False
            )
            self.model.load_state_dict(self.resumed["model"])
            self.start_step = int(self.resumed.get("step", 0))
            if self.is_main:
                self.log(
                    f"[dynamics] resumed from {self.latest_path} @ step "
                    f"{self.start_step}"
                )

    def _setup_optimizer(self):
        config = self.config
        lr, betas = config.lr, (0.9, config.beta2)
        # Decaying norms and biases is opt-in: it changes the param-group count,
        # which the optimizer state of an existing chain must match.
        if config.wd_exclude_norms and config.weight_decay > 0:
            parameters = [p for p in self.model.parameters() if p.requires_grad]
            decayed = [p for p in parameters if p.ndim >= 2]
            undecayed = [p for p in parameters if p.ndim < 2]
            self.optimizer = torch.optim.AdamW(
                [
                    {"params": decayed, "weight_decay": config.weight_decay},
                    {"params": undecayed, "weight_decay": 0.0},
                ],
                lr=lr,
                betas=betas,
            )
            if self.is_main:
                self.log(
                    f"[dynamics] wd groups: {len(decayed)} decayed "
                    f"(wd={config.weight_decay}) / {len(undecayed)} undecayed "
                    f"(norms+biases)"
                )
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                betas=betas,
                weight_decay=config.weight_decay,
            )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, self._lr_factor
        )
        self._restore_optimizer()

    def _lr_factor(self, step: int) -> float:
        """Linear warmup, then a cosine from the decay anchor to ``min_lr_ratio``.

        A pure function of the step, so resuming reproduces the schedule. The
        anchor lets a warm-started fork begin its cosine at full rate.
        """
        config = self.config
        if config.warmup_steps and step < config.warmup_steps:
            return (step + 1) / float(config.warmup_steps)
        anchor = float(config.lr_decay_from or config.warmup_steps)
        progress = (step - anchor) / max(1.0, config.steps - anchor)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return config.min_lr_ratio + (1.0 - config.min_lr_ratio) * cosine

    def _restore_optimizer(self):
        checkpoint, start_step = self.resumed, self.start_step
        scheduler_restored = False
        if checkpoint is not None and "opt" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["opt"])
            except (ValueError, KeyError) as e:
                self.log(
                    f"[dynamics] optimizer state INCOMPATIBLE ({type(e).__name__}: "
                    f"{e}); Adam moments restart from zero at step {start_step}"
                )
            if "sched" in checkpoint:
                try:
                    self.scheduler.load_state_dict(checkpoint["sched"])
                    scheduler_restored = True
                except Exception as e:
                    self.log(
                        f"[dynamics] scheduler state incompatible ({type(e).__name__}"
                        f": {e}); replaying {start_step} steps instead"
                    )
        if not scheduler_restored:
            for _ in range(start_step):
                self.scheduler.step()
        # The requested --lr wins over the one restored from the checkpoint.
        lr = self.config.lr
        restored_lr = float(self.optimizer.param_groups[0]["lr"])
        current_lr = lr * self._lr_factor(start_step)
        for group in self.optimizer.param_groups:
            group["lr"] = current_lr
            group["initial_lr"] = lr
        self.scheduler.base_lrs = [lr for _ in self.optimizer.param_groups]
        if (
            self.is_main
            and checkpoint is not None
            and abs(restored_lr - current_lr) > 1e-12
        ):
            self.log(
                f"[dynamics] LR OVERRIDE on resume: checkpoint carried "
                f"{restored_lr:.3e}, requested base lr={lr:.3e} -> {current_lr:.3e} "
                f"at step {start_step}"
            )

    def _setup_history(self):
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        if os.environ.get("IGNITE_DETECT_ANOMALY"):
            torch.autograd.set_detect_anomaly(True)
        self.history_path = Path(self.out_dir) / "loss_history.jsonl"
        if self.is_main and self.start_step and self.history_path.exists():
            # Drop entries past the restart step, so overlapping legs do not repeat.
            kept = [
                line
                for line in self.history_path.read_text().splitlines()
                if line.strip() and json.loads(line).get("step", 0) <= self.start_step
            ]
            self.history_path.write_text("\n".join(kept) + ("\n" if kept else ""))

    def _setup_validation(self):
        config = self.config
        val_sample = WindowSample(
            str(config.val_window_sample or config.window_sample).lower()
        )
        if val_sample is WindowSample.LIVE:
            val_sample = WindowSample.FIRST
            self.log(
                "[dynamics] window_sample=live is train-only; validation forced to "
                "'first'"
            )
        val_ds = self._dataset(
            self.val_shots,
            windows_per_shot=config.val_windows_per_shot,
            window_sample=val_sample,
        )
        # Fixed windows, identical on every rank and every evaluation.
        self.val_batches = []
        if len(val_ds):
            n_windows = min(config.val_windows, len(val_ds))
            if self.is_main:
                same = val_sample == str(config.window_sample).lower()
                self.log(
                    f"[dynamics] val protocol: {len(val_ds)} candidate windows "
                    f"(val_windows_per_shot={config.val_windows_per_shot}, "
                    f"sample={val_sample}{' = train' if same else ' != train'}), "
                    f"scoring {n_windows}"
                )
            positions = torch.linspace(0, len(val_ds) - 1, steps=n_windows).tolist()
            items = [val_ds[int(i)] for i in positions]
            for start in range(0, len(items), config.batch_size):
                self.val_batches.append(
                    _collate_frames(items[start : start + config.batch_size])
                )
        if self.is_main:
            n_fixed = sum(batch[1].shape[0] for batch in self.val_batches)
            self.log(
                f"[dynamics] loss history -> {self.history_path} (+ loss_curve.png "
                f"each ckpt); val: {n_fixed} fixed windows"
            )

    def _restore_best(self):
        self.best_metric = BestMetric(str(self.config.best_metric).lower())
        self.best_loss, self.best_step = float("inf"), -1
        self.evals_since_best = 0
        if not self.best_path.exists():
            return
        try:
            best = torch.load(self.best_path, map_location="cpu", weights_only=False)
            ranked_on = str(best.get("best_metric", "masked"))
            if ranked_on != self.best_metric:
                if self.is_main:
                    self.log(
                        f"[dynamics] best.pt was ranked on {ranked_on!r} but this "
                        f"leg selects on {str(self.best_metric)!r} — best tracking "
                        f"RESTARTS"
                    )
            else:
                self.best_loss = float(best.get("val_loss", float("inf")))
                self.best_step = int(best.get("step", -1))
                if self.is_main:
                    self.log(
                        f"[dynamics] resumed BEST {self.best_metric}_ce="
                        f"{self.best_loss:.4f} @ step {self.best_step}"
                    )
        except Exception as e:
            self.log(
                f"[dynamics] could not read {self.best_path} ({type(e).__name__}); "
                f"best tracking restarts"
            )

    # Training.

    def _train_loop(self) -> int:
        self.step = self.start_step
        # `step` counts optimizer steps, so schedule, cadence and resume ignore
        # accum_steps.
        self.micro_step = 0
        self.accumulated_loss = 0.0
        self.equal_weight_losses = []
        # Profiling only on logged steps, which already synchronise for the norm.
        self.timer = _PhaseTimer(self.device)
        self.model.train()
        stop = False
        while self.step < self.config.steps and not stop:
            if self.sampler is not None:
                self.sampler.set_epoch(self.step)
            # Must precede iterating the loader, which forks the workers.
            self.train_ds.set_epoch(self.step)
            for batch in self.loader:
                stop = self._train_on_batch(*batch)
                if stop:
                    break
        if self.is_main:
            best = (
                f" | best val {self.best_metric}_ce {self.best_loss:.4f} @ step "
                f"{self.best_step} -> dynamics_best.pt"
                if math.isfinite(self.best_loss)
                else ""
            )
            self.log(f"[dynamics] done @ step {self.step}{best}")
        return self.step

    def _is_logged_step(self) -> bool:
        return (self.step + 1) % _LOG_EVERY == 0

    def _train_on_batch(self, codes, actuators, present, live) -> bool:
        """Accumulate one micro-batch; step when the group is full. True = stop."""
        config, timer = self.config, self.timer
        profile = self._is_logged_step() or self.step < _PROFILED_FIRST_STEPS
        if profile:
            timer.lap("data")
        codes = {name: c.to(self.device) for name, c in codes.items()}
        actuators = self._noisy(actuators.to(self.device))
        present = present.to(self.device) if self.use_presence else None
        live = live.to(self.device) if self.use_live else None
        if profile:
            timer.lap("h2d", sync=True)
        if self.micro_step == 0:
            self.optimizer.zero_grad()
        self.ss_fraction = self.model.ss_fraction(self.step)
        # Unweighted per-modality terms give the train loss comparable across
        # arms; each forces a device sync, so collect them on logged steps only.
        per_modality = {} if self._is_logged_step() else None
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = self.model.training_loss(
                codes,
                actuators,
                generator=None,
                ss_frac=self.ss_fraction,
                label_smoothing=config.label_smoothing,
                present=present,
                live=live,
                mod_weights=self.mod_weights,
                per_modality=per_modality,
                block_forcing=config.block_forcing,
                seed_drop_p=config.seed_drop_p,
                seed_column_drop_p=config.seed_column_drop_p,
                act_drop_p=config.act_drop_p,
                forecast_depth=config.forecast_depth,
            )
        if per_modality:
            self.equal_weight_losses.append(
                sum(per_modality.values()) / len(per_modality)
            )
        (loss / config.accum_steps).backward()
        self.accumulated_loss += float(loss.item())
        if profile:
            timer.lap("fb")
        self.micro_step += 1
        if self.micro_step < config.accum_steps:
            timer.mark()
            return False
        self.micro_step = 0
        grad_norm = self._optimizer_step(profile)
        self.step += 1
        return self._after_step(grad_norm)

    def _noisy(self, actuators: torch.Tensor) -> torch.Tensor:
        """Add the actuator-noise augmentation to channels varying in the window.

        Channels constant over the window (dead or held) stay exact, so the model
        can still tell a missing actuator from a live one.
        """
        sigma = float(self.config.actuator_noise)
        if sigma <= 0.0:
            return actuators
        varies = (actuators.amax(dim=1) != actuators.amin(dim=1)).unsqueeze(1)
        return actuators + sigma * torch.randn_like(actuators) * varies

    def _optimizer_step(self, profile: bool):
        """All-reduce, optionally measure and clip, then step; return the norm."""
        self._all_reduce_gradients()
        if profile:
            self.timer.lap("sync", sync=True)
        grad_norm = None
        if profile:
            grad_norm = self._gradient_norm()
            self.timer.lap("gn")
        # Clip the averaged gradient. Measured norms are ~0.05, so 1.0 never fires.
        if self.config.grad_clip and self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip
            )
        self.optimizer.step()
        self.scheduler.step()
        if profile:
            self.timer.lap("opt", sync=True)
        return grad_norm

    def _all_reduce_gradients(self):
        world_size = self.ddp.world_size
        if world_size <= 1:
            return
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                parameter.grad /= world_size

    @torch.no_grad()
    def _gradient_norm(self) -> float:
        squared = torch.zeros((), device=self.device, dtype=torch.float32)
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                squared += parameter.grad.detach().float().pow(2).sum()
        return float(squared.sqrt())

    def _after_step(self, grad_norm) -> bool:
        """Validate, log and checkpoint after an optimizer step. True = stop."""
        config, step = self.config, self.step
        masked = generation = None
        if step % config.ckpt_every == 0:
            # Validation runs on every rank to keep them in lockstep.
            masked = self._validate()
            generation = self._validate(
                mask_ratio=1.0, history_frames=self.model_cfg.k0_seed
            )
        train_loss = self.accumulated_loss / max(1, config.accum_steps)
        equal_weight = (
            sum(self.equal_weight_losses) / len(self.equal_weight_losses)
            if self.equal_weight_losses
            else None
        )
        self.accumulated_loss = 0.0
        self.equal_weight_losses = []
        if self.is_main and step % _LOG_EVERY == 0:
            self._log_train_step(train_loss, equal_weight, grad_norm)
        if self.is_main and masked is not None:
            self._log_validation(masked, generation)
        self.timer.reset()
        stop = False
        if self.is_main and step % config.ckpt_every == 0:
            stop = self._save_checkpoints(masked, generation)
        # Re-arm after validation and checkpointing, so they are not charged to
        # the next step's data time.
        self.timer.mark()
        return stop or step >= config.steps

    def _validate(self, **mask_options) -> ValScore:
        """Equal-weight validation loss over the fixed windows.

        Training weights are never applied, so every arm is scored on the same
        quantity. ``cache_enabled=False`` stops the no-grad autocast pass from
        filling the bf16 weight cache with detached weights that the next
        training forward would reuse.
        """
        if not self.val_batches:
            return ValScore(float("nan"), {})
        self.model.eval()
        generator = torch.Generator(device=self.device)
        generator.manual_seed(_VALIDATION_SEED)
        total = 0.0
        sums: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        with (
            torch.no_grad(),
            torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
                cache_enabled=False,
            ),
        ):
            for codes, actuators, present, live in self.val_batches:
                per_modality: Dict[str, float] = {}
                total += float(
                    self.model.training_loss(
                        {name: c.to(self.device) for name, c in codes.items()},
                        actuators.to(self.device),
                        generator=generator,
                        ss_frac=0.0,
                        present=present.to(self.device) if self.use_presence else None,
                        live=live.to(self.device) if self.use_live else None,
                        mod_weights=None,
                        per_modality=per_modality,
                        gen_mask_p=0.0,
                        **mask_options,
                    )
                )
                for name, value in per_modality.items():
                    sums[name] = sums.get(name, 0.0) + value
                    counts[name] = counts.get(name, 0) + 1
        self.model.train()
        return ValScore(
            total / len(self.val_batches),
            {name: sums[name] / counts[name] for name in sorted(sums)},
        )

    def _log_train_step(self, train_loss, equal_weight, grad_norm):
        memory = ""
        if self.device.type == "cuda":
            allocated = torch.cuda.max_memory_allocated() / 2**30
            reserved = torch.cuda.max_memory_reserved() / 2**30
            memory = (
                f" mem_alloc={allocated:.1f}G mem_resv={reserved:.1f}G "
                f"frag={reserved - allocated:.1f}G"
            )
        norm = f" gnorm={grad_norm:.4f}" if grad_norm is not None else ""
        lr = self.scheduler.get_last_lr()[0]
        self.log(
            f"[dynamics] step {self.step}/{self.config.steps}{norm} "
            f"loss={train_loss:.4f} ss={self.ss_fraction:.3f} lr={lr:.2e}{memory}"
            f"{self.timer.summary()}"
        )
        record = {
            "step": self.step,
            "loss": train_loss,
            "loss_eqw": equal_weight,
            "ss": self.ss_fraction,
            "grad_norm": grad_norm,
            "t_total": round(self.timer.total(), 3),
        }
        record.update(
            {f"t_{phase}": round(s, 3) for phase, s in self.timer.seconds.items()}
        )
        record["lr"] = lr
        _append_jsonl(self.history_path, record)

    def _log_validation(self, masked: ValScore, generation: ValScore):
        self.log(
            f"[dynamics] step {self.step} VAL masked_ce={masked.loss:.4f} "
            f"gen_ce={generation.loss:.4f} (gap {generation.loss - masked.loss:+.4f}; "
            f"selecting on {self.best_metric}; EQUAL-WEIGHT metric)"
        )
        per_modality = generation.per_modality
        if per_modality:
            worst_first = sorted(per_modality, key=lambda name: -per_modality[name])
            self.log(
                "[dynamics]   per-modality gen_ce: "
                + "  ".join(f"{name} {per_modality[name]:.3f}" for name in worst_first)
            )
        _append_jsonl(
            self.history_path,
            {
                "step": self.step,
                "val_loss": masked.loss,
                "gen_loss": generation.loss,
                "val_permod": masked.per_modality,
                "gen_permod": generation.per_modality,
            },
        )
        _render_loss_curve(self.history_path, Path(self.out_dir) / "loss_curve.png")

    def _checkpoint_payload(self) -> dict:
        """Weights, optimizer state and the configuration to rebuild the model."""
        model_cfg, config = self.model_cfg, self.config
        return {
            "model": self.model.state_dict(),
            "opt": self.optimizer.state_dict(),
            "sched": self.scheduler.state_dict(),
            "step": self.step,
            "cfg_gen_mask_p": float(model_cfg.gen_mask_p),
            "cfg_gen_horizon_alpha": float(model_cfg.gen_horizon_alpha),
            "cfg_gen_horizon_max": int(model_cfg.gen_horizon_max),
            "cfg_ss_final_frac": float(model_cfg.ss_ramp_final_frac),
            "cfg_ss_ramp_steps": int(model_cfg.ss_ramp_steps),
            "cfg_act_cross_attn": bool(model_cfg.act_cross_attn),
            "cfg_latent_head": bool(model_cfg.latent_head),
            # "" = per-shot z-scored actuators; else the side cache the model
            # trained on. eval_dynamics refuses a mismatch.
            "cfg_act_global": _ACT_GLOBAL_DIR,
            "cfg_fsq_levels": [int(level) for level in model_cfg.fsq_levels],
            "cfg_grad_checkpointing": bool(model_cfg.grad_checkpointing),
            "cfg_dropout": float(model_cfg.dropout),
            "cfg_block_forcing": bool(config.block_forcing),
            "cfg_forecast_depth": int(config.forecast_depth),
            # > 0: the unconditional branch is trained, so eval may use --guidance.
            "cfg_act_drop_p": float(config.act_drop_p),
            # Training-only augmentation, kept for the record.
            "cfg_seed_drop_p": float(config.seed_drop_p),
            "cfg_seed_column_drop_p": float(config.seed_column_drop_p),
            "cfg_actuator_noise": float(config.actuator_noise),
            "cfg_window_jitter": int(config.window_jitter),
            "cfg_depth": config.depth,
            "cfg_d_model": config.d_model,
            "cfg_n_heads": model_cfg.n_heads,
            "cfg_k0": model_cfg.k0_seed,
            "cfg_n_predict": model_cfg.n_predict,
            "modalities": [
                (m.name, m.family, m.n_tok, m.codebook_size)
                for m in model_cfg.modalities
            ],
        }

    def _save_checkpoints(self, masked: ValScore, generation: ValScore) -> bool:
        """Save the rolling and best checkpoints; True when early stopping fires."""
        payload = self._checkpoint_payload()
        _save_atomically(payload, self.latest_path)
        selected = (generation if self.best_metric is BestMetric.GEN else masked).loss
        if not math.isfinite(selected):
            return False
        if selected < self.best_loss - 1e-6:
            self.best_loss, self.best_step = selected, self.step
            self.evals_since_best = 0
            best = {
                **payload,
                "val_loss": selected,
                "masked_ce": masked.loss,
                "gen_ce": generation.loss,
                "best_metric": str(self.best_metric),
            }
            _save_atomically(best, self.best_path)
            _append_jsonl(
                self.history_path,
                {
                    "step": self.step,
                    "best_val_loss": selected,
                    "best_metric": str(self.best_metric),
                },
            )
            self.log(
                f"[dynamics] new BEST {self.best_metric}_ce={selected:.4f} @ step "
                f"{self.step} (masked {masked.loss:.4f} / gen {generation.loss:.4f})"
            )
            return False
        self.evals_since_best += 1
        patience = self.config.patience
        if patience and self.evals_since_best >= patience:
            self.log(
                f"[dynamics] EARLY STOP: {self.evals_since_best} evals without "
                f"improvement (best {self.best_loss:.4f} @ step {self.best_step}); "
                f"stopping at step {self.step}"
            )
            return True
        return False


def train(
    cache_dir, out_dir, config: Optional[TrainConfig] = None, log=print, **options
):
    """Train the dynamics model on a frame-code cache, resuming from ``out_dir``.

    Writes ``dynamics_latest.pt``, ``dynamics_best.pt``, ``loss_history.jsonl``,
    ``loss_curve.png`` and ``split_shots.json``.

    Parameters
    ----------
    cache_dir : path-like
        Frame-code cache.
    out_dir : path-like
        Run directory.
    config : TrainConfig, default=None
        Options of the run; defaults to ``TrainConfig()``.
    log : callable, default=print
        Logger.
    **options
        Overrides of individual :class:`TrainConfig` fields.

    Returns
    -------
    step : int
        The last optimizer step.
    """
    config = dataclasses.replace(config or TrainConfig(), **options)
    return Trainer(cache_dir, out_dir, config, log=log).fit()


def _add_config_flags(parser: argparse.ArgumentParser) -> None:
    """Add one flag per :class:`TrainConfig` field that has one."""
    for field in dataclasses.fields(TrainConfig):
        meta = field.metadata
        if not meta["cli"]:
            continue
        default = meta["cli_default"]
        kwargs = {"help": meta["help"]}
        if isinstance(default, bool) and meta["cli_type"] is None:
            kwargs["action"] = "store_true"
        else:
            kwargs["default"] = default
            flag_type = meta["cli_type"]
            if flag_type is None and isinstance(default, (int, float, str)):
                flag_type = type(default)
            if flag_type is not None:
                kwargs["type"] = flag_type
            if meta["choices"] is not None:
                kwargs["choices"] = meta["choices"]
        parser.add_argument(f"--{field.name}", **kwargs)


def build_arg_parser():
    """Return the command-line parser of the trainer and cache tools."""
    parser = argparse.ArgumentParser("IGNITE Phase-B dynamics trainer")
    add = parser.add_argument
    add("--cache_dir", required=True, help="pre-encoded frame-code cache dir")
    add("--out_dir", default=None, help="run directory (not needed with --precompute)")
    _add_config_flags(parser)
    tools = parser.add_argument_group("cache tools (each runs instead of training)")
    tools.add_argument(
        "--precompute", action="store_true", help="build the frame-code cache"
    )
    tools.add_argument(
        "--build_presence", action="store_true", help="build the presence map"
    )
    tools.add_argument(
        "--patch_actuators",
        action="store_true",
        help="rewrite only the actuators of an existing cache with the aligned time "
        "base, verifying and backing up each entry",
    )
    tools.add_argument(
        "--data_dir",
        default=None,
        help="H5 directory (default: spike.DEFAULT_DATA_DIR)",
    )
    tools.add_argument(
        "--max_shots", type=int, default=0, help="--precompute: first N shots (0 = all)"
    )
    tools.add_argument(
        "--codec_tmpl",
        default=None,
        help="--precompute: codec checkpoint template tried before the frozen "
        "manifest, e.g. 'path/codecs/{m}/codec_best.pt'",
    )
    tools.add_argument(
        "--shot_sample",
        type=int,
        default=0,
        help="--precompute: seeded random sample of N shots (overrides --max_shots)",
    )
    tools.add_argument("--shot_seed", type=int, default=0, help="seed of --shot_sample")
    tools.add_argument(
        "--t0_start",
        type=float,
        default=1.0,
        help="--precompute: shot time [s] of frame 0",
    )
    tools.add_argument(
        "--shot_timeout_s",
        type=int,
        default=900,
        help="--precompute: per-shot encode budget [s]",
    )
    tools.add_argument(
        "--dry_run",
        action="store_true",
        help="--patch_actuators: verify and report, write nothing",
    )
    return parser


def main(argv=None):
    """Run the tool selected by the command line."""
    args = build_arg_parser().parse_args(argv)
    data_dir = args.data_dir or spike.DEFAULT_DATA_DIR
    if args.build_presence:
        out = args.presence_path or (Path(args.cache_dir) / "_presence.json")
        build_presence(args.cache_dir, out_path=out)
        return 0
    if args.patch_actuators:
        return run_patch_actuators(args.cache_dir, data_dir, dry_run=args.dry_run)
    if args.precompute:
        return run_precompute(
            args.cache_dir,
            data_dir,
            max_shots=args.max_shots,
            codec_tmpl=args.codec_tmpl,
            shot_sample=args.shot_sample,
            shot_seed=args.shot_seed,
            t0_start=args.t0_start,
            shot_timeout_s=args.shot_timeout_s,
        )
    if not args.out_dir:
        raise SystemExit(
            "--out_dir is required for training (omit only with --precompute)"
        )
    return train(args.cache_dir, args.out_dir, config=TrainConfig.from_cli(args))


if __name__ == "__main__":
    main()
