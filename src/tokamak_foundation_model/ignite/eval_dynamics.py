r"""Evaluation of the IGNITE dynamics model: rollouts, metrics and comparison figures.

Loads a checkpoint, seeds ``K0`` real frames from the frame-code cache, rolls the
model out and decodes the ground-truth and predicted codes through the frozen
codecs. Token accuracy and decoded nRMSE are reported against persistence (the last
seed frame held constant), and a ground-truth versus prediction figure is rendered.

Usage, in the ``stellarai`` pixi environment::

    python -m tokamak_foundation_model.ignite.eval_dynamics \
        --ckpt <run>/dynamics_latest.pt --shot 200729 --out_dir <dir>

Environment
-----------
IGNITE_EVAL_MODALITIES
    Comma-separated modalities to decode and score.
IGNITE_EVAL_RAW_GT
    ``0`` scores against the codec round-trip instead of the measured signal.
IGNITE_DATA_DIR
    H5 directory of the measured signals.
IGNITE_EVAL_BF16
    ``1`` runs the rollout under bf16 autocast, which lowers accuracy.
IGNITE_HISTORY_CACHE
    ``1`` (the default of this CLI) keeps the committed frames' temporal keys and
    values during a rollout, so each decode step runs one frame instead of the whole
    history (``MaskGITDynamics.rollout(history_cache=...)``); about 8x faster per
    shot, same result up to float rounding. ``0`` runs the full pass. Library calls
    (e.g. training's skill check) default to ``0``.
IGNITE_ACT_GLOBAL
    Must match training, see :func:`load_model`.
IGNITE_FIG_*, IGNITE_CMAP_*
    Figure options, see :class:`FigureOptions`.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, NamedTuple, Optional, Tuple

import numpy as np
import torch

from . import spike
from . import train_dynamics as td
from .dynamics_config import (
    FROZEN_MODALITIES,  # noqa: F401 (re-exported)
    DynamicsConfig,
    ModalitySpec,
)
from .maskgit import MaskGITDynamics

# Modalities decoded and scored. A codec that fails to decode aborts the whole
# evaluation, so IGNITE_EVAL_MODALITIES can restrict the set.
EVAL_MODALITIES: Tuple[str, ...] = tuple(
    name
    for name in os.environ.get(
        "IGNITE_EVAL_MODALITIES",
        "ece,bes,mhr,co2,ts_core_density,ts_core_temp,ts_tangential_density,"
        "ts_tangential_temp,cer_ti,tangtv_lower,tangtv_upper",
    ).split(",")
    if name.strip()
)
SPECTRO = ("ece", "bes", "mhr", "co2")
VIDEO = ("tangtv_lower", "tangtv_upper")
SLOWTS = (
    "ts_core_density",
    "ts_core_temp",
    "ts_tangential_density",
    "ts_tangential_temp",
    "cer_ti",
)

_DEFAULT_CACHE = "/lustre/orion/fus187/proj-shared/models/ignite_production/frame_codes"
_DECODE_CHUNK = 16
_SHUFFLE_SEED = 1234
_MERGE_TIMEOUT_S = 3600
_MERGE_POLL_S = 10
FRAME_S = 0.05
WARMUP_S = 1.0


def _env_flag(name: str, default: str) -> bool:
    return os.environ.get(name, default) != "0"


def _env_selection(name: str, default: Tuple[str, ...], allowed: Tuple[str, ...]):
    """Read a figure selection: comma-separated names, or ``"all"``."""
    value = os.environ.get(name)
    if not value:
        return default
    if value.strip().lower() == "all":
        return allowed
    return tuple(n for n in (x.strip() for x in value.split(",")) if n in allowed)


@dataclasses.dataclass(frozen=True)
class FigureOptions:
    """Options of the comparison figure, read once from ``IGNITE_FIG_*``.

    Parameters
    ----------
    spectro, slowts : tuple of str
        Modalities drawn (the metrics cover all of them).
    width_in, font_pt : float
        Printed width and font size; the canvas is the delivered size.
    show_skill, show_r, show_nrmse, show_channel, show_tag : bool
        Parts of the panel annotations, all off by default (``IGNITE_FIG_SKILL``,
        ``_R``, ``_NRMSE``, ``_CH``, ``_TAG`` = 1 turns one on).
    show_video_nrmse : bool
        Add the per-frame video nRMSE row.
    gt_only : bool
        Draw the same figure without any prediction artist.
    cmap_spectro, cmap_video, cmap_diff : str
        Colour maps.
    legend_loc : str
        Location of the legend in the first panel.
    """

    spectro: Tuple[str, ...]
    slowts: Tuple[str, ...]
    width_in: float
    font_pt: float
    show_skill: bool
    show_r: bool
    show_nrmse: bool
    show_channel: bool
    show_tag: bool
    show_video_nrmse: bool
    gt_only: bool
    cmap_spectro: str
    cmap_video: str
    cmap_diff: str
    legend_loc: str

    @classmethod
    def from_env(cls) -> FigureOptions:
        """Read the options from the environment."""
        env = os.environ.get
        return cls(
            spectro=_env_selection("IGNITE_FIG_SPECTRO", ("ece", "mhr"), SPECTRO),
            slowts=_env_selection(
                "IGNITE_FIG_SLOWTS",
                ("ts_core_density", "ts_core_temp", "cer_ti"),
                SLOWTS,
            ),
            width_in=float(env("IGNITE_FIG_WIDTH_IN", "6.5")),
            font_pt=float(env("IGNITE_FIG_FONT_PT", "12")),
            show_skill=_env_flag("IGNITE_FIG_SKILL", "0"),
            show_r=_env_flag("IGNITE_FIG_R", "0"),
            show_nrmse=_env_flag("IGNITE_FIG_NRMSE", "0"),
            show_channel=_env_flag("IGNITE_FIG_CH", "0"),
            show_tag=_env_flag("IGNITE_FIG_TAG", "0"),
            show_video_nrmse=_env_flag("IGNITE_FIG_VRMSE", "0"),
            gt_only=_env_flag("IGNITE_FIG_GT_ONLY", "0"),
            cmap_spectro=env("IGNITE_CMAP_SPECTRO", "Blues"),
            cmap_video=env("IGNITE_CMAP_VIDEO", "gray"),
            cmap_diff=env("IGNITE_CMAP_DIFF", "RdBu_r"),
            legend_loc=env("IGNITE_FIG_LEGEND_LOC", "upper left"),
        )


FIGURE_OPTIONS = FigureOptions.from_env()
FIG_SPECTRO = FIGURE_OPTIONS.spectro
FIG_SLOWTS = FIGURE_OPTIONS.slowts
FIG_W_IN = FIGURE_OPTIONS.width_in
FIG_PT = FIGURE_OPTIONS.font_pt
# Liberation Serif is metrically identical to Times New Roman, absent on Frontier.
FIG_SERIF = ["Times New Roman", "Liberation Serif", "STIXGeneral", "DejaVu Serif"]


_COORD_MAPS: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}


def _coords_to_quant(codec, coords: torch.Tensor) -> torch.Tensor:
    """Map [0, 1] FSQ grid coordinates (..., n_levels) to the codec's decoder input.

    ``indices_to_codes`` is affine in the level indices (scale and shift, optional
    rotation, linear projection), so the map is fitted once per codec on its own code
    table and checked to be exact. The expected coordinates of a token therefore decode
    from exactly the expected decoder input.
    """
    fsq = codec.quantizer.fsq
    key = id(codec)
    if key not in _COORD_MAPS:
        levels = fsq._levels.to(coords.device)
        index = torch.arange(int(levels.prod()), device=coords.device)
        level_index = fsq.indices_to_level_indices(index).float()
        target = fsq.indices_to_codes(index).float()
        design = torch.cat([level_index, torch.ones_like(level_index[:, :1])], dim=1)
        weight = torch.linalg.lstsq(design.double(), target.double()).solution.float()
        error = float((design @ weight - target).abs().max())
        if error > 1e-4:
            raise RuntimeError(f"FSQ code map is not affine (max error {error:.2e})")
        _COORD_MAPS[key] = (levels, weight)
    levels, weight = _COORD_MAPS[key]
    if coords.shape[-1] != levels.numel():
        raise ValueError(f"coords have {coords.shape[-1]} levels, codec has {levels.numel()}")
    level_index = coords.float() * (levels.float() - 1.0)
    return torch.cat([level_index, torch.ones_like(level_index[..., :1])], dim=-1) @ weight


@torch.no_grad()
def decode_flat(codec, flat: torch.Tensor) -> torch.Tensor:
    """Decode flat FSQ codes (T, n_tok) into the codec's normalised space.

    A floating-point ``flat`` of shape (T, n_tok, n_levels) is read as grid coordinates
    in ``[0, 1]`` (an expected frame, see ``MaskGITDynamics.rollout(return_mean=True)``)
    and decoded without rounding to a code.
    """
    if flat.is_floating_point():
        return codec.decode(_coords_to_quant(codec, flat))
    return codec.decode(codec.quantizer.fsq.indices_to_codes(flat))


def archived(codes, key: str) -> torch.Tensor:
    """One array of a ``codes_<shot>_step<N>.npz`` archive as a tensor for decode_flat.

    Code arrays (``gt__``, ``pred__``, ``real__``) come back as longs, expected frames
    (``mean__``, ``realmean__``) as float coordinates.
    """
    value = torch.from_numpy(np.asarray(codes[key]))
    return value.float() if value.is_floating_point() else value.long()


@torch.no_grad()
def decode_flat_chunked(
    codec, flat: torch.Tensor, device, chunk: int = _DECODE_CHUNK
) -> np.ndarray:
    """Decode (T, n_tok) codes in chunks of frames; same result, bounded memory.

    Also takes (T, n_tok, n_levels) float coordinates (see :func:`decode_flat`).
    """
    if not flat.is_floating_point():
        flat = flat.long()
    decoded = [
        decode_flat(codec, flat[start : start + chunk].to(device)).float().cpu()
        for start in range(0, flat.shape[0], chunk)
    ]
    return torch.cat(decoded, dim=0).numpy()


def _infer_actuator_dim(cfg, state_dict) -> None:
    weight = state_dict.get("backbone.act_embed.weight")
    if weight is None:
        return
    width = int(weight.shape[1])
    if width != int(cfg.actuator_dim):
        print(
            f"[eval] actuator_dim {cfg.actuator_dim} -> {width} "
            f"(inferred from backbone.act_embed.weight)",
            flush=True,
        )
        cfg.actuator_dim = width


def _infer_head_type(cfg, checkpoint) -> None:
    head = checkpoint["model"].get(
        f"backbone.tok.heads.{cfg.modalities[0].name}.weight"
    )
    if head is None:
        return
    width = int(head.shape[0])
    is_latent = width == 2 * len(cfg.fsq_levels)
    persisted = bool(checkpoint.get("cfg_latent_head", is_latent))
    if is_latent != persisted:
        print(
            f"[eval] WARNING cfg_latent_head={persisted} but head width {width} "
            f"says {is_latent}; trusting the weights",
            flush=True,
        )
    cfg.latent_head = is_latent
    if cfg.latent_head:
        print(
            f"[eval] LATENT HEAD checkpoint (head width {width}, "
            f"fsq_levels {cfg.fsq_levels}); predictions round to codes",
            flush=True,
        )


def _check_actuator_normalisation(checkpoint) -> None:
    """Refuse to score a model on actuators normalised differently from training."""
    # Checkpoints without the key were all trained on per-shot actuators ("").
    trained = str(checkpoint.get("cfg_act_global", "") or "")
    evaluated = os.environ.get("IGNITE_ACT_GLOBAL", "").strip()

    def describe(side_cache):
        if side_cache:
            return "IGNITE_ACT_GLOBAL=" + side_cache
        return "PER-SHOT z-scored actuators"

    if bool(trained) != bool(evaluated):
        raise RuntimeError(
            f"actuator normalisation mismatch: checkpoint trained with "
            f"{describe(trained)}, eval run with {describe(evaluated)}. "
            f"Set IGNITE_ACT_GLOBAL to match."
        )
    if trained:
        print(
            f"[eval] actuators: dataset-wide normalised side cache {evaluated} "
            f"(trained on {trained})",
            flush=True,
        )


def load_model(ckpt_path: Path, device) -> Tuple[MaskGITDynamics, DynamicsConfig, int]:
    """Rebuild a trained model from its checkpoint.

    The architecture is inferred from the weights where possible (actuator width,
    latent versus classifier heads), so a stale config default cannot break a
    load.

    Parameters
    ----------
    ckpt_path : Path
        ``dynamics_latest.pt`` or ``dynamics_best.pt``.
    device : torch.device
        Target device.

    Returns
    -------
    model : MaskGITDynamics
        The model in eval mode, with dropout and checkpointing off.
    cfg : DynamicsConfig
        Its configuration.
    step : int
        Training step of the checkpoint.

    Raises
    ------
    RuntimeError
        If ``IGNITE_ACT_GLOBAL`` is set for evaluation but not for training, or
        the reverse: the model would see actuators it was not trained on.
    """
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_kwargs = dict(
        depth=int(checkpoint["cfg_depth"]), d_model=int(checkpoint["cfg_d_model"])
    )
    if "modalities" in checkpoint:
        model_kwargs["modalities"] = tuple(
            ModalitySpec(*spec) for spec in checkpoint["modalities"]
        )
    for key, field in (
        ("cfg_n_heads", "n_heads"),
        ("cfg_k0", "k0_seed"),
        ("cfg_n_predict", "n_predict"),
    ):
        if key in checkpoint:
            model_kwargs[field] = int(checkpoint[key])
    cfg = DynamicsConfig(**model_kwargs)
    cfg.grad_checkpointing = False
    _infer_actuator_dim(cfg, checkpoint["model"])
    if "cfg_act_cross_attn" in checkpoint:
        cfg.act_cross_attn = bool(checkpoint["cfg_act_cross_attn"])
    if "cfg_dropout" in checkpoint:
        cfg.dropout = float(checkpoint["cfg_dropout"])
    if "cfg_fsq_levels" in checkpoint:
        cfg.fsq_levels = tuple(int(level) for level in checkpoint["cfg_fsq_levels"])
    _infer_head_type(cfg, checkpoint)
    _check_actuator_normalisation(checkpoint)
    cfg.dropout = 0.0
    model = MaskGITDynamics(cfg).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    return model, cfg, int(checkpoint.get("step", 0))


def load_shot_cache(cache_dir: Path, shot: str) -> Dict:
    """Load one shot's frame-code cache entry."""
    path = Path(cache_dir) / f"{shot}.pt"
    if not path.exists():
        raise FileNotFoundError(f"frame-code cache not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


# Accepted ``actuator_mode`` forms. On per-shot z-scored actuators a gain or offset
# edit is a no-op, so the physical-unit modes (scale, ramp, set, balance) need the
# dataset-wide normalisation of IGNITE_ACT_GLOBAL.
ACTUATOR_MODES = (
    "real",
    "zero",
    "freeze",
    "shuffle",
    "donor:<shot>",
    "group_zero:<group>",
    "scale:<group>=<factor>[@<frame>][+<group>=<factor>[@<frame>]...]",
    "ramp:<group>=<factor>@<frame>/<ramp_frames>",
    "set:<group>[<i>,...]=<value>@<frame>[-<end>][+...]",
    "balance:off=<i>,..;counter=<j>,..;nm_per_mw=<rate>;kv=<kV>@<frame>",
    "linear:<ramp_frames>[/<amount>]:<mode>",
    "file:<plan.npz>",
)


def _global_actuator_stats():
    """Per-channel ``mu`` and ``sd`` of the ``IGNITE_ACT_GLOBAL`` normalisation.

    Returns None when the variable or its stats file is missing.
    """
    side_cache = os.environ.get("IGNITE_ACT_GLOBAL", "").strip()
    stats = Path(side_cache) / "act_global_stats.npz" if side_cache else None
    if stats is None or not stats.exists():
        return None
    loaded = np.load(stats, allow_pickle=True)
    return (
        torch.as_tensor(loaded["mu"], dtype=torch.float32),
        torch.as_tensor(loaded["sd"], dtype=torch.float32),
    )


def _scale_actuators(edited, spec: str, K0: int, n_frames: int) -> None:
    """Scale actuator groups in physical units, in place, for ``scale:`` modes.

    ``spec`` is ``<group>=<factor>[@<frame>]`` terms joined by ``+``; each scales
    that group's physical value by ``factor`` from ``frame`` (default ``K0``, the
    first predicted frame) to the end of the window. ``factor`` 0 switches the
    actuator off. Needs the dataset-wide normalisation (``IGNITE_ACT_GLOBAL``):
    a normalised value g maps to physical ``g * sd + mu``.
    """
    stats = _global_actuator_stats()
    if stats is None:
        raise ValueError("scale: actuator modes need IGNITE_ACT_GLOBAL with its stats")
    mu, sd = stats
    for term in spec.split("+"):
        target, _, start = term.partition("@")
        group, _, factor = target.partition("=")
        columns = _actuator_group_columns(group.strip())
        first = int(start) if start else K0
        if not K0 <= first < n_frames:
            raise ValueError(f"scale start frame {first} outside [{K0}, {n_frames})")
        frames = slice(first, n_frames)
        m, d = mu[columns], sd[columns]
        varies = d > 0
        physical = edited[frames, columns] * d + m
        rescaled = (physical * float(factor) - m) / torch.where(varies, d, torch.ones_like(d))
        edited[frames, columns] = torch.where(varies, rescaled, edited[frames, columns])


def _set_actuators(edited, spec: str, K0: int, n_frames: int) -> None:
    """Set single actuator channels to physical values, in place, for ``set:`` modes.

    ``spec`` is ``<group>[<i>,<j>...]=<value>@<frame>[-<end>]`` terms joined by
    ``+``. The listed channels of the group (indices into its
    :data:`~.train_dynamics._ACT_SPEC` columns, e.g. the 8 beam lines 15L, 15R, 21L,
    21R, 30L, 30R, 33L, 33R) take ``value`` in physical units from ``frame`` to
    ``end`` (exclusive; default the end of the window). Needs ``IGNITE_ACT_GLOBAL``,
    as :func:`_scale_actuators`.
    """
    stats = _global_actuator_stats()
    if stats is None:
        raise ValueError("set: actuator modes need IGNITE_ACT_GLOBAL with its stats")
    mu, sd = stats
    for term in spec.split("+"):
        target, _, timing = term.partition("@")
        group_part, _, value = target.partition("=")
        group, _, index_text = group_part.partition("[")
        columns = _actuator_group_columns(group.strip())
        indices = [int(i) for i in index_text.rstrip("]").split(",")] if index_text else \
            list(range(columns.stop - columns.start))
        start, _, end = timing.partition("-")
        first = int(start) if start else K0
        last = int(end) if end else n_frames
        if not K0 <= first < last <= n_frames:
            raise ValueError(f"set frames {first}-{last} outside [{K0}, {n_frames}]")
        for i in indices:
            c = columns.start + i
            if sd[c] > 0:
                edited[first:last, c] = (float(value) - mu[c]) / sd[c]


def _balance_actuators(edited, spec: str, K0: int, n_frames: int) -> None:
    """Zero the net NBI torque with counter beams, in place, for ``balance:`` modes.

    ``spec`` is ``off=<i>,..;counter=<j>,..;nm_per_mw=<rate>;kv=<voltage>@<frame>``
    with indices into the 8 beam lines (15L, 15R, 21L, 21R, 30L, 30R, 33L, 33R) of
    the ``pinj``, ``beam_voltage`` and ``tinj`` groups. From ``frame`` on the ``off``
    beams are switched off and, frame by frame, the ``counter`` beams share the
    torque that cancels the remaining beams' torque: power ``torque / |nm_per_mw|``,
    voltage ``kv`` while on. The remaining beams keep their recorded waveform,
    ramp-down included, so the counter power follows it. Needs ``IGNITE_ACT_GLOBAL``.
    """
    stats = _global_actuator_stats()
    if stats is None:
        raise ValueError("balance: actuator modes need IGNITE_ACT_GLOBAL with its stats")
    mu, sd = stats
    options, _, start = spec.partition("@")
    fields = dict(term.split("=", 1) for term in options.split(";") if term)
    off = [int(i) for i in fields["off"].split(",")]
    counter = [int(i) for i in fields["counter"].split(",")]
    rate = abs(float(fields["nm_per_mw"]))
    volts = float(fields["kv"]) * 1e3
    first = int(start) if start else K0
    if not K0 <= first < n_frames:
        raise ValueError(f"balance start frame {first} outside [{K0}, {n_frames})")
    frames = slice(first, n_frames)
    groups = {name: _actuator_group_columns(name) for name in ("pinj", "beam_voltage", "tinj")}
    physical = {
        name: edited[frames, cols] * sd[cols] + mu[cols] for name, cols in groups.items()
    }
    kept = [i for i in range(physical["tinj"].shape[1]) if i not in off and i not in counter]
    share = (physical["tinj"][:, kept].sum(1).clamp_min(0.0) / len(counter)).unsqueeze(1)
    for name in groups:
        physical[name][:, off] = 0.0
    on = (share > 0).float()
    physical["tinj"][:, counter] = -share
    physical["pinj"][:, counter] = share / rate * 1e6
    physical["beam_voltage"][:, counter] = volts * on
    for name, cols in groups.items():
        d = sd[cols]
        varies = d > 0
        normed = (physical[name] - mu[cols]) / torch.where(varies, d, torch.ones_like(d))
        edited[frames, cols] = torch.where(varies, normed, edited[frames, cols])


def _ramp_actuators(edited, spec: str, K0: int, n_frames: int) -> None:
    """Hold an actuator group from ``frame`` on and ramp it, in place, for ``ramp:``.

    ``spec`` is ``<group>=<factor>@<frame>/<ramp_frames>``. From ``frame`` (default
    ``K0``) each channel follows, in physical units,
    ``I0 + (factor * I0 - I0) * clip((t - frame) / ramp_frames, 0, 1)``, where ``I0``
    is its value at ``frame``: factor 1 holds ``I0``, factor 0.5 ramps to half of it
    over ``ramp_frames`` frames and then holds. Needs ``IGNITE_ACT_GLOBAL``, as
    :func:`_scale_actuators`.
    """
    stats = _global_actuator_stats()
    if stats is None:
        raise ValueError("ramp: actuator modes need IGNITE_ACT_GLOBAL with its stats")
    mu, sd = stats
    target, _, timing = spec.partition("@")
    group, _, factor = target.partition("=")
    start, _, ramp = timing.partition("/")
    columns = _actuator_group_columns(group.strip())
    first = int(start) if start else K0
    if not K0 <= first < n_frames:
        raise ValueError(f"ramp start frame {first} outside [{K0}, {n_frames})")
    m, d = mu[columns], sd[columns]
    varies = d > 0
    scale = torch.where(varies, d, torch.ones_like(d))
    initial = edited[first, columns] * d + m
    steps = torch.arange(n_frames - first, dtype=edited.dtype)
    weight = (steps / max(1, int(ramp or 0))).clamp(0, 1).unsqueeze(1)
    physical = initial + (initial * float(factor) - initial) * weight
    edited[first:n_frames, columns] = torch.where(
        varies, (physical - m) / scale, edited[first:n_frames, columns]
    )


def _actuator_group_columns(group: str) -> slice:
    """Columns of one :data:`~.train_dynamics._ACT_SPEC` group."""
    offset = 0
    for key, n_channels in td._ACT_SPEC:
        if key == group:
            return slice(offset, offset + n_channels)
        offset += n_channels
    known = [key for key, _ in td._ACT_SPEC]
    raise ValueError(f"unknown actuator group {group!r}; known: {known}")


def apply_actuator_mode(
    act: torch.Tensor, mode: str, K0: int, cache_dir=None, F: int = None
) -> torch.Tensor:
    """Return a counterfactual copy of the actuators over the predicted frames.

    Parameters
    ----------
    act : torch.Tensor of shape (F, n_channels)
        Cached actuators; not modified.
    mode : str
        ``"real"`` (unchanged), ``"zero"``, ``"freeze"`` (hold the last seed
        frame), ``"shuffle"`` (seeded permutation of the predicted frames),
        ``"donor:<shot>"`` (another shot's actuators), ``"group_zero:<group>"``
        (zero one :data:`~.train_dynamics._ACT_SPEC` group) or
        ``"scale:<group>=<factor>[@<frame>]"`` (scale a group in physical units
        from ``frame`` on, :func:`_scale_actuators`) or
        ``"ramp:<group>=<factor>@<frame>/<ramp_frames>"`` (hold a group at its value at
        ``frame``, then ramp it linearly to ``factor`` times that value,
        :func:`_ramp_actuators`), or one of the ``set:``, ``balance:`` and
        ``linear:`` forms of :data:`ACTUATOR_MODES` (:func:`_set_actuators`,
        :func:`_balance_actuators`; ``linear:`` ramps into another mode), or
        ``"file:<plan.npz>"`` (the ``act_norm`` array of a shot-designer plan).
    K0 : int
        First predicted frame; the seed frames stay real.
    cache_dir : path-like, default=None
        Cache holding the donor shot.
    F : int, default=None
        End of the predicted region; defaults to ``len(act)``.

    Returns
    -------
    out : torch.Tensor of shape (F, n_channels)
        The edited actuators.
    """
    n_frames = int(act.shape[0]) if F is None else F
    predicted = slice(K0, n_frames)
    edited = act.clone()
    kind, has_argument, argument = mode.partition(":")
    if mode == "real":
        pass
    elif mode == "zero":
        edited[predicted] = 0.0
    elif mode == "freeze":
        edited[predicted] = act[K0 - 1 : K0]
    elif mode == "shuffle":
        generator = torch.Generator().manual_seed(_SHUFFLE_SEED + n_frames)
        edited[predicted] = act[torch.randperm(n_frames - K0, generator=generator) + K0]
    elif kind == "donor" and has_argument:
        if cache_dir is None:
            raise ValueError("donor mode needs cache_dir")
        donor_cache = load_shot_cache(Path(cache_dir), argument)
        # The donor must use the same normalisation as the edited shot.
        _load_global_actuators(donor_cache, argument)
        donor = donor_cache["actuators"].float()
        if donor.shape[0] < n_frames:
            raise RuntimeError(
                f"donor {argument} has {donor.shape[0]} frames < F={n_frames}"
            )
        edited[predicted] = donor[predicted]
    elif kind == "group_zero" and has_argument:
        edited[predicted, _actuator_group_columns(argument)] = 0.0
    elif kind == "scale" and has_argument:
        _scale_actuators(edited, argument, K0, n_frames)
    elif kind == "ramp" and has_argument:
        _ramp_actuators(edited, argument, K0, n_frames)
    elif kind == "set" and has_argument:
        _set_actuators(edited, argument, K0, n_frames)
    elif kind == "balance" and has_argument:
        _balance_actuators(edited, argument, K0, n_frames)
    elif kind == "file" and has_argument:
        # A whole actuator plan from the shot designer (designer.simulate.export): the
        # normalised actuators of every cache frame, measured except where edited.
        plan = torch.as_tensor(np.load(argument)["act_norm"], dtype=edited.dtype)
        if plan.shape[0] < n_frames or plan.shape[1] != edited.shape[1]:
            raise ValueError(f"plan {argument} has shape {tuple(plan.shape)}; needs at "
                             f"least ({n_frames}, {edited.shape[1]})")
        # The seed frames are never edited, so they identify the shot and normalisation.
        if not torch.allclose(plan[:K0], act[:K0].to(plan.dtype), atol=1e-3):
            raise ValueError(f"plan {argument} was built for another shot or normalisation")
        edited[predicted] = plan[predicted]
    elif kind == "linear" and has_argument:
        # Ramp from the measured actuators into another mode's: every channel moves
        # linearly, frame by frame, from its measured value to the target mode's value
        # over ramp_frames, starting where the target first differs, and stays at
        # ``amount`` of the way (default 1) after. Beam power, voltage and torque of a
        # balance target ramp together, as a modulated beam's frame averages do.
        timing, _, inner = argument.partition(":")
        ramp_text, _, amount = timing.partition("/")
        target = apply_actuator_mode(act, inner, K0, cache_dir, n_frames)
        changed = (target[:n_frames] != act[:n_frames]).any(dim=1).nonzero()
        if len(changed):
            first = int(changed[0])
            steps = torch.arange(n_frames - first, dtype=edited.dtype)
            weight = (steps / max(1, int(ramp_text))).clamp(0, 1) * float(amount or 1)
            edited[first:n_frames] = act[first:n_frames] + weight.unsqueeze(1) * (
                target[first:n_frames] - act[first:n_frames]
            )
    else:
        raise ValueError(
            f"unknown actuator_mode {mode!r}; expected one of {ACTUATOR_MODES}"
        )
    return edited


class Rollout(NamedTuple):
    """Ground-truth and predicted codes of one rollout, ``(F, n_tok)`` each.

    ``pred_mean`` holds the expected frames ``(F, n_tok, n_levels)`` in ``[0, 1]``
    coordinates when the rollout could produce them (categorical head, no consensus).
    """

    gt_codes: Dict[str, torch.Tensor]
    pred_codes: Dict[str, torch.Tensor]
    k0: int
    n_frames: int
    pred_mean: Optional[Dict[str, torch.Tensor]] = None


def _load_global_actuators(cache: Dict, shot) -> None:
    """Replace the cached actuators by the ``IGNITE_ACT_GLOBAL`` ones, in place."""
    side_cache = os.environ.get("IGNITE_ACT_GLOBAL", "").strip()
    if not side_cache:
        return
    if shot is None:
        raise ValueError("IGNITE_ACT_GLOBAL is set: rollout_shot needs the shot id")
    entry = Path(side_cache) / f"{shot}.pt"
    if not entry.exists():
        raise FileNotFoundError(
            f"IGNITE_ACT_GLOBAL={side_cache} has no entry for shot {shot}"
        )
    width = cache["actuators"].shape[-1]
    loaded = torch.load(entry, map_location="cpu", weights_only=False)
    cache["actuators"] = loaded["actuators_global"][:, :width]


def _rank_device() -> torch.device:
    """This rank's GPU, or the CPU when there is none.

    With one GPU visible per rank (Frontier) it is ``cuda:0``; with all of the node's
    GPUs visible (StellarAI, ``SRUN_GPU_BIND=none``) it is ``cuda:LOCAL_RANK``, as in
    ``train_codec._DDPState``.
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")
    index = int(os.environ.get("LOCAL_RANK", "0")) if torch.cuda.device_count() > 1 else 0
    torch.cuda.set_device(index)
    return torch.device("cuda", index)


def slowts_denorm(codecs) -> Dict:
    """Map decoded slow time series back to physical units, from the codecs' own stats.

    Each slow-TS codec standardises its input per channel with the global
    ``channel_mean`` / ``channel_std`` of its configuration, after
    ``log10(x + 1)`` for the Thomson signals (:data:`config.SLOWTS_PREPROCESS_METHOD`).
    The returned callables invert that on arrays of shape (F, C_pad, T) and drop the
    padded channels. Only the modalities of :data:`_SLOWTS_UNITS` (n_e, T_e, T_i) are
    converted; the others stay normalised.

    Parameters
    ----------
    codecs : dict
        ``{name: (codec, codec_cfg, family)}`` as from ``load_frozen_codecs``.

    Returns
    -------
    denorm : dict of str to callable
    """
    from .config import SLOWTS_PREPROCESS_METHOD

    denorm = {}
    for name, (_codec, codec_cfg, family) in codecs.items():
        method = SLOWTS_PREPROCESS_METHOD.get(name)
        mean = getattr(codec_cfg, "channel_mean", None)
        std = getattr(codec_cfg, "channel_std", None)
        if family != "slowts" or method not in ("standardize", "log_standardize"):
            continue
        # Only n_e, T_e and T_i, whose figure rows carry units (_SLOWTS_UNITS). cer_rot
        # has broken stats on some channels (inverts to ~1e27); mse has no unit row.
        if name not in _SLOWTS_UNITS:
            continue
        if mean is None or std is None:
            continue
        mean = np.asarray(mean, dtype=np.float64)
        std = np.clip(np.asarray(std, dtype=np.float64), 1e-3, None)

        def invert(x, mean=mean, std=std, log=method == "log_standardize"):
            x = np.asarray(x, dtype=np.float64)[:, : mean.size]
            shape = [1] * x.ndim
            shape[1] = -1
            y = x * std.reshape(shape) + mean.reshape(shape)
            return (np.power(10.0, y) - 1.0) if log else y

        denorm[name] = invert
    return denorm


@torch.no_grad()
def rollout_shot(
    model: MaskGITDynamics,
    cfg: DynamicsConfig,
    cache: Dict,
    K0: int,
    temperature: float,
    generator: torch.Generator,
    device,
    actuator_mode: str = "real",
    cache_dir=None,
    shot=None,
    guidance: float = 1.0,
    decode: Optional[Dict] = None,
    start: int = 0,
) -> Rollout:
    """Seed ``K0`` real frames of a shot and roll the model out.

    The rollout draws the same random numbers whatever ``actuator_mode`` is, so
    two runs with the same seed differ only through the actuators.

    Parameters
    ----------
    model : MaskGITDynamics
        Trained model.
    cfg : DynamicsConfig
        Its configuration.
    cache : dict
        The shot's cache entry. With ``IGNITE_ACT_GLOBAL`` set its actuators
        are replaced in place by the side-cache ones.
    K0 : int
        Seed frames.
    temperature : float
        Sampling temperature.
    generator : torch.Generator
        Source of randomness.
    device : torch.device
        Device of the model.
    actuator_mode : str, default="real"
        Counterfactual, see :func:`apply_actuator_mode`.
    cache_dir : path-like, default=None
        Cache directory, for donor counterfactuals.
    shot : str, default=None
        Shot id; required when ``IGNITE_ACT_GLOBAL`` is set.
    guidance : float, default=1.0
        Classifier-free guidance weight (:meth:`MaskGITDynamics.generate_frame`).
    decode : dict, default=None
        Further rollout options: ``gate``, ``consensus_n``, ``consensus_q``,
        ``consensus_mode`` (:meth:`MaskGITDynamics.rollout`); ``infer`` (modalities
        generated while all others keep their measured codes) and ``seed_from``
        (``{modality: shot}``, seed frames taken from another shot).
    start : int, default=0
        First frame of the window in the shot: the seed is frames
        ``start..start+K0``. Frame indices of ``actuator_mode`` count from it.

    Returns
    -------
    rollout : Rollout
        Codes over the whole window on the CPU, ``K0``, the window length
        ``F = min(cfg.max_frames, n_frames - start)`` and, where possible, the
        expected frames.
    """
    n_frames = min(cfg.max_frames, int(cache["n_frames"]) - int(start))
    if n_frames <= K0:
        raise RuntimeError(
            f"shot has F={n_frames} frames <= K0={K0}; nothing to predict"
        )
    names = [m.name for m in cfg.modalities]
    codes = {name: cache["codes"][name][start : start + n_frames] for name in names}
    seed_codes = {
        name: codes[name][:K0].long().unsqueeze(0).to(device) for name in names
    }
    _load_global_actuators(cache, shot)
    actuators = apply_actuator_mode(
        cache["actuators"][start : start + n_frames].float(),
        actuator_mode,
        K0,
        cache_dir=cache_dir,
        F=n_frames,
    )
    decode = dict(decode or {})
    for name, donor in (decode.pop("seed_from", None) or {}).items():
        # Seed one modality from another shot (it did not record in this one); the
        # measurement kept for scoring is still this shot's.
        if name not in names:
            raise ValueError(f"--seed_from: unknown modality {name}")
        if cache_dir is None:
            raise ValueError("--seed_from needs cache_dir")
        donor_codes = load_shot_cache(Path(cache_dir), str(donor))["codes"][name]
        if donor_codes.shape[0] < start + K0:
            raise ValueError(f"--seed_from: {donor} has fewer than {start + K0} frames")
        seed_codes[name] = (
            donor_codes[start : start + K0].long().unsqueeze(0).to(device)
        )
    infer = [n for n in decode.pop("infer", None) or () if n]
    hide = [n for n in decode.pop("hide", None) or () if n]
    # A diagnostic that did not record in the initial state (the same codes in every
    # seed frame: the codec's code of a zero-filled window) is [MASK] there, its
    # "not recorded" codes are never sampled, and it is predicted with all others,
    # as --hide does. No information from any other shot is used. On by default from
    # the CLI (IGNITE_MASK_ABSENT, read at call time), off for library calls.
    mask_absent = decode.pop("mask_absent", None)
    if mask_absent is None:
        mask_absent = os.environ.get("IGNITE_MASK_ABSENT", "0") == "1"
    # Forbid the masked diagnostic's "not recorded" codes (it must then be predicted as
    # recording). Off by default: the masked diagnostic is predicted freely, which may
    # again be "not recorded" (IGNITE_MASK_FORBID=1 or --mask_forbid 1 turns it on).
    mask_forbid = decode.pop("mask_forbid", None)
    if mask_forbid is None:
        mask_forbid = os.environ.get("IGNITE_MASK_FORBID", "0") == "1"
    absent_seed = {}
    if mask_absent and not infer and int(decode.get("consensus_n", 1)) <= 1:
        for name in names:
            if name in hide:
                continue
            seed = codes[name][:K0]
            if bool((seed == seed[:1]).all()):
                absent_seed[name] = seed[0].long().to(device)
        if absent_seed:
            print(f"[eval] not recorded in the initial state, masked and predicted: "
                  f"{sorted(absent_seed)}")
            hide = hide + sorted(absent_seed)
    if hide:
        # Missing diagnostics in a free rollout: the hidden modalities are [MASK] in
        # the seed; the first predicted frame generates them from the other
        # modalities' seed and then stands in for their seed, and every modality,
        # hidden or not, is generated autoregressively (nothing held at its
        # measurement).
        if infer:
            raise ValueError("--hide and --infer exclude each other")
        unknown = sorted(set(hide) - set(names))
        if unknown:
            raise ValueError(f"--hide: unknown modalities {unknown}")
        decode["blind"] = list(hide)
        decode["bootstrap_blind"] = True
    if decode.pop("infer_blind", False):
        if not infer:
            raise ValueError("--infer_blind needs --infer")
        # The inferred modalities are also hidden from the history ([MASK]).
        decode["blind"] = list(infer)
    if decode.pop("infer_bootstrap", False):
        if "blind" not in decode:
            raise ValueError("--infer_bootstrap needs --infer_blind")
        decode["bootstrap_blind"] = True
    absent_shot = decode.pop("forbid_absent", None)
    forbid_mods = [n for n in decode.pop("forbid_mods", None) or () if n]
    if forbid_mods and not absent_shot:
        raise ValueError("--forbid_modalities needs --forbid_absent")
    if absent_shot:
        # Never sample an inferred modality's "not recorded" frame: the constant codes
        # it has in a shot where it did not record.
        # In a plain rollout (complete initial state, nothing removed) the modalities
        # are given by --forbid_modalities.
        if not (infer or hide or forbid_mods) or cache_dir is None:
            raise ValueError("--forbid_absent needs --infer, --hide or "
                             "--forbid_modalities, and cache_dir")
        absent = load_shot_cache(Path(cache_dir), str(absent_shot))["codes"]
        decode["forbid"] = {}
        for name in infer or hide or forbid_mods:
            frames = absent[name].long()
            if not bool((frames == frames[:1]).all()):
                raise ValueError(
                    f"--forbid_absent: {name} records in {absent_shot} (codes change)"
                )
            decode["forbid"][name] = frames[0].to(device)
    for name, frame in absent_seed.items() if mask_forbid else ():
        decode.setdefault("forbid", {}).setdefault(name, frame)
    if infer:
        # Inference of missing modalities: every other modality keeps its measured
        # codes in every predicted frame; only ``infer`` is generated.
        unknown = sorted(set(infer) - set(names))
        if unknown:
            raise ValueError(f"--infer: unknown modalities {unknown}")
        decode["known_codes"] = {
            name: codes[name].long().unsqueeze(0).to(device)
            for name in names
            if name not in infer
        }
    with_mean = model.can_return_mean and int(decode.get("consensus_n", 1)) <= 1
    trajectory = model.rollout(
        seed_codes,
        actuators.unsqueeze(0).to(device),
        n_predict=n_frames - K0,
        temperature=temperature,
        generator=generator,
        guidance=guidance,
        return_mean=with_mean,
        **decode,
    )
    mean = None
    if with_mean:
        trajectory, mean = trajectory
        mean = {name: mean[name][0, :n_frames].float().cpu() for name in names}
    return Rollout(
        gt_codes={name: codes[name][:n_frames].long().cpu() for name in names},
        pred_codes={
            name: trajectory[name][0, :n_frames].long().cpu() for name in names
        },
        k0=K0,
        n_frames=n_frames,
        pred_mean=mean,
    )


# Core profiles of the rollout skill (analysis/checkpoint_scorecard.py and the
# trainer's --skill_every): T_e, n_e, T_i, v_tor.
PROFILE_SIGNALS = ("ts_core_temp", "ts_core_density", "cer_ti", "cer_rot")


def profile_converters(codecs, names=PROFILE_SIGNALS) -> Dict:
    """Per-signal converters of decoded slow time series to physical units.

    Unlike :func:`slowts_denorm` this covers cer_rot too: channels whose stats are
    broken (``|mean|`` or ``std`` above 1e4, or ``std`` at most 1e-3) become NaN.

    Parameters
    ----------
    codecs : dict
        ``{name: (codec, codec_cfg, family)}`` as from ``load_frozen_codecs``.
    names : sequence of str
        Slow-TS modalities to convert.

    Returns
    -------
    convert : dict of str to callable
        Each maps (F, C_pad, T) to (F, C, T) float64.
    """
    from .config import SLOWTS_PREPROCESS_METHOD

    convert = {}
    for name in names:
        cfg = codecs[name][1]
        mean = np.asarray(cfg.channel_mean, dtype=np.float64)
        raw_std = np.asarray(cfg.channel_std, dtype=np.float64)
        std = np.clip(raw_std, 1e-3, None)
        broken = (np.abs(mean) > 1e4) | (raw_std > 1e4) | (raw_std <= 1e-3)
        log = SLOWTS_PREPROCESS_METHOD[name] == "log_standardize"

        def invert(x, mean=mean, std=std, broken=broken, log=log):
            x = np.asarray(x, dtype=np.float64)[:, : mean.size]
            y = x * std[None, :, None] + mean[None, :, None]
            y = np.power(10.0, y) - 1.0 if log else y
            y[:, broken] = np.nan
            return y

        convert[name] = invert
    return convert


def core_channels(gt: np.ndarray, k0: int, n_core: int) -> list:
    """The ``n_core`` channels of (F, C) ``gt`` with the largest mean |value|.

    The mean is over the frames from ``k0`` on.
    """
    with np.errstate(invalid="ignore"):
        level = np.nanmean(np.abs(gt[k0:]), axis=0)
    finite = np.isfinite(level)
    order = [
        int(c) for c in np.argsort(np.where(finite, level, -np.inf))[::-1] if finite[c]
    ]
    return order[: max(1, n_core)]


def profile_skill(
    gt_codes, pred_codes, k0: int, n_frames: int, codecs, convert, device
) -> float:
    """RMSE skill against persistence of one rollout's core profiles.

    Each signal of :data:`PROFILE_SIGNALS` is decoded, converted to physical units,
    averaged over time within the frame and over its 3 strongest channels, and
    scored over frames ``k0..n_frames``: ``1 - rmse(prediction) / rmse(persistence)``,
    persistence holding the last seed frame. Unrecorded (flat) signals are skipped.

    Parameters
    ----------
    gt_codes, pred_codes : dict of str to torch.Tensor
        Real and predicted frames of each signal: codes ``(F, n_tok)`` or expected
        coordinates ``(F, n_tok, n_levels)``.
    k0 : int
        Seed frames.
    n_frames : int
        Frames of the rollout window.
    codecs : dict
        The signals' codecs, from ``load_frozen_codecs``.
    convert : dict
        From :func:`profile_converters`.
    device : torch.device
        Where to decode.

    Returns
    -------
    skill : float
        Mean over the scored signals; NaN when fewer than 2 could be scored.
    """
    per_signal = []
    for name in PROFILE_SIGNALS:
        gt, predicted = (
            np.nanmean(
                convert[name](
                    decode_flat_chunked(codecs[name][0], codes[name][:n_frames], device)
                ),
                axis=-1,
            )
            for codes in (gt_codes, pred_codes)
        )
        channels = core_channels(gt, k0, 3)
        with np.errstate(invalid="ignore"):
            real = np.nanmean(gt[:, channels], 1)
            model = np.nanmean(predicted[:, channels], 1)
        truth = real[k0:n_frames]
        if np.isfinite(truth).sum() < 10 or np.nanstd(truth) < 1e-6 * max(
            1.0, abs(np.nanmean(truth))
        ):
            continue
        rmse = np.sqrt(np.nanmean((model[k0:n_frames] - truth) ** 2))
        rmse_persistence = np.sqrt(np.nanmean((real[k0 - 1] - truth) ** 2))
        if np.isfinite(rmse) and rmse_persistence > 0:
            per_signal.append(1.0 - rmse / rmse_persistence)
    return float(np.mean(per_signal)) if len(per_signal) >= 2 else float("nan")


def task_state(name: str, family: str, decoded: np.ndarray, convert=None) -> np.ndarray:
    """Reduce one modality's decoded frames to the state the task scores.

    Slow time series are converted to physical units (where ``convert`` has the
    signal) and averaged over time within the frame, giving (F, C). Spectrograms
    are averaged over time within the frame, keeping the frequency profile
    (F, C, freq): the band power over time, not the turbulent texture. Filterscope
    envelopes and camera frames are averaged over everything but the channel,
    giving (F, C).
    """
    x = np.asarray(decoded, dtype=np.float64)
    if family == "slowts" and convert and name in convert:
        x = convert[name](x)
    if family == "spectro":
        return x.mean(axis=-1)
    with np.errstate(invalid="ignore"):
        return np.nanmean(x.reshape(x.shape[0], x.shape[1], -1), axis=-1)


def task_skill(
    gt_codes, pred_codes, k0: int, n_end: int, codecs, convert, device
) -> Dict[str, float]:
    """RMSE skill against persistence of every recorded modality of one rollout.

    Each modality is decoded, reduced by :func:`task_state` and scored over frames
    ``k0..n_end`` (the live part of the window): ``1 - rmse(prediction) /
    rmse(persistence)``, persistence holding the last seed frame. A modality whose
    codes do not change after the seed was not recorded and is skipped, as is one
    whose state persistence already matches exactly.

    Parameters
    ----------
    gt_codes, pred_codes : dict of str to torch.Tensor
        Real and predicted codes ``(F, n_tok)`` of each modality.
    k0 : int
        Seed frames.
    n_end : int
        End of the scored frames (exclusive), at most the window length.
    codecs : dict
        ``{name: (codec, cfg, family)}`` from ``load_frozen_codecs``; modalities
        without a codec are skipped.
    convert : dict
        Slow-TS converters to physical units, from :func:`profile_converters`.
    device : torch.device
        Where to decode.

    Returns
    -------
    skill : dict of str to float
        Per scored modality.
    """
    scores = {}
    if n_end - k0 < 2:
        return scores
    for name, (codec, _cfg, family) in codecs.items():
        truth_codes = gt_codes[name][:n_end]
        if bool((truth_codes[k0:] == truth_codes[k0 - 1 : k0]).all()):
            continue
        gt, predicted = (
            task_state(
                name, family, decode_flat_chunked(codec, c[name][:n_end], device), convert
            )
            for c in (gt_codes, pred_codes)
        )
        truth = gt[k0:n_end]
        with np.errstate(invalid="ignore"):
            rmse = np.sqrt(np.nanmean((predicted[k0:n_end] - truth) ** 2))
            rmse_persistence = np.sqrt(np.nanmean((gt[k0 - 1 : k0] - truth) ** 2))
        if np.isfinite(rmse) and rmse_persistence > 1e-12:
            scores[name] = float(1.0 - rmse / rmse_persistence)
    return scores


def _nrmse(pred: np.ndarray, gt: np.ndarray) -> float:
    """Return ``rmse(pred - gt) / std(gt)`` over finite entries, in float64.

    float64 is required: physical densities reach ~1e19 and overflow float32.
    """
    finite = np.isfinite(pred) & np.isfinite(gt)
    if not finite.any():
        return float("nan")
    predicted = pred[finite].astype(np.float64)
    truth = gt[finite].astype(np.float64)
    error = predicted - truth
    return float(np.sqrt(np.mean(error * error))) / (float(np.std(truth)) + 1e-12)


@torch.no_grad()
def raw_to_output_space(fam: str, raw, denorm_fn=None, mask=None) -> np.ndarray:
    """Map a codec's raw input window into the space of the denormalised decode.

    Video inputs are already camera counts and pass through; spectrogram and
    slow time-series inputs are standardised and go through ``denorm_fn``.

    Parameters
    ----------
    fam : str
        Codec family.
    raw : array-like
        Raw input window(s).
    denorm_fn : callable, default=None
        Inverse normalisation of the modality.
    mask : array-like of bool, default=None
        True where valid, of the shape of ``raw`` or of its leading axes. Invalid
        samples become NaN, which plotting and scoring treat as absent.

    Returns
    -------
    out : ndarray of float32
        The signal in output space.
    """
    signal = np.asarray(raw, dtype=np.float32)
    if mask is not None:
        valid = np.asarray(mask)
        if valid.shape == signal.shape:
            signal = np.where(valid.astype(bool), signal, np.nan)
        elif valid.shape == signal.shape[: valid.ndim]:
            trailing = (1,) * (signal.ndim - valid.ndim)
            signal = np.where(
                valid.astype(bool).reshape(valid.shape + trailing), signal, np.nan
            )
    if fam == "video":
        return signal
    return np.asarray(denorm_fn(signal), dtype=np.float32) if denorm_fn else signal


@torch.no_grad()
def load_raw_gt(
    shot: str,
    codecs: Dict,
    F: int,
    data_dir,
    t0_start: float = 1.0,
    denorm: Dict = None,
    log=print,
) -> Dict[str, np.ndarray]:
    """Return the measured input signal of each modality, ``{name: (F, ...)}``.

    These are the codec input windows, never encoded, so scoring against them
    includes the codec's reconstruction error. ``t0_start`` must match the
    cache manifest, or the frames shift against the codes. A modality that
    fails to load is logged and skipped.
    """
    measured = {}
    for name, (_codec, codec_cfg, family) in codecs.items():
        try:
            dataset = td._single_shot_dataset(
                name, family, codec_cfg, shot, data_dir, t0_start=t0_start
            )
            if len(dataset) < F:
                log(
                    f"[eval] raw GT {name}@{shot}: only {len(dataset)} frames < {F}; "
                    f"skipping"
                )
                continue
            frames, masks = [], []
            for t in range(F):
                item = dataset[t]
                frames.append(np.asarray(td._window(item), dtype=np.float32))
                if (
                    isinstance(item, tuple)
                    and len(item) > 1
                    and family in ("video", "slowts")
                ):
                    masks.append(np.asarray(item[1]))
            measured[name] = raw_to_output_space(
                family,
                np.stack(frames, 0),
                denorm_fn=(denorm or {}).get(name),
                mask=np.stack(masks, 0) if masks else None,
            )
        except Exception as e:
            log(f"[eval] raw GT {name}@{shot} unavailable ({type(e).__name__}: {e})")
    return measured


def bandpower_levels(edges: np.ndarray) -> np.ndarray:
    """Turn (N_LEV-1, D) interior quantile edges into (N_LEV, D) level values.

    Interior levels take the bin midpoint; the two open outer bins extrapolate by
    half the adjacent bin width, which keeps the values monotone in the level.
    """
    q = np.asarray(edges, dtype=np.float64)
    midpoints = 0.5 * (q[:-1] + q[1:])
    lowest = q[0] - 0.5 * (q[1] - q[0])
    highest = q[-1] + 0.5 * (q[-1] - q[-2])
    return np.concatenate([lowest[None, :], midpoints, highest[None, :]], axis=0)


def decode_bandpower(
    spec: Tuple[np.ndarray, int, int], codes: torch.Tensor
) -> np.ndarray:
    """Decode band-power levels to log power of shape (F, C, n_band, 1).

    The trailing axis is the intra-frame time axis, so a frame becomes one column
    of the stitched spectrogram.
    """
    level_values, n_channels, n_bands = spec
    levels = np.asarray(codes.detach().cpu(), dtype=np.int64)
    n_columns = levels.shape[1]
    values = level_values[
        np.clip(levels, 0, level_values.shape[0] - 1), np.arange(n_columns)[None, :]
    ]
    return values.reshape(levels.shape[0], n_channels, n_bands)[..., None].astype(
        np.float32
    )


def _scored(gt, pred, persistence, K0, gt_source, family, space) -> Dict:
    """Decoded arrays of one modality with their scores over the predicted frames."""
    nrmse = _nrmse(pred[K0:], gt[K0:])
    nrmse_persistence = _nrmse(persistence[K0:], gt[K0:])
    return {
        "gt": gt,
        "pred": pred,
        "gt_source": gt_source,
        "nrmse": nrmse,
        "nrmse_persistence": nrmse_persistence,
        "nrmse_skill": (
            1.0 - nrmse / nrmse_persistence if nrmse_persistence > 0 else float("nan")
        ),
        # Read before skill: skill alone prefers a flat prediction.
        "activity": _temporal_activity(pred[K0:], gt[K0:]),
        "family": family,
        "space": space,
    }


def _match_raw_width(raw: np.ndarray, gt: np.ndarray):
    """NaN-pad or truncate the channel axis of a raw window to the decode's width.

    Returns None if the shapes differ in any other way.
    """
    measured = np.asarray(raw, dtype=np.float32)
    if (
        measured.ndim != gt.ndim
        or measured.shape[0] != gt.shape[0]
        or measured.shape[2:] != gt.shape[2:]
    ):
        return None
    missing = gt.shape[1] - measured.shape[1]
    if missing > 0:
        pad_shape = (measured.shape[0], missing) + measured.shape[2:]
        return np.concatenate(
            [measured, np.full(pad_shape, np.nan, dtype=np.float32)], axis=1
        )
    if missing < 0:
        return measured[:, : gt.shape[1]]
    return None


def decode_all(
    codecs: Dict,
    gt_codes: Dict,
    pred_codes: Dict,
    K0: int,
    F: int,
    device,
    denorm: Dict = None,
    raw_gt: Dict = None,
    bandpower: Dict = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Decode and score the ground truth, prediction and persistence of each modality.

    Parameters
    ----------
    codecs : dict
        ``{name: (codec, cfg, family)}``; modalities without a codec are skipped.
    gt_codes, pred_codes : dict of str to torch.LongTensor of shape (F, n_tok)
        Codes over the whole window, seed frames included.
    K0 : int
        First predicted frame; scores cover ``[K0, F)``.
    F : int
        Window length.
    device : torch.device
        Device of the codecs.
    denorm : dict of str to callable, default=None
        Maps a decode to physical units. Per-window normalisations are inverted
        with the ground-truth frame's statistics.
    raw_gt : dict of str to ndarray, default=None
        Measured signals in output space (see :func:`load_raw_gt`), used as the
        reference when their shape matches the decode.
    bandpower : dict, default=None
        ``{name: (levels, C, n_band)}`` for band-power modalities, which have no
        learned decoder.

    Returns
    -------
    decoded : dict of str to dict
        Per modality: ``gt``, ``pred`` (leading axis F), ``gt_source``,
        ``nrmse``, ``nrmse_persistence``, ``nrmse_skill``, ``activity``,
        ``family`` and ``space``.
    """
    raw_gt = dict(raw_gt or {})
    decoded = {}
    for name in EVAL_MODALITIES:
        last_seed = slice(K0 - 1, K0)
        spec = (bandpower or {}).get(name)
        if spec is not None:
            gt = decode_bandpower(spec, gt_codes[name][:F])
            pred = decode_bandpower(spec, pred_codes[name][:F])
            persistence = np.repeat(
                decode_bandpower(spec, gt_codes[name][last_seed]), F, axis=0
            )
            decoded[name] = _scored(
                gt, pred, persistence, K0, "decoded", "spectro", "band power"
            )
            continue
        if name not in codecs:
            continue
        codec, _codec_cfg, family = codecs[name]
        gt = decode_flat_chunked(codec, gt_codes[name][:F], device)
        pred = decode_flat_chunked(codec, pred_codes[name][:F], device)
        # Persistence: decoding is deterministic, so tile the decoded last seed frame.
        persistence = np.repeat(
            decode_flat_chunked(codec, gt_codes[name][last_seed], device), F, axis=0
        )
        space = "normalized"
        if denorm and name in denorm:
            gt, pred = denorm[name](gt), denorm[name](pred)
            persistence = denorm[name](persistence)
            space = "physical"
        gt_source = "decoded"
        # Measured camera frames are raw counts; without a denorm the prediction stays
        # in the codec's normalised space, so the two would not share a colour scale.
        comparable = family != "video" or space == "physical"
        if name in raw_gt and comparable:
            # Compare after denorm, which drops the padded channels of slow-TS.
            if raw_gt[name].shape != gt.shape:
                matched = _match_raw_width(raw_gt[name], gt)
                if matched is not None:
                    raw_gt[name] = matched
            if raw_gt[name].shape == gt.shape:
                gt, gt_source = np.asarray(raw_gt[name], dtype=np.float32), "measured"
            else:
                print(
                    f"[eval] raw GT {name}: shape {raw_gt[name].shape} != decoded "
                    f"{gt.shape} — falling back to the decoded round-trip",
                    flush=True,
                )
        decoded[name] = _scored(gt, pred, persistence, K0, gt_source, family, space)
    return decoded


def _temporal_activity(pred: np.ndarray, gt: np.ndarray) -> float:
    """Return the ratio of mean temporal standard deviations, prediction over truth.

    0 means a prediction frozen in time and 1 the right temporal amplitude. The
    standard deviation is taken along time per element (axis 0), not over the
    flattened array, which static cross-channel structure would dominate.
    """
    predicted = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(gt, dtype=np.float64)
    if predicted.shape != truth.shape or predicted.shape[0] < 2:
        return float("nan")
    predicted_std = np.nanstd(predicted, axis=0)
    truth_std = np.nanstd(truth, axis=0)
    usable = np.isfinite(predicted_std) & np.isfinite(truth_std) & (truth_std > 0)
    if not usable.any():
        return float("nan")
    return float(np.nanmean(predicted_std[usable]) / np.nanmean(truth_std[usable]))


def self_check(decoded: Dict[str, Dict[str, np.ndarray]]) -> None:
    """Assert that decoded spectrogram and slow-TS ground truth is finite and varies.

    A constant or all-missing measured signal is an absent diagnostic and is only
    reported; for a decoded round-trip it indicates a broken decode path.

    Raises
    ------
    AssertionError
        If a decoded ground truth is constant or has no finite value, or no
        spectrogram or slow-TS modality was decoded.
    """
    n_checked, absent = 0, []
    for name, entry in decoded.items():
        if entry["family"] not in ("spectro", "slowts"):
            continue
        measured = entry.get("gt_source") == "measured"
        finite = entry["gt"][np.isfinite(entry["gt"])]
        if not finite.size:
            if measured:
                absent.append(f"{name}(all-missing)")
                continue
            raise AssertionError(
                f"[self-check] {name}: DECODED GT has NO finite values"
            )
        if float(finite.std()) > 0.0:
            n_checked += 1
        elif measured:
            absent.append(name)
        else:
            raise AssertionError(
                f"[self-check] {name}: DECODED GT is constant (std=0) — broken "
                f"decode path"
            )
    assert n_checked > 0 or absent, (
        "[self-check] no spectro/slowts modalities decoded — nothing to validate"
    )
    message = (
        f"[self-check] OK: {n_checked} spectro/slowts GT are finite + non-constant"
    )
    if absent:
        message += (
            f"; {len(absent)} constant because the diagnostic is ABSENT in "
            f"this shot ({', '.join(sorted(absent))})"
        )
    print(message, flush=True)


# A ground truth whose temporal std is below this fraction of its scale is static.
_STATIC_EPS = 1e-4


def _temporal_variation(gt: np.ndarray) -> Tuple[float, float]:
    """Return the mean temporal std and the max absolute value of finite entries."""
    values = np.asarray(gt, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return 0.0, 0.0
    scale = float(np.max(np.abs(values[finite])))
    with np.errstate(invalid="ignore"):
        temporal_std = float(
            np.nanmean(np.nanstd(np.where(finite, values, np.nan), axis=0))
        )
    if not np.isfinite(temporal_std):
        temporal_std = 0.0
    return temporal_std, scale


def _is_static(gt: np.ndarray) -> bool:
    """Return whether a ground truth is constant in time, relative to its scale."""
    temporal_std, scale = _temporal_variation(gt)
    return temporal_std <= _STATIC_EPS * max(scale, 1e-12)


# Okabe-Ito palette, shared with the learning-curve figure (epoch_curve.py).
OKABE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "black": "#000000",
}
_GT_COLOR, _PRED_COLOR = "#1A1A1A", OKABE["vermillion"]
_VIDEO_COLOR = {"tangtv_lower": OKABE["blue"], "tangtv_upper": OKABE["vermillion"]}
LW_PRIMARY, LW_SECONDARY = 1.6, 1.2
AX_LW, GRID_LW, GRID_ALPHA = 0.8, 0.5, 0.25
DASH_SECONDARY = (0, (3.2, 1.5))

_DISPLAY_NAMES = {
    "ece": "ECE",
    "mhr": "MHR",
    "bes": "BES",
    "co2": "CO2",
    "mirnov": "Mirnov",
    "tangtv_lower": "Lower\ndivertor",
    "tangtv_upper": "Upper\ndivertor",
}

# Slow-TS display: (scale, ylabel, region tag). The H5 stores eV and m^-3.
_SLOWTS_UNITS = {
    "ts_core_density": (1.0, r"$n_e$ (m$^{-3}$)", "core"),
    "ts_core_temp": (1e-3, r"$T_e$ (keV)", "core"),
    "ts_tangential_density": (1.0, r"$n_e$ (m$^{-3}$)", "tang."),
    "ts_tangential_temp": (1e-3, r"$T_e$ (keV)", "tang."),
    "cer_ti": (1e-3, r"$T_i$ (keV)", ""),
}

# One-line labels of the slow time series in normalised units.
_SLOWTS_SHORT = {
    "ts_core_density": r"Core $n_e$",
    "ts_core_temp": r"Core $T_e$",
    "ts_tangential_density": r"Tang. $n_e$",
    "ts_tangential_temp": r"Tang. $T_e$",
    "cer_ti": r"$T_i$",
    "cer_rot": r"$v_\phi$",
    "mse": "MSE",
}

# Row heights in inches at the 12 pt design point; scaled mildly with the font.
_ROW_HEIGHTS = {"slow": 0.58, "spec": 1.05, "vid": 0.90, "vrmse": 0.60}
_SECTION_GAP_IN = 0.46
_ROW_SPACING = 0.28
_MAX_STITCHED_COLUMNS = 4096


_PLASMA_END_CACHE: Dict[str, Optional[Dict[str, float]]] = {}


def _plasma_end(shot) -> Optional[float]:
    """Plasma end time [s] of a shot, or None if unknown.

    Read from ``IGNITE_PLASMA_END_JSON`` (``{shot: seconds}``) if it is set, for
    example the ``eval_runs/tend_all.json`` the trainer masks post-shot frames with.
    Off by default: that table covers only some shots, so figures would be shaded
    inconsistently.
    """
    path = os.environ.get("IGNITE_PLASMA_END_JSON", "").strip()
    if not path:
        return None
    if path not in _PLASMA_END_CACHE:
        try:
            _PLASMA_END_CACHE[path] = {
                str(k): float(v) for k, v in json.loads(Path(path).read_text()).items()
            }
        except (OSError, ValueError):
            _PLASMA_END_CACHE[path] = None
    table = _PLASMA_END_CACHE[path]
    return None if table is None else table.get(str(shot))


def _display_name(name: str) -> str:
    return _DISPLAY_NAMES.get(name, name)


def _best_channel(gt: np.ndarray, reduce_axes: Tuple[int, ...]) -> int:
    """Return the channel whose frame means vary most over time."""
    values = np.where(np.isfinite(gt), gt, np.nan)
    with np.errstate(invalid="ignore"):
        variance = np.nanvar(np.nanmean(values, axis=reduce_axes), axis=0)
    variance = np.where(np.isfinite(variance), variance, -np.inf)
    return int(np.argmax(variance)) if np.isfinite(variance).any() else 0


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r over the elements finite in both arrays."""
    usable = np.isfinite(a) & np.isfinite(b)
    if usable.sum() < 2:
        return float("nan")
    x = a[usable].astype(np.float64).ravel()
    y = b[usable].astype(np.float64).ravel()
    std_x, std_y = x.std(), y.std()
    if std_x <= 0 or std_y <= 0:
        return float("nan")
    return float(((x - x.mean()) * (y - y.mean())).mean() / (std_x * std_y))


def _stitch(spectrogram: np.ndarray, channel: int) -> np.ndarray:
    """Concatenate one channel's frames along time, keeping at most ~4096 columns."""
    image = np.concatenate(list(spectrogram[:, channel]), axis=-1)
    return image[:, :: max(1, image.shape[-1] // _MAX_STITCHED_COLUMNS)]


def _robust_limits(image: np.ndarray, upper_percentile: float):
    finite = image[np.isfinite(image)]
    if not finite.size:
        return 0.0, 1.0
    return float(np.percentile(finite, 1.0)), float(
        np.percentile(finite, upper_percentile)
    )


class _Figure:
    """Canvas, row layout and shared time axis of one comparison figure.

    Rows come in a fixed order, so panel k shows the same modality in every
    figure: slow-TS traces, spectrograms, cameras, then the optional video nRMSE.
    A gap row separates sections.
    """

    def __init__(
        self, decoded, K0, F, t_origin, options: FigureOptions, plasma_end=None
    ):
        import string

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        from .config import STFT_FS, STFT_N_FFT

        self.plt = plt
        self.options = options
        self.decoded = decoded
        self.K0, self.F = K0, F
        self.t_start, self.t_end = t_origin, t_origin + F * FRAME_S
        self.t_rollout = t_origin + K0 * FRAME_S
        self.times = t_origin + np.arange(F) * FRAME_S
        self.mid_frame = K0 + (F - K0) // 2
        self.t_origin = t_origin
        self.plasma_end = plasma_end
        if plasma_end is not None:
            # Show the camera frame in the middle of the predicted live plasma.
            last_live = int((plasma_end - t_origin) / FRAME_S)
            if K0 < last_live < F:
                self.mid_frame = K0 + (last_live - K0) // 2
        self.khz_per_bin = (STFT_FS / STFT_N_FFT) / 1e3
        self.pt = options.font_pt
        self._set_paper_style()
        self.spectro = [n for n in options.spectro if n in decoded]
        self.video = [n for n in VIDEO if n in decoded]
        self.slowts = [n for n in options.slowts if n in decoded]
        self.static = {name: _is_static(entry["gt"]) for name, entry in decoded.items()}
        self.unit = (
            "phys"
            if any(e.get("space") == "physical" for e in decoded.values())
            else "norm"
        )
        self.rows = self._layout_rows()
        scale = 0.55 + 0.45 * (self.pt / 12.0)
        heights = [
            _SECTION_GAP_IN if kind == "gap" else _ROW_HEIGHTS[kind] * scale
            for kind, _name in self.rows
        ]
        margin_left = 0.10 + 0.075 * self.pt
        margin_right = 0.10 + 0.030 * self.pt
        margin_bottom, margin_top = 0.06 + 0.046 * self.pt, 0.06 + 0.015 * self.pt
        n_rows = max(len(self.rows), 1)
        height = (
            sum(heights) * (1.0 + _ROW_SPACING * (n_rows - 1) / n_rows)
            + margin_bottom
            + margin_top
        )
        width = options.width_in
        self.fig = plt.figure(figsize=(width, height))
        self.grid = GridSpec(
            len(self.rows),
            1,
            figure=self.fig,
            height_ratios=heights,
            hspace=_ROW_SPACING,
            left=margin_left / width,
            right=1.0 - margin_right / width,
            top=1.0 - margin_top / height,
            bottom=margin_bottom / height,
        )
        self.letters = iter(string.ascii_lowercase)
        self.legend_host = None

    def _set_paper_style(self):
        pt = self.pt
        self.plt.rcParams.update(
            {
                "font.family": "serif",
                "font.serif": FIG_SERIF,
                "mathtext.fontset": "stix",
                "font.size": pt,
                "axes.labelsize": pt,
                "axes.titlesize": pt,
                "xtick.labelsize": pt - 1,
                "ytick.labelsize": pt - 1,
                "legend.fontsize": pt - 1,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.linewidth": AX_LW,
                "xtick.major.width": AX_LW,
                "ytick.major.width": AX_LW,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            }
        )

    def _layout_rows(self):
        rows = (
            [("slow", n) for n in self.slowts]
            + [("spec", n) for n in self.spectro]
            + [("vid", n) for n in self.video]
        )
        if self.video and self.options.show_video_nrmse and not self.options.gt_only:
            rows.append(("vrmse", None))
        section = {"vrmse": "vid"}
        with_gaps = []
        for kind, name in rows:
            if with_gaps and section.get(
                with_gaps[-1][0], with_gaps[-1][0]
            ) != section.get(kind, kind):
                with_gaps.append(("gap", None))
            with_gaps.append((kind, name))
        return with_gaps

    def rows_of(self, kind):
        """``(row index, modality)`` of every row of one kind."""
        return [(i, name) for i, (k, name) in enumerate(self.rows) if k == kind]

    def is_absent(self, name: str) -> bool:
        """Whether a diagnostic has nothing to show: no valid sample, or constant.

        An unrecorded diagnostic is missing when measured and constant when decoded;
        its prediction is then drawn from nothing and is not shown either.
        """
        entry = self.decoded[name]
        return bool(not np.isfinite(entry["gt"]).any() or self.static.get(name))

    def not_recorded(self, ax):
        """Grey panel standing in for a diagnostic the shot did not record."""
        ax.set_facecolor("0.94")
        ax.text(
            0.5,
            0.5,
            "not recorded",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=self.pt - 2,
            color="0.45",
        )
        ax.set_yticks([])

    def letter(self, ax):
        """Panel letter at the figure's left edge, level with the row top."""
        top = ax.get_position().y1
        self.fig.text(
            0.012,
            top + 0.004,
            next(self.letters),
            fontsize=self.pt,
            fontweight="bold",
            va="bottom",
            ha="left",
        )

    def time_axis(self, ax, ticks: bool, xlabel: bool = False):
        ax.set_xlim(self.t_start, self.t_end)
        ax.axvline(self.t_rollout, color="0.45", ls="--", lw=0.9)
        self.shade_after_plasma(ax)
        ax.tick_params(labelbottom=ticks)
        if xlabel:
            ax.set_xlabel("Time (s)")

    def shade_after_plasma(self, ax):
        """Grey out the frames after the plasma ended (the vessel is empty)."""
        if self.plasma_end is None or self.plasma_end >= self.t_end:
            return
        ax.axvspan(
            max(self.plasma_end, self.t_start),
            self.t_end,
            color="0.82",
            alpha=0.75,
            lw=0,
            zorder=5,
        )

    def metric_box(self, ax, x, y, prediction, truth, entry, va, alpha):
        """Boxed r / nRMSE / skill annotation of a prediction panel, if enabled."""
        options = self.options
        skill = entry.get("nrmse_skill") if options.show_skill else None
        parts = []
        if options.show_r:
            parts.append(f"r {_corr(prediction, truth):.3f}")
        if options.show_nrmse:
            parts.append(f"nRMSE {entry['nrmse']:.3f}")
        if skill is not None and np.isfinite(skill):
            parts.append(f"skill {skill:+.2f}")
        if not parts:
            return
        text = " · ".join(parts)
        ax.text(
            x,
            y,
            text,
            transform=ax.transAxes,
            ha="right",
            va=va,
            fontsize=self.pt - 3,
            color="0.15",
            bbox=dict(boxstyle="square,pad=0.18", fc="white", ec="none", alpha=alpha),
        )


def _draw_slowts_rows(figure: _Figure) -> None:
    """Most variable channel of each slow time series, truth and prediction."""
    options, pt = figure.options, figure.pt
    rows = figure.rows_of("slow")
    last_row = rows[-1][0] if rows else None
    for i, name in rows:
        entry = figure.decoded[name]
        ax = figure.fig.add_subplot(figure.grid[i])
        absent = figure.is_absent(name)
        tag = ""
        if entry.get("space") == "physical" and name in _SLOWTS_UNITS:
            scale, ylabel, tag = _SLOWTS_UNITS[name]
        else:
            scale, ylabel = 1.0, _SLOWTS_SHORT.get(name, name)
        channel = 0
        if absent:
            figure.not_recorded(ax)
        else:
            truth = np.where(np.isfinite(entry["gt"]), entry["gt"], np.nan)
            prediction = np.where(np.isfinite(entry["pred"]), entry["pred"], np.nan)
            channel = _best_channel(entry["gt"], reduce_axes=(2,))
            with np.errstate(invalid="ignore"):
                truth_trace = np.nanmean(truth[:, channel, :], axis=-1) * scale
                prediction_trace = (
                    np.nanmean(prediction[:, channel, :], axis=-1) * scale
                )
            # Missing measured samples are skipped, so the trace stays one line.
            valid = np.isfinite(truth_trace)
            ax.plot(
                figure.times[valid],
                truth_trace[valid],
                color=_GT_COLOR,
                lw=LW_PRIMARY,
                solid_capstyle="round",
                zorder=2,
            )
            if not options.gt_only:
                ax.plot(
                    figure.times,
                    prediction_trace,
                    color=_PRED_COLOR,
                    lw=LW_SECONDARY,
                    ls=DASH_SECONDARY,
                    zorder=3,
                )
            if figure.legend_host is None:
                figure.legend_host = ax
            ax.grid(alpha=GRID_ALPHA, lw=GRID_LW)
        ax.set_ylabel(ylabel)
        skill = entry.get("nrmse_skill") if options.show_skill else None
        parts = [
            part
            for part in (
                tag if options.show_tag else "",
                f"ch{channel:02d}" if options.show_channel and not absent else "",
                f"nRMSE {entry['nrmse']:.3f}"
                if options.show_nrmse and not absent
                else "",
                f"skill {skill:+.2f}"
                if skill is not None and np.isfinite(skill) and not absent
                else "",
            )
            if part
        ]
        if parts:
            ax.text(
                0.995,
                1.04,
                " · ".join(parts),
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=pt - 3,
                color="0.35",
            )
        figure.time_axis(ax, ticks=(i == last_row), xlabel=False)
        figure.letter(ax)
        if i == rows[0][0]:
            ax.text(
                figure.t_rollout + 0.02,
                1.04,
                "prediction start",
                transform=ax.get_xaxis_transform(),
                fontsize=pt - 3,
                color="0.45",
                va="bottom",
                ha="left",
            )
            if figure.plasma_end is not None and figure.plasma_end < figure.t_end:
                ax.text(
                    figure.plasma_end + 0.02,
                    1.04,
                    "plasma ended",
                    transform=ax.get_xaxis_transform(),
                    fontsize=pt - 3,
                    color="0.45",
                    va="bottom",
                    ha="left",
                )


def _draw_spectro_rows(figure: _Figure) -> None:
    """Stitched spectrogram strips; truth and prediction share scale and axes."""
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    from matplotlib.ticker import MaxNLocator

    options, K0 = figure.options, figure.K0
    rows = figure.rows_of("spec")
    first_row = rows[0][0] if rows else None
    last_row = rows[-1][0] if rows else -1
    shared_time_axis = None
    for i, name in rows:
        entry = figure.decoded[name]
        absent = figure.is_absent(name)
        panels = GridSpecFromSubplotSpec(
            1, 1 if options.gt_only else 2, subplot_spec=figure.grid[i], wspace=0.06
        )
        extent = (
            figure.t_start,
            figure.t_end,
            0.0,
            float(entry["gt"].shape[2]) * figure.khz_per_bin,
        )
        ax_truth = figure.fig.add_subplot(panels[0], sharex=shared_time_axis)
        ax_prediction = (
            None
            if options.gt_only
            else figure.fig.add_subplot(panels[1], sharex=ax_truth, sharey=ax_truth)
        )
        if shared_time_axis is None:
            shared_time_axis = ax_truth
        axes = [ax_truth] if options.gt_only else [ax_truth, ax_prediction]
        if absent:
            for ax in axes:
                figure.not_recorded(ax)
                ax.set_xlim(figure.t_start, figure.t_end)
                ax.set_ylim(extent[2], extent[3])
        else:
            channel = _best_channel(entry["gt"], reduce_axes=(2, 3))
            truth_image = _stitch(entry["gt"], channel)
            vmin, vmax = _robust_limits(truth_image, 99.5)
            images = [truth_image]
            if not options.gt_only:
                images.append(_stitch(entry["pred"], channel))
            for ax, image in zip(axes, images):
                ax.imshow(
                    image,
                    aspect="auto",
                    origin="lower",
                    cmap=options.cmap_spectro,
                    vmin=vmin,
                    vmax=vmax,
                    extent=extent,
                    interpolation="none",
                )
                ax.axvline(figure.t_rollout, color="w", ls="--", lw=0.9)
                figure.shade_after_plasma(ax)
        for ax in axes:
            ax.set_xlim(figure.t_start, figure.t_end)
            ax.tick_params(labelbottom=(i == last_row))
            if i == last_row:
                ax.set_xlabel("Time (s)")
        if not options.gt_only:
            ax_prediction.tick_params(labelleft=False)
            ax_prediction.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="lower"))
        ax_truth.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="upper"))
        ax_truth.set_ylabel(f"{_display_name(name)}\nFreq (kHz)")
        if absent:
            ax_truth.set_yticks([0, 10, 20] if extent[3] < 40 else [0, 100, 200])
        if not options.gt_only and not absent:
            figure.metric_box(
                ax_prediction,
                0.98,
                0.94,
                entry["pred"][K0:],
                entry["gt"][K0:],
                entry,
                "top",
                0.8,
            )
        if i == first_row:
            ax_truth.set_title("Ground truth")
            if not options.gt_only:
                ax_prediction.set_title("Prediction")
        figure.letter(ax_truth)


def _draw_video_rows(figure: _Figure) -> None:
    """Mid-rollout camera frame, truth and prediction, with a colour bar."""
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    options, K0, mid = figure.options, figure.K0, figure.mid_frame
    rows = figure.rows_of("vid")
    first_row = rows[0][0] if rows else None
    for i, name in rows:
        entry = figure.decoded[name]
        absent = figure.is_absent(name)
        if options.gt_only:
            panels = GridSpecFromSubplotSpec(
                1, 2, subplot_spec=figure.grid[i], wspace=0.30, width_ratios=[1.0, 0.05]
            )
        else:
            panels = GridSpecFromSubplotSpec(
                1,
                3,
                subplot_spec=figure.grid[i],
                wspace=0.30,
                width_ratios=[1.0, 1.0, 0.05],
            )
        truth, prediction = entry["gt"], entry["pred"]
        ax_truth = figure.fig.add_subplot(panels[0])
        ax_prediction = None if options.gt_only else figure.fig.add_subplot(panels[1])
        colorbar_ax = figure.fig.add_subplot(panels[1 if options.gt_only else 2])
        axes = [ax_truth] if options.gt_only else [ax_truth, ax_prediction]
        if absent:
            for ax in axes:
                figure.not_recorded(ax)
            colorbar_ax.set_axis_off()
        else:
            mid_time = truth.shape[2] // 2
            vmin, vmax = _robust_limits(truth[mid, 0, mid_time], 99.0)
            image_style = dict(
                cmap=options.cmap_video, vmin=vmin, vmax=vmax, aspect="auto"
            )
            image = ax_truth.imshow(truth[mid, 0, mid_time], **image_style)
            if not options.gt_only:
                image = ax_prediction.imshow(
                    prediction[mid, 0, mid_time], **image_style
                )
            figure.fig.colorbar(image, cax=colorbar_ax)
            colorbar_ax.tick_params(labelsize=5.5, length=2, pad=1)
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        ax_truth.set_ylabel(_display_name(name))
        if not options.gt_only and not absent:
            figure.metric_box(
                ax_prediction,
                0.99,
                0.04,
                prediction[K0:],
                truth[K0:],
                entry,
                "bottom",
                0.78,
            )
        if i == first_row:
            ax_truth.set_title(
                f"Ground truth (t = {figure.t_origin + mid * FRAME_S:.2f} s)"
            )
            if not options.gt_only:
                ax_prediction.set_title("Prediction")
        figure.letter(ax_truth)


def _video_nrmse_per_frame(truth: np.ndarray, prediction: np.ndarray, K0: int, F: int):
    errors = []
    for frame in range(K0, F):
        usable = np.isfinite(truth[frame]) & np.isfinite(prediction[frame])
        if not usable.any():
            errors.append(np.nan)
            continue
        t = truth[frame][usable].astype(np.float64)
        p = prediction[frame][usable].astype(np.float64)
        errors.append(float(np.sqrt(np.mean((p - t) ** 2)) / (np.std(t) + 1e-12)))
    return errors


def _draw_video_nrmse_row(figure: _Figure) -> None:
    for i, _name in figure.rows_of("vrmse"):
        ax = figure.fig.add_subplot(figure.grid[i])
        for name in figure.video:
            if figure.is_absent(name):
                continue
            entry = figure.decoded[name]
            ax.plot(
                figure.times[figure.K0 :],
                _video_nrmse_per_frame(entry["gt"], entry["pred"], figure.K0, figure.F),
                color=_VIDEO_COLOR.get(name, _GT_COLOR),
                lw=LW_PRIMARY,
                label=name.replace("tangtv_", "").capitalize() + " divertor",
            )
        ax.set_ylabel("video nRMSE")
        ax.set_ylim(bottom=0.0)
        ax.legend(frameon=False, fontsize=6.5, loc="upper left")
        ax.grid(alpha=GRID_ALPHA, lw=GRID_LW)
        figure.time_axis(ax, ticks=True, xlabel=True)
        figure.letter(ax)


def _draw_legend(figure: _Figure) -> None:
    """Ground-truth / prediction legend inside the first time panel."""
    if figure.legend_host is None:
        return
    line = figure.plt.Line2D
    handles = [line([], [], color=_GT_COLOR, lw=LW_PRIMARY, label="Ground truth")]
    if not figure.options.gt_only:
        handles.append(
            line(
                [],
                [],
                color=_PRED_COLOR,
                lw=LW_SECONDARY,
                ls=DASH_SECONDARY,
                label="Prediction",
            )
        )
    figure.legend_host.legend(
        handles=handles,
        loc=figure.options.legend_loc,
        frameon=True,
        ncol=1 if figure.options.gt_only else 2,
        fontsize=figure.pt - 2,
        borderaxespad=0.25,
        handlelength=1.6,
        columnspacing=0.8,
        framealpha=0.9,
        edgecolor="none",
    )


def render_figure(
    decoded: Dict[str, Dict[str, np.ndarray]],
    shot: str,
    step: int,
    temperature: float,
    K0: int,
    F: int,
    out_dir: Path,
    t_origin: float = WARMUP_S,
    plasma_end: Optional[float] = None,
) -> Tuple[Path, Path]:
    """Render the ground-truth versus prediction figure of one shot.

    Frames after the plasma end are greyed out; ``plasma_end`` defaults to the
    shot's entry in ``IGNITE_PLASMA_END_JSON`` (see :func:`_plasma_end`).

    Parameters
    ----------
    decoded : dict
        Output of :func:`decode_all`.
    shot : str
        Shot id, used in the file names.
    step : int
        Checkpoint step, used in the file names.
    temperature : float
        Sampling temperature (unused; kept for callers).
    K0 : int
        First predicted frame.
    F : int
        Window length.
    out_dir : Path
        Output directory.
    t_origin : float, default=WARMUP_S
        Shot time of frame 0.
    plasma_end : float, default=None
        Plasma end time [s]; looked up by shot if None.

    Returns
    -------
    png, pdf : Path
        Paths of the written figure files.
    """
    if plasma_end is None:
        plasma_end = _plasma_end(shot)
    figure = _Figure(decoded, K0, F, t_origin, FIGURE_OPTIONS, plasma_end=plasma_end)
    for draw in (
        _draw_slowts_rows,
        _draw_spectro_rows,
        _draw_video_rows,
        _draw_video_nrmse_row,
        _draw_legend,
    ):
        draw(figure)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"dynamics_eval_{shot}_step{step}.png"
    pdf = out_dir / f"dynamics_eval_{shot}_step{step}.pdf"
    figure.fig.savefig(png, dpi=200)
    figure.fig.savefig(pdf)
    figure.plt.close(figure.fig)
    return png, pdf


def _read_manifest(cache_dir):
    """Return ``(manifest, error)`` for a cache's ``_codec_manifest.json``."""
    path = Path(cache_dir) / "_codec_manifest.json"
    if not path.exists():
        return {}, None
    try:
        manifest = json.loads(path.read_text())
    except Exception as e:  # noqa: BLE001 - the manifest is advisory
        return {}, e
    if not isinstance(manifest, dict):
        return {}, TypeError(f"manifest is a {type(manifest).__name__}, not an object")
    return manifest, None


def _load_bandpower(cache_dir, codecs, log) -> Dict:
    """Band-power decode specs from a ``bin_edges*.npz`` beside the cache, if any."""
    cache_dir = Path(cache_dir)
    candidates = (
        cache_dir / "bin_edges_128.npz",
        cache_dir.parent / "bin_edges_128.npz",
        cache_dir / "bin_edges.npz",
        cache_dir.parent / "bin_edges.npz",
    )
    edges_file = next((path for path in candidates if path.exists()), None)
    bandpower: Dict = {}
    if edges_file is None:
        return bandpower
    edges = np.load(edges_file)
    for name in edges.files:
        if name not in EVAL_MODALITIES or name not in codecs:
            continue
        modality_edges = edges[name]
        n_channels = int(getattr(codecs[name][1], "channels", 0) or 0)
        if (
            n_channels <= 0
            or modality_edges.ndim != 2
            or modality_edges.shape[1] % n_channels
        ):
            log(
                f"[eval] band-power {name}: edges {modality_edges.shape} not divisible "
                f"by C={n_channels} — falling back to the FSQ codec",
                flush=True,
            )
            continue
        bandpower[name] = (
            bandpower_levels(modality_edges),
            n_channels,
            modality_edges.shape[1] // n_channels,
        )
    if bandpower:
        log(
            f"[eval] band-power decode from {edges_file.name} for "
            + ", ".join(
                f"{name} (C={spec[1]}, {spec[2]} bands)"
                for name, spec in sorted(bandpower.items())
            ),
            flush=True,
        )
    return bandpower


def _resolve_val_shots(ckpt, cache_dir, val_tail, val_n, split_seed, log) -> str:
    """Return the last ``val_tail`` validation shots of the run, comma-separated.

    The split frozen next to the checkpoint is preferred; recomputing it is only
    valid if the cache has not changed since training.
    """
    split_file = Path(ckpt).parent / "split_shots.json"
    if split_file.exists():
        split = json.loads(split_file.read_text())
        cached = {path.stem for path in Path(cache_dir).glob("*.pt")}
        val = [shot for shot in split["val"] if shot in cached]
        log(
            f"[eval] split from {split_file} (frozen by the run): val={len(val)} "
            f"test={len(split.get('test', []))} HELD OUT, "
            f"seed={split.get('split_seed')}",
            flush=True,
        )
    else:
        _train, val, _test = td.split_shots(
            cache_dir, val_n or val_tail, split_seed=split_seed
        )
        log(
            f"[eval] NO split_shots.json next to the ckpt — RECOMPUTED the split "
            f"(val_n={val_n or val_tail}, split_seed={split_seed}). Valid only if the "
            f"cache is unchanged since training.",
            flush=True,
        )
    shots = ",".join(val[-val_tail:])
    log(f"[eval] val_tail={val_tail} -> shots: {shots}", flush=True)
    return shots


def _merge_rank_parts(out_dir, per_shot, all_shots, rank, world_size, log) -> bool:
    """Write this rank's results; rank 0 waits for and merges every part.

    Returns False on non-zero ranks, which are done after writing their part.
    """
    part = out_dir / f"_eval_part_rank{rank}.json"
    tmp = out_dir / f"_eval_part_rank{rank}.json.tmp"
    with open(tmp, "w") as f:
        json.dump(per_shot, f)
    tmp.replace(part)
    if rank != 0:
        log(f"[eval] rank {rank}: part written, exiting", flush=True)
        return False
    deadline = time.time() + _MERGE_TIMEOUT_S
    while time.time() < deadline:
        if len(sorted(out_dir.glob("_eval_part_rank*.json"))) >= world_size:
            break
        time.sleep(_MERGE_POLL_S)
    for part_file in sorted(out_dir.glob("_eval_part_rank*.json")):
        per_shot.update(json.loads(part_file.read_text()))
        part_file.unlink()
    missing = [shot for shot in all_shots if shot not in per_shot]
    if missing:
        log(
            f"[eval] WARNING: no results from {len(missing)} shots: {missing}",
            flush=True,
        )
    return True


def _scored_values(per_shot: Dict, field: str) -> Dict[str, list]:
    """Collect the non-NaN values of ``field`` per modality across shots.

    An absent diagnostic gives NaN for its shot, which must not poison the mean.
    """
    modalities = set().union(*(per_shot[shot][field] for shot in per_shot))
    values = {}
    for modality in sorted(modalities):
        shot_values = (
            per_shot[shot][field][modality]
            for shot in per_shot
            if modality in per_shot[shot][field]
        )
        values[modality] = [v for v in shot_values if not math.isnan(v)]
    return values


def _fraction(names, compare) -> Dict[str, float]:
    """Mean of a per-token boolean comparison, per modality."""
    return {name: float(compare(name).float().mean()) for name in names}


@dataclasses.dataclass
class _EvalContext:
    """Everything the per-shot evaluation needs, resolved once per run."""

    model: MaskGITDynamics
    cfg: DynamicsConfig
    step: int
    codecs: Dict
    bandpower: Dict
    device: torch.device
    n_seed: int
    temperature: float
    seed: int
    actuator_mode: str
    cache_dir: str
    out_dir: str
    use_raw_gt: bool
    data_dir: object
    cache_t0: float
    log: object
    guidance: float = 1.0
    decode: Optional[Dict] = None
    denorm: Optional[Dict] = None


def _rollouts(context: _EvalContext, cache, shot):
    """The rollout, and the real-actuator baseline of a counterfactual (or None)."""

    def rollout(actuator_mode):
        generator = torch.Generator(device=context.device).manual_seed(
            int(context.seed)
        )
        return rollout_shot(
            context.model,
            context.cfg,
            cache,
            context.n_seed,
            context.temperature,
            generator,
            context.device,
            actuator_mode=actuator_mode,
            cache_dir=context.cache_dir,
            shot=shot,
            guidance=context.guidance,
            decode=context.decode,
        )

    # bf16 rollouts measurably lower accuracy, so they are opt-in.
    bf16 = context.device.type == "cuda" and os.environ.get("IGNITE_EVAL_BF16") == "1"
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):
        result = rollout(context.actuator_mode)
        # A counterfactual is measured against the same rollout under the real
        # actuators, with an identical random stream. IGNITE_EVAL_NO_BASELINE=1 skips
        # that baseline (half the cost) when the real rollout is evaluated anyway.
        skip = os.environ.get("IGNITE_EVAL_NO_BASELINE") == "1"
        baseline = None if context.actuator_mode == "real" or skip else rollout("real")
    return result, baseline


def _archive_codes(context: _EvalContext, shot, result: Rollout, baseline: Optional[Rollout] = None) -> None:
    """Save the codes, so a figure can be re-rendered without a new rollout.

    A counterfactual also keeps its real-actuator baseline (same random stream) as
    ``real__<modality>``, so the actuator response can be scored from the archive alone.
    Expected frames, where the rollout has them, go in as ``mean__`` / ``realmean__``
    (float16 coordinates in ``[0, 1]``; read them back with :func:`archived`).
    """

    def means(rollout, prefix):
        if rollout is None or rollout.pred_mean is None:
            return {}
        return {
            f"{prefix}__{name}": value.numpy().astype(np.float16)
            for name, value in rollout.pred_mean.items()
        }

    try:
        # The archive is written before the figures, which create out_dir otherwise.
        Path(context.out_dir).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            Path(context.out_dir) / f"codes_{shot}_step{context.step}.npz",
            K0=np.int32(result.k0),
            F=np.int32(result.n_frames),
            step=np.int32(context.step),
            shot=str(shot),
            temperature=np.float32(context.temperature),
            guidance=np.float32(context.guidance),
            decode=json.dumps(context.decode or {}),
            **{
                f"gt__{name}": codes.cpu().numpy().astype(np.int32)
                for name, codes in result.gt_codes.items()
            },
            **{
                f"pred__{name}": codes.cpu().numpy().astype(np.int32)
                for name, codes in result.pred_codes.items()
            },
            **{
                f"real__{name}": codes.cpu().numpy().astype(np.int32)
                for name, codes in (baseline.pred_codes.items() if baseline is not None else ())
            },
            **means(result, "mean"),
            **means(baseline, "realmean"),
        )
    except Exception as e:
        context.log(
            f"[eval] WARNING: could not archive codes for {shot}: "
            f"{type(e).__name__}: {e}"
        )


def _evaluate_shot(context: _EvalContext, shot, is_first: bool, render: bool) -> Dict:
    """Roll out, score and optionally render one shot; return its metrics entry."""
    cache = load_shot_cache(Path(context.cache_dir), shot)
    result, baseline = _rollouts(context, cache, shot)
    gt, pred = result.gt_codes, result.pred_codes
    K0, F = result.k0, result.n_frames
    predicted = slice(K0, F)
    token_accuracy = _fraction(gt, lambda n: pred[n][predicted] == gt[n][predicted])
    # Codes are highly persistent, so accuracy only means something as skill.
    persistence = _fraction(gt, lambda n: gt[n][predicted] == gt[n][K0 - 1 : K0])
    _archive_codes(context, shot, result, baseline)
    measured = (
        load_raw_gt(
            shot,
            context.codecs,
            F,
            context.data_dir,
            t0_start=context.cache_t0,
            log=context.log,
        )
        if context.use_raw_gt
        else None
    )
    decoded = decode_all(
        context.codecs,
        gt,
        pred,
        K0,
        F,
        context.device,
        raw_gt=measured,
        bandpower=context.bandpower,
        denorm=context.denorm,
    )
    if is_first:
        self_check(decoded)
    entry = {
        "token_accuracy": token_accuracy,
        "token_accuracy_persistence": persistence,
        "token_accuracy_skill": {
            n: token_accuracy[n] - persistence[n] for n in token_accuracy
        },
        "nrmse": {n: d["nrmse"] for n, d in decoded.items()},
        "nrmse_persistence": {n: d["nrmse_persistence"] for n, d in decoded.items()},
        "nrmse_skill": {n: d["nrmse_skill"] for n, d in decoded.items()},
        "K0": K0,
        "F": F,
    }
    if baseline is not None:
        real = baseline.pred_codes
        entry["actuator_mode"] = context.actuator_mode
        entry["divergence_vs_real"] = _fraction(
            pred, lambda n: pred[n][predicted] != real[n][predicted]
        )
        entry["token_accuracy_real_actuators"] = _fraction(
            pred, lambda n: real[n][predicted] == gt[n][predicted]
        )
    if render:
        png, _pdf = render_figure(
            decoded,
            shot,
            context.step,
            context.temperature,
            K0,
            F,
            Path(context.out_dir),
        )
        entry["figure"] = str(png)
    return entry


def _summarise(per_shot, ckpt, step, temperature, seed, codec_tmpl) -> Dict:
    """Run-level metrics: means over the shots where each modality was scored."""

    def mean(field):
        return {
            modality: float(np.mean(values)) if values else float("nan")
            for modality, values in _scored_values(per_shot, field).items()
        }

    return {
        "ckpt": str(ckpt),
        "step": step,
        "temperature": temperature,
        "seed": int(seed),
        "codec_tmpl": codec_tmpl,
        "n_shots": len(per_shot),
        "shots": sorted(per_shot),
        "mean_token_accuracy": mean("token_accuracy"),
        "mean_nrmse": mean("nrmse"),
        "mean_token_accuracy_persistence": mean("token_accuracy_persistence"),
        "mean_token_accuracy_skill": mean("token_accuracy_skill"),
        "mean_nrmse_persistence": mean("nrmse_persistence"),
        "mean_nrmse_skill": mean("nrmse_skill"),
        "n_shots_scored_nrmse": {
            modality: len(values)
            for modality, values in _scored_values(per_shot, "nrmse").items()
        },
        "per_shot": per_shot,
    }


def _log_summary(metrics: Dict, log) -> None:
    log(
        "[eval] predicted region — model vs PERSISTENCE (freeze last seed frame):",
        flush=True,
    )
    log(
        f"  {'modality':24s} {'tok_acc':>8s} {'pers':>8s} {'skill':>8s} "
        f"{'nRMSE':>8s} {'pers':>8s} {'skill':>7s}",
        flush=True,
    )
    for name, accuracy in metrics["mean_token_accuracy"].items():
        nrmse = metrics["mean_nrmse"].get(name)
        decoded_columns = (
            f"{nrmse:8.3f} {metrics['mean_nrmse_persistence'][name]:8.3f} "
            f"{metrics['mean_nrmse_skill'][name]:+7.3f}"
            if nrmse is not None
            else " " * 25
        )
        log(
            f"  {name:24s} {accuracy:8.3f} "
            f"{metrics['mean_token_accuracy_persistence'][name]:8.3f} "
            f"{metrics['mean_token_accuracy_skill'][name]:+8.3f} " + decoded_columns,
            flush=True,
        )


def _codec_template(codec_tmpl, manifest, manifest_error, log):
    """The requested template, else the one that encoded the cache."""
    if codec_tmpl is not None:
        return codec_tmpl
    if manifest_error is not None:
        log(
            f"[eval] WARNING unreadable _codec_manifest.json ({manifest_error})",
            flush=True,
        )
        return None
    template = manifest.get("codec_tmpl") or None
    if template:
        log(f"[eval] codec_tmpl from cache manifest: {template}", flush=True)
    return template


def run(
    ckpt: str,
    shot: str,
    cache_dir: str,
    out_dir: str,
    temperature: float = 1.0,
    seed: int = 0,
    k0: int = 0,
    codec_tmpl: str = None,
    render_all: bool = False,
    val_tail: int = 0,
    val_n: int = 0,
    split_seed: int = 0,
    actuator_mode: str = "real",
    log=print,
    guidance: float = 1.0,
    decode: Optional[Dict] = None,
) -> Dict:
    """Evaluate a checkpoint on one or more shots and write ``eval_metrics.json``.

    Per shot: a rollout from ``K0`` real frames, token accuracy of every modality
    over the predicted frames, decoded nRMSE of the :data:`EVAL_MODALITIES` whose
    codec resolves, each against persistence, and the archived codes. Shots are
    sharded over ``RANK``/``WORLD_SIZE`` and merged by rank 0.

    Parameters
    ----------
    ckpt : str
        Checkpoint path.
    shot : str
        Comma-separated shot ids.
    cache_dir, out_dir : str
        Frame-code cache and output directory.
    temperature : float, default=1.0
        Sampling temperature.
    seed : int, default=0
        Rollout seed.
    k0 : int, default=0
        Seed frames; 0 uses the checkpoint's ``k0_seed``.
    codec_tmpl : str, default=None
        Codec checkpoint template; defaults to the one in the cache manifest, so
        the decoders match the codecs that wrote the cache.
    render_all : bool, default=False
        Render a figure for every shot instead of the first only.
    val_tail : int, default=0
        If positive, evaluate the last ``val_tail`` validation shots of the run
        instead of ``shot``.
    val_n, split_seed : int, default=0
        Split parameters, used only when the run froze no split.
    actuator_mode : str, default="real"
        Counterfactual (:func:`apply_actuator_mode`); any other mode also runs
        the real-actuator rollout with the same seed and reports the divergence.
    log : callable, default=print
        Logger.
    guidance : float, default=1.0
        Classifier-free guidance weight on the actuators; ``1.0`` is off.
    decode : dict, default=None
        Confidence gate and consensus options of the rollout
        (``gate``, ``consensus_n``, ``consensus_q``); ``None`` is the plain rollout.

    Returns
    -------
    metrics : dict
        Contents of ``eval_metrics.json``; on non-zero ranks only this rank's
        per-shot results.
    """
    device = _rank_device()
    log(
        f"[eval] device={device} ckpt={ckpt} actuator_mode={actuator_mode} "
        f"guidance={guidance} decode={decode or {}}",
        flush=True,
    )
    if val_tail:
        shot = _resolve_val_shots(ckpt, cache_dir, val_tail, val_n, split_seed, log)
    # Measured ground truth needs the cache's frame origin.
    manifest, manifest_error = _read_manifest(cache_dir)
    try:
        cache_t0 = float(manifest.get("t0_start", 1.0))
    except (TypeError, ValueError):
        cache_t0 = 1.0
    use_raw_gt = os.environ.get("IGNITE_EVAL_RAW_GT", "1") != "0"
    if use_raw_gt:
        log(
            f"[eval] GROUND TRUTH = measured input (t0_start={cache_t0} from the cache "
            f"manifest); metrics therefore include codec reconstruction error and are "
            f"NOT comparable to decoded-GT runs. Set IGNITE_EVAL_RAW_GT=0 for the old "
            f"behaviour.",
            flush=True,
        )
    model, cfg, step = load_model(Path(ckpt), device)
    n_seed = int(k0) if k0 else cfg.k0_seed
    log(
        f"[eval] model loaded: d_model={cfg.d_model} depth={cfg.depth} "
        f"heads={cfg.n_heads} step={step} K0={n_seed} max_frames={cfg.max_frames} "
        f"frame_tokens={cfg.tokens_per_frame}",
        flush=True,
    )
    log(
        "[eval] history cache: "
        + ("on" if os.environ.get("IGNITE_HISTORY_CACHE", "0") == "1" else "off")
        + " (IGNITE_HISTORY_CACHE; applies to plain rollouts without guidance)",
        flush=True,
    )
    codec_tmpl = _codec_template(codec_tmpl, manifest, manifest_error, log)
    codecs = td.load_frozen_codecs(
        list(EVAL_MODALITIES), repo=Path.cwd(), tmpl=codec_tmpl
    )
    for codec, _codec_cfg, _family in codecs.values():
        codec.to(device)
    log(
        f"[eval] loaded {len(codecs)} frozen codecs "
        f"(tmpl={codec_tmpl or 'manifest'}): {sorted(codecs)}",
        flush=True,
    )
    context = _EvalContext(
        model=model,
        cfg=cfg,
        step=step,
        codecs=codecs,
        bandpower=_load_bandpower(cache_dir, codecs, log),
        device=device,
        n_seed=n_seed,
        temperature=temperature,
        seed=seed,
        actuator_mode=actuator_mode,
        cache_dir=cache_dir,
        out_dir=out_dir,
        use_raw_gt=use_raw_gt,
        data_dir=os.environ.get("IGNITE_DATA_DIR") or spike.DEFAULT_DATA_DIR,
        cache_t0=cache_t0,
        guidance=guidance,
        decode=decode,
        denorm=slowts_denorm(codecs),
        log=log,
    )

    all_shots = [s.strip() for s in str(shot).split(",") if s.strip()]
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    shots = all_shots[rank::world_size]
    log(
        f"[eval] rank {rank}/{world_size}: {len(shots)}/{len(all_shots)} shots",
        flush=True,
    )
    per_shot: Dict[str, Dict] = {}
    figures = []
    for index, shot_id in enumerate(shots):
        entry = _evaluate_shot(
            context, shot_id, is_first=index == 0, render=index == 0 or render_all
        )
        if "figure" in entry:
            figures.append(entry["figure"])
        per_shot[shot_id] = entry
        accuracy = entry["token_accuracy"]
        log(
            f"[eval] shot {shot_id}: acc(mean)="
            f"{sum(accuracy.values()) / max(len(accuracy), 1):.3f} "
            f"({index + 1}/{len(shots)})",
            flush=True,
        )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if world_size > 1 and not _merge_rank_parts(
        out, per_shot, all_shots, rank, world_size, log
    ):
        return {"per_shot": per_shot, "rank": rank}
    metrics = _summarise(per_shot, ckpt, step, temperature, seed, codec_tmpl)
    metrics["guidance"] = float(guidance)
    metrics["decode"] = dict(decode or {})
    with open(out / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    _log_summary(metrics, log)
    log(
        f"[eval] metrics: {out / 'eval_metrics.json'}"
        + (f"\n[eval] figure: {figures[0]}" if figures else ""),
        flush=True,
    )
    return metrics


def rerender(eval_dir: str, cache_dir: str, codec_tmpl: str = None, log=print) -> list:
    """Redraw the figures of a finished evaluation from its archived codes.

    No rollout is run: every ``codes_<shot>_step<N>.npz`` in ``eval_dir`` is
    decoded and rendered again, in place, with the current figure code. Shots are
    sharded over ``RANK``/``WORLD_SIZE``.

    Parameters
    ----------
    eval_dir : str
        Folder of a finished :func:`run`.
    cache_dir : str
        Frame-code cache the evaluation used (for the manifest and frame origin).
    codec_tmpl : str, default=None
        Codec checkpoint template; defaults to the cache manifest's.
    log : callable, default=print
        Logger.

    Returns
    -------
    figures : list of str
        PNG paths written by this rank.
    """
    device = _rank_device()
    manifest, manifest_error = _read_manifest(cache_dir)
    cache_t0 = float(manifest.get("t0_start", 1.0)) if manifest else 1.0
    codec_tmpl = _codec_template(codec_tmpl, manifest, manifest_error, log)
    codecs = td.load_frozen_codecs(
        list(EVAL_MODALITIES), repo=Path.cwd(), tmpl=codec_tmpl
    )
    for codec, _codec_cfg, _family in codecs.values():
        codec.to(device)
    denorm = slowts_denorm(codecs)
    use_raw_gt = os.environ.get("IGNITE_EVAL_RAW_GT", "1") != "0"
    data_dir = os.environ.get("IGNITE_DATA_DIR") or spike.DEFAULT_DATA_DIR
    archives = sorted(Path(eval_dir).glob("codes_*_step*.npz"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    figures = []
    for archive in archives[rank::world_size]:
        codes = np.load(archive, allow_pickle=True)
        shot, step = str(codes["shot"]), int(codes["step"])
        K0, F = int(codes["K0"]), int(codes["F"])
        gt = {
            key[4:]: torch.from_numpy(codes[key]).long()
            for key in codes.files
            if key.startswith("gt__")
        }
        pred = {
            key[6:]: torch.from_numpy(codes[key]).long()
            for key in codes.files
            if key.startswith("pred__")
        }
        measured = (
            load_raw_gt(shot, codecs, F, data_dir, t0_start=cache_t0, log=log)
            if use_raw_gt
            else None
        )
        decoded = decode_all(
            codecs, gt, pred, K0, F, device, raw_gt=measured, denorm=denorm
        )
        png, _pdf = render_figure(
            decoded, shot, step, float(codes["temperature"]), K0, F, Path(eval_dir)
        )
        figures.append(str(png))
        log(f"[rerender] {png}", flush=True)
    return figures


def curate_core(
    cache_dir,
    out_path=None,
    k0: int = 20,
    n_frames: int = 60,
    thresh: float = 0.999,
    shots=None,
    require_all: bool = True,
    log=print,
):
    """List the cached shots in which every modality changes over the eval window.

    A missing or parked diagnostic encodes to frozen codes, which hands model and
    persistence a free accuracy of 1. The statistic is the persistence accuracy
    of :func:`run`: the fraction of tokens in ``[k0, F)`` equal to frame
    ``k0 - 1``; a modality is dynamic below ``thresh``.

    Parameters
    ----------
    cache_dir : path-like
        Frame-code cache.
    out_path : path-like, default=None
        If given, write the list and report there as JSON.
    k0 : int, default=20
        First predicted frame.
    n_frames : int, default=60
        Window length.
    thresh : float, default=0.999
        Persistence accuracy at or above which a modality counts as frozen.
    shots : list of str, default=None
        Shots to screen; None screens the whole cache.
    require_all : bool, default=True
        Require every modality to be dynamic, instead of at least one.
    log : callable, default=print
        Logger.

    Returns
    -------
    curated : list of str
        Shots passing the screen.
    report : dict
        ``{shot: {modality: persistence_accuracy}}``.
    """
    cache = Path(cache_dir)
    stems = sorted(p.stem for p in cache.glob("*.pt")) if shots is None else list(shots)
    curated, report, n_too_short = [], {}, 0
    for shot in stems:
        try:
            entry = torch.load(cache / f"{shot}.pt", map_location="cpu")
        except Exception as e:
            log(f"[curate] skip {shot}: {type(e).__name__}: {e}")
            continue
        window = min(int(entry.get("n_frames", 0)) or n_frames, n_frames)
        if window <= k0:
            n_too_short += 1
            continue
        persistence = {}
        for modality, codes in entry["codes"].items():
            if codes.shape[0] < window:
                persistence[modality] = float("nan")
                continue
            persistence[modality] = float(
                (codes[k0:window] == codes[k0 - 1 : k0]).float().mean()
            )
        report[shot] = persistence
        scored = [v for v in persistence.values() if not math.isnan(v)]
        dynamic = [v for v in scored if v < thresh]
        passes = len(dynamic) == len(scored) if require_all else bool(dynamic)
        if passes and len(scored) == len(persistence):
            curated.append(shot)
    n_modalities = len(next(iter(report.values()), {}))
    log(
        f"[curate] {len(curated)}/{len(report)} shots have ALL {n_modalities} "
        f"modalities dynamic over frames [{k0},{n_frames}) (+{n_too_short} too short)"
    )
    if report:
        n_frozen = {}
        for persistence in report.values():
            for modality, value in persistence.items():
                frozen = not math.isnan(value) and value >= thresh
                n_frozen[modality] = n_frozen.get(modality, 0) + int(frozen)
        most_frozen = sorted(n_frozen.items(), key=lambda item: -item[1])[:6]
        log(
            "[curate] most frequently FROZEN modalities: "
            + ", ".join(f"{m} {count}/{len(report)}" for m, count in most_frozen)
        )
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(out_path) + ".tmp")
        record = {
            "cache_dir": str(cache),
            "k0": k0,
            "n_frames": n_frames,
            "threshold": thresh,
            "require_all": require_all,
            "curated": curated,
            "per_shot": report,
        }
        tmp.write_text(json.dumps(record, indent=1))
        tmp.replace(Path(out_path))
        log(f"[curate] -> {out_path}")
    return curated, report


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the command-line parser of the evaluation."""
    parser = argparse.ArgumentParser("IGNITE Phase-B dynamics evaluation")
    add = parser.add_argument
    add("--ckpt", help="dynamics checkpoint; not needed with --curate")
    add(
        "--shot",
        default="200729",
        help="comma-separated shots; the figure covers the first unless --render_all",
    )
    add("--cache_dir", default=_DEFAULT_CACHE, help="pre-encoded frame-code cache dir")
    add("--out_dir", required=True, help="output dir for figures and eval_metrics.json")
    add("--temperature", type=float, default=1.0, help="rollout sampling temperature")
    add(
        "--guidance",
        type=float,
        default=1.0,
        help="classifier-free guidance weight on the actuators; 1 = off. Needs a "
        "checkpoint trained with --act_drop_p > 0",
    )
    add(
        "--gate",
        type=float,
        default=0.0,
        help="confidence gate: tokens sampled with probability below it keep the "
        "previous frame's code; 0 = off",
    )
    add(
        "--consensus_n",
        type=int,
        default=1,
        help="consensus rollout: candidates per frame; 1 = off",
    )
    add(
        "--consensus_q",
        type=float,
        default=0.5,
        help="share of candidates that must agree (within one FSQ cell) to change a "
        "token",
    )
    add(
        "--consensus_mode",
        choices=("majority", "mean"),
        default="majority",
        help="combine the candidates by near-1 majority or by their mean FSQ "
        "coordinates",
    )
    add(
        "--consensus_independent",
        action="store_true",
        help="consensus over independent rollouts (an ensemble) instead of candidates "
        "that share the committed history",
    )
    add(
        "--infer",
        default="",
        help="comma list of modalities to infer from the others: every other "
        "modality keeps its measured codes in every predicted frame and only these "
        "are generated (missing-diagnostic inference); empty = plain rollout",
    )
    add(
        "--hide",
        default="",
        help="comma list of modalities missing from the initial state ([MASK] in the "
        "seed): the first predicted frame generates them from the other modalities "
        "and then stands in for their seed; every modality is then predicted "
        "autoregressively as in a plain rollout (excludes --infer)",
    )
    add(
        "--mask_absent",
        type=int,
        choices=(0, 1),
        default=None,
        help="1 (default from the CLI, IGNITE_MASK_ABSENT): a diagnostic that did not "
        "record in the initial state is [MASK] there and predicted with all others; "
        "0: it is continued as recorded (constant 'not recorded' frame). Not with "
        "--infer or consensus",
    )
    add(
        "--mask_forbid",
        type=int,
        choices=(0, 1),
        default=None,
        help="with --mask_absent: 1 never samples the masked diagnostic's 'not "
        "recorded' codes (it must be predicted as recording); 0 (default, "
        "IGNITE_MASK_FORBID) predicts it freely",
    )
    add(
        "--forbid_modalities",
        default="",
        help="with --forbid_absent and neither --infer nor --hide: comma list of "
        "modalities whose 'not recorded' codes are never sampled in a plain rollout "
        "(the initial state is kept as measured)",
    )
    add(
        "--infer_blind",
        action="store_true",
        help="with --infer: the inferred modalities are also [MASK] in every history "
        "frame, seed included, so they come from the other modalities alone",
    )
    add(
        "--infer_bootstrap",
        action="store_true",
        help="with --infer_blind: only the first predicted frame is generated blind; "
        "it then stands in for the inferred modalities' seed frames, so no later "
        "frame sees [MASK] in its history",
    )
    add(
        "--forbid_absent",
        default="",
        help="shot in which the --infer modalities did not record: their constant "
        "'not recorded' codes there are never sampled, so an inferred diagnostic "
        "cannot be generated as switched off",
    )
    add(
        "--seed_from",
        default="",
        help="comma list <modality>:<shot>: that modality's seed frames are taken from "
        "the other shot (same cache), e.g. tangtv_upper:203489, to give a modality "
        "that did not record a seed; the scored measurement stays this shot's",
    )
    add("--seed", type=int, default=0, help="rollout seed")
    add("--k0", type=int, default=0, help="seed frames; 0 = the checkpoint's k0_seed")
    add(
        "--codec_tmpl",
        default=None,
        help="codec checkpoint template; defaults to the cache manifest's",
    )
    add("--render_all", action="store_true", help="render a figure for every shot")
    add(
        "--val_tail",
        type=int,
        default=0,
        help="evaluate the last N validation shots of the run (overrides --shot)",
    )
    add("--val_n", type=int, default=0, help="validation size, with --val_tail")
    add(
        "--split_seed",
        type=int,
        default=0,
        help="split seed of the run, used if it froze no split",
    )
    add(
        "--actuator_mode",
        default="real",
        help="actuator counterfactual over the predicted frames: real | zero | "
        "freeze | shuffle | donor:<shot> | group_zero:<group> | "
        "scale:<group>=<factor>[@<frame>] (physical units, e.g. scale:pinj=0@50 "
        "turns the beams off from frame 50) | ramp:<group>=<factor>@<frame>/<frames> "
        "(hold the value at frame, ramp to factor times it over frames) | set:... | "
        "balance:... | linear:... (full forms in eval_dynamics.ACTUATOR_MODES)",
    )
    add(
        "--rerender",
        action="store_true",
        help="redraw the figures of the finished evaluation in --out_dir from its "
        "archived codes, without a new rollout",
    )
    curate = parser.add_argument_group("curation (runs instead of an evaluation)")
    curate.add_argument(
        "--curate",
        action="store_true",
        help="list the shots whose every modality changes over the eval window, "
        "without a model",
    )
    curate.add_argument(
        "--curate_out",
        default=None,
        help="curated list output (default <out_dir>/eval_core.json)",
    )
    curate.add_argument(
        "--curate_frames", type=int, default=60, help="window length of --curate"
    )
    curate.add_argument(
        "--curate_split",
        default=None,
        help="a run's split_shots.json; screen only one of its partitions",
    )
    curate.add_argument(
        "--curate_partition",
        default="val",
        choices=("train", "val", "test"),
        help="partition of --curate_split to screen",
    )
    return parser


def main(argv=None):
    """Run the evaluation or the curation selected by the command line."""
    os.environ.setdefault("IGNITE_HISTORY_CACHE", "1")  # inference: history cache on
    os.environ.setdefault("IGNITE_MASK_ABSENT", "1")  # inference: absent seeds masked
    args = build_arg_parser().parse_args(argv)
    if args.curate:
        shots = None
        if args.curate_split:
            split = json.loads(Path(args.curate_split).read_text())
            shots = split[args.curate_partition]
            print(
                f"[curate] screening the '{args.curate_partition}' partition of "
                f"{args.curate_split}: {len(shots)} shots",
                flush=True,
            )
        curated, _report = curate_core(
            args.cache_dir,
            out_path=args.curate_out or (Path(args.out_dir) / "eval_core.json"),
            k0=args.k0 or 20,
            n_frames=args.curate_frames,
            shots=shots,
        )
        return 0 if curated else 1
    if args.rerender:
        rerender(args.out_dir, args.cache_dir, codec_tmpl=args.codec_tmpl)
        return 0
    if not args.ckpt:
        raise SystemExit("--ckpt is required (omit only with --curate or --rerender)")
    return run(
        args.ckpt,
        args.shot,
        args.cache_dir,
        args.out_dir,
        temperature=args.temperature,
        seed=args.seed,
        k0=args.k0,
        codec_tmpl=args.codec_tmpl,
        render_all=args.render_all,
        val_tail=args.val_tail,
        val_n=args.val_n,
        split_seed=args.split_seed,
        guidance=args.guidance,
        decode={
            "gate": args.gate,
            "consensus_n": args.consensus_n,
            "consensus_q": args.consensus_q,
            "consensus_mode": args.consensus_mode,
            "consensus_shared": not args.consensus_independent,
            "infer": [x.strip() for x in args.infer.split(",") if x.strip()],
            "hide": [x.strip() for x in args.hide.split(",") if x.strip()],
            "forbid_mods": [x.strip() for x in args.forbid_modalities.split(",") if x.strip()],
            "mask_absent": None if args.mask_absent is None else bool(args.mask_absent),
            "mask_forbid": None if args.mask_forbid is None else bool(args.mask_forbid),
            "infer_blind": args.infer_blind,
            "infer_bootstrap": args.infer_bootstrap,
            "forbid_absent": args.forbid_absent.strip() or None,
            "seed_from": dict(
                x.strip().split(":", 1) for x in args.seed_from.split(",") if x.strip()
            ),
        },
        actuator_mode=args.actuator_mode,
    )


if __name__ == "__main__":
    main()
