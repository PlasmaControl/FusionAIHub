"""The DIII-D actuators IGNITE ingests: channel layout, names, units and data locations.

IGNITE conditions every frame (50 ms) on 88 actuator channels, the frame means of the
groups in :data:`~..train_dynamics._ACT_SPEC`. Nothing about the machine is written
here: channel names are the MDSplus/PTDATA signal names of the data config
(``modalities.yaml``), and every number (what each channel has delivered, torque per MW,
which sources and gyrotrons ran in a campaign) is measured from every tokenized shot by
:mod:`.index` and read with :func:`limits` and :func:`database_limits`. A plan is feasible
when it stays within what the database shows DIII-D has run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np

from .. import train_dynamics as td

# Frame grid of the frame-code caches: frame k covers [T0 + k * FRAME_S, T0 + (k+1) * FRAME_S).
FRAME_S = 0.05
T0 = 1.0
# IGNITE rollouts: 20 seed frames (1-2 s), predicted frames 20-99 (2-6 s).
K0 = 20
N_FRAMES = 100

MODELS = Path(os.environ.get("IGNITE_MODELS", "/scratch/gpfs/EKOLEMEN/ps9551/ignite/models"))
INDEX_DIR = Path(os.environ.get("IGNITE_DESIGNER_INDEX", MODELS / "designer_index"))
# Normalisation IGNITE was trained with (dataset-wide mu, sd per channel).
ACT_STATS = MODELS / "ignite_prod_v4" / "act_global" / "act_global_stats.npz"


class TokenSet(NamedTuple):
    """One directory of tokenized shots and its actuator side cache."""

    name: str
    frame_codes: Path
    act_global: Path
    plasma_end: Path


TOKEN_SETS: Tuple[TokenSet, ...] = tuple(
    TokenSet(name, MODELS / name / "frame_codes", MODELS / name / "act_global",
             MODELS / name / "plasma_end")
    for name in ("ignite_prod_v4", "ignite_fdp_205000_v4", "ignite_fdp_below190000_v4")
)

_MODALITIES = (Path(td.__file__).resolve().parents[1] / "data" / "config" / "modalities"
               / "modalities.yaml")


def _signal_name(key: str) -> str:
    """Short channel name from a signal path: the MDSplus node (``\\D3D::TOP.NB.NB21L:PINJ_21L``
    -> ``NB21L``, ``...ECH.LEIA:ECLEIFPWRC`` -> ``LEIA``) or the PTDATA point name without
    its trailing ``F`` (``IU30F`` -> ``IU30``)."""
    if ":" in key:
        return key.split(":")[-2].split(".")[-1]
    return key[:-1] if key.endswith("F") else key


def _signals() -> Dict[str, Tuple[str, ...]]:
    import yaml

    found: Dict[str, Tuple[str, ...]] = {}

    def walk(d):
        if isinstance(d, dict):
            for key, v in d.items():
                if key in dict(td._ACT_SPEC) and isinstance(v, dict) and "input_key" in v:
                    found[key] = tuple(v["input_key"])
                else:
                    walk(v)
        elif isinstance(d, list):
            for v in d:
                walk(v)

    walk(yaml.safe_load(_MODALITIES.read_text()))
    return found


# Signal paths and short names of the 88 channels, in IGNITE's order.
SIGNALS: Dict[str, Tuple[str, ...]] = _signals()
CHANNEL_NAMES: Dict[str, Tuple[str, ...]] = {
    g: tuple(_signal_name(k) for k in SIGNALS[g]) for g, _ in td._ACT_SPEC}
BEAMS = CHANNEL_NAMES["pinj"]
UNITS = {"ech_power": "W", "pinj": "W", "beam_voltage": "V", "tinj": "N m",
         "gas_flow": "Torr L/s", "gas_raw": "raw valve signal", "rmp": "A", "i_coil": "A"}
# Units the designer talks in, and the factor from the stored unit.
DISPLAY = {"ech_power": ("MW", 1e-6), "pinj": ("MW", 1e-6), "beam_voltage": ("kV", 1e-3),
           "tinj": ("N m", 1.0), "gas_flow": ("Torr L/s", 1.0), "gas_raw": ("raw", 1.0),
           "rmp": ("kA", 1e-3), "i_coil": ("kA", 1e-3)}
GROUPS = tuple(key for key, _ in td._ACT_SPEC)
assert all(len(CHANNEL_NAMES[g]) == n for g, n in td._ACT_SPEC), "channel names out of date"
assert all(len(set(v)) == len(v) for v in CHANNEL_NAMES.values()), "channel names not unique"
N_CHANNELS = sum(n for _, n in td._ACT_SPEC)


def columns(group: str) -> slice:
    """Columns of one actuator group in the 88-channel layout."""
    offset = 0
    for key, n in td._ACT_SPEC:
        if key == group:
            return slice(offset, offset + n)
        offset += n
    raise KeyError(f"unknown actuator group {group!r}; known: {GROUPS}")


def column(group: str, channel) -> int:
    """Column of one channel, given by name (``"NB21L"``, ``"GASA"``) or index."""
    names = CHANNEL_NAMES[group]
    if isinstance(channel, str) and not channel.isdigit():
        key = channel.upper().removesuffix("F")
        matches = [i for i, n in enumerate(names) if n.upper() == key]
        if not matches:
            raise KeyError(f"{group} has no channel {channel!r}; channels: {names}")
        index = matches[0]
    else:
        index = int(channel)
        if not 0 <= index < len(names):
            raise KeyError(f"{group} channel index {index} outside 0..{len(names) - 1}")
    return columns(group).start + index


def resolve(spec: str) -> List[int]:
    """Columns of ``"group"`` or ``"group:ch1,ch2"`` (names or indices)."""
    group, _, chans = spec.partition(":")
    group = group.strip()
    if not chans:
        cols = columns(group)
        return list(range(cols.start, cols.stop))
    return [column(group, c.strip()) for c in chans.split(",") if c.strip()]


def alias_columns(col: int) -> List[int]:
    """Columns that record the same physical signal (the 12 RMP coil currents appear
    in both ``rmp`` and ``i_coil``); an edit of one must be applied to all."""
    group = next(g for g in GROUPS if columns(g).start <= col < columns(g).stop)
    if group not in ("rmp", "i_coil"):
        return [col]
    name = CHANNEL_NAMES[group][col - columns(group).start]
    out = [columns(g).start + CHANNEL_NAMES[g].index(name)
           for g in ("rmp", "i_coil") if name in CHANNEL_NAMES[g]]
    return sorted(set(out))


def frame_of(t: float) -> int:
    """Cache frame whose window starts at shot time ``t`` (s), on the 50 ms grid."""
    k = (float(t) - T0) / FRAME_S
    if abs(k - round(k)) > 1e-6:
        raise ValueError(f"time {t} s is not on the 50 ms frame grid starting at {T0} s")
    return int(round(k))


def time_of(k) -> np.ndarray:
    """Start time (s) of cache frame(s) ``k``."""
    return T0 + FRAME_S * np.asarray(k, dtype=np.float64)


def act_stats() -> Tuple[np.ndarray, np.ndarray]:
    """Dataset-wide (mu, sd) of IGNITE's actuator normalisation."""
    loaded = np.load(ACT_STATS, allow_pickle=True)
    return np.asarray(loaded["mu"], np.float32), np.asarray(loaded["sd"], np.float32)


_LIMITS: Optional[dict] = None


def limits() -> dict:
    """Machine limits measured from all tokenized shots (written by :mod:`.index`)."""
    global _LIMITS
    if _LIMITS is None:
        path = INDEX_DIR / "machine_limits.json"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing; build it with "
                "`python -m tokamak_foundation_model.ignite.designer.index`"
            )
        _LIMITS = json.loads(path.read_text())
    return _LIMITS


def era_limits(shot: int) -> dict:
    """What the database shows for the 2000-shot block nearest ``shot`` (about one campaign):
    which sources, gyrotrons, valves and coils ran there, their maxima and each beam's
    torque per MW by direction; and, per channel, the most it delivered in the whole
    database (``*_db``, with the block it came from), which is the feasibility limit."""
    lim = limits()
    blocks = lim["eras"]
    key = min(blocks, key=lambda k: abs(int(k) - int(shot)) if blocks[k]["n_shots"] else 1e12)
    era = json.loads(json.dumps(blocks[key]))
    db = database_limits()
    for name, lb in era["beams"].items():
        lb.update(db["beams"][name])
    for group in ("ech_power", "gas_flow", "rmp", "i_coil"):
        for name, g in era[group].items():
            g.update(db[group][name])
    return {"era_start": int(key), "totals_db": db["totals"], **era}


_DB: Optional[dict] = None
# A database limit is the value reached in at least this many shots, so that one shot with
# a corrupt signal cannot set it.
DB_MIN_SHOTS = 3


def _nth(tops) -> dict:
    """Value reached in at least :data:`DB_MIN_SHOTS` shots, from per-block top lists."""
    pairs = sorted((tuple(x) for t in tops for x in t), reverse=True)
    if not pairs:
        return {"max_db": 0.0, "max_db_shots": []}
    n = min(DB_MIN_SHOTS, len(pairs))
    return {"max_db": pairs[n - 1][0], "max_db_shots": [s for _, s in pairs[:n]]}


def database_limits() -> dict:
    """Per channel, the most it delivered in at least :data:`DB_MIN_SHOTS` tokenized shots
    (frame means, each shot's 99th percentile; ``limits()["method"]``) and those shots;
    the same for the per-shot totals of the precedent series (beam power, counter-current
    beam power, ECH power, |net beam torque|)."""
    global _DB
    if _DB is not None:
        return _DB
    blocks = [b for b in limits()["eras"].values() if b["n_shots"]]
    out: dict = {"beams": {}, "ech_power": {}, "gas_flow": {}, "rmp": {}, "i_coil": {}}
    for name in BEAMS:
        d = _nth(b["beams"][name]["power_MW_top"] for b in blocks)
        kv = sorted((b["beams"][name]["kv_max"] or 0 for b in blocks), reverse=True)
        out["beams"][name] = {"power_MW_db": d["max_db"], "power_MW_db_shots": d["max_db_shots"],
                              "kv_db": kv[0] or None}
    for group in ("ech_power", "gas_flow", "rmp", "i_coil"):
        for name in CHANNEL_NAMES[group]:
            out[group][name] = _nth(b[group][name]["top"] for b in blocks)
    z = np.load(INDEX_DIR / "shots.npz")
    off, shot = z["offsets"], z["shot"]
    totals = {}
    for series in ("pnbi_MW", "counter_MW", "ech_MW", "torque_Nm"):
        x = np.abs(z[f"series__{series}"].astype(np.float64))
        per = np.full(len(shot), np.nan)
        for i in range(len(shot)):
            v = x[off[i]:off[i + 1]]
            v = v[np.isfinite(v)]
            if len(v):
                per[i] = np.percentile(v, 99)
        order = np.argsort(-np.nan_to_num(per, nan=-np.inf))[:DB_MIN_SHOTS]
        totals[series] = {"max_db": round(float(per[order[-1]]), 3),
                          "max_db_shots": [int(shot[i]) for i in order]}
    out["totals"] = totals
    _DB = out
    return out


def knowledge() -> str:
    """How the agent works with the tools (``KNOWLEDGE.md``); no machine facts."""
    return (Path(__file__).with_name("KNOWLEDGE.md")).read_text()
