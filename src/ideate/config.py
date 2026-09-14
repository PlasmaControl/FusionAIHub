"""Configuration loading: paths, YAML configs, and the signal/actuator registry.

This is the only module that reads YAML. Everything per-campaign (actuator members, signal
node names, thresholds) lives in configs/ideate/ so adding a gyrotron is a config change.


Ported from shot-recommender-system (shotrec) @565d548.
"""

from __future__ import annotations

import copy
import os
import re
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict

CONFIG_DIR = Path(os.environ.get("IDEATE_CONFIG_DIR", Path(__file__).resolve().parents[2] / "configs" / "ideate"))
_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _interpolate(values: dict[str, Any]) -> dict[str, Any]:
    """Resolve ${key} against the mapping itself, then against the environment."""
    out = dict(values)
    for _ in range(10):  # chains like c -> b -> a resolve within a few passes
        changed = False
        for k, v in out.items():
            if isinstance(v, str) and "${" in v:
                new = _VAR.sub(
                    lambda m: str(out.get(m.group(1), os.environ.get(m.group(1), m.group(0)))), v
                )
                if new != v:
                    out[k], changed = new, True
        if not changed:
            break
    return out


class Paths(BaseModel):
    data_root: Path
    raw_dir: Path
    db_dir: Path
    text_cache_dir: Path
    ignite_inputs_dir: Path
    models_dir: Path
    eval_dir: Path
    sessions_dir: Path
    actuations_dir: Path
    llm_cache_dir: Path
    staged_raw_dir: Path
    text_root: Path
    logs_jsonl: Path
    shot_index_json: Path
    shotsummary_raw_dir: Path
    per_shot_txt_dir: Path
    qh_database_csv: Path
    foundation_model_processed_dir: Path
    fdp_project_dir: Path
    sentence_transformers_model: str = "sentence-transformers/all-MiniLM-L6-v2"


_PATHS_OVERRIDE: ContextVar[Paths | None] = ContextVar("ideate_paths", default=None)


@contextmanager
def using_paths(paths: Paths):
    """Scope paths to one request, including its worker thread, without changing the env."""
    token = _PATHS_OVERRIDE.set(paths)
    try:
        yield
    finally:
        _PATHS_OVERRIDE.reset(token)


def load_paths(path: Path | None = None) -> Paths:
    override = _PATHS_OVERRIDE.get()
    if path is None and override is not None:
        return override
    if os.environ.get("IDEATE_DATA_ROOT") == "":
        raise ValueError("IDEATE_DATA_ROOT is empty; set it to a data directory or unset it")
    p = path or Path(os.environ.get("IDEATE_PATHS", CONFIG_DIR / "paths.yaml"))
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if os.environ.get("IDEATE_DATA_ROOT"):
        raw["data_root"] = os.environ["IDEATE_DATA_ROOT"]
    return Paths(**_interpolate(raw))


def _load_yaml_file(name: str) -> dict[str, Any]:
    return yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8")) or {}


# path -> (size, mtime_ns, parsed document)
_YAML_CACHE: dict[str, tuple[int, int, dict[str, Any]]] = {}


def load_yaml(name: str) -> dict[str, Any]:
    """A parsed config from configs/ideate/, cached on (path, size, mtime_ns), returned as a
    deep copy.

    Measured on the 105-shot database: signals.yaml parses in 24 ms and actuators.yaml in 9 ms,
    and one `ideate query` read them dozens of times -- once per result for the unit lookup, twice
    per result for the flag rules. Keying on size and mtime means an edited config is picked up on
    the next call with no restart; the deep copy means a caller that mutates what it got back
    cannot poison the next caller's view.
    """
    path = CONFIG_DIR / name
    st = path.stat()
    hit = _YAML_CACHE.get(str(path))
    if hit is None or hit[0] != st.st_size or hit[1] != st.st_mtime_ns:
        hit = _YAML_CACHE[str(path)] = (st.st_size, st.st_mtime_ns, _load_yaml_file(name))
    return copy.deepcopy(hit[2])


class CorpusAddress(BaseModel):
    """Where a signal lives in the FAITH corpus: a group, some of its channels, one reduction.

    Either `actuator` (an entry of actuators.yaml's `corpus:` block, which already says which
    group and channels an actuator is) or `group` -- never both spelled out twice. `scale` is the
    CORPUS's storage convention and is deliberately not `SignalSpec.scale`, which corrects the
    d3d_fusion_data layout's own conventions and is wrong here by exactly that factor.
    """

    model_config = ConfigDict(extra="forbid")

    actuator: str | None = None
    group: str | None = None
    channels: list[int] | None = None  # None is "all of them"
    reduce: Literal["sum", "mean", "first"] | None = None
    scale: float = 1.0
    units: str | None = None


class LabelmakerAddress(BaseModel):
    """One canonical feature of `$LABELMAKER_ROOT/features/<shot>_features.h5`.

    `reduce` says how a stored `(C, T)` array becomes one series: a scalar feature is `(1, T)` and
    takes `first`; a profile is `(33 rho, T)` and takes `core` (rho = 0), `edge` (rho = 1) or
    `peak` (the largest of the 33 at each time).
    """

    model_config = ConfigDict(extra="forbid")

    feature: str
    reduce: Literal["first", "core", "edge", "peak", "mean", "sum"] = "first"
    scale: float = 1.0
    units: str | None = None


class SignalSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str  # DB column prefix, e.g. "ip", "pnbi_15L", "betan"
    # The d3d_fusion_data address. Empty for a quantity that layout does not carry at all (a
    # corpus- or labelmaker-only signal): `legacy_raw` then finds no such group and reports it
    # `unavailable`, which is what that layout's answer honestly is.
    group: str = ""
    col: str = ""
    # The other two raw layouts. Absent means "this signal has no address there", which is a
    # coverage fact and not a fetchable gap -- see `corpus_signals` for the status rules.
    corpus: CorpusAddress | None = None
    labelmaker: LabelmakerAddress | None = None
    units: str | None = None
    tier: str = "raw"  # "raw" | "derived"
    abs: bool = False
    scale: float = 1.0  # multiplied into values on read, so units declares the unit *after*
    # scaling (e.g. a column stored in MA with scale: 1.0e+6 and units: A)
    null_value: float | list[float] | None = None  # a raw sample exactly equal to this -- or, as
    # a list, to any one of these -- means "no value", not a measurement; becomes NaN on read.
    # The list form exists for a symmetric clamp (e.g. drsep's +-0.4 m saturation) that a single
    # scalar can't express; a plain float still works unchanged for a single sentinel like
    # zxpt1's -9.99.
    stats: list[str] = ["mean"]
    fetch: dict[str, Any] | None = None
    provenance: dict[str, Any] | None = None
    system: str | None = None  # actuator system name when this is a member
    member: str | None = None
    installed: bool = True
    pcs: str | None = None  # PCS waveform path, e.g. "gas/gasstandard/GASA_FlowRate"


class SystemSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    prefix: str
    units: str | None = None
    stats: list[str] = ["mean", "peak"]
    max: float | None = None
    members: list[str] = []


def campaign_for_shot(shot: int) -> str:
    for c in load_yaml("actuators.yaml").get("campaigns", []):
        lo, hi = c["shots"]
        if lo <= shot <= hi:
            return str(c["id"])
    return "unknown"


def _fmt(template: str, member: dict[str, Any]) -> str:
    mid = str(member["id"])
    code = str(member.get("code", mid.lower()))
    return template.format(id=mid, id_lower=mid.lower(), code=code, code_upper=code.upper())


def _installed(member: dict[str, Any], shot: int) -> bool:
    since, until = member.get("since"), member.get("until")
    return (since is None or shot >= since) and (until is None or shot <= until)


def expand_registry(shot: int, include_not_installed: bool = False) -> list[SignalSpec]:
    """All signals for a shot: fixed signals from signals.yaml + actuator members installed for it."""
    specs: list[SignalSpec] = []
    for name, s in load_yaml("signals.yaml")["signals"].items():
        specs.append(SignalSpec(name=name, **s))
    for sys_name, sysdef in load_yaml("actuators.yaml")["systems"].items():
        for m in sysdef["members"]:
            inst = _installed(m, shot)
            if not inst and not include_not_installed:
                continue
            fetch = None
            if sysdef.get("fetch"):
                fetch = {
                    k: (_fmt(v, m) if isinstance(v, str) else v) for k, v in sysdef["fetch"].items()
                }
            specs.append(
                SignalSpec(
                    name=f"{sysdef['prefix']}_{m['id']}",
                    group=sysdef["staged"]["group"],
                    col=_fmt(sysdef["staged"]["col"], m),
                    units=sysdef.get("units"),
                    tier="raw",
                    stats=list(sysdef.get("stats", ["mean", "peak"])),
                    fetch=fetch,
                    system=sys_name,
                    member=str(m["id"]),
                    installed=inst,
                    pcs=m.get("pcs"),
                )
            )
    return specs


class CorpusActuator(BaseModel):
    """Which FAITH-corpus group carries one actuator, which of its channels, and how they add up.

    `system` and `members` are what let a per-member registry spec (`pech_LEIA`) find its channel:
    `members` lists the registry member ids in the CORPUS's channel order, with None for a channel
    no registry member names. It is a list and not a positional assumption because the two orders
    genuinely differ -- the corpus orders its twelve gyrotrons alphabetically and actuators.yaml
    orders them by installation date.
    """

    model_config = ConfigDict(extra="forbid")

    name: str  # ideate's name for the actuator, e.g. "nbi_torque"
    group: str  # the corpus group, e.g. "tinj"
    channels: list[int] | None = None  # None is the config's "all"
    reduce: Literal["sum", "mean", "first"] = "sum"  # what a TOTAL over those channels means
    system: str | None = None  # the actuators.yaml `systems:` entry this group carries, if any
    members: list[str | None] | None = None  # registry member ids, in the corpus channel order

    def channel_of(self, member: str) -> int | None:
        """The corpus channel index carrying `member`, or None when this group does not."""
        if not self.members:
            return None
        for i, m in enumerate(self.members):
            if m is not None and str(m) == str(member):
                return i
        return None


def corpus_actuators() -> dict[str, CorpusActuator]:
    """The `corpus:` block of actuators.yaml: ideate's actuator names -> corpus groups.

    A mapping and not a lookup table in code, because it is a claim about someone else's data
    that we may have to correct: the corpus groups are unnamed channel arrays with no units and
    no attributes, so which group is "the beam power" was established by comparison, not by
    reading a label, and a correction should be a config edit.
    """
    out: dict[str, CorpusActuator] = {}
    for name, spec in (load_yaml("actuators.yaml").get("corpus") or {}).items():
        channels = spec.get("channels", "all")
        out[name] = CorpusActuator(
            name=name,
            group=spec["group"],
            channels=None if channels in (None, "all") else [int(c) for c in channels],
            reduce=spec.get("reduce", "sum"),
            system=spec.get("system"),
            members=spec.get("members"),
        )
    return out


def actuator_systems(shot: int) -> dict[str, SystemSpec]:
    out: dict[str, SystemSpec] = {}
    for sys_name, sysdef in load_yaml("actuators.yaml")["systems"].items():
        out[sys_name] = SystemSpec(
            name=sys_name,
            prefix=sysdef["prefix"],
            units=sysdef.get("units"),
            stats=list(sysdef.get("stats", ["mean", "peak"])),
            max=sysdef.get("max"),
            members=[str(m["id"]) for m in sysdef["members"] if _installed(m, shot)],
        )
    return out


def load_shot_list(name: str = "poc_v1", path: Path | None = None) -> list[int]:
    """Shots from configs/ideate/shot_lists/<name>.yaml with the hand_review block applied.

    hand_review is the human gate on a generated list: `drop` removes a shot the reviewer
    rejected, `add` pins one the greedy allocator missed. Drops are applied before adds, so a
    shot named in both ends up selected -- `add` is the reviewer's later, more specific word.
    """
    p = path or (CONFIG_DIR / "shot_lists" / f"{name}.yaml")
    # Explicit utf-8: the generated lists carry non-ASCII operator text (the "\u2026" that marks a
    # cut excerpt), and labels.yaml carries "alfven"/"alfv\u00e9n", so a POSIX/C locale default
    # would raise here rather than at import time.
    doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    shots = {int(e["shot"]) for e in doc.get("shots", [])}
    review = doc.get("hand_review") or {}
    shots -= {int(e["shot"]) for e in review.get("drop") or []}
    shots |= {int(e["shot"]) for e in review.get("add") or []}
    return sorted(shots)
