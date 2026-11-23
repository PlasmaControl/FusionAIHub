"""Settings, ladders and selection rules of the fixed tokenizers: pure functions of numbers.

A `Setting` names one codec and its size for one signal. Each signal has a ladder of settings in
rising size (`rungs`): the cosine transform keeps 1 to 16 values per frame and channel, the principal
components of a profile or map keep 0 to 96 coefficients, and a spectrogram keeps a number of cosine
values per band. The rules of "Which number decides what" choose a rung from the floors (the NRMSE a
perfect predictor would score through the codec, in TokaMark's units, lower is better): the smallest
rung whose floor is at or under a target (`choose_rung`), or within a tolerance of a reference rung
(`choose_within`). The targets are a fifth of the best published score of the tasks that predict the
signal. Nothing here reads data or fits anything; `tokamark` is needed only by `predicted_by` and
`targets`, which read the task specs.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from tokamak_foundation_model.ignite.mast import registry, tasks
from tokamak_foundation_model.ignite.mast.tasks import SignalSpec
from tokamak_foundation_model.ignite.mast.tokenizers import frames

DCT_RUNGS = (1, 2, 3, 4, 6, 8, 12, 16)                  # cosine values kept per frame and channel
PCA_RUNGS = (0, 8, 16, 24, 32, 48, 64, 96)              # principal-component coefficients kept
BAND_RUNGS = (1, 2, 4, 8, 16, 32)                       # cosine values kept per band (all bands alike)
LOW_BAND_RUNGS = (4, 8, 16, 32)                         # values kept in the lowest band, first step
VALUES_PER_TOKEN = {"profile": 4, "map": 2}             # coefficients per token, by family
INPUT_ONLY_TARGET = 0.0176                              # NRMSE target of a signal no task predicts
N_BINS = 64                                             # bins of every value's grid
N_BANDS = 8                                             # bands of a spectrogram (256 bins of 32)
BAND_WHOLE = 32                                         # values of a band kept whole
TARGET_FRACTION = 0.2                                   # target = this times the best published NRMSE
CODECS = ("frame_dct", "linear", "band_dct")


@dataclass(frozen=True)
class Setting:
    """One codec of one signal.

    `codec` is ``"frame_dct"``, ``"linear"`` or ``"band_dct"``. `k` is ``(K,)`` for `frame_dct` (values
    kept per frame and channel), ``(n_values, values_per_token)`` for `linear` (principal-component
    coefficients kept, and how many go in one token) and one K per band for `band_dct`. `domain` is the
    unit the values are in (``"linear"``, ``"compressed"``, ``"signal"``; ``"band"`` for `band_dct`).
    `n_bins` is the number of bins of each value's grid. `source` names a fitted basis file or a
    checkpoint of a later plan; it is empty for the three codecs of this one, and it is part of the
    JSON that `assignment_sha256` hashes.
    """

    codec: str
    k: tuple[int, ...]
    domain: str = "linear"
    n_bins: int = N_BINS
    source: str = ""


# ---------------------------------------------------------------------------------------- ladders

def codec_kind(spec: SignalSpec) -> str:
    """``"linear"`` for the profile and map families, ``"band_dct"`` for the spectrograms, else ``"frame_dct"``."""
    if spec.name in tasks.SPECTROGRAM_SIGNALS:
        return "band_dct"
    if registry.family_of(spec) in VALUES_PER_TOKEN:
        return "linear"
    return "frame_dct"


def domains_of_kind(kind: str, compress: str) -> tuple[str, ...]:
    """The domains of a codec `kind` (``"frame_dct"``, ``"linear"`` or ``"band_dct"``); see `domains`."""
    if kind == "band_dct":
        return ("band",)
    if kind not in CODECS:
        raise ValueError(f"unknown codec {kind!r}; one of {CODECS}")
    base = ("signal", "linear") if kind == "linear" else ("linear",)
    return base + (("compressed",) if compress == "asinh" else ())


def domains(spec: SignalSpec, compress: str) -> tuple[str, ...]:
    """The domains a signal is fitted and tried in; `compress` is its `ChannelStats.compress`.

    `frame_dct`: ``("linear",)``, plus ``"compressed"`` for an asinh signal. `linear`: ``("signal",
    "linear")``, plus ``"compressed"`` for asinh. `band_dct`: ``("band",)``.
    """
    return domains_of_kind(codec_kind(spec), compress)


def band_rungs(low: int | None) -> tuple[Setting, ...]:
    """The two-step ladder of a spectrogram (rule 5), eight bands, in rising size.

    ``low=None``: the settings ``(k0, 32, 32, 32, 32, 32, 32, 32)`` for k0 in `LOW_BAND_RUNGS` (the
    lowest band's size is chosen first, the others kept whole). ``low=k0``: ``(k0, k, k, k, k, k, k, k)``
    for k in `BAND_RUNGS` (the other bands' size is chosen second).
    """
    if low is None:
        ks = [(k0,) + (BAND_WHOLE,) * (N_BANDS - 1) for k0 in LOW_BAND_RUNGS]
    else:
        ks = [(low,) + (k,) * (N_BANDS - 1) for k in BAND_RUNGS]
    return tuple(Setting("band_dct", k, "band") for k in ks)


def rungs(spec: SignalSpec, domain: str) -> tuple[Setting, ...]:
    """The ladder of a signal in `domain`, in rising size.

    `frame_dct`: one setting per K of `DCT_RUNGS` with ``K <= frame_len(spec.dt)``. `linear`: one per n of
    `PCA_RUNGS` with ``n <= prod(values_shape)`` and the family's values per token. `band_dct`:
    ``band_rungs(None)``.
    """
    kind = codec_kind(spec)
    if kind == "frame_dct":
        n = frames.frame_len(spec.dt)
        return tuple(Setting("frame_dct", (k,), domain) for k in DCT_RUNGS if k <= n)
    if kind == "linear":
        d = math.prod(spec.values_shape)
        vpt = VALUES_PER_TOKEN[registry.family_of(spec)]
        return tuple(Setting("linear", (n, vpt), domain) for n in PCA_RUNGS if n <= d)
    return band_rungs(None)


# ------------------------------------------------------------------------------- selection rules

def choose_rung(floors: Sequence[float], target: float) -> tuple[int, bool]:
    """The index of the first floor at or under `target`, and True; else the last index and False.

    `floors` are NRMSE in TokaMark's units, one per rung in rising size; `target` likewise. NaN (a
    rung that was not scored) never meets a target.
    """
    if not len(floors):
        raise ValueError("floors is empty")
    for i, f in enumerate(floors):
        if f <= target:
            return i, True
    return len(floors) - 1, False


def choose_within(floors: Sequence[float], reference: float, tolerance: float = 0.02) -> int:
    """The index of the first floor at or under ``reference * (1 + tolerance)``; the last if none.

    `floors` and `reference` are NRMSE in TokaMark's units; `tolerance` is a fraction. NaN never meets it.
    """
    if not len(floors):
        raise ValueError("floors is empty")
    limit = reference * (1.0 + tolerance)
    for i, f in enumerate(floors):
        if f <= limit:
            return i
    return len(floors) - 1


# ----------------------------------------------------------------------------------- assignments

def reference_assignment(signals: Sequence[SignalSpec]) -> dict[str, Setting]:
    """The "own mean" reference of every signal: one cosine value, no component, one value per band."""
    out: dict[str, Setting] = {}
    for s in signals:
        kind = codec_kind(s)
        if kind == "frame_dct":
            out[s.name] = Setting("frame_dct", (1,))
        elif kind == "linear":
            out[s.name] = Setting("linear", (0, VALUES_PER_TOKEN[registry.family_of(s)]), "signal")
        else:
            out[s.name] = Setting("band_dct", (1,) * N_BANDS, "band")
    return out


def uniform_assignment(signals: Sequence[SignalSpec], compress: Mapping[str, str],
                       rung: int = -1) -> dict[str, Setting]:
    """Every signal at `rung` of the ladder of its first domain.

    `rung` is a Python index (``-1`` the largest), limited to the ladder's length: a signal whose
    ladder is shorter gives its last rung. `compress` maps a signal name to its ``"none"`` or
    ``"asinh"``.
    """
    out: dict[str, Setting] = {}
    for s in signals:
        ladder = rungs(s, domains(s, compress[s.name])[0])
        out[s.name] = ladder[max(-len(ladder), min(rung, len(ladder) - 1))]
    return out


# ------------------------------------------------------------------------------------- targets

@functools.lru_cache(maxsize=1)
def _outputs() -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    """The 39 signal names (sorted) and, for each, the tasks that predict it, in `ALL_TASKS` order."""
    names = tuple(s.name for s in registry.all_signals())
    by: dict[str, list[str]] = {n: [] for n in names}
    for t in tasks.ALL_TASKS:
        for s in tasks.load_task(t).outputs:
            by[s.name].append(t)
    return names, {n: tuple(v) for n, v in by.items()}


def predicted_by() -> dict[str, tuple[str, ...]]:
    """Signal name to the tasks that predict it (it is an output of them), for the 39 signals. Needs `tokamark`."""
    return dict(_outputs()[1])


def targets() -> dict[str, float | None]:
    """The NRMSE target of each of the 39 signals, in TokaMark's units. Needs `tokamark`.

    0.2 times the smallest `best_published` over the tasks that predict the signal;
    `INPUT_ONLY_TARGET` for a signal no task predicts; None for the spectrograms (scored in a
    band domain with no published score of their own).
    """
    out: dict[str, float | None] = {}
    for name, ts in _outputs()[1].items():
        if name in tasks.SPECTROGRAM_SIGNALS:
            out[name] = None
        elif ts:
            out[name] = TARGET_FRACTION * min(tasks.best_published(t) for t in ts)
        else:
            out[name] = INPUT_ONLY_TARGET
    return out


# ----------------------------------------------------------------------------------------- files

def _settings_json(a: Mapping[str, Setting]) -> dict[str, dict]:
    return {name: {**asdict(s), "k": list(s.k)} for name, s in sorted(a.items())}


def assignment_sha256(a: Mapping[str, Setting]) -> str:
    """sha256 of the JSON of the settings with sorted keys (so of every field of every `Setting`)."""
    text = json.dumps(_settings_json(a), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def save_assignment(a: Mapping[str, Setting], path: Path,
                    notes: Mapping[str, str] | None = None) -> None:
    """Write ``{"settings", "notes", "sha256"}`` as JSON to `path` (replaced atomically)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"settings": _settings_json(a), "notes": dict(notes or {}), "sha256": assignment_sha256(a)}
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def load_assignment(path: Path) -> dict[str, Setting]:
    """Read a file of `save_assignment`; `ValueError` if its sha256 is not that of its settings."""
    doc = json.loads(Path(path).read_text())
    a = {name: Setting(d["codec"], tuple(d["k"]), d["domain"], d["n_bins"], d["source"])
         for name, d in doc["settings"].items()}
    if assignment_sha256(a) != doc["sha256"]:
        raise ValueError(f"{path}: the stored sha256 {doc['sha256']} is not that of its settings "
                         f"({assignment_sha256(a)}); the file was edited after it was written")
    return a
