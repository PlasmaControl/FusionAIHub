"""The signal registry resolved against the FAITH corpus and labelmaker's feature store.

`corpus.CorpusReader` answers for a NAMED GROUP: give it `pinj` and it gives you eight unnamed
channels on a millisecond axis. `build` does not speak that language -- it speaks `SignalSpec`,
the registry's vocabulary of named quantities (`pnbi_15L`, `dalpha`, `betan`) -- and the mapping
between the two is not a rename. It is three separate claims about someone else's data, and each
of them is a silent, plausible-looking error when it is wrong:

* **which group.** `actuators.yaml`'s `corpus:` block, established in Task I3 by comparing
  waveforms, because a corpus group has no units, no attributes and no channel names.
* **which channel.** The same block's `members:` list, in the CORPUS's channel order, which is
  not the registry's: the corpus orders its twelve gyrotrons alphabetically. A positional
  assumption reads LEIA's power off BORIS's channel and reports a plausible number.
* **which reduction.** Eight beam powers add up; eight accelerating voltages average; a
  line-integrated density is one chord and not a total at all. `reduce: sum|mean|first` is
  per address, never a default applied to whatever a group happens to hold.

The second source is labelmaker's `$LABELMAKER_ROOT/features/<shot>_features.h5`, which is where
the EFIT and plasma scalars live for corpus shots (the corpus itself carries no EFIT). Those are
read through `labelmaker.features.store`, not through h5py here, so that the resolver provenance
(`archive` / `corpus` / `fdp`, which differ by a few percent and by a 25 ms row lag -- see
`labelmaker.features.namespace`) and the recorded miss causes are the ones labelmaker itself
records rather than this module's reading of them.

**The status rules, which are the point of the module.** A build over 500 shots produces a
coverage table that people plan work from, so the difference between "DIII-D did not record this"
and "we have not fetched it yet" has to survive all the way into that table:

| what happened | status |
|---|---|
| the group / feature is there | `present` |
| the corpus group is absent or is a `(C, 1)` placeholder | `unavailable` |
| the signal has no corpus and no labelmaker address at all | `unavailable` |
| the stored feature holds no finite sample | `unavailable` |
| labelmaker recorded a miss that will fail again identically | `unavailable` |
| no feature file, no record of a miss, or a miss worth retrying | `pending` |
| the member is not installed on this shot | `not_installed` |

Never 0, and never NaN standing in for either. `pending` is a work order -- the shots it names
are exactly the ones `labelmaker.run features` (under the `fdp run` wrapper) can still fill in --
so it must not be spent on a quantity no source has.

A corpus file that does not open at all (~2.3 % of them, truncated writes) raises `ShotFailed`
and fails the whole shot rather than being reported as a shot with no diagnostics: the build
records it in `manifest.failed`, where it reads as the file-level fault it is.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import CorpusActuator, Paths, SignalSpec, corpus_actuators
from ..schema import Status
from .corpus import CorpusReader
from .reader import ShotFailed, Signal, Unavailable

#: The corpus is float32 on disk and so is `Signal.y` everywhere else in ideate; the arithmetic in
#: between (a sum over eight beams, a mean over eight filterscope views) is float64, because
#: D-alpha runs at ~1e15-1e18 and a float32 sum of squares of that overflows -- the same reason
#: `features._finite` casts.
DTYPE = np.float32


def default_features_dir(root: str | os.PathLike | None = None) -> Path | None:
    """`$LABELMAKER_ROOT/features`, or None when the variable is not set.

    None rather than labelmaker's own default root: a build that silently read another user's
    feature store would report coverage this repo's environment cannot reproduce. With no root
    set, every labelmaker-addressed signal is `pending`, which is true and is a work order.
    """
    root = root or os.environ.get("LABELMAKER_ROOT")
    return Path(root) / "features" if root else None


@dataclass(frozen=True)
class Address:
    """One spec's corpus address, with the actuator block and the spec's own block merged."""

    group: str
    channels: list[int] | None  # None is every channel of the group
    reduce: str
    scale: float = 1.0
    units: str | None = None


class CorpusSignalReader(CorpusReader):
    """The FAITH corpus + labelmaker features as a `SignalReader`.

    Constructed from `Paths` (not from two directories) so that a worker process can rebuild one
    from a picklable argument: `build._build_many` sends `reader_kind` and `paths` down to each
    worker and constructs the reader there, because an open HDF5 handle must never be forked.

    Holds no file handles between calls, exactly like `CorpusReader`. `read_shot` is the call the
    build makes, and it opens each corpus group ONCE however many specs share it -- eight beams
    share `pinj`, eighteen coils share `i_coil` -- which is the difference between one read of a
    40 MB group and eighteen.
    """

    kind = "corpus"

    def __init__(self, paths: Paths, features_dir: Path | None = None):
        super().__init__(Path(paths.foundation_model_processed_dir))
        self.paths = paths
        self.features_dir = Path(features_dir) if features_dir else default_features_dir()

    def __repr__(self) -> str:
        return f"CorpusSignalReader({str(self.corpus_dir)!r}, features_dir={self.features_dir!r})"

    # ------------------------------------------------------------------------ the spec level

    def read_shot(
        self, shot: int, specs: list[SignalSpec]
    ) -> tuple[dict[str, Signal | None], dict[str, Status]]:
        """Every spec's signal and every spec's status, in one pass over this shot's two files.

        Same shape as `legacy_raw.read_shot`'s answer, deliberately: `build_record` reads a shot
        through whichever reader it was handed and has no branch on which one it was.
        """
        signals: dict[str, Signal | None] = {s.name: None for s in specs}
        # Everything starts unavailable -- "no source carries this for this shot" -- and only a
        # labelmaker miss that is worth retrying moves a name to `pending`. The default is the
        # conservative one: `pending` is a promise that a fetch would help.
        status: dict[str, Status] = {}
        installed = [s for s in specs if s.installed]

        by_group: dict[str, list[tuple[SignalSpec, Address]]] = {}
        labelmaker: list[SignalSpec] = []
        for spec in installed:
            addr = self.address(spec)
            if addr is not None:
                by_group.setdefault(addr.group, []).append((spec, addr))
            elif spec.labelmaker is not None:
                labelmaker.append(spec)

        for group, items in by_group.items():
            for name, sig in self._read_group(shot, group, items).items():
                signals[name] = sig
        for name, sig, st in self._read_features(shot, labelmaker):
            signals[name] = sig
            if sig is None:
                status[name] = st

        coverage: dict[str, Status] = {}
        for spec in specs:
            if not spec.installed:
                coverage[spec.name] = "not_installed"
            elif signals[spec.name] is not None:
                coverage[spec.name] = "present"
            else:
                coverage[spec.name] = status.get(spec.name, "unavailable")
        return signals, coverage

    def read_signal(self, shot: int, spec: SignalSpec) -> Signal | None:
        return self.read_shot(shot, [spec])[0][spec.name]

    def signal_status(self, shot: int, spec: SignalSpec) -> Status:
        return self.read_shot(shot, [spec])[1][spec.name]

    # ------------------------------------------------------------------------ addressing

    def address(self, spec: SignalSpec) -> Address | None:
        """This spec's corpus address, or None when the corpus does not carry it.

        Two ways in, and both end in `actuators.yaml`'s `corpus:` block rather than duplicating
        it: an explicit `corpus:` block on the signal (which may name an `actuator:` instead of a
        group), and a per-member actuator spec, whose system names one of those entries.
        """
        actuators = corpus_actuators()
        block = spec.corpus
        if block is not None:
            act = actuators.get(block.actuator) if block.actuator else None
            if block.actuator and act is None:
                raise KeyError(f"{spec.name}: no corpus actuator {block.actuator!r}")
            group = block.group or (act.group if act else None)
            if group is None:
                raise ValueError(f"{spec.name}: corpus address names neither group nor actuator")
            channels = block.channels if block.channels is not None else (act.channels if act else None)
            how = block.reduce or (act.reduce if act else "sum")
            return Address(group, channels, how, block.scale, block.units or spec.units)
        if spec.system and spec.member:
            act = self._actuator_for_system(actuators, spec.system)
            if act is None:
                return None
            channel = act.channel_of(spec.member)
            # A member with no channel in this group is a coverage fact about the corpus (it
            # carries twelve gyrotrons and the registry knows of twelve, but the sets could
            # diverge), not a configuration error -- so it reads `unavailable`, not a raise.
            return None if channel is None else Address(act.group, [channel], "first", 1.0, spec.units)
        return None

    @staticmethod
    def _actuator_for_system(
        actuators: dict[str, CorpusActuator], system: str
    ) -> CorpusActuator | None:
        return next((a for a in actuators.values() if a.system == system), None)

    # ------------------------------------------------------------------------ the corpus

    def _read_group(
        self, shot: int, group: str, items: list[tuple[SignalSpec, Address]]
    ) -> dict[str, Signal | None]:
        """One read of `group` serving every spec addressed to it.

        The union of the wanted channels is read once and each spec is served by slicing that
        selection, so `pinj`'s eight beams cost one open and one read, not eight of each.
        `Unavailable` (the group is absent, or is the corpus's `(C, 1)` placeholder) leaves every
        spec of the group unread; `ShotFailed` propagates, because a file that will not open is
        not a diagnostic that was not recorded.
        """
        wants_all = any(a.channels is None for _, a in items)
        wanted: list[int] | None = None
        if not wants_all:
            wanted = sorted({c for _, a in items for c in (a.channels or [])})
        try:
            t_ms, vals = self.read(shot, group, wanted)
        except Unavailable:
            return {}
        except ValueError:
            # A video group addressed as a waveform: a configuration error, not a coverage fact.
            raise
        out: dict[str, Signal | None] = {}
        for spec, addr in items:
            if addr.channels is None:
                rows = vals
            else:
                index = {c: i for i, c in enumerate(wanted or [])}
                take = [index[c] for c in addr.channels if c in index]
                if len(take) != len(addr.channels):
                    out[spec.name] = None
                    continue
                rows = vals[take]
            y = _reduce(_prepare(rows, spec, addr.scale), addr.reduce)
            if not np.isfinite(y).any():
                # Every channel of this address recorded nothing. The group is there, the
                # diagnostic was not: `unavailable`, which is what a None here becomes.
                out[spec.name] = None
                continue
            out[spec.name] = Signal(
                t_ms, y.astype(DTYPE), addr.units, "corpus", group, spec.col or spec.name, "corpus"
            )
        return out

    # ------------------------------------------------------------------------ labelmaker

    def _read_features(
        self, shot: int, specs: list[SignalSpec]
    ) -> list[tuple[str, Signal | None, Status]]:
        """Every labelmaker-addressed spec, with the status a miss earns.

        The file is interrogated twice (what it holds, what it recorded as missed) and then read
        once per stored feature, all through `labelmaker.features.store`: the miss causes and the
        transient/permanent rule are labelmaker's own, and a copy of them here would be a second
        opinion about someone else's file.
        """
        if not specs:
            return []
        from labelmaker.features import store

        path = self.features_dir / f"{int(shot)}_features.h5" if self.features_dir else None
        if path is None or not path.exists():
            return [(s.name, None, "pending") for s in specs]
        try:
            present, missing = store.present(path), store.missing_names(path)
        except OSError:
            # The file is there but unreadable right now (a features job may be rewriting it --
            # `write_features` renames into place, so this is a narrow window). Retryable, which
            # is what `pending` means.
            return [(s.name, None, "pending") for s in specs]
        out: list[tuple[str, Signal | None, Status]] = []
        for spec in specs:
            feature = spec.labelmaker.feature
            if feature not in present:
                cause = missing.get(feature)
                retry = cause is None or store.is_transient(cause)
                out.append((spec.name, None, "pending" if retry else "unavailable"))
                continue
            try:
                arr = store.read_feature(path, feature)
            except (OSError, KeyError):
                out.append((spec.name, None, "pending"))
                continue
            y = _reduce(_prepare(np.atleast_2d(arr.y), spec, spec.labelmaker.scale),
                        spec.labelmaker.reduce)
            if not np.isfinite(y).any():
                out.append((spec.name, None, "unavailable"))
                continue
            resolver = str(arr.attrs.get("resolver", "unknown"))
            sig = Signal(
                np.asarray(arr.x, dtype=np.float64).ravel() * 1000.0,  # seconds -> ms
                y.astype(DTYPE),
                spec.labelmaker.units or arr.attrs.get("units") or spec.units,
                "labelmaker",
                feature,
                spec.col or spec.name,
                f"labelmaker:{resolver}",
            )
            out.append((spec.name, sig, "present"))
        return out


# ---------------------------------------------------------------------------------- arithmetic


def _prepare(rows: np.ndarray, spec: SignalSpec, scale: float) -> np.ndarray:
    """`rows` as float64 with the spec's per-sample conventions applied, before any reduction.

    Order matters and is the same as `legacy_raw._signal`'s, with one deliberate difference:
    `scale` comes from the ADDRESS, never from `spec.scale`. A registry `scale` corrects a
    d3d_fusion_data storage convention (`ipsip` is stored in megaamps in a column declared amps);
    applying it to labelmaker's `ip`, which is already amps, would report a 1.2 MA shot as 1.2 TA.

    `abs` is the spec's, because it is a claim about the QUANTITY (Ip's sign is a machine
    convention, an RMP coil's is a phase), and it runs per channel BEFORE the reduction -- which
    is what makes the RMP total a sum of magnitudes rather than the ~0 a signed sum of an n = 3
    coil pattern gives.
    """
    y = np.asarray(rows, dtype=np.float64)
    if spec.null_value is not None:
        nulls = spec.null_value if isinstance(spec.null_value, list) else [spec.null_value]
        for nv in nulls:
            y = np.where(y == nv, np.nan, y)
    if scale != 1.0:
        y = y * scale
    return np.abs(y) if spec.abs else y


def _reduce(rows: np.ndarray, how: str) -> np.ndarray:
    """`(C, n)` channels down to one `(n,)` series.

    `sum` and `mean` count only the channels that actually recorded a sample at each instant --
    the same rule `features.system_totals` applies to the legacy layout, and for the same reason:
    an unrecorded member contributes nothing and must not turn the whole total NaN, while a
    sample no channel recorded is NaN and not a confident zero.
    """
    if rows.ndim == 1:
        rows = rows[None, :]
    if how in ("first", "core"):
        return np.array(rows[0], dtype=np.float64)
    if how == "edge":
        return np.array(rows[-1], dtype=np.float64)
    if how == "peak":
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "All-NaN slice encountered", RuntimeWarning)
            return np.nanmax(rows, axis=0)
    if how not in ("sum", "mean"):
        raise ValueError(f"unknown reduction {how!r}")
    recorded = np.isfinite(rows)
    n = recorded.sum(axis=0)
    total = np.where(recorded, rows, 0.0).sum(axis=0)
    out = total if how == "sum" else total / np.maximum(n, 1)
    out[n == 0] = np.nan
    return out


__all__ = ["Address", "CorpusSignalReader", "ShotFailed", "default_features_dir"]
