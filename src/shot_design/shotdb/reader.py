"""The raw-signal layer as an interface, so that the rest of ideate names a contract rather than
a file layout.

There are two raw layouts, and they are not variants of each other. The d3d_fusion_data layout
(`legacy_raw.py`) is a pandas HDFStore of named columns on a millisecond index, in two locations
with a precedence rule. The FAITH corpus layout (`corpus.py`) is one `<shot>_processed.h5` per
shot holding unnamed channel arrays on a SECONDS axis, with absent diagnostics written as `(C, 1)`
placeholders and 2.3 % of the files unreadable. What they have in common -- and all they have in
common -- is that you can ask them for a shot's groups and for a group's samples on a time axis.

So the contract is split in two, deliberately:

* `Reader` is that common file layer, in MILLISECONDS, which is the unit every other module of
  ideate (features, segments, retrieval, the schema) already speaks. Both readers implement it.
* `SignalReader` adds the spec-level half -- `read_shot`, `read_signal`, `signal_status` -- which
  is what `build`, `cli` and `retrieval.actuation` call today. These take a `SignalSpec`: a
  registry entry naming a group, a column, a scale, a null sentinel. That vocabulary is the
  legacy layout's, and mapping it onto the corpus's unnamed channels is a design decision with
  real content (which corpus group is the beam power, and what a "total" means over its channels)
  that belongs to the corpus build, not here. `corpus.CorpusReader` therefore implements `Reader`
  only, and `corpus_signals.CorpusSignalReader` -- which is the one that reads `actuators.yaml`'s
  `corpus:` block and labelmaker's feature store -- implements `SignalReader` on top of it.

Both protocols are `runtime_checkable`, which for a Protocol means `isinstance` checks method
NAMES only -- not signatures. That is enough for what it is used for here: an assertion in the
test suite that an adapter did not silently lose a method, and a readable answer to "how far
along is this reader", never a dispatch decision.

Two exceptions cross the boundary, because both layouts have both failures and callers have to
tell them apart:

* `Unavailable` -- the file is fine, it just does not carry that group or signal for this shot.
  A coverage fact about DIII-D, to be recorded, not an error to be logged.
* `ShotFailed` -- the shot's file could not be read at all. On the corpus that is ~2.3 % of
  files (truncated writes), and it must never be reported as "the diagnostic was not recorded".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

import numpy as np

from ..config import SignalSpec
from ..schema import Status


class Unavailable(Exception):
    """This shot's raw file does not carry that group/signal -- a coverage fact, not a fault."""


class ShotFailed(Exception):
    """This shot's raw file could not be read at all. Chains the OSError that said so whenever
    there was one -- which is every case but a file whose datasets contradict each other."""


@dataclass
class Signal:
    """One registry signal of one shot: values on a millisecond axis, with the provenance the
    caller needs to interpret them.

    Defined here rather than in `legacy_raw` because it is the currency of `SignalReader`, which
    both raw layouts will eventually speak; `legacy_raw.Signal` re-exports this same class, so
    `isinstance` and pickling are unaffected by where it is written down.
    """

    t_ms: np.ndarray
    y: np.ndarray
    units: str | None
    # Which layer produced the samples. "staged"/"fetched" are the two d3d_fusion_data locations
    # (who WROTE the file, per `legacy_raw.is_ours`, not where it was found); "corpus" is a FAITH
    # <shot>_processed.h5 group and "labelmaker" a canonical feature of <shot>_features.h5.
    source: Literal["staged", "fetched", "corpus", "labelmaker"]
    group: str
    col: str
    # Finer provenance, when the layer has more than one source of its own: labelmaker resolves a
    # feature per shot from the archive, the corpus or fdp and records which, and those three do
    # not agree to better than a few percent (see labelmaker.features.namespace). None on the
    # legacy layout, where `source` is already the whole answer.
    resolver: str | None = None


@runtime_checkable
class Reader(Protocol):
    """A shot's raw file, whatever layout it is in. Time is milliseconds, always."""

    def path(self, shot: int) -> Path:
        """Where this shot's file is (or would be), whether or not it exists."""

    def available(self, shot: int) -> bool:
        """Is there a file for this shot that actually opens?

        Openable, not merely present: on the corpus 2.3 % of the files exist and do not open, and
        a caller asking "do we have this shot" is not helped by an answer that changes to an
        exception on the next line.
        """

    def groups(self, shot: int) -> list[str]:
        """The groups this shot actually recorded, sorted. Placeholders for a diagnostic that did
        not record are not groups this shot has, and are not listed.

        This is a coverage answer, not a promise that `read` will serve every name in it. On the
        corpus the video groups (`bolo`, `irtv`, `tangtv`) are listed here when the shot recorded
        them -- they are diagnostics that were looking, and a census that hid them would be
        wrong -- but `read` refuses them with `ValueError`, because their samples are frames and
        `(C, n)` is not their shape. A caller iterating `groups()` and reading each one must
        either filter on the census's `kind` column or expect that `ValueError`.
        """

    def read(
        self, shot: int, group: str, channels: Sequence[int] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """`group` as `(t_ms float64 (n,), y float32 (C, n))`, C in the order asked for.

        `channels` are indices into the group's own channel order; None means all of them.
        Raises `Unavailable` if this shot did not record the group, `ShotFailed` if its file
        cannot be read.
        """

    def coverage(self, shot: int, group: str) -> tuple[float, float]:
        """`(t0_ms, t1_ms)`: the span the group's time axis covers, from the time axis alone.

        This is the span of the DIAGNOSTIC, not of the samples `read` returns -- the two differ
        by the trailing pad sample on the corpus's fast groups -- and it is what an event needs
        in order to say "no ELMs here" rather than "no data here".
        """


@runtime_checkable
class SignalReader(Reader, Protocol):
    """A `Reader` that also answers in the vocabulary of the signal registry (`SignalSpec`).

    This is the half `shotdb.build` reads a shot through, so a build is retargeted at another raw
    layout by passing a different reader, not by editing an import.
    """

    def read_shot(
        self, shot: int, specs: list[SignalSpec]
    ) -> tuple[dict[str, Signal | None], dict[str, Status]]:
        """Every spec's signal (None when this shot has none) and every spec's coverage status,
        in one pass over the shot's files. The two cannot disagree: presence is the read."""

    def read_signal(self, shot: int, spec: SignalSpec) -> Signal | None:
        """One spec's signal, or None when this shot does not carry it."""

    def signal_status(self, shot: int, spec: SignalSpec) -> Status:
        """One spec's coverage status: present, unavailable, pending or not_installed."""
