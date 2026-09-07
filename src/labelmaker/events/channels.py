"""Which corpus channels a shot's mask run uses, and over what time.

The U-Net is run on ONE channel at a time, and a shot's fast groups are
3.85 GB between them, so the first decision of a mask run is which handful
of channels to spend the GPU on. Round 1 is eleven specs: two magnetics,
three ECE, two CO2 interferometer chords, two BES, and two `mirnov`
channels that stand in for `mhr` when `mhr` is absent. They are a spread
over the machine rather than a set of duplicates - a mode that is on one
diagnostic and not on another is itself evidence - and they are few because
2,000 shots x 2 passes has to finish.

Two facts from the 300-shot availability sample (plan V2, V3) shape this:

* group availability is very uneven - `mhr` 86%, `mirnov` 98.6%, `co2` 39%,
  `bes` 33% - so a plan is per shot, not per campaign, and `mirnov` is
  registered as `mhr`'s fallback rather than as an extra channel: on the
  86% of shots that have `mhr` it would be a near-duplicate of it;
* the groups do not cover the same time - `mhr` only 4.19 s, and the window
  SLIDES per shot; `ece` 6.19 s; `mirnov` 16.4 s - so "no EHO on this shot"
  is meaningless without the interval the detector could see. `coverage`
  is what every event's `t_cov0_s`/`t_cov1_s` is filled from.

Everything here reads headers only: `plan_for` touches `shape` and nothing
else, and `coverage` reads one channel row plus `xdata`. A corpus group is
ABSENT when its `ydata` last axis is shorter than 2 - the corpus writes a
`(C, 1)` placeholder rather than omitting the group (see
`features/resolve_corpus.py`) - and so is a group that is missing, has no
`ydata`, or is not two-dimensional: none of them is a record this pipeline
can read.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import h5py
import numpy as np

#: What a channel is there to see. Short strings, because they end up in an
#: event row and in a mask key: `magnetics` (a Mirnov coil or the `mhr`
#: high-rate magnetics - the broadest view of any MHD mode), `ece_core`,
#: `ece_mid`, `ece_edge` (electron-temperature fluctuations at three radii -
#: which one a mode shows on is where it lives), `density` (a CO2
#: interferometer chord - line-integrated density fluctuation), `bes` (beam
#: emission, local density fluctuation where the beam crosses).
ROLES = (
    "magnetics", "ece_core", "ece_mid", "ece_edge", "density", "bes",
)

#: Every reason `plan_for` can give for skipping a spec. Closed vocabulary:
#: a consumer counting why round 1 lost channels should not have to parse
#: prose, and anything unreadable is reported as "group absent" rather than
#: as a new fourth reason.
REASONS = ("group absent", "channel out of range", "fallback not needed")

#: The corpus' absent-signal sentinel: `ydata.shape[-1] < 2`.
MIN_SAMPLES = 2


@dataclass(frozen=True)
class ChannelSpec:
    """One (diagnostic, channel) the mask run may process.

    `fallback_for` names a group this spec stands in for: a fallback spec is
    planned only on the shots where that group is absent, so it never
    doubles up with the channel it replaces.
    """

    diag: str
    channel: int
    role: str
    fallback_for: str = ""

    @property
    def key(self) -> str:
        """`"{diag}:{channel}"` - how a spec is named in `reasons`."""
        return f"{self.diag}:{self.channel}"


#: Round 1. `mhr` 0/4 are a quarter of the array apart; `ece` 8/20/40 span
#: core to edge; `co2` 0/2 are two chords; `bes` 26/28 are neighbours in the
#: middle of the grid. `mirnov` 0/8 replace `mhr` when it is absent.
ROUND1_PLAN: tuple[ChannelSpec, ...] = (
    ChannelSpec("mhr", 0, "magnetics"),
    ChannelSpec("mhr", 4, "magnetics"),
    ChannelSpec("ece", 8, "ece_core"),
    ChannelSpec("ece", 20, "ece_mid"),
    ChannelSpec("ece", 40, "ece_edge"),
    ChannelSpec("co2", 0, "density"),
    ChannelSpec("co2", 2, "density"),
    ChannelSpec("bes", 26, "bes"),
    ChannelSpec("bes", 28, "bes"),
    ChannelSpec("mirnov", 0, "magnetics", fallback_for="mhr"),
    ChannelSpec("mirnov", 8, "magnetics", fallback_for="mhr"),
)


def _shape_of(handle, diag: str) -> tuple[int, int] | None:
    """`(channels, samples)` of a group's `ydata`, or None if unusable.

    Header only: `h5py` knows a dataset's shape from the file's metadata, so
    this reads no data at all.
    """
    if diag not in handle:
        return None
    group = handle[diag]
    if "ydata" not in group:
        return None
    shape = group["ydata"].shape
    if len(shape) != 2 or shape[-1] < MIN_SAMPLES:
        return None
    return (int(shape[0]), int(shape[1]))


def plan_for(
    corpus_file,
    plan: Sequence[ChannelSpec] = ROUND1_PLAN,
) -> tuple[list[ChannelSpec], dict[str, str]]:
    """Which specs this shot can run, and why the others cannot.

    Returns `(specs, reasons)`; `specs` keeps `plan`'s order and `reasons`
    maps the key of every skipped spec to one of `REASONS`. Together they
    partition `plan`, so a run can record what it did NOT do as well as what
    it did.

    A fallback's target group is looked up in the FILE, not among the plan's
    own specs: a plan of nothing but the mirnov fallback - which is what a
    re-run of the one channel that failed looks like - would otherwise find
    no `mhr` beside it and promote a duplicate of the channel it exists to
    replace.
    """
    plan = tuple(plan)
    diags = {spec.diag for spec in plan}
    diags |= {spec.fallback_for for spec in plan if spec.fallback_for}
    with h5py.File(corpus_file, "r", locking=False) as f:
        shapes = {diag: _shape_of(f, diag) for diag in sorted(diags)}
    specs: list[ChannelSpec] = []
    reasons: dict[str, str] = {}
    for spec in plan:
        if spec.fallback_for and shapes.get(spec.fallback_for) is not None:
            reasons[spec.key] = "fallback not needed"
            continue
        shape = shapes[spec.diag]
        if shape is None:
            reasons[spec.key] = "group absent"
        elif not 0 <= spec.channel < shape[0]:
            reasons[spec.key] = "channel out of range"
        else:
            specs.append(spec)
    return specs, reasons


def coverage(corpus_file, diag: str) -> tuple[float, float]:
    """`(t0_s, t1_s)` of the group's finite samples; `(nan, nan)` if absent.

    The span is where the diagnostic has data, which is not the same as the
    file's time axis: `mirnov` on shot 198658 carries 8,191,977 samples of
    which the first 1,669,828 and the last 1,443,111 are NaN, and an event
    detector that claimed coverage over the NaN would be claiming to have
    looked where nothing was recorded.

    Finiteness is read off channel 0 - one contiguous row, ~8 MB on the
    widest group - because the gaps are a property of when the digitiser
    ran, not of one channel. NaN rather than an exception for an absent
    group: a shot missing `co2` is the common case (39% availability), and
    the caller's job is to record the gap, not to stop.
    """
    with h5py.File(corpus_file, "r", locking=False) as f:
        if _shape_of(f, diag) is None:
            return (float("nan"), float("nan"))
        y = np.asarray(f[diag]["ydata"][0, :])
        x = np.asarray(f[diag]["xdata"][:], dtype=np.float64)
    if x.size != y.size:
        raise ValueError(
            f"{diag}: xdata has {x.size} samples, ydata {y.size}"
        )
    finite = np.flatnonzero(np.isfinite(y) & np.isfinite(x))
    if finite.size == 0:
        return (float("nan"), float("nan"))
    return (float(x[finite[0]]), float(x[finite[-1]]))
