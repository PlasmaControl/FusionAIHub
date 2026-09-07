"""The 88-channel IGNITE actuator contract: layout, per-frame means, z-scoring, and edits.

The production dynamics checkpoint (`cfg_actuator_dim = 88`) consumes one 88-wide control vector
per 50 ms frame, and the code that built those vectors is in no git branch: the committed
`train_dynamics._ACT_SPEC` sums to 70 channels, the previous generation, and the bundle's own
README says the extra 18 are undocumented. This module is the reconstruction, and it is not a
guess -- `scripts/ideate/g_enc.py` rebuilds the actuator block of the ten caches shipped in the
bundle and compares it with what production wrote. Measured 2026-09-07 over all ten:

    88/88 channels BIT-IDENTICAL in float16 on 190090, 190729, 192473, 200241, 201885,
    202537, 202793, 204346;  70/88 and 69/88 on 190735 and 190736, where `rmp[11]` and
    `i_coil[0:6]` differ (the corpus files for those two shots disagree with what the
    production builder read; the residual is ~1 z on i_coil[5] and ~2 z on rmp[11]).

So the layout below is measured fact for eight shots out of ten and the two exceptions are named
rather than smoothed away.

Three properties of that layout are load-bearing and each is a silent failure if ignored.

**Group order and width.** `ech_power 12 | pinj 8 | beam_voltage 8 | tinj 8 | gas_flow 11 |
gas_raw 11 | rmp 12 | i_coil 18`, in that order, offsets 0/12/20/28/36/47/58/70. A group the
shot does not have is not dropped -- it is ZERO-filled to its full width, so channel 70 is
`i_coil[0]` for every shot whether or not the coils were energised. That is measured, not
assumed: every one of the ten shipped caches carries at least one all-zero group at exactly
these offsets (190090/201885/202537/202793 zero `ech_power`+`pinj`+`beam_voltage`+`tinj`,
200241 zeroes those three without `ech_power`, and the remaining five zero `ech_power`), and
they reproduce bit-identically here.

**The time base.** A frame's window is resolved in SHOT time, `round((t - xdata[0]) * fs)` with
`fs = (n - 1) / (xdata[-1] - xdata[0])`, not by counting samples from the array start. The
actuator groups do not start at t = 0 (gas at -10 s, beam_voltage at ~-6.3 s, rmp at ~-1.0 s and
shot-dependent), so counting from the array start reads the gas trace entirely out of pre-shot
idle. And `xdata` is float32, so `1 / median(diff(x))` loses the step to cancellation and drifts
~1.8 frames over an 11 s window; the span-based rate is exact.

**The z-score.** Each channel is z-scored over the shot's own frames, ddof 0. That is why
`apply` exists: the cached actuators are ALREADY per-shot z-scored, so scaling a channel and
re-z-scoring it against ITS OWN statistics is an exact no-op -- a +20 % NBI edit disappears. Only
re-z-scoring against the REFERENCE shot's statistics keeps the edit (`policy="reference_stats"`,
the default). `policy="self_stats"` exists so a test can demonstrate the trap.

**A PRODUCTION QUIRK IS REPRODUCED HERE ON PURPOSE, AND NEW SHOTS INHERIT IT.** The corpus
writes a trailing all-NaN pad sample on nearly every actuator group; production averaged it in
as a zero, so whichever frame straddles the end of a record comes out divided by a window that
is longer than the samples it contains -- the end-of-record frame reads low, and a frame that is
only half recorded reads at half amplitude. `_record_length` recovers the untrimmed length and
divides by it deliberately, because bit-compatibility with the ten shipped caches is what makes
a Phase-5 seed mean anything (getting it wrong costs ~0.04 z on all twelve `rmp` channels; see
`_record_length`). The consequence is that EVERY shot encoded from now on carries production's
zero-averaged end-of-record frame, by design. `frame_means` is therefore the right function for
building a model input and the WRONG one to hand to anyone doing physics on the last frame of a
record -- for that, average over the samples that exist. `tests/ideate/test_design_actuators.py`
pins the quirk with an exact 100 * 200 / 201 case so it cannot be "simplified" away silently.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np

from ..config import load_yaml
from ..schema import ActuationSet
from ..shotdb.reader import Unavailable

FRAME_S = 0.05  # IGNITE's production frame duration
#: The z the training corpus was standardised to; beyond this the dynamics model is
#: extrapolating and `apply` says so (plan §3 V10).
EXTRAPOLATION_Z = 3.0
#: Production's `(x - mean) / (std + EPS)`. For a constant channel `x - mean` is exactly 0, so
#: this and the explicit "std 0 -> z 0" rule agree; the epsilon only matters for a channel whose
#: std is nonzero but far below the trace's own scale.
EPS = 1e-6

Policy = Literal["reference_stats", "self_stats"]


@dataclass(frozen=True)
class ActGroup:
    """One actuator group's slice of the 88-wide vector."""

    group: str  # the corpus group name, e.g. "ech_power"
    n_channels: int
    offset: int  # first row of this group in the 88-channel vector

    @property
    def rows(self) -> slice:
        return slice(self.offset, self.offset + self.n_channels)


def _spec(*groups: tuple[str, int]) -> tuple[ActGroup, ...]:
    out, off = [], 0
    for name, n in groups:
        out.append(ActGroup(name, n, off))
        off += n
    return tuple(out)


ACT_SPEC: tuple[ActGroup, ...] = _spec(
    ("ech_power", 12),
    ("pinj", 8),
    ("beam_voltage", 8),
    ("tinj", 8),
    ("gas_flow", 11),
    ("gas_raw", 11),
    ("rmp", 12),
    ("i_coil", 18),
)
N_CHANNELS = sum(g.n_channels for g in ACT_SPEC)  # 88
_BY_GROUP = {g.group: g for g in ACT_SPEC}


class Reader(Protocol):
    """The slice of `shotdb.corpus.CorpusReader` this module uses."""

    def read(
        self, shot: int, group: str, channels: Sequence[int] | None = None
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def coverage(self, shot: int, group: str) -> tuple[float, float]: ...


# ------------------------------------------------------------------------------- the layout


def group_spec(group: str) -> ActGroup:
    if group not in _BY_GROUP:
        raise KeyError(f"{group!r} is not an IGNITE actuator group ({', '.join(_BY_GROUP)})")
    return _BY_GROUP[group]


def channel_index(group: str, ch: int) -> int:
    """The row of `(group, ch)` in the 88-channel vector."""
    spec = group_spec(group)
    if not 0 <= ch < spec.n_channels:
        raise IndexError(f"{group} has {spec.n_channels} channels, asked for {ch}")
    return spec.offset + ch


def channel_names() -> tuple[str, ...]:
    """`("ech_power[0]", ..., "i_coil[17]")` -- 88 names in row order."""
    return tuple(f"{g.group}[{i}]" for g in ACT_SPEC for i in range(g.n_channels))


CHANNEL_NAMES = channel_names()


# ------------------------------------------------------------------------------- frame means


def _record_length(t_ms: np.ndarray, span_ms: float) -> int:
    """How many samples the group's time axis has BEFORE the reader strips its trailing pad.

    `CorpusReader.read` drops the samples at the end where every channel is non-finite, which is
    right for a percentile and wrong here: the corpus writes a one-sample NaN pad on almost every
    actuator group (measured: rmp, gas_flow, i_coil and beam_voltage each lose exactly one sample
    on 190090/202537/204346), production averaged that pad in as a zero, and dropping it shortens
    the divisor of whichever frame straddles the end of the record. On 190090 that alone moved
    all twelve `rmp` channels by ~0.04 z against the shipped cache -- small, systematic, and
    entirely invisible unless you compare. `coverage` reports the untrimmed span, so the full
    length is recoverable from the two together.
    """
    n_read = int(t_ms.size)
    read_span = float(t_ms[-1] - t_ms[0])
    if read_span <= 0 or span_ms <= 0:
        return n_read
    return max(n_read, round(span_ms * (n_read - 1) / read_span) + 1)


def _frame_bounds(
    t_ms: np.ndarray, n_record: int, span_ms: float, n_frames: int, frame_s: float, t0_s: float
) -> np.ndarray:
    """`(n_frames, 2)` sample indices, clamped into the record.

    Written over the millisecond axis the corpus reader returns, in the one arrangement that
    keeps the 1000x out of the arithmetic: `(t - x0) * (n - 1) / span` is `(t - x0) * fs` with
    the unit factor cancelled between numerator and denominator, so a reader that converts to ms
    and a producer that stayed in seconds land on the same index.
    """
    per_ms = (n_record - 1) / span_ms if span_ms > 0 else 10.0  # 10 kHz, the actuators' rate
    starts = (t0_s + np.arange(n_frames) * frame_s) * 1000.0 - float(t_ms[0])
    lo = np.rint(starts * per_ms)
    hi = np.rint((starts + frame_s * 1000.0) * per_ms)
    return np.clip(np.stack([lo, hi], axis=1), 0, n_record).astype(np.int64)


def frame_means(
    shot: int,
    reader: Reader,
    *,
    n_frames: int,
    frame_s: float = FRAME_S,
    t0_s: float = 0.0,
    dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    """`(88, n_frames)` float32: each actuator channel's mean over each 50 ms frame.

    A group the shot does not carry is left as NaN ROWS, which is a different statement from the
    zeros the cache ends up holding: "this shot has no I-coil record" versus "the coils were at
    zero". `build_actuators` collapses the first into the second because that is what production
    wrote and what the checkpoint was trained on; a caller reasoning about coverage wants the
    NaN. Within a group that IS present, a non-finite sample counts toward the frame's mean with
    the value 0 -- production's `np.nan_to_num(seg).mean(axis=1)` -- so a half-recorded frame
    reads as half amplitude, not as the average of what was recorded.

    `dtype` exists for exactly one caller. The means are computed in float64 and float32 is a
    perfectly good way to REPORT a mean of float32 samples, but production z-scored the
    un-rounded float64 means and only rounded at the very end, to float16. Rounding to float32
    first moves the last float16 bit of roughly one channel in ten (measured on 190090: 80/88
    actuator channels bit-identical to the shipped cache instead of 88/88), so
    `build_actuators` asks for float64 and everyone else gets the smaller array.
    """
    out = np.full((N_CHANNELS, n_frames), np.nan, dtype=np.float64)
    for spec in ACT_SPEC:
        try:
            t_ms, y = reader.read(shot, spec.group)
        except Unavailable:
            continue
        if t_ms.size < 2:
            continue
        try:
            t0_ms, t1_ms = reader.coverage(shot, spec.group)
        except Unavailable:  # pragma: no cover -- read() succeeded, so the axis is there
            t0_ms, t1_ms = float(t_ms[0]), float(t_ms[-1])
        span_ms = t1_ms - t0_ms
        n_record = _record_length(t_ms, span_ms)
        y = np.nan_to_num(np.asarray(y, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        bounds = _frame_bounds(t_ms, n_record, span_ms, n_frames, frame_s, t0_s)
        col = np.zeros((spec.n_channels, n_frames), dtype=np.float64)
        # The group can be narrower or wider than the contract: pad with zeros, never drop.
        width, n_read = min(spec.n_channels, y.shape[0]), y.shape[1]
        for f, (i0, i1) in enumerate(bounds):
            hi = min(int(i1), n_read)
            if i1 > i0 and hi > i0:
                # sum / window length, not mean of what survives: the samples the reader
                # stripped were non-finite, and production counted them as zeros.
                col[:width, f] = y[:width, i0:hi].sum(axis=1) / (i1 - i0)
        out[spec.rows] = col
    return out.astype(dtype)


# ------------------------------------------------------------------------------- z-scoring


def z_score(
    x: np.ndarray, *, stats: tuple[np.ndarray, np.ndarray] | None = None
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """`(z, (mean, std))` for `(C, T)` traces, standardised per channel over time with ddof 0.

    A channel with zero spread is returned as exactly 0 rather than as 0/eps noise, and `stats`
    is returned unchanged when it was supplied -- which is how `apply` re-standardises an edited
    trace against a DIFFERENT shot's statistics.
    """
    x = np.asarray(x, dtype=np.float64)
    if stats is None:
        stats = (x.mean(axis=1), x.std(axis=1))  # ddof 0: the production convention
    mean, std = (np.asarray(s, dtype=np.float64) for s in stats)
    z = np.where(std[:, None] > 0.0, (x - mean[:, None]) / (std[:, None] + EPS), 0.0)
    return z.astype(np.float32), stats


@dataclass
class Actuators:
    """One shot's control trajectory: the raw frame means and their per-shot z-score."""

    shot: int
    raw: np.ndarray  # (88, n_frames) float64, absent groups zero-filled
    z: np.ndarray  # (88, n_frames) float32
    stats: tuple[np.ndarray, np.ndarray]  # per-channel (mean, std) of `raw`
    missing: tuple[str, ...] = ()  # groups the corpus file did not carry
    frame_s: float = FRAME_S
    t0_s: float = 0.0

    @property
    def n_frames(self) -> int:
        return int(self.raw.shape[1])

    @property
    def frame_times(self) -> np.ndarray:
        """Seconds; frame f covers `[t0 + f * frame_s, t0 + (f + 1) * frame_s)`."""
        return self.t0_s + np.arange(self.n_frames) * self.frame_s


def build_actuators(
    shot: int,
    reader: Reader,
    n_frames: int,
    *,
    frame_s: float = FRAME_S,
    t0_s: float = 0.0,
) -> Actuators:
    """The 88-channel trajectory for `shot`, z-scored against its own frames.

    This is what goes into a frame-code cache. Absent groups become zeros here (see
    `frame_means`), which z-score to zeros because a constant channel has no spread.
    """
    raw = frame_means(
        shot, reader, n_frames=n_frames, frame_s=frame_s, t0_s=t0_s, dtype=np.float64
    )
    absent = tuple(g.group for g in ACT_SPEC if np.isnan(raw[g.offset]).all())
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    z, stats = z_score(raw)
    return Actuators(shot, raw, z, stats, absent, frame_s, t0_s)


# ------------------------------------------------------------------------------- edits


def corpus_systems() -> dict[str, tuple[str, tuple[str, ...]]]:
    """`{registry system: (corpus group, member ids in the corpus's channel order)}`.

    Read from `configs/ideate/actuators.yaml:corpus`, which is the only place the pairing lives:
    a corpus group is an unnamed `(C, n)` array with no units and no labels, and the member order
    is the corpus producer's, not the registry's (the corpus lists its twelve gyrotrons
    alphabetically while `systems.ech` lists them by installation date). Only the four entries
    that name a `system:` can be addressed per member; `beam_voltage`, `tinj`, `gas_flow` and
    `rmp` have no per-member registry specs, so no waveform key resolves to them.
    """
    out: dict[str, tuple[str, tuple[str, ...]]] = {}
    for entry in load_yaml("actuators.yaml")["corpus"].values():
        system, group = entry.get("system"), entry["group"]
        if system and entry.get("members") and group in _BY_GROUP:
            out[system] = (group, tuple(str(m) for m in entry["members"]))
    return out


def resolve_key(key: str) -> tuple[ActGroup, tuple[int, ...]]:
    """A waveform key (`"nbi.total"`, `"ech.LUKE"`) -> the group and the rows it drives."""
    system, _, member = key.partition(".")
    systems = corpus_systems()
    if system not in systems:
        raise KeyError(f"{key!r}: no corpus actuator group for system {system!r}")
    group, members = systems[system]
    spec = group_spec(group)
    if member in ("", "total"):
        return spec, tuple(range(spec.offset, spec.offset + spec.n_channels))
    if member not in members:
        raise KeyError(f"{key!r}: {system} has no member {member!r} in the corpus channel order")
    return spec, (spec.offset + members.index(member),)


@dataclass
class ApplyReport:
    """What an edit did, in the units the model actually sees."""

    policy: str
    edited_channels: tuple[str, ...] = ()
    z_shift: dict[str, float] = field(default_factory=dict)  # channel -> max |dz| vs the reference
    #: V10's warning, scoped to the EDIT: channels this actuation set wrote that end up beyond
    #: |z| = EXTRAPOLATION_Z. This is the list a caller has to act on.
    edited_extrapolated: tuple[str, ...] = ()
    #: Informational, and deliberately not mixed in with the above: channels of the REFERENCE
    #: shot that were already beyond the limit before the edit. Real reference shots are full of
    #: them -- 22/88 on 190090 (max 9.88 z), 2/88 on 202537, 26/88 on 204346 -- so reporting
    #: them as edit warnings buries the one warning that is about the caller's own change.
    reference_extrapolated: tuple[str, ...] = ()
    unresolved_keys: tuple[str, ...] = ()

    @property
    def max_z_shift(self) -> float:
        return max(self.z_shift.values(), default=0.0)


def _edit_raw(
    aset: ActuationSet, raw: np.ndarray, frame_times: np.ndarray
) -> tuple[np.ndarray, list[int], list[str]]:
    """Write the actuation set's waveforms into a copy of the raw traces.

    A `<system>.total` waveform is a SUM over the group's channels (that is `reduce: sum` in
    actuators.yaml for all four addressable systems), so it is applied as a per-frame gain that
    preserves the split between members -- scaling a total must not decide by itself which beam
    carries the extra power. Where the reference total is zero there is no split to preserve and
    the demand is spread evenly, which is the only answer that does not invent a beam.
    """
    from ..retrieval.actuation import values_at

    out = np.array(raw, dtype=np.float64, copy=True)
    edited: list[int] = []
    unresolved: list[str] = []
    for key, wf in sorted(aset.waveforms.items()):
        try:
            _, rows = resolve_key(key)
        except KeyError:
            unresolved.append(key)
            continue
        target = values_at(wf.vertices, frame_times)
        idx = np.asarray(rows, dtype=np.int64)
        if len(rows) == 1:
            out[idx[0]] = target
        else:
            current = out[idx].sum(axis=0)
            live = current != 0.0
            gain = np.divide(target, current, out=np.ones_like(target), where=live)
            out[idx] = np.where(live, out[idx] * gain, target / len(rows))
        edited.extend(int(i) for i in idx)
    return out, sorted(set(edited)), unresolved


def apply(
    aset: ActuationSet,
    raw: np.ndarray,
    stats: tuple[np.ndarray, np.ndarray],
    *,
    policy: Policy = "reference_stats",
    frame_s: float = FRAME_S,
    t0_s: float = 0.0,
) -> tuple[np.ndarray, ApplyReport]:
    """Edit `raw (88, T)` with `aset` and re-standardise it -> `(z (88, T), ApplyReport)`.

    `policy="reference_stats"` divides the edited traces by the REFERENCE shot's per-channel
    mean and std -- the ones `stats` carries -- so a +20 % NBI edit moves the z traces the model
    reads. `policy="self_stats"` recomputes the statistics from the edited traces, which cancels
    any pure scale or offset exactly; it is kept only so a test can show that it does.

    The report's two extrapolation lists are deliberately separate. `edited_extrapolated` is
    V10's warning -- channels THIS actuation set wrote that end past |z| = 3 -- and is the list
    to act on. `reference_extrapolated` is the reference shot's own baseline, which on a real
    shot is most of what a whole-vector check would report (22/88 channels on 190090, 26/88 on
    204346) and none of which the caller did.
    """
    if policy not in ("reference_stats", "self_stats"):
        raise ValueError(f"unknown policy {policy!r} (reference_stats | self_stats)")
    raw = np.asarray(raw, dtype=np.float64)
    if raw.shape[0] != N_CHANNELS:
        raise ValueError(f"raw actuators must be ({N_CHANNELS}, T), got {raw.shape}")
    frame_times = t0_s + np.arange(raw.shape[1]) * frame_s
    edited_raw, rows, unresolved = _edit_raw(aset, raw, frame_times)

    reference_z, _ = z_score(raw, stats=stats)
    z, _ = z_score(edited_raw, stats=stats if policy == "reference_stats" else None)

    shift = np.abs(z - reference_z).max(axis=1)
    report = ApplyReport(
        policy=policy,
        edited_channels=tuple(CHANNEL_NAMES[i] for i in rows),
        z_shift={CHANNEL_NAMES[i]: float(shift[i]) for i in rows},
        edited_extrapolated=_extrapolated(z, rows),
        reference_extrapolated=_extrapolated(reference_z),
        unresolved_keys=tuple(unresolved),
    )
    if report.unresolved_keys:
        warnings.warn(
            "actuation keys with no corpus actuator group were ignored: "
            + ", ".join(report.unresolved_keys),
            RuntimeWarning,
            stacklevel=2,
        )
    return z, report


def _extrapolated(
    z: np.ndarray, rows: Sequence[int] | None = None, limit: float = EXTRAPOLATION_Z
) -> tuple[str, ...]:
    """Channels of `z` (restricted to `rows` when given) whose peak |z| is past `limit`.

    `rows` is what makes the warning V10's rather than the shot's. Computed over all 88 rows it
    is dominated by the reference shot's own baseline -- 22 of 88 channels on 190090 -- and the
    one line that says "your edit left the envelope" is indistinguishable from twenty-one that
    say "this shot always was outside it". `apply` calls this twice: once over the edited rows
    of the edited z, once over every row of the unedited reference z.
    """
    peak = np.abs(z).max(axis=1)
    idx = np.flatnonzero(peak > limit)
    if rows is not None:
        idx = idx[np.isin(idx, np.asarray(rows, dtype=np.int64))]
    return tuple(
        f"{CHANNEL_NAMES[i]} reaches |z| = {peak[i]:.2f} (> {limit:g}): outside the range the "
        f"dynamics model was trained on, so its response there is extrapolation"
        for i in idx
    )
