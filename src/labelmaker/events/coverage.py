"""What was actually looked at, per source, per quantity.

`t_cov0_s`/`t_cov1_s` on an event row are what tells "no ELM here" from
"nobody looked", and they are only worth that if they are the coverage of
the thing the row was measured on. Two ways they stop being that, both
found by the iteration-0 critic on shot 198658:

* **A union of unrelated axes.** The old `pipeline._span` took the earliest
  start and latest end across EVERY actuator input and gave that one span
  to every actuator event. Gas runs -10 to 94.8576 s on that shot (the gas
  recorder really does have that long an axis), NBI 0 to 13.1001 s and the
  RMP coils -1.06286 to 10.20114 s - so an `nbi_on` row claimed 94.86 s of
  coverage it had no measurement over, and "the beams were off after 13 s"
  became indistinguishable from "nobody measured them". `feature_spans`
  gives each feature ITS OWN axis, and `intersect` gives a heuristic that
  needs several inputs at once - the L-H detector needs D-alpha AND the
  line density AND the injected power - the intersection of the ones it
  requires, which is the only interval in which its answer means anything.

* **Padding counted as observation.** A record's axis routinely runs past
  its samples: the corpus' fast groups end in NaN, a filterscope's head and
  tail are NaN, a dead coil is NaN throughout. `finite_span` therefore
  measures from the first to the last FINITE sample rather than from the
  first to the last time, so coverage is where there was something to see.
  For a multi-channel feature a sample counts as finite when ANY channel is
  - "the RMP coils are on" is a claim about the set of them, and one dead
  coil does not end the observation.

**Clipping is the other half of the same honesty.** A transform pads: a
track stitched across tile boundaries ended up to 2.052 ms past its own
`t_cov1_s` on the three pilot shots (9, 11 and 15 rows of them). That
overrun is transform-edge support, not plasma, so `clip_to_coverage`
trims the extent back into the coverage and says `attrs["clipped"] = true`
where it had to - and `schema.Event.__post_init__` now REFUSES a row that
ends after its own coverage, so the invariant is enforced at the point the
row is built rather than repaired later by whoever notices.

Nothing here opens a file or knows a locator; it takes arrays and spans.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

#: An unknown span. Not `(0, 0)` and not `(-inf, inf)`: "nobody looked" is
#: a third answer, and a consumer that treats it as either is wrong in a
#: direction that cannot be detected downstream.
UNKNOWN: tuple[float, float] = (float("nan"), float("nan"))


def _known(span) -> bool:
    return bool(math.isfinite(float(span[0])) and math.isfinite(float(span[1])))


def finite_span(t_s, y=None) -> tuple[float, float]:
    """`(first, last)` time at which there is a finite sample.

    `y` is the trace measured on `t_s`: `(samples,)` or
    `(channels, samples)`, in which case a sample counts wherever ANY
    channel is finite. Without `y` the finiteness of `t_s` itself is the
    test, which is what a bare time axis can be asked.

    `UNKNOWN` when nothing is finite - an all-NaN channel, an empty axis -
    because a span computed from no samples is not a span.
    """
    t = np.asarray(t_s, dtype=np.float64).ravel()
    ok = np.isfinite(t)
    if y is not None:
        arr = np.asarray(y, dtype=np.float64)
        if arr.ndim == 1:
            sample = np.isfinite(arr)
        elif arr.ndim == 2:
            sample = np.isfinite(arr).any(axis=0)
        else:
            raise ValueError(f"y must be 1-D or 2-D; got {arr.shape}")
        if sample.size != t.size:
            raise ValueError(
                f"{sample.size} samples against {t.size} times"
            )
        ok = ok & sample
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return UNKNOWN
    return (float(t[idx[0]]), float(t[idx[-1]]))


def feature_spans(
    features: Mapping[str, tuple[Any, Any]],
) -> dict[str, tuple[float, float]]:
    """`{name: (t0, t1)}`: each canonical feature's own finite coverage.

    `features` is what `pipeline._actuator_features` builds and
    `heuristics.actuator_intervals` reads - `{name: (t_s, y)}` in
    `features/namespace.py`'s terms - and this is the per-quantity coverage
    that replaces the union of all their axes.
    """
    return {
        name: finite_span(t_s, y) for name, (t_s, y) in features.items()
    }


def intersect(spans: Iterable[tuple[float, float]]) -> tuple[float, float]:
    """The interval every one of `spans` covers, or `UNKNOWN`.

    For a detector that needs several inputs AT ONCE. One unknown span
    makes the intersection unknown rather than being skipped: a heuristic
    whose density trace was never measured did not observe the shot's
    D-alpha stretch, it observed nothing, and answering with the D-alpha's
    span would claim otherwise. An empty iterable is `UNKNOWN` for the
    same reason, and so is an intersection that comes out backwards - two
    inputs that never ran at the same time cover no time together.
    """
    spans = list(spans)
    if not spans or not all(_known(s) for s in spans):
        return UNKNOWN
    lo = max(float(s[0]) for s in spans)
    hi = min(float(s[1]) for s in spans)
    return (lo, hi) if hi >= lo else UNKNOWN


def clip_to_coverage(
    t0_s: float, t1_s: float, cov: tuple[float, float],
) -> tuple[float, float, bool]:
    """`(t0, t1, clipped)`: the extent trimmed into `[cov0, cov1]`.

    An unknown or half-unknown coverage clips nothing on the side it does
    not know, because there is no bound there to trim to.

    An extent lying WHOLLY outside its coverage comes back untouched, with
    `clipped` false. That is not a transform edge - it is a detector
    claiming an event in a stretch it did not measure - and moving it onto
    the boundary would turn a bug into a plausible-looking row. Left
    unclipped, `schema.Event.__post_init__` refuses it, which is where a
    caller finds out.
    """
    t0, t1 = float(t0_s), float(t1_s)
    cov0, cov1 = float(cov[0]), float(cov[1])
    lo = max(t0, cov0) if math.isfinite(cov0) else t0
    hi = min(t1, cov1) if math.isfinite(cov1) else t1
    if hi < lo:
        return (t0, t1, False)
    return (lo, hi, lo != t0 or hi != t1)


def clipped_attrs(attrs: Mapping[str, Any], clipped: bool) -> dict[str, Any]:
    """`attrs` with `clipped` recorded when, and only when, it happened.

    Absent rather than `false` on the ordinary row: `attrs` is stored as a
    JSON string per row and a key that is false on 99.9% of a campaign's
    events costs more than it says. A consumer asks `attrs.get("clipped")`.
    """
    out = dict(attrs)
    if clipped:
        out["clipped"] = True
    return out


# ------------------------------------------------- per-source completion

#: A skip whose step name starts with one of these is about a `(diag,
#: channel[, pass])` block, and the rest of the key names it. `norm` is
#: its own source and not `tokeye_track`'s, because a norm fallback is
#: recorded WHILE the block runs: they are two facts about one block, and
#: collapsing them onto one key would lose whichever came second.
_BLOCK_STEPS = {
    "channel": "tokeye_track",
    "read": "tokeye_track",
    "mask": "tokeye_track",
    "norm": "norm",
}

#: Step name -> the sources it would have written. A step that produces two
#: sources gets a row for each, so "did `tokeye_transient` run" has an
#: answer whichever way the ELM clock went. A step with no entry here is
#: its own source name - `qh_flattop` and `nbi_counter` are recorded skips
#: that are not event sources at all, and inventing a source for them would
#: be worse than letting them name themselves.
_STEP_SOURCES = {
    "elm_clock": ("tokeye_transient", "elm_clock"),
    "sawtooth": ("ece_sawtooth",),
    "lh": ("dalpha_lh",),
    "actuator": ("actuator",),
    "qh": ("qh_proxy",),
    "text": ("text",),
}

#: The diagnostic a non-block step reads, where it reads one. Used only to
#: fill the `diag` column of a SKIPPED step's row, so that a skipped
#: sawtooth row and a sawtooth row that ran carry the same key.
_STEP_DIAGS = {
    "sawtooth": "ece",
    "lh": "filterscopes",
}


def _split_block_key(rest: str) -> tuple[str, int, str]:
    """`"mhr:4:wide"` -> `("mhr", 4, "wide")`; a missing part is the default."""
    parts = rest.split(":")
    diag = parts[0] if parts and parts[0] else ""
    try:
        channel = int(parts[1])
    except (IndexError, ValueError):
        channel = -1
    pass_name = parts[2] if len(parts) > 2 else ""
    return diag, channel, pass_name


def skip_key(step: str) -> list[tuple[str, str, int, str]]:
    """One `ShotResult.skipped` step name -> the `(source, diag, channel,
    pass)` keys it is a fact about.

    The step names are `pipeline.process_shot`'s: `"channel bes:26"`,
    `"mask mhr:4:wide"`, `"actuator gas"`, `"elm_clock"`, `"sawtooth"`.
    Anything unrecognised becomes a row under its own name rather than
    being dropped - a skip nobody can read is still better than a skip
    nobody can see.
    """
    head, _, rest = step.partition(" ")
    if head in _BLOCK_STEPS and rest:
        diag, channel, pass_name = _split_block_key(rest)
        return [(_BLOCK_STEPS[head], diag, channel, pass_name)]
    if head == "actuator" and rest:
        # `"actuator gas"`: the group is what was not read, and the group
        # name is what the ran rows carry in `diag`.
        return [("actuator", rest, -1, "")]
    sources = _STEP_SOURCES.get(step, (step,))
    diag = _STEP_DIAGS.get(step, "")
    return [(source, diag, -1, "") for source in sources]


def source_records(
    shot: int,
    *,
    ran: Mapping[tuple[str, str, int, str], tuple[float, float]],
    skipped: Mapping[str, str],
    events: Sequence[Any] = (),
) -> list[dict[str, Any]]:
    """One shot's per-source completion rows, for `schema.write_sources`.

    `ran` maps a `(source, diag, channel, pass_name)` key to the coverage
    that key was computed over - EVERY key that ran, whether or not it
    produced an event, which is the whole point: "the tracker ran on
    mhr_04_wide and saw no mode" and "mhr_04_wide was never processed" are
    different facts and an events file alone states neither.

    `skipped` is `ShotResult.skipped`; every entry becomes a row (two,
    where one step owns two sources), with its reason. A key that both ran
    and was skipped keeps the RAN row, because the step got far enough to
    look and `status == "ok"` carries an empty `reason` by contract; the
    step's own message is still in `ShotResult.skipped` and in the run's
    JSON. `norm` and `qh_flattop` have sources of their own precisely so
    that the caveats which fire while a step SUCCEEDS get rows rather than
    colliding with it.
    """
    counts: dict[tuple[str, str, int, str], int] = {}
    for event in events:
        key = (
            str(event.source), str(event.diag), int(event.channel),
            str(event.pass_name),
        )
        counts[key] = counts.get(key, 0) + 1

    rows: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for key, span in ran.items():
        rows[key] = _row(shot, key, "ok", "", span, counts.get(key, 0))
    for step, why in skipped.items():
        for key in skip_key(step):
            if key in ran:
                continue
            if key in rows:
                rows[key]["reason"] = f"{rows[key]['reason']}; {why}"
                continue
            rows[key] = _row(shot, key, "skipped", why, UNKNOWN,
                             counts.get(key, 0))
    # Any source that wrote events without declaring a key: recorded rather
    # than lost, with the coverage its own rows carry.
    for key, n in counts.items():
        if key not in rows:
            rows[key] = _row(shot, key, "ok", "", UNKNOWN, n)
    return [rows[key] for key in sorted(rows)]


def _row(shot: int, key, status: str, reason: str, span, n_events: int) -> dict:
    source, diag, channel, pass_name = key
    return {
        "shot": int(shot),
        "source": str(source),
        "status": str(status),
        "reason": str(reason),
        "t_cov0_s": float(span[0]),
        "t_cov1_s": float(span[1]),
        "n_events": int(n_events),
        "diag": str(diag),
        "channel": int(channel),
        "pass_name": str(pass_name),
    }


__all__ = [
    "UNKNOWN",
    "clip_to_coverage",
    "clipped_attrs",
    "feature_spans",
    "finite_span",
    "intersect",
    "skip_key",
    "source_records",
]
