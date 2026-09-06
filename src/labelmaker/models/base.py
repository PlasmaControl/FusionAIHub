"""What a model consumes and produces, in canonical terms.

A model folder contributes a `spec.py` holding these dataclasses and nothing
else: the mapping from the names the model was trained under to canonical
features, the lag structure, the training-time admissible domain, and how to
read its output columns. Fetching, sampling, ensembling and writing are
shared, so adding a model adds no pipeline code.

The domain is kept as data (a tuple of `DomainRule`) rather than a
hand-written predicate, so the same clauses can be printed into the model
card and checked by a test.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..features import namespace as ns
from ..features.store import FeatureArray
from ..timebase import sample_at, window_mean

#: Width of the boxcar the archive store's own build averaged over: two
#: 25 ms grid steps. MEASURED in Task 12 against raw PTDATA `ip` (8 random
#: overlap shots): archive row k is the mean of the raw signal over
#: `[STEP_S*(k-2), STEP_S*k]`, reproducing the RAW SIGNAL to a median
#: relative error of 3.9e-08 (float32 round-trip precision) under this
#: convention - three to five orders of magnitude better than nearest-sample
#: or any other window placement tried against that raw signal.
#:
#: Task 15 independently confirmed the SAME window convention wins on the
#: corpus and fdp paths (`ip`, `kappa`, `pinj_total`, `tinj_total`; see
#: `validate`'s module docstring and the Task 15 report for the
#: four-candidate scan) - but do not read 3.9e-08 as those paths' accuracy.
#: That figure is Task 12's raw-PTDATA number only. Measured against the
#: ARCHIVE column (not raw PTDATA) on the corpus and fdp paths, this
#: convention beats the best alternative candidate by 11x to 232x and
#: nearest-sample alone by 98x to 233x - ONE TO TWO orders of magnitude,
#: never three or four - landing at 5.9e-4 to 1.4e-2 median relative error
#: on corpus `pinj_total`/`tinj_total` and 9.9e-5 on fdp `ip`.
#:
#: This is a property of how the archive store was built, not of any one
#: model's `dt_s`, so it is fixed at two grid steps rather than derived from
#: a per-model value.
ARCHIVE_WINDOW_S = 2 * ns.STEP_S

#: ROW 0 AND COVERAGE, MEASURED (Task 15). Windowing
#: introduces two things a bare nearest-sample convention did not have.
#:
#: (a) Row 0's validity is now source-dependent. The window at `t=0` is
#: `[-ARCHIVE_WINDOW_S, 0)`, and whether that holds a sample depends on
#: where the SOURCE record starts, not on the physics. MEASURED: the
#: archive's own row 0 is a genuine finite reading, not NaN - `ip` on
#: shots 183224/183225/183226 is -1175.8 A / 1717.6 A / -3726.1 A, a noisy
#: near-zero pre-plasma baseline - because the raw PTDATA record it was
#: built from starts well before t=0 (fdp's own fetch of `ip` on shot
#: 185945 starts at t=-0.973 s), so upstream's window had real samples to
#: average. The corpus' `pinj`/`tinj` records, by contrast, are clipped to
#: start at EXACTLY t=0.0 s (measured on 10 of 10 random corpus shots), so
#: the SAME window at row 0 finds nothing and `window_mean` correctly
#: returns NaN, which `_validity` then correctly marks invalid. This is not
#: a bug: it is the resolver-dependence this task set out to remove
#: elsewhere in the record, surfacing at the one row of 240 the archive can
#: answer from data no corpus-served channel has (an fdp-served field -
#: PTDATA `ip`/`bt`, EFIT, ZIPFIT - generally does not have this problem,
#: since its raw record also predates t=0).
#:
#: (b) Coverage. `timebase.window_mean` has no minimum-sample requirement -
#: one finite sample in the window is enough to mark a row measured, where
#: the old `sample_at(max_gap=dt/2)` convention returned NaN whenever the
#: nearest sample was more than half a step away. MEASURED on a
#: corpus-served 10 kHz channel (`pinj`, 8 random corpus shots, `n=960`
#: rows): the interior of the record holds a median of exactly 500 samples
#: per 50 ms window, as the sample rate predicts, and the only rows below
#: that are the record's own edges - row 0 (0 samples, see (a) above) and
#: row 1 (half the window, ~250 samples) - 4 of 960 sampled rows (0.4%),
#: all at the very start of a shot. No row elsewhere in a shot was found
#: under-covered, so no minimum-coverage guard is added here: the looser
#: policy costs only the same edge rows (a) already accounts for, not an
#: unbounded fraction of the record.


def sample_by_resolver(
    x: np.ndarray, y: np.ndarray, t: np.ndarray, resolver: str, dt_s: float
) -> np.ndarray:
    """Sample one stored array the way its resolver's convention prescribes.

    Shared by `InputSpec.build` and `validate.reconstruction_fidelity`'s ECH
    diagnostic so the two cannot drift apart on what "the right way to
    read this resolver" means - both read `ns.SAMPLING_BY_SOURCE` rather than
    each hand-coding the `resolver in (...)` test that this replaces.

    `"window"` sources (corpus, fdp) get the archive's own 50 ms mean ending
    at `t`; anything else - `"nearest"` (archive) and `"unknown"` (an
    undeclared or future source, and every hand-built `FeatureArray` in the
    test suite that omits `attrs`) - is read nearest-sample. Falling through
    to nearest on `"unknown"` rather than raising is deliberate: raising here
    would cost a whole shot, which the per-shot-isolation rule this module's
    callers all honour forbids (see `validate.reconstruction_fidelity`'s
    docstring). Always returns a 2-D `(channels, len(t))` array - `sample_at`
    already does; `window_mean` is wrapped in `np.atleast_2d` because it
    squeezes a single-channel result to 1-D, and skipping that wrapper would
    let a scalar field's `vals[0]` silently index the first *timestep*
    instead of the whole series.
    """
    convention = ns.SAMPLING_BY_SOURCE.get(resolver, "unknown")
    if convention == "window":
        return np.atleast_2d(window_mean(x, y, t - ARCHIVE_WINDOW_S, ARCHIVE_WINDOW_S))
    # "nearest", and "unknown" as safe fallback. Half a step, not a whole
    # one: it is the correct nearest-neighbour rule (a query is trustworthy
    # only if a real sample lies within half a sampling interval), and a
    # whole step puts the record-edge case exactly on the boundary - for a
    # t+dt field on a 240-row 25 ms record the final query sits 0.025 s past
    # the last sample, so `gap > dt_s` is decided by a 3.5e-16 float residue.
    return sample_at(x, y, t, max_gap=dt_s / 2)


def _nonpositive_to_zero(a: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(a) & (a > 0.0), a, 0.0)


@dataclass(frozen=True)
class Transform:
    """A named pure mapping a spec applies to a field after sampling.

    `fills` is the entire reason this is a class and not a bare callable. It
    says whether the mapping stands in for a reading that cannot be used (a
    FILL, which INVENTS a value) or recovers the physical value from a reading
    that can (a CORRECTION, which does not). Two entries below compute
    identical arithmetic and differ only in this flag, so it is keyword-only
    and has no default: adding a transform forces the author to decide which
    kind it is, and `build` reads the answer to decide whether a row's value
    was measured. A flag that could be forgotten would fail the way this
    module already failed once - silently reporting an invented value as
    measured - so the classification is total by construction.
    """

    fn: Callable[[np.ndarray], np.ndarray]
    fills: bool = field(kw_only=True)


#: Named pure transforms a spec may apply after sampling - named rather than
#: inline lambdas so the model card can list them and a test can compare the
#: card against the spec.
TRANSFORMS: dict[str, Transform] = {
    "reciprocal": Transform(lambda a: 1.0 / a, fills=False),
    # The next two compute the SAME arithmetic and mean different things. Do
    # not merge them: only the name and the `fills` flag distinguish a
    # measurement correction from a fabrication.
    #
    # A CORRECTION. Its input is a reading that means something physical but
    # is out of range - a negative power is sensor baseline noise meaning
    # "off" - so clipping recovers the value rather than inventing one, and
    # the row stays MEASURED. The per-model measurements justifying a
    # particular field's use of it belong in that model's spec and card.
    "clip_negative_to_zero": Transform(_nonpositive_to_zero, fills=False),
    # A FILL. A negative deposition location is not a location, so overwriting
    # one invents a value. This is why train.py:81's filter DROPPED those rows rather
    # than correcting them; labelmaker keeps the row and marks it untrustworthy
    # instead.
    "nonneg_zero_fill": Transform(_nonpositive_to_zero, fills=True),
}

def _sigmoid(a: np.ndarray) -> np.ndarray:
    # `exp` of a large negative logit overflows to inf, which gives the right
    # answer (a probability of 0.0) but raises a RuntimeWarning that the test
    # suite's `-W error` would turn into an exception.
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-a))


ACTIVATIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "none": lambda a: a,
    "sigmoid": _sigmoid,
}

STATS = ("value", "min", "max", "absmax")
NAN_POLICIES = ("zero",)


@dataclass(frozen=True)
class InputField:
    """One model input: its trained-on name and where it now comes from."""

    model_name: str
    canonical: str
    lag: str = "t"
    transform: str | None = None
    scale: float = 1.0
    absent_ok: bool = False

    def __post_init__(self) -> None:
        if self.lag not in ("t", "t+dt"):
            raise ValueError(f"lag must be 't' or 't+dt', got {self.lag!r}")
        if self.transform is not None and self.transform not in TRANSFORMS:
            raise ValueError(f"unknown transform {self.transform!r}")
        if self.absent_ok and self.transform is None:
            raise ValueError(
                f"{self.model_name}: absent_ok needs a transform to turn the "
                "absence into a value"
            )
        ns.by_name(self.canonical)  # fail loudly on a typo, at import time

    @property
    def kind(self) -> str:
        return ns.by_name(self.canonical).kind


@dataclass(frozen=True)
class DomainRule:
    """One clause of the upstream training filter, kept as data.

    `stat` says how a profile is reduced along the radial axis before the
    bound is applied. Rows outside a clause are flagged, never dropped: a
    label file covers the whole shot and says where not to trust it.
    """

    canonical: str
    stat: str = "value"
    lo: float | None = None
    hi: float | None = None
    lo_inclusive: bool = False
    hi_inclusive: bool = False

    def __post_init__(self) -> None:
        if self.stat not in STATS:
            raise ValueError(f"stat must be one of {STATS}, got {self.stat!r}")
        # Model authors write these tuples by hand - the tearing spec alone has
        # thirteen - so a stat that does not match the field's kind is a
        # plausible copy-paste error. Caught here, at import, rather than as an
        # opaque numpy AxisError from reducing a 1-D array along axis 1.
        kind = ns.by_name(self.canonical).kind
        if self.stat == "value" and kind != "scalar":
            raise ValueError(
                f"stat='value' compares a field directly, so it needs a scalar; "
                f"{self.canonical!r} is a profile - reduce it with 'min', 'max' "
                "or 'absmax'"
            )
        if self.stat != "value" and kind != "profile":
            raise ValueError(
                f"stat={self.stat!r} reduces along the radial axis, so it needs a "
                f"profile; {self.canonical!r} is a scalar - use 'value'"
            )


@dataclass(frozen=True)
class UnknownWhenActive:
    """Flag rows where one input was never measured while another was active.

    A gap in an input is sometimes benign and sometimes a fabrication, and
    only a second field can tell you which. An ECH deposition location that
    nobody recorded is harmless while no ECH power is flowing - the
    zero-filled stand-in then matches a state the model trained on - and is a
    fabrication the moment power flows, because the fill asserts the power
    lands on axis. The rule flags the second case and leaves the first alone.

    A row is flagged when the `unknown` field's value was invented AND the
    `active` field either exceeds `threshold` or was itself invented: an
    unknown value is benign only when the partner is *known* to have been
    inactive. The per-model measurements that justify a particular pair
    belong in that model's spec and card, not here.
    """

    unknown: str
    active: str
    threshold: float = 0.0

    def __post_init__(self) -> None:
        for name in (self.unknown, self.active):
            ns.by_name(name)  # fail at import on a typo


@dataclass(frozen=True)
class BuiltInputs:
    """Model-ready arrays for one shot, plus what is trustworthy."""

    t: np.ndarray                  # (T,) seconds - the label time stamps
    scalars: np.ndarray            # (T, n_scalar)
    profiles: np.ndarray           # (T, n_rho, n_profile)
    valid: np.ndarray              # (T,) bool
    missing: tuple[str, ...]
    resolvers: dict[str, str]
    #: rows each rule ALONE rejects, keyed "<canonical> <stat>" for a domain
    #: rule, "<canonical> not finite" for a gap, "<unknown> unknown while
    #: <active> active" for a pair rule; only rules that rejected something.
    #: A row failing two rules is counted under both, so the values do not
    #: sum to the invalid count - they answer "why", not "how many".
    invalid_reasons: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class InputSpec:
    """How to turn a feature file into this model's input arrays."""

    fields: tuple[InputField, ...]
    dt_s: float
    rho_grid: np.ndarray = field(default_factory=lambda: ns.RHO_GRID)
    nan_policy: str = "zero"
    domain: tuple[DomainRule, ...] = ()
    unknown_when_active: tuple[UnknownWhenActive, ...] = ()

    def __post_init__(self) -> None:
        if self.nan_policy not in NAN_POLICIES:
            raise ValueError(
                f"nan_policy must be one of {NAN_POLICIES}, got {self.nan_policy!r}"
            )
        # A domain rule on a canonical that two fields share would check an
        # unspecified one of them. `canonical_names` deduplicates, so the same
        # physical quantity at two lags is an anticipated configuration - make
        # the ambiguity loud rather than arbitrary.
        carried = [f.canonical for f in self.fields]
        for rule in self.domain:
            shared = [f.model_name for f in self.fields if f.canonical == rule.canonical]
            if len(shared) > 1:
                raise ValueError(
                    f"domain rule on {rule.canonical!r} is ambiguous: fields "
                    f"{shared} all use it"
                )
        for pair in self.unknown_when_active:
            for role, name in (("unknown", pair.unknown), ("active", pair.active)):
                if name not in carried:
                    raise ValueError(
                        f"pair rule's {role} names {name!r}, which is not an input "
                        f"of this spec; inputs are {sorted(set(carried))}"
                    )
                if carried.count(name) > 1:
                    raise ValueError(
                        f"pair rule's {role} names {name!r}, which two fields "
                        "carry; disambiguate before adding the rule"
                    )

    @property
    def scalar_fields(self) -> tuple[InputField, ...]:
        return tuple(f for f in self.fields if f.kind == "scalar")

    @property
    def profile_fields(self) -> tuple[InputField, ...]:
        return tuple(f for f in self.fields if f.kind == "profile")

    @property
    def canonical_names(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(f.canonical for f in self.fields))

    def build(self, features: Mapping[str, FeatureArray], grid) -> BuiltInputs:
        """Sample, transform, flag, and stack - in that order.

        The flags are computed on the sampled values *before* the NaN policy
        runs, so a zero-filled hole is never mistaken for a real zero.
        """
        grid = np.asarray(grid, dtype=np.float64)
        n = grid.size
        sampled: dict[str, np.ndarray] = {}
        # What was never measured, recorded before any transform runs - a
        # zero-filling transform destroys exactly this information, and a
        # cross-field rule needs it to tell a benign gap from a fabrication.
        unmeasured: dict[str, np.ndarray] = {}
        missing: list[str] = []
        resolvers: dict[str, str] = {}
        for f in self.fields:
            arr = features.get(f.canonical)
            if arr is None:
                missing.append(f.canonical)
                shape = (n,) if f.kind == "scalar" else (n, self.rho_grid.size)
                v = np.full(shape, np.nan)
                unmeasured[f.model_name] = np.ones(n, dtype=bool)
                if f.absent_ok:
                    # This field says its transform knows how to represent
                    # absence. `unmeasured` still remembers the value was
                    # never measured, so a cross-field rule can adjudicate
                    # whether that matters. Without `absent_ok` the NaN
                    # survives and the finiteness check below invalidates the
                    # row, which is the right default.
                    with np.errstate(divide="ignore", invalid="ignore"):
                        v = TRANSFORMS[f.transform].fn(v)
                    v = v * f.scale
                sampled[f.model_name] = v
                continue
            resolver = str(arr.attrs.get("resolver", "unknown"))
            resolvers[f.canonical] = resolver
            t = grid + self.dt_s if f.lag == "t+dt" else grid
            # Sampling is keyed on the resolver (`sample_by_resolver`, which
            # reads `ns.SAMPLING_BY_SOURCE`) rather than on the feature:
            # the same canonical name means a different sampling rule
            # depending on which source actually produced the stored array.
            # A corpus- or fdp-served field is a true-time, high-rate record
            # and gets windowed into the archive's own 50 ms mean ending at
            # `t`; an archive-served field is read nearest-sample at its own
            # stamp, since it is ALREADY that boxcar - windowing it again
            # would average an already-averaged signal a second time (see
            # `validate.match_rows`, which aligns the un-windowed archive
            # path to 2.3e-7). MEASURED in Task 15, on the corpus
            # (`pinj_total`, `tinj_total`) and fdp (`ip`, `kappa`) paths, to
            # beat the best of three other candidates by 11x to 232x and
            # nearest-sample alone by 98x to 233x - see `ARCHIVE_WINDOW_S`'s
            # docstring for the corrected magnitude and the Task 15 report
            # for the full four-candidate scan.
            #
            # A label stamped at `t` is therefore computed from an input
            # window `[t - ARCHIVE_WINDOW_S, t)` for a windowed source - the
            # label's timestamp trails its input window's centre by half
            # that width. That is uniform across every source, causal, and
            # matches what upstream trained on; it is not corrected here for
            # the same reason `resolve_archive`'s own lag is not: doing so
            # would move every archive-derived figure already published.
            vals = sample_by_resolver(arr.x, arr.y, t, resolver, self.dt_s)
            v = vals[0] if f.kind == "scalar" else vals.T
            gap = ~np.isfinite(v)
            if f.transform is not None:
                with np.errstate(divide="ignore", invalid="ignore"):
                    filled = TRANSFORMS[f.transform].fn(v)
                if TRANSFORMS[f.transform].fills:
                    # `!=` rather than a finiteness test: NaN != NaN is True,
                    # which is wanted, and it also catches a finite reading the
                    # transform overwrote. That second case is the point - a
                    # negative deposition location is a reading, but it is not
                    # a location, so filling it invents a value just as surely
                    # as filling an absent one. An exact 0.0 is left alone,
                    # since the fill does not change it.
                    gap |= filled != v
                v = filled
            unmeasured[f.model_name] = gap if gap.ndim == 1 else gap.any(axis=1)
            sampled[f.model_name] = v * f.scale
        valid, invalid_reasons = self._validity(sampled, n, unmeasured)
        scalars = (
            np.stack([sampled[f.model_name] for f in self.scalar_fields], axis=1)
            if self.scalar_fields
            else np.zeros((n, 0))
        )
        profiles = (
            np.stack([sampled[f.model_name] for f in self.profile_fields], axis=2)
            if self.profile_fields
            else np.zeros((n, self.rho_grid.size, 0))
        )
        if self.nan_policy == "zero":
            scalars = np.nan_to_num(scalars, nan=0.0, posinf=0.0, neginf=0.0)
            profiles = np.nan_to_num(profiles, nan=0.0, posinf=0.0, neginf=0.0)
        return BuiltInputs(
            t=grid,
            scalars=scalars,
            profiles=profiles,
            valid=valid,
            missing=tuple(sorted(set(missing))),
            resolvers=resolvers,
            invalid_reasons=invalid_reasons,
        )

    def _validity(
        self,
        sampled: dict[str, np.ndarray],
        n: int,
        unmeasured: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, dict[str, int]]:
        """The row mask, and how many rows each rule alone rejected."""
        ok = np.ones(n, dtype=bool)
        reasons: dict[str, int] = {}

        def reject(label: str, bad: np.ndarray) -> None:
            nonlocal ok
            if bad.any():
                reasons[label] = int(bad.sum())
            ok &= ~bad

        canonical_of = {f.model_name: f.canonical for f in self.fields}
        for key, v in sampled.items():
            finite = np.isfinite(v) if v.ndim == 1 else np.isfinite(v).all(axis=1)
            reject(f"{canonical_of[key]} not finite", ~finite)
        by_canonical = {f.canonical: f.model_name for f in self.fields}
        # built once and shared with the pair-rule loop below
        for rule in self.domain:
            key = by_canonical.get(rule.canonical)
            if key is None:
                raise KeyError(
                    f"domain rule names {rule.canonical}, which is not an input"
                )
            v = sampled[key]
            # Non-finite rows are already invalid, so filling them keeps the
            # reductions warning-free without changing any verdict.
            v = np.where(np.isfinite(v), v, 0.0)
            if rule.stat == "value":
                red = v
            elif rule.stat == "min":
                red = v.min(axis=1)
            elif rule.stat == "max":
                red = v.max(axis=1)
            else:
                red = np.abs(v).max(axis=1)
            inside = np.ones(n, dtype=bool)
            if rule.lo is not None:
                inside &= (red >= rule.lo) if rule.lo_inclusive else (red > rule.lo)
            if rule.hi is not None:
                inside &= (red <= rule.hi) if rule.hi_inclusive else (red < rule.hi)
            reject(f"{rule.canonical} {rule.stat}", ~inside)
        for pair in self.unknown_when_active:
            gap = unmeasured[by_canonical[pair.unknown]]
            active_gap = unmeasured[by_canonical[pair.active]]
            active = sampled[by_canonical[pair.active]]
            if active.ndim > 1:
                # The fill assumes 0 means inactive, which holds for a
                # non-negative threshold; `max` suffices because `where` has
                # already removed every non-finite channel.
                active = np.where(np.isfinite(active), active, 0.0).max(axis=1)
            # An unknown value is benign only when the partner is known to
            # have been inactive. If the partner was itself never measured,
            # nothing is known about the pair and the row cannot be trusted -
            # otherwise a fill transform on the partner would quietly report
            # "inactive" and license the very gap this rule exists to catch.
            reject(
                f"{pair.unknown} unknown while {pair.active} active",
                gap & (active_gap | (active > pair.threshold)),
            )
        return ok, reasons


@dataclass(frozen=True)
class OutputField:
    """One label a model produces, and which output column carries it."""

    name: str
    task: str
    column: int
    activation: str = "none"
    units: str = ""
    classes: tuple[str, ...] = ()
    attrs: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation {self.activation!r}")


@dataclass(frozen=True)
class Decoded:
    """A label series and the ensemble's spread around it."""

    mean: np.ndarray
    lo: np.ndarray
    hi: np.ndarray


@dataclass(frozen=True)
class OutputSpec:
    fields: tuple[OutputField, ...]

    def decode(self, members: np.ndarray) -> dict[str, Decoded]:
        """`(M, T, C)` raw member outputs -> one `Decoded` per label.

        The activation is applied *after* the ensemble mean, matching the
        upstream harness (`test.py` plots `sigmoid(mean(members))`).
        Averaging probabilities instead would be a different - arguably
        better calibrated - number, but it would not be this model's output.
        """
        members = np.asarray(members, dtype=np.float64)
        if members.ndim != 3:
            raise ValueError(f"expected (M, T, C) members, got {members.shape}")
        out: dict[str, Decoded] = {}
        for f in self.fields:
            col = members[:, :, f.column]
            act = ACTIVATIONS[f.activation]
            out[f.name] = Decoded(
                mean=act(col.mean(axis=0)),
                lo=act(col.min(axis=0)),
                hi=act(col.max(axis=0)),
            )
        return out


@dataclass(frozen=True)
class ModelAdapter:
    """Everything labelmaker needs to run one trained model."""

    slug: str
    card_id: str
    framework: str
    time_step_ms: float
    artifacts: tuple[str, ...]
    upstream: str
    input_spec: InputSpec
    output_spec: OutputSpec
    load: Callable[[Any], Callable[[BuiltInputs], np.ndarray]]
    ensemble_n: int = 1
    #: The shots this model was trained on, where they are known. Empty means
    #: "not recorded", never "none": a model whose training shots are unknown
    #: cannot have its pool numbers split into held-out and in-sample halves,
    #: and `validate` reports every such shot as held out. The survival models
    #: know theirs (`training_shots.txt` beside their spec); the tearing CNN's
    #: training store IS the archive labelmaker scores against, which is a
    #: different problem its own card states.
    training_shots: frozenset[int] = frozenset()
