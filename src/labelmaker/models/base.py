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
from ..timebase import sample_at

#: Named pure transforms a spec may apply after sampling - named rather than
#: inline lambdas so the model card can list them and a test can compare the
#: card against the spec.
TRANSFORMS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "reciprocal": lambda a: 1.0 / a,
    # train.py:81 - NaN or negative ECH power was set to zero before the
    # training filter ran, so those rows were kept. Applied before the
    # validity flags so a zeroed ECH channel does not invalidate a row.
    "nonneg_zero_fill": lambda a: np.where(np.isfinite(a) & (a > 0.0), a, 0.0),
}

ACTIVATIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "none": lambda a: a,
    "sigmoid": lambda a: 1.0 / (1.0 + np.exp(-a)),
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

    def __post_init__(self) -> None:
        if self.lag not in ("t", "t+dt"):
            raise ValueError(f"lag must be 't' or 't+dt', got {self.lag!r}")
        if self.transform is not None and self.transform not in TRANSFORMS:
            raise ValueError(f"unknown transform {self.transform!r}")
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
    only a second field can tell you which. The ECH deposition location is
    the case this exists for. It is absent whenever ECH is off, where the
    zero-fill reproduces the upstream convention and costs nothing - and
    MEASURED over 400 archive shots, 74.7% of its gaps are exactly that. But
    it is also absent on 70.1% of the rows where ECH is genuinely injecting,
    and there the zero-fill tells the model the power lands on axis when
    nobody knows where it lands. Upstream dropped those rows, so the model
    never trained on that state; a label computed from one is an
    extrapolation and must say so.
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
        for rule in self.domain:
            shared = [f.model_name for f in self.fields if f.canonical == rule.canonical]
            if len(shared) > 1:
                raise ValueError(
                    f"domain rule on {rule.canonical!r} is ambiguous: fields "
                    f"{shared} all use it"
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
                sampled[f.model_name] = np.full(shape, np.nan)
                unmeasured[f.model_name] = np.ones(n, dtype=bool)
                continue
            resolvers[f.canonical] = str(arr.attrs.get("resolver", "unknown"))
            t = grid + self.dt_s if f.lag == "t+dt" else grid
            # Half a step, not a whole one, for two reasons. It is the
            # correct nearest-neighbour rule: a query is trustworthy only
            # if a real sample lies within half a sampling interval. And a
            # whole step puts the record-edge case exactly on the boundary
            # - for a t+dt field on a 240-row 25 ms record the final query
            # sits 0.025 s past the last sample, so `gap > dt_s` is decided
            # by a 3.5e-16 float residue. It happens to fall the right way
            # on this data; half a step clears it by 0.0125.
            vals = sample_at(arr.x, arr.y, t, max_gap=self.dt_s / 2)
            v = vals[0] if f.kind == "scalar" else vals.T
            unmeasured[f.model_name] = (
                ~np.isfinite(v) if v.ndim == 1 else ~np.isfinite(v).all(axis=1)
            )
            if f.transform is not None:
                with np.errstate(divide="ignore", invalid="ignore"):
                    v = TRANSFORMS[f.transform](v)
            sampled[f.model_name] = v * f.scale
        valid = self._validity(sampled, n, unmeasured)
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
        )

    def _validity(
        self,
        sampled: dict[str, np.ndarray],
        n: int,
        unmeasured: dict[str, np.ndarray],
    ) -> np.ndarray:
        ok = np.ones(n, dtype=bool)
        for v in sampled.values():
            ok &= np.isfinite(v) if v.ndim == 1 else np.isfinite(v).all(axis=1)
        by_canonical = {f.canonical: f.model_name for f in self.fields}
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
            if rule.lo is not None:
                ok &= (red >= rule.lo) if rule.lo_inclusive else (red > rule.lo)
            if rule.hi is not None:
                ok &= (red <= rule.hi) if rule.hi_inclusive else (red < rule.hi)
        by_canonical = {f.canonical: f.model_name for f in self.fields}
        for pair in self.unknown_when_active:
            gap = unmeasured[by_canonical[pair.unknown]]
            active = sampled[by_canonical[pair.active]]
            if active.ndim > 1:
                active = np.nanmax(np.where(np.isfinite(active), active, 0.0), axis=1)
            ok &= ~(gap & (active > pair.threshold))
        return ok


@dataclass(frozen=True)
class OutputField:
    """One label a model produces, and which output column carries it."""

    name: str
    task: str
    column: int
    activation: str = "none"
    units: str = ""
    classes: tuple[str, ...] = ()

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
