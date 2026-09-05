"""Building model inputs from canonical features, and reading outputs back."""
import numpy as np
import pytest

from labelmaker.features import namespace as ns
from labelmaker.features.store import FeatureArray
from labelmaker.models.base import (
    ARCHIVE_WINDOW_S,
    TRANSFORMS,
    BuiltInputs,
    Decoded,
    DomainRule,
    InputField,
    InputSpec,
    OutputField,
    OutputSpec,
    Transform,
    UnknownWhenActive,
)
from labelmaker.timebase import sample_at, window_mean

GRID = 0.025 * np.arange(6)          # 0.000 .. 0.125 s


def _scalar_feature(values, resolver="archive"):
    return FeatureArray(
        x=0.025 * np.arange(len(values)),
        y=np.asarray(values, dtype=float)[None, :],
        attrs={"resolver": resolver},
    )


def _profile_feature(rows):
    y = np.asarray(rows, dtype=float).T          # (33, T)
    return FeatureArray(x=0.025 * np.arange(y.shape[1]), y=y, attrs={"resolver": "archive"})


def test_lag_shifts_the_scalar_by_one_step():
    spec = InputSpec(
        fields=(
            InputField("bt_at_t", "bt", lag="t"),
            InputField("bt_next", "ip", lag="t+dt"),
        ),
        dt_s=0.025,
    )
    feats = {
        "bt": _scalar_feature([0, 1, 2, 3, 4, 5, 6]),
        "ip": _scalar_feature([0, 1, 2, 3, 4, 5, 6]),
    }
    built = spec.build(feats, GRID)
    assert built.scalars.shape == (6, 2)
    np.testing.assert_allclose(built.scalars[:, 0], [0, 1, 2, 3, 4, 5])
    np.testing.assert_allclose(built.scalars[:, 1], [1, 2, 3, 4, 5, 6])
    assert built.valid.all()
    assert built.resolvers == {"bt": "archive", "ip": "archive"}


def test_transform_and_scale_are_applied_after_sampling():
    spec = InputSpec(
        fields=(
            InputField("inv_q", "qpsi", transform="reciprocal"),
            InputField("ip_ma", "ip", scale=1e-6),
        ),
        dt_s=0.025,
    )
    q_rows = [np.full(33, 2.0)] * 6
    feats = {"qpsi": _profile_feature(q_rows), "ip": _scalar_feature([1e6] * 6)}
    built = spec.build(feats, GRID)
    assert built.profiles.shape == (6, 33, 1)
    np.testing.assert_allclose(built.profiles[:, :, 0], 0.5)
    np.testing.assert_allclose(built.scalars[:, 0], 1.0)


def test_profile_field_order_is_the_stacking_order():
    spec = InputSpec(
        fields=(
            InputField("ne", "ne_zipfit"),
            InputField("te", "te_zipfit"),
        ),
        dt_s=0.025,
    )
    feats = {
        "ne_zipfit": _profile_feature([np.full(33, 3.0)] * 6),
        "te_zipfit": _profile_feature([np.full(33, 7.0)] * 6),
    }
    built = spec.build(feats, GRID)
    np.testing.assert_allclose(built.profiles[:, :, 0], 3.0)
    np.testing.assert_allclose(built.profiles[:, :, 1], 7.0)


def test_missing_feature_is_recorded_and_zero_filled_but_invalid():
    spec = InputSpec(fields=(InputField("bt", "bt"), InputField("ip", "ip")), dt_s=0.025)
    built = spec.build({"bt": _scalar_feature([1.0] * 6)}, GRID)
    assert built.missing == ("ip",)
    np.testing.assert_allclose(built.scalars[:, 1], 0.0)   # nan_policy="zero"
    assert not built.valid.any()                            # nothing is trustworthy


def test_a_gap_larger_than_one_step_is_not_extrapolated():
    spec = InputSpec(fields=(InputField("bt", "bt"),), dt_s=0.025)
    short = FeatureArray(x=np.array([0.0, 0.025]), y=np.array([[1.0, 2.0]]),
                         attrs={"resolver": "archive"})
    built = spec.build({"bt": short}, GRID)
    assert built.valid.tolist() == [True, True, False, False, False, False]


def test_the_last_step_of_a_t_plus_dt_field_is_never_extrapolated():
    # The real archive record is exactly 240 rows at 25 ms, so a t+dt field's
    # final query lands one step past the end. Upstream training dropped that
    # row for the same reason (x0 = rows[1:], x1 = rows[:-1]); reusing the
    # last row would fabricate a label, so the step must be flagged.
    spec = InputSpec(fields=(InputField("bt", "bt", lag="t+dt"),), dt_s=ns.STEP_S)
    x = ns.STEP_S * np.arange(240)
    feats = {
        "bt": FeatureArray(
            x=x, y=np.arange(240, dtype=float)[None, :], attrs={"resolver": "archive"}
        )
    }
    built = spec.build(feats, ns.GRID_S)
    assert built.valid.sum() == 239
    assert not built.valid[-1]
    assert built.valid[:-1].all()


def test_the_trust_boundary_is_not_decided_by_float_residue():
    # A gap comfortably inside half a step is trusted, one comfortably
    # outside is not, and neither verdict sits on a knife edge.
    spec = InputSpec(fields=(InputField("bt", "bt"),), dt_s=0.025)
    x = np.array([0.0, 0.05])          # 50 ms apart, so 0.025 is the midpoint
    feats = {"bt": FeatureArray(x=x, y=np.array([[1.0, 2.0]]))}
    built = spec.build(feats, np.array([0.010, 0.020]))
    assert built.valid.tolist() == [True, False]


def test_domain_rules_flag_rows_without_dropping_them():
    spec = InputSpec(
        fields=(
            InputField("kappa", "kappa"),
            InputField("ne", "ne_zipfit"),
            InputField("rot", "rot_zipfit"),
        ),
        dt_s=0.025,
        domain=(
            DomainRule("kappa", "value", lo=1.6, hi=2.0),
            DomainRule("ne_zipfit", "min", lo=0.0, lo_inclusive=True),
            DomainRule("ne_zipfit", "max", hi=12.0),
            DomainRule("rot_zipfit", "absmax", hi=150.0),
        ),
    )
    kappa = _scalar_feature([1.8, 1.5, 1.8, 1.8, 1.8, 1.8])
    ne = [np.full(33, 3.0) for _ in range(6)]
    ne[2] = np.full(33, 20.0)                     # too high
    ne[3] = np.full(33, -1.0)                     # negative
    rot = [np.full(33, 10.0) for _ in range(6)]
    rot[4] = np.full(33, -200.0)                  # |rot| too large
    built = spec.build(
        {"kappa": kappa, "ne_zipfit": _profile_feature(ne),
         "rot_zipfit": _profile_feature(rot)},
        GRID,
    )
    assert built.valid.tolist() == [True, False, False, False, False, True]
    assert built.scalars.shape == (6, 1) and built.profiles.shape == (6, 33, 2)


def test_domain_rule_for_an_unused_feature_is_a_programming_error():
    spec = InputSpec(
        fields=(InputField("bt", "bt"),),
        dt_s=0.025,
        domain=(DomainRule("kappa", "value", lo=0.0),),
    )
    with pytest.raises(KeyError, match="kappa"):
        spec.build({"bt": _scalar_feature([1.0] * 6)}, GRID)


def test_a_profile_field_can_also_carry_the_t_plus_dt_lag():
    # The lag arithmetic is kind-agnostic, but only scalars were covered.
    spec = InputSpec(
        fields=(
            InputField("ne_now", "ne_zipfit", lag="t"),
            InputField("ne_next", "te_zipfit", lag="t+dt"),
        ),
        dt_s=0.025,
    )
    rows = [np.full(33, float(i)) for i in range(7)]
    feats = {"ne_zipfit": _profile_feature(rows), "te_zipfit": _profile_feature(rows)}
    built = spec.build(feats, GRID)
    np.testing.assert_allclose(built.profiles[:, 0, 0], [0, 1, 2, 3, 4, 5])
    np.testing.assert_allclose(built.profiles[:, 0, 1], [1, 2, 3, 4, 5, 6])


def test_an_unmeasured_input_is_flagged_only_when_its_partner_is_active():
    # A gap in one input can be benign or a fabrication, and only a second
    # field says which. Here `ech_rho` is never measured; the row survives
    # while power is zero and is flagged once power flows.
    spec = InputSpec(
        fields=(
            InputField("ech_pwr", "ech_power_total", transform="nonneg_zero_fill"),
            InputField("rho", "ech_rho", transform="nonneg_zero_fill"),
        ),
        dt_s=0.025,
        unknown_when_active=(
            UnknownWhenActive(unknown="ech_rho", active="ech_power_total"),
        ),
    )
    t = 0.025 * np.arange(6)
    power = np.array([0.0, 0.0, 0.0, 1.0e6, 1.0e6, 0.0])
    feats = {
        "ech_power_total": FeatureArray(x=t, y=power[None, :]),
        "ech_rho": FeatureArray(x=t, y=np.full((1, 6), np.nan)),
    }
    built = spec.build(feats, t)
    # the zero-fill still happens - the model gets 0.0 either way
    np.testing.assert_allclose(built.scalars[:, 1], 0.0)
    # but the two powered steps are no longer claimed as trustworthy
    assert built.valid.tolist() == [True, True, True, False, False, True]


def test_a_measured_input_is_never_flagged_by_the_pair_rule():
    spec = InputSpec(
        fields=(
            InputField("ech_pwr", "ech_power_total", transform="nonneg_zero_fill"),
            InputField("rho", "ech_rho", transform="nonneg_zero_fill"),
        ),
        dt_s=0.025,
        unknown_when_active=(
            UnknownWhenActive(unknown="ech_rho", active="ech_power_total"),
        ),
    )
    t = 0.025 * np.arange(4)
    feats = {
        "ech_power_total": FeatureArray(x=t, y=np.full((1, 4), 1.0e6)),
        "ech_rho": FeatureArray(x=t, y=np.full((1, 4), 0.35)),
    }
    assert spec.build(feats, t).valid.all()



def test_every_transform_declares_whether_it_fills():
    # `fills` decides whether a row's value counts as measured, so no entry may
    # leave it implicit. A `Transform` cannot be built without it, which is the
    # point: forgetting is a TypeError here rather than a silent claim that an
    # invented value was measured.
    assert all(isinstance(t.fills, bool) for t in TRANSFORMS.values())
    with pytest.raises(TypeError):
        Transform(lambda a: a)                       # no `fills`


def test_a_correction_and_a_fill_computing_the_same_arithmetic_differ():
    # `clip_negative_to_zero` and `nonneg_zero_fill` are the same function.
    # Only the classification separates a corrected reading from an invented
    # one, so pin the behavioural consequence: without this, a well-meaning
    # dedupe of the two entries is a green test run.
    assert TRANSFORMS["clip_negative_to_zero"].fn is TRANSFORMS["nonneg_zero_fill"].fn
    t = 0.025 * np.arange(3)
    feats = {
        "ech_power_total": FeatureArray(x=t, y=np.full((1, 3), 1.0e6)),
        # The negative location below is real data: 75 of 578,160 archive
        # rho readings are negative. The power here is a synthetic +1.0e6,
        # chosen only to make the partner unambiguously active.
        "ech_rho": FeatureArray(x=t, y=np.full((1, 3), -1.0)),
    }
    pair = (UnknownWhenActive(unknown="ech_rho", active="ech_power_total"),)

    def valid_with(transform):
        spec = InputSpec(
            fields=(
                InputField("ech_pwr", "ech_power_total",
                           transform="clip_negative_to_zero"),
                InputField("rho", "ech_rho", transform=transform),
            ),
            dt_s=0.025,
            unknown_when_active=pair,
        )
        return spec.build(feats, t).valid

    # As a fill, overwriting -1.0 invents a location: with power flowing, the
    # row cannot be trusted. As a correction, -1.0 would read as a measured
    # "off" and the row would stand - which for a location is wrong.
    assert not valid_with("nonneg_zero_fill").any()
    assert valid_with("clip_negative_to_zero").all()


def test_a_negative_reading_a_correction_maps_is_still_measured():
    # The other half of the same distinction, on the field that motivated it:
    # a negative ECH power is baseline noise meaning "off", so clipping it
    # must not make the row's partner gap look unadjudicable.
    spec = InputSpec(
        fields=(
            InputField("ech_pwr", "ech_power_total",
                       transform="clip_negative_to_zero"),
            InputField("rho", "ech_rho", transform="nonneg_zero_fill",
                       absent_ok=True),
        ),
        dt_s=0.025,
        unknown_when_active=(
            UnknownWhenActive(unknown="ech_rho", active="ech_power_total"),
        ),
    )
    t = 0.025 * np.arange(4)
    # Synthetic magnitudes; real corpus off-segments reach -112 kW.
    off = np.array([-40.0, -4086.0, 0.0, -1.0])
    built = spec.build({"ech_power_total": FeatureArray(x=t, y=off[None, :])}, t)
    assert built.valid.all(), "a corrected reading is measured, not invented"
    # and a power that was never measured at all still is invented
    nan_power = np.array([np.nan, np.nan, 0.0, -1.0])
    built = spec.build(
        {"ech_power_total": FeatureArray(x=t, y=nan_power[None, :])}, t
    )
    assert built.valid.tolist() == [False, False, True, True]


def test_absent_ok_routes_an_absent_field_through_its_transform():
    # Without absent_ok a wholly absent field stays NaN and invalidates every
    # row, which is the right default. With it, the transform represents the
    # absence and a pair rule decides whether that matters.
    fields = (
        InputField("ech_pwr", "ech_power_total", transform="nonneg_zero_fill"),
        InputField("rho", "ech_rho", transform="nonneg_zero_fill", absent_ok=True),
    )
    pair = (UnknownWhenActive(unknown="ech_rho", active="ech_power_total"),)
    t = 0.025 * np.arange(4)
    off = {"ech_power_total": FeatureArray(x=t, y=np.zeros((1, 4)))}
    on = {"ech_power_total": FeatureArray(x=t, y=np.full((1, 4), 1.0e6))}

    spec = InputSpec(fields=fields, dt_s=0.025, unknown_when_active=pair)
    built = spec.build(off, t)                      # rho absent, power off
    assert built.missing == ("ech_rho",)
    np.testing.assert_allclose(built.scalars[:, 1], 0.0)
    assert built.valid.all(), "a benign absence must not cost the shot its rows"
    assert not spec.build(on, t).valid.any(), "absence with power flowing is a fabrication"


def test_absent_ok_without_a_transform_is_rejected():
    with pytest.raises(ValueError, match="absent_ok"):
        InputField("rho", "ech_rho", absent_ok=True)


def test_an_absent_field_without_absent_ok_still_invalidates_the_row():
    spec = InputSpec(fields=(InputField("bt", "bt"), InputField("ip", "ip")), dt_s=0.025)
    built = spec.build({"bt": _scalar_feature([1.0] * 6)}, GRID)
    assert built.missing == ("ip",) and not built.valid.any()


def test_a_pair_rule_naming_a_non_input_is_rejected_descriptively():
    # The equivalent domain-rule mistake already raises a descriptive error;
    # this one used to surface as a bare KeyError from deep inside build.
    with pytest.raises(ValueError, match="not an input"):
        InputSpec(
            fields=(InputField("bt", "bt"),),
            dt_s=0.025,
            unknown_when_active=(
                UnknownWhenActive(unknown="ech_rho", active="bt"),
            ),
        )


def test_a_pair_rule_on_an_ambiguous_canonical_is_rejected():
    # Same hazard the domain-rule guard exists for: two fields carrying one
    # canonical means the rule would adjudicate against an arbitrary one.
    with pytest.raises(ValueError, match="two fields"):
        InputSpec(
            fields=(
                InputField("bt_now", "bt", lag="t"),
                InputField("bt_next", "bt", lag="t+dt"),
                InputField("ip", "ip"),
            ),
            dt_s=0.025,
            unknown_when_active=(UnknownWhenActive(unknown="ip", active="bt"),),
        )


def test_a_pair_rule_naming_an_unknown_feature_is_rejected():
    with pytest.raises(KeyError):
        UnknownWhenActive(unknown="no_such_feature", active="ech_power_total")


def test_a_domain_rule_whose_stat_mismatches_the_field_kind_is_rejected():
    # Otherwise this surfaces as numpy.exceptions.AxisError from reducing a
    # 1-D array along axis 1, well away from the typo that caused it.
    with pytest.raises(ValueError, match="needs a profile"):
        DomainRule("kappa", "max", hi=2.0)
    with pytest.raises(ValueError, match="needs a scalar"):
        DomainRule("ne_zipfit", "value", hi=12.0)


def test_an_unknown_nan_policy_is_rejected():
    with pytest.raises(ValueError, match="nan_policy"):
        InputSpec(fields=(InputField("bt", "bt"),), dt_s=0.025, nan_policy="zeros")


def test_an_ambiguous_domain_rule_is_rejected():
    with pytest.raises(ValueError, match="ambiguous"):
        InputSpec(
            fields=(
                InputField("bt_now", "bt", lag="t"),
                InputField("bt_next", "bt", lag="t+dt"),
            ),
            dt_s=0.025,
            domain=(DomainRule("bt", "value", lo=0.0),),
        )


def test_bad_field_definitions_are_rejected_at_construction():
    with pytest.raises(ValueError, match="lag"):
        InputField("bt", "bt", lag="tomorrow")
    with pytest.raises(ValueError, match="transform"):
        InputField("bt", "bt", transform="logarithm")


def test_decode_applies_the_activation_after_the_ensemble_mean():
    spec = OutputSpec(
        fields=(
            OutputField("betan", "regression", column=0),
            OutputField("tm_prob", "binary", column=1, activation="sigmoid"),
        )
    )

    def sigmoid(a):
        return 1.0 / (1.0 + np.exp(-np.asarray(a, dtype=float)))

    # Two members, three rows. The tearing logits are deliberately
    # asymmetric: with symmetric logits both orderings collapse to 0.5 and
    # the test would prove nothing.
    members = np.array(
        [[[1.0, 1.0], [2.0, -2.0], [3.0, 0.0]],
         [[3.0, 3.0], [4.0, -1.0], [5.0, 4.0]]]
    )
    got = spec.decode(members)
    np.testing.assert_allclose(got["betan"].mean, [2.0, 3.0, 4.0])
    np.testing.assert_allclose(got["betan"].lo, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(got["betan"].hi, [3.0, 4.0, 5.0])
    # mean over members in logit space, THEN the activation
    np.testing.assert_allclose(got["tm_prob"].mean, sigmoid([2.0, -1.5, 2.0]))
    np.testing.assert_allclose(got["tm_prob"].lo, sigmoid([1.0, -2.0, 0.0]))
    np.testing.assert_allclose(got["tm_prob"].hi, sigmoid([3.0, -1.0, 4.0]))
    # averaging probabilities instead would give a materially different answer
    averaged_probs = (sigmoid([1.0, -2.0, 0.0]) + sigmoid([3.0, -1.0, 4.0])) / 2
    assert np.abs(averaged_probs - got["tm_prob"].mean).max() > 0.01
    assert isinstance(got["tm_prob"], Decoded)


def test_built_inputs_is_the_declared_shape_contract():
    built = BuiltInputs(
        t=GRID, scalars=np.zeros((6, 2)), profiles=np.zeros((6, 33, 3)),
        valid=np.ones(6, bool), missing=(), resolvers={},
    )
    assert built.scalars.shape[0] == built.profiles.shape[0] == built.t.size
    assert built.profiles.shape[1] == ns.RHO_GRID.size


def test_a_non_archive_resolved_field_is_sampled_as_the_archives_window():
    # Per-resolver sampling (Task 15): a field resolved from anywhere but the
    # archive is a true-time record, not yet the 50 ms boxcar the archive's
    # own build averaged over, so it must be turned into that same window -
    # ending exactly at the query time - to be comparable to an
    # archive-resolved field in the same row. MEASURED on both the corpus
    # (`pinj_total`, `tinj_total`) and fdp (`ip`, `kappa`) paths to beat
    # nearest-sample and every other window placement by three to five
    # orders of magnitude; see the Task 15 report for the four-candidate
    # scan.
    spec = InputSpec(fields=(InputField("bt", "bt"),), dt_s=0.025)
    x = 0.001 * np.arange(201)              # 0 .. 0.200 s at 1 ms, high-rate
    y = (10.0 + x)[None, :]                 # a ramp, so window != nearest
    feats = {"bt": FeatureArray(x=x, y=y, attrs={"resolver": "corpus"})}
    grid = np.array([0.100, 0.150])
    built = spec.build(feats, grid)
    want = window_mean(x, y, grid - ARCHIVE_WINDOW_S, ARCHIVE_WINDOW_S)
    np.testing.assert_allclose(built.scalars[:, 0], want)
    # and it must actually differ from nearest-sample, or this could pass by
    # accident on a signal too flat to tell the two conventions apart
    nearest = sample_at(x, y, grid, max_gap=spec.dt_s / 2)[0]
    assert not np.allclose(built.scalars[:, 0], nearest)


def test_an_archive_resolved_field_is_read_nearest_not_windowed_again():
    # The archive path is unchanged by the per-resolver fix: its own row is
    # ALREADY the 50 ms boxcar, so windowing it a second time would average
    # an already-averaged signal.
    spec = InputSpec(fields=(InputField("bt", "bt"),), dt_s=0.025)
    feats = {"bt": _scalar_feature([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], resolver="archive")}
    built = spec.build(feats, GRID)
    np.testing.assert_allclose(built.scalars[:, 0], [1, 2, 3, 4, 5, 6])


def test_a_non_archive_profile_field_is_also_windowed():
    # The per-resolver rule is keyed on the resolver, not the feature kind:
    # a profile served from fdp or the corpus needs the same treatment as a
    # scalar.
    spec = InputSpec(fields=(InputField("ne", "ne_zipfit"),), dt_s=0.025)
    x = 0.001 * np.arange(201)
    ramp = 10.0 + x
    y = np.stack([ramp] * 33, axis=0)       # (33, T), every channel the ramp
    feats = {"ne_zipfit": FeatureArray(x=x, y=y, attrs={"resolver": "fdp"})}
    grid = np.array([0.100, 0.150])
    built = spec.build(feats, grid)
    want = window_mean(x, y, grid - ARCHIVE_WINDOW_S, ARCHIVE_WINDOW_S)
    np.testing.assert_allclose(built.profiles[:, 0, 0], want[0])
    # every radial channel was fed the same ramp, so they must all agree
    np.testing.assert_allclose(
        built.profiles[:, :, 0], np.broadcast_to(want[0][:, None], (2, 33))
    )


def test_invalid_reasons_count_the_rows_each_rule_alone_rejects():
    """A physicist asking "why is this shot 95% invalid" needs the answer per
    rule, not a bool. The 2024 tearing shots came out 4-15% valid because ECH
    power flows with no deposition location on ~90% of rows, and a script was
    needed to find that out; `build` should say it.
    """
    spec = InputSpec(
        fields=(
            InputField("kappa", "kappa"),
            InputField("ne", "ne_zipfit"),
            InputField("pech", "ech_power_total"),
            InputField("rho", "ech_rho", transform="nonneg_zero_fill", absent_ok=True),
        ),
        dt_s=0.025,
        domain=(
            DomainRule("kappa", "value", lo=1.6, hi=2.0),
            DomainRule("ne_zipfit", "max", hi=12.0),
        ),
        unknown_when_active=(
            UnknownWhenActive(unknown="ech_rho", active="ech_power_total"),
        ),
    )
    kappa = _scalar_feature([1.8, 1.5, 1.8, 1.8, 1.8, 1.8])          # row 1
    ne = [np.full(33, 3.0) for _ in range(6)]
    ne[2] = np.full(33, 20.0)                                       # row 2
    pech = _scalar_feature([0.0, 0.0, 0.0, 1e5, 1e5, 0.0])          # rows 3, 4
    built = spec.build(
        {"kappa": kappa, "ne_zipfit": _profile_feature(ne), "ech_power_total": pech},
        GRID,
    )
    assert built.valid.tolist() == [True, False, False, False, False, True]
    assert built.invalid_reasons == {
        "kappa value": 1,
        "ne_zipfit max": 1,
        "ech_rho unknown while ech_power_total active": 2,
    }


def test_invalid_reasons_name_a_non_finite_input_by_its_canonical_feature():
    spec = InputSpec(fields=(InputField("bt", "bt"), InputField("ip", "ip")), dt_s=0.025)
    built = spec.build({"bt": _scalar_feature([1, 1, np.nan, 1, 1, 1])}, GRID)  # ip absent
    assert built.valid.tolist() == [False] * 6
    assert built.invalid_reasons == {"bt not finite": 1, "ip not finite": 6}
