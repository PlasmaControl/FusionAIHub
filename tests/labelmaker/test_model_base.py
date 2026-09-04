"""Building model inputs from canonical features, and reading outputs back."""
import numpy as np
import pytest

from labelmaker.features import namespace as ns
from labelmaker.features.store import FeatureArray
from labelmaker.models.base import (
    BuiltInputs,
    Decoded,
    DomainRule,
    InputField,
    InputSpec,
    OutputField,
    OutputSpec,
    UnknownWhenActive,
)

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
