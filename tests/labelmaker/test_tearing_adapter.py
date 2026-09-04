"""The Phase 1 adapter: name mapping, lags, transforms, domain, decode."""
from pathlib import Path

import numpy as np
import pytest

from labelmaker.config import Paths
from labelmaker.features import namespace as ns
from labelmaker.features.store import FeatureArray
from labelmaker.models import registry
from labelmaker.models.d3d_tearing_onset_cnn1d import spec as tm

UPSTREAM = Path(
    "/projects/EKOLEMEN/simple_ae_predictor/models/rt_multi_io/mse_bin_os_w"
)

# train.py:31-32, verbatim.
INPUTS_0D = [
    "bt", "ip", "pinj", "tinj", "R0_EFITRT1", "kappa_EFITRT1",
    "tritop_EFIT01", "tribot_EFIT01", "gapin_EFIT01", "ech_pwr_total",
    "EC.RHO_ECH",
]
INPUTS_1D = [
    "thomson_density_mtanh_1d", "thomson_temp_mtanh_1d", "1/qpsi_EFITRT1",
    "pres_EFIT01", "cer_rot_csaps_1d",
]


def test_field_order_is_the_upstream_column_order():
    got_0d = [f.model_name for f in tm.ADAPTER.input_spec.scalar_fields]
    got_1d = [f.model_name for f in tm.ADAPTER.input_spec.profile_fields]
    assert got_0d == INPUTS_0D
    assert got_1d == INPUTS_1D


def test_lags_and_step():
    assert tm.ADAPTER.input_spec.dt_s == 0.025
    assert tm.ADAPTER.time_step_ms == 25.0
    assert all(f.lag == "t+dt" for f in tm.ADAPTER.input_spec.scalar_fields)
    assert all(f.lag == "t" for f in tm.ADAPTER.input_spec.profile_fields)


def test_outputs_are_betan_then_the_tearing_logit():
    fields = tm.ADAPTER.output_spec.fields
    assert [f.name for f in fields] == ["betan", "tm_prob"]
    assert [f.column for f in fields] == [0, 1]
    assert [f.activation for f in fields] == ["none", "sigmoid"]
    assert [f.task for f in fields] == ["regression", "binary"]


def test_every_upstream_filter_clause_is_a_domain_rule():
    rules = {(r.canonical, r.stat): r for r in tm.ADAPTER.input_spec.domain}
    assert rules[("ne_zipfit", "min")].lo == 0.0
    assert rules[("ne_zipfit", "min")].lo_inclusive
    assert rules[("ne_zipfit", "max")].hi == 12.0
    assert rules[("te_zipfit", "max")].hi == 10.0
    assert rules[("qpsi", "max")].hi == 3.0
    assert rules[("pres", "max")].lo == 0.0 and rules[("pres", "max")].hi == 2e5
    assert rules[("rot_zipfit", "absmax")].hi == 150.0
    assert (rules[("r0", "value")].lo, rules[("r0", "value")].hi) == (1.65, 1.9)
    assert (rules[("kappa", "value")].lo, rules[("kappa", "value")].hi) == (1.6, 2.0)
    assert (rules[("tritop", "value")].lo, rules[("tritop", "value")].hi) == (0.0, 1.0)
    assert (rules[("tribot", "value")].lo, rules[("tribot", "value")].hi) == (0.0, 1.0)
    assert rules[("gapin", "value")].hi == 0.2
    assert rules[("te_zipfit", "min")].lo == 0.0
    assert rules[("te_zipfit", "min")].lo_inclusive
    # No ech_rho rule: nonneg_zero_fill subsumes the upstream clause, so one
    # could never fire. See the comment in spec.py.
    assert ("ech_rho", "value") not in rules


def test_a_fabricated_ech_location_is_flagged_when_power_is_flowing():
    # MEASURED over 400 archive shots: 70.1% of ECH-powered rows have no
    # recorded deposition location. Zero-filling those says "on axis", a
    # state upstream dropped from training - so the row must not be claimed
    # as valid. With ECH off, the same gap is the upstream convention and
    # the row stands.
    for power, expect_valid in ((0.0, True), (1.0e6, False)):
        feats, grid = _features()
        t = feats["ech_rho"].x
        feats["ech_rho"] = FeatureArray(
            x=t, y=np.full((1, t.size), np.nan), attrs={"resolver": "archive"}
        )
        feats["ech_power_total"] = FeatureArray(
            x=t, y=np.full((1, t.size), power), attrs={"resolver": "archive"}
        )
        built = tm.ADAPTER.input_spec.build(feats, grid)
        np.testing.assert_allclose(built.scalars[:, 10], 0.0)   # zero-filled either way
        assert bool(built.valid.all()) is expect_valid, power


def test_the_qpsi_rule_bounds_the_reciprocal_not_qpsi_itself():
    # max(1/qpsi) < 3 flags a low-q profile and admits a high-q one. Asserting
    # only `hi == 3.0` cannot tell this from the opposite reading.
    for qpsi, expect_valid in ((5.0, True), (2.5, True), (0.3, False)):
        feats, grid = _features()
        t = feats["qpsi"].x
        feats["qpsi"] = FeatureArray(
            x=t, y=np.full((33, t.size), qpsi), attrs={"resolver": "archive"}
        )
        built = tm.ADAPTER.input_spec.build(feats, grid)
        assert bool(built.valid.all()) is expect_valid, qpsi


def _features(n=8, *, ech_nan=False):
    t = ns.STEP_S * np.arange(n + 1)          # one extra step for the t+dt lag
    scalars = {
        "bt": 2.0, "ip": 1.0e6, "pinj_total": 5000.0, "tinj_total": 4.0,
        "r0": 1.75, "kappa": 1.8, "tritop": 0.4, "tribot": 0.3,
        "gapin": 0.05, "ech_power_total": 1.0e6, "ech_rho": 0.4,
    }
    out = {
        name: FeatureArray(x=t, y=np.full((1, t.size), v), attrs={"resolver": "archive"})
        for name, v in scalars.items()
    }
    if ech_nan:
        y = np.full((1, t.size), np.nan)
        out["ech_power_total"] = FeatureArray(x=t, y=y, attrs={"resolver": "archive"})
    profiles = {
        "ne_zipfit": 3.0, "te_zipfit": 2.0, "qpsi": 2.5, "pres": 5.0e4,
        "rot_zipfit": 50.0,
    }
    for name, v in profiles.items():
        out[name] = FeatureArray(
            x=t, y=np.full((33, t.size), v), attrs={"resolver": "archive"}
        )
    return out, ns.STEP_S * np.arange(n)


def test_build_produces_the_model_input_shapes():
    feats, grid = _features()
    built = tm.ADAPTER.input_spec.build(feats, grid)
    assert built.scalars.shape == (8, 11)
    assert built.profiles.shape == (8, 33, 5)
    assert built.valid.all()
    assert built.missing == ()
    np.testing.assert_allclose(built.scalars[:, 0], 2.0)          # bt
    np.testing.assert_allclose(built.profiles[:, :, 2], 1 / 2.5)  # 1/qpsi
    np.testing.assert_allclose(built.profiles[:, :, 3], 5.0e4)    # pres, unscaled


def test_nan_ech_power_becomes_zero_without_invalidating_the_row():
    # Upstream rule (train.py:81): NaN or negative ECH power is set to zero
    # *before* the filter runs, so those rows survived training.
    feats, grid = _features(ech_nan=True)
    built = tm.ADAPTER.input_spec.build(feats, grid)
    np.testing.assert_allclose(built.scalars[:, 9], 0.0)
    assert built.valid.all()


def test_negative_ech_power_also_becomes_zero():
    feats, grid = _features()
    t = feats["ech_power_total"].x
    feats["ech_power_total"] = FeatureArray(
        x=t, y=np.full((1, t.size), -1.0e5), attrs={"resolver": "archive"}
    )
    built = tm.ADAPTER.input_spec.build(feats, grid)
    np.testing.assert_allclose(built.scalars[:, 9], 0.0)
    assert built.valid.all()


def test_a_finite_but_nonsensical_ech_location_becomes_zero_and_stays_valid():
    # -1.0 and -1e9 are MEASURED readings (finite), just physically
    # impossible, so nonneg_zero_fill zeroes them and the row stays usable -
    # there is no ech_rho DomainRule. NaN is a different case: it means the
    # deposition location was never measured at all, and whether that is
    # benign or a fabrication now depends on whether ECH power is flowing -
    # see test_a_fabricated_ech_location_is_flagged_when_power_is_flowing.
    for value in (-1.0, -1e9):
        feats, grid = _features()
        t = feats["ech_rho"].x
        feats["ech_rho"] = FeatureArray(
            x=t, y=np.full((1, t.size), value), attrs={"resolver": "archive"}
        )
        built = tm.ADAPTER.input_spec.build(feats, grid)
        np.testing.assert_allclose(built.scalars[:, 10], 0.0)
        assert built.valid.all(), value


def test_out_of_domain_kappa_is_flagged():
    feats, grid = _features()
    t = feats["kappa"].x
    y = np.full((1, t.size), 1.8)
    y[0, 4] = 1.2                                   # below the 1.6 floor
    feats["kappa"] = FeatureArray(x=t, y=y, attrs={"resolver": "archive"})
    built = tm.ADAPTER.input_spec.build(feats, grid)
    # kappa is a t+dt field, so grid step i reads feature sample i+1:
    # spoiling sample 4 flags grid step 3, not grid step 4.
    assert built.valid.sum() == 7 and not built.valid[3]


requires_upstream = pytest.mark.skipif(
    not UPSTREAM.exists(), reason=f"upstream weights not available: {UPSTREAM}"
)


def test_card_matches_the_spec():
    # Deliberately NOT gated on the upstream mount: this reads only the card
    # and the spec, so it is a pure-repo invariant. Gating it would let card
    # drift pass unnoticed in exactly the environments that lack /projects.
    assert registry.card_discrepancies("d3d_tearing_onset_cnn1d") == []
    assert registry.implemented() == ["d3d_tearing_onset_cnn1d"]


@requires_upstream
def test_predict_runs_end_to_end_from_the_upstream_directory():
    predict = tm.load(UPSTREAM)
    feats, grid = _features()
    built = tm.ADAPTER.input_spec.build(feats, grid)
    members = predict(built)
    assert members.shape == (10, 8, 2)
    assert np.isfinite(members).all()
    decoded = tm.ADAPTER.output_spec.decode(members)
    assert set(decoded) == {"betan", "tm_prob"}
    assert ((decoded["tm_prob"].mean >= 0) & (decoded["tm_prob"].mean <= 1)).all()
    assert (decoded["tm_prob"].lo <= decoded["tm_prob"].mean).all()
    assert (decoded["tm_prob"].hi >= decoded["tm_prob"].mean).all()


@pytest.mark.skipif(
    not (Paths.from_env().models / "d3d_tearing_onset_cnn1d").exists(),
    reason="weights not yet copied into the data root",
)
def test_verification_passes_against_the_copy_inference_will_load():
    # Every other weights test reads the upstream directory, but the global
    # constraint is that inference reads the copy in the data root. Exercise
    # those bytes.
    registry.verify_artifacts(
        "d3d_tearing_onset_cnn1d", Paths.from_env().models / "d3d_tearing_onset_cnn1d"
    )


@requires_upstream
def test_sha256_verification_rejects_a_tampered_artifact(tmp_path):
    import shutil

    for name in tm.ARTIFACTS:
        shutil.copy(UPSTREAM / name, tmp_path / name)
    registry.verify_artifacts("d3d_tearing_onset_cnn1d", tmp_path)  # passes
    with open(tmp_path / tm.ARTIFACTS[0], "ab") as fh:
        fh.write(b"\x00")
    # "sha256 mismatch", not "sha256": the latter also matches the
    # truncated-card error, so this test could pass on the wrong failure.
    with pytest.raises(ValueError, match="sha256 mismatch"):
        registry.verify_artifacts("d3d_tearing_onset_cnn1d", tmp_path)
