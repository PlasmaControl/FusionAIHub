"""d3d_tearing_onset_cnn1d - tearing-mode onset 25 ms ahead, plus betan.

Upstream: /projects/EKOLEMEN/simple_ae_predictor/models/rt_multi_io/mse_bin_os_w
(ten Keras 2.8 members, `best_model_{i}_4c.h5`), training code `../train.py`,
reference harness `../../test/test.py`.

The model takes eleven 0-D quantities at `t + 25 ms` and five 33-point
profiles at `t`, and emits two columns: `betan` and a tearing logit. Input
names and their order are `train.py:31-32` verbatim; the domain rules are
`train.py:81`. The profile branch opens with a BatchNormalization; the eleven
0-D inputs go straight into the Concatenate and are normalised just after it,
by the BatchNormalization on the 15-vector. Either way the graph normalises
its own inputs, so there is no external scaler to recover.

Substitutions, all measured in validation/d3d_tearing_onset_cnn1d/:
  R0_EFITRT1, kappa_EFITRT1, 1/qpsi_EFITRT1  <- offline EFIT01 equivalents
  thomson_*_mtanh_1d, cer_rot_csaps_1d       <- ZIPFIT fitted profiles
"""
from __future__ import annotations

from pathlib import Path

from ...features import namespace as ns
from ..base import (
    DomainRule,
    InputField,
    InputSpec,
    ModelAdapter,
    OutputField,
    OutputSpec,
    UnknownWhenActive,
)
from ..runners import keras_h5

SLUG = "d3d_tearing_onset_cnn1d"
CARD_ID = "plasmacontrol/d3d-tearing-onset-cnn1d"
UPSTREAM = Path(
    "/projects/EKOLEMEN/simple_ae_predictor/models/rt_multi_io/mse_bin_os_w"
)
ARTIFACTS = tuple(f"best_model_{i}_4c.h5" for i in range(10))
DT_S = 0.025

INPUT_SPEC = InputSpec(
    fields=(
        # 0-D block, at t + dt. Order is x0's column order.
        InputField("bt", "bt", lag="t+dt"),
        InputField("ip", "ip", lag="t+dt"),
        InputField("pinj", "pinj_total", lag="t+dt"),
        InputField("tinj", "tinj_total", lag="t+dt"),
        InputField("R0_EFITRT1", "r0", lag="t+dt"),
        InputField("kappa_EFITRT1", "kappa", lag="t+dt"),
        InputField("tritop_EFIT01", "tritop", lag="t+dt"),
        InputField("tribot_EFIT01", "tribot", lag="t+dt"),
        InputField("gapin_EFIT01", "gapin", lag="t+dt"),
        InputField(
            # A correction, not a fill, following train.py:80, which clipped
            # NaN and negative alike. On the ARCHIVE path there is nothing to
            # correct - MEASURED over the full store, `EC.PECH` has ZERO
            # negative readings in 578,160 samples - but on the CORPUS path
            # there is: ~3.1% of per-gyrotron samples are negative (3.1% over
            # 393 shots, 3.3-3.7% in other draws), down to -112 kW, which is
            # sensor baseline noise meaning "off", so clipping recovers the
            # physical value.
            #
            # A NON-FINITE reading is different - nothing is known - and
            # counts as invented, which is what lets the pair rule below flag
            # a row where neither the power nor the location was measured.
            # NOTE this is labelmaker's own conservatism, NOT upstream
            # fidelity: train.py:80 clips a NaN power to 0 and KEEPS the row,
            # because column 9 is de-NaN'd before the `isnan(x0).sum() == 0`
            # test at train.py:81. It coincides with upstream on the archive
            # only because the location is NaN on exactly the same rows.
            # `EC.PECH` is NaN on 56.8% of rows, and 23.0% of those NaNs
            # (13.05% of all rows) are INTERIOR to the measured window, not
            # merely before or after the shot.
            "ech_pwr_total", "ech_power_total", lag="t+dt",
            transform="clip_negative_to_zero",
        ),
        # absent_ok: the archive omits this column on 2,591 of its 5,000
        # shots, and absence carries no information about whether ECH ran, so
        # it is not assumed benign; it is zero-filled and then adjudicated by
        # the `unknown_when_active` pair below against the power field.
        #
        # `EC.PECH` and `EC.RHO_ECH` come from the same EC subtree and are
        # co-present by construction: MEASURED over the full store, 2,409
        # shots have both and 2,591 have neither - never one without the
        # other - and on all 2,409 their finite masks are IDENTICAL. Two
        # consequences:
        #
        # 1. A row with power flowing but no location is RARE here: 75 of
        #    56,658 powered rows (0.132%). It is exactly the negative-rho
        #    count, and necessarily so - all 75 negative readings have a
        #    finite positive power. An earlier comment claimed 70.1%, which
        #    was measured against `ech_pwr`, a single-gyrotron column whose
        #    coverage differs from the location's.
        # 2. What the rule mostly does here is invalidate the rows where BOTH
        #    are NaN, which upstream also dropped via `x0[:, 10] >= 0`.
        #    Verified: over all 2,409 paired shots, the set labelmaker
        #    invalidates and the set upstream drops are IDENTICAL - zero rows
        #    either way.
        #
        # `ech_rho` has `sources=("archive",)` and the corpus has no
        # deposition-location group at all (its ECH groups are `ech_power`,
        # `ech_pol_angle`, `ech_polarization`, `ech_tor_angle`). So on a
        # corpus-served shot the location is absent, and any row with corpus
        # power flowing is invalid. Whether fdp can fetch a deposition
        # location - what the scaling path needs - has not been investigated;
        # ECH is on hold.
        InputField(
            "EC.RHO_ECH", "ech_rho", lag="t+dt", transform="nonneg_zero_fill",
            absent_ok=True,
        ),
        # profile block, at t. Order is x1's channel order.
        InputField("thomson_density_mtanh_1d", "ne_zipfit", lag="t"),
        InputField("thomson_temp_mtanh_1d", "te_zipfit", lag="t"),
        InputField("1/qpsi_EFITRT1", "qpsi", lag="t", transform="reciprocal"),
        InputField("pres_EFIT01", "pres", lag="t"),
        InputField("cer_rot_csaps_1d", "rot_zipfit", lag="t"),
    ),
    dt_s=DT_S,
    rho_grid=ns.RHO_GRID,
    nan_policy="zero",
    # train.py:81, clause by clause. Rows outside these ranges are labelled
    # anyway and flagged: the model never saw such states in training.
    domain=(
        DomainRule("ne_zipfit", "min", lo=0.0, lo_inclusive=True),
        DomainRule("ne_zipfit", "max", hi=12.0),
        DomainRule("te_zipfit", "min", lo=0.0, lo_inclusive=True),
        DomainRule("te_zipfit", "max", hi=10.0),
        # NB reduces the TRANSFORMED array, so this is max(1/qpsi) < 3 - the
        # upstream clause. Read literally as max(qpsi) < 3 it would flag
        # nearly every H-mode slice, so do not "fix" it.
        DomainRule("qpsi", "max", hi=3.0),
        DomainRule("pres", "min", lo=0.0, lo_inclusive=True),
        DomainRule("pres", "max", lo=0.0, hi=2.0e5),
        DomainRule("rot_zipfit", "absmax", hi=150.0),
        DomainRule("r0", "value", lo=1.65, hi=1.9),
        DomainRule("kappa", "value", lo=1.6, hi=2.0),
        DomainRule("tritop", "value", lo=0.0, hi=1.0),
        DomainRule("tribot", "value", lo=0.0, hi=1.0),
        DomainRule("gapin", "value", hi=0.2),
    ),
    # `nonneg_zero_fill` reproduces the upstream ECH-off convention when the
    # deposition location is unusable. "Unusable" is two cases, not one: the
    # column is absent, or it holds a negative value - MEASURED over the full
    # store, 75 of 578,160 readings (0.0130%) in 15 shots - and a negative rho
    # is not a location, so the fill invents one either way. That rate is
    # heavy-tailed: 2,000-shot draws gave 8, 20, 48 and 60, so it must be
    # quoted from the population, not a sample. Where the location is unknown and power
    # is flowing, the zero-fill would tell the model the power lands on axis;
    # upstream dropped those rows, so the model never saw that state.
    #
    # The power field's `clip_negative_to_zero` is deliberately NOT a fill, so
    # a corrected reading stays MEASURED and this rule can read it. Only a
    # non-finite power counts as invented, and then the row is flagged
    # regardless of the location, because nothing is known about the pair.
    unknown_when_active=(
        UnknownWhenActive(unknown="ech_rho", active="ech_power_total"),
    ),
    # NOTE the removed clause, kept as a comment for the audit trail:
    # train.py:81's `x0[:, 10] >= 0` needs no DomainRule here, because a
    # DomainRule reads the value AFTER the transform and `nonneg_zero_fill`
    # has already mapped every negative and NaN location to 0.0 - so the rule
    # could never fire. The clause is not lost, though: it moved into the
    # pair rule above, which sees that the fill CHANGED the value and so
    # treats the row's location as unknown. Upstream dropped such rows;
    # labelmaker keeps the row, feeds the model the 0.0 that is the upstream
    # ECH-off convention, and marks the row invalid unless the power is known
    # to have been off. Said here rather than left as dead code that looks
    # live.
)

OUTPUT_SPEC = OutputSpec(
    fields=(
        OutputField("betan", "regression", column=0, activation="none", units=""),
        OutputField("tm_prob", "binary", column=1, activation="sigmoid",
                    classes=("no_tearing", "tearing")),
    )
)


def load(model_dir):
    """Load the ten members once and return a predictor over BuiltInputs."""
    graphs = keras_h5.load_ensemble(Path(model_dir) / name for name in ARTIFACTS)

    def predict(built):
        # Positional, not keyed by name: the ten members carry different
        # layer names (input_1/input_2 .. input_19/input_20), so only the
        # order - 0-D block first, profile block second - is common.
        return keras_h5.predict_members(graphs, [built.scalars, built.profiles])

    return predict


ADAPTER = ModelAdapter(
    slug=SLUG,
    card_id=CARD_ID,
    framework="keras_h5",
    time_step_ms=DT_S * 1000.0,
    artifacts=ARTIFACTS,
    upstream=str(UPSTREAM),
    input_spec=INPUT_SPEC,
    output_spec=OUTPUT_SPEC,
    load=load,
    ensemble_n=len(ARTIFACTS),
)
