"""d3d_tearing_onset_cnn1d - tearing-mode onset 25 ms ahead, plus betan.

Upstream: /projects/EKOLEMEN/simple_ae_predictor/models/rt_multi_io/mse_bin_os_w
(ten Keras 2.8 members, `best_model_{i}_4c.h5`), training code `../train.py`,
reference harness `../../test/test.py`.

The model takes eleven 0-D quantities at `t + 25 ms` and five 33-point
profiles at `t`, and emits two columns: `betan` and a tearing logit. Input
names and their order are `train.py:31-32` verbatim; the domain rules are
`train.py:83`. Each branch starts with a BatchNormalization holding the
training-set moving statistics, so no external scaler is needed.

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
            "ech_pwr_total", "ech_power_total", lag="t+dt",
            transform="nonneg_zero_fill",
        ),
        InputField(
            "EC.RHO_ECH", "ech_rho", lag="t+dt", transform="nonneg_zero_fill",
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
    # train.py:83, clause by clause. Rows outside these ranges are labelled
    # anyway and flagged: the model never saw such states in training.
    domain=(
        DomainRule("ne_zipfit", "min", lo=0.0, lo_inclusive=True),
        DomainRule("ne_zipfit", "max", hi=12.0),
        DomainRule("te_zipfit", "min", lo=0.0, lo_inclusive=True),
        DomainRule("te_zipfit", "max", hi=10.0),
        DomainRule("qpsi", "max", hi=3.0),
        DomainRule("pres", "min", lo=0.0, lo_inclusive=True),
        DomainRule("pres", "max", lo=0.0, hi=2.0e5),
        DomainRule("rot_zipfit", "absmax", hi=150.0),
        DomainRule("r0", "value", lo=1.65, hi=1.9),
        DomainRule("kappa", "value", lo=1.6, hi=2.0),
        DomainRule("tritop", "value", lo=0.0, hi=1.0),
        DomainRule("tribot", "value", lo=0.0, hi=1.0),
        DomainRule("gapin", "value", hi=0.2),
        DomainRule("ech_rho", "value", lo=0.0, lo_inclusive=True),
    ),
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
