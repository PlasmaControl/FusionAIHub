"""d3d_tearing_time_to_event_dsm - probability of a tearing-mode onset within
250 ms, 500 ms and 1 s, from a Deep Survival Machines model.

Upstream: /projects/EKOLEMEN/survival_tm_2/models/rt_fixed_rot.pkl (the
real-time-signal variant, 2024-11), normalisation constants
/projects/EKOLEMEN/survival_tm/data/rt_normalizations_dict.pkl, inference
reference /projects/EKOLEMEN/survival_tm/get_survival_from_shot.py
(`get_rt_survival_from_shot`), training /projects/EKOLEMEN/survival_tm/train_tm_model.py.

The model takes 14 scalars and 6 profiles at `t`, all real-time EFIT and
fitted-profile quantities. Preprocessing is upstream's, step by step: each
33-point profile is interpolated to 100 points (65 for 1/q and pressure),
projected on a stored 4-component PCA and z-scored; each scalar is z-scored;
the 38 numbers go through a two-layer ReLU6 embedding and the DSM heads
(`runners/dsm_pickle.py`). The label is 1 - S(horizon | x). PCBCOIL is
multiplied by 1.69861e-5 before normalising, as upstream does (the line
commented 'fix BT'); beam power is converted kW -> MW to match `bmspinj`.

Substitutions: offline EFIT01 for every EFITRT2 quantity; ZIPFIT fits for the
mtanh/csaps profiles; labelmaker's 25 ms grid and 50 ms input window for
upstream's 20 ms samples. Unlike the tearing CNN, this model's training rows
are not on disk in a form labelmaker reads, so none of these is priced against
an archive; `validate` cannot score it yet and its card says so.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ...features import namespace as ns
from ..base import InputField, InputSpec, ModelAdapter, OutputField, OutputSpec
from ..runners import dsm_pickle

SLUG = "d3d_tearing_time_to_event_dsm"
CARD_ID = "plasmacontrol/d3d-tearing-time-to-event-dsm"
UPSTREAM = Path("/projects/EKOLEMEN/survival_tm_2/models")
ARTIFACTS = ("rt_fixed_rot.pkl", "rt_normalizations_dict.pkl")
DT_S = 0.025
HORIZONS_MS = (250.0, 500.0, 1000.0)
BT_PER_AMP = 1.69861e-5

INPUT_SPEC = InputSpec(
    fields=(
        # `signals` in get_survival_from_shot.py, in order.
        InputField("bmspinj", "pinj_total", scale=1e-3),          # kW -> MW
        InputField("bmstinj", "tinj_total"),
        InputField("betan_EFITRT2", "betan"),
        InputField("qmin_EFITRT2", "qmin"),
        InputField("ech_pwr_total", "ech_power_total"),
        InputField("ip", "ip"),
        InputField("PCBCOIL", "pcbcoil", scale=BT_PER_AMP),
        InputField("li_EFITRT2", "li"),
        InputField("aminor_EFITRT2", "aminor"),
        InputField("rmaxis_EFITRT2", "r0"),
        InputField("tribot_EFITRT2", "tribot"),
        InputField("tritop_EFITRT2", "tritop"),
        InputField("kappa_EFITRT2", "kappa"),
        InputField("volume_EFITRT2", "volume"),
        # `prof_signals`, in order. q is taken raw here and inverted in
        # `predict`, where upstream's `inf -> 1` convention is applied.
        InputField("thomson_temp_mtanh_1d", "te_zipfit"),
        InputField("cer_temp_csaps_1d", "ti_zipfit"),
        InputField("thomson_density_mtanh_1d", "ne_zipfit"),
        InputField("cer_rot_csaps_1d", "rot_zipfit"),
        InputField("qpsi_EFITRT2", "qpsi"),
        InputField("pres_EFITRT2", "pres"),
    ),
    dt_s=DT_S,
    rho_grid=ns.RHO_GRID,
    nan_policy="zero",
    # Upstream applied no range filter, only `check_all_signals` (every input
    # finite), which is the finiteness part of labelmaker's validity mask.
    domain=(),
)

OUTPUT_SPEC = OutputSpec(
    fields=tuple(
        OutputField(name, "binary", column=i, activation="none",
                    classes=("no_onset", "onset"))
        for i, name in enumerate(("tm_risk_250ms", "tm_risk_500ms", "tm_risk_1s"))
    )
)

#: Normalisation-dict key per profile field, in `prof_signals` order. The
#: rotation profile is normalised under `rotation_kms`.
_NORM_KEYS = ("thomson_temp_mtanh_1d", "cer_temp_csaps_1d", "thomson_density_mtanh_1d",
              "rotation_kms", "qpsi_EFITRT2", "pres_EFITRT2")
_RHO_33 = np.linspace(0.0, 1.0, 33)
_RHO_100 = np.linspace(0.0, 1.0, 100)
_RHO_65 = np.linspace(0.0, 1.0, 65)


def preprocess(built, norm: dict) -> np.ndarray:
    """`(T, 38)` model inputs from built arrays, as get_survival_from_shot.py does it."""
    scalar_names = [f.model_name for f in INPUT_SPEC.scalar_fields]
    mean = np.array([norm[k]["mean"] for k in scalar_names], dtype=np.float64)
    std = np.array([norm[k]["std"] for k in scalar_names], dtype=np.float64)
    cols = [(np.asarray(built.scalars, dtype=np.float64) - mean) / std]
    for j, key in enumerate(_NORM_KEYS):
        prof = np.asarray(built.profiles[:, :, j], dtype=np.float64)
        if key == "qpsi_EFITRT2":
            with np.errstate(divide="ignore"):
                prof = 1.0 / prof
            prof = np.where(prof == np.inf, 1.0, prof)
        grid = _RHO_65 if key in ("qpsi_EFITRT2", "pres_EFITRT2") else _RHO_100
        prof = np.array([np.interp(grid, _RHO_33, row) for row in prof])
        pca = (prof - norm[key]["pca_mean"]) @ np.asarray(norm[key]["pca_matrix"]).T
        cols.append((pca - norm[key]["mean"]) / norm[key]["std"])
    return np.concatenate(cols, axis=1)


def load(model_dir):
    """Read the checkpoint and constants once; return a predictor over BuiltInputs."""
    model_dir = Path(model_dir)
    graph = dsm_pickle.load_dsm(model_dir / ARTIFACTS[0])
    with open(model_dir / ARTIFACTS[1], "rb") as fh:
        norm = dsm_pickle.RestrictedUnpickler(fh).load()

    def predict(built):
        x = preprocess(built, norm)
        risk = 1.0 - dsm_pickle.survival(graph, x, horizons_ms=HORIZONS_MS)
        return risk[None, :, :]                      # one member: (1, T, 3)

    return predict


ADAPTER = ModelAdapter(
    slug=SLUG,
    card_id=CARD_ID,
    framework="dsm_pickle",
    time_step_ms=DT_S * 1000.0,
    artifacts=ARTIFACTS,
    upstream=str(UPSTREAM),
    input_spec=INPUT_SPEC,
    output_spec=OUTPUT_SPEC,
    load=load,
    ensemble_n=1,
)
