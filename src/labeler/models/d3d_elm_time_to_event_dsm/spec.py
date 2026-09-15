"""d3d_elm_time_to_event_dsm - probability of an ELM within 5, 10, 20 and 50 ms,
from a Deep Survival Machines model labeler fitted itself.

The weights are **not** upstream's. Upstream's Keras graphs take 124 inputs and
64 of them are BES, which the FAITH corpus fills on 2 of 24 sampled shots, so a
model that needs BES cannot be served at corpus scale. Task 8a therefore fitted
the same architecture (`hiro_scripts/model.cfg`: k=3, layers=[128], LogNormal,
lr 1e-3, batch 1024, dropout 0.2, seed 0) on upstream's own split,
`/projects/EKOLEMEN/wpqh_elm_hiro/data/train_test_split_model10.pkl`, read as-is
and never re-split, twice: once on all 124 columns and once on the 60 that are
not BES. The 60-column fit is what this adapter serves
(`elm_dsm_no_bes.pkl`), and it is the BETTER of the two - see the card.

Column order was the whole difficulty and it is settled by measurement rather
than by reading: the split pickle is in `new_diagnostic_order`
(`elm_inputs.SPLIT_COLUMN_ORDER`), not `current_diagnostic_order`, which puts
ECE last and BES at slots 12-75. See `labeler.models.elm_inputs`.

The chain, per shot:

    11 canonical features on the 25 ms grid
      -> `preprocess`: laid into the 60 columns of `elm_inputs.COLUMN_SETS`
         ["no_bes"], unit-converted to what upstream fitted on
      -> 100 ms boxcar on the `pinj` and `tinj` columns (upstream's own filter;
         NBI is modulated) and mean-fill for any column this corpus cannot serve
      -> (raw - mean) / std from `normalization.json`, upstream's constants
      -> DSM heads (`runners/dsm_pickle.survival`) at `h + 1` ms
      -> `1 - S`                                  -> elm_risk_{5,10,20,50}ms

**Two of the 60 columns are always mean-filled.** `pcphd02` and `pcphd03`, the
two D-alpha photodiodes, come from a PTDATA pickle upstream built by hand
(`data/dalpha_wpqh.pkl`) and have no corpus group at all - the corpus'
`filterscopes` group is a different instrument and is the absent-signal
sentinel on every shot sampled. They are filled at the training mean, i.e.
exactly 0 after normalisation, on every row of every shot. That is NOT recorded
in the per-row `_valid` mask, deliberately: a flag that is 0 on every row of
every shot carries no information and would only hide the rows that are
genuinely untrustworthy. It is a property of this checkpoint's serving path, so
it lives in the card and in each label's `mean_filled_columns` attribute.

A feature that is absent on SOME shots is different and is flagged: `co2_r0`
and its three siblings are absent on half the sampled corpus shots, and there
`InputSpec._validity` marks every row invalid (the raw value was never
measured) while `predict` still mean-fills the column and still emits a
probability, so the series exists and says on its face where not to trust it.

Substitutions from what upstream fitted on, each measured on shot 185808, which
is in both the corpus and the staged `<shot>_slow.h5` store upstream read:

* `bt`: upstream computed `1.69861e-5 * pcbcoil`; labeler uses the canonical
  `bt` in tesla, which is archive- and fdp-served and therefore covers far more
  shots. At t = 2.0 s on shot 185808 the two read -2.0910 T and -2.0306 T, a
  3.0% difference (the archive column is its own 50 ms boxcar);
* `ech`: upstream took column 1 of the staged `ech` group (`echpwrc`);
  labeler uses `ech_power_total`, the 12-gyrotron corpus sum or the
  archive's `EC.PECH`. `namespace.py`'s `ech_power_total` note has the measured
  spread between those two;
* `gas`, `ece`, `co2_<chord>`: same instrument, same channel order, measured
  against the staged groups - see each feature's note in `namespace.py`;
* the grid: upstream's rows are 1 ms means, labeler's are 25 ms. That is a
  sampling change, not a model change; the card says so.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ...config import sha256_of
from ...features import namespace as ns
from ..base import (
    BuiltInputs,
    InputField,
    InputSpec,
    ModelAdapter,
    OutputField,
    OutputSpec,
)
from ..elm_inputs import COLUMN_SETS, SUFFIX
from ..runners import dsm_pickle

SLUG = "d3d_elm_time_to_event_dsm"
CARD_ID = "plasmacontrol/d3d-elm-time-to-event-dsm"
UPSTREAM = Path("/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/models/d3d_elm_time_to_event_dsm")
ARTIFACTS = ("elm_dsm_no_bes.pkl", "normalization.json")
COLUMN_SET = "no_bes"
COLUMNS: tuple[str, ...] = COLUMN_SETS[COLUMN_SET]
DT_S = ns.STEP_S
HORIZONS_MS = (5.0, 10.0, 20.0, 50.0)
#: Upstream fits on `t + 1` (`new_train_elm_model.py`), so a real horizon `h`
#: is queried at `h + 1` ms. Recorded in every `training.json` as well.
T_OFFSET_MS = 1.0
#: The 48 ECE radiometer channels are one `profile` feature whose axis is
#: channel number, not rho. Nothing interpolates it; this only fixes the shape.
ECE_CHANNELS = np.arange(1.0, 49.0)
#: Upstream low-passes the raw `pinj` and `tinj` columns with a 100-sample
#: boxcar on its 1 ms grid, because NBI is modulated
#: (`data_processing.ipynb` cell 34). 100 ms is four samples of labeler's
#: 25 ms grid.
NBI_BOXCAR_MS = 100.0
#: Columns no corpus group can serve, filled at the training mean on every row.
#: See the module docstring for why this is not a per-row validity flag.
ALWAYS_MEAN_FILLED: tuple[str, ...] = ("pcphd02" + SUFFIX, "pcphd03" + SUFFIX)

INPUT_SPEC = InputSpec(
    fields=(
        InputField("ip" + SUFFIX, "ip", scale=1e-6),               # A -> MA
        InputField("bt" + SUFFIX, "bt"),                           # T
        InputField("gas" + SUFFIX, "gas"),                         # V
        InputField("pinj" + SUFFIX, "pinj_total", scale=1e3),      # kW -> W
        InputField("tinj" + SUFFIX, "tinj_total"),                 # N m
        InputField("ech" + SUFFIX, "ech_power_total"),             # W
        *(
            InputField(f"co2_density_slow_{chord}{SUFFIX}", f"co2_{chord}")
            for chord in ("r0", "v1", "v2", "v3")
        ),
        InputField("ece_slow" + SUFFIX, "ece"),                    # 48 channels
    ),
    dt_s=DT_S,
    rho_grid=ECE_CHANNELS,
    nan_policy="zero",
    # Upstream's row filter was a |z| > 10 cut on every normalised column
    # except the two D-alpha photodiodes, plus CO2 outside [0, 1e15]. Neither
    # is expressible as a `DomainRule` on a canonical feature (the first is a
    # post-normalisation rule over 58 columns at once, the second names a
    # column this adapter mean-fills on half the corpus), so both are applied
    # in `predict` as an extra validity narrowing rather than declared here.
    domain=(),
)

OUTPUT_SPEC = OutputSpec(
    fields=tuple(
        OutputField(f"elm_risk_{int(h)}ms", "binary", column=i, activation="none",
                    classes=("no_elm", "elm"))
        for i, h in enumerate(HORIZONS_MS)
    )
)

#: Upstream's own row filter, kept as numbers so the card can quote them.
Z_LIMIT = 10.0
CO2_LO, CO2_HI = 0.0, 1e15


def _boxcar(v: np.ndarray, taps: int) -> np.ndarray:
    """Centred moving average, `np.convolve(..., 'same')` as upstream wrote it."""
    if taps < 2:
        return v
    return np.convolve(v, np.ones(taps) / taps, mode="same")


def preprocess(built: BuiltInputs, norm: dict) -> tuple[np.ndarray, np.ndarray]:
    """`(T, 60)` model inputs and a per-row "every column was measured" mask.

    Columns are addressed by NAME through `COLUMNS`, never by position: the
    upstream reorder that this model's first fit tripped over would then be a
    KeyError here rather than a silently wrong answer.
    """
    index = {name: i for i, name in enumerate(COLUMNS)}
    raw = np.zeros((built.t.size, len(COLUMNS)), dtype=np.float64)
    scalar_fields = INPUT_SPEC.scalar_fields
    for j, f in enumerate(scalar_fields):
        raw[:, index[f.model_name]] = built.scalars[:, j]
    if INPUT_SPEC.profile_fields:                    # `ece`, 48 channels
        ece = built.profiles[:, :, 0]
        for k in range(ece.shape[1]):
            raw[:, index[f"ece_slow_channel_{k + 1}{SUFFIX}"]] = ece[:, k]
    for name in ("pinj", "tinj"):
        col = index[name + SUFFIX]
        raw[:, col] = _boxcar(raw[:, col], round(NBI_BOXCAR_MS / (DT_S * 1e3)))

    mean = np.asarray(norm["mean"], dtype=np.float64)
    std = np.asarray(norm["std"], dtype=np.float64)
    x = (raw - mean) / std

    # Mean-fill: exactly 0 after normalisation. Two columns always (no corpus
    # group exists), plus every column whose feature was absent for this shot -
    # `build` has already turned those into a raw 0.0, which for CO2 line
    # density would normalise to -2.7 rather than to the training mean.
    filled: list[str] = list(ALWAYS_MEAN_FILLED)
    missing = set(built.missing)
    for f in INPUT_SPEC.fields:
        if f.canonical not in missing:
            continue
        if f.kind == "profile":
            filled += [f"ece_slow_channel_{k + 1}{SUFFIX}" for k in range(len(ECE_CHANNELS))]
        else:
            filled.append(f.model_name)
    x[:, [index[name] for name in filled]] = 0.0

    # Upstream's row filter, as a validity narrowing rather than a row drop:
    # |z| > 10 on every column it applied it to (all but the two photodiodes),
    # and CO2 outside [0, 1e15]. A mean-filled column is 0 and passes both, so
    # the filter only ever speaks about columns this shot actually measured.
    checked = [i for name, i in index.items() if name not in set(ALWAYS_MEAN_FILLED)]
    ok = (np.abs(x[:, checked]) <= Z_LIMIT).all(axis=1)
    co2 = [index[f"co2_density_slow_{c}{SUFFIX}"] for c in ("r0", "v1", "v2", "v3")]
    inside = (raw[:, co2] >= CO2_LO) & (raw[:, co2] <= CO2_HI)
    ok &= np.where(np.isin(co2, [index[n] for n in filled]), True, inside).all(axis=1)
    return x, ok


def load(model_dir: Path) -> Callable[[BuiltInputs], np.ndarray]:
    """Read the checkpoint and upstream's normalisation once; return a predictor."""
    model_dir = Path(model_dir)
    graph = dsm_pickle.load_dsm(model_dir / ARTIFACTS[0])
    norm = json.loads((model_dir / ARTIFACTS[1]).read_text())
    if list(norm["columns"]) != list(COLUMNS):
        raise ValueError(
            f"{ARTIFACTS[1]} names {len(norm['columns'])} columns that are not "
            f"{COLUMN_SET}'s; refusing to normalise with the wrong constants"
        )
    checkpoint_sha = sha256_of(model_dir / ARTIFACTS[0])

    def predict(built: BuiltInputs) -> np.ndarray:
        x, ok = preprocess(built, norm)
        # In place: `built` is frozen, and narrowing the mask is what
        # `BuiltInputs`' docstring sanctions a predictor doing.
        np.logical_and(built.valid, ok, out=built.valid)
        risk = 1.0 - dsm_pickle.survival(
            graph, x, horizons_ms=[h + T_OFFSET_MS for h in HORIZONS_MS]
        )
        return risk[None, :, :]                       # one member: (1, T, 4)

    predict.output_spec = OutputSpec(tuple(
        OutputField(
            f.name, f.task, f.column, f.activation, f.units, f.classes,
            f.attrs + (
                ("checkpoint", ARTIFACTS[0]),
                ("checkpoint_sha256", checkpoint_sha),
                ("column_set", COLUMN_SET),
                ("queried_at_ms", str(HORIZONS_MS[f.column] + T_OFFSET_MS)),
                ("mean_filled_columns", ",".join(ALWAYS_MEAN_FILLED)),
                ("trained_grid_ms", "1.0"),
            ),
        )
        for f in OUTPUT_SPEC.fields
    ))
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
    # The split's own `train_final_shots_list` holds 327 shots, but they are
    # wide-pedestal-QH shots from a different era than the FAITH corpus and
    # none of labeler's pool is in it. Left empty rather than half-written:
    # `validate` then reports every shot as held out, which is the honest
    # reading until the list is actually extracted and committed.
    training_shots=frozenset(),
)
