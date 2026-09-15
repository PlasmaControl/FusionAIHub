"""d3d_tearing_time_to_event_dsm_continued - the same tearing survival model,
trained past the point where its shipped fit stopped.

The shipped checkpoint (`d3d_tearing_time_to_event_dsm`) ran 20 epochs at
lr 1e-5 with its validation NLL still falling monotonically; auton-survival's
early stop never fired. `scripts/labeler/retrain_tearing_dsm.py` continues
that fit from those weights - no pretraining, so the learned per-component
shape/scale parameters survive - at lr 1e-4 until the fork's own patience rule
ends it, and writes `rt_fixed_rot_continued.pkl` beside a copy of the same
normalisation constants.

Everything except the weight file is the base model's: the same 14 scalars and
6 profiles in the same order, the same preprocessing, the same 20 output
columns, the same evaluator. This module therefore imports the base spec's
pieces and builds its loader with the base spec's own `make_load`, so the two
models can never drift apart in anything but their weights. The card carries
the comparison of the two on labeler's 500-shot pool.

The base `load` also reads `calibration.json` from the model directory when one
is there. This model has its own, fitted on its own scores on held-out shots by
`validate.calibration_study`; the base model's maps were fitted on the base
model's scores and never transfer.
"""
from __future__ import annotations

from pathlib import Path

from ..base import ModelAdapter
from ..d3d_tearing_time_to_event_dsm.spec import (
    DT_S,
    HORIZONS_MS,
    INPUT_SPEC,
    OUTPUT_SPEC,
    TRAINING_SHOTS,
    make_load,
    preprocess,
)

__all__ = ["ADAPTER", "ARTIFACTS", "CARD_ID", "HORIZONS_MS", "INPUT_SPEC", "OUTPUT_SPEC",
           "SLUG", "TRAINING_SHOTS", "UPSTREAM", "load", "preprocess"]

SLUG = "d3d_tearing_time_to_event_dsm_continued"
CARD_ID = "plasmacontrol/d3d-tearing-time-to-event-dsm-continued"
#: Trained here, not upstream: the artifact directory is labeler's own.
UPSTREAM = Path("/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/models/"
                "d3d_tearing_time_to_event_dsm_continued")
ARTIFACTS = ("rt_fixed_rot_continued.pkl", "rt_normalizations_dict.pkl")

load = make_load(ARTIFACTS)

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
    # The continuation ran on the base checkpoint's own rows, so its training
    # shots are the base model's: the same file, not a copy of it.
    training_shots=TRAINING_SHOTS,
)
