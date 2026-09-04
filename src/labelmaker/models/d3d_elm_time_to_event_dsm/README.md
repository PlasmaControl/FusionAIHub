---
language: en
license: other
library_name: keras
pipeline_tag: time-series-forecasting
tags:
  - diii-d
  - tokamak
  - elm
  - survival-analysis
datasets:
  - plasmacontrol/d3d-faith-corpus
metrics:
  - roc_auc
model-index:
  - name: d3d-elm-time-to-event-dsm
    results: []
labelmaker:
  status: scaffold
  slug: d3d_elm_time_to_event_dsm
  card_id: plasmacontrol/d3d-elm-time-to-event-dsm
  framework: keras
  time_step_ms: null
  ensemble_n: 1
  upstream:
    path: /projects/EKOLEMEN/wpqh_elm_hiro/hiro_scripts/
    artifacts:
      - wpqh1_embedding_with_norm.keras
      - wpqh1_gate.keras
      - wpqh1_scaleg.keras
      - wpqh1_shapeg.keras
    sha256: {}
  inputs: []
  outputs: []
  approximations: []
  blocked_on:
    - "recover the ~124 trained-on feature names and their order from `hiro_scripts/`"
    - "decide how a deep-survival-machines head (gate, scale, shape) becomes a label series: hazard at fixed horizons, or expected time to event"
    - "the embedding network bakes in its own input normalisation, so the constants need no recovery - confirm that"
---

# plasmacontrol/d3d-elm-time-to-event-dsm

**Status: scaffold.** Labelmaker cannot run this model yet. Nothing here loads
weights and no labels are produced; the folder exists so the roster, the naming
scheme and the known upstream location are recorded in one place.

## Model details

Predicts time to the next ELM from a wide 0-D and profile feature vector, as a mixture of Weibull distributions (deep survival machines). Four saved graphs: an embedding network with normalisation, and gate, scale and shape heads.

## Uses

Intended use is offline label generation over the FAITH shot corpus, for
comparison against IGNITE and against other models' labels. Not for real-time
control and not for physics conclusions without the reliability numbers in the
Evaluation section.

## Bias, risks and limitations

Unmeasured. This model has not been run through labelmaker's validation, so
nothing is known here about how its labels behave on corpus shots.

## Training details

Upstream, outside this repository: `/projects/EKOLEMEN/wpqh_elm_hiro/hiro_scripts/`. Recovering the trained-on
feature list and preprocessing constants is part of the work listed in
`blocked_on`.

## Evaluation

None yet. When implemented, `python -m labelmaker.run validate --models d3d_elm_time_to_event_dsm`
writes adapter fidelity, reconstruction fidelity and label quality into
`model-index` above.

## Technical specifications

Embedding 143,276 parameters; gate/scale/shape ~3,000 each. Legacy Keras format; `runners/keras_h5.py` may need additional layer support.

## Citation

Unpublished internal model. Attribute to the PlasmaControl group, Princeton.

## Contact

`nc1514@princeton.edu`.
