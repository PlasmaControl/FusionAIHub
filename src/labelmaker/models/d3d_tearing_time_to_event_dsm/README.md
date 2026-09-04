---
language: en
license: other
library_name: keras
pipeline_tag: time-series-forecasting
tags:
  - diii-d
  - tokamak
  - tearing-mode
  - survival-analysis
datasets:
  - plasmacontrol/d3d-faith-corpus
metrics:
  - roc_auc
model-index:
  - name: d3d-tearing-time-to-event-dsm
    results: []
labelmaker:
  status: scaffold
  slug: d3d_tearing_time_to_event_dsm
  card_id: plasmacontrol/d3d-tearing-time-to-event-dsm
  framework: keras
  time_step_ms: null
  ensemble_n: 1
  upstream:
    path: /projects/EKOLEMEN/survival_tm/
    artifacts: []
    sha256: {}
  inputs: []
  outputs: []
  approximations: []
  blocked_on:
    - "three candidate directories exist (`survival_tm`, `survival_tm_2`, `survival_tm_depreciated`) - identify the production checkpoint"
    - "recover the trained-on feature list"
    - "same survival-head-to-label-series question as the ELM model"
---

# plasmacontrol/d3d-tearing-time-to-event-dsm

**Status: scaffold.** Labelmaker cannot run this model yet. Nothing here loads
weights and no labels are produced; the folder exists so the roster, the naming
scheme and the known upstream location are recorded in one place.

## Model details

Time to tearing-mode onset as a survival problem, the same family as the ELM model. Complements `d3d_tearing_onset_cnn1d`, which answers the fixed-horizon binary question instead.

## Uses

Intended use is offline label generation over the FAITH shot corpus, for
comparison against IGNITE and against other models' labels. Not for real-time
control and not for physics conclusions without the reliability numbers in the
Evaluation section.

## Bias, risks and limitations

Unmeasured. This model has not been run through labelmaker's validation, so
nothing is known here about how its labels behave on corpus shots.

## Training details

Upstream, outside this repository: `/projects/EKOLEMEN/survival_tm/`. Recovering the trained-on
feature list and preprocessing constants is part of the work listed in
`blocked_on`.

## Evaluation

None yet. When implemented, `python -m labelmaker.run validate --models d3d_tearing_time_to_event_dsm`
writes adapter fidelity, reconstruction fidelity and label quality into
`model-index` above.

## Technical specifications

Not yet inspected.

## Citation

Unpublished internal model. Attribute to the PlasmaControl group, Princeton.

## Contact

`nc1514@princeton.edu`.
