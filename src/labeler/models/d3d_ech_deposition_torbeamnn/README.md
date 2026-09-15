---
language: en
license: other
library_name: keras
pipeline_tag: tabular-regression
tags:
  - diii-d
  - tokamak
  - ech
  - torbeam
  - surrogate
datasets:
  - plasmacontrol/d3d-faith-corpus
metrics:
  - rmse
model-index:
  - name: d3d-ech-deposition-torbeamnn
    results: []
labelmaker:
  status: scaffold
  slug: d3d_ech_deposition_torbeamnn
  card_id: plasmacontrol/d3d-ech-deposition-torbeamnn
  framework: keras
  time_step_ms: null
  ensemble_n: 1
  upstream:
    path: /projects/EKOLEMEN/torbeamNN/models_v2/
    artifacts:
      - s1_omode.h5
      - s1_xmode.h5
    sha256: {}
  inputs: []
  outputs: []
  approximations: []
  blocked_on:
    - "recover the input list and ordering from the torbeamNN training code"
    - "O-mode and X-mode are separate graphs, so the adapter needs the per-gyrotron polarisation, which the corpus does carry as `ech_polarization`"
    - "decide whether to emit one label per gyrotron or one aggregate"
---

# plasmacontrol/d3d-ech-deposition-torbeamnn

**Status: scaffold.** labeler cannot run this model yet. Nothing here loads
weights and no labels are produced; the folder exists so the roster, the naming
scheme and the known upstream location are recorded in one place.

## Model details

A neural surrogate for the TORBEAM ray-tracing code: given the equilibrium, profiles and launcher geometry, predicts where ECH power deposits and how much is absorbed. Would give a physics-consistent ECH deposition label for every corpus shot with ECH.

## Uses

Intended use is offline label generation over the FAITH shot corpus, for
comparison against IGNITE and against other models' labels. Not for real-time
control and not for physics conclusions without the reliability numbers in the
Evaluation section.

## Bias, risks and limitations

Unmeasured. This model has not been run through labeler's validation, so
nothing is known here about how its labels behave on corpus shots.

## Training details

Upstream, outside this repository: `/projects/EKOLEMEN/torbeamNN/models_v2/`. Recovering the trained-on
feature list and preprocessing constants is part of the work listed in
`blocked_on`.

## Evaluation

None yet. When implemented, `python -m labeler.run validate --models d3d_ech_deposition_torbeamnn`
writes adapter fidelity, reconstruction fidelity and label quality into
`model-index` above.

## Technical specifications

748,735 (O-mode) and 750,040 (X-mode) parameters. A third, smaller graph (`models/mini_torbeamNN_model_0.h5`, 32,454 parameters) exists and may be the better choice for bulk inference.

## Citation

Unpublished internal model. Attribute to the PlasmaControl group, Princeton.

## Contact

`nc1514@princeton.edu`.
