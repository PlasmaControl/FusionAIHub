---
language: en
license: other
library_name: keras
pipeline_tag: tabular-classification
tags:
  - diii-d
  - tokamak
  - ech
  - multitask
datasets:
  - plasmacontrol/d3d-faith-corpus
metrics:
  - roc_auc
model-index:
  - name: d3d-ech-beam-fate-mlp
    results: []
labelmaker:
  status: scaffold
  slug: d3d_ech_beam_fate_mlp
  card_id: plasmacontrol/d3d-ech-beam-fate-mlp
  framework: keras
  time_step_ms: null
  ensemble_n: 1
  upstream:
    path: /projects/EKOLEMEN/ECH_interlock/models_v15/
    artifacts:
      - multitask_skip.keras
      - norm_stats.npy
    sha256: {}
  inputs: []
  outputs: []
  approximations: []
  blocked_on:
    - "the file is a legacy Keras-2 HDF5 containing `TFOpLambda`, which `runners/keras_h5.py` does not implement - add that layer or re-save the graph once via `tf_keras`"
    - "its profiles are on a 101-point grid, so `ne` and `Te` need their own canonical features rather than the 33-point `RHO_GRID`"
    - "confirm whether the normalisation constants in `norm_stats.npy` are already baked into the graph"
---

# plasmacontrol/d3d-ech-beam-fate-mlp

**Status: scaffold.** Labelmaker cannot run this model yet. Nothing here loads
weights and no labels are produced; the folder exists so the roster, the naming
scheme and the known upstream location are recorded in one place.

## Model details

Classifies the fate of an ECH beam (absorbed, shine-through, or reflected) and regresses absorption efficiency, deposition height and toroidal angle. Four inputs: `ne` (101), `Te` (101), three scalars, fourteen EFIT scalars.

## Uses

Intended use is offline label generation over the FAITH shot corpus, for
comparison against IGNITE and against other models' labels. Not for real-time
control and not for physics conclusions without the reliability numbers in the
Evaluation section.

## Bias, risks and limitations

Unmeasured. This model has not been run through labelmaker's validation, so
nothing is known here about how its labels behave on corpus shots.

## Training details

Upstream, outside this repository: `/projects/EKOLEMEN/ECH_interlock/models_v15/`. Recovering the trained-on
feature list and preprocessing constants is part of the work listed in
`blocked_on`.

## Evaluation

None yet. When implemented, `python -m labelmaker.run validate --models d3d_ech_beam_fate_mlp`
writes adapter fidelity, reconstruction fidelity and label quality into
`model-index` above.

## Technical specifications

186,726 parameters. Four outputs returned as a dict; the config order (`logits, eta, z, phi`) differs from the tensor order, which the adapter must pin explicitly.

## Citation

Unpublished internal model. Attribute to the PlasmaControl group, Princeton.

## Contact

`nc1514@princeton.edu`.
