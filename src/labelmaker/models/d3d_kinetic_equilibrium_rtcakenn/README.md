---
language: en
license: other
library_name: keras
pipeline_tag: tabular-regression
tags:
  - diii-d
  - tokamak
  - equilibrium
  - cake
  - surrogate
datasets:
  - plasmacontrol/d3d-faith-corpus
metrics:
  - roc_auc
model-index:
  - name: d3d-kinetic-equilibrium-rtcakenn
    results: []
labelmaker:
  status: scaffold
  slug: d3d_kinetic_equilibrium_rtcakenn
  card_id: plasmacontrol/d3d-kinetic-equilibrium-rtcakenn
  framework: keras
  time_step_ms: null
  ensemble_n: 1
  upstream:
    path: /projects/EKOLEMEN/rtcakenn_optimization/
    artifacts: []
    sha256: {}
  inputs: []
  outputs: []
  approximations: []
  blocked_on:
    - "four candidate directories exist (`fall2024_rtcakenn`, `rtcakenn_optimization`, `rscake_nn`, `cake_nn`) - identify the production checkpoint and its author"
    - "recover the input list"
    - "profile outputs need canonical *output* features, which the label schema does not yet cover (it stores `(C, T)` per label, so a profile label is `C = n_rho`)"
---

# plasmacontrol/d3d-kinetic-equilibrium-rtcakenn

**Status: scaffold.** Labelmaker cannot run this model yet. Nothing here loads
weights and no labels are produced; the folder exists so the roster, the naming
scheme and the known upstream location are recorded in one place.

## Model details

A real-time surrogate for CAKE kinetic-equilibrium reconstruction. Its outputs are profiles rather than scalars, which makes it the roster's test of whether the label format handles profile-valued labels.

## Uses

Intended use is offline label generation over the FAITH shot corpus, for
comparison against IGNITE and against other models' labels. Not for real-time
control and not for physics conclusions without the reliability numbers in the
Evaluation section.

## Bias, risks and limitations

Unmeasured. This model has not been run through labelmaker's validation, so
nothing is known here about how its labels behave on corpus shots.

## Training details

Upstream, outside this repository: `/projects/EKOLEMEN/rtcakenn_optimization/`. Recovering the trained-on
feature list and preprocessing constants is part of the work listed in
`blocked_on`.

## Evaluation

None yet. When implemented, `python -m labelmaker.run validate --models d3d_kinetic_equilibrium_rtcakenn`
writes adapter fidelity, reconstruction fidelity and label quality into
`model-index` above.

## Technical specifications

Not yet inspected.

## Citation

Unpublished internal model. Attribute to the PlasmaControl group, Princeton.

## Contact

`nc1514@princeton.edu`.
