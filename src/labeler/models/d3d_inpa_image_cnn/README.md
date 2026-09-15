---
language: en
license: other
library_name: keras
pipeline_tag: image-to-image
tags:
  - diii-d
  - tokamak
  - inpa
  - fast-ion
  - cnn
datasets:
  - plasmacontrol/d3d-faith-corpus
metrics:
  - rmse
model-index:
  - name: d3d-inpa-image-cnn
    results: []
labelmaker:
  status: scaffold
  slug: d3d_inpa_image_cnn
  card_id: plasmacontrol/d3d-inpa-image-cnn
  framework: keras
  time_step_ms: null
  ensemble_n: 1
  upstream:
    path: /projects/EKOLEMEN/agarcia/inpa_net/
    artifacts:
      - cnn.keras
    sha256: {}
  inputs: []
  outputs: []
  approximations: []
  blocked_on:
    - "**there is no input source**: the corpus has no INPA group, so no corpus shot can be fed to this model. Label generation is impossible until INPA frames are added to the corpus or fetched separately"
    - "recover the expected image size and preprocessing"
---

# plasmacontrol/d3d-inpa-image-cnn

**Status: scaffold.** labeler cannot run this model yet. Nothing here loads
weights and no labels are produced; the folder exists so the roster, the naming
scheme and the known upstream location are recorded in one place.

## Model details

Maps INPA (Imaging Neutral Particle Analyzer) camera frames to a fast-ion phase-space quantity. Listed for completeness of the roster; it is the one candidate whose inputs the corpus does not contain at all.

## Uses

Intended use is offline label generation over the FAITH shot corpus, for
comparison against IGNITE and against other models' labels. Not for real-time
control and not for physics conclusions without the reliability numbers in the
Evaluation section.

## Bias, risks and limitations

Unmeasured. This model has not been run through labeler's validation, so
nothing is known here about how its labels behave on corpus shots.

## Training details

Upstream, outside this repository: `/projects/EKOLEMEN/agarcia/inpa_net/`. Recovering the trained-on
feature list and preprocessing constants is part of the work listed in
`blocked_on`.

## Evaluation

None yet. When implemented, `python -m labeler.run validate --models d3d_inpa_image_cnn`
writes adapter fidelity, reconstruction fidelity and label quality into
`model-index` above.

## Technical specifications

1,441,344 parameters. Legacy Keras format.

## Citation

Unpublished internal model. Attribute to the PlasmaControl group, Princeton.

## Contact

`nc1514@princeton.edu`.
