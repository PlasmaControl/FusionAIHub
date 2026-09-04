---
language: en
license: other
library_name: keras
pipeline_tag: tabular-classification
tags:
  - diii-d
  - tokamak
  - plasma
  - tearing-mode
  - neoclassical-tearing-mode
  - cnn
datasets:
  - plasmacontrol/d3d-faith-corpus
metrics:
  - roc_auc
  - f1
model-index:
  - name: d3d-tearing-onset-cnn1d
    results: []
labelmaker:
  status: implemented
  slug: d3d_tearing_onset_cnn1d
  card_id: plasmacontrol/d3d-tearing-onset-cnn1d
  framework: keras_h5
  time_step_ms: 25.0
  ensemble_n: 10
  upstream:
    path: /projects/EKOLEMEN/simple_ae_predictor/models/rt_multi_io/mse_bin_os_w
    trained: 2022-12
    training_code: /projects/EKOLEMEN/simple_ae_predictor/models/rt_multi_io/train.py
    reference_harness: /projects/EKOLEMEN/simple_ae_predictor/test/test.py
    artifacts:
      - best_model_0_4c.h5
      - best_model_1_4c.h5
      - best_model_2_4c.h5
      - best_model_3_4c.h5
      - best_model_4_4c.h5
      - best_model_5_4c.h5
      - best_model_6_4c.h5
      - best_model_7_4c.h5
      - best_model_8_4c.h5
      - best_model_9_4c.h5
    sha256:
      best_model_0_4c.h5: 8ce0711be7477e48ce6b7a342ef2c09342d0b8c7423d6ecb5f85f5bebd3f2b14
      best_model_1_4c.h5: 57fcd5dd2c2b45e9895a40eec072aa74d1cf0de338c11d90c9c44553018d23ed
      best_model_2_4c.h5: 0e19bd488b122713478b8d1c49810d014d874b3bf482176f13225ec11e435e39
      best_model_3_4c.h5: 63adfd5b07d23884fb1cb5fe2eb905c33232665300770aa27b08ac63d9d7468a
      best_model_4_4c.h5: 6e8cf67cfeecc2762f5b7ef985f3f7dd9e3bde3c976534a5d6ce71995ff9f08a
      best_model_5_4c.h5: 8c7eba982f0b5f9832f353ec1b0337ba62efaf51d1cddcd84969f7d48d0ed74e
      best_model_6_4c.h5: 383838eb500c25d95a9380c5e9e935dc1ab721c87017ad209f804467ab8c8d98
      best_model_7_4c.h5: fd1cbcf687090851358ff82ada79de25aefd87c9c68f16343c692912cc8b6aec
      best_model_8_4c.h5: 0b1a6ef87c8469b7cdc8dc3720da13cd97d190172d54f5decff6935282c408ca
      best_model_9_4c.h5: 9d6b4d78f212f0e6915f18cdcf51d990e1d022adb1f877db9e76f73577b4c7da
  inputs:
    - "bt <- bt"
    - "ip <- ip"
    - "pinj <- pinj_total"
    - "tinj <- tinj_total"
    - "R0_EFITRT1 <- r0"
    - "kappa_EFITRT1 <- kappa"
    - "tritop_EFIT01 <- tritop"
    - "tribot_EFIT01 <- tribot"
    - "gapin_EFIT01 <- gapin"
    - "ech_pwr_total <- ech_power_total"
    - "EC.RHO_ECH <- ech_rho"
    - "thomson_density_mtanh_1d <- ne_zipfit"
    - "thomson_temp_mtanh_1d <- te_zipfit"
    - "1/qpsi_EFITRT1 <- qpsi"
    - "pres_EFIT01 <- pres"
    - "cer_rot_csaps_1d <- rot_zipfit"
  outputs:
    - name: betan
      task: regression
      activation: none
      units: ""
    - name: tm_prob
      task: binary
      activation: sigmoid
      units: ""
  approximations:
    - "R0_EFITRT1 and kappa_EFITRT1 are served by offline EFIT01 (rmaxis, kappa);
      measured median relative difference 8.8e-3 and 3.1e-3 on shot 185945"
    - "1/qpsi_EFITRT1 is served by offline EFIT01 qpsi; measured 6.4e-2"
    - "the three kinetic profiles are ZIPFIT fits, not the pipeline's own mtanh
      and csaps fits; measured 2.0e-1 (ne), 1.8e-1 (Te), 1.7e-1 (rotation)
      median relative difference, correlation 0.98-0.99"
    - "ech_pwr_total is served by the archive column EC.PECH, the machine
      total. NOT ech_pwr, which is stored (1, 240) and holds a single
      gyrotron (ech_names is length 1: LEIA, LUKE or TINMAN). Established by
      exact row alignment to the model's own training array: matching x0.npy
      rows to archive time indices on the five bit-identical columns resolves
      9805 of 9805 rows uniquely, and at powered rows EC.PECH has median
      relative error 0.097% (91.2% within 1%) against ech_pwr's 61% (0.0%
      within 1%). Coverage: EC.PECH on 1,299 of the 1,497 PoC-pool shots
      against ech_pwr's 384"
    - "NaN or negative ECH power becomes 0 (upstream rule, train.py:80). This
      is a CORRECTION, not a fill, so a corrected row still counts as a
      MEASURED power, which is what lets the rule below tell a benign gap
      from a fabrication. On the archive path there is nothing to correct -
      measured, EC.PECH has zero negative readings in 578,160 samples - but
      on the corpus path ~3.1% of per-gyrotron samples are negative, to
      -112 kW"
    - "a NaN ECH power is treated as UNMEASURED, which is labelmaker's own
      conservatism and not upstream fidelity: train.py:80 clips a NaN to 0
      and keeps the row. It coincides with upstream on the archive only
      because the location is NaN on exactly the same rows. EC.PECH is NaN on
      56.8% of rows, and 23.0% of those NaNs (13.05% of all rows) are
      interior to the measured window rather than before or after the shot"
    - "a missing or negative EC.RHO_ECH becomes 0, the upstream ECH-off
      convention, and is then adjudicated against the ECH power rather than
      assumed benign. Measured over the full store: the column is absent on
      2,591 of 5,000 shots, and negative on 75 of 578,160 readings (0.0130%,
      in 15 shots) - a negative rho is not a location, so the fill invents a
      value either way. An exact 0.0 is left alone, being the genuine ECH-off
      reading"
    - "an unknown deposition location is benign only when the power is KNOWN
      to have been off. If the power itself was never measured, nothing is
      known about the pair and the row is flagged - otherwise a gap in the
      power would quietly read as inactive and license the very fabrication
      the rule exists to catch"
    - "EC.PECH and EC.RHO_ECH are co-present by construction: measured over
      the full store, 2,409 shots have both and 2,591 have neither, never one
      without the other, and on all 2,409 their finite masks are identical.
      So a powered row with no location is rare here - 75 of 56,658 powered
      rows (0.132%), exactly the negative-rho count, since all 75 negative
      readings carry a finite positive power. What the rule mostly does is
      invalidate the rows where both are NaN, and over all 2,409 paired shots
      that set is IDENTICAL to the set upstream drops via x0[:, 10] >= 0 -
      zero rows either way"
    - "the corpus cannot supply a deposition location at all - it has no such
      group - so on a corpus-served shot any row with ECH power flowing is
      invalid. The corpus is also not interchangeable with EC.PECH for power:
      time-aligned over a random 60 overlap shots, the corpus channel sum is
      0.779 of EC.PECH at the median (per-shot 0.444 to 1.406), because its
      12 channels miss gyrotrons EC.PECH counts. Both are in W; the shortfall
      is content, not units"
    - "49 PoC-pool shots carry only the old ech_pwr column and not EC.PECH,
      so the archive alone cannot serve their ECH power and every row of
      theirs is invalid: measured, 2,676 valid rows become 0. Falling back to
      ech_pwr would add nothing - measured, all 49 read identically 0.0 W.
      The rows are recoverable from the corpus source instead, and the corpus
      reads all 12 gyrotrons: it serves 38 of the 49 and shows zero power
      flowing on every one, which CONFIRMS ECH-off rather than assuming it.
      Upstream, by contrast, fabricated a 0.0 for an absent ECH signal and
      trained on those rows; labelmaker declines to invent the value and
      takes the evidence from the second source"
    - "inference evaluates the Keras graph in numpy (models/runners/keras_h5.py);
      equality with TensorFlow is checked to 1e-5 in validation"
---

# plasmacontrol/d3d-tearing-onset-cnn1d

Predicts, for a DIII-D discharge, the probability that a tearing mode is present
25 ms from now, together with the normalised beta at the same instant.

## Model details

A ten-member ensemble of small multi-input networks (12,086 parameters each).
The profile branch is two `Conv1D` layers over the 33-point radial axis, pooled
and compressed to four numbers; those are concatenated with the eleven 0-D
inputs and passed through three dense layers to a two-column output. Every
block is preceded by a `BatchNormalization` carrying training-set statistics -
though note the eleven 0-D inputs are normalised not before the concatenation
but immediately after it, on the 15-vector. Either way the graph normalises its
own inputs and needs no external scaler.

- Developed by: PlasmaControl group, Princeton (upstream author recorded in
  `PROVENANCE.json`)
- Model type: 1-D CNN plus MLP, multi-input, multi-task
- Trained: December 2022, on 639,555 filtered timeslices from 8,505 DIII-D shots
  (147000-190997)
- Tearing-positive fraction in training: 7.9%

## Uses

Offline label generation over the FAITH shot corpus: a per-shot tearing-mode
probability series at 25 ms resolution that can be compared against IGNITE and
against other models. The `_valid` companion series says where the inputs were
inside the training domain; a probability outside it is an extrapolation.

Not intended for real-time control (the real-time variant of this model is a
different artifact) and not a substitute for magnetics-based mode detection.

## Bias, risks and limitations

- The label answers "is a tearing mode present at t+25 ms", not "will one appear"
  - a mode already present is the easy majority of positives.
- Trained on 2011-2021 shots; corpus shots beyond 190997 are outside the
  training shot range even when their parameters are in domain.
- The three kinetic profiles are substituted (see `approximations`), which is
  the dominant reconstruction error. The measured effect on label quality is in
  the Evaluation section.
- `betan` is a secondary head; the upstream training filter kept only rows with
  `0 < betan < 5`, so predictions far outside that band are unreliable.

## Training details

Preprocessing, filtering and the ensemble loop are `train.py` in the upstream
directory. Loss: MSE on `betan`, binary cross-entropy `from_logits=True` on the
tearing column, with oversampling and class weighting (the `mse_bin_os_w`
variant). Inputs and outputs both at 25 ms; 0-D inputs read one step ahead of
the profiles.

## Evaluation

Written by `python -m labelmaker.run validate --models d3d_tearing_onset_cnn1d`
into `model-index` above and, in full, into
`<LABELMAKER_ROOT>/validation/d3d_tearing_onset_cnn1d/`:

- `adapter_fidelity.json` - numpy evaluator against TensorFlow on the upstream
  reference file, max absolute difference.
- `reconstruction.json` - per-feature agreement between labelmaker's features
  and the model's own training rows on the corpus/archive overlap shots.
- `label_quality.json` - AUROC, F1 at 0.5 and calibration against the archived
  labels, computed twice: with archived inputs and with labelmaker's inputs. The
  difference is the reconstruction penalty.

## Technical specifications

- Inputs: `input_1 (None, 11)` float32, `input_2 (None, 33, 5)` float32
- Output: `dense_4 (None, 2)` - column 0 `betan` (linear), column 1 tearing
  logit (apply sigmoid)
- Radial grid: 33 points, `rho = linspace(0, 1, 33)`
- Ensemble: mean over ten members in logit space, then the activation; the
  member min and max are stored as the label's spread
- Framework: Keras 2.8 legacy HDF5, evaluated in numpy

## Citation

Unpublished internal model. Attribute to the PlasmaControl group, Princeton.

## Contact

`nc1514@princeton.edu`.
