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
  results:
  - task:
      type: tabular-regression
      name: betan
    dataset:
      name: d3d overlap shots (n_shots=486/500 requested, n_rows=31257/35776 valid after labeler's
        validity mask)
      type: d3d-faith-corpus
    metrics:
    - name: rmse (archived inputs)
      type: rmse
      value: 0.1198152800935405
  - task:
      type: tabular-classification
      name: tm_prob
    dataset:
      name: d3d overlap shots (n_shots=486/500 requested, n_rows=31257/35776 valid after labeler's
        validity mask)
      type: d3d-faith-corpus
    metrics:
    - name: auroc (archived inputs)
      type: roc_auc
      value: 0.9315146853156802
    - name: f1_at_0.5 (archived inputs)
      type: f1
      value: 0.5211251565880879
  - task:
      type: tabular-regression
      name: betan
    dataset:
      name: d3d overlap shots (n_shots=486/500 requested, n_rows=31257/35776 valid after labeler's
        validity mask)
      type: d3d-faith-corpus
    metrics:
    - name: rmse (reconstructed inputs)
      type: rmse
      value: 0.1572815611360803
  - task:
      type: tabular-classification
      name: tm_prob
    dataset:
      name: d3d overlap shots (n_shots=486/500 requested, n_rows=31257/35776 valid after labeler's
        validity mask)
      type: d3d-faith-corpus
    metrics:
    - name: auroc (reconstructed inputs)
      type: roc_auc
      value: 0.8971832940727098
    - name: f1_at_0.5 (reconstructed inputs)
      type: f1
      value: 0.4858807709547288
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
  - bt <- bt
  - ip <- ip
  - pinj <- pinj_total
  - tinj <- tinj_total
  - R0_EFITRT1 <- r0
  - kappa_EFITRT1 <- kappa
  - tritop_EFIT01 <- tritop
  - tribot_EFIT01 <- tribot
  - gapin_EFIT01 <- gapin
  - ech_pwr_total <- ech_power_total
  - EC.RHO_ECH <- ech_rho
  - thomson_density_mtanh_1d <- ne_zipfit
  - thomson_temp_mtanh_1d <- te_zipfit
  - 1/qpsi_EFITRT1 <- qpsi
  - pres_EFIT01 <- pres
  - cer_rot_csaps_1d <- rot_zipfit
  outputs:
  - name: betan
    task: regression
    activation: none
    units: ''
  - name: tm_prob
    task: binary
    activation: sigmoid
    units: ''
  approximations:
  - 'R0_EFITRT1 and kappa_EFITRT1 are served by offline EFIT01 (rmaxis, kappa); measured against the model''s
    own training inputs over the 100-shot proof-of-concept pool (7,373 matched rows): median relative
    difference 2.5e-3 and 3.3e-3, correlation 0.88 and 0.97'
  - 1/qpsi_EFITRT1 is served by offline EFIT01 qpsi; measured 3.4e-2 median relative difference, correlation
    0.95, over the same pool (243,309 profile points)
  - 'the three kinetic profiles are ZIPFIT fits, not the pipeline''s own mtanh and csaps fits; measured
    against the training inputs over the same pool: 6.7e-2 (ne, correlation 0.92), 1.25e-1 (Te, 0.97)
    and 1.4e-1 (rotation, 0.93) median relative difference. This is the dominant reconstruction error,
    and the rotation profile is absent altogether on ~22% of shots'
  - bt and ip come from the archive where its group carries them (31 of the 100 proof-of-concept shots)
    and from PTDATA through fdp otherwise (69 of 100), windowed into the archive's own 50 ms mean ending
    at t; measured against the training inputs over the pool, 5.3e-6 (bt) and 9.6e-6 (ip) median relative
    difference, so the fdp path is not a visible substitution for these two. pinj, tinj, tritop, tribot,
    gapin, pres and the ECH pair are served by the archive and are exact (median relative difference 0)
  - 'ech_pwr_total is served by the archive column EC.PECH, the machine total. NOT ech_pwr, which is stored
    (1, 240) and holds a single gyrotron (ech_names is length 1: LEIA, LUKE or TINMAN). Established by
    exact row alignment to the model''s own training array: matching x0.npy rows to archive time indices
    on the five bit-identical columns resolves 9805 of 9805 rows uniquely, and at powered rows EC.PECH
    has median relative error 0.097% (91.2% within 1%) against ech_pwr''s 61% (0.0% within 1%). Coverage:
    EC.PECH on 1,299 of the 1,497 PoC-pool shots against ech_pwr''s 384'
  - 'NaN or negative ECH power becomes 0 (upstream rule, train.py:80). This is a CORRECTION, not a fill,
    so a corrected row still counts as a MEASURED power, which is what lets the rule below tell a benign
    gap from a fabrication. On the archive path there is nothing to correct - measured, EC.PECH has zero
    negative readings in 578,160 samples - but on the corpus path ~3.1% of per-gyrotron samples are negative,
    to -112 kW. NOTE: this percentage was measured before the per-resolver sampling fix, which now windows
    every corpus-served field (`ns.SAMPLING_BY_SOURCE`) rather than reading it nearest-sample; it describes
    the raw corpus samples, not a quantity `build()` computes today, and is pending re-measurement under
    the current convention. ECH''s rule, threshold and locator are unchanged and remain on hold - this
    note is about the number''s currency, not about the rule'
  - 'a NaN ECH power is treated as UNMEASURED, which is labeler''s own conservatism and not upstream
    fidelity: train.py:80 clips a NaN to 0 and keeps the row. It coincides with upstream on the archive
    only because the location is NaN on exactly the same rows. EC.PECH is NaN on 56.8% of rows, and 23.0%
    of those NaNs (13.05% of all rows) are interior to the measured window rather than before or after
    the shot'
  - 'a missing or negative EC.RHO_ECH becomes 0, the upstream ECH-off convention, and is then adjudicated
    against the ECH power rather than assumed benign. Measured over the full store: the column is absent
    on 2,591 of 5,000 shots, and negative on 75 of 578,160 readings (0.0130%, in 15 shots) - a negative
    rho is not a location, so the fill invents a value either way. An exact 0.0 is left alone, being the
    genuine ECH-off reading'
  - an unknown deposition location is benign only when the power is KNOWN to have been off. If the power
    itself was never measured, nothing is known about the pair and the row is flagged - otherwise a gap
    in the power would quietly read as inactive and license the very fabrication the rule exists to catch
  - 'EC.PECH and EC.RHO_ECH are co-present by construction: measured over the full store, 2,409 shots
    have both and 2,591 have neither, never one without the other, and on all 2,409 their finite masks
    are identical. So a powered row with no location is rare here - 75 of 56,658 powered rows (0.132%),
    exactly the negative-rho count, since all 75 negative readings carry a finite positive power. What
    the rule mostly does is invalidate the rows where both are NaN, and over all 2,409 paired shots that
    set is IDENTICAL to the set upstream drops via x0[:, 10] >= 0 - zero rows either way'
  - 'the corpus cannot supply a deposition location at all - it has no such group - so on a corpus-served
    shot any row with ECH power flowing is invalid. The corpus is also not interchangeable with EC.PECH
    for power: time-aligned over a random 60 overlap shots, the corpus channel sum is 0.779 of EC.PECH
    at the median (per-shot 0.444 to 1.406), because its 12 channels miss gyrotrons EC.PECH counts. Both
    are in W; the shortfall is content, not units'
  - '49 PoC-pool shots carry only the old ech_pwr column and not EC.PECH, so the archive alone cannot
    serve their ECH power and every row of theirs is invalid: measured, 2,676 valid rows become 0. Falling
    back to ech_pwr would add nothing - measured, all 49 read identically 0.0 W. The rows are recoverable
    from the corpus source instead, and the corpus reads all 12 gyrotrons: it serves 38 of the 49 and
    shows zero power flowing on every one, which CONFIRMS ECH-off rather than assuming it. Upstream, by
    contrast, fabricated a 0.0 for an absent ECH signal and trained on those rows; labeler declines
    to invent the value and takes the evidence from the second source. NOTE: ''38 of the 49 show zero
    power'' was measured against the nearest-sample series `build()` produced before the per-resolver
    sampling fix; `build()` now windows a corpus-served field into the archive''s 50 ms mean instead,
    a different derived series from the same raw samples, so this count is pending re-measurement under
    the current convention. ECH''s rule, threshold and locator are unchanged and remain on hold'
  - inference evaluates the Keras graph in torch (models/runners/keras_h5.py); checked against frozen
    real-TensorFlow outputs (tensorflow-cpu==2.15.1) on 1,673 reference rows x 10 members, gated on a
    scale-normalized max (measured 2.71e-06 against 1e-5), a median absolute difference (7.65e-07 against
    1e-6), and a label-space max on the published post-activation tm_prob (1.24e-06 against 1e-5). The
    raw absolute max is 5.6005e-05 - float32 arithmetic noise, not a semantic error (see labeler.validate.adapter_fidelity)
    - and the operational consequence is 1.2e-06 in published tm_prob probability
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
- A label stamped at time `t` is computed from inputs averaged over
  `[t-50ms, t]`, not from an instantaneous reading at `t`: the label's
  timestamp trails its input window's centre by 25 ms. This is uniform
  across every feature source (archive, corpus, fdp - see Task 15's
  per-resolver sampling fix in `models/base.py`), causal, and matches what
  the model trained on; it compounds with the `t+25 ms` label-semantics
  caveat above rather than replacing it. One row is a further, source-
  dependent special case: at `t=0` the window `[-50ms, 0]` predates the
  record, so an archive-served input (whose row already carries this
  average, computed from raw data that starts well before the shot) is
  measured there while a corpus-served actuator total (whose stored record
  is clipped to start at exactly `t=0`) is not - `models/base.py`'s
  `ARCHIVE_WINDOW_S` comment has the measurement.
- Trained on 2011-2021 shots; corpus shots beyond 190997 are outside the
  training shot range even when their parameters are in domain.
- The three kinetic profiles are substituted (see `approximations`), which is
  the dominant reconstruction error. The measured effect on label quality is in
  the Evaluation section.
- `betan` is a secondary head; the upstream training filter kept only rows with
  `0 < betan < 5`, so predictions far outside that band are unreliable.
- A shot missing any input for its whole record has no valid rows at all - the
  ZIPFIT rotation profile is absent on ~22% of shots - and its labels are
  published entirely as extrapolations: 5 of the 100 proof-of-concept shots.
- Outside the training archive there is no source for the ECH deposition
  location, so every row with ECH power flowing is invalid. On the 2024
  tearing-mode shots 199597-199607 ECH is on for ~90% of plasma rows: they come
  out 4-15% valid, and the ten members disagree widely on them (the ensemble
  spread spans 0 to 1 where the mean sits at 0.2-0.5). Their `tm_prob` peaks
  above 0.9 on 9 of the 11, but by this card's definition it is an
  extrapolation there.

## Training details

Preprocessing, filtering and the ensemble loop are `train.py` in the upstream
directory. Loss: MSE on `betan`, binary cross-entropy `from_logits=True` on the
tearing column, with oversampling and class weighting (the `mse_bin_os_w`
variant). Inputs and outputs both at 25 ms; 0-D inputs read one step ahead of
the profiles.

## Evaluation

Written by `python -m labeler.run validate --models d3d_tearing_onset_cnn1d`
into `model-index` above and, in full, into
`<LABELER_ROOT>/validation/d3d_tearing_onset_cnn1d/`:

- `adapter_fidelity.json` - torch evaluator against frozen real-TensorFlow
  outputs on the upstream reference file: a scale-normalized max, a median
  absolute difference, and a label-space max on the published `tm_prob`
  (measured 2.71e-06, 7.65e-07 and 1.24e-06 against gates of 1e-5, 1e-6 and
  1e-5), plus the raw absolute max (5.6005e-05, float32 arithmetic noise -
  see `labeler.validate.adapter_fidelity`).
- `reconstruction.json` - per-feature agreement between labeler's features
  and the model's own training rows on the corpus/archive overlap shots. On
  the 486 aligned shots of the 500-shot pool: the archive-served columns
  (`pinj`, `tinj`, `tritop`, `tribot`, `gapin`, `pres`, the ECH pair) are
  exact; `bt`/`ip` through fdp 7e-6 and 1e-5; the offline-EFIT01-for-EFITRT1
  substitutions 2.1e-3 (R0), 2.8e-3 (kappa) and 3.0e-2 (1/qpsi); the ZIPFIT
  profiles 6.0e-2 (ne), 1.24e-1 (Te) and 1.24e-1 (rotation) median relative
  difference. (On the 100-shot proof-of-concept pool the same figures were
  2.5e-3, 3.3e-3, 3.4e-2, 6.7e-2, 1.25e-1, 1.4e-1.)
- `label_quality.json` - AUROC, F1 at 0.5, precision, recall, best F1 and
  the threshold reaching it, Brier and calibration against the archived
  labels, scored two ways over the SAME rows: with archived (training) inputs
  and with labeler's own reconstructed inputs, both restricted to the rows
  labeler's own validity rule would actually publish a label for. The
  `model-index` numbers above are this row-matched pair; the difference
  between them is the reconstruction penalty - on the 500-shot pool (486
  aligned), `tm_prob` AUROC 0.932 -> 0.897 (-0.034), best F1 0.576 -> 0.490
  (-0.086, reached at 0.76 and 0.60), and `betan` RMSE 0.120 -> 0.157
  (+0.037) over 31,257 valid rows of 35,776 matched. Best F1 is reported
  because the AUROC gap understates the cost: the reconstruction loses
  positives the model had placed confidently, which a rank statistic barely
  registers. F1 at 0.5 is low on both sides for a reason unrelated to the
  reconstruction: oversampled, class-weighted training makes the model flag
  about twice the base rate of rows at 0.5 (the 0.5-0.6 reliability bin
  observes 8%); the ranking is fine, the operating threshold is a choice.
  `dataset.name` states how many of the requested shots were used and how
  many of the matched rows passed the validity mask - both denominators
  matter. The archived rows are aligned to labeler's timesteps by an exact
  match on the EFIT01 geometry columns `tritop`/`tribot`/`gapin` plus
  whichever of `bt`/`ip` the archive served for that shot, with the
  reconstructed columns breaking exact ties; 486 of the 489 archived shots
  in the pool aligned at median distance zero, and the other three have
  archived rows whose geometry appears nowhere in our EFIT01 series (the
  archive covers times the reconstruction does not). A shot that cannot be
  aligned is skipped and diagnosed, and
  the full JSON also reports each side scored over every matched row
  regardless of validity (`*_all`, diagnostic only, never the published
  number) and a `skip_reasons` histogram with a warning when one cause
  dominates - see `labeler.validate.label_quality`'s docstring for why
  scoring the two inputs over different row sets (an earlier version of this
  card) understates the penalty and can invert which direction a metric moved.

## Technical specifications

- Inputs: `input_1 (None, 11)` float32, `input_2 (None, 33, 5)` float32
- Output: `dense_4 (None, 2)` - column 0 `betan` (linear), column 1 tearing
  logit (apply sigmoid)
- Radial grid: 33 points, `rho = linspace(0, 1, 33)`
- Ensemble: mean over ten members in logit space, then the activation; the
  member min and max are stored as the label's spread
- Framework: Keras 2.8 legacy HDF5, evaluated in torch

## Citation

Unpublished internal model. Attribute to the PlasmaControl group, Princeton.

## Contact

`nc1514@princeton.edu`.
