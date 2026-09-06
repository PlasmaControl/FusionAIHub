---
language: en
license: other
library_name: pytorch
pipeline_tag: tabular-classification
tags:
  - diii-d
  - tokamak
  - tearing-mode
  - survival-analysis
  - deep-survival-machines
datasets:
  - plasmacontrol/d3d-faith-corpus
metrics:
  - roc_auc
model-index:
  - name: d3d-tearing-time-to-event-dsm
    results: []
labelmaker:
  status: implemented
  slug: d3d_tearing_time_to_event_dsm
  card_id: plasmacontrol/d3d-tearing-time-to-event-dsm
  framework: dsm_pickle
  time_step_ms: 25.0
  ensemble_n: 1
  upstream:
    path: /projects/EKOLEMEN/survival_tm_2/models
    trained: 2024-11
    training_code: /projects/EKOLEMEN/survival_tm/train_tm_model.py
    reference_harness: /projects/EKOLEMEN/survival_tm/get_survival_from_shot.py
    artifacts:
    - rt_fixed_rot.pkl
    - rt_normalizations_dict.pkl
    sha256:
      rt_fixed_rot.pkl: 2a7b65a9e7484c917c92d8394e5a4bce5a2b09e027da95e3832dd5f8b2d16f5b
      rt_normalizations_dict.pkl: fa515c7b591f3e1ea5f710d75a825b1a7831cbd63e06543f9ccb87ff87a73fbe
    notes: >-
      rt_normalizations_dict.pkl lives in /projects/EKOLEMEN/survival_tm/data/ upstream.
      Training shots: 8,923 unique DIII-D shots, 140444-193373, over 914,898 training rows,
      read from /projects/EKOLEMEN/survival_tm_2/data/rt_filtered_shots_pcb_rot.pkl
      (sha256 f89286ed88bdf20bfa6af0abd49a881e7902f77b78e6ecf811f3d9aab5f03288) and committed
      beside this card as training_shots.txt by scripts/labelmaker/write_training_shots.py.
      214 of labelmaker's 500 pool shots, 208 of the 463 scored shots and 41 of the 80 onset
      shots are in that list; every pool number below is split on it.
  inputs:
  - bmspinj <- pinj_total
  - bmstinj <- tinj_total
  - betan_EFITRT2 <- betan
  - qmin_EFITRT2 <- qmin
  - ech_pwr_total <- ech_power_total
  - ip <- ip
  - PCBCOIL <- pcbcoil
  - li_EFITRT2 <- li
  - aminor_EFITRT2 <- aminor
  - rmaxis_EFITRT2 <- r0
  - tribot_EFITRT2 <- tribot
  - tritop_EFITRT2 <- tritop
  - kappa_EFITRT2 <- kappa
  - volume_EFITRT2 <- volume
  - thomson_temp_mtanh_1d <- te_zipfit
  - cer_temp_csaps_1d <- ti_zipfit
  - thomson_density_mtanh_1d <- ne_zipfit
  - cer_rot_csaps_1d <- rot_zipfit
  - qpsi_EFITRT2 <- qpsi
  - pres_EFITRT2 <- pres
  outputs:
  - name: tm_risk_250ms
    task: binary
    activation: none
    units: ''
  - name: tm_risk_500ms
    task: binary
    activation: none
    units: ''
  - name: tm_risk_1s
    task: binary
    activation: none
    units: ''
  - name: tm_time_p10
    task: regression
    activation: none
    units: 'ms'
  - name: tm_time_p50
    task: regression
    activation: none
    units: 'ms'
  - name: tm_time_p90
    task: regression
    activation: none
    units: 'ms'
  - name: tm_time_iqr_log
    task: regression
    activation: none
    units: ''
  - name: tm_mix_w0
    task: regression
    activation: none
    units: ''
  - name: tm_mix_w1
    task: regression
    activation: none
    units: ''
  - name: tm_mix_w2
    task: regression
    activation: none
    units: ''
  - name: tm_mix_mu0
    task: regression
    activation: none
    units: 'ln ms'
  - name: tm_mix_mu1
    task: regression
    activation: none
    units: 'ln ms'
  - name: tm_mix_mu2
    task: regression
    activation: none
    units: 'ln ms'
  - name: tm_mix_sigma0
    task: regression
    activation: none
    units: ''
  - name: tm_mix_sigma1
    task: regression
    activation: none
    units: ''
  - name: tm_mix_sigma2
    task: regression
    activation: none
    units: ''
  - name: tm_gate_entropy
    task: regression
    activation: none
    units: 'nat'
  - name: tm_risk_250ms_isotonic
    task: binary
    activation: none
    units: ''
  - name: tm_risk_500ms_isotonic
    task: binary
    activation: none
    units: ''
  - name: tm_risk_1s_isotonic
    task: binary
    activation: none
    units: ''
  approximations:
  - every EFITRT2 quantity (betan, qmin, li, aminor, rmaxis, tribot, tritop, kappa, volume, qpsi, pres) is
    served by offline EFIT01; for the four quantities the tearing CNN also uses, that substitution was
    measured at 2.1e-3 (R0), 2.8e-3 (kappa) and 3.0e-2 (1/q) median relative difference against real-time
    EFIT on 486 shots; the others are unpriced
  - the four kinetic profiles (Te, Ti, ne, rotation) are ZIPFIT fits standing in for the pipeline's own mtanh
    and csaps fits; unpriced for this model (its training rows are not on disk in a form labelmaker reads);
    for the three the tearing CNN shares, 6.0e-2 to 1.24e-1 median relative difference
  - inputs are sampled on labelmaker's 25 ms grid as 50 ms window means, where upstream used the real-time
    data dictionary's 20 ms samples; the model has no temporal structure, so this changes which instants are
    labelled, not how
  - the rotation profile is fed in the archive column's units (kHz, see features/namespace.py) and normalised
    under upstream's `rotation_kms` constants, exactly as get_survival_from_shot.py does with the same column
---

# plasmacontrol/d3d-tearing-time-to-event-dsm

**Status: implemented** (2026-09-05). Labels are produced and `validate` scores their per-shot alarms
against archived onsets (see Evaluation).

## Model details

Time to tearing-mode onset as a survival problem: a Deep Survival Machines
model (auton-survival, the group's fork) with a k=3 log-normal mixture over
onset time, conditioned on a 38-dimensional embedding of 14 real-time scalars
and 6 fitted profiles. Labelmaker publishes the risk of an onset within 250 ms,
500 ms and 1 s: `1 - S(horizon | x)`. Complements `d3d_tearing_onset_cnn1d`,
which answers the fixed-horizon "is a mode present 25 ms from now" question
with a different architecture and input set.

Three further series, `tm_risk_250ms_isotonic`, `tm_risk_500ms_isotonic`, and
`tm_risk_1s_isotonic`, apply per-horizon isotonic maps fitted on **all_pre_onset**
valid rows of the calibration fit shots. The raw `tm_risk_*` series remain the
model's output. Isotonic series are always declared and are NaN when
`calibration.json` is absent. Each calibrated label group carries `calibration`
and `calibration_fit_on` attributes recording the fitting shots, row count,
prevalence, row set, date and git SHA.

The 14 additional series expose that distribution: `tm_time_p10`,
`tm_time_p50`, and `tm_time_p90` are event-time quantiles in ms;
`tm_time_iqr_log` is ln(p90) - ln(p10). Those quantiles are clamped to
[1e-3, 1e7] ms, so a value sitting on either end is not a predicted time but
"beyond the representable range" - 1e7 ms is 10,000 s, far longer than any
DIII-D discharge. `tm_mix_w0..2` are softmax gate
weights, `tm_mix_mu0..2` are locations in ln ms, and `tm_mix_sigma0..2`
are component log-scales: exp(sigma) is the standard deviation of ln t.
`tm_gate_entropy` is -sum w ln w in nats, from 0 to ln 3. Component order
is the checkpoint's own; components are not identifiable across retrainings.

This spread is the predictive distribution of event time (aleatoric variation
plus what the network learned), not epistemic uncertainty over weights.
The upstream `rt_models*.pkl` ensembles would give the latter. The p10–p90
band therefore differs from the stored single-member ensemble spread.

Architecture, read from the checkpoint: `Linear(38, 100, no bias) -> ReLU6 ->
Linear(100, 1000, no bias) -> ReLU6`, then a softmax gate (1000 -> 3) and
tanh-activated scale and shape heads (1000 -> 3) added to learned per-component
offsets; temperature 1.0. Preprocessing (from `get_survival_from_shot.py`):
profiles interpolated from 33 to 100 points (65 for 1/q and pressure), 1/q with
inf -> 1, a stored 4-component PCA per profile, then z-scores; scalars
z-scored; `PCBCOIL * 1.69861e-5` (used as toroidal field), beam power in MW.

## Uses

Offline label generation over the FAITH shot corpus, for comparison against
IGNITE and against other models' labels. Not for real-time control and not for
physics conclusions without reliability numbers, which this model does not yet
have (below).

## Bias, risks and limitations

- The same 2024-shot caveats as the tearing CNN apply to any model fed offline
  EFIT01 and ZIPFIT: the substitutions are unpriced here.
- Rows are valid where every input is finite; upstream applied no other filter,
  so labelmaker applies none, and out-of-domain inputs are not flagged.
- The rotation profile is absent on ~22% of shots and the ion-temperature fit
  on some more; a row missing either is invalid.
- Upstream's own evaluation is per **shot**, not per row (`metrics_helpers.py`):
  one verdict from the last label in the trace, a warning time from the final
  0-to-1 transition, a count of crossings that fail to persist 400 ms, FPR over
  quiet shots and FNR over tearing shots, at a default threshold of 0.7. The
  per-row F1 numbers below are therefore not comparable to any upstream figure.
- This checkpoint is under-trained (Training details), but training it to
  convergence did not help: `d3d_tearing_time_to_event_dsm_continued` continues
  this fit for 300 epochs, cutting validation NLL 0.471321 -> 0.287652 on the
  same (leaky, row-level) split, and **measured worse on the 500-shot pool** -
  AUROC 0.810 -> 0.681, 0.787 -> 0.670, 0.758 -> 0.680 at 250 ms / 500 ms / 1 s
  (on held-out shots alone, 0.839 -> 0.692, 0.817 -> 0.672, 0.801 -> 0.688: the
  same verdict), with raw ECE on all pre-onset rows worse at every horizon. This model stays the
  default; the comparison is in that card and in
  `outputs/labelmaker/presentation_continued/`.
- Calibration depends on the reported row set. The earlier full-pool 1 s
  figures were ECE 0.022 on all aligned pre-onset rows (5.5% positive), versus
  0.448 on onset-only shots (54% positive). The study below estimates the
  adjustment on separate **held-out** fit shots - never on shots this
  checkpoint was trained on - and measures it on the report shots, on the
  in-training shots and on both together.

Calibration study (2026-09-05, seed 0), **fitted on held-out shots only**: 463
shots have usable pooled rows from the 500-shot pool, 208 of them training
shots. The seeded permutation is applied to the **255 held-out shots alone** -
127 fit, 128 report - because an isotonic map fitted on rows the model
memorised is calibrated to memorisation. The fit half has 7,941
**all_pre_onset** rows and 741 **onset_shots_only** rows. Every metric is then
reported on three populations: `held_out` (the 128 report shots, 8,809 /
685 rows), `in_training` (all 208, 11,540 / 1,469 rows) and `all`, their union
(336 shots, 20,349 / 2,154 rows) - which is every scored shot the fit did not
use, not the whole pool. Target prevalence p1 is computed on the held-out FIT
half of each row set and applied to all three. Training prevalence q1(h) is
`mean((e == 1) & (t <= h_ms))` over 914,898 upstream rows; source paths and
SHA256s are in `validation/d3d_tearing_time_to_event_dsm/calibration_study.json`,
and `calibration.json`'s `fit_on` records `subset: held_out`.

Forced republication succeeded on all 500 shots in **746.97 s** (eight
workers, 2026-09-05); shot 187199's `tm_risk_1s_isotonic` carries
`calibration: isotonic, fit on held-out all_pre_onset rows` and a
`calibration_fit_on` with `subset: held_out`, so the published series come
from the held-out map.

The published isotonic maps use only **all_pre_onset fit rows**. The study
also fits onset-only maps for comparison; those maps are not published.
PAVA pools tied scores first; application interpolates linearly and clamps
at the end knots. In the table, each triplet is **raw / prior shift / isotonic**.

| Row set | Horizon | q1 training | p1 fit | subset | ECE (raw / prior / iso) | Brier (raw / prior / iso) |
|---|---|---:|---:|---|---|---|
| all_pre_onset | 250ms | 0.01811677 | 0.01183730 | all | 0.001672 / 0.003407 / 0.004342 | 0.013043 / 0.012965 / 0.013018 |
| all_pre_onset | 250ms | 0.01811677 | 0.01183730 | held_out | 0.002928 / 0.001904 / 0.004723 | 0.009524 / 0.009407 / 0.009452 |
| all_pre_onset | 250ms | 0.01811677 | 0.01183730 | in_training | 0.000714 / 0.005006 / 0.004052 | 0.015730 / 0.015682 / 0.015740 |
| all_pre_onset | 500ms | 0.03759982 | 0.02480796 | all | 0.005000 / 0.007875 / 0.008049 | 0.026986 / 0.026863 / 0.026969 |
| all_pre_onset | 500ms | 0.03759982 | 0.02480796 | held_out | 0.005542 / 0.005233 / 0.008640 | 0.019774 / 0.019546 / 0.019718 |
| all_pre_onset | 500ms | 0.03759982 | 0.02480796 | in_training | 0.005710 / 0.010473 / 0.009062 | 0.032491 / 0.032448 / 0.032504 |
| all_pre_onset | 1s | 0.07185282 | 0.05326785 | all | 0.019731 / 0.017428 / 0.027483 | 0.051189 / 0.050819 / 0.051871 |
| all_pre_onset | 1s | 0.07185282 | 0.05326785 | held_out | 0.014508 / 0.009268 / 0.021393 | 0.037235 / 0.036876 / 0.037276 |
| all_pre_onset | 1s | 0.07185282 | 0.05326785 | in_training | 0.023743 / 0.024076 / 0.032133 | 0.061840 / 0.061462 / 0.063013 |
| onset_shots_only | 250ms | 0.01811677 | 0.12685560 | all | 0.101148 / 0.057292 / 0.028080 | 0.115520 / 0.106379 / 0.103838 |
| onset_shots_only | 250ms | 0.01811677 | 0.12685560 | held_out | 0.093080 / 0.060444 / 0.034065 | 0.111981 / 0.106758 / 0.100489 |
| onset_shots_only | 250ms | 0.01811677 | 0.12685560 | in_training | 0.104910 / 0.066005 / 0.027741 | 0.117170 / 0.106203 / 0.105399 |
| onset_shots_only | 500ms | 0.03759982 | 0.26585695 | all | 0.220396 / 0.063889 / 0.028660 | 0.232118 / 0.176459 / 0.185617 |
| onset_shots_only | 500ms | 0.03759982 | 0.26585695 | held_out | 0.208588 / 0.090528 / 0.026287 | 0.226711 / 0.184016 / 0.184275 |
| onset_shots_only | 500ms | 0.03759982 | 0.26585695 | in_training | 0.225903 / 0.072478 / 0.029767 | 0.234639 / 0.172936 / 0.186243 |
| onset_shots_only | 1s | 0.07185282 | 0.57085020 | all | 0.435796 / 0.071241 / 0.043990 | 0.416344 / 0.216335 / 0.249538 |
| onset_shots_only | 1s | 0.07185282 | 0.57085020 | held_out | 0.420512 / 0.154653 / 0.048024 | 0.406911 / 0.236554 / 0.249751 |
| onset_shots_only | 1s | 0.07185282 | 0.57085020 | in_training | 0.442923 / 0.077276 / 0.042108 | 0.420743 / 0.206907 / 0.249439 |

The onset-only corrections substantially reduce calibration error and Brier
loss, but residual error remains. On all_pre_onset rows the raw scores are
already close to calibrated and isotonic makes ECE **worse** at every horizon
and on every subset (held-out 1 s: 0.014508 raw to 0.021393). Prior shift is
the only correction that ever beats raw there, and it does so at all three
horizons on held-out rows (1 s: 0.014508 to 0.009268) while worsening all
three on in-training rows. The map is fitted on 127 shots and
reported on different ones, so none of this establishes calibration on another
shot pool. Isotonic plateaus change ranking through ties: all_pre_onset AUROC
(raw to isotonic) on held-out rows is 0.8868 to 0.8814 at 250 ms, 0.8587 to
0.8433 at 500 ms and 0.8313 to 0.8248 at 1 s. Prior shift preserves ordering.
Splitting the report changes the conclusion nowhere: raw ECE is between
0.0007 and 0.024 on all_pre_onset for both halves, and the isotonic penalty
is the same sign on both.

## Training details

Upstream, outside this repository: `train_tm_model.py` over the `rt_*` pickles
in `/projects/EKOLEMEN/survival_tm_2/data/`. The full hyperparameter dict,
decoded from the shipped pickle's own opcode stream on 2026-09-05:

```
iters=20  k=3  layers=[100,1000]  distribution=LogNormal  learning_rate=1e-05
batch_size=1000  discount=1.0  temp=1.0  activation=ReLU6  random_seed=0
```

The pickle also carries its loss curves: 20 entries, validation NLL 0.534 to
0.471, **monotone and still falling at the last epoch**. auton-survival's early
stop (`train_patience = 5`) never fired, so this fit was ended by its `iters`
setting, not by convergence. It is under-trained.
`d3d_tearing_time_to_event_dsm_continued` continues this fit from these
weights; see that card and the comparison in Evaluation below.

**No class balancing was applied.** `losses._conditional_lognormal_loss` scores
events with `log f(t)` and censored rows with `alpha * log S(t)`, normalised by
the total row count; `alpha = model.discount = 1.0`, so censored rows are not
down-weighted. The survival likelihood is the whole of the imbalance handling.
`train_tm_model.py` applies no resampling and no time cut.

**Correction, 2026-09-05: the validation split behind the 0.471 above is a
row-level holdout, not the by-shot split `train_tm_model.py` performs today.**
That script's 80/10/10 by-shot rule reproduces 0.4517 on these weights, not
0.4713, and the parameter dict inside the pickle has no `seed` for that rule to
use; the config that names this model
(`/projects/EKOLEMEN/survival_tm/outputs/rt_fixed_rotconfig`) does not even name
a shots list. Searching the plausible rules against the pickle's own stored
final validation loss found the one that reproduces it to machine precision
(|diff| < 1e-15): the whole 914,898-row dataset went into `SurvivalModel.fit`
with no `val_data`, so `estimators.py` applied its own default,
`data.sample(frac=1 - 0.15, random_state=0)`, training on 777,663 rows and
validating on the 137,235-row complement. 8,685 of the 8,690 shots with a
validation row also have training rows, so **the stored validation curve is
optimistic**: it is not a by-shot holdout and does not measure generalisation
to unseen shots. Everything else in this section - the hyperparameters, the
absence of balancing, the row statistics - is unaffected. The measurement is
`$LABELMAKER_ROOT/runs/task4_split_search.py`, and
`scripts/labelmaker/retrain_tearing_dsm.py` gates on reproducing 0.4713 before
it will continue the fit.

Training rows (`rt_filtered_{e,t}_bms_pcb_rot.pkl`): 914,898 rows, **15.07%
events**, `t` in ms with median 1,920 and max 5,740, and only 14.7% of rows
within 600 ms of their event. Labels come from `format_survival_labels.py`,
which truncates each shot at onset, so no post-onset row is ever a training row.

The 1:1 event/censor undersampling and the `t < 600 ms` cut that appear in
`train_survival_study.py` and `tm_survival.ipynb` belong to an Optuna study and
were **not** used for this model. Details in
`docs/superpowers/specs/2026-09-05-labelmaker-phase3-design.md` section 2.1.

## Evaluation

Adapter fidelity is checked as a test rather than a `validate` report:
`tests/labelmaker/test_dsm_pickle.py` compares labelmaker's evaluator against
the fork's own `predict_survival` on 256 inputs
(`tests/labelmaker/data/tearing_dsm_golden.npz`, made once by
`make_tearing_dsm_golden.py`) to 1e-9 in float64, and
`test_tearing_dsm_adapter.py` checks the preprocessing step by step against
the upstream script. Reconstruction fidelity and label quality need a truth
series - "onset within the next horizon", derivable from the tearing archive's
`tm_label` on the 1,503 overlap shots. `validate.alarm_quality` now builds
that truth for the published-label alarm report below.

**Roughly half of the pool is in this checkpoint's own training set, and every
report now splits on it.** The training set is the 8,923 unique DIII-D shots of
`training_shots.txt` (140444-193373, 914,898 training rows); **214 of the 500
pool shots, 208 of the 463 aligned scored shots and 41 of the 80 shots with an
archived onset** are in it. `validate.alarm_quality` and
`validate.calibration_study` therefore report every metric three ways - `all`,
`held_out`, `in_training` - taking the split from the adapter's own
`training_shots`, and the published isotonic map is fitted on held-out shots
only. Any number below labelled `all` still mixes the two.

### Held out against in training

Measured 2026-09-05 by `validate.alarm_quality` on the 500-shot pool. Row set:
**pre-onset valid rows of every aligned shot**. Membership is constant within a
shot, so each subset is a whole number of shots.

| subset | shots | quiet | tearing | rows | positive at 1 s |
|---|---:|---:|---:|---:|---:|
| all | 463 | 383 | 80 | 28,290 | 1,565 |
| held_out | 255 | 216 | 39 | 16,750 | 783 |
| in_training | 208 | 167 | 41 | 11,540 | 782 |

| horizon | AUROC all | AUROC held_out | AUROC in_training | IPCW held_out | IPCW in_training |
|---|---:|---:|---:|---:|---:|
| 0.25 s | 0.810043 | 0.838724 | 0.760792 | 0.837937 | 0.756859 |
| 0.5 s | 0.786852 | 0.816826 | 0.738049 | 0.815260 | 0.733874 |
| 1 s | 0.758441 | 0.800649 | 0.696373 | 0.801975 | 0.691228 |

| subset | final FPR/FNR at 0.2 | any-row FPR/FNR at 0.2 | median warning (s) | horizon-integrated FPR/FNR at 0.2 |
|---|---|---|---:|---|
| all | 0.065274 / 0.662500 | 0.146214 / 0.600000 | 0.350 | 0.054504 / 0.603125 |
| held_out | 0.069444 / 0.589744 | 0.101852 / 0.512821 | 0.288 | 0.043981 / 0.580128 |
| in_training | 0.059880 / 0.731707 | 0.203593 / 0.682927 | 0.375 | 0.068114 / 0.625000 |

**The in-sample half is the worse half, not the better one.** Ranking is 0.04
to 0.10 of AUROC *higher* on the shots this checkpoint never saw, at all three
horizons and under both AUC definitions. The contamination therefore did not
inflate the headline numbers; it depressed them. The two subsets are different
populations - 41 of 208 in-training shots tear (19.7%, 6.8% of their pre-onset
rows positive at 1 s) against 39 of 255 held-out shots (15.3%, 4.7%) - so this
is a statement about which shots the pool sampled, not evidence that training
hurt. What it does settle is that the pool numbers were not being carried by
memorisation. Whether an honest by-shot retrain scores differently is Task 4b,
not something these rows answer.

Measured once outside `validate` (2026-09-05, `outputs/labelmaker/dsm_onset_quality.py`
in the FusionAIHub checkout): on the 486 aligned shots of the 500-shot pool,
86 of which have an onset in their archived window, taking rows before the
onset only (28,290 rows) and truth "onset within the next horizon":

| label | positives | AUROC | best F1 (at) | ECE | CNN `tm_prob` on the same truth, AUROC |
|---|---|---|---|---|---|
| `tm_risk_250ms` | 369 | 0.810 | 0.134 (0.08) | 0.002 | 0.786 |
| `tm_risk_500ms` | 780 | 0.787 | 0.169 (0.11) | 0.007 | 0.749 |
| `tm_risk_1s` | 1,565 | 0.758 | 0.211 (0.16) | 0.022 | 0.671 |

The risk ranks pre-onset rows better than the CNN's present-mode probability
does, and is calibrated, but the absolute discrimination is modest and the
best F1 is low because onsets are rare in the windows (1.3% to 5.5% of rows).
Inputs were labelmaker's reconstruction (offline EFIT01, ZIPFIT), so this is
the published label's quality, not the model's ceiling; the truth is the
archive's 25 ms `tm_label`, whose own onset timing is unexamined.

### Published event-time distribution

Re-published the 500-shot pool on 2026-09-05 in **755.23 s** wall time
(eight workers; 500 successful shots). Row set: **all valid published rows**,
including post-onset rows, without requiring archive alignment: 55,407 of
120,000 grid rows, from 484 shots with at least one valid row.
The pooled median `tm_time_p50` is **17,344.333984 ms** and the pooled median
`tm_time_iqr_log` is **3.519858122**. **0 / 55,407 (0%)** valid rows have
`tm_gate_entropy > 0.9 ln 3` (0.988751060 nat). A decisive gate does not
imply a narrow event-time distribution.

Default `analyze` on the two example shots produced the following `tm_time_p50`
scores. Row set: **valid aligned pre-onset rows of each tearing shot only**;
errors are ln(predicted ms / archived time remaining in ms).

| shot | archived onset (s) | n | median absolute log ratio | bias_log | rmse_log | p50 at onset minus 1 s (ms) |
|---|---|---|---|---|---|---|
| 187199 | 3.825 | 92 | 2.141511 | 2.163416 | 2.245654 | 11,779.271484 |
| 186545 | 3.275 | 51 | 1.751241 | 2.181940 | 2.356755 | — (nearest row invalid) |

The logarithmic p10–p90 panels show medians well above the remaining time
on both shots. On 187199 the median remains around 10,000 ms approaching
onset while the archived countdown falls below the band. On 186545 the
band widens markedly near onset, reaching much shorter lower-tail times
while the median remains several thousand ms. Neither panel is evidence
of epistemic uncertainty or a new held-out evaluation. The figures and full
per-shot scores are in `outputs/labelmaker/analysis/{187199,186545}/` in the
FusionAIHub checkout.

### Per shot, upstream's way

Measured 2026-09-05 from existing published labels on the 500-shot pool.
Row set: **pre-onset valid rows of every aligned shot**: 28,290 rows from
463 scored shots, **383 quiet and 80 tearing**. There are 486 aligned shots
(86 with onset), but 23 have no valid pre-onset predictions and are excluded
from shot-rate denominators; 14 other pool shots fail archive alignment.

Final-label calls use `risk >= threshold` on the last retained row; any-row
calls use any retained row. Warning time is onset minus the start of the
final on-run, summarized over final-label TPs only.

| `tm_risk_1s` threshold | final FPR | final FNR | any-row FPR | any-row FNR | median warning (s) |
|---|---|---|---|---|---|
| 0.1 | 0.180157 | 0.475000 | 0.321149 | 0.350000 | 0.675 |
| 0.2 | 0.065274 | 0.662500 | 0.146214 | 0.600000 | 0.350 |
| 0.3 | 0.000000 | 0.987500 | 0.015666 | 0.950000 | 0.050 |
| 0.7 | 0.000000 | 1.000000 | 0.000000 | 1.000000 | — (no TPs) |

The final-label rule has lower FPR and higher FNR than the any-row rule at
thresholds 0.1–0.3; at 0.7 both rules miss every tearing shot.

| horizon (s) | plain AUROC | IPCW AUC |
|---|---|---|
| 0.25 | 0.810043 | 0.808217 |
| 0.5 | 0.786852 | 0.784507 |
| 1.0 | 0.758441 | 0.758210 |

Both AUCs start from the same 28,290 rows. IPCW uses reverse Kaplan–Meier
censoring weights and excludes quiet rows censored before the horizon from
case/control pairs. Both metrics share the same onset-within case truth,
including the 1 ns tolerance for floating-point grid subtraction. Plain
AUROC exceeds IPCW AUC by 0.001826, 0.002344 and 0.000231 at the three
horizons, respectively. These are descriptive pool
results, not a new held-out evaluation or a threshold recommendation.
The complete threshold sweep, warning quartiles, jump histograms and
horizon integrals are in `$LABELMAKER_ROOT/validation/d3d_tearing_time_to_event_dsm/alarm_quality.json`,
each of them under `labels.<name>.subsets.{all,held_out,in_training}`. The two
tables above are the `all` subset; "Held out against in training" gives the
same numbers split, and the held-out figures are in
`outputs/labelmaker/presentation/held_out/`.

## Technical specifications

104,800 embedding weights plus 9,009 head weights, float64, in a pickle read
by `runners/dsm_pickle.py` with a restricted unpickler (only the three
auton-survival classes and torch/numpy are accepted). Both artifacts verified
against the sha256 above before every load.

## Citation

Unpublished internal model. Attribute to the PlasmaControl group, Princeton.

## Contact

`nc1514@princeton.edu`.
