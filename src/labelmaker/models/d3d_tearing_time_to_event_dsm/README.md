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
    notes: rt_normalizations_dict.pkl lives in /projects/EKOLEMEN/survival_tm/data/ upstream
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

The 14 additional series expose that distribution: `tm_time_p10`,
`tm_time_p50`, and `tm_time_p90` are event-time quantiles in ms;
`tm_time_iqr_log` is ln(p90) - ln(p10). `tm_mix_w0..2` are softmax gate
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
- **The calibration depends on which rows you ask about, and both numbers below
  are correct.** Over every aligned shot's pre-onset rows (5.5% positive at 1 s)
  the risk is calibrated, ECE 0.022. Restricted to the 86 shots that do get an
  onset (54% positive) it is badly under-confident, ECE 0.448. That is a
  base-rate shift, not miscalibration: the model was fit to a population that is
  85% censored with a median 1.92 s to event. Any report has to name its row set.

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

**No class balancing was applied.** `losses._conditional_lognormal_loss` scores
events with `log f(t)` and censored rows with `alpha * log S(t)`, normalised by
the total row count; `alpha = model.discount = 1.0`, so censored rows are not
down-weighted. The survival likelihood is the whole of the imbalance handling.
`train_tm_model.py` applies no resampling and no time cut, and splits 80/10/10
**by shot**.

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
horizon integrals are in `$LABELMAKER_ROOT/validation/d3d_tearing_time_to_event_dsm/alarm_quality.json`.

## Technical specifications

104,800 embedding weights plus 9,009 head weights, float64, in a pickle read
by `runners/dsm_pickle.py` with a restricted unpickler (only the three
auton-survival classes and torch/numpy are accepted). Both artifacts verified
against the sha256 above before every load.

## Citation

Unpublished internal model. Attribute to the PlasmaControl group, Princeton.

## Contact

`nc1514@princeton.edu`.
