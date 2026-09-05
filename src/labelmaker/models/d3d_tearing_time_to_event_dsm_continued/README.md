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
  - retrained
datasets:
  - plasmacontrol/d3d-faith-corpus
metrics:
  - roc_auc
model-index:
  - name: d3d-tearing-time-to-event-dsm-continued
    results: []
labelmaker:
  status: implemented
  slug: d3d_tearing_time_to_event_dsm_continued
  card_id: plasmacontrol/d3d-tearing-time-to-event-dsm-continued
  framework: dsm_pickle
  time_step_ms: 25.0
  ensemble_n: 1
  upstream:
    path: /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/models/d3d_tearing_time_to_event_dsm_continued
    trained: 2026-09-05
    training_code: scripts/labelmaker/retrain_tearing_dsm.py
    reference_harness: /projects/EKOLEMEN/survival_tm/get_survival_from_shot.py
    continued_from: /projects/EKOLEMEN/survival_tm_2/models/rt_fixed_rot.pkl (sha256 2a7b65a9e7484c91)
    artifacts:
    - rt_fixed_rot_continued.pkl
    - rt_normalizations_dict.pkl
    sha256:
      rt_fixed_rot_continued.pkl: 4b1745ddd641c61bb826edd2f4969647213a6d94e365b3f8fd3c671470bfa5be
      rt_normalizations_dict.pkl: fa515c7b591f3e1ea5f710d75a825b1a7831cbd63e06543f9ccb87ff87a73fbe
    notes: trained by labelmaker, not upstream; rt_normalizations_dict.pkl is byte-identical to the base
      model's copy of /projects/EKOLEMEN/survival_tm/data/rt_normalizations_dict.pkl
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

# plasmacontrol/d3d-tearing-time-to-event-dsm-continued

**Status: implemented** (2026-09-05). A variant of
`d3d_tearing_time_to_event_dsm`: the same architecture, inputs, preprocessing
and 20 output columns, with weights that continue the shipped fit instead of
stopping at its 20th epoch. The default tearing survival label is still the
base model's; this one is published beside it so the two can be compared on
the same shots.

## Model details

Everything about the forward pass is the base card's: 14 real-time scalars and
6 fitted profiles at `t`, upstream's preprocessing (33 -> 100 point profiles,
65 for 1/q and pressure, 1/q with `inf -> 1`, a stored 4-component PCA per
profile, z-scores; `PCBCOIL * 1.69861e-5`; beam power in MW), a k=3 log-normal
mixture over onset time, and `1 - S(horizon | x)` at 250 ms, 500 ms and 1 s.
`spec.py` imports `INPUT_SPEC`, `OUTPUT_SPEC`, `preprocess` and `HORIZONS_MS`
from the base module and builds its loader with the base module's own
`make_load`, so the only thing that differs between the two models is the
weight file.

The three `*_isotonic` columns are declared, as they are for the base model,
and are **NaN for this model**: they are filled only when a `calibration.json`
sits in the model's own directory, and this model's directory has none. A
calibration study was run for it and its report is published (Evaluation), but
the maps that study writes were removed rather than installed, so the labels on
disk and this card agree. The base model's isotonic maps were fitted on the
base model's scores and do not transfer to different weights.

## Uses

Offline label generation over the FAITH shot corpus, for the comparison in
Evaluation below. Not a drop-in replacement for the base model: the default in
`analyze_default.yaml` is unchanged, and the decision of which checkpoint
should be labelmaker's tearing survival label is open (phase3 design section 7,
question 5).

## Training details

`scripts/labelmaker/retrain_tearing_dsm.py`, run on one stellar CPU node
(SLURM job 2923879, partition `all`, 8 CPUs, 32 GB). It continues the
shipped checkpoint's own `torch_model` rather than refitting: auton-survival's
`train_dsm` begins with `pretrain_dsm` and then **overwrites** the learned
per-component `shape` and `scale` parameters with a one-dimensional fit, so
only the part of that function from `model.double()` onward is reproduced -
Adam, per-epoch `shuffle(..., random_state=epoch)`, minibatch
`conditional_loss(..., elbo=True)`, full-validation `conditional_loss(...,
elbo=False)`, every epoch's `state_dict` kept and the argmin reloaded at the
end, and the same "no improvement for `patience` epochs" stop.

```
continued from rt_fixed_rot.pkl (20 epochs, lr 1e-5, val NLL 0.534 -> 0.471)
lr 1e-4   batch 1000   max epochs 300   patience 5   seed 0   k=3   layers=[100,1000]
distribution LogNormal   discount 1.0   temp 1.0   elbo True   torch 2.14.0+cu126 (CPU)
```

Two continuation details are recorded rather than hidden: the epoch counter
continues from 20, so no epoch repeats a shuffle ordering the shipped fit
already used; and Adam's moment estimates are not in the pickle, so the
optimiser restarts cold, which a single uninterrupted run would not do.

**The split this model was trained on was measured, not assumed.** The config
that produced `rt_fixed_rot` names no shots list, and the parameter dict inside
the pickle has no `seed`, so the by-shot 80/10/10 split in today's
`train_tm_model.py` cannot be the rule that ran in 2024-11 (it reproduces
0.4517, not the stored 0.4713). Searching the plausible rules against the
checkpoint's own stored final validation loss found the answer to machine
precision: the whole 914,898-row dataset went into `SurvivalModel.fit` with no
`val_data`, so the estimator applied its own default,
`data.sample(frac=0.85, random_state=0)`, and validated on the 137,235-row
complement. That is a **row-level** split: 8,685 of the 8,690 shots with a
validation row also have training rows in it. The continuation uses the same
split, because it is the only one on which "before" and "after" mean the same
thing, and the gate in the training script refuses to run if
`compute_nll` on the rebuilt split does not reproduce the stored loss within
2e-3 (measured |diff| 0.0, exactly).

**Measured curve.** 300 epochs, 20 through 319, in 29 min 38 s on one node
(job 2923879, 8 CPUs at 95.1% utilisation over the run, memory high-water
4.6 GB of the 32 GB requested; the right-sized request is 8 CPUs, 8 GB, 1 h).

| | epochs | validation NLL |
|---|---|---|
| shipped fit, lr 1e-5 | 0-19 | 0.534448 -> 0.471321 |
| continuation, lr 1e-4 | 20-319 | 0.463622 -> best **0.287652** at epoch 301 |

The improvement is **0.183669 nats (39.0%)** on the split the checkpoint was
fitted on. Most of it comes early: the curve is at 0.3265 by continuation epoch
100 and 0.2992 by epoch 200, and the last 70 epochs oscillate between 0.287652
and 0.322409 with no trend. The run therefore ended on its 300-epoch cap rather
than on the patience rule (five consecutive non-improving epochs never
occurred), but the curve is flat enough that a longer run at this learning rate
would gain little. The full curve is in `training.json` and `loss_curve.png` in
the artifact directory.

As a second reading on the same weights, the NLL over the by-shot validation
shots of `train_tm_model.py`'s split falls 0.451674 -> 0.223317. That number is
**not** leak free either - the model trained on 85% of those shots' rows - and
it is recorded only because it is the same measurement both checkpoints can be
put through.

## Bias, risks and limitations

- **The validation NLL below is optimistic and cannot be read as a
  generalisation gain.** Its rows share shots with the training rows (above).
  The out-of-sample evidence in this card is the 500-shot pool comparison, and
  even there 214 of the 500 pool shots contributed rows to the training set.
- Every caveat of the base model applies unchanged: offline EFIT01 substituted
  for EFITRT2, ZIPFIT fits for the pipeline's own mtanh/csaps profiles,
  labelmaker's 25 ms grid and 50 ms window for upstream's 20 ms samples, rows
  valid only where every input is finite, no out-of-domain flagging, and the
  rotation profile missing on ~22% of shots.
- The isotonic columns are NaN (above), so any consumer that reads
  `tm_risk_*_isotonic` gets nothing from this model.
- The mixture components are not identifiable across retrainings: this model's
  `tm_mix_*` component order is its own and has no relation to the base
  model's.

## Evaluation

Published for the 500-shot pool on 2026-09-05 (`infer --workers 8`, 500 shots
in **813.99 s**, 31,550 valid rows), then scored exactly as the base model was:
`validate.alarm_quality` and `validate.calibration_study` with the same seed-0
shot split. Reports:
`$LABELMAKER_ROOT/validation/d3d_tearing_time_to_event_dsm_continued/{alarm_quality,calibration_study}.json`.

Row sets: **AUROC, IPCW AUC, lead time, FPR and FNR** are over all **28,290
pre-onset valid rows of the 463 scored shots** (383 quiet, 80 tearing);
**ECE** is over the **report half** of that pool (232 shots, 14,005
all_pre_onset rows of which 1,702 are onset-shot rows), raw risks with no
post-hoc map.

| quantity | shipped | continued | difference |
|---|---:|---:|---:|
| validation NLL (the shipped model's own split) | 0.471321 | **0.287652** | **-0.183669** |
| AUROC `tm_risk_250ms` | 0.810043 | 0.681327 | **-0.128716** |
| AUROC `tm_risk_500ms` | 0.786852 | 0.669523 | **-0.117329** |
| AUROC `tm_risk_1s` | 0.758441 | 0.680208 | **-0.078233** |
| IPCW AUC `tm_risk_250ms` / `500ms` / `1s` | 0.808217 / 0.784507 / 0.758210 | 0.679597 / 0.664393 / 0.673280 | -0.128620 / -0.120115 / -0.084930 |
| ECE all_pre_onset, 250 ms / 500 ms / 1 s | 0.004434 / 0.014819 / 0.024092 | 0.009956 / 0.027431 / 0.055926 | +0.005522 / +0.012612 / +0.031834 |
| ECE onset_shots_only, 250 ms / 500 ms / 1 s | 0.109420 / 0.230784 / 0.459990 | 0.104915 / 0.220567 / 0.438905 | -0.004505 / -0.010217 / -0.021085 |
| median lead time, `tm_risk_1s` at 0.2 | 0.350 s | 0.500 s | +0.150 s |
| final-label FPR at 0.2, `tm_risk_1s` | 0.065274 | 0.117493 | +0.052219 |
| final-label FNR at 0.2, `tm_risk_1s` | 0.662500 | 0.625000 | -0.037500 |

**The continued model did not measure better where it counts.** It gained
0.1837 nats of validation likelihood and lost 0.078 to 0.129 of AUROC at every
horizon, with IPCW AUC agreeing and calibration on all pre-onset rows getting
worse. It calls more shots at threshold 0.2 - FNR down, FPR up by more, lead
time longer - which is the behaviour of a model whose risks moved up, not of a
model that ranks rows better. Firing earlier is not ranking better.

The explanation this card can support is the leaky split above: the objective
that improved is a likelihood measured on rows whose shots are in training.
Nothing here recommends replacing the shipped checkpoint, and the default in
`analyze_default.yaml` is unchanged; phase3 design section 7 question 5 stays
open.

**These pool numbers are roughly half in-sample for both models.** The
survival training set is 8,923 unique DIII-D shots spanning 140444-193373; 214
of the 500 pool shots, 208 of the 463 aligned scored shots and 41 of the 80
shots with an archived onset are training shots. Separating in-training from
held-out shots is a separate task; these numbers pool the two.

The seven Phase 2 figures re-rendered for this model, with the comparison
table and the CNN panels unchanged, are in
`outputs/labelmaker/presentation_continued/` in the FusionAIHub checkout.

A `calibration_study` was run for this slug and its report is published, but
its isotonic maps were **not** installed: the `calibration.json` the study
writes was removed from the model directory, so the labels on disk and the
card agree that this model's `*_isotonic` columns are NaN. For the record, the
study's isotonic ECE on the report half would be 0.025317 (all_pre_onset) and
0.047409 (onset_shots_only) at 1 s.

## Technical specifications

The same graph as the base model - 104,800 embedding weights plus 9,009 head
weights, float64 - in a pickle of the same shape
(`[[SurvivalModel, train_losses, val_losses, params]]`), read by
`runners/dsm_pickle.py` with the same restricted unpickler. The stored loss
lists carry the whole trajectory: the first 20 entries are the shipped fit's,
the rest this continuation's. Both artifacts are verified against the sha256
above before every load. `training.json` and `PROVENANCE.json` sit beside the
weights in the artifact directory.

## Citation

Unpublished internal model, retrained by labelmaker from the PlasmaControl
group's `rt_fixed_rot` checkpoint. Attribute to the PlasmaControl group,
Princeton.

## Contact

`nc1514@princeton.edu`.
