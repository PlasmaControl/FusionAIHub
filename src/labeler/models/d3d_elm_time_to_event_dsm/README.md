---
language: en
license: other
library_name: pytorch
tags:
  - diii-d
  - tokamak
  - elm
  - survival-analysis
datasets:
  - plasmacontrol/d3d-faith-corpus
metrics:
  - concordance_index
model-index:
  - name: d3d-elm-time-to-event-dsm
    results: []
labelmaker:
  status: implemented
  slug: d3d_elm_time_to_event_dsm
  card_id: plasmacontrol/d3d-elm-time-to-event-dsm
  framework: dsm_pickle
  time_step_ms: 25.0
  ensemble_n: 1
  display_name: elm-dsm refit
  membership:
    source: training_membership.json
    physical_training_shots: 300
    physical_early_stopping_shots: 80
    physical_normalization_shots: 365
    physical_shots_shared_by_training_and_early_stopping: 15
    exposed_physical_shots: 365
    blind_cohort_normalization_shots: [190532, 190646]
    validation_policy: >-
      adapter.training_shots includes weight-training, early-stopping and
      pre-split normalization exposure. General validation's in_training subset
      means source-exposed for this adapter; no exposed shot is held_out.
  serving:
    interpretation: offline risk score, not a causal forecast
    trained_row_ms: 1.0
    serving_window_ms: 50.0
    serving_grid_ms: 25.0
    centered_nbi_lookahead_ms: 25.0
  preprocessing_exposure:
    applies_to: [elm-dsm refit, elm-dsm detection exposed, elm-dsm detection init]
    scope: upstream means and standard deviations computed before the source split
    blind_cohort_shots: [190532, 190646]
    role: feature-statistics exposure, independent of reviewed-label CV
    decision: retain historical exposed fits as supplemental; isolated detection refits preprocessing
  isolated_detection:
    normalization: measured usable labeled optimizer-training rows within each fold
    inner_validation_in_normalization: false
    source_normalization_reused: false
    initialization: independent seeded random weights
    source_parameters_reused: false
    epochs_per_fold: 40
    folds: 5
  upstream:
    path: /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/models/d3d_elm_time_to_event_dsm
    artifacts:
      - elm_dsm_no_bes.pkl
      - normalization.json
    sha256:
      elm_dsm_no_bes.pkl: 2c28518eb1934c542ba007f7971d6ad2e38543be6f61ce57508b8629443aa9c3
      normalization.json: 0d3e99e3e2e7f1aee5bb3aa9d40a9393d81e12789428b3e1ae9c23ab1598e45e
    notes:
      - "2026-09-06: the weights are labeler's own, not upstream's Keras
        graphs. Same architecture and hyperparameters as hiro_scripts/model.cfg,
        fitted on upstream's own split (train_test_split_model10.pkl, 629,023
        train / 142,745 test rows, never re-split) on the 60 columns that are
        not BES. Artifacts and PROVENANCE_no_bes.json in the path above."
      - "2026-09-06, CORRECTION to the 2026-09-06 note this replaces: the split
        pickle's columns are in `new_diagnostic_order`, NOT
        `current_diagnostic_order`. Cell 43 of data_processing.ipynb writes
        reordered_model10.pkl in the new order and cell 47 builds the split from
        THAT file. MEASURED: the pickle carries both *_final_x and
        *_final_x_normalized, so upstream's per-column transform is recoverable
        exactly; all 124 columns of both sides match new_diagnostic_order to a
        worst relative error of 1.2e-12, and only 6 of 124 (the shared first
        six) match current_diagnostic_order. Under the real order BES is slots
        12-75 and ECE is 76-123."
      - "2026-09-06: the first no_bes fit (commit 54d201d) took slots 0-59
        believing them to be the non-BES columns. Under the real order those
        slots are 12 non-BES columns plus bes_slow_channel_1..48, so that model
        dropped every ECE channel and required 48 BES ones. It has been
        refitted on the correct 60 columns; the numbers below are the refit."
      - "2026-09-06: with the columns identified correctly, dropping BES is not
        a cost - it is an improvement. no_bes beats all124 on test NLL by 0.101
        and on 20 ms AUROC by 0.0019 (Evaluation). BES is 64 of 124 inputs and
        the fit at these hyperparameters overfits from epoch 1, so removing them
        removes more variance than signal."
      - "2026-09-06: both fits still reach their best test NLL after ONE epoch
        and get worse after it, at lr 1e-3 AND at lr 1e-4 (job 2924075,
        discarded). The optimum lies inside the first pass over 540k rows and a
        once-per-epoch validation curve cannot see it; the fix is sub-epoch
        checkpointing, not a smaller step. These are one-epoch models."
      - "2026-09-06: upstream passes the TEST split as val_data, which this fit
        reproduces for comparability, so the early stop selects on the test rows
        and test NLL is a selection-informed number, not a held-out one."
      - "2026-09-06: hiro's auton-survival fork inserts nn.Dropout(0.2) after
        every embedding ReLU6 (survival_tm_2's does not), so
        runners/dsm_pickle.load_dsm skips Dropout layers - identity in eval
        mode, no weights."
  inputs:
    - "ip_downsampled <- ip"
    - "bt_downsampled <- bt"
    - "gas_downsampled <- gas"
    - "pinj_downsampled <- pinj_total"
    - "tinj_downsampled <- tinj_total"
    - "ech_downsampled <- ech_power_total"
    - "co2_density_slow_r0_downsampled <- co2_r0"
    - "co2_density_slow_v1_downsampled <- co2_v1"
    - "co2_density_slow_v2_downsampled <- co2_v2"
    - "co2_density_slow_v3_downsampled <- co2_v3"
    - "ece_slow_downsampled <- ece"
  outputs:
    - name: elm_risk_5ms
      task: binary
      activation: none
    - name: elm_risk_10ms
      task: binary
      activation: none
    - name: elm_risk_20ms
      task: binary
      activation: none
    - name: elm_risk_50ms
      task: binary
      activation: none
  approximations:
    - "pcphd02 and pcphd03, the two D-alpha photodiodes, are 2 of the 60 columns
      and have NO corpus group: upstream read them from a hand-built PTDATA
      pickle (data/dalpha_wpqh.pkl). They are filled at the training mean -
      exactly 0 after normalisation - on every row of every shot."
    - "bt: upstream computed 1.69861e-5 * pcbcoil; labeler uses the canonical
      `bt` in tesla for its much wider shot coverage. Measured at t = 2.0 s on
      shot 185808: -2.0910 T against -2.0306 T, a 3.0% difference."
    - "ech: upstream took `echpwrc`, column 1 of the staged `ech` group;
      labeler uses ech_power_total (the 12-gyrotron corpus sum, or the
      archive's EC.PECH). See namespace.py's ech_power_total note for the
      measured spread between those two."
    - "gas is channel 0 of the corpus `gas_raw` group (PTDATA `gasa`), matched to
      the staged `gas` column at correlation +1.0000 and mean ratio 1.000 on
      shot 185808; no other gas_raw channel exceeds +0.17."
    - "ece is the corpus `ece` group's 48 channels decimated to 1 ms, matched to
      the staged `ece_slow` group channel for channel on shot 185808 (0.3-7%
      over a 100 ms window; the corpus record is ~3 MHz, the staged one 65 kHz)."
    - "co2_<chord> is the 1 ms mean of the SAME 500 kHz corpus `co2` record the
      AE model reads as a waveform - the brief's `co2_slow` - not a separate
      fetch. Chord order r0/v1/v2/v3 confirmed by rank: on shot 199421 the
      corpus chords order v1 > v2 > r0 > v3, which is upstream's own ordering of
      those four columns' training means."
    - "the serving inputs are 50 ms means ending at each timestamp on a 25 ms
      grid; the refit was trained on upstream's 1 ms rows."
    - "upstream's 100 ms boxcar on the raw pinj and tinj columns (NBI is
      modulated) is reproduced as a 4-tap centred moving average on the 25 ms
      grid."
  resolved:
    - "2026-09-05: the 124 trained-on names and their order ARE recovered - see Input order below"
    - "2026-09-06: and the order of the SPLIT PICKLE is new_diagnostic_order, measured to 1.2e-12; see upstream.notes"
    - "2026-09-05: the embedding does bake in its own normalisation in the Keras graphs; labeler's own fit uses upstream's normalizations dict instead, written to normalization.json"
    - "2026-09-05: the graph is structurally identical to d3d_tearing_time_to_event_dsm, so runners/dsm_pickle.survival() applies verbatim"
    - "The 1 ms-trained refit is served with 50 ms means on a 25 ms grid; risk is read at h + 1 ms. Centered NBI smoothing includes the row 25 ms later, so scores are offline."
    - "2026-09-06: the DSM head becomes a label series as 1 - S(h + 1) at four horizons, not as an expected time to event"
    - "The 60-column refit and the native 124-column checkpoint use different validation sets and preprocessing; their reported AUROCs do not establish that either input set outperforms the other."
  blocked_on:
    - "the fit is a one-epoch model at lr 1e-3 and at lr 1e-4 alike; a model worth trusting numerically needs sub-epoch checkpointing (validate every N minibatches), which no run has done yet"
    - "the adapter still mean-fills pcphd02 / pcphd03; the separate native evaluation uses exported or fetched original photodiodes without changing this adapter"
    - "catalog-wide adapter and reconstruction validation remains incomplete; reviewed occupancy and native diagnostics are reported separately below with source-exposure limits"
    - "corpus coverage: only 6 of the 24 sampled corpus shots have all 11 inputs, so 18 produce labels with no valid row at all. co2 is corpus:SignalAbsent below shot 198279 (12 of 24) and pinj_total/tinj_total have no fdp source in namespace.py, only archive+corpus (11 of 24). Serving the corpus properly needs an fdp NBI fetch and a decision about pre-198279 CO2"
---

# elm-dsm refit

**Status: implemented.** Offline ELM risk scores at 5, 10, 20 and 50 ms, from a
Deep Survival Machines refit. Serving uses 50 ms means on a 25 ms grid for a
model trained on 1 ms rows. The centered NBI boxcar incorporates the row 25 ms
later, so these scores do not support causal forecasting claims. The paper's
short names are `elm-dsm refit`, `elm-dsm detection` (isolated),
`elm-dsm detection exposed` and `elm-dsm detection init` (historical supplemental);
the existing adapter slug remains `d3d_elm_time_to_event_dsm` for compatibility.

## Model details

The weights are **not** upstream's. Upstream's four Keras graphs take 124
inputs and 64 of them are BES, which the FAITH corpus fills on 2 of 24 sampled
shots, so a model that needs BES cannot be served at corpus scale. labeler
therefore fitted the same architecture on upstream's own rows twice - once on
all 124 columns (`all124`) and once on the 60 that are not BES (`no_bes`) - and
serves the second. Architecture and hyperparameters are `hiro_scripts/model.cfg`
verbatim: `k = 3`, one 128-unit ReLU6 embedding layer with dropout 0.2,
LogNormal mixture, Adam at lr 1e-3, batch 1024, seed 0.

Inference is `1 - S(h + 1 | x)` through `runners/dsm_pickle.survival`, the same
reader the tearing survival models use. The `+ 1` ms is upstream's fit-time
offset (`new_train_elm_model.py` fits on `t + 1`), so a real horizon `h` is
queried at `h + 1`; each label's `queried_at_ms` attribute records it.

### Input order - the trap, and how it was settled

There are two orders and they differ in 118 of 124 columns. Both are written
out in cell 43 of `hiro_scripts/data_processing.ipynb`:

| order | layout | where it lives |
|---|---|---|
| `current_diagnostic_order` | ip, bt, gas, pinj, tinj, ech, **ece 1-48**, pcphd02, pcphd03, co2 x4, bes 1-64 | `compiled_model10.pkl`, the Keras graphs |
| `new_diagnostic_order` | ip, bt, gas, pinj, tinj, ech, pcphd02, pcphd03, co2 x4, **bes 1-64**, ece 1-48 | `reordered_model10.pkl`, **`train_test_split_model10.pkl`** |

Cell 47 builds the split pickle from `reordered_model10.pkl`, so the rows
labeler fits on are in the **second** order. That is measured, not read: the
pickle carries both `*_final_x` (raw) and `*_final_x_normalized`, so upstream's
per-column transform is recoverable as `s = std(raw)/std(norm)`,
`m = mean(raw) - s*mean(norm)` and can be matched against the named
`normalizations` dict in `data/testing_model.pkl`. All 124 columns of both the
train and the test side match `new_diagnostic_order` to a worst relative error
of **1.2e-12**; only 6 of 124 - the first six, which the two orders share -
match `current_diagnostic_order`. Reproduce with
`python scripts/labeler/elm_write_normalization.py --verify-split`.

The first `no_bes` fit (commit `54d201d`) took slots 0-59 of the pickle
believing them to be the non-BES columns. Under the real order those slots are
the 12 non-ECE, non-BES columns plus `bes_slow_channel_1..48`: that model
dropped every ECE channel and *required* 48 BES ones. It has been refitted.
`labeler.models.elm_inputs` now pins both orders and selects a column set by
name, so the same mistake becomes a `KeyError` rather than a silently wrong
model.

### The 60 columns, and where each comes from

| slots (of `new_diagnostic_order`) | column | canonical feature | source |
|---|---|---|---|
| 0 | `ip` (MA) | `ip` x 1e-6 | archive or fdp |
| 1 | `bt` (T) | `bt` | archive or fdp |
| 2 | `gas` (V) | `gas` = corpus `gas_raw` channel 0 | corpus |
| 3 | `pinj` (W) | `pinj_total` x 1e3 | archive or corpus |
| 4 | `tinj` (N m) | `tinj_total` | archive or corpus |
| 5 | `ech` (W) | `ech_power_total` | archive or corpus |
| 6-7 | `pcphd02`, `pcphd03` | **none** - mean-filled | - |
| 8-11 | `co2_density_slow_{r0,v1,v2,v3}` | `co2_r0..co2_v3`, 1 ms mean of the corpus `co2` record | corpus |
| 76-123 | `ece_slow_channel_1..48` | `ece`, 48 channels at 1 ms | corpus |

## Uses

Offline label generation over the FAITH shot corpus, for comparison against
IGNITE and against other models' labels. Not for real-time control and not for
physics conclusions without further validation. Historical survival-target and
ablation results describe the upstream population; reviewed occupancy and
native-input development panels below have separate coverage and exposure limits.

## Bias, risks and limitations

**Validity caveats, in the order they will bite:**

1. **Survival refit trained on wide-pedestal QH only.** Its training rows are
   inside a wide-pedestal
   QH-mode phase
   (`WPQHphases-tau_min500-tau_inter250-pewid_max4.0-pewid_min3.0-tinj_wpqh2.1`).
   The survival refit has never seen an ELMing H-mode, an L-mode or a ramp. Labels on a
   corpus shot outside that regime are extrapolation, and nothing in the
   per-row `_valid` mask says so - the regime is a property of the shot, not of
   a row's inputs.
2. **Two of 60 columns are always fabricated.** `pcphd02` and `pcphd03` sit at
   the training mean on every row of every shot. This is deliberately NOT
   folded into `_valid`: a flag that is 0 everywhere carries no information and
   would hide the rows that are genuinely untrustworthy. Each label carries it
   as the `mean_filled_columns` attribute instead.
3. **CO2 is absent on half the corpus.** `co2_r0..co2_v3` come from a group the
   corpus fills on 12 of 24 sampled shots. Where it is absent, all four columns
   are mean-filled and **every row is marked invalid** - the probability is
   still written, so the series exists and says on its face where not to trust
   it.
4. **One-epoch model.** See Evaluation.
5. **Catalog-wide adapter validation remains incomplete.** The reviewed
   occupancy benchmark and native-input audits below evaluate separate panels;
   they do not establish fidelity across the full corpus.
6. **Pre-split feature exposure in historical DSM variants.** The normalization
   constants were computed over all 365 source physical shots before the
   upstream split, including blind-cohort shots 190532 and 190646. The refit,
   historical scratch detection model and initialized detection model use them.
   This is feature-statistics exposure, independent of reviewed-label CV. Their
   scores remain supplemental. The confirmatory detector instead extracts raw
   rows without source normalization or clipping, fits measured-column statistics
   inside each optimizer-training partition, then fills missing columns and clips
   with that fold's statistics. It initializes independent random weights and
   reuses no source model parameters. Inner-validation and outer-test shots do
   not enter its preprocessing fit.

`training_membership.json` records source hashes and separate physical-shot
roles: 300 weight-training shots, 80 early-stopping shots, 15 physical shots on
both sides, and 365 normalization-exposed shots. These IDs are decoded from
upstream `<shot>_<phase>` identifiers. General validation's `in_training` subset
uses the union of all three roles, so source-exposed shots are never labelled
held out. The refit overlaps reviewed labels on shots 190637, 190643, 192721,
192751 and 196541; comparisons on those shots are in sample for the survival
weights and labels. The normalization source is `compiled_model10.pkl`, with
`model10_norms` computed over `final_x` in `data_processing.ipynb` before split.

## Training details

Rows are upstream's: 1 ms samples inside wide-pedestal QH phases, `t` is ms to
the next ELM as a sawtooth and `e = 1` at an ELM; NaN to 0, CO2 outside
`[0, 1e15]` dropped, a 100 ms boxcar on `pinj` and `tinj`, and `|z| > 10` rows
dropped on every column except the two photodiodes. The split is upstream's own
phase-record 80/20 at seed 0, read as-is and never re-split. It is not a disjoint
physical-shot split: 15 physical shots have phases on both sides. Its test rows
were used for early stopping and model/input-set selection, so evaluation on
them is selection-informed validation evidence. Normalization was computed
before this split.

Two upstream defects worth knowing: `train_elm_model.py` has train and test
**swapped** (it trains on the last 10% of shots), and an earlier version of
this card claimed a Weibull mixture, which nothing supports -
`hiro_scripts/model.cfg` says LogNormal.

The trainer is `scripts/labeler/elm_dsm_train.py`. It reproduces the fork's
loop (`pretrain_dsm`, Adam, per-epoch shuffle at `random_state=i`,
`conditional_loss(elbo=True)` on minibatches, full-set validation, argmin
reload, `train_patience = 5`) with three changes, each forced by a measurement
recorded in its docstring: a wall-clock deadline, a validation pass in `eval()`
mode, and a stop at the first non-finite loss.

## Evaluation

### Isolated reviewed-label detection

The confirmatory detector completes 40 epochs in each of the five fixed,
shot-grouped folds used by `elm-ours`. Inner-validation AUPRC selects the
checkpoint and inner-validation F1 selects the threshold. The selected epochs
(zero based) are 14, 34, 9, 30 and 1; thresholds are 0.207, 0.376, 0.544,
0.017 and 0.451. This variability is development evidence, and the task is
reviewed ELMy-phase occupancy rather than independently validated individual
ELM onset detection. Source parameter reuse and source normalization reuse are
both false for every fold, recorded with its fitting shot IDs, normalization
constants, raw-row hashes and checkpoint hash.

| Common panel | Shots | 50 ms bins | AUROC [95% shot CI] | F1 [95% shot CI] |
|---|---:|---:|---|---|
| All reviewed shots | 119 | 11,653 | 0.855 [0.813, 0.896] | 0.754 [0.696, 0.811] |
| BES-covered shots | 73 | 6,527 | 0.856 [0.806, 0.902] | 0.780 [0.705, 0.842] |

The source is [the detector evaluation JSON](../../../../outputs/labeler/elm/dsm/evaluation.json),
`sets.{all119,bes73}.methods.elm-dsm-detect`, produced by
`scripts/labeler/elm_dsm_evaluate.py --run cv2 --epochs 40 --device cuda`.
The historical normalization-exposed scratch and source-initialized scores are
retained as `elm-dsm-detect-exposed` and `elm-dsm-detect-init`; the latter also
inherits survival-weight and source checkpoint-selection exposure.

The limited-input survival refit's own-target AUROCs at 5, 10, 20 and 50 ms
are 0.758 [0.700, 0.811], 0.764 [0.708, 0.817], 0.770 [0.716, 0.822] and
0.777 [0.722, 0.830], on 80 physical early-stopping validation shots. These
are selection-informed results. Its served checkpoint is **after one epoch**,
selected as `best_epoch=0` from a **seven-epoch run**, rather than a run that
stopped after one epoch. Exact values and physical-shot bootstrap intervals
are in that JSON's `own_target.horizons`; the source training record is
`training_no_bes.json` beside the installed checkpoint.

### Native original-checkpoint audit

`scripts/labeler/elm_dsm_native.py` evaluates parameter setting 1 from the
original `hiro_scripts/models/model9.pkl`, identified by the upstream export
notebook and verified against the shipped `wpqh1` embedding and head kernels.
It retains all 124 inputs, including 64 BES channels and PCPHD02/03, at 1 ms,
and uses the original PyTorch ReLU6 checkpoint. The Keras conversion contains
unbounded ReLU activations, so it is not silently substituted for this checkpoint.

Exact source exports exist on five reviewed shots. Four have scored review
overlap (190637, 190643, 192721 and 196541), totaling 11,565 one-millisecond
rows; 192751's exported phase rows do not overlap a scored review span.
The exact-export 50 ms risk AUROC is 0.465 [0.139, 0.653]. Every shot in this
small panel is source exposed: 196541 entered optimizer fitting and the other
three entered checkpoint selection, and all entered source normalization.
No threshold is tuned for this native panel. It is a separate continuous-score
comparison with limited phase coverage, not a held-out confirmatory result.

Original diagnostic H5 records plus existing or paced, isolated PCPHD02/03
fetches make 45 reviewed shots reconstructable. The reconstruction is reported
separately because the source smoothed concatenated filtered phase rows whereas
serving smooths each shot's normalized NBI trace. It retains original 1 ms
sampling and all 124 columns without mean filling or clipping. Per-column
agreement with exact source exports, remaining missing columns for every
reviewed shot, filter-domain failures and coverage are recorded in
[the native evaluation JSON](../../../../outputs/labeler/elm/dsm/native_evaluation.json).
The fetch record is [native_fetch.json](../../../../outputs/labeler/elm/dsm/native_fetch.json).

The original checkpoint's own-target AUROCs at 5/10/20/50 ms are
0.926 [0.915, 0.937], 0.931 [0.919, 0.941], 0.939 [0.927, 0.949] and
0.955 [0.945, 0.964], on 326 physical source validation shots. The original
trainer's reversed chronological partition is preserved: the last 10% of phase
records train the model, and the first 90% select it. These are historical
checkpoint-selection results, not untouched evaluation; exact values and
row/case/control counts are in `native_evaluation.json:own_target.horizons`.

### BES ablation, refitted on the correct columns (2026-09-06)

`all124` is upstream's input set; `no_bes` is the 60 columns of the split
pickle that are not BES (slots 0-11 and 76-123 of `new_diagnostic_order`).
Rows are upstream's 1 ms wide-pedestal-QH test rows, **not** the FAITH corpus
and not labeler's 500-shot pool: these numbers describe upstream's
population. The score is `1 - S(h + 1)` through
`runners/dsm_pickle.survival`; AUROC cases are `e == 1 and t <= h` against
controls `t > h`, rows censored inside `h` excluded; IPCW is
`labeler.alarm.ipcw_auc` on the same score.

| set | inputs | test NLL | AUROC 5 ms | 10 ms | 20 ms | 50 ms | IPCW AUC 20 ms |
|---|---|---|---|---|---|---|---|
| `all124`‡ | 124 | 1.2553 | 0.7562 | 0.7624 | 0.7680 | 0.7807 | 0.7680 |
| **`no_bes`**‡ | **60** | **1.1542** | **0.7581** | **0.7643** | **0.7699** | **0.7771** | **0.7699** |

‡ marks supplemental source exposure: these rows selected the checkpoints and
input set, and were included in upstream normalization.

**Decision rule, written before the run:** adopt `no_bes` if its test NLL is
within 0.02 of `all124`'s *and* its 20 ms AUROC is within 0.02. It passes both
by improving on both: **ΔNLL -0.1010** and **Δ20 ms AUROC +0.0019** in
`no_bes`'s favour. **Verdict: adopt `no_bes`; BES is droppable.** (50 ms is the
one horizon where `all124` is ahead, by 0.0036.)

This reverses the verdict this card carried on 2026-09-06 before the column
order was measured. The earlier "`no_bes` costs +0.051 NLL and -0.030 AUROC"
compared `all124` against a model that had been fitted on 12 non-BES columns
plus 48 BES ones and no ECE at all. That comparison is void; this one replaces
it.

Two caveats that belong with the numbers rather than under them:

* **best epoch is 0** for every fit run so far. Test NLL rises monotonically
  from epoch 1 (`no_bes` 1.154 -> 1.372 by epoch 6, where the fork's patience
  fired; `all124` diverged to NaN at epoch 16). A rerun at lr 1e-4 with a
  by-shot validation carve out of the training shots (job 2924075, discarded)
  had best epoch 0 as well and was *worse* on the held-out test split, so this
  is not a learning-rate problem: the optimum lies inside the first pass over
  540k rows and a once-per-epoch curve cannot see it. The fix is sub-epoch
  checkpointing;
* upstream passes the **test** split as `val_data`, which this fit reproduces,
  so the early stop selects on the test rows. `test NLL` is a
  selection-informed number, not held out. Both sets are selected identically,
  so the comparison between them is fair.

### Coverage on the corpus

`features` + `infer` (under `fdp run`) on the 24 corpus shots of
`sample_shots(corpus_shots, 24, seed=0)`, measured 2026-09-06. All 24 shots
produce all four label series over 240 rows each; **6 of 24 have a non-zero
valid fraction** and 18 have none, because a canonical feature is missing:

| feature | absent on |
|---|---|
| `co2_r0..co2_v3` | 12 of 24 (`corpus:SignalAbsent`, every shot below 198279) |
| `pinj_total`, `tinj_total` | 11 of 24 (`archive:ShotNotInArchive,corpus:SignalAbsent`; neither has an fdp source in `namespace.py`) |
| `ece` | 4 of 24 |
| `gas`, `ech_power_total` | 2 of 24 |
| `ip`, `bt` | 0 of 24 - but only under `fdp run`; an un-wrapped run reports `fdp:PtDataError` on 19 of 24 and every row then goes invalid |

Valid fractions on the six complete shots: 199025 0.542, 199421 0.621, 200417
0.971, 201828 0.517, 201881 0.992, 203898 0.092. Peak `elm_risk_20ms` over the
24 shots ranges 0.000-0.345.

On a shot where every input is present the remaining invalidity is upstream's
own `|z| > 10` row filter, i.e. the row is outside the wide-pedestal-QH
training distribution - not a missing column. MEASURED on the demo shot
**199597** (all 11 features present, 240 rows, valid fraction **0.192**): the
filter fires on `tinj` for 139 rows (max |z| 14.8) and on `gas` for 55 rows
(max |z| 16.0), and on nothing else. That is the filter doing its job; it is
also why a corpus shot far from wide-pedestal QH will show a mostly-grey panel.

Artifacts, with `training_no_bes.json`, `PROVENANCE_no_bes.json`,
`ablation.json`, `normalization.json` and the loss curves:
`/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/models/d3d_elm_time_to_event_dsm/`.
The refit ran on `stellar-vis2` (`sbatch` was unavailable to the session; the
original 8-CPU SLURM run of the same script is job 2924069): 82 s wall, 350% of
4 threads, 4.33 GB peak RSS, deterministic at seed 0.

## Technical specifications

60 -> 128 ReLU6 embedding (dropout 0.2, identity in eval mode), then gate,
scale and shape heads at `(128, 3)`, LogNormal mixture, all float64. Read by
`runners/dsm_pickle.py`, not by Keras.

## Citation

Unpublished internal model. The architecture and the training rows are Hiro
Farre-Kaga's (`/projects/EKOLEMEN/wpqh_elm_hiro/`); the fitted weights served
here are labeler's. Attribute to the PlasmaControl group, Princeton.

## Contact

`nc1514@princeton.edu`.
