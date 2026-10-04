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
  display_name: elm-dsm-survival
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
      pre-split normalization. General validation's in_training subset
      includes every shot used in those roles; none is marked held_out.
  serving:
    interpretation: offline risk score, not a causal forecast
    trained_row_ms: 1.0
    serving_window_ms: 50.0
    serving_grid_ms: 25.0
    centered_nbi_lookahead_ms: 25.0
  preprocessing_exposure:
    applies_to:
      - elm-dsm-survival
      - elm-dsm-detect (source statistics)
      - elm-dsm-detect (source weights and statistics)
    scope: upstream means and standard deviations computed before the source split
    blind_cohort_shots: [190532, 190646]
    role: feature statistics reuse upstream shots regardless of reviewed-label folds
    decision: historical fits are supplemental; isolated detection refits preprocessing
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
      - "2026-10-03: the no_bes/all124 comparison is source-validation selection evidence only; it does not establish that BES is dispensable or that one architecture outperforms the native [100,1000] checkpoint."
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

# elm-dsm-survival

**Status: implemented.** Offline ELM risk scores at 5, 10, 20 and 50 ms, from a
Deep Survival Machines refit. Serving uses 50 ms means on a 25 ms grid for a
model trained on 1 ms rows. The centered NBI boxcar incorporates the row 25 ms
later, so these scores do not support causal forecasting claims. The paper's
short names are `elm-dsm-survival`,
`elm-dsm-detect (60-input 1×128)` (separate preprocessing per fold),
`elm-dsm-detect (source statistics)` and
`elm-dsm-detect (source weights and statistics)` (historical supplemental);
the existing adapter slug remains `d3d_elm_time_to_event_dsm` for compatibility.

## Model details

The weights are **not** upstream's. Upstream's four Keras graphs take 124
inputs and 64 of them are BES, which the FAITH corpus fills on 2 of 24 sampled
shots, so a model that needs BES cannot be served at corpus scale. labeler
therefore fitted a smaller architecture on upstream's own rows twice - once on
all 124 columns (`all124`) and once on the 60 that are not BES (`no_bes`) - and
serves the second. The native checkpoint uses layers `[100, 1000]`; both
refits use one 128-unit layer. Refit hyperparameters are `k = 3`, ReLU6,
dropout 0.2,
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
6. **Pre-split feature statistics in historical DSM variants.** The normalization
   constants were computed over all 365 source physical shots before the
   upstream split, including blind-cohort shots 190532 and 190646. The refit,
   historical random-weight detection model and pretrained detection model use them.
   These statistics reuse upstream shots regardless of reviewed-label folds.
   Their scores remain supplemental. The reviewed detection adaptation instead extracts raw
   rows without source normalization or clipping, fits measured-column statistics
   inside each optimizer-training partition, then fills missing columns and clips
   with that fold's statistics. It starts from independent random weights and
   reuses no source model parameters. Inner-validation and outer-test shots do
   not enter its preprocessing fit.

`training_membership.json` records source hashes and separate physical-shot
roles: 300 weight-training shots, 80 early-stopping shots, 15 physical shots on
both sides, and 365 shots used for normalization. These IDs are decoded from
upstream `<shot>_<phase>` identifiers. General validation's `in_training` subset
uses the union of all three roles, so shots used for any role are never labelled
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

### Reduced-input reviewed-label detection adaptation

These are developmental shot-CV occupancy estimates on five fixed folds. Preliminary
outer-fold predictions were available before the reported recipe was fixed and could
have informed inputs, scaling, architecture, selection or evaluation; the saved records
do not establish their influence. Inner-validation AUPRC selects checkpoints at
zero-based epochs 35, 31, 2, 24, 0; inner-validation F1 selects thresholds 0.067, 0.122,
0.482, 0.021, 0.395. Each fold fits its own normalization and starts from random
weights. No blind-cohort shots enter these fits.

| Common panel | Shots | 50 ms bins | AUROC [95% shot CI] | AUPRC | F1 |
|---|---:|---:|---|---|---|
| All reviewed | 119 | 11,653 | 0.845 [0.796, 0.893] | 0.748 [0.651, 0.832] | 0.742 [0.681, 0.798] |
| BES subset only | 73 | 6,527 | 0.840 [0.777, 0.892] | 0.731 [0.612, 0.836] | 0.783 [0.711, 0.842] |

Source: [dsm/evaluation.json](../../../../outputs/labeler/elm/dsm/evaluation.json),
`detectors.elm-dsm-detect` and `sets`. These DSM detection rows are lower bounds on DSM
detection skill under our recipe, not the best achievable DSM performance. With
post-warm-up checkpoint selection and 3 further seeds the all119 AUROC has mean 0.865
(range 0.857–0.870); the native comparator below is refitted the same way.

The detector uses 60 input columns. PCPHD02/03 means come from a fresh fetch on 111 of
119 shots and from the upstream WPQH PCPHD02/03 export on 8; DENV2F and DENV3F means
supply the two density columns on 115 and 115 shots, with 4 and 4 rejected (failed
digitiser) and mean-filled. This is elm-dsm-detect (60-input 1×128), a reduced-input
adaptation trained and evaluated on 50 ms rows. The source model trained on native 1 ms
rows with 124 inputs and layers [100, 1000] for WPQH breakthrough-ELM forecasting; this
is not an objective-only retrain of that model. A fresh fetch of the 8 swap shots'
PCPHD02/03 equals the upstream export sample for sample in 16 of 16 records (maximum
absolute difference 0; `dsm/swap_photodiode_agreement.json`), so the two sources agree
in scale and the detection AUROC gap on those shots is not an input-source artefact.
Fast-density units and filterscope sightlines are unverified in retained metadata; fixed
input scaling, clipping and magnitude screening do not establish physical calibration.
Over the flat-top window (1–4 s) of the 119 reviewed shots the divided fast-density
input has median 0.84, 5–95% range 0.17–1.93, and 0.4% of cells sit at the upper clip
(12; 9 of 238 chords are zeroed or empty in the window; `density_range.json`,
`scripts/labeler/elm_density_range.py`). This is a numerical range only: the ordinate
units and the FS02–04 sightlines are unverified, and no sightline list is retained in
the signal records or the literature digests.

The companion occupancy U-Net omits FS01 because its retained cache contains FS02–04
only. Its fast-density inputs divide native values by `1e14`, clip to `[-3, 12]`, and
clip ten times the 0.2 s high-pass to `[-10, 10]`; chords with median absolute native
magnitude above `1e16` are zeroed by a heuristic failed-digitiser screen. Offline
metadata audits made no new fetches and changed no saved inputs or weights. Sources:
`density_units.json`, `filterscope_metadata.json` and `src/labeler/elm/inputs.py`. No
independently validated physical-onset detector is delivered; run days cross folds in
the review analysis (16 of 94 review days) and the Smith onset folds are grouped by run
day (none of 31 days crosses folds).

### Limited-input survival refit (selection evidence)

The served checkpoint is after 1 epoch of a 7-epoch run, selected at best_epoch=0. Its
80 physical source-validation shots selected the checkpoint; they are not independent
test evidence.

| Horizon | AUROC [95% physical-shot CI] |
|---|---|
| 5 ms | 0.758 [0.700, 0.811] |
| 10 ms | 0.764 [0.708, 0.817] |
| 20 ms | 0.770 [0.716, 0.822] |
| 50 ms | 0.777 [0.722, 0.830] |

Source: `dsm/evaluation.json:own_target.horizons`.

### Native original-checkpoint audit

The original 124-input, 1 ms checkpoint has embedding layers [100, 1000]. The
limited-input survival refit and reviewed detector instead use one 128-unit layer.
Native evaluation uses model9 parameter setting 1 with ReLU6; the Keras conversion's
unbounded ReLU is not substituted.

The corrected forward presence target asks whether a reviewed present span intersects
(t, t+h]; the separate onset target asks whether a non-crowd start lies in that
interval. Reviewed non-crowd starts are annotation boundaries without independent
physical-onset truth.

Exact-export 50 ms AUROC is 0.607, CI null: descriptive only on 4 shots reused in source
fitting and 11,565 rows (190637, 190643, 192721, 196541). 196541 entered optimizer
fitting; the other three entered checkpoint selection; all entered source normalization.
Five shots have exact exports, but 192751 has no scored overlap, so the exact-export
JSON contains four shots. No operating threshold is selected.

| Native panel | Horizon | Shots | Rows | AUROC [95% physical-shot CI] |
|---|---|---:|---:|---|
| Reconstructed presence | 5 ms | 33 | 138,049 | 0.610 [0.541, 0.705] |
| Reconstructed presence | 10 ms | 33 | 137,986 | 0.613 [0.544, 0.708] |
| Reconstructed presence | 20 ms | 33 | 137,825 | 0.620 [0.549, 0.715] |
| Reconstructed presence | 50 ms | 33 | 137,282 | 0.709 [0.630, 0.789] |
| Source selection target | 5 ms | 326 | 699,512 | 0.926 [0.915, 0.937] |
| Source selection target | 10 ms | 326 | 697,855 | 0.931 [0.919, 0.941] |
| Source selection target | 20 ms | 326 | 694,596 | 0.939 [0.927, 0.949] |
| Source selection target | 50 ms | 326 | 685,337 | 0.955 [0.945, 0.964] |

Reconstruction is a sensitivity: source smoothing of concatenated phase rows differs
from within-shot NBI smoothing. Source validation retains the original reversed
chronological split and selected weights; it is not independent evaluation. Coverage,
inputs, targets and memberships are in
[native_evaluation.json](../../../../outputs/labeler/elm/dsm/native_evaluation.json).
Reproduce scoring without training with `elm_dsm_evaluate.py --rescore` and
`elm_dsm_native.py`; render this block with `elm_protocol.py`.

### Native DSM detection comparator

The timestamp-aware 1 ms audit finds at least 112/124 inputs on 48 reviewed shots; 37
have complete 124-input scored bins. Missing column names and shot counts are in
`dsm/native_detection.json:coverage,missing_column_shot_counts`.

The native [100,1000] ReLU6 architecture was refitted for occupancy on complete-input
shots only, with random weights and optimizer-training-only normalization. The fixed
25-epoch recipe uses the original five outer folds and their inner-validation shot
partitions; checkpoint AUPRC and F1 thresholds are selected only on inner validation. No
source weights or statistics and no blind-test shots are reused. The native refit trains
on 24–28 shots per fold with 2–8 inner-validation shots (train/inner-validation by fold:
24/2, 28/3, 24/7, 24/3, 25/8), against 81–82 and 14 for the headline elm-ours; its
thresholds, chosen on so few shots, are erratic (3.1e-05–0.9999).

Inputs are timestamp-aware 1 ms means from stored original corpus H5 records and
retained PCPHD02/03. Standardized inputs are clipped at ±10; NBI uses the source's
100-row centered smoothing within each shot rather than across concatenated source
phases. This reconstructs native diagnostic inputs, not bit-identical historical
exported rows. Bin scores average the 50 native row probabilities; measured support and
target are identical for every compared method.

| Matched panel / method | Shots / bins | AUROC [95% shot CI] | AUPRC | F1 |
|---|---|---|---|---|
| all119 / elm-ours | 37 / 3,576 | 0.936 [0.862, 0.981] | 0.853 [0.632, 0.976] | 0.833 [0.727, 0.908] |
| all119 / elm-ours (native-fold training shots) | 37 / 3,576 | 0.863 [0.795, 0.923] | 0.785 [0.599, 0.907] | 0.760 [0.654, 0.841] |
| all119 / elm-dsm-detect (60-input 1×128) | 37 / 3,576 | 0.823 [0.735, 0.899] | 0.704 [0.525, 0.849] | 0.759 [0.628, 0.852] |
| all119 / elm-dsm-native-detect (124-input [100,1000]) | 37 / 3,576 | 0.734 [0.610, 0.856] | 0.598 [0.396, 0.822] | 0.664 [0.509, 0.784] |
| bes73 / elm-ours | 37 / 3,339 | 0.929 [0.848, 0.980] | 0.854 [0.632, 0.976] | 0.837 [0.730, 0.913] |
| bes73 / elm-ours (native-fold training shots) | 37 / 3,339 | 0.852 [0.780, 0.917] | 0.789 [0.603, 0.910] | 0.766 [0.661, 0.845] |
| bes73 / elm-dsm-detect (60-input 1×128) | 37 / 3,339 | 0.805 [0.708, 0.889] | 0.706 [0.530, 0.850] | 0.763 [0.634, 0.854] |
| bes73 / elm-dsm-native-detect (124-input [100,1000]) | 37 / 3,339 | 0.715 [0.584, 0.844] | 0.604 [0.401, 0.825] | 0.667 [0.512, 0.789] |
| bes73 / elm-elmo | 37 / 3,339 | 0.890 [0.818, 0.948] | 0.808 [0.677, 0.910] | 0.818 [0.722, 0.889] |

`elm-ours (native-fold training shots)` repeats the elm-ours network, recipe and seeds
on the native folds' own train and inner-validation shots
(`scripts/labeler/elm_native_ours.py`; its fold thresholds span 0.013–0.782). On the
same bins it scores AUROC 0.863 [0.795, 0.923] against 0.936 [0.862, 0.981] for the
headline elm-ours, 0.734 [0.610, 0.856] for the native refit and 0.823 [0.735, 0.899]
for the 60-input adaptation. At equal training shots elm-ours has the higher point AUROC
than the native refit, with overlapping intervals. The headline's lead over the native
refit therefore mixes training size with architecture. This smaller support panel is a
secondary control and does not replace the primary all119/bes73 benchmark.

### DSM baselines retrained with post-warm-up selection

Both detection variants were refitted with three further seeds on the same outer folds
and recipe, choosing the epoch that ends the best three-epoch mean inner-validation
AUPRC window lying wholly at or after the warm-up (6 of 40 epochs for the 60-input
adaptation, 4 of 25 for the native refit); the reported fit takes the raw best epoch.
Ranges are over seeds, not intervals. Selected epochs span 8–37 and 6–24; native
thresholds span 0.0002–1.000, an erratic operating point.

| Detector | Shots / bins | Fit | AUROC | AUPRC | F1 |
|---|---|---|---|---|---|
| elm-dsm-detect (60-input 1×128) | 119 / 11,653 | reported (raw selection) | 0.845 | 0.748 | 0.742 |
| elm-dsm-detect (60-input 1×128) | 119 / 11,653 | post-warm-up repeats, mean (range) | 0.865 (0.857–0.870) | 0.740 (0.722–0.751) | 0.755 (0.749–0.763) |
| elm-dsm-native-detect (124-input [100,1000]) | 37 / 3,576 | reported (raw selection) | 0.734 | 0.598 | 0.664 |
| elm-dsm-native-detect (124-input [100,1000]) | 37 / 3,576 | post-warm-up repeats, mean (range) | 0.715 (0.707–0.725) | 0.598 (0.595–0.600) | 0.626 (0.603–0.642) |

These DSM detection rows are lower bounds on DSM detection skill under our recipe, not
the best achievable DSM performance. Source: `dsm/baseline_seeds.json`.

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

‡ marks supplemental reuse of upstream data: these rows selected the checkpoints and
input set, and were included in upstream normalization.

**Decision rule, written before the run:** adopt `no_bes` if its test NLL is
within 0.02 of `all124`'s *and* its 20 ms AUROC is within 0.02. It passes both
by improving on both: **ΔNLL -0.1010** and **Δ20 ms AUROC +0.0019** in
`no_bes`'s favour. **Selection: serve `no_bes` under this validation rule.** (50 ms is the
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
  so this remains a selection comparison; it does not establish that BES is
  dispensable in other populations or architectures.

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
