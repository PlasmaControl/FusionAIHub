---
language: en
license: other
library_name: pytorch
pipeline_tag: time-series-forecasting
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
  upstream:
    path: /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/models/d3d_elm_time_to_event_dsm
    artifacts:
      - elm_dsm_no_bes.pkl
      - normalization.json
    sha256:
      elm_dsm_no_bes.pkl: 2c28518eb1934c542ba007f7971d6ad2e38543be6f61ce57508b8629443aa9c3
      normalization.json: 0d3e99e3e2e7f1aee5bb3aa9d40a9393d81e12789428b3e1ae9c23ab1598e45e
    notes:
      - "2026-09-06: the weights are labelmaker's own, not upstream's Keras
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
    - "bt: upstream computed 1.69861e-5 * pcbcoil; labelmaker uses the canonical
      `bt` in tesla for its much wider shot coverage. Measured at t = 2.0 s on
      shot 185808: -2.0910 T against -2.0306 T, a 3.0% difference."
    - "ech: upstream took `echpwrc`, column 1 of the staged `ech` group;
      labelmaker uses ech_power_total (the 12-gyrotron corpus sum, or the
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
    - "the grid: upstream's rows are 1 ms means, labelmaker's are 25 ms means
      (the archive's 50 ms boxcar ending at t). A sampling change, not a model
      change - no weight and no normalisation constant differs."
    - "upstream's 100 ms boxcar on the raw pinj and tinj columns (NBI is
      modulated) is reproduced as a 4-tap centred moving average on the 25 ms
      grid."
  resolved:
    - "2026-09-05: the 124 trained-on names and their order ARE recovered - see Input order below"
    - "2026-09-06: and the order of the SPLIT PICKLE is new_diagnostic_order, measured to 1.2e-12; see upstream.notes"
    - "2026-09-05: the embedding does bake in its own normalisation in the Keras graphs; labelmaker's own fit uses upstream's normalizations dict instead, written to normalization.json"
    - "2026-09-05: the graph is structurally identical to d3d_tearing_time_to_event_dsm, so runners/dsm_pickle.survival() applies verbatim"
    - "2026-09-06: the aggregation from the 1 ms training grid to labelmaker's 25 ms grid is a sampling change; the risk is read at h + 1 ms"
    - "2026-09-06: the DSM head becomes a label series as 1 - S(h + 1) at four horizons, not as an expected time to event"
    - "2026-09-06: BES is droppable after all - with the columns identified correctly the 60-column fit BEATS the 124-column one"
  blocked_on:
    - "the fit is a one-epoch model at lr 1e-3 and at lr 1e-4 alike; a model worth trusting numerically needs sub-epoch checkpointing (validate every N minibatches), which no run has done yet"
    - "pcphd02 / pcphd03 have no corpus group, so 2 of 60 columns are mean-filled on every row of every shot; serving them would need an fdp/toksearch PTDATA fetch of the two photodiodes"
    - "labelmaker's own validation (adapter fidelity, reconstruction fidelity, label quality) has not been run for this slug; the numbers below are upstream-population numbers only"
    - "the training-shot list is not committed, so `validate` reports every pool shot as held out"
    - "corpus coverage: only 6 of the 24 sampled corpus shots have all 11 inputs, so 18 produce labels with no valid row at all. co2 is corpus:SignalAbsent below shot 198279 (12 of 24) and pinj_total/tinj_total have no fdp source in namespace.py, only archive+corpus (11 of 24). Serving the corpus properly needs an fdp NBI fetch and a decision about pre-198279 CO2"
---

# plasmacontrol/d3d-elm-time-to-event-dsm

**Status: implemented.** Probability that an ELM occurs within 5, 10, 20 and
50 ms, on labelmaker's 25 ms grid, from a Deep Survival Machines model
labelmaker fitted itself.

## Model details

The weights are **not** upstream's. Upstream's four Keras graphs take 124
inputs and 64 of them are BES, which the FAITH corpus fills on 2 of 24 sampled
shots, so a model that needs BES cannot be served at corpus scale. Labelmaker
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
labelmaker fits on are in the **second** order. That is measured, not read: the
pickle carries both `*_final_x` (raw) and `*_final_x_normalized`, so upstream's
per-column transform is recoverable as `s = std(raw)/std(norm)`,
`m = mean(raw) - s*mean(norm)` and can be matched against the named
`normalizations` dict in `data/testing_model.pkl`. All 124 columns of both the
train and the test side match `new_diagnostic_order` to a worst relative error
of **1.2e-12**; only 6 of 124 - the first six, which the two orders share -
match `current_diagnostic_order`. Reproduce with
`python scripts/labelmaker/elm_write_normalization.py --verify-split`.

The first `no_bes` fit (commit `54d201d`) took slots 0-59 of the pickle
believing them to be the non-BES columns. Under the real order those slots are
the 12 non-ECE, non-BES columns plus `bes_slow_channel_1..48`: that model
dropped every ECE channel and *required* 48 BES ones. It has been refitted.
`labelmaker.models.elm_inputs` now pins both orders and selects a column set by
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
physics conclusions without the reliability numbers below - which, for this
slug, are upstream-population numbers only.

## Bias, risks and limitations

**Validity caveats, in the order they will bite:**

1. **Wide-pedestal QH only.** Every training row is inside a wide-pedestal
   QH-mode phase
   (`WPQHphases-tau_min500-tau_inter250-pewid_max4.0-pewid_min3.0-tinj_wpqh2.1`).
   The model has never seen an ELMing H-mode, an L-mode or a ramp. Labels on a
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
5. **Never validated on labelmaker's own pool.** `validate` has not been run
   for this slug, so nothing here says how these labels behave on corpus shots.

## Training details

Rows are upstream's: 1 ms samples inside wide-pedestal QH phases, `t` is ms to
the next ELM as a sawtooth and `e = 1` at an ELM; NaN to 0, CO2 outside
`[0, 1e15]` dropped, a 100 ms boxcar on `pinj` and `tinj`, and `|z| > 10` rows
dropped on every column except the two photodiodes. The split is upstream's own
by-shot 80/20 at seed 0, read as-is and never re-split.

Two upstream defects worth knowing: `train_elm_model.py` has train and test
**swapped** (it trains on the last 10% of shots), and an earlier version of
this card claimed a Weibull mixture, which nothing supports -
`hiro_scripts/model.cfg` says LogNormal.

The trainer is `scripts/labelmaker/elm_dsm_train.py`. It reproduces the fork's
loop (`pretrain_dsm`, Adam, per-epoch shuffle at `random_state=i`,
`conditional_loss(elbo=True)` on minibatches, full-set validation, argmin
reload, `train_patience = 5`) with three changes, each forced by a measurement
recorded in its docstring: a wall-clock deadline, a validation pass in `eval()`
mode, and a stop at the first non-finite loss.

## Evaluation

### BES ablation, refitted on the correct columns (2026-09-06)

`all124` is upstream's input set; `no_bes` is the 60 columns of the split
pickle that are not BES (slots 0-11 and 76-123 of `new_diagnostic_order`).
Rows are upstream's 1 ms wide-pedestal-QH test rows, **not** the FAITH corpus
and not labelmaker's 500-shot pool: these numbers describe upstream's
population. The score is `1 - S(h + 1)` through
`runners/dsm_pickle.survival`; AUROC cases are `e == 1 and t <= h` against
controls `t > h`, rows censored inside `h` excluded; IPCW is
`labelmaker.alarm.ipcw_auc` on the same score.

| set | inputs | test NLL | AUROC 5 ms | 10 ms | 20 ms | 50 ms | IPCW AUC 20 ms |
|---|---|---|---|---|---|---|---|
| `all124` | 124 | 1.2553 | 0.7562 | 0.7624 | 0.7680 | 0.7807 | 0.7680 |
| **`no_bes`** | **60** | **1.1542** | **0.7581** | **0.7643** | **0.7699** | **0.7771** | **0.7699** |

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
here are labelmaker's. Attribute to the PlasmaControl group, Princeton.

## Contact

`nc1514@princeton.edu`.
