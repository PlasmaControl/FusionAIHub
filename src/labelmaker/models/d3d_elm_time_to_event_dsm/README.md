---
language: en
license: other
library_name: keras
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
  status: scaffold
  slug: d3d_elm_time_to_event_dsm
  card_id: plasmacontrol/d3d-elm-time-to-event-dsm
  framework: keras
  time_step_ms: null
  ensemble_n: 1
  upstream:
    path: /projects/EKOLEMEN/wpqh_elm_hiro/hiro_scripts/
    artifacts:
      - wpqh1_embedding_with_norm.keras
      - wpqh1_gate.keras
      - wpqh1_scaleg.keras
      - wpqh1_shapeg.keras
    sha256: {}
    notes:
      - "2026-09-06: labelmaker fitted its own DSMs on upstream's own split
        (train_test_split_model10.pkl, 629,023 train / 142,745 test rows,
        never re-split), same architecture and hyperparameters, once on all 124
        inputs and once on the 60 non-BES ones. See Evaluation for the ablation
        table; artifacts in
        /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/models/d3d_elm_time_to_event_dsm/
        (elm_dsm_all124.pkl, elm_dsm_no_bes.pkl, ablation.json, SLURM 2924069)."
      - "2026-09-06: BES is NOT droppable. The decision rule (no_bes adopted if
        its test NLL is within 0.02 and its 20 ms AUROC within 0.02 of all124's)
        fails on both legs: dNLL +0.0510, d20 ms AUROC -0.0298. Fetching BES is
        now a separate decision rather than an assumption."
      - "2026-09-06: `current_diagnostic_order` is NOT stored inside
        train_test_split_model10.pkl (the pickle holds only the six train_*/test_*
        arrays). The 124-name order is taken from cell 43 of
        hiro_scripts/data_processing.ipynb and is pinned in
        labelmaker.models.elm_inputs, whose tests hold BES at slots 60-123."
      - "2026-09-06: hiro's auton-survival fork inserts nn.Dropout(0.2) after every
        embedding ReLU6 (survival_tm_2's does not), so runners/dsm_pickle.load_dsm
        now skips Dropout layers - identity in eval mode, no weights."
      - "2026-09-06: at lr 1e-3 both fits reach their best test NLL after ONE epoch
        and get monotonically worse after it; all124 diverged to NaN at epoch 16.
        The numbers below are therefore one-epoch models. A useful ELM model needs a
        lower learning rate and a validation split carved out of the training shots
        (upstream passes the test split as val_data, so the early stop reads the
        test rows)."
  inputs: []
  outputs: []
  approximations: []
  resolved:
    - "2026-09-05: the 124 trained-on names and their order ARE recovered - see Input order below"
    - "2026-09-05: the embedding does bake in its own normalisation (identity kernel with bias = -mean, then diag(1/std)); the constants need no recovery"
    - "2026-09-05: the graph is structurally identical to d3d_tearing_time_to_event_dsm (k=3, 100 then 1000 hidden, gate/scaleg/shapeg heads at (1000,3)), so runners/dsm_pickle.survival() applies verbatim"
  blocked_on:
    - "BES is 64 of the 124 inputs and the corpus fills it on only 2 of 24 sampled shots; serving this model at corpus scale needs an fdp/toksearch BES fetch plus a 1 ms downsample. MEASURED 2026-09-06: dropping BES costs +0.051 test NLL and -0.030 20 ms AUROC, so the fetch cannot simply be skipped"
    - "the two labelmaker-trained DSMs both stop at their first epoch (see Evaluation); a model worth serving needs a lower learning rate and a validation split taken from the training shots"
    - "the upstream grid is 1 ms, labelmaker's is 25 ms; decide the aggregation"
    - "decide how a deep-survival-machines head becomes a label series: hazard at fixed horizons, or expected time to event"
    - "DECIDED 2026-09-05: reproducing these exact weights is no longer the goal - labelmaker will train its own model from the same inputs and labels, so this folder may be superseded by a labelmaker-owned slug (phase3-design section 6)"
---

# plasmacontrol/d3d-elm-time-to-event-dsm

**Status: scaffold.** Labelmaker cannot run this model yet. Nothing here loads
weights and no labels are produced; the folder exists so the roster, the naming
scheme and the known upstream location are recorded in one place.

## Model details

Predicts time to the next ELM from a wide 0-D and diagnostic-channel feature
vector as a mixture of parametric distributions (deep survival machines). Four
saved graphs: an embedding network with normalisation, and gate, scale and shape
heads.

The mixture family is **not** decidable from the weights. The only configuration
in the upstream directory (`hiro_scripts/model.cfg`) says `LogNormal`; an earlier
version of this card said Weibull, which nothing supports. Treat it as LogNormal
and confirm before use.

### Input order (measured 2026-09-05)

124 columns, in `current_diagnostic_order` from cell 43 of
`hiro_scripts/data_processing.ipynb` (cell 34 in an earlier reading of the
notebook; re-checked 2026-09-06). Pinned in `labelmaker.models.elm_inputs`:

| slots | contents |
|---|---|
| 0-5 | `ip`, `bt`, `gas`, `pinj`, `tinj`, `ech` |
| 6-53 | `ece_slow_channel_1..48` |
| 54-55 | `pcphd02`, `pcphd03` |
| 56-59 | `co2_density_slow_{r0,v1,v2,v3}` |
| 60-123 | `bes_slow_channel_1..64` |

Proved rather than assumed: slots 56-59 carry baked normalisation means of
8.6e13 to 1.3e14, which is CO2 line-integrated density. Under the alternative
order those slots would hold BES volts near -1.

**There is a second order and it is a trap.** `new_diagnostic_order` puts all 76
non-ECE columns first and `ece_slow_channel_1..48` last; it is materialised in
`wpqh_elm_hiro/data/reordered_model10.pkl` and it is what the PCS wants. Feeding
it to these weights produces silent garbage.

Upstream also applies `t + 1` at fit time, so a horizon `h` must be queried as
`S(h + 1)`.

## Uses

Intended use is offline label generation over the FAITH shot corpus, for
comparison against IGNITE and against other models' labels. Not for real-time
control and not for physics conclusions without the reliability numbers in the
Evaluation section.

## Bias, risks and limitations

Unmeasured. This model has not been run through labelmaker's validation, so
nothing is known here about how its labels behave on corpus shots.

## Training details

Upstream, outside this repository:
`/projects/EKOLEMEN/wpqh_elm_hiro/hiro_scripts/`. Read on 2026-09-05 and
recorded in full in
`docs/superpowers/specs/2026-09-05-labelmaker-phase3-design.md` section 2.2.
In brief: rows are restricted to wide-pedestal QH phases
(`WPQHphases-tau_min500-tau_inter250-pewid_max4.0-pewid_min3.0-tinj_wpqh2.1`) on
a 1 ms grid; `t` is ms to the next ELM as a sawtooth and `e = 1` at an ELM;
filters are NaN to 0, CO2 outside `[0, 1e15]` dropped, a 100 ms boxcar on `pinj`
and `tinj` because NBI is modulated, and `|z| > 10` rows dropped on every column
except the two D-alpha photodiodes; the split is by shot with `seed 0`, 80/20.
`model.cfg` records `k=3, iters=1000, LogNormal, lr=1e-3, batch=1024,
layers=[128], dropout=0.2`. No event/censor resampling is applied anywhere.

Two upstream defects worth knowing: `train_elm_model.py` has train and test
**swapped** (it trains on the last 10% of shots), and the card's former Weibull
claim came from nowhere. `new_train_elm_model.py` with
`data/train_test_split_model10.pkl` is the correct entry point.

## Evaluation

### BES ablation, labelmaker's own fits (2026-09-06)

Two DSMs, same architecture and hyperparameters as `model.cfg` (`k=3,
layers=[128]`, LogNormal, lr 1e-3, batch 1024, dropout 0.2, seed 0), fitted on
**upstream's own split**, read as-is from
`/projects/EKOLEMEN/wpqh_elm_hiro/data/train_test_split_model10.pkl`: 629,023
training rows, 142,745 test rows, event rate 51.3% / 44.5%. `all124` is the
upstream input set; `no_bes` is slots 0-59, everything except
`bes_slow_channel_1..64`. Rows are 1 ms wide-pedestal-QH samples, not the FAITH
corpus and not labelmaker's 500-shot pool: these numbers describe upstream's
population.

Upstream fits on `t + 1` ms, so a horizon `h` is queried as `S(h + 1)`. The
score below is `1 - S(h + 1)` computed through labelmaker's own
`runners/dsm_pickle.survival`. AUROC cases are `e == 1 and t <= h` against
controls `t > h`, with rows censored inside `h` excluded; the IPCW column is
`labelmaker.alarm.ipcw_auc` on the same score.

| set | inputs | test NLL | AUROC 5 ms | 10 ms | 20 ms | 50 ms | IPCW AUC 20 ms |
|---|---|---|---|---|---|---|---|
| `all124` | 124 | **1.2553** | 0.7562 | 0.7624 | **0.7680** | 0.7807 | 0.7680 |
| `no_bes` | 60 | **1.3062** | 0.7381 | 0.7384 | **0.7382** | 0.7446 | 0.7382 |

**Decision rule, written before the run:** adopt `no_bes` if its test NLL is
within 0.02 of `all124`'s *and* its 20 ms AUROC is within 0.02. It fails both:
ΔNLL **+0.0510** and Δ20 ms AUROC **-0.0298**, each above tolerance.
**Verdict: keep BES.** Fetching BES for the corpus is a separate decision, not
an assumption that can be dropped.

Two caveats that belong with the numbers rather than under them:

* both fits' best epoch is **epoch 0**. The test NLL rises monotonically from
  epoch 1 in both sets and `all124` diverged to NaN at epoch 16, so these are
  one-epoch models and the table compares two one-epoch models. The comparison
  between them is fair - identical data, seed, schedule and selection rule - but
  the absolute quality is not what this architecture can do;
* upstream passes the **test** split as `val_data`, which this fit reproduces,
  so the early stop selects on the test rows. `test NLL` is a
  selection-informed number, not held out.

Artifacts, with `training_<set>.json`, `PROVENANCE_<set>.json`, `ablation.json`,
per-set normalisation dicts and loss curves:
`/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/models/d3d_elm_time_to_event_dsm/`.
Produced by `scripts/labelmaker/elm_dsm_train.py` under SLURM job 2924069
(8 CPUs, 32 GB, 2 min 15 s and 1 min 38 s, 68.6% / 64.5% CPU utilisation,
2.8 GB / 2.1 GB memory high-water).

Labelmaker's own validation - adapter fidelity, reconstruction fidelity, label
quality - is still unmeasured: `python -m labelmaker.run validate --models
d3d_elm_time_to_event_dsm` needs the adapter this folder does not have yet.

## Technical specifications

Embedding 143,276 parameters; gate/scale/shape ~3,000 each. Legacy Keras format; `runners/keras_h5.py` may need additional layer support.

## Citation

Unpublished internal model. Attribute to the PlasmaControl group, Princeton.

## Contact

`nc1514@princeton.edu`.
