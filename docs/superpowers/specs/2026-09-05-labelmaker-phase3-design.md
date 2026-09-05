# Labelmaker Phase 3 design

Status: **design only.** Nothing in this document is built. It records what was
read out of the four upstream training loops, the decisions Nathan made on
2026-09-05, and the requirements for the one genuinely new model - an Alfvén
eigenmode activity label.

Predecessor: `2026-09-05-labelmaker-phase2-design.md`. Every number below was
measured on 2026-09-05; the commands that produced them are named inline so they
can be re-run.

---

## 1. What was asked

1. Study how the tearing survival model's training loop handles its imbalanced
   data, and get more reliable results from it. **Decision: do A, B and C of the
   options below.**
2. The survival model's distinguishing property is that it is Bayesian - it can
   say *when* a mode is coming, with a spread, not just a risk. Publish that.
   TabPFN is also Bayesian, so a comparison is worth having. **Decision:
   implement. Use the TabPFN *regressor*, because the spread is easier to read;
   it does not have to match the DSM, it has to be visually comparable.**
3. Study the ELM detector's training loop for the input ordering. **Decision:
   the exact upstream model is not required. Given its inputs and its labels,
   train our own.**
4. Add an AE (Alfvén eigenmode) activity label, documented with everything
   needed to build it. Section 5.

---

## 2. What the upstream training loops actually do

### 2.1 Tearing survival DSM (`d3d_tearing_time_to_event_dsm`)

Sources: `/projects/EKOLEMEN/survival_tm/{train_tm_model.py,train_survival_study.py,format_survival_labels.py,metrics_helpers.py}`,
`/projects/EKOLEMEN/survival_tm_2/train_models/{tm_survival.ipynb,auton-survival/}`.

Hyperparameters of the shipped `rt_fixed_rot.pkl`, decoded from the pickle's own
opcode stream (`pickletools.genops`):

```
iters=20  k=3  layers=[100,1000]  distribution=LogNormal  learning_rate=1e-05
batch_size=1000  discount=1.0  temp=1.0  activation=ReLU6  random_seed=0
```

The pickle also stores its loss curves: 20 entries, validation NLL
0.534 -> 0.471, **monotone and still falling at the last epoch**. auton-survival's
early stop (`utilities.py: train_patience = 5`) never fired. The model is
under-trained by its own configuration, not converged.

**There is no class balancing.** `losses._conditional_lognormal_loss` scores
events with `log f(t)` and censored rows with `alpha * log S(t)`, normalised by
`len(uncens)+len(cens)`; `alpha = model.discount`, which is `1.0` here. The
survival likelihood *is* the imbalance handling. `train_tm_model.py` applies no
resampling and no time cut, and splits 80/10/10 **by shot** from the seed.

Training rows (`survival_tm_2/data/rt_filtered_{e,t}_bms_pcb_rot.pkl`):

| | value |
|---|---|
| rows | 914,898 |
| events (`e == 1`) | 137,864 (**15.07%**) |
| `t` units | ms; min 20, median 1920, max 5740 |
| rows with `t < 600 ms` | 14.66% |

Label construction (`format_survival_labels.py`): for each flat-top row of a
shot, `t = tm_time - time[ind]`, `e = 1` if the shot tears at all, otherwise
`t = end_of_shot - time[ind]`, `e = 0`. Crucially the shot is **truncated at
onset** (`if end_ind > tm_ind and tm_ind != 0: end_ind = tm_ind`), so no
post-onset row is ever a training row. Flat-top bounds come from
`t_ip_flat_sql` / `ip_flat_duration_sql`; a row with any NaN input is skipped.

The two balancing tricks exist only in the Optuna study
(`train_survival_study.py`, and cells 36-39 of `tm_survival.ipynb`), which the
shipped model **did not use**:

1. keep only rows with `t < 600 ms`;
2. `np.random.choice` the `e == 0` rows down to exactly `len(e == 1)` - a 1:1
   event/censor ratio - then shuffle (the arrays arrive sorted by class).

The study's objective is validation NLL (`model.compute_nll`), searched over
activation, `k` (2-10), lr (1e-7 to 1e-1 log), 1-5 layers of 25-250 units, and
batch 10,000-40,000.

**Upstream scores per shot, not per row** (`metrics_helpers.py`):
`get_classification` collapses a shot to one TP/FP/TN/FN from the *last* label in
the trace; `warning_time` is the final 0 -> 1 transition; `jumps` counts crossings
that do not persist 400 ms; FPR is over quiet shots and FNR over tearing shots.
`fnr_fpr_calculator` sweeps the prediction horizon and integrates FPR/FNR over
it. Its default threshold is 0.7. The notebook also uses
`sksurv.metrics.cumulative_dynamic_auc` (IPCW time-dependent AUC).

**This explains the calibration split measured in Phase 2.** Over every aligned
shot's pre-onset rows (5.5% positive at 1 s) the published risk is calibrated,
ECE 0.022. Restricted to the 86 shots that do get an onset (54% positive) it is
badly under-confident, ECE 0.448. That is a base-rate shift, not miscalibration:
the model was fit to a population that is 85% censored with a median 1.92 s to
event, and it is right about that population.

### 2.2 ELM survival DSM (`d3d_elm_time_to_event_dsm`)

Sources: `/projects/EKOLEMEN/wpqh_elm_hiro/hiro_scripts/{data_processing.ipynb,train_elm_model.py,new_train_elm_model.py,model.cfg}`.

Graph, read from the weight files:

```
wpqh1_embedding_with_norm.keras :
    Input(124) -> Dense(124, kernel = I, bias = -mean)
               -> Dense(124, kernel = diag(1/std))
               -> Dense(100) -> ReLU -> Dense(1000) -> ReLU
wpqh1_gate.h5   : (1000, 3), no bias
wpqh1_scaleg.h5 : (1000, 3) + bias (3,)
wpqh1_shapeg.h5 : (1000, 3) + bias (3,)
```

`k = 3`, and the graph is **structurally identical to the tearing DSM** (which is
38 -> 100 -> 1000 with the same three heads). `runners/dsm_pickle.survival()`'s
mixture math therefore applies verbatim; only a Keras weight loader is absent.

**The normalisation is baked in** - this settles the card's third `blocked_on`
item. Constants read back from the two Dense layers: `ip` mean -0.885 (MA, and
signed), `bt` -1.782 T, `gas` 0.106, `pinj` 3.977e6 W, `tinj` 0.586,
`ech` 4.331e5 W, ECE 0.69-1.73 keV, `pcphd02`/`pcphd03` 0.046/0.217 V,
CO2 8.6e13-1.3e14 cm^-2, BES -1.2 to -3.1 V.

**The input order is `current_diagnostic_order`** (cell 34 of
`data_processing.ipynb`), 124 columns:

| slots | contents |
|---|---|
| 0-5 | `ip`, `bt`, `gas`, `pinj`, `tinj`, `ech` |
| 6-53 | `ece_slow_channel_1..48` |
| 54-55 | `pcphd02`, `pcphd03` |
| 56-59 | `co2_density_slow_{r0,v1,v2,v3}` |
| 60-123 | `bes_slow_channel_1..64` |

Proof, not inference: slots 56-59 carry baked means of 8.6e13 to 1.3e14, which
is CO2 line-integrated density. Under the alternative order those slots would
hold BES volts near -1.

There **is** a second order, `new_diagnostic_order` - all 76 non-ECE columns
first, `ece_slow_channel_1..48` last - materialised in
`wpqh_elm_hiro/data/reordered_model10.pkl`. That one is for the PCS. Feeding it
to these weights silently produces garbage.

Rest of the pipeline:

- **1 ms grid** (`times = np.arange(6000)`), not labelmaker's 25 ms.
- Rows restricted to wide-pedestal QH phases:
  `WPQHphases-tau_min500-tau_inter250-pewid_max4.0-pewid_min3.0-tinj_wpqh2.1`.
- `t` = ms to the next ELM (a sawtooth; `metrics_helpers.find_peaks_in_data(t)`
  recovers ELM boundaries), `e = 1` at an ELM.
- **`t + 1` is applied at fit time** in both training scripts, so a horizon `h`
  must be queried as `S(h + 1)`.
- Row filters: NaN -> 0; drop rows where CO2 (56-59) is `< 0` or `> 1e15`;
  box-smooth `pinj` and `tinj` with a 100-sample (100 ms) boxcar because NBI is
  modulated; drop rows with `|z| > 10` on any column **except** the two D-alpha
  photodiodes, which spike at every ELM by design.
- No event/censor resampling anywhere in the ELM pipeline.
- Split by shot, `np.random.seed(0)`, 80/20 -> `data/train_test_split_model10.pkl`.
- `model.cfg`: `k=3, iters=1000, LogNormal, lr=1e-3, batch=1024, layers=[128],
  dropout=0.2`. The card's current "Weibull" claim is unsupported - the weights
  cannot decide a distribution, and the only config in the directory says
  LogNormal.
- `train_elm_model.py` has **train and test swapped** (it trains on the last 10%
  of shots by index and validates on the first 90%). `new_train_elm_model.py`
  plus the pre-split pickle is the correct entry point.
- Dataset variants: `compiled_model3` drops rapid-ELM sequences (gaps < 100 ms);
  `compiled_model7` keeps only pre-first-ELM rows; `compiled_model10` is the one
  in use; `compiled_model11` stacks 10 consecutive timesteps into 1,240 features.

**Data availability is the real blocker, not the adapter.** Sampling 24 corpus
shots for the groups this model needs:

| corpus group | channels | filled |
|---|---|---|
| `ece` | 48 (exact match) | 20/24 |
| `co2` | 4 (exact match) | 12/24 |
| `bes` | 64 (exact match) | **2/24** |

The channel counts line up exactly, but `bes` is a `(64, 1)` placeholder on 22
of 24 shots and BES is 64 of the model's 124 inputs. Serving this model at
corpus scale means an fdp/toksearch fetch for BES, and for CO2 on about half the
shots, then a 1 ms downsample.

---

## 3. Survival reliability: A, B and C

### 3.1 A - score the way upstream scores

Add a per-shot alarm scorer beside the existing per-row metrics, ported from
`metrics_helpers.py` so the numbers are comparable to the upstream paper:

- alarm = risk >= threshold on the final valid finite row; a 0 -> 1 run
  that later reverts to 0 and lasted at least 400 ms is a *jump*. Every
  reverted 0 -> 1 run is an excursion; the final, unreverted run is neither;
- `warning_time` = onset time minus the time of the last 0 -> 1 transition;
- FPR over shots with no archived onset, FNR over shots that have one;
- sweep the horizon and integrate FPR/FNR, as `fnr_fpr_calculator` does;
- report `sksurv`-style IPCW time-dependent AUC alongside the plain AUROC, since
  plain AUROC on censored data is optimistic.

This changes no weights and no labels. It belongs in `validate`, keyed off the
same `ARCHIVE_TRUTH` table `archived_truth` already uses, and the per-shot block
`analyze` writes gains `alarm`, `jumps` and `warning_time_s` fields.

### 3.2 B - recalibrate to the row set being reported

Two mechanisms, both cheap, and the report must say which one produced a number:

1. **Prior shift**, closed form. When moving from the training population
   (prevalence `q1`) to a reported subset (prevalence `p1`), shift the log-odds
   by `log[(p1/p0) * (q0/q1)]`. This is the correct fix for the ECE 0.448 on
   onset-only shots and it introduces no fitted parameters.
2. **Isotonic regression per horizon**, fitted on the aligned-shot pool with the
   shot-level split honoured (fit on one shot set, report on the other), for the
   case where the miscalibration is not a pure base-rate effect.

Both are post-hoc maps on the published risk. The raw risk stays in the label
file; the calibrated series is published beside it, named, with its fitting set
recorded in the label attributes. A calibration that cannot say what it was fit
on is not usable.

### 3.3 C - finish training the model

Continue from the shipped weights rather than starting over - the loss curve
says the fit was cut short, so this is the cheapest real improvement available:

- same data (`rt_filtered_*_bms_pcb_rot`), same 80/10/10 **by-shot** split, same
  seed, so the comparison is honest;
- `lr = 1e-4` (upstream's 1e-5 is what made 20 epochs insufficient), `iters` high
  enough that `train_patience = 5` decides when to stop rather than the config;
- keep `discount = 1.0` and apply no resampling, so the population - and
  therefore the calibration this model is valuable for - is unchanged;
- report the new validation NLL against 0.471 and re-run every Phase 2 figure.

The result is a *new artifact* with its own sha256, its own card row, and its own
`model-index` entry. The shipped upstream pickle stays the default until the
retrained one is measured better on the same shots.

Explicitly **not** doing: the study's `t < 600 ms` cut and 1:1 resampling. They
raise the risk scale so a 0.2 threshold becomes meaningful, but they destroy the
population calibration, which is the one thing this model has that the CNN does
not. If a shot-level alarm is what is wanted, A gives it without touching the
weights.

SLURM: the upstream launcher asked for 32 CPUs / 48 GB / no GPU for a
914k x 38 fit. That is over-provisioned. Per the standing instruction, poll
`squeue -u nc1514` and `jobstats <jobid>` during the run and report CPU/GPU
utilisation and memory high-water marks; size the next request from what the
first one actually used.

---

## 4. Uncertainty: publish the spread

### 4.1 What the DSM already computes and throws away

At every row the model produces a **k=3 mixture of log-normals over
time-to-onset**: `pi = softmax(gate(h)/temp)`, `mu_g = tanh(shapeg(h)) + shape_g`,
`sigma_g = tanh(scaleg(h)) + scale_g`. `runners/dsm_pickle.survival()` computes
all of it and returns three scalars, `1 - S(h)`. New label series, no retraining:

| label | meaning |
|---|---|
| `tm_time_p10`, `tm_time_p50`, `tm_time_p90` | quantiles of predicted time to onset, by bisection on the monotone `S` |
| `tm_time_iqr_log` | spread: `log(p90) - log(p10)`, the natural width for a log-normal mixture |
| `tm_mix_w{0,1,2}`, `tm_mix_mu{0,1,2}`, `tm_mix_sigma{0,1,2}` | the mixture itself |
| `tm_gate_entropy` | how undecided the gate is between components |

The mixture series are what make multimodality visible - one component saying
200 ms while another says "never" is exactly the about-to-tear signature, and it
is invisible in `1 - S(1s)`.

Be precise in the card about what this spread is: it is the **predictive spread
of the event time** (aleatoric plus whatever the network learned), *not*
epistemic uncertainty over weights. Epistemic spread needs an ensemble, and
upstream has one ready: `survival_tm_2/models/{rt_models.pkl,
rt_models_final_1.pkl, rt_models_final_2.pkl, rt_models_final_5.pkl}` are lists
of `[model, train_loss, val_loss, param]` from the parameter grid. Loading a few
and reporting the inter-member spread costs one adapter change.

`analyze` gains a panel form for this: the median as a line, `p10..p90` as a
band, on the same time axis as the risk.

### 4.2 TabPFN regressor, for comparison

Not required to beat or match the DSM. The point is that it is a second Bayesian
model whose predictive distribution can be drawn next to the DSM's on the same
shot.

- **Target**: time to onset in ms, on the CNN's own archived input rows (which is
  where a truth exists), restricted to **pre-onset rows of shots that do tear**.
  `TabPFNRegressor` cannot represent censoring, so censored rows are excluded and
  the card must say so - this is a *conditional* prediction ("given that this
  shot tears, when"), not a hazard.
- **Output**: the regressor's full predictive distribution, summarised as
  `p10/p50/p90` so it plots identically to the DSM's band.
- **Environment**: the existing uv venv at `<scratchpad>/tabpfn-venv`, TabPFN
  8.5.0, torch 2.14.0+cu126 (cu130 has no sm_70 kernels for stellar-vis2's
  V100S). Use `create_default_for_version("v2", ...)`; v2.5/v3 weights are
  license-gated behind a `TABPFN_TOKEN` from ux.priorlabs.ai.
- **Honest framing in the docs**: the DSM answers a hazard question over all
  rows; the TabPFN regressor answers a conditional timing question over a biased
  subset. They are drawn together to be *looked at*, and the figure caption must
  say that.

If a like-for-like comparison is wanted later, the correct construction is a
discrete-time hazard: one binary TabPFN per horizon over rows still at risk,
which handles censoring properly. Recorded here, not planned.

---

## 5. New model: `d3d_ae_activity_seldnet`

An Alfvén eigenmode **activity** label: is an AE present at this instant, and at
what frequency. Slug follows the roster grammar
`<device>_<phenomenon>_<predicted>_<arch>`.

This is the one model on the roster that labelmaker **trains itself**, because
no upstream artifact answers this question - the upstream artifacts either
classify five mode types in time (`aemodes`, noisy labels) or segment a
spectrogram into pixels (`tokeye`, wrong I/O contract for a label series).

### 5.1 Two outputs

| output | task | units | definition |
|---|---|---|---|
| `ae_active` | binary | - | probability that an AE is present in this 25 ms window |
| `ae_frequency` | regression | kHz | intensity-weighted mean frequency of the AE activity in that window; undefined (and masked invalid) where `ae_active` is below its threshold |

**All AE mode types collapse to one class.** The upstream label has five, in this
order (`aemodes/.archive/early_fusion.py:50`):

```
labels = ['lfm', 'bae', 'eae', 'rsae', 'tae']
              0      1      2      3      4
```

so `ae_active = label_1 | label_2 | label_3 | label_4` - **BAE, EAE, RSAE, TAE -
and `label_0` (LFM) is excluded**, being a low-frequency mode rather than an
Alfvén eigenmode. Note that `aemodes/src/aemodes/pipeline/step_1_make_semantic.py`
collapses with `y.sum(axis=0) > 0` over *all five*; the new label must drop
channel 0 first. LFM may be published later as its own separate label; it is not
folded into this one.

### 5.2 Data

| item | value |
|---|---|
| diagnostic | `co2_density`, 4 channels `r0`, `v1`, `v2`, `v3` (CO2 interferometer) |
| training stores | `/scratch/gpfs/EKOLEMEN/d3d_fusion_data/<shot>.h5` and `/scratch/gpfs/EKOLEMEN/hackathon/raw_h5_files/<shot>.h5`, pandas HDFStore, group `co2_density` |
| labels | `/scratch/gpfs/nc1514/aemodes/data/co2_250_detector.pkl` (original) and `co2_250_detector_2.pkl` (soft-relabelled, 2.9 GB each) |
| pickle layout | `[train_shots, X_train, y_train, valid_shots, X_valid, y_valid]`; `X[i]` is keyed `r0/v1/v2/v3`, each `(3905, 128)`; `y[i]` is `(3905, 5)` |
| shots | **180 total, 170659-178879** (120 train, 60 valid) |
| native rate | 1.667 MHz in the store; `resample_poly(up=3, down=10)` -> **500 kHz**, Nyquist **250 kHz** (the "250" in the file name) |
| cached time series | `aemodes/data/.cache/ae_timeseries/<shot>_{train,valid}.arrow`, 1,000,001 rows over 0-2000 ms, columns `r0,v1,v2,v3,label_0..label_4` |

**The 180 training shots have zero overlap with the FAITH corpus** (checked
shot by shot against `/scratch/gpfs/EKOLEMEN/foundation_model/<shot>_processed.h5`:
0 hits). The corpus is 2019+; these are 2016 shots. So the model trains off the
staged/hackathon stores and is *applied* to the corpus - which means the corpus
CO2 must be checked for rate and availability. It was:

| corpus `co2` | value |
|---|---|
| shape | `(4, 4.5-5.5e6)` |
| time span | -1.4 s to 7.6-9.6 s |
| sample rate | **~500 kHz** (median dt 2.0 µs), Nyquist 250 kHz |
| availability | filled on 12 of 24 randomly sampled corpus shots |

The corpus rate matches the trained-on rate exactly - **no resampling is needed
at inference**, only the 4-channel presence check. About half of corpus shots
have no CO2 and get no AE label; that is a validity-mask outcome, not an error.

### 5.3 Spectrogram - regenerated, ours

The pickle's spectrograms were made under settings that are not recorded with
them, so they are not reused. The regeneration recipe is
`aemodes/src/aemodes/pipeline/step_0a_make_spectrogram.py`, and its output is on
disk for all 180 shots (`aemodes/data/.cache/step_0a/spectrograms/<shot>_<mode>.tif`):

| setting | value |
|---|---|
| window | Hann, `nperseg = 1024` |
| hop | 128 |
| transform | `scipy.signal.ShortTimeFFT(...).spectrogram(...)[1:]` - DC bin dropped - then `log1p` |
| result | `(4, 512, 7820)` float32 per shot |
| frequency resolution | 250 kHz / 512 = **0.4883 kHz per bin** |
| time resolution | hop 128 at 500 kHz = **0.256 ms per frame** |
| global standardisation | mean **54.637**, std **2.732** (`step_0a/stats.json`, over 2,882,764,800 values) |

That is 2x finer in time and 4x finer in frequency than the pickle's
`(3905, 128)`. Band edges in bin numbers: **80 kHz = bin 164**, 250 kHz = bin 512.

Note `tokeye.transforms.compute_stft` uses the same `n_fft = 1024` / `hop = 128`
recipe and additionally (a) forms a **cross-spectrum** `sxx[0] * conj(sxx[1])`
when handed two rows, and (b) clips to the 1st/99th percentiles. Whichever of
the two is used for the AE model, the same transform must be used for training
and for corpus inference, and the card must name it.

### 5.4 Band and instrument-line rejection

Two hard physics/instrument constraints, both from Nathan:

1. **AE modes live roughly 80-300 kHz.** Anything below 80 kHz is not an AE, so
   bins 0-163 are excluded from both the activity decision and the frequency
   centroid. At 250 kHz Nyquist the reachable band is **80-250 kHz**; the
   300 kHz upper end of the physical range is *not observable* in this
   diagnostic at this rate, and the card must say so - a mode above 250 kHz is
   invisible, not absent.
2. **Receiver pickup shows as a horizontal line** - a single frequency bin lit
   across the whole shot - and is removed with a notch. Detection recipe: for
   each frequency bin, the fraction of time frames in which it is active (or its
   power relative to the local median across neighbouring bins); a bin active
   across a large majority of the record is an instrument line, not a mode.
   Notch it (zero the row, or subtract the per-bin median) **before** the
   activity decision and before the centroid, and record which bins were notched
   per shot in the label attributes so the removal is auditable.

Nathan removes these by hand today. The automatic rule must be written down with
its threshold, and its output has to be reviewable per shot, because a wrongly
notched bin silently deletes real modes.

### 5.5 Labels - the reason to build this at all

Three label sources, in increasing quality:

1. **The pickle's own `y`** - Bill Heidbrink's time annotations, organised by
   Azaraksh Jalalvand. Time-only, five classes, and **noisy**: this is the "labels
   are very bad" that motivates everything below.
2. **`co2_250_detector_2.pkl`** - the same labels replaced by the smoothed
   predictions of five per-class SELDNets trained on them
   (`aemodes/scripts/improve_dataset.py`). Soft values in [0,1] rather than
   binary; Nathan's own viewer thresholds them at 0.4.
3. **Coherent-mode-filtered labels** - the intended source here.

Source 3 is the point. `tokeye`'s `big_tf_unet` is a strong coherent-mode filter:
a 1 -> 2 channel U-Net (`out_channels = 2`, 5 layers, first layer 32, dropout 0.2)
whose **channel 0 is coherent activity and channel 1 is transient activity**
(`tokeye/README.md:205-206`). Its released weights are
`tokeye/model/big_tf_unet_251210.pt`, registry name `big_tf_unet`, HF repo
`nc1/big_tf_unet`. The predecessor of the same idea inside `aemodes` is
`model/big_mode_v1-5.pt`, and `step_1_make_semantic.py` already demonstrates the
construction:

```python
out = torch.sigmoid(model(inp)) > 0.2      # coherent-mode mask
ae_true = dataset[idx]['y'].numpy().sum(axis=0) > 0   # time-only AE label
overlap = out[i][0] * ae_true               # channel 0 = coherent modes
```

The mask says *where in time-frequency* coherent activity is; the annotation says
*when* an AE was present. Their product is a time-frequency AE mask that is far
better localised than either - and it is generated automatically, which is what
makes a better label affordable. The new label differs from that snippet in one
respect: `ae_true` must exclude `label_0` (LFM), per 5.1.

**From the mask to the two published series**, per 25 ms labelmaker window:

- restrict to bins 164-511 (80-250 kHz) after notching;
- `ae_active` = a coherent-AE mask occupancy statistic over the window's frames,
  above a threshold that is fitted, recorded, and reported - not assumed;
- `ae_frequency` = the spectrogram-intensity-weighted centroid of the masked
  pixels, in kHz; masked invalid where `ae_active` is below threshold, because a
  centroid of an empty mask is not a number.

Nathan's framing to preserve in the card: *if* tokeye pulls the eigenmode out
correctly, then the mask is more informative about AEs than the hand annotation,
and it can be produced automatically. He added (2026-09-05) that **the hand
annotations under-count the real AE activity**, so activity the mask finds outside
an annotated window is expected, and a label with more positives than the
annotation is acceptable. That changes the construction: the primary label is the
band-restricted, notched mask itself; the annotation is kept for evaluation (recall
of annotated frames is the test the mask must pass; its precision against the
annotation is reported but is not a failure). The conditional is load-bearing - the mask's
own quality on these 180 shots has not been measured, and measuring it is the
first task, not an assumption.

### 5.6 Model and training loop

The existing time-classification architectures in `aemodes` are the starting
point; both already emit one prediction per time frame, which is what a label
series needs.

| | `SELDNetModel` | `BaselineModel` |
|---|---|---|
| file | `aemodes/src/aemodes/models/detection/seldnet.py` | `.../baseline.py` |
| input | `(4, 355, 128)` = 4 channels x T frames x F bins | same |
| output | `(355, 5)` per-frame logits | same |
| body | 3x [LazyConv2d 3x3 -> BN -> ReLU -> MaxPool(1,pool) -> Dropout] with `pool_sizes=[9,8,2]`, then 2 bidirectional GRUs (multiplicative halves), then an FNN | 3 conv blocks then a per-frame linear head |

Changes for this model:

- **output head**: `(T, 2)` - one activity logit and one frequency value - instead
  of `(T, 5)`. Two losses: BCE-family on the activity, and a masked regression
  (Huber, in kHz, or on a normalised band coordinate) on the frequency, evaluated
  only on frames the *label* calls active. Weight the two terms and record the
  weight.
- **input width**: the frequency axis is now 512 bins, not 128, and the band
  restriction removes the bottom 164. Either feed bins 164-511 (348 bins) or feed
  all 512 and let the loss ignore the rest; the former is cheaper and states the
  physics in the architecture.
- **noisy-label loss**: keep upstream's `BinarySCELoss` for the activity term -
  symmetric cross entropy, `alpha=1.0` BCE plus `beta=0.5` reverse CE
  (`improve_dataset.py`), which exists precisely because these labels are bad.
  Compare against plain BCE and report both; if the tokeye-filtered labels are as
  much better as expected, SCE's advantage should shrink, and that shrinkage is
  itself the evidence that the new labels are better.
- **windowing**: `ShotDataset` hardcodes `lenshot = 3905`, `nwin = 11`,
  `lenwin = 355`. With 7820 frames the window length changes; read the length
  from the data instead of hardcoding it.

Upstream training settings to inherit, all from `improve_dataset.py`:

```
epochs 30   batch 16   AdamW lr 1e-4 weight_decay 1e-4
CosineAnnealingLR T_max=30 eta_min=1e-6   precision bf16-mixed
EarlyStopping(val_loss, patience=5)   ModelCheckpoint(save_top_k=1, min val_loss)
inference: sigmoid -> moving average, kernel 5 frames
```

Split **by shot** (120/60 already fixed in the pickle's own train/valid lists) -
never by frame, or adjacent frames of one shot land on both sides.

SLURM, from `improve_dataset.sh`: 1 GPU, 2 CPUs, 8 GB, 10 h. That was for five
small SELDNets; with 512-bin inputs the memory and dataloader needs go up. Per
the standing instruction, poll `squeue -u nc1514` and `jobstats <jobid>` during
training and report CPU/GPU utilisation and memory, and raise
`--cpus-per-task` if the dataloader is the bottleneck.

### 5.7 What labelmaker needs on its side

- A `co2` feature in `features/namespace.py` with 4 channels - a **raw waveform**
  feature, unlike every existing scalar/profile feature, so the feature layer
  needs a shape it has not carried before. This is the largest piece of
  integration work.
- A resolver from the corpus `co2` group; no `fdp` path is required for the
  training shots, but corpus shots without `co2` must produce invalid rows rather
  than an error.
- A `DomainRule` for the band and for the notch outcome, so rows whose
  spectrogram was heavily notched are flagged rather than silently trusted.
- 25 ms windowing of a 0.256 ms frame grid: 97.6 frames per window. Fix the
  aggregation (mean over frames, or max) and write it in the card; it decides
  what `ae_active` means.

### 5.8 Blocked on

1. Measure `big_tf_unet`'s coherent mask against the 180 annotated shots before
   trusting it as a label source. If the mask misses modes the annotation finds,
   the product label is worse than either input.
2. Fix and record the notch rule's threshold, with a per-shot review of which
   bins it removes.
3. Fix the `ae_active` occupancy threshold and the 25 ms aggregation.
4. Decide the frequency target's treatment when two AEs are present at different
   frequencies in one window - a single mean frequency is a lossy summary and the
   card must say what it reports (intensity-weighted mean over all masked pixels
   is the current plan; the alternative is the strongest mode's frequency).
5. Confirm whether the AE model's spectrogram uses `aemodes`' plain `log1p` STFT
   or `tokeye`'s cross-spectrum + percentile-clipped variant. Training and corpus
   inference must agree.
6. Corpus CO2 coverage at scale - the 12/24 sample needs to become a real count
   over 16,909 shots.

### 5.9 Roster bookkeeping

The roster's "Deliberately excluded" list rules out TokEye and `ae_tf_maskrcnn`
as labelmaker models, on the grounds that image-in/mask-out is the wrong I/O
contract. That stays true and this model does not contradict it: `big_tf_unet` is
used **offline, as a label-construction tool**, and never publishes a label.
What labelmaker publishes is two per-timestep series from a model it trained
itself. The exclusion note should be amended to say exactly that, so the two
facts do not later read as a contradiction.

---

## 6. ELM: our own model

Decision: do not reproduce `wpqh1_*`. Use its inputs and its labels and train a
model labelmaker owns, the same way the AE model is owned.

What that buys:

- freedom from the 124-column order trap of 2.2, and from the `t + 1` offset;
- freedom to drop BES if the corpus cannot serve it - a model trained on
  `ip, bt, gas, pinj, tinj, ech, ece_slow_1..48, pcphd02, pcphd03, co2_r0..v3`
  (60 columns) needs only groups the corpus fills on 12-20 of 24 sampled shots,
  where the full 124 needs BES on 2 of 24;
- a shot-level split we choose, avoiding the swapped-split bug.

What it costs: the upstream evaluation no longer transfers, so the model needs
its own reliability numbers from scratch. Retain from upstream: the WPQH phase
restriction, the label construction (`t` = ms to next ELM, `e` = 1 at an ELM),
the 100 ms boxcar on `pinj`/`tinj`, the CO2 range filter, the `|z| > 10` outlier
drop that spares the D-alpha photodiodes, and the by-shot split. Drop: the `+1`
offset (use a distribution that tolerates `t = 0`, or add the offset explicitly
and document it).

Ablate BES first - if the 60-column model is close to the 124-column one on the
same shots, the fdp BES fetch is not worth building.

---

## 7. Open questions for Nathan

1. AE: is the intensity-weighted mean frequency over all masked pixels the right
   `ae_frequency`, or should it be the strongest mode's frequency when several
   are present at once?
2. AE: publish LFM as its own separate label later, or discard it?
3. AE: which spectrogram - `aemodes`' `log1p` STFT (settings measured above) or
   `tokeye`'s cross-spectrum with percentile clipping?
4. Notch: is "a bin active across most of the record" the rule you use by eye, or
   is there a specific frequency list per campaign?
5. Survival: ship the retrained (C) model as the default once it measures better,
   or keep the upstream pickle as default and publish both?
6. Still open from Phase 2: the `tm_risk_1s` default threshold (0.2 shipped, 0.10
   measured better), an ECH deposition-location source, `TABPFN_TOKEN`, the
   rtCAKENN production checkpoint owner, and whether the 2024 ZIPFIT rotation of
   1,200-2,000 is a units change or a fit blow-up.

## 8. Order of work

1. Survival A - the per-shot alarm scorer. No weights, immediate payoff.
2. Uncertainty 4.1 - quantiles, mixture and gate entropy from
   `dsm_pickle.survival()`, plus the `analyze` band panel.
3. Survival B - prior shift and per-horizon isotonic, with the fitting set
   recorded.
4. Survival C - continue training on SLURM; re-run the Phase 2 figures.
5. Uncertainty 4.2 - TabPFN regressor, for the side-by-side figure.
6. AE 5.8.1 - measure the coherent mask against the 180 annotated shots. This is
   the go/no-go for the whole AE label.
7. AE - notch rule, thresholds, then train.
8. ELM - the 60-column model and the BES ablation.
