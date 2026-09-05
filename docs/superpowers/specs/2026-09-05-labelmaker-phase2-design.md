# Labelmaker Phase 2 design

**Date:** 2026-09-05
**Status:** proposed, with the items marked *(built)* implemented in the same session; Nathan
reviews before anything else is started
**Location:** `src/labelmaker/` in FusionAIHub, branch `labelmaker`
**Author:** Claude, for Nathaniel Chen. Written in a non-interactive session, so every
decision below is an assumption stated for review rather than an agreed answer.
**Builds on:** `2026-09-03-labelmaker-design.md` (Phase 1, built).

## 1. What was asked

After the 100-shot proof of concept, four requests:

1. Improve the tearing model, and try TabPFN to see whether it does better.
2. Run labelmaker on more shots, about 500, including the tearing-mode shots 199597-199607.
3. Add the other roster models.
4. End goal: type in a shot, name the labels wanted in a config file, and get a file with the
   labels plus one plot with a subplot per label.

Standing instructions still in force: ECH is on hold; commits go to `labelmaker`, no push,
no co-author trailer.

## 2. Measurements that shape the design

Measured today, details in `.superpowers/sdd/progress.md`:

- **500 shots ran end to end in six minutes of wall time** (400 new shots, 8 fdp workers).
  Validation used 486: the 11 famous shots have no archived truth (they are 2024 shots; the
  training archive ends at 190997) and 3 shots were rejected by the row matcher. Headline on
  31,257 valid rows: `tm_prob` AUROC 0.9315 archived vs 0.8972 reconstructed (penalty
  -0.034, the same as on 100 shots); `betan` RMSE 0.120 vs 0.157.
- **The famous shots come out 85-95 % invalid.** On 199597, ECH power is on for 221 of 238
  plasma rows and the deposition location `ech_rho` exists only in the archive, so the
  `unknown_when_active` rule flags every ECH-on row. The model itself fires: `tm_prob`
  peaks above 0.9 on 9 of the 11 shots. This is the ECH hold landing on exactly the shots
  Nathan wants to look at (Section 8).
- **TabPFN's current weights are license-gated.** Package 8.5.0 installs and runs on the
  stellar-vis2 V100S, but the default v3 and the v2.5 checkpoints require an account at
  ux.priorlabs.ai and a `TABPFN_TOKEN`. The v2 checkpoint downloads from Hugging Face
  without one and is what the study uses until Nathan sets a token.
- **The six scaffold models split three ways.** Two have their inputs fully recovered
  (tearing survival, ECH beam fate); one has its input names but a different data pipeline
  (ELM survival, 124 downsampled raw diagnostics from the hackathon "slow" files); one needs
  an author to name the production checkpoint (rtCAKENN, three candidate directories); one
  has no input source at all (INPA); one is ECH with unrecovered inputs (torbeamNN).

## 3. Decisions

| Question | Decision | Why |
|---|---|---|
| Where `analyze` writes | `<root>/analysis/<shot>/<shot>_analysis.json` and `<shot>_labels.png`; the labels themselves stay in `labels/<shot>_labels.h5` | one canonical label file; the analysis directory is a view, cheap to regenerate |
| Config format | YAML, keys `labels`, `context`, `threshold`; a default file ships in the package | the cards are already YAML; a physicist edits a list, not code |
| Validity explanation | `BuiltInputs.invalid_reasons`: per-rule counts of rows each rule alone rejects | the famous-shot finding took a script to diagnose; the runner should say it |
| Metrics | add `f1_max` and `threshold_at_f1_max` to `binary_metrics` | the -0.14 best-F1 penalty found on 2026-09-05 is invisible in the reports |
| Row matcher | when no column is archive-served, match on geometry and tie-break on bt/ip | recovers the 3 rejected fully-fdp shots; same logic that fixed the 100-shot pool |
| Next adapters, in order | tearing survival DSM, then ELM survival DSM; ECH pair gated on the hold; rtCAKENN after an author confirms the checkpoint; INPA moved to excluded | inputs at hand first; blocked ones documented, not guessed |
| Reading pickled models | a restricted `pickle.Unpickler` with stub classes for `auton_survival`, `sklearn`, `sksurv`; torch tensors real; weights verified by sha256 of the upstream pickle | upstream bytes only, like the Keras-2 H5 path; no fork of auton-survival in the env |
| TabPFN | Phase 3 pulled forward: study A on the CNN's archived inputs (done with v2), study B on labelmaker-reconstructed features; results into the card | Nathan asked; the reconstructed-feature study is the one that can change what labelmaker publishes |
| "Improve the model" | no retraining inside labelmaker; the study reports what a substitute would gain, and the card gets a tuned threshold | labelmaker publishes trained models' outputs; a new model is a new folder with a card |

## 4. `analyze` *(built)*

### 4.1 Command

```bash
pixi run -e labelmaker fdp run python -m labelmaker.run analyze \
    --shots 199597 199606 --config my_labels.yaml [--out DIR] [--force]
```

`analyze` is a fifth stage. `--models` is not accepted with it: the models come from the
labels named in the config. `--config` defaults to the packaged
`src/labelmaker/analyze_default.yaml`. `--out` defaults to `<root>/analysis/`.

### 4.2 Config

```yaml
labels:                                  # plotted top to bottom, in this order
  - d3d_tearing_onset_cnn1d/tm_prob
  - d3d_tearing_onset_cnn1d/betan
context:                                 # corpus signals drawn above the labels
  - ip
  - pinj
threshold: 0.5                           # drawn on binary labels only
```

A label is `<slug>/<name>`; an unknown slug or name is a config error (exit 2) before any
work starts. `context` names are canonical features or corpus groups; a missing one is drawn
as an empty panel with the reason in its title, never an error.

### 4.3 Flow

For each shot, in one process: `features_for_shot` for the union of the named models'
inputs, `infer_for_shot` per model (both skip work that is complete unless `--force`), then
read the label file back and write the two outputs. Shots are isolated exactly as in the
other stages. The run gets a `runs/analyze-<id>/` manifest like every other run.

### 4.4 Outputs

`<shot>_analysis.json`:

```json
{"shot": 199597, "config": {...}, "written": "2026-09-05T07:00:00+00:00",
 "labels": {"d3d_tearing_onset_cnn1d/tm_prob": {
     "card_id": "...", "artifact_sha256": "...", "task": "binary", "time_step_ms": 25.0,
     "n_rows": 240, "n_valid": 11, "valid_fraction": 0.046,
     "invalid_reasons": {"ech_rho unknown while ech_power_total active": 221, "kappa value": 8, ...},
     "max": 0.910, "t_at_max": 3.25, "first_above_threshold": 3.10, "threshold": 0.5,
     "resolvers": {"bt": "fdp", ...}, "missing_inputs": ["ech_rho"]}},
 "labels_file": ".../labels/199597_labels.h5"}
```

`<shot>_labels.png`: one column of panels sharing the time axis. Context panels first, then
one panel per label in config order. A label panel draws the series in blue, the ensemble
spread as a light band, invalid rows as a grey wash, and the threshold as a dashed line for
binary labels. Title: shot and the list of card ids. Colours follow the dataviz reference
palette; a series never needs colour alone to be read (the legend names it).

### 4.5 Errors

Shot not in the corpus and not in the archive: the analysis JSON is still written, with
every label reporting `n_rows: 0` and the features-stage misses; the plot has the context
panels only. A model that fails to load is exit 4 before any shot runs, as in `infer`.

### 4.6 Tests

Config parsing (good, unknown label, missing file); the per-label summary on a synthetic
label file (max, first crossing, valid fraction, invalid reasons); the figure is produced
(file exists, non-trivial size) with a non-interactive backend; `analyze` end to end on the
synthetic archive and corpus fixtures from `test_run.py`.

## 5. Small changes to Phase 1 *(built)*

- `InputSpec._validity` returns the per-rule rejection counts alongside the mask;
  `BuiltInputs.invalid_reasons` carries them; `infer_for_shot`'s log row includes them.
- `validate.binary_metrics` gains `f1_max`, `threshold_at_f1_max`, computed from the
  precision-recall curve; reported in every series and in the reconstruction penalty.
- `validate._match_columns`: no archive-served column at all -> match on the geometry
  columns (6, 7, 8) with bt/ip (0, 1) as tie-breakers. Same `_TIE_EPS` logic.
- Docs and the card's prose move from the 100-shot to the 500-shot pool.

## 6. New adapters

### 6.1 `d3d_tearing_time_to_event_dsm` *(built if time allowed; otherwise this is the plan)*

Upstream: `/projects/EKOLEMEN/survival_tm_2/models/rt_fixed_rot.pkl` (Hf8585, 2024-11), the
real-time-signal variant; constants `survival_tm/data/rt_normalizations_dict.pkl`; inference
reference `survival_tm/get_survival_from_shot.py::get_rt_survival_from_shot`.

Graph, read from the pickle: `Linear(38,100, no bias) -> ReLU6 -> Linear(100,1000, no
bias) -> ReLU6`, then heads `gate (1000->3)`, `scaleg (1000->3, bias)`, `shapeg (1000->3,
bias)`, and parameters `shape (3,)`, `scale (3,)`; k = 3, LogNormal, temperature 1.0.
Survival at horizon `t` (ms), following `losses._lognormal_cdf` and `_init_dsm_layers`:

```
mu    = tanh(shapeg(h)) + shape          sigma = tanh(scaleg(h)) + scale
w     = log_softmax(gate(h) / temp)
S(t)  = sum_k exp(w_k) * (0.5 - 0.5 erf((ln t - mu_k) / (exp(sigma_k) sqrt 2)))
```

Inputs, in upstream order, and their canonical features:

| upstream | canonical | source | note |
|---|---|---|---|
| bmspinj, bmstinj | pinj_total, tinj_total | corpus/archive | units to confirm against the normalisation means (5.48, 3.62 -> MW and N m) |
| betan_EFITRT2, qmin_EFITRT2, li_EFITRT2, aminor_EFITRT2, volume_EFITRT2 | betan, qmin, li, aminor, volume | new, fdp EFIT01 aeqdsk | offline EFIT01 stands in for EFITRT2, priced like R0/kappa |
| ech_pwr_total | ech_power_total | archive/corpus | |
| ip | ip | archive/fdp | upstream multiplies its stored `ip` by 1.69861e-5 before normalising (mean 8.9e5 after) - measure which units the fetch returns |
| PCBCOIL | pcbcoil | new, fdp PTDATA | |
| rmaxis, tribot, tritop, kappa (EFITRT2) | r0, tribot, tritop, kappa | archive/fdp | |
| thomson_temp_mtanh_1d, thomson_density_mtanh_1d, cer_rot_csaps_1d | te_zipfit, ne_zipfit, rot_zipfit | archive/fdp | rotation normalised as `rotation_kms` |
| cer_temp_csaps_1d | ti_zipfit | new, fdp ZIPFIT01 itemp | |
| qpsi_EFITRT2, pres_EFITRT2 | qpsi (as 1/q), pres | archive/fdp | |

Preprocessing inside the adapter: each 33-point profile linearly interpolated to 100 points
(65 for 1/q and pressure), projected on the stored PCA (4 components each), z-scored; the 14
scalars z-scored; concatenated to 38. Output label `tm_risk_1s` = 1 - S(1000 ms), task
`binary`-like probability on a 20 ms step (upstream's grid), plus `tm_survival_1s` = S. Truth
for label quality: from the archive's `tm_label`, "an onset occurs within the next 1 s".
Adapter fidelity: a golden file made once in the throwaway venv with the auton-survival fork,
compared to labelmaker's evaluator on the same rows. Also compare to upstream's own outputs
on 199597-199607 (`survival_tm/data/199596199610_data_outputs.pkl`) once their layout is
understood.

### 6.2 `d3d_elm_time_to_event_dsm` (next)

Inputs recovered: the 124 keys of `compiled_model10.pkl['normalizations']`, in order: ip,
bt, gas, pinj, tinj, ech, 48 ECE channels, pcphd02, pcphd03, 4 CO2 chords, 64 BES
channels, all `_downsampled` from `/scratch/gpfs/EKOLEMEN/hackathon/<shot>_slow.h5`. What is
still unknown: the downsampling grid, and the mapping of `ece_slow_channel_k` and
`bes_slow_channel_k` onto the corpus' `ece` and `bes` arrays. Graph: Keras embedding
(`wpqh1_embedding_with_norm_flat.keras`, a zip; needs a `.keras` reader beside the H5 one)
plus three H5 heads; LogNormal k = 3. Label: ELM within a fixed horizon, on the downsampled
step. Truth: `data/elm_labels_dict.pkl`.

### 6.3 The rest

- `d3d_ech_beam_fate_mlp` and `d3d_ech_deposition_torbeamnn`: gated on the ECH hold. The
  beam-fate inputs are fully known (Section 2); its training data are TORBEAM simulations, so
  only adapter fidelity can be measured.
- `d3d_kinetic_equilibrium_rtcakenn`: three candidate checkpoints found; the card lists them
  and asks for an author.
- `d3d_inpa_image_cnn`: moved from scaffold to the excluded list - no corpus input.

## 7. TabPFN study (Phase 3, pulled forward)

**A. On the CNN's own inputs.** Test rows: the 7,373 archived rows of the 100 PoC shots.
TabPFN (v2, GPU) fits on 10k and 50k rows sampled from the other 8,405 training shots;
the CNN's numbers on the same rows are its ceiling, because it trained on them. Metrics:
AUROC, AUPRC, F1 at 0.5, best F1 and its threshold, Brier, ECE; `betan` RMSE for the
regressor. Also the CNN with an isotonic recalibration fitted on the other shots, to
separate "better model" from "better calibrated". Results:
`outputs/labelmaker/tabpfn/`, summarised in the card's Evaluation.

**B. On labelmaker's reconstruction.** Train on the reconstructed rows of the 386 non-PoC
validatable shots with the archive's truth; test on the PoC 100's reconstructed rows; compare
to the CNN fed the same reconstructed rows. This answers whether a model trained on what
labelmaker can actually reconstruct closes the -0.14 best-F1 gap.

When Nathan sets `TABPFN_TOKEN`, both studies rerun with v2.5/v3.

## 8. Open questions for Nathan

1. **ECH.** The famous shots are unusable as published because the tearing model needs a
   deposition location. Lifting the hold means: find whether fdp/MDSplus serves `EC.RHO_ECH`
   (or the per-gyrotron `rho` the pipeline computed) and add an fdp source for `ech_rho`.
   Until then, `tm_prob` on 2024 ECH shots is an extrapolation by the card's own definition.
2. **TabPFN token.** Register at ux.priorlabs.ai and export `TABPFN_TOKEN` to run the
   current models; v2 results stand in.
3. **rtCAKENN checkpoint.** Which of `fall2024_rtcakenn/best_model_42.h5`,
   `rscake_nn/best_model_{0..9}.h5`, `rtcakenn_optimization/models/keras2c_converted` is the
   production model, and who owns it.
4. **Rotation on 2024 shots.** ZIPFIT rotation reaches 1.2e3-2.0e3 on 199602/199606 (km/s
   is expected); the `absmax < 150` rule catches it, but is that units or a fit blow-up?

## 9. Order of work

1. Section 5 changes with tests. 2. `analyze` with tests, run on 199597-199607 and a PoC
shot. 3. Study A results and figure. 4. Tearing survival adapter. 5. Docs and card. 6. Study
B. 7. ELM survival adapter. 8. Everything ECH, after the hold is lifted.
