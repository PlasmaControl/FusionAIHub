# Labelmaker design

**Date:** 2026-09-03
**Status:** approved design, pre-implementation
**Location:** `src/labelmaker/` in FusionAIHub (FAITH), branch `nathan_fm`
**Author:** Nathaniel Chen, with Claude

## 1. Context

The PlasmaControl group has many trained ML models that predict plasma phenomena
(tearing-mode onset, ELM timing, ECH beam fate, kinetic equilibria, ...). Each has its own
input pipeline, framework, output convention, and storage location. None of them can be run
over the FusionAIHub shot corpus today, and their labels cannot be compared to each other or
to IGNITE.

Labelmaker standardizes this. Given a shot list, it resolves the physics quantities each model
needs, runs the trained models, and writes per-shot labels in the FusionAIHub HDF5 layout so
IGNITE loaders can read a label like any other signal. It records, per model, how reliable
those labels are against available ground truth.

Two facts found during exploration shape the design:

- **"The FAITH database" is the FusionAIHub corpus itself**: FAITH is the repo's acronym
  (Fusion AI Toolkit & Hub), the installable package is `faith`, and the shot corpus is
  `/scratch/gpfs/EKOLEMEN/foundation_model/<shot>_processed.h5` (16,909 files on 2026-09-03).
- **No candidate model can be fed from the corpus alone.** The corpus holds raw diagnostics
  and actuators. Every usable model consumes per-timeslice physics features: EFIT scalars,
  fitted ne/Te/rotation profiles on a radial grid, actuator totals. Those features exist on
  disk only for older shots (the profile-predictor store ends at shot 180927 and has zero
  overlap with the corpus). They must be reconstructed. That reconstruction, not the model
  wrappers, is the substantive work.

## 2. Goals

1. One folder per model with a machine-readable I/O contract and a HuggingFace-style model
   card, so a physicist can tell what the model is, where its weights came from, what it
   consumes, and how well its labels held up.
2. A shared canonical feature layer so adding a model adds no fetch code.
3. Per-shot label files in the corpus layout (`xdata`/`ydata` groups), with provenance.
4. A validation stage that measures label reliability against ground truth, and separates
   wrapper error from input-reconstruction error.
5. Proof of concept on ~100 shots that scales to the full corpus by a flag, not a rewrite.

## 3. Non-goals and exclusions

- **No TokEye** (`tokeye_unet`, `ae_tf_maskrcnn`). They map spectrograms to pixel masks, a
  different I/O contract, and they already have their own packaged app.
- **No TokaMind** (`tokamind_base_v2`): MAST-pretrained; tokenizer and inverse decode live
  outside the graph.
- **No diag2diag**: excluded by decision.
- **No artifacts from `tokamak_deploy_bench`.** That repo is a latency benchmark that feeds
  random noise into models and stores only timings. It served as a reference list of what
  models exist. Labelmaker loads every model from its upstream source of truth.
- **No LLM or keyword sentiment**; labels come from trained models only.
- **Not all shots in the proof of concept.** ~100 shots, chosen so validation is possible.
- **TabPFN substitution study is Phase 3**, not part of the first implementation plan.
- **IGNITE embedding probes are Phase 4**; no IGNITE codec checkpoint exists on stellar today.

## 4. Decisions (recorded)

| Question | Decision |
|---|---|
| Shot universe | The FusionAIHub corpus `<shot>_processed.h5`. Labels written as a sibling per-shot HDF5 in the same layout. |
| First models | Tearing-onset first; ELM time-to-event and ECH beam fate next. All other roster models scaffolded as empty folders with cards. |
| Input source | Reconstruct features (corpus HDF5 + fdp efit01/PTDATA + own profile fits) and validate against the archived training arrays on the 1,503 overlapping shots. |
| Architecture | Shared feature layer with thin per-model adapters; three stages with per-shot HDF5 between them. |
| Model weights | Upstream artifacts only, copied once with sha256; native framework runners (Keras, torch); no ONNX conversion. |
| Naming | HuggingFace model-card standard; slug grammar `<device>-<phenomenon>-<predicted>-<arch>[-<variant>]`. |
| Task types | Classification or regression (plus survival for the DSM models). |

## 5. Architecture

Three stages, files between them, each independently rerunnable. Dependencies are by file,
like the rest of FusionAIHub.

```
corpus <shot>_processed.h5 ──┐
fdp (efit01, PTDATA) ────────┼─► [features] ─► <shot>_features.h5 ─► [infer] ─► <shot>_labels.h5
own profile fits ────────────┘        │                                    │
                                      └────────── [validate] ◄─────────────┘
                                                      │
                                     archived training arrays, ground-truth label series
```

- **features**: resolve the union of canonical features the requested models need, per shot,
  into `<shot>_features.h5`. Runs in the `labelmaker` pixi environment (which includes the
  `fdp` feature).
- **infer**: for each model, build inputs from the feature file, predict, decode, write
  `<shot>_labels.h5`. Runs in the `labelmaker` pixi environment.
- **validate**: adapter fidelity, reconstruction fidelity, label quality. Writes JSON and
  updates model cards.

## 6. Package layout

About 18 source files. Plain functions and frozen dataclasses. No Pydantic (repo convention
is `@dataclass`). No torch import in labelmaker's own code; framework runners are
import-guarded. Labelmaker sits on the `ignite/gate.py` side of the IGNITE reuse boundary:
it imports nothing from `tokamak_foundation_model` model packages.

```
src/labelmaker/
  __init__.py
  config.py              Paths and RunConfig dataclasses; env-var overrides for roots
  catalog.py             corpus shot enumeration; (C,1) absent-signal check; archive overlap
  timebase.py            span-based sample rate, clamped index, window means, resample-to-grid
  features/
    __init__.py
    namespace.py         FeatureSpec registry (canonical name, kind, units, grid, source)
    store.py             read/write <shot>_features.h5; missing/complete bookkeeping
    resolve_corpus.py    actuator totals, raw Thomson, raw CER from <shot>_processed.h5
    resolve_fdp.py       efit01 aeqdsk/geqdsk and PTDATA ip/bt via toksearch (fdp env only)
    resolve_fits.py      mtanh (ne, Te) and csaps (rotation) onto the 33-point rho grid
    geometry.py          Thomson/CER channel positions; rho mapping via efit01 psirz
  models/
    __init__.py
    README.md            roster, naming scheme, exclusions
    base.py              InputField/InputSpec/OutputField/OutputSpec/ModelAdapter
    registry.py          discovers model folders; parses card front matter; cross-checks spec
    runners/
      __init__.py
      keras.py           import-guarded tf.keras loader and batch predictor
      torch.py           import-guarded torch loader and batch predictor
    d3d_tearing_onset_cnn1d/      README.md  spec.py  __init__.py     (implemented, Phase 1)
    d3d_elm_time_to_event_dsm/    README.md  spec.py  __init__.py     (scaffold)
    d3d_tearing_time_to_event_dsm/ ...                                (scaffold)
    d3d_ech_beam_fate_mlp/ ...                                        (scaffold)
    d3d_ech_deposition_torbeamnn/ ...                                 (scaffold)
    d3d_kinetic_equilibrium_rtcakenn/ ...                             (scaffold)
    d3d_inpa_image_cnn/ ...                                           (scaffold)
  labels/
    __init__.py
    schema.py            LabelSpec: task, classes, units, activation, time base
    store.py             write <shot>_labels.h5 in corpus layout with provenance attrs
  run.py                 python -m labelmaker.run {features|infer|validate|all}
  validate.py            reconstruction-vs-archive and label-vs-truth reports
tests/labelmaker/        synthetic-first, module-mirrored; skipif on the corpus path
```

## 7. Environment and packaging

- FusionAIHub is pixi-managed (Python 3.11). Labelmaker uses that, not a separate uv project.
- `pyproject.toml` changes:
  - `[tool.hatch.build.targets.wheel] packages = ["src/faith", "src/tokamak_foundation_model", "src/labelmaker"]`
    so `import labelmaker` works through the editable install. Today hatchling autodiscovers
    only `src/faith`.
  - New pixi feature `labelmaker` with `tensorflow-cpu` (Keras models load natively),
    `pyarrow` (for `labels_index.parquet`), and `pyyaml` if not already transitively present
    (hydra-core pulls it in via omegaconf; verify). `scipy`, `h5py`, `numpy`, `pandas` are
    already shared dependencies.
  - New environment `labelmaker = ["labelmaker", "fdp", "cuda"]`, so one environment runs
    every stage. The existing `fdp` feature supplies `toksearch` and `toksearch_d3d` from the
    `ga-fdp` channel. The default environment is untouched.
  - Later phases: optional `tabpfn` feature.
- Tests run as bare `pytest tests/labelmaker`, matching the repo (no pytest config exists).
- Ruff `line-length = 88` is the only lint rule; follow it.

**Repo-state caution.** On 2026-09-03 the FusionAIHub working tree is mid-merge (461 changed
files, `scripts/training/train_e2e_stage1.py` unmerged). This spec is committed alone by
path. Implementation starts on a fresh branch off `nathan_fm` after that merge is resolved
or stashed.

## 8. Data root

All artifacts under `/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/` (Nathan's scratch is near
quota; group storage is the rule). Overridable by `LABELMAKER_ROOT`.

```
features/<shot>_features.h5
labels/<shot>_labels.h5
labels_index.parquet
models/<slug>/{artifacts..., PROVENANCE.json}
runs/<run_id>/{manifest.json, log.txt}
validation/<slug>/{adapter_fidelity.json, reconstruction.json, label_quality.json, *.png}
```

## 9. Canonical feature layer

### 9.1 Namespace

Every quantity a model can ask for has one canonical name encoding physics and provenance:
`<quantity>_<source>`. Examples: `ip_ptdata`, `bt_ptdata`, `pinj_total_corpus`,
`tinj_total_corpus`, `ech_power_total_corpus`, `kappa_efit01`, `tritop_efit01`,
`tribot_efit01`, `gapin_efit01`, `r0_efit01`, `betan_efit01`, `qpsi_efit01` (profile),
`pres_efit01` (profile), `ne_mtanh_ts` (profile), `te_mtanh_ts` (profile),
`rot_csaps_cer` (profile), `ech_rho_torbeam`.

`FeatureSpec(name, kind, units, source, grid=None, notes="")` where `kind` is `scalar` or
`profile`. Profiles live on the shared 33-point grid `rho = linspace(0, 1, 33)`, matching the
`spatial_coordinates` array in the upstream test file. The registry is a module-level tuple
of frozen specs, the same pattern as `FROZEN_MODALITIES` in `ignite/dynamics_config.py`.

A model's `InputSpec` maps each trained-on feature name to a canonical name. Substitutions
are therefore explicit and reviewable, for example `kappa_EFITRT1 <- kappa_efit01`
(real-time EFIT was used in training; offline efit01 is what fdp serves).

### 9.2 Storage

`<shot>_features.h5`, corpus layout: one group per canonical feature, `xdata` float64
seconds, `ydata` float32 `(C, T)` with C = 1 for scalars and 33 for profiles. Features keep
their native time base; high-rate PTDATA scalars are decimated to 1 ms means to bound size.

Group attrs: `units`, `source`, `resolver`, `fetched_at`, `complete`, and `rho` for profiles.
File attrs: `shot`, `missing` (every requested feature that failed, with the exception class),
`labelmaker_version`, `git_sha`. Availability is measured per shot and recorded, following
`/scratch/gpfs/nc1514/fdp/scripts/omnimode.py`. A group is (re)fetched only if absent,
incomplete (interrupted run), or newly requested.

### 9.3 Time-base conventions

Copied from `ignite/train_dynamics.py` (bug found and fixed there on 2026-08-11):

- sample rate from the span, `fs = (n - 1) / (x[-1] - x[0])`, never `1/median(diff(x))`
  (float32 `xdata` loses the step to cancellation);
- sample index `round((t - x[0]) * fs)`, clamped into the record (actuator groups start at
  negative times; a negative index would silently wrap);
- corpus `xdata` is in seconds; upstream archives are in milliseconds; convert at the
  boundary and store seconds.

### 9.4 Resolvers

Plain functions `resolve(shot, specs, ctx) -> dict[name, FeatureArray | Missing]`, one module
per source, selected by `FeatureSpec.source`.

- **corpus**: opens `<shot>_processed.h5`; a group whose `ydata.shape[-1] < 2` is absent (the
  `(C, 1)` sentinel documented in `data/multi_file_dataset.py`). Sums `pinj` (8 beams),
  `tinj`, and `ech_power` (12 gyrotrons) into totals; passes raw `ts_core_*`,
  `ts_tangential_*`, `cer_rot` through for the fits resolver. Units of `pinj` in the corpus
  are verified against the archive (archive `pinj` is kW) in Phase 1.
- **fdp**: `MdsSignal` on the `efit01` tree (aeqdsk scalars; geqdsk `qpsi`, `pres`, `psirz`,
  grid) and `PtDataSignal('ip')`, `PtDataSignal('bt')`. Import-guarded so the default
  environment never touches toksearch. Worker pool forked before the first fetch (ptserver is
  not fork-safe). Expects a cached token from `pixi run fdp login`; sets
  `FDP_NO_AUTO_LOGIN` in batch. No retries exist in fdp; labelmaker retries once per signal
  then records the miss. PTDATA index coverage ends near shot 201,299; newer shots need
  `ImasSignal`, handled in Phase 2 if the corpus sample reaches them.
- **fits**: maps Thomson and CER channels to rho via efit01 `psirz` (from `geometry.py`),
  fits mtanh to ne and Te and a smoothing spline (csaps) to rotation, evaluates on the
  33-point grid at each efit01 time. Matches what the original profile pipeline did in kind;
  exactness is measured, not assumed (Section 12).
- **derived**: `ech_rho_torbeam`. For Phase 1 the value comes from
  `/projects/EKOLEMEN/profile_predictor/DATA/ech_rho.pkl` where the shot exists, else 0,
  which the tearing model's training filter admitted (`x0[:, 10] >= 0`). Declared as an
  approximation in the model card.

### 9.5 Geometry (highest-risk piece)

Thomson and CER channel `(R, Z)` are not in the corpus. Two routes, verified in order during
Phase 1: (1) the IMAS `thomson_scattering` and `charge_exchange` IDS through fdp
`ImasSignal`; (2) a static geometry table checked into `features/geometry/` and verified
against one reference shot from the profile-predictor archive (`thomson_density_psi_raw_1d`
gives per-channel psi for old shots, which pins the mapping). Whichever route works is
recorded in the feature card of `ne_mtanh_ts`, `te_mtanh_ts`, `rot_csaps_cer`.

## 10. Model folders

### 10.1 Naming scheme

Slug grammar: `<device>-<phenomenon>-<what-is-predicted>-<architecture>[-<variant>]`.
Hyphenated as the HuggingFace-style id in the card (`plasmacontrol/<slug>`), underscored as
the Python folder name. Labelmaker output groups use the folder name.

| Folder | Card id | Task | Status |
|---|---|---|---|
| `d3d_tearing_onset_cnn1d` | `plasmacontrol/d3d-tearing-onset-cnn1d` | binary + regression | implemented (Phase 1) |
| `d3d_elm_time_to_event_dsm` | `plasmacontrol/d3d-elm-time-to-event-dsm` | survival | scaffold (Phase 2) |
| `d3d_ech_beam_fate_mlp` | `plasmacontrol/d3d-ech-beam-fate-mlp-v15` | 3-class + 3 regression | scaffold (Phase 2) |
| `d3d_tearing_time_to_event_dsm` | `plasmacontrol/d3d-tearing-time-to-event-dsm` | survival | scaffold |
| `d3d_ech_deposition_torbeamnn` | `plasmacontrol/d3d-ech-deposition-torbeamnn-s1` | regression + binary | scaffold |
| `d3d_kinetic_equilibrium_rtcakenn` | `plasmacontrol/d3d-kinetic-equilibrium-rtcakenn-v1` | profile regression | scaffold |
| `d3d_inpa_image_cnn` | `plasmacontrol/d3d-inpa-image-cnn` | image regression | scaffold |

`models/README.md` lists this roster, the grammar, and the exclusions (TokEye,
`ae_tf_maskrcnn`, TokaMind, diag2diag) with one-line reasons so they are not re-added.

### 10.2 Model card (`README.md`)

HuggingFace model-card template: YAML front matter, then Model Details, Uses, Bias/Risks/
Limitations, Training Details, Evaluation, Technical Specifications, Citation, Contact.
Standard front-matter keys used: `pipeline_tag` (`tabular-classification`,
`tabular-regression`, `time-series-forecasting`), `tags` (device, phenomenon, task),
`library_name`, `datasets`, `metrics`, `model-index`. One custom block:

```yaml
labelmaker:
  status: implemented            # implemented | scaffold
  upstream:
    path: /projects/EKOLEMEN/simple_ae_predictor/models/rt_multi_io/
    artifacts: [best_model_0_4c.h5, ..., best_model_9_4c.h5]
    sha256: {best_model_0_4c.h5: "...", ...}
    author: ""
    trained: 2022-11
  framework: keras
  time_step_ms: 25
  inputs:                        # trained-on name <- canonical name
    - "bt <- bt_ptdata"
    - "kappa_EFITRT1 <- kappa_efit01"
  outputs:
    - {name: tm_prob, task: binary, activation: sigmoid}
    - {name: betan, task: regression, units: ""}
  approximations:
    - "EFITRT1 inputs served by offline efit01"
    - "EC.RHO_ECH from archived ech_rho.pkl, else 0"
```

The registry parses the front matter. A test asserts the card's `inputs` and `outputs`
match `spec.py` for every implemented model, and that every scaffold card parses. The
validate stage writes metrics into `model-index`, so the card is the single place to read
what the model is, where it came from, and how its labels performed.

### 10.3 `spec.py` contract

Frozen dataclasses in `models/base.py`:

- `InputField(model_name, canonical, kind, lag="t", transform=None)`; `lag` is `"t"` or
  `"t+dt"`; `transform` is a named pure function (e.g. `reciprocal` for `1/qpsi`).
- `InputSpec(fields, dt_s, rho_grid, nan_policy, domain)` builds `(T, ...)` arrays from the
  feature file with `timebase.py`. `domain` holds the training-time value ranges (taken from
  the upstream preprocessing filter); timesteps outside it are flagged, not dropped.
- `OutputField(name, task, activation, classes, units, output_index)`.
- `OutputSpec(fields)`.
- `load(paths) -> Predictor`; `Predictor(inputs) -> dict[name, (T, ...)]`. Ensembles return
  `mean`, `min`, `max`.
- `ModelAdapter(slug, card, input_spec, output_spec, load)`.

Runners in `models/runners/` are the only place a framework is imported, each guarded.

### 10.4 Phase 1 adapter: `d3d_tearing_onset_cnn1d`

Upstream: `/projects/EKOLEMEN/simple_ae_predictor/models/rt_multi_io/best_model_{0..9}_4c.h5`
(Keras; ten-member ensemble; 12,086 params each), training code `train.py` in the same
directory, reference harness `test/test.py`, reference features `test/test_shots.h5`.

Inputs, from `train.py:31-33`. 0-D at `t+dt`, `dt = 25 ms`:

| Trained-on | Canonical | Notes |
|---|---|---|
| `bt` | `bt_ptdata` | |
| `ip` | `ip_ptdata` | |
| `pinj` | `pinj_total_corpus` | archive is kW; verify corpus units |
| `tinj` | `tinj_total_corpus` | |
| `R0_EFITRT1` | `r0_efit01` | verify which efit01 node the profile pipeline used (filter admits 1.65..1.9 m) |
| `kappa_EFITRT1` | `kappa_efit01` | |
| `tritop_EFIT01` | `tritop_efit01` | |
| `tribot_EFIT01` | `tribot_efit01` | |
| `gapin_EFIT01` | `gapin_efit01` | m |
| `ech_pwr_total` | `ech_power_total_corpus` | W; NaN or negative -> 0 (upstream rule) |
| `EC.RHO_ECH` | `ech_rho_torbeam` | approximation, see 9.4 |

1-D at `t`, 33 points each:

| Trained-on | Canonical | Transform |
|---|---|---|
| `thomson_density_mtanh_1d` | `ne_mtanh_ts` | 1e19 m^-3 |
| `thomson_temp_mtanh_1d` | `te_mtanh_ts` | keV |
| `1/qpsi_EFITRT1` | `qpsi_efit01` | reciprocal |
| `pres_EFIT01` | `pres_efit01` | Pa |
| `cer_rot_csaps_1d` | `rot_csaps_cer` | krad/s |

Preprocessing: NaN -> 0 on both input blocks (upstream `test.py`). Outputs: index 0
`betan` (regression, no activation); index 1 `tm_prob` (binary; the head emits a logit
trained with `from_logits=True`, so apply sigmoid). Labels are stamped at `t+dt`.
Domain ranges come from the `idx` filter in `train.py:83`.

## 11. Label format

`<shot>_labels.h5`, corpus layout. Group path `<folder_slug>/<label>`, e.g.
`d3d_tearing_onset_cnn1d/tm_prob`. `xdata` float64 seconds on the model's native time step
(consumers resample; the IGNITE frame is 50 ms). `ydata` float32 `(C, T)`: C = 1 for binary
and regression, `n_classes` for multiclass probabilities. Probabilities are stored, never
thresholded labels. Companion groups: `<label>_spread` `(2, T)` with ensemble min and max,
and `<label>_valid` `(1, T)` uint8, 1 where all inputs were present and inside the training
domain.

Group attrs: `task`, `classes`, `units`, `activation`, `card_id`, `artifact_sha256`,
`ensemble_n`, `time_step_ms`. File attrs: `shot`, `labelmaker_version`, `git_sha`,
`features_sha256`, `run_id`. Files are written atomically (`tmp.replace(final)`).

`labels_index.parquet` at the data root: one row per (shot, card_id, label) with `n_valid`,
`n_total`, `run_id`, so "which shots have a tearing label" is one query.

## 12. Runner

`python -m labelmaker.run <stage>`:

- `features --models ... (--shots N N | --shot-file F | --corpus [--sample N --seed S])`. Resolves the union of canonical features for the models,
  forks a pool before the first fetch, writes each feature file atomically, skips complete
  groups.
- `infer --models ...`. Loads each model once per worker,
  builds inputs, predicts, writes labels atomically, skips shots already labeled by the same
  `artifact_sha256`.
- `validate --models ...` (Section 13).
- `all` chains the three.

All stages run in the `labelmaker` pixi environment (`pixi run -e labelmaker python -m labelmaker.run ...`).

Every shot is isolated: per-shot `try/except` that logs the miss with the exception class,
and a per-shot `SIGALRM` timeout (IGNITE measured ~56 of 3,000 shots hanging on HDF5 reads).
Each invocation writes `runs/<run_id>/manifest.json`: resolved config, git sha, shot list,
model card ids and sha256s, hostname and CPU/GPU info.

Proof-of-concept shot list: 100 shots sampled (fixed seed) from the 1,503 shots present in
both the corpus and the tearing-mode training archive
(`/projects/EKOLEMEN/tm_data/z.npy`; shots 185945..190997), so Section 13 has something to
compare against. `--corpus` scales to all 16,909 with no code change.

## 13. Validation and reliability

Outputs under `validation/<slug>/`; headline metrics written into the card's `model-index`.

1. **Adapter fidelity.** Run the adapter's `Predictor` on the feature arrays in the upstream
   `test/test_shots.h5` (7 shots, 240 timesteps at 25 ms) and compare against a direct Keras
   ensemble call on the same arrays. Tolerance 1e-5. Proves the wrapper before any
   reconstruction enters. The test file's columns differ slightly from the training names
   (`*_EFITRT1` variants of tritop/tribot/gapin/pres, no ECH columns, no `tm_label`); the
   check feeds identical arrays to both paths and zero-fills the absent columns on both
   sides, so it tests the wrapper, not the column mapping.
2. **Reconstruction fidelity.** On overlap shots, align reconstructed features to the
   archived training rows (`x0.npy`, `x1.npy`, `z.npy`; rows carry no timestamps) by
   cross-correlating `ip` and `betan` trajectories to recover the time offset. Report
   per-feature relative error and a distribution-shift statistic (KS) per shot and pooled.
   This is where `EFITRT1 <- efit01` and the home-grown profile fits are measured.
3. **Label quality.** Against ground truth (`/projects/EKOLEMEN/profile_predictor/DATA/ntm_labels.pkl`,
   50 ms label series for 5,834 shots; and archived `y.npy` on overlap shots), report AUROC,
   F1 at 0.5, and calibration for `tm_prob`, computed twice: with archived inputs and with
   reconstructed inputs. The gap is the reconstruction penalty and is the number that
   answers "is this reliable".

## 14. Testing

Synthetic-first and test-driven, one test module per source module under
`tests/labelmaker/`. Real-data tests use `pytest.mark.skipif` on
`/scratch/gpfs/EKOLEMEN/foundation_model`; fdp tests run only with `LABELMAKER_FDP=1`.

- `test_timebase.py`: span-based rate, clamping, window means, resample on constructed arrays.
- `test_feature_store.py`, `test_label_store.py`: round-trips through a temp HDF5, attrs,
  atomic write, skip-if-complete.
- `test_resolve_corpus.py`: a tiny synthetic `_processed.h5` with one `(C, 1)` absent group.
- `test_resolve_fits.py`: mtanh and csaps recover known synthetic profiles.
- `test_registry.py`: every card parses; implemented cards match their `spec.py`.
- `test_tearing_adapter.py`: inputs assembled from synthetic features have the right shapes,
  lags, and transform; decode applies sigmoid to index 1 only; domain flags set.
- `test_run.py`: `features` and `infer` on synthetic inputs end to end in a temp root.

## 15. Phases

1. **Phase 1 (first implementation plan):** package, feature layer, geometry route,
   tearing-onset adapter, scaffolds and cards for the roster, runner, validation on 100
   shots, pyproject changes.
2. **Phase 2:** `d3d_elm_time_to_event_dsm` (recover the 124 feature names from
   `/projects/EKOLEMEN/wpqh_elm_hiro/hiro_scripts/`) and `d3d_ech_beam_fate_mlp` (inputs
   fully specified upstream; PCA constants exist); `ImasSignal` path for shots past the
   PTDATA index cap.
3. **Phase 3:** TabPFN substitution study. For each implemented model, fit TabPFN on the
   feature cache against the model's outputs and against ground truth; compare; write
   results into the cards.
4. **Phase 4:** IGNITE embedding probes once codec checkpoints are on stellar.

## 16. Risks and verifications scheduled for Phase 1

- Channel geometry for Thomson and CER (Section 9.5). Blocking for the profile features.
- Which efit01 nodes correspond to `R0_EFITRT1` and the shape scalars the profile pipeline
  used; confirm from the pipeline source if it is on disk, else by matching archived values
  on an overlap shot.
- Corpus `pinj`/`tinj`/`ech_power` units versus archive units (kW vs W).
- fdp throughput: omnimode's log shows ~38 min for 24 shots at 12 workers with ~215 channels
  each; labelmaker needs ~20 channels per shot, so 100 shots is minutes to an hour. Corpus
  scale is a scheduling question, not a design one.
- Mid-merge FusionAIHub tree (Section 7).

## 17. References

- Corpus: `/scratch/gpfs/EKOLEMEN/foundation_model/<shot>_processed.h5`; loader
  `src/tokamak_foundation_model/data/data_loader.py` (`SIGNAL_CONFIGS` at line 305); absent
  sentinel `src/tokamak_foundation_model/data/multi_file_dataset.py:845-861`.
- Time-base convention: `src/tokamak_foundation_model/ignite/train_dynamics.py:170-186`.
- Dependency posture to copy: `src/tokamak_foundation_model/ignite/gate.py:26-29`.
- fdp fetch template: `/scratch/gpfs/nc1514/fdp/scripts/omnimode.py`; toksearch docs in
  `toksearch/__init__.py` and `toksearch_d3d/__init__.py` of the fdp pixi env.
- Tearing model upstream: `/projects/EKOLEMEN/simple_ae_predictor/{models/rt_multi_io,test}`;
  training arrays `/projects/EKOLEMEN/tm_data/{x0,x1,y,z}.npy`; ground truth
  `/projects/EKOLEMEN/profile_predictor/DATA/{ntm_labels.pkl,ntm_labeled_shots.txt,tm_shots.npy}`;
  `ech_rho.pkl` in the same directory.
- Model reference list (not a source of artifacts): `/scratch/gpfs/nc1514/tokamak_deploy_bench/MODEL_ROSTER.md`.
- Shot catalog with per-shot HDF5 group availability:
  `/scratch/gpfs/nc1514/shotsearch/data/manifest.parquet` (42,809 shots).
