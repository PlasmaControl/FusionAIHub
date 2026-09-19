# Labelmaker design

**Date:** 2026-09-03
**Status:** Phase 1 built (2026-09-04); sections 6-16 describe what was built, not what was planned
**Location:** `src/labelmaker/` in FusionAIHub (FAITH), branch `labelmaker`
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
| Input source | Reconstruct features (the 25 ms archive store where it has the shot, else corpus HDF5 + fdp efit01/zipfit01/PTDATA; ZIPFIT stands in for the pipeline's own profile fits) and validate against the archived training arrays on the 3,246 corpus/archive overlap shots. |
| Architecture | Shared feature layer with thin per-model adapters; three stages with per-shot HDF5 between them. |
| Model weights | Upstream artifacts only, copied once with sha256, verified before every load; the Keras-2 artifacts are read directly and evaluated in torch against a frozen real-Keras reference; no ONNX conversion. |
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

About 20 source files. Plain functions and frozen dataclasses. No Pydantic (repo convention
is `@dataclass`). torch is imported in exactly two places - `models/runners/keras_h5.py`,
which evaluates the Keras-2 graphs, and `validate.py`, which checks that evaluator against
frozen real-Keras outputs - and reaches nothing downstream: both take and return numpy.
Labelmaker sits on the `ignite/gate.py` side of the IGNITE reuse boundary: it imports nothing
from `tokamak_foundation_model` model packages.

```
src/labelmaker/
  __init__.py
  config.py              Paths dataclass with env-var overrides; atomic_path; sha256_of; git_sha
  catalog.py             corpus shot enumeration; (C,1) absent-signal check; archive overlap; seeded sample
  timebase.py            span-based sample rate, clamped index, window means, decimate-to-step
  features/
    __init__.py
    namespace.py         FeatureSpec registry (canonical name, kind, units, ordered sources + locators)
    store.py             read/write <shot>_features.h5; missing/complete bookkeeping; transient causes
    resolve_archive.py   the 25 ms archive store the Phase 1 model was trained from (primary source)
    resolve_corpus.py    actuator totals (pinj, tinj, ech_power) from <shot>_processed.h5
    resolve_fdp.py       efit01 aeqdsk/geqdsk, zipfit01 profiles and PTDATA ip/bt via toksearch
  models/
    __init__.py
    README.md            roster, naming scheme, exclusions
    base.py              InputField/InputSpec/DomainRule/UnknownWhenActive/OutputSpec/ModelAdapter
    registry.py          discovers model folders; parses cards; cross-checks spec; writes model-index
    runners/
      __init__.py
      keras_h5.py        reads a Keras-2 legacy .h5 (config + weights) and evaluates the graph in torch
    d3d_tearing_onset_cnn1d/      README.md  spec.py  __init__.py     (implemented, Phase 1)
    d3d_elm_time_to_event_dsm/    README.md  spec.py  __init__.py     (scaffold)
    d3d_tearing_time_to_event_dsm/ ...                                (scaffold)
    d3d_ech_beam_fate_mlp/ ...                                        (scaffold)
    d3d_ech_deposition_torbeamnn/ ...                                 (scaffold)
    d3d_kinetic_equilibrium_rtcakenn/ ...                             (scaffold)
    d3d_inpa_image_cnn/ ...                                           (scaffold)
  labels/
    __init__.py
    schema.py            LabelSpec: task, classes, units, activation, time base, artifact digest
    store.py             write <shot>_labels.h5 in corpus layout with provenance attrs; parquet index
  run.py                 python -m labelmaker.run {features|infer|validate|all}
  validate.py            adapter fidelity, reconstruction-vs-archive, label-vs-truth reports
tests/labelmaker/        synthetic-first, module-mirrored; skipif on the corpus path; live fdp opt-in
```

Planned here and not built in Phase 1: `resolve_fits.py` and `geometry.py` (Section 9.5 - the
three kinetic profiles are served as ZIPFIT fits instead), and `runners/keras.py` /
`runners/torch.py` (the one runner reads the artifact format directly; a torch-native model
would get its own `runners/torch_pt.py`).

## 7. Environment and packaging

- FusionAIHub is pixi-managed (Python 3.11). Labelmaker uses that, not a separate uv project.
- `pyproject.toml`, as built:
  - `[tool.hatch.build.targets.wheel] packages = ["src/faith", "src/tokamak_foundation_model", "src/labelmaker"]`
    so `import labelmaker` works through the editable install.
  - pixi feature `labelmaker`: `pyarrow` (for `labels_index.parquet`), `pyyaml` (declared
    directly rather than relied on transitively), a protobuf pin for the fresh solve, and CPU
    torch wheels. **No `tensorflow-cpu`**: the Keras artifacts are evaluated in torch
    (`runners/keras_h5.py`), and the one-off real-Keras reference was produced outside this
    environment and frozen into `tests/labelmaker/data/tearing_golden.npz`.
  - environment `labelmaker = ["labelmaker", "fdp"]` - no `cuda`; the heaviest model is 12k
    parameters on CPU. The `fdp` feature supplies `fdp`, `toksearch` and `toksearch_d3d` from
    the `ga-fdp` channel, and carries an activation table setting
    `LD_LIBRARY_PATH=$CONDA_PREFIX/lib:...`. Without it `import torch` binds the system
    `libstdc++`, which lacks `GLIBCXX_3.4.29`, and every compiled extension loaded after it
    that needs the symbol (`fdp`, `toksearch`, `toksearch_d3d`, `scipy`) fails to import.
    That once disabled the whole fdp path silently; the activation table is the fix, a test
    imports torch and toksearch in one process, and `resolve_fdp._import_diagnosis` reports
    the reason if it ever recurs.
  - Later phases: optional `tabpfn` feature.
- Live fetches run under the wrapper: `pixi run -e labelmaker fdp run python -m labelmaker.run ...`.
  Without it PTDATA fails (`getservbyname failed for task 'PTSERVER'`) and MDSplus fails
  (`TREE-E-FOPENR`).
- Tests: `pixi run -e labelmaker python -m pytest tests/labelmaker -q -W error`; the two live
  fdp tests are opt-in (`--run-live` or `LABELMAKER_FDP=1`).
- Lint: `pixi run -e labelmaker ruff check src/labelmaker tests/labelmaker` with the repo's
  default rule set. `E501` is not enabled; do not reflow for line length.

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

Every quantity a model can ask for has one canonical name naming the **physics quantity**, not
the source: `ip`, `bt`, `pinj_total`, `tinj_total`, `ech_power_total`, `r0`, `kappa`,
`tritop`, `tribot`, `gapin`, `betan`, `qpsi` and `pres` (profiles), `ne_zipfit`, `te_zipfit`,
`rot_zipfit` (profiles), `ech_rho`. The design originally encoded the source in the name
(`ip_ptdata`, `kappa_efit01`); that was dropped because most quantities are served by more
than one source with different shot coverage, and which one actually produced a feature is a
property of the shot, not of the name.

`FeatureSpec(name, kind, units, sources, locators, step=0.0, notes="")` where `kind` is
`scalar` or `profile`, `sources` is an **ordered preference tuple** over `("archive",
"corpus", "fdp")` - cheapest first; `run.features_for_shot` walks them in that global order,
which a test asserts every spec agrees with - and `locators` is the parallel tuple of names in
each source: the archive column, the corpus group, the MDSplus node or PTDATA point.
Provenance is recorded per shot in the stored group's `resolver` attribute. Profiles live on
the shared 33-point grid `RHO_GRID = linspace(0, 1, 33)`, which for the two EFIT profiles is
uniform normalized psi rather than rho - measured, Section 9.4.

`namespace.SAMPLING_BY_SOURCE` declares, totally over the sources, how a stored array from each
is read at model-input time: an archive-served field is already the archive's 50 ms boxcar and
is read nearest-sample; a corpus- or fdp-served field is a true-time record and is windowed
into that same boxcar (`models.base.ARCHIVE_WINDOW_S`), so a mixed-source row refers to one
physical interval (Section 9.3).

A model's `InputSpec` maps each trained-on feature name to a canonical name, so substitutions
are explicit and reviewable: `kappa_EFITRT1 <- kappa` (real-time EFIT in training, offline
efit01 served now), `thomson_density_mtanh_1d <- ne_zipfit`.

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

And one convention measured in Phase 1 rather than copied:

- **the archive store's time base.** Row `k` of the 25 ms archive is the mean of the raw
  signal over `[25(k-2), 25k]` ms - a 50 ms boxcar centred on `25(k-1)` ms - reproducing raw
  PTDATA `ip` to 3.9e-08 median relative error (float32 round-trip precision); every other
  offset and width tried is 1e-4 or worse. `resolve_archive` stamps row `k` at `25k` ms, one
  step late, and that stamp is deliberately left alone: it is what the model trained on, and
  every published archive-derived figure depends on it. `InputSpec.build` reconciles the
  sources instead, per `SAMPLING_BY_SOURCE`, so a label at `t` is computed from inputs averaged
  over `[t-50ms, t]` uniformly across sources - causal, and as trained. That convention beats
  nearest-sample and every other window placement by one to two orders of magnitude on the
  corpus and fdp paths (measured against the archive column on random overlap shots).

### 9.4 Resolvers

Plain functions `resolve(shot, names, ...) -> (arrays, missing)`, one module per source, tried
cheapest-first per feature; `missing` maps a feature to a short cause, and
`store.TRANSIENT_CAUSES` decides which causes a later run retries.

- **archive** (primary): `/projects/EKOLEMEN/profile_predictor/DATA/new_h5_data/example_191450_183224.h5`,
  5,000 shots (183224..191450), 209 columns each, 240 rows at 25 ms - the store the Phase 1
  model's training features were built from. Matching `x0.npy` rows to archive rows on the
  five bit-identical columns (bt, ip, tritop, tribot, gapin) resolves 9,805 of 9,805 rows
  uniquely, so this store is the row-level provenance of the training set. Columns vary per
  shot: `ip`/`bt` are absent on 2,192 of 5,000, `EC.RHO_ECH` on 2,591. It covers 3,246 of the
  16,909 corpus shots (19%): the proof-of-concept path, not the scaling path.
- **corpus**: opens `<shot>_processed.h5`; a group whose `ydata.shape[-1] < 2` is absent (the
  `(C, 1)` sentinel). Sums `pinj` (8 beams; **W in the corpus, canonical kW**), `tinj` (N m,
  no scaling) and `ech_power` (12 gyrotrons, W) into totals, decimated to 1 ms. 1.3-3.7% of
  corpus files are truncated on disk and are recorded as a per-shot `OSError` miss. The corpus
  `ech_power` sum under-reports the archive's `EC.PECH` by ~22% at the median over a random 60
  overlap shots - its 12 channels miss gyrotrons `EC.PECH` counts - and the corpus has no
  deposition-location group; ECH handling is on hold (Section 16).
- **fdp** (the scaling path): `PtDataSignal('ip'|'bt')`, `MdsSignal` on `efit01` (aeqdsk
  scalars; geqdsk `qpsi`, `pres`, `rmaxis`; time from `atime`/`gtime` - `aeqdsk:time` is
  poloidal flux, a trap) and on `zipfit01` (`EDENSFIT`, `ETEMPFIT`, `TROTFIT`, the ZIPFIT
  substitutes for the pipeline's own mtanh/csaps fits). Every axis is identified by its units
  and length, never by position. No unit factor is needed on any of the thirteen features
  (every median ratio to the archive column is 1.000 within 6e-4); `rot_zipfit` is in kHz, read
  off the node's units field. The geqdsk profiles arrive on 65 points of normalized psi and are
  taken at the even indices, **not** converted to rho via `rhovn`: against the archive that
  conversion is 7x worse on `qpsi` and 10x worse on `pres`. Against the archive, under the
  convention `build()` uses (50 ms window ending at `t`), on a random 10 overlap shots:

  | feature | median rel | p90 rel |
  |---|---|---|
  | ip | 9.8e-05 | 1.4e-03 |
  | bt | 7.2e-05 | 1.9e-04 |
  | r0, kappa, tritop, tribot, gapin, betan | ~3e-08 | 2e-04 .. 3e-03 |
  | qpsi, pres | ~3.5e-08 | 5e-03, 1.9e-02 |
  | ne_zipfit, te_zipfit, rot_zipfit | ~2.4e-08 | 2e-03 .. 3e-03 |

  Coverage gaps are the store's, not fdp's: fdp served `ip`/`bt` on all 120 of a random
  120-shot sample. `TROTFIT` is absent on ~22% of shots and `EDENSFIT`/`ETEMPFIT` on ~9%. The
  `fdp run` wrapper is mandatory; the worker pool is forked before any fetch (ptserver is not
  fork-safe) and before toksearch is imported in the parent.
- **`ech_rho`**: the archive column `EC.RHO_ECH` (TORBEAM deposition location), archive-only.
  Zero-filled where absent (upstream's ECH-off convention) and then adjudicated by an
  `UnknownWhenActive` pair rule against the ECH power, so a fabricated location with power
  flowing invalidates the row instead of being assumed benign. `ech_rho.pkl` is not used.

### 9.5 Geometry - moved to Phase 2

Thomson and CER channel `(R, Z)` are not in the corpus - measured: no corpus group carries
channel positions at all - and neither route planned here was built in Phase 1, because ZIPFIT
already serves the three kinetic profiles from `zipfit01` at a measured substitution cost
against the model's own mtanh/csaps training inputs (Section 13, proof-of-concept pool):
density 6.7e-2, temperature 1.3e-1, rotation 1.4e-1 median relative difference,
correlation 0.92-0.97 (100 shots, 243,309 profile points). Fitting the profiles from raw Thomson and CER needs the geometry and
is the first thing Phase 2 should price against those numbers.

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
    - "bt <- bt"
    - "kappa_EFITRT1 <- kappa"
  outputs:
    - {name: tm_prob, task: binary, activation: sigmoid}
    - {name: betan, task: regression, units: ""}
  approximations:
    - "EFITRT1 inputs served by offline efit01"
    - "EC.RHO_ECH from the archive column; zero-filled where absent and the row flagged when ECH power flows"
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

Runners in `models/runners/` are the only place a model framework is loaded; torch also appears in `validate.py`, where the runner is checked against the frozen reference.

### 10.4 Phase 1 adapter: `d3d_tearing_onset_cnn1d`

Upstream: `/projects/EKOLEMEN/simple_ae_predictor/models/rt_multi_io/mse_bin_os_w/best_model_{0..9}_4c.h5`
(Keras 2.8; ten-member ensemble; 12,086 params each), training code `train.py` in
`rt_multi_io/`, reference harness `test/test.py`, reference features `test/test_shots.h5`.
Each `_4c.h5` member has two inputs, `(None, 11)` and `(None, 33, 5)`, and one two-column
output: column 0 `betan` (linear), column 1 the tearing logit. The ten members carry different
layer names (`input_1`/`input_2` .. `input_19`/`input_20`), so inputs are fed positionally.

Inputs, from `train.py:31-33`. 0-D at `t+dt`, `dt = 25 ms`:

| Trained-on | Canonical | Notes |
|---|---|---|
| `bt` | `bt` | archive bit-identical; fdp PTDATA, T |
| `ip` | `ip` | archive bit-identical; fdp PTDATA, A |
| `pinj` | `pinj_total` | archive kW; corpus W, scaled |
| `tinj` | `tinj_total` | N m |
| `R0_EFITRT1` | `r0` | offline efit01 `rmaxis` stands in |
| `kappa_EFITRT1` | `kappa` | offline efit01 stands in |
| `tritop_EFIT01` | `tritop` | bit-identical |
| `tribot_EFIT01` | `tribot` | bit-identical |
| `gapin_EFIT01` | `gapin` | m, bit-identical |
| `ech_pwr_total` | `ech_power_total` | W; negative -> 0 is a correction, NaN stays unmeasured |
| `EC.RHO_ECH` | `ech_rho` | archive-only; zero-fill plus pair rule, Section 9.4 |

1-D at `t`, 33 points each:

| Trained-on | Canonical | Transform |
|---|---|---|
| `thomson_density_mtanh_1d` | `ne_zipfit` | 1e19 m^-3 |
| `thomson_temp_mtanh_1d` | `te_zipfit` | keV |
| `1/qpsi_EFITRT1` | `qpsi` | reciprocal |
| `pres_EFIT01` | `pres` | Pa, bit-identical |
| `cer_rot_csaps_1d` | `rot_zipfit` | kHz - measured from the node's units, not krad/s |

Preprocessing: NaN -> 0 on both input blocks (upstream `test.py`), applied after the validity
flags are computed so a filled hole is never mistaken for a real zero. Outputs: index 0
`betan` (regression, no activation); index 1 `tm_prob` (binary; the head emits a logit trained
with `from_logits=True`, so sigmoid, applied after the ensemble mean as the upstream harness
does). Labels are stamped at the grid time `t`, on the 240-row 25 ms grid, with the 0-D inputs
read at `t+dt` - so a label answers "is a mode present at t+25 ms". Domain ranges come from
the `idx` filter in `train.py:81`, kept as `DomainRule` data and printed into the card.

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

Proof-of-concept shot list: 100 shots sampled (fixed seed) from the 3,246 shots present in
both the corpus and the tearing-mode training archive (`/projects/EKOLEMEN/tm_data/z.npy`),
written to `$LABELMAKER_ROOT/poc_shots.txt`, so Section 13 has something to compare against.
`--corpus` scales to all 16,909 with no code change.

## 13. Validation and reliability

Outputs under `validation/<slug>/`; headline metrics written into the card's `model-index`.

1. **Adapter fidelity.** The torch evaluator (`runners/keras_h5.py`) against real-Keras outputs
   frozen once into `tests/labelmaker/data/tearing_golden.npz` (1,673 rows from
   `test/test_shots.h5` x 10 members, `tensorflow-cpu==2.15.1`), so no framework is needed at
   run time. Four conjunctive gates replace the planned single 1e-5 absolute tolerance, which
   assumed outputs of order 1 when the tearing logit reaches 20.7: scale-normalized max 1e-5
   (measured 2.7e-6), median abs diff 1e-6 (7.65e-7), label-space max on the published
   `tm_prob` 1e-5 (1.2e-6), absolute max 1e-4 (5.6e-5). The residual is float32 rounding, not
   a semantic error: the evaluator's own float64-vs-float32 self-disagreement is the same size.
2. **Reconstruction fidelity.** On overlap shots, the archived training rows (`x0.npy`,
   `x1.npy`, `z.npy`; no timestamps) are aligned to our timesteps by an **exact
   nearest-neighbour match on the columns the archive served bit-identically** (bt, ip,
   tritop, tribot, gapin, each scaled by its within-shot spread), not by cross-correlating
   trajectories: it resolves every row uniquely at distance zero. Where fdp backfilled
   `bt`/`ip`, the archive-served geometry columns match and the reconstructed ones only break
   exact ties (EFIT01 at 50 ms holds the geometry for two grid steps on some shots). Per-feature
   median relative error, correlation and KS statistic are pooled over the matched rows, with
   per-feature shot counts; a feature absent for a shot is skipped, not priced as a zero-fill.
3. **Label quality.** Against the archived `y.npy` truth on the same matched rows
   (`ntm_labels.pkl` is not used): AUROC, F1 at 0.5, precision/recall, Brier and calibration
   for `tm_prob`, RMSE for `betan`, computed twice - with archived inputs and with
   reconstructed inputs - **over the same rows**, restricted to those labelmaker's own
   validity rule would publish. The difference is the reconstruction penalty. Scoring the two
   over different row sets understated the penalty by a third and inverted the F1 comparison
   on an early run, so the report carries both denominators and a `computed_from` note.

Every per-shot step is isolated (try/except plus the same SIGALRM budget as the runner); skips
are diagnosed and histogrammed, with a warning when one cause dominates or a whole source
served nothing - which is how a dead fdp path shows up in the JSON rather than in a separate
investigation.

## 14. Testing

Synthetic-first and test-driven, one test module per source module under
`tests/labelmaker/`. Real-data tests use `pytest.mark.skipif` on
`/scratch/gpfs/EKOLEMEN/foundation_model`; live fdp tests run only with `LABELMAKER_FDP=1` or
`--run-live`. The whole suite runs under `-W error`.

- `test_timebase.py`: span-based rate, clamping, window means, resample on constructed arrays.
- `test_feature_store.py`, `test_label_store.py`: round-trips through a temp HDF5, attrs,
  atomic write, skip-if-complete.
- `test_resolve_corpus.py`: a tiny synthetic `_processed.h5` with one `(C, 1)` absent group.
- `test_resolve_archive.py`: one shot's columns off a synthetic archive file; a missing column
  is a per-feature miss, a wrong radial length a named one.
- `test_resolve_fdp.py`: axis identification by units and length, the profile coordinate, the
  miss diagnosis, and torch followed by toksearch in one process; live fetches opt-in.
- `test_keras_h5.py`, `test_adapter_fidelity.py`: twelve hand-computed per-layer oracles, TF
  `'same'` padding, and the frozen real-Keras golden file with its four gates.
- `test_reconstruction.py`, `test_label_quality.py`, `test_ks_statistic.py`: row alignment
  (subset match, tie-break), metrics against scipy as an oracle, per-shot isolation, the
  card's model-index round trip.
- `test_registry.py`: every card parses; implemented cards match their `spec.py`.
- `test_tearing_adapter.py`: inputs assembled from synthetic features have the right shapes,
  lags, and transform; decode applies sigmoid to index 1 only; domain flags set.
- `test_run.py`: `features` and `infer` on synthetic inputs end to end in a temp root.

## 15. Phases

1. **Phase 1 (built):** package, feature layer (archive, corpus and fdp resolvers),
   tearing-onset adapter, scaffolds and cards for the roster, runner, validation on the
   100-shot proof-of-concept pool, pyproject changes. Geometry and the pipeline's own profile
   fits moved to Phase 2 (Section 9.5).
2. **Phase 2:** channel geometry and mtanh/csaps fits from raw Thomson and CER, priced
   against the ZIPFIT numbers; ECH deposition location off the archive (whether fdp can serve
   one); `d3d_elm_time_to_event_dsm` (recover the 124 feature names from
   `/projects/EKOLEMEN/wpqh_elm_hiro/hiro_scripts/`) and `d3d_ech_beam_fate_mlp` (inputs
   fully specified upstream; PCA constants exist); `ImasSignal` path for shots past the
   PTDATA index cap.
3. **Phase 3:** TabPFN substitution study. For each implemented model, fit TabPFN on the
   feature cache against the model's outputs and against ground truth; compare; write
   results into the cards.
4. **Phase 4:** IGNITE embedding probes once codec checkpoints are on stellar.

## 16. Risks and verifications - status after Phase 1

- Channel geometry for Thomson and CER: **retired for Phase 1** (Section 9.5); ZIPFIT serves
  the profiles at a measured cost. Phase 2's first job.
- Which efit01 nodes correspond to `R0_EFITRT1` and the shape scalars: **settled by
  measurement against the archive** - `rmaxis` and the aeqdsk `kappa`, `tritop`, `tribot`,
  `gapin`; the geqdsk profiles on uniform normalized psi (Section 9.4).
- Corpus `pinj`/`tinj`/`ech_power` units: **measured** - `pinj` W (canonical kW), `tinj` N m,
  `ech_power` W but under-reporting `EC.PECH` by ~22% at the median. ECH is **on hold** by the
  repo owner's instruction: the corpus cannot supply a deposition location, whether fdp can
  has not been investigated, and the card's corpus-path ECH figures are marked as pending
  re-measurement under the current sampling convention.
- fdp throughput, **measured** on the proof-of-concept pool (8 workers; 13 features through
  fdp plus the archive and corpus reads, per shot): 3.5 s median, 7.3 s worst, ~35 s wall for
  70 fdp-served shots. Inference is 0.1 s per shot. Corpus scale (16,909 shots) is a
  scheduling question, not a design one.
- The `import torch` / system `libstdc++` loader-order failure that silently disabled the fdp
  path (Section 7): fixed by the pixi activation table, made self-diagnosing, and covered by a
  test that imports torch and toksearch in one process.
- Mid-merge FusionAIHub tree: resolved; the work is on the `labelmaker` branch.

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
