# labelmaker

Runs the group's trained models over the FAITH shot corpus and writes their
predictions as per-shot label files in the corpus HDF5 layout, with a measured
statement of how much each model's labels can be trusted.

Design: `docs/superpowers/specs/2026-09-03-labelmaker-design.md`.
Phase 1 build log: `docs/superpowers/plans/2026-09-03-labelmaker-phase1.md`.
Code: `src/labelmaker/`. Tests: `tests/labelmaker/`.

## Running it

```bash
pixi install -e labelmaker              # once
pixi run -e labelmaker fdp login        # once per token; only the fdp resolver needs it

pixi run -e labelmaker label 199597     # the one-shot demo: every servable model on one shot
```

`label` runs the `analyze` stage over every model in `analyze_default.yaml`, then
leaves four files in `outputs/labelmaker/analysis/<shot>/` in this repo (override the
directory with `LABELMAKER_DEMO_OUT`): `<shot>_labels.h5` (the labels file, `<slug>/<label>/{xdata,ydata}`
plus `_spread`/`_valid`), `<shot>_labels.npz` (`time_s` plus one `"<slug>/<label>"` array
per label), `<shot>_analysis.json` (per-label summary) and `<shot>_labels.png`, one
panel per prediction. Pass several shots to
label them all; set `LABELMAKER_DEMO_FORCE=1` to recompute instead of reusing the
features and labels cached under `$LABELMAKER_ROOT`. Shots 199597-199607 (2024) have every input the models need; older shots
lack the CO2 interferometer channels the AE model reads, and some lack EFIT via fdp.

The full pipeline, stage by stage:

```bash
pixi run -e labelmaker fdp run python -m labelmaker.run all \
    --models d3d_tearing_onset_cnn1d \
    --overlap --sample 100 --seed 20260903 --workers 8 --timeout 300
```

The `fdp run` wrapper is required whenever a feature may come through fdp, which
for the tearing model is every shot whose archive group lacks `ip`/`bt` (2,192 of
the archive's 5,000) and every shot outside the archive. Without it PTDATA fails
with `getservbyname failed for task 'PTSERVER'` and MDSplus with `TREE-E-FOPENR`,
and those misses are recorded as transient, so the next correctly wrapped run
retries exactly them.

Shot selection is one of `--shots N N`, `--shot-file PATH`, `--corpus`, or
`--overlap` (shots present in both the corpus and the tearing model's training
archive), with `--sample N --seed S` to take a reproducible subset and
`--limit N` to keep the first N of it. Stages are
`features`, `infer`, `validate`, `all`, `analyze` and `events`; each is independently
rerunnable and skips work that is already complete unless given `--force`. The
100-shot proof-of-concept pool is `$LABELMAKER_ROOT/poc_shots.txt`; the
500-shot pool the reliability numbers below come from is `shots_500.txt` (the
100, 389 more validatable shots, and the tearing-mode shots 199597-199607).

Exit codes: 0 ok; 1 no shots selected; 2 usage; 3 the weights on disk do not
match the card's sha256 (nothing ran); 4 a requested model cannot be loaded; 5
adapter fidelity failed - the evaluator disagrees with the framework it is
supposed to reproduce, and no label from that model should be trusted; 6 a
validation report raised.

## One shot, all the labels you asked for

```bash
pixi run -e labelmaker fdp run python -m labelmaker.run analyze \
    --shots 199597 --config my_labels.yaml --out /some/dir
```

`analyze` takes its models from the config rather than `--models`. The config
is YAML (the default is `src/labelmaker/analyze_default.yaml`):

```yaml
labels:                                  # plotted top to bottom, in this order
  - d3d_tearing_onset_cnn1d/tm_prob
  - d3d_tearing_onset_cnn1d/betan
context:                                 # canonical features drawn above them
  - ip
  - pinj_total
threshold: 0.5                           # drawn on binary labels
thresholds:                              # per-label overrides of that value
  d3d_tearing_time_to_event_dsm/tm_risk_1s: 0.2
```

It runs `features` and `infer` for the models those labels need, then writes
`<out>/<shot>/<shot>_analysis.json` - per label: card id, weights digest, rows,
valid fraction, *why* rows are invalid (per rule), the peak and when the label
first crosses the threshold - and `<shot>_labels.png`, one panel per context
feature and per label, invalid rows washed grey, the ensemble spread as a
band.

When the shot is in the tearing model's training archive, the archived truth is
overlaid and scored: the rows the archive calls a tearing mode are shaded, the
first of them marked as the onset, `betan`'s archived column drawn as a dashed
line, and each label's `truth` block in the JSON carries AUROC, precision,
recall and F1 at the label's threshold, plus `lead_time_s` - how far before the
archived onset the label first crossed. A survival label is scored only on rows
before the onset, against "an onset occurs within the horizon". A shot with no
archived truth says so instead, per label. `--out` defaults to `$LABELMAKER_ROOT/analysis/`. The label file under
`labels/` stays the one canonical output; the analysis directory is a view of
it.

## What it writes

Everything under `$LABELMAKER_ROOT` (default
`/scratch/gpfs/EKOLEMEN/nc1514/labelmaker`; `$LABELMAKER_CORPUS` overrides the
corpus location):

| path | contents |
|---|---|
| `features/<shot>_features.h5` | canonical features, one group each, `xdata`/`ydata`, with the `resolver` that served it |
| `labels/<shot>_labels.h5` | `<slug>/<label>` plus `_spread` (ensemble min/max) and `_valid` companions |
| `labels_index.parquet` | one row per shot, model and label - the "which shots have labels" query |
| `models/<slug>/` | the weights, copied once; verified against the card's sha256 before every load |
| `masks/<shot>_masks.npz` | one `(diag, channel, pass)` block per planned channel: packed coherent/transient masks, their row and column summaries, 16-band log-power, the column times, a `_meta` JSON |
| `events/<shot>_events.parquet` | discrete events - one row per thing that happened, and the source that says so |
| `events/<shot>_sources.parquet` | one row per (source, diag, channel, pass) that RAN or was SKIPPED - the "did anybody look" query |
| `events_index.parquet` | one row per (shot, source, phenomenon) - the "which shots have EHOs" query |
| `text/logs_subset.jsonl` | the shots-in-hand slice of the 616 MB logbook dump, beside `logs_subset.missing` |
| `runs/events/<run_id>.json` | one mask run: settings, per-shot rows, per-source event totals |
| `runs/<run_id>/` | `manifest.json` (config, git sha, shot list), `log.txt` (one JSON row per shot), `summary.json` |
| `validation/<slug>/` | reliability reports (below) |
| `validation/<slug>/alarm_quality.json` | per-shot final-label and any-row FPR/FNR, warning times, jumps, and IPCW AUC for labels with archive truth |
| `analysis/<shot>/` | `analyze`'s JSON summary and figure for that shot |

Labels are probabilities, never thresholded, on the model's own time step.
`<label>_valid` is 1 where every input was present and inside the model's
training domain; a probability where it is 0 is an extrapolation. A label
stamped at `t` is computed from inputs averaged over `[t - 50 ms, t]`, the
convention the archive the model trained on was built with, uniformly across
sources; for the tearing model the 0-D inputs are read one step ahead, so the
label answers "is a mode present at t + 25 ms".

## Events: what happened, and on whose word

A label is a probability at every time step. An **event** is a single thing
with a start, an end, an optional frequency band and a named source, and the
`events` stage is what produces them:

```bash
pixi run -e labelmaker python -m labelmaker.run events \
    --shot-file $LABELMAKER_ROOT/recommender_v1.txt --limit 5 \
    --device cpu --tile-batch 8 --passes wide --timeout 900
```

No `--models`: the stage reads the corpus and one pinned U-Net checkpoint
(`models/tokeye/big_tf_unet_251210.pt`, refused unless its sha256 matches).
For each shot it plans eleven channels (`events/channels.py` - two magnetics,
three ECE, two CO2 chords, two BES, and two `mirnov` that stand in for `mhr`
where `mhr` is absent), runs the network over each channel's spectrogram, and
turns what it sees into rows:

| source | evidence_kind | what it claims |
|---|---|---|
| `tokeye_track` | detector | a coherent mode (or a `pickup` line) with a band, a chirp and a confidence |
| `tokeye_transient` | detector | class-agnostic `transient` points from one mask reference channel |
| `elm_clock` | heuristic | D-alpha `elm` points and the `elm_free` intervals implied by those same peaks |
| `ece_sawtooth` | heuristic | one point per inversion-qualified crash, with dropping-channel bounds; radius is not mapped |
| `dalpha_lh` | heuristic | L->H and H->L transitions |
| `actuator` | heuristic | the intervals NBI, ECH, the RMP coils and the gas valves were on for |
| `qh_proxy` | heuristic | an EHO inside an ELM-free NBI-heated flat-top - a proxy, and its `attrs` say so |
| `text` | text | a phenomenon this shot's own logbook entries name |
| `qmin_rule` | heuristic | the q-min regime bands of the Ip flat-top (`qmin_hybrid`/`qmin_elevated`/`qmin_high`) |
| `database:<table>` | database | a row of a curated table somebody sent us, e.g. `database:rwm_onsets_2017` |

The D-alpha clock reads filterscopes channels 0-7 independently of the U-Net,
using the first channel with at least two adjacent finite samples (prefer 0). It scales the finite
range to [0, 1], smooths for 0.64 ms, and picks peaks with prominence ≥ 0.03
and separation ≥ 3 ms. Within each contiguous finite run it rejects peaks
whose half-prominence width exceeds `DALPHA_MAX_WIDTH_MS = 5.0` ms. D-alpha
ELMs are millisecond bursts (a 200 Hz train is 5 ms apart); humps hundreds
of ms wide are baseline excursions. This shape guard is independent of
production-shot counts and applies only to the D-alpha clock. Smoothing
and width measurement never span NaN gaps or padding.
Point attributes are `prominence` (normalised),
`width_ms` (half-prominence width), `channel`, and `rate_hz_local` (centred
100 ms count / 0.1 s). Confidence is NaN: this is a heuristic that still needs
manual ELM validation. Padding and internal gaps cannot generate peaks or
quiet intervals; source coverage uses the first and last finite sample.
All-NaN filterscopes are skipped, never called ELM-free. `transient` is in
both registries and never supplies an ELM window feature or phenomenon hit.
ELM window rates, quiet fractions and ages require `source=elm_clock` and
`diag=filterscopes`; legacy magnetics clock rows are excluded before point
clustering or interval unions, including in mixed legacy/new tables.
Re-running events replaces old `tokeye_transient` rows by source; existing
read-only products are not migrated by changing the code. See the
[L-A assessment](superpowers/specs/2026-09-13-labels-assessment-A.md) for the
500-shot census and validation limits.

Flags: `--passes {wide,zoom}` (wide is 0.49 kHz/bin and 0.256 ms/column, zoom
is four times finer in frequency and four times coarser in time),
`--tile-batch` (tiles per forward pass; about memory, not speed),
`--amp` (fp16, CUDA only), `--norm {record,plasma}`, `--limit N`,
`--refresh-text`, `--run-id`. The stage runs one shot at a time whatever
`--workers` says - the network is one large torch module and the parallelism
that matters is inside the batched forward pass - and each shot still gets its
own SIGALRM budget, so a hung read costs one shot.

### Evidence kinds, and what may become a feature

`evidence_kind` is the column that says what KIND of claim a row is, and it
is load-bearing rather than decorative:

| evidence_kind | what the row is | may become a diagnostic feature |
|---|---|---|
| `detector` | a network or a threshold, on a measured signal | yes |
| `heuristic` | arithmetic on one or more measured signals | yes |
| `forecast` | a model's estimate of what was ABOUT to happen; carries `horizon_s` | no |
| `text` | a phrase in this shot's own logbook entries | no |
| `human` | an annotation somebody made | no |
| `database` | a value restated from another store | no |
| `model` | a model's output, not raised to a forecast | no |

**A `text` row does not describe what a diagnostic showed.** It records
that a phrase appeared in the logbook, and a phrase may be about the
DETECTOR rather than about the plasma: shot 185980's operator wrote
"Updated ELM detector tuning.", and shot 193348's "Sawtooth piggyback:
Density higher than desired but good shot." names an experiment, not a
crash. Both are true rows and neither is an observation of the phenomenon.
Because a text row inherits the shot's span - `[0, PULSE-LENGTH)`, or a
point at zero when the bundle has no length - an untimed mention lands at
shot start, which is where the iteration-0 critic found both of them being
counted as events at `t = 0`.

`events/windows.py` therefore selects the rows it reduces into the 46
`phenomenon_window_features` by an explicit policy, not by `phenomenon`:

* `DIAGNOSTIC_EVIDENCE` - the row's `evidence_kind` must be `detector` or
  `heuristic`;
* `FAMILY_SOURCES` - and its `source` must be one the family's own detector
  writes: `coherent_mode` and `pickup` from `tokeye_track`, `elm` from
  `elm_clock`, `elm_free` from `elm_clock`, `sawtooth` from
  `ece_sawtooth`, `lh_transition` from `dalpha_lh`.

Both, together, applied once before anything is clustered or unioned. Text
and forecasts stay in the file - retrieval and the text channel want them -
and reach none of the 46 numbers; the served `FeatureArray` stamps the
policy into its `attrs` so a stored feature can be re-checked when the
policy changes.

### Coverage: no coverage is not absence

Every row carries `t_cov0_s`/`t_cov1_s`, the span of the thing the row was
measured on. Three rules:

* **Per source and per quantity.** An `nbi_on` row's coverage is the NBI
  digitiser's, a `gas_on` row's is the gas recorder's, and `diag` on an
  actuator row names which. They are not the same span: on shot 198658 the
  gas axis runs -10 to 94.86 s and the NBI's stops at 13.10 s.
* **Over finite samples.** A record's axis runs past its samples - the fast
  groups end in NaN, a filterscope's head and tail are NaN - and padding is
  not observation. A multi-input heuristic gets the INTERSECTION of the
  inputs it needs at once: the L-H detector covers only where the D-alpha,
  the line density and the injected power were all measured.
* **NaN means unknown, not zero.** `(NaN, NaN)` is "nobody looked", which is
  a third answer beside "looked and saw nothing" and "looked and saw
  something". A consumer that reads it as an empty interval, or as an
  infinite one, is wrong in a direction nothing downstream can detect.

An event's extent is clipped into its own coverage at the point the row is
built, with `attrs["clipped"] = true` where it had to be: a track stitched
across tile boundaries carries the transform's edge support and ran up to
2 ms past the record on the pilot shots. `schema.Event` refuses a row whose
`t1_s` is after its own `t_cov1_s`, so the invariant is enforced rather
than repaired. A POINT event the detector's own grid or gate window put past
the bound - an ELM found on the transform's edge column, an L→H transition
within 5 ms of the beam record's end - is written at the bound with
`attrs["clipped"] = true` and the measured column or instant kept in
`attrs` (`col`, `t_measured_s`): the pipeline isolates failures per step,
so refusing the one row would cost the shot its whole ELM clock or its
every L→H claim.

### `events/<shot>_sources.parquet`: did anybody look

An events file says what was FOUND. A detector that ran and found nothing
writes no row to it, so "no ELMs on this shot" and "nobody has run the ELM
clock on this shot" are the same empty query. The sources file is the
missing half - one row per `(source, diag, channel, pass_name)` that ran or
was skipped:

| column | meaning |
|---|---|
| `shot`, `source`, `diag`, `channel`, `pass_name` | which producer, on which block or quantity |
| `status` | `ok` (it ran to completion), `skipped` (it did not), `error` |
| `reason` | why, for a skip; `""` when `ok` |
| `t_cov0_s`, `t_cov1_s` | what it ran over; NaN when unknown |
| `n_events` | how many rows it put in the events file - **0 is a real answer** |
| `run_id`, `git_sha`, `written_at` | which run wrote this row |

So: `status == "ok"` with `n_events == 0` is observed silence, `status ==
"skipped"` with a reason is the absence of an observation, and NO ROW AT
ALL is "not processed". Written merged and atomically on the same key, so a
re-run of one channel replaces that channel's rows and leaves the rest.

Three things worth knowing before reading a row:

- **A skip is ordinary.** `bes` is absent on 67% of shots and `mirnov` is not
  planned where `mhr` is present, so every shot's row carries a `skipped` dict
  saying which channel or step did not run and why. It is the answer to "was
  there no ELM here, or did nobody look" - and so is
  `events/<shot>_sources.parquet`, per source, on disk.
- **The corpus has no `ip` and no `betan`.** So `actuator` never claims
  `nbi_counter` (it is a comparison of the injected torque's sign with the
  current's), `qh_proxy` has no flat-top to intersect and claims nothing, and
  an L->H row's `attrs["betan"]` is `null`. All three are recorded as skips
  rather than left looking like an absence of the phenomenon.
- **Text is never a label by itself, and never a detection.** A `text` row's
  confidence is capped at `TEXT_ONLY_CEILING`, it is excluded from every
  diagnostic feature by the evidence policy above, and only the shot's OWN
  logbook entries are read - the session context is the run's, not this
  shot's. The shot-scope text comes
  from a subset of the logbook dump that the stage builds once per run;
  `--refresh-text` re-asks about shots a previous run found no record for
  (deleting `text/logs_subset.missing` forgets all of them).

### Curated label tables

Curated lists use the user's `data/events/<category>/{raw,format,extend_<model>}/`
layout. `raw/` holds byte-identical originals; deterministic adapters in
`scripts/labelmaker/labels_format.py` use `tables.yaml` to convert them into
`format/`. `events/databases.py` reads only the common format CSVs, with parsed JSON
attributes and the stored evidence kind. Its `FORMAT_COLUMNS` and validator define
one schema for every dataset and producer: `shot, t0_s, t1_s, phenomenon,
evidence_kind, source, confidence, attrs`. Every CSV has a `.meta.json` sidecar
with schema version, input SHA-256 or producer/run/git provenance, writer,
creation time, row count, and shot count.

Add an untouched raw file, declare its columns and format stem in the manifest,
register an adapter if needed, run `PYTHONPATH=src python
scripts/labelmaker/labels_format.py`, and commit raw, format CSV, sidecar, and
converter changes together. Manifest `made_at` fixes the conversion revision time;
sorted rows, stable float/JSON formatting, and that timestamp make the CSV and
sidecar byte-reproducible. See [data/events/README.md](../data/events/README.md) for
the complete schema and examples. `config.Paths.label_tables` is the root;
`LABELMAKER_LABEL_TABLES` overrides it.

Each `extend_<model>/` belongs to one producer task and holds that producer's result
on the 500 `recommender_v1` shots. `scripts/labelmaker/labels_extend.py` reads
per-shot events and source records without modifying them. Specify `--category`,
`--producer` (source or phenomenon), `--shot-list`, optional `--events-root`, and
`--out data/events/<category>/extend_<source>/recommender_v1.csv`. Phenomenon
selectors spanning sources are refused; even a single-source phenomenon export
uses the actual source's directory. An empty scan with no producing source may
use `extend_<phenomenon>/`. The common
schema preserves evidence kinds and source identities. Above **50,000** selected
rows, the writer instead emits `recommender_v1.summary.csv` with
`shot, n_events, t_first_s, t_last_s, t_cov0_s, t_cov1_s`, and metadata pointing to
the full `$LABELMAKER_ROOT/events` products. Summary coverage gives bounds, not
continuous coverage; missing files and source statuses remain visible. A producer
rerun regenerates its directory's table and removes a stale full/summary alternate.
Categories without a producer have no `extend_*` directory.

**A listing is not a coverage claim**: every row a table produces has
`t_cov0_s = t_cov1_s = NaN` and `confidence = NaN`, and a shot no table names
gets no event row *and* no source record — so a shot's absence from a curated
list is never a negative, and nothing downstream may read it as one.

A shot a table *does* name gets one row in `events/<shot>_sources.parquet` per
naming table, on the same contract as every detector's: `status = "ok"`,
`reason = ""` (an `ok` row carries no reason), `n_events`, `diag = ""`,
`channel = -1`, `pass_name = ""` — and coverage NaN, which is the whole
signal. It is unambiguous because no detector writes `ok` with NaN coverage,
so `ok` + NaN coverage + a `source` beginning `database:` *is* the curated
source. The sentence itself lives in `databases.COVERAGE_REASON` for the docs
to quote, not in the row.

The events stage reads the tables as one more guarded step, but the usual way
to ingest one is the standalone mode, which needs no corpus file, no U-Net and
no GPU:

```bash
pixi run -e labelmaker python -m labelmaker.run events --databases-only \
    --shot-file $LABELMAKER_ROOT/recommender_v1.txt
```

It prints `N of M shots are named by any table`. Zero is a normal answer and
exits 0: the two RWM tables span 156785–176092, the corpus starts at 185601,
and `0 of 500 shots are named by any table` is what an honest run says. The
committed `resistive_wall_mode/extend_rwm/recommender_v1.csv` records exactly that
empty result, with its successful scan's run ID and root-relative paths in the
metadata. Missing shots are represented by a count, the first 20, and a pointer
to the complete `.missing_event_shots.json` sidecar beside the CSV.
A `--run-id` supplied to the extension writer must identify a successful zero scan
of the selected curated tables. The two committed format tables still yield 56
events and 33 source rows on their original 33 shots. An invalid
`tables.yaml` or an unreadable or invalid format CSV stops the run before any
shot with `EXIT_BAD_LABEL_TABLE` (exit code 7). All tables are validated before
any per-shot output is written.

The cost is per *named* shot, not per shot in the list: a shot no table names
is a dictionary lookup, and a shot one does names costs ~0.11 s (measured:
write the events, read them back, append the index, write the sources row).
The 33 RWM shots take about 4 s and `recommender_v1`'s 500 unnamed shots about
1.7 s; a table naming all 16,909 corpus shots would take roughly half an hour,
serially — `databases_stage` ignores `--workers`, which at this cost is a
choice and not an oversight.

Curated rows can never become window features. `windows.py` applies **both**
halves of the evidence policy (`DIAGNOSTIC_EVIDENCE` *and* `FAMILY_SOURCES`,
above) in `diagnostic_mask`, and `database` is not a diagnostic evidence kind —
so a curated row is excluded even when its `phenomenon` is one the windows do
count and even if its `source` were one a family allows.
`test_a_database_row_inside_the_window_changes_no_feature` pins this with three
rows, one per half of the policy plus the realistic case, so the test fails if
either filter is dropped.


## Running the events job

Use the three L12 scripts in order: a CPU pre-pass over the **whole** shot list,
one GPU per array element, then a single dependent index rebuild and utilisation
gate. The array reads the text subset with `--text-subset readonly`, writes
per-shot products with `--no-index`, and binds the parent plus prep/tail workers
to its allocated CPUs. `PREFETCH` must be at least `PREP_WORKERS`, and request
exactly `PREP_WORKERS + 2` CPUs. Both scripts and the gate import `$REPO/src`;
GPU inference uses the phase3 Python, while CPU commands and jobstats use the
main checkout's labelmaker pixi environment.

The pre-pass also stages the site's unchanged `jobstats` client and its support
files in a private `runs/slurm/jobstats-client/` directory. Compute nodes do not
have `/usr/local/bin/jobstats`; the companion puts this shared copy on `PATH`.
Run the pre-pass from a login node where the site command is installed.

For a first, at-most-20-shot pilot (these commands submit work):

```bash
cd /scratch/gpfs/nc1514/FusionAIHub-L
export REPO=$PWD
export LABELMAKER_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker
ROOT=$LABELMAKER_ROOT
mkdir -p "$ROOT/runs/slurm"
head -20 "$ROOT/recommender_v1.txt" > "$ROOT/runs/slurm/pilot20.txt"

# 1. Once for all 500 shots, on the login node; no GPU allocation.
SHOT_FILE="$ROOT/recommender_v1.txt" bash scripts/labelmaker/tokeye_text_subset.sh

# 2. One array element. Defaults and their measured basis are atop the sbatch.
JOBID=$(SHOT_FILE="$ROOT/runs/slurm/pilot20.txt" N_CHUNKS=1 \
  sbatch --parsable --array=0-0%1 scripts/labelmaker/tokeye_masks.sbatch)
# Poll squeue until this array has left the queue before inspecting its gate.
while [[ -n $(squeue -h -j "$JOBID" -o %i) ]]; do sleep 20; done

# 3. Once the array has finished, rebuild then gate; afterok covers every element.
CHECKID=$(sbatch --parsable --dependency=afterok:"$JOBID" \
  scripts/labelmaker/tokeye_masks_afterok.sbatch "$JOBID" --pilot)
while [[ -n $(squeue -h -j "$CHECKID" -o %i) ]]; do sleep 20; done
# Once CHECKID has left squeue, also gate the CPU companion (GPU metrics are N/A).
PYTHONPATH="$REPO/src" pixi run --manifest-path \
  /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker \
  python -m labelmaker.jobstats --job-id "$CHECKID" --pilot \
  --wait-for-data 300 --preserve-dir "$ROOT/runs/slurm" \
  --out "$ROOT/runs/slurm/jobstats.json"
```

`tokeye_masks_afterok.sbatch` invokes `jobstats_check.py` with the GPU array ID,
which expands and gates every element. The equivalent manual GPU gate is
`python -m labelmaker.jobstats --job-id "$JOBID" --pilot --wait-for-data 300
--preserve-dir "$ROOT/runs/slurm" --out "$ROOT/runs/slurm/jobstats.json"`
under the same pixi environment. Read the JSON verdict and preserve raw jobstats
and sacct captures: the pilot exemption gives exit 0 even if thresholds fail.
If the array fails, `afterok` does not run: gate that failed array manually and
cancel its pending companion rather than treating the missing gate as success.

L12's write boundary permits only `masks/`, `events/`, `text/`, and `runs/`
under the data root. Its rebuild therefore uses
`--index-out "$ROOT/events/events_index.parquet"`; it **does not refresh** the
canonical `$ROOT/events_index.parquet` read by existing consumers. Publishing
that canonical index requires a later task with the appropriate write scope.
Without `--index-out`, the driver keeps its existing canonical output path.

The stop rule is binding: **L12 submits no production job.** Production requires
CPU, CPU-memory, GPU and GPU-memory each ≥70%; CPU-only jobs have two applicable
gates. A ≤20-shot pilot is exempt but fully reported. Diagnose any miss from
`prep_wait_s`, `infer_s`, `describe_s`, `finish_s`, and `tail_wait_s`; permit at
most one additional ≤20-shot pilot when a knob change is indicated. Record both
memory measurements (sampled jobstats and sacct MaxRSS), parent and per-PID worker
RSS, tiles/s, every shot's wall time and status, `text_subset_missing`, and counts
from all `*_sources.parquet` files. File existence alone is insufficient evidence
of completion. The measured L12 report and exact **unsubmitted** production
commands are in `.superpowers/sdd/task-L12-report.md`.

### Rule labels from the features store

Two of the quantities the events stage needs are not corpus groups at all.
`ip` is archive- and fdp-served and `qmin` is fdp-only
(`\efit01::top.results.aeqdsk:qmin`), so both come out of the **features
store** — `$LABELMAKER_ROOT/features/<shot>_features.h5`, written by the
`features` stage — which the events stage reads through
`features/store.read_feature`, opening the file separately for each quantity.
Until it did, three things were dead: `nbi_counter` was
a recorded skip on 100 % of shots, `qh_proxy` intersected an EHO with an
empty flat-top and claimed nothing, and there was no q-min label at all.

**The Ip flat-top** (`heuristics.ip_flattop`) is the longest contiguous
stretch whose |Ip| is over 90 % of the record's own peak, as `(first
sample, last sample)`. Magnitude and not sign, because DIII-D runs both
current directions; ties go to the earlier stretch. It is a *gate*, not a
precision measurement of flat-top boundaries: a 100 ms boundary error spans
five 20 ms q-min intervals, while a missing flat-top prevents the rule from
running. `actuator_intervals` gets the same `ip`
and can finally compare the injected torque's sign with the current's,
which is what `nbi_counter` is.

**The q-min bins** (`heuristics.qmin_regimes`) are exclusive and are the
label sheet's own: `qmin_hybrid` 0.95 < q ≤ 1.5, `qmin_elevated`
1.5 < q ≤ 2, `qmin_high` q > 2. A band is claimed over a contiguous run of
q-min samples that lies inside the flat-top and lasts **at least 500 ms**
(`>=`, edge to edge on the sample times — 25 intervals between 26 samples
at a 20 ms cadence). The comparison has no floating-point tolerance.
A sample at 1.5 or 2 belongs to the band *below* it; 0.95 belongs to none. A
non-finite sample belongs to none, which splits a band around a dropout
rather than interpolating over it.

**The gate is the rule.** MEASURED over the 500 shots of `recommender_v1`:
gated, hybrid 271 / elevated 60 / high 50 shots. The ungated condition
`qmin > 0.95` for at least 500 ms anywhere in the record fires on
**497 of 500 (99.4 %)**; it does not require staying in one of the three
bands. Including the current ramps therefore makes that condition nearly
universal on this list. `scripts/labelmaker/qmin_regime_census.py` measured it, and
it writes `tests/labelmaker/data/qmin_regimes_recommender_v1.json`, which
the suite reads on every run — so the gate is defended by a failing test
rather than by a comment.

**`confidence` is NaN, deliberately.** A threshold on a reconstructed
scalar has no calibrated probability behind it, and a 1.0 would let a
ranker read a rule as a perfectly-confident detector. `attrs` carries
`qmin_min`, `qmin_max`, the `thresholds` that produced the row, and
`efit: "efit01"` — the last because the sheet asks for EFIT02 or CAKE and
EFIT01 is what the features store holds today. Changing the equilibrium
requires updating the namespace resolver/locator, refreshing the stored
feature, and updating `heuristics.QMIN_EFIT`: the event attribute currently
comes from that constant, not from the feature's metadata.

The coverage of a `qmin_rule` row is the intersection of the flat-top and
the span from the first to last finite q-min sample. This single span does
not represent internal dropouts; those still split event bands. Outside
the flat-top the rule
deliberately does not look, and declaring the ramp as covered would turn an
abstention into an observed absence. A shot whose flat-top holds no band at
all writes no event row and still writes its `qmin_rule` source row with
`n_events = 0`, meaning no band met the duration requirement. This does not
prove q-min stayed below 0.95: short excursions also produce no event.
A shot with no features
file is `skipped["features"]`; one whose file carries no `qmin` is
`skipped["qmin"]` and still gets its `ip`-dependent steps in the full stage.
The quantity source rows are `("features", "ip")` and
`("features", "qmin")`, each with its own finite span; the rule's row is
`("qmin_rule", "qmin")`. A missing file instead produces a skipped
`("features", "")` row. In the full stage, missing counter-injection
inputs produce a skipped `("actuator", "tinj_total")` row, and missing
QH inputs produce a skipped `qh_proxy` row. These steps get no successful
`ran` row when an input is missing; skipped rows have NaN coverage and a
path-free reason.

Because none of this needs the corpus or the network, there is a standalone
mode for it, beside `--databases-only` and running the curated tables too:

```bash
pixi run -e labelmaker python -m labelmaker.run events --rules-only \
    --shot-file $LABELMAKER_ROOT/recommender_v1.txt
```

For this list it prints `shots with a q-min band: qmin_elevated=60,
qmin_high=50, qmin_hybrid=271`. It writes rule/table events to
`events/<shot>_events.parquet`, their completion records to
`events/<shot>_sources.parquet`, and `events_index.parquet`, through the
same writers as the full stage. Only evaluated sources or attempted steps
that were skipped are recorded: `--rules-only` does not evaluate
`nbi_counter` or `qh_proxy` and writes no source rows for them. The 500-shot
run takes minutes on a login node, depending on storage latency.

`--root` moves **both** ends: the features it reads and the events it
writes are both under it, so a scratch root has to be given the store to
read. To write somewhere disposable while reading the real store, point
its `features/` at the real one and leave everything else under the
scratch root:

```bash
mkdir -p /tmp/ld2/rules500
ln -s "$LABELMAKER_ROOT/features" /tmp/ld2/rules500/features
pixi run -e labelmaker python -m labelmaker.run events --rules-only \
    --shot-file $LABELMAKER_ROOT/recommender_v1.txt --root /tmp/ld2/rules500
```


## Reading a label

```python
import h5py

with h5py.File(".../labels/190000_labels.h5") as f:
    g = f["d3d_tearing_onset_cnn1d/tm_prob"]
    t, p = g["xdata"][:], g["ydata"][0]          # seconds, probability
    valid = f["d3d_tearing_onset_cnn1d/tm_prob_valid/ydata"][0].astype(bool)
    print(g.attrs["card_id"], g.attrs["task"], g.attrs["artifact_sha256"][:12])
```

## How reliable are the labels

`python -m labelmaker.run validate` writes four reports for models with archived truth and folds the headline
numbers into the model card's `model-index`, so the card is the one place to
read how a model performed. For `d3d_tearing_onset_cnn1d`, measured on the
500-shot pool (486 aligned to their archived training rows - the 11 tearing-mode
shots of 2024 have no archived truth and 3 shots have archived rows our
reconstruction does not cover; 31,257 published rows of 35,776 matched):

1. **Adapter fidelity** - our torch evaluation of the Keras-2 graphs against
   real-Keras outputs frozen for the same weight files. Passes all four gates;
   the published `tm_prob` differs from TensorFlow's by at most 1.2e-6.
2. **Reconstruction fidelity** - our features against the model's own training
   inputs, feature by feature. Archive-served columns are exact; `bt`/`ip`
   through fdp agree to 1e-5; offline EFIT01 standing in for real-time EFIT
   costs 2.1e-3 (R0), 2.8e-3 (kappa), 3.0e-2 (1/qpsi); ZIPFIT standing in for
   the pipeline's own profile fits costs 6.0e-2 (ne), 1.24e-1 (Te), 1.24e-1
   (rotation) median relative difference.
3. **Label quality** - against the archived labels, scored twice over the same
   rows: with the training inputs (the model's ceiling) and with our
   reconstruction (what labelmaker publishes).
4. **Alarm quality** (`alarm_quality.json`) - final-label and any-row shot
   FPR/FNR, warning times, jumps, and per-horizon IPCW AUC alongside plain
   AUROC. Available for slugs with archived truth; survival uses pre-onset
   valid rows, while column labels use all valid aligned rows. Every metric
   appears three times, under `labels.<name>.subsets.{all,held_out,in_training}`,
   split by the adapter's own `training_shots` - a model that records none
   reports every shot as held out.

| `tm_prob` | AUROC | F1 at 0.5 | precision | recall | best F1 (at) |
|---|---|---|---|---|---|
| archived inputs | 0.932 | 0.521 | 0.373 | 0.862 | 0.576 (0.76) |
| reconstructed inputs | 0.897 | 0.486 | 0.403 | 0.613 | 0.490 (0.60) |

The gap - AUROC -0.034, best F1 -0.086, `betan` RMSE 0.120 -> 0.157 - is the
reconstruction penalty, and the number that answers "is this reliable". The
ceiling is scored on rows the model trained on, so read it as optimistic; the
honest reading is the gap. Best F1 is reported because the AUROC gap
understates the cost: the reconstruction loses positives the model had placed
confidently, which a rank statistic barely registers.

F1 at 0.5 is low for a reason that is not the reconstruction: the model was
trained oversampled and class-weighted (`mse_bin_os_w`), so at 0.5 it flags
about twice the base rate of rows (reliability bin 0.5-0.6 observes 8%). Its
ranking is fine; the threshold is the operator's choice, and `analyze` takes
one in its config.

## Adding a model

1. `src/labelmaker/models/<device>_<phenomenon>_<predicted>_<arch>/`, with
   `README.md` (HuggingFace card plus the `labelmaker:` block), `spec.py`
   (`ADAPTER`), and an empty `__init__.py`. Copy the closest existing folder.
2. Map each trained-on input name to a canonical feature in
   `features/namespace.py`. Add a `FeatureSpec` only if the quantity is
   genuinely new, and give it every source that can serve it, cheapest first.
3. Put the training-time filter in as `DomainRule`s. Labelmaker flags rows
   outside them instead of dropping them.
4. Load the weights from their upstream location, copy them into
   `<root>/models/<slug>/`, and record the sha256 in the card - inference
   refuses to run against bytes the card does not know.
5. `pixi run -e labelmaker python -m pytest tests/labelmaker -q -W error` - the
   registry tests check that the card and `spec.py` agree - and
   `pixi run -e labelmaker ruff check src/labelmaker tests/labelmaker`.

## Known limits

- **Shot-scope text is thin, and a hit is not an assertion.** Measured over
  the 500 shots of `recommender_v1.txt`, a shot's OWN logbook entries name a
  sawtooth on 3 of them (0.6%) - a clear MISS against the >=5% condition the
  plan set. The run-scope session text would give 14.6%, and is a fact about
  the session rather than about the shot, so it is not substituted. Separately,
  a lexicon hit is a mention and not a claim that the phenomenon occurred:
  shot 185980's "Updated ELM detector tuning." is a positive ELM hit and is
  about the detector. This is why `text` rows are capped at
  `TEXT_ONLY_CEILING` and excluded from every diagnostic feature.
- The archive resolver covers 3,246 of 16,909 corpus shots (19%). Everything
  else goes through fdp, which is built, measured against the archive on
  random overlap shots, and exercised on 69 of the 100 proof-of-concept shots -
  but has not been run at corpus scale (measured 3.5 s per shot median at 8
  workers, so the corpus is hours, not days).
- The three kinetic profiles are ZIPFIT fits standing in for the tearing model's
  own mtanh/csaps fits - the dominant reconstruction error, priced above. The
  rotation profile is absent on ~22% of shots, and a shot missing any input for
  its whole record has no valid rows: 5 of the 100 proof-of-concept shots.
  Fitting the profiles from raw Thomson and CER needs channel geometry the
  corpus does not carry, and is Phase 2.
- ECH is on hold. The corpus has no deposition-location group, its
  `ech_power` sum under-reports the archive's `EC.PECH` by ~22% at the median,
  and whether fdp can serve a deposition location has not been investigated;
  outside the archive, any row with ECH power flowing is invalid. This is what
  decides the 2024 tearing-mode shots 199597-199607: ECH is on for ~90% of
  their plasma rows, so they come out 4-15% valid, even though `tm_prob` peaks
  above 0.9 on 9 of the 11 (`analyze` shows the members disagree widely on
  them too). Until a deposition-location source exists, their labels are
  extrapolations by the card's own definition.
- Three of the 500 validated shots (190859, 190861, 190863) are rejected by the
  row matcher: 4-12 of their archived rows have EFIT geometry that appears
  nowhere in our EFIT01 series, so the archive covers times our reconstruction
  does not. Refusing the whole shot is the conservative choice; a per-row
  refusal is the planned fix.
- ZIPFIT rotation on some 2024 shots reaches 1,200-2,000 in units the model
  reads as km/s; the `|rotation| < 150` domain rule catches those rows, but
  whether that is a units change or a fit blow-up has not been checked.
- The label-quality numbers come from shots the archive holds, because only
  there is there an archived truth to score against; they are a measurement of
  the reconstruction, not of the model on unseen shots.
- Two models are implemented: the tearing-onset CNN and the tearing
  time-to-event survival model (`d3d_tearing_time_to_event_dsm`, onset risk at
  250 ms, 500 ms and 1 s). The survival model's labels are produced but not yet
  scored by `validate`, whose reports are specific to the CNN's training
  archive; its adapter fidelity is a test against the upstream fork's own
  outputs. The other five roster folders are scaffolds whose cards say what
  blocks each of them; `docs/superpowers/specs/2026-09-05-labelmaker-phase2-design.md`
  records what the upstream archaeology found for each.
- The survival model's calibration depends on which rows a report covers, and
  that is a property of its training population, not a defect: over all aligned
  shots' pre-onset rows (5.5% positive at 1 s) it is calibrated, ECE 0.022;
  restricted to shots that do get an onset (54% positive) it is under-confident,
  ECE 0.448. It was fit on rows that are 85% censored with a median 1.92 s to
  event. Every figure and every report states its row set for this reason.
- Roughly half the 500-shot pool is in the survival models' own training set:
  214 of the 500 shots, 208 of the 463 scored shots and 41 of the 80 shots with
  an archived onset are among the 8,923 shots of
  `models/d3d_tearing_time_to_event_dsm/training_shots.txt`, which both survival
  checkpoints share. `alarm_quality` and `calibration_study` therefore report
  every metric on `all`, `held_out` and `in_training`, and the published
  isotonic maps are fitted on held-out shots only; a number labelled `all` is
  still part in-sample. Measured, the contamination did not flatter the shipped
  checkpoint - its held-out AUROC is 0.839 / 0.817 / 0.801 against 0.761 /
  0.738 / 0.696 in-sample - because the two subsets are different populations
  (19.7% of in-training pool shots tear, against 15.3% of held-out ones). The
  tearing CNN is a separate case and not covered by this split: the archive it
  is scored against IS its training store, as its card says.
- `docs/superpowers/specs/2026-09-05-labelmaker-phase3-design.md` carries the
  next round: what the tearing-survival, ELM and Alfven-eigenmode training loops
  upstream actually do (measured, with the shipped survival model's
  hyperparameters decoded from its own pickle), the three agreed reliability
  fixes, the uncertainty series to publish from the survival mixture, and the
  requirements for `d3d_ae_activity_seldnet` - the one model labelmaker will
  train itself.
