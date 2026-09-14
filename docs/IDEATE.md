# ideate

A shot database and retrieval layer for DIII-D: build a queryable record of a set of discharges
from the local raw stores, then find shots by what happened in them — a description, a reference
shot, a set of conditions — and read back what the diagnostics, the labels and the operators say.
It exists to answer "has DIII-D done anything like this, and what happened?" before an experiment
is proposed, so `ideate` is the retrieval half of the recommender: no model is trained here.

The source is `src/ideate/`; the tests are `tests/ideate/`. Two pixi environments run it,
`ideate` (CUDA) and `ideate-cpu` (identical but for the torch wheel — only `design_rollout` needs
a GPU). Both set `IDEATE_DATA_ROOT`, which is where the built database lives.

## The command line

```bash
pixi run -e ideate-cpu ideate <command>          # or: python -m ideate <command>
```

| command | what it does |
| --- | --- |
| `build` | full rebuild of the database from a shot list (atomic: writes `db.tmp`, swaps) |
| `add` | incremental upsert of one or more shots |
| `query` | find similar shots — `--ref SHOT`, `--text`, `--where`, `--actuator` |
| `show` | print one shot's record |
| `coverage` | present/unavailable/pending per registry field |
| `corpus` | the FAITH corpus: what each shot file carries; `corpus select` draws the shot list |
| `logs` | the shot-log contract (`logs missing`, `logs import`) |
| `labels` | labelmaker's labels and events → `labels_wide.parquet`, `events.parquet` |
| `phenomenon` | which shots show a phenomenon, and what kind of evidence says so (`--list` prints the registry) |
| `export` | `ShotSummary` rows as JSON or Parquet |
| `actuation`, `blurb`, `llm`, `model` | actuator waveform sets, per-shot blurbs, LLM reachability, the IGNITE bundle |

`ideate <command> --help` is authoritative; the table is a map, not a specification.

The development universe is `configs/ideate/shot_lists/recommender_v1.yaml` — 500 shots drawn by
`ideate corpus select` under the rule in `src/ideate/shotdb/select.py`'s module docstring. The
database built from it is what the MCP server below serves.

## Phenomena: the evidence classes and the caveat vocabulary

```bash
ideate phenomenon "edge harmonic oscillation" --n 20            # ranked shots + caveats
ideate phenomenon "tearing mode" --avoid phenomenon:elm --json  # the hits as JSON
ideate phenomenon --list                                        # the registry
```

`ideate phenomenon` answers "which shots had one of these?" — and the answer is only usable
because it says *who claims so*. Four classes of evidence can name a phenomenon on a shot, and
they are not interchangeable:

| class | where it comes from | what it is |
| --- | --- | --- |
| **observed** | `events.parquet` rows whose `evidence_kind` is `detector` or `heuristic` — a `tokeye_track` in the right band, an `ece_sawtooth` crash, a `dalpha_lh` transition | a diagnostic showed it. The only class that is a measurement, and the only one that can rule the phenomenon *out* — inside the window the diagnostic covered. An allow-list, not "anything that is not a forecast": a `text` or `model` row that matches a registry rule is counted as neither, and says so |
| **label** | `labels_wide.max_valid` for a detection head named in the registry, **at or above its floor** | a model's opinion about the present. `None`, never `0`, where the model was not run or had no valid samples. A score *below* the floor (`thr` from the model card, else `retrieval.yaml`'s `label_floor`, 0.5) is reported with a caveat naming it and is not evidence: "the model ran" and "the model said yes" are different facts, and a quarter of this database's scored tearing shots are at or under 0.010 |
| **forecast** | `events.parquet` rows with `evidence_kind == "forecast"` — a risk curve crossed at a threshold, with a `horizon_s` | a model's estimate of what was about to happen. It arrives in `PhenomenonHit.forecasts`, never in `intervals`, and it is never reported as an observation |
| **text** | `text_claims.parquet`, `polarity=pos` and `temporality=observed` | an operator wrote it down. Never a label by itself: a hit resting on nothing else is capped at 0.25 and carries **TEXT ONLY** |

Ranking is the class order first and the score second: an observed hit outranks a label-only hit,
which outranks a forecast-only hit, which outranks a curated-list hit, which outranks a text-only
hit, whatever the weights in `configs/ideate/retrieval.yaml` say. **On the `recommender_v1`
database today every one of the 1,037 event rows is a forecast**, so `ideate phenomenon` returns
forecast-, label- and text-class hits and no observed ones at all until the production masks land
and `ideate labels join` ingests labelmaker's detector output — and the command prints that fact
under its `resolved:` line rather than leaving it here.

### Coverage: the four states

`coverage` answers "did anyone look, over the window I asked about?", and `coverage_state` says
which of four situations a `null` is. Only the last makes an empty `intervals` a negative:

| `coverage_state` | what it means | `--avoid` |
| --- | --- | --- |
| `unindexed` | nothing in the pipeline looks for this phenomenon at all (`rwm`, `detachment`) | keeps the shot, with a caveat |
| `unprocessed` | its detectors exist and none of them has run on this shot | keeps the shot, with a caveat |
| `uncovered` | they ran, but elsewhere in the record — not over the segment searched | keeps the shot, with a caveat |
| `observed` | they covered the window, or part of it; `coverage` is that intersection | drops only on an observed interval; caveats anything short of full cover |

Coverage is **clipped to the window searched** before any of this is decided. On labelmaker's own
shots the ELM clock's coverage ends at ~4.3 s while other detectors run to 6-7 s, so a flat top
extending past 4.3 s is partly unlooked-at for ELMs — and a `coverage` that reported the clock's
own window would have made that shot a clean ELM negative. `coverage_windows` is the real union
(gaps and all) and `coverage` is its hull; a hull spanning a gap says so in a caveat.

### The ELM case: a transient detector is not an ELM detector

`elm` now resolves observed `elm_clock` point events with `evidence_kind="heuristic"`.
The clock reads D-alpha filterscopes 0-7 and publishes a point-event family beside
its `elm_free` intervals. These points are measured peak-picking results, not a
trained ELM classifier; manual validation remains a follow-up. The rule has weight
1.0 and does not carry the class-agnostic transient caveat.

`tokeye_transient` publishes `phenomenon="transient"`, with its own lexicon and
registry entry. It cannot count as ELM evidence, including legacy rows whose
phenomenon is still `elm`. Re-run labelmaker's events stage and the labels join to
publish new products; a code change alone cannot populate the database. L-A's
read-only census found zero observed events on the 500 recommender shots, and
its CPU demonstrations under `/tmp` do not change that production count.

The registry is `configs/ideate/phenomena.yaml` — one entry per phenomenon, saying which
`labels_wide` series, which event sources, which frequency band and which descriptors count.
It states **no aliases**: the ids and the phrases that name them live in
`src/labelmaker/events/lexicons.yaml`, which both packages read, and an `aliases:` key in
ideate's file is an error rather than a second vocabulary.

### The caveat vocabulary

Every hit carries `caveats`, and a hit with none is a hit that lacks nothing — which is what
makes the others mean something. They are constants in `ideate.retrieval.phenomena`, so a caller
can key on them:

| caveat | what it tells you |
| --- | --- |
| `TEXT ONLY` | nothing but operator text; the score is capped at labelmaker's `TEXT_ONLY_CEILING` |
| `ranked on forecasts: ...` | the strongest evidence is a model's estimate of what was about to happen; no diagnostic saw anything |
| `ranked on model labels: ... (<p>) ...` | the strongest evidence is a model's score, and the caveat says what the score was |
| `ranked on a curated human list: ...` | a human list names the shot and nothing else does |
| `no diagnostic coverage recorded; absence is not evidence` | `unindexed`: nothing looks for this phenomenon at all. **Not** "it did not happen" |
| `no detector for <title> has run on this shot; ...` | `unprocessed`: its detectors exist and none ran here |
| `the <title> detectors ran on this shot but not over the <segment> window; ...` | `uncovered`: they looked somewhere else in the record |
| `the <title> detectors covered only <windows> of the <segment> window; ...` | partial cover: outside those stretches, absence is unmeasured |
| `the <title> coverage of the <segment> window has <n> gap(s): ...` | `coverage` is a hull over stretches nobody read; `coverage_windows` has the union |
| `observed via tokeye_transient, a class-agnostic transient detector: ...` | retained caveat vocabulary for custom/legacy rules; the shipped ELM rule uses the D-alpha clock |
| `<n> row(s) of evidence_kind <kinds> match this phenomenon's rules and are counted as neither observation nor forecast` | a `text`/`model`/`human` row matched a rule; it is not a sighting |
| `<key> scored <p>, below the <floor> evidence floor: ...` | the model ran and said no; the number is reported and does not count |
| `the quote is this shot's most informative logbook entry and does not mention <title>` | the quotation beside the hit is not the reason for the hit |
| `no observed evidence: ...` | no detector claims it on this shot |
| `no label evidence: ...` | no model in the registry emits a label for this phenomenon at all |
| `label not run on this shot, or no valid samples: unavailable, not 0` | the label row is missing or `n_valid == 0` |
| `no operator text names this phenomenon on this shot` | the logbook is silent |
| `text evidence is run scope: ...` | the sentence is the session's, shared by every shot of the run. Measured over 500 shots: `elm` fires on 99.4 % of shots at run scope and 11.0 % at shot scope |
| `operator log says NOT <title>` | somebody wrote that it was absent |
| `no <segment> segment on this shot; the whole record was searched` | the window was widened, and you are told |
| `kept despite --avoid <token>: ...` | the shot survived an `--avoid` filter because the avoided phenomenon's coverage is not a full cover of the window — one variant per coverage state, and no caveat at all in the one case that is a real negative |
| `<n> event(s) not shown: ...` | events the source never scored, dropped by `--min-confidence` |

`--avoid phenomenon:elm` drops the shots an ELM detector fired on and **keeps**, with the caveat
for that shot's coverage state, the shots no ELM detector covered. Dropping those would read a gap in the diagnostic
coverage as a physics result. The clock's source record states whether it ran and
over which filterscope span, even when it found no points. Its `elm_free` intervals
also provide coverage evidence through `coverage_sources`; a missing clock remains
unknown. A phenomenon without a detector declares no diagnostic coverage.

One field name is a promise it cannot yet keep: `actuators_at_onset` holds the **segment's** own
summary columns (`pnbi_total_mean`, `pech_total_mean`, `gas_total_mean`, `irmp_total_peak`), not
the values at the phenomenon's first interval. A per-onset lookup needs the raw waveform, which
the database does not carry; the keys are the column names, so the field cannot misdescribe
itself in the meantime.

## The MCP server

`ideate` exposes its retrieval over the Model Context Protocol, so an assistant can search the
database directly instead of being handed a transcript of a CLI run. stdio transport:

```bash
pixi run -e ideate-cpu ideate-mcp        # == python -m ideate.mcp
```

### Attaching it in Claude Code

`.mcp.json` at the repo root is the project-scoped configuration; Claude Code reads it for any
session started in this checkout and asks once whether to trust it.

```json
{
  "mcpServers": {
    "ideate": {
      "command": "pixi",
      "args": ["run", "-e", "ideate-cpu", "python", "-m", "ideate.mcp"],
      "cwd": "/scratch/gpfs/nc1514/FusionAIHub"
    }
  }
}
```

`cwd` is absolute because the server is started by whatever directory the client happens to be
in. `IDEATE_DATA_ROOT` comes from the `ideate-cpu` environment's own activation, so the config
names no data path; add an `"env"` block to point one session at a different database.

`/mcp` in a session lists the connected servers and their tools. Outside Claude Code, any MCP
client works — the transport is stdio and the command above is the whole contract.

### The tools

| tool | arguments | returns |
| --- | --- | --- |
| `search_shots` | `text`, `ref_shot`, `segment`, `constraints`, `actuators`, `require_labels`, `avoid_labels`, `n` | ranked shots with a description, an explanation naming which channel found each one, and the operating-limit flags for the proposed actuators |
| `describe_shot` | `shot`, `segment` | the prose description and the whole stored record: segments and their scalars, labels and their source, outcome, and the operator logbook verbatim |
| `get_events` | `shot`, `phenomenon`, `t0_s`, `t1_s` | a `status` — `unindexed`, `unprocessed`, `uncovered` or `observed` — and four lists kept apart: `events` (what a diagnostic showed), `text_mentions` (a lexicon hit in the logbook), `database_intervals` (a curated table's rows), `forecasts` (a model's estimate); plus `coverage`, the per-source table of what ran over which span |

Plus one resource, `ideate://manifest`: the built database's manifest, which is how a caller
finds out which shots the tools can see at all.

Four things hold for the replies and are worth knowing before reading one:

* **`caveats` is on every reply the tools themselves produce, and it changes what the reply
  means.** An empty result carrying "no channel had anything to search on" means the query was
  empty — not that no such shot exists. A result whose constraint excluded shots for having no
  recorded value says so, because "not measured" is not "out of range". The promise is scoped:
  a call whose *arguments* fail the tool schema (a string where a shot number belongs) is
  rejected by the MCP framework before any tool code runs, and comes back as a protocol
  validation error with no `caveats` field.
* **`get_events` keeps four kinds of claim in four lists.** An `events` row is a detector's or
  a heuristic's claim about what a diagnostic showed, with `source` saying who. A `forecasts`
  row (`evidence_kind == "forecast"`) is a model's estimate of what was about to happen, raised
  from a risk curve at a threshold. A `text_mentions` row (`evidence_kind == "text"`) is a
  lexicon hit in the operator logbook — somebody wrote the word — and is *not* evidence that the
  phenomenon occurred: shot 185980's "Updated ELM detector tuning." is a positive ELM hit about
  the detector. They arrive in different lists and must stay in different sentences.
* **`get_events` keeps a curated list out of `events` too.** A row with
  `evidence_kind == "database"` comes from a table somebody sent us — the first two are Jeremy
  Hansen's RWM onset databases, under `data/events/resistive_wall_mode/` and declared in
  `data/events/tables.yaml` — and it arrives in a fourth list, `database_intervals`. It names a
  shot and a time, not a measurement: its `confidence` is `null`, because a human list has no
  calibrated probability, and its coverage (`t_cov0_s`/`t_cov1_s`) is `null`, because nobody
  recorded which interval of the shot was examined. That second null is the load-bearing one — a
  shot's **absence** from a curated list is not a negative, and nothing downstream may read it as
  one. `database` is outside `OBSERVED_KINDS`, so a shot whose only rows come from a spreadsheet
  never answers `status: observed`, and labelmaker's own `database:<stem>` source row — `ok`,
  coverage NaN — matches `event_sources.NON_DIAGNOSTIC_SOURCE_PREFIXES`, so it is not counted
  as a diagnostic having looked and cannot donate coverage either. `ideate labels join` needs no rule for these rows: `events_union` reads each
  shot's `events/<shot>_events.parquet` wholesale, so they reach `db/events.parquet` as they
  are.
* **`status` says what an empty `events` means**, and the four values are not degrees of one
  thing. `unindexed`: the shot is not in the database (the error dict `describe_shot` gives).
  `unprocessed`: it is indexed, but `db/event_sources.parquet` records no detector as having
  completed over it — absence is not evidence. `uncovered`: detectors ran, but not over the
  window asked about; the caveat names the covered span. `observed`: some one detector's *own*
  finite coverage overlaps the window, and an empty list is a real finding of nothing, said in as
  many words — and the caveat counts only the sources that covered the window, not everything
  that ran. **A source that completed without recording its coverage (`ok` with NaN `t_cov`, as
  real shot 198658's `actuator/ech_power_total` is) can neither cover nor un-cover a window: it
  keeps the shot out of `unprocessed` because it did run, it can never make a window `observed`,
  and a shot whose every completed source has unknown coverage answers `uncovered` with a caveat
  naming each one.** A `text` row, a `database:<stem>` row and an `evidence_kind="database"` row
  are not diagnostics having looked either, and none of them can make a shot `observed`. A
  reversed, zero-width or non-finite window is an error dict, never a silent empty. (On the
  `recommender_v1` database today *all* 1,037 event rows are forecasts and no shot has an
  observed-event product, so `get_events` answers `unprocessed` for every one of the 500 — which
  is the truth the old empty list hid.)

## Frame-code provenance

Each `frame_codes/<shot>.pt` has a JSON sibling saying how it was made — device, thread count,
encode clock, run manifest — because the encoder is not device-independent (185955's `bes` and
`mhr` codes differ between cuda and cpu, and between cpu at four threads and at eight). The 500
production caches predate the sidecar and were reconstructed by
`scripts/ideate/frame_codes_provenance.py --backfill` from the run manifests under `runs/encode/`.

**A backfilled sidecar carries `device`/`device_source`, `torch_threads`/`torch_threads_source`
(the `OMP_NUM_THREADS` the sbatch exports, not a measurement of the run), `encoded_at` (the cache
file's mtime), `run_manifest` where one names the shot, and `backfilled: true`. It carries NO
input fingerprint — `input_file` null and `input_fingerprint.kind` `"unknown"`, with
`size_bytes`, `mtime_ns` and `sha256` null — and null `torch_version`, `git_sha`,
`ignite_bundle`, `ignite_bundle_sha`, `ignite_revision`, `n_frames`, `modalities` and
`include_video`, because the run manifests record none of them and stat-ing the corpus file now
would describe it today rather than at encode time.** So "every cache has a sidecar" is true and
"every cache's encode is reproducible from its sidecar" is not.

`--audit` is the read-only form of that claim: it prints the census — caches, sidecars, missing
sidecars, `input_fingerprint.kind`, device, `backfilled`, and the null count per field — so a
reader checks the counts instead of trusting the sentence.

Nothing in the tools re-implements retrieval: they call `ideate.retrieval.rank` and
`ideate.retrieval.describe` over the same `ShotDB` the CLI opens, so an assistant and
`ideate query` cannot disagree about a shot. `src/ideate/mcp/tools.py` is the whole contract —
the function signatures and docstrings there *are* the tool schemas the assistant sees.
