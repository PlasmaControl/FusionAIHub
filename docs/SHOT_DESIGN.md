# shot_design

Environment settings use the `SHOT_DESIGN_*` prefix. Legacy `IDEATE_*` settings
remain fallbacks when the corresponding new name is unset; an explicitly empty
new setting still wins. A fallback emits one log line naming the replacement.
The pixi environments remain `ideate` and `ideate-cpu` because pixi rejects
underscores in environment names. Tasks and Python modules use `shot_design`.
The production data directory remains `/scratch/gpfs/EKOLEMEN/nc1514/ideate`.

A shot database and retrieval layer for DIII-D: build a queryable record of a set of discharges
from the local raw stores, then find shots by what happened in them — a description, a reference
shot, a set of conditions — and read back what the diagnostics, the labels and the operators say.
It exists to answer "has DIII-D done anything like this, and what happened?" before an experiment
is proposed, so `shot_design` is the retrieval half of the recommender: no model is trained here.

The source is `src/shot_design/`; the tests are `tests/shot_design/`. Two pixi environments run it,
`shot_design` (CUDA) and `shot_design-cpu` (identical but for the torch wheel — only `design_rollout` needs
a GPU). Both set `SHOT_DESIGN_DATA_ROOT`, which is where the built database lives.

## The command line

```bash
pixi run -e ideate-cpu shot_design <command>          # or: python -m shot_design <command>
```

| command | what it does |
| --- | --- |
| `build` | full rebuild of the database from a shot list (atomic: writes `db.tmp`, swaps) |
| `add` | incremental upsert of one or more shots |
| `query` | find similar shots — `--ref SHOT`, `--text`, `--where`, `--actuator` |
| `show` | print one shot's record |
| `describe` | describe a stored shot, with phenomenon evidence, forecast ranges, coverage states and attributed quotes |
| `coverage` | present/unavailable/pending per registry field |
| `corpus` | the FAITH corpus: what each shot file carries; `corpus select` draws the shot list |
| `logs` | the shot-log contract (`logs missing`, `logs import`) |
| `labels` | labeler's labels and events → `labels_wide.parquet`, `events.parquet` |
| `phenomenon` | which shots show a phenomenon, and what kind of evidence says so (`--list` prints the registry) |
| `eval` | the frozen evaluation harness: `eval prompts`, `eval latency`, `eval recall` |
| `export` | `ShotSummary` rows as JSON or Parquet |
| `actuation`, `blurb`, `llm`, `model` | actuator waveform sets, per-shot blurbs, LLM reachability, the IGNITE bundle |

`shot_design <command> --help` is authoritative; the table is a map, not a specification.

The development universe is `configs/shot_design/shot_lists/recommender_v1.yaml` — 500 shots drawn by
`shot_design corpus select` under the rule in `src/shot_design/shotdb/select.py`'s module docstring. The
database built from it is what the MCP server below serves.

## Scratch databases and the pixi activation env

`pixi run -e ideate` and `-e ideate-cpu` set `SHOT_DESIGN_DATA_ROOT`, `LABELER_ROOT` and
`SHOT_DESIGN_CORPUS` from `[tool.pixi.feature.shot_design.target.unix.activation.env]` in `pyproject.toml`.
Activation runs *after* your shell, so a value you exported is replaced without a word. On
2026-09-14 a one-shot scratch build, run through `pixi run` with `SHOT_DESIGN_DATA_ROOT` exported to a
`/tmp` directory, published itself over the 500-shot production database.

So a scratch build must not go through `pixi run`. Call the environment's interpreter directly,
with the variables exported:

```bash
export SHOT_DESIGN_DATA_ROOT=/tmp/scratch-db HF_HUB_OFFLINE=1
/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/ideate-cpu/bin/python -m shot_design build --shots 190000
```

or give it a paths file of its own — `SHOT_DESIGN_PATHS=<file>` — remembering that `SHOT_DESIGN_DATA_ROOT`
still wins over it, so it has to be out of the environment:

```bash
pixi run -e ideate-cpu env -u SHOT_DESIGN_DATA_ROOT SHOT_DESIGN_PATHS=/tmp/my-paths.yaml \
    python -m shot_design build --shots 190000
```

**How to check.** Every command that writes under the root — `build`, `add`, `labels join`,
`encode`, `corpus scan`, `corpus select` — prints one line on stderr before its first write:

```
shot_design build: data root /tmp/scratch-db (SHOT_DESIGN_DATA_ROOT env) -> db /tmp/scratch-db/db
```

The bracket is the source that won, and it is one of `SHOT_DESIGN_DATA_ROOT env`,
`SHOT_DESIGN_PATHS=<file>` or `<repo>/configs/shot_design/paths.yaml default` — the third names the
packaged paths file in full, because `SHOT_DESIGN_CONFIG_DIR` can move it and a label is only useful
if it names the file that was actually read. If it names a root you did not mean, stop there.

**What the guard refuses.** Before publishing, `build` compares the existing
`<db_dir>/manifest.json` with what it is about to write and refuses — nothing touched, exit 1,
a message naming both sources and both counts — when the `shot_source` differs, when the new
`n_shots` is smaller than the existing one (a `--limit` pilot, a one-shot build), or when the
existing manifest cannot be read. `shot_design build --force` overrides it and records the database
it replaced under `forced_over` in the new manifest. Growing the same selection needs no flag:
the 500 → 504 rebuild over `list:recommender_v1` publishes as it always did. `shot_design add` and
`shot_design labels join` are upserts and are not guarded.

## Search

```bash
shot_design query "edge harmonic oscillation" --n 5
shot_design query "edge harmonic oscillation" --avoid phenomenon:elm --n 5
shot_design query --ref 198658 --require phenomenon:tearing
shot_design query "tearing mode" --require label:d3d_tearing_onset_cnn1d/tm_prob
shot_design query "tearing mode" --require source:detector
shot_design describe 198658
```

Positional query text and `--text` are equivalent. The `phenomenon` channel resolves aliases
through the existing lexicon and uses the same evidence tiers and four-term score as
`shot_design phenomenon` below. Its RRF weight is **1.2**. Within this channel, observed evidence
ranks above labels, forecasts, curated lists and text-only evidence; the final search ranking
also combines the scalar, dense-text, BM25 and IGNITE channels. A query with no resolved
phenomenon contributes no vote, leaving its ranking identical without this channel.

`--require` requires every supplied token; `--avoid` rejects any supplied token present in a
segment. Existing regime and operational tokens (`QH`, `L`, `dud`, etc.) still work.

| token | presence means |
| --- | --- |
| `phenomenon:<id>` | registered detector or heuristic evidence overlaps this segment; a forecast, model label or text mention alone cannot create it |
| `label:<slug>/<name>` | valid label probability at or above its published operating point; for registered detection labels without one, the registry threshold or declared label floor applies |
| `source:<kind>` | an event of that evidence kind overlaps this segment, e.g. `source:detector` or `source:forecast` |
| `source:<producer>` | an event from that producer overlaps this segment, e.g. `source:tokeye_track` |

Label summaries are shot scoped and their tokens apply to each segment; they do not establish
when within a segment a label crossed its threshold. Event and phenomenon tokens are segment
scoped. Token sets and evidence are cached lazily in the loaded database snapshot.

**Query `--avoid phenomenon:<id>` requires observed coverage of the segment by a relevant
source.** It rejects observed matches and excludes shots whose relevant sources are unprocessed
or uncovered, with counts and per-state caveats in CLI text, CLI JSON and MCP search results.
No data is not an established negative. At least 50% of the segment must be measured
(`AVOID_MIN_COVERED_FRACTION = 0.5`), counting the union of covered stretches, never their hull
or double-counting overlap. Smaller fractions are excluded. Each retained partial negative
carries its covered percentage in the shot's `caveats` field (also printed under that shot in
CLI text); silence outside those stretches is unmeasured. This filter threshold does not
change the coverage states: even a boundary instant can be `observed`. An indexed phenomenon with no detector registered
(such as `rwm`) is unprocessed and cannot satisfy this filter. This coverage requirement applies
to phenomenon tokens; ordinary label/source tokens retain set membership semantics.

`describe` and MCP `describe_shot` add a line for each phenomenon resolved from the shot's
quotable logbook or supported by indexed evidence. Observed intervals, model label probabilities,
forecast ranges and text-only evidence are named separately, with the coverage state. Every
operator quotation comes from one `best_quote` logbook entry; narrow excerpts retain the
phenomenon mention, with ellipses marking omitted context.

## Phenomena: the evidence classes and the caveat vocabulary

```bash
shot_design phenomenon "edge harmonic oscillation" --n 20            # ranked shots + caveats
shot_design phenomenon "tearing mode" --avoid phenomenon:elm --json  # the hits as JSON
shot_design phenomenon --list                                        # the registry
```

`shot_design phenomenon` answers "which shots had one of these?" — and the answer is only usable
because it says *who claims so*. Four classes of evidence can name a phenomenon on a shot, and
they are not interchangeable:

| class | where it comes from | what it is |
| --- | --- | --- |
| **observed** | `events.parquet` rows whose `evidence_kind` is `detector` or `heuristic` — a `tokeye_track` in the right band, an `ece_sawtooth` crash, a `dalpha_lh` transition | a diagnostic showed it. The only class that is a measurement, and the only one that can rule the phenomenon *out* — inside the window the diagnostic covered. An allow-list, not "anything that is not a forecast": a `text` or `model` row that matches a registry rule is counted as neither, and says so |
| **label** | `labels_wide.max_valid` for a detection head named in the registry, **at or above its floor** | a model's opinion about the present. `None`, never `0`, where the model was not run or had no valid samples. A score *below* the floor (`thr` from the model card, else `retrieval.yaml`'s `label_floor`, 0.5) is reported with a caveat naming it and is not evidence: "the model ran" and "the model said yes" are different facts, and a quarter of this database's scored tearing shots are at or under 0.010 |
| **forecast** | `events.parquet` rows with `evidence_kind == "forecast"` — a risk curve crossed at a threshold, with a `horizon_s` | a model's estimate of what was about to happen. It arrives in `PhenomenonHit.forecasts`, never in `intervals`, and it is never reported as an observation |
| **text** | `text_claims.parquet`, `polarity=pos` and `temporality=observed` | an operator wrote it down. Never a label by itself: a hit resting on nothing else is capped at 0.25 and carries **TEXT ONLY** |

One registry id is a **topic, not a mode**: `fast_ion` (fast ions, energetic particles, beam ions,
FIDA) has no detector, no label head, no band and no coverage source, so `locate("fast_ion")` can
only ever return TEXT ONLY hits. It exists so that those words do not have to be aliased onto
`ae` — `ae` is one specific MHD mode with a ≥ 40 kHz detector band and an `--avoid` path read as
"the AE detector looked and saw nothing", and a diagnostic name resolving to it would turn a topic
match into a mode observation.

Ranking is the class order first and the score second: an observed hit outranks a label-only hit,
which outranks a forecast-only hit, which outranks a curated-list hit, which outranks a text-only
hit, whatever the weights in `configs/shot_design/retrieval.yaml` say. **On the `recommender_v1`
database today every one of the 1,037 event rows is a forecast**, so `shot_design phenomenon` returns
forecast-, label- and text-class hits and no observed ones at all until the production masks land
and `shot_design labels join` ingests labeler's detector output — and the command prints that fact
under its `resolved:` line rather than leaving it here.

### Coverage: the four states

`coverage` answers "did anyone look, over the window I asked about?", and `coverage_state` says
which of four situations a `null` is. Only the last makes an empty `intervals` a negative:

| `coverage_state` | what it means | query `--avoid phenomenon:<id>` |
| --- | --- | --- |
| `unindexed` | the shot is absent from `db.shots`; checked first by retrieval and MCP | excluded |
| `unprocessed` | no relevant detector completed on this indexed shot, or none is registered | excluded with a caveat |
| `uncovered` | relevant detectors completed, but no finite coverage overlaps the window; includes `ok` rows with NaN coverage | excluded with a caveat |
| `observed` | a relevant source's finite coverage overlaps the window | kept only without observed matching intervals; partial coverage is caveated |

For a phenomenon, relevant sources are its registered event sources plus `coverage_sources`.
`get_events` without a phenomenon considers all sources; with a phenomenon it uses precisely
that phenomenon's covering sources. Another detector cannot make RWM observed. Both tools say
`no detector registered for rwm; text/database evidence only` on an indexed shot with no RWM
detector. Unknown coverage gets its own `ran; coverage unknown` caveat.

Each source stores `intervals`, a JSON list of disjoint finite `[t0, t1]` pairs, and
`min_gap_s`, its detector-derived gap resolution. Finite runs separated by less than that
resolution may merge. Leading/trailing NaNs never extend coverage. Multi-input heuristics
intersect their required inputs' interval sets; any-channel steps use their union.
`t_cov0_s`/`t_cov1_s` are display hulls only.

Coverage is clipped to the requested window. A window wholly inside an interior gap is
`uncovered`, with a caveat listing the covered intervals around it. A window crossing a gap
is `observed` with `coverage_partial=True` and the “covered only” qualification; a window
inside one interval is `observed`. Retrieval's `coverage_windows` and MCP's
`coverage_windows` carry the actual union; hulls never decide observation.

`ShotDB.load` reads `event_sources.parquet` once per snapshot. An absent table may fall back
to detector/heuristic event rows: new pipeline rows preserve the source interval set in
`attrs.coverage_intervals` and its resolution in `attrs.coverage_min_gap_s`. Explicit empty
or unreadable source tables cannot borrow event coverage. Older source files, older joined
tables, and older events without interval metadata use their finite hull as one interval,
always with this caveat: “coverage recorded as a hull by an older writer; interior gaps unknown”.
They do not establish whether an interior dropout occurred. Reload the snapshot to see updates.

### The ELM case: a transient detector is not an ELM detector

`elm` now resolves observed `elm_clock` point events with `evidence_kind="heuristic"`.
The clock reads D-alpha filterscopes 0-7 and publishes a point-event family beside
its `elm_free` intervals. These points are measured peak-picking results, not a
trained ELM classifier; manual validation remains a follow-up. The rule has weight
1.0 and does not carry the class-agnostic transient caveat.

`tokeye_transient` publishes `phenomenon="transient"`, with its own lexicon and
registry entry. It cannot count as ELM evidence, including legacy rows whose
phenomenon is still `elm`. Re-run labeler's events stage and the labels join to
publish new products; a code change alone cannot populate the database. L-A's
read-only census found zero observed events on the 500 recommender shots, and
its CPU demonstrations under `/tmp` do not change that production count.

The registry is `configs/shot_design/phenomena.yaml` — one entry per phenomenon, saying which
`labels_wide` series, which event sources, which frequency band and which descriptors count.
It states **no aliases**: the ids and the phrases that name them live in
`src/labeler/events/lexicons.yaml`, which both packages read, and an `aliases:` key in
shot_design's file is an error rather than a second vocabulary.

### The caveat vocabulary

Every hit carries `caveats`, and a hit with none is a hit that lacks nothing — which is what
makes the others mean something. They are constants in `shot_design.retrieval.phenomena`, so a caller
can key on them:

| caveat | what it tells you |
| --- | --- |
| `TEXT ONLY` | nothing but operator text; the score is capped at labeler's `TEXT_ONLY_CEILING` |
| `ranked on forecasts: ...` | the strongest evidence is a model's estimate of what was about to happen; no diagnostic saw anything |
| `ranked on model labels: ... (<p>) ...` | the strongest evidence is a model's score, and the caveat says what the score was |
| `ranked on a curated human list: ...` | a human list names the shot and nothing else does |
| `no diagnostic coverage recorded; absence is not evidence` | diagnostic coverage is unknown; silence does not establish absence |
| `no detector registered for <id>; text/database evidence only` | the indexed shot is `unprocessed` for a phenomenon with no registered covering source |
| `<source>: ran; coverage unknown; absence is not evidence` | a relevant source completed without finite coverage |
| `required corpus group <group> is absent for <id>` | an inventoried corpus shot lacks a required group; unknown legacy inventories stay silent |
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

`shot_design phenomenon "tearing mode" --avoid phenomenon:elm` retains its exploratory behavior:
it drops observed ELM matches and keeps unknown coverage with a caveat. Use
`shot_design query "tearing mode" --avoid phenomenon:elm` when the result must satisfy the coverage
requirement described in Search above. The registry's `coverage_sources` identifies sources
that establish where a detector looked even when it found nothing. The ELM clock reads D-alpha
(`filterscopes`); its source record states whether it ran and over which filterscope span,
even when it found no points. Its `elm_free` intervals also provide coverage evidence through
`coverage_sources`; a missing clock remains unknown. A phenomenon without a detector (`rwm`,
`detachment`) declares no diagnostic coverage, so its coverage stays unknown and every hit for
it says so.

One field name is a promise it cannot yet keep: `actuators_at_onset` holds the **segment's** own
summary columns (`pnbi_total_mean`, `pech_total_mean`, `gas_total_mean`, `irmp_total_peak`), not
the values at the phenomenon's first interval. A per-onset lookup needs the raw waveform, which
the database does not carry; the keys are the column names, so the field cannot misdescribe
itself in the meantime.

## Evaluation

```bash
shot_design eval prompts --split eval            # the frozen 200 against the eval side of the split
shot_design eval prompts --split all --json      # the whole development universe, as JSON
shot_design eval latency --repeats 20            # the Appendix B budget table, warm
shot_design eval recall eho                      # detector recall vs a human annotation sheet
```

**The harness is frozen before anything is tuned.** `configs/shot_design/evalsets/` holds 200 prompts
(**v1.1**) authored from this corpus's own vocabulary — the 500 shots' mini-proposal titles, the
operators' logbook sentences, the 13 phenomenon ids and the 14 curation themes — and a dev/eval
split cut by **run day**, not by shot. `tests/shot_design/test_evalset_frozen.py` asserts the CSV's
sha256 against a literal in the test *and* against the hash in
`configs/shot_design/evalsets/README.md`, so editing the set takes three deliberate edits in three
files. **A prompt the retrieval cannot answer is a line in the report, not a rewrite of the
prompt.**

v1.1 is the one re-freeze: an independent review found eleven prompts with defects in the
**question** — an expectation naming the AE *mode* for a fast-ion *topic*, a row whose hard filter
selected against its own expected phenomenon, a duplicate pair, a garbled sentence — and those are
corrections to what is asked, not to what was answered. The README carries a prompt-by-prompt
changelog and keeps the superseded v1.0 hash, so a number published against either is
identifiable. Always quote the version beside the number.

**Some prompts cannot be answered by this corpus, and say so.** Eight rows return nothing on every
split because the 500 shots have no such discharge: `betan_mean` tops out at 3.006, `pech_total_mean`
at 2.298 MW, `pnbi_total_mean` never falls below 0.834 MW or reaches 12 MW, and `q95_mean` is NaN
on all 500 shots (the builder writes the column; nothing records it). They are kept, with an
annotation in `notes`, because they are honest things for a physicist to type and "nothing here"
is the correct answer — but they cost coverage, so coverage alone never tells the whole story.
Three further rows are satisfiable corpus-wide and empty only on the eval side; that is
split thinness, a different finding, and is annotated as such.

| what `eval prompts` measures | why it is separate |
| --- | --- |
| **coverage** — prompts with ≥ 1 result | a system that answers nothing is absent, not inaccurate. Plan bar: ≥ 95 % |
| **hard-filter survival** — candidates ÷ segment rows | a top-5 out of six and a top-5 out of three hundred are different answers |
| **channel participation** — which of `CHANNELS` contributed | a channel firing on 3 % of prompts is mis-wired or very narrow, and a score shows neither |
| **category → phenomenon resolution** | measures the **lexicon** against the words physicists type, not retrieval. Plan bar: ≥ 80 % on `qh_mode`, `elm_rmp`, `fast_ions` |
| **run diversity, duplicate rate** — over the top-10 | ten results from one run day are one experiment shown ten times |
| **the proxy grade** — over the 20 hand-graded prompts | see below |

`--split eval` (the default) restricts the database to the 110 eval shots **before** searching, so
the hard filter, the BM25 corpus statistics and the k-NN neighbourhoods are computed on one side
only. Every report says which split produced it, because a number from `all` and a number from
`eval` are not comparable. **Resolution is the exception**: it is computed from the prompt text
and the lexicon alone, so that column is identical on every split, and the table says so.

Exit codes: **0** measured and inside both bars, **2** refused (nothing to measure), **3**
measured and outside a bar. Both bars gate the exit, not only coverage.

### What the proxy grade is NOT

Twenty prompts carry `hand_graded=1` and a rubric in the evalset's `notes` column saying what a
correct top-5 looks like. The harness computes a **proxy** for them — does any of the top 5 carry
evidence of every expected phenomenon, or do all 5 satisfy the expected constraints — and every
report repeats, in its own `proxy.caveat` field:

> PROXY, NOT A HUMAN GRADE … It does not judge whether the shots are the right ones; the rubric in
> the evalset's `notes` column does, and only a human can.

A prompt can pass the proxy with five useless shots and fail it while returning the five a
physicist would have picked. A prompt that states no machine-checkable expectation scores `None`,
never a free pass.

### Latency

`eval latency` times five operations against the plan's Appendix B budgets (load < 3 s, search
with text < 400 ms, search without text < 200 ms, `phenomenon locate` < 300 ms; `describe` has no
budget and so gets the verdict `n/a` rather than a free PASS).

**Three separate blocks of N=20, interleaved, and a verdict only where they agree.** One block on
a shared login node measures the node, not the code: re-running this harness on the same database
gave `search_no_text` at 126 / 156 / 235 ms against a 200 ms budget and `search_text` anywhere
from 174 ms to 2.3 s. A row that straddles its budget across the blocks therefore reports
**`load-dependent`** — not a pass, not a failure, but the statement that the node would not let
the measurement be made — and only a row over budget in *every* block is a FAIL. The headline
median is the median of the block medians; the spread is printed. **Warm**: every operation runs
once and that run is discarded, so the MiniLM load and the first parquet read are not in the
numbers. The report records the load average, the core count and `torch.get_num_threads()`, so
the next reader's run is comparable with this one. Exit 3 means a row was over budget in every
block.

### Phenomenon recall

`eval recall <phenomenon>` scores the detectors against `$LABELER_ROOT/annotate/<phenomenon>/`
— `sheet.csv` joined to `manifest.parquet`, `split=test` rows only, `y`/`n` labels only. It
**refuses with exit 2** below 20 *scorable* test rows — counted after the windows on shots the
database does not hold are dropped, so a sheet of 22 rows with 20 absent shots cannot publish a
number from two windows. An empty table would read as "recall 0", and the two are opposite
answers. It also refuses, **before it reads the sheet**, a phenomenon `phenomena.yaml` gives no
event rule and no label head — `fast_ion`, `detachment`, `rwm`. Nothing writes an observed
interval for those, so their recall would be 0.0 over any sheet at all: a number about the
registry's shape, not about a detector. No sheet exists yet, so today every invocation refuses
for one reason or the other. Only an
*observed* interval overlapping the annotated window counts as a detection — a forecast and an
operator's sentence do not, or the detectors' recall would be inflated with the label models'
confidence.

### Out of scope here

Whether a detector is *physically* right. Agreement with a TokEye track or a teacher label is
agreement, not physical accuracy (plan V17: the existing AE model scores 0.99 against its teacher
and **0.62** against human annotation). Establishing physical quality is the labeler
workstream's task; this harness measures the retrieval layer built on top of whatever evidence
exists.

## The MCP server

`shot_design` exposes its retrieval over the Model Context Protocol, so an assistant can search the
database directly instead of being handed a transcript of a CLI run. Four tools — `search_shots`,
`describe_shot`, `get_events` and `phenomenon_locate` — and one resource. stdio transport:

```bash
pixi run -e ideate-cpu shot_design-mcp        # == python -m shot_design.mcp
```

### Attaching it in Claude Code

`.mcp.json` at the repo root is the project-scoped configuration; Claude Code reads it for any
session started in this checkout and asks once whether to trust it.

```json
{
  "mcpServers": {
    "shot_design": {
      "command": "pixi",
      "args": ["run", "-e", "shot_design-cpu", "python", "-m", "shot_design.mcp"],
      "cwd": "/scratch/gpfs/nc1514/FusionAIHub"
    }
  }
}
```

`cwd` is absolute because the server is started by whatever directory the client happens to be
in. `SHOT_DESIGN_DATA_ROOT` comes from the `shot_design-cpu` environment's own activation, so the config
names no data path; add an `"env"` block to point one session at a different database.

`/mcp` in a session lists the connected servers and their tools. Outside Claude Code, any MCP
client works — the transport is stdio and the command above is the whole contract.

### The tools

| tool | arguments | returns |
| --- | --- | --- |
| `search_shots` | `text`, `ref_shot`, `segment`, `constraints`, `actuators`, `require_labels`, `avoid_labels`, `n` | ranked shots with a description, an explanation naming which channel found each one, and the operating-limit flags for the proposed actuators |
| `describe_shot` | `shot`, `segment` | the prose description and the whole stored record: segments and their scalars, labels and their source, outcome, and the operator logbook verbatim |
| `get_events` | `shot`, `phenomenon`, `t0_s`, `t1_s` | a `status` — `unindexed`, `unprocessed`, `uncovered` or `observed` — and four lists kept apart: `events` (what a diagnostic showed), `text_mentions` (a lexicon hit in the logbook), `database_intervals` (a curated table's rows), `forecasts` (a model's estimate); plus `coverage`, the per-source table of what ran over which span |
| `phenomenon_locate` | `phenomenon`, `n`, `segment`, `constraints`, `min_confidence`, `avoid` | the shots carrying evidence of a phenomenon named in free text or by id, ordered by evidence class before score, each hit keeping its classes apart (`intervals`, `label_evidence`, `forecasts`, `text_snippets`) and its own caveats; plus `resolved` (what the text matched) and `notes` (what `avoid` dropped, about shots that are *not* in `hits`) |

Plus one resource, `shot_design://manifest`: the built database's manifest, which is how a caller
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
  never answers `status: observed`, and labeler's own `database:<stem>` source row — `ok`,
  coverage NaN — matches `event_sources.NON_DIAGNOSTIC_SOURCE_PREFIXES`, so it is not counted
  as a diagnostic having looked and cannot donate coverage either. `shot_design labels join` needs no rule for these rows: `events_union` reads each
  shot's `events/<shot>_events.parquet` wholesale, so they reach `db/events.parquet` as they
  are.
* **`status` says what an empty `events` means**, and the four values are not degrees of one
  thing. `unindexed`: the shot is not in the database (the error dict `describe_shot` gives).
  `unprocessed`: it is indexed, but no relevant source is recorded as having
  completed over it (the phenomenon's covering sources when filtered, all sources otherwise) — absence is not evidence. `uncovered`: detectors ran, but not over the
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
`scripts/shot_design/frame_codes_provenance.py --backfill` from the run manifests under `runs/encode/`.

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

Nothing in the tools re-implements retrieval: they call `shot_design.retrieval.rank` and
`shot_design.retrieval.describe` over the same `ShotDB` the CLI opens, so an assistant and
`shot_design query` cannot disagree about a shot. `src/shot_design/mcp/tools.py` is the whole contract —
the function signatures and docstrings there *are* the tool schemas the assistant sees.

## Local LLM and per-shot blurbs

`configs/shot_design/llm.yaml` enables `provider: ollama`. Without an endpoint file,
the client is unavailable and builds use the existing header + outcome template.
The UI reads stored blurbs; loading a page does not generate them.

Prompt v5 asks for exactly **three plain sentences**, at most **90 words**:

1. What the shot or experiment set out to do.
2. Whether it succeeded, plainly naming a disruption, fast current quench or early
   termination when the source records one; say when success is unknown.
3. One interesting finding from the operator entries or shot brief, or the literal
   sentence **No notable findings were logged.**

The model uses the mini-proposal title/purpose, run title, shot brief, up to eight
quotable operator entries, chief-operator status, outcome and verdict. It writes
in its own words, without quotation marks, headings, lists or invented/rounded
numbers. The gate rejects empty text, overlong text, double/curly quotation marks,
numbers or shot references absent from the source, and anything other than three
complete sentences. Sentence terminators are `.`, `!` and `?` followed by whitespace
or end of text; decimal points do not split sentences. A rejected reply retains
the header + outcome template. These mechanical checks do not establish semantic
accuracy: inspect the dry-run candidates before the full backfill.

Every generated table row carries `blurb`, `blurb_source` (`llm` or `template`),
`blurb_model` (resolved tag, e.g. `gemma4:26b`) and `blurb_prompt_version` (integer).
The latter two record the configuration used for the attempt, including template
fallbacks; `blurb_source` says whether the model supplied the final text. Legacy
rows retain unknown provenance until processed. `manifest.json["blurbs"]` contains
the whole table's `llm` and `template` counts plus the latest pass's `model` and
`prompt_version`; a limited pass may leave a mix of row versions.

`blurb` normally selects templates, empty blurbs, and rows with missing or older
prompt versions. `--all` selects every row, including current model blurbs.
`--limit N` takes the first N eligible shots in shot order (`0` does nothing).
`--dry-run` prints each shot number, source-text character count, candidate, gate
verdict/reason and final text; it writes neither database files nor request cache.
A normal pass atomically replaces `shots.parquet`, then atomically updates the
manifest. Those are two file replacements, not a transaction across both files.

### Operator runbook (after merge)

The existing Ollama 0.33.3 binary and both Gemma models are reused in place:

| `llm.yaml` key | default path |
| --- | --- |
| `ollama_bin_dir` | `/scratch/gpfs/EKOLEMEN/nc1514/shot-recommender/bin/ollama` |
| `ollama_models_dir` | `/scratch/gpfs/EKOLEMEN/nc1514/shot-recommender/models/ollama` |
| `ollama_home_dir` | `/scratch/gpfs/EKOLEMEN/nc1514/shot-recommender/ollama_home` |

`serve_llm.sh` requires `<ollama_bin_dir>/bin/ollama`; a missing binary exits 2
with its path. Installed model tags bypass pulling; a missing tag uses the retained
pull loop. No binary installation or copying occurs. Defaults are a 16384-token
context, 24h keep-alive and two loaded models. The Slurm allocation is one GPU,
eight CPUs, 64 GB RAM and four hours. See the upstream
[Ollama serving documentation](https://github.com/ollama/ollama/blob/main/docs/cli.mdx)
and [environment settings](https://github.com/ollama/ollama/blob/main/docs/faq.mdx).

Run these commands as the operator, from the merged checkout, on the login node:

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

# 1. Create the Slurm log directory, then submit the model server.
mkdir -p /scratch/gpfs/EKOLEMEN/nc1514/ideate/llm
sbatch scripts/shot_design/serve_llm.sbatch

# 2. Wait for readiness; check the job/log if this does not appear.
until test -s /scratch/gpfs/EKOLEMEN/nc1514/ideate/llm/endpoint.json; do sleep 2; done
cat /scratch/gpfs/EKOLEMEN/nc1514/ideate/llm/endpoint.json
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e ideate-cpu python -m shot_design llm

# 3. Inspect five candidates, gate verdicts and final summaries without writing.
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e ideate-cpu python -m shot_design blurb --all --dry-run --limit 5

# 4. After reviewing the preview, backfill all shots from the login node.
bash scripts/shot_design/blurb_all.sh

# 5. Stop the existing UI serve process with Ctrl-C in its terminal, then restart
#    it so its loaded database snapshot contains the new blurbs.
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e ideate-cpu python -m shot_design serve --port 8765
```

The blurb wrapper runs the mandated frozen Pixi command with
`python -m shot_design blurb --all`; additional flags are forwarded. The client
runs on the CPU and discovers the GPU server through `llm/endpoint.json` under
`paths.data_root`. A configured `base_url` takes precedence over that file.
Match the endpoint's `job_id` to the submitted job and check its log for `ready:`;
the `llm` command prints discovery metadata and does not probe the server.

The Slurm script binds `$(hostname -s):11434` and uses the default data root in its
literal `#SBATCH -o` directive. For another root, override `sbatch --output` and
create its `llm` directory first; account for Pixi activation overriding
`SHOT_DESIGN_DATA_ROOT` as described above. Endpoint JSON contains `url`, `models`,
`host`, `job_id`, `started` and `version`, and is published with an atomic rename.
An answering port is refused before any write. On exit, the script removes the
endpoint only if its URL and start time still match this run, then stops its child.
To stop the GPU allocation, use `scancel JOB_ID` with the ID printed by `sbatch`;
to renew it, wait for that job to exit and submit the same script again.

## Shot Designer: Search, Shot and Locate

Shot Designer uses a white page, a dark teal-green banner (`#2f6f66`), and light
neutral panels. The header shows the shot count and range; the build SHA remains
available in `/api/meta`. The local browser UI wraps the existing MCP tool functions and the
`shot_design phenomenon --json` retrieval path. It reads the database without building or
updating it. Start it on Stellar from this worktree:

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/src
export HF_HUB_OFFLINE=1 SHOT_DESIGN_DATA_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/ideate \
       LABELER_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker \
       SHOT_DESIGN_CORPUS=/scratch/gpfs/EKOLEMEN/foundation_model
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml \
  -e ideate-cpu python -m shot_design serve --port 8765
```

On your computer, run `ssh -L 8765:localhost:8765 stellar`, then open the token URL
printed by the server. The server binds only to `127.0.0.1`. A valid `?token=` link
sets an HttpOnly cookie and redirects to remove the token from the URL; requests
without that cookie receive 401. `--db-dir PATH` selects another built database;
`--token TOKEN` supplies a token instead of generating one. There are no external
browser assets or tunnel services.

Search supports text, a reference shot, segment, result count, JSON constraints,
and require/avoid labels. Search scores come from `search_shots`; titles come from
`describe_shot`. The results table has a Summary column in place of the quote.
Both replies' caveats stay available in the result row.

Summary uses the existing `blurb` and `blurb_source` columns in `shots.parquet`.
The record, search results and Locate hits all carry both fields. When
`blurb_source != "llm"`, a small muted **auto** tag appears next to the text;
template blurbs contain the factual header + outcome line. The offline
`shot_design blurb --all` pipeline writes LLM text into the same column, so a
three-sentence blurb needs no UI or table changes. The browser never generates
text. Missing columns, nulls or blank text yield `None`, displayed as `—` in
results; the shot Summary block is hidden when there is no text. No separate
summary table is loaded. Restart the UI after backfilling `shots.parquet` to load
the updated snapshot.

Shot starts with the available summary, followed by server-built `describe_parts`:
header, selected segment's scalar grid, labels, outcome, one complete attributed
operator quote, and a phenomena table. Each phenomenon row has its observed count,
first three observed intervals, forecast count, and coverage with qualifications.
Missing coverage does not turn an absent observation into a measured zero. Run/MP
metadata, stored flags, all segment scalars and the full logbook follow. The original
`describe()` string remains available to CLI/MCP callers; the browser does not parse it.

Measured numbers use one formatter: four significant digits without trailing zeros,
scientific notation for magnitudes at least `1e5` or below `1e-3` (except zero).
Times use seconds with three decimals, and confidence uses three decimals.
Identifiers, dates and counts retain their original digits; shot numbers never wrap.
Units come from the signal/actuator registry by scalar column. Slopes are per second
(`stat_slope` fits time in seconds), while fractions and unregistered quantities
stay unitless. Explicit registry qualifications such as `[?]` remain visible.
Operator text and offline summaries retain their original words and numbers.

Long text has a two-line / roughly 140-character preview with **more / less**.
Tall sections, including scalar cards and coverage, initially show at most 260 px
with **Show all / Show less**. Timeline colours identify the
reported evidence kind, and forecast lanes, curated database intervals and text
mentions remain separate. Event bars expose time, phenomenon, evidence kind and
confidence in hover titles; there are no per-event bullet lists. Coverage rows
without intervals retain their status and reason. Missing values display as `—`.
Every events reply shows its status and caveats in concise browser wording; the
MCP wording and `get_events` status semantics are unchanged. When there is no status
explanation in the reply, the explanation displays `—` too.

The Shot phenomenon filter matches literal event names, while Locate uses registry
classification rules. For example, an EHO can be classified from a `coherent_mode`
row. Selecting a phenomenon therefore also shows an **all event names** timeline
with its own status and caveats, so the literal filter cannot hide that context.
Locate hits show summaries, intervals, duration and caveats, and open Shot with the
phenomenon selected.

`/api/search` and `/api/shot/{shot}/events` preserve the MCP tool JSON.
`/api/shot/{shot}` retains `description`, `record`, `frame_codes` and
`caveats`, and exposes `blurb` and `blurb_source` both at top level and in `record`.
`/api/search` rows and `SearchHit` also carry `blurb` and `blurb_source`. The shot
response adds `units: {scalar_column: unit_string}` for all stored segments and:

```text
describe_parts: {
  blurb: string | null,
  blurb_source: string | null,
  header: string,
  segment: {name, t0_s, t1_s} | null,
  scalars: [{name, value: number | null, units: string}],
  labels: {regime, regime_source, operational, cluster},
  outcome: {...stored Outcome fields, end_time_s: number | null},
  operator_quote: {text, role, author, time} | null,
  phenomena: [{id, title, n_observed: integer | null, first_intervals: [Interval],
               n_forecast, coverage_note, coverage_windows: [[t0_s, t1_s]],
               coverage_partial, caveats: [string]}],
  caveats: [string]
}
```

`coverage_note` carries the registry coverage state. `first_intervals` contains at
most three observed `Interval` records; forecasts never enter it. Missing selected
segments return no scalars, without substituting another segment. Units are
unscaled registry strings: `ip_mean: "A"`, `ip_slope: "A/s"`,
`ne_line_mean: "m/cm3"`; dimensionless or unregistered quantities use `""`.

Error dictionaries remain HTTP 200. `/api/locate` preserves the CLI's
bare JSON list; CLI stderr notes travel as the JSON-encoded `X-Ideate-Caveats`
response header and are displayed above the hits. `/api/meta` summarizes the
manifest and registry; `/api/phenomena` lists registry IDs, titles, aliases and
sources. Unknown API paths return JSON 404.

The browser reads only `server` from `configs/shot_design/ui.yaml`; the file's `landing` and
`actuation` blocks belong to `retrieval.scenarios` and `retrieval.actuation`, which
predate the thin UI and still take their defaults from it. Request-scoped paths keep app factories separate;
Locate calls serialize the upstream curated-list cache reset because that cache
uses a configuration key instead of the selected CSV path.
