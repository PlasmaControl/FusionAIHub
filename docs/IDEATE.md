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
| `eval` | the frozen evaluation harness: `eval prompts`, `eval latency`, `eval recall` |
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
| **observed** | `events.parquet` rows with `evidence_kind` ≠ `forecast` — a `tokeye_track` in the right band, an `ece_sawtooth` crash, a `dalpha_lh` transition | a diagnostic showed it. The only class that is a measurement, and the only one that can rule the phenomenon *out* — inside the window the diagnostic covered |
| **label** | `labels_wide.max_valid` for a detection head named in the registry | a model's opinion about the present. `None`, never `0`, where the model was not run or had no valid samples |
| **forecast** | `events.parquet` rows with `evidence_kind == "forecast"` — a risk curve crossed at a threshold, with a `horizon_s` | a model's estimate of what was about to happen. It arrives in `PhenomenonHit.forecasts`, never in `intervals`, and it is never reported as an observation |
| **text** | `text_claims.parquet`, `polarity=pos` and `temporality=observed` | an operator wrote it down. Never a label by itself: a hit resting on nothing else is capped at 0.25 and carries **TEXT ONLY** |

One registry id is a **topic, not a mode**: `fast_ion` (fast ions, energetic particles, beam ions,
FIDA) has no detector, no label head, no band and no coverage source, so `locate("fast_ion")` can
only ever return TEXT ONLY hits. It exists so that those words do not have to be aliased onto
`ae` — `ae` is one specific MHD mode with a ≥ 40 kHz detector band and an `--avoid` path read as
"the AE detector looked and saw nothing", and a diagnostic name resolving to it would turn a topic
match into a mode observation.

Ranking is the class order first and the score second: an observed hit outranks a label-only hit,
which outranks a forecast-only hit, which outranks a text-only hit, whatever the weights in
`configs/ideate/retrieval.yaml` say. **On the `recommender_v1` database today every one of the
1,037 event rows is a forecast**, so `ideate phenomenon` returns forecast-, label- and text-class
hits and no observed ones at all until the production masks land and `ideate labels join` ingests
labelmaker's detector output.

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
| `ranked on model labels: ...` | the strongest evidence is a model's score, not a diagnostic |
| `ranked on a curated human list: ...` | a human list names the shot and nothing else does |
| `no diagnostic coverage recorded; absence is not evidence` | nobody looked. **Not** "it did not happen" |
| `no observed evidence: ...` | no detector claims it on this shot |
| `no label evidence: ...` | no model in the registry emits a label for this phenomenon at all |
| `label not run on this shot, or no valid samples: unavailable, not 0` | the label row is missing or `n_valid == 0` |
| `no operator text names this phenomenon on this shot` | the logbook is silent |
| `text evidence is run scope: ...` | the sentence is the session's, shared by every shot of the run. Measured over 500 shots: `elm` fires on 99.4 % of shots at run scope and 11.0 % at shot scope |
| `operator log says NOT <title>` | somebody wrote that it was absent |
| `no <segment> segment on this shot; the whole record was searched` | the window was widened, and you are told |
| `kept despite --avoid <token>: ...` | the shot survived an `--avoid` filter because nothing looked for the avoided phenomenon — no data is not a negative |
| `<n> event(s) not shown: ...` | events the source never scored, dropped by `--min-confidence` |

`--avoid phenomenon:elm` drops the shots an ELM detector fired on and **keeps**, with that
caveat, the shots no ELM detector ran on. Dropping those would read a gap in the diagnostic
coverage as a physics result. What separates the two is `coverage_sources` in the registry: a
detector that finds nothing writes no rows, so on a quiet shot the only record that the mhr data
was read for ELMs at all is `elm_clock`'s `elm_free` interval — a *different* source from the one
that would have reported an ELM. A phenomenon nothing detects (`rwm`, `detachment`) declares
none, so its coverage stays unknown and every hit for it says so.

One field name is a promise it cannot yet keep: `actuators_at_onset` holds the **segment's** own
summary columns (`pnbi_total_mean`, `pech_total_mean`, `gas_total_mean`, `irmp_total_peak`), not
the values at the phenomenon's first interval. A per-onset lookup needs the raw waveform, which
the database does not carry; the keys are the column names, so the field cannot misdescribe
itself in the meantime.

## Evaluation

```bash
ideate eval prompts --split eval            # the frozen 200 against the eval side of the split
ideate eval prompts --split all --json      # the whole development universe, as JSON
ideate eval latency --repeats 20            # the Appendix B budget table, warm
ideate eval recall eho                      # detector recall vs a human annotation sheet
```

**The harness is frozen before anything is tuned.** `configs/ideate/evalsets/` holds 200 prompts
(**v1.1**) authored from this corpus's own vocabulary — the 500 shots' mini-proposal titles, the
operators' logbook sentences, the 13 phenomenon ids and the 14 curation themes — and a dev/eval
split cut by **run day**, not by shot. `tests/ideate/test_evalset_frozen.py` asserts the CSV's
sha256 against a literal in the test *and* against the hash in
`configs/ideate/evalsets/README.md`, so editing the set takes three deliberate edits in three
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

`eval recall <phenomenon>` scores the detectors against `$LABELMAKER_ROOT/annotate/<phenomenon>/`
— `sheet.csv` joined to `manifest.parquet`, `split=test` rows only, `y`/`n` labels only. It
**refuses with exit 2** below 20 *scorable* test rows — counted after the windows on shots the
database does not hold are dropped, so a sheet of 22 rows with 20 absent shots cannot publish a
number from two windows. An empty table would read as "recall 0", and the two are opposite
answers. No sheet exists yet, so today every invocation refuses. Only an
*observed* interval overlapping the annotated window counts as a detection — a forecast and an
operator's sentence do not, or the detectors' recall would be inflated with the label models'
confidence.

### Out of scope here

Whether a detector is *physically* right. Agreement with a TokEye track or a teacher label is
agreement, not physical accuracy (plan V17: the existing AE model scores 0.99 against its teacher
and **0.62** against human annotation). Establishing physical quality is the labelmaker
workstream's task; this harness measures the retrieval layer built on top of whatever evidence
exists.

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
| `get_events` | `shot`, `phenomenon`, `t0_s`, `t1_s` | the shot's events, filtered by phenomenon and by time overlap — and, in a separate list, the forecasts |

Plus one resource, `ideate://manifest`: the built database's manifest, which is how a caller
finds out which shots the tools can see at all.

Two things hold for every reply and are worth knowing before reading one:

* **`caveats` is always there, and it changes what the reply means.** An empty result carrying
  "no channel had anything to search on" means the query was empty — not that no such shot
  exists. A result whose constraint excluded shots for having no recorded value says so, because
  "not measured" is not "out of range".
* **`get_events` never mixes forecasts into `events`.** A row with `evidence_kind == "forecast"`
  is a model's estimate of what was about to happen, raised from a risk curve at a threshold;
  every other row is a claim about what a diagnostic showed. They arrive in different lists and
  must stay in different sentences. (On the `recommender_v1` database today, *all* 1,037 event
  rows are forecasts — so a tool that concatenated them would report nothing but model output as
  observation.)

Nothing in the tools re-implements retrieval: they call `ideate.retrieval.rank` and
`ideate.retrieval.describe` over the same `ShotDB` the CLI opens, so an assistant and
`ideate query` cannot disagree about a shot. `src/ideate/mcp/tools.py` is the whole contract —
the function signatures and docstrings there *are* the tool schemas the assistant sees.
