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
| **observed** | `events.parquet` rows with `evidence_kind` ≠ `forecast` — a `tokeye_track` in the right band, an `ece_sawtooth` crash, a `dalpha_lh` transition | a diagnostic showed it. The only class that is a measurement, and the only one that can rule the phenomenon *out* — inside the window the diagnostic covered |
| **label** | `labels_wide.max_valid` for a detection head named in the registry | a model's opinion about the present. `None`, never `0`, where the model was not run or had no valid samples |
| **forecast** | `events.parquet` rows with `evidence_kind == "forecast"` — a risk curve crossed at a threshold, with a `horizon_s` | a model's estimate of what was about to happen. It arrives in `PhenomenonHit.forecasts`, never in `intervals`, and it is never reported as an observation |
| **text** | `text_claims.parquet`, `polarity=pos` and `temporality=observed` | an operator wrote it down. Never a label by itself: a hit resting on nothing else is capped at 0.25 and carries **TEXT ONLY** |

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
