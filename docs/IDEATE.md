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
| `export` | `ShotSummary` rows as JSON or Parquet |
| `actuation`, `blurb`, `llm`, `model` | actuator waveform sets, per-shot blurbs, LLM reachability, the IGNITE bundle |

`ideate <command> --help` is authoritative; the table is a map, not a specification.

The development universe is `configs/ideate/shot_lists/recommender_v1.yaml` — 500 shots drawn by
`ideate corpus select` under the rule in `src/ideate/shotdb/select.py`'s module docstring. The
database built from it is what the MCP server below serves.

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
| `get_events` | `shot`, `phenomenon`, `t0_s`, `t1_s` | the shot's events, filtered by phenomenon and by time overlap — and, in separate lists, the forecasts and the curated-table rows |

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
* **`get_events` keeps a curated list out of `events` too.** A row with
  `evidence_kind == "database"` comes from a table somebody sent us — the first two are Jeremy
  Hansen's RWM onset databases, under `data/labels/resistive_wall_mode/` and declared in
  `data/labels/tables.yaml` — and it arrives in a fourth list, `database_intervals`. It names a
  shot and a time, not a measurement: its `confidence` is `null`, because a human list has no
  calibrated probability, and its coverage (`t_cov0_s`/`t_cov1_s`) is `null`, because nobody
  recorded which interval of the shot was examined. That second null is the load-bearing one — a
  shot's **absence** from a curated list is not a negative, and nothing downstream may read it as
  one. `ideate labels join` needs no rule for these rows: `events_union` reads each shot's
  `events/<shot>_events.parquet` wholesale, so they reach `db/events.parquet` as they are.

Nothing in the tools re-implements retrieval: they call `ideate.retrieval.rank` and
`ideate.retrieval.describe` over the same `ShotDB` the CLI opens, so an assistant and
`ideate query` cannot disagree about a shot. `src/ideate/mcp/tools.py` is the whole contract —
the function signatures and docstrings there *are* the tool schemas the assistant sees.
