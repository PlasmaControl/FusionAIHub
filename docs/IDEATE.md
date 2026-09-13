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
| `get_events` | `shot`, `phenomenon`, `t0_s`, `t1_s` | a `status` — `unindexed`, `unprocessed`, `uncovered` or `observed` — and three lists kept apart: `events` (what a diagnostic showed), `text_mentions` (a lexicon hit in the logbook), `forecasts` (a model's estimate); plus `coverage`, the per-source table of what ran over which span |

Plus one resource, `ideate://manifest`: the built database's manifest, which is how a caller
finds out which shots the tools can see at all.

Three things hold for the replies and are worth knowing before reading one:

* **`caveats` is on every reply the tools themselves produce, and it changes what the reply
  means.** An empty result carrying "no channel had anything to search on" means the query was
  empty — not that no such shot exists. A result whose constraint excluded shots for having no
  recorded value says so, because "not measured" is not "out of range". The promise is scoped:
  a call whose *arguments* fail the tool schema (a string where a shot number belongs) is
  rejected by the MCP framework before any tool code runs, and comes back as a protocol
  validation error with no `caveats` field.
* **`get_events` keeps three kinds of claim in three lists.** An `events` row is a detector's or
  a heuristic's claim about what a diagnostic showed, with `source` saying who. A `forecasts`
  row (`evidence_kind == "forecast"`) is a model's estimate of what was about to happen, raised
  from a risk curve at a threshold. A `text_mentions` row (`evidence_kind == "text"`) is a
  lexicon hit in the operator logbook — somebody wrote the word — and is *not* evidence that the
  phenomenon occurred: shot 185980's "Updated ELM detector tuning." is a positive ELM hit about
  the detector. The three arrive in different lists and must stay in different sentences.
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

Nothing in the tools re-implements retrieval: they call `ideate.retrieval.rank` and
`ideate.retrieval.describe` over the same `ShotDB` the CLI opens, so an assistant and
`ideate query` cannot disagree about a shot. `src/ideate/mcp/tools.py` is the whole contract —
the function signatures and docstrings there *are* the tool schemas the assistant sees.
