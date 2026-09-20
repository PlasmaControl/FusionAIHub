---
title: "The MCP server"
sidebar_position: 3
---

## The MCP server

`shot_design` exposes its retrieval over the Model Context Protocol, so an assistant can search the
database directly instead of being handed a transcript of a CLI run. Four tools — `search_shots`,
`describe_shot`, `get_events` and `phenomenon_locate` — and one resource. stdio transport:

```bash
pixi run -e shot-design-cpu shot_design-mcp        # == python -m shot_design.mcp
```

### Attaching it from an MCP client

`.mcp.json` at the repo root is a project-scoped MCP configuration; any MCP client that reads
that convention picks it up for a session started in this checkout.

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

The transport is stdio and the command above is the whole contract — any MCP client that speaks
stdio can list the connected server's tools and call them the same way.

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

