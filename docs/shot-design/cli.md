---
title: "The command line"
sidebar_position: 2
---

## The command line

```bash
pixi run -e shot-design-cpu shot_design <command>          # or: python -m shot_design <command>
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



Two subcommands added by this port are documented on their own pages:
[Simulation](./simulation.md) (`simulate`) and
[Database build](./database-build.md) (`corpus scan`/`select`, `build`,
`labels join`, `blurb`). `model --pin`/`--check`/`--download` (the IGNITE v4
bundle) is covered in [IGNITE — v4 generation](../models/ignite.md#12-v4-generation).
