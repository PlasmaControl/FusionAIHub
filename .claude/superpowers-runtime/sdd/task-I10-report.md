# Task I10 — report: `phenomenon_locate`, the fourth MCP tool

Branch `recommender-I10` in `/scratch/gpfs/nc1514/FusionAIHub-I10`, cut from `recommender` @ a3db1e7.

## What changed, and why

**`src/ideate/mcp/tools.py`** — new `phenomenon_locate(phenomenon, n=20, segment="flat_top",
constraints=None, min_confidence=0.0, avoid=None) -> dict`, plus three module constants
(`NOTHING_RESOLVED`, `_TIER_CAVEAT`, `_NOTES_CAVEAT`). It is what `cli.cmd_phenomenon` does —
`phenomena.resolve` then `phenomena.locate` — returned as a payload instead of printed as a
table, so an assistant can ask "which shots show this" without being handed a CLI transcript.

The three places where the CLI's shape is the *claim* and not a formatting choice are kept:

* **Unresolved text is an error, not an empty list.** `resolve` returning nothing is a fact about
  the question; an empty `hits` reads as a fact about the database, and the two are opposite
  answers. The reply is `{"error": "no phenomenon resolved from <text>; try one of: <id (title)>,
  …", "caveats": [NOTHING_RESOLVED]}` — the CLI's exit-2 case, with every registry id listed.
* **`notes` stays its own list.** `locate`'s notes are about shots that are *not* in the result
  (what `avoid` dropped, on what evidence); a dropped shot cannot carry a caveat, so this is the
  one claim no hit can report. The CLI sends them to stderr for the same reason. A top-level
  caveat says `notes` describes shots that are NOT in `hits`.
* **`ALL_FORECASTS`** is added to `caveats` when the database holds only forecast rows, exactly
  as the CLI prints it under `resolved:` — on such a database no hit below can be an observed one.

Top-level `caveats` also carry the standing tier rule, built by *formatting*
`phenomena.RANKING_SENTENCE`, `TEXT_ONLY` and `DATABASE_ONLY` rather than restating them (the
brief's "do not invent new sentences where one exists"; a caveat spelled two ways is one nobody
can filter on), and the segment correction `_segment` records when it reads `flattop` as
`flat_top`. Each hit's own `caveats` are passed through untouched.

Failure paths are the siblings' verbatim: `_segment` for the segment name, `_range` for the
constraints (so a model that learnt the constraint shape from `search_shots` need not learn a
second one), `KeyError` from `db.mask` reworded the way `search_shots` words it, `PhenomenaError`
(an `avoid` token that is not a phenomenon) as a sentence, `_db()` for the missing database, and
`never_raises` + `_incomplete_db()` for a build caught mid-publish.

**`src/ideate/mcp/server.py`** — `phenomenon_locate` appended to `TOOLS` (behind `never_raises`,
as the others are) and one paragraph added to `INSTRUCTIONS`: what the class order means, that a
hit's rank is therefore not its strength, that unresolved text comes back as an error rather than
an empty result, and that `notes` is about excluded shots. The `get_events` text is not restated.

**`docs/IDEATE.md`** — one row in the "### The tools" table in the same format as the other
three, and the "The MCP server" prose now names four tools where it named three.

**`tests/ideate/test_mcp.py`** — ten new tests (below) plus three existing ones extended to the
fourth tool name (`TOOLS`, the in-process server's tool list, the stdio roundtrip's).

## The tests

New, over a `phenomenon_db` fixture giving `ideate_db`'s four shots one evidence class each
(100 observed on three `tokeye_track` rows in the tearing band plus an ELM a detector saw, 101
forecast-only, 200 the operators' word only, 201 silent — the shape `test_phenomena.phen_db`
has, because only a database holding every class at once can pin that the ORDER reaches the
caller):

| test | what it pins |
| --- | --- |
| `…resolves_an_alias_and_ranks_by_evidence_class_first` | "an NTM at 2 s" → `tearing`; hits 100, 101, 200; `FORECAST_ONLY`/`TEXT_ONLY` still on the hits; `RANKING_SENTENCE` in `caveats` |
| `…on_text_naming_nothing_is_an_error_not_an_empty_list` | no `hits` key, the error lists `id (title)` pairs, the caveat says this is not "no shot has one" |
| `…avoid_drops_a_shot_and_the_drop_is_in_notes` | `avoid=["phenomenon:elm"]` removes 100; "dropped 1 shot" in `notes`; `caveats` says `notes` is about shots NOT in `hits` |
| `…rejects_an_avoid_token_that_is_not_a_phenomenon` | `PhenomenaError` → error dict |
| `…reports_an_unknown_constraint_column_as_an_error` | `KeyError` → error dict naming the column |
| `…takes_the_constraints_search_shots_takes` | `{"lo": …}` and `[lo, hi]` are one constraint |
| `…says_when_the_database_holds_only_forecasts` | `ALL_FORECASTS` in `caveats` |
| `…on_an_unknown_segment_lists_the_ones_there_are` | the segment error names the four |
| `…on_a_half_published_database_is_the_shared_error` | the REGISTERED tool: `FileNotFoundError`/`shots.parquet` + the "rebuilt … `ideate build`" caveat |
| `…without_a_database_at_all_names_the_build_command` | `_db()`'s sentence |

`caveats` is asserted non-empty on the success payload and on the no-resolution error. The
unknown-segment and unknown-column errors keep the siblings' convention (`_error` with whatever
caveats were accumulated), which is what `search_shots` does for the same two failures.

## Production smoke (read-only)

```
$ PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/src HF_HUB_OFFLINE=1 pixi run --frozen --no-install \
    --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e ideate-cpu \
    python -c 'from ideate.mcp import tools; import json; r = tools.phenomenon_locate("edge harmonic oscillation", n=5); print(...)'
{
 "phenomenon": "eho",
 "title": "Edge harmonic oscillation",
 "resolved": [
  {"id": "eho", "title": "Edge harmonic oscillation", "weight": 1.0}
 ],
 "n": 5,
 "hits": [
  {"shot": 186036, "score": 0.9030280321355949, "caveats": [
    "tokeye_track: coverage recorded as a hull by an older writer; interior gaps unknown",
    "required corpus group co2 is absent for eho",
    "no label evidence: no model in the registry emits a label for this phenomenon",
    "no operator text names this phenomenon on this shot",
    "the quote is this shot's most informative logbook entry and does not mention Edge harmonic oscillation"]},
  {"shot": 186034, "score": 0.8646647167633873, "caveats": [ … the same five … ]},
  {"shot": 186055, "score": 0.7364028618842733, "caveats": [ … the same five … ]},
  {"shot": 186059, "score": 0.7364028618842733, "caveats": [ … the same five … ]},
  {"shot": 185956, "score": 0.6321205588285577, "caveats": [ … the same five … ]}
 ],
 "notes": [],
 "caveats": [
  "hits are ordered by evidence CLASS before score -- an observed hit outranks a label-only hit,
   which outranks a forecast-only hit, which outranks a curated-list hit, which outranks a
   text-only hit -- so a hit's rank is not its strength: one carrying 'TEXT ONLY' or 'ranked on a
   curated human list: no detector, model or logbook claims this shot' rests on no observation at
   all"
 ]
}
```

(The per-hit caveat lists are identical across the five and are elided above; the first is shown
in full. Exit 0. Nothing was written: the call loads the database and reads.)

Two things in that output are worth reading twice. The five hits are **observed** — `tokeye_track`
rows, not forecasts — so `ALL_FORECASTS` is correctly absent, which means the database has gained
detector rows since `docs/IDEATE.md`'s "all 1,037 event rows are forecasts" note was written.
And every hit carries `required corpus group co2 is absent for eho`: the EHO rule wants mhr and
co2, and these shots' co2 flag is false, so the silence of a diagnostic that is not there is not
being counted as evidence.

## Latency (warm, same process, cached database)

`phenomenon_locate("edge harmonic oscillation", n=5)` after one warm-up call:

| run | wall |
| --- | --- |
| 1 | 175.0 ms |
| 2 | 174.5 ms |
| 3 | 173.4 ms |

`src/ideate/eval/latency.py`'s `BUDGETS_S["phenomenon_locate"]` is **0.3 s**, so all three are
inside it, with ~42 % headroom. Nothing was tuned to get there.

Read this next to the harness rather than as a replacement for it: `latency.measure` times
`phenomena.locate` itself over `elm` (the phenomenon with by far the most rows, chosen as the slow
case) and its own docstring records `phenomenon_locate` over budget in 6 runs out of 6 on a loaded
node. This is the MCP tool call, on `eho`, at `n=5`, on a quiet node — a different and gentler
measurement, and it says nothing about the harness's verdict.

## Verification

```
$ … -e ideate-cpu python -m pytest tests/ideate -q -W error -p no:cacheprovider
1262 passed in 212.44s (0:03:32)          exit 0
$ … ruff check src/labelmaker src/ideate scripts/labelmaker tests/labelmaker tests/ideate
All checks passed!                        exit 0
```

## Deviations

1. **`ruff` is not in the `ideate-cpu` environment** (`ruff: command not found`, and
   `python -m ruff` finds no module); it lives in the `labelmaker` and `fdp` envs. Rather than run
   a `pixi run` with an environment other than the mandated `-e ideate-cpu`, ruff was invoked as
   the binary directly:
   `/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/ruff check --no-cache <paths>`
   (`--no-cache` so it writes no `.ruff_cache` into the worktree). Same binary the mandated
   command would have reached; no environment was solved or installed.
2. **Two tests beyond the brief's list** — the `avoid`-token rejection and the two-spellings
   constraint test — because both are contract the brief names in prose ("`constraints` parsed
   with the existing `_range` exactly as `search_shots` does", "`avoid`") and neither was covered.
3. The timing script lives at `/tmp/i10/time_locate.py` (scratch under `/tmp/i10`, as required);
   the smoke itself is the brief's command, unmodified.

Nothing else departs from the brief. No ideate write command was run; no file was created under
`/scratch/gpfs/nc1514` outside this worktree; `configs/ideate/phenomena.yaml`,
`tests/ideate/test_phenomena.py`, `data/events/**`, `src/labelmaker/**`, `scripts/labelmaker/**`,
`tests/labelmaker/**` and `docs/superpowers/plans/**` are untouched.

## Commits

```
$ git log --oneline recommender..recommender-I10
<this>    ideate: I10 report - phenomenon_locate, the smoke and the timings
c7dd9b0  ideate: I10 - phenomenon_locate, the fourth MCP tool
dc110d2  ideate: I10 tests - phenomenon_locate, failing first
f404f9a  ideate: I10 brief - MCP phenomenon_locate tool to the existing tools' contract
```
