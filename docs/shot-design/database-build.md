---
title: Database build
sidebar_position: 7
---

How the Frontier `shot_design` database — the shot list `recommender_frontier_v1`
— gets built: census the corpus, select a shot list, join labels, build the
database, backfill blurbs, encode IGNITE frame codes. This mirrors the
Stellar `recommender_v1` build but starts from the Frontier `shotsummary`
text bundles directly (no `sql/` logbook import), and targets roughly 5000
shots instead of 500. All commands below run through
`scripts/slurm_frontier/_shot_design_common.sh` (`$PY`, `$ROOT`) or directly
as `sbatch` jobs from the repo root, per [Frontier](../clusters/frontier.md#slurm-wrappers).

Building this database is ongoing work; the shot count and every count below
are targets, not measured results — read `db/manifest.json` and the job logs
for the actual numbers once a stage completes.

## 1. Census

```bash
sbatch scripts/slurm_frontier/shot_design_census.sh
pixi run --frozen -e shot-design-frontier python -m shot_design corpus summary \
    "$SHOT_DESIGN_DATA_ROOT/db/corpus_coverage.parquet" | head -40
```

One CPU node scans every shot file's header in parallel
(`corpus scan --workers 48`) and writes `db/corpus_coverage.parquet` — the
table every later selection, coverage question and `frame_codes_dirs` lookup
reads. `corpus summary` prints the resulting availability table; read it to
know how many shots are actually eligible before selecting.

## 2. Select

```bash
pixi run --frozen -e shot-design-frontier python -m shot_design corpus select \
    --n 5000 --name recommender_frontier_v1 --seed 20260919
```

`corpus select` draws a shot list from the census under the same rule as
`recommender_v1` (theme quotas, ×10 for the larger target), preferring shots
that already have text. If fewer than 5000 shots are eligible, it takes all
of them — the list is not padded to hit a number. The result is committed as
`configs/shot_design/shot_lists/recommender_frontier_v1.yaml`.

## 3. Logs (informational only)

```bash
pixi run --frozen -e shot-design-frontier python -m shot_design logs missing \
    --list recommender_frontier_v1 | tail -3
```

`logs missing` reports how many selected shots have no `per_shot_txt`
bundle. Nothing is imported by this step — there is deliberately no `sql/`
logbook on Frontier for this database — and `corpus select` already prefers
shots that have text, so this is a coverage check, not a required action.

## 4. Labels

```bash
pixi run --frozen -e shot-design-frontier python -m shot_design labels join \
    --list recommender_frontier_v1
```

Joins `labeler`'s labels and events (from `data/events/` and whatever is
under `$LABELER_ROOT`) into `labels_wide.parquet`, `events.parquet` and
`text_claims.parquet` for the selected shots. This is an upsert, not a
guarded production write, unlike `build` below.

## 5. Build

```bash
SHOT_LIST=recommender_frontier_v1 sbatch scripts/slurm_frontier/shot_design_build.sh
pixi run --frozen -e shot-design-frontier python -m labeler.jobstats <jobid>
```

`shot_design_build.sh` runs on the `extended` partition (a 6-hour job
exceeds `batch`'s 2-hour cap) and does the full atomic rebuild — `db.tmp`,
then swap — for the named shot list. `BUILD_ARGS` (e.g. `--limit 20` for a
pilot, `--reader corpus --no-encode`) is forwarded as an extra environment
variable. Gate the job afterwards with `labeler.jobstats`, same as any other
Frontier job (Frontier has no `jobstats` command; the utilization half of
the gate comes from `rocm-smi` samples written alongside the job).

## 6. Blurbs

```bash
bash scripts/shot_design/blurb_frontier.sh --dry-run --limit 5   # preview, login node
WORKERS=8 bash scripts/shot_design/blurb_frontier.sh              # full backfill
```

`blurb_frontier.sh` and `blurb`'s `--workers` flag are part of the
concurrent LLM-provider work (see [LLM providers](./llm-providers.md)) and
may not be present in every checkout yet; the underlying command is
`python -m shot_design blurb [--workers N] [--dry-run] [--limit N]`. The
blurb backfill runs on the login node, not through `sbatch` — the
`agy`-based Gemini Flash provider needs outbound network and the login
node's OAuth cache (see
[LLM providers](./llm-providers.md#frontier-gemini-flash-via-agy)). With
`WORKERS=8`, roughly 5000 shots is expected to take on the order of an hour;
after it finishes, read `db/manifest.json`'s `blurbs` counts (target: over
90% `blurb_source: llm`) and re-run `--dry-run --shots <N failing>` on any
gate failures to see why they fell back to the template.

## 7. Encode

```bash
SHOT_LIST=recommender_frontier_v1 sbatch scripts/slurm_frontier/shot_design_encode.sh
pixi run --frozen -e shot-design-frontier python -m shot_design coverage
pixi run --frozen -e shot-design-frontier python -m shot_design describe 190736
```

`shot_design_encode.sh` is an 8-task array, one GCD each, one contiguous
slice of the sorted shot list per task (`--chunk`/`--n-chunks`); the work is
I/O-bound rather than GPU-bound, so 8 GCDs on one node is enough. After it
completes, `coverage` and `describe <shot>` are the same commands used on
Stellar — compare a `describe` record against the equivalent Stellar shot to
sanity-check the Frontier build.
