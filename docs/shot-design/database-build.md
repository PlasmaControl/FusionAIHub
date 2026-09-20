---
title: Database build
sidebar_position: 7
---

How the Frontier `shot_design` database — the shot list `recommender_frontier_v1`
— gets built: census the corpus, select a shot list, check logs, join labels,
fetch features from fdp, build the database, encode IGNITE frame codes,
backfill blurbs, then sanity-check coverage/describe. This mirrors the
Stellar `recommender_v1` build but starts from the Frontier `shotsummary`
text bundles directly (no `sql/` logbook import). The census target was
5,000 shots instead of 500; the Frontier corpus
supplies only 3,031 eligible shots out of 8,753 total shot files, so the
selected list is all 3,031 of them. All commands below run through
`scripts/slurm_frontier/_shot_design_common.sh` (`$PY`, `$ROOT`) or directly
as `sbatch` jobs from the repo root, per [Frontier](../clusters/frontier.md#slurm-wrappers).

Building this database is ongoing work; read `db/manifest.json` and the job
logs for the actual numbers once a stage completes.

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
that already have text. If fewer than `--n` shots are eligible, it takes all
of them — the list is not padded to hit a number: `--n 5000` on this corpus
selected all 3,031 eligible shots. The result is committed as
`configs/shot_design/shot_lists/recommender_frontier_v1.yaml`
(`n_eligible`/`n_selected`: 3031/3031 in its own `summary` block).

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

## 5. Features (fdp)

Frontier has no plasma current in the FAITH corpus and no labeler feature
store, so `ip` — and the 14 other labeler-addressed features (`bt betan
kappa li qmin tritop tribot gapin aminor r0 volume pcbcoil ne_zipfit
te_zipfit`) — come from GA's `fdp`, run from the Frontier **login node**
over Pelican/OSDF (HTTPS 443):

```bash
PIXI_CACHE_DIR=/tmp/pixi-cache-nchen pixi install --frozen -e fdp   # once
fdp login                                                            # writes ~/.fdp/cache/d3d.token
pixi run --frozen -e fdp fdp -D d3d run python -m labeler.run features \
    --features ip bt betan kappa li qmin tritop tribot gapin aminor r0 \
    volume pcbcoil ne_zipfit te_zipfit \
    --shot-file <shots.txt> --workers 12 \
    --root "$LABELER_ROOT" --corpus-dir "$SHOT_DESIGN_CORPUS"
```

`fdp login` writes the token to `~/.fdp/cache/d3d.token`; a JWT at
`~/.fdp/token` (mode 600) works too. Expect roughly 35 s per shot per
worker. Output lands at `$LABELER_ROOT/features/<shot>_features.h5`, one
file per shot, which `build --reader corpus` (below) reads.

**Fallback when a shot has no `ip` feature**: `build` derives segments from
the shot table's `PULSE-LENGTH` instead (ramp-up 1.0 s / ramp-down 0.27 s,
summing to `select.PROXY_RAMP_S` = 1.27 s) and marks the record honestly —
`coverage_reasons["ip"]` is set, `end_reason` is `no_ip_signal`, and `ip` is
absent from `raw_sources`. See `src/shot_design/shotdb/features.py`'s
`proxy_segments` and `configs/shot_design/retrieval.yaml`'s `segments:`
block (`proxy_ramp_up_s`/`proxy_ramp_down_s`) for the exact keys.

## 6. Build

```bash
SHOT_LIST=recommender_frontier_v1 sbatch scripts/slurm_frontier/shot_design_build.sh
pixi run --frozen -e shot-design-frontier python -m labeler.jobstats <jobid>
```

`shot_design_build.sh` runs on the `extended` partition by default (a
6-hour job exceeds `batch`'s 2-hour cap) and does the full atomic rebuild —
`db.tmp`, then swap — for the named shot list; its default `BUILD_ARGS` is
`--reader corpus --no-encode` (Frontier has no `d3d_fusion_data` raw layer,
so the FAITH corpus is the raw layer and the IGNITE channel is filled
separately by `shot_design_encode.sh`). Override `BUILD_ARGS` for a pilot
(e.g. `--reader corpus --limit 20 --no-encode`) and submit with `sbatch`'s
own `-p`/`-q`/`-t` overrides to run it in debug QOS instead of waiting on
`extended`: `sbatch -p batch -q debug -t 02:00:00 scripts/slurm_frontier/shot_design_build.sh`
(debug QOS caps at 2 h, which is enough for a `--limit`-ed pilot). Gate the
job afterwards with `labeler.jobstats`, same as any other Frontier job
(Frontier has no `jobstats` command; the utilization half of the gate comes
from `rocm-smi` samples written alongside the job).

## 7. Encode

```bash
SHOT_LIST=recommender_frontier_v1 sbatch scripts/slurm_frontier/shot_design_encode.sh
```

`shot_design_encode.sh` is an 8-task array, one GCD each, one contiguous
slice of the sorted shot list per task (`--chunk`/`--n-chunks`); the work is
I/O-bound rather than GPU-bound, so 8 GCDs on one node is enough.

## 8. Blurbs

```bash
bash scripts/shot_design/blurb_frontier.sh --dry-run --limit 5   # preview, login node
WORKERS=8 bash scripts/shot_design/blurb_frontier.sh              # full backfill
```

`blurb_frontier.sh` wraps `python -m shot_design blurb --workers "${WORKERS:-8}"
[--dry-run] [--limit N]`. The blurb backfill runs on the login node, not
through `sbatch` — the `agy`-based Gemini Flash provider needs outbound
network and the login node's OAuth cache (see
[LLM providers](./llm-providers.md#frontier-gemini-flash-via-agy)). With
`WORKERS=8`, the 3,031-shot list is expected to take on the order of an
hour; after it finishes, read `db/manifest.json`'s `blurbs` counts (target:
over 90% `blurb_source: llm`) and re-run `--dry-run --shots <N failing>` on
any gate failures to see why they fell back to the template.

## 9. Coverage / describe

```bash
pixi run --frozen -e shot-design-frontier python -m shot_design coverage
pixi run --frozen -e shot-design-frontier python -m shot_design describe 190736
```

Same commands used on Stellar, run once encode and blurbs have both
finished — compare a `describe` record against the equivalent Stellar shot
to sanity-check the Frontier build.
