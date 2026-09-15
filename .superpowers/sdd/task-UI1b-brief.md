# Task UI1b — Summary slot reads `shots.parquet` blurb columns (fix on `recommender-UI1`)

Worktree: `/scratch/gpfs/nc1514/FusionAIHub-UI1`, branch `recommender-UI1` (HEAD a041abb). Work only here.
Read `.superpowers/sdd/task-UI1-brief.md` and `.superpowers/sdd/task-UI1-report.md` first.

## Problem

UI1 item 4 built a read-only `summaries.parquet` join (`ShotDB.summaries`, `ShotDB.summary()`,
`ShotRecord.summary`, `ResultItem.summary`, `PhenomenonHit.summary`, `load_errors['summaries']`,
`describe_parts.summary`, browser `row.summary` / `record.summary` / `hit.summary`, docs section,
tests in `tests/shot_design/test_ui1_data.py` / `test_ui1_browser.py`). Nothing produces that table.

The three-sentence summary (goal / success / finding) is produced by the sibling task LLM1
(branch `recommender-LLM1`, worktree `/scratch/gpfs/nc1514/FusionAIHub-LLM1`, read-only for you;
see its `.superpowers/sdd/task-LLM1-report.md`). It writes INTO `shots.parquet` — columns
`blurb` (text), `blurb_source` (`llm` | `template`), `blurb_model`, `blurb_prompt_version` — via
`shot_design blurb` (`shotdb/build.py::write_blurbs`). `blurb`/`blurb_source` already exist on
this branch (`build.py::SHOT_COLS`, `_shot_row`); LLM1 adds the two provenance columns.

Second defect: `ShotDB.get()` deserialises `ShotRecord` from the `record_json` column, and
`write_blurbs` does NOT rewrite `record_json`. So after a backfill `rec.blurb` (schema.py:239)
still holds the build-time template. The parquet columns are authoritative.

## Required changes

1. Remove the `summaries.parquet` machinery entirely: constructor argument, attribute, loader
   branch and validation, `summary()` method, `ShotRecord.summary`, `ResultItem.summary`,
   `PhenomenonHit.summary`/`SearchHit.summary`, `load_errors['summaries']` caveat, the
   `describe_parts.summary` key, docs section "summary contract", and every test of it.
2. `ShotRecord` gains `blurb_source: Literal["llm", "template"] | None = None` next to `blurb`.
   `ShotDB.get()` overwrites `rec.blurb` / `rec.blurb_source` from the `shots.parquet` columns
   when present (blank/NaN → `None` / `None`); a table without the columns leaves both `None`.
   `blurb_model` / `blurb_prompt_version` are NOT exposed to the UI.
3. Everywhere `summary` was exposed, expose `blurb` and `blurb_source` instead:
   `ResultItem`, `PhenomenonHit`/`SearchHit` (populated from the DB at hit construction),
   `/api/search` rows, `/api/locate` hits, `describe_shot()` top level and `record`,
   `describe_parts.blurb` + `describe_parts.blurb_source`. Additive names only; no other key
   renames. MCP tool descriptions untouched.
4. Browser: the results column stays titled "Summary" and shows `row.blurb` through `longText`;
   the shot page's top Summary block and Locate cards show `blurb`. When `blurb_source ===
   "template"` render a small muted tag `auto` after the text (class `blurb-auto`, `title`
   "Deterministic header + outcome; no model summary yet"); when `llm`, no tag. Missing/blank
   blurb → `—` in the column, block hidden, as before.
5. Docs (`docs/SHOT_DESIGN.md`): replace the summaries.parquet section with three or four
   sentences: the UI reads `blurb`/`blurb_source` from `shots.parquet`; `template` rows show the
   `auto` tag; `shot_design blurb` fills them (link to the LLM section by heading name; do not
   duplicate LLM1's runbook); restart `serve` after a backfill because the snapshot is loaded once.
6. Tests (TDD: write/adjust first, watch fail, then implement): store overrides record_json
   blurb from the columns; missing columns → None; blank → None; `describe_shot`/`/api/shot`
   carry `blurb` + `blurb_source`; search/locate rows carry both; browser renders `auto` tag
   only for `template`; no `summary`/`summaries` attribute or key remains
   (`grep -rn "summar" src/shot_design` should only hit unrelated words: `ShotSummary`,
   `to_summary`, `_sources_summary`, `CoverageSummary`, `shot_summary`, label-summary comments).
7. Update `.superpowers/sdd/task-UI1-report.md`: replace "## Summary table contract" and the
   item-4 paragraph with the blurb contract; update the API shapes block; add "## UI1b
   verification" with the exact commands and counts.

## Rules (binding)

- Every `pixi run` MUST be
  `pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e ideate-cpu ...`
  (labeler suite: `-e labelmaker`). Never `pixi install|lock|update`, never bare `pixi run`.
- Suite command (from the worktree root):
  `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/src HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 <pixi prefix> -e ideate-cpu python -m pytest tests/shot_design -q -W error -p no:cacheprovider`
  must be green (baseline on this branch: 1,399 passed at a041abb; expect fewer after removing
  summaries tests). Ruff:
  `/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/ruff check --no-cache src/shot_design tests/shot_design`.
- No production data writes (nothing under `/scratch/gpfs/EKOLEMEN`), nothing new under
  `/scratch/gpfs/nc1514` except this worktree's source. Tests use `tmp_path` only.
- Do not edit `docs/superpowers/plans/**`. Do not touch `recommender-LLM1` or the main checkout.
- Commit on `recommender-UI1` as you go, subjects prefixed `shot_design:`, bodies stating what
  was verified, trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. Leave
  `git status --short` clean. Finish by printing the final suite and ruff output lines.
