# Task UI2 — Shot Designer round-2 layout fixes (columns, one toggle per cell, longer clamp, timeline axis)

Worktree: `/scratch/gpfs/nc1514/FusionAIHub-UI2`, branch `recommender-UI2` (from `recommender` @ 5f36c2e). Work only here.
Read `.superpowers/sdd/task-UI1-brief.md` and `.superpowers/sdd/task-UI1-report.md` first (UI1 + UI1b are merged).
Files: `src/shot_design/ui/static/{app.js,style.css,index.html}`, `src/shot_design/ui/app.py`,
`src/shot_design/retrieval/describe.py` (`describe_parts`), tests `tests/shot_design/test_ui1_browser.py`,
`test_ui1_data.py`, `test_ui.py`, docs `docs/SHOT_DESIGN.md` (Shot Designer section).

## Owner feedback (verbatim intent, 2026-09-15)

"the score is still weird and cut off, as with some other columns. rule of thumb: the column title
should fill the column. i prefer for each of them just having one 'more' -- there's too many. the
text limit is also too short. for the event timeline the time is really off, or a weird scale:
shots are almost always up to 6-8 seconds long with a -2 to 0 second initial time. also the time
is hidden when i don't show all."

Screenshots showed: Score header wrapped as "Sc/or/e" and the value `0.03062` one digit per line;
"Phenomeno/n", "Obs/erve/d", "Fore/cast/s" headers wrapped; Run / mini-proposal cell with two
separate "more" toggles; Caveats cell with one "more" per caveat; phenomena "First intervals" cell
with both a "more" toggle and a "+241 more" note; every event lane's bars packed into the left
~10% of the lane; no time axis visible while the section is collapsed.

## Diagnosis (verified against the production DB, shot 199607)

- `style.css:45` gives `th, td { overflow-wrap: anywhere }`, so headers and numbers break inside
  words. `.shot-number` alone is exempt.
- `app.js:395` builds the timeline `domain` as `bounds([...events, ...forecasts,
  ...database_intervals, ...coverage])`. Coverage rows include the `actuator` source whose
  recorded coverage is `-10.000 .. 94.857 s` on 199607 (features `-3.99 .. 20.02 s`), so the axis
  runs -10..95 s while every event lies in -1.45..7.55 s. The shot's own `full` segment is
  0.013..6.944 s; other segments: ramp_up 0.013..0.987, flat_top 0.987..5.145, ramp_down 5.145..6.944.
- `timeline()` appends the axis AFTER the lanes, so the 260 px collapse hides it.
- `longText` clamps at 140 characters and is applied per item, giving one toggle per item.

## Required changes

1. **Columns.** Headers never wrap inside a word: `th { white-space: nowrap; overflow-wrap: normal }`;
   each column is at least as wide as its header (rule of thumb from the owner). Numeric cells
   (score, counts, times, confidences) are `white-space: nowrap` with tabular numerals; shot
   numbers keep their existing rule. Prose cells (Summary, Run / mini-proposal, Caveats, First
   intervals, Coverage) take the remaining width and wrap at word boundaries only
   (`overflow-wrap: normal; word-break: normal`), except that a single token longer than the
   cell may break. Tables sit in a container with `overflow-x: auto`; the page body never scrolls
   horizontally (phone width ~400 px still works: the table scrolls inside its container).
   Apply to Results, Locate, Phenomena, Scalars and any other table.
2. **One toggle per cell.** Add a cell-level helper (e.g. `cellText(items, {limit})`) that renders
   the cell's items (run title + MP title; all caveats; all intervals; multiple notes) as one
   block: items separated by line breaks, the WHOLE block clamped to the limit with a single
   `…` and a single more/less toggle that expands the whole cell. `longText` stays for
   single-item blocks. Remove the separate "+N more" notes where a toggle now exists (the
   expanded state shows everything; the count may appear inside the toggle label, e.g.
   "more (244)"). Row navigation propagation rules and aria attributes from UI1 stay.
3. **Longer clamp.** Table cells clamp at 320 characters / 5 CSS lines; shot-page blocks
   (Summary, Outcome, operator quote, caveats list) at 600 characters / 8 lines. One constant
   each, documented next to the formatter. Toggle hidden when nothing is clipped (as now).
4. **Timeline axis.** Domain rule (deterministic, documented in the code and docs):
   `t_lo = min(-2, floor(full.t0_s))`, `t_hi = max(8, ceil(full.t1_s + 0.5))` seconds, where
   `full` is the shot's `full` segment (fall back to the union of segments, then to -2..8 when
   no segment exists). Events, forecasts, database intervals and coverage outside the domain
   are CLIPPED to the edge with a small edge marker (class `clipped-left` / `clipped-right`)
   and the hover title states the true span (e.g. "coverage -10.000–94.857 s, drawn -2–8 s").
   Coverage is never allowed to widen the axis. Ticks every 1 s with labels (0 s emphasised),
   the axis drawn ABOVE the lanes and repeated below them; the top axis stays visible in the
   collapsed state (place it outside the collapsible body or make it sticky). The
   `/api/shot/{shot}/events` response gains `domain: {t0_s, t1_s, source: "full segment" |
   "segments" | "default"}` computed server-side from the record, so CLI/MCP consumers see the
   same numbers; the browser uses it and never recomputes from coverage.
5. **Tests** (TDD): CSS/DOM tests in the Node harness for nowrap headers/numbers and the single
   toggle per cell (counts of `.text-toggle` per cell = 1), clamp constants, clipped markers and
   titles, axis present before the lanes and containing the 0 s label; API test for `domain`
   on a record with a `full` segment, with only other segments, and with none; 199607-like
   fixture with actuator coverage -10..95 s must yield domain -2..8. Existing UI1 tests updated,
   not deleted, where behaviour changed intentionally (state which and why in the report).
6. **Docs**: update the Shot Designer section (table rules, clamp limits, timeline domain rule
   and clipping) in `docs/SHOT_DESIGN.md`. Do not touch the MCP tool descriptions or
   `get_events` semantics.
7. **Report** `.superpowers/sdd/task-UI2-report.md`: what changed per item, the domain rule,
   before→after for any wording, full verification output.

## Rules (binding)

- Every `pixi run` MUST be
  `pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e ideate-cpu ...`;
  never `pixi install|lock|update`, never bare `pixi run`.
- Suite (from the worktree root), must be green:
  `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/src HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 <pixi prefix> -e ideate-cpu python -m pytest tests/shot_design -q -W error -p no:cacheprovider`
  (baseline 1,440 passed / 0 skipped at 87311e1). Ruff:
  `/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/ruff check --no-cache src/shot_design tests/shot_design`.
  `node --check` every JS module.
- Read-only access to the production DB is allowed for diagnosis
  (`/scratch/gpfs/EKOLEMEN/nc1514/ideate/db`); NO writes under `/scratch/gpfs/EKOLEMEN`, nothing
  new under `/scratch/gpfs/nc1514` except this worktree's source; tests use `tmp_path` only.
  Do not start a server on port 8765 (the owner's instance may be running); use the TestClient
  or another port and stop it.
- Do not edit `docs/superpowers/plans/**`. Do not touch the main checkout.
- Commit on `recommender-UI2` as you go, subjects prefixed `shot_design:`, bodies stating what
  was verified, trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. Leave
  `git status --short` clean; end by printing the final suite, ruff and node --check lines.
