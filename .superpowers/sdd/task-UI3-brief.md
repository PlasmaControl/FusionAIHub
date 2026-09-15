# Task UI3 — phenomenon lanes in the event timeline, stable numeric inputs, Info tab (scoring)

Worktree: `/scratch/gpfs/nc1514/FusionAIHub-UI3`, branch `recommender-UI3` (from `recommender` @ a6eda38). Work only here.
Read `.superpowers/sdd/task-UI1-brief.md`, `task-UI1-report.md`, `task-UI2-brief.md`, `task-UI2-report.md` first
(UI1, UI1b, UI2 are merged; keep their rules: number formatting, one toggle per cell, clamp limits,
timeline domain rule and clipping, terse wording, no build SHA in the banner).

## Owner feedback (verbatim intent, 2026-09-15)

1. "the phenomena i would like in the event timeline along with all the events."
2. "i dont know why but when i type a number in reference shot the number sometimes changes."
3. "also a tab that says either info or settings. and it should show how the search scores are calculated."

## Item 1 — phenomenon lanes in the timeline

- `/api/shot/{shot}/events` gains an additive `phenomena` list built server-side from the SAME
  registry evidence `describe_parts()` uses (share the code path; do not re-derive in JS):
  `[{id, title, intervals: [Interval], n_forecast, forecast_intervals: [Interval], coverage_note,
  coverage_windows, caveats}]`, registry order, only phenomena with ≥1 observed or forecast
  interval or a named text/database claim (same row selection as `describe_parts.phenomena`).
  Respect the request's `phenomenon`, `t0_s`, `t1_s`, `min_confidence` filters exactly as the
  `events` list does; `get_events` MCP semantics and tool descriptions unchanged.
- Browser (shot page, and the Locate context view where the same data is loaded): the Events
  timeline gets a **Phenomena** lane group ABOVE the per-source lanes, one lane per phenomenon,
  label = title with the id muted underneath, bars from `intervals`, coloured by evidence kind
  like source bars, hover title = phenomenon, source, evidence kind, span, confidence, caveats.
  Forecast intervals per phenomenon go into the Forecasts timeline as their own phenomenon lanes.
  Same domain, clipping, ticks and collapse behaviour as UI2. Frequency-resolved intervals
  (f0/f1 present) show the band in the title only.
- The Phenomena table keeps Phenomenon / Observed / Forecasts / Coverage and DROPS the "First
  intervals" column (the lanes now show every interval). `describe_parts.first_intervals` and
  `.intervals` stay in the API for other clients.

## Item 2 — numeric inputs changing by themselves

Diagnosis: every form input is `type="number"` (`index.html` lines 27, 29, 46, 53, 54, 100, 101).
Browsers change a focused `type="number"` value on mouse-wheel scroll and on ArrowUp/ArrowDown,
so scrolling the page with the cursor over the focused Reference shot field increments it.
Fix: shot identifiers (`ref_shot`, `shot`) become `type="text" inputmode="numeric"
pattern="[0-9]*" autocomplete="off"` with the existing identifier styling; other numeric fields
(`n`, hits, `t0_s`, `t1_s`, `min_confidence`) become `type="text" inputmode="decimal"` with
explicit parsing (`optionalNumber` / a shared `parseNumberField`) and inline validation
messages in the terse wording style (e.g. "Shot must be a whole number"). No wheel or arrow key
may alter any field. Confirm nothing rewrites a field the user is typing in: the only
programmatic writes are navigation (`renderShot` setting `shot`/`segment`), which stay.

## Item 3 — Info tab: how search scores are calculated

- New top-level tab **Info** (after Locate). Content is generated from the configuration the
  server actually loaded, never hardcoded in JS: new `GET /api/scoring` returning
  `{method, formula, k0, channels: [{name, weight, compares}], dedup_threshold,
  run_diversity_decay, outcome_penalty, hard_filters: [...], phenomenon: {formula, weights,
  saturation_n, class_order, text_only_ceiling}, db: {n_shots, shot_range, git_sha, built}}`.
  Sources: `configs/shot_design/retrieval.yaml` (`retrieval.k0`, `weights`, `dedup_threshold`,
  `run_diversity_decay`, `outcome_penalty`), `shot_design.retrieval.rank` (`rrf_fuse`,
  `search`), `shot_design.retrieval.channels.CHANNELS` (every registered channel appears; a
  channel absent from `weights` shows 1.0 and "default"), the phenomenon score formula and
  class order documented in retrieval.yaml / `retrieval.phenomena` (`OBSERVED > LABELLED >
  FORECAST > DATABASE > TEXTUAL`) and `configs/shot_design/phenomena.yaml`, `/api/meta` for db.
- Page text, plain and short, no hedging or AI-style qualifiers. Structure:
  1. **Search score** — "Each channel ranks every candidate; the score is a weighted reciprocal
     rank fusion: score = Σ_c w_c / (k0 + rank_c), k0 = 60. Scores are small (about 0.01–0.04)
     and only their order matters." Table of channels: name, weight, what it compares
     (scalar_knn: segment scalar embedding of the reference shot; text_knn: MiniLM embedding of
     mini-proposal and logbook text; bm25: exact words in that text; ignite_knn: IGNITE codec
     embeddings, needs a reference shot and an encoded database; phenomenon: resolved
     phenomenon evidence, ordered by evidence class then score). Then: hard filters
     (constraints, segment, require/avoid labels) apply before fusion; duplicate shots above
     cosine 0.97 in logbook-text space collapse; the m-th hit from one run day is multiplied by
     0.9^(m−1); with "prefer successful outcomes" a failed verdict or missed Ip target is
     multiplied by 0.5. Every number printed comes from the endpoint.
  2. **Phenomenon score** — the formula from retrieval.yaml with the configured weights and
     saturation count, the class order sentence, the text-only ceiling.
  3. **Database** — shot count, shot range, build SHA, build time (the SHA the banner no longer
     shows lives here).
- Tests: endpoint values equal the loaded YAML values (load the YAML in the test and compare,
  no literals); every `CHANNELS` key appears; Node harness renders the tab from a fixture and
  shows each channel name and weight, k0 and the class order; input type/inputmode assertions
  for every form field; phenomenon lanes appear above source lanes with the right titles and
  count; forecasts lanes per phenomenon; "First intervals" column absent; `/api/shot/{shot}/events`
  `phenomena` matches `describe_parts.phenomena` ids and interval counts on the shared fixture
  and honours the filters.
- Docs: `docs/SHOT_DESIGN.md` Shot Designer section: Info tab, `/api/scoring`, phenomena lanes,
  input behaviour. Report `.superpowers/sdd/task-UI3-report.md` with per-item changes, exact
  Info-tab wording, verification output.

## Rules (binding)

- Every `pixi run` MUST be
  `pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e ideate-cpu ...`;
  never `pixi install|lock|update`, never bare `pixi run`.
- Suite (from the worktree root), must be green:
  `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/src HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 <pixi prefix> -e ideate-cpu python -m pytest tests/shot_design -q -W error -p no:cacheprovider`
  (baseline 1,453 passed / 0 skipped at 1b409d3). Ruff:
  `/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/ruff check --no-cache src/shot_design tests/shot_design`.
  `node --check` every JS module. TDD: tests first.
- Read-only access to the production DB is allowed for diagnosis
  (`/scratch/gpfs/EKOLEMEN/nc1514/ideate/db`); NO writes under `/scratch/gpfs/EKOLEMEN`, nothing
  new under `/scratch/gpfs/nc1514` except this worktree's source; tests use `tmp_path` only.
  Do not start a server on port 8765 (the owner's instance is running); use the TestClient or
  another port and stop it.
- Do not edit `docs/superpowers/plans/**`. Do not touch the main checkout.
- Commit on `recommender-UI3` as you go, subjects prefixed `shot_design:`, bodies stating what
  was verified, trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. Leave
  `git status --short` clean; end by printing the final suite, ruff and node --check lines.
