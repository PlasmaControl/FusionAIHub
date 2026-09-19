# UI3 report — Shot Designer

- Worktree: `/scratch/gpfs/nc1514/FusionAIHub-UI3`
- Branch: `recommender-UI3`
- Starting HEAD: `9563f9f`
- Date: 2026-09-15

Read the committed UI3 brief first, followed by the complete UI1 and UI2 briefs
and reports. Implemented all three UI items, their tests, and documentation.

## 1. Phenomenon lanes

`describe.phenomenon_rows()` now builds both structured description rows and
HTTP timeline rows. `_selected_phenomena()` shares registry-order selection with
the prose description as well. It retains the existing selection of interval,
model-label, named text and curated-list evidence; unknown coverage never becomes
a measured zero. The existing `first_intervals` and complete `intervals` fields
remain available, and `forecast_intervals` is additive.

Successful `GET /api/shot/{shot}/events` replies add `phenomena` rows containing:

```text
id, title, intervals, n_forecast, forecast_intervals,
coverage_note, coverage_windows, caveats,
n_observed, first_intervals, coverage_partial
```

The endpoint searches the whole shot by default. Descriptions continue to use
their selected segment; comparisons use the same window in tests. Request time
filters select inclusive overlaps and retain true endpoints; coverage windows
intersect the requested window. One-sided filters preserve the omitted coverage
bound, so they cannot manufacture an infinite unmeasured region.

The literal `phenomenon` filter selects that registry ID and classifies only the
event rows returned by the existing exact-name filter. For example, filtering
`tearing` does not bring back a `coherent_mode` row even when registry rules classify
it as tearing. The all-event context contains those classifications. This retains
the UI1/UI2 distinction between literal filtering and Locate classification.

`min_confidence` is HTTP-only because MCP `get_events` has no such argument. It
filters events and forecasts and updates their counts and forecast caveat. A
positive threshold excludes missing confidence as well as lower scores, with an
exclusion note. Zero retains unscored rows. Text/database claims and the original
coverage-derived status remain available. Invalid thresholds return a validation
error. No MCP implementation, description, signature or status semantics changed.
Untimed legacy rows remain in the unwindowed API payload and cannot make drawable
phenomenon intervals or break the response.

The shared browser timeline renders:

- **Phenomena** above **Sources**, one lane per returned phenomenon, with title
  and muted ID. Empty claim lanes retain their identity without invented bars.
- Separate phenomenon lanes in **Forecasts (model estimates)** for phenomena
  with forecast intervals. Forecasts never enter observed lanes.
- Source, phenomenon, evidence kind, true span, confidence and caveats in hover
  titles. Frequency bands appear only in titles.
- The same server domain, clipping markers, one-second axes, top axis outside the
  collapse body, and one section expansion control used by UI2.

Both the Shot view and the context opened from Locate use `renderEvents()` and
receive these lanes. The Phenomena table now has exactly **Phenomenon / Observed /
Forecasts / Coverage**. Coverage retains one disclosure per cell; the obsolete
**First intervals** column is removed.

## 2. Stable numeric input

Shot and Reference shot are `type="text" inputmode="numeric" pattern="[0-9]*"
autocomplete="off"`. Results, Hits, both time bounds and both minimum-confidence
fields are `type="text" inputmode="decimal"`. The event form gains its confidence
field so the HTTP filter can be used directly.

`parseNumberField()` accepts finite decimal/scientific notation for numeric
values, requires digits and safe integers for identifiers, positive whole result
counts, confidence within 0..1, and End after Start. Blank optional values remain
absent. Forms use explicit validation and an inline, live error message. No
validation path or asynchronous response writes back to a field being edited.
Navigation retains `openShot()`'s Shot/segment population and event-form reset;
initialization still fills the registry and segment choices.

Validation wording:

- `<field> is required`
- `<field> must be a whole number`
- `<field> must be a number`
- `<field> must be at least <minimum>`
- `<field> must be between <minimum> and <maximum>`
- `End must be after start`

Invalid submissions make no request. A Node submission test changes Reference
shot while a search response is pending and confirms the typed value survives.
Real Chromium checks confirm ArrowUp, ArrowDown and mouse-wheel scrolling leave
all eight numeric-input values unchanged.

## 3. Info tab and scoring endpoint

**Info** is the fourth top-level tab. Its route fetches `GET /api/scoring`; all
scoring numbers and channel names rendered by JS come from that response.

The endpoint uses `rank.load_cfg()` for RRF/reranking settings, `channels.CHANNELS`
for every registered channel, and the database's cached `_phenomenon_config` for
the weights/saturation Search actually uses. The last choice prevents Info from
advertising newly edited settings while Search still uses its loaded snapshot.
The text ceiling is the shared labeler lexicon constant; class order is sorted
from the actual phenomenon tier constants. A new unweighted channel receives
numeric weight 1.0 and `weight_source: "default"`, rendered **1 (default)** under
the existing number-formatting rule. Comparison descriptions live server-side.

Database count, range, build SHA and build time use the same helper as `/api/meta`.
The banner still shows only count/range. `score_range: [0.01, 0.04]` is the brief's
illustrative scale, supplied by the server; it is not a bound or a probability.

### Exact Info wording with the checked-in configuration

#### Search score

> Each channel ranks every candidate; the score is a weighted reciprocal rank fusion: score = Σ_c w_c / (k0 + rank_c), k0 = 60. Scores are small (about 0.01–0.04) and only their order matters.

| Channel | Weight | What it compares |
| --- | --- | --- |
| scalar_knn | 1 | Segment scalar embedding of the reference shot; requested scalar targets when no reference is given |
| text_knn | 1 | MiniLM embedding of mini-proposal and logbook text |
| bm25 | 0.6 | Exact words in mini-proposal and logbook text |
| ignite_knn | 1 | IGNITE codec embeddings; needs a reference shot and an encoded database |
| phenomenon | 1.2 | Resolved phenomenon evidence, ordered by evidence class then score |

> Hard filters (constraints, segment, require labels, avoid labels) apply before fusion.

> Duplicate shots above cosine 0.97 in logbook-text space collapse.

> The m-th hit from one run day is multiplied by 0.9^(m−1).

> With “prefer successful outcomes”, a failed verdict or missed Ip target is multiplied by 0.5.

#### Phenomenon score

> score = label × max_p + event × (1 − exp(−n_events / saturation_n)) + text × tanh(hits / 2) + database

The displayed key/value grid is:

```text
label          1
event          1
text           0.5
database       1
saturation_n   3
```

> max_p is the strongest qualifying model label. n_events is the sum of matching event-rule weights from phenomena.yaml. hits counts operator mentions. The database term applies when a curated list names the shot. Forecasts set the evidence class; they add no event term.

> Evidence class comes first: OBSERVED > LABELLED > FORECAST > DATABASE > TEXTUAL. Score orders hits within each class.

> Text-only ceiling: 0.25.

#### Database

The read-only production check displayed:

```text
Shots        504
Shot range   185786–204925
Build SHA    2d20398
Built        2026-09-14T15:19:57.046986+00:00
```

These values are not hardcoded in the page. Unavailable metadata renders `—`.

## 4. Tests, review and documentation

Added 26 regressions in `test_ui3.py` and `test_ui3_browser.py`. They exercise real
TestClient routes and the existing dependency-free Node renderer harness:

- Description/timeline registry ID and interval agreement; every interval,
  forecasts, named claims, literal/time/confidence filters, invalid thresholds,
  preserved MCP contracts, untimed rows, unchanged cached evidence and one-sided
  coverage semantics.
- Loaded YAML values, every channel, a newly registered default channel, changed
  configuration, warmed Search configuration, and shared database metadata.
- Field type/inputmode attributes, lane order/counts/titles/frequency bands,
  forecast separation, clipping/axis/collapse behavior, dynamic Info contents,
  explicit parsing, actual form submission, failed-validation request suppression,
  typing during an in-flight response, and confidence wording.

Existing tests were retained and adjusted for intentional changes:

- `test_ui.event_payload()` and the domain test in `test_ui1_data.py` remove/assert
  the new `phenomena` field before their exact unchanged MCP-payload comparison.
- The UI2 interval-cell expansion test now checks four Phenomena columns,
  retained observed/forecast counts, and one expansion control for all coverage
  qualifications. Complete intervals are now tested as timeline bars instead of
  a removed table cell. The description's complete/first-three interval tests
  remain in place.

TDD output:

```text
Initial new-feature red: 20 failed in 3.57s
Removed-column red: 1 failed, 9 deselected in 0.21s
First implementation: 8 failed, 94 passed in 9.32s
Corrected fixture/transport expectations: 60 passed in 5.60s
Untimed-event regression red: 1 failed, 21 passed in 3.37s
First full suite: 1475 passed in 145.65s (0:02:25)
Review regressions red: 4 failed, 22 deselected in 2.16s
Final focused UI3: 26 passed in 3.46s
```

The first implementation's remaining test failures were additive-field transport
comparisons, the curated list actually naming `qh` rather than `eho`, and the
float32 fixture representation of 0.9. The boundary test now uses exactly
representable 0.875. No confidence tolerance was invented in production code.

An independent read-only reviewer found and then verified fixes for the Search
configuration cache, false partial coverage for one-sided windows, and confidence
formatting. Re-review: **Ready from code review. All three findings are resolved.
No remaining findings.**

Updated `docs/SHOT_DESIGN.md`'s Shot Designer section with Info, `/api/scoring`,
phenomenon lanes, input behavior, HTTP filters, API shapes and configuration
snapshot behavior. The UI1/UI2 table, formatter, clamp and domain rules remain.

## Production/browser verification

Used TestClient for read-only production calls, with no application server:
`/api/shot/199607`, `/api/shot/199607/events`, `/api/scoring`, `/api/meta`,
`/api/phenomena`, and ELM over 1..2 s with minimum confidence 0.5.

```text
events=2009, forecasts=44
domain={t0_s:-2, t1_s:8, source:"full segment"}
phenomena (observed intervals, forecasts):
ae (344,0), eho (10,0), elm (773,0), transient (507,0),
tearing (236,44), sawtooth (65,0), fishbone (108,0), qcm (178,0),
lh (1,0), qmin_elevated (2,0), qmin_high (1,0)
filtered_events=0
```

Those whole-shot interval counts can exceed UI2's flat-top description counts;
both retain their original window contract. Confidence-filtered ELM rows are
absent because the source has not supplied confidence clearing that threshold.
The coverage status remains independent of the display filter.

Cached headless Chromium loaded the actual worktree HTML/CSS/JS and TestClient
production payloads through the DevTools Protocol. It checked all eight inputs,
every phenomenon lane's bar count, forecast lane counts, missing interval column,
top axes, channel names, absence of banner SHA, and page width at 1440 px and
400 px. Screenshots of Info and the timeline were inspected. No horizontal page
overflow; the existing table wrappers handle narrow widths.

```text
Chromium input checks: 8 text fields unchanged by ArrowUp, ArrowDown and mouse wheel
Chromium production lanes and Info: no page overflow at 1440 px or 400 px
```

Transient JSON, scripts, browser profiles, screenshots and metrics live under
`/tmp/shot-design-ui3-*`, outside the checkout. The browser terminates after each
check; it uses an ephemeral debugging port and no application server or port 8765.

## Repository boundaries

Every Python command used the required frozen/no-install main-manifest
`ideate-cpu` prefix, with bytecode disabled. Ruff used the explicitly prescribed
read-only executable and `--no-cache`. No install, lock or dependency update,
production write, main-checkout working-tree change, plan-directory edit, or new
worktree `.pixi` directory occurred. Tests write through `tmp_path` only.

The commit is made on `recommender-UI3` only after the full suite has stopped,
with a `shot_design:` subject, verification in its body, and trailer:

```text
Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
```

## Final verification

Full suite from the UI3 worktree root, exit **0**, **1479 passed / 0 skipped**
(1453 baseline + 26 UI3 regressions):

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/src HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e ideate-cpu python -m pytest tests/shot_design -q -W error -p no:cacheprovider
```

Complete pytest stdout:

```text
........................................................................ [  4%]
........................................................................ [  9%]
........................................................................ [ 14%]
........................................................................ [ 19%]
........................................................................ [ 24%]
........................................................................ [ 29%]
........................................................................ [ 34%]
........................................................................ [ 38%]
........................................................................ [ 43%]
........................................................................ [ 48%]
........................................................................ [ 53%]
........................................................................ [ 58%]
........................................................................ [ 63%]
........................................................................ [ 68%]
........................................................................ [ 73%]
........................................................................ [ 77%]
........................................................................ [ 82%]
........................................................................ [ 87%]
........................................................................ [ 92%]
........................................................................ [ 97%]
.......................................                                  [100%]
1479 passed in 143.94s (0:02:23)
```

Pixi emitted its existing network-filesystem cache notice before Python started,
redirecting the repodata cache to `/tmp/pixi-cache-nc1514/repodata`. Pytest ran
with `-W error` and reported no warning or skipped test.

Final Ruff, JavaScript syntax and whitespace checks each exited **0**:

```text
$ /scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/ruff check --no-cache src/shot_design tests/shot_design
All checks passed!

$ node --check src/shot_design/ui/static/app.js
(no output; exit 0)

$ git diff --check
(no output; exit 0)
```

`app.js` is the only JavaScript module under the Shot Designer source/tests.
The final Chromium pass used `/tmp/shot-design-ui3-browser-O2JSkn/` and exited 0.
The commit includes source, tests, documentation and this report. Final handoff
checks verify branch `recommender-UI3`, the required trailer and empty
`git status --short` after committing. No task item remains unfinished.
