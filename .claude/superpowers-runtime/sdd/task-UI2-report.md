# UI2 report — Shot Designer

- Worktree: `/scratch/gpfs/nc1514/FusionAIHub-UI2`
- Branch: `recommender-UI2`
- Starting HEAD: `a8ed58684c075ec8bd7cdfebac707d9b858fe86e`
- Date: 2026-09-15

Read the complete UI2 brief and the UI1 brief/report before implementation.

## Changes by brief item

1. **Columns and scrolling.** Tables use automatic layout and single-line
   headers, making each header the minimum width of its column. Numeric table
   cells, numeric fields, scalar values, scores in Locate, and inline numbers/time
   spans use nowrap and tabular numerals. Shot identifiers retain their existing
   treatment. Prose uses normal word boundaries with an emergency break for a
   token longer than its text container. The table wrappers constrain their own
   horizontal scrolling, and the containing grid can shrink on phone screens.
   Results and Phenomena are the existing tables; Locate and Scalars use cards
   and grids, to which the corresponding numeric/text rules also apply.

2. **One disclosure per cell.** `cellText()` joins a cell's items with line
   breaks before making one preview and one more/less control. Results combine
   run ID/run title/MP title and combine caveats/flags/asynchronous API notes or
   errors. Phenomena combine all intervals and all coverage qualifications. Cell
   content no longer contains nested section controls or one control per item.
   Expansion restores the complete text and preserves click propagation guards,
   `aria-controls`, and `aria-expanded`. Short text's toggle remains hidden.
   The shot caveats list is likewise one block with one disclosure.

   `describe_parts().phenomena[].intervals` is an additive field containing every
   observed interval. The existing `first_intervals` field retains its first-three
   contract for other clients. Forecasts remain separate. The browser uses the
   complete list, with a fallback to the legacy field, so expanding the preview
   does not merely expose another `+N more` note.

3. **Longer previews.** `CELL_TEXT_LIMIT = 320` and `BLOCK_TEXT_LIMIT = 600` are
   documented beside the formatters. CSS limits table text to five lines and
   shot blocks to eight. Summary, Outcome text fields, the attributed operator
   quote, and the combined caveats list use the block limit. Character clipping
   respects whitespace when possible and restores original text on expansion;
   the existing layout measurement reveals controls for CSS-only clipping.

4. **Timeline scale, clipping, and ticks.** `_timeline_domain()` computes the
   deterministic domain from the record on the server, converting milliseconds
   to seconds. Successful `/api/shot/{shot}/events` replies add:

   ```text
   domain: {t0_s, t1_s, source: "full segment" | "segments" | "default"}
   ```

   With the valid `full` segment:

   ```text
   t_lo = min(-2, floor(full.t0_s))
   t_hi = max(8, ceil(full.t1_s + 0.5))
   ```

   Fallback: apply the same rule to the union of valid record segments; use
   −2..8 s when none exist. Nonfinite/reversed spans are ignored. The full
   segment takes precedence even when another segment is wider. Coverage and
   event filters cannot affect the axis.

   Each HTTP Locate hit also adds this same domain, so observed and forecast
   timelines use the record's scale even for longer shots. This addition resolved
   the independent review's finding that Locate was initially using a fixed
   default. Existing Locate fields and its bare-list response shape remain;
   CLI/MCP models, tool descriptions and `get_events` semantics are unchanged.

   All drawable categories clip to the domain. Both partial and wholly external
   spans have `clipped-left`/`clipped-right` edge markers. Titles and accessible
   labels retain true endpoints, evidence metadata and the clipped drawn span.
   Right-edge points remain inside the track. Each nonempty timeline has a
   one-second tick axis above and below its lanes; the top axis is outside the
   collapsed body. Empty timelines retain a top axis. Zero is emphasised.
   Negative endpoint labels stay centred to avoid overlap at phone widths.

5. **Tests and TDD.** Extended the dependency-free Node DOM harness and real
   TestClient fixtures. New regressions exercise complete cell expansion after
   asynchronous success/failure, interval/coverage grouping, clamp boundaries,
   hidden short-text controls, numeric classes and CSS, clipped spans/points,
   axis order/ticks, and full/union/default domains with −10..95 s coverage.
   Additional tests cover invalid segment spans and Locate/Shot consistency for
   a −3.4..10.501 s full segment, yielding −4..12 s.

   Existing tests were preserved and updated where intended behavior changed:

   - `test_rendered_disclosures_summary_scalars_and_event_tooltips`: increased
     its quote from 15 to 30 repetitions so it still exercises expansion past
     the new 600-character threshold; retained the original rendering,
     attribution, formatting and accessibility assertions. Extracted its DOM
     setup into the shared harness used by the new tests.
   - `test_parts_separate_forecasts_and_bound_observed_intervals`: retained the
     first-three checks and added checks for the full observed interval list.
   - `test_transport_does_not_drop_new_fields_or_coerce_values`,
     `test_events_is_tool_json_and_preserves_evidence`, and
     `test_each_event_status_survives_transport`: assert/remove only the new
     domain before the existing exact tool-payload comparison. Error replies
     still compare directly.
   - `test_locate_is_cli_json_with_reply_notes`: assert/remove each hit's domain,
     then retain the exact CLI field comparison and response-note assertions.
   - `test_rendered_source_bars_do_not_fill_a_gap_or_an_empty_interval_set`:
     supplies −2..8 and expects the two disjoint bars at 20%/10% and 40%/20%.
     It still rejects bars for empty/skipped coverage and preserves the gap.

   Initial red run: **20 failed, 60 passed in 7.21s**. Failures showed duplicate
   controls, missing limits/nowrap/axes/domain/complete intervals, and old
   geometry. First implementation: **6 failed, 74 passed in 7.62s**; domain
   checks passed, but the new fixture incorrectly expected actuator coverage
   after an EHO filter. Restricting that assertion to the unfiltered reply
   preserved the existing filter semantics. Focused green: **80 passed in 7.86s**.
   The first full run passed **1451 tests in 143.93s**.

   Review follow-up red: **4 failed in 3.14s**, proving absent Locate domains and
   wrongly clipped 9..10 s intervals. After the fix, all focused tests passed:
   **82 passed in 7.40s**. The final full-suite transcript follows below.

6. **Documentation.** Updated only the Shot Designer section of
   `docs/SHOT_DESIGN.md`: table sizing/scrolling, grouped disclosures, both limits,
   the exact domain rule and fallbacks, clipping/axes, and additive HTTP fields.
   No edits under `docs/superpowers/plans/` or to MCP descriptions/semantics.

7. **Report and handoff.** This report records the changes, wording, TDD evidence,
   production/browser verification, final checks and intended commit boundary.

## Wording and presentation: before → after

| Before | After |
| --- | --- |
| Run/MP cell: a separate `more` for each title | One `more` / `less` for the whole cell, including its run ID |
| Caveats/coverage: one `more` per note, plus nested `Show all` controls | One `more` / `less` for all content in that cell |
| First intervals: preview plus `+241 more` | One `more` / `less` that reveals all 244 intervals; no separate count note |
| Three endpoint/midpoint labels such as `-10.000 s`, `42.429 s`, `94.857 s` | One-second labels `-2 s` through `8 s`, with `0 s` emphasised, above and below lanes |
| Coverage title starting `-10.000–94.857 s · …` | `coverage -10.000–94.857 s, drawn -2–8 s · …` |
| Arbitrarily wrapped headers and digits | Whole headers and unbroken numeric values, e.g. `Score` and `0.03062` |

Existing headings, `more`/`less`, `Show all`/`Show less`, evidence caveat wording,
and operator/summary text remain. Item line breaks replace per-item wrappers or
comma/semicolon separators in grouped cells. Only coverage titles add the word
`coverage`; clipped titles add `drawn … s`. Axis containers add the accessible
label `Time (seconds)`.

## Production and browser verification

Used authenticated TestClient calls against the production DB **read-only**:
`GET /api/shot/199607`, `GET /api/shot/199607/events`, and the event route filtered
to ELM over 1..2 s. No server was started and port 8765 was not used.

- Full segment: **0.01330908651351903..6.94430908651352 s**.
- Actuator gas coverage: **−10..94.85749816894531 s**, retained unchanged in JSON.
- Domain, both filtered and unfiltered:
  **`{"t0_s": -2, "t1_s": 8, "source": "full segment"}`**.
- Unfiltered payload: **2009 events**, **44 forecasts**, still separate.
- The 244-observation row is **AE**; the ELM row has **740** observed intervals.

Loaded the actual HTML/CSS/JS renderer and the TestClient production payload into
cached headless Chromium using the DevTools Protocol. The agent-browser CLI was
unavailable; no browser/dependency installation or Python environment update was
needed. Inspected screenshots and measured Search, Shot and Locate at **1440 px**
and **400 px**. Checks confirmed:

- No horizontal document overflow in any view. At 400 px: Search document 400 px;
  Shot/Locate 385 px plus the browser's 15 px vertical scrollbar.
- Search table: **632 px** of content scrolls inside **346 px** on mobile.
  Phenomena table: **710 px** scrolls inside **331 px**.
- Score `0.03062` stays on one line in a **73.90625 px** column. All header and
  numeric table-cell computed styles are nowrap. Prose cells have one toggle;
  numeric cells have none. A short Summary's toggle is hidden after measurement.
- The top timeline axes remain outside all collapsed bodies with every one-second
  label and bold `0 s` visible. Events use the width of the shot-scale lane.
- Expanding AE reveals all **244 intervals**; expanding ELM reveals all **740**.
  Each cell has one control, and neither expansion causes page overflow.
- Production gas coverage shows both 5 px edge markers and the title
  `coverage -10.000–94.857 s, drawn -2–8 s`, preserving its original span.

Transient production JSON, Chromium profiles, screenshots, browser checks and logs
were placed under `/tmp`; none are committed. Automated test data use `tmp_path`.
The browser and its ephemeral debugging port were closed after each check.

## Final verification

Final full suite, from the UI2 worktree root (exit **0**):

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/src HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e ideate-cpu python -m pytest tests/shot_design -q -W error -p no:cacheprovider
```

Complete stdout/stderr:

```text
WARN cache for Repodata at /home/nc1514/.cache/rattler/cache/repodata is on a network/parallel filesystem (NFS/SMB/FUSE/BeeGFS/Lustre/GPFS/CephFS), redirected to /tmp/pixi-cache-nc1514/repodata for this run. Set [cache.repodata] in config.toml or PIXI_CACHE_DIR to override, or [cache.netfs-redirect] = "never" to keep the original path.
........................................................................ [  4%]
........................................................................ [  9%]
........................................................................ [ 14%]
........................................................................ [ 19%]
........................................................................ [ 24%]
........................................................................ [ 29%]
........................................................................ [ 34%]
........................................................................ [ 39%]
........................................................................ [ 44%]
........................................................................ [ 49%]
........................................................................ [ 54%]
........................................................................ [ 59%]
........................................................................ [ 64%]
........................................................................ [ 69%]
........................................................................ [ 74%]
........................................................................ [ 79%]
........................................................................ [ 84%]
........................................................................ [ 89%]
........................................................................ [ 94%]
........................................................................ [ 99%]
.............                                                            [100%]
1453 passed in 139.12s (0:02:19)
```

**1453 passed; 0 failed; 0 skipped.** This is the 1440-test baseline plus 13 new
regressions. The Pixi cache-location warning above is emitted by Pixi before
Python starts; pytest runs with `-W error` and passes. No dependency changes.

Final lint (exit **0**):

```text
$ /scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/ruff check --no-cache src/shot_design tests/shot_design
All checks passed!
```

Final JavaScript syntax and whitespace checks (each exit **0**):

```text
$ node --check src/shot_design/ui/static/app.js
(no output; exit 0)
$ git diff --check
(no output; exit 0)
```

`app.js` is the only JavaScript module in `src/shot_design` and the only touched
JavaScript file. The independent read-only review found one Important issue
(Locate domain consistency), verified its fix, and reported no remaining findings.

Final Chromium production checks (exit **0**):

```text
Expanded production ELM cell: {"count":"740","intervalLines":740,"buttons":1,"expanded":"true","pageWidth":385}
Production clipped coverage: {"title":"coverage -10.000–94.857 s, drawn -2–8 s · status: ok; reason: —; diag: gas; channel: -1; pass_name: —; n_events: 15; min_gap_s: 0.020","classes":"mark clipped-left clipped-right","leftEdge":"5px","rightEdge":"5px"}
```

Chromium screenshots and metrics from this final browser run are under
`/tmp/shot-design-ui2-browser-mYAr2P/`. Earlier AE expansion also verified all 244
intervals. Browser scripts read the actual worktree assets; production replies
were obtained through TestClient, without starting an HTTP application server.

## Repository boundaries and commit

Only this worktree's source, tests, `docs/SHOT_DESIGN.md`, and this report changed.
Every Python invocation used the required frozen/no-install `ideate-cpu` prefix,
with bytecode disabled. Ruff used the prescribed read-only executable and
`--no-cache`. No environment install/lock/update, production writes, main-checkout
working-tree edits, worktree `.pixi`, or generated run artifacts are included.

The implementation, tests, documentation and report form one intended commit on
`recommender-UI2`, with a `shot_design:` subject, verification in its body and the
required trailer:

```text
Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
```

The commit is made only after all suites stop. Final handoff checks require a
clean `git status --short` on `recommender-UI2` and exactly that intended commit
after `a8ed586`.
