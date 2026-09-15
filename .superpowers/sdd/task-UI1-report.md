# UI1 report — Shot Designer

Worktree: `/scratch/gpfs/nc1514/FusionAIHub-UI1`
Branch: `recommender-UI1`
Base: `e54490a` (brief commit, following `3a08ba0`)
Date: 2026-09-15

## Changes by brief item

1. **Title:** document title and banner read **Shot Designer**. The production metadata displays `504 shots · 185786–204925`.
2. **Theme:** white page; `#2f6f66` banner, links and buttons; `#24574f` hover; `#f8faf9` panels; `#d6e2de` borders. White banner text and visible focus outlines. Mobile layout remains a single column.
3. **Shot identifiers:** nowrap, tabular/monospace shot digits and an 8ch minimum Shot column. Result and Locate links, shot headings, reference inputs and shot references embedded in long text preserve whole shot numbers. Dates, counts, run IDs and MP IDs bypass numeric rounding.
4. **Summary slot (amended):** reads the existing `blurb` and `blurb_source` columns in `shots.parquet`. `ShotRecord`, `ResultItem`, `PhenomenonHit` and its public alias `SearchHit` carry both optional fields. Search/Locate and the shot Summary block render `blurb`; a small muted `auto` tag appears when text is present and its source is not `llm`. Missing text displays `—` in results and hides the shot Summary block. No separate summary table is loaded or generated; no summary generation or production writes.
5. **Text expansion:** shared `longText` helper, about 140 characters and two CSS lines, with ellipsis and more/less buttons. Used for summaries, caveats, titles, notes, fields and logbook text across all views. Toggle buttons stop row-navigation propagation and expose expanded state and controlled element IDs. Quoted text is never rewritten; expansion restores the complete entry.
6. **Numbers:** shared `formatNumber` applies four significant digits, strips trailing zeros, and uses scientific notation at the specified magnitude boundaries. Times convert milliseconds to seconds and use three decimals; confidences use three decimals. Invalid/nonfinite/missing values show `—`, never zero. Rules and eight examples are documented beside the formatter. Original numbers in operator prose and offline summary text remain verbatim.
7. **Units:** API maps stored scalar columns through `split_stat` to the signal/actuator registry. `stat_slope` uses `np.polyfit(t / 1000.0, y, 1)`, so slopes use unit/s. Fractions and unregistered quantities use the empty unit string. Values remain unscaled, e.g. `ip_mean 8.924e5 A`.
8. **Tall sections:** shared 260px collapse body, fade and Show all/Show less. Applies to selected scalar grid, per-segment cards, their containing section, outcome, phenomena table, caveats, event/forecast/database/coverage lanes, text mentions, logbook, frame codes, run/MP metadata, result-row notes and complete record. Disclosure measurement is batched after DOM layout and refreshed on resizing, view changes and expansion. Clipped controls are removed from keyboard navigation until revealed.
9. **Timeline:** lane graphs retained; per-event bullet lists removed. Bars carry hover titles with time span, phenomenon, evidence kind, confidence and any caveats. Coverage rows without drawable intervals retain their status/reason. Exact interval sets preserve coverage gaps. Forecasts stay in their own lanes.
10. **Structured shot page:** optional Summary first, header, selected segment scalar grid, labels, Outcome, one attributed operator quote, phenomena table; metadata, remaining scalars and logbook follow. `describe_parts()` builds the data server-side and shares the registry evidence selection used by `describe()`. The existing description string remains unchanged; no browser paragraph parsing. Missing selected segments are not replaced with another segment.
11. **Wording:** all shortened shared caveats are mapped only in the browser and tested against the original Python constants/templates. Coverage bounds, gap counts, forecast/observation distinction, run scope, unknown coverage and unprocessed-versus-quiet meaning remain. Unknown wording passes through. MCP tool descriptions and `get_events` status semantics are unchanged. Full before/after ledger below.
12. **Build SHA:** removed from visible banner; `/api/meta.db.git_sha` remains available and was checked against production metadata (`2d20398`).

`docs/SHOT_DESIGN.md` now documents the title, theme, summary contract, structured response, number/units rules and disclosure controls. Its launch directory is the UI1 worktree and the caveat response-header spelling is corrected to the existing `X-Ideate-Caveats`.

## Blurb column contract (amended)

Location: existing `<db_dir>/shots.parquet` rows. No new table:

| Column | Type | Meaning |
| --- | --- | --- |
| blurb | string or null | Stored Summary text: LLM sentences or factual header + outcome template |
| blurb_source | string or null | `llm` or `template`; unknown on legacy rows |

`ShotDB.get()` overlays both current table values on the stored `record_json`, because the offline backfill updates columns independently of that JSON. Missing columns/null/blank values become `None`. The same normalization supplies Locate hits. The renderer uses `blurb` directly and tags any non-LLM source `auto`, including unknown legacy provenance. `shot_design blurb --all` can replace the text in place with three LLM sentences. Restart the UI after backfilling to load the new snapshot. This amendment writes no production table. An obsolete `summaries.parquet`, if present, is ignored.

## API shapes

`/api/search` retains the existing search result shape with `results[].blurb` and `results[].blurb_source`. `/api/locate` retains its bare hit list with both fields; `SearchHit` aliases `PhenomenonHit`. `describe_shot()` exposes both fields at top level and in `record`; its description string and docstring remain unchanged. The superseded `summary` fields were removed.

`GET /api/shot/{shot}?segment=flat_top` preserves all of `describe_shot()` and adds:

```text
units: { scalar_column: unit_string }
describe_parts: {
  blurb: string | null,
  blurb_source: string | null,
  header: string,
  segment: {name: string, t0_s: number, t1_s: number} | null,
  scalars: [{name: string, value: number | null, units: string}],
  labels: {regime, regime_source, operational, cluster},
  outcome: {...stored Outcome fields, end_time_s: number | null},
  operator_quote: {text: string, role: string, author: string | null,
                   time: string | null} | null,
  phenomena: [{
    id: string, title: string,
    n_observed: integer | null,
    first_intervals: [Interval],
    n_forecast: integer,
    coverage_note: string,
    coverage_windows: [[t0_s, t1_s]],
    coverage_partial: boolean,
    caveats: [string]
  }],
  caveats: [string]
}
```

`units` covers every scalar column in all stored segments. Examples: `ip_mean: A`, `ip_slope: A/s`, `ne_line_mean: m/cm3`, `dalpha_std: ph/cm2/sr/s`, `pnbi_total_mean: W`, `betan_mean: ''`, `pech_total_on_frac: ''`. The selected segment's scalar values remain raw numbers. Nonfinite scalar values become null in `describe_parts`.

`first_intervals` contains at most the first three observed `Interval` records (`t0_s`, `t1_s`, optional frequency endpoints, source, evidence_kind, confidence, event_id). `n_observed` is null when there are no observed rows and the relevant coverage state is not observed; measured empty coverage yields 0. Forecast counts are counts of separately indexed forecast rows. `coverage_note` is the registry coverage state; interval unions, partial status and qualifications remain separate fields. One row per phenomenon named by quotable shot text or supported by indexed evidence, in registry order. All construction is deterministic.

## Wording ledger

### Shared caveats: original source → browser display

The examples below exercise the actual upstream constants and dynamic templates. Only the rendered wording changes; Python source strings remain intact.

- **Before:** text evidence is run scope: a session-wide sentence, not this shot's logbook
  **After:** Run-level text, not shot-specific

- **Before:** no diagnostic coverage recorded; absence is not evidence
  **After:** Coverage unrecorded; absence unmeasured

- **Before:** no detector registered for rwm; text/database evidence only
  **After:** rwm: no detector; text/database evidence only

- **Before:** no detector for ELM has run on this shot; absence is not evidence
  **After:** ELM: detectors not run; absence unmeasured

- **Before:** the ELM detectors ran on this shot but not over the flat_top window; absence there is unmeasured, not established
  **After:** ELM: no coverage of flat_top; absence unmeasured

- **Before:** the Transient activity detectors covered only [0.9873090865135192, 3.797149896621704] s of the flat_top window; absence outside that is unmeasured
  **After:** Transient activity: covered 0.987–3.797 s of flat_top only; outside unmeasured

- **Before:** the ELM coverage of the flat_top window has 2 gap(s): the covered intervals are disjoint; see `coverage_windows` for the observed portions
  **After:** ELM: 2 gaps in flat_top coverage; see covered intervals

- **Before:** observed via tokeye_transient, a class-agnostic transient detector: a sawtooth crash or a disruption precursor is transient too, so this is an ELM-LIKE TRANSIENT and not a classified ELM
  **After:** Class-agnostic transient; may be ELM-like, a sawtooth or a disruption precursor

- **Before:** 3 row(s) of evidence_kind human match this phenomenon's rules and are counted as neither observation nor forecast
  **After:** 3 human rows; neither observations nor forecasts

- **Before:** tm/p scored 0.123, below the 0.50 evidence floor: reported, not counted as evidence
  **After:** tm/p: 0.123, below evidence floor 0.500

- **Before:** the quote is this shot's most informative logbook entry and does not mention ELM
  **After:** Shot logbook quote; does not mention ELM

- **Before:** this database holds 44 event row(s), all of them forecasts: no observed-class hit is possible yet
  **After:** 44 indexed rows, all forecasts; no observations

- **Before:** TEXT ONLY
  **After:** Text only

- **Before:** ranked on forecasts: the strongest evidence here is a model's estimate of what was about to happen, not an observation
  **After:** Ranked on forecasts (model estimates)

- **Before:** ranked on model labels: the strongest evidence here is a model's score (0.900), not a diagnostic
  **After:** Ranked on model labels (0.900); no diagnostic evidence

- **Before:** ranked on a curated human list: no detector, model or logbook claims this shot
  **After:** Curated list only; no detector, model or logbook evidence

- **Before:** no observed evidence: no detector claims this phenomenon on this shot
  **After:** No detector evidence for this phenomenon

- **Before:** no label evidence: no model in the registry emits a label for this phenomenon
  **After:** No label model for this phenomenon

- **Before:** label not run on this shot, or no valid samples: unavailable, not 0
  **After:** Label unavailable: not run or no valid samples

- **Before:** no operator text names this phenomenon on this shot
  **After:** No shot-specific operator mention

- **Before:** operator log says NOT ELM
  **After:** Operator reports no ELM

- **Before:** no flat_top segment on this shot; the whole record was searched
  **After:** No flat_top segment; searched whole record

- **Before:** 44 forecast row(s) are in `forecasts`, not in `events`: a forecast is a model's claim about what was about to happen, not an observation of what did
  **After:** 44 forecasts (model estimates) shown separately

- **Before:** 2 row(s) are in `database_intervals`, not in `events`: a curated list names a shot and a time, not a measurement. Its coverage is null because nobody recorded which interval of the shot was examined, so a shot's ABSENCE from such a list is not a negative
  **After:** 2 database intervals (curated lists); coverage unknown; absence unmeasured

- **Before:** 3 row(s) are in `text_mentions`, not in `events`: a text row is a LEXICON HIT in the operator logbook -- somebody wrote a word -- and is not an assertion that the phenomenon occurred, nor a claim about what any diagnostic showed
  **After:** 3 logbook word matches; not observations

- **Before:** 2 row(s) have no recorded time and were not considered for the window: unknown when, which is not the same as outside it
  **After:** 2 rows excluded: time unknown, not outside window

- **Before:** 1 source(s) ran over shot 199607 and recorded NO coverage -- actuator/ech_power_total: ran; coverage unknown -- says nothing about this window, neither that it was looked at nor that it was not
  **After:** actuator/ech_power_total: coverage unknown

- **Before:** no observed-event product for shot 199607: no relevant detector source is recorded as having completed over it. Returned event rows do not establish that a registered covering source ran. Absence is not evidence -- this is not a quiet shot, it is an unexamined one
  **After:** Shot 199607: no completed covering detector; unprocessed, not quiet

- **Before:** no source with recorded coverage ran over shot 199607 over [1.0, 2.0] s: the sources that completed did not record what span they read, so nothing establishes that anybody looked -- an empty result here is not an observation of nothing happening
  **After:** Shot 199607: coverage unknown over 1.000–2.000 s; absence unmeasured

- **Before:** 2 source(s) ran over shot 199607 and reported 0 detections inside their coverage over [1.0, 2.0] s. This IS an observation of nothing happening, unlike an unprocessed shot
  **After:** Shot 199607: 2 sources, 0 detections within coverage over 1.000–2.000 s

- **Before:** 1 source(s) FAILED on shot 199607: whatever they would have seen is missing from this reply
  **After:** Shot 199607: 1 sources failed; evidence missing

- **Before:** coverage recorded as a hull by an older writer; interior gaps unknown
  **After:** Legacy coverage hull; interior gaps unknown

- **Before:** elm_clock: coverage recorded as a hull by an older writer; interior gaps unknown
  **After:** elm_clock: Legacy coverage hull; interior gaps unknown

- **Before:** actuator/ech: ran; coverage unknown; absence is not evidence
  **After:** actuator/ech: coverage unknown; absence unmeasured

- **Before:** 2 event(s) not shown: the source recorded no confidence, so they cannot be shown to reach min_confidence 0.5
  **After:** 2 events excluded: confidence unrecorded; minimum 0.500

- **Before:** kept despite --avoid phenomenon:elm: nothing looked for ELM on this shot, so its absence is unmeasured, not established
  **After:** Kept with avoid phenomenon:elm: ELM unexamined; absence unmeasured

- **Before:** kept despite --avoid phenomenon:elm: no detector for ELM has run on this shot, so its absence is unmeasured, not established
  **After:** Kept with avoid phenomenon:elm: ELM detectors not run; absence unmeasured

- **Before:** kept despite --avoid phenomenon:elm: the ELM detectors ran on this shot but not over the window searched, so its absence there is unmeasured, not established
  **After:** Kept with avoid phenomenon:elm: ELM coverage outside window; absence unmeasured

- **Before:** kept despite --avoid phenomenon:elm: the ELM detectors covered only part of the window searched, so outside that its absence is unmeasured
  **After:** Kept with avoid phenomenon:elm: ELM coverage partial; outside unmeasured

- **Before:** --avoid phenomenon:elm: dropped 2 shot(s) with observed ELM evidence
  **After:** Avoid phenomenon:elm: excluded 2 shots with observed ELM

- **Before:** 2 of those drops rest on evidence that carries: observed via tokeye_transient, a class-agnostic transient detector: a sawtooth crash or a disruption precursor is transient too, so this is an ELM-LIKE TRANSIENT and not a classified ELM
  **After:** 2 excluded shots: Class-agnostic transient; may be ELM-like, a sawtooth or a disruption precursor

- **Before:** --avoid phenomenon:elm: excluded 5 flat_top segment(s) with unprocessed coverage; absence is not evidence
  **After:** Avoid phenomenon:elm: excluded 5 flat_top segments; unprocessed coverage, absence unmeasured

- **Before:** --avoid phenomenon:elm: detectors covered 85.5% of the flat_top window; absence outside that coverage is unmeasured
  **After:** Avoid phenomenon:elm: covered 85.5% of flat_top; outside unmeasured

- **Before:** the window [7.0, 8.0] s is outside every source's coverage of shot 199607, whose display hull is 0.0 to 6.0 s; covered intervals: [0.0, 2.0] s, [3.0, 6.0] s -- nobody looked there, so an empty result says nothing about the window you asked about
  **After:** Shot 199607: 7.000–8.000 s outside coverage; hull 0.000–6.000 s; covered intervals: 0.000–2.000 s, 3.000–6.000 s; requested window unmeasured

- **Before:** the frame-code cache for shot 199607 has no provenance sidecar: the device and thread count it was encoded on are not recorded, and the codes are not bit-reproducible across either
  **After:** Shot 199607: frame-code device and thread count unrecorded; codes vary with both

- **Before:** shot 199607 has no ramp_up segment; the description falls back to `full`
  **After:** Shot 199607: no ramp_up segment scalars

- **Before:** no channel had anything to search on -- give ref_shot, text, constraints or actuators
  **After:** Enter a reference shot, text, constraints or actuators

- **Before:** excluded for having no recorded value: ip_mean (5 shots)
  **After:** Missing values excluded: ip_mean (5 shots)

### Result flag wording and numbers: before → after

Flags use their structured raw `value` and `limit` through the shared number formatter. Remaining envelope bounds already rounded in the source message are formatted without inventing precision. Counts and rule identifiers retain their original digits.

- `ip_mean = 8.92e+05 > 1e+05: configured limit` → `ip_mean = 8.924e5 > 1e5: configured limit` (raw value 892400, limit 100000).
- `ip_mean = 8.92e+05 is above the observed 1e+03-1e+05 range of 199607 shots in the database -- outside what has been run, not necessarily outside what is possible` → `ip_mean = 8.924e5; above observed range 1000–1e5 (199607 shots); not an operating limit`.
- The envelope template also handles `below`, unknown bounds (`—`), and an unrecorded shot count (`database` instead of inventing a count). Unmatched flag text, including skipped-rule diagnostics, is retained.

### UI captions and headings: before → after

- `shot_design: DIII-D shots` (document title) → `Shot Designer`
- `shot_design` (banner) → `Shot Designer`
- `Loading database summary…` → `Loading shots…`
- `504 shots · 185786, 204925 · build 2d20398` → `504 shots · 185786–204925`
- `Number of results` → `Results`
- `Enter a query to search the database.` → `Enter a query.`
- `Inspect a shot` → `Open a shot`
- `Event phenomenon (exact name)` → `Event name (exact match)`
- `Start (seconds)` / `End (seconds)` → `Start (s)` / `End (s)`
- `Open a shot from a result or enter its number.` → `Enter a shot number or open a result.`
- `This filter matches stored event names exactly. Locate also uses registry rules: its phenomenon ID can differ from the event name. The all-event context below keeps those observations visible.` → `Exact event-name match. Locate uses registry classification; names may differ. All event names appear below.`
- `forecasts — a model's risk estimate, not an observation` → `Forecasts (model estimates)` (both Shot timelines and Locate)
- `Curated database intervals — entries in a list, not measurements` → `Database intervals (curated lists)`
- `Per-source coverage` → `Coverage by source`
- `Logbook claims; these are not events on the timeline.` → `Logbook mentions, not observations.`
- `All event names — context for registry classification` → `All event names (registry context)`
- `Number of hits` → `Hits`
- `Ranked hits` → `Results`
- `Choose a phenomenon to locate its evidence.` → `Choose a phenomenon.`
- `Quote` (results column) → `Summary`
- `No rows returned.` → `No indexed intervals`
- `Stored flags and groups` → `Flags and groups`
- `Logbook quotes` → `Logbook`
- Generic millisecond field labels `*_ms` → `* (s)` with corresponding unit conversion.
- Dense generated description paragraph → structured `Summary`, header, segment/grid, `Labels`, `Outcome`, `Operator quote`, `Phenomena`, `Run / mini-proposal` sections. Added phenomena columns: `Phenomenon`, `Observed`, `First intervals`, `Forecasts`, `Coverage`; extra observed rows read `+N more`. Partial coverage reads `Partial coverage; outside unmeasured`.
- Raw outcome keys → `Ip target`, `Ip error`, `NBI target`, `NBI error`, `NBI target units`, `Flat top`, `Ended early`, `Fast quench`, `End reason`, `End time`, `Faults`; target booleans become `hit` / `missed`, absent remains `—`, and end-reason underscores become spaces.
- Unbounded text/sections → `more` / `less` and `Show all` / `Show less` controls. Locate's quote/role block moves to the shot's attributed logbook; its new Summary slot displays offline text or `—`.

New browser label from the amendment: `auto` (small and muted), next to a present blurb whose source is not `llm`. The superseded `summaries.parquet unavailable: <error>` diagnostic was removed with that loader. The structured missing-segment caveat remains `No <segment> scalars recorded`. No existing shared evidence wording was changed.

## Original UI1 verification (before the Summary amendment)

### Test-first sequence

- Before API implementation: new summary/units/parts tests failed as expected (`24 failed in 2.99s`).
- After API implementation: focused transport/description/new-data checks: `85 passed in 6.45s`.
- Before formatter/wording implementation: `2 failed in 0.26s`, for the missing functions.
- Before remaining avoid/provenance wording maps: `1 failed, 1 passed in 0.29s`.
- Focused browser/data/API/interval checks: `59 passed in 10.04s`.
- An intermediate static-title check failed while the HTML edit had not yet applied; the subsequent transport checks passed (`34 passed in 4.54s`).

- Before final flag formatting: `1 failed, 3 deselected in 0.21s` (missing formatter).
- After flag formatting and remaining section bounds: `36 passed in 5.47s`.

### Required complete suites

Commands ran from `/scratch/gpfs/nc1514/FusionAIHub-UI1`:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/src HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e ideate-cpu python -m pytest tests/shot_design -q -W error -p no:cacheprovider -rs

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/src HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker python -m pytest tests/labeler -q -W error -p no:cacheprovider -rs
```

First full pass, before the final browser-only flag fix:

```text
1402 passed in 145.07s (0:02:25)
```

Final Shot Design pass:

```text
1403 passed in 130.91s (0:02:10)
0 skipped; exit 0
```

Labeler (unchanged by the final browser-only fix):

```text
SKIPPED [1] tests/labeler/test_l14perf_real_identity.py:45: set L14PERF_REAL_ROOT to an l14perf scratch directory
SKIPPED [1] tests/labeler/test_resolve_fdp.py:454: live fdp fetch is opt-in: --run-live or LABELER_FDP=1
SKIPPED [1] tests/labeler/test_resolve_fdp.py:476: live fdp fetch is opt-in: --run-live or LABELER_FDP=1
1775 passed, 3 skipped in 293.47s (0:04:53)
```

Both commands exited 0 with `-W error`. After labeler pytest completed, XRootD's interpreter-finalization callback printed this upstream shutdown warning (outside the pytest summary):

```text
/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/lib/python3.11/site-packages/XRootD/client/finalize.py:46: FutureWarning: `torch.distributed.reduce_op` is deprecated, please use `torch.distributed.ReduceOp` instead
  if isinstance(obj, File) and obj.is_open():
```

Pixi also emitted its existing network-filesystem cache notice and redirected its repodata cache to `/tmp/pixi-cache-nc1514/repodata`. All invocations used `--frozen --no-install` and the main checkout manifest.

### Syntax and lint

```text
node --check src/shot_design/ui/static/app.js
(exit 0, no output)

/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/ruff check --no-cache src/labeler src/shot_design scripts/labeler tests/labeler tests/shot_design
All checks passed!
```

### Served-page API checks (read-only production)

Started the current worktree source with the required frozen/no-install main-manifest `ideate-cpu` command, `python -m shot_design serve --port 8768`. Used httpx against localhost; the token gate first returned 401, then accepted the token and served the title/assets. Read-only request output:

```text
meta: n_shots=504, shot_range=[185786, 204925], n_segments=1993, git_sha=2d20398
shot: 199607 summary: None units: 160
parts: summary, header, segment, scalars, labels, outcome, operator_quote, phenomena, caveats
scalar units: ip_mean=A, ip_slope=A/s, ne_line_mean=m/cm3
phenomena: ae (244 observed), eho (10), elm (740), transient (466),
          tearing (140 observed / 42 forecasts), sawtooth (50), fishbone (69),
          qcm (126), qmin_elevated (2), qmin_high (1)
events: observed 2009; forecasts: 44
search: [(199790, None), (187071, None)]
locate: [(186535, observed), (186528, observed)]
Served API smoke: PASS; no database writes
```

The browser's existing Node renderer harness verifies disjoint coverage bars. Added dependency-free DOM tests execute the real rendering functions for summary presence/absence, scalar units, time and confidence formatting, whole shot IDs, event tooltip content, absence of bullet lists, full-text expansion, Show all/less and click propagation. Operator text is inserted as text nodes.

Final served HTML, JS and CSS checks also passed after the flag formatter change. The temporary server then shut down cleanly (exit 0). `git diff --check` passed; no `.pixi`, Python bytecode, pytest cache or Ruff cache was found in the worktree.

## Boundaries and remaining work

- Offline language-model blurb generation remains the follow-up task. The UI displays the existing factual template blurb with `auto` until LLM text arrives in that same column; missing text still displays `—` and hides the shot Summary block.
- No browser engine is installed in this environment, so no pixel screenshot/manual browser layout review was performed. Node syntax/rendering checks and the served HTTP API were exercised; CSS layout/focus behavior was reviewed in source.
- Production DB, labeler root, datasets and model artifacts were read-only. Test files were written only through existing/new tmp_path fixtures. No installs, lock/update commands, worktree `.pixi`, or generated runtime artifacts were added. No edits under `docs/superpowers/**`.
- No MCP tool descriptions or event-status logic changed. The amendment replaces the optional summary response field and summary-table error caveat in `mcp/tools.py` with `blurb` and `blurb_source` response fields.
- Commits use `shot_design:` and the required `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` trailer. No commits occurred while a suite was running.

## Summary amendment verification

The user amended item 4 after the original UI1 commits: render the existing
`shots.parquet` blurb/source fields and a muted `auto` tag for non-LLM text.
The brief, current contracts and documentation above now reflect that amendment.
The original verification transcript is retained above as historical evidence;
its `summary` keys and absent-summary production observations describe the
superseded implementation.

Tests were updated before implementation. The first focused run failed as
expected: `13 failed, 19 passed in 3.99s`, covering missing record/hit fields,
the old sidecar loader/API and the browser still reading `summary`. The
absent-column fixture was then corrected to remove the template columns already
present in the shared synthetic DB. After implementation, the focused API and
Node renderer tests passed: `32 passed in 3.53s` with `-W error`.

The new tests exercise authenticated `/api/search` and `/api/shot` calls through
the existing httpx TestClient, using tmp_path tables. They verify stored LLM,
template and unknown sources; absent/null/blank text; both hit models; read-only
column hydration; and ignoring an obsolete sidecar. The Node harness executes
the real render functions for Search, Locate and Shot, verifying the text and
the presence/absence of the small muted `auto` tag for each source.

Full shot_design verification (exit 0):

```bash
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e ideate-cpu env PYTHONPATH=/scratch/gpfs/nc1514/FusionAIHub-UI1/src PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 pytest /scratch/gpfs/nc1514/FusionAIHub-UI1/tests/shot_design -q -W error -p no:cacheprovider -rs
```

```text
1407 passed in 135.67s (0:02:15)
0 skipped; exit 0
```

The full labeler result follows after completion. Every actual browser JS module
passed `node --check`. Ruff passed across `src/labeler src/shot_design
scripts/labeler scripts/shot_design tests/labeler tests/shot_design` using the main
checkout's labelmaker Ruff binary with `--no-cache` (`All checks passed!`).
No production server, database, LLM or Slurm operation was needed for this amendment.

## Final disposition

All twelve brief items are implemented. Required suites and lint/syntax checks passed. The worktree contains source, tests, GUI documentation and this report only; runtime checks used tmp_path or read-only production access. The report and documentation are committed after suite completion.
