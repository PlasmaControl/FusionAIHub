# Review — task C3fix (`recommender-C3fix`, 3 commits, 25 files, +1408/-174)

**VERDICT: MERGE WITH FIXES: (1) L→H `pinj` coverage silently changed from any-channel to
all-channel finiteness — untested, undocumented, can empty `dalpha_lh` coverage on a real shot;
(2) `event_sources.legacy_hull` raises on an ndarray-valued `intervals` column; (3) `text`
source coverage dropped to `[]` without mention in report or docs.** The cap path itself is
genuinely fixed end to end and the regression chain is real, hermetic and would have failed on
9bc7ce6. Nothing under `docs/superpowers/plans/**` changed. All three requested gates are green
from the worktree.

---

## Findings

### 1. MEDIUM — L→H `pinj` coverage silently narrowed from any-channel to all-channel
`src/labelmaker/events/pipeline.py:1178-1184`

```python
coverage.Coverage.measured(
    pinj_t_s, np.asarray(pinj_y).sum(axis=0), min_gap_s=LH_MIN_GAP_S)
```

Before, the same input was `coverage.finite_span(pinj_t_s, pinj_y)` — a 2-D trace, sample counted
where **any** channel is finite (`coverage._finite_mask`, coverage.py:78). `sum(axis=0)` is NaN
wherever **any** beam channel is NaN, so this is the opposite rule. Consequences:

* One all-NaN beam channel in the 8-channel `pinj` group empties the whole set →
  `dalpha_lh` row gets `intervals == "[]"`, hull NaN → `CoverageSummary.from_rows` puts it in
  `unknown` (event_sources.py:172-175) → every L→H window answers `uncovered` on that shot.
* The detector does not need every beam: its gate is `np.any(pinj >= min_pinj_kw)`
  (`heuristics.py:855`), i.e. one beam above threshold suffices.
* Undocumented — `docs/LABELMAKER.md` (new coverage section) says "Required inputs intersect
  their interval sets … any-channel steps use their union"; nothing says pinj switched rule.
  The report's `min_gap_s` table does not mention it either.
* Untested — `tests/labelmaker/test_events_pipeline.py:1489-1516` NaNs **all** pinj channels over
  `[.2, .25]`, which any-channel and all-channel both exclude, so the change is unpinned.

Why it matters: this is the opposite failure mode to the one being fixed (valid observation →
`uncovered`), on the 504 shots that will be regenerated after this merge.

Fix: pass `pinj_y` directly (`Coverage.measured(pinj_t_s, pinj_y, min_gap_s=LH_MIN_GAP_S)`) to
restore the prior semantics, or keep the sum and (a) say so in `docs/LABELMAKER.md`, (b) add a
test with one channel NaN and another finite, (c) check a real `pinj` group for all-NaN channels
before regenerating.

### 2. LOW — `legacy_hull` returns an array (then raises) for a list-typed `intervals` column
`src/ideate/labels/event_sources.py:76-79`

```python
return value is None or (not isinstance(value, (str, list, tuple)) and pd.isna(value))
```

Verified in the worktree (`-e ideate-cpu`): with `row["intervals"] = np.array([[0.,1.],[2.,4.]])`
this returns `array([[False, False],[False, False]])`, so the caller's `if legacy_hull(row):`
(`row_intervals`, :86) raises `ValueError: truth value of an array … is ambiguous`. `row_intervals`
itself *does* accept list-valued intervals (`Coverage(intervals, 0.0)` at :93), so the two
functions disagree about which encodings are supported. Any parquet that stores `intervals` as a
real list column instead of the JSON string — the other persistence encoding a future writer might
pick — breaks every coverage read. Fix: `value is None or (isinstance(value, float) and pd.isna(value))`,
or `np.ndarray` in the isinstance tuple.

### 3. LOW — `text` source coverage silently dropped; not in the report or docs
`src/labelmaker/events/pipeline.py:1287`

`ran[("text", "", -1, "")] = coverage.Coverage((), 0.0)` replaces
`text_weak.shot_span_s(shot, paths=paths)`. The persisted `text` row now has NaN `t_cov0_s/1_s`
and `intervals == "[]"` instead of the shot span. Harmless for coverage decisions (`text` is in
`NON_DIAGNOSTIC_SOURCES`, event_sources.py:119), but it is a visible change in every regenerated
`_sources.parquet` and in the MCP `coverage.sources` block, no test pins it, and the report only
claims "Curated sources remain non-diagnostic, with NaN hulls". Either keep the span (display-only
anyway) or record the change.

### 4. LOW — the "one shared coverage path" is now two implementations
`src/ideate/mcp/tools.py:601-615` vs `src/ideate/retrieval/phenomena.py:840-846`

`get_events` re-derives `_merge`/`_clip`/`partial` inline instead of calling `ph._coverage_for`.
The two are not identical in shape (`_coverage_for` takes a `(t0, t1)` tuple or `None`;
`get_events` allows one-sided bounds), and `get_events` never emits `COVERAGE_GAPS`
(phenomena.py:142) while `_coverage_for` does. They agree today only because
`tests/ideate/test_coverage_agreement.py:33-83` asserts it. The brief asked for one shared path;
this is a drift risk, not a current defect. Fix: give `phenomena` a helper taking
`(t0|None, t1|None)` and call it from both.

### 5. LOW — `format_intervals` rounds coverage boundaries in user-visible caveats
`src/ideate/labels/event_sources.py:100-101` — `f"[{a:g}, {b:g}] s"` is 6 significant digits, so
`-0.049879997…` prints `-0.04988` and a boundary near 100 s rounds to 1e-4 s. These numbers appear
in the `uncovered` caveat (tools.py:626-631) and the partial caveat, i.e. exactly where a caller
compares them with their own window. Use `:.4f`/`:.6g` or the same `.3f` the existing
`_coverage_for` partial caveat uses (phenomena.py:850).

### 6. LOW — every event row now carries the full interval set in `attrs`
`src/labelmaker/events/coverage.py:286-300` (`attach_intervals`), consumed by
`src/ideate/shotdb/store.py:212-220`. The fallback it enables is correct, but the payload is
O(intervals) per event: an ELM clock emitting several thousand points on a source with many finite
runs repeats the same JSON list on every row. Consider storing it once (e.g. only on the first row
per key) or accept and note the size. Not a correctness issue.

### 7. NIT
* `src/ideate/cli.py:1032-1045` — the `hand_review: +N (...), -M` line is printed on **every**
  `corpus select` run, not only with `--txt-out`. Harmless but it is a new unconditional stdout
  line that any caller parsing this output will see. Everything else about `corpus select` is
  unchanged (the `shots:` cohort and the carried `hand_review` block are untouched —
  `tests/ideate/test_select.py:1535-1536`).
* `src/ideate/labels/event_sources.py:389-406` — `__all__` was not extended with
  `LEGACY_HULL_CAVEAT`, `row_intervals`, `legacy_hull`, `format_intervals`,
  `with_interval_columns`, although `phenomena.py` and `tools.py` use them.
* `coverage.feature_spans` now has no production caller (only tests). Retained deliberately per the
  brief; flagging so it is not mistaken for live code.
* `src/labelmaker/events/schema.py:374` does `from .coverage import Coverage` inside the function
  although there is no import cycle (coverage.py imports nothing from schema).

---

## Answers

**1. `finite_intervals` / intersection / union correctness.**
`coverage.py:116-152`. Intervals are **closed** `(first finite sample, last finite sample)`.
That is consistent with the rest of the system because the ideate overlap tests are closed too —
`coverage_state` (`hi >= t0_s and lo <= t1_s`, event_sources.py:184-185), `observing_rows` (:353-355)
and the MCP event filter (`t1_s >= t0`, `t0_s <= t1`, tools.py:586-588) use the identical rule, so
the invariant "a window the filter would return an event from is never `uncovered`" holds. There is
no half-open/closed mismatch introduced here; `windows.py:251`'s half-open tiling is a separate,
unrelated concern.
`min_gap_s`: merge iff `lo - prev_hi < gap` (:148) measured last-finite-before to first-finite-after,
so a gap **shorter** than `min_gap_s` merges and one equal or longer splits — matches the brief, and
`0.0` bridges nothing (`t[i+1]-t[i] > 0`). Tested at `tests/labelmaker/test_events_coverage.py:387-402`.
Constants are the detectors' own and verified present with those values: `transients.MIN_DISTANCE_MS = 3.0`
(transients.py:60), `heuristics.STEP_SPAN_MS = 8.0` (:97), `LH_DROP_WINDOW_MS = 5.0` (:116),
`ACTUATOR_MIN_MS = 20.0` (:177), wired at pipeline.py:84-92. All are ≥3 orders of magnitude below
the fixture's 1 s dropout, so none is fitted to it. The TokEye value is one grid column
(`_block_coverage`, pipeline.py:96-99), justified because `masks.read_waveform` refuses interior
non-finite samples (`masks.py:146-150`) — verified, so hull == intervals there.
Edges: leading/trailing NaN produce no interval (:129-132, test :417-425); all-NaN and empty axis →
`()` (:138, test :428-433); single sample → one degenerate `(t, t)` interval; non-monotonic finite
samples raise (:140-141); `min_gap_s` NaN/negative raises (`_check_min_gap`, :103-113).
`union_intervals` merges touching (`lo <= prev_hi`, :189) and validates finiteness/order;
`intersect_intervals` is the standard two-pointer sweep (:196-208), returns `()` for any empty input
and for no input (:223-229), and `Coverage.intersect` keeps the **coarsest** `min_gap_s` via `max`
(:282). Determinism: plain float comparisons, no tolerances anywhere in this module; the only
epsilon (`phenomena._EPS = 1e-9`) is on the partial-coverage test and predates this branch.

**2. Persistence.**
`SOURCE_KEY = ("source","diag","channel","pass_name")` (schema.py:338) is **unchanged**; design (b)
was chosen, one row per key, so `write_sources`' replacement semantics (schema.py:432-446) and the
"two records for one key" guard (:429-431) are untouched. Merge across shapes is tested at
`tests/labelmaker/test_events_coverage.py:505-528`: an old hull-only file merged with a new row keeps
`min_gap_s` NaN on the old row and `intervals == "[]"` on the new one, two rows out.
`schema._source_row:373-378` derives `t_cov0_s/t_cov1_s` from `Coverage.hull` whenever `intervals` is
present, so the display hull can never disagree with the set. `read_sources` back-fills the two
columns when absent (:471-477), `ideate.labels.event_sources.read_sources`/`with_interval_columns`
(:66-73, :258-263) do the same on the ideate side, and `_frame` (:191-193) preserves the dtype
contract, asserted at `tests/ideate/test_event_sources.py:39-50` and
`tests/labelmaker/test_events_schema.py:369-380`. The join passes the columns through untouched —
`join.py` only comments the guarantee (:566-568) and uses `es.sources_union`, which is a concat of
`read_sources`. Robustness of the JSON column: `None`/NaN/absent column = legacy (finding 2 is the
one hole); explicit `"[]"` is authoritative and is **not** silently upgraded to the hull
(`row_intervals` docstring and :86-95, tested at `test_event_sources.py:40-41` where a stale finite
hull with `intervals="[]"` yields `n_sources_unknown_coverage == 1` and `coverage_span is None`).
Backward compatibility is explicit and produces a CAVEAT, not a silent hull: `LEGACY_HULL_CAVEAT`
(`event_sources.py:64`) is surfaced in `_coverage_for` (phenomena.py:828-829), in `get_events`
(tools.py:616-617), in `describe` (describe.py:338) and in the UI (app.js:190-192), tested for source
tables, old joins and old event-only DBs at `test_coverage_agreement.py:141-163` and
`test_event_sources.py:44-52`.

**3. Consumers, end to end.**
`get_events(shot, ph, t0, t1)` → `es.for_shot` (event_sources.py:151, reads `db.coverage_sources`) →
`es.coverage_state` (:180-189, interval sets only) → `CoverageSummary.from_rows` +
`ph._merge`/`ph._clip` for the windows and the partial flag (tools.py:601-615).
(a) inside a gap of every covering source → `uncovered`, caveat names the covered intervals
(tools.py:626-632; retrieval equivalent phenomena.py:837-838). Verified by
`tests/ideate/test_coverage_agreement.py` `[flat_top 1.2–1.8]`.
(b) crossing an edge → `observed` + `coverage_partial=True` + `COVERAGE_PARTIAL` "covered only …"
(tools.py:605-615). Verified `[ramp_up 0.5–1.5]`.
(c) whole record (no window) → `observed`, and now **`coverage_partial=True` whenever the merged
coverage is disjoint** — phenomena.py:844-846 dropped the `window is not None` guard, matching
tools.py:606. Pinned by `test_whole_record_searches_agree_that_interior_gaps_are_partial`
(test_coverage_agreement.py:120-127). Note for the controller: this widens `coverage_partial=True`
across the corpus once production is regenerated; it is the honest direction but it is an
answer-changing behaviour, not only a gap fix.
`evidence`/`locate`/`describe` all go through the single `_coverage_for` (phenomena.py:986, :1339,
describe.py:332), so they agree by construction; `get_events` is the second implementation (finding 4).
`grep -rn "finite_span\|t_cov0_s\|t_cov1_s" src/`: no consumer still decides coverage from the hull.
The surviving hits are (i) labelmaker **event-row** coverage columns and clip bounds
(heuristics.py:573/947/1094/1306/1425, transients.py:441/564, tracks.py:679, text_weak.py:666,
databases.py:419/431) — display/clip, explicitly allowed by the brief; (ii)
`event_sources.py:81` — the caveated legacy path only; (iii) `tools.py:763-774` — the display block,
now labelled "display hull" in the caveat text; (iv) `store.py:208` — a groupby key for the
event-row fallback, superseded by `intervals` when present; (v) `app.js:189` — a defensive branch
that the new payload never reaches (`intervals` is always a list). `pipeline.py` has **zero**
remaining `finite_span` / `feature_spans` / `coverage.intersect` calls (grep clean).

**4. `shotdb/store.py`.**
Fix confirmed at `store.py:209-221`: `evidence_kind in ('detector','heuristic')` rows are copied,
`attrs.coverage_intervals` / `coverage_min_gap_s` are lifted into `intervals` / `min_gap_s`, and both
are added to the groupby keys so `es.source_row(**row)` reconstructs a row with real intervals;
legacy rows with no such attrs get `None` → caveated hull. `_EVIDENCE_COLUMNS['coverage_sources']`
was extended to carry the two columns through `evidence_rows` (:39-41). Nothing else in the module
changed — `segments`, `shapes`, the embedding matrices, `_coverage_positions` and the token/mask
algebra are byte-identical (`git diff` shows exactly these two hunks). Pinned by
`test_new_event_fallback_cannot_reintroduce_the_pipeline_hull`
(test_coverage_agreement.py:110-118), which deletes `event_sources.parquet` and still gets
`uncovered` with no older-writer caveat.

**5. UI `app.js`.**
`app.js:184-195` draws one bar **per interval** from the MCP payload's
`coverage.sources[].intervals` (tools.py:774-776), skips non-diagnostic sources by the same
`text`/`database`/`database:` rule as `es.is_non_diagnostic`, falls back to the hull only when
`intervals` is not an array, and attaches the legacy caveat text; `min_gap_s` was added to the
hover detail (:156). The contract matches the payload. `tests/ideate/test_ui_intervals.py` is **not**
a string match: it loads the real `app.js` into a Node `vm` with a stub DOM, calls the real
`renderEvents` with a three-source payload, and asserts the emitted mark geometry
(`left:0%;width:25%` and `left:50%;width:50%` on the domain `[0,4]`) — two bars, not three, and the
empty interval set and the skipped row draw nothing. It `pytest.skip`s when Node is absent; Node was
present in my run (341 passed, 0 skipped), so it did execute.

**6. Deliverable 2.**
`cli.py:1032-1048`: `effective = (generated - drop) | add`, which is exactly
`config.load_shot_list`'s order (config.py:326-329, drops then adds, so a shot in both survives).
Written sorted numerically and inherently deduplicated (set); `hand_review: +2 (190998 190999), -1 (190003)`
format matches the brief. `shots:` and the carried `hand_review` block are untouched
(`test_select.py:1535-1536`). The regression asserts equality with `load_shot_list` on the emitted
YAML (:1539) and the drop/re-add precedence (:1540). The only other output change is that the
`hand_review:` line is printed unconditionally (nit 7). Minor divergence: the CLI accepts bare-int
review entries, `load_shot_list` does not — pre-existing asymmetry, superset, harmless.
Production cross-check (read-only, no writes): `configs/ideate/shot_lists/recommender_v1.yaml` has
500 generated, drop `[]`, add `[190736, 199597, 199607, 200729]` → 504 effective;
`/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/recommender_v1.txt` has 504 lines, sorted and unique, and
is **exactly equal** to what the new code would write (no missing, no extra).

**7. Tests.**
The real chain exists in the critic's shape:
`tests/ideate/test_coverage_agreement.py:13-41` (`gap_chain`) builds a corpus-layout HDF5 with only a
`filterscopes` group on a 2^16+1, 10 kHz axis, all 8 channels NaN over `[1, 2]` s plus NaN first and
last sample (`tests/labelmaker/coverage_fixture.py:1-21`), runs the **real**
`pipeline.process_shot(198658, …, model=None)`, then the **real** `cli.main(["labels","join", …])`
into a tmp DB, then `tools.get_events` (:44-83), `evidence`/`_coverage_for`/`locate`/`describe`
(:66-78) and a **real `python -m ideate.mcp` stdio subprocess** (:86-107, the `tests/ideate/test_mcp.py`
pattern). Hermetic: `Paths(root=tmp_path/"products", corpus=tmp_path/"corpus")`,
`monkeypatch.setenv("LABELMAKER_ROOT", …)`, `ideate_db` is `tmp_path/"ideate"` with
`IDEATE_DATA_ROOT` monkeypatched and `IDEATE_PATHS` deleted (conftest.py:758-768); the subprocess env
is built from `get_default_environment()` with `IDEATE_DATA_ROOT`/`LABELMAKER_ROOT` pointed at
tmp_path and `IDEATE_PATHS` popped. No changed or new test reads `/scratch/gpfs/EKOLEMEN/` (the grep
hits are pre-existing, unrelated modules that skip when the stores are absent). All gates ran
`-W error`.
Would it have failed on 9bc7ce6: yes, twice over —
`tests/labelmaker/test_events_pipeline.py:57-70` asserts `"intervals" in clock` with the message
"the source writer still publishes only a hull", and `SOURCE_COLUMNS` had no such column at the base;
and the `[flat_top]` parametrisation asserts `uncovered` where the base answered `observed, n=0`.
The two former cap paths are re-pinned at `test_coverage_agreement.py:224-241` (sawtooth 14–15 s
`uncovered` despite gas to 94.9 s; QH 3–4 s `unprocessed`).
No test looked tuned to the fixture: expected intervals come back from the fixture builder rather
than being hard-coded, the min_gap tests use the detector constants
(`test_events_coverage.py:398`, `test_events_pipeline.py:67`), and the UI geometry is derived from
the payload domain.

**8. Report honesty.**
Worktree `git status --porcelain` shows exactly one entry, the untracked brief
`.superpowers/sdd/task-C3fix-brief.md`. `git diff -- pixi.lock` in the worktree is **empty** — the
lock is at HEAD, i.e. the restoration the report describes is real. There is **no** `.pixi`
directory anywhere in the worktree (`find . -maxdepth 3 -name .pixi` → nothing), so the
"worktree-local `.pixi` artifacts" the report observed are gone. Blast radius on the shared envs
looks nil: `/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/{labelmaker,ideate-cpu}` are dated Sep 4 and
Sep 7, `ideate` Sep 14 00:15, all before the task's 11:29 start; the main checkout's `pixi.lock` is
`M` but its mtime is 10:16, predating the task and matching the pre-existing modification recorded
in both the session snapshot and the critic's own honesty section. So the disclosed deviation left
no detectable residue — the disclosure stands and I found nothing it hides.
Claims vs diff: I could verify every substantive one. The `min_gap_s` derivation table matches the
constants (see answer 1); "no `finite_span`/`feature_spans`/scalar `intersect` remains in
pipeline.py" is true by grep; "TokEye … waveform loading already refuses interior NaNs" is true
(masks.py:146-150); "coarsest input resolution" is `max` (coverage.py:282); "the browser renders
separate source bars" and "`shotdb/store.py` was necessary" are both in the diff. Two omissions
rather than false claims: the `pinj` any→all change (finding 1) and the `text` span removal
(finding 3) are not mentioned anywhere.
Red→green counts are plausible and exact: report says labelmaker 1580 → 1596 (+16) and ideate
1176 → 1192 (+16). Counting collected items in the diff: labelmaker 12 new in
`test_events_coverage.py` + 4 in `test_events_pipeline.py` = 16; ideate 9 items in
`test_coverage_agreement.py` (one 3-way and one 2-way parametrisation) + 1 `test_coverage_diags` +
2 `test_event_sources` + 2 `test_phenomena_deferrals` (parametrised) + 1 `test_select` +
1 `test_ui_intervals` = 16. Both match.

**9.** `git diff --name-only 9bc7ce6..recommender-C3fix -- 'docs/superpowers/plans/**'` → empty.
No plan file changed. All three commit messages end with
`Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` as the brief required.

---

## Commands I ran (from `/scratch/gpfs/nc1514/FusionAIHub-C3fix`)

```
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/src HF_HUB_OFFLINE=1

pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml \
  -e labelmaker python -m pytest tests/labelmaker/test_events_coverage.py \
  tests/labelmaker/test_events_pipeline.py tests/labelmaker/test_events_schema.py \
  -q -W error -p no:cacheprovider
  -> 168 passed in 30.91s        exit 0

pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml \
  -e ideate-cpu python -m pytest tests/ideate/test_coverage_agreement.py \
  tests/ideate/test_coverage_diags.py tests/ideate/test_event_sources.py \
  tests/ideate/test_phenomena.py tests/ideate/test_phenomena_deferrals.py \
  tests/ideate/test_select.py tests/ideate/test_ui_intervals.py tests/ideate/test_mcp.py \
  -q -W error -p no:cacheprovider
  -> 341 passed in 22.95s        exit 0

pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml \
  -e labelmaker ruff check src/labelmaker src/ideate scripts/labelmaker tests/labelmaker tests/ideate
  -> All checks passed!          exit 0
```

No skips in either run (the Node-dependent UI test executed). Read-only throughout: no edit, stage,
commit, checkout, rebase or merge; no `ideate build/add/labels join/corpus …` and no
`labelmaker.run` outside the test suites; the only `/scratch/gpfs/EKOLEMEN/` access was a read of
`recommender_v1.txt` for answer 6.
