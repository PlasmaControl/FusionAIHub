# C3fix implementation report

Worktree: `/scratch/gpfs/nc1514/FusionAIHub-C3fix`; branch: `recommender-C3fix`.
Base: `9bc7ce6`. The task brief and critic defect 1, reproduction, and must-change
items 1 and 3 were read before designing the implementation.

## Deliverable 1: finite interval coverage

Chose persistence design **(b)**: append JSON `intervals` and float64 `min_gap_s`
to the source schema. One row still represents one `(source, diag, channel,
pass_name)`; merge/replacement keys and source counts stay unchanged. The hull
columns remain useful display bounds but cannot determine observation. JSON also
matches the existing event-attribute persistence convention and passes unchanged
through the ideate join.

The inherited edits were **kept and extended**, not accepted without review.
The initial diff was inspected and its test module passed **34 tests**. Its finite
run, union/intersection, and immutable `Coverage` implementation matched the
brief. I revised the stale hull documentation and pairwise loop, added measured
coverage/intersection methods, source persistence and event fallback metadata,
and extended its tests. The previous agent's eleven tests remain.

All pipeline `ran` assignments now publish coverage objects. No `finite_span`,
`feature_spans`, or scalar `coverage.intersect` call remains in pipeline.py.
The per-source gap constants are beside the detector orchestration and derive
from the detector constants, never the controlled dropout:

| Source | `min_gap_s` derivation |
| --- | --- |
| ELM clock | `transients.MIN_DISTANCE_MS / 1000` = 0.003 s |
| Sawtooth | `heuristics.STEP_SPAN_MS / 1000` = 0.008 s |
| L→H | `heuristics.LH_DROP_WINDOW_MS / 1000` = 0.005 s |
| Actuators | `heuristics.ACTUATOR_MIN_MS / 1000` = 0.020 s |
| Features and q-min | 0: individual-sample evaluation bridges no missing samples |
| TokEye tracks/transients | one column of the pass's time grid; waveform loading already refuses interior NaNs |
| QH/counter-injection | coarsest participating input resolution; required sets intersect; QH unions accepted track blocks first |

Tests cover leading/trailing NaNs, all-NaN inputs, short/long gaps, multi-input
holes, q-min/feature holes, disjoint accepted track blocks, persistence/merging,
old source shapes, and registry eligibility. Event extents retain outer-hull
clipping; an extent spanning a gap is not split at that gap.

`CoverageSummary`, `coverage_state`, `covers`, `coverage_span`, `observing_rows`,
and source summaries use interval sets. Retrieval evidence/locate/avoid/describe
and MCP preserve them, including whole-record partial coverage. The browser
renders separate source bars. Known interval metadata is also carried in event
`attrs.coverage_intervals` and `attrs.coverage_min_gap_s`; the optional event-only
fallback reads it instead of reviving a known gap from event hulls. Empty or
unreadable source tables remain authoritative.

Rows without interval metadata retain the old single finite hull interpretation
and disclose: **coverage recorded as a hull by an older writer; interior gaps unknown**.
Tests cover old source files, old joins, old event-only databases, new event-only
fallbacks, and explicit empty interval sets with misleading finite display bounds.
Curated sources remain non-diagnostic, with NaN hulls.

### Red → green evidence

All commands below ran from the worktree. The prescribed environment setup was:

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/src
```

Writer red (after correcting a fixture assertion from `error is None` to the
actual empty-string success contract):

```bash
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker python -m pytest tests/labelmaker/test_events_pipeline.py::test_real_clock_persists_the_finite_intervals_around_a_dropout -q -W error -p no:cacheprovider
```

Failed: **the source writer still publishes only a hull**. The helper regression
also failed with `observed != uncovered` for an explicit interval-set gap.

Exact controlled chain:

```bash
HF_HUB_OFFLINE=1 pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e ideate-cpu python -m pytest tests/ideate/test_coverage_agreement.py -k 'real_pipeline or real_stdio' -q -W error -p no:cacheprovider
```

After fixture setup corrections (valid segment names and an empty tmp labels
directory), **4 failed**: the in-process and actual stdio MCP gap answers were
`observed`, and joined source rows had no intervals. The green capture used the
same command with `-s`: **4 passed, 31 deselected in 6.40s**, exit 0.
The synthetic corpus has only eight filterscope channels on a 10 kHz, 2^16+1 axis;
all channels are NaN over [1,2] s, and the first/last samples are NaN. The real
`pipeline.process_shot` produces two coverage intervals, **[-0.0499, 0.9999] s**
and **[2.0001, 6.5035] s**. No model or clock substitution is used. Real CLI
`labels join` writes the fixture DB. There are 35 ELM points and 26 ELM-free
intervals, with no event intersecting 1.2–1.8 s.

The in-process outputs, verbatim:

```json
{"window": [1.2, 1.8], "status": "uncovered", "coverage_partial": false, "caveats": ["the window [1.2, 1.8] s is outside every source's coverage of shot 198658, whose display hull is -0.0499 to 6.5035 s; covered intervals: [-0.0499, 0.9999] s, [2.0001, 6.5035] s -- nobody looked there, so an empty result says nothing about the window you asked about"]}
{"window": [0.5, 1.5], "status": "observed", "coverage_partial": true, "caveats": ["the Edge localised mode detectors covered only [0.5, 0.9999] s of the requested window; absence outside that is unmeasured"]}
{"window": [2.5, 3.0], "status": "observed", "coverage_partial": false, "caveats": []}
```

The actual `python -m ideate.mcp` stdio subprocess answers the same three
questions with states `[uncovered, observed, observed]`, partial flags
`[false, true, false]`, and the covered-interval/partial caveats. The transport
regression checks `is_error=False` and uses this worktree's `PYTHONPATH`.
`evidence`, `_coverage_for`, `locate`, and `describe` are tested on the identical
windows. New event-only fallback red was `observed != uncovered`; green preserves
the gap without an older-writer caveat.

Independent review found two additional gaps, each reproduced before fixing:
Node execution of the actual timeline rendered three full-hull bars instead of
two interval bars; unbounded MCP said partial while whole-record retrieval did
not. Both regressions now pass and the reviewer confirmed the fixes read-only.
The rendering test requires Node and ran here; it did not start a server.

The two former cap controls are retained and explicitly rechecked: sawtooth at
14–15 s is `uncovered` despite gas extending to 94.9 s; QH at 3–4 s is
`unprocessed` when its own source is skipped for missing Ip flat-top. The existing
writer-side missing-flat-top and coverage-diag tests remain.

## Deliverable 2: effective text work order

`--txt-out` writes sorted `(generated − drops) ∪ adds`; stdout names effective
additions and drops separately. `shots:` and the carried `hand_review` block
remain unchanged. Drops followed by adds preserve an explicitly re-added shot.
Bare integer review entries accepted by older selection documents still work;
the new regression uses the structured entries consumed by `load_shot_list`.

The initial regression was in `tests/ideate/test_select_effective_list.py`, then
moved into `test_select.py` to reuse its fixture without imported-fixture lint
suppression. Red command:

```bash
HF_HUB_OFFLINE=1 pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e ideate-cpu python -m pytest tests/ideate/test_select_effective_list.py -q -W error -p no:cacheprovider
```

Failed because the text retained dropped 190003 and omitted the additions.
The green selection run passed **117 tests**. Final test location:

```bash
HF_HUB_OFFLINE=1 pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e ideate-cpu python -m pytest tests/ideate/test_select.py::test_finalize_txt_out_applies_review_and_names_the_effective_changes -q -W error -p no:cacheprovider
```

The regression checks equality with `config.load_shot_list` on the emitted YAML,
unchanged generated rows, sorted unique IDs, drop/re-add precedence, and stdout
`hand_review: +2 (190998 190999), -1 (190003)`.

## Deliverable 3: documentation and this report

Updated `docs/IDEATE.md` “Coverage: the four states” and LABELMAKER's coverage,
ELM, q-min, and source-schema text. Documentation red was a Python assertion
requiring `min_gap_s` and the exact older-writer caveat in both docs plus existence
of this report; it failed on `docs/IDEATE.md` before the edits. Final check:

```bash
python3 - <<'PYDOC'
from pathlib import Path
for name in ('docs/IDEATE.md', 'docs/LABELMAKER.md'):
    text = Path(name).read_text()
    assert 'min_gap_s' in text
    assert 'coverage recorded as a hull by an older writer; interior gaps unknown' in text
assert Path('.superpowers/sdd/task-C3fix-report.md').exists()
PYDOC
```

## Full verification

The first full pass found only three outdated assertions: new event coverage
attributes were excluded by an exact key-set check; MCP span wording changed;
an older-event fixture expected no gap caveat. The assertions were extended to
verify the new attributes and both modern/legacy caveat behavior, and MCP retains
its display-hull wording alongside the actual intervals. Those runs were
**1594 passed / 1 failed / 2 skipped** and **1186 passed / 2 failed**.

Final exact gates:

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/src
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker python -m pytest tests/labelmaker -q -W error -p no:cacheprovider
HF_HUB_OFFLINE=1 pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e ideate-cpu python -m pytest tests/ideate -q -W error -p no:cacheprovider
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker ruff check src/labelmaker src/ideate scripts/labelmaker tests/labelmaker tests/ideate
```

- Labelmaker: **1596 passed, 2 skipped in 249.05s**, exit **0**.
- Ideate: **1192 passed in 227.57s**, exit **0**.
- Ruff: **All checks passed**, exit **0**, over the exact requested scope.
- Documentation/report assertion: **PASS**, exit **0**; `git diff --check`: clean.

XRootD printed the known atexit FutureWarning after the labelmaker summary;
it did not change the successful exit code. The final suites ran with both
code deliverables present; neither source changed between those runs and the
commits.

Local ignored evidence logs: `.superpowers/sdd/C3fix-{labelmaker,ideate}-final.log`
and `C3fix-chain.log`. No suite was running when any deliverable was committed.

## Commits

`git log --oneline recommender..recommender-C3fix` immediately before committing
the documentation/report deliverable (its own hash cannot be embedded in itself):

```text
68e0742 ideate: write the reviewed cohort to corpus text output
867f6da labelmaker: preserve disjoint source coverage through ideate
```

The third commit is this documentation/report deliverable. Every commit has an
evidence-stating body and ends with
`Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## Deviations and limits

- **Startup command deviation:** after reading the brief, the very first test
  invocation mistakenly used `pixi run -e labelmaker pytest ...` rather than the
  frozen/no-install command. It had already exited when I attempted to stop it.
  It changed the worktree `pixi.lock`, which was restored; worktree-local `.pixi`
  environment artifacts were observed. I cannot claim that initial invocation
  kept environments read-only. Every subsequent validation used the brief's
  frozen/no-install invocation against the existing main-checkout manifest.
  No explicit `pixi install` command was run.
- One focused mixed-environment invocation tried an ideate fixture in the
  labelmaker environment and failed because sklearn is absent there; it was
  rerun in ideate-cpu. Fixture setup mistakes and red runs are not counted as
  successful evidence.
- Scope extensions: `shotdb/store.py` was necessary to retain interval fields in
  retrieval records and eliminate the event-hull fallback. The small browser
  timeline fix was necessary because it too consumed the hull as coverage.
  No changes to masks.py, TokEye mask scripts, or the block inference loop.
- No production writes or production application runs. All pipeline/join and
  output-list writes were inside pytest tmp_path fixtures. No real 198658 data
  was copied or mutated; this is a controlled synthetic reproduction, not a claim
  about an actual dropout in that shot. No files were created under
  `/scratch/gpfs/nc1514` outside this worktree. No plans were edited; `recommender`
  was never checked out or committed to. The supplied untracked brief is left
  as provided. Worktree-local ignored Pixi/test artifacts are not committed.

## Post-review fixes (2026-09-14)

Read the independent `review-C3fix.md` (MERGE WITH FIXES), this report and the
brief before editing. Starting HEAD was `5de8680` on `recommender-C3fix`.
Findings 1–5 are addressed below. The earlier report and captured outputs above
describe the original implementation; this section records the corrections.

### Finding 1: any-channel beam coverage

`src/labelmaker/events/pipeline.py:1182` now passes the 2-D `pinj_y` directly to
`Coverage.measured`: any finite beam covers that sample. It still intersects
the beam intervals with D-alpha and density and uses `LH_MIN_GAP_S`. This
restores the previous coverage rule; summing the channels had accidentally
made a single NaN invalidate the other seven. Documented at
`docs/LABELMAKER.md:306`.

`tests/labelmaker/test_events_pipeline.py:1505`
(`test_lh_coverage_needs_any_finite_beam_at_each_sample`) first records the real
pipeline's baseline intervals, then NaNs exactly one or all eight beam channels
over 0.2–0.25 s and republishes within `tmp_path`. The one-channel case requires
identical coverage; the all-channel case requires two intervals ending/starting
at the actual surrounding samples.

Red: **1 failed, 1 passed, 80 deselected in 4.42s**, exit **1**; one missing beam
incorrectly split the baseline. Green: **2 passed, 80 deselected in 4.51s**,
exit **0**, with the same command:

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/src HF_HUB_OFFLINE=1
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker python -m pytest tests/labelmaker/test_events_pipeline.py -k test_lh_coverage_needs_any_finite_beam_at_each_sample -q -W error -p no:cacheprovider
```

### Finding 2: interval encodings

`src/ideate/labels/event_sources.py:68` recognizes ndarray, list, tuple and JSON
string values before testing scalar missingness, so `legacy_hull` always returns
a boolean. `row_intervals` (`:76`) accepts those authoritative sets, including
empty sets, while None/NaN retain the explicitly caveated legacy hull.

`tests/ideate/test_event_sources.py:43`
(`test_interval_encodings_have_consistent_legacy_and_authoritative_semantics`)
checks populated list/tuple/ndarray/JSON, None, Python and NumPy NaN, and empty
list/tuple/ndarray/JSON. It verifies parsing, boolean legacy classification,
summary provenance and the gap decision against deliberately misleading hulls.
Red: **2 failed, 9 passed, 15 deselected in 0.14s**, exit **1** (both ndarray
variants returned arrays). Green: **11 passed, 15 deselected in 0.06s**, exit **0**:

```bash
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e ideate-cpu python -m pytest tests/ideate/test_event_sources.py -k test_interval_encodings -q -W error -p no:cacheprovider
```

### Finding 3: explicit text-source decision

Retained the empty source coverage as an explicit decision. The `text` source is
non-diagnostic and carries no coverage: `intervals=[]`, NaN display hull. This
replaces its older shot-span hull and records only that the logbook search ran.
The contract is now stated at `docs/LABELMAKER.md:310` and beside the writer at
`src/labelmaker/events/pipeline.py:1289`; the stale source-reader comment was
corrected at `src/ideate/labels/event_sources.py:114`.

`tests/labelmaker/test_events_pipeline.py:739`
(`test_text_source_has_documented_non_diagnostic_empty_coverage`) runs the real
text pipeline in `tmp_path`, checks `ok`/one event/empty intervals/NaN hull,
and requires the explicit documentation. Red: **1 failed, 82 deselected in
3.39s**, exit **1**, specifically on the missing documentation; the existing
empty-coverage behavior already passed. After documenting the decision, green:
**1 passed, 82 deselected in 3.02s**, exit **0**.

`tests/ideate/test_coverage_agreement.py:209`
(`test_text_only_empty_coverage_is_unprocessed_and_never_unknown`) pins the
already-correct consumer behavior: text-only rows contribute zero unknown
coverage sources, no coverage windows, and no partial flag. Bounded and unbounded
MCP calls are `unprocessed`; evidence and describe never turn them into
`uncovered` evidence or unknown diagnostic coverage. **1 passed, 35 deselected
in 1.37s**, exit **0**. Commands:

```bash
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker python -m pytest tests/labelmaker/test_events_pipeline.py -k test_text_source_has_documented_non_diagnostic_empty_coverage -q -W error -p no:cacheprovider
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e ideate-cpu python -m pytest tests/ideate/test_coverage_agreement.py -k test_text_only_empty_coverage -q -W error -p no:cacheprovider
```

### Finding 4: one shared window decision

`src/ideate/retrieval/phenomena.py:807` (`coverage_for_sources`) owns the window
decision and gap/partial qualifications. It calls the existing interval-based
`event_sources.coverage_state`, unions and clips the eligible source sets, and
returns status, hull, windows, partial flag and qualification strings. Optional
bounds use the recorded edge; no bounds searches the whole record and is partial
when the union has an interior gap. Existing closed-boundary overlap and 1 ns
partial-edge tolerance are preserved.

MCP calls it at `src/ideate/mcp/tools.py:601`; `_coverage_for` calls it at
`src/ideate/retrieval/phenomena.py:862`, reached by evidence at `:1013` and
describe through `src/ideate/retrieval/describe.py:356`. Registry source and
diagnostic selection still happen before this shared decision. Per-interface
shot/provenance explanations remain outside it; partial and gap caveats now
come from exactly one implementation, with the same precise boundaries.

`tests/ideate/test_coverage_agreement.py:155` tests eleven combinations: inside
a gap, crossing an edge or gap, fully covered, whole record, and either one-sided
bound both within and outside the recorded intervals. It checks identical
status/windows/partial and exact qualification strings. `:186` wraps the actual
shared function to verify MCP, evidence and describe really call it.

The initial ten-window run was **7 failed, 4 passed** (different partial strings,
one-sided retrieval TypeErrors, absent shared function). Adding an explicit
bounded gap-spanning window and using `None` for the original whole-record API
gave the final red: **8 failed, 4 passed, 36 deselected in 6.21s**, exit **1**,
including missing MCP gap caveats. Green: **12 passed, 36 deselected in 5.91s**,
exit **0**, with the same final command:

```bash
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e ideate-cpu python -m pytest tests/ideate/test_coverage_agreement.py -k 'share_window_decisions or call_the_same_window_function' -q -W error -p no:cacheprovider --tb=short
```

The follow-up read-only review caught one consequence of sharing the existing
gap string: MCP displays a whole-source hull alongside clipped window intervals,
so the string could not say that `coverage` is the hull of `coverage_windows`.
The regression at `tests/ideate/test_coverage_agreement.py:175` checks both
payloads for a bounded gap-spanning request and rejects that false relationship.
Red: **1 failed, 48 deselected in 3.03s**, exit **1**. The shared wording at
`src/ideate/retrieval/phenomena.py:142` now says the covered intervals are disjoint
and points to `coverage_windows`. Green: **1 passed, 48 deselected in 3.22s**,
exit **0**:

```bash
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e ideate-cpu python -m pytest tests/ideate/test_coverage_agreement.py -k test_gap_qualification_does_not_confuse_source_and_requested_hulls -q -W error -p no:cacheprovider --tb=short
```

The reviewer confirmed the correction and found no remaining issues in scope.
That review performed no tests, edits or writes.

### Finding 5: round-trippable displayed boundaries

`src/ideate/labels/event_sources.py:103` uses float `repr` formatting, preserving
the actual interval endpoints. The shared partial-caveat implementation at
`src/ideate/retrieval/phenomena.py:838` uses that same formatter, removing the
retrieval-side three-decimal rounding too.

`tests/ideate/test_event_sources.py:20`
(`test_formatted_bounds_do_not_exclude_a_window_the_source_covers`) displays
[99.9999999, 100.0000499] s and parses it back. A contained window must still fit
the displayed bounds, which must round-trip exactly. Red: **1 failed, 26
deselected in 0.12s**, exit **1**, because `%g` printed `[100, 100] s`. Green:
**1 passed, 26 deselected in 0.06s**, exit **0**:

```bash
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e ideate-cpu python -m pytest tests/ideate/test_event_sources.py -k test_formatted_bounds -q -W error -p no:cacheprovider
```

### Finding 6: known storage cost, deferred as requested

`src/labelmaker/events/coverage.py:286` still repeats the full source interval
set in every event's `attrs`, enabling correct event-only fallback. Storage is
O(events × intervals), so a source with many finite runs and thousands of ELM
events repeats substantial JSON. No deduplication or persistence change was
made in this post-review pass.

### Final verification for all five fixes

Every finding above is covered by these complete-suite gates, all run from the
worktree with the requested environment and warning policy:

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/src HF_HUB_OFFLINE=1
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker python -m pytest tests/labelmaker -q -W error -p no:cacheprovider
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e ideate-cpu python -m pytest tests/ideate -q -W error -p no:cacheprovider
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker ruff check src/labelmaker src/ideate scripts/labelmaker tests/labelmaker tests/ideate
```

- Labelmaker: **1599 passed, 2 skipped in 250.11s (0:04:10)**, exit **0**.
- Ideate: **1218 passed in 198.42s (0:03:18)**, exit **0**.
- Ruff: **All checks passed!**, exit **0**, over the entire requested scope.

The first full ideate run passed **1217 passed in 230.90s (0:03:50)**, exit **0**,
before the follow-up review's caveat wording correction and extra regression.
The final ideate run above includes that correction. Labelmaker source/tests/docs
did not change after its full successful run. Both the original real
pipeline → tmp DB join → in-process/stdio MCP chain and former cap-path controls
remain in the passing suites. The known XRootD atexit FutureWarning followed the
labelmaker summary without changing its successful exit code.

Logs: `/tmp/C3fix-post-labelmaker.log`, `/tmp/C3fix-post-ideate.log`, and
`/tmp/C3fix-post-ideate-final.log`. `git diff --check` is clean. A report
completeness assertion failed while result fields were pending and passed after
the final result and commit evidence were filled in.

### Scope and commits

All post-review edits and Git mutations occurred in this worktree, on
`recommender-C3fix`. No other branch/worktree was inspected or modified. The
required Pixi manifest was used solely to execute the existing environments;
every invocation used `--frozen --no-install`. No local `.pixi` directory was
created and `pixi.lock` has no changes. All application writes occurred within
the test suites' `tmp_path` fixtures; scratch logs are under `/tmp`. There were
no manual production commands or accesses to `/scratch/gpfs/EKOLEMEN/`, no
changes under `docs/superpowers/plans/**`, and no edits to masks, mask scripts,
or the pipeline block-inference loop.

The supplied review and brief were untracked on entry despite the stated clean
starting state. They are preserved verbatim with the committed report so the
worktree finishes clean. Finding 6 remains deferred as requested; no other
requested fix is left undone. All commits have red-to-green evidence bodies and
the required Co-Authored-By trailer. No suite was running at commit time.

`git log --oneline 5de8680..HEAD` immediately before the report commit:

```text
d8388a1 ideate: share coverage decisions and preserve interval boundaries
b8d3d71 labelmaker: restore any-beam coverage and document text semantics
```

The final commit is `ideate: record C3fix post-review evidence`; its own
hash cannot be embedded in itself. The final response includes the full
three-commit log, including this report commit.
