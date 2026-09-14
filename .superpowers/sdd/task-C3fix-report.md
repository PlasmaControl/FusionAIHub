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
