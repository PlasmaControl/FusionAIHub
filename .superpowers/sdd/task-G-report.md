# Part G implementation report

Worktree: `/scratch/gpfs/nc1514/FusionAIHub-flow`
Branch: `sd-flow`
Base: `696260f`
Date: 2026-09-22

The whole iteration-2 spec was read, including F6, together with the flow audit's
UI description, code-quality notes and UI-flow priorities. G1–G3 are complete in
three commits; this report is a fourth commit.

## G1 — search to design

Commit: `5f86f33` — `shot_design: connect search selections to actuator designs`

- Ordered selection of up to six shots; first selected is the reference, others
  comparisons, and search text becomes Notes:
  `src/shot_design/ui/static/app.js:337`,
  `src/shot_design/ui/static/design.js:882`.
- Submitted text, reference, segment, result count, constraints and require/avoid
  labels live in the URL hash. Back/reload restores controls and reruns retrieval:
  `src/shot_design/ui/static/app.js:416`,
  `src/shot_design/ui/static/app.js:770`.
- Rank and contributing-channel chips replace the fused score. Row explanations,
  measured differences, flags and caveats are visible. Query-level proposal flags
  appear above results; row-level proposal flags are supported too. Removed the
  extra `/api/shot` request for every row:
  `src/shot_design/ui/static/app.js:337`.
- Field/minimum/maximum rows replace JSON entry. Label pickers use the database,
  schema regime vocabulary, and phenomenon registry used by the server's filters:
  `src/shot_design/ui/app.py:50`,
  `src/shot_design/ui/static/app.js:374`.
- One shared DOM factory serves the three page controllers:
  `src/shot_design/ui/static/assistant.js:2`. Non-finite response numbers become
  JSON null on the server; removed the browser's NaN/Infinity regex parser:
  `src/shot_design/ui/app.py:33`.
- UI documentation: `docs/shot-design/overview.md:431`.

## G2 — simulation state and results

Commit: `a798244` — `shot_design: retain simulation results by revision and render metrics`

- Simulation state is keyed by saved design ID in a separate controller; one
  renderer derives submission eligibility, status, report links and results:
  `src/shot_design/ui/static/simulation.js:47`.
- Opening a saved revision fetches its status. Edits retain the result and report,
  identify the saved revision, and state that unsaved edits are excluded:
  `src/shot_design/ui/static/design.js:127`,
  `src/shot_design/ui/static/design.js:746`.
- Existing results offer **Run again**, with confirmation. Pending submissions
  cannot be repeated; queued/running states show elapsed time. Delayed responses
  cannot repaint another revision. Failed initial status reads do not offer to
  submit an unknown job:
  `src/shot_design/ui/static/simulation.js:58`,
  `src/shot_design/ui/static/simulation.js:105`,
  `src/shot_design/ui/static/simulation.js:144`.
- A submission timestamp/job ID is recorded atomically in `submission.json`, so
  reopening while queued works and a rerun does not appear to have completed
  because the previous worker's `status.json` remains. Worker status files are
  not overwritten by the UI:
  `src/shot_design/ui/simulate_routes.py:55`.
- Authenticated `GET /api/design/{id}/simulate/metrics` serves `metrics.json`,
  with a plain 404 detail if absent. Panel names now support `co2.png`:
  `src/shot_design/ui/simulate_routes.py:101`,
  `src/shot_design/ui/simulate_routes.py:118`.
- Inline modality tables render the exact F6 schema: errors, negative skill,
  edit effect, noise and unresolved effects. Held diagnostics appear once as
  “not simulated (diagnostic absent)”; panel images follow their tables:
  `src/shot_design/ui/static/simulation.js:10`.
- Polling configuration is served through metadata:
  `src/shot_design/ui/app.py:171`.
- UI documentation: `docs/shot-design/programs.md:62`.

## G3 — assistant

Commit: `5fcd860` — `shot_design: reattach assistant jobs and simulate saved results`

- A saved exportable result offers **Simulate**; otherwise it offers **Open in
  actuator editor**. Reference and comparison shots link to the Shot view:
  `src/shot_design/ui/static/assistant.js:74`,
  `src/shot_design/ui/static/assistant.js:237`.
- The job ID is retained at `#create/{job}`. Reload fetches and reattaches to the
  job; an unavailable job is explicitly described as no longer on the server:
  `src/shot_design/ui/static/assistant.js:137`,
  `src/shot_design/ui/static/assistant.js:157`,
  `src/shot_design/ui/static/app.js:734`.
- The assistant simulation action opens the saved revision, checks its status and
  uses the editor's submission behavior. Its URL contains no automatic-submit
  instruction, so reloading cannot resubmit:
  `src/shot_design/ui/static/app.js:741`,
  `src/shot_design/ui/static/design.js:901`.
- UI documentation: `docs/shot-design/programs.md:12`.

## Verification

The exact requested command was run from the worktree before implementation and
after the final code and test changes:

```bash
PYTHONPATH=$PWD/src pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e shot-design-cpu python -m pytest tests/shot_design -q -W error
```

Before:

```text
1855 passed, 4 skipped in 178.40s (0:02:58)
```

After:

```text
1882 passed, 4 skipped in 165.11s (0:02:45)
```

Both exited 0. The skip count is unchanged. An earlier postimplementation full
run also passed (`1881 passed, 4 skipped in 188.26s`); the final run above includes
the additional pending-rerun status regression.

Ruff reported **All checks passed!** for every changed Python file, using:

```bash
mapfile -t TASK_PYTHON_FILES < <(git diff --name-only 696260f -- '*.py')
/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/ruff check "${TASK_PYTHON_FILES[@]}"
```

`git diff --check 696260f` and `node --check` on app.js, assistant.js, design.js
and simulation.js passed. Node was already on PATH. No dependency installation,
lockfile operation, bare-environment suite, or production job was run.

### New tests

27 additional tests; each item's initial behavior tests were run red before its
implementation. The pending-rerun status regression was also observed red before
its fix and folded into the G2 commit.

- G1: three browser contracts in `tests/shot_design/test_ui_flow_browser.py:7`
  cover ordered selection/limit, explanations/channels/no row fetch, filter hash
  restoration and validation, and editor comparisons/Notes. Two transport tests
  at `tests/shot_design/test_ui.py:359` cover metadata and strict finite JSON.
- G2: five route tests at `tests/shot_design/test_ui_simulate.py:238` cover exact
  metrics, missing metrics, queue reattachment, rerun status and numbered panels.
  Existing route auth/path-validation tests include the new metrics route.
  Nine browser tests in `tests/shot_design/test_ui_simulation_browser.py:33`
  cover rendering, edits, confirmation/double submission, elapsed time and
  revision isolation, missing metrics, failed status/submission, delayed replies,
  and pending-rerun wording.
  `tests/shot_design/test_ui_metrics.py:3` defines the exact F6 fixture, including
  negative skill, an unresolved effect and a held modality; the route tests write
  it as a real `metrics.json` in the temporary simulation directory.
- G3: eight tests in `tests/shot_design/test_ui_assistant_flow.py:7` cover action
  eligibility, shot links, running-to-complete reattachment, missing/stale jobs,
  job hashes, simulation navigation without reload submission, and HTTP status
  propagation.
- Existing fake-DOM tests were updated for the shared helper and the intentional
  replacement of raw scores and per-row title requests.

### Browser and manual checks

Ran the actual HTML/CSS/JavaScript in cached headless Chromium, using a temporary
localhost Node server with fixture API responses and a fixture PNG. Inspected
screenshots of Search and the editor's metrics table. Checked:

- Constraint/label submission, selection in reverse row order, comparison order
  and Notes in the editor, Back, and full reload restoring filters and results.
- Negative skill and unresolved-effect wording, held diagnostic shown once,
  successful panel image loading, and retained results after a notes edit.
- Assistant reference links, result reattachment after reload, one simulation
  submission through the editor, queued reattachment without a second submission,
  and the explicit missing-job message.
- Search at 1440 px and 390 px; no document-wide horizontal overflow at 390 px.
- No browser script errors. The temporary server and Chromium were stopped.

No `shot_design serve` process was started. Browser API data and the panel were
fixtures; no real model inference or physics validation was claimed.

### Existing fixes verified and left intact

- B6: 2–6 s defaults already matched in index.html, design.js and
  `src/shot_design/design/program.py:81`.
- B8: `configs/shot_design/paths.yaml:23` already selected the Stellar simulation
  wrapper; the Frontier paths retain their own wrapper.
- `src/shot_design/simulate/cli.py:115` already derived prediction length as
  `min(n_frames, trained) - k0` when not explicitly supplied.
- `configs/shot_design/ignite_modalities.yaml:43` already pinned generation v4;
  no active v2 UI/model path was reintroduced.

## Scope and remaining work

No G1–G3 item is left undone. All implementation changes are within the authorized
UI, UI-test and UI-documentation paths. No Part F/R/H file, labeler file,
pyproject.toml, pixi.lock, or data/events path was edited. No production data was
written, no sbatch/srun/fdp command was executed, and neither HF_HOME nor
HF_HUB_CACHE was set.

Real ensemble generation and scoring remain with Part F. The UI integration is
verified against its specified F6 artifact schema. Assistant jobs still use the
existing in-memory server store; disk persistence was not requested by Part G,
and unavailable jobs are handled explicitly. Broader audit refactors and
additional product features outside G1–G3 were intentionally left outside scope.
