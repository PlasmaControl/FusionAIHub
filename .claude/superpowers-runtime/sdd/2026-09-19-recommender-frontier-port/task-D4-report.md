# Task D4 report: Frontier sbatch wrapper + UI routes for the Shot Designer simulation stage

## Status: DONE

Commit: `91381ec` "shot_design ui: simulate via sbatch submit + status polling; Frontier simulate wrapper"

## What was built

### 1. `src/shot_design/ui/simulate_routes.py` (new)

`APIRouter(prefix="/api/design")` with four handlers, all guarded by `_check_ident`
(re-uses `shot_design.design.program.ID_PATTERN`, `^[0-9a-f]{32}$`, to reject
traversal/garbage idents with 404 before any path is built):

- `POST /{ident}/simulate` -> 202. Loads `configs/shot_design/ui.yaml`'s `simulate`
  block via `config.load_yaml("ui.yaml")["simulate"]` (KeyError -> 500 "... is missing
  its simulate: block", raised inside the handler so a missing block never crashes
  import). Formats `submit_cmd.format(ident=ident)`, calls
  `request.app.state.submit(cmd)`, parses `Submitted batch job (\d+)`; no match -> 502
  with the raw output as `detail`. Success -> `{"ident", "job_id", "status_url"}`.
- `GET /{ident}/simulate` -> contents of `outputs/<ident>/simulation/status.json`, or
  `{"state": "not_started"}` if the file doesn't exist yet.
- `GET /{ident}/simulate/report` -> `FileResponse` of `.../simulation/report.md`
  (`text/markdown`); 404 if missing.
- `GET /{ident}/simulate/panels/{name}` -> `FileResponse` of
  `.../simulation/panels/<name>` (`image/png`); `name` validated against
  `^[a-z_]+\.png$` (rejects `.jpg`, uppercase, bare names, and traversal segments
  that survive URL-decoding into the path param) -> 404 for anything else, and 404
  again if the (validated) file doesn't exist. Panels live under a `panels/`
  subdirectory, per `simulate/report.py`'s `report.write` (`panel_dir = out_dir /
  "panels"`; `report.md` and `status.json` sit directly in `out_dir`) -- confirmed by
  reading `report.write`'s body, not just the brief's paraphrase.

`default_submit(cmd)`: the production `app.state.submit`. `REPO_ROOT =
Path(__file__).resolve().parents[3]` (verified: `ui/simulate_routes.py` ->
`[0]=ui, [1]=shot_design, [2]=src, [3]=repo root`; a test asserts
`(REPO_ROOT / "pyproject.toml").exists()`). Runs
`subprocess.run(shlex.split(cmd), cwd=REPO_ROOT, capture_output=True, text=True,
check=True)` and returns `.stdout`.

### 2. `src/shot_design/ui/app.py`

- `app.state.submit = default_submit` set in `create_app`, alongside
  `app.state.paths`/`app.state.token` (imported locally inside the function, matching
  the existing lazy-import style used for `design_router`/`assistant_router` a few
  lines down, rather than a top-level import -- kept consistent with the file's own
  convention rather than introducing a new one).
- `app.include_router(simulate_router)` added next to the other two routers.

### 3. `configs/shot_design/ui.yaml`

```yaml
simulate:
  submit_cmd: "sbatch scripts/slurm_frontier/shot_design_simulate.sh {ident}"
  poll_s: 15
```

### 4. `scripts/slurm_frontier/shot_design_simulate.sh`

Stub body replaced with the brief's Step 5 verbatim (header untouched, already
correct from the stub): sources `_shot_design_common.sh` via
`${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}`, then
`srun "$PY" -m shot_design simulate "$IDENT" --device cuda "${@:2}"`.
`tests/shot_design/test_slurm_frontier_scripts.py` already had
`shot_design_simulate.sh` in its `SCRIPTS` list (house-style + submit-dir-sourcing
checks), so no test file changes were needed there; both parametrized tests
were re-run and are green against the new body.

### 5. Static UI (`design.js` + `index.html`)

- `index.html`: added a `Simulate` button (`#design-simulate`, disabled by default),
  a status line (`#design-simulate-status`), and a report link
  (`#design-simulate-report`, hidden until complete), placed next to the existing
  JSON/IGNITE download links.
- `design.js`: added `state.simulateIdent` / `state.simulateTimer` (module state, not
  DOM `dataset` -- see "quality note" below). `invalidateDownloads()` now also resets
  the simulate button/status/report and stops any pending poll; `updateDownloads()`
  enables the Simulate button and records the current id exactly when the IGNITE
  download link becomes available (`validation.can_export`), since `cli.py`'s
  `run()` calls `program.export_ignite` itself -- simulating doesn't require the user
  to have downloaded the `.pt` first, only that the design *can* export.
  `simulateDesign()` POSTs, then `pollSimulateStatus()` polls
  `GET .../simulate` every `SIMULATE_POLL_MS` (15000, a literal client-side echo of
  `ui.yaml`'s `poll_s: 15` -- there is no route exposing `ui.yaml` to the browser, so
  hardcoding with a comment pointing at the source of truth was the YAGNI choice over
  adding one). On `complete`, shows the report link; on `failed`, shows the error;
  polling is guarded by `state.simulateIdent !== id` checks so a stale poll from a
  previous design can't clobber a newer one's UI after a re-render.

Followed the existing pattern exactly: same `state.api(path, options)` wrapper
(throws `Error(message)` from `data.error ?? data.detail` on non-2xx, matching
`app.js`'s `api()`), same `node(id)` helper, same click-handler wiring style in
`initDesign()`.

## Tests

### `tests/shot_design/test_ui_simulate.py` (new, 24 tests)

Uses the shared `paths` fixture from `tests/shot_design/conftest.py` (not
`program_source` -- no design program needed, just a data root). `client` fixture
builds `create_app(paths=paths, token="secret")` and injects
`app.state.submit = lambda cmd: "Submitted batch job 4242"` so the real
`default_submit` (and therefore `sbatch`) is never invoked by any route test.

Covers: submit -> 202 with job_id/status_url; submit passes the formatted
`submit_cmd` through to `app.state.submit`; unparseable submit output -> 502 with
raw output in `detail`; missing `simulate` block in `ui.yaml` -> 500 (via
monkeypatching `shot_design.config.load_yaml`, not the yaml file on disk); status
`not_started` / echoing `status.json`; report served / 404 when missing; panel
served as `image/png` / 404 when missing; panel name validation rejecting
`.jpg`, uppercase, bare names, `mhr..png`, and encoded traversal segments;
malformed/unsafe idents -> 404 across all four routes; all four routes require
auth (matching `test_design_routes.py`'s pattern); `REPO_ROOT` resolves to the
actual repo root; `default_submit` calls `subprocess.run` with `cwd=REPO_ROOT`,
`check=True`, and the `shlex.split` command (via `monkeypatch.setattr(subprocess,
"run", fake)`).

### `tests/shot_design/test_design_browser.py` (modified, 1 line)

The Node-VM browser-contract harness's stub DOM only creates `Element`s for a
fixed `ids` list; `design.js`'s `invalidateDownloads()`/`updateDownloads()` are
exercised by nearly every existing test in this file (save, preview, reset,
merge, ...), and my changes call `node("design-simulate")` etc. unconditionally
inside those two functions. Without adding the three new ids, every existing
test in the file broke with `TypeError: Cannot read properties of null`. Added
`'design-simulate', 'design-simulate-status', 'design-simulate-report'` to the
`ids` list. No other line in this file changed; no assertions were weakened.

(Quality note: my first draft stored the in-flight simulate ident on
`button.dataset.ident`. The harness's mock `Element` class has no `.dataset`
implementation at all, so that broke every test with `TypeError: Cannot convert
undefined or null to object` on `delete simulate.dataset.ident`. Moved that state
into `state.simulateIdent` instead -- cleaner than a DOM data attribute anyway,
consistent with how `state.program`/`state.dirty` already track editor state.)

## TDD evidence

RED (before wiring `simulate_routes.router` into `app.py`; the file
`simulate_routes.py` existed but was not yet imported/included, plans/handlers
correct):

```
$ pixi run --frozen -e shot-design-frontier python -m pytest \
    tests/shot_design/test_ui_simulate.py -q -p no:cacheprovider
...
FAILED test_submit_returns_job_id_and_status_url
FAILED test_submit_runs_the_configured_command_with_ident_filled_in
FAILED test_unparseable_submit_output_is_502
FAILED test_missing_simulate_block_in_ui_yaml_is_500
FAILED test_status_is_not_started_when_no_status_file_exists
FAILED test_status_returns_the_status_json_contents
FAILED test_report_is_served_when_present
FAILED test_panel_is_served_as_png
8 failed, 16 passed in 3.89s
```

(The 16 "passed" at RED were tests whose expected result was itself 404/401 --
auth-gate and unknown-route both happen to return those codes before any route
exists, so they were trivially green; the 8 failures are the ones that only pass
once the real handlers run, which is the meaningful RED signal.)

GREEN (after `app.py` wiring + `ui.yaml` + slurm script + JS/HTML):

```
$ pixi run --frozen -e shot-design-frontier python -m pytest \
    tests/shot_design/test_ui_simulate.py -q -p no:cacheprovider
........................                                                 [100%]
24 passed, 1 warning in 3.83s
```

Regression check (design routes + slurm house-style, unaffected by D4):

```
$ pixi run --frozen -e shot-design-frontier python -m pytest \
    tests/shot_design/test_slurm_frontier_scripts.py \
    tests/shot_design/test_design_routes.py -q -p no:cacheprovider
............................                                             [100%]
28 passed, 1 warning in 4.37s
```

Browser-contract RED (after design.js change, before fixing the harness's `ids`
list):

```
$ pixi run --frozen -e shot-design-frontier python -m pytest \
    tests/shot_design/test_design_browser.py -q -p no:cacheprovider
...
TypeError: Cannot read properties of null (reading 'addEventListener')
24 failed, 1 passed in 1.38s
```

GREEN (after adding the three ids):

```
$ pixi run --frozen -e shot-design-frontier python -m pytest \
    tests/shot_design/test_design_browser.py -q -p no:cacheprovider
.........................                                                [100%]
25 passed in 1.18s
```

Full scoped re-check right before commit:

```
$ pixi run --frozen -e shot-design-frontier python -m pytest \
    tests/shot_design/test_ui_simulate.py tests/shot_design/test_design_browser.py \
    tests/shot_design/test_design_routes.py tests/shot_design/test_slurm_frontier_scripts.py \
    -q -p no:cacheprovider
77 passed, 1 warning in 6.36s
```

Full `tests/shot_design` (per environment rules, before committing):

```
$ pixi run --frozen -e shot-design-frontier python -m pytest tests/shot_design -q -p no:cacheprovider
15 failed, 1713 passed, 20 skipped, 2 warnings in 128.48s
```

All 15 failures are in files I was told never to touch (`src/shot_design/cli.py`,
`src/shot_design/simulate/cli.py`, `tests/shot_design/test_simulate_cli.py`) plus
the one pre-existing known failure (`test_mcp.py::test_the_project_mcp_config_...`)
plus a cascading `test_cli.py` failure from the same broken import. Root cause
visible in the traceback: `ImportError: cannot import name 'DEFAULT_DECODE' from
'shot_design.simulate'` -- the concurrent fix-round implementer's in-progress edit
to `src/shot_design/simulate/__init__.py`/`cli.py` (also touching
`src/shot_design/shotdb/select.py` and `tests/shot_design/test_select.py`, neither
of which I touched or staged). None of D4's own files or tests are implicated;
re-running my four scoped test files alone (above) is 77/77 green. I did not stage
or commit any of those files.

## Files changed (staged and committed)

- `/lustre/orion/fus187/scratch/nchen/FusionAIHub/configs/shot_design/ui.yaml`
- `/lustre/orion/fus187/scratch/nchen/FusionAIHub/scripts/slurm_frontier/shot_design_simulate.sh`
- `/lustre/orion/fus187/scratch/nchen/FusionAIHub/src/shot_design/ui/app.py`
- `/lustre/orion/fus187/scratch/nchen/FusionAIHub/src/shot_design/ui/simulate_routes.py` (new)
- `/lustre/orion/fus187/scratch/nchen/FusionAIHub/src/shot_design/ui/static/design.js`
- `/lustre/orion/fus187/scratch/nchen/FusionAIHub/src/shot_design/ui/static/index.html`
- `/lustre/orion/fus187/scratch/nchen/FusionAIHub/tests/shot_design/test_design_browser.py`
- `/lustre/orion/fus187/scratch/nchen/FusionAIHub/tests/shot_design/test_ui_simulate.py` (new)

Explicitly NOT touched (per instructions, and confirmed still unstaged/untouched by
this work): `src/shot_design/simulate/cli.py`, `src/shot_design/cli.py`,
`tests/shot_design/test_simulate_cli.py`.

## Self-review

**Completeness:** all four routes implemented and tested; `ui.yaml` `simulate:`
block added and read the same way `landing`/`actuation`/`server` are (
`config.load_yaml("ui.yaml")["simulate"]`); JS button + polling + report link
added and wired; Slurm script body matches the brief verbatim (header was
already correct from the D3 stub, untouched); `app.state.submit` default wired
with a dedicated unit test for its `cwd`/`shlex.split`/`check=True` behavior.

**Quality:** ident validation reuses `program.ID_PATTERN` rather than
reinventing a regex, matching the house style already visible in
`design_routes.py`'s `_id_path`. Panel path correctly targets the `panels/`
subdirectory after reading `report.write`'s actual body rather than trusting the
brief's paraphrase ("panels are PNG files in that same directory") at face
value -- the code and the brief disagree on this detail and the code wins.

**Discipline (YAGNI):** did not add a config-exposure endpoint just to let JS
read `poll_s` at runtime; a hardcoded 15000ms with a comment pointing at the
yaml key was the smaller, sufficient change. Did not restyle any existing part
of `design.js`/`index.html` beyond the new button/status/report elements and
their wiring. Did not add ident-existence checks beyond format validation (GET
`/simulate` legitimately returns `not_started` for a syntactically valid ident
with no simulation directory yet -- it does not require the design itself to
exist, matching the brief's stated contract).

**Testing:** `TestClient` used exactly as in `test_design_routes.py`; every
test that could invoke `sbatch` instead injects `app.state.submit`; the one
test that exercises the real subprocess path monkeypatches `subprocess.run`
rather than shelling out. Output was captured directly from real `pytest` runs
(pasted above), not asserted from memory.

## Concerns

1. The full-suite run has 15 failures, all outside my scope (see above) and
   caused by a concurrently-edited, currently-broken import in files I was
   explicitly told not to touch. This should resolve when that other fix round
   lands; flagging in case it's still broken when this task is reviewed.
2. `SIMULATE_POLL_MS` in `design.js` is a hardcoded literal (15000) mirroring
   `ui.yaml`'s `poll_s: 15` rather than reading it live, since no route exposes
   `ui.yaml` to the browser and the brief's four listed routes don't include
   one. If a future task wants the poll interval genuinely configurable from
   the browser, that would need a small addition (e.g. folding it into
   `/api/meta`) -- out of scope here.

## Fix round 1

Commit: `20f4b9d` "shot_design ui: map a failed sbatch submission to 502 with its stderr"

### Important -- unguarded `submit_fn(cmd)`

`src/shot_design/ui/simulate_routes.py`'s `submit()` called `submit_fn(cmd)` (now
`request.app.state.submit(cmd)`, see Minor below) with no exception handling. A
real failing `sbatch` -- `subprocess.CalledProcessError` from `default_submit`'s
`check=True`, or `FileNotFoundError`/other `OSError` when `sbatch` isn't on
`PATH` -- propagated straight through FastAPI's default handler as a bare-text
500 with a full traceback body, not JSON. The frontend's `state.api()` helper
(`static/app.js:232-233`) does an unconditional `JSON.parse(await
response.text())` on every response regardless of status, so that plain-text
500 would throw a `SyntaxError` inside `simulateDesign()`'s `catch` block
instead of the intended "Submit failed: ..." message -- the operator would see
a cryptic client-side parse error with no hint that sbatch itself failed, let
alone why.

Fixed by wrapping the call:

```python
try:
    output = request.app.state.submit(cmd)
except subprocess.CalledProcessError as exc:
    raise HTTPException(502, exc.stderr or exc.stdout or str(exc)) from exc
except OSError as exc:
    raise HTTPException(502, str(exc)) from exc
```

`CalledProcessError` is checked first (it is a `SubprocessError`, not an
`OSError`, so order doesn't strictly matter here, but it groups the two
sbatch-specific failure shapes -- "ran and exited nonzero" vs. "couldn't even
run it" -- next to each other in the diff). `detail` prefers `stderr` (where
sbatch actually puts its error message, e.g. `sbatch: error: invalid qos`),
falls back to `stdout`, then to `str(exc)` if both are empty, matching
`design_routes.py`'s `_call()` convention of putting the underlying exception's
own text straight into the `HTTPException` detail.

Added two tests to `tests/shot_design/test_ui_simulate.py`:
- `test_failing_sbatch_maps_to_502_with_its_stderr`: injects
  `app.state.submit` raising
  `subprocess.CalledProcessError(1, ["sbatch"], output="", stderr="sbatch: error: invalid qos")`;
  asserts 502 and `detail == "sbatch: error: invalid qos"`.
- `test_sbatch_not_on_path_maps_to_502`: injects `app.state.submit` raising
  `FileNotFoundError(...)`; asserts 502 and the message text is present in
  `detail`.

### Minor -- unreachable `getattr` fallback

`getattr(request.app.state, "submit", default_submit)` was dead code:
`create_app` always sets `app.state.submit = default_submit` before any
request can reach this handler, so the fallback branch could never execute.
Replaced with `request.app.state.submit` directly (now wrapped in the
try/except above).

### Verification

TDD RED, before the fix (stashed `simulate_routes.py` only, kept the new
tests):

```
$ pixi run --frozen -e shot-design-frontier python -m pytest \
    tests/shot_design/test_ui_simulate.py -q -p no:cacheprovider \
    -k "failing_sbatch or not_on_path"
...
src/shot_design/ui/simulate_routes.py:65: in submit
    output = submit_fn(cmd)
FileNotFoundError: [Errno 2] No such file or directory: 'sbatch'
...
FAILED tests/shot_design/test_ui_simulate.py::test_failing_sbatch_maps_to_502_with_its_stderr
FAILED tests/shot_design/test_ui_simulate.py::test_sbatch_not_on_path_maps_to_502
2 failed, 24 deselected, 1 warning in 4.28s
```

(The `CalledProcessError` case failed the same way, as an unhandled 500-causing
exception inside the request -- not shown twice above for brevity, but both
tests were in the RED run.)

GREEN, after the fix:

```
$ pixi run --frozen -e shot-design-frontier python -m pytest \
    tests/shot_design/test_ui_simulate.py -q -p no:cacheprovider
..........................                                               [100%]
26 passed, 1 warning in 3.31s
```

Full `tests/shot_design`, once, per environment rules:

```
$ pixi run --frozen -e shot-design-frontier python -m pytest tests/shot_design -q -p no:cacheprovider
1 failed, 1731 passed, 20 skipped, 2 warnings in 125.22s
FAILED tests/shot_design/test_mcp.py::test_the_project_mcp_config_points_at_this_server
```

That is the known pre-existing failure named in the task instructions
(`git -C /scratch/gpfs/nc1514/FusionAIHub rev-parse --show-toplevel` -- a stale
path from a different machine, unrelated to this repo checkout or to D4). No
other failures: the concurrent fix round's `DEFAULT_DECODE` import breakage
that was present during the initial D4 run has since landed and resolved
cleanly, and nothing in `src/shot_design/cli.py`,
`src/shot_design/design/assistant.py`, `configs/shot_design/evalsets/*`,
`scripts/shot_design/*`, or `tests/shot_design/test_assistant_cli.py` was
touched by this fix round.

### Files changed (staged and committed)

- `/lustre/orion/fus187/scratch/nchen/FusionAIHub/src/shot_design/ui/simulate_routes.py`
- `/lustre/orion/fus187/scratch/nchen/FusionAIHub/tests/shot_design/test_ui_simulate.py`
