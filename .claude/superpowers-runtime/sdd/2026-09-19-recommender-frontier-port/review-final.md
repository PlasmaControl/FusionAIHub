# Final whole-branch review — recommender Frontier port

Range `a978f0e..77a0678` (branch `nathan_dev`), reviewed 2026-09-20 while build 5516457 and
encode 5516504 sit in the queue. Read-only: no working-tree, index, HEAD or branch state was
touched.

## Passes run

**Pass 1 (the plan's own logic)** — `git diff a978f0e..77a0678 -- src/shot_design
scripts/slurm_frontier scripts/shot_design configs/shot_design src/labeler`, 69 files.
Read file by file, with the current file pulled up wherever a hunk needed context:
`shotdb/{build,features,select,ignite,legacy_raw}.py`, `design/{assistant,seed,
program_reference,provenance}.py`, `llm/{client,agy}.py`, the whole of `simulate/`
(`core`, `decode`, `report`, `cli`, `__init__`), `cli.py` (all three diff chunks),
`ui/{app,simulate_routes,static/design.js,static/index.html}`, `labeler/jobstats.py`,
`scripts/slurm_frontier/*` (wrappers + `_shot_design_common.sh` + `_frontier_settings.sh`),
`scripts/shot_design/{demo_frontier,demo_frontier_collect,blurb_frontier,g_enc.py}`, and
`configs/shot_design/{ignite_modalities,llm,paths.frontier,ui,retrieval,evalsets}.yaml`.
Cross-checked against the merged IGNITE code where a contract is shared
(`maskgit.rollout`, `frame_layout`, `eval_dynamics.load_model`, `dynamics_config`).

**Pass 2 (tests)** — `tests/shot_design` (52 files touched) and `tests/labeler`. Checked for
Frontier data roots (only `test_paths_frontier.py` names them, and only as string assertions
about the committed yaml — no I/O), for `paths`/`tmp_path` usage, and for assert-free tests
(AST scan of every new test module: the only "no assert" hits are `pytest.raises` blocks and
`np.testing.assert_*` calls — no empty tests). Did not re-run the suites; the controller's
1705-pass run stands, and nothing I read raised a doubt that a focused run would settle.

**Pass 3 (light)** — `docs/`, `website/`, `pyproject.toml`, `.github/workflows/docs.yml`.
Checked for Claude/superpowers material under `docs/` (none: `grep -rli
"superpowers\|claude" docs/ website/` is empty), for secrets (none; only `LLM_API_KEY` read
from the environment and prose references to `~/.fdp/token` / the OAuth cache), for bare
`pixi` in anything Frontier-facing (none — the bare `pixi run` lines left in `docs/` are
Stellar sections and two pre-existing `scripts/train_*_ast.sh`), and the Pages workflow
(`npm ci` against a committed `website/package-lock.json`, `onBrokenLinks: 'throw'`,
`baseUrl: /FusionAIHub/`).
Skipped `35cfed1`/`bd0ea9e`'s own content except where later commits touch it.

## Findings

### Critical

None.

### Important

**I1. The demo cannot submit its three simulations — `scripts/shot_design/demo_frontier.sh:13`
with `scripts/slurm_frontier/shot_design_simulate.sh:4`.**
`shot_design_simulate.sh` carries `#SBATCH -q debug`, and Frontier's `debug` QOS has
`MaxSubmitJobsPU = 1` (verified: `sacctmgr -n show qos format=Name,MaxSubmitJobsPU,...` →
`debug 1 … 02:00:00`; the ledger hit the same wall at 12:40 with job 5516504). The demo loop
submits three of these back to back under `set -euo pipefail`, so the second `sbatch` is
refused with `QOSMaxSubmitJobPerUserLimit`, the command substitution fails, and the run dies
after one design — the plan's §7 deliverable. The same wrapper is what the UI submits
(`configs/shot_design/ui.yaml:18`), so a second concurrent simulation from the browser
returns a 502 for the same reason. The controller logged this at 12:50 as "decide at run
time"; it is a code defect, not a run-time choice.
*Fix:* take the QOS out of the wrapper and make it opt-in — drop line 4 of
`shot_design_simulate.sh` (the job is `-t 01:00:00` on `batch`, which is inside the 2 h cap
without `debug`) and let a single short run ask for it explicitly (`sbatch -q debug …` or
`SBATCH_QOS=debug`). Alternatively leave the wrapper alone and have `demo_frontier.sh` pass
`sbatch --parsable -q "${DEMO_QOS:-normal}"`. Either is one line, and both keep `-q debug`
available for the genc/one-off case.

**I2. `--k0`/`--n-predict` beyond the checkpoint's trained horizon fails as an opaque index
error — `src/shot_design/simulate/cli.py:107`.**
`cfg.k0_seed, cfg.n_predict = args.k0, args.n_predict` mutates the config *after*
`core.load_dynamics` built the model, and `frame_layout.py:46` sizes
`nn.Embedding(cfg.max_frames, d)` at construction (100 rows for this checkpoint). The only
guard downstream, `frame_layout.py:81`, compares against `cfg.max_frames`, which is a
property that re-computes to the *new* `k0 + n_predict` — so it passes and the embedding
lookup then indexes out of range (a device-side assert on an MI250X, which is the worst
place to read one). The CLI's own guard (`simulate/cli.py:82`) only checks the design seed's
frame count, so `--n-predict 120` on a 200-frame seed reaches this.
*Fix:* capture `trained = cfg.max_frames` immediately after `load_dynamics` and raise
`ValueError(f"--k0 + --n-predict = {total} exceeds the checkpoint's trained horizon
{trained}")` before the assignment; `status.json` then carries a readable error.

### Minor

**M1. `src/shot_design/design/assistant.py:305`** — `client.cfg.get("provider") == "off"`
re-implements `LLMClient.off` and misses its other spellings (`""`, `None`, `"none"`,
`"false"`, any casing/whitespace). Those cases fall through and fail later inside `chat()`
with the right message, so nothing is broken, but the duplicate is the kind that drifts.
*Fix:* `if client.off:`.

**M2. `src/shot_design/llm/client.py:133`** — `available()` is provider-aware for `agy` and
`off` but falls through to the ollama endpoint logic for an unregistered provider, while
`chat()` (line 224) now refuses one by name. A typo'd provider with an endpoint file present
therefore reports "available" and then raises at call time.
*Fix:* after the `agy` branch, `if provider not in self._providers: return False, f"unknown
llm provider {provider!r}; known: …"`.

**M3. `src/shot_design/cli.py:870`** — `design show --references --actuation-csv` silently
prints only the CSV. *Fix:* put the two flags in an `add_mutually_exclusive_group()` on the
`design show` parser (~1993).

**M4. `src/shot_design/cli.py:704-706`** — `_evalsets_prompt` catches `OSError` but not
`yaml.YAMLError`, and `entry["text"]` (line 718) raises `KeyError` for an entry that is a
mapping without `text`. A hand-edited evalset gives a traceback instead of a message.
*Fix:* `except (OSError, yaml.YAMLError)`, and `entry.get("text", "")` with an explicit
"prompt has no text" message.

**M5. `src/shot_design/simulate/cli.py:74-88`** — a job killed by the wall clock, OOM or
`scancel` leaves `status.json` at `state: "running"` forever, and `design.js:190` polls it
every 15 s indefinitely with the Simulate button disabled. *Fix (cheap):* in
`shot_design_simulate.sh`, `srun … || python -c "…mark failed…"`, or have the status route
report `stale` when `started` is older than the wrapper's wall clock.

**M6. `src/shot_design/shotdb/build.py:462-464`** — the proxy reason overwrites a reader's
own `reasons["ip"]` (e.g. "corpus address missed"), which is the one explanation of *why*
there was no trace. *Fix:* append rather than replace when a reader reason exists.

**M7. `src/shot_design/cli.py:141-142`** — `_buildable` reads every shot's text bundle up
front (`read_bundles`) even for shots whose Ip is present, and line 142 rebuilds `set(have)`
once per element (O(n²) for a 3 k list). Both are invisible at today's sizes.
*Fix:* move the bundle read inside the `ip is missing` branch, and hoist `skipped = set(have)`
out of the comprehension.

**M8. `pyproject.toml:401`** — the `shot-design-frontier` activation hardcodes
`SHOT_DESIGN_PATHS=/lustre/orion/fus187/scratch/nchen/FusionAIHub/configs/shot_design/
paths.frontier.yaml`, so a second checkout (or a colleague's) activates *this* repo's config
file. The three roots are genuinely machine-global, but the paths file is repo-relative.
*Fix:* `"$PIXI_PROJECT_ROOT/configs/shot_design/paths.frontier.yaml"`, or note the constraint
in `docs/clusters/frontier.md` if pixi will not expand it.

**M9. `src/shot_design/cli.py:682`** — `client.cfg.get(provider, {}).get("bin", provider)`
raises `AttributeError` when the provider block is present but null, which is exactly the
style `llm.yaml` uses for absent settings elsewhere. *Fix:* `(client.cfg.get(provider) or {})`.

**M10. `src/shot_design/shotdb/build.py:333-352`** — `_bundle_pulse_length_s` re-implements
`select._num` (verified character-for-character equivalent: `float(str(v).replace(" ", ""))`,
`None` on `ValueError`). Two copies of one parsing rule. *Fix:* call `select._num` (or
promote it to a shared helper) — two lines, no behaviour change.

**M11. `tests/shot_design/test_simulate_report.py:48`** — meta key `dynamics_sha256`; the
shipped producer (`simulate/cli.py:124`) writes `bundle_manifest_sha256`. Inert (the test
neither asserts it nor does `report.write` interpret meta), but it documents a contract that
does not exist. *Fix:* rename the literal.

**M12. `src/shot_design/ui/static/design.js:131-140`** — reopening a design whose simulation
already completed resets the panel and disables the button, so the only affordance is to
submit a *new* Slurm job; the existing `report.md` is unreachable from the UI. *Fix:* on
design load, poll once and render `complete`/`failed` without scheduling the timer for
`not_started`.

**M13. `src/shot_design/shotdb/ignite.py:158-186`** — the manifest↔checkpoint cross-check
compares the two modality lists positionally, so a pin whose manifest key order differs from
the checkpoint's would report a false "manifest modalities != the checkpoint's own". Correct
today (both are the canonical token order), fragile if a future pin is written from a dict
that is not. *Fix:* compare as sorted lists and report ordering separately.

**M14. `src/shot_design/shotdb/build.py:934-950`** — `write_blurbs(workers>1)` submits every
shot up front, waits for all of them, then writes the parquet once; one worker exception
(e.g. `db.get`) surfaces at `fut.result()` and discards an hour of accepted blurbs. Same
exposure as the sequential path, so not a regression. *Fix, if it ever bites:* write in
batches, or catch per-future and count failures.

**M15. `src/shot_design/simulate/core.py:139-147`** — the paired rollout relies on the global
`torch.manual_seed` although `rollout()` accepts `generator=`. Correct for the single-threaded
CLI; it would silently stop being a paired comparison if simulate were ever run in-process
next to other torch work. *Fix:* pass an explicit `torch.Generator` per arm.

## Parked-findings adjudication

**a. `tests/shot_design/test_simulate_report.py:48` stale `dynamics_sha256` — ACCEPT (not
stale in the breaking sense), with a one-word cleanup.** `report.write` copies whatever meta
it is handed into the h5 attrs and interprets only `dynamics_step`; the test never asserts
the key, so it passes and will keep passing. But the shipped caller writes
`bundle_manifest_sha256` (`simulate/cli.py:124`), so the fixture advertises a key nothing
produces. Rename it (M11); not a merge blocker.

**b. G1 `cli.py` minors — FIX NOW for the two one-liners, ACCEPT the size.** The missing
`yaml.YAMLError` catch (M4) and the silent `--references` + `--actuation-csv` precedence (M3)
are each a one-line change with an obvious right answer; do them in the same round as I1/I2.
`cli.py` at ~2 300 lines is the repo's established single-file CLI convention (every command
in this codebase lives there, and the heavy imports are already deferred per-command);
splitting it in this branch would collide with every other open lane for no behavioural gain.
Accept, and treat "split `cli.py` into `cli/` sub-modules" as its own task.

**c. F2b minors — ACCEPT all three; two are cheap cleanups.** `_bundle_pulse_length_s`
duplicating `select._num` is real but the two are semantically identical (checked), so it is
a DRY nit, not a correctness risk (M10). "No test for a bundle without a shot-table row" is
covered by contract: `text.shot_table_row` returns `{}` for a bundle with no table, `.get`
then yields `None`, and `test_build_record_without_a_bundle_still_has_no_segments` pins the
resulting shape. `_buildable` reading bundles for Ip-present shots is a wasted read only
(M7). None blocks merge. Worth recording: the 2026-09-20 04:56 fdp run gave all 3 031 list
shots a real `ip`, so the proxy path is not exercised by the current build at all — its
tests are its only coverage today.

**d. E2 `available()` — the parked concern is RESOLVED; the brief prose was the thing that
was wrong.** The shipped code is provider-aware and strict: `client.py:136-146` checks
`shutil.which(agy.bin)` for the agy provider and never consults the ollama endpoint file, and
`chat()` refuses an unregistered provider by name instead of falling back
(`client.py:224-233`). Both are pinned by tests
(`test_available_checks_the_agy_binary_on_path`,
`test_unknown_provider_raises_llmunavailable_naming_the_known_ones`). The one residual is
that `available()` itself is not strict about an *unknown* provider (M2).

**e. `tests/labeler/test_labels_layout_integration.py` 5 820-shot union — ACCEPT as
pre-existing and out of scope.** `git diff 702aaae..nathan_dev -- data/events tests/labeler`
shows this branch only touches the import; the 56/33 literals are RWM-era and the shot union
grew at `55a9076` (2026-09-19, pre-plan). Do not fix it here — re-deriving the expectation
across six event tables is labeler work. Recommend (follow-up, not a blocker) marking it
`@pytest.mark.real_data` (the marker already exists in `pyproject.toml`) so a default
`pytest tests/labeler` is not hostage to a 30-minute test whose numbers are known wrong.

## Spec alignment (global constraints)

| Constraint | Verdict | Evidence |
|---|---|---|
| Work on `nathan_dev`; never force-push; `nathan_fm`/`dev-nathan`/`main` untouched | satisfied | `git branch -v`: `nathan_dev` ahead 95, `nathan_fm` still at `702aaae`, `main` at `abdf2e0`; lane branches `nathan_dev-e`/`-h` merged, not rebased over anything shared |
| Every pixi call on Frontier `--frozen` | satisfied | no `pixi` at all in `scripts/slurm_frontier` or `scripts/shot_design` (wrappers call `$PY` = `.pixi/envs/shot-design-frontier/bin/python` directly); `docs/clusters/frontier.md:23,30,140` all use `--frozen` and state the rule; the bare `pixi run` lines left in `docs/` are Stellar sections, plus two pre-existing `scripts/train_*_ast.sh` |
| Frontier roots `SHOT_DESIGN_DATA_ROOT` / `LABELER_ROOT` / `SHOT_DESIGN_CORPUS`; tests never write there | satisfied | `configs/shot_design/paths.frontier.yaml` + `_shot_design_common.sh:18-21` + `pyproject.toml:399-404`; tests use the `paths` fixture (tmp_path) and the new autouse `_no_production_frame_codes_cache` fixture strips `frame_codes_cache` so no test can reach the read-only production corpus; `test_paths_frontier.py` only asserts strings |
| v4: 15 modalities, every vocab 1000, `t0_start_s: 1.0`, 1209 tok/frame, actuators `(F, 88)` | satisfied | `ignite_modalities.yaml` `model:` block (`generation: v4`, `frame_tokens: 1209`, `t0_start_s: 1.0`, 15-entry `production_vocabs`/`families`/`n_tok`) + `mirnov` modality entry; `design/seed.py:74-80` derives `MODALITIES`/`VIDEO_MODALITIES` from `model.families`; `program_reference.validate_cache` checks the generation first and `(F, 88)` float16; `tests/shot_design/test_ignite_v4.py`, `test_phaseb_frame_layout.py` |
| Pinned copies with sha256 under `<models_dir>/IGNITE_v4/` | satisfied | `ignite.pin_bundle` copies resolved (not symlinked) files and writes `codecs/MANIFEST.json` with a `sha256` per file; `local_name: IGNITE_v4`; `load_codecs` re-hashes the codecs before loading and `model --check` re-hashes everything plus the checkpoint cross-check |
| Production frame-code cache read first | satisfied | `build.frame_codes_dirs` puts `model.frame_codes_cache` first and is now the single source for `has_frame_codes`, `corpus select` and `program_reference._cache_path`; `ignite.frame_codes(use_cache=True)` serves from it and *raises* on a vocab mismatch; the G-ENC gate passes `use_cache=False` |
| Slurm `-A fus187 -p batch`; short jobs `-q debug` (≤2 h, one at a time); logs to `$SHOT_DESIGN_DATA_ROOT/runs/slurm/%j.out` | satisfied with one documented deviation, one defect | all five wrappers carry `-A fus187` and log to `.../runs/slurm/%j.out` (`%A_%a` for the array) and are pinned by `test_slurm_frontier_scripts.py`. Deviation: `shot_design_build.sh`/`shot_design_encode.sh` use `-p extended`, justified in-file by a measured `sbatch --test-only` refusal (batch caps a 1-node job at 2 h) — accepted. Defect: `-q debug` baked into `shot_design_simulate.sh` collides with the "one at a time" limit the constraint itself states (finding I1) |
| No production writes without the owner's go-ahead | satisfied | ledger records the owner's 00:45 go-ahead for the F-phase; nothing in the reviewed code writes to `proj-shared` outside the operator commands (`model --pin`, `build`, `encode`), and `production_cache_path` is documented and used read-only |
| No Claude/superpowers material under `docs/` | satisfied | `grep -rli "superpowers\|claude" docs/ website/` → empty; the SDD workspace lives under `.claude/superpowers-runtime/` |
| Ruff `line-length = 88` | satisfied for this plan's code | every new file (`simulate/*`, `llm/agy.py`, the new tests) is ≤88 columns; the long-line counts in older modules are pre-existing and unenforced (`[tool.ruff]` sets only `line-length`, and the default rule set excludes E501) |
| One-line imperative commit subjects | satisfied (spot check) | `git log --oneline a978f0e..77a0678` reads as `<area>: <what>`; the one non-imperative subject (`25667aa`) was already parked in the ledger |
| TDD (failing test first) | satisfied by evidence | every new behaviour in Pass 1 has a matching behavioural test (agy provider, simulate core/decode/report/cli, UI simulate routes, proxy segments, quotas, v4 pin/validate, Frontier jobstats); the ledger records the red→green order per task |
| Nothing copied from Stellar; no `sql/`; Gemini Flash via `agy` the only LLM | satisfied | `paths.frontier.yaml` still *points* at `${text_root}/sql/...` (so a future copy needs no code change) but nothing on the branch copies or requires it; `llm.yaml` `provider: agy` with `models: gemini-3.8-flash-{high,low}`; the `ollama:` block is retained deliberately per spec §5 as the Stellar provider, and `design/assistant.py` no longer names Gemma anywhere |

## Verdict

**Needs fix round.**

Must fix before merge to `main`:

1. **I1** — `scripts/shot_design/demo_frontier.sh:13` / `scripts/slurm_frontier/shot_design_simulate.sh:4`: `-q debug` (MaxSubmitJobsPU=1) makes the three-prompt demo abort after one design and caps the UI at one queued simulation.
2. **I2** — `src/shot_design/simulate/cli.py:107`: guard `k0 + n_predict` against the checkpoint's trained `max_frames` before overriding the config.

Worth folding into the same round (each one line, no design decisions): **M1** (use
`client.off`), **M2** (strict `available()`), **M3** (mutually exclusive `design show` flags),
**M4** (`yaml.YAMLError`), **M9** (null provider block).

Everything else is optional polish or accepted as out of scope. The v4 contract, the
prod-cache-first precedence, the real-Ip-over-proxy precedence, the agy provider and its
error mapping, the Slurm submit-dir pattern and the test hermeticity are all correct and
well covered; the branch is in good shape apart from the two items above.
