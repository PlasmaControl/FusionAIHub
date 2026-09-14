# L-D1b implementation report

Worktree: `/scratch/gpfs/nc1514/FusionAIHub-LD1b`
Branch: `recommender-LD1b`
Base: `d649520`

## Delivered

1. `databases.FORMAT_COLUMNS`, schema version 1, and `validate_format` define the
   eight-column common CSV projection. Validation rejects wrong columns/order,
   invalid lexicon/evidence/source values, malformed/non-object/nonfinite JSON,
   fractional shots, nonfinite/backwards times, and invalid confidence.
2. Eleven category directories mirror the user's layout, with one-line inventory
   mappings and tracked raw/format placeholders where needed. Both RWM originals
   were moved with `git mv`, byte-identical to `6de489d`. The two generated format
   CSVs and sidecars are committed. `labels_format.py` uses manifest-registered
   adapters, sorted rows without deduplication, fixed float/JSON formatting and a
   recorded manifest revision timestamp; regeneration tests compare all bytes.
3. `databases.py` loads `format/<format_stem>.csv` exclusively, retaining stored
   evidence and confidence and parsing JSON attrs. The source names, event IDs,
   NaN coverage and absent-shot/no-source contract are preserved. Neither `run.py`
   nor `pipeline.py` nor `heuristics.py` was changed.
4. `labels_extend.py` reads the selected per-shot event/source products, preserves
   all contributing producer/run/git triples, and writes the common table. More
   than 50,000 events selects a per-shot summary (50,000 itself remains a full
   table). Summaries include zero/missing shots with unknown values left empty;
   coverage is a bounding range, not an assertion of continuity. A rerun removes
   only the stale alternate output and sidecar. Input event stores are never written.
5. Schema/converter/loader/extension regressions and two real RWM CLI integration
   tests cover the specified behavior. IDEATE fixtures now use format/. This
   worktree's `.mcp.json` cwd was corrected from the main checkout to this worktree,
   fixing the artifact explicitly identified in the brief. Two pre-existing Ruff
   style findings in the labelmaker golden-fixture generator were fixed without
   regenerating its data.
6. `data/labels/README.md` documents the convention, schema, metadata, add-table
   process, producer lifecycle and summary rule. The curated-tables subsection of
   `docs/LABELMAKER.md` matches the new implementation.

## Data evidence

- Original RWM tables still yield **56 events / 33 source rows / 33 named shots**
  through `events --databases-only` and format/ (integration-tested).
- The actual 500-shot RWM scan completed successfully with **zero events and zero
  source rows**. Its run ID is `ld1b-rwm-recommender-v1`, with run products only in
  `/tmp/ld1b-rwm-run` and log `/tmp/ld1b-rwm-run.log`.
- Committed output:
  `data/labels/resistive_wall_mode/extend_rwm/recommender_v1.csv` (header only) and
  `recommender_v1.meta.json`. The sidecar records the successful scan's run/git
  provenance and 500 requested shots. This is "ran, zero", not a detector claim
  that RWM is absent.
- No `extend_*` placeholder directories were created for missing producers.
- Raw byte identity is checked directly against git `6de489d` and by tests; the
  raw SHA-256 values are recorded in the committed format sidecars.

Commands used for the actual RWM run (the YAML's 500 shots were first written to
`/tmp/ld1b-rwm-run/recommender_v1.txt`):

```bash
PYTHONPATH=$PWD/src /scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/python \
  -m labelmaker.run events --databases-only \
  --shot-file /tmp/ld1b-rwm-run/recommender_v1.txt --root /tmp/ld1b-rwm-run \
  --run-id ld1b-rwm-recommender-v1
PYTHONPATH=$PWD/src /scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/python \
  scripts/labelmaker/labels_extend.py --category resistive_wall_mode --producer rwm \
  --shot-list configs/ideate/shot_lists/recommender_v1.yaml --root /tmp/ld1b-rwm-run \
  --run-id ld1b-rwm-recommender-v1 \
  --out data/labels/resistive_wall_mode/extend_rwm/recommender_v1.csv
```

## Test-first and review evidence

- Baseline database tests: 38 passed.
- Before implementations: 20 schema tests failed for the absent validator;
  11 converter tests failed for the absent raw/format artifacts/adapter fields;
  the loader migration produced 24 expected failures; all six original extension
  cases failed because the writer did not exist.
- Focused green checkpoints: schema/database 58; schema/converter 31;
  loader/pipeline/schema/converter 116; IDEATE phenomena 79; original extension
  boundary tests 6; final review-regression/CLI integration set 11, all with
  `-W error`.
- Independent read-only review found two provenance defects. Three new regressions
  failed before fixes: zero/error-only producers were lost by a phenomenon
  selector, and unrelated successful runs could certify empty exports. Known
  family/manifest producers now supply zero-only provenance, and `--run-id`
  requires a matching successful databases-only zero scan of the selected tables.
  The reviewer rechecked the fixes and reported no remaining important/critical
  issue in that diff.

## Final verification

The exact commands required by the brief were run from this worktree with
`PYTHONDONTWRITEBYTECODE=1` exported to avoid writing bytecode into shared
installation directories:

```bash
cd /scratch/gpfs/nc1514/FusionAIHub-LD1b && PYTHONPATH=$PWD/src pixi run --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker python -m pytest tests/labelmaker -q -W error
HF_HUB_OFFLINE=1 PYTHONPATH=$PWD/src IDEATE_DATA_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/ideate LABELMAKER_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker IDEATE_CORPUS=/scratch/gpfs/EKOLEMEN/foundation_model /scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/ideate-cpu/bin/python -m pytest tests/ideate -q -W error
```

- **Labelmaker: 1,446 passed, 2 skipped in 630.43 s; exit 0.** Log:
  `/tmp/ld1b-labelmaker-final.log`.
- **IDEATE: 930 passed in 632.48 s; exit 0.** Log:
  `/tmp/ld1b-ideate-final.log`.
- HEAD remained `db86a3e` throughout both runs; commits 5 and 6 were created only
  after the full labelmaker process exited. No source/test code changed after
  these final suite runs.
- After pytest's successful summary, the external XRootD shutdown finalizer emitted
  a PyTorch `reduce_op` deprecation FutureWarning. It did not fail a test or change
  the process's exit code; no warning filter or dependency patch was added.
- `git diff --check` passed.

Ruff 0.16.5 passes on the affected package trees and every labelmaker script:

```bash
/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/ruff check \
  src/labelmaker src/ideate tests/labelmaker tests/ideate scripts/labelmaker
```

**The unrestricted repository-wide Ruff requirement is not met by the existing
baseline.** `ruff check src tests scripts/labelmaker` reports 1,222 remaining
findings, all pre-existing outside these clean package scopes. A read-only
`git archive d649520` baseline into `/tmp/ld1b-lint-baseline` reported 1,224.
Comparing `(relative file, rule, message)` multisets found **zero introduced and
two fixed findings**. No diagnostics were suppressed, no lint configuration was
weakened, and unrelated model/test code was not mass-reformatted. Logs:
`/tmp/ld1b-ruff-final.log`, `/tmp/ld1b-ruff-baseline.json`,
`/tmp/ld1b-ruff-current.json`. This is a documented baseline limitation, not a claim
that the unrestricted check passed.

## Literal interpretations and boundaries

- The common metadata's conversion `made_at` is manifest-controlled revision time,
  so rerunning the converter reproduces the sidecar too. Extension `made_at` is
  wall-clock export time.
- Categories with no specific lexicon ID yet say "pending" in their README.
  H/L/I-mode intervals, q-min bands and WPQH do not borrow a scientifically
  different existing ID. Their producer tasks must register IDs and category
  mappings when implementing those labels.
- `--producer rwm` is the brief's permitted phenomenon selector: it aggregates the
  RWM curated sources while preserving actual sources whenever there are rows.
- Source selectors support arbitrary source IDs. Phenomenon selection learns
  sources from matching events and the manifest; unambiguous zero-only families
  are also registered in `PHENOMENON_SOURCES`. A new producer with only zero/error
  records must register that mapping or be exported by exact source ID.
- Full-table `n_shots` means represented (event-bearing) shots; summary `n_shots`
  means represented summary shots. Metadata also carries requested-shot and
  event-bearing-shot counts explicitly.
- The per-shot run products are temporary `/tmp` evidence, not committed data.
  Production roots were read-only; no SLURM commands/jobs were used. All source,
  artifact and temporary writes stayed in the supplied worktree or `/tmp`, apart
  from the shared Git metadata inherently required by the requested worktree
  commits. The test command used the shared Pixi environment specified by the brief.
- The user-provided `.superpowers/sdd/task-LD1b-brief.md` remains untouched and
  untracked. No `docs/superpowers/plans/**` edits, merges, pushes or PRs were made.

## Deliverable commits

Every commit uses `labelmaker:` and the exact trailer
`Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

1. `99c6749` — define and validate the common label table schema.
2. `8db24b6` — preserve raw labels and generate reproducible format tables.
3. `88f232a` — load curated events exclusively from format tables.
4. `db86a3e` — export producer tables with bounded per-shot summaries.
5. `caf3be1` — verify format ingestion and extension provenance.
6. This documentation/report commit — document the label table lifecycle and verification.

## Review fixes

This section supersedes the original implementation notes above. Fix base: `8d01d5a`; worktree: `/scratch/gpfs/nc1514/FusionAIHub-LD1b`; branch: `recommender-LD1b`. The six requested items were completed in order. The main checkout was only read; no other worktrees or plans were edited, and no SLURM, merge, push, or external publication was used. All writes were confined to this worktree and `/tmp/ld1b-fix`, apart from the shared Git metadata inherent in the requested worktree commits.

### Finding → commit → verification

| Review finding | Fix commit | Evidence |
|---|---|---|
| 1: worktree MCP cwd | `6e9700e` | `.mcp.json` is byte-identical to `recommender`; the IDEATE cwd assertion is the expected worktree artifact. |
| 2: authored README preservation | `85e706b` | Four original byte prefixes verified against the main checkout; exactly one appended `## Tables` each; hashes below. |
| 3: category layout/inventory rename | `85e706b` | Twelve category names match the main checkout; `minimum_safety_factor` replaces `q_min`; `poloidal_beta` added; inventory bytes unchanged; 11 format tests passed at this checkpoint. |
| 4: corrupt format CSV exit | `d712216` | `test_an_invalid_format_csv_stops_the_run_before_any_shot`: all four cases (invalid phenomenon, missing column, malformed CSV, missing file) failed with `0 != 7` before the change. All 89 pipeline/database tests then passed. |
| 5: temporary provenance path | `33372e6`, `b7c56e3` | `test_rwm_500_scan_exports_empty_table_with_completed_run_provenance` asserts `full_events_root=events` and `run_metadata=runs/events/<run-id>.json`. Committed metadata contains no `/tmp` pointer. |
| 6: long missing-shot list | `33372e6`, `b7c56e3` | `test_missing_event_shots_are_bounded_with_a_complete_list_sidecar`; metadata contains count + first 20 + relative full-list filename. Committed metadata shrank from 6,933 to 1,393 bytes. |
| 7: mixed producing sources | `33372e6` | `test_extend_refuses_a_phenomenon_spanning_sources_before_writing` covers detector/text evidence on different shots; `test_single_source_phenomenon_requires_the_actual_source_directory` and both zero/error-source cases enforce source directories. |
| 8: hard-coded identity-test stem | `33372e6` | `test_rwm_raw_bytes_match_original_commit_and_format_regenerates` now uses `raw_path`, `path`, and `raw_file`; the existing adapter test also uses distinct raw/format stems. |
| 9: inventory count and old paths | `85e706b`, `33372e6` | Parse confirms 37 rows, 15 at Priority 5; spec and README use `discrete_labels.csv` and current `raw/` paths. Historical Git comparisons deliberately retain the old commit paths. |
| 10: missing guards | `33372e6` | `test_gitignore_allows_new_label_tables_to_be_tracked` checks tracked and future paths with `git check-ignore --no-index`; `test_discrete_label_inventory_has_the_roadmap_input_schema` checks exact columns, at least 37 rows, and valid priorities. |

Seven housekeeping regression cases failed before their implementation; afterward the complete format/extend/layout set passed (26 tests, `-W error`). The two requested guard tests cover existing correct behavior.

### Regeneration

`/tmp/ld1b-fix/regeneration.log` records two byte-identical regenerations of both format CSVs and both metadata files. Both raw files also matched the user's main-checkout originals and Git commit `6de489d`. The new 500-shot run `ld1b-fix-rwm-recommender-v1` completed with zero events and source rows; its run JSON is `/tmp/ld1b-fix/rwm-run/runs/events/ld1b-fix-rwm-recommender-v1.json`.

The RWM extension CSV is byte-identical to its previous committed version. Its metadata intentionally changed for the review fixes and the fresh run. The extension CSV, metadata, and complete missing-shot sidecar all reproduced byte-for-byte when replaying the recorded export timestamp with the same run input; `made_at` remains wall-clock time in normal operation. SHA-256 evidence is in `/tmp/ld1b-fix/regeneration-sha256.json`. No production event store was changed.

### Authored README SHA-256 comparisons

The user files have no final newline. Each complete original was copied unchanged as a byte prefix, then `\n\n## Tables\n` and the technical section were appended. Whole-file hashes therefore differ from the originals as required by the appended section. Sparse user READMEs were preserved as authored; only the other categories received the full template.

| Category | Original SHA-256 = copied prefix SHA-256 | Complete README with Tables SHA-256 |
|---|---|---|
| `alfven_eigenmode` | `de6c5ddc97aa184e653d191eb6fccaa69e2801602f204917e59b53bcdda93466` | `e34f037514d44724b5de5933b2f9c29d9be23d054bbd96ca612c0b3f5703ed0f` |
| `low_confinement_mode` | `0fc9e32049723a6bc2a8506d77b6a4749fed234f662a6512c54e40f218887f42` | `6b73e36aa61f22b52352e5a2fdf8a84375ce219276b22a1683cc30be2c6abc22` |
| `minimum_safety_factor` | `5f2cf1ad383f23b76e79e92c5d53799c6bc96370a665840f3a0aea2257fad53e` | `02f907954a43a3f43dd7f3b06db72469434c474bf3ad621569edbc3a1adfe461` |
| `resistive_wall_mode` | `c219b3f95b665715a481f7d8367dcb82ccf504f95d998ac44e9b303dcd3dc44c` | `0644663d0a31d512c86010b63a2e7739e33cb929139b7598dfc3053cd0bdf63d` |

### Final verification

Both full suites ran with `-W error` at HEAD `b7c56e3`. No commit was made while either labelmaker run was active. Both final subprocesses have exited; the only subsequent change is this report.

| Check | Result | Log |
|---|---|---|
| Labelmaker | **1,454 passed, 2 skipped, 0 failed**, exit 0, 78.08 s | `/tmp/ld1b-fix/labelmaker-verified.log` |
| IDEATE | **929 passed, 1 expected worktree failure**, exit 1, 99.31 s | `/tmp/ld1b-fix/ideate-verified.log` |
| Ruff, requested scope | **All checks passed**, exit 0 | `/tmp/ld1b-fix/ruff-final.log` |
| Byte-identity regeneration | **Passed**, exit 0 | `/tmp/ld1b-fix/regeneration.log` |
| `git diff --check` and MCP identity to `recommender` | **Passed** | Verified after the final suites |

The IDEATE failure is exactly `tests/ideate/test_mcp.py::test_the_project_mcp_config_points_at_this_server` at line 891: the restored cwd is `/scratch/gpfs/nc1514/FusionAIHub`, while the running worktree is `/scratch/gpfs/nc1514/FusionAIHub-LD1b`. It is explicitly expected by the fix brief and was not suppressed or repaired with a worktree-specific commit. Both skipped labelmaker cases are live FDP fetch tests gated on `--run-live` or `LABELMAKER_FDP=1`. The external XRootD shutdown finalizer emitted its existing PyTorch `reduce_op` FutureWarning after the successful labelmaker summary; process exit remained 0.

Commands (from this worktree):

```bash
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$PWD/src
export TMPDIR=/tmp/ld1b-fix
export XDG_CACHE_HOME=/tmp/ld1b-fix/cache
export MPLCONFIGDIR=/tmp/ld1b-fix/matplotlib
export HF_HOME=/tmp/ld1b-fix/huggingface
export HF_HUB_OFFLINE=1
export TORCH_HOME=/tmp/ld1b-fix/torch

pixi run --frozen --no-install \
  --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker \
  python -m pytest tests/labelmaker -q -W error -ra \
  --basetemp=/tmp/ld1b-fix/pytest-labelmaker-verified

IDEATE_DATA_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/ideate \
LABELMAKER_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker \
IDEATE_CORPUS=/scratch/gpfs/EKOLEMEN/foundation_model \
  /scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/ideate-cpu/bin/python \
  -m pytest tests/ideate -q -W error -ra \
  --basetemp=/tmp/ld1b-fix/pytest-ideate-verified

/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/ruff check --no-cache \
  src/labelmaker src/ideate tests/labelmaker tests/ideate scripts/labelmaker
```


### Decisions and deviations

- `poloidal_beta` has no row in the 37-row inventory and no registered phenomenon ID; its README says both explicitly. No inventory row or lexicon ID was invented.

- The allowed refusal strategy was used for multi-source phenomenon selectors. A single-source phenomenon selector can still be used with the actual source directory. The requested empty `extend_rwm` scan remains valid because it contains no producing-source rows.

- Whole authored README hashes cannot equal originals after appending a Tables section; preservation is verified on the complete original byte prefix, with both hashes reported above. This is the literal copy-then-append interpretation.

- Pixi used `--frozen --no-install` with the main checkout's existing manifest/environment, preventing installation or lockfile changes. Bytecode was disabled; temporary files and caches were redirected to `/tmp/ld1b-fix`. A copy of the already cached MiniLM checkpoint was staged in that scratch cache for offline tests.

- An initial attempt to run labelmaker with an empty scratch data root produced one existing model-weight lookup failure and 13 skips (1,442 passed). The final run restores the brief's read-only input roots rather than modifying or skipping that test; all test/run output stays in scratch. IDEATE was also rerun with the exact input roots from the brief.

- The original brief/review input files remain unmodified and untracked. No product code or test was changed to suppress the expected MCP worktree assertion.

### Fix commits

- `6e9700e` — labelmaker: restore shared MCP configuration from recommender
- `85e706b` — labelmaker: adopt user category layout and preserve authored READMEs
- `d712216` — labelmaker: reject invalid format tables before processing shots
- `33372e6` — labelmaker: isolate extension sources and compact export provenance
- `b7c56e3` — labelmaker: regenerate RWM extension with portable scan metadata
- This report commit: `labelmaker: document LD1b review fixes and verification`.

Every fix commit has the exact trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
