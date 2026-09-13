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
