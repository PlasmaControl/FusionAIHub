# Task LLM1 report

- Worktree: `/scratch/gpfs/nc1514/FusionAIHub-LLM1`
- Branch: `recommender-LLM1`
- Starting commit: `7f46126` (the committed LLM1 brief)
- Date: 2026-09-15

## Delivered

- Prompt v5 requests three sentences: goal, success/failure, one finding (or the
  literal no-findings sentence), with a 90-word cap. The fallback stays the
  existing header + outcome line.
- `Blurb.candidate` preserves the normalized model reply for preview, including
  rejected candidates. `make(..., cache=False)` bypasses request caching for a
  preview; ordinary deterministic requests retain the existing cache behavior.
- Builds, adds and backfills include `blurb_model` and `blurb_prompt_version`.
  The version column uses nullable integers so adding to an older table cannot
  silently turn versions into floating-point values.
- `blurb --limit N --dry-run` selects eligible shots in shot order, prints the
  shot, source length in characters, candidate, gate verdict/reason and final
  text, and writes no database or request-cache files. `--all` still selects all
  rows; `--limit 0` does nothing; negative limits are rejected.
- The default provider is Ollama. A missing endpoint leaves the client unavailable
  without connecting; the template remains usable. The start hint names the new
  Slurm script and retains explicit `base_url` setup as an alternative.
- Ported `serve_llm.sh`, `serve_llm.sbatch` and `blurb_all.sh`. The serving script
  uses one mandated frozen Pixi invocation to resolve paths, all model tags and
  the environment interpreter. It then uses that interpreter for endpoint JSON.
  The shell's HOME is unchanged; only Ollama child processes receive the configured
  keypair home. No binary download/copy path exists. Installed model tags are
  checked with `ollama show` before the retained pull loop, avoiding registry
  downloads when the models are already present.
- Updated the LLM section of `docs/SHOT_DESIGN.md` with the contract and runbook.

## Exact prompt v5 (max_words = 90)

```text
You summarise DIII-D tokamak shots for physicists scanning a list of past shots to find one worth looking at. Write exactly three plain sentences: the first says what the experiment or this shot set out to do; the second says whether it succeeded (say when success is unknown); the third gives one interesting finding from the operator entries or shot brief. When nothing notable is recorded, the third sentence must be exactly: No notable findings were logged. Use only the text given. Do not invent or round numbers; prefer no numbers at all. Do not use quotation marks -- write in your own words, never quote the operators. If the shot ended in a disruption, a fast current quench or was terminated early, the second sentence must say so plainly. No headings, no lists, no preamble, exactly three sentences, and never more than 90 words.
(prompt v5)
```

The user message remains `source_text(rec)`: shot/date/campaign; mini-proposal
name/purpose; run title; shot brief; up to eight quotable operator entries;
chief-operator status; outcome line; verdict. The resolved default model is
`gemma4:26b`, with `reasoning_effort: none` and deterministic temperature zero.
The prompt version remains part of the cache key through the system message.

## Gate rules

In order, reject:

1. Empty/whitespace-only text.
2. More than the configured maximum number of whitespace-separated words (90).
3. Straight double or curly double quotation marks.
4. Shot references absent from the source (allowing the record's own shot number).
5. Numbers absent from the source, using the existing `describe.extract_facts`.
6. Anything other than three complete sentences: count `[.!?]+` terminator groups
   followed by whitespace or end of text and require final punctuation. Decimal
   points inside a number do not count. A unit such as `kA.` at sentence end does.

Failure retains the template and its reason; a dry run also prints the candidate.
The gate is mechanical: it does not prove semantic fidelity, independently detect
all rounded numbers, or enforce the meaning of each sentence. Those instructions
are in the prompt, and the operator's five-shot preview remains necessary for a
real model quality check. Templates are intentionally exempt from three sentences.

## Table and manifest contract

| Column | Meaning |
| --- | --- |
| `blurb` | Final accepted reply or unchanged header + outcome fallback |
| `blurb_source` | `llm` if accepted, otherwise `template` |
| `blurb_model` | Resolved configured model tag for the generation attempt, including a template outcome |
| `blurb_prompt_version` | Integer prompt version for the attempt; legacy untouched rows can be null |

`only_missing=True` selects template/non-model rows, empty text, and missing or
older prompt versions. Current/future model versions are retained unless `--all`
is used. The limit applies after selection. A normal write updates all four
columns for selected rows, atomically replaces `shots.parquet`, and then atomically
replaces `manifest.json` (separate replacements, not a joint transaction).

`manifest.json["blurbs"]` contains `llm` and `template` counts for the entire table,
plus the latest writing pass's `model` and integer `prompt_version`. A limited run
may therefore leave mixed per-row versions; row provenance is authoritative for
individual shots. No-op and dry runs do not rewrite files or metadata.

## Serving contract

Configuration defaults reuse the existing installation:

- `ollama_bin_dir`: `/scratch/gpfs/EKOLEMEN/nc1514/shot-recommender/bin/ollama`
- `ollama_models_dir`: `/scratch/gpfs/EKOLEMEN/nc1514/shot-recommender/models/ollama`
- `ollama_home_dir`: `/scratch/gpfs/EKOLEMEN/nc1514/shot-recommender/ollama_home`

Missing/non-executable `<ollama_bin_dir>/bin/ollama` exits 2 naming the path.
An already answering HTTP port exits 2 before any write or child startup.
Defaults: `OLLAMA_CONTEXT_LENGTH=16384`, `OLLAMA_KEEP_ALIVE=24h`,
`OLLAMA_MAX_LOADED_MODELS=2`. Endpoint publication follows child readiness and
model checks, using a temporary file + rename. JSON keys are exactly `url`,
`models`, `host`, `job_id`, `started`, `version`. On exit/signal, cleanup removes
only a handle with this run's URL AND start time, preserving a replacement even
at the same URL, then kills/reaps its own child.

Slurm: `-J shot-design-llm -p gpu --gres=gpu:1 -c 8 --mem=64G -t 04:00:00`,
output `/scratch/gpfs/EKOLEMEN/nc1514/ideate/llm/slurm-%j.out`, submit-root working
directory, bind `$(hostname -s):11434`. For another data root, override the literal
Slurm output directive with `sbatch --output` and create that root's `llm` first.
Pixi activation's data-root override is documented in `docs/SHOT_DESIGN.md`.

## Exact operator runbook (after merge; not executed by this task)

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

# Submit Ollama on a GPU node; note the printed job ID.
mkdir -p /scratch/gpfs/EKOLEMEN/nc1514/ideate/llm
sbatch scripts/shot_design/serve_llm.sbatch

# Wait for publication. If delayed, check the job and its slurm-JOB_ID.out log.
until test -s /scratch/gpfs/EKOLEMEN/nc1514/ideate/llm/endpoint.json; do sleep 2; done
cat /scratch/gpfs/EKOLEMEN/nc1514/ideate/llm/endpoint.json
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e ideate-cpu python -m shot_design llm

# Inspect five samples, including any rejected candidate and its template fallback.
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e ideate-cpu python -m shot_design blurb --all --dry-run --limit 5

# After reviewing the preview, backfill all shots from the login node.
bash scripts/shot_design/blurb_all.sh

# Ctrl-C the existing UI serve process in its terminal, then reload the DB snapshot.
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e ideate-cpu python -m shot_design serve --port 8765
```

Match the endpoint's job ID to the submitted job and look for `ready:` in the
server log before the preview. The `llm` command prints discovery metadata and
is not a live health probe. The backfill wrapper expands to:

```bash
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e ideate-cpu python -m shot_design blurb --all
```

The model server is independent of the UI. To stop/renew the GPU allocation,
`scancel JOB_ID`, wait for that job to exit, then submit `serve_llm.sbatch` again.
The endpoint is rediscovered by the client. Restarting the UI after a backfill is
what refreshes the already loaded table snapshot.

## Verification

Tests were written and run before implementation. The initial focused run showed
20 failures / 35 passes for blurb/client requirements; all five initial shell
checks failed because the scripts did not yet exist. The additional legacy-add
regression failed with `float64` versions (`NaN`, `5.0`) before the nullable-integer
fix, then passed.

The first full shot_design run passed: `1401 passed in 267.31s (0:04:27)`.
The final run below also includes the legacy-add fix and same-URL replacement
cleanup regression.

The initial labeler run, without numerical thread limits, reported `2 failed,
1773 passed, 3 skipped in 447.92s (0:07:27)`. Both failures were existing pooled
inference checks in `test_l14perf_identity.py`: an exact probability comparison
and the second shot's two-second deadline. With `OMP_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, and `OPENBLAS_NUM_THREADS=1` (also used in the project's
L14perf report), that entire file passed: `36 passed in 56.40s` with `-W error`.
No labeler implementation or tests were changed. The final whole-suite run uses
those numerical thread limits.

Final shot_design command (exit 0):

```bash
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e ideate-cpu env PYTHONPATH=/scratch/gpfs/nc1514/FusionAIHub-LLM1/src PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 pytest /scratch/gpfs/nc1514/FusionAIHub-LLM1/tests/shot_design -q -W error -p no:cacheprovider
```

```text
1403 passed in 135.38s (0:02:15)
```

Zero skipped. This includes fake-httpx v5 requests and fact gating, current/stale/
unknown/future version selection, configured model overrides, temporary-database
backfills, dry-run byte/mtime equality including the cache, legacy incremental
adds, and shim-only serving/cleanup/double-start/missing-binary tests.

Final labeler command (exit 0):

```bash
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker env PYTHONPATH=/scratch/gpfs/nc1514/FusionAIHub-LLM1/src PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 pytest /scratch/gpfs/nc1514/FusionAIHub-LLM1/tests/labeler -q -W error -p no:cacheprovider
```

```text
1775 passed, 3 skipped in 277.28s (0:04:37)
```

After pytest's successful summary, the installed XRootD client's interpreter
shutdown finalizer printed this existing dependency warning (also present in the
initial run); it was not suppressed and the command exited 0:

```text
.../XRootD/client/finalize.py:46: FutureWarning: `torch.distributed.reduce_op` is deprecated, please use `torch.distributed.ReduceOp` instead
  if isinstance(obj, File) and obj.is_open():
```

Static validation (all exit 0):

```bash
bash -n scripts/shot_design/serve_llm.sh
bash -n scripts/shot_design/serve_llm.sbatch
bash -n scripts/shot_design/blurb_all.sh
/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/ruff check --no-cache src/labeler src/shot_design scripts/labeler scripts/shot_design tests/labeler tests/shot_design
git diff --check
```

Ruff output: `All checks passed!`; syntax and diff checks produced no output.
The report's exact prompt block was compared to `_system(90) + "\n(prompt v5)"`
using the mandated environment interpreter; they match.
The mandated frozen launcher emits its existing network-filesystem cache notice
and redirects repodata to `/tmp`; no install/lock/update was requested.

## Scope and remaining operator work

No real Ollama process was started; lifecycle tests run a fake Python HTTP shim
under `tmp_path`. No Slurm jobs were submitted and no production backfill was run.
No binary/model files were copied or downloaded, and no production data, endpoint,
shot-recommender or labeler stores were written. Tests use temporary databases.
No dependency installation, lock or update commands were run through Pixi.

The find-docs skill was used to query current official Ollama docs via Context7;
its npm cache was directed to `/tmp`, outside the project/data roots. The port also
used the two specified previous-project scripts as read-only reference.

No UI files, `retrieval/describe.py`, `shotdb/store.py`, `schema.py`, `SearchHit`,
or `docs/superpowers/**` were changed. Table provenance needs no schema additions;
the sibling UI task owns presentation and schema integration.

The real Gemma five-shot quality check, production backfill, Slurm deployment and
UI restart remain operator actions as required by the brief.

All suites exited before commits. Handoff checks require clean `git status
--short` and no `.pixi` in this worktree; dependency manifests and lockfiles remain
unchanged.
