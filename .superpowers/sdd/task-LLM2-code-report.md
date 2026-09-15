# Task LLM2 code report

## Implemented

- Prompt v6 preserves the v5 three-sentence contract and instructs the model to copy
  abbreviations/symbols verbatim, never expand or introduce them, and never spell out quantities.
- The gate rejects unsupported number words and abbreviations after the existing shot/number
  checks and before sentence counting. It handles whitespace/hyphen tokenization, Unicode boundary
  punctuation, acronym plurals, source substring matching with hyphens removed, and DIII-D.
- `write_blurbs(..., shots=...)` validates the complete requested set before generation, sorts and
  deduplicates targets, ignores `only_missing` for targeted runs, and applies `limit` afterward.
- All build, add, and backfill manifest paths now include a non-null prompt-version histogram.
- `shot_design blurb --shots N [N ...]` supports targeted writes/dry runs and rejects `--all` with
  exit 2 before opening the model.
- Prompt-version pins were updated to v6, including the LLM config contract test. Mixed v5/v6 and
  nullable legacy provenance remain covered.

## TDD evidence

Initial focused RED command:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/scratch/gpfs/nc1514/FusionAIHub-LLM2/src HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e ideate-cpu python -m pytest tests/shot_design/test_blurb.py tests/shot_design/test_blurb_provenance.py -q -W error -p no:cacheprovider
```

Result: `14 failed, 36 passed`. Expected failures covered the new gate stages/order, v6 prompt and
config provenance, histogram, targeted API and validation/limit behavior, CLI targeting/conflict,
and the add path.

Additional edge-case RED runs proved ASCII/Unicode punctuation stripping, normalized DIII plural
allowance, and checking hyphenated abbreviations. The last such run was `2 failed, 4 passed` for
curly-single-quoted `thirteen` and an absent `L-H`.

Fresh focused GREEN command:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/scratch/gpfs/nc1514/FusionAIHub-LLM2/src HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e ideate-cpu python -m pytest tests/shot_design/test_blurb.py tests/shot_design/test_blurb_provenance.py tests/shot_design/test_llm_client.py -q -W error -p no:cacheprovider
```

Result: `81 passed in 11.71s`.

Assigned-file lint:

```text
/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/ruff check --no-cache src/shot_design/retrieval/blurb.py src/shot_design/shotdb/build.py src/shot_design/cli.py tests/shot_design/test_blurb.py tests/shot_design/test_blurb_provenance.py tests/shot_design/test_llm_client.py
```

Result: `All checks passed!`

## Files changed by this code task

- `src/shot_design/retrieval/blurb.py`
- `configs/shot_design/llm.yaml`
- `src/shot_design/shotdb/build.py`
- `src/shot_design/cli.py`
- `tests/shot_design/test_blurb.py`
- `tests/shot_design/test_blurb_provenance.py`
- `tests/shot_design/test_llm_client.py`
- `.superpowers/sdd/task-LLM2-code-report.md`

## Remaining verification

The parent agent owns the full `tests/shot_design` run, full scoped Ruff command, documentation,
final report, and commit. No production stores were accessed and no dependency/lock operations ran.
