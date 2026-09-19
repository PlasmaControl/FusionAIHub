# Task LLM2 report

Requirements: `.superpowers/sdd/task-LLM2-brief.md`.
Worktree: `/scratch/gpfs/nc1514/FusionAIHub-LLM2`.
Branch: `recommender-LLM2`.
Initial HEAD: `32913d113ee776139b3d94a3246d95b03fb1c3d2` (clean).

## Progress

| Deliverable | Integration check | Status |
| --- | --- | --- |
| Prompt v6 and config | Prompt rules and YAML version must agree; existing client config pin also needs v6. | Tests and implementation in progress. |
| Gate rules | New checks follow number/shot checks and precede sentence counting. | Tests and implementation in progress. |
| Targeted regeneration and CLI | CLI passes shots to the writer; selection ignores only_missing, validates before model calls, then applies limit. | Tests and implementation in progress. |
| Manifest provenance | build, add and write_blurbs already use the shared _blurb_counts helper. | Tests and implementation in progress. |
| Documentation | Describe rules, reasons, histogram, CLI selection and audit example. | Updated; awaiting final review. |
| Verification and commit | Required full suite and ruff checks must pass before the requested commit. | Pending. |

All work uses the supplied worktree and existing environment. The supplied brief
is the implementation specification; no new design or plan file is needed.

## Tests-first evidence

Before production edits, the focused blurb and provenance tests produced
`14 failed, 36 passed`. The failures covered the absent number-word and acronym
checks, ordering, v6 prompt/config, manifest histogram, targeted regeneration,
and CLI options. Additional integration coverage and implementation are in progress.

## Verification

Pending the completed implementation.

## Anything left

Implementation, review, required verification, and commit remain in progress.
