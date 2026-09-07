# recommender — execution ledger

Plan: docs/superpowers/plans/2026-09-07-recommender-plan.md (specs: docs/superpowers/specs/2026-09-07-recommender-{labelmaker-v2,ideate,critique}.md)
Repo: /scratch/gpfs/nc1514/FusionAIHub, branch `recommender` (cut from labelmaker@879fc17 on 2026-09-07)
Workstream L (labelmaker v2) runs in worktree /scratch/gpfs/nc1514/FusionAIHub-L on branch `recommender-L`, merged into `recommender` after each task.
Data roots: $LABELMAKER_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker, $IDEATE_DATA_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/ideate. Nothing new under /scratch/gpfs/nc1514 except source.

Per task: command run, output summary, jobstats (for SLURM jobs), reviewer verdict. Per iteration: critic score (Codex gpt-6-astra --effort xhigh).

## Iteration 0 — Foundations & truth (critic bar ≥ 6)

| id | task | owner model | status |
|---|---|---|---|
| I1 | pyproject/pixi: hatch packages + `ideate` feature, envs `ideate`, `ideate-cpu`; `src/ideate/__init__.py` | opus | complete (b93ec76, review clean) |
| I2 | port shotrec → src/ideate (no UI, no fetch/curate/pcslayout) + tests/ideate + configs/ideate, green `-W error` | codex gpt-6-astra | pending |
| I3 | `shotdb/reader.py` Reader protocol, `CorpusReader`, `build` on Reader, `ideate corpus scan` census CLI | opus | pending |
| I4 | census SLURM job → `corpus_coverage.parquet` (16,909 rows) + jobstats (CPU/CPU-mem) | controller | pending |
| I5 | `ideate corpus select` → `recommender_v1` (500 shots, §5.7), `ideate logs missing/import` | opus | pending |
| I6 | `labels/join.py`, `labels_wide`, events union, `ideate build --no-encode` on recommender_v1 | opus | pending |
| I7 | MCP skeleton (`search_shots`, `describe_shot`, `get_events`), `tools/list` from Claude Code | opus | pending |
| I8 | `design/actuators.py`, `encode_frame_codes`, G-ENC gate ×3 shots, encode sbatch + jobstats | opus/fable | pending |
| L1 | `events/schema.py`, `config.Paths` additions, `labels/store.append_index(keys=)` | opus | complete (4765062 on recommender-L, review clean) |
| L2 | `events/unet.py` vendored U-Net + `pin_unet.py` golden | opus | pending |
| L3 | `events/channels.py` + `events/masks.py`; smoke on 198658 reproduces A1 constants | opus | pending |
| L4 | `events/tracks.py` | opus | pending |
| L5 | `events/transients.py` (ELM clock) | opus | pending |
| L6 | `events/heuristics.py` (sawtooth, L-H, actuators, qh proxy) | opus | pending |
| L7 | `events/lexicons.yaml` + `events/text_weak.py` | opus | pending |
| L8 | `events/windows.py`, `features/resolve_events.py`, namespace/run/config edits | opus | pending |

## Log
- 2026-09-07 branch `recommender` created; plan + specs + ledger committed.
- I1 complete: commit b93ec76. `pixi install -e ideate-cpu` / `-e ideate` ok (torch 2.14.0+cpu / 2.6.0+cu124, cuda True on login V100S); `from mcp.server import MCPServer` (mcp 2.1.1); MiniLM `sentence-transformers/all-MiniLM-L6-v2` loads offline → 384-d; envs 83k inodes. Deviation: `sentence-transformers` on pypi (conda recipe forces torch 2.12 vs feature.cuda <2.11). Reviewer: approved; Minor: comment wording at pyproject "single exception"; test style (`from __future__`); `ideate`/`ideate-mcp` pixi tasks fail until the port lands (expected).
- L1 complete: commit 4765062 (recommender-L). Suite 494 → 534 collected, green `-W error`. Reviewer approved. Deviations accepted: `write_events` returns file contents; `sources=` unions with the events' sources; short git sha via `config.git_sha()`. Minor (3 folded into L2 housekeeping: `allow_nan=False`, `match=` tightening, forecast round-trip row); remaining Minor for final review: int16/int32 narrowing on channel/shot not validated; rows stored source-major (document); `index_rows` latest-pick vacuous; `INDEX_KEYS` new surface in `labels/store.py`.
