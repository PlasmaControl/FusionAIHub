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
| I2 | port shotrec → src/ideate (no UI, no fetch/curate/pcslayout) + tests/ideate + configs/ideate, green `-W error` | codex gpt-6-astra | complete (684880c + fix 0895634; 422 tests; re-review approved) |
| I3 | `shotdb/reader.py` Reader protocol, `CorpusReader`, `build` on Reader, `ideate corpus scan` census CLI | opus | pending |
| I4 | census SLURM job → `corpus_coverage.parquet` (16,909 rows) + jobstats (CPU/CPU-mem) | controller | pending |
| I5 | `ideate corpus select` → `recommender_v1` (500 shots, §5.7), `ideate logs missing/import` | opus | pending |
| I6 | `labels/join.py`, `labels_wide`, events union, `ideate build --no-encode` on recommender_v1 | opus | pending |
| I7 | MCP skeleton (`search_shots`, `describe_shot`, `get_events`), `tools/list` from Claude Code | opus | pending |
| I8 | `design/actuators.py`, `encode_frame_codes`, G-ENC gate ×3 shots, encode sbatch + jobstats | opus/fable | pending |
| L1 | `events/schema.py`, `config.Paths` additions, `labels/store.append_index(keys=)` | opus | complete (4765062 on recommender-L, review clean) |
| L2 | `events/unet.py` vendored U-Net + `pin_unet.py` golden | opus | complete (8542db3, 9ee4f9b; review approved, Important fix folded into L3 housekeeping) |
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
- L2 complete: 8542db3 (schema strict JSON attrs) + 9ee4f9b (vendored U-Net). sha256 4afc3948…29a9, 7,852,002 params, `big_tf_unet_251210.pt` is the bare state_dict (the `_weights.pt` file is an older-key export); copied to $LABELMAKER_ROOT/models/tokeye/. Golden 128×128: max_abs_diff 0.0 vs golden and vs tokeye original (torch 2.14.0+cpu, 1 thread). Suite 534 → 549 green. Reviewer approved; Important (pin_unet re-pin path unreachable) → L3 housekeeping; Minor: `probabilities()` no inference_mode (→ L3), torch_version not compared (→ L3), `path` unannotated, zeros-input forward test weak, config drops `**kwargs`.
- I2 implemented by Codex gpt-6-astra: 684880c "ideate: port shot recommendation library and CLI" — 55 files, +20,379; 407 tests green `-W error`, 15 IGNITE tests green in the CUDA env with exact frame-code equality on 3 live modalities; no Python 3.12-only syntax found. Review pending.
- I2 review (opus, delta-vs-upstream method): `frame_codes()` byte-identical; drop list and attribution exact; acceptance checks 3–5 + `--help` re-run clean. Needs fixes → fix dispatched: (Important) `configs/`→`configs/ideate/` rename unfinished in llm/client.py, flags/rules.py, retrieval/rank.py, config.py docstring; (Important) trace helpers moved ui/waveforms.py → retrieval/actuation.py lost their tests. Minor for final review: `cli.py` broad `except (OSError, ValueError)` swallows pydantic ValidationError tracebacks; inert `# noqa` codes (BLE001/TRY004/C408) under this repo's ruff config; non-mechanical edits (re.I→IGNORECASE, zip→pairwise, int(len)) add upstream re-sync noise; `text.py` EMBED_MODEL dead vs `Paths.sentence_transformers_model`; `features.py:164` comment vs `invalid="ignore"`; `ignite.py` takes `open_h5` from legacy_raw; `test_ignite.py` module-scope `load_paths()` errors at collection when IDEATE_DATA_ROOT=""; `config.py:21` 112-char line.
- I2 complete: fix 0895634 closes both Important findings (config-path grep clean; 13 upstream trace tests restored verbatim + 4 new); 422 tests green `-W error`; `frame_codes()` byte-identical (md5). Re-review approved. Minor: 28 comment re-wraps (re-sync noise); `-0.0` assertion weak in test_actuation_traces.py:96; `cli.py` `except SystemExit` branch defensive/unreachable — decide at final review.
- L3 implemented: e75826c (pin_unet re-pin path, inference_mode) + 309f480 (channels + masks). Suite 549 → 665 green. 198658 reproduces A1 for mhr/ece/co2 (16,391/24,199/35,164 cols; 37/54/79 wide, 10/14/20 zoom tiles; prep 0.99/1.44/2.11 s); one real U-Net tile on mhr:0 p ∈ [0.073, 0.391]. Corrections to A1: bes/mirnov column counts should be 24,583 / 64,007 (7-frame STFT border; tile counts unchanged); bes absent on 198658. New facts: mirnov on 198658 has 1,669,828 LEADING NaNs → read_waveform must strip leading runs too (→ L4 housekeeping); truncated files raise OSError (→ pipeline catches per shot). Review pending.
