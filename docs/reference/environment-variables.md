---
title: Environment variables
sidebar_position: 1
---

Every environment variable `shot_design`, `labeler` and the Frontier SLURM
wrappers read, gathered in one place. Most are set by a pixi activation
block (`shot-design`, `shot-design-cpu`, `shot-design-frontier` in
`pyproject.toml`) rather than by hand — see
[Adding a cluster](../clusters/adding-a-cluster.md) for how a new cluster
wires its own values in.

## The three roots (`shot_design` / `labeler`)

| Variable | Purpose | Stellar value | Frontier value |
|---|---|---|---|
| `SHOT_DESIGN_DATA_ROOT` | the `shot_design` workstream root (database, frame codes, caches, outputs) | `/scratch/gpfs/EKOLEMEN/nc1514/ideate` | `/lustre/orion/fus187/proj-shared/nchen/shot_design` |
| `LABELER_ROOT` | the `labeler` workstream root (detector features, masks, labels, models) | `/scratch/gpfs/EKOLEMEN/nc1514/labelmaker` | `/lustre/orion/fus187/proj-shared/nchen/labeler` |
| `SHOT_DESIGN_CORPUS` | the read-only DIII-D per-shot HDF5 corpus | `/scratch/gpfs/EKOLEMEN/foundation_model` | `/lustre/orion/fus187/proj-shared/foundation_model` |

An explicit `SHOT_DESIGN_DATA_ROOT` always wins over whatever a paths file
(below) would resolve `data_root` to — this is what lets a scratch build
never touch production by accident, and also what makes it dangerous to run
`pixi run` (which re-asserts the pixi activation values) against a scratch
target: use the environment's interpreter directly instead.

## Paths file selection

| Variable | Purpose |
|---|---|
| `SHOT_DESIGN_PATHS` | selects a whole paths file (`configs/shot_design/paths.yaml` default, `paths.frontier.yaml` on Frontier) that resolves every other path the code needs, including the three roots above |
| `SHOT_DESIGN_CONFIG_DIR` | moves where packaged config YAML (`llm.yaml`, `retrieval.yaml`, etc.) is read from |

See [Adding a cluster](../clusters/adding-a-cluster.md#the-paths-file-pattern)
for the paths-file convention itself.

## Fallback / legacy names

| Variable | Fallback for |
|---|---|
| `LABELER_CORPUS`, `LABELER_LOGS_JSONL`, `LABELER_RAW_CACHE`, `LABELER_TEXT_ROOT` | narrower overrides of paths that would otherwise come from `LABELER_ROOT` / the paths file |
| `LABELER_LABEL_TABLES` | overrides which label tables `labeler` reads |
| `SHOT_DESIGN_TEXT_ROOT` | overrides the text-corpus root independent of `SHOT_DESIGN_CORPUS` |

Legacy `IDEATE_*`/`LABELMAKER_*` names (from before the 2026-09-15 package
rename) remain fallbacks when the corresponding new name is unset; an
explicitly empty new value still wins over the legacy one, and a fallback
emits one log line naming the replacement.

## Model / offline behaviour

| Variable | Effect |
|---|---|
| `SHOT_DESIGN_HF_ONLINE` | when unset (the default), Hugging Face calls (the `sentence-transformers/all-MiniLM-L6-v2` embedder) run offline from a pre-populated cache |
| `HF_HUB_OFFLINE` | set to `1` by the `shot-design-frontier` pixi activation; Frontier compute nodes have no outbound network |
| `TOKENIZERS_PARALLELISM` | set to `false` by the `shot-design-frontier` activation to silence the tokenizers fork warning |
| `HDF5_USE_FILE_LOCKING` | set to `FALSE` by the `shot-design-frontier` activation — the corpus lives on Lustre, where h5py's default locking fails to open a read-only file the way it does on GPFS |
| `HF_HOME` | Hugging Face cache location; **never redirect it** when running `shot_design`/`labeler` test suites |

## LLM provider

| Variable | Effect |
|---|---|
| `OLLAMA_*` (family) | set by `scripts/shot_design/serve_llm.sh` on Stellar for the Ollama server (context length, keep-alive, etc.) |

The provider itself (`ollama` on Stellar, `agy` on Frontier) is selected in
`configs/shot_design/llm.yaml`, not by an environment variable — see
[LLM providers](../shot-design/llm-providers.md).

## Frontier / ROCm job environment

| Variable | Set by | Effect |
|---|---|---|
| `RCCL_PLUGIN` | `_shot_design_common.sh` (`=0`) | single-GPU/CPU `shot_design` jobs need no RCCL network plugin |
| `PYTORCH_ROCM_ARCH`, `FLASH_ATTENTION_TRITON_AMD_ENABLE` | `_frontier_settings.sh` | ROCm build target and the Triton flash-attention backend |
| `NCCL_SOCKET_IFNAME` | training scripts via `_frontier_common.sh` | forces RCCL onto the Slingshot HSN interface (`hsn0`) |
| `MASTER_ADDR`, `MASTER_PORT` | `_frontier_common.sh` | multi-node `torch.distributed` rendezvous |
| `PIXI_CACHE_DIR` | set by hand before any `pixi` invocation on Frontier | keeps the pixi cache off `$HOME` |

Always invoke `pixi` on Frontier with `--frozen`; a bare `pixi run`/`pixi
install` re-solves every environment for every platform and fails on the
`default`/`win-64` PyPI solve.
