---
title: Environment variables
---

Every environment variable `shot_design`, `labeler` and the Frontier SLURM
wrappers read, gathered in one place. The three data roots differ per cluster
and come from the cluster table `labeler.env.CLUSTERS` (below); the rest get
their value from a pixi activation block (`shot-design`, `shot-design-cpu`,
`shot-design-frontier` in `pyproject.toml`) or by hand. A value you export
always wins ([Pixi environments](./pixi-environments.md)). See
[Adding a cluster](../clusters/adding-a-cluster.md) for how a new cluster
wires its own values in.

## The three roots (`shot_design` / `labeler`)

| Variable | Purpose | Stellar | StellarAI | Frontier |
|---|---|---|---|---|
| `SHOT_DESIGN_DATA_ROOT` | the `shot_design` workstream root (database, frame codes, caches, outputs) | `/scratch/gpfs/EKOLEMEN/nc1514/shot_design` | `/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/shot_design` | `/lustre/orion/fus187/proj-shared/nchen/shot_design` |
| `LABELER_ROOT` | the `labeler` workstream root (detector features, masks, labels, models, the review page's rows in `spectrograms/`, the fetch cache in `raw/`) | `/scratch/gpfs/EKOLEMEN/nc1514/labeler` | `/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler` | `/lustre/orion/fus187/proj-shared/nchen/labeler` |
| `LABELER_LABEL_TABLES` | the label-table directory (see below) | `/scratch/gpfs/EKOLEMEN/nc1514/events` | `/scratch/gpfs/EKOLEMEN/D3D/events` | not set; export it |
| `SHOT_DESIGN_CORPUS` | the read-only DIII-D per-shot HDF5 corpus | `/scratch/gpfs/EKOLEMEN/foundation_model` | `/scratch/gpfs/EKOLEMEN/D3D/signals` | `/lustre/orion/fus187/proj-shared/foundation_model` |

The Stellar and StellarAI values of the first three are the two rows of
`labeler.env.CLUSTERS`; the packages and the sbatch launchers need nothing exported on either
(some scripts do: see [Stellar](../clusters/stellar.md#environment-variables)). Frontier's come from
its pixi activation block. `SHOT_DESIGN_CORPUS` is not in the table: the
`shot-design` pixi activation defaults it to Stellar's value, so on StellarAI
export the path above.

An explicit `SHOT_DESIGN_DATA_ROOT` always wins over whatever a paths file
(below) or the cluster table would resolve `data_root` to — this is what lets a
scratch build never touch production by accident. `pixi run` keeps a root you
exported: the activation blocks that still set one (Frontier's) do so only as
defaults (`${SHOT_DESIGN_DATA_ROOT:-...}`).

## Which cluster

The table is picked by `labeler.env.cluster()`, in this order:

1. `LABELER_CLUSTER` (`stellar` or `stellarai`). Set but empty, or any other
   name, means no cluster: the Python constants apply (`labeler.config`'s
   `DEFAULT_*`, `configs/shot_design/paths.yaml`). The test suites set it empty. A
   non-empty name that is not a row (`Stellar`, a typo) is noted once on stderr.
2. The hostname's prefix, longest first (`stellarai-*` before `stellar-*`; StellarAI's
   compute nodes are `sai-*`).
3. Slurm's `SLURM_CLUSTER_NAME`, when it is a table key.

For one variable the order is: the exported value, else the cluster's row,
else the constant. `python src/labeler/env.py NAME [DEFAULT]` (which the sbatch
scripts call) prints it in that order, with the argument last, and
`python src/labeler/env.py --cluster` prints the cluster. For
`shot_design`, the row applies only when neither `SHOT_DESIGN_DATA_ROOT` nor
`SHOT_DESIGN_PATHS` is set, and the origin line a write command prints then reads
`cluster stellar default`.

The labeler launchers under `scripts/labeler/` run that command with the pixi `labeler`
environment of the Stellar main checkout, `/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labeler`;
`LABELER_PIXI_ENV` names another checkout's environment (on StellarAI, export it).

The command line does not read `SHOT_DESIGN_PATHS`. A launcher that resolves
`SHOT_DESIGN_DATA_ROOT` through it (`simulate.sbatch`, `build.sbatch`) exports the cluster's
root even when only `SHOT_DESIGN_PATHS` is set, and an exported root outranks the paths
file. To run such a launcher against a paths file's root, export `SHOT_DESIGN_DATA_ROOT` too.

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
| `LABELER_CORPUS`, `LABELER_LOGS_JSONL`, `LABELER_TEXT_ROOT` | narrower overrides of paths that would otherwise come from `LABELER_ROOT` / the paths file |
| `LABELER_RAW_CACHE` | where fetched raw signals park (default `$LABELER_ROOT/raw`) |
| `LABELER_LABEL_TABLES` | the label-table directory (the cluster's row above, outside the checkout; `scripts/labeler/hf_publish.py pull` restores it): what `labeler` reads, and where the review page saves (`<event>/review/`) |
| `LABELER_CONFIG_DIR` | the labeler's tracked configuration (default `configs/labeler` in the checkout): `events.yaml` (taxonomy, label grid, dataset manifest) and `best_models.yaml` |
| `SHOT_DESIGN_TEXT_ROOT` | overrides the text-corpus root independent of `SHOT_DESIGN_CORPUS` |

The legacy `IDEATE_*` and `LABELMAKER_*` names (from before the 2026-09-15
package rename) are no longer read: setting one has no effect. Use the
`SHOT_DESIGN_*` and `LABELER_*` names above.

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
