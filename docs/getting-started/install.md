---
title: Install
sidebar_position: 1
---

FusionAIHub uses [pixi](https://pixi.sh) for environment management, with
`uv` handling PyPI dependencies. Python 3.11 is required
(`>=3.11,<3.12`); the installed package name is `faith`, though the source
tree is still `src/tokamak_foundation_model/` (see
[the foundation model](../models/foundation-model.md)).

## Pick your environment

| Env | Platform | Use |
|---|---|---|
| `default` | NVIDIA/CUDA | foundation-model training (`pixi install`) |
| `fdp` | NVIDIA/CUDA + `ga-fdp` channel | direct DIII-D access via `toksearch` (needs MDSplus) |
| `labelmaker` | CPU only | the `labeler` package |
| `shot-design` | NVIDIA/CUDA | `shot_design` with GPU (`design_rollout`) |
| `shot-design-cpu` | CPU | `shot_design` on a login node |
| `shot-design-frontier` | AMD/ROCm (Frontier) | `shot_design` and `labeler` on Frontier — the only environment installed there for that workstream |
| `frontier` | AMD/ROCm (Frontier) | foundation-model / IGNITE training on Frontier MI250X |
| `rocm` | AMD/ROCm (della-milan) | MI210 variant |

Environment names use the dash form (`shot-design`, not `shot_design`) —
pixi rejects underscores in an environment name — while the Python package
installed by every one of them is `shot_design`.

## NVIDIA / CUDA (Stellar and similar)

```bash
pixi install               # default env: foundation-model training
pixi shell                 # activate it
python scripts/run_demo.py # basic inference demo
```

`fdp`, `labelmaker`, `shot-design` and `shot-design-cpu` install the same
way: `pixi install -e <env>`. See [Stellar](../clusters/stellar.md) for the
Stellar-specific paths and modules these environments assume.

## AMD / ROCm — Frontier

```bash
# 1. Clone to scratch
cd /lustre/orion/fus187/scratch/$USER
git clone git@github.com:PlasmaControl/FusionAIHub.git
cd FusionAIHub

# 2. Install pixi
curl -fsSL https://pixi.sh/install.sh | bash
source ~/.bashrc

# 3. Always --frozen on Frontier, and keep the pixi cache off $HOME
export PIXI_CACHE_DIR=/tmp/pixi-cache-$USER

# 4. Install the environments this workstream needs (~5 min each)
pixi install --frozen -e frontier
pixi install --frozen -e shot-design-frontier

# 5. Build flash-attention 2 for the frontier env (~2-5 min)
pixi run --frozen -e frontier setup-flash-attn
```

A bare `pixi install`/`pixi run` (no `--frozen`) re-solves every environment
for every platform, including a `default`/`win-64` PyPI solve that needs a
Windows interpreter — it will fail on Frontier. See
[Frontier](../clusters/frontier.md) for the roots, environment variables and
SLURM wrappers this environment expects, and
[Environment variables](../reference/environment-variables.md) for the full
list.

## AMD / ROCm — della-milan (MI210)

```bash
bash scripts/slurm_della_milan/setup_rocm_env.sh
```

Scripts for this platform live in `scripts/slurm_della_milan/`.

## Running the demos

Once an environment with the foundation-model training stack is active:

```bash
python scripts/run_demo.py      # basic inference demo
python scripts/run_demo_2.py    # demo with torchinfo summary
python scripts/run_demo_3.py    # demo with trainer.train() + validation
```

See [the foundation model](../models/foundation-model.md) for the
architecture these demos exercise, and
[Database build](../shot-design/database-build.md) /
[Simulation](../shot-design/simulation.md) for the `shot_design` side once
one of the `shot-design*` environments is installed.
