# FusionAIHub (FAITH)

FusionAIHub (FAITH — Fusion AI Toolkit & Hub) is a multi-modal foundation
model for tokamak fusion plasma data: it fuses DIII-D time series,
spectrograms, video and text logs to predict future plasma states, and
includes `shot_design`, a retrieval and simulation layer for proposing and
checking new shots.

## Install

```bash
# NVIDIA/CUDA (default env)
pixi install
pixi shell
python scripts/run_demo.py

# AMD/ROCm (Frontier)
curl -fsSL https://pixi.sh/install.sh | bash
export PIXI_CACHE_DIR=/tmp/pixi-cache-$USER
pixi install --frozen -e frontier
pixi run --frozen -e frontier setup-flash-attn
```

Full docs, including every platform, are published at
<https://plasmacontrol.github.io/FusionAIHub/> — start with
[Getting started](docs/getting-started/install.md) and
[Frontier](docs/clusters/frontier.md).
