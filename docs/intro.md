---
title: FusionAIHub (FAITH)
sidebar_position: 1
slug: /
---

FusionAIHub (FAITH — Fusion AI Toolkit & Hub) is a multi-modal foundation
model for tokamak fusion plasma data. It processes diverse diagnostics from
the DIII-D tokamak — time series, spectrograms, video, and text logs — and
fuses them to predict future plasma states. Alongside the foundation model,
the repository holds `shot_design`, a retrieval and simulation layer that
answers "has DIII-D done anything like this, and what happened?" and lets a
proposed shot design get an IGNITE-rollout sanity check before it runs on
the machine; and `labeler`, which runs the group's trained detector models
over the corpus to produce per-shot labels.

## Repository layout

- `src/tokamak_foundation_model/` — the foundation-model source (the
  installable package is named `faith`; the source tree kept its working
  name through the transition — see [the foundation
  model](./models/foundation-model.md)).
- `src/shot_design/`, `src/labeler/` — the retrieval/simulation and
  detector-label packages, with their own pixi environments.
- `configs/shot_design/` — signal, modality, path and LLM configuration read
  by `shot_design`.
- `scripts/slurm/`, `scripts/slurm_frontier/`, `scripts/slurm_della_milan/`
  — per-cluster SLURM job scripts.
- `data/` — small, git-tracked label and event tables; the multi-terabyte
  HDF5 corpus itself lives outside git (see [the corpus](./data/corpus.md)).

## Where to go next

- **[Getting started](./getting-started/install.md)** — pixi environments,
  one per platform, and the demo scripts.
- **[Clusters](./clusters/stellar.md)** — Stellar (NVIDIA) and
  [Frontier](./clusters/frontier.md) (AMD/ROCm) setup, paths and SLURM
  conventions, plus the [general pattern](./clusters/adding-a-cluster.md)
  for porting to a new one.
- **[Data](./data/corpus.md)** — the DIII-D HDF5 corpus, its modalities and
  where each cluster finds it.
- **[Models](./models/foundation-model.md)** — the foundation-model
  architecture and [IGNITE](./models/ignite.md), the tokenized dynamics
  model behind shot-design simulation, plus the tokenizer and rollout-quality
  design notes.
- **[Shot Design](./shot-design/overview.md)** — the `shot_design` CLI, MCP
  server, LLM-backed blurbs, actuator programs, simulation and database
  build.
- **[Labeler](./labeler/overview.md)** — the detector-label pipeline.
- **[Examples](./examples/index.md)** — a worked demo walkthrough.
- **[Reference](./reference/research-plan.md)** — the research plan,
  environment-variable and SLURM-script tables.

## Install quickly

```bash
pixi install            # NVIDIA/CUDA (default env)
pixi shell
python scripts/run_demo.py
```

See [Getting started](./getting-started/install.md) for every other
platform, including [Frontier](./clusters/frontier.md).
