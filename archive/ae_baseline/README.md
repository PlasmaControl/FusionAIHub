# AE-Based Aurora Baseline (archived snapshot)

Point-in-time snapshot of the autoencoder-based Aurora codebase. Serves as the
controlled baseline for contribution **C3** of the research plan
(`ResearchPlan.MD`, §2, §6.0): the demonstration that reconstruction-trained
latent spaces are geometrically incompatible with temporal prediction, and that
end-to-end tokenizers resolve this.

## Snapshot provenance

- **Date:** 2026-04-22
- **Working-tree snapshot from git HEAD:** `4f68b7c` (`Merge branch 'dev-peter'
  of https://github.com/PlasmaControl/FusionAIHub into dev-peter`)
- **Includes uncommitted modifications** in the working tree at snapshot time
  (AE hyperparameter unification, profile decoder double-pool fix, preprocessing
  stats fixes from the 2026-04-20 session). Originals remain live in the
  repository and may continue to evolve; this copy does not.

## What's inside

```
src/tokamak_foundation_model/
  models/              Aurora foundation model, per-modality autoencoders,
                       perceiver, fusion, prediction, loss, model_factory
  trainer/             MultimodalTrainer (AE + Aurora training loop)
scripts/training/      AE reconstruction scripts, train_aurora,
                       train_foundation_model, debug_latent_continuity
                       (produces the C3 scatter plots), diagnostics
scripts/slurm/         SLURM launchers for Aurora and per-modality AE training
tests/                 test_aurora, test_aurora_impulse, test_dynamics_rollout,
                       test_model_shapes
```

## What's NOT included (and why)

- **AE checkpoints** (~2.7 GB, live at
  `src/tokamak_foundation_model/models/latent_feature_space/checkpoints/`):
  not duplicated for size. The live path is stable; refer to it when
  regenerating C3 plots via `scripts/training/debug_latent_continuity.py`.
- **Shared infrastructure:** `data/`, `utils/`, data-preparation scripts,
  `preprocessing_stats.pt`, shot-list YAMLs, `pyproject.toml`, pixi lockfile.
  The end-to-end replacement reuses these unchanged; they do not need a frozen
  baseline copy.

## Reproducing the C3 evidence

The Spearman rank correlation measurements (§1.1 of `ResearchPlan.MD`) are
produced by `scripts/training/debug_latent_continuity.py` against AE
checkpoints under
`src/tokamak_foundation_model/models/latent_feature_space/checkpoints/` and
`scripts/slurm/runs/`. Finding reported in `ResearchPlan.MD`: Spearman ≤ −0.1
across all eight modalities.
