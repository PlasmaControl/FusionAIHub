# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FAITH (Fusion AI Toolkit & Hub) is a multimodal foundation model for tokamak plasma physics data. It processes heterogeneous fusion diagnostics (spectrograms, time series, spatial profiles, video, text) from the DIII-D tokamak via modality-specific autoencoders that produce fixed-size latent token representations, which are then fused by a transformer.

## Environment Setup

Uses [Pixi](https://pixi.sh) for environment management. On the Stellar cluster, `pixi` is not on PATH by default — always run `module load pixi` before any `pixi` command:

```bash
module load pixi
pixi install          # Install dependencies
pixi shell            # Activate environment
```

For FDP (Fusion Data Platform) features (toksearch access):
```bash
pixi install -e fdp
pixi shell -e fdp
```

The package is installed in editable mode (`faith` and `tokamak_foundation_model`).

## Running Tests

```bash
# Run all tests
pixi run python -m pytest tests/

# Run a single test file
pixi run python -m pytest tests/test_model_shapes.py

# Run a single test
pixi run python -m pytest tests/test_model_shapes.py::test_autoencoder_output_shape
```

## Training

### Single-GPU / Local Training

```bash
pixi run python scripts/training/train_unimodal_autoencoder.py \
    --signal ece \
    --data_dir /scratch/gpfs/EKOLEMEN/big_d3d_data/dummy_foundation_model_data \
    --d_model 64 --epochs 10
```

Supported `--signal` values: `gas`, `ech`, `pin`, `tin`, `d_alpha`, `mse`, `ts_core_density`, `mhr`, `ece`, `co2`, `bolo`, `irtv`, `tangtv`

### Multi-GPU (SLURM / torchrun)

SLURM scripts are in `scripts/slurm/`. Submit with:
```bash
sbatch scripts/slurm/train_ece.sh
```

The `DistributedManager` in `src/tokamak_foundation_model/utils/distributed.py` auto-detects DDP via `MASTER_PORT`/`RANK` environment variables (set by `torchrun` or SLURM). No code changes needed to switch between single and multi-GPU.

### Data Preparation

```bash
# Process raw HDF5 shots into training-ready files (uses Hydra config)
pixi run python -m tokamak_foundation_model.data.prepare_data

# Override config from CLI
pixi run python -m tokamak_foundation_model.data.prepare_data shot_list=train_full
```

Hydra config lives in `src/tokamak_foundation_model/data/config/`. Shot lists are in `config/shot_list/`; modality configs in `config/modalities/`.

## Architecture

### Two-Stage Design

**Stage 1 — Unimodal Autoencoders** (`src/tokamak_foundation_model/models/modality/`): Each diagnostic signal has a dedicated autoencoder that compresses it to `n_tokens` latent vectors of dimension `d_model`. All autoencoders share the interface `(n_channels, d_model, n_tokens)` and inherit from `ModalityAutoEncoder` (`models/modality/base.py`). The `forward()` method runs `decoder(encoder(x))`.

**Stage 2 — Fusion Transformer** (`src/tokamak_foundation_model/models/fusion/baseline_fusion_transformer.py`): Concatenates latent tokens from all modalities (with learned modality and positional embeddings), then runs a causal transformer. Input is `list[tuple[Tensor, int]]` — (tokens `[B, n_tokens, d_model]`, modality_id).

### Modality-to-Model Mapping

Defined in `src/tokamak_foundation_model/models/model_factory.py`:

| Signal type | Model class | Input shape |
|---|---|---|
| `actuator` (gas, ech, pin, tin) | `ActuatorBaselineAutoEncoder` | `(C, T)` |
| `fast_time_series` (d_alpha) | `FastTimeSeriesBaselineAutoEncoder` | `(C, T)` |
| `slow_time_series` | `SlowTimeSeriesBaselineAutoEncoder` | `(C, T)` |
| `profile` (mse, ts_core_density) | `SpatialProfileBaselineAutoEncoder` | `(spatial, time)` |
| `spectrogram` (mhr, ece, co2) | `SpectrogramBaselineAutoEncoder` | `(C, freq, time)` |
| `video` (bolo, irtv, tangtv) | `VideoBaselineAutoEncoder` | `(frames, H, W)` |

Use `build_model(model_name, d_model, n_tokens, n_channels)` from `model_factory.py` to instantiate.

### Data Pipeline

`TokamakH5Dataset` (`src/tokamak_foundation_model/data/data_loader.py`) loads shot `.h5` files. Each shot is chunked into `chunk_duration_s`-length windows. Two modes:
- **Standard mode** (`prediction_mode=False`): returns `{signal_name: tensor}` — used for unimodal autoencoder training.
- **Prediction mode** (`prediction_mode=True`): returns `{"inputs": {...}, "targets": {...}}` — splits the extended window at `chunk_duration_s`.

Spectrogram signals (`mhr`, `ece`, `co2`) are STFT-processed; time-series are linearly interpolated to `target_fs`. Preprocessing methods per signal: `none`, `standardize`, `normalize`, `log_standardize`, `log` — stats loaded from a `.pt` file (`--stats_path`).

The HDF5 file handle is excluded from pickling (`__getstate__`/`__setstate__`) and reopened per worker, so use `worker_init_fn` with `DataLoader`.

### Trainers

- `UnimodalTrainer`: for single-signal autoencoder training; DDP-aware via `DistributedManager`.
- `MultimodalTrainer`: for joint multimodal training with `inputs`/`targets` batch format.

Checkpoints save `{model_state_dict, optimizer_state_dict, scheduler_state_dict, epoch, loss}`. Best-validation checkpoint saved as `<name>_best.pth`.

## Key File Locations

| Path | Purpose |
|---|---|
| `src/tokamak_foundation_model/models/model_factory.py` | Signal→model registry and `build_model()` |
| `src/tokamak_foundation_model/models/modality/` | Per-modality encoder/decoder/autoencoder implementations |
| `src/tokamak_foundation_model/models/fusion/` | Multimodal fusion transformer |
| `src/tokamak_foundation_model/data/data_loader.py` | `TokamakH5Dataset`, signal configs, preprocessing |
| `src/tokamak_foundation_model/data/prepare_data.py` | Shot processing pipeline (Hydra) |
| `src/tokamak_foundation_model/trainer/trainer.py` | `UnimodalTrainer` and `MultimodalTrainer` |
| `src/tokamak_foundation_model/utils/distributed.py` | `DistributedManager` for DDP |
| `scripts/training/train_unimodal_autoencoder.py` | Main training entry point |
| `scripts/slurm/` | SLURM job scripts for each signal |
| `tests/test_model_shapes.py` | Shape regression tests for all autoencoders |
