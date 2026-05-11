# Training Pipeline

Three stages, run in order. All commands assume you're in the repo root with `pixi shell` active (or prefix with `pixi run`).

## Stage 1: Train Unimodal Autoencoders

Trains one autoencoder per signal. Each learns to compress its modality into a fixed-size token representation `(n_tokens, d_model)`.

```bash
# Submit all training jobs (one per signal)
for f in scripts/slurm/train_*.sh; do sbatch "$f"; done

# Or train a single signal interactively
python scripts/training/train_unimodal_autoencoder.py \
    --signal ece --epochs 50 --batch_size 4 \
    --data_dir /scratch/gpfs/EKOLEMEN/foundation_model \
    --stats_path data/preprocessing_stats.pt \
    --checkpoint_dir runs/ece
```

**Outputs:** `runs/{signal}/checkpoint.pth`, `config.json`, `plots/`

**Data:** Raw HDF5 shot files in `/scratch/gpfs/EKOLEMEN/foundation_model/`

## Stage 2: Generate Token Dataset

Runs trained autoencoders in inference mode on all shots, saving compressed tokens to HDF5 files. This decouples encoding from prediction — Stage 3 loads pre-computed tokens instead of raw data.

```bash
# Submit as SLURM array job (9 tasks, ~1000 shots each)
sbatch scripts/slurm/generate_tokens.sh

# Or run interactively (test with a few shots first)
python scripts/training/generate_token_dataset.py --max_shots 10
python scripts/training/generate_token_dataset.py  # all shots
```

**Inputs:** `runs/{signal}/checkpoint.pth` from Stage 1

**Outputs:** `data/tokens/{shot_id}_tokens.h5` — one file per shot containing all signals

**HDF5 layout:**
```
{shot_id}_tokens.h5
  {signal}/tokens        [n_chunks, n_tokens, d_model]
  {signal}/observations  [n_chunks, ...]
```

## Stage 3: Train Multimodal Predictor

Trains a fusion transformer that takes tokens from all modalities at time `t` and predicts tokens at time `t+1`. Uses frozen decoders from Stage 1 for observation-space loss.

```bash
# Submit SLURM job
sbatch scripts/slurm/train_multimodal.sh

# Or run interactively
python scripts/training/train_multimodal_predictor.py \
    --token_dir data/tokens \
    --checkpoint_dir runs \
    --output_dir runs/multimodal \
    --signals ich filterscopes vib
```

**Inputs:** `data/tokens/` from Stage 2, `runs/{signal}/` from Stage 1 (for frozen decoders)

**Outputs:** `runs/multimodal/checkpoint.pth`

## Directory Layout

```
runs/                  # Stage 1 checkpoints (per signal)
data/tokens/           # Stage 2 token files (per shot)
runs/multimodal/       # Stage 3 checkpoint
logs/                  # SLURM job logs
data/preprocessing_stats.pt  # Precomputed normalization stats
```

## SLURM Scripts

All in `scripts/slurm/`:

| Script | Stage | Notes |
|--------|-------|-------|
| `train_{signal}.sh` | 1 | One per signal, `--time=00:20:00` |
| `generate_tokens.sh` | 2 | Array job, 9 tasks |
| `train_multimodal.sh` | 3 | Single job, `--time=12:00:00` |
