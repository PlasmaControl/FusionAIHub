# shellcheck shell=bash
# Sourced by evaluation SLURM wrappers AFTER cd'ing to the repo root.
# Reuses the training env setup, then overrides for single-process eval.

source scripts/slurm_frontier/_frontier_settings.sh

# Eval runs 8 INDEPENDENT single-GCD processes — no DDP. Any code that
# auto-initializes torch.distributed off MASTER_ADDR must not see it.
unset MASTER_ADDR MASTER_PORT

# _frontier_settings.sh pins OMP_NUM_THREADS=1 for DDP ranks; eval shards
# get a few threads for CPU-side STFT/resample work.
export OMP_NUM_THREADS=4
