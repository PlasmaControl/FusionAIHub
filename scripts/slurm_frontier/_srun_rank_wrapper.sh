#!/bin/bash
# Per-rank wrapper executed by `srun`. Maps SLURM rank vars into the env vars
# that torch.distributed (and src/.../utils/distributed.py) reads, then execs
# python on the script + args passed in $@.
#
# This is the srun analogue of what torchrun does internally — but srun gives
# us NUMA-aware GPU binding (--gpu-bind=closest) which torchrun does not.

set -uo pipefail
export RANK="${SLURM_PROCID}"
export LOCAL_RANK="${SLURM_LOCALID}"
export WORLD_SIZE="${SLURM_NTASKS}"
exec python -u "$@"
