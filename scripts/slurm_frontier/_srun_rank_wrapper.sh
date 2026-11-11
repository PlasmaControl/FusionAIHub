#!/bin/bash
# Per-rank wrapper run by `srun`. Maps the SLURM rank variables to the ones
# torch.distributed reads, then runs python with the arguments in $@.
# This replaces torchrun so that srun can do the GPU binding.

set -uo pipefail
export RANK="${SLURM_PROCID}"
export LOCAL_RANK="${SLURM_LOCALID}"
export WORLD_SIZE="${SLURM_NTASKS}"
exec python -u "$@"
