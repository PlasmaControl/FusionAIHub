# Sourced by the mode_fdp_*.sbatch jobs (scripts/labeler/mode_all_shots.py, runbook
# docs/labeler/mode_fdp_allshots.md): one run root and omnimode's paths under it.
# The OMNIMODE_* paths are forced, not defaulted, so a job never reads or writes another
# omnimode tree; the seal's override variables are unset, so omnimode's batch keeps refusing
# the sealed lists (an exported OMNIMODE_ACCESS or OMNIMODE_UNSEALED never reaches a job).
# mode_all_shots.omnimode_env() sets the same paths for the login-node commands.
REPO="${REPO:-/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub}"
export LABELER_ROOT="${LABELER_ROOT:-$REPO/.cache/labeler}"
export MODE_FDP_ROOT="${MODE_FDP_ROOT:-$LABELER_ROOT/mode_fdp}"
export OMNIMODE_REPO="${OMNIMODE_REPO:-/scratch/gpfs/EKOLEMEN/nc1514/integratedmode}"
export OMNIMODE_DATA="$MODE_FDP_ROOT/omnimode"
export OMNIMODE_CORPUS="$MODE_FDP_ROOT/corpus"
export OMNIMODE_RAW_CACHE="$OMNIMODE_DATA/none/raw_cache"
export OMNIMODE_FEATURES="$OMNIMODE_DATA/none/features"
export OMNIMODE_LABELER_ROOT="$OMNIMODE_DATA/none/labeler"
export OMNIMODE_TOKEYE_WEIGHTS="$MODE_FDP_ROOT/tokeye/big_tf_unet_251210.pt"
unset OMNIMODE_ACCESS OMNIMODE_UNSEALED OMNIMODE_PROTOCOL
export HDF5_USE_FILE_LOCKING=FALSE PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "job ${SLURM_JOB_ID:-none}${SLURM_ARRAY_TASK_ID:+_$SLURM_ARRAY_TASK_ID} on $(hostname) at $(date -Is): list ${LIST:-none}, root $MODE_FDP_ROOT, omnimode $OMNIMODE_REPO ($(git -C "$OMNIMODE_REPO" rev-parse --short HEAD 2>/dev/null || echo '?'))"
