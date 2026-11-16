#!/bin/bash
#SBATCH --job-name=mode-fdp-masks
#SBATCH --account=nstx-u
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler/runs/slurm/%x_%j.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler/runs/slurm/%x_%j.out

# TokEye 1.0 masks for one list (omnimode's scripts/tokeye_masks.py, unchanged): PARTS processes
# share the GPU, each on shots[p::PARTS], into <root>/omnimode/frontend/masks/tokeye-1.0.0/.
# Existing masks are skipped; a shot that fails is written to <masks>/_errors/ and labelled
# "no TokEye mask" by the labels job (mode_all_shots.py redo retries it).
# tokeye and torch come from `uv run --with` offline: warm the uv cache on the login node first
# (docs/labeler/mode_fdp_allshots.md). The seal guard in tokeye_masks.py refuses a sealed shot.
#   LIST=... [PARTS=4] sbatch scripts/labeler/mode_fdp_masks.sbatch
# GPU rule: the log ends with the mean GPU utilisation (nvidia-smi every 0.5 s); below 50 %
# in the pilot, stop and report before the full run.
set -uo pipefail
source "${REPO:-/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub}/scripts/labeler/mode_fdp_env.sh"
export HF_HUB_OFFLINE=1 UV_OFFLINE=1
cd "$OMNIMODE_REPO"
UV=(uv run --no-project --python 3.13 --with tokeye==1.0.0 --with torch==2.9.1 --with-editable .)
"${UV[@]}" python -I -c "import tokeye, torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
SMI_LOG="$LABELER_ROOT/runs/slurm/mode-fdp-masks_${SLURM_JOB_ID:-none}.nvsmi.csv"
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -lms 500 > "$SMI_LOG" &
SMI=$!
trap 'kill "$SMI" 2>/dev/null || true' EXIT
PARTS="${PARTS:-4}"
ARGS=(--shots-file "$LIST" --task-count "$PARTS")
[ -f "$OMNIMODE_TOKEYE_WEIGHTS" ] && ARGS+=(--weights "$OMNIMODE_TOKEYE_WEIGHTS")
PIDS=()
for (( p = 0; p < PARTS; p++ )); do
    "${UV[@]}" python -I scripts/tokeye_masks.py "${ARGS[@]}" --task-index "$p" &
    PIDS+=($!)
done
RC=0
for pid in "${PIDS[@]}"; do wait "$pid" || RC=1; done
kill "$SMI" 2>/dev/null || true
awk '{s+=$1; n++} END {if (n) printf "GPU utilisation: mean %.0f %% over %d samples%s\n", s/n, n, (s/n < 50 ? " (BELOW 50 %: stop and report)" : "")}' "$SMI_LOG"
echo "finished at $(date -Is), exit $RC (1 = a shot failed: see <masks>/_errors/)"
exit $RC
