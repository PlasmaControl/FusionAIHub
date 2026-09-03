#!/bin/bash
# Launch the RESIDUAL-FSQ Stage-1 production chain: identical to the live
# raw-FSQ chain (allshots_b32) except the spectro codec dir points at the
# baseline-subtracted residual codecs. The residual behavior is self-declared
# by the codec cfg (bg_subtract=True) → forward_batch runs the whole spectro
# pathway in R-space (modes are the dominant signal → no broadband-dominated
# code collapse). Cold-start; N chained jobs; multi-partition each.
#
# Usage:  bash scripts/slurm_frontier/launch_resid_fsq_chain.sh [N_JOBS]
set -euo pipefail
cd /lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub

N_JOBS="${1:-10}"
LAUNCHER=scripts/slurm_frontier/train_e2e_stage1_d1024_48L.sh

# --- config copied verbatim from the live raw-FSQ chain (allshots_b32 ckpt args),
#     only CHECKPOINT_DIR + SPEC_FSQ_CODEC_DIR changed ------------------------
export CHECKPOINT_DIR=/lustre/orion/fus187/proj-shared/models/e2e_stage1_allshots_b32_resid
export BATCH_SIZE=32
export MAX_STEPS=105500
export VAL_EVERY=300
export VAL_MAX_BATCHES=100
export SPECTRO_PATCH_F=32
export SPECTRO_PATCH_T=16
export LR=7e-4
export WARMUP_STEPS=4000
export USE_VIDEO="tangtv_lower tangtv_upper"
# spectro FSQ -> RESIDUAL codecs (the only substantive change)
export SPEC_FSQ=1
export SPEC_FSQ_CODEC_DIR=/lustre/orion/fus187/proj-shared/models/fsq_spectro_residual_codecs
export SPEC_CODE_CLASS_WEIGHT=4.0
export SPEC_CODE_WEIGHT_BATCHES=50
# other 3 FSQ families (unchanged from the raw chain)
export VIDEO_FSQ=1
export VIDEO_FSQ_CODEC_DIR=/lustre/orion/fus187/proj-shared/models/fsq_video_codecs_2ch
export FASTTS_FSQ=1
export FASTTS_FSQ_CODEC_DIR=/lustre/orion/fus187/proj-shared/models/fsq_fastts_codec_tok80
export SLOW_TS_FSQ=1
export SLOW_TS_FSQ_CODEC_DIR=/lustre/orion/fus187/proj-shared/models/fsq_slowts_codecs
export ALL_SHOTS=1
export LAZY_OPTIMIZER_LOAD=1

mkdir -p "$CHECKPOINT_DIR"

PREV=""
for i in $(seq 1 "$N_JOBS"); do
    if [ -z "$PREV" ]; then
        JID=$(sbatch --parsable -J "e2e_resid_c$i" "$LAUNCHER")
    else
        JID=$(sbatch --parsable -J "e2e_resid_c$i" --dependency=afterany:"$PREV" "$LAUNCHER")
    fi
    echo "submitted residual chain job $i: $JID (dep=${PREV:-none})"
    scontrol update job="$JID" Partition=extended,batch,g1 >/dev/null 2>&1 || true
    PREV="$JID"
done
echo "residual-FSQ chain launched: $N_JOBS jobs -> $CHECKPOINT_DIR"
