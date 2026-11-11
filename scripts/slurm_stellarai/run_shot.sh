#!/bin/bash
#SBATCH --job-name=ignite_shot
#SBATCH --account=nstx-u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --exclude=sai-n01g4
#SBATCH --output=ignite_shot_%j.out
#
# Simulate one DIII-D shot with IGNITE on StellarAI.
#
# The model is given the measured shot from 1 to 2 s and predicts all diagnostics
# from 2 s on, under the measured actuators or a changed set (SCENARIO).
#
#   sbatch --export=ALL,SHOT=204011 run_shot.sh
#   sbatch --export=ALL,SHOT=204011,SCENARIO=ech_off,SEEDS="1 2 3 4" run_shot.sh
#
# SCENARIO   real         measured actuators (default)
#            ech_off      ECH ramped to zero over 1 s from 3.5 s
#            ech_half     ECH ramped to half
#            torque_zero  torque ramped to zero through the beam mix: 150L/R, 30R, 330R off,
#                         210L/R counter-current (only for shots where 210 injects counter)
#            torque_half  torque ramped to half
#            anything else is passed to eval_dynamics --actuator_mode as it is
# SEEDS      sampled rollouts, one folder each (default 1)
# OUT_DIR    results (default ./ignite_<shot>_<scenario>): a figure of measured vs
#            predicted signals, the predicted codes and eval_metrics.json per seed
#
# The shot must be in the frame-code cache (shots 190000-204999).
set -euo pipefail

SHOT=${SHOT:?set SHOT, e.g. sbatch --export=ALL,SHOT=204011 run_shot.sh}
SCENARIO=${SCENARIO:-real}
SEEDS=${SEEDS:-1}
OUT_DIR=${OUT_DIR:-$PWD/ignite_${SHOT}_${SCENARIO}}

REPO=/scratch/gpfs/EKOLEMEN/ps9551/FusionAIHub
MODELS=/scratch/gpfs/EKOLEMEN/ps9551/ignite/models
CKPT=${CKPT:-$MODELS/ignite_prod_v4/runs/soup_lu18_h16_30/dynamics_step30000.pt}

# Frame 50 is 3.5 s; ramps take 20 frames (1 s).
beams="balance:off=0,1,5,7;counter=2,3;nm_per_mw=-0.979;kv=72@50"
case $SCENARIO in
    real)        mode=real ;;
    ech_off)     mode="linear:20:scale:ech_power=0@50" ;;
    ech_half)    mode="linear:20:scale:ech_power=0.5@50" ;;
    torque_zero) mode="linear:20:$beams" ;;
    torque_half) mode="linear:20/0.5:$beams" ;;
    *)           mode=$SCENARIO ;;
esac

cd "$REPO"
source scripts/slurm_stellarai/_stellarai_common.sh
export IGNITE_ACT_GLOBAL=$MODELS/ignite_prod_v4/act_global  # actuator scaling used in training
export IGNITE_EVAL_RAW_GT=0       # compare with the measurements in the cache
export IGNITE_MASKGIT_GUMBEL=3    # sampling noise; without it spectrogram modes go flat
export IGNITE_EVAL_MODALITIES=ece,bes,mhr,co2,mirnov,ts_core_temp,ts_core_density,ts_tangential_temp,ts_tangential_density,cer_ti,cer_rot,mse,filterscopes,tangtv_lower,tangtv_upper

for seed in $SEEDS; do
    python -m tokamak_foundation_model.ignite.eval_dynamics \
        --ckpt "$CKPT" \
        --shot "$SHOT" \
        --cache_dir "$MODELS/ignite_prod_v4/frame_codes" \
        --codec_tmpl "$MODELS/ignite_codecs_v4/{m}/codec_best.pt" \
        --actuator_mode "$mode" \
        --temperature 1.0 \
        --seed "$seed" \
        --out_dir "$OUT_DIR/seed$seed"
done
echo "results in $OUT_DIR"
