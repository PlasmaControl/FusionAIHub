#!/bin/bash
#SBATCH --account=kolemen
#SBATCH --job-name=paper_rollouts
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=600G
#SBATCH --time=02:00:00
#SBATCH --qos=gpu-short
#SBATCH --exclude=sai-n01g4
#SBATCH --output=/scratch/gpfs/EKOLEMEN/ps9551/FusionAIHub/logs/%x_%j.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/ps9551/FusionAIHub/logs/%x_%j.err
# Paper figures from the soup (1:1 average of ss75g_live_unroll@18000 and ss75g_spec3_h16@30000),
# Gumbel reveal noise 3, seeds 1-4 (the protocol of the existing soup archives).
#  1. Rollouts: 201081 under its measured actuators (again: its beams were repaired in act_global
#     on 2026-11-10, after the old archives) and 201081 under the actuators of the session shots
#     201079, 201083 and 201084 (actuator mode donor:<shot>, from the measured 1-2 s state).
#  2. Figure data in ens4_soup/: 204011 replay and ramps (re-extracted with the camera-view fix),
#     201081 with the donor runs, 201085 (gas added).
set -uo pipefail
cd /scratch/gpfs/EKOLEMEN/ps9551/FusionAIHub
source scripts/slurm_stellarai/_stellarai_common.sh >/dev/null 2>&1
export PYTHONPATH=/scratch/gpfs/EKOLEMEN/ps9551/ignite/pyvendor
M=/scratch/gpfs/EKOLEMEN/ps9551/ignite/models
export IGNITE_ACT_GLOBAL=$M/ignite_prod_v4/act_global IGNITE_EVAL_RAW_GT=0 IGNITE_EVAL_NO_BASELINE=1
export IGNITE_MASKGIT_GUMBEL=3
export IGNITE_EVAL_MODALITIES="bes,cer_rot,cer_ti,co2,ece,filterscopes,mhr,mirnov,mse,tangtv_lower,tangtv_upper,ts_core_density,ts_core_temp,ts_tangential_density,ts_tangential_temp"
STEP=30000; R=$M/ignite_prod_v4/runs/soup_lu18_h16_30
FC=$M/ignite_prod_v4/frame_codes; TM="$M/ignite_codecs_v4/{m}/codec_best.pt"
D=/scratch/gpfs/EKOLEMEN/ps9551/ignite/demo_figures_live_unroll/ens4_soup; mkdir -p $D/ramps
F=analysis/demo_controllability_fig.py
SEEDS="1 2 3 4"
N_GPU=$(nvidia-smi -L 2>/dev/null | wc -l); [ "$N_GPU" -gt 0 ] || N_GPU=1

rollout() {  # gpu shot name mode seed
  local out=$R/_demo_$2_$3_s${STEP}_g3_seed$5; mkdir -p $out
  CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=8 python -u -m tokamak_foundation_model.ignite.eval_dynamics \
    --ckpt $R/dynamics_step$STEP.pt --shot $2 --cache_dir $FC --out_dir $out --codec_tmpl "$TM" \
    --temperature 1.0 --guidance 1 --seed $5 --actuator_mode "$4" --render_all > $out/eval.log 2>&1
}
gpu=0
for name in real donor201079 donor201083 donor201084; do
  mode=real; [ $name != real ] && mode="donor:${name#donor}"
  for seed in $SEEDS; do
    rollout $((gpu % N_GPU)) 201081 $name "$mode" $seed &
    gpu=$((gpu + 1))
    [ $((gpu % N_GPU)) -eq 0 ] && wait
  done
done
wait
echo "[paper] rollouts done $(date +%H:%M)"
grep -l Traceback $R/_demo_201081_*_s${STEP}_g3_seed*/eval.log && echo "[paper] TRACEBACK above"

ens() { local k out=(); for k in $SEEDS; do out+=("$R/_demo_$1_$2_s${STEP}_g3_seed$k"); done; (IFS=+; echo "${out[*]}"); }
ch() { python -c "import numpy as n;d=n.load('$1');print(','.join(f'{m}={int(d[\"spectro_channel__\"+m])}' for m in ('co2','mirnov')))"; }
OLD=/scratch/gpfs/EKOLEMEN/ps9551/ignite/demo_figures_live_unroll
ex() {  # gpu shot data spectro channels [extra args...]
  local gpu=$1 shot=$2 data=$3 spec=$4 chan=$5; shift 5
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=4 python $F extract --shot $shot --step $STEP --spectro $spec \
    --spectro_channel "$chan" --spectro_member medoid --real "$(ens $shot real)" --nbi_off "$(ens $shot real)" \
    --ech_off "$(ens $shot real)" --torque0 "$(ens $shot real)" "$@" --cache_dir $FC --codec_tmpl "$TM" \
    --data $data 2>&1 | grep -E "wrote|Traceback|Error" -A3
}
ex $((0 % N_GPU)) 204011 $D/demo_204011_val_data.npz ece,mhr ece=24,mhr=0 &
ex $((1 % N_GPU)) 204011 $D/ramps/demo_204011_ramps_data.npz ece,mhr ece=24,mhr=0 \
  --ech_lin_off "$(ens 204011 ech_lin_off)" --ech_lin_half "$(ens 204011 ech_lin_half)" \
  --tq_lin_zero "$(ens 204011 tq_lin_zero)" --tq_lin_half "$(ens 204011 tq_lin_half)" &
ex $((2 % N_GPU)) 201081 $D/demo_201081_cm_data.npz co2,mirnov "$(ch $OLD/demo_201081__demo_201081_real_s18000_c1_cm_data.npz)" \
  --donor "201079=$(ens 201081 donor201079)" --donor "201083=$(ens 201081 donor201083)" \
  --donor "201084=$(ens 201081 donor201084)" &
ex $((3 % N_GPU)) 201085 $D/demo_201085_cm_data.npz co2,mirnov "$(ch $OLD/demo_201085__demo_201085_real_s18000_seed5_cm_data.npz)" &
wait
echo "[paper] extracts done $(date +%H:%M)"
