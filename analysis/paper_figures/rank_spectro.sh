#!/bin/bash
#SBATCH --job-name=rank_spectro
#SBATCH --account=nstx-u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=1:30:00
#SBATCH --exclude=sai-n01g4
#SBATCH --qos=gpu-short
#SBATCH --error=/scratch/gpfs/EKOLEMEN/ps9551/FusionAIHub/logs/rank_spectro_%j.err
#SBATCH --output=/scratch/gpfs/EKOLEMEN/ps9551/FusionAIHub/logs/rank_spectro_%j.out
# Rank spectrogram channels of the paper rollouts (rank_spectro.py), one shot per GPU.
cd /scratch/gpfs/EKOLEMEN/ps9551/FusionAIHub
source scripts/slurm_stellarai/_stellarai_common.sh >/dev/null 2>&1
export PYTHONPATH=/scratch/gpfs/EKOLEMEN/ps9551/ignite/pyvendor OMP_NUM_THREADS=4
P="python -u analysis/paper_figures/rank_spectro.py"
O=${PAPER_FIG_DATA:-/scratch/gpfs/EKOLEMEN/ps9551/ignite/paper_figures/data}
CUDA_VISIBLE_DEVICES=0 $P --shot 204011 --runs real,tq_lin_zero,tq_lin_half,ech_lin_off,ech_lin_half --switch 50 --out $O/rank_204011 &
CUDA_VISIBLE_DEVICES=1 $P --shot 201081 --runs real,donor201079,donor201083,donor201084 --switch 20 --out $O/rank_201081 &
CUDA_VISIBLE_DEVICES=2 $P --shot 201085 --runs real --out $O/rank_201085 &
CUDA_VISIBLE_DEVICES=3 $P --shot 201240 --runs real,ech_lin_off,ech_lin_half --switch 50 --out $O/rank_201240 &
wait
echo "[rank] done $(date)"
