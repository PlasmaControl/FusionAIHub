#!/bin/bash
#SBATCH -A fus187
#SBATCH -J specport_stats
#SBATCH -o /lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/logs/%x_%j.out
#SBATCH -e /lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/logs/%x_%j.err
#SBATCH -t 01:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
set -e
# NVIDIA-Spectral-Codec port: recompute the per-(channel,freq) log-power stats on the NEW
# STFT grid (n_fft 512 -> 256 DC-dropped bins). The existing codec_mhr_perfreq_stats.pt is
# 512-bin and is REJECTED by train_codec's grid check for a 256-bin codec.
cd "${SLURM_SUBMIT_DIR}"
source scripts/slurm_frontier/_frontier_common.sh
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:$PYTHONPATH}"
OUT=/lustre/orion/fus187/proj-shared/models/ignite_codecs_mhr_specport/stats
mkdir -p "${OUT}"
srun -N1 -n1 -c "${SLURM_CPUS_PER_TASK}" --gpus-per-task=1 --gpu-bind=closest \
  python -m tokamak_foundation_model.ignite.train_codec \
    --modality mhr --n_shots 400 --eval_n_shots 16 \
    --stft_n_fft 512 --stft_hop 256 --freq_bins 256 --time_frames 96 \
    --patch_f 8 --patch_t 16 \
    --out_dir "${OUT}" \
    --compute_logpow_stats "${OUT}/codec_mhr_perfreq_stats_nfft512.pt"
echo "=== SPECPORT STATS DONE (exit $?) ==="
