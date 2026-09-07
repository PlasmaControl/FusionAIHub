#!/bin/bash
#SBATCH --job-name=ideate-build-rv1
#SBATCH --partition=pppl
#SBATCH --qos=pppl-short-stellar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=2800M
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/ideate/runs/slurm/%j.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/ideate/runs/slurm/%j.out

# Task I6b: the recommender_v1 shot database, built from the FAITH corpus + labelmaker's feature
# store -- `ideate build --list recommender_v1 --reader corpus --no-encode`.
#
# `--no-encode` skips the IGNITE waveform channel ONLY; the scalar and MiniLM text embeddings are
# part of the database and are written either way (the manifest's ignite block says so). The
# IGNITE encode is its own GPU job (plan section 8).
#
# The build was ALREADY RUN on the login node on 2026-09-07 (500 shots / 1977 segments in 143.4 s;
# see .superpowers/sdd/task-I6b-report.md). This script exists so the run is reproducible on
# SLURM -- for a rebuild after a config change, and for the full-corpus build of iteration 4.
#
# ------------------------------------------------------------------ sizing (measured 2026-09-07)
#
# Login node (stellar-vis2, load average 1-7), `/usr/bin/time -v` plus a 0.5 s process-tree RSS
# sampler (`ps -eo pid,ppid,rss` summed over the subtree; `ps` reports rss in KiB, and the
# sampler divided by 1e6 -- so every "GB" in the last two columns is 1e6 KiB = 0.954 GiB, which
# is the conversion the --mem arithmetic below uses), `.pixi/envs/ideate-cpu/bin/python` called
# directly (the `pixi run` launcher, ~1 GiB by itself, is bypassed so the memory measured is the
# build's):
#
#   shots  workers   build s   s/shot   wall     CPU     max RSS (main)   tree peak RSS
#      20        4       8.8    0.44    10.6 s   348 %        733 MB          1.00 GB
#      20        8       7.5    0.375    9.3 s   444 %        757 MB          1.99 GB
#      20       16       7.0    0.35     8.9 s   514 %        740 MB          3.37 GB
#     100        8      25.9    0.259   28.1 s   492 %       1188 MB          1.84 GB
#     500        8     143.4    0.287   148.7 s  552 %       1394 MB          3.32 GB   <- production
#     500        8     154.1    0.308   159.2 s  506 %       1327 MB          3.33 GB   (repeat)
#
# The 20-shot rows are warm-cache repeats of the same first 20 shots; the 100- and 500-shot rows
# are the honest ones (400 of the 500 files had never been read). Marginal cost from the 100-shot
# row against the 20-shot one: (25.9 - 7.5) / 80 = 0.23 s per shot at 8 workers, and the 500-shot
# run came in at 0.287 s per shot -- the difference is the ~200 co2/bes shots, whose extra
# channels are tens of MB each.
#
# --cpus-per-task: the 500-shot run consumed 552 % of ONE cpu at 8 workers -- 5.52 cpu-seconds per
#   wall-second -- so 6 cpus is the allocation that matches the work: 5.52 / 6 = 92 % expected CPU
#   utilisation, against 69 % if 8 were requested for 8 workers. `--workers 8` is kept because the
#   workers block on GPFS as much as they compute; asking for 8 cpus to run them would fail the
#   fairshare gate on a job that is partly I/O.
#
# --mem: 1.3 x the measured process-tree peak. THE UNIT MATTERS AND IS KiB: the sampler summed
#   `ps -eo rss`, which reports KiB, and divided by 1e6, so the "3.32 GB" in the table above is
#   3.32e6 KiB = 3,242 MiB = 3.166 GiB. 1.3 x 3,242 MiB = 4,215 MiB -> --mem=4600M (which SLURM
#   reads as 4600 MiB), i.e. measured/requested = 3,242 / 4,600 = 70.5 %. That is half a point
#   above the plan section 13 gate, not the 72 % an earlier version of this comment claimed by
#   mixing GB and MiB in one ratio. Do not shrink the request on the strength of that margin.
#
#   Expect jobstats to report LESS. SLURM samples MaxRSS periodically (tens of seconds) and this
#   job runs ~150 s: the eight workers' tree peak is short-lived (each holds one shot's channel
#   arrays while it builds it) and may fall between two samples entirely, so a sub-70 % CPU-mem
#   line on this job is a sampling artefact of a short job and not evidence of an over-request.
#   The number to compare against is the sampled peak above, measured every 0.5 s.
#
#   The tree peak is what matters and not the 1.39 GB `time -v` reports: that is the MAIN process
#   alone, which peaks at the end (PCA + embeddings + parquet) while the eight workers peak
#   earlier, each holding one shot's channel arrays -- i_coil alone is (18, 563k) float32 = 40 MB,
#   co2 one channel of 4.5 M samples = 18 MB.
#
# --time: 148.7 s measured, cold. 6x margin for a colder corpus (the I3 census measured 0.123 s
#   per FILE of pure GPFS metadata latency on a cold cache) -> 00:15:00.

set -euo pipefail

# MEASURED (job 2925318, 2026-09-07, stellar-i07n8, 6 cpus / 4600M, the sizing above this note):
#   COMPLETED in 3:44; jobstats CPU 49.8 % (11:09 cpu-time / 22:24 cpu-wall), CPU-mem 47 %
#   (2.1 / 4.5 GB) -- BOTH below the 70 % gate. Diagnosis: the login-node pilot ran WARM (the same
#   500 corpus files had just been read twice), so it finished in 148 s at 552 % cpu; the compute
#   node read them cold from GPFS and took 224 s for ~the same cpu work (669 vs ~817 cpu-s), i.e.
#   the job is I/O-bound cold and 3.0 cores were busy on average. SLURM's sampled memory (2.1 GB)
#   is also below the sampler's process-tree peak (3.2 GiB), as the earlier review warned.
#   Re-sized for the next run: 4 cpus (3.0 busy / 4 = 75 %), --mem=2800M (2.1 / 2.8 = 75 %).
#   Lesson recorded in the ledger: size from a COLD pilot (drop_caches is not available; use shots
#   not read in the last hour, or a compute-node pilot).
# REPO may point at a clean worktree snapshot (the controller does this while another task edits
# the main checkout); IDEATE_PY is the interpreter, which always lives in the main checkout.
REPO="${REPO:-/scratch/gpfs/nc1514/FusionAIHub}"
IDEATE_PY="${IDEATE_PY:-/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/ideate-cpu/bin/python}"
IDEATE_DATA_ROOT="${IDEATE_DATA_ROOT:-/scratch/gpfs/EKOLEMEN/nc1514/ideate}"
LIST="${LIST:-recommender_v1}"

cd "$REPO"
echo "job ${SLURM_JOB_ID:-none} on $(hostname) at $(date -Is)"

# HF_HUB_OFFLINE: the MiniLM checkpoint for the text embeddings is in the user cache and must not
# be re-fetched from a node with no route to the Hub. TOKENIZERS_PARALLELISM: the tokenizer forks
# with the build's own pool otherwise and warns on every shot.
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export IDEATE_DATA_ROOT="$IDEATE_DATA_ROOT"
export LABELMAKER_ROOT="${LABELMAKER_ROOT:-/scratch/gpfs/EKOLEMEN/nc1514/labelmaker}"
# One thread per worker: the work is numpy over one shot's arrays, and 8 workers x 6 OpenMP
# threads on 6 allocated cpus is pure contention.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE
# The env python has the main checkout on its path; put THIS snapshot's src first.
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

srun "$IDEATE_PY" -m ideate build \
    --list "$LIST" \
    --reader corpus \
    --no-encode \
    --workers 8

echo "finished at $(date -Is)"
