#!/bin/bash
#SBATCH --job-name=lm-labels-rv1
#SBATCH --partition=pppl
#SBATCH --qos=pppl-short-stellar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=20G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%j.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%j.out

# Task I6b: labeler over the 500 recommender_v1 shots -- features, then infer for the four
# shipped models (`shot_design labels join` wants the LABELS; the features stage alone writes none).
# Validate is deliberately NOT run: see the model-card note at the bottom of this header.
#
# All 500 shots already have a <shot>_features.h5 (Task I5 wrote them and back-filled `ip` through
# fdp): 0 of 500 lack one, checked 2026-09-07. What this run still does per shot is retry the
# misses labeler recorded as TRANSIENT (`store.TRANSIENT_CAUSES` -- TreeFOPENR, PtDataError,
# timeouts), which is where its time goes, and then infer.
#
# ------------------------------------------------------ !! fdp reachability, read before submitting
#
# The features stage's remaining work is fdp fetches, and fdp is why this job may be worth less on
# a compute node than on the login node:
#
#   * `fdp run` is an environment wrapper (fdp.cli execs the command with the device config and a
#     bearer token in the environment), not a proxy, so it does not create a route that is not
#     there. `store.TRANSIENT_CAUSES`' comment records the measured signature of running WITHOUT
#     it: PtDataError for every PTDATA point, TreeFOPENR for every MDSplus node.
#   * Task I5's report states that the compute nodes have no route to PTDATA, which is why its
#     `ip` back-fill ran on the login node, and its own features sbatch notes "EFIT resolves from
#     the archive (no fdp)".
#
# So: this job will produce labels for all 500 shots from the features already on disk either way.
# If the node cannot reach fdp, the features stage records the same transient misses again (a few
# seconds per shot, not a failure) and the fdp-only scalars stay missing -- which `shot_design build`
# reports as `pending`, honestly, rather than as data. The ~23 s per shot of fdp work measured
# below is LOGIN-NODE work; if fdp is unreachable the job is much shorter, not longer.
# The alternative, if those scalars are wanted, is to run the same command on the login node,
# which is what the pilot below did.
#
# ------------------------------------------------------------------ sizing (measured 2026-09-07)
#
# Login node (stellar-vis2), `/usr/bin/time -v` plus a 0.5 s process-tree RSS sampler, over
# `pixi run -e labelmaker fdp run python -m labeler.run all --models <the four> ...`, ten shots
# spread across the list (every 50th), each set of ten never run before:
#
#   workers   wall      s/shot   CPU    max RSS (main)   tree peak RSS   features   infer (40 pairs)
#         1   5:20.9     32.1    33 %      1717 MB          2.22 GB       228.3 s        82.2 s
#         4   1:41.1     10.1    97 %      1596 MB          6.23 GB       237.0 s        66.4 s
#
# The two runs do the same total work (features 228 s vs 237 s single-threaded) and differ only in
# how much of it overlaps: 3.2x less wall for 4x the workers, i.e. the stage is network-bound, as
# `--workers` scaling on an fdp fetch should be. Per shot, single-threaded: 23.7 s features +
# 6.6 s infer = 30.3 s.
#
# Per worker, from the two rows: 0.24 cpu (0.97 / 4, and 0.33 for one) and ~1.16 GB
# ((6.23 - 1.6) / 4), on top of a ~1.6 GB parent that has torch and the four models loaded.
#
# --cpus-per-task: 12 workers x 0.24 = 2.9 cpu -> 3 (97 % expected CPU utilisation).
# --mem: 1.6 + 12 x 1.16 = 15.5 GB measured-equivalent; 1.3x -> 20.2 GB -> 20G (measured/requested
#   = 78 %, above the 70 % gate).
# --time: 500 x 30.3 s / 12 workers = 1263 s = 21 min, plus ~2 min of validate and start-up.
#   A 2x margin for fdp contention across twelve concurrent fetchers -> 00:45:00.
#
# A re-run over shots whose features are already complete costs ~1 s per shot (measured: a second
# `all` over the same ten shots reported "features: 10 skipped" and finished in 10.8 s), so a
# resubmission after a partial run is cheap.
#
# ---------------------------------------------------------------- stage `all` REWRITES model cards
#
# The validate stage writes its results into the model card in the REPO
# (src/labeler/models/<slug>/README.md, `model-index`), keyed by nothing but the run: the
# 10-shot pilot for this script replaced d3d_tearing_onset_cnn1d's committed 486-shot metrics with
# 3-shot ones and dropped its tm_prob AUROC entry outright. That revert was made by hand.
# So: run stage `all` at FULL scale only. A small-N smoke test of this pipeline should use
# `features` + `infer`, never `all`, and `git status src/labeler` is worth a look after any run
# of it.

set -euo pipefail
REPO=/scratch/gpfs/nc1514/FusionAIHub
ROOT="$("/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/python" "$REPO/src/labeler/env.py" LABELER_ROOT "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")"
ENV="$REPO/.pixi/envs/labelmaker"
MODELS="d3d_tearing_onset_cnn1d d3d_tearing_time_to_event_dsm d3d_elm_time_to_event_dsm d3d_ae_activity_seldnet"

# The activation variable the fdp feature needs (GLIBCXX_3.4.29 for toksearch's pyxrootd), set
# here so the environment's python can be called directly and the ~1 GB `pixi run` launcher is not
# counted against --mem.
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export LABELER_ROOT="$ROOT" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE
cd "$REPO"
echo "job ${SLURM_JOB_ID:-none} on $(hostname) at $(date -Is)"

# `fdp run` for the device configuration and bearer token; without it every fdp fetch fails per
# signal (see the note above). It execs the command, so `srun` still owns the process tree.
# Controller decision (2026-09-07): stages `features` then `infer`, NEVER `all` -- the validate
# stage rewrites the committed model cards (see above) and `shot_design labels join` needs only the
# labels. The EFIT/plasma scalars shot_design reads (betan, kappa, ... ne_zipfit) are not inputs of these
# four models and are fetched separately on the login node (scripts/labeler/fetch_features.py).
for STAGE in features infer; do
    srun "$ENV/bin/fdp" run "$ENV/bin/python" -m labeler.run "$STAGE" \
        --models $MODELS \
        --shot-file "${SHOT_FILE:-$ROOT/recommender_v1.txt}" \
        --workers 12 --timeout 600
done

echo "finished at $(date -Is)"
