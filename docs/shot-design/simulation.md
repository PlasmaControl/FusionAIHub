---
title: Simulation
sidebar_position: 6
---

`shot_design simulate` runs one saved design's paired real/proposed IGNITE
rollout: what the dynamics model predicts if the reference shot's own
actuators had continued (`real`), against what it predicts under the
design's proposed actuators (`proposed`), both compared to the reference
shot's own recorded codes (`gt`). It is the way a proposed shot design gets
a rollout-based sanity check before anyone runs it on the machine.

## What it computes

`src/shot_design/simulate/core.py` builds two actuator arms that share their
first `k0` frames (the seed, using the reference shot's measured controls)
and diverge for the predicted frames — `real` keeps the reference's own
actuators, `proposed` substitutes the design's. Both arms are rolled out
from the same pinned IGNITE v4 dynamics checkpoint
(`core.load_dynamics`/`core.run_paired`), with the same RNG seed, so any
difference between them is attributable to the actuator change and not to
sampling noise. Three metrics come out per modality:

- `divergence_vs_real` — fraction of predicted tokens that differ between
  the `proposed` and `real` rollouts (a measure of *did the actuator change
  do anything*).
- `token_accuracy` — predicted tokens matching the reference shot's own
  ground-truth codes (`gt`) over the predicted region.
- `persistence_accuracy` — a trivial baseline (the last seed frame repeated)
  compared to `gt`, so `token_accuracy - persistence_accuracy` is a *skill*
  score: does the model beat "nothing changes" at all.

`src/shot_design/simulate/decode.py` turns the predicted/real/gt token
triples into normalized reconstruction values. Spectrograms use signed mean z
in the selected frequency band; slow/fast time series use the intra-frame mean.
Then
`src/shot_design/simulate/report.py` writes `simulation.h5` (tokens for all
three arms, decoded series, and the `real`/`proposed` actuator trajectories,
plus provenance attributes `design_id`, `bundle_manifest_sha256`,
`codec_generation`, `frame_origin_s`, `frame_s`, `window_s`, `dynamics_step`), one
`panels/<modality>.png` three-line plot per decoded modality, and a
`report.md` with a `modality | frac_static | token_acc | persistence_acc |
skill | divergence_vs_real | frac_static_real` table. `frac_static` measures
repeated predicted tokens in the proposed arm; `frac_static_real` applies the
same calculation to the real arm. Both remain visible in the report.

## Running it

```bash
python -m shot_design simulate <design-ident> \
    --seed 0 --k0 20 \
    --decode filterscopes,mhr,mirnov,ts_core_density,ts_core_temp \
    --device cuda --out <data_root>/outputs/<ident>/simulation
```

`--k0` defaults to 20 seed frames. Without `--n-predict`, prediction uses
`min(seed_frames_available, checkpoint_max_frames) - k0`: at most 80 frames
with the pinned checkpoint's 100-row frame embedding. An explicit horizon
must fit both the seed and checkpoint. Frame 0 is at 1.0 s, so the earliest
design start is 2.0 s; a new design defaults to 2–6 s. `--decode` defaults
to the five modalities shown above. Output defaults to
`<data_root>/outputs/<ident>/simulation`.

The command writes `status.json` in that directory **first and last** —
before doing any work, with `state: "running"`, and again on every exit
path, success or failure, with `state: "complete"` (plus `report:
"report.md"`) or `state: "failed"` (plus the exception text in `error`). The
write is atomic (temp file + `os.replace`), so a poller reading the file
mid-run never sees a half-written document. That contract exists because the
cluster wrapper (below) submits this command through `sbatch` and has nothing
else to poll.

## On Frontier

`scripts/slurm_frontier/shot_design_simulate.sh` is the one-GPU `batch`
wrapper (1 GPU, 7 CPUs, 1 hour): `sbatch
scripts/slurm_frontier/shot_design_simulate.sh <ident>`, submitted with the
repo root as the working directory (every `slurm_frontier` wrapper locates
`_shot_design_common.sh` relative to `$SLURM_SUBMIT_DIR`, which is only set
correctly when `sbatch` is invoked from there). The design's Actuator editor
UI submits this same job and polls its `status.json` over HTTP instead of
requiring an operator to run `sbatch` by hand:

- `POST /api/design/{ident}/simulate` runs the app's cluster paths file's
  `simulate_submit_cmd` (`sbatch scripts/slurm_frontier/shot_design_simulate.sh
  {ident}` on Frontier) and returns `{ident, job_id, status_url}`.
- `GET /api/design/{ident}/simulate` reads back `status.json`
  (`{"state": "not_started"}` before the job has written one); the UI polls
  it every `simulate.poll_s` (15) seconds.
- `GET /api/design/{ident}/simulate/report` and
  `GET /api/design/{ident}/simulate/panels/{name}` serve `report.md` and a
  `panels/<modality>.png` once they exist, 404 until then.

Running the CLI directly (above) or through the sbatch wrapper works the same
way outside the UI.

## On Stellar

Stellar runs IGNITE v4 from
`/scratch/gpfs/EKOLEMEN/nc1514/shot-recommender/models/IGNITE_v4`, the
sha256-pinned `nc1/IGNITE-v4` bundle at revision `d2f12b82`.
Every reader uses `model:` in `configs/shot_design/ignite_modalities.yaml`:
15 codecs with 1000 codes each, 1209 tokens per frame and 219 frames per shot.

The UI uses `paths.yaml`'s `simulate_submit_cmd` to submit a single design:

```bash
sbatch scripts/shot_design/simulate.sbatch <ident>
```

This wrapper requests one A100, 4 GB host memory and three hours. It runs
against the configured data root and does not source `_stellar_common.sh`.
Size memory and wall time from the first v4 pilot's measured usage.

The Stellar batch scripts under `scripts/shot_design/` run the interpreter
directly (never `pixi run`, whose activation would re-point the data root at
production) and REQUIRE `SHOT_DESIGN_DATA_ROOT` to name a batch root such as
`/scratch/gpfs/EKOLEMEN/nc1514/ideate/experiments/<name>`; the production
root is refused (`_stellar_common.sh`). A full batch, from a shot list to a
summary table:

```bash
export SHOT_DESIGN_DATA_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/ideate/experiments/stellar_v4_batch
PY=.pixi/envs/shot-design/bin/python
# 1. a diverse list (login node, ~20 s); §5.7's per-run-day/per-mpid caps set the ceiling
$PY -m shot_design corpus select --n 1000 --name stellar_v4_batch --seed 20260920 \
    --census /scratch/gpfs/EKOLEMEN/nc1514/ideate/db/corpus_coverage.parquet \
    --frame-codes /scratch/gpfs/EKOLEMEN/nc1514/ideate/frame_codes
# 2. database + labels join into the batch root (CPU job)
LIST=stellar_v4_batch sbatch scripts/shot_design/build_batch.sbatch
# 3. frame codes for the shots that have none (4 GPUs, one job)
LIST=stellar_v4_batch sbatch scripts/shot_design/encode_batch.sbatch
# 4. one prompt per shot, then the assistant over all of them (login node: agy needs the network;
#    when the Gemini quota is spent, Gemma through Ollama on the login GPU -- see below)
$PY scripts/shot_design/batch_prompts.py --list stellar_v4_batch --out $SHOT_DESIGN_DATA_ROOT/prompts/prompts.jsonl
$PY scripts/shot_design/batch_design.py --prompts $SHOT_DESIGN_DATA_ROOT/prompts/prompts.jsonl \
    --out $SHOT_DESIGN_DATA_ROOT/designs/designs.jsonl --provider agy --workers 2 --retry-failed
# 5. paired rollouts, 4 GPUs per job, two jobs (the QOS cap), resumable (complete/failed ids are
#    skipped). batch_idents.py keeps a theme-stratified, append-only id list growing while step 4
#    still runs; FOLLOW=1 makes the jobs re-read it until <idents>.final appears.
$PY scripts/shot_design/batch_idents.py --designs $SHOT_DESIGN_DATA_ROOT/designs/designs.jsonl \
    --prompts $SHOT_DESIGN_DATA_ROOT/prompts/prompts.jsonl --out $SHOT_DESIGN_DATA_ROOT/runs/idents.txt --follow &
for i in 1 2; do IDENTS=$SHOT_DESIGN_DATA_ROOT/runs/idents.txt FOLLOW=1 sbatch --time=20:00:00 \
    scripts/shot_design/simulate_batch.sbatch --n-predict 40 --decode-steps 4; done
# 6. one row per prompt: design, scales, simulation state and the report.md metrics
$PY scripts/shot_design/batch_collect.py --designs $SHOT_DESIGN_DATA_ROOT/designs/designs.jsonl \
    --out $SHOT_DESIGN_DATA_ROOT/summary
```

`batch_design.py` passes each prompt's shot as `assistant --ref-shot N`, which
anchors the retrieval on that shot and offers it first among the candidates;
the model still chooses the reference, so `batch_collect.py` reports how many
designs kept their source shot. The `gpu-stellar` QOS runs at most two jobs and
eight GPUs per user, so a batch is two `simulate_batch` jobs of four GPUs.

**Rollout cost.** `maskgit.rollout` re-runs the full trajectory for every
reveal pass of every predicted frame (no KV cache), so wall time grows with
the trajectory length and linearly with `--decode-steps`. The single-design
wrapper's initial sizing assumes roughly 70–130 minutes for 80 predicted
frames and 10 passes on an A100; this is an estimate awaiting a v4 pilot.
The `k0`, resolved `n_predict`, and `decode_steps` values are recorded in
`simulation.h5`.

**LLM on Stellar.** `agy` is for Gemini only (house rule; Claude runs through
Claude Code itself, GPT through Codex). When the Gemini quota is exhausted
(`RESOURCE_EXHAUSTED ... Resets in 135h`, seen 2026-09-21 after 23 designs)
the batch falls back to Gemma through Ollama: start the server on a login GPU
(`scripts/shot_design/serve_llm.sh`, or `ollama serve` by hand with an
`llm/endpoint.json` written into the batch root), then `batch_design.py
--provider ollama`. `llm.yaml`'s `ollama:` block carries its own `models:`
(gemma4:26b / e4b, used automatically whenever the provider is ollama) and
`reasoning_effort: "none"`, without which gemma4:26b spends the whole token
budget on a `reasoning` field and the propose stage fails. Measured: 28 s per
design, two model calls, about 3 designs per minute with two workers.
`assistant --llm-model TAG` overrides every alias for one run.

## Reading the result

Read `report.md` before trusting a skill number: the currently pinned
dynamics checkpoint is `mskfull/dynamics_best.pt` at step 3200, an early
checkpoint (see [IGNITE — v4 generation](../models/ignite.md#12-v4-generation)).
A `report.md` generated from it is evidence that the v4 rollout stack runs
end to end for a real design, not a validated accuracy claim — a negative
skill score does not necessarily mean the proposed actuators were bad, and a
positive one does not yet mean they were good.
