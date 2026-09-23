---
title: Simulation
sidebar_position: 6
---

`shot_design simulate` rolls IGNITE out from a saved design's seed frames many times
over and scores the result against what the reference shot actually did. It answers two
questions: does the model forecast this shot better than holding the last measured frame,
and does the design's actuator change move the forecast by more than the model's own
run-to-run variation?

## What it computes

`src/shot_design/simulate/core.py` rolls out three arms from the same `k0` measured seed
frames, `--members` times each (default 8), with the exact KV cache:

- `real`: the reference shot's measured actuators.
- `proposed`: the design's actuators (identical to `real` over the seed frames), on the
  same random numbers as `real`.
- `null`: the real actuators again, on fresh random numbers.

MaskGIT samples by Gumbel-max, so `real` and `proposed` draw the same token wherever their
distributions are close; they part only where the actuators moved a distribution far
enough. After that a rollout carries the difference forward, and it can also grow for
reasons the edit did not cause. The `null` arm measures how far fresh random numbers alone
move an ensemble, and an edit counts only when it moves the ensemble mean well beyond that.
A diagnostic that was absent from the seed (its codes identical in every seed frame, the
codec's placeholder) is held at that code and not scored.

`src/shot_design/simulate/decode.py` decodes every trajectory through the modality's own
frozen codec and reduces it to one feature per frame and channel, in the codec's
normalised space: mean z over 10–60 kHz for `mhr` and `mirnov` and over the full band for
the other spectrograms, the frame mean for cameras, the intra-frame mean for time series
and profiles. `src/shot_design/simulate/score.py` then scores frames `[k0, F)`:

- `skill`: 1 − CRPS / CRPS(persistence). CRPS is the fair (unbiased for any ensemble size)
  estimator; persistence holds the last seed frame. Above 0 the ensemble beats it.
- `nrmse`: the ensemble mean's RMSE over the measured standard deviation of the predicted
  frames, next to the same for persistence and the seed mean.
- `spread_error`: ensemble spread over the ensemble mean's error, with the finite-ensemble
  correction; 1 is calibrated, below 1 overconfident.
- `effect`, `noise`, `effect_to_noise`, `resolved`: the RMS gap between the `proposed` and
  `real` ensemble means, the same for `null` against `real`, their ratio, and whether the
  ratio is at least 2.

`src/shot_design/simulate/report.py` writes, from the same arrays:

- `simulation.h5`: `tokens/{gt,real,proposed,null}/<m>` (members × frames × tokens),
  `features/<m>/{gt,real,proposed,null}`, `actuators/{real,proposed}`, and attributes
  `design_id`, `bundle_manifest_sha256`, `codec_generation`, `dynamics_step`, `window_s`,
  `frame_origin_s`, `frame_s`, `t0_s`, `k0`, `n_predict`, `members`, `decode_steps`,
  `temperature`, `precision`, `held`, `seeds`, `batch`.
- `metrics.json` (`schema: "shot-design-simulation-metrics-v1"`): the scores above per
  modality, `{"held": true}` for a held one; an undefined score is `null`.
- `report.md`: one table (skill, spread / error, edit effect / noise) in plain words.
- `panels/<m>.png`: shot time on x, each arm's ensemble mean with its 10–90 % band, the
  measurement in black.

## Running it

```bash
python -m shot_design simulate <design-ident> --members 8 --seed 0 --device cuda
```

`--k0` defaults to 20 seed frames. Without `--n-predict`, prediction uses
`min(seed_frames_available, checkpoint_max_frames) - k0`: at most 80 frames
with the pinned checkpoint's 100-row frame embedding. An explicit horizon
must fit both the seed and checkpoint. Frame 0 is at 1.0 s, so the earliest
design start is 2.0 s; a new design defaults to 2–6 s. `--decode-steps` (10) sets the
MaskGIT reveal passes per frame and `--temperature` (1) the sampling temperature.
`--batch` sets how many members share one batched rollout (default: as many as the GPU's
free memory holds, about 16 GB each in fp32 for 80 frames), and `--bf16` runs the rollouts
under bfloat16 autocast. `--decode` limits scoring to a comma-separated list; every
modality is scored by default. `real` and `proposed` share `--seed`; `null` takes
`seed + members`. Output defaults to `<data_root>/outputs/<ident>/simulation`.

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
- `GET /api/design/{ident}/simulate/report`,
  `GET /api/design/{ident}/simulate/metrics` and
  `GET /api/design/{ident}/simulate/panels/{name}` serve `report.md`,
  `metrics.json` and a `panels/<modality>.png` once they exist, 404 until then.

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

This wrapper requests one A100, 4 GB host memory and one hour. It runs
against the configured data root and does not source `_stellar_common.sh`.
At the defaults (8 members × 3 arms, 80 frames, 10 passes, fp32) the pilot,
job 2939099, took 29 minutes at a 3.2 GiB host peak and passed all four
utilisation gates. Time grows about linearly with `--members` and
`--n-predict`.

The Stellar batch scripts under `scripts/shot_design/` run the interpreter
directly and REQUIRE `SHOT_DESIGN_DATA_ROOT` to name a batch root such as
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
# 5. rollout ensembles, 4 GPUs per job, two jobs (the QOS cap), resumable (complete/failed ids are
#    skipped). batch_idents.py keeps a theme-stratified, append-only id list growing while step 4
#    still runs; FOLLOW=1 makes the jobs re-read it until <idents>.final appears.
$PY scripts/shot_design/batch_idents.py --designs $SHOT_DESIGN_DATA_ROOT/designs/designs.jsonl \
    --prompts $SHOT_DESIGN_DATA_ROOT/prompts/prompts.jsonl --out $SHOT_DESIGN_DATA_ROOT/runs/idents.txt --follow &
for i in 1 2; do IDENTS=$SHOT_DESIGN_DATA_ROOT/runs/idents.txt FOLLOW=1 sbatch --time=20:00:00 \
    scripts/shot_design/simulate_batch.sbatch --n-predict 40 --decode-steps 4; done
# 6. one row per prompt: design, scales, simulation state and the metrics.json scores
$PY scripts/shot_design/batch_collect.py --designs $SHOT_DESIGN_DATA_ROOT/designs/designs.jsonl \
    --out $SHOT_DESIGN_DATA_ROOT/summary
```

`batch_design.py` passes each prompt's shot as `assistant --ref-shot N`, which
anchors the retrieval on that shot and offers it first among the candidates;
the model still chooses the reference, so `batch_collect.py` reports how many
designs kept their source shot. The `gpu-stellar` QOS runs at most two jobs and
eight GPUs per user, so a batch is two `simulate_batch` jobs of four GPUs.

**Rollout cost.** With the KV cache the seed is encoded once and each reveal pass runs
only the new frame's 1,209 tokens, so a pass costs the same late in a rollout as early.
On a V100S in fp32 one 80-frame, 10-pass member takes 106 s at a 16.6 GiB peak; the
uncached path it replaced took 75 minutes for one real and one proposed rollout on an A100
(job 2939073). On an A100 the default ensemble, 24 such members two at a time plus their
decoding, takes 29 minutes (job 2939099). The `k0`, resolved `n_predict`, `members`, `decode_steps`, `temperature`
and precision are recorded in `simulation.h5`.

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

Start with `skill`: the pinned dynamics checkpoint (`mskfull` step 3200) is an early one
(see [IGNITE — v4 generation](../models/ignite.md#12-v4-generation) and
[the diagnosis](../models/ignite-v4-diagnosis.md)), and a signal it forecasts worse than
persistence tells you nothing about an edit. Where skill is positive, read
`effect_to_noise`: an unresolved edit is one the model cannot tell apart from rerunning the
same actuators, whatever the panels seem to show.
