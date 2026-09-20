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
triples back into physical units for a handful of modalities (band power
for spectro modalities, raw values for slow/fast time series), and
`src/shot_design/simulate/report.py` writes `simulation.h5` (tokens,
decoded series and actuators for all three arms plus provenance attributes:
`dynamics_sha256`, `codec_generation`, `design_id`, `window_s`), one
`panels/<modality>.png` three-line plot per decoded modality, and a
`report.md` with a `modality | frac_static | token_acc | persistence_acc |
skill | divergence_vs_real` table.

## Running it

```bash
python -m shot_design simulate <design-ident> \
    --seed 0 --k0 20 --n-predict 80 \
    --decode filterscopes,mhr,mirnov,ts_core_density,ts_core_temp \
    --device cuda --out <data_root>/outputs/<ident>/simulation
```

`--k0` (seed frames) and `--n-predict` (predicted frames) default to 20 and
80; `--decode` defaults to the same five-modality list shown above. Output
defaults to `<data_root>/outputs/<ident>/simulation`.

The command writes `status.json` in that directory **first and last** —
before doing any work, with `state: "running"`, and again on every exit
path, success or failure, with `state: "complete"` (plus `report:
"report.md"`) or `state: "failed"` (plus the exception text in `error`). The
write is atomic (temp file + `os.replace`), so a poller reading the file
mid-run never sees a half-written document. That contract exists because the
Frontier path (below) submits this command through `sbatch` and has nothing
else to poll.

## On Frontier

`scripts/slurm_frontier/shot_design_simulate.sh` is the one-GPU, debug-QOS
wrapper: `sbatch scripts/slurm_frontier/shot_design_simulate.sh <ident>`,
submitted with the repo root as the working directory (every
`slurm_frontier` wrapper locates `_shot_design_common.sh` relative to
`$SLURM_SUBMIT_DIR`, which is only set correctly when `sbatch` is invoked
from there). The design's Actuator editor UI is expected to submit this same
job and poll its `status.json` over HTTP (`POST`/`GET
/api/design/{ident}/simulate`) rather than requiring an operator to run
`sbatch` by hand — that route is part of the concurrent Frontier UI work and
had not landed in this checkout at the time this page was written; until it
does, the CLI and the sbatch wrapper above are the supported way to run a
simulation.

## Reading the result

Read `report.md` before trusting a skill number: the currently pinned
dynamics checkpoint is `mskfull/dynamics_best.pt` at step 3200, an early
checkpoint (see [IGNITE — v4 generation](../models/ignite.md#12-v4-generation)).
A `report.md` generated from it is evidence that the v4 rollout stack runs
end to end for a real design, not a validated accuracy claim — a negative
skill score does not necessarily mean the proposed actuators were bad, and a
positive one does not yet mean they were good.
