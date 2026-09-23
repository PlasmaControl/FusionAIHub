# Iteration 2: IGNITE Stage 1, a simulation that can say no, the search-to-design flow

**Date:** 2026-09-22. **Branch base:** `nathan_dev` @ 609f658.

Iteration 1 (label review, v4 only, pixi; spec `2026-09-22-label-review-v4-only-pixi-design.md`)
scored **7** (Opus 5.5) and **7** (GPT-6-sol), average **7**. Both scorers named the same gaps:
no IGNITE code changed, the product and the paper still score rollouts by exact token match,
and the search-to-design flow is untouched. Opus also found that the review page's colour
depends on zoom. This iteration closes those gaps.

## Part F: IGNITE Stage 1 (worktree `FusionAIHub-stage1`, branch `ignite-stage1`)

Everything in [the diagnosis](../../../docs/models/ignite-v4-diagnosis.md) Stage 1, as code.

**F1. Exact KV cache, opt-in.** `MaskGITDynamics.rollout(..., kv_cache=False)`. With it on, the
seed is prefilled once and each decode pass runs only the new frame's 1,209 tokens, attending
over cached per-layer temporal keys and values. Exact: spatial attention stays inside a frame,
temporal attention is causal, actuators are added per frame. After a frame is committed, one
more pass writes its keys and values. Supported with `cfg_scale == 1` and `lag_embed_k == 0`
(v4); anything else raises. Default path unchanged, bit for bit (`test_phaseb_compat.py`).
Tests: cached equals uncached token for token on tiny configs, for both reveal policies, with
revision rounds and with text conditioning.

**F2. Placeholder clamp.** `rollout(..., hold=None)`: `{modality: (B,) bool}`; a held modality
keeps its last seed frame's codes and is revealed from the first pass. The multinomial draws
still happen, so the RNG stream position is unchanged. `placeholders(seed_codes)` names the
modalities whose seed tokens are identical in every seed frame (the codec's constant null code
for an absent diagnostic).

**F3. Sampler.** `SamplerConfig.gumbel` replaces the import-time `IGNITE_MASKGIT_GUMBEL`
(no script sets it). Default 0.

**F4. Bugs.** B1 (`rollout_shot` NameError under `IGNITE_ACT_GLOBAL`, and it no longer
overwrites the caller's cache); B3 (`load_model` restores `lag_embed_k` from the weights and
every saved `cfg_*` field); B5 (`freq_axis_khz` gives the pooled bands' centres when
`band_pool > 0`); B6 (the simulate docstrings say what pairing actually guarantees); B8 (one
scheduled-sampling default, 0.75 over 2,000 steps, in `DynamicsConfig`; the Frontier script
stops restating it); B11 (`actuator_dim` defaults to the actuator spec's width, 88). The two
IGNITE tests that need Frontier-only files skip when the files are absent.

**F5. Scoring.** `shot_design/simulate/score.py`, numpy only:
- `crps(members, obs)` = E|X−y| − ½E|X−X′| per element (exact for a finite ensemble).
- nRMSE of the ensemble mean, normalised by the measured std per channel.
- skill = 1 − CRPS / MAE(persistence); persistence holds the last seed frame.
- spread/error ratio; edit effect RMS(mean proposed − mean real) against the noise floor
  RMS(mean null − mean real), where the null arm reruns the real actuators on fresh seeds.
- Decoded features come from `simulate/decode.py` (band power per channel for spectrograms,
  intra-frame mean for profiles and filterscopes; cameras reduced to a frame mean).

**F6. Simulate uses it.** `core.run_ensemble(model, cfg, codes, arms, members, seed)`:
`members` rollouts per arm (default 8) with the KV cache and the clamp, the real and proposed
arms on common random numbers, a `null` arm on fresh seeds. `simulate --members N`. Outputs:
`simulation.h5` (tokens and decoded features per member), `metrics.json` (schema below),
`report.md` (decoded table, plain language), `panels/<m>.png` (shot time on x, ensemble mean
and 10–90 % band per arm, measured in black). `simulate.sbatch` resized from the pilot.

`metrics.json` (`schema: "shot-design-simulation-metrics-v1"`):

```json
{"schema": "shot-design-simulation-metrics-v1", "members": 8, "k0": 20, "n_predict": 80,
 "frame_s": 0.05, "t0_s": 2.0, "decode_steps": 10, "temperature": 1.0, "held": ["co2"],
 "modalities": {"mhr": {"family": "spectro", "feature": "10-60 kHz band power",
   "nrmse": {"real": 1.21, "persistence": 1.05, "seed_mean": 1.30},
   "crps": {"real": 0.41, "persistence": 0.38}, "skill": -0.08, "spread_error": 0.83,
   "effect": 0.12, "noise": 0.10, "effect_to_noise": 1.2, "resolved": false}}}
```

`resolved` is `effect_to_noise >= 2`. A held modality has `"held": true` and no scores.

**F7. Evaluation harness.** `python -m shot_design.simulate.evaluate`: held-out shots (the v4
frame-code caches with shot < 190000; v4 trained on 190000–204999), seed 1.0–2.0 s, 80
predicted frames. Arms per shot: `real`, `null` (fresh seeds), `nudge` (real + 1e-3 σ on every
actuator, common random numbers: the H3 null edit), and three +0.5 σ edits on the predicted
region (`gas_flow`, `pinj`, `ech_power`) for the physics sign checks (more gas raises
density and D-alpha, more beam power raises Ti and rotation, more ECH raises core Te).
Baselines: persistence, seed mean, climatology (pool mean at each frame), nearest analog (the
pool shot whose seed features are nearest, its measured future). Outputs `results.json`,
`per_shot.csv` and a markdown summary under
`/scratch/gpfs/EKOLEMEN/nc1514/ideate/experiments/ignite_eval_v4/<run>/`. Horizons 1–10,
11–40, 41–80 frames. Bootstrap 95 % intervals over shots.

**F8. Runs** (every job gated by `labeler.jobstats` at >= 70 % on CPU, CPU-mem, GPU, GPU-mem):
a decode sweep on 6 held-out shots (passes {4, 10, 20} × T {1.0, 0.7} × Gumbel {0, 1}, bf16
and fp32 at the default), then the evaluation on 40 held-out shots with the default and the
sweep's best setting. The diagnosis doc gains the measured Stage 1 results and the commands.

## Part G: shot_design flow (one Codex gpt-6-astra agent, worktree `FusionAIHub-flow`, branch `sd-flow`)

From the flow audit (breaks B3, B4, B5, B9, B10, B11):
- **Search → design.** Multi-select rows, "Design from selected" (first = reference, the rest
  comparisons, up to six), the query carried into Notes. Search state in the URL hash, so Back
  restores results. Each row shows its `explanation` and `proposal_flags`; rank and channel
  chips replace the raw fused score. A constraint builder (field, min, max rows) replaces raw
  JSON; a label picker lists the vocabulary.
- **Simulation state.** Keyed by design id; reopening a design fetches its status; an edit keeps
  the last result, labelled with the revision it belongs to; Simulate becomes "Run again" with
  a confirmation once a result exists. `GET /api/design/{id}/simulate/metrics` returns
  `metrics.json`; the editor renders it inline (plain-language names, skill against
  persistence, effect against noise) with the panels as images. Elapsed time while running.
- **Assistant.** A Simulate button on the result, reference shots as links, the job id in the
  URL so a reload reattaches.
- Tests for each; suites through pixi with `--frozen` and `-W error`.

## Part R: review colour that does not depend on zoom (this session)

Image rows pool by **mean** (in dB) at every pyramid level and at read time, so the noise
floor stays at the level-1 median at any zoom; max pooling raised it 1.1 → 5.6 → 7.8 dB on
shot 170659. Stores carry `image_pool = "mean"`; an older store is repooled from its own
level 1 on open (no raw data needed). The 180 AE store files are repooled by one CPU job.
Traces keep (min, max).

## Part H: paper (this session, after F8)

Back up `dev/paper` (untracked) under the ideate experiments root first. Regenerate Extended
Data Table 1 with persistence, seed-mean, climatology and analog columns and CRPS skill from
F7; put the null line on Fig. 3b. Correct what the evidence rejects (the BES rise, divergence
read as an effect, "40–55 %", "11 of 13"), each change marked `<!-- 2026-09-22: ... -->` for
the owner. State which rollouts are held out.

## Out of scope

Stage 2/3 retraining (compute and architecture are the owner's decisions; the diagnosis
recommends option b). Spec item 11 (renaming v2 roots) stays with the owner.
