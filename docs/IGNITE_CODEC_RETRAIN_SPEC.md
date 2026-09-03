# IGNITE spectro codec retrain spec (post gate-campaign, 2026-08-01)

Outcome of the 2026-07-31/08-01 FSQ-AE gate campaign (see
`tests/ignite/test_fsq_overfit_realshot.py` and memory
`project-ignite-fsq-overfit-gates`): locked architecture + input standardization per
modality, all user-validated on shot 200729 figures. This spec maps those decisions to
the four production retrain launches. **No code changes required to launch** — every
knob is a CLI flag; the only recommended pre-launch patch is the entropy-DDP fix
(decision 2 below).

## Locked configuration → launch flags

Launcher: `scripts/slurm_frontier/ignite_codec_prod.sh` (8 nodes × 1 rank, batch 8/rank,
20k steps). Env per launch: `MODALITY`, `N_SHOTS=9000` (**mandatory** — the shared
lengths caches are exact-path-keyed; any other N cold-scans ~8k files into the NCCL
watchdog), `LR` (decision 1), `OUT_DIR`, `EXTRA_ARGS`.

META=/lustre/orion/fus187/proj-shared/foundation_model_meta

| modality | EXTRA_ARGS | notes |
|---|---|---|
| bes | `--fsq_levels 8,8,8,5,5,5` | no input standardization (per-freq z REJECTED for bes: whitened bes is speckle) |
| co2 | `--fsq_levels 8,8,8,5,5,5 --logpow_stats_path $META/codec_co2_perfreq_stats.pt --input_instance_norm` | raw-z auto-applies and now COMPOSES (fix in main()); activity + adv-warmup overrides auto-apply |
| ece | `--patch_f 8 --patch_t 8 --logpow_stats_path $META/codec_ece_perfreq_stats.pt --input_instance_norm` | 768 tokens/window; codebook stays 1k |
| mhr | `--logpow_stats_path $META/codec_mhr_perfreq_stats.pt --input_instance_norm` | adv-warmup overrides auto-apply |

Launch pattern (each; all four can run in parallel on `-p batch`):

    MODALITY=<m> N_SHOTS=9000 LR=3e-4 OUT_DIR=eval_runs/ignite_codec_<m>_v3 \
      EXTRA_ARGS="<from table>" sbatch scripts/slurm_frontier/ignite_codec_prod.sh
    scontrol update job=<id> Partition=extended,batch,g1   # standing rule

Stats files (already generated + gate-validated, `std_kind=within_shot`, co2 in the
compose space): `$META/codec_{co2,ece,mhr}_perfreq_stats.pt`. Do NOT regenerate with
the old pooled convention; the degenerate pre-fix co2 file lives in `.bak`.

## Pre-launch decisions

1. **LR (recommended: `LR=3e-4`).** The launcher default 1e-3 is the measured
   stuck-regime for FSQ settling (gate calibration: FSQ at 1e-3 plateaued at nmae 0.37
   with churning codes; 3e-4 converged). Evidence is single-window/recon-only — if
   preferred, run one modality at each LR as a canary before committing all four.
2. **Entropy-DDP fix (recommended: apply before launch).**
   `quantizer.entropy_loss`'s FIX-1 all-reduce is not autograd-aware, so the
   batch-diversity gradient is attenuated by 1/world_size (8×) while the per-sample
   confidence term keeps full strength. Fix = scale the batch-entropy term by
   world_size (or use `torch.distributed.nn.functional.all_reduce`). Small patch +
   regression test; without it the anti-collapse reward is 8× weaker than configured.
3. **Objective weights: unchanged** (pixel 0.05 / consistency 1.0 / entropy 1.0 /
   fm 1.0 / adaptive adv, per-modality overrides auto). Known-risk baseline — the gate
   validated architecture+preprocessing under pure recon, not this objective. The new
   standardization removes the measured collapse mechanisms (DC plates, saturation,
   loss-invisible structure), so the recipe gets one shot as-is; revisit weights only
   if the oracle gate fails.

## Known behavior changes to expect

* **co2 activity stratification becomes inert**: `min_activity` compares the built
  window's std, which is ≡1.0 under instance norm → every window counts as active.
  Acceptable: instance norm itself removes the degenerate-window domination mechanism
  (quiet windows normalize instead of pulling toward a constant). Shot-level presence
  filtering still applies.
* z-modalities train in the normalized target space — val losses/pixel metrics are not
  comparable to any pre-v3 run. Judge by the trainer's gate metrics + rendered figures.

## Acceptance / promotion

* Trainer gate per checkpoint: utilization ≫ 1 code (gate-observed: co2 167, ece 419,
  mhr 79 on single windows), envelope corr above `gate_recon_floor`, and — per the
  mode-audit mandate — **stability ≥ 0.8** before any Phase-B training on the codes.
* Render per-modality recon figures and judge structure visually (session lesson:
  pixel metrics under-discriminate; figures decide).
* On promotion, Phase-B prep (separate work item): int32 code cache (64k > int16 max),
  factorized per-dim prediction heads for bes/co2, frame layout for ece's 768-token
  slice (spectro frame slice 768 → 1344 tokens).
