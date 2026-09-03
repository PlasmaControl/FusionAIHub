# Stage 2 / extended Stage 2 integration — generative spectro head + resize-conv video

## Context
The Stage-1 POC (plan `dapper-pondering-backus.md`) validates that the new heads
*can* produce coherent spectrogram modes and a clean single-frame video. But two
things make Stage 2 the real test:

1. **The checkerboard is an autoregressive artifact.** The per-patch `ConvTranspose`
   seams are barely visible in a Stage-1 single-window decode; they compound over
   the K-step rollout and only become obvious in Stage 2 (user-confirmed). So the
   **resize-conv video fix can only be validated in Stage 2** (a K≥8 block-mode render).
2. **The paper's headline figures are the long-rollout (K=10 block) animations** —
   produced by the Stage 2 / extended models. A head that works at K=1 must also
   work through the rollout.

The heads + `model.py` flags are already built (stage-agnostic). This plan wires the
**loss/eval** into the Stage 2 trainers. **Gated on the Stage-1 POC**: the Stage 2 POC
inits from the Stage-1 generative best.pt, so it only runs if Stage 1 shows real modes.

## Approach
- **Expose per-step backbone token slices from the rollout** so the per-K flow loss has
  its conditioning (the rollout currently discards them).
- **Per-K flow loss** in both Stage 2 loss loops; for a generative spectro modality,
  **replace the cos+mag displacement loss with `MAE(μ) + λ·flow`** (displacement was the
  *deterministic* mode-fix attempt that failed — the flow head owns mode structure now;
  μ owns the envelope/dynamics). Keep displacement for any non-generative spectro.
- **Temporal coherence**: share ONE noise draw across all K rollout steps at eval so the
  sampled residual evolves smoothly with the conditioning instead of flickering frame-to-frame.
- **TVR + collapse-aware best.pt** in the Stage 2 validators (mirroring Stage 1).

## Changes

### 1. Rollout exposes per-step token slices — `src/tokamak_foundation_model/e2e/rollout.py`
- `_decode_diagnostics()` (~96–107): also return the per-modality backbone token slice
  it already slices to feed each head (`out_tokens[:, slice_]`). Add a flag so the
  default (predictions-only) path is unchanged for non-generative runs.
- `RolloutResult` (~line 20): add `diag_token_slices_per_step: List[Dict[str, Tensor]]`
  (populated only when any head is a `SpectrogramFlowHead`).
- `TokenSpaceRollout.forward` (109–220): collect the slices per step.
- **Eval temporal coherence**: thread an optional per-rollout `flow_noise` (drawn once,
  reused each step) into the head `sample()` call so the K decoded frames share noise.
  Add `noise: Optional[Tensor]=None` to `SpectrogramFlowHead.sample`/`forward`
  (`output_heads.py`); default (None) = independent draw (current behavior).

### 2. Per-K flow loss — `train_e2e_stage2_delta.py` (loss loop ~622–696)
- In the per-step, per-modality loop: if `isinstance(head, SpectrogramFlowHead)`:
  - `mae = masked_mae(pred=μ, target_k, mask)` (pred is μ in train mode),
  - `flow = head.flow_loss(tokens_k[name], μ, target_k, mask)` using the exposed slice,
  - `step_loss += mae + head.flow_lambda * flow`; **skip** the `displacement_losses` call
    for this modality (gated by `--spec_gen_keep_displacement`, default off).
  - Log `{name}_flow`.
- Non-generative modalities: unchanged (MAE + displacement for spectro, MAE[+smoothness] for video).

### 3. Per-K flow loss — `train_e2e_stage2_extended.py` (inside `_make_chunk_fn` ~399–474)
- Same branch, but the `flow_loss` call must live **inside** `_make_chunk_fn` (heads +
  tokens are recomputed there under gradient checkpointing). The chunk returns its
  accumulated loss; adding the flow term inside keeps the autograd graph checkpoint-correct.
- Note: extended uses **free-rollout ctx** (k≥1 ctx = detached previous *prediction* = μ),
  which is fine — flow loss doesn't use ctx; it uses (tokens_k, μ_k, target_k).

### 4. TVR + collapse-aware best.pt — both Stage 2 validators
- Add a temporal-variance-ratio accumulator to each trainer's `validate()` (compute at the
  reported K, e.g. k=1, or average over k) — same formula as Stage 1
  (`var_t(pred)/var_t(GT)` over valid bins).
- best.pt (delta ~1793, ext ~1864): behind `--collapse_aware_best`,
  `sel = Σ_k Σ_m MAE + λ·Σ_{gen} max(0, 1 − tvr)`. Default off → unchanged Σ MAE.

### 5. DDP / find_unused_parameters
- `find_unused_parameters=False` (distributed.py:78) requires every param to get grads each
  step. The velocity net runs once per K-step inside the loss loop every training step →
  satisfied. (Verified analogous in the Stage-1 CPU test: velocity grads present.)

## POC run (gated on Stage-1 POC success)
Clone the delta sbatch into `train_e2e_stage2_poc_genvid.sh`:
- `--init_checkpoint <stage1 genvid best.pt>` (from `e2e_poc_genvid/`),
- small: d_model 512 / 12L (match the Stage-1 POC so the init loads), `--max_files 200`,
  short curriculum (`--K_max 8` or curriculum `10`), ~2–4k steps, ECE+CO2+tangtv,
  `--video_resize_conv --spec_generative --collapse_aware_best --no_amp_val`,
- `--backbone_grad_checkpoint` (extended needs it at K=80; delta POC at low K may not).
- Fresh `--checkpoint_dir e2e_stage2_poc_genvid`, distinct `MASTER_PORT` (29531),
  POC-specific `--lengths_cache_dir`.
- 4 nodes, `-p batch`. Est. hours (low K, small model, 200 shots).

## Verification (end-to-end — this is where the checkerboard is judged)
1. Train the Stage 2 delta POC from the Stage-1 genvid init.
2. **Video (the key one)**: render **block mode** (`EVAL_K=8 EVAL_ROLLOUT_STEP=-1
   EVAL_EXTRA_ARGS="--comparison_figure --no_spec_fusion"`). The RAW video panel must show
   **no 12×12 checkerboard** across the K-step rollout (vs the obvious checkerboard in the
   current Stage 2 renders). This is the validation Stage 1 could not give.
3. **Spectro modes in rollout**: same render; RAW pred-spectro shows mode bands (TVR ↑) and
   they stay temporally coherent across the block (shared-noise check — no per-frame flicker).
4. **Quant**: TVR ≥ 0.6 on ECE+CO2 in the Stage 2 val logs; best.pt selected by the
   collapse-aware scalar.
5. Decision: clean checkerboard + modes through the rollout → bake into the full retrain
   (Stage 1 → delta → extended). Flicker but sharp → tune sampling/noise-sharing. Still
   collapsed → the generative head doesn't survive the rollout; reconsider.

## Risks / open
- **Flicker**: independent per-step sampling could make the block animation shimmer. Shared
  noise (change #1) is the mitigation; may still need fewer Euler steps or
  previous-frame-conditioned noise. The token recurrence stays deterministic regardless, so
  mode *locations* are coherent; only fine stochastic texture varies.
- **Displacement vs flow**: defaulting displacement OFF for generative spectro is a judgment
  call (the `--spec_gen_keep_displacement` flag lets us ablate). Risk that μ under plain MAE
  is *too* smooth and the flow must do too much; if so, re-enable displacement on μ.
- **Extended gradient-checkpoint**: flow loss inside `_make_chunk_fn` adds a velocity-net
  forward to each recomputed chunk → more recompute at K=80. Acceptable; monitor step time.
- **Cost**: a true extended (K=80) generative run is the most expensive; do the **delta**
  POC first (low K) to validate, then extended only if needed for the long-horizon figure.

## Critical files
- `src/tokamak_foundation_model/e2e/rollout.py` — expose per-step token slices; eval shared noise
- `src/tokamak_foundation_model/e2e/output_heads.py` — `sample(..., noise=None)` for coherence
- `scripts/training/train_e2e_stage2_delta.py` — per-K flow loss, displacement gate, TVR, best.pt
- `scripts/training/train_e2e_stage2_extended.py` — same, inside `_make_chunk_fn`
- `scripts/slurm_frontier/train_e2e_stage2_poc_genvid.sh` — new (clone delta sbatch)
