# FSQ-spectro → e2e wiring scope (task #32)

Goal: put the validated **adversarial-FSQ spectrogram recipe** (sharp modes at the
**production 24-token budget**) into the e2e model, **warm-starting** from the existing
model (TS + video already work), to hit the 14-day deliverable. No token/memory redesign
— fold24 works, so the backbone sequence length is unchanged.

## Two-phase, production-faithful (standard discrete-AR: freeze tokenizer, then predict)

### Phase 1a — pre-train + FREEZE the adversarial FSQ spectro codec  (cheap, ~1–2 days)
Per spectro modality (ece, co2, bes, mhr), train an FSQ-AE = `SpectrogramTokenizer`
(patch 64×32 → 24 tokens) → `FSQBottleneck` (dim 24) → `SpectrogramOutputHead`, with the
**VQ-GAN recipe** (`SpectroDiscriminator` + hinge + feature-matching + mode-weighted recon
+ **R1 γ10 / D-lr 1e-4 rebalance**). Reconstruction only. Save a frozen `spectro_codec_<mod>.pt`.
- New file `scripts/training/train_fsq_codec.py` (lift the AE-adversarial loop + discriminator
  straight out of `poc_fsq_stageB.py` — already written & validated).
- Promote `SpectroDiscriminator` from the POC into `e2e/quantizers/`.

### Phase 1b — main multimodal run: backbone predicts the frozen codes (the ~10-day run, warm-started)
Backbone predicts per-dim spectro **codes via class-weighted CE**; video + TS stay continuous.

## Files & changes
1. `e2e/output_heads.py` — new **`SpectrogramCodeHead`**: holds the FROZEN codec (encoder+FSQ+decoder,
   loaded from Phase 1a). Methods: `code_logits(tokens)`→(B,n_tok,dim,levels) [the prediction head,
   NEW weights]; `encode_target(spectro)`→per-dim codes [frozen, makes CE targets]; `decode(codes)`→
   spectrogram [frozen, for viz/rollout]. Sampling at inference.
2. `model.py:~307` — `--spec_fsq` → build `SpectrogramCodeHead(load frozen codec)` instead of
   `SpectrogramFlowHead` for spectro modalities. Backbone/other heads unchanged.
3. `train_e2e_stage1.py`
   - `compute_step_loss:~881` — add a `SpectrogramCodeHead` branch: `tgt_codes = head.encode_target(targets[cfg])`
     (frozen); `logits = head.code_logits(token_slices[cfg])`; **class-weighted CE** (per-dim inverse-freq,
     data-normalized — the poc_fsq_stageB implementation). Log code-acc. Replaces the MAE+flow branch.
   - flags: `--spec_fsq`, `--fsq_codec_dir`, `--spec_code_class_weight` (cap), `--spec_mode_weight`.
   - class weights precomputed once from the training code distribution.
4. `rollout.py` — MINIMAL. Token-space recurrence (backbone output tokens fed back) is UNCHANGED —
   the code head is a decode-time layer, orthogonal to recurrence. Per step: `decode(sample(code_logits))`
   for viz. (Full code-space recurrence = a Stage-2 refinement; note, not needed for Stage-1 deliverable.)
5. eval `eval_e2e_animation_tokamak.py load_model` — reconstruct `SpectrogramCodeHead` + load frozen
   codec from ckpt args; sample→decode for the comparison figure.

## Warm-start (the 14-day enabler)
Phase 1b `--init_checkpoint <existing model>` loads **backbone + TS + video heads** (they work); the
FSQ codec is loaded **frozen** (Phase 1a); the **code-prediction head inits fresh**. Continuous spectro
INPUT tokenization is kept (backbone input distribution preserved → gentle warm-start); only the spectro
OUTPUT path changes continuous→code. `load_checkpoint_with_refine_tolerance` already tolerates head-key
diffs. Use `--lazy_optimizer_load` + right batch (resume-OOM lesson).

## Risks / open
- Code-head predicts from backbone FORECAST tokens — sharpness comes from the frozen adversarial DECODER
  (renders sharp modes from any plausible codes), so imperfect code-acc still yields mode-bearing output
  (seen in the POC). Class-weighting keeps rare mode-codes in the loss.
- Generalization: relying on the video precedent (single-shot overfit sufficed → generalized on scale-up);
  light 2nd-shot check queued before committing.
- Stage-2 (K-step rollout) full code-space feedback = later; Stage-1 single-step code prediction first.

## Sequence
1a codec pre-train (adversarial, per modality, frozen) → 1b warm-start main run (code CE) → eval figures.
Split-video-codec (upper/lower divertor) folds into 1b's model construction (separate task, same run).
