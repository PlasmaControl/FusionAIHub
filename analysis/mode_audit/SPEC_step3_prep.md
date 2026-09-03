# Step-3 prep specs (paper — reviewed diffs, not yet wired)

Three items so Step 3 lands as reviewed diffs, not improvised code. (A) is already
implemented+tested; (B) and (C) are design specs to review now.

---

## A. Soft/ordinal CE head loss — DONE (implemented + unit-tested)
`src/tokamak_foundation_model/e2e/ordinal_loss.py` · test `analysis/mode_audit/test_ordinal_ce.py` (17/17).
- Target: `q(k-1,k,k+1) = [eps, 1-2eps, eps]`; out-of-range neighbour mass clamped onto the
  true level (`k=0 -> [1-eps, eps]`, `k=L-1 -> [eps, 1-eps]`), renormalized. `eps` default 0.1.
- `soft_ordinal_ce(logits, codes, eps, weight, reduction)` — CE against `q`; reduces to hard CE
  as `eps->0`; optional per-element `weight` (compose with class weights).
- `tol1_codeacc` / `exact_codeacc` — the training-log metrics (tol1 = the gate quantity).
- Step-3 wiring: in `compute_step_loss`, for `SpectrogramCodeHead`, swap `F.cross_entropy(...)`
  for `soft_ordinal_ce(logits, tgt_codes, eps=SPEC_ORDINAL_EPS, weight=<existing cw>)`; log
  `{name}_tol1acc` alongside `{name}_codeacc`. Gate reads mode-band tol1acc.

---

## B. Per-modality loss normalization — SPEC

**Requirement (from the gradient-share audit):** ~4 orders-of-magnitude per-parameter
gradient disparity between slow-TS (dominant) and spectro (starved) paths; ece backbone-token
gradient ~4e-5. Every modality must contribute O(1) to the total so no path is starved.

**Scheme chosen: per-modality EMA magnitude normalization.** All heads are now CE
(commensurable nats), so normalizing each modality's loss by a running EMA of its own
magnitude is sufficient and has NO learnable machinery to destabilize. (Rejected: Kendall
learned log-variances — extra params that can drift/degenerate; GradNorm — needs per-task
grad-norm computation = extra backward passes. EMA is the simplest defensible O(1) scheme.)

**Math.** Per modality `m`, maintain a detached running EMA of its raw loss:
```
ema_m <- beta * ema_m + (1 - beta) * detach(L_m)      # beta = 0.99; init ema_m = first L_m
w_m    = priority_m / (ema_m + 1e-8)                   # effective weight
L_total = sum_m  w_m * L_m
```
Each term `w_m * L_m ~ priority_m` = O(1) → equal footing. `priority_m` default 1.0;
set `priority_spectro > 1` (e.g. 2-4) if we want to *over*-drive modes (the audit says
spectro is the goal). Warmup: first ~50 steps use `w_m = priority_m / (L_m.detach()+eps)`
(no EMA lag). Orthogonal to class weighting (that's within a modality's CE).

**Logged (every log step):** per-modality `w_m`, `ema_m`, and `w_m * L_m` (the O(1) check).
Plot `w_m(t)` over training = the "effective weight over time" panel.

**Sanity gate (its own tripwire):** flag if any `w_m(t)` drifts `>10x` from its post-warmup
value without an explained cause (e.g. a modality's loss legitimately collapsing as it learns).
Rationale: if loss magnitudes are stable, weights should be stable; a >10x drift means a
modality is collapsing/exploding — catch it before it silently re-starves another path.

**Flags:** `--loss_norm_ema {off|on}`, `--loss_norm_beta 0.99`, `--loss_priority_spectro 1.0`.
Default off → byte-identical to current runs.

---

## C. Cross-modality oracle audit — JOB SPECS (ready; DO NOT RUN until after the branch decision)

Gives every modality's *learnability floor* (persistence/oracle) — Step 6's
per-modality CE-vs-oracle-floor instrument needs it. Keep the node uncontended now.

**MHR (spectrogram) = pure config.** MHR uses the spectro codec family, so the existing
harnesses run as-is:
```
# oracle + stability + persistence-tol on MHR (finer or production codec)
MODALITIES=mhr CODEC_DIR=<...fsq_resid_p8_all or fsq_spectro_residual_codecs> \
  sbatch scripts/slurm_frontier/mode_audit_oracle.sh      # persistence_oracle.py
# and persistence_tol: point persistence_tol_s16.py's CODEC_DIR at the mhr codec, MODALITIES=mhr
```

**filterscopes (fast-TS) + tangtv (video) = config + small adapter, NOT pure config.**
- The CORE metric is modality-agnostic: encode consecutive windows through the frozen codec
  → per-dim int codes → exact/tol1 agreement at lag = the prediction stride. Reuse the
  persistence_tol harness's pair logic verbatim.
- What differs: (1) codec loaders — fast-TS = `fsq_fastts_codec_tok80` (FastTS codec class),
  video = `fsq_video_codecs_2ch` (Video codec class), NOT `load_frozen_codec` (spectro). Add a
  2-line loader switch keyed on modality. (2) "active" stratification — there is no 5-40 kHz
  band; use a modality-appropriate activity criterion: fast-TS = ELM/burst amplitude
  (p99-p50 per channel, the ELM shot-selection metric); video = frame-to-frame motion /
  per-window pixel variance. (3) input preprocessing — fast-TS = per-(window,channel) z-score
  (as its codec was trained), video = the video normalization; NO baseline_residual / no
  freq-smoothing.
- Deliverable per modality: same 2x2 table (active/quiescent × in/out) + exact/tol1 + chances +
  shuffled control + lag curve. Compare each modality's tol1 floor to its trained head codeacc
  (Step 6 instrument).

**Run order (day after branch decision):** MHR first (pure config, ~10 min), then the fast-TS
and video adapters (~1 h to add the loader switch + activity criterion, then run). Output to
`analysis/mode_audit/persistence_tol_{mhr,filterscopes,tangtv}.json`.
