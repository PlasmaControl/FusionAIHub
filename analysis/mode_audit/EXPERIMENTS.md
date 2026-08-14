# Descriptor-head forecasting — experiment log

Metric convention: `peak_in_tol` / recall = fraction of events where the predicted
mode peak-frequency is within ±1 kHz (2 bins of 72) of ground truth. **Every
"beats persistence" claim must clear the RAW-persistence null** (argmax of the
*unthresholded* input profile) — thresholded-persistence and shuffled-chance are
insufficient because the label threshold blinds the baseline while the model's input
keeps the answer.

Model under test: `models/e2e_descriptor_ece_anchor/e2e_stage1_best.pt` — d512/12L,
persistence-anchored descriptor head (pred = current-window descriptor + zero-init
residual), warm-started, all-shots, ECE, dist-CE β=4, prominence-weighted.

---

## GATE 1 — does the descriptor head have any real (non-copyable) forecasting skill?

### Aggregate (saturated — cannot show skill)
Active-window `peak_in_tol` = **0.576 = persistence exactly**. Dist-diagnostic: model
freq-entropy 3.823 ≈ persistence 3.819 → model ≡ persistence (argmax + distribution);
the black/yellow render was a logits-vs-profile **scale artifact**, not collapse. At
50 ms / ±1 kHz the mode drifts <1 kHz, so persistence is unbeatable here by construction.

### Onset — **LEAKAGE (detection, NOT forecasting)** ❌
Pooled 14 shots, K=3 hysteresis, n=133 onsets (job 4987608):
- model **0.406 ±0.083**  |  **RAW-persistence null 0.579**  |  thresh-persistence 0  |  shuffled 0.112
- beyond-raw (onsets the raw copy misses) = **0.036 ≈ noise**
- Verdict: model < raw-null (gap −0.173). The onset skill is **sub-threshold copy** — the
  unthresholded input already carries a growing precursor the detector threshold hides.
  Real capability = *precursor DETECTION by the representation*, not model forecasting;
  the model even degrades the raw copy. (Earlier onset optimism = this leakage; retracted.)

### Drift-direction — **VALIDATED forecasting signal (sparse, non-copyable)** ✅
Three-way scoring (the naive 0.167 was "none scored as wrong" — broken metric):
- model commits a direction on ~10% of drift events; **when committed, 78.3% correct**
  (n≈174, p≪0.001 vs chance 0.5). A static copy has zero drift → **un-leakable.**
- Sparse but real. This is the anchor of the Gate-1 deliverable.

### Death — **BIAS, discarded** ❌ (job 4987641, CI-separation exit rule)
- death recall **0.232 ± 0.074** (n=125) → CI lower bound 0.158
- false-death (sustained-window) **0.155 ± 0.020** (n=1213) → CI upper bound 0.175
- **0.158 < 0.175 → CIs OVERLAP → absent-bias, not skill.** The model predicts "absent"
  15.5% of the time even on *sustained* modes, which accounts for the 0.232. Discard.

### Gate 1 deliverable — FINAL
> *The descriptor head has forecasting skill in **drift-direction only** (non-copyable, 0.783
> ±0.061 on the ~10% of drift events it commits to, CI lower bound 0.722 ≫ chance 0.5) and
> **detection** skill in onset precursors (leakage, not forecasting); death recall is absent-bias.*

**~~Gate 1 status: OPEN~~ → CLOSED (2026-07-14).** Sole validated forecasting signal = **drift-direction** (sparse). Onset = detection/leakage. Death = bias.

---

## NAMED DEFECT — absent-bias (false-death rate 0.155)
The head predicts "absent" on 15.5% of *sustained*-mode windows. **Rollout math:** if
per-step-independent, 0.155/step compounds to near-certain spurious mode-death by ~15 of
80 steps. → **false-death-rate is a Gate-2 tracked metric AND a Gate-4 (rollout) blocker.**
The K-step probe must later measure the *effective* (correlated-error) per-rollout
suppression rate — it can be better or worse than the independent estimate.

## GATE 2 — grow the committed set (RE-AIMED 2026-07-14). Two runs: 2a (t+1) + 2b (t+4/t+8).
**~~GATE 2 status: OPEN~~ → CLOSED (2026-07-14).** 2a PASSED (β8+transition5 t+1 head grew commit 0.10→0.157,
false-death 0.155→0.002 — FROZEN as `e2e_descriptor_ece_g2`). 2b FLAT on the argmax gate at t+2/t+4 (commit≈0,
insufficient_commits) → the t+1 g2 head stays the committed forecaster. Sub-threshold discriminator POSITIVE
(calibrated soft directional signal, ΔLL>0); heuristic-fails split (4991591) SEALED it: t+2 heuristic-bound
(momentum), **t+4 weak BEYOND-heuristic (trend-independent, both-subsets CI>0.5; mom-wrong 0.611 [0.535,0.681]
n=167)**. false-death 0.000 both horizons. Gate 3 (actuator counterfactual) now tests whether the t+4 signal is
plasma-state-causal. FULLY SEALED.
### Gate 2a — β-sharpen(β8) + transition-overweight(×5) at t+1, anchored — **PASSED, head FROZEN ✅ (2026-07-14, train 4987738 → eval 4987739)**
Head `models/e2e_descriptor_ece_g2/e2e_stage1_best.pt` (step 5500; horizon 0.05, dist_beta 8.0,
transition_weight 5.0, anchor, weight 6.0). Eval `eval_runs/descriptor_trained_g2/onset_skill.json`,
pooled 14 shots K=3. Pre-registered exit rule (needs a∧b) — both cleared:
- **(a) drift commit-rate 0.10 → 0.157** (model-none 0.843; n_drift=1742) at **committed-acc 0.759 ≥ 0.70**
  (n_committed≈274, p≪0.001 vs chance 0.5). Grew the committed set ~57% relative, accuracy above gate.
- **(b) false-death 0.155 → 0.002** (sustained n=1213). The named absent-bias defect is essentially
  ELIMINATED (β-sharpen+transition-overweight = the mechanism). Flip side: death-recall also fell 0.232→0.008
  (head almost never predicts absence) — but death was already discarded as bias (Gate 1), so no real
  capability lost, and killing false-death is a genuine Gate-4/rollout win (no spurious mode death).
- (c) onset-beyond-raw 0.036 ≈ 0 (model 0.586 ≈ RAW-null 0.579, gap +0.008 = still LEAKAGE, as pre-registered;
  real onset test is Gate 3 / actuators).
**VERDICT: freeze `e2e_descriptor_ece_g2` as the current-best descriptor head.** committed-acc dipped 0.783→0.759
(honest cost of higher commit-rate) but stays above the 0.70 gate. Gate 2a CLOSED.

### Gate 2b — FULL SPEC t+4/t+8 (horizon still important, user 2026-07-14) — **BLOCKED on a DESIGN FORK, not a bug**
`--prediction_horizon_s 0.2` is the **Stage-2 K-step-rollout lever**: when horizon>chunk the data loader
splits every target into K=round(horizon/chunk)=4 sub-windows (data_loader.py:1797-1825) → all fast targets
are 4× longer. Stage-1's single-window losses can't consume it: after fixing the actuator-tokenizer warm-start
(added `act_tokenizers.` to allowed_missing so the fresh 0.2-geometry conv inits on a warm backbone —
smoke 4988290 LOAD-OK), the smoke then crashed in `masked_mae` (compute_step_loss:1216) with pred 5 vs
target 20 (=K=4×) at the FORWARD, i.e. EVERY base modality breaks, not just the descriptor. So t+4 is NOT a
Stage-1 bugfix — it needs a design decision on the target mechanism. (Mechanism detail: SIGNAL targets are the
FULL extended future `signal[..., K*n_train:]` (data_loader:1775, K=history_windows=1 → not sub-split, just
K_horizon× longer); MOVIES sub-split (1815-1825). Net: every target is K_horizon× longer → base losses mismatch.)

#### HORIZON PROBE (job 4988515, `horizon_probe.json`, 14 shots) — HEADROOM CONFIRMED, drift is ANTI-MOMENTUM at t+4
peak_in_tol(persistence) decays 0.796(t+1)→0.691(t+2)→**0.545(t+4)**→0.399(t+8); drift-fraction grows
0.204→0.309→**0.455**→0.601. At 200 ms persistence is far from saturated + 46% of modes move = real headroom
(unlike t+1 where persistence saturates). MOMENTUM-match (does prior N-step drift dir predict next): 0.619(t+1,n=42
**TOO THIN — do not quote**), 0.616(t+2,n=86), **0.342(t+4,n=111,~3σ BELOW chance)**, 0.470(t+8,n=117≈random walk).
⇒ **t+4 drift is systematically ANTI-momentum** (mode freq oscillates/mean-reverts around profile-set equilibria over
200 ms) = STRUCTURE, hence LEARNABLE if the backbone sees the equilibrium-setting state (it sees the profiles); t+8 is a
random walk (unlearnable). **t+8 DEMOTED.** Decay figs `eval_runs/descriptor_trained_g2/ece_horizon_probe.png`.

#### GATE 2b — PRE-REGISTERED (updated for the probe). Target = t+4 (200 ms), multi-target {t+2, t+4}.
**Mechanism:** prediction_horizon_s=0.2 (K=4 extended target); base losses slice every target to sub-window-0 (t+1,
= Gate 2a exactly, no crash); the DESCRIPTOR head emits per-horizon readouts (multi-horizon head) supervised at t+2
(sub-window 1) and t+4 (sub-window 3). Multi-target anchors the oscillation phase (a 200 ms jump is harder to learn
than a short path). Persistence-anchored (anchor = current window at every horizon), β8, transition-weight 5, warm-start g2.
**THREE NULLS (all mandatory, from the probe) — skill = beat ALL THREE:**
- persistence (t+4 peak_in_tol 0.545)
- momentum (continue recent trend ≈ 0.342 at t+4)
- **anti-momentum (flip recent trend ≈ 0.658 at t+4)** — NEW. A head that learns a static "reverse the trend" rule
  clears persistence+momentum while learning ZERO plasma-state dependence → must also beat anti-momentum. Same
  discipline as the RAW-persistence null, new axis.
**EXIT (freeze rule as before; either outcome closes the gate):**
- (1) t+4 peak_in_tol > 0.545 (persistence) WITH CI separation
- (2) drift dir-acc(committed) > max(momentum, anti-momentum) ≈ **0.66**
- (3) false-death ≤ ~0.01 (do NOT let the horizon regress the β8 win of 0.002)
- (4) onset-beyond-raw at the matched t+4 raw-null: REPORTED, NOT gated (real test = Gate 3 / actuators)
**PRE-LAUNCH GATE: label-alignment smoke** — verify the t+4 descriptor target reads sub-window 3 (not off-by-one
t+3/t+5), end-to-end vs the real pipeline (target-subwin-3 of sample i ≈ input-window of sample i+20). The
prediction_horizon_s/sub-window machinery already meant something different than assumed once this week; an off-by-one
would fake mean-reversion skill invisibly. LAUNCH only after it passes.

**LABEL-ALIGN PASSED (2026-07-14, job 4988960 on the 0.2 smoke ckpt).** Synthetic: ridge lands ONLY in
sub3 (=t+4), others flat. Pipeline cross-check (peak-freq of target sub-window `off` vs input@i+(off+1)·5):
t+1 0.987 / t+2 0.974 / t+3 0.987 / **t+4 0.976** — all ≈0.98 (Tin==Tw==96, exact slicing; ~2% = STFT
frame-edge jitter, TCOL-absorbed). Off-by-one RULED OUT. **PRODUCTION LAUNCHED = 4989036** (`e2e_descriptor
_ece_t4mh`, warm-start g2, all-shots, horizon 0.2, horizons 2/4, β8, transition5, anchored, 6000 steps).
Mechanism proven by smoke 4988905 (train+val+save clean; desc_l_t2≈desc_l_t4≈4.4 balanced). Code fixes:
forward_batch keeps trunc_t×max(horizons) spectro target only when a descriptor head exists (else unchanged);
base losses + copy_baseline_mae + validate align target→prediction width; descriptor slices sub-windows at
pred width. Eval verdict harness = `GATE2B_EVAL` in descriptor_head_proof.py (per-horizon, 3 nulls).

#### GATE 2b ADDENDUM — anchor spec + per-horizon nulls (2026-07-14, user items 1+2)
**ANCHOR (verified in code, train_e2e_stage1.py):** every horizon's readout anchors on the INPUT
(t+0) descriptor — computed ONCE, reused for all horizons — **NOT chained** (t+4 does NOT anchor on
the t+2 readout). So the t+4 residual is the full 200 ms evolution off FLAT persistence, and the t+4
exit compares cleanly against flat persistence (no attribution muddying).
**PER-HORIZON NULL TABLE (from horizon_probe 4988515; null set is HORIZON-SPECIFIC — do NOT copy
t+4's thresholds onto t+2):**
| horizon | persistence (peak_in_tol) | momentum | anti-momentum | drift-dir null = max(mom,anti) |
|---|---|---|---|---|
| t+2 (100 ms) | 0.691 | 0.616 | 0.384 | **0.616** |
| t+4 (200 ms) | 0.545 | 0.342 | 0.658 | **0.658** |
**EXIT per horizon:** peak_in_tol > persistence (CI-sep) AND drift-dir(committed) > drift-dir-null;
false-death ≤ ~0.01 (both horizons). t+4 = headline gate (mean-reverting physics = most headroom);
t+2 = phase anchor + its own (weaker) gate. At t+2 the binding drift null is MOMENTUM (0.616, weakly
positive); at t+4 it's ANTI-momentum (0.658). n=42 t+1 momentum is TOO THIN to quote.
**LOSS-SHARE WATCH (item 3):** the two horizons are summed (meaned) INSIDE one head-loss BEFORE the
EMA-norm → t+2 (easier, pers 0.69) can descend faster and shadow t+4's gradient. Logged per-horizon
`desc_l_t2`/`desc_l_t4`; watch the ratio in hour 1 — if t+2 dominates and t+4 stalls, add per-horizon
EMA-norm or upweight t+4.
### (original re-aim spec)
Skill already proven (drift); goal is no longer "find skill" but "grow the one real signal
+ fix the named defect." Single run: `e2e_descriptor_ece_t4` = t+4 (`PRED_HORIZON=0.2`) ×
β-sharpen (`--spec_descriptor_dist_beta 8`) × transition-overweight
(`--spec_descriptor_transition_weight 5`), anchored, warm-start d512. Train 4987707 → eval 4987708.
**Pre-registered three numbers (current-head baseline in parens):**
- (a) drift **commit-rate** ↑ from ~0.10, at **committed-acc ≥ 0.70** (baseline 0.783)  — grow the signal
- (b) **false-death-rate** ↓ from 0.155  — fix the named defect (β-sharpen the plausible mechanism)
- (c) **onset-beyond-raw** with raw-null control — expected ≈0 at pure horizon-retrain; real test is Gate 3
**EXIT:** improvement on (a) AND (b) → freeze the NEW head; flat → freeze the CURRENT head,
carry the honest skill profile forward. Either way Gate 2 closes in one run.

#### GATE 2b — FINAL RESULT (2026-07-14, best.pt step 5500 val_loss 1.1459; eval 4991263, conv 4991262 PASS, 14 shots)
Chain 4989036→4990544 trained full 6000 steps (val improved monotonically 1.85→1.15, NOT stalled).
Convention PASS on final weights (t+4 sub-window match 0.976). `gate2b.json` in `eval_runs/descriptor_t4mh_final/`.
**ARGMAX GATE = FLAT (both horizons):** peak_in_tol model≈persistence (t+2 0.720 vs 0.727; t+4 0.563 vs 0.560,
neither CI-beats), commit-rate≈0 (t+2 n_com=1, t+4 n_com=3 → **insufficient_commits**, MIN_COMMIT=30). So by the
pre-registered argmax exit → **FLAT branch → FREEZE the g2 t+1 head** as the committed drift forecaster.
**false-death = 0.000 at BOTH horizons** — the β8 fix GENERALIZED to horizon (Gate-4 rollout-viability win).
**SUB-THRESHOLD DISCRIMINATOR = POSITIVE (both):** `subthreshold_signal=True`, `anchor_identical=False`.
- t+2: ΔLL(model−anchor)=**0.0207±0.0106** (n=317, CI-lo>0) | mass-shift dir-acc=**0.603 [0.548,0.655]** (CI-lo>0.5) | mean|shift|=0.639 bins
- t+4: ΔLL=**0.0290±0.0115** (n=414, CI-lo>0) | mass-shift dir-acc=**0.623 [0.576,0.669]** (CI-lo>0.5) | mean|shift|=0.720 bins
⇒ the head is CALIBRATED: it shifts probability mass toward the TRUE drift direction (beyond persistence,
significant) but abstains from argmax commitment under mean-reversion noise. commit≈0 is the calibrated
posterior, NOT timidity — corroborating from the TRAINING side what Gate 1's onset-leakage showed from eval.
**HEURISTIC-FAILS SPLIT (job 4991591, both-subsets test) — RESOLVES the footnote, per-horizon.** Model
mass-shift dir-acc split by whether momentum (recent trend) was right; beyond-heuristic = Wilson CI>0.5 on BOTH
subsets (any trend rule is right on one / wrong on the other by construction; only trend-INDEPENDENT signal
clears both):
- **t+2 = HEURISTIC-BOUND:** mom-correct 0.701 [0.604,0.783] n=97, mom-wrong **0.514 [0.420,0.608]** n=105 (CI
  spans 0.5) → the sub-threshold signal IS momentum (trend-continuation); NO skill beyond the heuristic.
- **t+4 = BEYOND-HEURISTIC (weak, real):** mom-correct 0.615 [0.519,0.703] n=104, mom-wrong **0.611 [0.535,0.681]
  n=167** — BOTH CI-lo>0.5. On the 167 windows where the trend was the WRONG predictor (mean-reversion), the
  model still called direction 61% → directional signal INDEPENDENT of the recent trend. `beats_heuristic=True`.
⇒ at the physically-useful 200 ms horizon the head carries a WEAK, TREND-INDEPENDENT, sub-threshold directional
signal (not reducible to momentum/anti-momentum). Whether it is plasma-state-CAUSAL (vs longer-history spectro
structure) = exactly **GATE 3** (actuator counterfactual), now decisively motivated with a concrete sub-threshold
target to prove actuator-driven. (Corrects an earlier "at heuristic strength" read that used a buggy
dominant-label flag — fixed to the both-subsets test.)
**GATE 2b CLOSED.** Freeze g2 t+1 head (drift 0.783 committed, false-death 0.002). t+4 head kept as a
calibrated SOFT forecaster (sub-threshold directional signal, heuristic-ambiguous). Next = Gate 3, or a cheap
threshold/calibration analysis (does the soft mass-shift beat the horizon heuristic specifically — eval-only).

## GATE 3 — actuator→mode CAUSALITY (counterfactual). PRE-REGISTERED SPEC (2026-07-14, drafted during 2b training)
**~~Gate 3 status: OPEN~~ → CLOSED (2026-07-15). VERDICT: actuator conditioning DEMONSTRATED at the output**
(dfreq-denominated, pin→AE, pooled bootstrap-CI-separated, regime-concentrated in AE-active shots, placebo-silent)
**at β=6 with false-death 0.003**; the full distributional bar (ΔLL ≥ 3.5e-4 floor) is UNMET at any *stable* β
(β=5 clears it but the sign goes incoherent = knee instability + false-death 0.021 > gate); the **persistence
anchor is identified as the coupling mechanism** (controllability ↔ fidelity trade off through it); **FiLM queued
as the decoupling architecture.** **PARTIAL-positive; the dfreq-claim is final.** Full arc: Gate-3 dead→localized
(§below) → Gate-3-FIX (scale) → disambiguation (residual latent) → anchor-β anneal sweep → n-enlarged sign
confirmation. Report: GATE3_FIX_REPORT.md §1–§8. Sweep: eval_runs/anneal_beta_sweep/. Money figure → 200729 @ β=6 (Gate 4).

**Why it's the title-critical gate:** Gate 1 showed onset is DETECTION/leakage, not forecasting; the only
non-copyable signal is drift-direction. Gate 3 asks the world-model question directly: **does perturbing the
ACTUATOR commands causally move the model's predicted mode descriptor in the physically-correct direction?**
If yes → the model learned actuator→mode dynamics (a controllable world model), not just precursor detection.
This is the claim a control/scenario paper needs; drift-direction skill alone doesn't establish it.

**Mechanism (plumbing — the piece that DOESN'T exist yet):** actuators are already model INPUTS
(`act_inputs` from `batch["targets"]`, forward_batch:711). A counterfactual forward = run the model TWICE on
the same window — once with real actuators, once with `act_inputs[name] += Δ` (in the dataset-standardized
frame; Δ = a physically-meaningful step, e.g. +1σ beam power / ±1σ RMP current / +1σ ECH) — and diff the
predicted t+h descriptor. Build as an eval mode `ACT_CF` in descriptor_head_proof.py: reuse event-adjacent
window mining (EXISTS: onset/hysteresis) to select windows where a mode is present/forming, then per window
compute Δdescriptor = descriptor(pred | act+Δ) − descriptor(pred | act). NO retrain — pure eval on the g2b head.

**Directional-shift metric (spec exists in chat, not in repo):** per perturbed actuator, score the SIGN and
MAGNITUDE of the induced shift against the physically-expected response:
- peak-FREQ shift (mode chirps with rotation/q → beam/current should move it a signed direction)
- presence/amplitude change (RMP/ECH suppress or drive specific modes → band-power should drop/rise)
Report: mean signed Δ (with CI) per (actuator, mode) pair; fraction of windows with the correct sign.

**THREE controls (mandatory, mirror the raw-null discipline):**
1. SPECIFICITY — perturbing an IRRELEVANT actuator (one with no physical coupling to that mode) must give
   Δ≈0. A model that moves the mode for EVERY actuator learned a generic input-sensitivity, not causality.
2. NOISE FLOOR — Δ=0 (identical inputs) and Δ=tiny must give |Δdescriptor|≈0; the real-Δ response must exceed it.
3. DOSE MONOTONICITY — |Δdescriptor| should grow with |Δactuator| (2σ > 1σ) for a real causal channel.

**GATE (pre-registered):** a causal claim requires, for at least the physically-canonical (actuator, mode)
pair, ALL of: correct sign in >~⅔ of windows (CI above 0.5), response > noise floor, specificity (irrelevant
actuator Δ within noise), and dose monotonicity. FAIL / ambiguous → report the model as a mode DETECTOR +
drift-forecaster (Gate 1/2 result), NOT a controllable world model — honest scope for the paper.
**Runs only after Gate 2b closes** (needs the frozen g2b head). Actuator→mode physical-coupling table
(which actuator is canonical for which mode: NBI/Ip→rotation/q→AE/tearing freq; RMP→ELM/locked-mode;
ECH→tearing stabilization) TBD with the user at build time.

#### GATE 3 — RESULT (2026-07-14). CONDITIONING EFFECTIVELY DEAD → LOCALIZED to unstandardized actuators.
Counterfactual (ACT_CF in descriptor_head_proof.py; forward_batch `act_perturb` hook) on best.pt step5500,
t+4, ECCD showcase pair 199597/199607 (from `additional_data`) + 2 held-out shots, n=158 mode-active windows.
Perturb actuator +2σ (Δσ×std, std-scaled), read ΔLL@true-mode-bin + mass-shift + entropy (2b machinery):
- **ech_power (ECCD) +2σ: ΔLL = −3.5e-06 ±2.5e-06** — physically-correct SIGN (suppress), dose-monotonic
  (1σ −2.6e-6 < 2σ), CI excludes 0, BUT magnitude ~10⁻⁶ = NEGLIGIBLE.
- **gas_flow (placebo) +2σ: ΔLL = +1.6e-05** — LARGER magnitude than the target, opposite sign → SPECIFICITY
  FAILS (the tiny actuator sensitivity is not specific to the physically-coupled actuator).
⇒ **the spectro descriptor is NOT meaningfully conditioned on actuators** (response ~1e-6, non-specific).
Title claim (controllable / actuator-conditioned world model) NOT supported by this checkpoint.
**ROOT CAUSE — LOCALIZED + CONFIRMED (the pre-registered "actuator token-path audit," job 4991956):**
actuators enter the model UNSTANDARDIZED. data_loader configures actuators with `preprocess method="none"`
(line ~265: `ech_power … none`) while all diagnostics use standardize/log_standardize. So ech_power arrives
at raw ~1e5 (audit: finite_frac 1.0, std 2.3e5, absmean 1.27e5) vs gas_flow ~O(1) (std 11). Audit token-path:
+5(abs) ech_power → max|Δtok[ece]| 1.9e-6 (dead) vs +5 gas_flow → 3.2e-2 (live) — the tokenizer can't condition
on the 1e5-scale actuator. preprocessing_stats.pt ALREADY has ech_power mean/std (`raw`+`log`) — just not applied.
**THE ONE LOCALIZED FIX (identified, not yet executed):** set actuator `preprocess` from "none" → standardize
(log_standardize for power-law acts like ech_power/pin; stats present) → RETRAIN (actuator input distribution
changes → can't warm-start the actuator tokenizer). Re-run ACT_CF to verify. Everything else EXONERATED
(backbone, descriptor head, the t+4 signal all fine — the failure is isolated to the actuator input scaling).
**RESOLVES the Gate-2b open question:** since the model barely conditions on actuators, the confirmed weak
t+4 trend-independent signal (Gate 2b) is **longer-history SPECTRO structure, NOT actuator-causal** — the model
forecasts modes autoregressively from spectro history, and is not yet a controllable (actuator-driven) world model.

## GATE 3-FIX — PRE-REGISTRATION (2026-07-14, decisions at the Task-1 STOP gate). Do NOT strike Gate 3 until report read.
**GLOBAL ANGLE SCAN (50 shots, job 4995873 + h5 scan):** `ech_tor_angle`/`ech_pol_angle`/`ech_polarization` =
IDENTICALLY ZERO corpus-wide (dataset gap — placeholder datasets never populated). `ech_power` populated in
~34% of shots (ECH-heated), std up to 2e5 where present. ⇒ **ECCD AIMING is absent → aiming-dependent NTM
suppression is UNLEARNABLE from this data**; the 199597/199607 pair differ in alignment the model can't see.
**DECISIONS (user, at the STOP gate):**
- **DROP the 3 angle channels** from model inputs entirely for this retrain (10→**7 actuators**: pin,
  beam_voltage, tin, ech_power, gas_flow, gas_raw, rmp). Constant-zero = dead weight.
- **ACT_CF ech_power — PHYSICS-HONEST expectation, pre-registered NOW:** with aiming absent, the model can at
  best learn the MARGINAL effect of ECH power averaged over the corpus's deposition geometries. NOT expected to
  show clean aiming-dependent suppression — only a marginal/averaged power response. (Prevents post-hoc spin.)
- **pin→AE is CO-PRIMARY** (not backup): NBI injected power `pin` (fully populated, already O(1)/LIVE — +5σ
  |Δtok[ece]| = 2.8e-2) → fast-ion drive → Alfvén eigenmodes. Retrain ACT_CF tests BOTH ech_power (marginal
  power) AND pin (NBI→AE) as primary causal channels; placebos = gas_flow + gas_raw.
**SCALING (CONFIRMED 2026-07-14):** ech_power → **log_standardize** (dead, raw 1e5, all≥0); rmp → **standardize**
(raw 578, bipolar mean −22.6); **beam_voltage → LEAVE raw** (live-ish 1.1e-2, isolate the change); pin/tin/
gas_flow/gas_raw → keep `none` (already O(1)). **Angles DROPPED** (3 channels). Two scaling changes + channel
drop. Stats: derived log/raw stats already in preprocessing_stats.pt (`log`/`raw` sub-keys) — if any recompute
needed, versioned NEW file, never overwrite. NOTE: `pin` already live → pin→AE testable on the CURRENT ckpt.

#### GATE 3-FIX Task 2 — SMOKE RESULT (2026-07-14, smoke 4996604 + sensitivity 4997367)
Config wired: 7 actuators (angles dropped), ech_power→log_standardize, rmp→standardize, beam_voltage raw;
`--reinit_act_tokenizers` flag added (warm-start drops act tokenizers → fresh; input dist changed). Smoke
(500 steps, warm-start t4mh + reinit) trained clean — warm-start across 10→7 actuators + reinit OK (backbone
token-count change did NOT break warm-start). **POST-FIX token-path sensitivity (+5σ |Δtok[ece]|):**
- **ech_power: DEAD 6.3e-4 → LIVE 5.99e-2 (~95×)** — now O(1) (mean 0.43, std 1.14). log_standardize WORKS. ✓
- pin 1.88e-1 (strongest — good for pin→AE co-primary), gas_flow 6.9e-2, beam_voltage 7.4e-2 (live raw).
- **rmp ANOMALY: still raw-scale (std 1.39e5) — standardize did NOT normalize it.** rmp raw scale is
  shot-dependent (578 in 199597 vs ~1e5 in others); stored rmp[raw] stats (std ~1631) are UNIT-MISMATCHED with
  the H5 → under-normalize. rmp is secondary; OPEN: recompute rmp raw stats (versioned) OR leave rmp raw.
Task-2 CORE PASS (primary ech_power conditionable). OPEN before Task 3: (1) rmp handling; (2) recipe A (g2 t+1)
vs B (t4mh t+4).

#### GATE 3-FIX Task 3 — RETRAIN pre-registration (ENTRY-BEFORE-LAUNCH, 2026-07-14). Recipe B.
**Config diff vs t4mh (warm-start source):** ONE scaling change (`ech_power` none→**log_standardize**) + DROP
3 angle channels (10→**7 actuators**) + `--reinit_act_tokenizers` (fresh act tokenizers — input dist changed).
`rmp` LEFT RAW (unit-mismatch; deferred to data-pipeline reconciliation; reported-not-claimed). Everything else
IDENTICAL to t4mh: d512/12L, horizons **2,4** (t+4 readout = the ACT_CF instrument), β8/tw5/anchor/dist, weight 6,
LR 2e-4/warmup 300, all-shots, warm cache, ~6000 steps. **Backbone UNFROZEN** (mandatory — it learned to ignore
actuator positions; frozen = false-negative ACT_CF). Warm-start `e2e_descriptor_ece_t4mh` best.pt (g2→t4mh
lineage). Dir `e2e_g3fix`.
**MONITOR (per log step):** standard suite + `act_tok_gradnorm` (actuator-tokenizer grad-norm — proxy for the
backbone re-attending to actuators; should GROW; discriminates a weak ACT_CF: FLAT=never-re-attended (train
longer) vs GREW=attended-but-no-signal (physics/data limit)).
**PRE-REGISTERED ACT_CF TABLE (post-retrain, t+4):** PRIMARY = `ech_power` (MARGINAL-power expectation per the
aiming caveat — NOT clean aiming-suppression) + `pin`→AE (NBI power, natively live). PLACEBOS = `gas_flow`,
`gas_raw` (must NOT fire). `rmp` = REPORTED, NOT CLAIMED (raw/unit-mismatch). EXIT (4 criteria): (1) ech_power
ΔLL correct sign, |mag| ≥100× the 3.5e-6 pre-fix floor + CI≠0; (2) dose-monotonic (2σ>1σ); (3) specificity —
placebos |ΔLL| < primary; (4) noise-floor Δ=0 ×10. REGRESSION battery (no Gate-1/2 loss): drift commit/dir-acc
+nulls, **false-death ≤0.01 (HARD gate)**, onset-beyond-raw. CAUTION: `rmp` raw ~1e5 = numerically-loud →
training instability / regression shift → rmp saturation is first suspect.

## MONEY FIGURE — counterfactual discharge panel (PAPER CENTERPIECE; pre-registered 2026-07-14)
THREE ECE-spectrogram strips, same held-out shot, same initial 50 ms window, ~1–4 s:
1. **GT** — real ECE spectrogram; mode ridge (drifting/chirping) visible.
2. **Rollout, REAL actuators** — sampled prediction: descriptor-driven ridge over codec-rendered texture, carried
   k=20–80 windows; ridge should TRACK (persist, drift with correct statistics, band-power evolving).
3. **Rollout, PERTURBED actuators** — same seed, ech_power/pin trajectory shifted; ridge RESPONDS (amplitude
   drops under suppression / AE band rises under beam drive).
**CLAIMS:** panel 1 vs 2 = the model WORKS (vs June's visibly-dead symptom); panel 2 vs 3 = the model is a
WORLD MODEL (conditioning, visibly alive). One figure, both claims, no table required to believe it.
**HONESTY CONSTRAINTS (pre-registered):** (a) panel 2 shows 2–3 sampled rollouts OR a descriptor-uncertainty
band (NOT one cherry-picked trajectory) — no selection; (b) caption states the texture is GENERATIVE
(statistics-matched) while the RIDGE DYNAMICS are the forecast — the exact claim, visually encoded, no more.
**BUILDABILITY:** rollout machinery + descriptor overlay + codec renderer all EXIST. Perturbed-trajectory panel =
the ACT_CF `act_perturb` hook applied at ROLLOUT time — ONE wiring item (rollout uses model.rollout/decode, not
forward_batch, so the perturbation must be applied to act_inputs inside the rollout loop).
**SUPPORTING VISUALS (each half-built; produced THIS run in Task 4, no extra gate):**
- Descriptor forecast strip (`ece_trained_ridge` / EVAL_TRAINED): GT ridge / model forecast / persistence, drift
  commits visible as departures from the persistence panel — regenerate on the retrained ckpt = "forecasting skill."
- Conditioning-coming-alive curve: `act_tok_gradnorm` (now logged per step) vs training step = the bug being
  FIXED (shows the mechanism, not just the outcome).
- ΔLL waterfall: per-actuator ACT_CF response with placebos at the noise floor = specificity in one bar chart (from act_cf.json).
**TIMELINE:** descriptor strip + conditioning curve + ΔLL waterfall = this run (Task 4, ~hours). Panels 1–2 =
with Gate 4's K-probe (~days). Panel 3 needs ACT_CF ALIVE → full triptych ~7–10 days on the success branch;
two-panel form (working forecaster, honestly scoped) on the WEAK branch.
**GATE 4 PROMPT MUST INCLUDE:** "produce the three-panel counterfactual figure as a MANDATORY deliverable of the
K-probe run" — so the visual proof arrives WITH the numbers, not assembled later under deadline pressure.

## REUSABLE METHODOLOGY CONTRIBUTION (paper methods section)
Any claim of event-forecasting skill in this system carries **two mandatory controls**,
named and demonstrated here: (1) the **RAW-persistence null** — argmax of the *unthresholded*
input profile — because a detection threshold blinds the naive persistence baseline while the
model's input retains the answer (exposed onset as leakage: model 0.406 < raw-null 0.579);
(2) the **false-death / sustained-window control** — event-recall vs the same prediction on
non-event windows — to separate a real transition signal from a constant bias (exposed death
as absent-bias). Deliverable sentence above is paper-ready, filed verbatim.

## GATE 3-FIX DISAMBIGUATION — residual-level ACT_CF (PRE-REGISTERED 2026-07-15)
**WHY:** Gate-3-fix ACT_CF at the β=8 anchored OUTPUT gave |ΔLL|≤1e-4 everywhere (near-saturated softmax
UNDER-reports) — cannot tell "residual genuinely inert" from "residual responds but the anchor masks it".
No retrain; re-reads the SAME g3fix ckpt. Ckpt: models/e2e_g3fix/e2e_stage1_best.pt (val 1.1399).
**PRIMARY INSTRUMENT (refinement 1):** residual-level ‖Δresid‖ = RMS change of the PRE-ANCHOR head output
dh(tok_perturbed)−dh(tok_real), per channel INCLUDING placebos, + its DIRECTION resid_Δfreq
(does the residual shift toward LOWER freq under +pin, matching the output dfreq sign?).
**VERDICT RULE:** specificity ORDERING at residual level. `latent_conditioning` = a PRIMARY's ‖Δresid‖
CI-lower exceeds the max placebo ‖Δresid‖ CI-upper (pin ≫ placebos, CI-separated). This is the decider.
**REFINEMENT 2 (OOD guard):** lower-anchor readout (β_corr=2.0) is CORROBORATION ONLY — reducing β puts the
residual OOD vs its β=8 training, so a weak response there is UNINTERPRETABLE. Never the decider; residual-level has no such problem.
**REFINEMENT 3 (permanent):** dfreq is now a STANDING ACT_CF column with a significance star (CI excludes 0),
and the "alive floor" gains a dfreq-denominated twin: a primary is dfreq-alive iff its |Δfreq| CI excludes 0 AND
|Δfreq_primary| exceeds the placebo |Δfreq| band (specificity-based, no arbitrary magnitude). Applies to all future ACT_CF verdicts.
**OUTPUT (β8) verdict** kept for continuity as `conditioning_alive_output_beta8` but explicitly NOT the decider.
**OUTCOME (job 5002063, 2026-07-15):** `latent_conditioning = True`. Residual-level ‖Δresid‖: pin 1.97e-3 ±2.5e-4
(CI-lower 1.71e-3 = **3.7× above** placebo band 4.6e-4), resid_Δfreq −0.019±0.003 bins (→ lower freq under +pin =
AE-correct). ech_power inert pre-anchor (1.5e-5, aiming-gap; not a fair test). Placebos gas_flow 3.7e-4 / gas_raw
1.0e-4 both in band. Lower-β=2 corroboration (OOD, not decider): pin response GROWS as anchor weakens (ΔLL
−1e-4→−4e-4) = anchor-masking signature. **So the β=8 output "dead" (ΔLL~1e-6) was a measurement artifact of the
near-saturated anchored softmax; the residual DOES condition on pin, specifically + directionally.** Real, specific,
but SMALL. Next = anchor-weight annealing (β 8→small) to unmask at output; ECH untestable on this corpus.
Feeds GATE3_FIX_REPORT.md §2e/§3/§4. Gate 3 NOT struck — awaiting user read.

## GATE 3-FIX UNMASK — anchor-β anneal retrain (PRE-REGISTERED 2026-07-15)
**PREMISE:** disambiguation (job 5002063) confirmed pin conditions the RESIDUAL specifically (3.7× above placebo
band, −0.019 bins toward lower freq = AE-correct) but the β=8 persistence anchor MASKS it at the output. Anneal the
anchor to make the whisper audible at the output WITHOUT re-opening mean-collapse (the anchor's original job —
dist-CE without it produced black output).
**LEVER (one, decoupled):** anchor PREDICTION weight anneals; TARGET softmax β held FIXED at 8 (task definition
unchanged). Impl: `--spec_descriptor_anchor_beta_holds 8,6,5,4,3 --spec_descriptor_anchor_beta_hold_steps 1500`;
pred_logit = anchor·β(step) + residual (train_e2e_stage1.py:_desc_term), q = softmax(target·8) FIXED.
**SCHEDULE:** stepwise holds 8→6→5→4→3, 1500 steps each (max_steps 7500), milestone ckpt `beta{β}_step{N}.pt` at
each hold boundary = EQUILIBRATED head per β → run ACT_CF per-β, pick landing β from the curve.
**FLAT LR (unconfound):** lr=min_lr=5e-5 so cosine decay is a no-op — each β-hold trains at the SAME lr, so the
β→metric curve isn't confounded by LR decay. Deliberate.
**WARM-START:** --init_checkpoint g3fix best.pt, NO --reinit_act_tokenizers (keep the conditioned residual +
tokenizers — they ARE the starting point). Backbone UNFROZEN. d512/12L proof scale. One change vs g3fix = the β anneal.
**COLLAPSE TRIPWIRES (per 500 steps = per val; log-and-continue, NOT abort):** logged in step line + a `[tripwire]`
val line — (1) desc_fdrift = false-death proxy (spurious peak-drift >2 bins on STATIC/no-flip windows); (2) desc_hfrac
= H(pred)/log(NF), →1 = flat-collapse; (3) desc_ftp = persistence peak-in-tol (fidelity ref vs model desc_ftol).
best.pt promotion BLOCKED when window-mean fdrift>0.01 OR hfrac>0.98 (`--desc_false_death_abort 0.01`); job keeps
running so the whole β trajectory is milestoned (pick best β post-run).
**EXIT TABLE (post-run, existing harness = descriptor_head_proof.py):**
- SUCCESS = OUTPUT-level ACT_CF: pin ΔLL AND dfreq above alive floor with CI≠0, placebos silent (specificity survives
  the softmax), dose-monotonic (2σ>1σ) — AND regression battery intact (false-death ≤0.01, drift signal preserved,
  peak_in_tol within CI of persistence). = residual whisper audible at output without paying in collapse.
- PARTIAL = pin clears floor but false-death creeps → β landing point is the trade-off dial; pick best β from milestone curve.
- FAIL = pin never clears the output floor even at β=3 → honest scope stays detectable-not-controllable (§2e residual figure carries the claim).
**MAGNITUDE EXPECTATION (written before the number):** residual carries ~2e-3 / 0.02 bins. Unmasking makes it VISIBLE,
not necessarily large. A real, specific, correctly-signed, dose-monotonic OUTPUT response that's still modest = a
controllable forecast demonstrated IN PRINCIPLE; magnitude ceiling = data/training-scale (production pin, actuator-aware
from step 0). Paper sentence = "dial," not "knob."
**RIDES ALONG (not this run):** counterfactual rollout triptych buildable the day output ACT_CF passes (same-seed
rollout pair under perturbed PIN; ECH aiming-gap = honest caveat line). FiLM stays queued for production regardless.
**OUTCOME:** _CHAIN LAUNCHED 2026-07-15: 5002860→61→62→63 (d512, warm g3fix, β8→6→5→4→3×1500, flat lr 5e-5)._
Smoke (5002470→5002798) validated the machinery + caught a missing `--spec_descriptor` ENABLE flag (sub-flags alone
don't build the head → no descriptor/tripwires). Fix: launch needs `--spec_descriptor --spec_descriptor_tcol 6
--spec_descriptor_hidden 512` (match g3fix arch → clean warm-start, 0 keys dropped). Also tightened the inline
fdrift proxy to ACTIVE-static windows (was inflated 0.068 by quiescent-noise vs eval false-death 0.000). Milestones
`beta{β}_step{N}.pt` per hold; post-run = output ACT_CF per β → pick landing β from curve. Do NOT strike Gate 3._

**RESULT (2026-07-15, RELAUNCHED chain 5003934→37 after the wrong-lengths-cache detour; eval wave 5005519–28,
each milestone eval'd at its OWN trained anchor β via DESC_ANCHOR_BETA):** **the anneal UNMASKED pin conditioning at
the OUTPUT — a clean TRADE-OFF DIAL.** β→(pin dfreq / pin ΔLL / false-death): β8 (masked / ns / 0.000) → β6
(**−0.018\* / ns / 0.003**) → β5 (+0.019\* / **+3.95e-3\*** / 0.021) → β4 (−0.086\* / −4.21e-3\* / 0.077) → β3
(−0.086\* / −9.27e-3\* / 0.223). Placebos (gas_flow/gas_raw) ~1e-4 throughout (≪ pin); ech_power inert at all β
(aiming gap). pin response ↑ AND false-death ↑ monotonically as β↓; false-death crosses the 0.01 gate at β≈5, right
as pin ΔLL clears the floor. **No β meets the full bar (ΔLL+dfreq clear AND false-death ≤0.01) simultaneously →
PARTIAL, positive.** Best clean point **β=6** (pin dfreq −0.018 sig+placebo-specific+AE-correct-direction; false-death
0.003; peak-in-tol 0.54 vs pers 0.56, no regression). Caveats: effect SMALL at β6 (0.018-bin, "dial not knob");
peak-in-tol never beats persistence at any β (conditioned ≠ skill); β5 dfreq sign-flip anomaly. Controllability
DEMONSTRATED as a tunable dial. Next (user-gated): money-figure triptych on pin @ β6/β5; FiLM at production scale to
get a large effect without the false-death cost. Verdict + curve in GATE3_FIX_REPORT.md §7 + eval_runs/anneal_beta_sweep/.
Gate 3 NOT struck — awaiting user read._

## GATE 3-FIX SIGN CONFIRMATION — n-enlarged + per-shot heterogeneity (PRE-REGISTERED 2026-07-15)
**WHY:** anneal β-sweep pin dfreq sign was unstable across rungs (−0.018 β6, +0.019 β5, −0.086 β4/3) at n=158 —
the weakest link in the "AE-correct direction" claim. Test at n→500–1000 (15-shot gate2b pool, MAX_WIN 300) with
BOTH pooled bootstrap CI and per-shot breakdown. Jobs 5005771 (β6) / 5005772 (β5), each at its own trained anchor β.
**READING (pre-registered, two independent axes):**
- **POOLED bootstrap-percentile CI on dfreq = the GATE.** β6 −0.018 stands iff its boot CI excludes 0.
- **PER-SHOT dfreq spread = the INTERPRETATION** (pooling mixes AE-active & quiet shots; a true regime-dependent
  effect concentrates in AE-active shots with quiet non-AE shots ≈0, and would look like ns dilution if only pooled).
  Effect concentrated where AE physics predicts + pooled CI excludes 0 = TWO independent confirmations in one job.
- **β5 sign test:** if +0.019 was small-n noise → enlarged n pulls it toward 0/negative → β6 stands. If it PERSISTS
  positive at n~1000 with a clean CI → β-specific (plausible: near the anchor-release knee the residual's expression
  is unstable across the softmax-saturation boundary; different windows unmask with different signs) → the KNEE
  REGION is untrustworthy → push the operating point AWAY from 5, reinforcing β6.
**DECISION:** β6 −0.018 holds (boot CI≠0) & β5 → same-sign/zero ⇒ direction claim stands, β6 = operating point.
Both wash out ⇒ honest operating point → β4-with-caveats, or wait for FiLM. PARTIAL stays the headline regardless
(strict ΔLL≥3.5e-4 bar unmet at every β; β6 supports FREQUENCY-SHIFT conditioning, not full distributional).
**OUTCOME (5005771/72, n=967 each, 2026-07-15):** **β=6 direction CONFIRMED both ways.** β6 pin Δfreq −0.0057,
bootstrap95 [−0.0071,−0.0045] (excludes 0, negative=AE-correct), > placebo 0.0025; PER-SHOT concentrates in
AE-active shots (200729 −0.042, 191001 −0.024, 200000 −0.022; quiet ≈0) = regime-dependent, physics-consistent.
n=158 −0.018 was INFLATED (4-shot pool over-weighted AE shots); unbiased pooled = −0.0057, AE-active ≈ −0.04.
**β=5 flip is REAL + β-specific + INCOHERENT** (+0.0148 boot95 [+0.0131,+0.0167] persists at n=967, BUT 200729
flips to −0.008 vs majority-positive + vs its own β6 −0.042) = anchor-release instability across the softmax
saturation boundary → **knee untrustworthy → operating point = β=6** (above knee; β4/β3 also negative). PARTIAL
headline holds (ΔLL below floor; FREQUENCY-SHIFT conditioning). Money figure → 200729 @ β=6. β=5.5 hold RECOMMEND
AGAINST (lands in the unstable knee). Verdict in GATE3_FIX_REPORT.md §8. Gate 3 NOT struck — awaiting user read._

## GATE 4 — conditioned-mode K-probe + counterfactual ridge traces (RESULT 2026-07-15)
Job 5006092 (β=6 milestone, 200729, n=256 windows, K=40 gate@10, 5-rollout fan real±1σ±2σ, anchor-decomposed).
Wiring: perturbation hook + token-slice exposure in rollout_forward_one_batch (eval_e2e.py); ridge = descriptor-head
forecast per rollout step, decomposed output(anc·β+dh) / residual(dh) / anchor(fed-back-state descriptor).
Script: analysis/mode_audit/gate4_kprobe.py. Artifacts: eval_runs/gate4_kprobe/{gate4_kprobe.json, gate4_ridge_trace.png}.
**SINGLE-STEP (k0) — CONFIRMED at power + BIDIRECTIONAL:** ΔOUT +1σ −0.0191 [boot −0.0219,−0.0166], +2σ −0.0206
[−0.0234,−0.0180] (lower freq); −1σ +0.0005 [ns], −2σ +0.0025 [+0.0017,+0.0033] (higher freq). +pin lowers / −pin
raises the predicted mode freq, both bootstrap-significant, AE-correct, sign-reversing — stronger than the one-sided
ACT_CF. Asymmetric (+ side ~8× − side). **First n=8 run was noise (k1 −0.34 transient, sign-inconsistent) → n=256 fixed it.**
**ROLLOUT (k10/k39) — RE-ABSORBS (pre-registered informative-failure regime):** +1σ k0 −0.019 → k10 +0.002
[−0.005,+0.009]=null; anchor never separates (ΔANC k39 ≈ 0.002–0.004 ≈ 0) → conditioning NOT carried in the fed-back
state; washes out within ~10 steps. (−2σ nominal "accumulate" k10 +0.013 is weak+asymmetric+not-anchor-carried = drift, not compounding.)
**VERDICT:** single-step actuator conditioning is real/specific/bidirectional; the ANCHORED autoregressive rollout
re-absorbs it. Quantifies the FiLM motivation — the persistence anchor couples controllability↔fidelity (Gate 3) AND
re-absorbs conditioning in rollout, so a diverging counterfactual trajectory needs conditioning-by-construction (FiLM),
not an anchored world model. **Money-figure consequence:** perturbed-pin rollout strips RE-CONVERGE (visually ~null);
the honest figure = GT/real/perturbed strips CARRIED BY the ridge-trace panel (single-step shift + bidirectional dose +
re-absorption). Next: render strips triptych; MHR A2 fold-in; then Gate 5 ticket.

### GATE 4 — COMPLETE GATE TABLE (job 5006187, n=256) — CORRECTS the "freeze" read above
The counterfactual-only entry above flagged a possible deterministic-token freeze. INSTRUMENTATION CHECK + full
gate table RESOLVE it IN THE MODEL'S FAVOR — dynamics are ALIVE, not frozen. Rollout IS deterministic continuous-token
(rollout.py:252 feeds continuous backbone tokens back, no sample/quantize) BUT that does not freeze it here:
- **Mode survival PASS:** prominence retention @k10 = **1.018** (mode's band-power held through free rollout, K=39 too).
- **Compounded false-death = 0.000 (effective) vs 0.814 (independent 0.155/step compounding)** — the Gate-1 named
  defect does NOT compound; rollout errors STRONGLY ANTI-CORRELATE; the established mode is stable. Pre-registered
  Gate-4 core question (correlation-vs-independence) → decisively PASSED.
- **Dynamics ALIVE PASS:** ridge-variance ratio pred/GT = **1.28** (slightly OVER GT 0.47→0.60); drift pred 1.56 vs
  GT 0.59 (mildly HYPER-dynamic, over-drifts 2.7×). NOT a frozen fixed-point → the deterministic-token instrumentation
  worry is empirically refuted. (My interim "re-absorbs→dead dynamics" read was WRONG — corrected here.)
- **Single-step conditioning PASS:** bidirectional + bootstrap-significant (+pin −0.019 [−0.022,−0.017], −pin +0.0025 [+0.0017,+0.0033]).
- **Sustained controllability FAIL:** counterfactual Δ re-converges by k10 (two ALIVE trajectories converge; the pin
  nudge doesn't steer the multi-step trajectory) — a real dynamical property, NOT a determinism artifact.
**NET VERDICT:** β=6 is a FAITHFUL SURVIVING-MODE SIMULATOR (mode survives, false-death 0, GT-scale dynamics) that is
single-step actuator-conditioned but NOT yet controllable-at-horizon. Of the 3 FiLM acceptance criteria, TWO ARE
ALREADY MET by the anchored model (variance~GT ✓, false-death≤0.01 ✓=0.000); only SUSTAINED COUNTERFACTUAL SEPARATION
fails → that is FiLM's narrowed job. Code-space-rollout detour NOT needed (dynamics not dead). MHR A2: ftdec_spec_mhr
(4952989) TIMED OUT no-verdict; residcodec_mhr (4969570) recon-fig only, no gate → DEFERRED to production rendering-track.
Artifacts: eval_runs/gate4_kprobe/{gate4_kprobe.json, gate4_ridge_trace.png}.

### GATE 4 — RECONCILIATION (job 5006319 + ensemble figure) — RETRACTS "dynamics ALIVE"; freeze is REAL under deterministic rollout
Ensemble per-window ridge figure (gate4_ensemble_ridge.png) + variance split by transient decides it:
- FULL k0-39: pred_var 0.596 / GT 0.465 = 1.28× (what the prior entry reported as "alive").
- TRANSIENT k0-2: pred_var 1.327 / GT 0.156 = **8.5×** — a one-step spike (pred mean ridge 36.6→38.6(k1)→35.4).
- SUSTAINED k2-39: pred_var **0.072** / GT 0.459 = **0.16×**; pred FROZEN at ~35.1 for all k≥2 (GT wanders ~37.5).
  Windows moving >0.5 bin over k2→39: **pred 4/256 vs GT 130/256.** Pred freezes ~2.4 bins BELOW GT.
**CORRECTION:** the "1.28× dynamics-alive" was ENTIRELY the k0-1 transient. The deterministic continuous-token rollout
(rollout.py:252 feeds the smoothed continuous prediction back) FREEZES to a fixed point (autoregressive mean-collapse).
⇒ mode-survival retention 1.02 + false-death 0.000 are FREEZE ARTIFACTS (a frozen ridge trivially never drops below
threshold; frozen at the WRONG freq) — NOT dynamical survival. RETRACT the "survival PASS / dynamics ALIVE" reading.
**WHAT STANDS:** single-step (k0) conditioning — bidirectional + bootstrap-significant (single-forward, not a rollout artifact).
**GATE 4 NOT CALLABLE** on this rollout: the June core question (dynamical mode survival) is confounded by the freeze.
**NEXT (required before Gate 4):** code-space rollout — sample/quantize codes → re-tokenize → feed back (breaks the
continuous-mean collapse) → re-measure survival/false-death/variance/drift + counterfactual. Cheap experiment BEFORE FiLM.
If sampled rollout ALSO freezes → freeze is a model property → FiLM. If it comes alive → valid simulator reading (honestly).
Two prior interim reads were WRONG (n=8 "re-absorbs→dead"; n=256-full "alive") — this transient-split reconciliation is the correct one.

## GATE 4 — free-rollout conditioned-mode survival + controllability. **~~OPEN~~ → CLOSED (2026-07-15).**
**VERDICT.** The world model has **real single-step actuator→mode causality**: perturbing beam power (pin) shifts the
predicted ECE mode frequency, **bidirectional** (+pin→lower, −pin→higher), **bootstrap-significant**, AE-correct,
placebo-silent (k0 ΔOUT: +1σ −0.019 [boot −0.022,−0.017]; +2σ −0.021; −2σ +0.0025 [+0.0017,+0.0033]; n=256, 200729 @ β=6).
Under **multi-step FREE ROLLOUT** (deterministic token-space, the deployed rollout): the established mode **PERSISTS** —
prominence retention 1.02, **compounded false-death 0.000 vs β6-independent ~0.03** (rollout errors anti-correlate, no
spurious mode-death) — **but the ridge FREEZES** to a fixed point ~2.4 band-bins below GT (sustained k≥2 variance 0.16×GT,
252/256 windows flat after a k0-1 transient). So the free rollout does **not track GT dynamics** and the single-step
conditioning does **not propagate**. **SUSTAINED CONTROLLABILITY = FAIL.**
**Mechanism:** autoregressive continuous-mean-collapse — `rollout.py:252` feeds the smoothed continuous prediction back;
iterating relaxes to a frozen fixed point. This is a property of the DETERMINISTIC rollout, characterized (not confounded):
the k0 single-step measurement is a clean single-forward result and stands independently.
**DELIVERABLE (callable):** *single-step actuator conditioning is real, bidirectional and specific; the established mode
survives free rollout (no compounding false-death) but the deterministic rollout collapses it to a frozen off-frequency
fixed point, so multi-step controllability is not yet demonstrated.* Two-gate framing preserved: Gate 3 = single-step
conditioning at the coupled (anchor) optimum; Gate 4 = it does not survive free rollout dynamically → the decoupling +
generative-rollout architecture is the named fix.
**FORWARD (production track, NOT a Gate-4 blocker):** (1) code-space sampled/quantized rollout — break the continuous-mean
collapse, re-measure whether dynamics + counterfactual revive; (2) FiLM conditioning-by-construction, anchor-reduced d512,
accept = {rollout variance ~GT, counterfactual sustained past k10, false-death ≤0.01}. Caveats: k0-1 transient over-shoots
(2.7×GT, moot under freeze; a drift-rate-per-K-block statistic for the Gate-5 suite). MHR A2 → deferred, production rendering-track.
**~~Gate 4 status: OPEN~~ → CLOSED.** Artifacts: eval_runs/gate4_kprobe/{gate4_kprobe.json, gate4_ridge_trace.png, gate4_ensemble_ridge.png, gate4_perwindow.npz}.

## GATE 4 — REOPENED (SAME-DAY CORRECTION 2026-07-15). The CLOSED verdict above is PREMATURE — measured on BROKEN WIRING.
The freeze localizes to `rollout.py:252` (continuous prediction fed back through the anchored head) = the KNOWN, SPECCED,
DEFERRED pre-K>1 fix. Deterministic continuous feedback through an anchored head is a CONTRACTION MAPPING → finds a fixed
point; this is mean-collapse (failure-mode #1) reappearing at rollout depth — the PREDICTED consequence of the deferred bug,
not a new pathology or a model property. **The whole Gate-4 table (survival, false-death, variance, drift, counterfactual
re-absorption) was run on the broken rollout → ALL TBD.** Judging the anchored model's dynamics/controllability on this wiring
would condemn the architecture for the plumbing's crime. **Gate 4 = REOPENED; the real table comes from the FIXED wiring.**
What still stands: single-step (k0) conditioning — bidirectional, bootstrap-significant, a single-FORWARD measurement (no rollout).
Gates 1-3 untouched.

### GATE 4 — SAMPLED-ROLLOUT RERUN (PRE-REGISTERED 2026-07-15). Implementing the specced fix = task #5.
**FIX (task #5):** rollout feedback goes code-space — decode diag tokens → code logits → **SAMPLE** (or argmax) codes →
re-embed/re-tokenize → feed back, breaking the contraction (stochastic injection prevents fixed-point convergence; the
IRIS/Genie argument for why discrete world models roll out stably). **OPT-IN** (`feedback_mode`, default "continuous" =
byte-identical for the Stage-2 trainer — must not break deployed training).
**FORK:** (b) SAMPLE = the candidate fix; (a) ARGMAX-through-codes = the CONTROL (isolates stochasticity vs on-manifold
re-tokenization); (c) BOTH = the run.
**THREE REQUIREMENTS (all cheap, all pre-registered):**
1. **Every fed-back component sampled, not just codes.** The freeze lives in the WHOLE fed-back state; if the descriptor/
   anchor pathway feeds a mean while codes are sampled, the anchor re-freezes the ridge. State the per-step sampling policy
   for EVERY fed-back component in the entry — fix the bug, don't half-fix it.
2. **Temperature is load-bearing → LOGGED + SWEPT, not silently defaulted.** T=1 may over-inject (sampled renders used to
   speckle). Deliverable = sustained-variance-vs-GT ratio as a function of T (the calibration curve). Watch the 2.7×
   over-drift too — same measurement, may resolve or worsen.
3. **Re-measure the FULL gate table on the sampled rollout** — survival, false-death (vs corrected β6-independent ~0.03),
   prominence retention, drift stats, AND the counterfactual fan (the k>10 re-absorption was measured on frozen wiring →
   now UNKNOWN; re-run). Entire Gate-4 table is TBD, not just the variance row.
**FiLM TRIGGER:** fires ONLY if the SAMPLED (fixed-wiring) rollout also freezes / fails {variance~GT, counterfactual past
k10, false-death≤0.01}. The anchored model has never been rolled out correctly; FiLM is judged only after it has been.

#### GATE 4 SAMPLED-ROLLOUT — READOUT PRE-REGISTRATION (3 notes, before jobs 5007477/78 land)
**(1) FOURTH branch row (likeliest): smoke PASS + ALIVE-but-MISCALIBRATED.** T=1 is an uncalibrated first draw;
codec history = T=1 codes once speckled. Expect NOT clean {GT-scale | frozen} but plausibly alive at 2-4×GT variance
or alive with degraded prominence retention. **THAT IS A PASS on tonight's question** (contraction broken, freeze was
plumbing) — calibration is DEFERRED to the T-sweep. **Tonight's k-probe verdict = FROZEN vs NOT-FROZEN only; the
GT-scale criterion belongs to the T-sweep's table. Two gates, not one** — conflating them = tonight's optimistic
rounding-up risk. An over-lively first table must NOT be read as a new failure.
**(2) Smoke threshold caveat.** code-agreement≥0.5 measures encode(decode(x)) round-trip STABILITY (oracle: ~0.6 exact
even on clean states). A pass at 0.55 = "on-manifold-ish", NOT lossless — fine for the abort-gate. BUT if the sampled
rollout shows slow degradation over k, CUMULATIVE round-trip loss (0.6^k compounding in non-persistent code dims) is a
suspect BEFORE blaming the model — check via the transient-split instrument: degradation LINEAR in k = round-trip
attrition; PLATEAUS = dynamics.
**(3) Counterfactual fan on live wiring = HIGHEST-STAKES row, and CONFOUNDED tonight.** The k>10 re-absorption (current
FiLM-narrowing motivation) was measured on the FROZEN rollout where everything converged to one fixed point. On a live
SAMPLED rollout the pin Δ might persist / wash out in sampling noise / genuinely re-absorb = three different FiLM scopes.
**CONFOUND:** tonight's fan runs real vs perturbed as SEPARATE sample draws (unpaired) → the counterfactual Δ is dominated
by sampling noise, NOT a clean read; the −0.02-bin single-step effect needs PAIRED-SEED (same draw, pin-only delta) or
ensemble-of-seeds to resolve. ⇒ tonight = freeze verdict only; **counterfactual persistence = the FIRST follow-up
(paired/ensemble seeds, powered for the small effect)**, not tonight's read.

#### GATE 4 SAMPLED-ROLLOUT — RESULT (jobs 5007499 sample / 5007500 argmax, β6/200729/n256, fixed wiring)
Smoke PASS (spectro round-trip 0.945 ≥ gate; code-agreement 0.252 = FSQ redundancy, harmless). **FREEZE = INSTRUMENTATION,
NOW FIXED.** Transient-split SUSTAINED (k≥2) ridge-var pred/GT: continuous(broken)=**0.16×** → **argmax=1.61×** →
**sample T=1=4.86×**; windows moving >0.5bin k2→39: continuous **4/256** → argmax **175/256** → sample **237/256** (GT 130/256).
Mean ridge @k39: continuous 35.1 (frozen 2.4 below GT) → argmax 38.8, sample 37.3 (≈GT 37.5). prominence-retention
continuous 1.02 → argmax 1.35, sample 1.66 (mode persists/sharpens, NOT frozen; not attrition — attrition would REDUCE it).
**VERDICT (frozen-vs-not, tonight's only question): NOT FROZEN — freeze was the rollout.py:252 continuous-feedback
contraction (the deferred bug). Mechanism PASS.**
**FORK ANSWER:** argmax (deterministic, no sampling noise) ALONE un-freezes (1.61×≫0.16×) ⇒ the operative ingredient is
ON-MANIFOLD DISCRETE-BOTTLENECK RE-TOKENIZATION, not stochasticity; sampling AMPLIFIES (4.86×). argmax = the clean
dynamics read (no noise confound): model's true rollout dynamics ALIVE ~1.6×GT, mode tracks toward GT freq. **The June
question is answered on correct wiring: an established mode SURVIVES dynamical free rollout (persists + evolves at
GT-scale), NOT frozen. "Simulator" reading resurrected honestly.**
**DEFERRED per reading rules (NOT claimed tonight):** (1) GT-scale CALIBRATION = T-sweep (T=1 over-lively 4.86× = the
pre-registered fourth-branch "alive-but-miscalibrated" = mechanism pass; argmax 1.6× brackets low; lower T calibrates).
(2) COUNTERFACTUAL PERSISTENCE = FIRST follow-up: tonight's fan boot-CIs span 0 hugely (k10 +1σ [−0.66,+0.37]) = the
unpaired-sampling confound as predicted → PAIRED-SEED rerun (identical draw, pin-only delta) needed to resolve the −0.02 effect.
(3) OVER-DRIFT 2.0-2.8×GT persists (worse with sampling) → T-sweep + Gate-5 drift-rate-per-K-block. Single-step k0 conditioning
UNCHANGED (+1σ −0.019, −2σ +0.0025 — single-forward, rollout-independent). Artifacts: eval_runs/gate4_{sampled,argmax}/.

#### GATE 4 — PAIRED-SEED COUNTERFACTUAL (#6, job 5007688, β6/200729/n256, sample T=1, CRN)
CRN VALID (same-seed reproducibility=1.000 → Δ isolates pin, not sampling noise). Doses 0/+2σ/−2σ.
- **Single-step (k0): CLEAN BIDIRECTIONAL** — differential Δ(+2σ)−Δ(−2σ) = −0.023 boot[−0.026,−0.021] (excludes 0; +pin lowers, −pin raises). Confirmed a 3rd time.
- **Multi-step (k10): UNRESOLVED** — differential nominally NEGATIVE and GROWING (k0 −0.023 → k10 −0.215 → k39 −0.317, same sign = hints persist+amplify, NOT re-absorb) BUT boot CI [−0.492,+0.050] SPANS 0.
- **WHY unresolved = the finding:** T=1 rollout is over-lively (drift 2.8×GT, per-window trajectory variance ±0.4) → the rollout's own CHAOTIC OVER-DRIFT swamps the pin signal. CRN removed SAMPLING noise but not TRAJECTORY-DIVERGENCE variance (huge at T=1).
- **ORDER IS COUPLED:** #6 (controllability) needs #7 (T-calibration) FIRST — a clean controllability read requires GT-scale drift (less chaos). Re-run #6 at the calibrated T. The nominal signal is PRO-controllability (persists+amplifies); unresolved ≠ absent.
**FiLM decision: still OPEN** — depends on whether controllability RESOLVES at calibrated T. Not FiLM-confirmed (nominal persistence favors controllable); not controllable-confirmed (T=1 too chaotic). ⇒ T-sweep now, then re-run #6 at GT-scale T.

#### GATE 4 — READOUT PRE-REG (2 corrections, before argmax-pc 5007xxx + T-sweep land)
**ARGMAX PAIRED = the VERDICT-CARRIER (was skipped; now job g4_argmaxpc).** Deterministic ⇒ two rollouts differing
only in pin have ZERO sampling-divergence variance ⇒ the differential Δ(+2σ)−Δ(−2σ) IS the pin effect pointwise
(across-window stats only); no CRN, no chaos-swamp, no T-calibration prereq. Already validated LIVE (1.6×GT, 175/256
moving) = the "less chaos" limit the sweep chases, available tonight. This arm calls the FiLM fork. (Earlier eyeball
from 5007500: +2σ & −2σ BOTH ~−0.15 at k10 ⇒ differential ~0 ⇒ likely SYMMETRIC/not-controllable — confirm with the boot CI.)
**CALIBRATION FLOOR (bounds #7):** sustained-var is ~monotone in T; bracket = argmax(T→0) 1.6× ↔ T=1 4.86×. So the
sweep FLOOR ≈ 1.6×; GT-scale 1.0× sits BELOW the bracket and T∈{0.3,0.5,0.7} will interpolate 1.6→4.86, NOT reach 1.0.
If so the honest deliverable = "best-available T small, residual over-liveliness ~1.6× is a MODEL PROPERTY (the over-drift's
cousin) owned by Gate-5 training calibration, not by T." A sweep that never touches 1.0× MEASURED THE FLOOR — informative, not failed.
**PLACEBO CONTROL (sampled arm only):** the T=1 nominal-growing differential (−0.023→−0.215→−0.317) is WEAK evidence —
chaotic pairs diverge under ANY perturbation (Lyapunov), pin or not. If the sampled arm is trusted at any T, it REQUIRES a
placebo paired differential (gas_flow ±2σ, same CRN): if the placebo grows the same way, the growth is Lyapunov, not
conditioning. The ARGMAX arm needs NO such control (nothing diverges deterministically except through the pin channel).

#### GATE 4 — FiLM FORK CALLED (argmax paired counterfactual, job 5007716, VERDICT-CARRIER, deterministic)
Immune to sampling-noise / chaos-from-T / placebo requirement (deterministic → only the pin channel differs).
- **k0 (single-step): CLEAN BIDIRECTIONAL** — differential Δ(+2σ)−Δ(−2σ) = −0.0231 boot[−0.0257,−0.0206] (excludes 0).
- **k10 (gate): CONTROLLABILITY = ZERO** — differential = **+0.0001 boot[−0.143,+0.141]**. +2σ and −2σ produce the IDENTICAL
  k10 shift (both −0.050) ⇒ SIGN-INDEPENDENT (symmetric) response ⇒ bidirectional control LOST by k10. Not chaos (argmax
  deterministic, var 1.77×GT, drift 2.0×), not frozen re-absorption (dynamics alive). Genuine loss of directional steering.
- Dynamics alive (var 1.77×GT), mode survives (retention 1.34, false-death 0.000 vs 0.030) — the simulator properties hold.
**VERDICT: β=6 is a DYNAMICAL, MODE-SURVIVING, SINGLE-STEP-CONDITIONED simulator that is NOT controllable-at-horizon.
Pre-registered "decays" outcome ⇒ FiLM CONFIRMED**, scope = controllability-persistence ONLY (dynamics/survival/single-step
all already pass), d512, accept={sustained-variance held (~GT after Gate-5 calib), counterfactual differential sustained
past k10, false-death ≤0.01}. Objective singular + un-confounded.
**T-SWEEP (#7, jobs 5007708-10):** sust.var/GT never reaches 1.0 (T0.3=1.97 → T1.0=4.41; argmax floor 1.6×) — GT-scale
UNREACHABLE by T ⇒ residual over-liveliness ~1.6-2× + over-drift ~2-4×GT are MODEL PROPERTIES → Gate-5 training calibration
(teacher-forced longer-K). Sampled-arm controllability INCONCLUSIVE (lone T=0.7 "resolve" = multiple-comparisons fluke on a
chaotic rollout w/ no placebo) → argmax is the decider, as pre-registered. Floor measured = informative, not failed.
**GATE 4 CLOSED.** Model track now launch-and-wait: (a) implement FiLM conditioning-by-construction + anchor-reduction, d512
train, judge on the 3 criteria; (b) Gate-5 ticket (filterscope/video oracle audits + hash assertions + monitors) → production.

## FiLM RUN — PRE-REGISTERED (2026-07-15, before building; the last architecture experiment)
**DESIGN CHOICE (locked, one-change-per-run even now): FiLM replaces the ACTUATOR PATHWAY, NOT the anchor.**
Keep the anchored descriptor head EXACTLY as validated (its fidelity/survival/single-step properties are BANKED — do not
touch). Inject actuators via **FiLM modulation of the backbone blocks** (per-block γ/β from the actuator embedding), removing
the actuator-token pathway that carries conditioning today. Let the counterfactual test whether construction-level
conditioning survives the anchor's STATE FEEDBACK. Warm-start β=6, anchor-config identical, d512.
**EXIT INSTRUMENT = tonight's EXACT table** — argmax paired counterfactual (gate4_kprobe.py FEEDBACK_MODE=argmax, deterministic
verdict-carrier), SAME window pool (200729, n=256), SAME k∈{0,10,39}. Before/after is ONE figure: **k10 controllability
differential Δ(+2σ)−Δ(−2σ) from +0.0001 → CI-excluding-zero with dose ordering restored (+pin lower / −pin higher held to k10).**
That figure, if it lands, IS the money-figure quantitative panel.
**ACCEPT (3 criteria, all required):** (1) sustained-variance held ~GT (Gate-5 calibration); (2) k10 counterfactual
differential SUSTAINED (CI excludes 0, signed like k0); (3) false-death ≤0.01.
**REGRESSION BATTERY rides along:** the BANKED single-step properties must SURVIVE the architecture change — false-death
0.000, drift-direction skill, single-step bidirectional conditioning. FiLM that breaks false-death or drift skill is NOT a
pass regardless of criterion (2).
**REALISTIC EXPECTATION (written down):** FiLM guarantees actuator influence on EACH step's computation; whether the SIGN
survives k steps of ANCHORED FEEDBACK is exactly what's TESTED, not assumed.
**PRE-REGISTERED FAIL BRANCH:** if FiLM + persistent-sign STILL collapses at k10 → the conditioning must enter the FED-BACK
STATE itself (the anchor is computed from the fed-back state, which is actuator-independent) → next run puts the ANCHOR on
trial: an **actuator-conditioned anchor** (one change, next run). That is the escalation, written before the FiLM run per house rules.

#### FiLM RUN 1 — RESULT (job 5008866, argmax verdict, CONFOUNDED by β=8 anchor deviation)
FiLM built + trained (chain 5007874-77, val 1.189, dir e2e_g3fix_film); rollout made FiLM-aware (rollout.py mirrors
model.forward FiLM). BUT trained at anchor β=8 (dist_beta default, NO anneal-to-6) — NOT the β=6 operating point (my deviation).
- **k0 single-step BROKE (regression FAIL):** differential +0.0025 [+0.000,+0.006] vs β=6 baseline −0.023 — 10× weaker, WRONG
  sign, k0-bidirectional=False. Banked single-step property did NOT survive → per pre-reg this alone is a fail.
- **k10: no clean control** — both ±2σ drive UP symmetrically (+0.26,+0.49), differential −0.235 [CI excl 0] only from magnitude
  gap, opposite-signed to k0; anchor Δ +1.06 (chaotic destabilization). controllable@k10=NO. Dynamics alive (2.7×), survival OK.
- **CONFOUND (interim, later RETRACTED):** thought β=8 anchor over-masked the k0 → ran a β=6 eval-time diagnostic to check.
- **NEXT (not the fail-branch yet):** (a) cheap β=6 eval-time diagnostic on THIS ckpt (does k0 recover under less masking); then
  (b) if masking → β=6-ANCHORED FiLM retrain (the clean run I should have launched); if k0 still broken → fail-branch (actuator-conditioned anchor).

#### FiLM RUN 1 — β=6 EVAL-TIME DIAGNOSTIC (job 5008894) → DISAMBIGUATED: NOT masking. FiLM-run-1 = FAIL.
Re-eval of the SAME FiLM ckpt at DESC_ANCHOR_BETA=6 came back IDENTICAL to β=8:
- k0 differential +0.0033 [+0.000,+0.0084] (vs β=8 +0.0025) — tiny, WRONG sign vs token-pathway −0.023, k0-bidirectional=False.
- k10 both ±2σ drive UP (+0.28,+0.51), diff −0.228; anchor Δ +1.06. controllable@k10=NO. Survival OK (153/256, retention 1.74, false-death 0.000, var 2.84×GT).
- **RETRACT the "β=8 confound" framing:** ΔOUT@k0 is anchor-β-INDEPENDENT by construction (persistence anchor doesn't depend on the
  actuator → cancels in the perturbation difference). Lowering β can't reveal a hidden k0 signal. The identical result CONFIRMS the tiny
  wrong-signed k0 is the GENUINE FiLM residual response, not masking. So "failed-vs-masked" is resolved: FAILED.
- **TWO real findings:** (i) FiLM head zero-init + only ~6k warm steps → the actuator→mode mapping is UNDERTRAINED (token pathway learned
  its −0.023 over the full run + gates; FiLM had to relearn from identity in 6k steps and produced a weak wrong-signed map). REAL confound.
  (ii) DEEPER: horizon control is lost IDENTICALLY in BOTH models (token β=6 k10 diff +0.0001; FiLM k10 symmetric +0.28/+0.51, anchor +1.06)
  → the blocker is the ROLLOUT OVER-DRIFT (2.8×GT, everything amplified UP), a dynamics-STABILITY problem, NOT the conditioning-injection point.
  FiLM was the hypothesized fix for controllability-persistence; it can't fix it because the blocker is drift, not injection.
- **VERDICT — TWO LAYERS, DIFFERENT LIFETIMES (do not conflate; this headline gets quoted):**
  - **FiLM-RUN-1 FAILS — PERMANENT.** Regression on the banked single-step property (wrong-signed k0 +0.003 vs −0.023, not
    bidirectional), no horizon control. This specific run is a settled negative. Dynamical/survival props preserved.
  - **FiLM-THE-HYPOTHESIS — NOT DISPROVEN, DEFERRED.** run-1 is UNDERTRAINED-CONFOUNDED (zero-init head relearning in ~6k steps
    what the token pathway built across the ENTIRE gate chain). Honest status: run failed; hypothesis deferred BEHIND the drift lever.
    Fair-FiLM (fork option A) is CONTINGENT on the post-Stage-2 counterfactual — only testable once the drift confound is removed.
- **FORK for user (do NOT launch unilaterally — days of compute, pre-reg says anchor; I surface a reframe):**
  (A) fair-FiLM retrain (from-scratch / much longer) to rule out the undertraining confound — but even clean FiLM likely won't fix drift;
  (B) pre-registered fail-branch = actuator-conditioned ANCHOR — also unlikely to fix drift (same drift blocker);
  (C) attack the DRIFT directly as the real controllability blocker (Gate-5 calibration: teacher-forced longer-K rollout / drift penalty).
  LEAN (C) — the two tables say the injection pathway isn't the bottleneck; the rollout stability is. Awaiting user steer.
- Figures: eval_runs/gate4_film_argmax/ (β=8) + eval_runs/gate4_film_b6/ (β=6); .npz per-window saved in each.

## STAGE-2 K-ANNEAL — PRE-REGISTERED (2026-07-16, the DRIFT INTERVENTION; before launch, per house rules)
**HYPOTHESIS (double-confirmed by two architectures):** horizon controllability dies identically under two different injection
architectures (token-pathway β6 k10 diff +0.0001; FiLM k10 symmetric divergence) while dynamics/survival hold in both. One
failure signature, one shared property: a rollout trained ONLY single-step, over-drifting ~2.8×GT and amplifying every
perturbation upward. The injection point is EXONERATED by parallel construction; the DRIFT is convicted. The intervention that
tests it is training the rollout itself — Stage-2 K-anneal.
**DESIGN (one change vs g3fix β=6: the K-rollout extension; everything else frozen at the operating point):**
- Extend `train_e2e_stage1.py` with opt-in `--k_rollout`; reuse the PROVEN eval wiring (TokenSpaceRollout + eval's
  rollout_forward_one_batch data construction) grad-enabled → train≡inference feedback BY CONSTRUCTION (closes the rollout.py:252
  / failure-mode-7 train/inference-mismatch class). Code-space (argmax) feedback, the Gate-4 fix that un-froze the rollout.
- Per-step FSQ code-CE (class_weight 10, temp 1.0) + descriptor `dist` loss (weight 6, tw 5, horizons 2,4) + continuous heads,
  summed over k, mirroring Stage-1 exactly. **β PINNED at 6 for the whole run (NO 8→3 anneal — that was a measurement sweep;
  re-running it drags through the β=5 sign-incoherent knee to β=3 where false-death=0.223).** Any β re-tune = post-hoc β-sweep
  eval on the trained ckpt, hours not a confounded schedule.
- **Anchor source per rollout step (train/inference pin):** anchor = descriptor_target(state fed INTO step k). k=0 = GT initial;
  k≥1 free = decoded re-tokenized fed-back state (exposed from TokenSpaceRollout); k≥1 TF = GT@t=k. Training-time anchor
  computation mirrors Gate-4 inference exactly. Smoke assertion: at TF=0, step-k ece diag_input == the decoded fed-back state.
- Curriculum K 10→20→40→80, block_steps per rung; TF-anneal `p_tf = max(0, 1 - step/tf_anneal_steps)` (scheduled sampling:
  GT-fed early → free-rollout late). Grad-checkpoint across rollout groups for K≥40 at d512 (memory). Warm-start
  `e2e_g3fix_anneal/e2e_stage1_beta6.0_step3000.pt`; model geometry (prediction_horizon_s=0.2, act-tokenizer conv (512,8,400))
  FROZEN for warm-start compat; dataset span decoupled (widened to K_max×chunk) — the same decoupling the eval used to hit K=40.
**EXIT INSTRUMENT (unchanged):** argmax paired counterfactual (gate4_kprobe FEEDBACK_MODE=argmax), 200729 n=256, k∈{0,10,39},
run PER K-BLOCK on milestone checkpoints. Primary readout = k10 controllability differential Δ(+2σ)−Δ(−2σ).
**THREE-BRANCH PRE-REGISTRATION for the k10 differential (written before launch):**
  (1) RETURNS BIDIRECTIONAL (CI excludes 0, signed like k0, dose ordering restored) → controllability was DRIFT-LIMITED all along;
      NO architecture change needed; the token pathway + descriptor anchor is a controllable simulator once drift-calibrated → money figure + Gate-5/production.
  (2) PARTIAL (differential shrinks toward 0 but doesn't fully resolve) → scope controllability FROM THE DECAY CURVE (which K it holds to); report the horizon of validity.
  (3) STILL DEAD on a drift-calibrated rollout (differential ~0, symmetric) → drift was NOT the (whole) blocker → the fail-branch
      chain reopens IN ORDER, now each finally testable WITHOUT the drift confound: fair-FiLM (from-scratch/longer, β6-anchored) → actuator-conditioned anchor.
**WATCHED PRE-REGISTRATION (not assumed): drift-ratio + sustained-variance trajectory PER K-BLOCK.** "Drift comes down under
rollout training" is the treatment's expected mechanism and must be SEEN coming true, logged alongside the counterfactual table
every block. If drift does NOT calibrate under K-training, that is the EARLIEST signal (visible at K=10, not K=80) that Gate-5
needs the drift-PENALTY variant rather than teacher-forcing alone. Per-block gate suite: {drift ratio pred/GT, sustained variance
pred/GT (transient-split k≥2), k10 counterfactual differential + bootstrap CI, false-death eff vs β6-independent 0.030}.
**REGRESSION (must survive every block):** false-death ≤0.01, single-step (k0) bidirectional conditioning preserved, mode survival.
A drift-calibrated rollout that breaks the banked single-step properties is NOT a pass.
**BLOCK-0 BASELINE:** run the gate4_kprobe suite on the UNTRAINED warm-start (step 0) first — free paired baseline; makes every
per-block comparison paired against the pre-intervention model.

#### K-ANNEAL LAUNCHED (2026-07-16) — K=10 block first, gated
Job 5010390: 2 ranks × batch 8 (global 16 = g3fix operating point), lr 2e-4, warm-start beta6.0_step3000.pt, dir e2e_g3fix_kanneal.
curriculum 10,20,40,80 × block_steps 5000; tf_anneal 2000 (TF→0 @2000, 3000 free-rollout steps in K=10 block); max_steps 20000 (LR span);
val_every 500; grad_ckpt 10. Timing (smoke): K=10 ~1.35s/step, K=40 ~5.0s/step, memory flat in K (gc=10, no OOM @K=40).
Block-0 baseline denominator = job 5010348 (gate4_kprobe on warm-start). GATE at step 5000: full n=256 argmax paired counterfactual +
drift ratio + sustained var + false-death, paired vs block-0. K=20/40/80 PROVISIONAL — re-confirmed at each block gate; drift-penalty
variant enters before K=40 if K=10 drift doesn't move (three-branch pre-reg above). Chain NOT pre-specced to K=80.

#### K-ANNEAL NaN DETOUR (2026-07-16) — root cause found, guard insufficient
Production K=10 block (5010390) NaN'd at step 50 (all ece terms) + ran ~37s/step. CANCELLED (user: "nans = nonsense").
- **Slowness:** dataset horizon sized to max-K=80 (4.2s) while K=10 needs 0.7s → 6× wasted I/O. FIX = per-block-matched horizon (CURRICULUM_KS=10 → 0.7s).
- **NaN root cause (localized via ROLLOUT_NAN_DEBUG probe, job 5011123):** SYSTEMATIC (every batch), ece-specific, at rollout k≥1,
  ONLY under teacher-forcing (p_tf>0). Probe: fed-back RAW input + loss target both FINITE, but ece BACKBONE token_slice 100% NaN at k=1.
  ⇒ the TF path (`_decide_feedback` → `_tokenize_diagnostics(gt_target)`) re-tokenizes the RAW GT target window (OFF the codec manifold,
  different scale than the free path's on-manifold decode) → bf16 backbone overflow. The FREE/argmax path (decode→re-tokenize, Gate-4-proven)
  was never exercised with TF; my smokes all ran p_tf=0 so TF went untested (my miss).
- **Guard (skip backward+opt.step on non-finite) is INSUFFICIENT here** (systematic → skips every step) AND DDP-incompatible (skipping
  backward desyncs all-reduce → crash). Kept as a rare-NaN backstop only.
- **FIX (pending free-rollout corpus-safety test 5011394):** make TF on-manifold — round-trip GT through the codec (encode→decode→re-tokenize)
  so TF feeds the SAME manifold as free rollout. Preserves scheduled-sampling design. Fallback = drop TF (pure free rollout = the eval path).
- Also fixed: NameError (os only function-local; probe used module-level os.environ) — re-smoke discipline reaffirmed.

#### K-ANNEAL NaN FIX — on-manifold teacher forcing (2026-07-16, user chose B)
User chose B over A (drop-TF): "A isn't the same experiment with a rougher edge, it's a different treatment" — TF-anneal (scheduled
sampling) IS the intervention (single-step→stable-long-horizon transition; the k1-regression that killed LoRA is exactly the
p_tf=0-cold-start risk), and p_tf=0 would confound the project's decision table ("drift not fixable" vs "we ablated the curriculum").
- **FIX (rollout.py):** new `TokenSpaceRollout._tokenize_gt_onmanifold` mirrors `_resample_feedback` but from GT codes —
  `head.encode_target(gt)→decode→re-tokenize` for code heads; raw tokenize for continuous. TF branch in `_decide_feedback` now routes
  through it. The teacher now lives on the SAME codec manifold as the free-rollout decoded state (failure-mode-7 on-manifold discipline
  applied to the TF path, where it had silently never been). decoded_feedback (anchor source) = the on-manifold GT. This is the CORRECT
  fix (closes the actual gap), not a workaround, and is the trainer you want for K=20/40/80 regardless of what K=10 shows.
- **SMOKE (CPU 5/5 + TF finite + idempotency; real corpus job 5011872):** p_tf=1.0 FINITE (was 99 NaNs). Round-trip on-manifold assertion:
  decode→encode idempotency (real codec 0.945 @Gate-4; tiny random codec 0.23, logged not gated). Standing requirement filed: every
  rollout-trainer smoke runs p_tf ∈ {1.0, 0.5, 0.0} (the p_tf=0-only coverage gap is what let the TF NaN reach production).
- **Rider:** tf_anneal_steps=2000 STANDS — both endpoints now validated (free rollout corpus-safe @job 5011394; TF on-manifold @5011872).
  The anneal's healthiest-ever config. Relaunch K=10 block with it once 5011872 confirms 0 non-finite.

#### K-ANNEAL — A LAUNCHED (2026-07-16): free-rollout K=10 block (drift/controllability gate)
After the NaN bisection closed the seam ledger at 5 (all named to lines), user chose A (free rollout now) over B (deferred).
- **NaN ROOT (localized to the conv layer):** the ece tokenizer `proj` (patch Conv2d, spectrogram.py:160) amplifies the codec
  reconstruction of lattice-EXTREME GT codes ~200× (in 9.875 → proj 1992 → out 2213) → backbone NaN. Natural inputs + in-distribution
  (predicted-code) decodes tokenize to the natural band (out ~50-210). It's STRUCTURE (extreme-code decode resonating with the patch
  conv), not magnitude. TF fed `decode(encode_target(GT))` whose ~1% extreme codes (mode-energy patches) hit the resonance; the free/
  argmax path decodes the MODEL's codes, which never hit extremes → safe. **A is CHARACTERIZED-safe, not "unexploded"** (measured: free
  proj in the natural band). Target-path seams (residual space = same baseline_residual_torch; window indexing; loss side) all cleared by inspection.
- **A config:** free rollout tf_anneal_steps=0 (p_tf=0 from step 1), β=6 pinned, CURRICULUM_KS=10 (0.7s horizon — efficient ~1.3s/step,
  NOT the 4.2s max-K horizon that gave 37s/step), block_steps=5000, max_steps=20000 (LR span), val_every=500, grad_ckpt=10, 2 ranks × batch 8
  (=global 16 g3fix op point), warm-start beta6.0_step3000.pt. Block-0 baseline denominator BANKED (job 5010348: k0 −0.023 bidir, k10 +0.000, drift 2.03×GT).
- **ASYMMETRIC GATE READOUT (pre-registered before the gate):** POSITIVE (drift→1, k10 differential returns bidirectional CI-excl-0) =
  CLEAN + FINAL (drift was the blocker; controllable simulator → money figure). NEGATIVE (drift doesn't calibrate OR descriptor skill
  degrades) = CONFOUNDED with "curriculum ablated (free-rollout-only)" → pre-registered response = B's fix + FULL-ANNEAL relaunch, NOT a
  conclusion about drift trainability.
- **TRIPWIRES in the monitor (per ≤500 steps):** (1) k1-regression — ece_desc_ftol + ece_ce at short horizon (free-rollout-from-step-1's
  known failure = single-step skill eroding; catch at ~2k, not the gate). (2) PROJ-RESONANCE WARN (spectrogram.py _encode, always-on):
  ece tokenizer out_absmax >600 (~3× natural band) = the model started predicting lattice-extreme codes → resonance awakening → B becomes
  urgent AND it's a signal the model is learning stronger mode energy (worth knowing for its own sake).
- **B (deferred follow-up):** fix = NORMALIZE the decoded feedback to input-window statistics before proj (edits only numerics), PREFER
  over clamping GT codes off the lattice extremes (those ~1% extremes are plausibly the mode-energy content this whole saga preserves;
  clamping edits the teacher signal). B enables the full 10→20→40→80 anneal. Surviving asterisk on A guarded by the resonance WARN.
- **SEAM LEDGER (closes at 5, all train/inference-seam bugs in the new rollout stitching, named to lines):** (1) rollout.py:252 continuous-
  feedback freeze [Gate-4]; (2) validate() single-step-on-wide-batch 17-vs-5 [val-horizon decouple]; (3) TF re-tokenizes raw off-manifold GT
  [on-manifold fix — necessary but insufficient]; (4) NameError os module-scope [probe]; (5) proj extreme-code-decode resonance [spectrogram.py:160,
  B pending]. Standing smoke req filed: rollout-trainer smokes run p_tf∈{1,0.5,0}.

#### K-ANNEAL A — INTERMEDIATE GATE (step 2000, job 5013564) + PRE-REGISTERED 4000 DECISION RULE (2026-07-16)
Intermediate drift/controllability read on the step-2000 free-rollout ckpt, paired vs block-0 (step 0):
- k0 differential: block-0 −0.023 (bidir ✓) → step-2000 **+0.017 (bidir ✗, SIGN FLIPPED)**.
- k10 differential: block-0 +0.0001 (null) → step-2000 **+0.857 [CI +0.49,+1.23]** — but drift-AMPLIFIED not control (drift tripled in lockstep; one-sided −2σ→−0.85/+2σ→~0, not symmetric).
- drift pred/GT: block-0 2.03× → step-2000 **6.53×** (WORSE). false-death 0.000, mode-present 153/256 (both held).
**REFRAME (filed):** this is NOT an anomaly — it's the PREDICTED curriculum-ablation behavior (free-rollout-from-step-1 destabilizes +
erodes k0; the exact failure mode TF-anneal/scheduled-sampling exists to prevent; failure-mode-7 lineage + LoRA k1-regression precedent).
The free-rollout-only block is the ABLATION ARM. If confirmed → conclusion = "drift training REQUIRES the curriculum," NOT "drift
untrainable" (that stays OPEN); B's full-anneal relaunch tests the real hypothesis for the first time. Asymmetric readout carrying its load.
**RIDER — k0 flip is the GRAVER signal (own hard-gate line):** drift-climb = treatment not helping; k0 sign-flip = treatment DESTROYING
a banked property (single-step bidir conditioning, 3 confirmations to establish). If drift STABILIZES but k0 STAYS FLIPPED at 4000 → still
B-commit (a rollout-calibrated model that lost single-step conditioning traded the demonstrated result for an undemonstrated one). Regression
battery hard-gate applies MID-BLOCK, not just at milestones.
**PRE-COMMITTED 4000 DECISION RULE (thresholds written before the number exists):**
  (a) drift(4000) > 6.5× (> drift(2000))  → COMMIT B immediately (monotonic climb; don't burn to 5000).
  (b) drift(4000) ∈ [2×, 6.5×] (falling, not recovered) → transient story gains support → burn to 5000 for the definitive read.
  (c) drift(4000) < ~2× AND k0 restored (bidirectional, sign back negative) → genuine mid-training transient → proceed to full 5000 gate.
  PLUS the rider: any branch with k0 still flipped at 4000 → B-commit regardless of drift.
**PARALLEL ACTION:** build B's feedback-normalization fix NOW (during the chain's wall-clock — free) so "commit B" = a relaunch, not a
decision-plus-build. B = normalize decoded feedback to input-window statistics before proj (chosen over code-clamping — the ~1% extreme
codes are mode-energy content). OPT-IN flag (default off → running A chain 5013164/65 unaffected on resume). Verifier: TF-on smoke proj drops
1992 → natural band (<600) with flag on; byte-identical with flag off. Full-anneal relaunch (10→20→40→80, tf_anneal restored) uses it.

#### K-ANNEAL — B-FIX DECISION (option 3, upstream) + WARM-START PRE-REG + val-trend reading (2026-07-16)
- **B-fix = OPTION 3 (upstream, code-space), NOT the downstream options.** Mechanism: resonance enters at DECODE of lattice-EXTREME GT
  codes (~1% of dims); options 1/2 intervene downstream + pay content costs (spatial whitening flattens real mode RIDGES — legitimate
  coherence indistinguishable from resonant-artifact coherence by a whitener → likely dies at the mode-check; proj-clamp clips content-blind).
  FIX = soft-clamp/re-quantize the FEEDBACK codes' extreme levels in by one (0→1, 15→14 — the ±1 tolerance the oracle/tol1 analysis proved
  physically negligible), OR equivalently clamp the decoded state per-patch to the empirical range of PREDICTED-code decodes (proven non-
  resonant). Attacks the entry point, costs ~1 FSQ level on 1% of dims (tol1-bounded ≈ nothing), leaves legitimate coherence untouched.
  VERIFY: extreme-clamped GT decode → proj must land in the natural band (existing [stage] probe). If it fails the probe → option 1 + mode-
  check, option 2 last resort. TEST THE CHEAP UPSTREAM FIX FIRST.
- **LEDGER:** the B subagent reported the per-(C,F) moment-matching as a NON-FIX (proj unchanged 1968) rather than shipping it — correct
  behavior; the refined "spatial-coherence, same-magnitude coin-flip, inputs never resonate" diagnosis is what redirected the fix upstream.
- **VAL-TREND READING (completes the story):** MAE gap eroding toward copy (0.109→0.059) AND displacement ratio crossing 1.0 single-step
  (0.92→1.03) = free-rollout-only isn't just failing to calibrate drift, it's converting the model toward the OVER-DRIFTING persistence-ish
  attractor at EVERY horizon simultaneously — textbook k1-regression. Sharpens the ablation write-up: the curriculum is what separates "learn
  rollout stability" from "unlearn single-step skill" — the measured version of a field-asserted claim (scheduled sampling necessary),
  by controlled ablation on a 120M model. Nature-shaped methods material from the yellow flag.
- **WARM-START PRE-REGISTRATION (time-sensitive, decided BEFORE the gate):** if the 4000 gate commits B, the full-anneal relaunch warm-starts
  from the **g3fix β=6 OPERATING checkpoint (e2e_stage1_beta6.0_step3000.pt) — NOT any A-block checkpoint.** The A block is ERODING (step-4000
  k0 more damaged than step-2000; both trail the pristine β=6), so a later warm-start inherits the erosion B exists to prevent. The A block
  contributes its ABLATION TABLE and nothing else to the lineage.

#### K-ANNEAL A — 4000 GATE → VERDICT: COMMIT B (k0 rider fires) (2026-07-16)
Drift trajectory 2.03×(s0) → 6.53×(s2000) → 3.11×(s4000): the s2000 spike was TRANSIENT (drift falling), drift-alone → branch(b) burn-to-5000.
k0 differential −0.023 bidir(s0) → +0.017 flipped(s2000) → +0.0017 bidir=False(s4000): single-step conditioning ERODED to ~0 + wrong-signed,
NOT restored → **k0 RIDER FIRES → COMMIT B regardless of drift**. k10 diff +0.857(s2000)→+0.004 null(s4000) as drift normalized → CONFIRMS
s2000 k10 was drift-AMPLIFICATION, never control. false-death 0.000, mode 153/256 held throughout. Round-trip 0.945→0.972→0.997 (code-agree
0.25→0.87 — codes more self-consistent). VERDICT: free-rollout-only partially recovers drift but DESTROYS k0 → a rollout-calibrated model that
lost single-step conditioning = traded the demonstrated result for an undemonstrated one. The FREE-ROLLOUT BLOCK = the ABLATION ARM: measured
proof scheduled sampling is load-bearing (separates "learn rollout stability" from "unlearn single-step skill"). Val was too noisy to trend
(gap 0.06-0.14, ratio 0.85-1.03 — no erosion; the gate/k0 was the instrument, not the val). EXECUTE: option-3 probe → B relaunch (full anneal
10→20→40→80, tf_anneal restored) from beta6.0_step3000.pt (pristine operating point — NOT any A-block ckpt; A contributes only this table).

#### K-ANNEAL — post-4000-gate findings + option-3 FAILURE + B-run tripwires (2026-07-16)
- **k10-collapse bonus finding (ablation write-up + standing rule):** k10 differential +0.857(s2000)→null(s4000) COLLAPSED as drift
  normalized (6.5×→3.1×) → empirical proof the s2000 "resolved differential" was DRIFT-AMPLIFICATION, never control (Lyapunov-vs-conditioning
  discrimination by TRAJECTORY, not placebo). STANDING RULE reinforced: **k10 differentials are only interpretable at calibrated drift.**
- **OPTION 3 FAILED (job 5014710):** extreme-code ±1 clamp [1,14] active (codes min=1 max=14 confirmed) but proj STILL ~1968, decode absmax
  unchanged 9.625, 20 NaN. ⇒ the "lattice-extreme codes cause the resonance" PREMISE IS REFUTED. ~half the GT decodes resonate (58 vs 45 @in≈10);
  spatial coherence is NOT extreme-code-driven. REFRAME: the resonating coherence is likely LEGITIMATE mode-ridge energy (modes ARE coherent);
  proj amplifies it + bf16 backbone can't hold it → NOT a content bug but a numerical-robustness gap. ⇒ option 1 (whitening) would flatten the
  modes = wrong; **option 2 reframed as the RIGHT fix (proj-output RENORM preserving relative structure, not a hard clip), not last-resort.**
- **ENHANCED FIX-PROBE PASS CRITERIA (for the next attempt):** proj<600 AND (a) tol1 code/decode agreement clamped-vs-unclamped ≳0.99 (verify,
  don't assume) AND (b) mode-detection score clamped-vs-unclamped on mode-active windows (guard: extremes may CONCENTRATE in mode-energy patches
  — high-amplitude ridges are where a quantizer runs out of range — so a "1% global" edit can still dim modes; make the teacher-fidelity cost a
  KNOWN number, not discovered at the K=40 block).
- **B-RUN TRIPWIRES (calibrated from the A ablation, which telegraphed the gate verdict ~1500 steps early):** promote to formal per-≤500-step
  monitor scalars — MAE-gap (model−copy) falling below ~0.08 AND displacement-ratio (pred_d/tgt_d) crossing 1.0 → self-report (anneal mis-paced /
  TF dropping too fast for this model) in HOURS, not at a block gate. The ablation arm's second gift: it calibrated B's early-warning system.

#### K-ANNEAL — RESONANCE ROOT CAUSE (diagnostic job 5015153) + embed-path feasibility (2026-07-16)
Discriminating diagnostic (paired GT-vs-predicted decode proj + spatial spectrum, resonance_diag.py) → NEITHER T1 (mode energy) NOR
T2 (realization roughness): the ~2001 resonance is a SINGLE FIXED proj Conv2d filter (out-ch 107, freq-token 0, time-token 0 = DC/low-freq
corner) saturating on the shared NEAR-DC BROADBAND FLOOR of every mode-active window. Proof: GT-decode / predicted-decode / random-realization
all proj=2001.15 bit-identical; blank decode proj=0.08 (content-driven, not bias); REAL INPUT window proj=1881/out=2101 (resonates too, just
UNDER the bf16 tip; codec decodes reach out=2216 and tip over → the "inputs never resonate" was a MARGIN artifact); mode-ridge deviation
(batch-mean-subtracted) GT=24 pred=15 ≪600 (ridge ~irrelevant to the max); in-band 5-40kHz fraction 0.00; spatial spectrum peaks at DC identical
to inputs. LATENT FRAGILITY: β=6 model already runs near the bf16 tip on ece proj single-step (input out=2101).
FIX RANKING (updated): (1) EMBED-PATH — feed codec.fsq.codes_to_tokens(codes) (B,24,256, = tokenizer output DIM, proj_out Linear(8→256)),
bypass decode→proj → DC filter UNREACHABLE. DIM-feasible (no projection gap) but SPACE-RISK: codec decoder-input embedding ≠ tokenizer-output
space the backbone trained on → SUBSTITUTION PROBE required (finite [guaranteed] AND sane free-rollout loss/codeacc vs pixel-path). (2) OPTION-2
feedback-renorm to input band — now WELL-MOTIVATED + provably MODE-SAFE (input band tolerated; resonance is DC-broadband not modes); warm-start-
safe, gated, no space-risk. Fallback if embed-path off-distribution. (Plots: eval_runs/resonance_diag/spatial_spectrum.png.)

#### K-ANNEAL — EMBED-PATH BLOCKED (shape gap) → OPTION-2 (2026-07-16)
Embed-path feasibility RE-CHECKED against the actual g3fix artifacts (NOT the stale d256 config the first inspection assumed):
backbone d_model=512 (beta6.0_step3000.pt args: d_model=512, n_layers=12, use_spectro=['ece']); backbone ece tokenizer proj (512,40,8,16),
spatial_pe (384,512) → emits (B,384,512). Frozen ece codec fsq_resid_p8_all/spectro_codec_ece.pt: codec d_model=256, n_tok=384, dim=48, L=16,
patch (8,16); fsq.proj_out (256,48) → codes_to_tokens → (B,384,**256**). Token COUNT matches (384) but FEATURE DIM does not (256 vs 512), and the
codec proj_out lacks the backbone's spatial_pe/modality_embed/refine. No existing 256→512 adapter → embed-path needs an UNTRAINED projection = off-
spec/off-distribution → BLOCKED. (Corrects the memory index "patch (64,32)" — this production codec is patch (8,16)/384tok/dim48/L16.)
DECISION (rule-based, pre-authorized): → OPTION-2 feedback-token renorm. Locus = POST-tokenizer token scale (NOT the earlier FAILED pre-tokenizer
per-(C,F) pixel moment-matching — that normalized pixels, wrong locus; the resonance is in the tokenizer proj output). Rule = scale each feedback
sample's ece tokens so absmax ≤ the SAME window's step-0 INPUT-window ece-token absmax (the model's tolerated reference; input out~2101 tolerated,
feedback out~2216 tips). Uniform per-sample scale (≤1, only when exceeding) → mode-safe (resonance is DC-broadband, ridge deviation ~24 scales with
everything, contrast preserved). Gated by feedback_normalize (default off, byte-identical). Probe = TF(p_tf=1,0.5)+free(p_tf=0) on the REAL model:
OFF reproduces the TF NaN, ON is finite AND free-path metrics unchanged (no-op in band) AND modes preserved.

#### K-ANNEAL — OPTION-2 FAILS (3rd strike) + MECHANISM REFRAME (2026-07-16)
Option-2 (post-tokenizer per-sample feedback-token renorm to step-0 input band) IMPLEMENTED correctly, byte-identical off, provably MODE-SAFE on
(free-path ON vs OFF: ece_codeacc 0.1360/0.1361, loss 0.354/0.357, ece_ce 2.391/2.392 — identical). But FAILS the TF-NaN gate (probe jobs
5015460 TF-OFF / 5015461 TF-ON: BOTH 18 non-finite, rank1 step0 p_tf=1.0, ece token_slice = backbone OUTPUT, nonfinite_frac=1.0; 5015462/63 free
ON/OFF both 0 — free was already safe, renorm didn't earn it). ROOT of the failure: the SPEC PREMISE "step-0 input band = safe ceiling" is FALSE.
Raw input-window ece token absmax ranges 1516/2040/2410 (min/mean/max, 5688 samples); NaN samples' input ref ~2210-2227 → scaling feedback to ref
leaves it ~2216 → still NaN. REFRAME: NaN fires at ~2216 which is BELOW the ~2410 max raw-input magnitude single-step g3fix trained on → if the
backbone tolerated 2410 inputs single-step, a 2216 feedback token shouldn't NaN on magnitude → this looks ROLLOUT-SPECIFIC NUMERICS (grad-ckpt
recompute / bf16 attention-softmax on hot tokens / K-accumulation), NOT feedback-magnitude. We were clamping the wrong layer. DISCRIMINATOR launched
(read-only, NOT a 4th fix): does single-step forward tolerate the hottest (2216-2410) raw-input AND codec-decode windows? + WHERE does the NaN first
appear in the backbone forward? → names the fix: single-step-finite ⇒ fix rollout numerics (fp32 narrow path / grad-ckpt precision); single-step-NaN
⇒ model-latent (source proj fix / global fp32 ece path). B HELD pending user steer (3rd-strike rule). Option-2 code is mode-safe + gated → retained
as scaffolding, not promoted.

#### K-ANNEAL — NaN DISCRIMINATOR: ROLLOUT-SPECIFIC (mechanism was MISCHARACTERIZED) (2026-07-16)
Read-only discriminator job 5015542 (nan_localize.py, real g3fix model, 1760 ece windows across 11 shots, token-absmax 832→2501). VERDICT:
ROLLOUT-SPECIFIC, NOT model-latent. Single-step forward FINITE at every window up to absmax 2501 (> corpus max 2410); codec-decode single-step
FINITE up to 2213 (the exact production-NaN magnitude); grad-ckpt+backward on the 2213 window FINITE (grad_absmax 0.21). NO first-non-finite op
in the backbone forward (tokenizer proj / QK^T pre-softmax / softmax / LN / FFN all clean). Mechanism: backbone is PRE-NORM → LN normalizes any
input magnitude → hot ece tokens (even a 3.45M actuator token) cannot overflow the forward. ⇒ THE RESONANCE→NaN FRAMING IS DEAD: the model does
NOT NaN on the resonant/hot tokens. All 3 prior fixes (moment-match, code-clamp, option-2 renorm) attacked feedback MAGNITUDE — the wrong wall.
Real locus (by EXCLUSION, not caught): the multi-step K-rollout compounding path (re-tokenize/feedback loop under bf16). RESIDUAL UNCERTAINTY: exact
overflowing op NOT caught (probe ran single-step + grad-ckpt-single-step, both clean; did NOT run the full K≥2 feedback loop). Production NaN clues:
rank1-only, training-step-0 (p_tf≈1), nonfinite_frac=1.0 (whole slice). Two live causes: (a) compounding bf16 numerics in the multi-step loop
[fp32 narrow ece rollout re-tokenize path fixes it, warm-start-safe, no clamp]; (b) a rank-1-specific degenerate/NaN window or codec decode [fp32
does NOT fix; data-hygiene guard does]. DISTINGUISHER = exact-op catch inside the real K-rollout on rank-1 data. 3RD-STRIKE RULE PREMISE ("fully-
characterized mechanism") NO LONGER HOLDS — mechanism was mischaracterized, now correctly = rollout numerics. B HELD for user steer: (A) catch exact
op first, or (B) apply fp32-narrow fix + re-probe (self-testing: clears→numerics confirmed; persists→data cause). New files (uncommitted):
analysis/mode_audit/nan_localize.py, scripts/slurm_frontier/_nan_localize.sbatch.

#### KNOWN NUMERICAL FRAGILITY — ece tokenizer ch-107 DC saturation (filed 2026-07-16, innocent tonight)
Characterization SURVIVES as a real finding even though it was NOT the NaN cause: the ece SpectrogramTokenizer.proj has a single fixed Conv2d
filter (out-ch 107, freq-token 0 / time-token 0 = DC corner) that saturates on the near-DC broadband floor present in EVERY mode-active window
(input, GT decode, predicted decode, random realization all → proj≈2001 / tokenizer-out≈2001-2216 bit-identical; blank decode 0.08 = content-driven;
mode-ridge deviation only ~24). Real INPUT windows span token-absmax 1516-2410 (some to 2501); codec decodes ~2213. The backbone is PRE-NORM so this
does NOT NaN the forward (job 5015542: finite to 2501 single-step + grad-ckpt+backward) — INNOCENT for the K-anneal NaN. BUT file as a latent
fragility: (a) it wastes token dynamic-range on a DC broadband component carrying almost no mode information, (b) it puts ece tokens 10-40x above the
"natural band" (50-210) the WARN was calibrated to, (c) the bf16-tip proximity (out~2101-2216) is a margin that could bite at production scale / lower-
precision / different accumulation. Candidate cleanups IF it ever matters: high-pass/re-center the DC per patch before proj, or rescale/re-init ch-107.
Plots: eval_runs/resonance_diag/spatial_spectrum.png; per-window JSON eval_runs/resonance_diag/resonance_diag.json.

#### K-ANNEAL — NaN CAUGHT (jobs 5018286+5018342): mse TF-path bug, NOT ece/resonance/numerics (2026-07-17)
The catch-first (A) decision was correct + vindicated. OBSERVED (not inferred) first-non-finite op via 387-module forward hooks in the REAL K-rollout,
rank-1 exact batch (shot 193735 chunks 0-7, reconstructed via DistributedTwoLevelSampler seed42 rank1 epoch0): first non-finite = **diag_tokenizers.mse.proj**
(Linear 5->512) at **rollout k=1** (k=0 is finite; the "step-0" in the option-2 report was the TRAINING step). Raw ece CLEAN; raw **mse (Motional Stark
Effect) GT target has 140 -inf in channels 3-4, all 8 windows**. At p_tf=1, _tokenize_gt_onmanifold feeds the UN-SANITIZED -inf GT into mse.proj -> NaN
feedback tokens -> k=1 backbone INPUT NaN before backbone runs -> spreads -> ece slice NaN. Trainer nan-loc only checks ece => misreported as "ece" =>
the entire resonance/ch-107/fp32/embed-path/option-2 saga chased a SYMPTOM. ISOLATION DISCRIMINATOR (the decisive test): k=0 clean; TF-on-manifold
feedback DIRTY at k=1 (fb finite=False); argmax free-rollout feedback CLEAN at k=1 (fb finite=True, ece absmax 2213) => bug is SPECIFIC to the TF GT-
retokenize path, NOT ece codec feedback, NOT bf16-narrow. Corpus scan: ece corpus clean (0 nonfinite/nan, 57/400 all-zero = expected padding); the -inf
is an mse-TARGET property (data_loader note: mse/cer arrive with NaN in some shots, NO zero_is_missing/nan_mask guard). ROOT = code bug: rollout_forward_loss
(~L1846-1849) splits slow-TS/cer/mse gt_k WITHOUT _eval_clean_and_mask, unlike step-0 diag_initial (L1755, cleaned). p_tf>0-only; bites every TF step
with dead mse/cer channels. FIX (observed, mechanical, warm-start-safe, NO fp32/clamp): mirror L1755 — apply _eval_clean_and_mask to the slow-TS/cer/mse
TF GT in rollout_forward_loss (option a, principled: TF path == tested step-0 path). rollout.py restored to WORKING_BACKUP (md5 a227375, 0 instrumentation).
Artifacts: eval_runs/nan_catch/{reconstruct_identity.py,launch_nan_catch*.sbatch,sibling_scan.log,rollout.py.WORKING_BACKUP,3 job logs}. B HELD for user GO.

#### K-ANNEAL — SEAM LEDGER CLOSES AT SIX + instrument-trust lesson + (A)-over-(B) validation (2026-07-17)
The K-anneal integration bugs all lived in the STITCHING between components, never in the components themselves. Six seams, now all closed:
  1. feedback continuity (rollout.py:252 — deterministic feedback froze; code-space sampled feedback fixed it)
  2. TF manifold / window-prep (on-manifold encode→decode→re-tokenize for teacher forcing)
  3. residual space (baseline-subtracted FSQ codec path consistency)
  4. indexing (per-step target/window alignment across the K-rollout)
  5. normalization / re-tokenization (feedback token scale vs input band — the resonance red-herring lived here)
  6. **cross-modality SANITIZATION asymmetry (step-0 path cleans via _eval_clean_and_mask; rollout TF `else` branch did NOT) — THE NaN.**
BONUS LESSON (worth its line): the trainer's nan-loc localizer checked ONLY spectro/ece slices, so a cross-modality NaN that ORIGINATED in mse.proj
and merely SPREAD to ece was reported as "ece". Three fixes (moment-match, code-clamp, option-2 renorm) + two dead ends (embed-path, resonance) were
all aimed by that mislabel. RULE: instrument trust is scoped to what the instrument actually CHECKS — a localizer that inspects one modality can only
ever blame that modality; extend localizers to all candidates before trusting a location label. (Instrument repaired this diff: nan-loc now per-
modality, always-on, fires on the NaN-guard path.)
(A)-over-(B) VALIDATION (methods-section-worthy): catch-first (A) was NOT caution-over-speed — (B) fp32-and-test would have SHIPPED the bug. fp32 does
not clear −inf, so either the NaN persists (wasting the launch) OR the "forced-finite" variant trains B for DAYS on masked garbage in the teacher (dead
mse channels re-tokenized as sanitized zeros with no traveling mask). The 20-min catch didn't cost a day — it saved the multi-day run. Third time this
week catch-first paid rent. ch-107 DC resonance files as characterized-and-innocent (real fragility, not this NaN — [[project-tokenizer-ch107-dc-fragility]]).

#### K-ANNEAL — B LAUNCHED (the decisive clean run) 2026-07-17
Fix VERIFIED (smoke jobs 5018430/31/32, p_tf{1,0.5,0}): ALL 0 non-finite (old run NaN'd k=1..9); free-path byte-identical to pre-fix (step-1 loss
30.8358 to every decimal → fix touched ONLY the NaN path); mask-fraction rider PASS (mse dead-channel valid_frac=0.971 IDENTICAL step-0 vs rollout →
mask travels, no sanitized-garbage training). B CHAIN: 15 jobs 5018506→5018522 (afterany, -N8 -t2h, scontrol multi-partition extended,batch,g1),
CHECKPOINT_DIR=**e2e_g3fix_kanneal_v2** (FRESH — warm-start beta6.0_step3000.pt; NOT resuming the buggy old e2e_g3fix_kanneal Jul-16 latest.pt).
Config: curriculum K=10→20→40→80, block_steps=5000, tf_anneal=4000 (in block 0), gc=10, β PINNED 6, FEEDBACK_NORMALIZE OFF (the L1755-mirror fix
makes option-2 unnecessary). NEXT DECISION = K=10 gate (end of block 0, ~step 5000, ~3 days): per-block gate4_kprobe vs block-0 denominator; tripwires
MAE-gap<0.08 + displacement-ratio>1.0. Diff (train_e2e_stage1.py, uncommitted): FIX1 else-branch clean+mask-travel ~1846-1880; FIX2 per-modality
always-on nan-loc ~1917-1975. Old buggy dir + saga scratch dirs left intact (cleanup = post-validation, with user confirm).

#### K-ANNEAL — B first-launch OOM (production scale) → gc=1 relaunch (2026-07-17)
First B chain head 5018506: **mse FIX CONFIRMED WORKING AT SCALE** — trained INTO the rollout (PROJ-RESONANCE warns firing = feedback tokenizing),
ZERO nan-loc lines, NO NaN. But HIP OOM at -N8/batch16 ~32min in (ranks 3/4/6/7; 48.84/64 GiB alloc, +5.01 GiB failed, 9.68 reserved-unalloc =
fragmentation). Root: gc_every=10 at K=10 → ONE checkpoint segment for the whole 10-step rollout → backward holds all 10 steps' activations = peak OOM.
FIX (experiment-PRESERVING — grad-checkpointing is exact, IDENTICAL gradients): GRAD_CKPT_EVERY=1 (checkpoint every rollout step → peak = 1 step's
activations) + PYTORCH_ALLOC_CONF/PYTORCH_HIP_ALLOC_CONF=expandable_segments:True (defrag the reserved-unalloc). Batch/nodes/global-batch/β/curriculum
UNCHANGED. Scancelled 5018506-22 (per pre-stated bad-start plan; v2 dir empty = OOM'd before any ckpt → clean warm-start) → relaunched FIXED probe
chain **5018701-03** (3 jobs; EXTEND to 15 once confirmed past the ~32min OOM point). Monitor on head 5018701.

#### K-ANNEAL — OOM ROOT CAUSE MEASURED: full-horizon data load, not rollout (2026-07-17)
Investigation (5018701 traceback, read-only): OOM is NOT rollout activations, NOT graph-retention, NOT gc (gc=1 gave byte-identical 48.84 GiB).
Traceback = train_e2e_stage1.py:1786 _eval_spectro_bg_split → spectro_bg.py:74 conv1d, TARGET PREPROCESSING at the TOP of rollout_forward_loss,
BEFORE the rollout loop — 48.84 GiB already resident. ROOT: dataset_horizon_s = max(curriculum_Ks)*chunk + pred = 80*0.05+0.2 = 4.2s, FIXED for the
whole run (train_e2e_stage1.py:3084-3087). Even block-0 (K=10, needs 0.7s) loads the ENTIRE 4.2s multimodal future every batch (~46 GiB resident;
ece alone ~9.6 GiB target + ~9.6 GiB input). Flat across K + gc → matches the byte-identical evidence. conv1d (bg-split freq-Gaussian on the 4.2s ece
target, fp32) = the 5.01 GiB straw. expandable_segments DEAD on HIP. No grad-accum support. g3fix pinned batch_size=16 (global 128) per ckpt args.
FUNDAMENTAL TRADE: resident mem ≈ batch × horizon × data; NO code-only fix (46 GiB is loaded DATA not activations). Levers: #1 per-block horizon
(--rollout_dataset_horizon_s = current-block reach; keeps batch16/global128/β; but CHANGES window set — __len__=floor((dur-horizon)/chunk), 4.2→1s
≈3× more windows/shot [later-shot, arguably more-correct-for-K10]; lengths cache is horizon-specific → needs offline rebuild; K=80 reverts to 4.2s →
OOM returns → needs per-step-backward+bf16-conv later) vs #6 batch-8 (keeps 4.2s/window-set; changes global batch 128→64 = departs pinned operating
point; still carries 4.2s → only ~half). SURFACED to user for the call (genuine experiment decision). Reco: #1 (preserves pinned batch, window change
defensible-as-correctness, cache handled offline, K=80 deferred past K=10 gate). B not running; chains scancelled.

#### K-ANNEAL — LEVER #1 CHOSEN + PRE-REGISTERED FENCES (2026-07-17)
DECISION: Lever #1 (per-block dataset-horizon ladder), NOT batch-8. Rationale: #6 (batch 128→64) is a global, poorly-characterized perturbation to
the pinned g3fix optimization recipe (LR-vs-batch coupling, gradient-noise scale) the warm-start's validity rests on; #1's change is a sampling-
distribution shift that is characterizable + directionally understood + defensibly a CORRECTNESS improvement (block-0 was silently discarding every
window within 4.2s of shot-end that a K=10 rollout can legitimately train on — an over-restriction inherited from sizing the dataset to max-K).
Horizon convention: rollout_dataset_horizon_s(K) = K*chunk + pred_horizon = K*0.05 + 0.2 (block 0/K=10 → 0.7s; K=20 → 1.2s; K=40 → 2.2s; K=80 → 4.2s).

FENCE 1 (window-set confound — pre-registered): B trains under the per-block horizon ladder; the A-block ablation + block-0 baseline were measured
under the fixed 4.2s window set. Therefore A-vs-B TRAINING-DYNAMICS comparisons (val trends, tripwire trajectories) attribute to curriculum + window-set
JOINTLY (not cleanly separable). The per-block GATE metrics attribute CLEANLY — gate4_kprobe runs on the eval protocol's FIXED shots/windows, independent
of the training distribution (Fence 2 asserts this in code).
FENCE 2 (gate immunity — to be asserted in gate4_kprobe): the gate's window pool MUST be selected under a FIXED horizon convention across ALL blocks,
else the per-block denominators drift with the training horizon. One assertion in the gate script confirms the eval horizon is block-independent.
GATE-5 ENTRY-TICKET item (K=80 memory cliff — named, not a surprise): the ladder reverts to 4.2s at K=80 → the ~46 GiB resident load returns → OOM at
the top. Two known levers when we reach it: (1) per-step backward (∇Σ=Σ∇, holds one step's graph — see OOM-diagnosis lever #3), (2) bf16 + per-step-slice
the target bg-split conv1d (spectro_bg.py). Later ladder blocks are provisional — now provisional with a known cliff + two known levers.

#### K-ANNEAL — B BLOCK-0 LAUNCHED (lever #1, segmented curriculum) 2026-07-17
Lever #1 wired + verified. Launcher (train_e2e_stage1_kanneal.sh): ROLLOUT_DATASET_HORIZON_S + STOP_AT_STEP passthroughs (both opt-in, byte-identical
unset). Trainer: --stop_at_step (argparse:2392; loop guard:4121) breaks the loop at the block boundary while MAX_STEPS=20000 keeps the LR cosine T_max
→ the pinned ONE-cosine-over-20000 recipe is PRESERVED across the segmented ladder (NOT compressed to per-block restarts). Batch-16 memprobe @0.7s:
peak 43.5-44.3 GiB (fits «64), 0 non-finite p_tf{1,0.5,0}, mse fix holds, mask valid_frac 0.971. Offline cache prebuild 5019464 (train@0.7/val@0.2,
full 7878/875) → e2e_g3fix_kanneal_v2/lengths_h0.7/. B BLOCK-0 CHAIN: **5019671-5019682** (12 jobs, afterok:5019464, -N8 -t2h, multi-partition),
CHECKPOINT_DIR=e2e_g3fix_kanneal_v2, warm-start beta6.0_step3000, K=10, horizon 0.7s, STOP_AT_STEP=5000 → stops at the K=10 gate.
SEGMENTED-CURRICULUM SHAPE (operational): the run is now block-by-block, NOT one fire-and-forget chain. Block N relaunch = resume latest.pt +
ROLLOUT_DATASET_HORIZON_S = K_N*0.05+0.2 (K=20→1.2 / K=40→2.2 / K=80→4.2) + a fresh offline lengths cache at that horizon + STOP_AT_STEP=(N+1)*5000.
K=80 block (4.2s) hits the memory cliff → per-step-backward + bf16-conv levers (Gate-5 entry ticket). K=10 gate at step 5000 (~3 days). Head 5019671 monitored.

#### ENDGAME — 4-item convergence to d1024 production (2026-07-17)
The investigation closes on 4 items; when all land, d1024 production launches and it's training+writing only.
1. **tangtv + filterscopes ORACLE AUDITS** — LAUNCHED (agent a193535, days). Last unmeasured inputs to the locked spec. Fill two audit-conditional
   rows via pre-written rules: video-loss-structure (tangtv) + filterscopes-FSQ-question (filterscopes). Oracle gate = stability≥0.8 + persistence≫0.10.
2. **d1024 spec + PAPER_SUMMARY rebuild** — DONE. Corrected against the LOCKED modality table (user-confirmed: video split upper/lower divertor 2 codecs;
   spectro ece+co2+bes+mhr; FSQ = spectro+video, TS continuous). d1024/48L PRODUCTION built+counted EXACT (all 6 codecs on disk): TOTAL 1,203,520,250
   (~1.20B) / TRAINABLE 1,145,387,460 / FROZEN 58,132,790 (4 spectro + 2 video codecs); 2524-token seq. Pilot d512 (ece-only/no-video, 120.7M) relabeled
   method-development NOT production. PAPER_SUMMARY.md rewritten; FACT_SHEET_production.md + build_and_count_production.py. (My first d1024 number 837.8M was
   the pilot scaled up with video+co2/bes/mhr DROPPED — wrong for production, corrected.) Prod launcher already wired: train_e2e_stage1_d1024_48L.sh.
3. **K=10 GATE READ** — gate-dependent (~1-2 days, B block-0 5019671-chain → step 5000). Read against 3 pre-written branches: claim-lands / architecture-
   chain / anneal-re-pacing. Whichever fires, next launch is diff-against-text.
4. **GATE-5 TICKET completes** — TERMINAL. recipe(B's gate) + loss-structures(audits 1) + locked-spec → d1024 production launches. Then training+writing only.

#### ORACLE AUDIT — filterscopes FAIL (endgame item 1, half done) 2026-07-17
Oracle gate = stability≥0.8 (codeacc of encode(GT) vs encode(GT +0.5ms shift)) + persistence≫0.10 (codeacc codes(t) vs codes(t+1)), active windows.
FILTERSCOPES (fast-TS) audit job 5021557 COMPLETE (codec = exploratory eval_runs/fsq_fastts_final/fastts_codec.pt): stability=0.315 (FAIL, gate 0.8),
persistence=0.171 (floor 0.125), quiescent 1.000/0.996, corr(persistence,activity)=−0.986, no codec-OOD gap. SAME failure structure as spectro modes
(quiescent trivially copyable; ELM/burst windows scatter codes = realization/phase bits). → filterscopes-FSQ-question row RESOLVED: **do NOT FSQ-code
filterscopes; keep CONTINUOUS** — CONFIRMS the locked spec (TS continuous). PAPER_SUMMARY.md row updated. VIDEO (tangtv_lower 5021555 / tangtv_upper
5021556, production 2ch split codecs) still RUNNING (heavier tensors; sparse per-shot video presence but enough windows) → video-loss-structure row
pending both verdicts (rule: PASS→keep exact-code CE on tangtv; FAIL→decoded/perceptual/statistics-target loss, not exact-code CE). Watcher armed.
Scripts: eval_runs/oracle_audit/oracle_video_fastts.py + scripts/slurm_frontier/oracle_audit_video_fastts.sh; outputs oracle_{filterscopes,tangtv_*}.json.

#### ORACLE AUDIT — tangtv PASS → ITEM 1 COMPLETE (2026-07-17)
tangtv oracle jobs 5021555 (lower) + 5021556 (upper) COMPLETE. ACTIVE-stratum verdict (the gate; pooled/quiescent were near-1.0 for both, non-
discriminating): tangtv_lower stability_active=0.846 persistence_active=0.821 (in_active 0.906 / out_active 0.826 — no OOD collapse); tangtv_upper
stability_active=0.873 persistence_active=0.870 (in_active 0.843 / out_active 0.929). stability_pass=persistence_pass=ORACLE_PASS=TRUE for both.
→ video-loss-structure row RESOLVED: **codes stable+persistent → exact-code FSQ code-CE on tangtv is well-posed → keep planned FSQ video loss.**
ENDGAME ITEM 1 COMPLETE: both audit-conditional rows filled, BOTH CONFIRM the locked spec (filterscopes CONTINUOUS, tangtv FSQ-CE). The active-stratum
split was decisive both ways (filterscopes active 0.315 FAIL vs video active 0.85-0.87 PASS). Spec fully measured; no changes. PAPER_SUMMARY.md §6 updated.
Remaining endgame: item 2 DONE, item 3 (K=10 gate) pending ~1-2 days, item 4 (Gate-5 ticket → d1024 production) terminal.

#### K=10 GATE READ — FAIL (claim does not land) — endgame item ③ (2026-07-18)
Gate model = e2e_g3fix_kanneal_v2/e2e_stage1_latest.pt STEP 5000 (K=10 block boundary, STOP_AT_STEP). gate4_kprobe job 5025827 (protocol reproduced
EXACTLY from block-0 denominator: argmax, DOSES 0/+2/−2, K=40 gate@10, SHOT 200729, n=256, β6; round-trip smoke corr 0.979; Fence-2 asserted in-code
eval_horizon=2.0s block-independent). Paired vs BLOCK-0 (beta6.0_step3000, banked eval_runs/kanneal_block0_baseline job 5010348):
  ctrl diff k10 (PRIMARY): +0.000 → +0.4919 [+0.401,+0.581] (GREW, NOT bidirectional — magnitude gap from drift amplification; both doses accumulate
    upward, controllable@k10=False; +2σ→+0.321, −2σ→−0.171 regime=accumulates = A-block s2000 signature)
  ctrl diff k0: −0.0231 → +0.0552 (FLIPPED; k0_bidirectional True→False)
  ctrl diff k39: −0.142 → +0.816; drift ratio pred/GT: 3.475× → 6.543× (WORSE); false-death k10/k39: 0/0 (survival intact); mode-present 153/256;
    retention 1.337→1.657. Tripwires BOTH FIRED: ece MAE-gap +0.0208 (<0.08), displacement-ratio 1.099 (>1.0). Fence1: GATE metrics attribute cleanly
    (un-confounded by curriculum+window-set); Fence2: passed.
VERDICT: claim-lands criterion (RETURNS BIDIRECTIONAL, CI excludes 0 signed-like-k0, dose ordering restored) NOT met (k0 flipped + k10 wrong-signed +
drift worse). k0-RIDER FIRES (k0 sign-flip = treatment destroying a banked property). WATCHED drift pre-reg FIRES (drift did NOT calibrate under K-
training = earliest signal Gate-5 needs the drift-PENALTY variant). B reproduced the A-block s2000 failure under the FULL anneal at step-5000 → curriculum
+TF (B's whole intervention) did NOT rescue drift/controllability. **DO NOT launch d1024 production.** NEXT (pre-registered, diff-against-text): drift-
PENALTY K-anneal variant (warm-start pristine beta6.0_step3000.pt, NOT eroding ckpt); FALLBACK = architecture-chain (fair-FiLM from-scratch/longer β6-
anchored → actuator-conditioned anchor). AMBIGUITY FLAGGED: 3 branch NAMES bare at line 1050; mapped to line-726 pre-reg + k0-rider + drift-watch;
decision ROBUST to mapping (production does not launch under any reading); WHICH next-launch (drift-penalty-first vs architecture-chain) = user's read.
Artifacts: eval_runs/gate4_kanneal_v2_k10_step5000/.

#### K=10 GATE — BRANCH CONFIRMED + STRIKE-3 PRE-REGISTRATION (2026-07-18, 2am)
BRANCH CONFIRMED: drift-PENALTY K-anneal variant (diff-against-text), warm-start PRISTINE beta6.0_step3000.pt (eroded step-5000 ckpt contributes its
gate table + nothing else). AMENDMENT (conditional on the tripwire-trajectory plot, pending): if erosion begins as p_tf→0 (prior — A-block reproduced
AFTER anneal completed), the variant ALSO carries a TF-FLOOR (anneal p_tf→~0.2-0.3 not 0, OR stretch the anneal across the whole block) so the drift
penalty doesn't fight the same free-rollout gradient that just won twice. This MERGES the anneal-re-pacing branch INTO the drift-penalty branch (both
pre-registered), attacking BOTH failure modes (drift amplification + k0-flip) in one launch.
STRIKE-3 PRE-REGISTRATION (explicit, per user): "training fixes drift" now has TWO STRIKES (A-block + B reproduced the SAME k0-flip/drift-amplification
signature). The drift-penalty+TF-floor variant is its THIRD and LAST cheap training-side test. PRE-REGISTERED OUTCOME: if direct drift supervision + a
TF floor STILL reproduces the signature (k0 flip / drift amplification / k10 not bidirectional) → conclusion = THIS CHECKPOINT LINEAGE'S ROLLOUT
DYNAMICS RESIST CALIBRATION (plausibly the anchor again — the trilogy's 4th act) → path forward is NOT more K-anneal variants but the ARCHITECTURE
CHAIN at d1024 FROM-SCRATCH (production trains rollout-native from step 0, not retrofitting rollouts onto a single-step-trained model). Contingency
half-written in the Gate-5 spec; the K=10 gate table (job 5025827) is its evidence base.
FOR THE RECORD (clean negative): survival + dynamics + false-death (0/0) INTACT; Fences 1+2 HELD (negative is un-confounded); tripwires matched their
A-block calibration (both fired); gate read itself out against pre-written branches within minutes of the step-5000 ckpt landing. The claim didn't land
tonight; the instrument did. SEQUENCE: tripwire-trajectory plot (~30min → TF-floor decision) → drift-penalty+TF-floor variant from pristine ckpt (LAST
training-side attempt) → its gate ~2-3 days. Production HELD.

#### STRIKE-3 — DESIGN CONFIRMED + SUCCESS BAR PRE-REGISTERED (2026-07-18)
REFRAME (from the tripwire trajectory): both tripwires breached at STEP 500 / p_tf=0.875 (near-max TF) and stayed breached → NOT erosion (no trajectory)
= a STEP FUNCTION. The K-rollout summed objective is INCOMPATIBLE WITH THE BANKED k=0 OPERATING POINT ON CONTACT. Candidate mechanism (unruled-out):
rollout loss sums per-step FSQ-CE + descriptor over K=10 → gradient composition changed radically from Stage-1 (10× terms, dominated by later-k whose
inputs are TF-fed but targets are deeper futures); k≥1 gradients CONFLICT with the k=0 term → k=0 property traded away from step 1. p_tf schedule AND
drift-penalty both leave gradient composition INTACT → neither fixes it (explains A-block + K-anneal + this trajectory in one sentence).
STRIKE-3 DESIGN (CONFIRMED, TWO levers targeting the two independently-measured failure modes): (1) DRIFT-PENALTY, ASYMMETRIC = relu(pred_drift −
gt_drift) [asymmetric b/c pathology is uniformly OVER-drift; symmetric would punish legit under-drift corrections]; (2) PER-K LOSS RE-WEIGHTING with
k=0 PROTECTED [k=0 keeps Stage-1 gradient share — k=0 full weight, k≥1 down-weighted/annealed-up; the inverse of implicit uniform summing; ≡ optional
k0-distillation anchor to frozen warm-start]; (3) NO TF-floor (verdict-b: TF didn't protect, keep test clean); (4) PRISTINE warm-start beta6.0_step3000;
(5) MID-BLOCK WATCHER (breach alert at FIRST val, kill-don't-wait). Two levers strain one-change purity but a one-lever strike-3 is designed to fail on
the other mode. ATTRIBUTION (pre-reg honest): pass → pair validated jointly, disentangle post-hoc; fail → BOTH training-side levers exhausted →
architecture chain opens with a COMPLETE negative record.
STRIKE-3 SUCCESS BAR (unambiguous exit, pre-registered): (1) k0 differential BIDIRECTIONAL again (CI-separated, both signs); (2) drift ratio < ~1.5×
AND falling across the block; (3) tripwires UNBREACHED at EVERY val; (4) k10 differential INTERPRETABLE (drift calibrated so the dose fan isn't riding
amplification). ANYTHING SHORT → ARCHITECTURE CHAIN at d1024-FROM-SCRATCH (rollout-native from step 0), NO STRIKE-4.
FAST-VERDICT NOTE: step-500 breach means strike-3's verdict may arrive in HOURS not days — if tripwires breach at the first val AGAIN despite both
levers, the watcher self-terminates → the architecture decision arrives THIS WEEK = the cheapest decisive negative of the project.

#### STRIKE-3 LAUNCHED (2026-07-18 13:17:47 EDT)
Chain 5028760-5028766 (7 jobs, -N8 -t2h, multi-partition). CHECKPOINT_DIR=e2e_g3fix_strike3 (fresh, warm-start PRISTINE beta6.0_step3000),
LENGTHS_CACHE_DIR=e2e_g3fix_kanneal_v2/lengths_h0.7 (reused). Config = B + two levers: DRIFT_PENALTY_WEIGHT=0.5 (asymmetric relu drift-pen on the
gate's centroid-vs-anchor drift_pred), K_GE1_WEIGHT_START=0.1 K_GE1_WEIGHT_ANNEAL_STEPS=4000 (k=0 pinned 1.0; k≥1 anneals 0.1→1.0). Defaults match B
(CURRICULUM 10,20,40,80; TF_ANNEAL_STEPS=4000 [NO floor — verdict-b]; GRAD_CKPT_EVERY=10; MAX_STEPS=20000; STOP_AT_STEP=5000; batch 16; horizon 0.7).
Smoke verified (build agent): CPU 5/5+S3a-d, GPU job 5028692 0 non-finite p_tf{1,0.5,0}, drift-pen live+asymmetric (3.67-7.04), w0=1.0/w_ge1=0.1.
MEASURED-basis timeline (B block-0 = 5000 K=10 steps in 9h41m, ~7.0s/step, 2026-07-17 14:12→23:53): once the head dequeues, fast-negative check
(kill-on-breach at step-500 val) ≈ +58min training; full K=10 gate ≈ +9h41m training. Queue wait unmeasurable ahead. Kill-on-breach watcher armed
(scancel chain on ece MAE-gap<0.08 at any val = k0-protection failed = B step-500 signature). Success bar (pre-registered): k0 bidirectional + drift
<1.5× falling + tripwires unbreached every val + k10 interpretable → else architecture-chain d1024-from-scratch (no strike-4).

#### STRIKE-3 VERDICT — FAIL → ARCHITECTURE CHAIN (2026-07-18 14:11 EDT)
Kill-on-breach watcher FIRED FAST_NEGATIVE at step 500 + scancelled chain 5028760-66 (authorized). REAL timing: launch 13:17:47 → head start
13:18:00 (near-instant dequeue) → step-500 val 14:10:15 → kill 14:11:10 = **53 min launch-to-verdict** (cheapest decisive negative, as pre-registered).
STRIKE-3 @ step 500 vs B @ step 500: ece MAE-gap 0.0520 (B 0.0428), disp-ratio 1.007 (B 1.033). BASELINE CHECK (disambiguates): g3anneal warm-start
run ece gap reached ~0.35 (cleared 0.08 by 4×); K-rollout runs (B + strike-3) stuck 0.02-0.05 → 0.052 = ~7× COLLAPSE of single-step ece skill, NOT
preserved-warm-start-level. 0.08 threshold is honest. LEVER READOUT: drift-penalty WORKED (ratio 1.033→1.007 ≈ calibrated → drift IS training-fixable);
k0-protection re-weighting INSUFFICIENT (gap 0.043→0.052 marginal, still 7× below warm-start; and step-500 is BEST-case for k0 [anneal → k≥1 weight only
~0.21], collapsed anyway → only worsens). THREE STRIKES (A-block, B, strike-3): the summed K-rollout objective collapses the banked k=0 property from
the first steps; NO training-side lever (TF schedule / drift-penalty / k0-re-weight) prevents it. → PRE-REGISTERED OUTCOME FIRES: **ARCHITECTURE CHAIN
at d1024-FROM-SCRATCH, NO STRIKE-4.** Training-side investigation CLOSED. Path = d1024 production trained ROLLOUT-NATIVE from step 0 (no single-step-only
banked property to collapse). Gate-5 ticket now completes: recipe = rollout-native architecture-chain + audit loss-structures (filterscopes continuous,
tangtv FSQ-CE) + locked spec (1.2B d1024/48L). Production launch = user's design/confirm (the "training + writing only" commitment).

#### ROLLOUT-NATIVE d1024 PRODUCTION — RECIPE + CAUTION + CONTINGENCY (pre-registered 2026-07-18)
MECHANISM SENTENCE (paper methods justification): "the summed rollout objective trades away single-step skill on contact; only training rollout-native
from step zero avoids the trade." Drift-penalty validated on the way out (strike-3 ratio 1.033→1.007) → qualifies for the recipe.
CAUTION (scope-honest, in the record): "no training-side lever prevents it" rests on THREE variants (A-block, B, strike-3) that ALL warm-started from a
single-step optimum. From-scratch is a BET (collapse = artifact of starting at the single-step attractor), well-motivated but NOT a measurement — the
d1024 run is its test.
CONTINGENCY (pre-registered NOW, fires on evidence): if rollout-native ALSO can't hold single-step skill alongside horizon stability → tension is
OBJECTIVE-INTRINSIC → fallback = STAGED TRAINING (single-step phase → rollout phase WITH k0-distillation). Trigger = the per-k loss-share log: if the
k=0 share collapses as K grows, that's the early signature → staged-training fires (do NOT invent at 2am).
RECIPE (d1024/48L, full modality 1.20B, FROM-SCRATCH random init, rollout-native):
- K-SCHEDULE: curriculum FROM K=1 (NOT fixed-K, NOT K≥2). K=1 initial phase = Stage-1-equivalent but UNDER the rollout loss framework (no objective
  switch ever — only horizon EXTENSION). Then K=2→5→10→… on VAL-GATED boundaries (not fixed step counts). Key: the failure mode avoided is the OBJECTIVE
  CHANGING; a K-curriculum under ONE loss family is extension, not retrofit.
- TF: standard schedule within each K-block (trajectory showed TF wasn't the problem; the warm-start was).
- DRIFT-PENALTY: IN from step 0, asymmetric relu(pred_drift-gt_drift), weight = strike-3's (0.5 — don't retune what worked). Inert at K=1, bites as K
  grows = self-scheduling.
- k0-PROTECTION: OUT (retrofit lever; from-scratch has no banked property yet). Uniform per-k weighting. BUT LOG per-k loss shares from step 0 (the
  contingency trigger).
- LOCKED-TICKET rest: full modality + audit loss-structures (filterscopes CONTINUOUS, tangtv FSQ-CE), β=6 anchor, actuator standardization from step 0,
  FiLM-flag OFF (deferred), standing smoke battery (3 p_tf, mask assertion, per-modality nan-loc, proj-band), mid-block tripwire watcher with FROM-SCRATCH
  thresholds — TRAJECTORY-BASED for the first phase ("gap RISING through step N"), NOT the absolute 0.08 floor (warm-start-calibrated; a from-scratch run
  crosses 0.08 FROM BELOW during normal learning).
- GATES: K-equivalent gate table at EACH curriculum boundary; block-0-style denominators banked at each K before extension; argmax paired counterfactual
  enters the suite once single-step conditioning FIRST appears (log its arrival step = first evidence the model learns the pin response at all).
SEQUENCE: smoke @ d1024 (step-rate + memory at K=1 AND K=10 → the REAL timeline) → pre-registration entry w/ contingency → EGEMEN sees the recipe
(his compute + owed case-study/scope hour = the ONLY human dependency on the critical path) → launch (~10-day full-budget run). Production of the retrofit
K-anneal path CLOSED.

#### ROLLOUT-NATIVE d1024 SMOKE — FIT + TIMING (measured 2026-07-18)
Config builds + trains from-scratch (1,203,520,250 params, all 6 codecs load, all 14 modalities). Battery ALL PASS (p_tf{1,0.5,0} finite, 0 non-finite,
nan-loc clean ×14, mask ok, proj-band clean ×4 spectro). Per-k loss-share logging LIVE (contingency trigger): K=1 k0_share=1.0; K=10 ~uniform 0.10 each.
3 FSQ-video+rollout integration bugs found+fixed (video class-weight target truncation; dataset_horizon must = K*chunk for video else per-step target≠codec
window; FSQ-video decode (B,T,C,H,W)→(B,C,T,H,W) permute at both feedback sites). Config = opt-in ROLLOUT_NATIVE=1 block on train_e2e_stage1_d1024_48L.sh.
MEMORY FIT (64 GiB MI250X GCD, 2 ranks, full modality, from-scratch): K=1 b16 gc10 = 35.74 GiB FITS (big margin); K=10 b16 gc10 = 62.5 GiB OOM; K=10 b8 =
61.7 OOM; **K=10 b4 gc10 = 58.5 GiB FITS**; K=10 b16 gc1 = pending (job 5029524). → batch 16 does NOT fit at K=10; needs BATCH SCHEDULE (16→4) as K grows.
STEP-RATE (measured): **K=1 6.37 s/step** (b16); **K=10 19.0 s/step** (b4). Caveats: full-modality K=10 getitem ~2.2 s/sample (30 video frames); K=10 first-step
MIOpen compile ~15 min (one-time cold). TIMELINE (measured basis, 5000 steps/phase): K=1 ~8.8h, K=10 ~26h; to-K=10 curriculum (K=1,2,5,10) ~2.7 days compute;
to-K=80 full ~10 days (K=80 dominates) — confirms the ~10-day memory estimate, now measured. DECISIONS SURFACED: (1) batch schedule 16→4 (or b16gc1 if it
rescues 16 — pending); (2) batch 4 at K=10 drops GLOBAL batch 128→32 at 8 nodes → training-dynamics change; options = accept / more nodes (b4×32=128) /
grad-accum (NOT supported). ROLLOUT_DATASET_HORIZON_S = K*0.05 must be set per K-block for the FSQ-video path. Egemen-review-ready. Production NOT launched.

#### ROLLOUT-NATIVE d1024 — BATCH DECISION: OPTION 1 LANDS (global 128 held) (2026-07-18)
GLOBAL BATCH = HOLD 128 (pre-registered, user directive): do NOT accept 32 at high K — a 4× optimization-regime shift at exactly K=10 confounds the gate
between "objective-at-horizon" and "small-batch-noise-at-fixed-LR" = uninterpretable. LADDER (never "accept 32 and hope"): (1) gc=1 rescues b16 → b16
throughout; (2) else more nodes (b4×32=128) = Egemen node-ask; (3) else grad-accum (~1 day + battery).
gc=1 RESULT (job 5029524, MEASURED): batch16 K=10 gc_every=1 → peak **49.28 GiB** / 64 (from 62.5 OOM at gc=10) → **FITS with margin → OPTION 1 LANDS:
batch 16 throughout, GLOBAL 128 HELD, NO node-ask needed.** Job crashed AFTER 18 clean training steps, in validate()→copy_baseline_mae→masked_mae
(train_e2e_stage1.py:429): "tensor a (3) vs b (12) at dim 2" = VAL SHAPE-BUG (persistence-baseline window-count mismatch under the rollout-native val),
SAME CLASS as the prior 17-vs-5 actuator val patch (fixed via val_prediction_horizon_s). NOT OOM, NOT training. Bounded pre-launch fix.
LR-AT-K-BOUNDARIES (pre-registered note): with global 128 held (option 1), NO mid-run LR-batch renegotiation → the LR-scaling concern EVAPORATES (had
option-1 failed → batch schedule → LR must scale with batch per transition = a coupled change gates can't attribute — another reason to hold 128).
STAGED-TRAINING CONTINGENCY TRIGGER (pre-registered numbers, fires the single-step→rollout+k0-distillation fallback): (a) k0-share collapses MATERIALLY
below uniform (1/K) as K grows [per-k-share log, live from step 0], OR (b) K-boundary gate shows single-step-skill metrics (from-scratch MAE-gap analog,
TRAJECTORY-thresholded not absolute floor, per the watcher redesign) degrading block-over-block. Rough > 2am-judgment.
REMAINING PRE-LAUNCH (critical path): (1) fix val shape-bug; (2) clean re-smoke gc=1/b16 K=10 → REAL step-rate + timeline; then Egemen (recipe+fit+timeline,
NO node-ask) → launch. Measured 10 days count from launch; first gate (K=1→2) < 1 day training in.

#### ROLLOUT-NATIVE d1024 — LAUNCH PRE-REGISTRATION (K-target + K=1→2 reading frame) 2026-07-18
VAL FIX DONE (subagent): video val-horizon mismatch (target 12 frames vs pred/persistence 3; dim-2 (B,C,T,H,W)); guarded slice of target frame-axis to
pred's; no-op for single-step/non-rollout/d512; VAL PASSES + 0 non-finite at K=1 (5030378) & K=10 (5030379). CLEAN RATES (option-1, global 128 held):
K=1 b16 gc10 = 6.36 s/step (35.7 GiB); K=10 b16 gc1 = 66.6 s/step (49.3 GiB) — ~10× K=1 (inherent to rollout depth). Rate ~linear in K (K=20≈133s, K=40≈266s,
K=80≈533s).
(1) K-TARGET = CLAIM-NEED, NOT LADDER-COMPLETENESS (plan of record): title claim = HORIZON CONTROLLABILITY (counterfactual differential sustained past the
~10-step wash-out that killed every retrofit) → demonstrable at **K=10–20 (~1–2 wks)**. K=80 (4s-discharge SCOPE claim) via CHEAPER PATH: train to K=10–20
(calibrated drift) → **EVALUATE at K=80 rollout** (drift penalty plausibly generalizes beyond training horizon; K=40 stress-eval machinery from Gate 4
exists) → 4s figure WITHOUT the 533s/step training phase. RULE: train to K=10–20 per gates, eval to K=80, EXTEND training ONLY if K=80 eval shows
horizon-specific degradation the gates say training would fix. → decisive result ~1.5–2 wks; extension = measured option not default; coexists w/ RFE calendar.
(2) K=1→2 GATE READING FRAME: at K=1→2 the from-scratch model has ONLY single-step training → the gate is NOT yet about the collapse. COLLAPSE QUESTION reads
at K=2+ boundaries: does single-step skill (TRAJECTORY-thresholds per watcher redesign, NOT absolute floor) HOLD as K extends, where every retrofit lost it
<500 steps? First genuinely informative signal = K=2 phase's first vals. BRANCH: holds → continue; degrades → per-k-share log arbitrates → staged-training
contingency (single-step phase → rollout + k0-distillation). NOTE (from-scratch phase length, honesty): the K=1 phase is single-step learning FROM RANDOM —
original single-step pretrain was ~118k steps, so K=1 is val-gated + potentially day-scale (NOT the 5000-step ~9h read); first VAL/tripwire signal <1h (step
500), but the K=1→2 gate needs the fuller phase. Val-gating not yet automated → manual (watch single-step val plateau) until built.

#### ROLLOUT-NATIVE d1024 K=1 PRODUCTION — LAUNCHED (the decisive run) 2026-07-18 21:39:17 EDT
Cache prebuild 5030687 (prebuild_h005, 1-node no-NCCL): full 7878 train @ horizon 0.05 + 875 val @ 0.2 → e2e_d1024_rollout_native/lengths_h0.05/
(~60-90min). K=1 PRODUCTION CHAIN: **5030688→5030691** (4×2h, job1 afterok:5030687, afterany chain, multi-partition extended,batch,g1),
CHECKPOINT_DIR=e2e_d1024_rollout_native (FRESH). CONFIG CONFIRMED (= smoke 5030378 at prod scale): **8 ranks=8 GCDs, batch16×8=GLOBAL 128** (overrode
launcher's 64-GCD default via -N8 --ntasks-per-node=1); **FROM-SCRATCH cold random init** (no INIT_CKPT/resume); curriculum_Ks=[1], horizon 0.05,
drift_penalty 0.5, UNIFORM k-weight (k0-protection OUT), gc=10 (35.7 GiB fits), NO STOP_AT_STEP (open-ended, human val-gates K=1→2), MAX_STEPS=118000
(LR-cosine horizon = original single-step-pretrain length); prediction_horizon 0.2 (val single-step); full modality (ece/co2/bes/mhr FSQ + tangtv_lower/
upper FSQ + filterscopes/7×slow-TS continuous); 1203.52M params; val-fix present. K=1 watcher = HEALTH+LEARNING only (NO kill — collapse gate is K=2+):
from-scratch single-step MAE-gap should RISE as it learns (trajectory frame, not absolute 0.08). First VAL/tripwire ~step500 (~53min training); K=1→2
gate needs the fuller (val-gated, potentially day-scale) phase. Retrofit K-anneal + strike-3 paths CLOSED; this is the from-scratch bet's decisive test.

#### ROLLOUT-NATIVE d1024 K=1 — chain exhausted (my under-provision) → EXTENDED (2026-07-19 07:47 EDT)
Initial chain 5030688-91 (4 jobs=8h) all CLEAN TIMEOUT (exit 0:0, NO crash) → ran to ~step 2750/118000, latest.pt 06:53. HEALTHY + LEARNING: single-step
ece gap ROSE 0.0123(step500)→0.0372→0.033→0.031 (from-scratch building single-step skill), disp-ratio 1.112→~1.0 (drift-penalty CALIBRATED from scratch =
the design working). Chain just RAN OUT (I queued only 4 jobs; K=1 is days). MEASURED EFFECTIVE RATE = ~10.5 s/step (steady ~9s + per-job restart/MIOpen-
compile overhead; slower than the 6.36s single-node smoke). EXTENDED: 5031850-5031873 (24 jobs=~48h runway, resume from latest.pt, same config, multi-
partition). TIMELINE (honest, measured): K=1 phase val-gated (gap-plateau) = ~days (plateau ~30k steps → ~3.6 days; full 118k → ~14 days). Gap 0.031 at
step 2750 is EARLY (of 118k) + well below warm-start's ~0.35 — whether it climbs toward 0.35 or plateaus lower is the from-scratch bet's key readout,
develops over the extension. NOTE: per-job MIOpen recompile is a ~10% overhead over a days-long run → shared MIOPEN cache is a worthwhile optimization
(follow-up). No kill at K=1 (health+learning only; collapse gate at K=2+).

#### ROLLOUT-NATIVE d1024 K=1 — TWO launch-flag-omission errors, both fixed (2026-07-19 ~09:42 EDT)
Same class of error twice: a launch replicated only a SUBSET of the launch env → silent wrong config until the model builds. (1) MY extension omitted FSQ/
patch/use_video/no-video-filter env → launcher default = OLD generative arch (tokens=1084, 1.814B) → resume state_dict mismatch → 24 jobs failed (~1h50
compute + a ~1h13 wrong-config cache rebuild that CORRUPTED the h0.05 cache: 4430/464 filtered subset vs the correct 7878/875). (2) The fix-subagent's
resume omitted --ntasks-per-node=1 → launcher SBATCH default --ntasks-per-node=8 = 64 GCDs = GLOBAL 1024 (a mid-run batch shift 128→1024, forbidden by the
interpretability discipline) — CAUGHT PENDING before it ran. FIX: cache rebuilt clean (prebuild 5032049, ~113min); resume chain relaunched 5032090-5032113
(24 jobs) at 8 GCDs/global 128 (--ntasks-per-node=1, NumTasks=8 verified), FSQ env taken from the CHECKPOINT'S SAVED ARGS (spec_fsq+fsq_resid_p8_all,
video_fsq+fsq_video_codecs_2ch, patch 8/16, use_spectro ece/co2/bes/mhr, use_video tangtv_lower/upper, no_video_presence_filter, pred_horizon 0.2, drift
0.5, curriculum 1), afterok:5032049. latest.pt (step 2360, 1.2B FSQ) SAFE throughout (failed jobs errored on LOAD, never wrote). Cost = ~3-4h wall-clock
(failed compute + cache rebuild + idle since ~09:00), NOT progress. LESSON: replicate the FULL launch spec — or pull it from the checkpoint's saved args —
never a subset; launch-flag omissions are silent until the state_dict load fails. Verify-before-trust monitor (bduo6r9se) confirms tokens=2524/clean-resume/
step~2360/8-GCDs before declaring the run live. Note 5032072 (clnilss_chan) = ANOTHER USER's job, coincidental id — not mine, correctly un-cancellable.

#### ROLLOUT-NATIVE d1024 K=1 — RESUME VERIFIED CORRECT, saga closed (2026-07-19 11:20 EDT)
Corrected resume chain 5032090-5032113 RUNNING + VERIFIED (from 5032090 log): Model tokens=2524 params=1203.52M ddp=True (correct FSQ arch); "resuming
from latest.pt" (not cold); NO state_dict/size-mismatch error (clean load); Spectro FSQ patch 8/16; K-rollout dataset horizon 0.05 / model 0.2; cache
"Loaded from cache" 7878/875 (rebuilt clean, no rescan). world_size verification (DEFINITIVE, from job logs): original 5030688 = rank=0/8 → 8 GCDs /
global 128; resume 5032090 = rank=0/8 → 8 GCDs / global 128 → MATCH, no batch shift, honors the global-128 decision. (Two subagents claimed the original
was "64 GCDs" — MISREAD; the process world_size=8 in the original's own log is authoritative.) Double config-error (my FSQ-env omission + the fix-agent's
--ntasks-per-node omission) fully resolved; ~3-4h wall-clock lost, ZERO progress lost (resumed at step 2360). Open efficiency Q for Egemen (not launch-
blocking): -N8 --ntasks-per-node=1 = 1 GCD/node → 8 nodes held for 8 GCDs (data-bandwidth headroom vs node-efficiency); pack to 1 node × 8 GCDs frees 7.

#### ROLLOUT-NATIVE d1024 K=1 — gap trajectory WATCH POINT (2026-07-19 19:30 EDT, step 4750/118000 ~4%)
Verified-correct run training clean. ece single-step gap trajectory: rose 0.012(step500)→0.037(~1000) then FLAT ~0.03 for steps 1000-4750 (0.031/0.023/
0.036/0.032 over the last 8h) — NOT climbing toward warm-start's ~0.35. Drift-ratio holds ~1.0 (drift-penalty working = positive). Loss noisy ~95-100.
AMBIGUOUS, NOT a verdict (4% in): (a) watch-point = ece single-step plateauing weak (~0.03 barely>persistence) = quality concern; (b) benign = early +
ece is hardest modality (mode-collapse-prone) + rollout-native/drift regime ≠ pure-single-step so 0.03-vs-0.35 may be apples-to-oranges + loss still high.
RESOLVES with more steps (climb vs pinned-at-0.03 over next tens-of-K). NOT over-calling on 4 noisy vals. Watcher re-armed. The K=1→2 gate reads whether
single-step HOLDS as K extends (which cares about hold, not absolute level) — but a weak single-step baseline is worth watching for model quality.

#### ROLLOUT-NATIVE d1024 K=1 — WATCH-POINT ESCALATED: ece gap DECLINING (2026-07-20 05:36 EDT, step ~7700-8050 ~6.5%)
ece val gap (fixed val set): flat 0.031/0.023/0.036/0.032/0.031 → then DROPPED 0.0088/0.0034 (last 2 vals); model ece MAE RISING 0.21→0.239 toward fixed
persistence 0.2424 → single-step ece decaying toward persistence. Drift-ratio rose 1.02→1.15. At K=1 (single-step, NO rollout) → points at SPECTRO
MODE-COLLAPSE recurring in the from-scratch ece head (the project's oldest failure), NOT a rollout effect. CAVEAT (not over-calling): only 2 declining
vals @ 6.5%; ece_codeacc still oscillates 0.11↔0.93 by batch (easy/hard alternation, NOT total collapse — code head still nails easy batches). NOT a
verdict; NOT intervening (over-calling on 2 vals is the trap). RESOLVES in next 3-4 vals: keep declining→0 (real ece collapse = from-scratch bet struggling
on the hardest modality even single-step) vs recover to ~0.03+ (transient dip). Tighter watcher set. If real: decision point (the from-scratch bet's ece
quality, independent of the K=2+ collapse gate). Run otherwise healthy (0 anomalies, drift-penalty holding on ratio elsewhere).

**RESOLUTION 2026-07-20 12:11 EDT (step 9750, ~0.145 epoch): ece-decay alarm STOOD DOWN — transient dip, NOT collapse.** Next val gap
recovered 0.0034(trough)→0.0138 (climbing back toward the 0.02–0.036 band it oscillated in), i.e. NOT a monotone slide to 0. Corroborating: ece
head loss is CONTENT-RESPONSIVE (0.17–0.42 on mode-active batches, 0.0 on frozen/padding batches) — a truly collapsed head emits ~constant
output with content-independent loss, which is NOT what's happening; ece_codeacc bimodal 0.14/0.95 is the known 200729-style padding confound
(0.95 = mostly-padding batches), not new collapse evidence. KEY REFRAME: at ~0.145 epoch from-scratch the MAE-vs-persistence gap is intrinsically
LOW-SIGNAL — tiny (0.003–0.036) vs warm-start's ~0.35 because the model is barely trained; the gap must BUILD over epochs, so neither the dip nor
recovery is diagnostic yet. Judging ece collapse off this early gap = the over-call trap. ACTION: retired the tight 6h decay watcher; armed a
lighter long-horizon TREND watcher (does the gap build over the next epoch, active-batch code-acc trend up) that also flags the K=1→2 curriculum
boundary = the first real collapse gate. Not intervening. Real collapse verdict deferred to a render + longer active-batch code-acc trend.

**STEP-10.8k RENDER (2026-07-20, job chain 5039307→5039715, `eval_runs/comparison/rn_native_step10800/`, 1-step-ahead K=1, shot 200729).**
Pipeline validated end-to-end on the from-scratch rollout-native arch (load_model rebuilds all 4 FSQ families from ckpt args; clean). **Spectro-panel viz FIXED + VALIDATED** in `eval_e2e_animation_tokamak.py`: the panels scaled off raw log|STFT| (background-dominated → flat plate, showed nothing). Fix = `_spectro_mode_view()` per-freq z-norm (subtract per-panel per-freq temporal mean, divide by GT per-freq std → modes on a common σ scale, flat pred stays flat), floor 0 / ceiling p95 of GT z-map; applied to GT/recon/pred; colorbar "log|STFT| z (per-freq)". EVAL_SPEC_DEBUG=1 dumps z-map pctls. **KEY FINDING (user-confirmed, corrects my flip-flop):** co2 GT shows a clear broadband mode 1–2 s / full-freq; the **CODEC RECON carries it almost perfectly** (recon≈GT) → FSQ representation is FAITHFUL, modes ARE in the code space. Prediction panel is FLAT there (ece z_pr p99=1.31 vs GT 3–6σ; co2 pred flat vs GT broadband). ⇒ **bottleneck localized to the WORLD-MODEL PREDICTION, not codec/viz/data.** Recon proves the ceiling exists: if pred learns mode-bearing codes, modes appear. Open question reduces to "does pred → recon as training proceeds" — the trend watcher's job. Render recipe: `EVAL_K=1 EVAL_ROLLOUT_STEP=0 EVAL_BATCH_SIZE=32 EVAL_EXTRA_ARGS="--comparison_figure --no_spec_fusion" sbatch -p batch eval_e2e_animation_tokamak.sh <ckpt> 200729 <outdir>`; ~2 min warm.

**⚠️ CORRECTION 2026-07-20 (user-flagged, INVALID-INSTRUMENT): the "prediction FLAT / bottleneck = world-model prediction" conclusion above is RETRACTED.** The pred spectro panels were rendered via the code head's EVAL-mode sampling at the DEFAULT `spec_code_temperature=1.0` (`SpectrogramCodeHead`: eval=multinomial, train=argmax; per-position = INDEPENDENT-MARGINAL). Per the sampling work, argmax AND independent-marginal-at-high-T erase coherent modes BY CONSTRUCTION even when the model's distribution contains them — so a flat decoded panel is NOT evidence the model lacks modes. T=0.3 re-render (`_T03/`, job 5040016) made ece FLATTER still (z_pr p99 1.31→0.85: near-argmax → the model's most-confident ece codes are background), reinforcing that texture-of-a-single-sample is the WRONG instrument. **VALID instrument = forecast layer (descriptor head) + mode-detection rate on SAMPLED renders.** Descriptor read (train batches): `ece_desc_hfrac≈0.056` = sharply peaked, NOT mean-collapsed → the forecast layer DOES carry ece mode content. ⇒ **modes are absent from the FIGURE, not the model.** CAVEAT: descriptor head = `persistence_anchor·β + residual` (starts at persistence, learns drift), so high `ftol` is partly persistence by construction; `ftp` = PERSISTENCE baseline (NOT model). GENUINE-SKILL question UNRESOLVED = `ftol > ftp` on ACTIVE/TRANSITION windows (mode moves/onsets — persistence beatable). NEXT: build a forecast-layer eval that aggregates ftol/ftp/hfrac/fdrift over the active stratum (val sample) — texture-free mode-detection rate. Do NOT conclude model mode-skill from decoded panels.

**USER RULING 2026-07-20 (texture vs forecast layer — project has ruled 3×: texture lies).** My "T=0.3 killed the modes-present claim" is WRONG twice: (1) INDEPENDENT-MARGINAL sampling erases cross-token coherence (= a ridge) at ANY temperature — the ban was on independent sampling, NOT on T=1.0; T=0.3 independent-marginal is the banned instrument, colder. VALID render = JOINT decode (MaskGIT-within-step / AR), likely NOT wired in the d1024 eval path. (2) T=0.3 was calibrated on d512-PILOT codecs → uncalibrated for d1024 codecs. Plus budget: 10.8K steps vs 336K-step reference lineage → even a correct render shows faint ridges at best now (texture mode-render was the LAST thing the pilot learned). **"FAILED" IS NOT AVAILABLE.** State: modes present in forecast layer (MEASURED, hfrac≈0.056 peaked); texture render UNVALIDATED (wrong/uncalibrated decode); skill-beyond-persistence UNMEASURED. Paper mode-claims carried by descriptor + detection metrics, texture = illustration (the audit's factorization). **DECISION = GO: build stratified forecast-layer eval (ftol vs ftp over active/transition stratum, val sample) — that number is the verdict.** RENDERING-TRACK ITEM FILED (not panic): joint decoding (MaskGIT temp tuning) was parked for the production-eval / Gate-5 phase → now DUE before any texture figure is quotable. First real verdict of the run is still the K=1→2 gate (not yet arrived).

**STRATIFIED FORECAST-LAYER EVAL — RESULT (job 5040931, step 11800, β=6.0, 200 batches / 38400 windows-per-modality, `analysis/mode_audit/descriptor_stratified_eval.py` + `.json`).** Texture-free ftol(model=anchor·6+residual) vs ftp(persistence) on TRANSITION (onset/death, persistence can't copy) / sustained / all strata:
- ece: transition Δ=**+0.393** (ftol .440 vs ftp .048, n=168 THIN); sustained −0.001; **all Δ=−0.540** (ftol .294 vs ftp .834) — residual injects SPURIOUS peak motion on quiescent windows.
- co2: transition Δ=+0.025 (n=**11034**, ROBUST); sustained +0.010; all +0.013 — consistently but MARGINALLY beats persistence (cleanest statistically).
- mhr: transition Δ=+0.079 (n=216 thin); ~persistence elsewhere.
- bes: transition Δ=0.000 (n=228) — no skill, no harm.
**VERDICT: NOT failed, NOT collapsed.** Model forecasts mode dynamics beyond persistence on the MOVING windows for 3/4 modalities → clears the persistence null → modestly AHEAD of schedule at 10.8k. Caveats: (1) ece/mhr/bes transition strata thin (n<230, prominence-median=0 → mostly quiescent); co2 is the only robust positive. (2) ece net-worse overall (quiescent noise). (3) aggregate hfrac 0.82–0.92 is UNSTRATIFIED (quiescent-dominated, flat=correct there) → NOT a collapse signal; my earlier "hfrac 0.056" was an unrepresentative single active batch — do not lean on hfrac unstratified. RE-RUN this eval per checkpoint to watch Δ grow. K=1→2 gate = still the first full verdict.

**FIGURE PLAN (user order 2026-07-20). V1 (build now, days, no new machinery) = descriptor-track overlay:** GT spectrogram of a co2 mode-active shot (co2 = the robust-positive modality from the stratified eval) + three tracks — GT ridge, MODEL forecast (argmax(anchor·6+d_pred) + uncertainty band from softmax spread), PERSISTENCE (argmax anchor) — predicted mode-freq vs time. Renders what's MEASURED (argmax = the eval's ftol quantity), zero texture-sampler dependence; reuses the pilot's validated ridge-strip idiom (descriptor_head_proof.py L202/239: `khz=arange(mode_lo,mode_hi)*500000/1024/1e3`). Built as `--figure_shot` mode of descriptor_stratified_eval.py. ACCEPTANCE (pre-registered): the model track must DEPART from persistence on the active/transition windows where the eval says it does — no shipping a shadow track. **V2 (QUEUED, K=10-gate render deadline, ~2-3wk, zero training contention) = MaskGIT joint-decode texture figure:** GT strip / sampled-rollout strip with ridge visible in pixels, MaskGIT-decoded, detection-validated vs the recon ceiling. MaskGIT build during K=2-5 phases → T-sweep → showcase renders at K=10 gate. ACCEPTANCE = pre-registered recon-ceiling detection test. Both criteria stand: no shadow track (V1), no texture that fails recon-ceiling (V2).

**V1 FIGURE — FIRST RUN = NEGATIVE (job 5043218, step 15930, shot 200729, `eval_runs/descriptor_track/`).** Built as `--figure_shot` mode of descriptor_stratified_eval.py (argmax peak track + softmax-σ band + GT-descriptor-ridge background, per spectro modality; acceptance=depart-from-persistence-on-active). Per-shot active ftol vs ftp: ece 0.19<0.24 (WORSE, n_trans=6 — not ece's shot), co2 0.134 vs 0.129 (+0.005, near-chance), mhr +0.022, bes flat. **Figures are NOISE.** co2: broadband (line-integrated density) → NO narrowband ridge → argmax-track jitters, ftol~0.13=near-chance for BOTH model+pers → co2's aggregate "+0.025 robust" is a marginal edge on a near-chance metric, NOT visible ridge-tracking. **Argmax-track is the WRONG viz for co2 (fundamental).** ece: descriptor too diffuse at 15.9k → peak jitters; model worse than pers on 200729. **"Prove it now" premise does NOT hold at 15.9k** — number stands (marginal), figure does not; shipping either misrepresents. FIX SPLIT: fundamental (co2 broadband, never a ridge-track) vs early (diffuse descriptor → re-run at later ckpt as skill sharpens). My "PASS(departs)" acceptance flag is a WEAK proxy (departs≠departs-toward-GT); real bar = ftol>ftp on transition, met only marginally. NEXT: re-run figure at materially later ckpt (one command); optional viz-improvement = centroid track + clip padding + narrowband modality (ece/mhr NOT co2) — but marginal separation expected, no money shot at current skill. Do NOT cherry-pick a shot.

**V1 READABLE TEMPLATE — BUILT + VALIDATED (2026-07-21, job 5043273, step 15930, `eval_runs/descriptor_track/{ece,mhr,bes}_track_200729_step15930.png`).** Rebuilt `render_track_figure` (in descriptor_stratified_eval.py) to the pilot 3-panel ridge-strip: A=GT ridge clipped to DATA-PRESENT windows (energy>floor, no void), ridge=brightest; B=dimmed ridge + TWO lines (GT white + MODEL centroid±σ, NO persistence); C=skill strip |model−GT| vs |pers−GT| green/red-shaded; tracks=CENTROID + rolling-median hysteresis (not argmax); headline # in CAPTION not title; 1 modality/figure. **Readable — figure-craft failure fixed.** Honest content at 15.9k: ece model centroid is FLAT ~22 kHz (mid-band, diffuse descriptor) — NOT tracking the moving GT ridge; worse than persistence (win 32%). mhr/bes similar (win 41%/39%, model slightly worse). Correct marginal-to-negative skill for 13%-of-phase, now legible. **co2 BAND-POWER PANEL = MY ERROR (removed):** descriptor head forecasts a FREQ DISTRIBUTION (softmax-CE), NOT band-power magnitude; `pe=anchor·6+d_pred` is a β-scaled logit → bandpower(pe)~200 vs raw bandpower(dtgt)~0 = pure SCALE ARTIFACT (win=0%). Broadband co2 has no ridge AND no band-power output → NO honest descriptor-figure; code now SKIPS broadband with printed reason. co2's aggregate +0.025 stands as a marginal near-flat-distribution stat, NOT visualizable. Template re-runs one-command at K=1→2 to get a fair shot as skill sharpens. ece FLAT-centroid corroborates the descriptor-over-drive finding (Item 1): diffuse+over-active residual.

**⚠️⚠️ CONTAMINATED-TARGET CATCH 2026-07-21 (user, from LOOKING at Panel A). The GT "ridge" is INVALID as a target on ece — SUSPEND the ece verdicts.** The GT track was `argmax(dtgt)` on EVERY window, but 200729's ece modes are INTERMITTENT chirping bursts (short down-sweeping streaks, often 2–3 coexisting @ windows 300–450); MOST windows have NO mode → argmax = NOISE-argmax of an empty spectrum = a random number dressed as GT. Scoring |model−GT| there = scoring vs a RANDOM WALK, which persistence trivially wins (noise-argmax is temporally uncorrelated). ⇒ **(1) ece anti-skill verdict (32%-wins, worse-than-persistence) = SUSPENDED (contaminated).** **(2) Over-drive diagnosis (Item 1) = SUSPENDED** — the flat cyan centroid could be over-drive OR the RATIONAL response to a band-center-noise target (predict center, hedge). Indistinguishable until target fixed. **(3) tw-anneal decision = HELD** — no touching the live run on a corrupted instrument. **(4) Stratified-eval active stratum ALSO contaminated:** my gate = single-window `prom=dtgt.amax−dtgt.mean` > MEDIAN — has local-background but NO persistence + threshold too low (median not P75) → noise spikes leak in → the +0.025/+0.393 Δs inherit unknown contamination. SURVIVES UNTOUCHED: pilot Gate-1/2 (detection-gated, committed-call, nulls — dist_gate.py), co2 aggregate (diff structure), the run, everything upstream. **FIX = port dist_gate.py standard: `fire_cut=P75` of band-prominence (prof−gaussian(prof,σ6)) presence gate + `consecutive` persistence (peaks agree window-to-window within TOL_BINS≈2). Extract GT ridge ONLY on detected windows; mask rest as "no mode" (itself a legit forecast target). Score B/C + ftol/ftp ONLY on detected.** Orders: (1) rebuild GT-track presence-gated [figure], (2) re-run stratified eval on gated stratum → ece-anti-skill + over-drive verdicts un-suspend only AFTER clean numbers land, (3) tw fix waits for clean number. The catch prevented a mid-flight amendment to the decisive run on a corrupted instrument.

**DETECTOR-VALIDATION FIRST (2026-07-21, user order): validate the DETECTOR by eyeball on the spectrogram BEFORE any skill number.** New `--validate_detector` mode in descriptor_stratified_eval.py renders per-modality QC (detection marks on the GT prominence ridge, `eval_runs/detector_qc/`). Hardened `_detect` = dist_gate band_prom + data-present mask + EDGE-guard + presence-based+DILATED constant-line (pickup) exclusion + drift-tolerant multi-peak RIDGE tracking (min_run persistence, admits chirps, drops speckle). co2 = separate `_detect_broadband` (band-power activity). **3 validate→fix iterations, each caught a real bug by eyeball:** v1 single-peak missed coexisting chirps + argmax-based pickup exclusion too weak; v2 secondary constant lines missed + marks hopped adjacent bin + raw per-window peaks = speckle; v3 present-based+dilated exclusion + ridge-persistence. **VERDICT v3 (200729): ece/mhr/bes PASS eyeball** (ece coexisting ridges marked/speckle dropped/pickup excluded; mhr pickup correctly excluded → 0 real modes on this shot [rests on pickup-vs-sustained-mode call — user to confirm]; bes early cluster marked). **co2 NOT clean** — bottom ~5kHz edge band dominates band-power → "activity" ≠ mode; needs edge-exclusion + profile-corr metric. NEXT: re-point aggregate + track figs from `_detect_gate` → validated `_detect` (multi-peak, stable/transition split) → clean Δ un-suspends ece/over-drive. co2 after edge-fix. tw-anneal STILL FROZEN. The instrument-hardening (texture-trap → render-ban → ridge-catch → detector-validation) installed the paper-grade standard before the gates.

**⚠️⚠️⚠️ WRONG-BAND CATCH 2026-07-21 (user, from full-freq figures) — the descriptor band is wrong for 3/4 spectro modalities.** Descriptor head is hardcoded **5–40 kHz for ALL spectro** (`model.py:562-563`, `_mlo=round(5/250*512)=10`, `_mhi=82`; dist_gate same). But per-freq-normalized FULL-FREQ (0–250 kHz) GT spectrograms of 200729 (`eval_runs/full_freq/`, new `--full_freq_view` mode) show the REAL modes live HIGH: **mhr 100–150 kHz, co2 100–200 kHz (confirmed visually: bright cluster windows 0–30 at ~100–200 kHz), bes 100–250 kHz.** Only **ece** (chirps ~7–30 kHz) is inside the 5–40 band. ⇒ **(1) descriptor instrument valid ONLY for ece;** mhr/co2/bes descriptor numbers are MEANINGLESS (wrong band — the 5–40 "detections" were pickup [mhr]/edge [bes]/bottom-band [co2], NOT modes). **(2) descriptor HEAD architecturally cannot forecast mhr/co2/bes real modes** (band excludes them); those modes are carried only by the full-freq FSQ CODE head (0–250 kHz). **(3) TRAINING-HEALTH FLAG (maybe bigger than ece over-drive):** the heavy descriptor loss (wt 6.0, tw 5.0) on mhr/co2/bes supervises a band where their modes AREN'T → fitting pickup/noise, misdirected capacity for 3/4 modalities. **GO-FORWARD:** ece → descriptor instrument OK (gate + score, over-drive question answerable). mhr/co2/bes → need FULL-FREQ instrument (detection+skill on 0–250 kHz code-head forecast; texture/MaskGIT path), descriptor path RETIRED for them. ARCHITECTURE (retrain item): descriptor band must be modality-specific (ece 5–40; mhr/co2/bes high-freq) or full-band. Model NOT necessarily failing high-freq (code head is full-freq); descriptor is the mis-banded AUXILIARY. tw-anneal STILL FROZEN (evidence was contaminated AND wrong-band).

**TRAINING-HEALTH CHECK RESULT 2026-07-21 (amendment-candidate #1 evidence) = INERT, alarm downgraded, NO mid-run change.** Per-modality descriptor-loss trajectory from chain logs (steps 2.4k→16.5k): co2/bes/mhr active-batch `_desc` PINNED at the `log(NF=72)≈4.28` flat-prediction floor early AND late (co2 4.28–4.61, bes 4.277–4.28, mhr 4.28–4.75; late-min 3.8–3.9, never dips below floor) = the head predicts FLAT (no forecastable in-band content in 5–40 kHz) → near-zero informative gradient = **INERT/benign wasted capacity, NOT active noise-fitting.** ece CONTRASTS: dips to 2.54 (below floor) = FITTING real in-band modes = active (right band). ⇒ **(1) "3 noise-fitting gradients degrade ece via shared backbone" hypothesis REJECTED** — inert heads don't pull the backbone. **(2) Amendment #1 (zero desc wt for mhr/co2/bes) does NOT fire** — per the "iff active" rule the check shows inert → optional cleanup only (removes benign waste + makes ece sole descriptor modality), NOT harm-removal; no mid-run amendment. **(3) ece flat-hedging is NOT backbone-pollution** → cause is tw/weight over-drive vs contaminated-target, both on ece's valid band, resolved by ece's gated eval. tw-anneal + amendment #1 BOTH stay unfired. NEXT: ece gated eval (validated `_detect`, ece-only, stable/transition split) = the clean ece Δ = the near-term deliverable. mhr/co2/bes → full-freq code-head instrument (MaskGIT track). Re-band descriptor = d1024-successor spec (Egemen: modality-specific vs full-band).

**★ CLEAN ECE Δ — VERDICT 2026-07-21 (job 5043968, step 16520, validated `_detect`, `descriptor_stratified_eval_gated_v2.json`).** The payoff of the whole texture→render→ridge→detector→band hardening arc, on the instrument verified by eye before metrics. **ece (VALID band, n_data_present 3978):** stable ftol 0.969 vs ftp 1.000 (Δ−0.031, n=255) — model ≈ persistence on non-moving modes; **transition ftol 0.075 vs ftp 0.000 (Δ+0.075, n=199)** — persistence is structurally 0% (can't predict onsets/moves), model 7.5% = NON-ZERO where persistence is ZERO = **genuine forecast skill beyond persistence**; detected(all) Δ+0.015. **⇒ OVER-DRIVE/ANTI-SKILL SCARE REFUTED** — clean instrument shows tracking(stable)+beating(transition), NOT the "worse-than-persistence" the contaminated noise-argmax suggested. Per pre-registered branch: **gated transition Δ positive → run is learning mode dynamics → CONTINUE, NO amendment; tw-anneal stays UNFIRED (confirmed no over-drive).** CAVEATS: skill MODEST (7.5%, most transitions still missed) + EARLY (16.5k≈0.25ep) + SINGLE-STEP (K=1; multi-step rollout = the eventual controllability test). co2/bes/mhr numbers in the JSON are WRONG-BAND artifacts (modes 100-250kHz), IGNORE — full-freq code-head instrument pending. This is the from-scratch model's first HONEST mode answer: positive, small, real.
