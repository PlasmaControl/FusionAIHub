# IGNITE spectrogram mode-loss audit — REPORT

**Question.** The FSQ world model predicts spectrogram codes of a frozen adversarial
codec; modes (tearing modes / AEs, 5–40 kHz) are missing from predictions. Is the
blocker (a) codec faithfulness, (b) CE class imbalance, or (c) representation loss
upstream of the code head — and what is the single highest-value next intervention?

**Answer (headline).** The codec is **faithful and decode-stable**; class weighting is
**over-provisioned**; the problem is **target-side**: the exact FSQ-code target is an
**intrinsically jittery, redundant** representation of modes, so exact-code cross-entropy
forces the world model to hit an unpredictable, arbitrarily-chosen member of a large
equivalence class of codes-that-decode-to-the-same-mode. It collapses to the code-space
conditional mean → dampened/absent modes. The jitter is the tell: a 0.5 ms shift of the
same plasma state scrambles ~74% (ECE) / ~82% (active CO2) of code dims, so the codes are
dominated by **STFT-phase/realization bits** the reconstruction codec faithfully preserves
but no model can forecast. **Fix = make the codec encode STATISTICS, not realizations, and
gate on the oracle before any world-model training. Not model size, not class weights, not
the codec's fidelity.**

---

## Ground truth (from the checkpoint — NOT memory)
Printed by `analysis/mode_audit/checkpoint_facts.py` → `ground_truth*.json`. This header
exists because carried-memory facts went stale THREE times this session (ECE missingness,
backbone size, and which model/codec is production). TWO distinct models matter:

**(A) PRODUCTION model — d1024/48L FSQ** `e2e_stage1_allshots_b32_resid/e2e_stage1_latest.pt`
(step 4800) — `ground_truth_d1024_fsq.json`:
- d_model 1024, n_layers 48, n_heads 8 · use_spectro [ece,co2,bes,mhr] · use_video
  [tangtv_lower,tangtv_upper] · chunk 50/step 10/horizon 50 ms · lr 7e-4 · batch 32.
- **1,188 M params:** backbone 609.1 · diag_tokenizers 478.6 (ece 121.9/bes 109.3/mhr
  104.1/co2 103.1/filterscopes 36.9/tangtv 1.5×2) · diag_heads 89.7 · act 10.9.
  **spectro tokens = 96/modality** (patch 32×16).
- **spec codec = `fsq_spectro_residual_codecs` (patch 32×16 COARSE)**, fsq_dim48/L16,
  bg_subtract True, smooth_frames None · **spec_code_class_weight = 4.0** · spec_generative
  False · video+fast-TS+slow-TS FSQ codecs all wired · freeze_* 0.

**(B) AUDIT Task-3 world model — d512/12L proof** `e2e_step2_fsq_finer` (step 4000) —
`ground_truth.json`: d512/12L, use_spectro [ece] only, ece=384 tokens, **codec
`fsq_resid_p8_all` (patch 8×16 FINER)**, **class_weight 20**. 109.5 M params.

⚠️ **Model mismatch (correct any transfer of numbers):** the codec-side tasks (0/1/2/5/6/7)
were measured on the **FINER 8×16** codec; the world-model Task 3 on the **d512 proof**.
**PRODUCTION runs the COARSER 32×16 codec at cw=4.** The MECHANISM conclusion (codes encode
realizations → unpredictable → encode statistics) is codec-objective-generic and holds for
both; the SPECIFIC numbers (stability 0.26 / capture 0.69) are the finer-codec's and must be
re-measured on the production 32×16 codec. Also: cw=4 (production) is BELOW the data-driven
inverse-freq max (ece 10.5 / co2 15.7 / mhr 13.8) — so "cw over-provisioned" (Task 1) is true
only for the proof's cw=20, NOT production.

## Scope, method, limitations
- Frozen finer codec `fsq_resid_p8_all` (patch 8×16, residual/bg_subtract, dim 48, L 16).
- World model = `models/e2e_step2_fsq_finer/e2e_stage1_latest.pt` — audited (below): a
  genuinely-trained, cold-start CE-code-head checkpoint (d512/12L, 4000 steps, ece-only).
- Diagnostic only — **no training, no model/loss/rollout edits**; new scripts under
  `analysis/mode_audit/`; each writes JSON + PDF.
- Mode band 5–40 kHz, prominence-above-`gaussian(σ=6)`-baseline detector (the ranker's).
- **No human per-window mode labels exist** → mode-positive/-free is detector-derived
  (relative top/bottom quartile of band prominence — the absolute z-threshold over-fired
  in residual space: every ece/bes window read as mode-active). This is a limitation:
  ece and bes have **no genuinely quiescent windows**, which makes some tests ill-posed
  for them (noted where relevant).

## Checkpoint identity audit
Verified the checkpoint is a real trained CE-code-head, not an MAE ckpt with a fresh head:
`spec_generative=False`; code-head predictor weights present (trunk 512² + 48 per-dim
16×512 logit heads) and trained (training drove ce→0.088, codeacc→0.91 on easy batches —
impossible for a fresh head); no `init_from`/`resume_from` (cold). Eval output is
structured, not random → weights loaded, not silently reset. **Task 3's low mode-codeacc
is real.** Caveat: small/brief proof model, so absolute codeacc is partly undertraining —
but the *pattern* (below) is decisive independent of scale.

---

## Task 0 — metric confound check ✅ clean
decode(encode(GT)) on mode-free windows, does the detector false-fire (patch-grid
checkerboard)? **FP rate 0.0 for every modality that has mode-free windows (co2, mhr, and
ece/bes under the quartile split).** No patch-grid alignment. **The mode metric is
trustworthy — not confounded by the ConvTranspose checkerboard.**

## Task 1 — code histogram + derived class weights
| modality | per-dim top-1 | dominant tuple | inverse-freq weight (max/mean) | eff-num max |
|---|---|---|---|---|
| ece | 0.107 | 1.5% | 10.5× / 1.0 | 4.0 |
| bes | 0.127 | ~0% | 14.7× / 1.0 | 14.4 |
| co2 | 0.296 | 30% | 15.7× / 1.0 | 14.8 |
| mhr | 0.654 | **65%** | 13.8× / 1.0 | 9.4 |

Imbalance is modality-specific (ece/bes diverse; co2 notable; mhr severe). **But the
current flat `class_weight=20` already exceeds the data-driven inverse-freq max for every
modality** → class weighting is **not under-provisioned**. Ruled out.

## Task 2 — splice faithfulness ✅ faithful
Graft mode-patch codes into a mode-free grid (forward) / erase them (inverse):
| modality | forward | inverse |
|---|---|---|
| co2 | 1.00 (z) / 0.50 (quartile) | 1.00 |
| mhr | 0.90 (z) / 1.00 (quartile) | 1.00 |
| bes | 1.00 (quartile) | 1.00 |
| ece | 0.00 (forward ill-posed*) | 1.00 |

\*ece has no genuinely mode-free windows to graft onto, and a grafted mode can't clear
ece's high top-quartile prominence bar against an already-active background (figure
`task2_ece_pair0.pdf` shows chimera ≈ free). Inverse passes. co2/mhr/bes pass both;
same per-patch decoder architecture. **Codes causally control mode content — codec faithful.**

## Task 3 — k1 teacher-forced render triad (ece, 20 strongest-mode windows)
| render | mode-capture | peak-match | profile-corr | tvr |
|---|---|---|---|---|
| argmax | **0.00** | 0.20 | 0.59 | 0.24 |
| independent sample (T=1) | **−0.04** | 0.20 | 0.41 | 0.35 |
| **GT-codes** | **0.69** | **0.95** | **0.92** | 0.41 |

codeacc: **mode-patch 0.097 vs background 0.127** (both ~2× the 1/16 random floor).
Figure `task3_ece.pdf`: GT-codes recover the ~8 kHz ridge; **argmax dampens it to a
low-freq smear; sample scatters incoherent blocks at wrong frequencies** (speckle).
Interpretation row selected: *codec fine (GT-codes good) + argmax-deletes + sample-speckles*
→ refined by Tasks 5/7 to **redundant-code / wrong-loss** (see below), not a capacity or
imbalance failure.

## Task 4 — persistence oracle (ceiling)
Confirmatory (rerun `4982455` in flight; core numbers already produced by Tasks 5/6):
persistence codeacc (copy codes t→t+1) on **active** windows ≈ **0.10** (co2 0.098,
ece 0.119) — i.e. the model's ~0.10 mode-codeacc **already matches the persistence
ceiling**. On **quiescent** windows persistence ≈ **0.99** (co2). The model is not
under-performing an achievable exact-code target on modes; the achievable target is ~0.10.

## Task 5 — stability (0.5 ms pre-STFT time shift) — **not OOD, intrinsic jitter**
codeacc between encode(GT) and encode(GT shifted 1 frame ≈ 0.5 ms). Random = 0.062.
| modality | window set | in-subset | out-subset |
|---|---|---|---|
| ece | all | 0.256 | 0.277 |
| ece | active | 0.255 | 0.276 |
| co2 | all | 0.434 | 0.510 |
| co2 | active | **0.179** | **0.177** |

Out-subset ≥ in-subset → **not codec-OOD scatter** (codec no worse on unseen shots).
On mode-active windows a negligible 0.5 ms shift flips **74–82% of code dims** → the
**exact-code target is intrinsically jittery**, uniformly in and out of the codec's
training set. (co2's "all" 0.43–0.51 is inflated by shift-stable quiescent background.)

## Task 6 — bimodality scatter — **"easy" = quiescent, not subset**
Per-window persistence codeacc vs residual band-variance (`task56_{ece,co2}.pdf`):
| modality | quiescent | active | corr(codeacc, activity) |
|---|---|---|---|
| ece | 0.128 | 0.119 | −0.52 |
| co2 | **0.990** | **0.098** | −0.92 |

The training-time bimodal codeacc (~0.9 "easy" / ~0.1 "hard") is **quiescent-vs-active**,
not easy-vs-hard-prediction and not in-vs-out-of-subset (points overlap). The headline
codeacc was inflated by trivially-persistable quiescent windows; on real modes it sits at
the ~0.10 jitter floor.

## Task 7 — decoded-output stability — **the decider: codes redundant**
decode(encode(GT)) vs decode(encode(shifted GT)), band-corr, active windows:
| modality | decoded-stability | code-stability | recon-fidelity | verdict |
|---|---|---|---|---|
| **ece** | **0.933** | 0.258 | 0.622 | DECODE STABLE → codes redundant → **loss fix** |
| co2 | 0.706 | 0.178 | 0.406 | DECODE MODERATE (loss fix + weaker co2 codec) |

For ece (the deployed modality): the **decoded spectrogram is 93% stable while the codes
are only 26% stable** → many different code-sets decode to the same mode → **the codes are
a redundant, overcomplete representation.** co2 is moderate (0.71) and its recon is weaker
(0.41) → co2 additionally needs a better codec, but ece is clean.

---

## Interpretation (which row the evidence selects)
> *GT-code render is fine + model argmax deletes / sample speckles + mode-patch codeacc ≈
> background ≈ persistence ceiling (~0.10) + code jitter intrinsic (not OOD, not imbalance)
> + decode stable despite code jitter (codes redundant).*

**Mechanism.** For each mode there is a large equivalence class of code-sets that all
decode to it. Exact-code CE forces the world model to reproduce **one arbitrary,
jitter-selected member** — an unpredictable target (0.26 shift-stability). Failing that,
its per-dim argmax settles on the **code-space conditional mean**, which decodes to a
dampened/absent mode (the mean-collapse problem, relocated from pixel space into code
space). Independent sampling instead scatters incoherent blocks (speckle). Neither is a
capacity, imbalance, or codec-faithfulness failure — it is a **wrong-objective** failure.

## THE single recommended next intervention (NOT implemented — plan only)
**Make the codec encode STATISTICS, not REALIZATIONS — then re-run the oracle as the
acceptance gate BEFORE any world-model training touches it.**

Root cause, stated exactly: a +1 STFT-frame shift is **0.5 ms of the same physical plasma
state**, yet it scrambles **~74% of ECE code dims (stability 0.26)** and **~82% on active
CO2 windows (0.18)**. The current adversarial *reconstruction* codec is trained to
reproduce the exact 2-D spectrogram, so it dutifully spends code capacity on
**STFT-phase / realization bits** — which are, by construction, unpredictable 50 ms ahead
(they don't even survive a half-millisecond shift of the *input*). These are not dynamics
targets; they are realization noise. No world model can or should predict them.

The fix is therefore **codec-side representation, not the world-model predictor**: retrain
(or re-target) the spectro codec so its codes encode the **shift-invariant statistics** of
the window — mode frequency, amplitude/band-power, envelope — rather than the exact
realization. Candidate directions (to design, not yet build): a shift/phase-invariant
target (e.g. power/PSD-domain or magnitude-statistics reconstruction, time-pooled within
the window), so that a 0.5 ms shift maps to (near-)identical codes.

**Acceptance gate (the oracle, re-run on the NEW codec) — pass BEFORE training a world model:**
1. **Stability ≥ ~0.8** on active windows (codes survive the 0.5 ms shift) — the primary gate.
2. **Persistence oracle on active windows ≫ 0.10** (ideally ≳ 0.5) — codes now carry a
   forecastable, dynamics-bearing signal.
3. Codec still **faithful** (Task-2 splice) and reconstructs modes (recon ≥ current).

**Kill criterion.** If a statistics-targeted codec's codes **still don't survive the 0.5 ms
shift** (stability stays ~0.26 on active windows) or the oracle stays ~0.10, the new target
is still realization-bound → the statistics parameterization is wrong; **do not proceed to
world-model training** — rethink the invariant. Only once the gate passes does exact-code
prediction (or code-CE) become a well-posed objective worth a world-model run.

*(Interim workaround, NOT the primary rec, if a codec retrain is not yet possible: train
the world model on a decoded mode-band perceptual loss via a soft expected-code-embedding
decode through the frozen decoder — Task 7 shows decode is 0.93-stable for ece, so this is
realization-tolerant. This treats the symptom; the codec-statistics fix removes the cause.)*

## Explicitly ruled out (do NOT spend budget here)
- **Codec redesign for better FIDELITY** — the codec is already faithful (Task 2) and
  decode-stable (Task 7); it reconstructs modes fine. (The recommendation is a *different*
  axis: change WHAT it encodes — statistics vs realization — not how well it reconstructs.)
- **Higher class weights** — cw=20 already exceeds data-driven inverse-freq (Task 1).
- **Bigger / longer code-CE predictor, or joint decoding (MaskGIT) alone** — all still
  target the unpredictable exact-realization codes (Tasks 3/5/7); MaskGIT already failed once.
- **Codec-OOD retraining** — no in/out-of-subset gap (Task 5).
- **Any world-model training on the current codes** — blocked until the oracle gate passes.

## Concrete pre-registered plan (do NOT implement yet)
**Step 1 — Denoise the magnitude before encoding.** Temporal averaging of `|STFT|` across
adjacent frames (or Welch-style segment averaging within the 50 ms window), then retrain the
FSQ codec on the smoothed representation. Rationale: coherent ridges (tearing modes, AEs)
are exactly the content that survives temporal averaging; STFT-phase/realization speckle is
exactly what dies. This is the operational form of "encode statistics, not realizations."
Averaging degree is the knob; escalation ladder = smooth harder → toward full within-window
time-pooling (per-window PSD) → if still not predictable, the factorization option
(separately encode predictable statistics vs discard realization). Note: complementary to
the existing *frequency* baseline-subtraction (residual codec) — this adds *time* smoothing.

**Step 2 — Pre-registered gate on the new codec (NO world model):**
1. **Stability ≥ ~0.9** on active windows (a 0.5 ms shift must be near-invariant — the
   definition of encoding structure not realization).
2. **Quiescent persistence ~0.99 retained** (don't break the easy background).
3. **Active-window 50 ms persistence well clear of chance** — if smoothed active persistence
   is still ~0.1, smooth harder or invoke the factorization option.
4. **Faithfulness preserved** (already instrumented): gt-codes render **mode-capture ≥ 0.69**
   (the current ceiling — smoothing must NOT drop it) + splice test on the new codec.

   ⚠️ **The crux/risk = the stability↔fidelity tradeoff.** More smoothing → higher
   stability/predictability but lower mode-capture; less → the reverse. The plan passes only
   if a smoothing level exists that is simultaneously stable (≥0.9) AND still renders the
   mode (capture ≥0.69). Whether that sweet spot exists is the empirical question the gate
   answers — physically favorable because mode FREQUENCY persists 0.85–0.99 over 400 ms
   (longmode_shots), i.e. the *statistics* are forecastable at 50 ms even though the
   realization is not; but not guaranteed. The pre-registered gate + escalation ladder is
   exactly the right way to find out without committing a world-model run.

**Step 3 — Only after the gate passes:** retrain the code head, re-run the Task 3 triad
**verbatim**. Success is redefined: **head codeacc ≈ the NEW oracle** (model back at the
ceiling — but the ceiling now *contains* the modes) **AND** argmax/sample renders that
**keep the ridge** (judged by eye, per standing rule). Raw codeacc in isolation is no longer
the criterion.

## Persistence-tol1 decision test (s16, 50 ms) — AMBIGUOUS, no branch picked
`persistence_tol_s16.py` (job 4984283), frozen s16 codec, encoder-only. 50 ms pair =
window i vs i+5 (step 0.01 s); stratum by target(i+5) band-prominence quartile; stats =
sweep-default `preprocessing_stats.pt` (s16-matched). Files: `persistence_tol_s16.json` + `.pdf`.

| cell | n | exact | tol1 | tol1-chance | shuffled tol1 |
|---|---|---|---|---|---|
| active in | 646 | 0.123 | 0.320 | 0.255 | 0.301 |
| active out | 225 | 0.135 | 0.356 | 0.278 | 0.339 |
| quiescent in | 460 | 0.130 | 0.346 | 0.283 | 0.328 |
| quiescent out | 393 | 0.144 | 0.377 | 0.290 | 0.356 |

**Headline — mode-band active tol1 = 0.336** (n=871; exact 0.133; tol1-chance 0.213; shuffled 0.247).
Lag curve (active tol1): 50 ms 0.329 → 100 0.327 → 200 0.323 → 400 0.320 (**flat** — no
decaying dynamics). Per-dim tol1: 0.296–0.375, **uniform** (no persistent subset → does NOT
support dim-weighted CE). `persistence_tol_s16.pdf` = lag curve + per-dim.

**VERDICT (pre-registered rule): tol1 = 0.336 ∈ [0.30, 0.50] → AMBIGUOUS → report + STOP, pick
no branch.** Reading: signal-above-shuffled ~0.09 (non-zero, so not a clean floor / not the
≤0.25 "retrain" call) but far below the ≥0.5 "skip-retrain" call; the flat lag curve + uniform
per-dim say the small signal is near-static background, not forecastable mode dynamics.
**Lean (NOT a decision):** ordinal tolerance alone does not carry 50 ms mode prediction → a
shift-stable-statistics codec is likely still needed (possibly combined with ordinal CE). No
retrain launched. Decision deferred to the user per the pre-registered AMBIGUOUS guard.

## Artifacts
`analysis/mode_audit/`: `tasks012_*.json`, `task3_ece.{json,pdf}`, `task56_{ece,co2}.{json,pdf}`,
`task7_decstab_*.json`, `task2_ece_pair0.pdf`; scripts `codec_tasks.py`, `triad_task.py`,
`persistence_oracle.py`, `stability_scatter.py`, `decoded_stability.py`.
