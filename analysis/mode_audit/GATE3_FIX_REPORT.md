# GATE 3-FIX REPORT

**Status: COMPLETE (fix → disambiguation → anchor-β anneal). Awaiting user read. Gate 3 is NOT struck.**
Date: 2026-07-15. Checkpoint: `/lustre/orion/fus187/proj-shared/models/e2e_g3fix/e2e_stage1_best.pt` (val_loss 1.1399 @ step 5750).
**LATEST (§7 anneal + §8 sign-confirmation): the anchor-β anneal UNMASKED pin conditioning at the output —
controllability demonstrated as a TRADE-OFF DIAL, with the direction now n-confirmed at β=6.**
Operating point β=6: pin dfreq **−0.0057** (n=967, bootstrap CI [−0.0071,−0.0045], excludes 0, AE-correct),
placebo-separated, and CONCENTRATED in AE-active shots (200729 −0.042). false-death 0.003, no peak-in-tol regression.
Response grows toward β=3 but false-death crosses the 0.01 gate at β≈5, where the sign also goes INCOHERENT (knee
instability) — so β=6 (above the knee) is the honest operating point. No β meets the full ΔLL+dfreq+false-death bar
→ **PARTIAL** (β=6 supports FREQUENCY-SHIFT conditioning; ΔLL below floor). Money figure → 200729 @ β=6.
Artifacts: `eval_runs/anneal_beta_sweep/` (`beta_tradeoff_curve.png`, `nsign_b{6,5}.0/`). Read §7+§8 for verdict + caveats.
Verdict artifacts: `eval_runs/gate3_fix_after/{act_cf.json, gate2b.json, gradnorm_curve.png, dLL_waterfall.png, train_chain.log}`
and `eval_runs/gate3_fix_disambig/{act_cf.json, resid_specificity.png}` (residual-level disambiguation).

**HEADLINE (revised after disambiguation):** the scale fix worked *more deeply than the output ACT_CF showed*.
At the β=8 anchored OUTPUT, forecast conditioning looked dead (ΔLL ~1e-6, §2c) — but that was a **measurement
artifact of a near-saturated anchored softmax**. At the RESIDUAL level (pre-anchor, §2e), the clean primary **`pin`
conditions specifically and directionally-correctly** (‖Δresid‖ 3.7× above the placebo band, freq-shift toward lower
frequency under +pin = AE-drive physics). `latent_conditioning = True`. The controllable signal EXISTS and is masked
by the persistence anchor at the output. It is **real, specific, but small in magnitude** — not yet a demonstrated
controllable forecast. The next move is a training change (anchor-weight annealing), gated on the user.

---

## 1. What was tested (pre-registration recap)

Gate 3 diagnosed that actuator conditioning was **dead at the input**: raw actuators (`ech_power` ~O(1e5), `beam_voltage`/`rmp` similar) entered the backbone unstandardized, so their tokens contributed ~0 to `tok[ece]`. Gate-3-FIX = the single localized fix + verification retrain:

- **Drop 3 ECH angle channels** (`ech_tor_angle`, `ech_pol_angle`, `ech_polarization`) — globally zero in the corpus.
- **`ech_power` → `log_standardize`** — the one scaling change under test.
- `rmp` left **raw** (reverted; unit-mismatch, reported-not-claimed).
- `beam_voltage` left raw. Recipe **B**: warm-start from `t4mh`, **fresh act tokenizers** (`--reinit_act_tokenizers`, 7 channels), **backbone UNFROZEN**, t+4 multi-horizon, ~6000 steps.

Pre-registered "conditioning alive" bar: a **primary** actuator (ech_power, pin) must move the descriptor forecast (|ΔLL@true-bin| ≥ ~100× the 3.5e-6 noise floor ⇒ ≥ ~3.5e-4) AND **placebos** (gas_flow, gas_raw) must stay silent. Regression hard gate: **false-death ≤ 0.01** at both horizons.

---

## 2. Results

### 2a. Input → backbone token path: **FIXED** ✓
Standardizing `ech_power` brought it from raw O(1e5) to O(1), and its influence on `tok[ece]` (+5σ perturbation) went from **dead → live**:

| channel | +5σ \|Δtok[ece]\| BEFORE (raw) | AFTER (log-std) |
|---|---|---|
| **ech_power** | **6.30e-04** (dead) | **5.99e-02** (~95× ↑) |

The 3 angle channels were confirmed globally zero (mean=std=absmax=0, Δtok=1.4e-6) → dropped. **Actuators now reach the backbone.** The diagnosed Gate-3 root cause is genuinely addressed.

### 2b. Actuator-tokenizer grad-norm proxy: **CONFOUNDED, not decisive**
`act_tok_gradnorm` over the retrain: first-3 mean 3.51e-2 → last-3 mean 5.19e-3 (median 9.07e-3). It **declines**, consistent with fresh-tokenizer convergence rather than rising attention — as pre-registered, this proxy cannot distinguish "backbone re-attends" from "tokenizer just settles." Not used as a verdict. Fig: `gradnorm_curve.png`.

### 2c. ACT_CF at the β=8 anchored OUTPUT: looks dead / non-specific — but this is a MEASUREMENT ARTIFACT (see §2e)
Perturbing each actuator (±σ-scaled) and measuring the descriptor forecast at t+4 (n=158, shots 199597/199607/200729/191001).
The output softmax is near-saturated at β=8, so ΔLL under-reports the residual's actuator response — §2e is the decider:

| channel | ΔLL @ true bin | flag |
|---|---|---|
| **ech_power** +2σ (primary) | **+2.0e-6** ±1.9e-6 | ~unchanged vs pre-fix; ≪ 3.5e-4 floor |
| **pin** +2σ (co-primary) | **−1.2e-4** ±1.5e-4 | **ns** (CI spans 0) |
| gas_flow +2σ (placebo) | **−5.8e-5** ±4.4e-5 | fires — **larger than ech_power** |
| gas_raw +2σ (placebo) | −1.2e-5 ±1.6e-5 | ns |

**`conditioning_alive = False`.** Every response is 1e-6–1e-4 (≪ the 3.5e-4 alive floor), and the placebo `gas_flow` (5.8e-5) exceeds `ech_power` (2e-6) and rivals `pin` — **no specificity**. Fig: `dLL_waterfall.png`.

### 2d. Regression battery: **PASS** ✓
| horizon | false-death | sub-threshold signal | beats momentum heuristic (both subsets) |
|---|---|---|---|
| t+2 | **0.000** ✓ | dLL(model−anchor)=0.031±0.012, mass-shift dir-acc 0.603[0.548,0.655] | mom-correct 0.701 / mom-wrong 0.514 → **False** |
| t+4 | **0.000** ✓ | dLL=0.044±0.013, mass-shift dir-acc 0.623[0.576,0.669] | mom-correct 0.615 / mom-wrong 0.611 → **True** |

The retrain introduced **no false-death regression** (hard gate met at both horizons) and **preserved** the closed Gate-2b findings: a real sub-threshold mass-shift signal, and t+4 still beats the momentum heuristic on both subsets. Peak-in-tol still does not beat persistence (0.72 vs 0.73 @ t+2; 0.562 vs 0.560 @ t+4) and commits are ~0 — the head remains persistence-anchored, unchanged from g2/t4mh.

### 2e. DISAMBIGUATION — residual-level ACT_CF (pre-anchor): **latent conditioning CONFIRMED for pin** ✓
The §2c output measurement cannot separate "residual inert" from "residual responds but the β=8 anchor masks it."
Resolved by measuring the **pre-anchor** head change `dh(tok_perturbed) − dh(tok_real)` directly (t+4, n=158). Artifacts: `eval_runs/gate3_fix_disambig/`.

| channel | ‖Δresidual‖ (RMS, pre-anchor) | vs placebo band (4.6e-4) | residual freq-shift |
|---|---|---|---|
| **pin** (clean primary) | **1.97e-3** ±2.5e-4 | **3.7× above** (CI-separated) | **−0.019 ±0.003 bins*** (→ lower freq, AE-correct) |
| ech_power (primary, aiming-gap) | 1.5e-5 ±2e-6 | below band | ~0 |
| gas_flow (placebo) | 3.7e-4 ±0.9e-4 | (in band) | −0.003 |
| gas_raw (placebo) | 1.0e-4 ±0.2e-4 | (in band) | +0.001 |

**`latent_conditioning = True` (pin).** The clean, fully-populated actuator moves the residual specifically (pin ≫ both placebos, CI-separated) and in the **physically-correct direction** (mass toward lower frequency under +pin, matching AE drive). Lower-β corroboration (β=2, OOD — not a decider): pin's output response *grows* as the anchor weakens (ΔLL −1e-4→−4e-4, Δfreq −0.008→−0.017), exactly the signature of anchor-masking. `ech_power` stays inert even pre-anchor — consistent with the pre-registered aiming-data gap (no beam geometry to detect suppression), not a model failure.

---

## 3. Interpretation — the fix worked at the input AND left a real (masked) residual signal

The scale fix worked exactly where it was aimed (**input → backbone: dead → live, ~95×**). The §2c output ACT_CF *looked* dead, but §2e shows that was a **measurement artifact of the near-saturated β=8 anchored softmax**: the residual — the learned, actuator-sensitive part of the head — **does condition on the clean actuator (pin), specifically and in the correct direction.** The persistence anchor masks it at the output, so the forecast the model actually emits is still persistence-dominated.

Two honest qualifiers on the positive result:
- **Magnitude is small.** ‖Δresid‖ ~2e-3 and a residual freq-shift of only −0.019 bins (≈2% of one bin). The signal is real, specific, and directionally correct — but it is a *whisper*, not a large controllable knob. Even fully unmasked, the forecast shift would be small at present.
- **Only the clean channel.** `pin` (AE drive, natively live, fully populated) is where conditioning shows. `ech_power` is inert because its beam-aiming geometry channels are globally zero (data limitation, pre-registered) — so ECH controllability cannot be tested with this corpus, full stop.

**Verdict:** an actuator-conditioned / controllable *forecast* is **not yet demonstrated at the output**, but the underlying mechanism is **present and specific** — the model has learned a (small) pin→mode-frequency dependence that the anchor currently hides. This is materially more hopeful than the output-level read: the blocker is now a known, addressable architectural knob (anchor weight), not absent conditioning.

---

## 4. Next step (RESULT of the disambiguation): unmask the residual via anchor-weight annealing

The disambiguation is done and it points one clear direction. The residual carries a real, specific pin→mode signal that the β=8 anchor suppresses at the output. To turn latent conditioning into a *demonstrable controllable forecast*:
1. **Anchor-weight annealing (training change, gated on user):** schedule β from 8 → a small value over training, so the residual's actuator response reaches the forecast without losing the persistence prior that keeps false-death at 0. Re-run ACT_CF at the OUTPUT afterward — success = pin ΔLL/dfreq clears the alive floor *and* placebos stay silent, with false-death still ≤0.01.
2. **Amplify the signal (optional, same retrain):** the residual response is small; a modest increase in descriptor-head capacity and/or a light actuator-forecast auxiliary loss could grow the pin effect. Keep to one change at a time vs the annealing run.
3. **ECH remains untestable** on this corpus (aiming-gap) — do not spend effort on ech_power controllability until beam-geometry channels are populated; report it as a data limitation.

If annealing brings the pin effect to the output with specificity preserved → the counterfactual triptych (money figure) is back in reach, scoped to **pin/AE drive** (honest, physics-anchored). If it does not → the honest scope is a forecaster with *detectable but not controllable* conditioning (two-panel figure + the residual-specificity plot as the "mechanism is present" evidence).

---

## 5. Figure inventory (this report)
- `eval_runs/gate3_fix_disambig/resid_specificity.png` — **the decider:** residual-level ‖Δresid‖, pin ≫ placebos (CI-separated) → latent conditioning.
- `eval_runs/gate3_fix_after/dLL_waterfall.png` — β=8 output ACT_CF (looks dead — the masked view; keep for the "why the artifact" story).
- `eval_runs/gate3_fix_after/gradnorm_curve.png` — grad-norm proxy (confounded, shown for completeness).
- Pending (GPU render, on go-ahead): descriptor strip on the g3fix ckpt (GT vs pred mode ridge).

---

## 6. Decision gate (for the user)
- **Do NOT strike Gate 3** until you have read this. The retrain is clean (val 1.1399, no regression), the input fix is real, and the disambiguation confirms **latent, specific, directionally-correct conditioning on pin** — masked at the output by the persistence anchor, and small in magnitude.
- Choose next step: **(A)** anchor-weight annealing retrain (§4.1) to unmask the pin signal at the output — the direct path toward the controllable-forecast claim; **(B)** accept the honest scope now (detectable-not-controllable) and design the two-panel + residual-specificity figure; **(C)** other.

---

## 7. ANCHOR-β ANNEAL RESULT — controllability is a TRADE-OFF DIAL (2026-07-15)

Retrain: warm-start g3fix, anchor β annealed 8→6→5→4→3 (1500 steps/hold, flat-ish g3fix recipe, full
7878-shot corpus, cache reused). One equilibrated milestone per β (`models/e2e_g3fix_anneal/beta{β}_step*.pt`).
Each milestone evaluated at its OWN trained anchor β (`DESC_ANCHOR_BETA`). Artifacts:
`eval_runs/anneal_beta_sweep/{actcf_b*,g2b_b*}/`, curve `beta_tradeoff_curve.png`, `beta_sweep_summary.json`.

| β | pin ΔLL@2σ | pin dfreq@2σ (bins) | max placebo \|ΔLL\| | false-death (max t2/t4) | peak-in-tol m/pers |
|---|---|---|---|---|---|
| 8 | −6.8e-5 (ns) | +0.001 (ns) | 2.0e-4 | **0.000** | 0.56 / 0.56 |
| 6 | −2.7e-4 (ns) | **−0.018 \*** | 4.0e-4 | **0.003** | 0.54 / 0.56 |
| 5 | **+3.95e-3 \*** | **+0.019 \*** | 2.6e-5 | 0.021 | 0.56 / 0.56 |
| 4 | **−4.21e-3 \*** | **−0.086 \*** | 7.7e-5 | 0.077 | 0.55 / 0.56 |
| 3 | **−9.27e-3 \*** | **−0.086 \*** | 6.9e-5 | 0.223 | 0.55 / 0.56 |

**The unmask worked.** The disambiguation's prediction — pin conditioning present in the residual but masked
by the β=8 anchor — is confirmed at the OUTPUT: as β drops, pin's output response emerges from masked
(β=8: dfreq +0.001 ns) to significant + placebo-specific + physically-correct (β=6: dfreq −0.018, mode →
lower frequency under +pin, the SAME sign the residual showed), then grows monotonically (β=3: dfreq −0.086,
ΔLL −9.3e-3). Placebos (gas_flow/gas_raw) stay at ~10⁻⁴ throughout — always ≪ pin once unmasked. `ech_power`
is inert at every β (ΔLL 2e-7→1.7e-4 < placebo) — the aiming-data gap, untestable, as pre-registered.

**But it is a trade-off dial, not a free lunch** — exactly the risk flagged ("annealing inflates false-death as it
unmasks pin"). False-death climbs in lockstep with the pin response: 0.000 → 0.003 → 0.021 → 0.077 → 0.223, crossing
the 0.01 gate between β=6 and β=5 — right where pin's ΔLL fully clears the floor. **No single β meets the full
pre-registered SUCCESS bar** (pin ΔLL *and* dfreq clear *and* false-death ≤0.01 simultaneously): at β=6 only dfreq
clears (false-death clean, 0.003); at β=5 both clear but false-death is 0.021 > gate.

**VERDICT: PARTIAL, leaning positive — controllability DEMONSTRATED as a dial.** Best clean operating point =
**β=6**: pin (AE drive) produces a significant, placebo-specific, physically-correct-direction shift in the
predicted mode frequency (−0.018 bins), at false-death 0.003 and no peak-in-tol regression (0.54 vs 0.56,
CI-overlapping). The response is tunable via the anchor weight and strengthens toward β=3 at the cost of
false-death. This is the pre-registered "dial, not knob" outcome, now with a quantified β→response↔false-death curve.

**Honest scope (no overclaim):**
- The clean-operating-point effect is SMALL (0.018-bin shift). It grows 5× by β=3 but only by paying false-death.
- **peak-in-tol never beats persistence at any β** (0.54–0.56 vs 0.56). The model is actuator-*conditioned* but still
  persistence-*dominated* in absolute skill — controllability ≠ forecast-skill improvement. State it as controllability.
- β=5 shows a dfreq SIGN FLIP (+0.019) coincident with its ΔLL-positive regime; the consistent AE-correct negative
  direction holds at β=6/4/3. Flag as an anomaly, not the headline.

**Next (user-gated, NOT auto-run):**
1. Adopt β=6 (clean) or β=5 (stronger, false-death 0.021) as the controllable-forecast demonstration ckpt; build the
   counterfactual money-figure triptych on **pin** at that β (ECH aiming-gap = the honest caveat line).
2. To get a large effect WITHOUT the false-death cost: **FiLM** (conditioning-by-construction) at production scale —
   now with hard evidence it has a real, specific, correctly-signed pin→mode signal to amplify.

**Gate 3 remains NOT struck** — awaiting your read of this result.

---

## 8. SIGN CONFIRMATION (n-enlarged + per-shot) — β=6 CONFIRMED, knee condemned (2026-07-15)

Jobs 5005771 (β=6) / 5005772 (β=5): 15-shot gate2b pool, MAX_WIN 300, n=**967** each, each at its own trained
anchor β, with nonparametric bootstrap CIs + per-shot dfreq breakdown. Artifacts: `eval_runs/anneal_beta_sweep/nsign_b{6,5}.0/`.

**β=6 — pin dfreq direction CONFIRMED (two independent axes):**
- POOLED GATE: pin Δfreq = **−0.0057**, bootstrap95 **[−0.0071, −0.0045]** (excludes 0, negative = AE-correct),
  > placebo band (0.0025). Sign holds at n=967.
- PER-SHOT (interpretation): 10 neg / 4 pos; effect CONCENTRATES in AE-active shots — **200729 −0.042** (canonical
  16 kHz coherent-mode shot), 191001 −0.024, 200000 −0.022 — quiet shots ≈0, positives tiny (≤+0.013). Real
  regime-dependent effect diluted by quiet shots, NOT a uniform small shift. Matches the residual-level sign (−0.019).
- CORRECTION: the earlier n=158 −0.018 was INFLATED (4-shot ACT_CF pool over-weighted the AE-active shots 200729/191001).
  Unbiased pooled effect = −0.0057; AE-active-shot effect ≈ −0.04. Even smaller in the pool than first reported, but significant.

**β=5 — the flip is REAL, β-specific, and INCOHERENT (condemns the knee):**
- POOLED: +0.0148, bootstrap95 [+0.0131, +0.0167] — persists positive at n=967, NOT small-n noise.
- PER-SHOT: 12/14 positive, but **200729 flips to −0.008** — the strongest-AE shot goes opposite to the majority AND
  opposite to its own β=6 sign (−0.042). Signature of anchor-release instability across the softmax-saturation boundary:
  near the knee, different windows unmask with different signs. **⇒ the knee region (β≈5) is untrustworthy; operating
  point pushed AWAY from 5, reinforcing β=6.** (β=4/β=3 also negative — the coherent physical sign is negative
  EVERYWHERE except the unstable β=5 knee.)

**UPDATED VERDICT — β=6 is the confirmed operating point.** The controllability claim, precisely stated and now
sign-confirmed: *perturbing beam power (pin) shifts the predicted ECE mode frequency DOWNWARD (AE-drive-correct), an
effect that is bootstrap-significant, placebo-separated, and concentrated in AE-active shots — at false-death 0.003
with no peak-in-tol regression.* This is FREQUENCY-SHIFT conditioning (not full distributional: ΔLL below the 3.5e-4
floor). **PARTIAL remains the headline** (strict ΔLL bar unmet at every β; controllability & fidelity coupled through
the anchor). The money-figure demonstration should be on an AE-active shot (**200729**, effect ≈−0.04) at β=6, where
the physics concentrates — the honest strong case, not the diluted pool average.

**β=5.5 hold — now RECOMMEND AGAINST:** the knee is demonstrably sign-incoherent at β=5; β=5.5 sits inside that
unstable transition, so it is unlikely to yield a clean stronger operating point and risks muddying the claim. β=6
(above the knee, coherent) is the honest operating point. (Decision deferred to user.)

**Gate 3: sign confirmation LANDED — β=6 direction established. Still NOT struck pending your read of §7+§8.**
