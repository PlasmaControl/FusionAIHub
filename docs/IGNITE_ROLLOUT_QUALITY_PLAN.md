# IGNITE Rollout Quality — Applying Self-Forcing / Self-Forcing++, Cosmos, and PAN

**Status:** analysis + recommendations, no implementation. Grounded in the code as of
`nathan_fm` @ 6de6fbe (2026-08-17) and the production run `prod_d512L8` (step 13.5k).
Companion to `docs/IGNITE_DESIGN.md`.

**Question answered:** IGNITE memorizes single shots but degrades hard on many shots,
and modalities drift mutually inconsistent during rollout. Can the Self-Forcing line
(arXiv 2506.08009, 2510.02283), NVIDIA Cosmos (gen 2.5 / Cosmos 3), or MBZUAI PAN
(arXiv 2511.09057) fix this — and what are the easy vs. hard changes?

**Answer in one line:** yes — IGNITE's failure signature is exactly the train/test gap
Self-Forcing was built for, the discrete-MaskGIT transfer is unusually clean (masking
*is* the discrete forward process), and there is a graded ladder from zero-retrain
sampler fixes to a self-forcing post-training stage; cross-modal clashing additionally
has a specific mechanical cause in the sampler that can be attacked today.

---

## 1. Diagnosis — three separable failure axes

### 1A. Exposure bias (the rollout-quality killer)

The training objective and the inference procedure ask the model different questions:

| | training (`maskgit.py:91-142`) | rollout (`maskgit.py:145-218`) |
|---|---|---|
| context frames | ground truth, **~64% masked** (`_random_mask`, per-frame cosine ratio) | **complete** (0% masked), **model-generated** |
| target frame | partially masked, visible tokens are GT | starts **fully masked**, visible tokens are the model's own commits |
| conditional trained/used | p(masked GT tokens \| masked GT everything) | p(frame \| committed self-generated history) |

The model is never trained on the conditional it samples from at inference. Evidence
this is the binding constraint, not a nice-to-have:

- **Scheduled sampling never ran.** Every step of `prod_d512L8/loss_history.jsonl` has
  `"ss": 0.0`; the launcher pins `--ss_final_frac 0`
  (`scripts/slurm_frontier/train_dynamics.sh:166`). The design doc mandated "light
  scheduled-sampling from the start" (IGNITE_DESIGN.md §5.6, decision #8) — it was
  never activated, and as written it is memory-infeasible anyway (materializes full
  `(B,F,N,vocab)` logits ≈ 400 GB at F=100, `train_dynamics.py:872-876`).
- **More training makes rollouts worse.** Controlled bp128 measurement (2026-08-15,
  shot 199597, same seed/arms): TM-band skill **+0.153 at step 11k → −0.862 at step
  20k**, with variance over-prediction. This inversion — teacher-forced CE improves
  while rollout skill collapses — is the textbook exposure-bias signature: sharper
  conditionals around GT contexts go further off-distribution when fed their own
  slightly-off tokens.
- **Single-shot memorization works** (`runs/overfit_full*_d256` reach CE ~0.01-0.3),
  so representation capacity is not the rollout bottleneck.
- Even the *disabled* scheduled-sampling hook is the weakest form of the idea: its
  substitution samples come from one forward pass over **clean GT codes**
  (`maskgit.py:79`), i.e. one-step errors under a GT context — not the structured
  drift statistics of real rollouts (see §2, SF++ ablation).

### 1B. Cross-modal incoherence ("clashing")

Mechanical causes, all in `generate_frame` (`maskgit.py:145-193`):

1. **Per-modality independent reveal schedules.** Each modality reveals the same
   *fraction* of its own tokens each step, ranked by its own confidence
   (`maskgit.py:167-185`). There is no global confidence pool over the frame's 1593
   tokens — an uncertain modality (say mhr during a mode transition) cannot defer
   while confident modalities commit first and anchor it; it must commit ~19 tokens
   at step 1 regardless.
2. **Independent marginal sampling within a step.** Tokens committed in the same
   decode step — across and within modalities — never see each other; coherence only
   propagates between steps (10 rounds of re-conditioning via spatial attention).
   Early-step incoherent commits are **never revisited** (no remasking of committed
   tokens).
3. **One global temperature** (`maskgit.py:171`), though confidences are not
   comparable across vocab sizes (1 000 vs 64 000) or token counts (4 vs 768).
4. **Cross-modal temporal coupling is indirect.** Temporal attention runs per token
   position (`dynamics.py:85`), so modality A's history reaches modality B's future
   only via alternating spatial↔temporal blocks — 8 alternations at depth 8.
5. The only trained coupling signal is spatial attention over the joint frame; the
   loss has no cross-modal consistency term, and modalities are averaged with equal
   weight regardless of token count (`maskgit.py:118-142`) — a 4-token slow-TS
   modality gets ~192× the per-token gradient of ece (768 tokens).

### 1C. Multi-shot generalization

- **89% of the production model is vocabulary tables** (measured on
  `prod_d512L8/dynamics_latest.pt`: 273.5 M of 307 M params are embeddings/heads; the
  four 64k-vocab modalities cost ~131 M in, ~131 M out). The actual dynamics core is
  **33.6 M params** (8 × 4.2 M blocks). The 815 M capacity arm "gave no rollout gain"
  (`train_dynamics.sh:32-33`) — capacity went to tables, and the teacher-forced
  objective can't convert capacity into rollout skill anyway (§1A).
- **Undertrained:** step 13.5k of 55 399 planned (10 epochs), val masked-CE 1.7124
  and still falling when the run stopped (Aug 13).
- **~37% of inputs are null.** Mean per-modality absence over 8 753 shots is 36.9%
  (presence map). `--mask_absent` now zero-weights these in the loss, but they still
  occupy input tokens and (by count) dominate several modalities' training signal.
- Stride-1 window sampling (`train_dynamics.py:699-700`) makes consecutive samples
  99%-redundant; effective diversity is ~8 753 shots × ~12 s, far less than the
  nominal 1.42 M windows suggests.

---

## 2. What the external work actually says (and what transfers)

### Self-Forcing (arXiv 2506.08009, NeurIPS 2025 spotlight)

Runs the *inference* procedure (autoregressive rollout, KV-cached, few-step) inside
the training loop, then applies a **video-level distribution-matching loss** (DMD /
SiD / GAN) to the completed rollout. Two facts matter most for IGNITE:

- **On-policy context is the load-bearing ingredient, not the fancy loss.** With the
  *identical* DMD loss: teacher-forced context 82.32, diffusion-forced 82.76,
  self-rollout context **84.31** VBench. The GAN variant (83.88) nearly matches DMD —
  loss choice is secondary.
- **No backprop-through-time.** Gradients flow only through the final denoising step
  of one frame (stochastic truncation); the KV cache/context is detached. The signal
  is *whose distribution the context comes from*, not gradients through the rollout.
  This makes the whole family memory-feasible.

### Self-Forcing++ (arXiv 2510.02283)

Fixes long-horizon collapse beyond the teacher's 5 s window: roll the student out
**far beyond the training horizon** (up to 20×) with the production cache mechanics,
sample **uniform contiguous windows from the student's own degraded rollouts**,
re-noise them along the diffusion schedule (**backward noise initialization** — keeps
windows temporally coupled to rollout context), and apply DMD there. A 5 s teacher
legitimately scores any window of a long video because short windows are marginals of
valid long sequences. Results: visual stability at 50 s **90.94 vs 40.12** for
Self-Forcing; 4 min 15 s videos ≈ positional-embedding capacity. Two ablations of
direct consequence for IGNITE:

- **Synthetic context corruption (noise injected into the KV cache) gives only slight
  improvement** — random corruption does not mimic real rollout-error statistics
  (drift toward stasis, saturation). So scheduled sampling / token corruption is a
  *bridge*, not the fix.
- Optional GRPO stage; notably, discrete models get exact token log-probs for free,
  making sequence-level RL *easier* for IGNITE than it was for them.

### Transfer to discrete MaskGIT (the mapping is clean)

Copilot4D (ICLR 2024) formalized MaskGIT as **discrete diffusion**: masking = forward
process, iterative unmasking = reverse process. Hence:

| Self-Forcing/++ concept | IGNITE analog |
|---|---|
| re-noise own rollout latents (backward noise init) | **re-mask the student's own rolled-out codes** (cosine schedule) — transfers 1:1 |
| few-step denoise per frame | the 10-step confidence unmask (already few-step) |
| video-level DMD/GAN on rollout windows | frozen "real" masked model + online critic on token windows, or R3GAN token-window discriminator, or GRPO with exact log-probs |
| gradient truncation + detached KV | grads only through the final reveal step's logits; context frames detached |
| rolling KV cache parity train↔test | window/positional-encoding parity (IGNITE currently consistent at ≤100 frames; parity work only needed for horizon extension) |

**Nobody has published the full self-forcing recipe for a discrete MaskGIT world
model** (search Aug 2026: only corruption-based approaches — Copilot4D, Masked-HWM —
plus MAGI's Complete Teacher Forcing). This is an open niche; IGNITE has the frozen
teacher (the pretrained ckpt), exact log-probs, physics-grounded reward functions
(the eval metrics), and a paired-counterfactual eval harness already built.

### MAGI — Complete Teacher Forcing (arXiv 2501.12389, CVPR 2025)

Nearest published attack on §1A's *structural* mismatch for masked video models:
condition masked target frames on **complete** (not masked) previous frames — +23%
FVD, stable 100+-frame rollouts from 16-frame training. Directly indicts IGNITE's
mask-everything-everywhere training scheme.

### Cosmos (Predict 2.5 / Cosmos 3, arXiv 2511.00062 / 2606.02800)

- **Noise the conditioning context during training** — Cosmos 1 Video2World noise-
  augments condition frames; PAN does the same (k=0.055). Independent double
  confirmation of context corruption as cheap hardening (with SF++'s caveat above).
- **Critic-guided best-of-N**: generate N rollouts, score each ~4 times with a
  physical-plausibility judge, keep the argmax; the Cosmos 3 critic was trained on
  ~1K human-graded generations. Operational anti-drift with **zero dynamics retrain**.
- **GRPO with an auxiliary base loss** to prevent reward hacking (their RL recipe).
- **Inverse dynamics as an auxiliary task** (Cosmos 3 trains forward + inverse +
  policy jointly): forces actuator-relevant physics into the representation.
- **Mixture-of-Transformers**: per-modality parameter towers, attention over the
  union of tokens — the template for 14 heterogeneous modalities.
- **Curation beats algorithms for cross-domain generalization** (they keep ~4% of
  clips after filtering; long-horizon gains gen1→2.5 came from base model + data,
  not a rollout algorithm).
- **Cautionary tale:** Cosmos 1's discrete-token AR line needed a 7B diffusion
  decoder to clean token artifacts and was dropped in gen 2 — for *pixel fidelity*
  reasons that matter less for diagnostics, but it says: don't expect the token
  bottleneck to be free; IGNITE's statistics-first codecs are the right mitigation.

### PAN (arXiv 2511.09057)

- **Long-range consistency lives in a compact latent state** predicted by the
  backbone (LLM over 256 query tokens/step); the diffusion decoder only renders
  locally. Cross-modal/temporal coherence by *common cause*, not pairwise attention.
- **Re-encoding anchor:** each decoded chunk is re-encoded and concatenated with the
  predicted latent — rollouts are continually re-grounded in what was actually
  emitted. (IGNITE's closed code space already commits sampled codes back as context,
  which is this anchor's discrete twin — a design choice to keep, not change.)
- **Never latent-match** (JEPA-style) — collapse risk; supervise through decoding.
  IGNITE's CE-through-heads already respects this.

---

## 3. Recommendations, ranked

Ordered by effort within each tier. Each item names the target file(s), the source
technique, and which failure axis (§1A/B/C) it attacks.

### Tier 0 — inference-only; no retrain; test on existing checkpoints (days)

**R0. Slice heads to the last frame in `generate_frame`.** `backbone.forward`
(`dynamics.py:133-134`) projects *every* frame and token to full vocab each of the
800 rollout forwards; only `[:, -1]` is used (`maskgit.py:171`). Add a
`logits_last(h)`/frame-slice path (mirror of `masked_logits`,
`frame_layout.py:97-115`): ~15 GB transient fp32 → ~150 MB, and rollouts get
dramatically cheaper. *Prerequisite for everything below* (batched rollout in
training, best-of-N, faster eval). No behavior change. [enabler]

**R1. Global cross-modal confidence pool.** Replace the per-modality reveal quota
with one frame-wide schedule over all 1593 tokens: normalize each modality's
confidence to a comparable scale (within-modality quantile rank is the robust
choice — raw probabilities are incomparable across vocab 1 000 vs 64 000), merge,
reveal the top-K overall per step. Uncertain modalities defer; confident ones anchor.
~30 lines in `maskgit.py:167-185`. [1B]

**R2. Per-modality temperature + top-p.** Expose dict-valued temperature and add
nucleus filtering at `maskgit.py:171-174`. Spectrogram vocabs at 64k sampled at T=1.0
from marginals are a variance firehose; slow-TS at vocab 1000/4 tokens wants
different treatment. Cheap grid on existing checkpoints. [1B, 1A]

**R3. Revision pass (draft-and-revise).** After the 10-step schedule completes,
re-mask the lowest-confidence ~15-30% of the *committed* frame (or run 1-2 extra
Gibbs-style sweeps) and re-decode conditioned on the survivors. Fixes early incoherent
commits that the current never-revisit rule locks in. ~1.3× rollout cost after R0.
[1B]

**R4. Best-of-N rollout reranking (Cosmos rejection-sampling pattern).** Sample N
rollouts (different seeds), score each window by **masked pseudo-likelihood under the
frozen pretrained model** (re-mask ~30% of the rollout's own codes, measure CE — the
discrete twin of "teacher scores the window") plus cheap physics scores already in
the tree (`gate.py` per-position Markov forecastability; persistence-skill). Keep the
argmax. Zero retrain; also becomes the reward function for R10. [1A, 1B]

*Tier-0 validation:* run R1-R4 on `prod_d512L8` @13.5k **and both bp128 checkpoints
(11k and 20k)**. If sampler fixes shrink the 11k→20k skill inversion, that confirms
exposure bias as the mechanism and calibrates how much Tier 1/2 must deliver.

### Tier 1 — training fixes; cheap retrain of d512L8-class arms (1–2 weeks incl. A/B)

**R5. Complete-context training mode (MAGI CTF).** Give a fraction q of training
windows the *rollout's* conditional structure: sample a boundary c ~ U[1, F−1];
frames < c fully visible (unmasked), frames ≥ c masked at high ratio (include ratio
1.0), loss only on frames ≥ c. ~20 lines in `_random_mask` (`maskgit.py:43-65`).
This is the cheap variant; the full MAGI trick (duplicated clean/masked frame pairs
with a custom temporal mask so *every* frame trains against complete context in
parallel) costs 2× frames and a custom SDPA mask — do the cheap variant first, it
removes the worst of the mismatch (context frames at rollout are never masked).
Expected effect size: MAGI reports +23% FVD from exactly this fix. [1A]

**R6. Fix scheduled sampling properly (bridge, not destination).** Unblock memory by
sampling through the same head-slicing path as R0 (project only the frames being
substituted); substitute with *multi-step* samples (2-4 unmask iterations, not
one-pass-under-GT); ramp per the design doc (`ss_ramp_final_frac=0.15` exists,
`dynamics_config.py`). Keep expectations calibrated: SF++'s ablation says corruption
alone under-delivers; this exists to harden the model before R10 and to finally
exercise the design-doc knob. [1A]

**R7. Loss reweighting + input hygiene.** Weight per-modality CE by token count (or
√count) so ece isn't out-gradiented 192:1 by 4-token modalities
(`maskgit.py:118-142`); keep `mask_absent`. Optionally: presence-balanced shot
sampling, and window stride > 1 (or random offsets) to cut the 99% window redundancy.
[1C]

**R8. Actuator-conditioning dropout + inverse-dynamics auxiliary.** Drop the additive
actuator embedding with p≈0.1 during training (`dynamics.py:116-117`) — this enables
**classifier-free guidance on actuators at eval** (amplify controllability at rollout
time, the standard fix for conditioning being re-absorbed — the exact failure the old
e2e model showed). Add a small inverse-dynamics head (predict `actuator_t` from frame
hidden states; Cosmos 3 pattern) as an auxiliary loss to force actuator-relevant
physics into the representation. Both are small diffs but need a retrain to matter.
[1A→controllability, 1C]

### Tier 2 — discrete self-forcing post-training (the main event; 2–4 weeks)

Post-training on top of the existing pretrained checkpoint (SF's own paradigm:
"parallel pre-training + sequential post-training"). Two stages, increasing
machinery:

**R9. Stage A — rollout-context fine-tune (DAgger-style scheduled sampling done
right).** In the trainer: for each window, pick c and m (m ramping 1→8-16); run a
**no-grad batched rollout** of m frames from GT prefix [0,c) (needs R0; use
bf16 + 4 decode steps for training rollouts — SF uses few-step rollouts in training);
then one supervised step: predict frame c+m (masked at the standard schedule, or
fully) conditioned on [GT[0:c] ⊕ rollout[c:c+m]], CE target = GT[c+m]. Gradients only
through the final prediction (context detached — SF's truncation). This trains the
model to *recover from its own drift* with paired supervision and no distribution-
matching machinery. Cost ≈ 2-4× per step at m≈8 with R0 + few-step rollouts.
Caveat: after large divergence GT stops being the right target — keep m modest,
optionally weight by rollout-vs-GT agreement. [1A, produces the on-policy contexts
SF++ says corruption can't fake]

**R10. Stage B — distribution-level objective on self-rollout windows.** The full
SF++ analog:

1. Periodically generate long rollouts with the *production* sampler (no grad).
2. Sample contiguous K-frame windows from them (uniform offset).
3. **Re-mask the student's own tokens** per the cosine schedule (backward-noise-init
   analog — keeps the window coupled to rollout statistics).
4. Update with one of (pick ONE to start):
   - **GRPO (recommended first):** N=8 rollouts per seed context; reward = frozen-
     teacher masked pseudo-likelihood of the window (R4's scorer) + decoded-space
     skill terms (GT is available in training) + cross-modal consistency probes;
     group-normalized advantages on the **exact log-probs of committed tokens**
     (store log p at commit time in `generate_frame` — trivial); auxiliary standard
     masked-CE on real windows to prevent reward hacking (Cosmos's aux-loss trick).
   - **R3GAN token-window critic (SF's GAN variant, 83.88 vs DMD 84.31):**
     bidirectional transformer discriminator over token embeddings of K-frame
     windows, real (cache) vs student rollouts, relativistic pairing + finite-
     difference R1/R2 on embeddings; generator grads via straight-through Gumbel on
     the final reveal step only.
   GRPO first: it needs no new network (the frozen teacher is the scorer), exploits
   the discrete-token log-prob advantage, and its rewards are IGNITE's existing eval
   metrics — the reward *is* the thing we want to go up. [1A, 1B via consistency
   rewards]

**Compute sanity:** SF post-training converged in ~1.5 h on 64 H100s for a 1.3B
model. IGNITE is 307 M on 128 MI250X GCDs — the post-training stage is cheap relative
to pretraining; the dominant cost is training-time rollouts, which R0 + few-step
decoding keeps at a small multiple of a teacher-forced step.

**R11. Horizon extension (SF++ proper) — after R9/R10 hold.** Roll out beyond 80
frames in post-training and supervise sampled windows. Blocked today by the learned
absolute `frame_embed` hard-capped at 100 frames and the unused `frame_offset`
(`frame_layout.py:46,55-67`) — requires switching the temporal axis to relative
encoding (RoPE in the temporal `_MHA`) or training with random frame offsets, then
keeping train/test window mechanics identical (SF's rolling-cache-parity lesson).
This is what turns 4 s rollouts into 10 s+ rollouts. [1A at long horizon]

### Tier 3 — architecture and data (bigger bets, roughly ordered by leverage/cost)

**R12. Factorize the 64k-vocab heads by FSQ digit.** Predict 6 sub-digits of
(8,8,8,5,5,5) instead of one 64 000-way softmax: the four 64k modalities' ~262 M of
table parameters collapse to ~2 M, and (at constant budget) the dynamics core can
grow ~8× (33.6 M → ~250 M+, e.g. d768-1024, deeper). Already named as future work in
`docs/IGNITE_CODEC_RETRAIN_SPEC.md:74`. This is the single biggest *generalization*
lever: right now 89% of parameters memorize vocabularies instead of modeling
dynamics — which is exactly "overfits a shot, fails across shots." Also softens the
confidence-comparability problem R1 works around. [1C, 1B]

**R13. State/register tokens (PAN GLP-lite).** Add S learned state tokens per frame
(S ≈ 16-64) that participate in spatial attention; run temporal attention **only over
state tokens** (or state + per-position). Long-range consistency then travels through
a compact world state (PAN's central claim), every modality decodes against a common
cause (anti-clash by construction), and temporal attention cost drops ~1593/S-fold,
buying budget for longer windows. Bigger rewrite of `dynamics.py` + retrain. [1B, 1A]

**R14. Per-family parameter towers (Cosmos MoT-lite).** Shared attention over the
union of tokens, per-family (spectro/video/slow-TS/fast-TS) QKV/FFN weights — respects
wildly different statistics without splitting the joint model. Consider only if R12's
reallocated capacity still underfits families differentially. [1C, 1B]

**R15. Data curation and sampling (Cosmos's actual answer to cross-domain
generalization).** Presence-aware curriculum (train first on shots where most
diagnostics recorded), drop segments where most modalities are frozen/absent,
shot-balanced batching. Cosmos kept 4% of raw clips; IGNITE currently trains on
everything, 37% null. [1C]

### R16 — TokEye activity masks as a physics side-channel (data + reward + eval)

[PlasmaControl/TokEye](https://github.com/PlasmaControl/TokEye) (arXiv 2602.20317)
— the group's U-Net that separates coherent modes and transient bursts from
broadband turbulence on fluctuation spectrograms — is now wired into the data:
the adjacent project `/lustre/orion/fus187/scratch/nchen/tokeye/` runs it over
the raw H5 traces and writes per-shot SIDECAR files `{shot}_tokeye.h5` (next
to the processed files, which stay read-only) holding `tokeye_<signal>/activity`
— uint8 `(C, 512, T)` maps of `sigmoid(logit_coherent + logit_transient)` on
TokEye's native grid (n_fft 1024, hop 128 = 2× finer than the codec's hop-256
grid; `tvals`/`freqs` datasets make pooling exact; ~8 MB/shot). Verified on the control
shots: mode tracks traced continuously through ELM striations, burst columns
captured, background ≈ 0 (fused mean 0.035, 1% of pixels > 0.5 on 190735
mhr ch3); ~1 s/channel on a GCD after MIOpen warmup, so the full 8,753-shot
mhr sweep is ~22 GPU-hours. Example panels:
`tokeye/out/example_190735 (full/zoom) .png`.

Uses, in increasing ambition:

1. **Activity-weighted rewards and metrics (feeds R4/R10 and §5).** The GT
   activity mask says *where the physics is*; weight decoded-vs-GT agreement
   by it. This directly fixes the dilution problem noted in
   `ignite_bp_cases.py` (a tearing mode occupying ~8% of bins vanishes in a
   512-bin average) and generalizes the hand-picked TM/AE bands — no TokEye
   inference needed in the training loop, the masks are precomputed.
2. **Curation and probe labels (feeds R15, gate.py).** Per-frame, per-band
   activity summaries = mode-presence labels for probes, and a principled
   "dynamic core" shot selector (replace `curate_core`'s all-modality-changes
   heuristic with measured MHD activity).
3. **Tokenizer input (statistics-first codec v2 / bp-line upgrade).** The bp
   tokenizer currently bins raw band log-power; binning *activity-masked*
   power (or adding pooled activity as an extra token stream) outsources the
   "statistic, not realization" extraction to a trained denoiser — aimed
   squarely at the degenerate mhr modality.

Caveats: masks are model outputs, not ground truth (treat as weak labels);
`--fusion-center` shifts the operating point (default 0 puts a coherent-only
pixel near 0.5, center ≈ −2.4 restores per-channel calibration).

### Anti-recommendations (things that look tempting but the evidence says no)

- **Don't just crank `--ss_final_frac` with the current implementation** — it OOMs by
  construction (full-logit materialization) and its one-pass-under-GT samples are the
  weakest corruption variant (SF++ ablation: minor gains).
- **Don't scale d_model/depth first.** The 815 M arm already showed no rollout gain;
  under teacher forcing, capacity mostly sharpens the wrong conditional (§1A), and
  89% of new params would go to tables anyway (fix R12 first).
- **Don't reintroduce continuous/decoded feedback into the rollout loop.** The old
  e2e model's deterministic continuous feedback froze to a fixed point; IGNITE's
  commit-discrete-codes loop is PAN's re-encoding anchor in discrete form — keep it.
- **Don't headline `divergence_vs_real`** (established 2026-08-15: token churn
  anti-correlates with decoded effect size); judge every change in decoded,
  band-restricted space with the majority-token guard.
- **Keep rollouts fp32** (bf16 measurably degrades accuracy via confidence-order
  perturbation, `eval_dynamics.py:1032-1044`) — note R1's rank-based confidences
  should also *reduce* this sensitivity.

---

## 4. Symptom → remedy map

| symptom | now (Tier 0) | near (Tier 1) | structural (Tier 2/3) |
|---|---|---|---|
| rollout degrades over 80 frames; more training makes it worse | R3 revision, R4 best-of-N | **R5 CTF**, R6 | **R9/R10 self-forcing**, R11 horizon |
| modalities clash / drift apart | **R1 global pool**, R2 temps, R3 | R7 reweighting | R10 consistency rewards, R13 state tokens |
| overfits one shot, poor across shots | — | R7, R8 aux | **R12 head factorization**, R15 curation |
| actuator response washed out in rollout | — | **R8 CFG dropout** | R10 controllability rewards |

## 4b. Where to pilot: the band-power line first

The `bp*` family (`/lustre/orion/fus187/proj-shared/models/ignite_bandpower/`) is the
same `MaskGITDynamics` class with a deterministic 8-level band-power tokenizer —
mhr(192) + co2(128) = **320 tokens/frame, vocab 8, 33.9 M params, ~all of them
transformer** (no vocab-table pathology, no 15 GB transient logits). Rollout-in-
training is computationally trivial there, it targets exactly the MHD channel where
the physics questions live, and it owns the documented skill inversion (+0.153 →
−0.862) that R5/R9/R10 must fix. **Pilot the whole Tier 1→2 ladder on bp_d512L8
(hours per experiment), then port the winners to the 14-modality production model**
(where R0/R12 make them affordable). Note bp is Peter's active pipeline — coordinate
rather than fork it.

## 5. Validation protocol (uses only existing machinery)

- **The regression test for exposure-bias fixes is the skill-vs-step curve.** Train an
  arm with R5(+R6), checkpoint every ~2k steps, run the fixed eval set; success =
  monotone (or at least non-inverting) rollout skill where bp128 showed +0.153→−0.862.
- Fixed comparison set: curated dynamic shots (`eval_dynamics.py:1194-1257`),
  `split_seed=42` untouched, token skill AND decoded band-restricted skill at
  k ∈ {10, 40, 80}, per modality, with the majority-token degeneracy guard (mhr must
  beat the constant-code baseline before any mhr claim).
- Tier-0 changes need no training: A/B on `prod_d512L8` @13.5k and bp128 @11k/@20k,
  same seeds, paired rollouts.
- Controllability: donor-shot counterfactuals (in-distribution) with the existing
  paired-RNG machinery; after R8, sweep CFG scale.

## 6. Publication note

A discrete self-forcing recipe for multi-modal *scientific* world models appears
unpublished (checked Aug 2026: corruption-based approaches and MAGI's CTF are the
closest). IGNITE already owns the ingredients a paper needs — frozen teacher, exact
token log-probs, physics-grounded rewards, paired actuator-counterfactual evals, and
a documented failure case (the bp128 inversion) that the method should visibly fix.
R5 + R9/R10 + the §5 protocol is a NeurIPS-shaped story if it works.

## 7. Sources

- Self-Forcing: arXiv 2506.08009, github.com/guandeh17/Self-Forcing
- Self-Forcing++: arXiv 2510.02283, self-forcing-plus-plus.github.io,
  github.com/justincui03/Self-Forcing-Plus-Plus
- MAGI (Complete Teacher Forcing): arXiv 2501.12389 · Copilot4D: arXiv 2311.01017 ·
  Masked-HWM: arXiv 2506.01182 · Diffusion Forcing: arXiv 2407.01392
- Cosmos 3: arXiv 2606.02800, github.com/NVIDIA/cosmos · Predict 2.5: arXiv
  2511.00062 · Reason-as-critic: docs.nvidia.com/cosmos (video_critic) · Cosmos
  Tokenizer: arXiv 2501.03575
- PAN: arXiv 2511.09057, ifm.mbzuai.ac.ae/pan (NeurIPS 2025 LAW workshop invited
  talk — the neurips.cc/virtual/2025/loc/san-diego/137008 link)
- TokEye: github.com/PlasmaControl/TokEye, arXiv 2602.20317; local ingest project
  `/lustre/orion/fus187/scratch/nchen/tokeye/` (README documents schema + sweep)
- Concurrent long-horizon work: LongLive arXiv 2509.22622, Rolling Forcing arXiv
  2509.25161, Causal-rCM arXiv 2606.25473, OPSD-V arXiv 2607.08766

## Ops

- Frontier's `/lustre/orion/.../scratch` **purges files by access time**: after long idle
  periods the pixi env loses stdlib files and `.pixi/envs/frontier/bin/python` will not start —
  rebuild with `pixi install` then
  `.pixi/envs/frontier/bin/pip install --no-deps x-transformers vector-quantize-pytorch loguru einops einx torch-einops-utils`,
  and keep `nathan_fm` pushed (unpushed git objects live on the same purging filesystem).
