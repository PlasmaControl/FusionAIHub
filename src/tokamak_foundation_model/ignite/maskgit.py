"""Phase-B MaskGIT dynamics: masked-token training, iterative-unmask decode, commit-code rollout.

Wraps :class:`DynamicsBackbone`. Training masks a random fraction of each frame's tokens
(replacing them with the per-modality ``[MASK]`` id) and predicts them by per-modality
categorical CE — conditioned, via the causal temporal attention, on past frames + visible tokens
+ actuators. Inference generates a frame from all-masked by iterative confidence-based unmasking
(cosine schedule). Rollout seeds ``K₀`` real frames, then generate → **commit the discrete
codes** → append → repeat to 80 frames. See docs/IGNITE_DESIGN.md §5.3/5.5/5.6.

Codes are dicts {modality_name: LongTensor (B, F, n_tok_m)}; a rollout frame's per-modality codes
are the committed integers. Randomness is injected via an optional ``generator`` so tests are
deterministic and resume is reproducible (main scripts pass one; Date/rand globals are avoided).
"""

from __future__ import annotations

import math
import os as _os
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dynamics import DynamicsBackbone
from .dynamics_config import DynamicsConfig
from .sampling import SamplerConfig, apply_top_p, norm_log_confidence
from .selfforce import rollout_context


def _cosine_keep_fractions(n_steps: int) -> list:
    """Fraction of a frame's tokens that remain MASKED after decode step i (cosine schedule).

    Starts fully masked (1.0 before step 0), ends fully revealed (0.0 after the last step).
    """
    return [math.cos(math.pi / 2 * (i + 1) / n_steps) for i in range(n_steps)]


# Reveal-order noise scale for generate_frame (0 = the historical greedy reveal, so every
# existing run and queued job is byte-identical unless this is set). See generate_frame.
_GUMBEL_SCALE = float(_os.environ.get("IGNITE_MASKGIT_GUMBEL", "0"))


class MaskGITDynamics(nn.Module):
    def __init__(self, cfg: DynamicsConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = DynamicsBackbone(cfg)

    # ---------------------------------------------------------------------------- training ---
    def _random_mask(self, codes: Dict[str, torch.Tensor], gen: Optional[torch.Generator],
                     ratio_override: Optional[float] = None, history_frames: int = 0,
                     gen_mask_p: Optional[float] = None):
        """Mask a per-frame random fraction of tokens. Returns (masked_codes, mask) with the same
        keys; masked positions in masked_codes hold the modality's MASK id, mask[.]=True there.

        ``ratio_override`` pins the mask ratio instead of drawing it, for DIAGNOSIS only: the
        training loss averages over the cosine prior, but a rollout builds every frame starting
        from ratio 1.0 (nothing in the frame known). Scoring at a fixed ratio separates
        "infills well" from "generates well" — they are not the same task and only the second
        one is what a 4 s prediction actually performs.
        """
        cfg = self.cfg
        ref = codes[cfg.modalities[0].name]
        B, Fr, _ = ref.shape
        dev = ref.device
        protect_gen = None          # set by generation mode below; frames it must not mask
        # one mask ratio per (B, frame), in [0, 1] via a cosine of a uniform (MaskGIT prior)
        if ratio_override is not None:
            ratio = torch.full((B, Fr), float(ratio_override), device=dev).clamp(1e-3, 1.0)
        else:
            u = torch.rand((B, Fr), generator=gen, device=dev)
            ratio = torch.cos(math.pi / 2 * (1.0 - u)).clamp(1e-3, 1.0)  # bias toward high masking
            # ---- GENERATION MODE (cfg.gen_mask_p) --------------------------------------------
            # MEASURED 2026-08-15 on bp128_big best: under the TRUE rollout condition (history
            # visible, target frame empty) this model scores 1.5804 vs a model-free bigram's
            # ~1.70 — 0.12 nats better — while the tracked val CE is 0.9228. It learned to
            # INTERPOLATE inside a half-given frame, not to predict the next one, because the
            # cosine prior almost never presents the generation task: ~9% of frames land near
            # ratio 1.0 and even those come with a masked history.
            # Here, for a fraction gen_mask_p of SAMPLES, the batch element is turned into
            # exactly what rollout() does: everything before a split point t is real and
            # unscored, everything from t on is masked along the reveal ladder.
            # VALIDATION MUST NOT INHERIT THIS. _val_loss passes gen_mask_p=0.0 so the val metric
            # stays a FIXED protocol across arms — otherwise an arm trained with generation mode
            # would also be *scored* on a harder mask distribution, inflating its val CE and
            # making it incomparable to every run that came before it.
            p = float(cfg.gen_mask_p if gen_mask_p is None else gen_mask_p)
            if p > 0.0:
                k0 = int(min(max(cfg.k0_seed, 0), Fr - 1))
                use = torch.rand((B, 1), generator=gen, device=dev) < p          # (B,1) samples
                span = max(Fr - k0, 1)
                # ---- HORIZON SAMPLING (cfg.gen_horizon_alpha) --------------------------------
                # rollout() calls the model as a ONE-STEP predictor, 80 times:
                #     for k in 1..n_predict: x[K0+k] <- generate(x[0 .. K0+k-1])
                # so the function it needs is p(x_f | x_0..x_{f-1}). A UNIFORM split point
                # t ~ U{k0..Fr-1} instead trains a ~21-frame-ahead predictor:
                #     scored frame f has horizon h = f - t + 1, mean (Fr-t+1)/2,
                #     E[h] = (101 - 59.5)/2 ~ 21 frames ~ 1.05 s at Fr=100, k0=20.
                # At 21 frames ahead the MARGINAL is near-optimal, so the model correctly learns a
                # marginal-like map -- MEASURED 2026-08-17 on shot 200144 given 20 real frames:
                # GT band2 0.277, persistence 0.277, model 0.007; one-step token acc 0.533 vs
                # persistence 0.652. The rollout then steps off the seed and flatlines, while
                # gen_ce (which scores this same uniform mixture) keeps improving -- 0.29 nats
                # across three arms with no change in the figure.
                # alpha > 0 samples the HORIZON h ~ h^-alpha and sets t = Fr - h, concentrating
                # gradient on the one-step conditional while keeping a heavy tail so the model
                # still learns to recover from a drifted history. alpha = 0 -> the old uniform
                # split, byte-identical.
                # gen_horizon_max: horizon UNIFORM in [1, hmax]. MEASURED 2026-08-17: alpha=2
                # puts 61% of draws at h=1, and h=1 scores ONE frame of 100 -- ~20x fewer scored
                # tokens per step than the uniform split, so the arm trained the right objective
                # on a fraction of the signal and lost 0.067 one-step accuracy. A bounded uniform
                # keeps the horizon short (mean (hmax+1)/2) while scoring hmax/2 frames per
                # sample, trading far less gradient for the same train/inference match.
                hmax = int(getattr(cfg, "gen_horizon_max", 0) or 0)
                alpha = float(getattr(cfg, "gen_horizon_alpha", 0.0) or 0.0)
                if hmax > 0:
                    hi = max(1, min(hmax, span))
                    h = (torch.randint(1, hi + 1, (B, 1), generator=gen, device=dev))
                    t = (Fr - h).clamp_(k0, Fr - 1)
                elif alpha > 0.0:
                    hs = torch.arange(1, span + 1, device=dev, dtype=torch.float32)
                    w = hs.pow(-alpha)
                    h = torch.multinomial(w / w.sum(), B, replacement=True,
                                          generator=gen).view(B, 1) + 1          # h in [1, span]
                    t = (Fr - h).clamp_(k0, Fr - 1)                              # (B,1)
                else:
                    t = k0 + (torch.rand((B, 1), generator=gen, device=dev) * span).long().clamp_(
                        0, span - 1)                                             # (B,1)
                idx = torch.arange(Fr, device=dev).view(1, Fr)
                before = idx < t                                                 # (B,Fr) history
                # target frames ride the reveal ladder, biased hard toward a cold start
                v = torch.rand((B, Fr), generator=gen, device=dev)
                ladder = (1.0 - v.pow(3.0)).clamp(1e-3, 1.0)     # ~50% of draws above 0.8
                gen_ratio = torch.where(before, torch.zeros_like(ratio), ladder)
                ratio = torch.where(use, gen_ratio, ratio)
                protect_gen = use & before        # (B,Fr) real, unscored history per sample
        # PROTECTED HISTORY. rollout() hands frames [0, k0_seed) to the model as real codes and
        # only ever predicts [k0_seed, F). Training masks and scores all 100 frames, so ~20% of
        # every gradient is spent on frames that are free at inference — and they are the EASY
        # ones (little history, no accumulated drift). Pinning ratio 0 here makes the scored
        # region match what rollout actually predicts.
        hf = int(max(0, min(history_frames, Fr)))
        if hf:
            ratio = ratio.clone()
            ratio[:, :hf] = 0.0
        # every frame that must stay FULLY VISIBLE: the diagnostic history_frames prefix, plus
        # each generation-mode sample's own pre-split history. The >=1-masked-token guarantee
        # below would otherwise punch a mask into them and reintroduce the very mismatch this
        # mode exists to remove.
        protect = torch.zeros((B, Fr), dtype=torch.bool, device=dev)
        if hf:
            protect[:, :hf] = True
        if protect_gen is not None:
            protect = protect | protect_gen
        masked, mask = {}, {}
        for m in cfg.modalities:
            c = codes[m.name]
            r = torch.rand(c.shape, generator=gen, device=dev)
            mk = r < ratio.unsqueeze(-1)                              # (B, F, n_tok) bool
            # guarantee >=1 masked token per frame so CE always has a target — but NEVER in the
            # protected history, which must stay fully visible to mirror rollout()
            none = ~mk.any(dim=-1, keepdim=True)
            none = none & ~protect.unsqueeze(-1)
            if none.any():
                mk = mk | (F.one_hot(torch.zeros(1, dtype=torch.long, device=dev), c.shape[-1])
                           .bool().view(1, 1, -1) & none)
            mc = torch.where(mk, torch.full_like(c, self.backbone.tok.mask_ids[m.name]), c)
            masked[m.name], mask[m.name] = mc, mk
        return masked, mask

    def _boundary_mask(self, codes: Dict[str, torch.Tensor], gen: Optional[torch.Generator],
                       min_boundary: int = 1):
        """CTF layout: frames < c are COMPLETE context; frames >= c are heavily masked targets.

        This is the conditional rollout actually uses (see docs/IGNITE_ROLLOUT_QUALITY_PLAN.md
        §1A). The boundary c is per-sample so one batch spans many context lengths.

        ``min_boundary`` floors that draw, so a caller can protect a prefix that MUST stay fully
        visible: self-forcing passes the end of its self-rolled window, keeping those frames as
        complete context instead of letting the mask overwrite the model's own output. The default
        1 reproduces the original bounds exactly (>=1 complete context frame).
        """
        cfg = self.cfg
        ref = codes[cfg.modalities[0].name]
        B, Fr, _ = ref.shape
        dev = ref.device
        c = torch.randint(min_boundary, max(min_boundary + 1, Fr), (B,),
                          generator=gen, device=dev)                          # >=1 context frame
        idx = torch.arange(Fr, device=dev).view(1, Fr)
        is_target = idx >= c.view(B, 1)                                        # (B, F)
        lo = float(cfg.ctf_min_target_ratio)
        u = torch.rand((B, Fr), generator=gen, device=dev)
        ratio = lo + (1.0 - lo) * u                                            # in [lo, 1]
        masked, mask = {}, {}
        for m in cfg.modalities:
            cd = codes[m.name]
            r = torch.rand(cd.shape, generator=gen, device=dev)
            mk = (r < ratio.unsqueeze(-1)) & is_target.unsqueeze(-1)
            none = (~mk.any(dim=-1, keepdim=True)) & is_target.unsqueeze(-1)   # keep >=1 target
            if none.any():
                first = torch.zeros_like(mk)
                first[:, :, 0] = True
                mk = mk | (first & none)
            masked[m.name] = torch.where(
                mk, torch.full_like(cd, self.backbone.tok.mask_ids[m.name]), cd)
            mask[m.name] = mk
        return masked, mask

    def ss_fraction(self, step: int) -> float:
        """Scheduled-sampling own-code fraction: linear ramp 0 -> ss_ramp_final_frac."""
        cfg = self.cfg
        return min(1.0, step / max(1, cfg.ss_ramp_steps)) * cfg.ss_ramp_final_frac

    @torch.no_grad()
    def _scheduled_sample_context(self, codes, actuators, ss_frac, gen, text=None):
        """Return a copy of ``codes`` with a ``ss_frac`` fraction of (B, frame) cells replaced by
        the model's OWN one-pass sampled codes — so the model learns to consume its own outputs
        (drift mitigation). Substitution is on the CONTEXT only; the CE target stays the real code.

        Two defects kept ``ss_final_frac=0`` on every run since the build; both are fixed:

        * FULL logits. ``self.backbone(codes, actuators)`` projected EVERY position to vocab —
          the (B, F, tokens_per_frame, vocab) tensor that :meth:`FrameTokenizer.masked_logits`
          exists to avoid (~B*400 GB at F=100), and it crashed outright with
          ``mat1 and mat2 shapes cannot be multiplied (128000x512 and 1x128000)``, killing all
          four SS legs in 43 s. Only the substituted frames are ever read, so gather them
          through ``masked_logits`` exactly as the CE path does.
        * AUTOCAST CACHE. This method is ``@torch.no_grad()`` and the caller runs under
          autocast, so its forward filled the bf16 weight cache with DETACHED copies which the
          real forward below then reused — gradients never reached the fp32 parameters.
          Measured: grad-norm 0.926 -> 0.087 with the cache on, 0.911 with it off, at an
          IDENTICAL loss. Silent, and invisible in the loss curve. Hence ``cache_enabled=False``.
        """
        if ss_frac <= 0.0:
            return codes
        ref = codes[self.cfg.modalities[0].name]
        B, Fr, _ = ref.shape
        sub = torch.rand((B, Fr), generator=gen, device=ref.device) < ss_frac   # (B, F) frames
        if not bool(sub.any()):
            return codes
        sel = {m.name: sub.unsqueeze(-1).expand_as(codes[m.name]).contiguous()
               for m in self.cfg.modalities}
        # cache_enabled=False is LOAD-BEARING (see the docstring) — do not "simplify" it away.
        with torch.no_grad(), torch.autocast(device_type=ref.device.type, dtype=torch.bfloat16,
                                             enabled=torch.is_autocast_enabled(),
                                             cache_enabled=False):
            h = self.backbone.encode(codes, actuators, text=text)
            slog = self.backbone.tok.masked_logits(h, sel)       # {name: (n_sel_m, vocab_m)}
        out = {}
        for m in self.cfg.modalities:
            lg = slog[m.name]
            if lg.numel() == 0:
                out[m.name] = codes[m.name]
                continue
            # chunk the categorical draw: vocab reaches 64000, so a one-shot softmax over every
            # selected position is hundreds of MB in fp32 for no benefit.
            samp = torch.empty(lg.shape[0], dtype=torch.long, device=lg.device)
            for i in range(0, lg.shape[0], 4096):
                p = lg[i:i + 4096].float().softmax(-1)
                samp[i:i + 4096] = torch.multinomial(p, 1, generator=gen).squeeze(-1)
            c = codes[m.name].clone()
            c[sel[m.name]] = samp.to(c.dtype)
            out[m.name] = c
        return out

    def _modality_weight(self, m) -> float:
        mode = getattr(self.cfg, "modality_loss_weight", "uniform")
        if mode == "uniform":
            return 1.0
        if mode == "tokens":
            return float(m.n_tok)
        if mode == "sqrt_tokens":
            return float(m.n_tok) ** 0.5
        raise ValueError(f"unknown modality_loss_weight {mode!r}")

    def training_loss(self, codes: Dict[str, torch.Tensor], actuators: torch.Tensor,
                      generator: Optional[torch.Generator] = None,
                      ss_frac: float = 0.0,
                      present: Optional[torch.Tensor] = None,
                      text: Optional[torch.Tensor] = None,
                      per_modality: Optional[Dict[str, float]] = None,
                      mask_ratio: Optional[float] = None,
                      history_frames: int = 0,
                      mod_weights: Optional[Dict[str, float]] = None,
                      gen_mask_p: Optional[float] = None,
                      label_smoothing: float = 0.0) -> torch.Tensor:
        """Masked-token cross-entropy over the masked positions, summed per modality, mean-reduced.

        ``label_smoothing`` is an EXPLICIT ARGUMENT, never read from cfg, because _val_loss calls
        this same method: smoothing the validation CE would raise it by ~eps*ln(vocab) (1.15 nats at
        eps=0.1 for this frame layout) and make the metric incomparable to every arm ever run. Only
        the training call site passes it; validation keeps the default 0.0.

        ``ss_frac`` > 0 applies scheduled sampling: a fraction of context frames are replaced by
        the model's own codes before masking; the CE target remains the REAL codes.

        ``cfg.ctf_frac`` > 0 makes that fraction of windows use the complete-context (CTF) layout
        (:meth:`_boundary_mask`) instead of the random per-frame mask, i.e. the clean-prefix /
        masked-suffix conditional that rollout actually samples from.

        ``present`` (B, n_modalities), 1.0 = the diagnostic recorded in that sample. Absent
        diagnostics encode to a constant null codeword; scoring them teaches the model to
        reproduce a constant and, since this loss weights every modality EQUALLY regardless of
        token count, they account for ~38% of the terms on the production cache. Their tokens
        are excluded from the CE while REMAINING as model input (that a diagnostic was off is
        real information about the shot).

        Absent modalities are zero-WEIGHTED rather than skipped: a skipped modality's output
        head would receive no gradient, which trips DDP's unused-parameter detection. A
        zero-weighted term keeps every parameter in the graph with a zero gradient.
        """
        context = self._scheduled_sample_context(codes, actuators, ss_frac, generator, text=text)
        # A caller that pins the mask (``mask_ratio`` given: _gen_val_loss's cold-start
        # diagnostic) asked for exactly that layout. The SF and CTF arms below are
        # TRAINING curricula; letting them win here would silently score a random
        # CTF/SF window under --ctf_frac>0 / --sf_frames>0 and call it the generation
        # loss. Skip both arms without touching the RNG so the pinned path is a bare
        # _random_mask stream.
        pinned = mask_ratio is not None
        n_sf = 0 if pinned else int(getattr(self.cfg, "sf_frames", 0))
        sf_at = None                  # end of the self-rolled window, once one has been built
        if n_sf > 0:
            ref_ctx = context[self.cfg.modalities[0].name]
            Fr = ref_ctx.shape[1]
            # A usable window needs >=1 real seed frame, n_sf rolled frames, and >=1 frame after
            # them. Anything shorter could only draw and then no-op, so bail before touching the
            # RNG rather than perturbing the stream for nothing.
            if Fr >= n_sf + 2:
                # Draw on the CONTEXT's device, same convention as the CTF gate below: a
                # device-typed generator (as _val_loss passes on GPU) only accepts draws on its
                # own device, so an unqualified draw raises there. On CPU the explicit device
                # leaves the stream bit-identical. Both draws stay .item()-converted so the
                # control flow below is plain Python scalars.
                sdev = ref_ctx.device
                use_sf = (self.cfg.sf_prob >= 1.0
                          or bool(torch.rand((), generator=generator, device=sdev).item()
                                  < self.cfg.sf_prob))
                if use_sf:
                    lo = max(1, min(self.cfg.k0_seed, max(1, Fr - n_sf - 1)))
                    b = int(torch.randint(lo, max(lo + 1, Fr - n_sf), (1,),
                                          generator=generator, device=sdev).item())
                    context = rollout_context(self, context, actuators, boundary=b,
                                              n_roll=n_sf, generator=generator, text=text)
                    # The CTF mask must not overwrite what we just rolled — those frames ARE the
                    # on-policy context this arm exists to train on. Floor the CTF boundary at the
                    # window's end so the rolled frames stay complete context and the supervised
                    # suffix follows them: the Self-Forcing conditional, structurally.
                    sf_at = min(b + n_sf, Fr - 1)
        use_ctf = False
        if not pinned and self.cfg.ctf_frac > 0.0:
            # Draw on the CODES' device: a device-typed generator (as _val_loss passes on GPU)
            # only accepts draws on its own device. Same convention as _random_mask; on CPU the
            # explicit device leaves the RNG stream bit-identical.
            cdev = codes[self.cfg.modalities[0].name].device
            use_ctf = bool(torch.rand((), generator=generator, device=cdev).item()
                           < self.cfg.ctf_frac)
        if use_ctf:
            masked, mask = (self._boundary_mask(context, generator, min_boundary=sf_at)
                            if sf_at is not None else self._boundary_mask(context, generator))
        else:
            masked, mask = self._random_mask(context, generator, ratio_override=mask_ratio,
                                             history_frames=history_frames,
                                             gen_mask_p=gen_mask_p)
        # Memory-critical: encode to hidden states, then project ONLY masked positions to vocab.
        # Materializing full (B, F, tokens_per_frame, vocab) logits is ~B*400 GB at F=100 (OOM);
        # masked_logits gathers first -> ~B*200 MB. See FrameTokenizer.masked_logits.
        h = self.backbone.encode(masked, actuators, text=text)         # (B, F, N, d)
        mlogits = self.backbone.tok.masked_logits(h, mask)            # {name: (n_masked, vocab)}
        total, count = codes[self.cfg.modalities[0].name].new_zeros((), dtype=torch.float32), 0.0
        for mi, m in enumerate(self.cfg.modalities):
            mk = mask[m.name]
            if not bool(mk.any()):
                continue
            lg = mlogits[m.name]                                      # (n_masked, vocab)
            tg = codes[m.name][mk]                                    # (n_masked,)
            w_m = self._modality_weight(m)
            if present is None:
                term = F.cross_entropy(lg, tg, label_smoothing=label_smoothing)
                wt = w_m * (1.0 if mod_weights is None else float(mod_weights.get(m.name, 1.0)))
                total = total + wt * term
                count += wt
                if per_modality is not None:
                    per_modality[m.name] = float(term.detach())
                continue
            # Per-SAMPLE presence: a batch mixes shots, so weight each masked token by
            # whether its own shot recorded this diagnostic. Broadcast (B,) over (B, F, n_tok)
            # and gather with the same boolean mask used to gather the logits, so weights and
            # tokens stay aligned.
            B = mk.shape[0]
            w = present[:, mi].view(B, *([1] * (mk.dim() - 1))).expand_as(mk)[mk].to(lg.dtype)
            denom = w.sum()
            if not bool(denom > 0):        # absent in EVERY sample of this batch
                total = total + 0.0 * lg.sum()        # keep the head in the autograd graph
                continue
            ce = F.cross_entropy(lg, tg, reduction="none",
                                 label_smoothing=label_smoothing)   # (n_masked,)
            term = (ce * w).sum() / denom
            wt = w_m * (1.0 if mod_weights is None else float(mod_weights.get(m.name, 1.0)))
            total = total + wt * term
            count += wt
            if per_modality is not None:
                per_modality[m.name] = float(term.detach())
        return total / max(count, 1e-8)

    # ---------------------------------------------------------------------------- inference ---
    def _decode_logits(self, seq, actuators, sampler, text=None):
        """Per-modality last-frame logits, with optional classifier-free guidance.

        cfg_scale == 1.0 short-circuits to ONE forward pass, so guidance costs nothing
        when it is off (and the default path stays bit-identical).
        """
        h = self.backbone.encode(seq, actuators, text=text)
        cond = self.backbone.tok.logits_last(h)
        if sampler.cfg_scale == 1.0:
            return cond
        # fully unconditional: the guidance baseline drops BOTH actuators and text.
        hu = self.backbone.encode(seq, actuators, drop_actuators=True, text=text, drop_text=True)
        uncond = self.backbone.tok.logits_last(hu)
        s = float(sampler.cfg_scale)
        return {n: uncond[n] + s * (cond[n] - uncond[n]) for n in cond}

    @torch.no_grad()
    def generate_frame(self, past_codes: Dict[str, torch.Tensor], actuators: torch.Tensor,
                       temperature: float = 1.0,
                       generator: Optional[torch.Generator] = None,
                       sampler: Optional[SamplerConfig] = None,
                       text: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Generate ONE next frame's committed codes given committed past frames.

        past_codes[.]: (B, P, n_tok) real codes for P past frames. actuators: (B, P+1, actuator_dim)
        (through the frame being generated — causal). Returns {name: (B, n_tok)} committed codes.
        """
        cfg = self.cfg
        sampler = SamplerConfig(temperature=temperature) if sampler is None else sampler
        ref = past_codes[cfg.modalities[0].name]
        B, P, _ = ref.shape
        dev = ref.device
        # start the new frame fully masked, append it to the past
        cur = {m.name: torch.full((B, m.n_tok), self.backbone.tok.mask_ids[m.name],
                                  dtype=torch.long, device=dev) for m in cfg.modalities}
        revealed = {m.name: torch.zeros((B, m.n_tok), dtype=torch.bool, device=dev)
                    for m in cfg.modalities}
        keep_masked = _cosine_keep_fractions(cfg.maskgit_decode_steps)
        for step, frac in enumerate(keep_masked):
            seq = {n: torch.cat([past_codes[n], cur[n].unsqueeze(1)], dim=1) for n in cur}
            logits = self._decode_logits(seq, actuators, sampler, text=text)  # {name:(B, n_tok, vocab)}
            # Three passes — sample all / decide reveals / commit. Splitting the old
            # single-pass loop lets the reveal POLICY see every modality's confidence at once
            # while leaving the sampling RNG order untouched (same multinomial calls, same
            # modality order), which is what keeps the default path bit-identical.
            # PASS 1 — sample every modality
            samp, conf = {}, {}
            for m in cfg.modalities:
                # .float(): the backbone may run under bf16 autocast (eval speed, matches
                # training numerics) but softmax/multinomial sample in fp32 — multinomial
                # does not support bf16 and low-precision probs would skew sampling.
                lg = logits[m.name].float() / max(sampler.temp_for(m.name), 1e-6)
                prob = apply_top_p(lg.softmax(-1), sampler.top_p)
                s = torch.multinomial(prob.reshape(-1, prob.shape[-1]), 1,
                                      generator=generator).reshape(B, m.n_tok)
                samp[m.name] = s
                c = prob.gather(-1, s.unsqueeze(-1)).squeeze(-1)   # (B, n_tok)
                # ANNEALED-GUMBEL REVEAL (canonical MaskGIT; opt-in via
                # IGNITE_MASKGIT_GUMBEL). Ranking purely on p(sampled) is
                # deterministic-greedy: the tokens revealed first are those whose
                # sample happened to land on the MARGINAL MODE, and the remaining
                # steps then condition on that seed. Measured 2026-08-13 on the
                # production ckpt at step 12500: p_max is only 0.15-0.21 for the
                # turbulent spectros (perplexity 27-57 against a 64000 vocab), so
                # the ranking carries no real confidence signal and the decode
                # cascades into one code -- co2 emitted a SINGLE code for 95% of
                # tokens and the rendered spectrograms/cameras were flat. The
                # Gumbel term makes the reveal order stochastic rather than a
                # deterministic function of probability; it anneals to 0 so late
                # steps still prefer genuinely confident tokens.
                #   Perturb in LOG space (log c + ann*g) but hand the policies a
                # POSITIVE score: _global_reveal ranks on norm_log_confidence
                # (1 + log c / log V), which needs c > 0. exp() of the perturbed
                # log is the same ordering under BOTH reveal policies, and the
                # clamp keeps the exponent inside fp32 for a large scale setting.
                # Default 0 draws no rand at all, so the historical greedy path
                # stays bit-identical (same RNG stream).
                if _GUMBEL_SCALE > 0.0:
                    u = torch.rand(c.shape, generator=generator,
                                   device=dev).clamp_(1e-9, 1 - 1e-9)
                    ann = _GUMBEL_SCALE * (1.0 - step / max(1, len(keep_masked)))
                    g = -torch.log(-torch.log(u))
                    c = (c.clamp_min(1e-12).log() + ann * g).clamp_(max=80.0).exp()
                conf[m.name] = c
            # PASS 2 — choose what to reveal
            if sampler.global_pool:
                take = self._global_reveal(conf, revealed, frac)
            else:
                take = self._per_modality_reveal(conf, revealed, frac)
            # PASS 3 — commit
            for m in cfg.modalities:
                cur[m.name] = torch.where(take[m.name], samp[m.name], cur[m.name])
                revealed[m.name] = revealed[m.name] | take[m.name]
        # REVISION (draft-and-revise): the schedule above never revisits a committed token,
        # so an incoherent early commit is permanent. Re-mask the least-confident fraction
        # and re-decode it against the tokens that survived.
        for _ in range(int(sampler.revision_rounds)):
            seq = {n: torch.cat([past_codes[n], cur[n].unsqueeze(1)], dim=1) for n in cur}
            logits = self._decode_logits(seq, actuators, sampler, text=text)
            samp, conf = {}, {}
            for m in cfg.modalities:
                lg = logits[m.name].float() / max(sampler.temp_for(m.name), 1e-6)
                prob = apply_top_p(lg.softmax(-1), sampler.top_p)
                s = torch.multinomial(prob.reshape(-1, prob.shape[-1]), 1,
                                      generator=generator).reshape(B, m.n_tok)
                samp[m.name] = s
                # confidence of the CURRENTLY COMMITTED code, not of the fresh draw:
                # that is what decides which commits look weakest in context.
                conf[m.name] = prob.gather(-1, cur[m.name].unsqueeze(-1)).squeeze(-1)
            for m in cfg.modalities:
                k = int(round(sampler.revision_frac * m.n_tok))
                if k <= 0:
                    continue
                weakest = conf[m.name].argsort(dim=-1)[:, :k]          # lowest confidence
                redo = torch.zeros_like(revealed[m.name])
                redo.scatter_(1, weakest, True)
                cur[m.name] = torch.where(redo, samp[m.name], cur[m.name])
        # any still-masked (numeric edge) -> final argmax
        for m in cfg.modalities:
            still = ~revealed[m.name]
            if bool(still.any()):
                seq = {n: torch.cat([past_codes[n], cur[n].unsqueeze(1)], dim=1) for n in cur}
                h = self.backbone.encode(seq, actuators, text=text)
                lg = self.backbone.tok.logits_last(h)[m.name]
                cur[m.name] = torch.where(still, lg.argmax(-1), cur[m.name])
        return cur

    # ---- reveal policies: given this step's confidences, which tokens to commit ---------
    def _per_modality_reveal(self, conf, revealed, frac):
        """Original policy: each modality reveals the same FRACTION of its own tokens."""
        take = {}
        for m in self.cfg.modalities:
            c = conf[m.name].masked_fill(revealed[m.name], float("inf"))
            n_reveal = m.n_tok - int(round(frac * m.n_tok))
            order = c.argsort(dim=-1, descending=True)
            new_rev = torch.zeros_like(revealed[m.name])
            new_rev.scatter_(1, order[:, :n_reveal], True)
            take[m.name] = new_rev & ~revealed[m.name]
        return take

    def _global_reveal(self, conf, revealed, frac):
        """Pooled policy: score the WHOLE frame on one comparable scale, reveal the global top-K.

        Lets an uncertain modality defer while confident ones commit first and anchor it
        through the next step's spatial attention — the cross-modal-coherence lever.

        Scoring is ``norm_log_confidence`` (1 + log c / log V), NOT a within-modality rank:
        ranks put every modality on the same {0..1} grid, which makes the allocation
        token-count-proportional — identical to the fixed quota — for ANY confidences. The
        log-vocab form removes the vocab-size scale while keeping the between-modality signal.

        The vocab term only bites on MIXED-vocab layouts (the cache-derived production set: four
        64k-vocab spectro modalities beside 1k-vocab slow-TS). Under a uniform vocab — the static
        ``FROZEN_MODALITIES`` table and the bp pilot line — ``log V`` is a common divisor, so the
        pool reduces to raw-probability top-K over the frame; still cross-modal and
        confidence-driven, just with no vocab skew left to correct.
        """
        parts = [norm_log_confidence(conf[m.name], m.codebook_size)
                 .masked_fill(revealed[m.name], float("inf")) for m in self.cfg.modalities]
        flat = torch.cat(parts, dim=1)                       # (B, tokens_per_frame)
        total = flat.shape[1]
        n_reveal = total - int(round(frac * total))
        # stable: exact score ties (equal vocab AND equal confidence) must not resolve by
        # sort-backend luck, or pooled reveal order stops being reproducible across devices.
        order = flat.argsort(dim=-1, descending=True, stable=True)
        sel = torch.zeros_like(flat, dtype=torch.bool)
        sel.scatter_(1, order[:, :n_reveal], True)
        take, off = {}, 0
        for m in self.cfg.modalities:
            take[m.name] = sel[:, off:off + m.n_tok] & ~revealed[m.name]
            off += m.n_tok
        return take

    @torch.no_grad()
    def rollout(self, seed_codes: Dict[str, torch.Tensor], actuators: torch.Tensor,
                n_predict: Optional[int] = None, temperature: float = 1.0,
                generator: Optional[torch.Generator] = None,
                sampler: Optional[SamplerConfig] = None,
                text: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Seed K₀ real frames -> generate + COMMIT n_predict frames. Closed code space, no
        decode/re-tokenize round-trip.

        seed_codes[.]: (B, K0, n_tok). actuators: (B, K0 + n_predict, actuator_dim).
        Returns full trajectory {name: (B, K0 + n_predict, n_tok)}.
        """
        cfg = self.cfg
        n_predict = cfg.n_predict if n_predict is None else n_predict
        traj = {n: v.clone() for n, v in seed_codes.items()}
        K0 = traj[cfg.modalities[0].name].shape[1]
        if actuators.shape[1] < K0 + n_predict:
            raise ValueError(
                f"actuators has {actuators.shape[1]} frames; need K0+n_predict={K0 + n_predict}"
            )
        for t in range(n_predict):
            nxt = self.generate_frame(
                traj, actuators[:, : K0 + t + 1], temperature=temperature,
                generator=generator, sampler=sampler, text=text
            )
            traj = {n: torch.cat([traj[n], nxt[n].unsqueeze(1)], dim=1) for n in traj}
        return traj
