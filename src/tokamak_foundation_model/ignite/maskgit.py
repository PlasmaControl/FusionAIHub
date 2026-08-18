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
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dynamics import DynamicsBackbone
from .dynamics_config import DynamicsConfig
from .sampling import SamplerConfig, apply_top_p, rank_normalize


def _cosine_keep_fractions(n_steps: int) -> list:
    """Fraction of a frame's tokens that remain MASKED after decode step i (cosine schedule).

    Starts fully masked (1.0 before step 0), ends fully revealed (0.0 after the last step).
    """
    return [math.cos(math.pi / 2 * (i + 1) / n_steps) for i in range(n_steps)]


class MaskGITDynamics(nn.Module):
    def __init__(self, cfg: DynamicsConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = DynamicsBackbone(cfg)

    # ---------------------------------------------------------------------------- training ---
    def _random_mask(self, codes: Dict[str, torch.Tensor], gen: Optional[torch.Generator]):
        """Mask a per-frame random fraction of tokens. Returns (masked_codes, mask) with the same
        keys; masked positions in masked_codes hold the modality's MASK id, mask[.]=True there."""
        cfg = self.cfg
        ref = codes[cfg.modalities[0].name]
        B, Fr, _ = ref.shape
        dev = ref.device
        # one mask ratio per (B, frame), in [0, 1] via a cosine of a uniform (MaskGIT prior)
        u = torch.rand((B, Fr), generator=gen, device=dev)
        ratio = torch.cos(math.pi / 2 * (1.0 - u)).clamp(1e-3, 1.0)   # bias toward higher masking
        masked, mask = {}, {}
        for m in cfg.modalities:
            c = codes[m.name]
            r = torch.rand(c.shape, generator=gen, device=dev)
            mk = r < ratio.unsqueeze(-1)                              # (B, F, n_tok) bool
            # guarantee >=1 masked token per frame so CE always has a target
            none = ~mk.any(dim=-1, keepdim=True)
            if none.any():
                mk = mk | (F.one_hot(torch.zeros(1, dtype=torch.long, device=dev), c.shape[-1])
                           .bool().view(1, 1, -1) & none)
            mc = torch.where(mk, torch.full_like(c, self.backbone.tok.mask_ids[m.name]), c)
            masked[m.name], mask[m.name] = mc, mk
        return masked, mask

    def ss_fraction(self, step: int) -> float:
        """Scheduled-sampling own-code fraction: linear ramp 0 -> ss_ramp_final_frac."""
        cfg = self.cfg
        return min(1.0, step / max(1, cfg.ss_ramp_steps)) * cfg.ss_ramp_final_frac

    @torch.no_grad()
    def _scheduled_sample_context(self, codes, actuators, ss_frac, gen):
        """Return a copy of ``codes`` with a ``ss_frac`` fraction of (B, frame) cells replaced by
        the model's OWN one-pass sampled codes — so the model learns to consume its own outputs
        (drift mitigation). Substitution is on the CONTEXT only; the CE target stays the real code."""
        if ss_frac <= 0.0:
            return codes
        logits = self.backbone(codes, actuators)
        ref = codes[self.cfg.modalities[0].name]
        B, Fr, _ = ref.shape
        sub = torch.rand((B, Fr), generator=gen, device=ref.device) < ss_frac   # (B, F) frames
        out = {}
        for m in self.cfg.modalities:
            prob = logits[m.name].softmax(-1)                       # (B, F, n_tok, vocab)
            samp = torch.multinomial(prob.reshape(-1, prob.shape[-1]), 1,
                                     generator=gen).reshape(codes[m.name].shape)
            out[m.name] = torch.where(sub.unsqueeze(-1), samp, codes[m.name])
        return out

    def training_loss(self, codes: Dict[str, torch.Tensor], actuators: torch.Tensor,
                      generator: Optional[torch.Generator] = None,
                      ss_frac: float = 0.0,
                      present: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Masked-token cross-entropy over the masked positions, summed per modality, mean-reduced.

        ``ss_frac`` > 0 applies scheduled sampling: a fraction of context frames are replaced by
        the model's own codes before masking; the CE target remains the REAL codes.

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
        context = self._scheduled_sample_context(codes, actuators, ss_frac, generator)
        masked, mask = self._random_mask(context, generator)
        # Memory-critical: encode to hidden states, then project ONLY masked positions to vocab.
        # Materializing full (B, F, tokens_per_frame, vocab) logits is ~B*400 GB at F=100 (OOM);
        # masked_logits gathers first -> ~B*200 MB. See FrameTokenizer.masked_logits.
        h = self.backbone.encode(masked, actuators)                   # (B, F, N, d)
        mlogits = self.backbone.tok.masked_logits(h, mask)            # {name: (n_masked, vocab)}
        total, count = codes[self.cfg.modalities[0].name].new_zeros((), dtype=torch.float32), 0
        for mi, m in enumerate(self.cfg.modalities):
            mk = mask[m.name]
            if not bool(mk.any()):
                continue
            lg = mlogits[m.name]                                      # (n_masked, vocab)
            tg = codes[m.name][mk]                                    # (n_masked,)
            if present is None:
                total = total + F.cross_entropy(lg, tg)
                count += 1
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
            ce = F.cross_entropy(lg, tg, reduction="none")            # (n_masked,)
            total = total + (ce * w).sum() / denom
            count += 1
        return total / max(count, 1)

    # ---------------------------------------------------------------------------- inference ---
    @torch.no_grad()
    def generate_frame(self, past_codes: Dict[str, torch.Tensor], actuators: torch.Tensor,
                       temperature: float = 1.0,
                       generator: Optional[torch.Generator] = None,
                       sampler: Optional[SamplerConfig] = None) -> Dict[str, torch.Tensor]:
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
            h = self.backbone.encode(seq, actuators)                  # (B, P+1, N, d)
            logits = self.backbone.tok.logits_last(h)                 # {name:(B, n_tok, vocab)}
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
                conf[m.name] = prob.gather(-1, s.unsqueeze(-1)).squeeze(-1)   # (B, n_tok)
            # PASS 2 — choose what to reveal
            if sampler.global_pool:
                take = self._global_reveal(conf, revealed, frac)
            else:
                take = self._per_modality_reveal(conf, revealed, frac)
            # PASS 3 — commit
            for m in cfg.modalities:
                cur[m.name] = torch.where(take[m.name], samp[m.name], cur[m.name])
                revealed[m.name] = revealed[m.name] | take[m.name]
        # any still-masked (numeric edge) -> final argmax
        for m in cfg.modalities:
            still = ~revealed[m.name]
            if bool(still.any()):
                seq = {n: torch.cat([past_codes[n], cur[n].unsqueeze(1)], dim=1) for n in cur}
                h = self.backbone.encode(seq, actuators)
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
        """Pooled policy: rank confidence across the WHOLE frame, reveal the global top-K.

        Lets an uncertain modality defer while confident ones commit first and anchor it
        through the next step's spatial attention — the cross-modal-coherence lever.
        """
        names = [m.name for m in self.cfg.modalities]
        parts = [rank_normalize(conf[n]).masked_fill(revealed[n], float("inf")) for n in names]
        flat = torch.cat(parts, dim=1)                       # (B, tokens_per_frame)
        total = flat.shape[1]
        n_reveal = total - int(round(frac * total))
        order = flat.argsort(dim=-1, descending=True)
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
                sampler: Optional[SamplerConfig] = None) -> Dict[str, torch.Tensor]:
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
                generator=generator, sampler=sampler
            )
            traj = {n: torch.cat([traj[n], nxt[n].unsqueeze(1)], dim=1) for n in traj}
        return traj
