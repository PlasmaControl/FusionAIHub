"""MaskGIT dynamics: masked-token training, iterative decoding and code rollout.

Training masks a random fraction of each frame's tokens and predicts them, through
the causal temporal attention, from past frames, visible tokens and actuators.
Inference generates a frame from all-masked by iterative confidence-ordered
unmasking. A rollout seeds ``K0`` real frames, then repeatedly generates a frame,
commits its discrete codes and appends it. See ``docs/IGNITE_DESIGN.md`` sections
5.3 to 5.6.

All randomness goes through an optional ``torch.Generator`` so tests and resumed
runs are reproducible.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dynamics import DynamicsBackbone, KVCache
from .dynamics_config import DynamicsConfig
from .frame_layout import code_to_coords, coords_to_code
from .sampling import SamplerConfig, apply_top_p, norm_log_confidence
from .selfforce import rollout_context

# Positions per chunk when sampling categorically; vocabularies reach 64000.
_SAMPLE_CHUNK = 4096


def _split_gaussian(out: torch.Tensor, levels):
    """Split latent-head output into a mean and a clamped log-variance."""
    n_dims = len(levels)
    mean = out[..., :n_dims].float()
    log_var = out[..., n_dims:].float().clamp(-10.0, 10.0)
    return mean, log_var


def _latent_nll(out: torch.Tensor, target: torch.Tensor, levels) -> torch.Tensor:
    """Gaussian negative log-likelihood of each target's normalised FSQ coordinate.

    Parameters
    ----------
    out : torch.Tensor of shape (n, 2 * len(levels))
        Latent-head output: mean, then log-variance, per FSQ dimension.
    target : torch.LongTensor of shape (n,)
        Target codes.
    levels : sequence of int
        FSQ levels.

    Returns
    -------
    nll : torch.Tensor of shape (n,)
        Per-token loss in nats, up to a constant. The log-variance is clamped so a
        single confident miss cannot dominate.
    """
    mean, log_var = _split_gaussian(out, levels)
    coords = code_to_coords(target, levels)
    return 0.5 * (((mean - coords) ** 2) * torch.exp(-log_var) + log_var).sum(-1)


def _latent_sample(out: torch.Tensor, levels, temperature: float, generator):
    """Sample codes from the latent head.

    Parameters
    ----------
    out : torch.Tensor of shape (..., 2 * len(levels))
        Latent-head output.
    levels : sequence of int
        FSQ levels.
    temperature : float
        Scale of the sampling noise; ``0`` returns the rounded mean.
    generator : torch.Generator or None
        Source of the sampling noise.

    Returns
    -------
    codes : torch.LongTensor of shape (...)
        Sampled codes, rounded to the nearest grid cell.
    confidence : torch.Tensor of shape (...)
        Negative summed log-variance, used as the reveal order.
    mean_codes : torch.LongTensor of shape (...)
        Codes of the rounded mean.
    """
    mean, log_var = _split_gaussian(out, levels)
    if temperature > 0.0:
        noise = torch.randn(
            mean.shape, generator=generator, device=mean.device, dtype=mean.dtype
        )
        coords = mean + float(temperature) * torch.exp(0.5 * log_var) * noise
    else:
        coords = mean
    return (
        coords_to_code(coords, levels),
        -log_var.sum(-1),
        coords_to_code(mean, levels),
    )


def _cosine_keep_fractions(n_steps: int) -> list:
    """Return the fraction of a frame still masked after each decode step.

    The schedule falls from just below 1 after the first step to 0 after the last.
    """
    return [math.cos(math.pi / 2 * (i + 1) / n_steps) for i in range(n_steps)]


def placeholders(seed_codes: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Find the rows whose seed frames all carry identical codes.

    An absent diagnostic encodes to one constant frame, so a modality that does not
    change once in the ``K0`` real frames is a placeholder; ``rollout(hold=...)``
    keeps it at that code instead of sampling a signal that was never measured.

    Parameters
    ----------
    seed_codes : dict of str to torch.LongTensor of shape (B, K0, n_tok)
        Real seed frames.

    Returns
    -------
    hold : dict of str to torch.BoolTensor of shape (B,)
        True for the placeholder rows; modalities without one are left out.
    """
    out = {}
    for name, codes in seed_codes.items():
        rows = (codes == codes[:, :1]).flatten(1).all(dim=1)
        if bool(rows.any()):
            out[name] = rows
    return out


def _text_kwargs(text) -> dict:
    """``{"text": text}``, or nothing without text: the default path then calls the
    backbone exactly as it did before text conditioning existed."""
    return {} if text is None else {"text": text}


class _CategoricalOutput:
    """Codebook classifier head: logits over the modality's codes."""

    def loss(self, logits, target, label_smoothing, reduction):
        """Cross-entropy per token (``reduction="none"``) or its mean."""
        return F.cross_entropy(
            logits, target, reduction=reduction, label_smoothing=label_smoothing
        )

    @staticmethod
    def _probs(logits, temperature, top_p):
        # Sample in fp32: the backbone may run under bf16 autocast, and multinomial
        # does not take bf16.
        return apply_top_p((logits.float() / max(temperature, 1e-6)).softmax(-1), top_p)

    def sample(self, logits, temperature, generator, top_p=None):
        """Sample (B, n_tok) codes; return them with their probability."""
        probs = self._probs(logits, temperature, top_p)
        codes = torch.multinomial(
            probs.reshape(-1, probs.shape[-1]), 1, generator=generator
        ).reshape(probs.shape[:-1])
        confidence = probs.gather(-1, codes.unsqueeze(-1)).squeeze(-1)
        return codes, confidence

    def resample(self, logits, temperature, generator, top_p, current):
        """Sample fresh codes; return them with the probability of ``current``."""
        probs = self._probs(logits, temperature, top_p)
        codes = torch.multinomial(
            probs.reshape(-1, probs.shape[-1]), 1, generator=generator
        ).reshape(probs.shape[:-1])
        return codes, probs.gather(-1, current.unsqueeze(-1)).squeeze(-1)

    def sample_flat(self, logits, generator):
        """Sample one code per row of (n, vocab) logits at temperature 1."""
        codes = torch.empty(logits.shape[0], dtype=torch.long, device=logits.device)
        for start in range(0, logits.shape[0], _SAMPLE_CHUNK):
            stop = start + _SAMPLE_CHUNK
            probs = logits[start:stop].float().softmax(-1)
            codes[start:stop] = torch.multinomial(
                probs, 1, generator=generator
            ).squeeze(-1)
        return codes

    def mode(self, logits, generator):
        """Most likely code."""
        return logits.argmax(-1)

    def reveal_score(self, confidence):
        """Confidence on the log scale, to which the Gumbel noise is added."""
        return confidence.clamp_min(1e-12).log()

    def from_reveal_score(self, score):
        """Back to a positive confidence, as the pooled reveal needs.

        ``exp`` keeps the order under both reveal policies; the clamp keeps the
        exponent inside fp32.
        """
        return score.clamp_(max=80.0).exp()


class _GaussianOutput:
    """Latent head: a Gaussian over the token's FSQ grid coordinate."""

    def __init__(self, levels):
        self.levels = levels

    def loss(self, out, target, label_smoothing, reduction):
        """Gaussian NLL per token (``reduction="none"``) or its mean."""
        nll = _latent_nll(out, target, self.levels)
        return nll.mean() if reduction == "mean" else nll

    def sample(self, out, temperature, generator, top_p=None):
        """Sample codes; return them with the negative summed log-variance."""
        if top_p is not None:
            raise ValueError("top_p needs a categorical head; the latent head has none")
        codes, confidence, _ = _latent_sample(out, self.levels, temperature, generator)
        return codes, confidence

    def sample_flat(self, out, generator):
        """Sample one code per row at temperature 1."""
        return _latent_sample(out, self.levels, 1.0, generator)[0]

    def mode(self, out, generator):
        """Code of the rounded mean."""
        return _latent_sample(out, self.levels, 0.0, generator)[2]

    def reveal_score(self, confidence):
        """The confidence is already on a log scale."""
        return confidence

    def from_reveal_score(self, score):
        """The per-modality reveal ranks the log-scale score directly."""
        return score


class MaskGITDynamics(nn.Module):
    """Masked-token dynamics model over frozen codec codes.

    Parameters
    ----------
    cfg : DynamicsConfig
        Model configuration.
    """

    def __init__(self, cfg: DynamicsConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = DynamicsBackbone(cfg)
        tokenizer = self.backbone.tok
        self.output = (
            _GaussianOutput(tokenizer.fsq_levels)
            if tokenizer.latent_head
            else _CategoricalOutput()
        )

    def _mask_id(self, name: str) -> int:
        return self.backbone.tok.mask_ids[name]

    def _gen_split_points(self, batch: int, n_frames: int, gen, device) -> torch.Tensor:
        """Draw the generation-mode split frame of each sample, shape (B, 1).

        The horizon ``h = n_frames - t`` is uniform in ``[1, gen_horizon_max]`` if
        that is set, follows ``h**-gen_horizon_alpha`` if that is positive, and
        otherwise the split frame is uniform in ``[k0_seed, n_frames)``.
        """
        cfg = self.cfg
        first = int(min(max(cfg.k0_seed, 0), n_frames - 1))
        span = max(n_frames - first, 1)
        max_horizon = int(cfg.gen_horizon_max or 0)
        alpha = float(cfg.gen_horizon_alpha or 0.0)
        if max_horizon > 0:
            high = max(1, min(max_horizon, span))
            horizon = torch.randint(
                1, high + 1, (batch, 1), generator=gen, device=device
            )
            return (n_frames - horizon).clamp_(first, n_frames - 1)
        if alpha > 0.0:
            horizons = torch.arange(1, span + 1, device=device, dtype=torch.float32)
            weights = horizons.pow(-alpha)
            draws = torch.multinomial(
                weights / weights.sum(), batch, replacement=True, generator=gen
            )
            horizon = draws.view(batch, 1) + 1
            return (n_frames - horizon).clamp_(first, n_frames - 1)
        uniform = torch.rand((batch, 1), generator=gen, device=device)
        return first + (uniform * span).long().clamp_(0, span - 1)

    def _random_mask(
        self,
        codes: Dict[str, torch.Tensor],
        gen: Optional[torch.Generator],
        ratio_override: Optional[float] = None,
        history_frames: int = 0,
        gen_mask_p: Optional[float] = None,
    ):
        """Mask a random fraction of the tokens of every frame.

        Parameters
        ----------
        codes : dict of str to torch.LongTensor of shape (B, F, n_tok)
            Input codes.
        gen : torch.Generator or None
            Source of randomness.
        ratio_override : float, default=None
            Fixed mask ratio instead of the cosine prior, for diagnosis.
        history_frames : int, default=0
            Leading frames kept fully visible and unscored.
        gen_mask_p : float, default=None
            Fraction of samples masked like a rollout; ``None`` uses
            ``cfg.gen_mask_p``. Validation passes ``0.0`` to keep a fixed protocol.

        Returns
        -------
        masked : dict of str to torch.LongTensor of shape (B, F, n_tok)
            Codes with masked positions set to the modality's ``[MASK]`` id.
        mask : dict of str to torch.BoolTensor of shape (B, F, n_tok)
            True at masked positions. Every unprotected frame has at least one.
        """
        cfg = self.cfg
        reference = codes[cfg.modalities[0].name]
        batch, n_frames, _ = reference.shape
        device = reference.device
        generation_history = None
        if ratio_override is not None:
            ratio = torch.full(
                (batch, n_frames), float(ratio_override), device=device
            ).clamp(1e-3, 1.0)
        else:
            uniform = torch.rand((batch, n_frames), generator=gen, device=device)
            ratio = torch.cos(math.pi / 2 * (1.0 - uniform)).clamp(1e-3, 1.0)
            p_generation = float(cfg.gen_mask_p if gen_mask_p is None else gen_mask_p)
            if p_generation > 0.0:
                # Generation mode: frames before the split are real and unscored,
                # later frames follow a reveal ladder biased towards a cold start.
                in_generation_mode = (
                    torch.rand((batch, 1), generator=gen, device=device) < p_generation
                )
                split = self._gen_split_points(batch, n_frames, gen, device)
                before_split = torch.arange(n_frames, device=device).view(1, -1) < split
                ladder_draw = torch.rand(
                    (batch, n_frames), generator=gen, device=device
                )
                ladder = (1.0 - ladder_draw.pow(3.0)).clamp(1e-3, 1.0)
                generation_ratio = torch.where(
                    before_split, torch.zeros_like(ratio), ladder
                )
                ratio = torch.where(in_generation_mode, generation_ratio, ratio)
                generation_history = in_generation_mode & before_split
        n_history = int(max(0, min(history_frames, n_frames)))
        if n_history:
            ratio = ratio.clone()
            ratio[:, :n_history] = 0.0
        protected = torch.zeros((batch, n_frames), dtype=torch.bool, device=device)
        if n_history:
            protected[:, :n_history] = True
        if generation_history is not None:
            protected = protected | generation_history
        masked, mask = {}, {}
        for m in cfg.modalities:
            frame_codes = codes[m.name]
            draw = torch.rand(frame_codes.shape, generator=gen, device=device)
            modality_mask = draw < ratio.unsqueeze(-1)
            # At least one target per frame, except in the protected history.
            without_target = ~modality_mask.any(
                dim=-1, keepdim=True
            ) & ~protected.unsqueeze(-1)
            if without_target.any():
                first_token = F.one_hot(
                    torch.zeros(1, dtype=torch.long, device=device),
                    frame_codes.shape[-1],
                ).bool()
                modality_mask = modality_mask | (
                    first_token.view(1, 1, -1) & without_target
                )
            masked[m.name] = torch.where(
                modality_mask,
                torch.full_like(frame_codes, self._mask_id(m.name)),
                frame_codes,
            )
            mask[m.name] = modality_mask
        return masked, mask

    def _seed_dropout(self, codes, drop_p: float, column_p: float, gen):
        """Draw the seed tokens hidden from the input, or None if nothing is hidden.

        Per sample and modality, the whole modality is hidden in the seed frames
        with probability ``drop_p``; otherwise, with probability ``column_p``, a
        random fraction (uniform in ``[0, 1)``) of its token positions is hidden in
        every seed frame, so those columns cannot be copied from their own history.
        A modality constant over the seed (absent or static) is never hidden.

        Parameters
        ----------
        codes : dict of str to torch.LongTensor of shape (B, F, n_tok)
            Real codes.
        drop_p, column_p : float
            Probabilities of hiding the whole modality or some of its columns.
        gen : torch.Generator or None
            Source of randomness.

        Returns
        -------
        hidden : dict of str to torch.BoolTensor of shape (B, F, n_tok), or None
            True at hidden positions, all within the first ``k0_seed`` frames.
        """
        if drop_p <= 0.0 and column_p <= 0.0:
            return None
        cfg = self.cfg
        reference = codes[cfg.modalities[0].name]
        batch, n_frames, _ = reference.shape
        device = reference.device
        n_seed = int(min(max(cfg.k0_seed, 0), n_frames))
        if n_seed == 0:
            return None
        in_seed = (torch.arange(n_frames, device=device) < n_seed).view(1, -1, 1)
        hidden = {}
        for m in cfg.modalities:
            seed = codes[m.name][:, :n_seed]
            varies = ~(seed == seed[:, :1]).flatten(1).all(dim=1)
            whole = torch.rand(batch, generator=gen, device=device) < drop_p
            partial = ~whole & (
                torch.rand(batch, generator=gen, device=device) < column_p
            )
            fraction = torch.rand((batch, 1), generator=gen, device=device)
            columns = (
                torch.rand((batch, m.n_tok), generator=gen, device=device) < fraction
            )
            columns = whole.view(-1, 1) | (partial.view(-1, 1) & columns)
            columns = columns & varies.view(-1, 1)
            hidden[m.name] = columns.unsqueeze(1) & in_seed
        return hidden

    def _hide(self, codes, hidden):
        """Set the ``hidden`` positions of ``codes`` to each modality's ``[MASK]``."""
        return {
            name: torch.where(
                hidden[name], torch.full_like(frames, self._mask_id(name)), frames
            )
            for name, frames in codes.items()
        }

    def _boundary_mask(
        self,
        codes: Dict[str, torch.Tensor],
        gen: Optional[torch.Generator],
        min_boundary: int = 1,
    ):
        """Mask like a rollout for complete-context training (``cfg.ctf_frac``).

        Frames before a per-sample boundary ``c`` are complete context; frames from
        ``c`` on are heavily masked targets (ratio in ``[ctf_min_target_ratio, 1]``),
        the conditional a rollout samples from. One batch spans many context
        lengths. See ``docs/IGNITE_ROLLOUT_QUALITY_PLAN.md`` section 1A.

        Parameters
        ----------
        codes : dict of str to torch.LongTensor of shape (B, F, n_tok)
            Input codes.
        gen : torch.Generator or None
            Source of randomness.
        min_boundary : int, default=1
            Floor of the boundary. Self-forcing passes the end of its self-rolled
            window, so those frames stay complete context instead of being masked.

        Returns
        -------
        masked, mask : dict of str to torch.Tensor of shape (B, F, n_tok)
            As :meth:`_random_mask`; every target frame has at least one target.
        """
        cfg = self.cfg
        reference = codes[cfg.modalities[0].name]
        batch, n_frames, _ = reference.shape
        device = reference.device
        boundary = torch.randint(
            min_boundary,
            max(min_boundary + 1, n_frames),
            (batch,),
            generator=gen,
            device=device,
        )
        is_target = torch.arange(n_frames, device=device).view(1, -1) >= boundary.view(
            batch, 1
        )
        low = float(cfg.ctf_min_target_ratio)
        uniform = torch.rand((batch, n_frames), generator=gen, device=device)
        ratio = low + (1.0 - low) * uniform
        masked, mask = {}, {}
        for m in cfg.modalities:
            frame_codes = codes[m.name]
            draw = torch.rand(frame_codes.shape, generator=gen, device=device)
            modality_mask = (draw < ratio.unsqueeze(-1)) & is_target.unsqueeze(-1)
            without_target = ~modality_mask.any(
                dim=-1, keepdim=True
            ) & is_target.unsqueeze(-1)
            if without_target.any():
                first_token = torch.zeros_like(modality_mask)
                first_token[:, :, 0] = True
                modality_mask = modality_mask | (first_token & without_target)
            masked[m.name] = torch.where(
                modality_mask,
                torch.full_like(frame_codes, self._mask_id(m.name)),
                frame_codes,
            )
            mask[m.name] = modality_mask
        return masked, mask

    def ss_fraction(self, step: int) -> float:
        """Return the scheduled-sampling fraction at ``step`` (linear ramp)."""
        cfg = self.cfg
        return min(1.0, step / max(1, cfg.ss_ramp_steps)) * cfg.ss_ramp_final_frac

    @torch.no_grad()
    def _scheduled_sample_context(self, codes, actuators, ss_frac, gen, text=None):
        """Replace a fraction of context frames by the model's own one-pass samples.

        Only the context changes; the loss targets stay the real codes.

        Notes
        -----
        The forward pass runs with ``cache_enabled=False``. Under the caller's
        autocast, a cached no-grad forward fills the bf16 weight cache with
        detached copies that the training forward then reuses, so gradients never
        reach the parameters. Logits are gathered with ``masked_logits`` because
        full logits do not fit in memory.
        """
        if ss_frac <= 0.0:
            return codes
        reference = codes[self.cfg.modalities[0].name]
        batch, n_frames, _ = reference.shape
        replaced = (
            torch.rand((batch, n_frames), generator=gen, device=reference.device)
            < ss_frac
        )
        if not bool(replaced.any()):
            return codes
        selection = {
            m.name: replaced.unsqueeze(-1).expand_as(codes[m.name]).contiguous()
            for m in self.cfg.modalities
        }
        with torch.autocast(
            device_type=reference.device.type,
            dtype=torch.bfloat16,
            enabled=torch.is_autocast_enabled(),
            cache_enabled=False,
        ):
            hidden = self.backbone.encode(codes, actuators, **_text_kwargs(text))
            selected_logits = self.backbone.tok.masked_logits(hidden, selection)
        out = {}
        for m in self.cfg.modalities:
            logits = selected_logits[m.name]
            if logits.numel() == 0:
                out[m.name] = codes[m.name]
                continue
            samples = self.output.sample_flat(logits, gen)
            context = codes[m.name].clone()
            context[selection[m.name]] = samples.to(context.dtype)
            out[m.name] = context
        return out

    @torch.no_grad()
    def _forecast_context(self, codes, actuators, ss_frac, gen, text=None):
        """Replace a fraction of history frames by the model's own one-step forecasts.

        Every frame is forecast in one block-forcing pass from the real frames
        before it, with nothing of the frame itself visible, and sampled at
        temperature 1. Training on such a history scores forecasts made from the
        model's own earlier forecasts. ``cache_enabled=False`` as in
        :meth:`_scheduled_sample_context`.
        """
        if ss_frac <= 0.0:
            return codes
        reference = codes[self.cfg.modalities[0].name]
        batch, n_frames, _ = reference.shape
        replaced = (
            torch.rand((batch, n_frames), generator=gen, device=reference.device)
            < ss_frac
        )
        if not bool(replaced.any()):
            return codes
        selection = {
            m.name: replaced.unsqueeze(-1).expand_as(codes[m.name]).contiguous()
            for m in self.cfg.modalities
        }
        blank = {
            m.name: torch.full_like(codes[m.name], self._mask_id(m.name))
            for m in self.cfg.modalities
        }
        with torch.autocast(
            device_type=reference.device.type,
            dtype=torch.bfloat16,
            enabled=torch.is_autocast_enabled(),
            cache_enabled=False,
        ):
            hidden = self.backbone.encode_block_forcing(
                codes, blank, actuators, **_text_kwargs(text)
            )
            selected_logits = self.backbone.tok.masked_logits(hidden, selection)
        history = {}
        for m in self.cfg.modalities:
            frames = codes[m.name].clone()
            frames[selection[m.name]] = self.output.sample_flat(
                selected_logits[m.name], gen
            ).to(frames.dtype)
            history[m.name] = frames
        return history

    def _self_forcing_context(self, context, actuators, generator, text):
        """Roll out ``cfg.sf_frames`` frames of ``context`` from the model's own output.

        Self-forcing (arXiv 2506.08009): training otherwise conditions on masked
        ground truth while a rollout conditions on its own complete output. The
        rollout runs under ``no_grad`` and is detached (:func:`.selfforce.
        rollout_context`), so there is no backpropagation through time.

        Returns
        -------
        context : dict of str to torch.LongTensor of shape (B, F, n_tok)
            The context with a self-rolled window, or unchanged.
        end : int or None
            End of the self-rolled window, the floor of the CTF boundary; None if no
            window was rolled.
        """
        n_roll = int(self.cfg.sf_frames)
        reference = context[self.cfg.modalities[0].name]
        n_frames = reference.shape[1]
        # A usable window needs a real seed frame, the rolled frames and one frame
        # after them; anything shorter would draw and then do nothing.
        if n_roll <= 0 or n_frames < n_roll + 2:
            return context, None
        # Draws on the context's device: a device-typed generator only accepts its own.
        device = reference.device
        if self.cfg.sf_prob < 1.0 and not bool(
            torch.rand((), generator=generator, device=device).item() < self.cfg.sf_prob
        ):
            return context, None
        low = max(1, min(self.cfg.k0_seed, max(1, n_frames - n_roll - 1)))
        boundary = int(
            torch.randint(
                low,
                max(low + 1, n_frames - n_roll),
                (1,),
                generator=generator,
                device=device,
            ).item()
        )
        context = rollout_context(
            self,
            context,
            actuators,
            boundary=boundary,
            n_roll=n_roll,
            generator=generator,
            text=text,
        )
        # The CTF mask must not overwrite the rolled frames: they are the on-policy
        # context this arm trains on, so the supervised suffix follows them.
        return context, min(boundary + n_roll, n_frames - 1)

    def _modality_weight(self, m) -> float:
        """Loss weight of a modality under ``cfg.modality_loss_weight``."""
        mode = self.cfg.modality_loss_weight
        if mode == "uniform":
            return 1.0
        if mode == "tokens":
            return float(m.n_tok)
        if mode == "sqrt_tokens":
            return float(m.n_tok) ** 0.5
        raise ValueError(f"unknown modality_loss_weight {mode!r}")

    def training_loss(
        self,
        codes: Dict[str, torch.Tensor],
        actuators: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        ss_frac: float = 0.0,
        present: Optional[torch.Tensor] = None,
        per_modality: Optional[Dict[str, float]] = None,
        mask_ratio: Optional[float] = None,
        history_frames: int = 0,
        mod_weights: Optional[Dict[str, float]] = None,
        gen_mask_p: Optional[float] = None,
        live: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        block_forcing: bool = False,
        seed_drop_p: float = 0.0,
        seed_column_drop_p: float = 0.0,
    ) -> torch.Tensor:
        """Masked-token loss, averaged per modality and then across modalities.

        Parameters
        ----------
        codes : dict of str to torch.LongTensor of shape (B, F, n_tok)
            Real codes; they are both the input and the targets.
        actuators : torch.Tensor of shape (B, F, actuator_dim)
            Actuator conditioning.
        generator : torch.Generator, default=None
            Source of randomness for masking and scheduled sampling.
        ss_frac : float, default=0.0
            Fraction of context frames replaced by the model's own samples.
        present : torch.Tensor of shape (B, n_modalities), default=None
            1.0 where the shot recorded the diagnostic. Absent diagnostics stay in
            the input but are not scored.
        per_modality : dict, default=None
            If given, filled with each modality's loss.
        mask_ratio : float, default=None
            Fixed mask ratio, for diagnosis.
        history_frames : int, default=0
            Leading frames kept visible and unscored.
        mod_weights : dict of str to float, default=None
            Weight of each modality in the average; missing names weigh 1.
        gen_mask_p : float, default=None
            Generation-mode fraction; ``None`` uses the configuration.
        live : torch.Tensor of shape (B, F), default=None
            1.0 for frames before the end of the discharge. Multiplies ``present``.
        text : torch.Tensor of shape (B, text_embed_dim), default=None
            Per-shot text embedding, required when ``cfg.text_embed_dim > 0``.
        label_smoothing : float, default=0.0
            Label smoothing of the cross-entropy. Passed explicitly so that
            validation, which calls this method too, is never smoothed.
        block_forcing : bool, default=False
            Block teacher forcing: every frame is predicted from a fully visible
            history of strictly earlier frames (:meth:`DynamicsBackbone.
            encode_block_forcing`), the condition of a rollout. Scheduled sampling
            then replaces history frames by the model's own one-step forecasts
            (:meth:`_forecast_context`) instead of samples that saw the answer.
        seed_drop_p : float, default=0.0
            Augmentation: probability of hiding a whole modality from the input in
            the seed frames (:meth:`_seed_dropout`). Hidden positions are always
            scored, so the model learns to predict a modality missing from the seed.
        seed_column_drop_p : float, default=0.0
            Augmentation: probability of hiding a random fraction of a modality's
            token positions in the seed frames, scored likewise.

        Returns
        -------
        loss : torch.Tensor of shape ()
            Weighted mean of the per-modality losses.

        Notes
        -----
        Without block forcing, ``cfg.sf_frames > 0`` replaces a window of the
        context by the model's own rollout (:meth:`_self_forcing_context`), and a
        ``cfg.ctf_frac`` fraction of the batches is masked like a rollout
        (:meth:`_boundary_mask`). Both are training curricula: a pinned
        ``mask_ratio`` skips them without drawing, so a validation call with a
        fixed mask scores exactly that mask.

        Each modality's weight is ``cfg.modality_loss_weight`` times its
        ``mod_weights`` entry. A modality with zero total weight contributes
        ``0 * logits.sum()`` so its head stays in the autograd graph, as DDP
        requires.
        """
        pinned = mask_ratio is not None
        curricula = not pinned and (self.cfg.sf_frames > 0 or self.cfg.ctf_frac > 0.0)
        if block_forcing and curricula:
            raise ValueError(
                "block_forcing already trains the rollout condition; it does not "
                "combine with ctf_frac or sf_frames"
            )
        dropped = self._seed_dropout(
            codes, seed_drop_p, seed_column_drop_p, generator
        )
        inputs = codes if dropped is None else self._hide(codes, dropped)
        if block_forcing:
            history = self._forecast_context(
                inputs, actuators, ss_frac, generator, text=text
            )
            # The query stream reveals real codes, as MaskGIT decoding of the
            # target frame does; the history carries the model's forecasts.
            query, mask = self._random_mask(
                codes,
                generator,
                ratio_override=mask_ratio,
                history_frames=history_frames,
                gen_mask_p=gen_mask_p,
            )
            if dropped is not None:
                history, query = self._hide(history, dropped), self._hide(query, dropped)
            hidden = self.backbone.encode_block_forcing(
                history, query, actuators, **_text_kwargs(text)
            )
        else:
            context = self._scheduled_sample_context(
                inputs, actuators, ss_frac, generator, text=text
            )
            self_forced_end = None
            if not pinned:
                context, self_forced_end = self._self_forcing_context(
                    context, actuators, generator, text
                )
            use_ctf = False
            if not pinned and self.cfg.ctf_frac > 0.0:
                device = codes[self.cfg.modalities[0].name].device
                use_ctf = bool(
                    torch.rand((), generator=generator, device=device).item()
                    < self.cfg.ctf_frac
                )
            if use_ctf:
                masked, mask = self._boundary_mask(
                    context,
                    generator,
                    min_boundary=1 if self_forced_end is None else self_forced_end,
                )
            else:
                masked, mask = self._random_mask(
                    context,
                    generator,
                    ratio_override=mask_ratio,
                    history_frames=history_frames,
                    gen_mask_p=gen_mask_p,
                )
            if dropped is not None:
                # Scheduled sampling may have refilled hidden positions.
                masked = self._hide(masked, dropped)
            hidden = self.backbone.encode(masked, actuators, **_text_kwargs(text))
        if dropped is not None:
            mask = {name: mask[name] | dropped[name] for name in mask}
        masked_logits = self.backbone.tok.masked_logits(hidden, mask)
        total = codes[self.cfg.modalities[0].name].new_zeros((), dtype=torch.float32)
        total_weight = 0.0
        for index, m in enumerate(self.cfg.modalities):
            modality_mask = mask[m.name]
            if not bool(modality_mask.any()):
                continue
            logits = masked_logits[m.name]
            target = codes[m.name][modality_mask]
            token_weight = self._token_weights(modality_mask, index, present, live)
            if token_weight is None:
                term = self.output.loss(logits, target, label_smoothing, "mean")
            else:
                token_weight = token_weight.to(logits.dtype)
                weight_sum = token_weight.sum()
                if not bool(weight_sum > 0):
                    total = total + 0.0 * logits.sum()
                    continue
                losses = self.output.loss(logits, target, label_smoothing, "none")
                term = (losses * token_weight).sum() / weight_sum
            weight = self._modality_weight(m) * (
                1.0 if mod_weights is None else float(mod_weights.get(m.name, 1.0))
            )
            total = total + weight * term
            total_weight += weight
            if per_modality is not None:
                per_modality[m.name] = float(term.detach())
        return total / max(total_weight, 1e-8)

    @staticmethod
    def _token_weights(modality_mask, index, present, live):
        """Loss weight of each masked token, or None if every token weighs 1.

        The per-sample presence and per-frame liveness are broadcast over the
        tokens and gathered with the same mask as the logits.
        """
        if present is None and live is None:
            return None
        batch = modality_mask.shape[0]
        weights = None
        if present is not None:
            weights = present[:, index].view(batch, *([1] * (modality_mask.dim() - 1)))
            weights = weights.expand_as(modality_mask)
        if live is not None:
            live_weights = live.view(
                batch, live.shape[1], *([1] * (modality_mask.dim() - 2))
            ).expand_as(modality_mask)
            weights = live_weights if weights is None else weights * live_weights
        return weights[modality_mask]

    def _decode_logits(self, sequence, actuators, sampler, text=None):
        """Last-frame head outputs, with optional classifier-free guidance.

        ``cfg_scale == 1`` is one forward pass, so guidance costs nothing when off.
        The guidance baseline drops both the actuators and the text.
        """
        hidden = self.backbone.encode(sequence, actuators, **_text_kwargs(text))
        conditional = self.backbone.tok.logits_last(hidden)
        if sampler.cfg_scale == 1.0:
            return conditional
        hidden = self.backbone.encode(
            sequence, actuators, drop_actuators=True, text=text, drop_text=True
        )
        unconditional = self.backbone.tok.logits_last(hidden)
        scale = float(sampler.cfg_scale)
        return {
            name: unconditional[name] + scale * (conditional[name] - unconditional[name])
            for name in conditional
        }

    def _check_sampler(self, sampler: SamplerConfig) -> None:
        if self.backbone.tok.latent_head and (
            sampler.top_p is not None
            or sampler.global_pool
            or sampler.revision_rounds
            or sampler.cfg_scale != 1.0
        ):
            raise ValueError(
                "the latent head samples with temperature and gumbel only; top_p, "
                "global_pool, revision_rounds and cfg_scale need a categorical head"
            )

    @torch.no_grad()
    def generate_frame(
        self,
        past_codes: Dict[str, torch.Tensor],
        actuators: torch.Tensor,
        temperature: float = 1.0,
        generator: Optional[torch.Generator] = None,
        sampler: Optional[SamplerConfig] = None,
        text: Optional[torch.Tensor] = None,
        cache: Optional[KVCache] = None,
        hold: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Generate the codes of the next frame by iterative unmasking.

        Parameters
        ----------
        past_codes : dict of str to torch.LongTensor of shape (B, P, n_tok)
            Committed codes of the past frames.
        actuators : torch.Tensor of shape (B, P + 1, actuator_dim)
            Actuators up to and including the generated frame.
        temperature : float, default=1.0
            Sampling temperature, used when ``sampler`` is None.
        generator : torch.Generator, default=None
            Source of randomness.
        sampler : SamplerConfig, default=None
            Decode policy; ``None`` is ``SamplerConfig(temperature=temperature)``,
            the original sampler.
        text : torch.Tensor of shape (B, text_embed_dim), default=None
            Per-shot text embedding.
        cache : KVCache, default=None
            From ``backbone.prefill`` over the same ``P`` frames: each pass then
            runs over the new frame only, and the committed frame is appended to
            the cache. Needs ``cfg_scale == 1``.
        hold : dict of str to torch.BoolTensor of shape (B,), default=None
            Rows kept at their last past frame's codes (see :func:`placeholders`).
            They are still sampled, so the random stream does not change.

        Returns
        -------
        codes : dict of str to torch.LongTensor of shape (B, n_tok)
            Committed codes of the new frame.

        Notes
        -----
        Each step samples every modality, then reveals the most confident
        still-masked tokens, per modality or pooled over the frame
        (``sampler.global_pool``). ``sampler.gumbel > 0`` adds annealed Gumbel
        noise to the confidence so the reveal order is not a greedy cascade onto
        the mode. ``sampler.revision_rounds`` then re-masks and re-decodes the
        least confident committed tokens.
        """
        cfg = self.cfg
        sampler = SamplerConfig(temperature=temperature) if sampler is None else sampler
        self._check_sampler(sampler)
        reference = past_codes[cfg.modalities[0].name]
        batch, n_past, _ = reference.shape
        device = reference.device
        if cache is None:

            def _next_frame_logits(frame):
                sequence = {
                    name: torch.cat([past_codes[name], frame[name].unsqueeze(1)], dim=1)
                    for name in frame
                }
                return self._decode_logits(sequence, actuators, sampler, text=text)

        else:
            if sampler.cfg_scale != 1.0:
                raise ValueError("the KV cache supports cfg_scale == 1 only")
            if cache.n != n_past:
                raise ValueError(
                    f"the cache holds {cache.n} frames; past_codes has {n_past}"
                )

            def _next_frame_logits(frame):
                hidden = self.backbone.step(cache, frame, actuators[:, -1], text=text)
                return self.backbone.tok.logits_last(hidden.unsqueeze(1))

        frame = {
            m.name: torch.full(
                (batch, m.n_tok), self._mask_id(m.name), dtype=torch.long, device=device
            )
            for m in cfg.modalities
        }
        revealed = {
            m.name: torch.zeros((batch, m.n_tok), dtype=torch.bool, device=device)
            for m in cfg.modalities
        }
        fixed = {}
        for name, rows in (hold or {}).items():
            fixed[name] = rows.view(batch, 1).expand_as(revealed[name])
            frame[name] = torch.where(fixed[name], past_codes[name][:, -1], frame[name])
            revealed[name] = fixed[name].clone()
        keep_masked = _cosine_keep_fractions(cfg.maskgit_decode_steps)
        for step, masked_fraction in enumerate(keep_masked):
            logits = _next_frame_logits(frame)
            # Sample every modality first, so the reveal policy sees all of the
            # frame's confidences; the sampling order is unchanged.
            samples, confidence = {}, {}
            for m in cfg.modalities:
                samples[m.name], score = self.output.sample(
                    logits[m.name], sampler.temp_for(m.name), generator, sampler.top_p
                )
                if sampler.gumbel > 0.0:
                    uniform = torch.rand(score.shape, generator=generator, device=device)
                    uniform = uniform.clamp_(1e-9, 1 - 1e-9)
                    anneal = sampler.gumbel * (1.0 - step / max(1, len(keep_masked)))
                    gumbel = -torch.log(-torch.log(uniform))
                    score = self.output.from_reveal_score(
                        self.output.reveal_score(score) + anneal * gumbel
                    )
                confidence[m.name] = score
            if sampler.global_pool:
                take = self._global_reveal(confidence, revealed, masked_fraction)
            else:
                take = self._per_modality_reveal(confidence, revealed, masked_fraction)
            for m in cfg.modalities:
                frame[m.name] = torch.where(take[m.name], samples[m.name], frame[m.name])
                revealed[m.name] = revealed[m.name] | take[m.name]
        # Revision: the schedule never revisits a committed token, so re-mask the
        # least confident fraction and re-decode it against the tokens that stayed.
        for _ in range(int(sampler.revision_rounds)):
            logits = _next_frame_logits(frame)
            samples, confidence = {}, {}
            for m in cfg.modalities:
                samples[m.name], confidence[m.name] = self.output.resample(
                    logits[m.name],
                    sampler.temp_for(m.name),
                    generator,
                    sampler.top_p,
                    frame[m.name],
                )
            for m in cfg.modalities:
                n_redo = int(round(sampler.revision_frac * m.n_tok))
                if n_redo <= 0:
                    continue
                weakest = confidence[m.name].argsort(dim=-1)[:, :n_redo]
                redo = torch.zeros_like(revealed[m.name])
                redo.scatter_(1, weakest, True)
                if m.name in fixed:
                    redo &= ~fixed[m.name]
                frame[m.name] = torch.where(redo, samples[m.name], frame[m.name])
        # Fill anything still masked (rounding edge cases) with the mode.
        for m in cfg.modalities:
            still_masked = ~revealed[m.name]
            if bool(still_masked.any()):
                fill = self.output.mode(_next_frame_logits(frame)[m.name], generator)
                frame[m.name] = torch.where(still_masked, fill, frame[m.name])
        if cache is not None:
            # Commit the final frame's keys and values.
            self.backbone.step(cache, frame, actuators[:, -1], text=text)
            cache.n += 1
        return frame

    def _per_modality_reveal(self, confidence, revealed, masked_fraction):
        """Reveal the same fraction of every modality's own tokens."""
        take = {}
        for m in self.cfg.modalities:
            score = confidence[m.name].masked_fill(revealed[m.name], float("inf"))
            n_reveal = m.n_tok - int(round(masked_fraction * m.n_tok))
            order = score.argsort(dim=-1, descending=True)
            newly_revealed = torch.zeros_like(revealed[m.name])
            newly_revealed.scatter_(1, order[:, :n_reveal], True)
            take[m.name] = newly_revealed & ~revealed[m.name]
        return take

    def _global_reveal(self, confidence, revealed, masked_fraction):
        """Reveal the top tokens of the whole frame on one comparable scale.

        An uncertain modality can defer while confident ones commit first and anchor
        it through the next step's spatial attention. The score is
        :func:`.sampling.norm_log_confidence` (``1 + log c / log V``), not a
        within-modality rank: ranks would make the allocation proportional to the
        token count for any confidences, i.e. the per-modality quota. Under a
        uniform vocabulary ``log V`` is a common divisor and the pool is a plain
        probability top-K over the frame.
        """
        parts = [
            norm_log_confidence(confidence[m.name], m.codebook_size).masked_fill(
                revealed[m.name], float("inf")
            )
            for m in self.cfg.modalities
        ]
        flat = torch.cat(parts, dim=1)
        total = flat.shape[1]
        n_reveal = total - int(round(masked_fraction * total))
        # Stable: exact ties must not resolve by sort-backend luck, or the pooled
        # order is not reproducible across devices.
        order = flat.argsort(dim=-1, descending=True, stable=True)
        selected = torch.zeros_like(flat, dtype=torch.bool)
        selected.scatter_(1, order[:, :n_reveal], True)
        take, offset = {}, 0
        for m in self.cfg.modalities:
            take[m.name] = selected[:, offset : offset + m.n_tok] & ~revealed[m.name]
            offset += m.n_tok
        return take

    @torch.no_grad()
    def rollout(
        self,
        seed_codes: Dict[str, torch.Tensor],
        actuators: torch.Tensor,
        n_predict: Optional[int] = None,
        temperature: float = 1.0,
        generator: Optional[torch.Generator] = None,
        sampler: Optional[SamplerConfig] = None,
        text: Optional[torch.Tensor] = None,
        kv_cache: bool = False,
        hold: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Roll out ``n_predict`` frames autoregressively in code space.

        Parameters
        ----------
        seed_codes : dict of str to torch.LongTensor of shape (B, K0, n_tok)
            Real seed frames.
        actuators : torch.Tensor of shape (B, >= K0 + n_predict, actuator_dim)
            Actuators of the whole trajectory.
        n_predict : int, default=None
            Frames to generate; ``None`` uses ``cfg.n_predict``.
        temperature : float, default=1.0
            Sampling temperature, used when ``sampler`` is None.
        generator : torch.Generator, default=None
            Source of randomness.
        sampler : SamplerConfig, default=None
            Decode policy (see :meth:`generate_frame`).
        text : torch.Tensor of shape (B, text_embed_dim), default=None
            Per-shot text embedding.
        kv_cache : bool, default=False
            Prefill the seed once and decode each frame against cached keys and
            values: the same trajectory at a fraction of the cost (see
            :class:`.dynamics.KVCache`).
        hold : dict of str to torch.BoolTensor of shape (B,), default=None
            :func:`placeholders` output, or any such mask.

        Returns
        -------
        trajectory : dict of str to torch.LongTensor of shape (B, K0 + n_predict, n_tok)
            Seed followed by the committed predictions.
        """
        cfg = self.cfg
        n_predict = cfg.n_predict if n_predict is None else n_predict
        trajectory = {name: codes.clone() for name, codes in seed_codes.items()}
        n_seed = trajectory[cfg.modalities[0].name].shape[1]
        if actuators.shape[1] < n_seed + n_predict:
            raise ValueError(
                f"actuators has {actuators.shape[1]} frames; "
                f"need K0+n_predict={n_seed + n_predict}"
            )
        cache = (
            self.backbone.prefill(
                trajectory, actuators[:, :n_seed], n_seed + n_predict, text=text
            )
            if kv_cache
            else None
        )
        for t in range(n_predict):
            next_frame = self.generate_frame(
                trajectory,
                actuators[:, : n_seed + t + 1],
                temperature=temperature,
                generator=generator,
                sampler=sampler,
                text=text,
                cache=cache,
                hold=hold,
            )
            trajectory = {
                name: torch.cat([codes, next_frame[name].unsqueeze(1)], dim=1)
                for name, codes in trajectory.items()
            }
        return trajectory
