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
import os
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dynamics import DynamicsBackbone
from .dynamics_config import DynamicsConfig
from .frame_layout import code_to_coords, coords_to_code

# Scale of the annealed Gumbel noise on the reveal order (0 = greedy reveal). Read
# once at import from ``IGNITE_MASKGIT_GUMBEL``.
_GUMBEL_SCALE = float(os.environ.get("IGNITE_MASKGIT_GUMBEL", "0"))

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


def _straight_through_sample(logits, temperature: float, generator):
    """One-hot Gumbel-max samples with the gradient of the Gumbel softmax.

    Parameters
    ----------
    logits : torch.Tensor of shape (..., codebook)
        Unnormalised log-probabilities.
    temperature : float
        Temperature of both the sample and the softmax; must be positive.
    generator : torch.Generator or None
        Source of randomness.

    Returns
    -------
    weights : torch.Tensor of shape (..., codebook)
        Exactly one-hot in value; differentiable through the softmax.
    """
    if temperature <= 0.0:
        raise ValueError("straight-through sampling needs a positive temperature")
    uniform = torch.rand(
        logits.shape, generator=generator, device=logits.device
    ).clamp_(1e-9, 1.0 - 1e-9)
    perturbed = (logits.float() - torch.log(-torch.log(uniform))) / temperature
    soft = perturbed.softmax(dim=-1)
    hard = F.one_hot(perturbed.argmax(dim=-1), logits.shape[-1]).to(soft.dtype)
    return hard - soft.detach() + soft


class _CategoricalOutput:
    """Codebook classifier head: logits over the modality's codes."""

    def loss(self, logits, target, label_smoothing, reduction):
        """Cross-entropy per token (``reduction="none"``) or its mean."""
        return F.cross_entropy(
            logits, target, reduction=reduction, label_smoothing=label_smoothing
        )

    def sample(self, logits, temperature, generator):
        """Sample (B, n_tok) codes; return them with their probability."""
        # Sample in fp32: the backbone may run under bf16 autocast.
        probs = (logits.float() / max(temperature, 1e-6)).softmax(-1)
        codes = torch.multinomial(
            probs.reshape(-1, probs.shape[-1]), 1, generator=generator
        ).reshape(probs.shape[:-1])
        confidence = probs.gather(-1, codes.unsqueeze(-1)).squeeze(-1)
        return codes, confidence

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


class _GaussianOutput:
    """Latent head: a Gaussian over the token's FSQ grid coordinate."""

    def __init__(self, levels):
        self.levels = levels

    def loss(self, out, target, label_smoothing, reduction):
        """Gaussian NLL per token (``reduction="none"``) or its mean."""
        nll = _latent_nll(out, target, self.levels)
        return nll.mean() if reduction == "mean" else nll

    def sample(self, out, temperature, generator):
        """Sample codes; return them with the negative summed log-variance."""
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

    def ss_fraction(self, step: int) -> float:
        """Return the scheduled-sampling fraction at ``step`` (linear ramp)."""
        cfg = self.cfg
        return min(1.0, step / max(1, cfg.ss_ramp_steps)) * cfg.ss_ramp_final_frac

    @torch.no_grad()
    def _scheduled_sample_context(self, codes, actuators, ss_frac, gen):
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
            hidden = self.backbone.encode(codes, actuators)
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
    def _forecast_context(self, codes, actuators, ss_frac, gen, depth: int = 1):
        """Replace a fraction of history frames by the model's own forecasts.

        Every frame is forecast in one block-forcing pass from the frames before
        it, with nothing of the frame itself visible, and sampled at temperature 1.
        Training on such a history scores forecasts made from the model's own
        earlier forecasts. ``cache_enabled=False`` as in
        :meth:`_scheduled_sample_context`.

        ``depth > 1`` repeats the pass on its own output for the same frames, so a
        replaced frame is forecast from earlier frames that are themselves
        forecasts, up to ``depth`` levels deep: the history a ``depth``-step rollout
        produces (the pushforward trick). Gradients flow through the final
        prediction only, as for ``depth == 1``.
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
        history = codes
        for _ in range(max(1, int(depth))):
            with torch.autocast(
                device_type=reference.device.type,
                dtype=torch.bfloat16,
                enabled=torch.is_autocast_enabled(),
                cache_enabled=False,
            ):
                hidden = self.backbone.encode_block_forcing(history, blank, actuators)
                selected_logits = self.backbone.tok.masked_logits(hidden, selection)
            forecast = {}
            for m in self.cfg.modalities:
                frames = codes[m.name].clone()
                frames[selection[m.name]] = self.output.sample_flat(
                    selected_logits[m.name], gen
                ).to(frames.dtype)
                forecast[m.name] = frames
            history = forecast
        return history

    def _history_gap(self, history, gap_p: float, gap_max: int, gen):
        """Hide a stretch of history frames, per sample with probability ``gap_p``.

        The gap starts at a random frame after the first and spans 1 to ``gap_max``
        frames; every modality is set to ``[MASK]`` there. Frames in and just after
        the gap are then predicted without their recent state, so the earlier state
        and the actuators over the gap must carry the forecast instead of
        persistence. Only the history changes; the query stream and the loss
        targets keep the real codes.
        """
        if gap_p <= 0.0 or gap_max < 1:
            return history
        reference = history[self.cfg.modalities[0].name]
        batch, n_frames, _ = reference.shape
        device = reference.device
        on = torch.rand((batch,), generator=gen, device=device) < gap_p
        if n_frames < 2 or not bool(on.any()):
            return history
        start = 1 + (
            torch.rand((batch,), generator=gen, device=device) * (n_frames - 1)
        ).long()
        length = 1 + (torch.rand((batch,), generator=gen, device=device) * gap_max).long()
        frames = torch.arange(n_frames, device=device).view(1, -1)
        gap = on.view(-1, 1) & (frames >= start.view(-1, 1)) & (
            frames < (start + length).view(-1, 1)
        )
        return {
            name: torch.where(
                gap.unsqueeze(-1), torch.full_like(frames_, self._mask_id(name)), frames_
            )
            for name, frames_ in history.items()
        }

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
        label_smoothing: float = 0.0,
        block_forcing: bool = False,
        seed_drop_p: float = 0.0,
        seed_column_drop_p: float = 0.0,
        act_drop_p: float = 0.0,
        forecast_depth: int = 1,
        hist_gap_p: float = 0.0,
        hist_gap_max: int = 10,
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
        act_drop_p : float, default=0.0
            Probability of training a sample without any actuator term, the
            unconditional branch that classifier-free guidance
            (:meth:`generate_frame` ``guidance``) needs. The own-forecast history
            stays conditioned.
        forecast_depth : int, default=1
            With ``block_forcing``: levels of own forecasts in the history
            (:meth:`_forecast_context`); ``1`` is one-step scheduled sampling.
        hist_gap_p : float, default=0.0
            With ``block_forcing``: probability per sample of hiding a stretch of up
            to ``hist_gap_max`` history frames (:meth:`_history_gap`), so the
            forecast across the gap has to come from the actuators.
        hist_gap_max : int, default=10
            Longest hidden stretch, in frames.

        Returns
        -------
        loss : torch.Tensor of shape ()
            Weighted mean of the per-modality losses.

        Notes
        -----
        A modality with zero total weight contributes ``0 * logits.sum()`` so its
        head stays in the autograd graph, as DDP requires.
        """
        dropped = self._seed_dropout(
            codes, seed_drop_p, seed_column_drop_p, generator
        )
        inputs = codes if dropped is None else self._hide(codes, dropped)
        act_null = None
        if act_drop_p > 0.0:
            act_null = (
                torch.rand(
                    (actuators.shape[0],), generator=generator, device=actuators.device
                )
                < act_drop_p
            )
        if block_forcing:
            history = self._forecast_context(
                inputs, actuators, ss_frac, generator, depth=forecast_depth
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
                history = self._hide(history, dropped)
                query = self._hide(query, dropped)
            history = self._history_gap(history, hist_gap_p, hist_gap_max, generator)
            hidden = self.backbone.encode_block_forcing(
                history, query, actuators, act_null
            )
        else:
            context = self._scheduled_sample_context(
                inputs, actuators, ss_frac, generator
            )
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
            hidden = self.backbone.encode(masked, actuators, act_null)
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
            weight = 1.0 if mod_weights is None else float(mod_weights.get(m.name, 1.0))
            total = total + weight * term
            total_weight += weight
            if per_modality is not None:
                per_modality[m.name] = float(term.detach())
        return total / max(total_weight, 1e-8)

    def rollout_loss(
        self,
        codes: Dict[str, torch.Tensor],
        actuators: torch.Tensor,
        horizon: int,
        start: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        temperature: float = 1.0,
        present: Optional[torch.Tensor] = None,
        live: Optional[torch.Tensor] = None,
        mod_weights: Optional[Dict[str, float]] = None,
        per_modality: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        """Cross-entropy of an autoregressive rollout, differentiable through the chain.

        The frames before ``start`` are real. Each of the next ``horizon`` frames is
        predicted in one pass from all-``[MASK]`` input, conditioned on the real
        frames and on the frames already generated, as a rollout is. Each
        prediction is scored against the real frame. The generated frame is a
        straight-through Gumbel sample: the forward pass feeds back the embedding
        of a hard code, sampled at ``temperature``, and the backward pass uses the
        gradient of the softmax, so later errors train earlier predictions.

        Parameters
        ----------
        codes : dict of str to torch.LongTensor of shape (B, F, n_tok)
            Real codes, the seed and the targets.
        actuators : torch.Tensor of shape (B, F, actuator_dim)
            Actuators of each frame.
        horizon : int
            Frames generated and scored.
        start : int, default=None
            First generated frame; ``None`` uses ``cfg.k0_seed``.
        generator : torch.Generator, default=None
            Source of the Gumbel noise.
        temperature : float, default=1.0
            Temperature of the samples and of their softmax.
        present, live, mod_weights, per_modality
            As in :meth:`training_loss`; ``live`` covers the whole window.

        Returns
        -------
        loss : torch.Tensor of shape ()
            Weighted mean over modalities of the per-modality mean cross-entropy.

        Notes
        -----
        Every step re-encodes the whole prefix, so a step costs a forward pass over
        ``start + j + 1`` frames; there is no key-value cache to differentiate
        through.
        """
        if self.backbone.tok.latent_head:
            raise ValueError("rollout_loss supports the categorical head only")
        cfg, tok = self.cfg, self.backbone.tok
        reference = codes[cfg.modalities[0].name]
        batch, n_frames, _ = reference.shape
        start = int(cfg.k0_seed if start is None else start)
        if not (0 < start and start + int(horizon) <= n_frames):
            raise ValueError(
                f"rollout start {start} + horizon {horizon} outside {n_frames} frames"
            )
        device = reference.device
        history = {
            m.name: tok.code_vectors(m.name, codes[m.name][:, :start])
            for m in cfg.modalities
        }
        query = {
            m.name: tok.code_vectors(
                m.name,
                torch.full(
                    (batch, 1, m.n_tok), self._mask_id(m.name), device=device
                ),
            )
            for m in cfg.modalities
        }
        step_logits = {m.name: [] for m in cfg.modalities}
        for offset in range(int(horizon)):
            frame = start + offset
            inputs = {
                name: torch.cat([history[name], query[name]], dim=1) for name in history
            }
            hidden = self.backbone.encode(inputs, actuators[:, : frame + 1])
            logits = tok.logits(hidden[:, -1:])
            for m in cfg.modalities:
                frame_logits = logits[m.name]
                step_logits[m.name].append(frame_logits)
                if offset + 1 < horizon:
                    vectors = tok.soft_code_vectors(
                        m.name,
                        _straight_through_sample(frame_logits, temperature, generator),
                    )
                    history[m.name] = torch.cat([history[m.name], vectors], dim=1)
        stop = start + int(horizon)
        window_live = None if live is None else live[:, start:stop]
        total = reference.new_zeros((), dtype=torch.float32)
        total_weight = 0.0
        for index, m in enumerate(cfg.modalities):
            logits = torch.cat(step_logits[m.name], dim=1)
            target = codes[m.name][:, start:stop]
            every = torch.ones_like(target, dtype=torch.bool)
            losses = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                target.reshape(-1),
                reduction="none",
            )
            token_weight = self._token_weights(every, index, present, window_live)
            if token_weight is None:
                term = losses.mean()
            else:
                token_weight = token_weight.to(losses.dtype)
                weight_sum = token_weight.sum()
                if not bool(weight_sum > 0):
                    total = total + 0.0 * logits.sum()
                    continue
                term = (losses * token_weight).sum() / weight_sum
            weight = 1.0 if mod_weights is None else float(mod_weights.get(m.name, 1.0))
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

    @torch.no_grad()
    def generate_frame(
        self,
        past_codes: Dict[str, torch.Tensor],
        actuators: torch.Tensor,
        temperature: float = 1.0,
        generator: Optional[torch.Generator] = None,
        guidance: float = 1.0,
        gate: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """Generate the codes of the next frame by iterative unmasking.

        Parameters
        ----------
        past_codes : dict of str to torch.LongTensor of shape (B, P, n_tok)
            Committed codes of the past frames.
        actuators : torch.Tensor of shape (B, P + 1, actuator_dim)
            Actuators up to and including the generated frame.
        temperature : float, default=1.0
            Sampling temperature.
        generator : torch.Generator, default=None
            Source of randomness.
        guidance : float, default=1.0
            Classifier-free guidance weight ``w``: the logits are
            ``l_uncond + w * (l_cond - l_uncond)``, with the unconditional branch
            run without any actuator term. ``1.0`` is plain conditioning and costs
            one forward pass per step; any other value costs two. Needs a model
            trained with ``act_drop_p > 0``, else the unconditional branch is
            untrained.
        gate : float, default=0.0
            Confidence gate: a token whose sampled code had probability below
            ``gate`` when it was revealed keeps the code of the previous frame,
            so the frame changes only where the model is sure. ``0.0`` is off.

        Returns
        -------
        codes : dict of str to torch.LongTensor of shape (B, n_tok)
            Committed codes of the new frame.

        Notes
        -----
        Each step reveals the most confident still-masked tokens. With
        ``IGNITE_MASKGIT_GUMBEL`` set, annealed Gumbel noise is added to the
        confidence so the reveal order is not a greedy cascade onto the mode.
        """
        cfg = self.cfg
        reference = past_codes[cfg.modalities[0].name]
        batch = reference.shape[0]
        device = reference.device
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

        guided = float(guidance) != 1.0
        if guided and not isinstance(self.output, _CategoricalOutput):
            raise ValueError("classifier-free guidance needs the categorical head")
        if gate > 0.0 and not isinstance(self.output, _CategoricalOutput):
            raise ValueError("the confidence gate needs the categorical head")
        # Probability of each committed code when it was revealed (for the gate).
        committed_p = {
            m.name: torch.ones((batch, m.n_tok), device=device) for m in cfg.modalities
        }
        if guided:
            # One doubled batch: the first half conditioned, the second without
            # any actuator term.
            both_actuators = torch.cat([actuators, actuators], dim=0)
            act_null = torch.arange(2 * batch, device=device) >= batch

        def _next_frame_logits():
            sequence = {
                name: torch.cat([past_codes[name], frame[name].unsqueeze(1)], dim=1)
                for name in frame
            }
            if not guided:
                return self.backbone(sequence, actuators)
            doubled = {name: torch.cat([c, c], dim=0) for name, c in sequence.items()}
            logits = self.backbone(doubled, both_actuators, act_null)
            out = {}
            for name, both in logits.items():
                # Only the generated (last) frame is read; keep its axis.
                cond, uncond = both[:batch, -1:].float(), both[batch:, -1:].float()
                out[name] = uncond + float(guidance) * (cond - uncond)
            return out

        keep_masked = _cosine_keep_fractions(cfg.maskgit_decode_steps)
        for step, masked_fraction in enumerate(keep_masked):
            logits = _next_frame_logits()
            for m in cfg.modalities:
                samples, confidence = self.output.sample(
                    logits[m.name][:, -1], temperature, generator
                )
                probability = confidence.float()
                if _GUMBEL_SCALE > 0.0:
                    uniform = torch.rand(
                        confidence.shape, generator=generator, device=device
                    )
                    uniform = uniform.clamp_(1e-9, 1 - 1e-9)
                    anneal = _GUMBEL_SCALE * (1.0 - step / max(1, len(keep_masked)))
                    gumbel = -torch.log(-torch.log(uniform))
                    confidence = self.output.reveal_score(confidence) + anneal * gumbel
                confidence = confidence.masked_fill(revealed[m.name], float("inf"))
                n_reveal = m.n_tok - int(round(masked_fraction * m.n_tok))
                reveal_index = confidence.argsort(dim=-1, descending=True)[:, :n_reveal]
                newly_revealed = torch.zeros_like(revealed[m.name])
                newly_revealed.scatter_(1, reveal_index, True)
                take = newly_revealed & ~revealed[m.name]
                frame[m.name] = torch.where(take, samples, frame[m.name])
                committed_p[m.name] = torch.where(
                    take, probability, committed_p[m.name]
                )
                revealed[m.name] = revealed[m.name] | newly_revealed
        # Fill anything still masked (rounding edge cases) with the mode.
        for m in cfg.modalities:
            still_masked = ~revealed[m.name]
            if bool(still_masked.any()):
                logits = _next_frame_logits()[m.name][:, -1]
                fill = self.output.mode(logits, generator)
                frame[m.name] = torch.where(still_masked, fill, frame[m.name])
        if gate > 0.0:
            for m in cfg.modalities:
                unsure = committed_p[m.name] < gate
                frame[m.name] = torch.where(
                    unsure, past_codes[m.name][:, -1], frame[m.name]
                )
        return frame

    def _consensus(
        self, candidates, previous, n_candidates: int, quorum: float, mode: str = "majority"
    ):
        """Commit one code per token from ``n_candidates`` candidate frames.

        ``candidates`` is (B * n_candidates, n_tok), candidates of a sample
        adjacent; ``previous`` (B, n_tok) is the last committed frame. ``majority``
        commits the near-1 majority where at least ``quorum`` of the candidates
        agree, else ``previous``. ``mean`` commits the code nearest to the mean FSQ
        coordinates of the candidates, so activity drawn at random times averages
        out while changes common to all candidates remain.
        """
        levels = list(self.cfg.fsq_levels)
        batch = previous.shape[0]
        codes = candidates.view(batch, n_candidates, -1)
        if mode == "mean":
            return coords_to_code(code_to_coords(codes, levels).mean(dim=1), levels)
        if mode != "majority":
            raise ValueError(f"consensus mode {mode!r}: expected 'majority' or 'mean'")
        scale = torch.tensor([level - 1 for level in levels], device=codes.device)
        digits = torch.round(code_to_coords(codes, levels) * scale)  # (B, N, n, D)
        near = (digits.unsqueeze(2) - digits.unsqueeze(1)).abs().le(1).all(-1)
        support = near.sum(dim=2)  # (B, N, n): candidates within one cell, self too
        best_support, best = support.max(dim=1)  # (B, n)
        majority = codes.gather(1, best.unsqueeze(1)).squeeze(1)
        agreed = best_support.float() >= quorum * n_candidates
        return torch.where(agreed, majority, previous)

    @torch.no_grad()
    def rollout(
        self,
        seed_codes: Dict[str, torch.Tensor],
        actuators: torch.Tensor,
        n_predict: Optional[int] = None,
        temperature: float = 1.0,
        generator: Optional[torch.Generator] = None,
        guidance: float = 1.0,
        gate: float = 0.0,
        consensus_n: int = 1,
        consensus_q: float = 0.5,
        consensus_mode: str = "majority",
        consensus_shared: bool = True,
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
            Sampling temperature.
        generator : torch.Generator, default=None
            Source of randomness.
        guidance : float, default=1.0
            Classifier-free guidance weight of every frame (:meth:`generate_frame`).
        gate : float, default=0.0
            Confidence gate of every frame (:meth:`generate_frame`).
        consensus_n : int, default=1
            Consensus rollout: draw ``consensus_n`` candidates of each frame from
            the same committed history and commit, per token, the candidate with
            the most others within one FSQ grid cell in every dimension, if at
            least ``consensus_q`` of all candidates agree with it; otherwise the
            token keeps the previous frame's code. ``1`` is the plain rollout.
        consensus_q : float, default=0.5
            Share of candidates that must agree.
        consensus_mode : {"majority", "mean"}, default="majority"
            How the candidates are combined (:meth:`_consensus`); ``mean`` ignores
            ``consensus_q``.
        consensus_shared : bool, default=True
            ``True``: every frame's candidates continue the one committed history,
            so a mode committed once is continued by all of them. ``False``: an
            ensemble of ``consensus_n`` independent rollouts, combined frame by frame
            only at the end, so a mode one member commits by chance does not carry
            into the others.

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
        n_candidates = max(1, int(consensus_n))
        if n_candidates > 1 and not consensus_shared:
            members = self.rollout(
                {name: c.repeat_interleave(n_candidates, dim=0) for name, c in seed_codes.items()},
                actuators.repeat_interleave(n_candidates, dim=0),
                n_predict=n_predict,
                temperature=temperature,
                generator=generator,
                guidance=guidance,
                gate=gate,
            )
            combined = {name: codes.clone() for name, codes in seed_codes.items()}
            for t in range(n_seed, n_seed + n_predict):
                frame = {
                    name: self._consensus(
                        members[name][:, t],
                        combined[name][:, -1],
                        n_candidates,
                        consensus_q,
                        consensus_mode,
                    )
                    for name in combined
                }
                combined = {
                    name: torch.cat([codes, frame[name].unsqueeze(1)], dim=1)
                    for name, codes in combined.items()
                }
            return combined
        for t in range(n_predict):
            history = trajectory
            frame_actuators = actuators[:, : n_seed + t + 1]
            if n_candidates > 1:
                history = {
                    name: codes.repeat_interleave(n_candidates, dim=0)
                    for name, codes in trajectory.items()
                }
                frame_actuators = frame_actuators.repeat_interleave(n_candidates, dim=0)
            next_frame = self.generate_frame(
                history,
                frame_actuators,
                temperature=temperature,
                generator=generator,
                guidance=guidance,
                gate=gate,
            )
            if n_candidates > 1:
                next_frame = {
                    name: self._consensus(
                        candidates,
                        trajectory[name][:, -1],
                        n_candidates,
                        consensus_q,
                        consensus_mode,
                    )
                    for name, candidates in next_frame.items()
                }
            trajectory = {
                name: torch.cat([codes, next_frame[name].unsqueeze(1)], dim=1)
                for name, codes in trajectory.items()
            }
        return trajectory
