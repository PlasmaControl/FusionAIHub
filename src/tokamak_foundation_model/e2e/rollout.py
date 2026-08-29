"""Token-space autoregressive rollout.

At each step ``k``, the diagnostic-token slice output by the backbone at
step ``k-1`` is fed directly as the diagnostic-token input at step ``k``
(no detokenize-then-retokenize). Actuator tokens are recomputed from fresh
per-step actuator commands. Output heads fire only so a loss can be computed
against raw ground truth — their output is never fed back (``ResearchPlan.MD``
§3.6, §5.9).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint as torch_ckpt

from .model import E2EFoundationModel
from .output_heads import SpectrogramFlowHead


@dataclass
class RolloutResult:
    """Everything the training loop or a §5.9 test needs from one rollout.

    Attributes
    ----------
    predictions
        Length ``K`` list; entry ``k`` is a ``{modality_name: raw_signal}``
        dict of head-decoded predictions for step ``k+1``.
    diagnostic_tokens
        Length ``K + 1`` list of ``(batch, n_diag_tokens, d_model)`` tensors.
        Index 0 is the tokenized initial state; index ``k + 1`` is the
        diagnostic slice of the backbone output after step ``k``.
    backbone_outputs
        Length ``K`` list of full ``(batch, n_total_tokens, d_model)``
        backbone outputs, covering diagnostic and actuator slots, one per
        step.
    """

    predictions: List[Dict[str, torch.Tensor]]
    diagnostic_tokens: List[torch.Tensor]
    backbone_outputs: List[torch.Tensor]
    # Length-``K`` list; entry ``k`` maps ``modality_name -> (batch, n_tokens,
    # d_model)`` backbone token slice fed to that modality's head at step
    # ``k`` — the conditioning a generative head (SpectrogramFlowHead) needs
    # to compute its per-step flow loss. Empty unless ``collect_token_slices``.
    diag_token_slices: List[Dict[str, torch.Tensor]] = field(default_factory=list)
    # Length-``n_steps`` list; entry ``k`` maps ``modality_name -> raw decoded
    # diagnostic state`` used as the INPUT to step ``k``. Index 0 is the GT
    # step-0 input (``initial_diag_inputs``); index ``k >= 1`` is the state the
    # feedback produced for step ``k`` (for FSQ/spectro modalities the
    # ``head.decode(codes)`` on-manifold state; for continuous heads the model's
    # own step ``k-1`` prediction). This is exactly the state the descriptor
    # anchor must read at step ``k`` so train == the Gate-4 inference wiring.
    # Empty unless ``collect_decoded_feedback``.
    decoded_feedback: List[Dict[str, torch.Tensor]] = field(default_factory=list)


class TokenSpaceRollout(nn.Module):
    """Autoregressive rollout wrapper around :class:`E2EFoundationModel`.

    Parameters
    ----------
    model
        The end-to-end foundation model providing tokenizers, backbone, and
        heads.
    dt_s
        Per-step time increment passed into the step-conditioning MLP.
        Defaults to 0.05 (50 ms, matching the Phase A window).
    """

    def __init__(self, model: E2EFoundationModel, dt_s: float = 0.05) -> None:
        super().__init__()
        self.model = model
        self.dt_s = dt_s
        self.n_diag_tokens = sum(
            layout.slice_.stop - layout.slice_.start
            for layout in model.token_layout
            if layout.is_diagnostic
        )

    def _diag_token_slice_offsets(self) -> Dict[str, Tuple[int, int]]:
        """Map each diagnostic modality name to its ``(start, stop)`` token
        offsets inside the concatenated ``(B, n_diag_tokens, d_model)`` stream
        (the same order ``_tokenize_diagnostics`` / ``_resample_feedback``
        iterate). Used to slice out one modality's tokens for the option-2
        per-sample renorm."""
        offsets: Dict[str, Tuple[int, int]] = {}
        off = 0
        for cfg in self.model.diagnostics:
            n = cfg.n_tokens()
            offsets[cfg.name] = (off, off + n)
            off += n
        return offsets

    @staticmethod
    def _renorm_feedback_tokens(
        name: str,
        tokens: torch.Tensor,
        ref_absmax: Optional[Dict[str, torch.Tensor]],
        *,
        eps: float = 1e-6,
        debug: bool = False,
    ) -> torch.Tensor:
        """OPTION-2 feedback-token renorm (POST-tokenizer, per-sample, uniform).

        ``tokens`` is one modality's ``(B, n_tok, d_model)`` feedback slice
        AFTER the backbone tokenizer (``tokenizer(decode(codes))``). Given the
        step-0 INPUT-window reference band ``ref_absmax[name]`` shape ``(B,)``
        (the model's tolerated per-sample token magnitude), scale each sample
        DOWN uniformly so its feedback token absmax never exceeds the input
        band: ``scale = clamp(ref / (fb + eps), max=1.0)``,
        ``tokens = tokens * scale[:, None, None]``.

        A single scalar per sample → the relative token structure (mode ridges
        vs floor) is preserved exactly → mode-safe by construction. Only scales
        DOWN (never amplifies), so in-band feedback is untouched (scale==1)."""
        if ref_absmax is None or name not in ref_absmax:
            return tokens
        ref = ref_absmax[name].to(tokens.dtype)                 # (B,)
        fb_absmax = tokens.abs().amax(dim=(1, 2))               # (B,)
        scale = torch.clamp(ref / (fb_absmax + eps), max=1.0)   # (B,)
        out = tokens * scale.view(-1, 1, 1)
        if debug:
            import os as _os
            if _os.environ.get("ROLLOUT_STATS_DEBUG") == "1":
                post = out.abs().amax(dim=(1, 2))
                print(
                    f"[opt2-renorm] {name}: ref_absmax={ref.tolist()} "
                    f"fb_absmax={fb_absmax.detach().tolist()} "
                    f"scale={scale.detach().tolist()} "
                    f"post_absmax={post.detach().tolist()}",
                    flush=True,
                )
        return out

    def _tokenize_diagnostics(
        self, diag_inputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        pieces: List[torch.Tensor] = []
        for cfg in self.model.diagnostics:
            x = diag_inputs[cfg.name]
            if cfg.kind == "video":
                # Video tokenizers honour a per-row camera-validity mask
                # (False rows are replaced with the learned missing_token).
                # Mirrors E2EFoundationModel.tokenize so missing-camera
                # samples don't get encoded as if a real camera frame
                # were present during step-0 init or TF re-tokenisation.
                valid = diag_inputs.get(f"{cfg.name}_valid")
                mask = valid.bool() if valid is not None else None
                pieces.append(
                    self.model.diag_tokenizers[cfg.name](x, mask=mask)
                )
            else:
                pieces.append(self.model.diag_tokenizers[cfg.name](x))
        return torch.cat(pieces, dim=1)

    def _tokenize_actuators(
        self, act_inputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        pieces: List[torch.Tensor] = []
        for cfg in self.model.actuators:
            pieces.append(self.model.act_tokenizers[cfg.name](act_inputs[cfg.name]))
        return torch.cat(pieces, dim=1)

    def _decode_diagnostics(
        self,
        diag_tokens: torch.Tensor,
        *,
        flow_noise: Optional[Dict[str, torch.Tensor]] = None,
        return_slices: bool = False,
    ):
        """Decode per-modality predictions from the diagnostic token slice.

        ``flow_noise`` (eval only): a per-modality noise tensor passed to a
        generative head's sampler so the SAME draw can be reused across all
        K rollout steps → temporally coherent block-mode frames (no per-step
        flicker). ``return_slices``: also return the per-modality token slice
        (the conditioning the per-step flow loss needs). Both default off →
        byte-identical to the original predictions-only path.
        """
        out: Dict[str, torch.Tensor] = {}
        slices: Dict[str, torch.Tensor] = {}
        offset = 0
        for cfg in self.model.diagnostics:
            n = cfg.n_tokens()
            sl = diag_tokens[:, offset : offset + n]
            head = self.model.diag_heads[cfg.name]
            if flow_noise is not None and isinstance(head, SpectrogramFlowHead):
                out[cfg.name] = head(sl, noise=flow_noise.get(cfg.name))
            else:
                out[cfg.name] = head(sl)
            if return_slices:
                slices[cfg.name] = sl
            offset += n
        if return_slices:
            return out, slices
        return out

    def _resample_feedback(
        self, diag_tokens: torch.Tensor, mode: str, temperature: float,
        *, return_decoded: bool = False, feedback_normalize: bool = False,
        ref_absmax: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """Code-space feedback: per diag modality, decode the predicted token slice to code logits,
        SAMPLE (mode='sample', at ``temperature``) or ARGMAX (mode='argmax') the codes, decode to the
        on-manifold state, and RE-TOKENIZE — returning the re-tokenized tokens to feed into the next
        step. Breaks the continuous-feedback contraction (fixed-point freeze). Heads without a code
        pathway (flow/descriptor) keep their continuous slice. Whole fed-back state is resampled, so
        the descriptor/anchor reads the same sampled tokens (no residual mean pathway).

        ``return_decoded`` (default ``False`` → byte-identical to the prior signature): also return a
        ``{modality_name: raw decoded state}`` dict — the on-manifold ``head.decode(codes)`` for
        code-path modalities. Continuous-head modalities get no entry here (the caller fills those
        from the model's own prediction). This is the raw fed-back state the descriptor anchor reads."""
        pieces: List[torch.Tensor] = []
        decoded_raw: Dict[str, torch.Tensor] = {}
        offset = 0
        for cfg in self.model.diagnostics:
            n = cfg.n_tokens()
            sl = diag_tokens[:, offset: offset + n]
            offset += n
            head = self.model.diag_heads[cfg.name]
            if all(hasattr(head, m) for m in ("code_logits", "sample_codes", "decode")):
                logits = head.code_logits(sl)
                codes = head.sample_codes(logits, temperature=temperature, hard=(mode == "argmax"))
                # Feedback tokens the CURRENT (production) way: decode codes to
                # the on-manifold pixel state, then re-tokenize through the
                # backbone tokenizer. The anchor reads this same decoded state.
                decoded = head.decode(codes)
                # VIDEO layout fix (mirror _tokenize_gt_onmanifold): decode returns
                # (B, T, C, H, W); the VideoTokenizer Conv3d expects (B, C, T, H, W).
                _re = decoded
                if getattr(cfg, "kind", None) == "video" and _re.dim() == 5:
                    _re = _re.permute(0, 2, 1, 3, 4).contiguous()
                tok = self.model.diag_tokenizers[cfg.name](_re)
                if feedback_normalize:
                    # OPTION-2 (default off → byte-identical): POST-tokenizer,
                    # per-sample UNIFORM token renorm to the step-0 input band.
                    # Only scales DOWN when the feedback token absmax exceeds the
                    # tolerated input reference (never amplifies), so in-band
                    # feedback is untouched. The anchor state (``decoded`` below)
                    # is UNCHANGED — only the tokens fed to the backbone scale.
                    tok = self._renorm_feedback_tokens(
                        cfg.name, tok, ref_absmax, debug=True
                    )
                pieces.append(tok)
                if return_decoded:
                    decoded_raw[cfg.name] = decoded
            else:
                pieces.append(sl)
        feedback_tokens = torch.cat(pieces, dim=1)
        if return_decoded:
            return feedback_tokens, decoded_raw
        return feedback_tokens

    def _tokenize_gt_onmanifold(
        self, gt_inputs: Dict[str, torch.Tensor], *, return_decoded: bool = False,
        feedback_normalize: bool = False,
        ref_absmax: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """Teacher-forcing feedback ON THE CODEC MANIFOLD.

        Mirrors :meth:`_resample_feedback` but starts from the GROUND-TRUTH state
        instead of the model's predicted logits: for code-path heads, round-trip
        the raw GT through the codec (``encode_target`` → ``decode``) before
        re-tokenizing, so the teacher state lives on the SAME manifold as the
        free-rollout decoded state. Re-tokenizing the RAW GT (the previous TF
        path) feeds an off-manifold-scale tensor into a backbone trained on
        codec-manifold inputs → bf16 overflow → the systematic k>=1 ece NaN.
        This applies failure-mode-7's on-manifold discipline to the TF path.

        Continuous heads (no codec) tokenize the raw GT directly (that IS their
        space). With ``return_decoded`` also returns ``{name: on-manifold state}``
        — the state the descriptor anchor reads, matching the free path."""
        pieces: List[torch.Tensor] = []
        decoded_raw: Dict[str, torch.Tensor] = {}
        for cfg in self.model.diagnostics:
            x = gt_inputs[cfg.name]
            head = self.model.diag_heads[cfg.name]
            if all(hasattr(head, m) for m in
                   ("code_logits", "sample_codes", "decode", "encode_target")):
                codes = head.encode_target(x)
                # On-manifold TF: decode the GT codes to the pixel state, then
                # re-tokenize. This is the hottest NaN locus (GT-code decode).
                decoded = head.decode(codes)
                # VIDEO layout fix: VideoCodeHead.decode returns the
                # VideoOutputHead-native (B, T, C, H, W), but the VideoTokenizer
                # (Conv3d patch_embed) expects (B, C, T, H, W). Permute before
                # re-tokenizing (only fires for video code-heads; spectro/fast/
                # slow-TS decode already returns the tokenizer-native layout).
                # Without this, the FSQ-video TF feedback path crashes at K>=1
                # ("expected 2 channels, got 3" — C/T axes swapped).
                _re = decoded
                if getattr(cfg, "kind", None) == "video" and _re.dim() == 5:
                    _re = _re.permute(0, 2, 1, 3, 4).contiguous()
                _tok = self.model.diag_tokenizers[cfg.name](_re)
                if feedback_normalize:
                    # OPTION-2 (default off → byte-identical): POST-tokenizer,
                    # per-sample UNIFORM token renorm to the step-0 input band —
                    # scale DOWN only when the GT-decode feedback exceeds the
                    # tolerated input reference. The anchor state (``decoded``) is
                    # UNCHANGED; only the tokens fed to the backbone scale.
                    _tok = self._renorm_feedback_tokens(
                        cfg.name, _tok, ref_absmax, debug=True
                    )
                pieces.append(_tok)
                if return_decoded:
                    decoded_raw[cfg.name] = decoded
                import os as _os
                if _os.environ.get("ROLLOUT_STATS_DEBUG") == "1":
                    _c = codes
                    _cmin, _cmax = int(_c.min()), int(_c.max())
                    _dec_am = float(decoded.abs().max())
                    print(
                        f"[bisect-TF] {cfg.name}: gt(mean={float(x.mean()):.3f} "
                        f"std={float(x.std()):.3f} absmax={float(x.abs().max()):.3f}) "
                        f"codes(dtype={_c.dtype} shape={tuple(_c.shape)} min={_cmin} "
                        f"max={_cmax} frac@min={float((_c==_cmin).float().mean()):.3f} "
                        f"frac@max={float((_c==_cmax).float().mean()):.3f}) "
                        f"decoded(absmax={_dec_am:.3f}) "
                        f"fbnorm={feedback_normalize} "
                        f"tok(absmax={float(_tok.abs().max()):.3f} finite={bool(torch.isfinite(_tok).all())})",
                        flush=True,
                    )
            elif cfg.kind == "video":
                valid = gt_inputs.get(f"{cfg.name}_valid")
                mask = valid.bool() if valid is not None else None
                pieces.append(self.model.diag_tokenizers[cfg.name](x, mask=mask))
                if return_decoded:
                    decoded_raw[cfg.name] = x
            else:
                pieces.append(self.model.diag_tokenizers[cfg.name](x))
                if return_decoded:
                    decoded_raw[cfg.name] = x
        feedback_tokens = torch.cat(pieces, dim=1)
        if return_decoded:
            return feedback_tokens, decoded_raw
        return feedback_tokens

    def _step(
        self,
        diag_tokens: torch.Tensor,
        act_inputs_k: Dict[str, torch.Tensor],
        *,
        k: int,
        batch: int,
        device: torch.device,
        start_time_s: torch.Tensor,
        use_film: bool,
        flow_noise: Optional[Dict[str, torch.Tensor]],
        collect_token_slices: bool,
    ):
        """One rollout step: backbone forward on ``diag_tokens`` (the input state
        for step ``k``) + per-modality decode. Returns
        ``(out_tokens, pred_diag_tokens, preds_k, slices_k)``. Pure of the
        feedback/TF decision (that lives in ``forward`` so it can gate on the
        next step). Split out so a contiguous group of steps can be wrapped in
        ``torch.utils.checkpoint`` for activation-memory savings."""
        if use_film:
            # FiLM model: actuators modulate the backbone (γ,β), NOT appended as tokens — must
            # mirror model.forward or the rollout evaluates a pathway the model never trained on.
            film_k = self.model._actuator_film_params(act_inputs_k)
            all_tokens = diag_tokens
        else:
            film_k = None
            act_tokens = self._tokenize_actuators(act_inputs_k)
            all_tokens = torch.cat([diag_tokens, act_tokens], dim=1)
        step_idx = torch.full((batch,), k, dtype=torch.long, device=device)
        time_s = start_time_s + k * self.dt_s
        if use_film:
            out_tokens = self.model.backbone(all_tokens, step_idx, time_s, film_params=film_k)
        else:
            out_tokens = self.model.backbone(all_tokens, step_idx, time_s)

        # Predictions are always the model's real backbone output — the TF /
        # feedback decision in ``forward`` only affects what flows into the
        # *next* iteration's backbone, not what's scored.
        pred_diag_tokens = out_tokens[:, : self.n_diag_tokens]
        if collect_token_slices:
            preds_k, slices_k = self._decode_diagnostics(
                pred_diag_tokens, flow_noise=flow_noise, return_slices=True,
            )
        else:
            preds_k = self._decode_diagnostics(pred_diag_tokens, flow_noise=flow_noise)
            slices_k = None
        return out_tokens, pred_diag_tokens, preds_k, slices_k

    def forward(
        self,
        initial_diag_inputs: Dict[str, torch.Tensor],
        act_inputs_per_step: List[Dict[str, torch.Tensor]],
        *,
        start_time_s: Optional[torch.Tensor] = None,
        collect_history: bool = True,
        gt_target_per_step: Optional[
            List[Dict[str, torch.Tensor]]
        ] = None,
        p_tf: float = 0.0,
        collect_token_slices: bool = False,
        collect_decoded_feedback: bool = False,
        grad_checkpoint_every: int = 0,
        flow_noise: Optional[Dict[str, torch.Tensor]] = None,
        feedback_mode: str = "continuous",
        feedback_temperature: float = 1.0,
        feedback_normalize: bool = False,
    ) -> RolloutResult:
        """Run a ``K``-step rollout.

        Parameters
        ----------
        initial_diag_inputs
            Ground-truth raw signals at step 0, one entry per diagnostic.
        act_inputs_per_step
            Length-``K`` list of actuator-input dicts, one per rollout step.
        start_time_s
            Optional ``(batch,)`` absolute-time tensor for step 0. Defaults
            to zeros.
        collect_history
            When ``False``, skip appending to ``diagnostic_tokens`` and
            ``backbone_outputs`` (returned lists are empty). Saves ~4 GB of
            GPU memory at K=80, batch=128. Default ``True`` preserves prior
            §5.9 test behaviour.
        gt_target_per_step
            Optional length-``K`` list of ground-truth diagnostic dicts;
            ``gt_target_per_step[k]`` is the GT state at ``t = (k+1)*dt_s``
            (i.e. the rollout target of step ``k``). Required when
            ``p_tf > 0``; ignored otherwise. Predictions and history are
            unaffected — they always reflect the model's actual outputs.
        p_tf
            Teacher-forcing probability at each step ``k >= 1``. With
            probability ``p_tf`` the next-step diagnostic input is the
            re-tokenized GT state; otherwise it is the backbone's
            previous output (the default free-rollout behaviour). The
            coin is flipped per ``(rollout-step, training-step)`` and
            applies uniformly across the batch. Default ``0.0`` (pure
            free-rollout, byte-identical to prior behaviour).
        collect_decoded_feedback
            When ``True``, populate ``RolloutResult.decoded_feedback`` — a
            length-``n_steps`` list where entry ``k`` is the RAW decoded
            diagnostic state used as the INPUT to step ``k`` (index 0 = the GT
            ``initial_diag_inputs``; index ``k >= 1`` = the on-manifold decoded
            state the feedback produced). This is the state the descriptor
            anchor must read at step ``k`` so the K-step trainer's anchor
            wiring matches the Gate-4 inference feedback. Default ``False`` →
            behaviour byte-identical to before (empty list).
        grad_checkpoint_every
            When ``> 0`` (and grad is enabled), wrap each contiguous group of
            this many rollout steps in ``torch.utils.checkpoint`` so activation
            memory scales with the group size instead of ``K``. ``0`` (default)
            runs the loop exactly as before. Results are unchanged (argmax /
            deterministic feedback recomputes identically); only memory differs.
        feedback_normalize
            When ``True``, apply OPTION-2 feedback-token renorm in BOTH the
            free/argmax ``_resample_feedback`` path and the teacher-forcing
            ``_tokenize_gt_onmanifold`` path. The feedback is tokenized the
            CURRENT way (``decode(codes)`` → backbone ``tokenizer``), then, POST
            tokenizer, each code-path modality's ``(B, n_tok, d_model)`` slice is
            scaled per-sample by a single scalar
            ``scale = clamp(ref_absmax / (fb_absmax + eps), max=1.0)`` so its token
            absmax never exceeds the step-0 INPUT-window reference band
            (``ref_absmax``, captured per sample from the initial tokenized
            diagnostics before any feedback). This only scales DOWN — in-band
            feedback (``fb <= ref``) is untouched (``scale == 1``, byte-identical).
            A UNIFORM per-sample scalar preserves the relative token structure
            (mode ridges vs floor) exactly, so it is mode-safe by construction.
            It targets the localized ece proj-conv NaN: the patch ``proj`` Conv2d
            has ONE fixed near-DC filter that saturates on the broadband floor of
            a codec-decoded window (feedback proj reaches ~2216, past the ~2101
            input band the single-step model tolerated) → bf16 overflow, hottest
            in the TF GT-code-decode path. The descriptor anchor still reads the
            unscaled PIXEL state (``decode(codes)`` in ``decoded_raw``); only the
            tokens fed to the backbone change. (Supersedes the FAILED
            pre-tokenizer per-(C,F) moment-matching — wrong locus, it normalized
            pixels not tokens — and the failed lattice-extreme code clamp.)
            Default ``False`` → byte-identical to the prior feedback (the running
            production chain, which does not set this flag, is unaffected).

        Returns
        -------
        RolloutResult
        """
        batch = next(iter(initial_diag_inputs.values())).shape[0]
        device = next(iter(initial_diag_inputs.values())).device
        n_steps = len(act_inputs_per_step)
        if start_time_s is None:
            start_time_s = torch.zeros(batch, device=device)

        # Teacher-forcing setup. ``use_tf`` is gated on both inputs being
        # supplied AND p_tf being non-zero, so the TF code path is fully
        # dormant when the trainer doesn't ask for it (preserves
        # byte-identity for existing tests / Aurora trainer / impulse
        # tests, none of which pass these args).
        use_tf = (
            p_tf > 0.0
            and gt_target_per_step is not None
            and len(gt_target_per_step) >= n_steps
        )

        diag_tokens = self._tokenize_diagnostics(initial_diag_inputs)

        # OPTION-2 reference band: per-sample INPUT-window token absmax at step 0,
        # per code-path modality. This is the tolerated reference each window's
        # feedback tokens are scaled DOWN to (never up). Captured from the SAME
        # step-0 tokenized diagnostics (before any feedback), sliced per modality
        # via the diagnostic token layout. Only computed under
        # ``feedback_normalize`` (else None → renorm helpers are a no-op → the
        # feedback is byte-identical). ``detach`` so it is a fixed reference, not
        # a gradient source.
        ref_absmax: Optional[Dict[str, torch.Tensor]] = None
        if feedback_normalize:
            ref_absmax = {}
            offsets = self._diag_token_slice_offsets()
            for cfg in self.model.diagnostics:
                head = self.model.diag_heads[cfg.name]
                if all(hasattr(head, m) for m in
                       ("code_logits", "sample_codes", "decode")):
                    s, e = offsets[cfg.name]
                    ref_absmax[cfg.name] = (
                        diag_tokens[:, s:e].detach().abs().amax(dim=(1, 2))
                    )
            import os as _os_rr
            if _os_rr.environ.get("ROLLOUT_STATS_DEBUG") == "1":
                for _nm, _v in ref_absmax.items():
                    print(f"[opt2-ref] step0 {_nm} ref_absmax={_v.tolist()}",
                          flush=True)

        diagnostic_tokens_history: List[torch.Tensor] = (
            [diag_tokens] if collect_history else []
        )
        predictions: List[Dict[str, torch.Tensor]] = []
        backbone_outputs: List[torch.Tensor] = []
        diag_token_slices_history: List[Dict[str, torch.Tensor]] = []
        # decoded_feedback[k] = raw decoded state fed as the INPUT to step k.
        # Index 0 is always the GT step-0 input.
        decoded_feedback_history: List[Dict[str, torch.Tensor]] = (
            [dict(initial_diag_inputs)] if collect_decoded_feedback else []
        )

        use_film = getattr(self.model, "use_actuator_film", False)

        def _decide_feedback(k: int, pred_diag_tokens: torch.Tensor):
            """Return ``(next_diag_tokens, decoded_raw_or_None)``: the tokens fed
            into step ``k+1`` and (when collecting) the RAW decoded state behind
            them for code-path modalities. Mirrors the original inline feedback
            block exactly — the ``torch.rand`` TF coin is drawn HERE, per step,
            so the default (``p_tf==0`` → ``use_tf`` False) path consumes zero
            draws and is byte-identical. On the last step there is no next step;
            the caller falls through to recording ``pred_diag_tokens``."""
            if k + 1 >= n_steps:
                return pred_diag_tokens, None
            if use_tf and torch.rand((), device=device).item() < p_tf:
                # Teacher-force ON THE CODEC MANIFOLD: round-trip the GT state at
                # t=(k+1)*dt_s through the codec (encode_target->decode) before
                # re-tokenizing, so the teacher lives on the SAME manifold as the
                # free-rollout decoded state. Re-tokenizing the RAW GT overflows
                # the backbone in bf16 (off-manifold scale) → the systematic
                # k>=1 ece NaN. decoded_raw (for the anchor) = the on-manifold GT.
                if collect_decoded_feedback:
                    return self._tokenize_gt_onmanifold(
                        gt_target_per_step[k], return_decoded=True,
                        feedback_normalize=feedback_normalize,
                        ref_absmax=ref_absmax,
                    )
                return self._tokenize_gt_onmanifold(
                    gt_target_per_step[k], feedback_normalize=feedback_normalize,
                    ref_absmax=ref_absmax,
                ), None
            if feedback_mode == "continuous":
                return pred_diag_tokens, None    # default: byte-identical to prior behaviour
            # GATE-4 FIX: code-space feedback. Feeding the CONTINUOUS prediction back is a
            # contraction → fixed-point freeze (rollout.py mean-collapse). Instead decode →
            # sample/argmax codes → re-tokenize the decoded (on-manifold) state → feed THAT back.
            # Sampling injects per-step stochasticity that breaks the contraction.
            if collect_decoded_feedback:
                return self._resample_feedback(
                    pred_diag_tokens, feedback_mode, feedback_temperature,
                    return_decoded=True, feedback_normalize=feedback_normalize,
                    ref_absmax=ref_absmax,
                )
            nxt = self._resample_feedback(
                pred_diag_tokens, feedback_mode, feedback_temperature,
                feedback_normalize=feedback_normalize,
                ref_absmax=ref_absmax,
            )
            return nxt, None

        def _run_group(group_start: int, group_end: int, diag_tokens_in: torch.Tensor):
            """Run steps [group_start, group_end) starting from ``diag_tokens_in``.

            Collects per-step tensors into flat lists so the whole group can be
            the body of ``torch.utils.checkpoint`` (which recomputes the forward
            in backward — every grad-carrying tensor is returned). Returns
            ``(next_diag_tokens, out_tokens[], pred_dicts[], slice_dicts[],
            feedback_tokens[], decoded_raw[])``. The feedback+TF decision runs
            INSIDE the group so the wiring is identical to the ungrouped loop."""
            dt = diag_tokens_in
            g_out_tokens: List[torch.Tensor] = []
            g_pred_dicts: List[Dict[str, torch.Tensor]] = []
            g_slice_dicts: List[Optional[Dict[str, torch.Tensor]]] = []
            g_feedback_tokens: List[torch.Tensor] = []
            g_decoded: List[Optional[Dict[str, torch.Tensor]]] = []
            for k in range(group_start, group_end):
                out_tokens, pred_diag_tokens, preds_k, slices_k = self._step(
                    dt, act_inputs_per_step[k], k=k, batch=batch, device=device,
                    start_time_s=start_time_s, use_film=use_film,
                    flow_noise=flow_noise, collect_token_slices=collect_token_slices,
                )
                g_out_tokens.append(out_tokens)
                g_pred_dicts.append(preds_k)
                g_slice_dicts.append(slices_k)
                dt, decoded_raw = _decide_feedback(k, pred_diag_tokens)
                g_feedback_tokens.append(dt)
                g_decoded.append(decoded_raw)
            return (dt, g_out_tokens, g_pred_dicts, g_slice_dicts,
                    g_feedback_tokens, g_decoded)

        use_ckpt = grad_checkpoint_every > 0 and torch.is_grad_enabled()
        group_size = max(1, grad_checkpoint_every)

        for group_start in range(0, n_steps, group_size):
            group_end = min(group_start + group_size, n_steps)
            if use_ckpt:
                results = torch_ckpt.checkpoint(
                    _run_group, group_start, group_end, diag_tokens,
                    use_reentrant=False,
                )
            else:
                results = _run_group(group_start, group_end, diag_tokens)
            (dt_next, g_out_tokens, g_pred_dicts, g_slice_dicts,
             g_feedback_tokens, g_decoded) = results
            for i, k in enumerate(range(group_start, group_end)):
                if collect_history:
                    backbone_outputs.append(g_out_tokens[i])
                predictions.append(g_pred_dicts[i])
                if collect_token_slices:
                    diag_token_slices_history.append(g_slice_dicts[i])
                # Diagnostic-token history: index k+1 = the feedback tokens fed
                # into step k+1 (identical to the original loop, grouping-blind).
                if collect_history:
                    diagnostic_tokens_history.append(g_feedback_tokens[i])
                # Decoded-feedback history: the RAW state that is the INPUT to
                # step k+1 (matching diagnostic_tokens_history's index k+1).
                if collect_decoded_feedback and k + 1 < n_steps:
                    dr = g_decoded[i]
                    if dr is None:
                        # Continuous head / continuous feedback: no decode →
                        # use the model's step-k prediction so every diagnostic
                        # the loss touches has an entry.
                        decoded_feedback_history.append(dict(g_pred_dicts[i]))
                    else:
                        # Code-path modalities carry their on-manifold decode;
                        # fill any continuous head from the step-k prediction so
                        # the dict is complete for the loss.
                        merged = dict(g_pred_dicts[i])
                        merged.update(dr)
                        decoded_feedback_history.append(merged)
            diag_tokens = dt_next

        return RolloutResult(
            predictions=predictions,
            diagnostic_tokens=diagnostic_tokens_history,
            backbone_outputs=backbone_outputs,
            diag_token_slices=diag_token_slices_history,
            decoded_feedback=decoded_feedback_history,
        )