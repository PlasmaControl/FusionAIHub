"""IGNITE Phase-B (MaskGIT dynamics) configuration + the frozen-codec frame layout.

Phase B is a factorized ST-transformer that models DYNAMICS over the FROZEN Phase-A discrete
codes — a "language model over plasma-state tokens". It does no perception (the codecs already
encode/decode raw signals), so it is lighter than the 48L continuous FAITH model; depth is a
first-class knob here (the point of the depth study).

Frame layout (frozen codec set, 2026-07-27):
    spectro  ece / bes / mhr / co2   192 tok each  -> 768
    video    tangtv_lower / _upper   108 tok each  -> 216
    slow-TS  7 signals                 4 tok each  ->  28
    ------------------------------------------------------
    TOTAL                                            1012 tokens / frame
Each token is a single integer code in [0, codebook_size); MaskGIT predicts a categorical over
each modality's own vocab. See docs/IGNITE_DESIGN.md §5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass(frozen=True)
class ModalitySpec:
    """One frozen codec's contribution to a frame's token layout.

    name           : codec identifier (e.g. "ece", "tangtv_lower", "mse").
    family         : "spectro" | "video" | "slowts" (drives within-modality position semantics).
    n_tok          : number of tokens this modality contributes per frame.
    codebook_size  : vocab size for this modality's tokens (= prod(fsq_levels), default 1000).
    """

    name: str
    family: str
    n_tok: int
    codebook_size: int = 1000


# The FROZEN Phase-A codec set that composes one plasma-state frame. Order is the canonical
# token order within a frame. mse is INCLUDED here (part of the state); its codec must clear the
# Phase-A gate before Phase-B TRAINING launches, but the layout/shape is fixed now.
FROZEN_MODALITIES: Tuple[ModalitySpec, ...] = (
    # spectro — 192 tokens each (patch_f=16 -> 32 freq, patch_t=16 -> 6 time)
    ModalitySpec("ece", "spectro", 192, 1000),
    ModalitySpec("bes", "spectro", 192, 1000),
    ModalitySpec("mhr", "spectro", 192, 1000),
    ModalitySpec("co2", "spectro", 192, 1000),
    # video — 108 tokens each divertor
    ModalitySpec("tangtv_lower", "video", 108, 1000),
    ModalitySpec("tangtv_upper", "video", 108, 1000),
    # slow-TS — 4 radial-zone tokens each (7 signals)
    ModalitySpec("ts_core_density", "slowts", 4, 1000),
    ModalitySpec("ts_core_temp", "slowts", 4, 1000),
    ModalitySpec("ts_tangential_density", "slowts", 4, 1000),
    ModalitySpec("ts_tangential_temp", "slowts", 4, 1000),
    ModalitySpec("cer_ti", "slowts", 4, 1000),
    ModalitySpec("cer_rot", "slowts", 4, 1000),
    ModalitySpec("mse", "slowts", 4, 1000),
    # fast-TS (filterscopes) — the ELM ACTIVITY ENVELOPE codec (E=5). The 4th codec family; was
    # ERRONEOUSLY omitted from the first layout. Codec is still PENDING (no passing best-ckpt yet —
    # collapsed like co2/mse were pre-fix), but the STATE includes it, so the layout does too.
    ModalitySpec("filterscopes", "fastts", 5, 1000),
)


@dataclass
class DynamicsConfig:
    """Phase-B MaskGIT dynamics model config.

    DEPTH is deliberately a plain field (not derived) so the depth study is a one-line sweep:
    ``DynamicsConfig(depth=k)`` for k in {12, 24, 36, 48}.
    """

    # --- frame layout (from the frozen codec set) --------------------------------------------
    modalities: Tuple[ModalitySpec, ...] = FROZEN_MODALITIES

    # --- backbone (confirmed 2026-07-27: d_model 1024, depth 24 start) -----------------------
    d_model: int = 1024
    depth: int = 24                # factorized blocks; each = spatial attn + temporal attn + FFN
    n_heads: int = 16              # d_model / 64
    ffn_mult: int = 4
    dropout: float = 0.0
    grad_checkpointing: bool = True  # recompute each block in backward (training only). REQUIRED at
    # d1024/depth24: one (B,F,N,d) activation is ~3 GB at bs8, ~24 blocks would exceed a 64 GB GCD.
    # Checkpointing keeps only block INPUTS (~10 GB) + one block's live activations -> fits.

    # --- temporal horizon --------------------------------------------------------------------
    k0_seed: int = 20              # real seed frames (~1.0 s) before prediction begins
    n_predict: int = 80            # predicted frames (80 * 50 ms = 4 s)

    # --- MaskGIT ------------------------------------------------------------------------------
    maskgit_decode_steps: int = 10         # iterative confidence-unmask steps at inference
    maskgit_mask_schedule: str = "cosine"  # tokens-kept-per-step schedule

    # --- scheduled sampling (rollout-drift mitigation; ramp over training) -------------------
    ss_ramp_final_frac: float = 0.15       # final own-sampled-code substitution fraction
    ss_ramp_steps: int = 40_000            # linear ramp 0 -> final over this many steps

    # --- complete-context training (CTF; MAGI arXiv 2501.12389) -------------------------------
    # Fraction of training windows that use the ROLLOUT's conditional structure: a clean
    # (fully visible) prefix and a heavily-masked suffix, loss on the suffix only. The default
    # random-mask objective masks ~64% of EVERY frame, so the model never trains on the
    # "complete context -> fully masked next frame" conditional that rollout actually uses.
    # 0.0 = off (original behaviour, bit-identical).
    ctf_frac: float = 0.0
    ctf_min_target_ratio: float = 0.8      # min mask ratio applied at/after the boundary

    # --- per-modality loss weighting ----------------------------------------------------------
    # "uniform" (default, original): every modality's masked CE counts the same, so a 4-token
    # slow-TS modality gets ~192x the per-token gradient of 768-token ece. "tokens" weights by
    # n_tok, "sqrt_tokens" by sqrt(n_tok) (a compromise that still protects small modalities).
    modality_loss_weight: str = "uniform"

    # --- actuator conditioning dropout (enables classifier-free guidance at rollout) ----------
    # Probability of zeroing the actuator embedding for a training sample. Without it the model
    # has no unconditional branch, so guidance cannot be applied at inference. Adds NO parameters.
    # 0.0 = off (original behaviour, bit-identical).
    actuator_dropout_p: float = 0.0

    # --- actuator conditioning (additive; causal) --------------------------------------------
    actuator_dim: int = 70                 # 7 modalities / 70 channels

    # --------------------------------------------------------------------------------------- #
    @property
    def n_modalities(self) -> int:
        return len(self.modalities)

    @property
    def tokens_per_frame(self) -> int:
        return sum(m.n_tok for m in self.modalities)

    @property
    def max_frames(self) -> int:
        return self.k0_seed + self.n_predict

    def modality_token_slices(self) -> List[Tuple[int, int]]:
        """Contiguous [start, end) token index range per modality within a frame (canonical order)."""
        out, off = [], 0
        for m in self.modalities:
            out.append((off, off + m.n_tok))
            off += m.n_tok
        return out
