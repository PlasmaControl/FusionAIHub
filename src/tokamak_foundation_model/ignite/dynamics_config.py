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
    # mirnov joined with the band-power codecs. cache_modality_specs() iterates THIS tuple and
    # skips any name the cache lacks, so a modality missing here is dropped SILENTLY: the
    # all-spectrogram union cache (4000 tok/frame) built a 3072-token frame and trained on 4 of
    # its 5 modalities with no warning. Caches without mirnov are unaffected by this entry.
    ModalitySpec("mirnov", "spectro", 192, 1000),
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

    # --- generation-mode masking (train the task rollout actually performs) ------------------
    # Fraction of SAMPLES whose mask is built like rollout(): frames before a split point t
    # (drawn in [k0_seed, F)) stay fully visible and unscored, frames from t on ride the reveal
    # ladder including a full cold start. 0.0 = the historical cosine-prior-on-every-frame
    # scheme, bit-for-bit. Motivation (measured 2026-08-15, bp128_big best): under the true
    # rollout condition the model scores 1.5804 vs a model-free bigram's ~1.70, while its
    # tracked val CE is 0.9228 — it interpolates inside a half-given frame instead of predicting
    # the next one. 20% of val weight sits at mask ratio > 0.95 and carries a third of the loss.
    gen_mask_p: float = 0.0

    # Lag-k own-column code embeddings added in FrameTokenizer.embed (0 = off, no new params).
    # At the generation condition a per-column count table over a column's own last 3 codes beats
    # the 270M model (1.5539 vs 1.6090), and the per-modality deficit tracks own-history value
    # (co2 gains 0.188 nats from it and the model loses by 0.143; mhr gains 0.064 and the model
    # already wins). k=3 hands the head that exact context so the table is representable rather
    # than something optimization has to rediscover through 16 shared-parameter layers.
    lag_embed_k: int = 0

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
