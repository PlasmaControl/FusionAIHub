"""Configuration and frozen-codec frame layout of the IGNITE dynamics model.

The dynamics model is a factorized spatio-temporal transformer over the discrete
codes of the frozen IGNITE codecs: a language model over plasma-state tokens. It
performs no perception (the codecs encode and decode the raw signals), so depth is
a first-class knob. See ``docs/IGNITE_DESIGN.md`` section 5.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


@dataclass(frozen=True)
class ModalitySpec:
    """One frozen codec's contribution to the token layout of a frame.

    Parameters
    ----------
    name : str
        Codec identifier, e.g. ``"ece"``, ``"tangtv_lower"`` or ``"mse"``.
    family : str
        Codec family: ``"spectro"``, ``"video"``, ``"slowts"`` or ``"fastts"``.
    n_tok : int
        Number of tokens the modality contributes per frame.
    codebook_size : int, default=1000
        Vocabulary size of the modality's tokens (product of the FSQ levels).
    """

    name: str
    family: str
    n_tok: int
    codebook_size: int = 1000


# Canonical token order within a frame. `cache_modality_specs` iterates this tuple
# and skips any name a cache lacks, so a modality missing here is dropped silently.
FROZEN_MODALITIES: Tuple[ModalitySpec, ...] = (
    ModalitySpec("ece", "spectro", 192, 1000),
    ModalitySpec("bes", "spectro", 192, 1000),
    ModalitySpec("mhr", "spectro", 192, 1000),
    ModalitySpec("co2", "spectro", 192, 1000),
    ModalitySpec("mirnov", "spectro", 192, 1000),
    ModalitySpec("tangtv_lower", "video", 108, 1000),
    ModalitySpec("tangtv_upper", "video", 108, 1000),
    ModalitySpec("ts_core_density", "slowts", 4, 1000),
    ModalitySpec("ts_core_temp", "slowts", 4, 1000),
    ModalitySpec("ts_tangential_density", "slowts", 4, 1000),
    ModalitySpec("ts_tangential_temp", "slowts", 4, 1000),
    ModalitySpec("cer_ti", "slowts", 4, 1000),
    ModalitySpec("cer_rot", "slowts", 4, 1000),
    ModalitySpec("mse", "slowts", 4, 1000),
    ModalitySpec("filterscopes", "fastts", 5, 1000),
)


def modalities_from_manifest(path: Path | str) -> Tuple[ModalitySpec, ...]:
    """The frame layout of a PINNED codec bundle, read from its ``codecs/MANIFEST.json``.

    FROZEN_MODALITIES above describes one codec generation; a bundle pinned later (v4 adds
    mirnov, 15 modalities, 1209 tokens/frame) carries its own table, and a dynamics checkpoint
    agrees with THAT one. JSON objects preserve insertion order, so the manifest's key order is
    the canonical token order -- it is what the checkpoint was trained with.
    """
    entries = json.loads(Path(path).read_text())["modalities"]
    return tuple(
        ModalitySpec(name, e["family"], int(e["n_tok"]), int(e["codebook_size"]))
        for name, e in entries.items()
    )


@dataclass
class DynamicsConfig:
    """Hyper-parameters of :class:`~.maskgit.MaskGITDynamics`.

    Parameters
    ----------
    modalities : tuple of ModalitySpec, default=FROZEN_MODALITIES
        Frame layout, in canonical token order.
    d_model : int, default=1024
        Width of the transformer.
    depth : int, default=24
        Number of factorized blocks (spatial attention, temporal attention, FFN).
    n_heads : int, default=16
        Attention heads per attention layer.
    ffn_mult : int, default=4
        FFN hidden width as a multiple of ``d_model``.
    dropout : float, default=0.0
        Dropout on the attention outputs and inside the FFN.
    grad_checkpointing : bool, default=True
        Recompute each block in the backward pass (training only). Needed once
        ``d_model * depth >= 4096`` for the activations to fit on one GCD.
    k0_seed : int, default=20
        Real seed frames before prediction starts (1 s at 50 ms per frame).
    n_predict : int, default=80
        Predicted frames (4 s).
    maskgit_decode_steps : int, default=10
        Iterative confidence-unmask steps when generating one frame.
    maskgit_mask_schedule : str, default="cosine"
        Unmasking schedule.
    ss_ramp_final_frac : float, default=0.75
        Final fraction of context frames replaced by the model's own samples
        (scheduled sampling). 0.75 is the best single setting measured (CE 0.1053,
        token accuracy 0.9792); it is a peak, not a monotone knob (1.0 collapses,
        leaving no teacher signal). v4 trained with 0.
    ss_ramp_steps : int, default=2000
        Steps of the linear scheduled-sampling ramp; a ramp longer than the run only
        ever reaches part of the final fraction.
    ctf_frac : float, default=0.0
        Fraction of training windows built like a rollout (complete-context
        training, MAGI arXiv 2501.12389): a clean, fully visible prefix and a heavily
        masked suffix, loss on the suffix only. ``0.0`` is off.
    ctf_min_target_ratio : float, default=0.8
        Minimum mask ratio at and after the CTF boundary.
    sf_frames : int, default=0
        Self-forcing (arXiv 2506.08009): frames the model rolls out from its own
        output, under ``no_grad`` and detached, before the supervised frame. ``0`` is
        off.
    sf_decode_steps : int, default=4
        MaskGIT steps of the training-time self-forcing rollout (inference uses
        ``maskgit_decode_steps``).
    sf_prob : float, default=1.0
        Probability that a window uses the self-rollout context when
        ``sf_frames > 0``.
    modality_loss_weight : str, default="uniform"
        Per-modality CE weight: ``"uniform"`` (every modality counts the same, so a
        4-token slow-TS modality gets ~192x the per-token gradient of a 768-token
        one), ``"tokens"`` (by ``n_tok``) or ``"sqrt_tokens"`` (by ``sqrt(n_tok)``).
    actuator_dropout_p : float, default=0.0
        Probability of zeroing a sample's actuator conditioning in training, which
        trains the unconditional branch classifier-free guidance needs. No
        parameters; ``0.0`` is off.
    gen_mask_p : float, default=0.0
        Fraction of samples masked like a rollout: frames before a split point stay
        visible and unscored, later frames follow the reveal ladder. ``0.0`` keeps
        the cosine-prior masking on every frame.
    latent_head : bool, default=False
        Replace each modality's codebook classifier by a Gaussian head over the
        token's FSQ grid coordinate, so a miss by one grid cell is cheap.
        Predictions are rounded back to codes.
    fsq_levels : tuple of int, default=(8, 5, 5, 5)
        FSQ levels of the codecs, used by the latent head to unpack codes.
    lag_embed_k : int, default=0
        Number of own-column history embeddings per token (0 adds no parameters).
    gen_horizon_alpha : float, default=0.0
        Generation-mode horizon exponent, ``p(h) ~ h**-alpha``. ``0.0`` draws the
        split point uniformly.
    gen_horizon_max : int, default=0
        If positive, draw the generation-mode horizon uniformly in
        ``[1, gen_horizon_max]``. Takes precedence over ``gen_horizon_alpha``.
    act_cross_attn : bool, default=False
        Add per-token cross-attention over the actuator channels.
    actuator_dim : int, default=88
        Number of actuator channels (``train_dynamics._ACT_SPEC``: 8 groups, 88
        channels).
    text_embed_dim : int, default=0
        Width of the precomputed per-shot text embedding, added like the actuators.
        ``0`` is off: no module is built and no RNG is drawn.
    text_dropout_p : float, default=0.0
        Per-sample probability of replacing the text embedding by a learned null
        embedding (missing text, classifier-free guidance).
    """

    modalities: Tuple[ModalitySpec, ...] = FROZEN_MODALITIES
    d_model: int = 1024
    depth: int = 24
    n_heads: int = 16
    ffn_mult: int = 4
    dropout: float = 0.0
    grad_checkpointing: bool = True
    k0_seed: int = 20
    n_predict: int = 80
    maskgit_decode_steps: int = 10
    maskgit_mask_schedule: str = "cosine"
    ss_ramp_final_frac: float = 0.75
    ss_ramp_steps: int = 2_000
    ctf_frac: float = 0.0
    ctf_min_target_ratio: float = 0.8
    sf_frames: int = 0
    sf_decode_steps: int = 4
    sf_prob: float = 1.0
    modality_loss_weight: str = "uniform"
    actuator_dropout_p: float = 0.0
    gen_mask_p: float = 0.0
    latent_head: bool = False
    fsq_levels: Tuple[int, ...] = (8, 5, 5, 5)
    lag_embed_k: int = 0
    gen_horizon_alpha: float = 0.0
    gen_horizon_max: int = 0
    act_cross_attn: bool = False
    actuator_dim: int = 88
    text_embed_dim: int = 0
    text_dropout_p: float = 0.0

    @property
    def n_modalities(self) -> int:
        """Number of modalities in the frame."""
        return len(self.modalities)

    @property
    def tokens_per_frame(self) -> int:
        """Total number of tokens in one frame."""
        return sum(m.n_tok for m in self.modalities)

    @property
    def max_frames(self) -> int:
        """Window length: seed frames plus predicted frames."""
        return self.k0_seed + self.n_predict

    def modality_token_slices(self) -> List[Tuple[int, int]]:
        """Return the ``[start, end)`` token range of each modality in a frame."""
        out, off = [], 0
        for m in self.modalities:
            out.append((off, off + m.n_tok))
            off += m.n_tok
        return out
