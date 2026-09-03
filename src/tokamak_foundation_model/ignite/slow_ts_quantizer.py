"""Phase-A FSQ bottleneck for the slow-TS codec.

The FSQ math is IDENTICAL to the spectrogram codec's (``quantizer.SpectroQuantizer``): the
quantizer only touches ``cfg.d_model`` / ``cfg.fsq_levels`` / ``cfg.diversity_weight`` /
``cfg.fsq_dim`` — fields the :class:`~ignite.config.SlowTSCodecConfig` carries with the same
names and meaning as :class:`~ignite.config.SpectroCodecConfig`. So the slow-TS quantizer
REUSES :class:`~ignite.quantizer.SpectroQuantizer` verbatim (straight-through FSQ + the
Genie/LFQ entropy anti-collapse term) — a thin, documented alias rather than a copy, exactly
as ``video_quantizer.VideoQuantizer`` does. Deliberate infra reuse (docs/IGNITE_DESIGN.md §7).

Only external libs + config.py + sibling ``quantizer`` are used; no FAITH model code.
"""
from __future__ import annotations

from .config import SlowTSCodecConfig
from .quantizer import SpectroQuantizer


class SlowTSQuantizer(SpectroQuantizer):
    """FSQ bottleneck for the slow-TS codec — same STE-FSQ + entropy machinery as spectro.

    ``quantize(feats) -> (quant (B,n_tok,d_model), codes (B,n_tok,fsq_dim) long)`` and
    ``entropy_loss(feats) -> scalar`` are inherited UNCHANGED from
    :class:`~ignite.quantizer.SpectroQuantizer`; only the config *type* differs (a
    ``SlowTSCodecConfig``, which exposes the identical ``d_model`` / ``fsq_levels`` /
    ``fsq_dim`` / ``diversity_weight`` fields the parent reads).
    """

    def __init__(self, cfg: SlowTSCodecConfig) -> None:  # noqa: D401 (thin init)
        super().__init__(cfg)  # SpectroQuantizer.__init__ only reads cfg.d_model / cfg.fsq_levels
