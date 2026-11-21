"""Phase-A FSQ bottleneck for the fast-TS (filterscopes) codec.

The quantizer only reads ``cfg.d_model``, ``cfg.fsq_levels``, ``cfg.diversity_weight``
and ``cfg.fsq_dim``, which :class:`~ignite.config.FastTSCodecConfig` defines with the
same meaning as :class:`~ignite.config.SpectroCodecConfig`. So the fast-TS quantizer is
:class:`~ignite.quantizer.SpectroQuantizer` (straight-through FSQ plus the Genie/LFQ
entropy term) under another name, as :class:`~ignite.video_quantizer.VideoQuantizer` is
for video (docs/IGNITE_DESIGN.md §7).
"""
from __future__ import annotations

from .config import FastTSCodecConfig
from .quantizer import SpectroQuantizer


class FastTSQuantizer(SpectroQuantizer):
    """FSQ bottleneck for the fast-TS codec.

    ``quantize(feats) -> (quant (B,n_tok,d_model), codes (B,n_tok,fsq_dim) long)`` and
    ``entropy_loss(feats) -> scalar`` are inherited from
    :class:`~ignite.quantizer.SpectroQuantizer`; only the config type differs.
    """

    def __init__(self, cfg: FastTSCodecConfig) -> None:  # noqa: D401 (thin init)
        super().__init__(cfg)  # SpectroQuantizer.__init__ only reads cfg.d_model / cfg.fsq_levels
