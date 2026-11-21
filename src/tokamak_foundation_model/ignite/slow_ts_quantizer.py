"""FSQ bottleneck for the slow-TS codec.

The FSQ math is the spectrogram codec's. ``quantizer.SpectroQuantizer`` reads only
``d_model``, ``fsq_levels``, ``diversity_weight`` and ``fsq_dim``, which
:class:`~ignite.config.SlowTSCodecConfig` carries with the same meaning, so this class is
a thin subclass rather than a copy, as in ``video_quantizer`` (docs/IGNITE_DESIGN.md §7).
"""
from __future__ import annotations

from .config import SlowTSCodecConfig
from .quantizer import SpectroQuantizer


class SlowTSQuantizer(SpectroQuantizer):
    """FSQ bottleneck for the slow-TS codec.

    ``quantize`` and ``entropy_loss`` are inherited unchanged from
    :class:`~ignite.quantizer.SpectroQuantizer`; only the config type differs.
    """

    def __init__(self, cfg: SlowTSCodecConfig) -> None:  # noqa: D401 (thin init)
        super().__init__(cfg)
