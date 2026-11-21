"""FSQ bottleneck for the tangtv video codec.

The FSQ math is the spectrogram codec's. ``quantizer.SpectroQuantizer`` reads only
``d_model``, ``fsq_levels``, ``diversity_weight`` and ``fsq_dim``, which
:class:`~ignite.config.VideoCodecConfig` carries with the same meaning, so this class is
a thin subclass rather than a copy (docs/IGNITE_DESIGN.md §7).
"""
from __future__ import annotations

from .config import VideoCodecConfig
from .quantizer import SpectroQuantizer


class VideoQuantizer(SpectroQuantizer):
    """FSQ bottleneck for the video codec.

    ``quantize`` and ``entropy_loss`` are inherited unchanged from
    :class:`~ignite.quantizer.SpectroQuantizer`; only the config type differs.
    """

    def __init__(self, cfg: VideoCodecConfig) -> None:  # noqa: D401 (thin init)
        super().__init__(cfg)
