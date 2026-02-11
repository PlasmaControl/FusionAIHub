from .base import ModalityEncoder, ModalityDecoder
from .time_series_baseline import TimeSeriesEncoder, TimeSeriesDecoder
from .fast_time_series_baseline import FastTimeSeriesEncoder, FastTimeSeriesDecoder
from .spectrogram_baseline import SpectrogramEncoder, SpectrogramDecoder
from .video_baseline import VideoEncoder, VideoDecoder
from .text_baseline import TextEncoder, TextDecoder

PROCESSOR_REGISTRY = {
    "spectrogram": SpectrogramEncoder,
    "timeseries": TimeSeriesEncoder,
    "fast_timeseries": FastTimeSeriesEncoder,
    "video": VideoEncoder,
    "text": TextEncoder,
}

DECODER_REGISTRY = {
    "spectrogram": SpectrogramDecoder,
    "timeseries": TimeSeriesDecoder,
    "fast_timeseries": FastTimeSeriesDecoder,
    "video": VideoDecoder,
    "text": TextDecoder,
}

__all__ = [
    "ModalityEncoder", "ModalityDecoder",
    "TimeSeriesEncoder", "TimeSeriesDecoder",
    "FastTimeSeriesEncoder", "FastTimeSeriesDecoder",
    "SpectrogramEncoder", "SpectrogramDecoder",
    "VideoEncoder", "VideoDecoder",
    "TextEncoder", "TextDecoder",
    "PROCESSOR_REGISTRY",
    "DECODER_REGISTRY",
]
