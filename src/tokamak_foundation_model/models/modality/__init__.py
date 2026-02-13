from .fast_time_series_baseline import (
    FastTimeSeriesEncoder,
    FastTimeSeriesDecoder,
    FastTimeSeriesAutoEncoder,
)
from .slow_time_series_baseline import (
    SlowTimeSeriesEncoder,
    SlowTimeSeriesDecoder,
    SlowTimeSeriesAutoEncoder,
)
from .profile_baseline import (
    SpatialProfileEncoder,
    SpatialProfileDecoder,
    SpatialProfileAutoEncoder,
)
from .spectrogram_baseline import SpectrogramAutoEncoder
from .video_baseline import VideoAutoEncoder