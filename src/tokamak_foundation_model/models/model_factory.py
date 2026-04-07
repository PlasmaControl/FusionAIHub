from torch import nn
from typing import Optional

from tokamak_foundation_model.models.modality import (
    ActuatorBaselineAutoEncoder,
    SlowTimeSeriesBaselineAutoEncoder,
    FastTimeSeriesBaselineAutoEncoder,
    SpatialProfileBaselineAutoEncoder,
    SpectrogramBaselineAutoEncoder,
    VideoBaselineAutoEncoder,
)


SIGNAL_MODEL_DEFAULTS = {
    "gas": "actuator",
    "gas_flow": "actuator",
    "gas_raw": "actuator",
    "ech": "actuator",
    "pin": "actuator",
    "tin": "actuator",
    "ich": "fast_time_series",
    "i_coil": "fast_time_series",
    "filterscopes": "fast_time_series",
    "d_alpha": "fast_time_series",
    "sxr": "fast_time_series",
    "neutron_rate": "fast_time_series",
    "bolo_raw": "fast_time_series",
    "mse": "profile",
    "ts_core_density": "profile",
    "ts_core_temp": "profile",
    "ts_tangential_density": "profile",
    "ts_tangential_temp": "profile",
    "cer_ti": "profile",
    "cer_rot": "profile",
    "vib": "slow_time_series",
    "mhr": "spectrogram",
    "ece": "spectrogram",
    "co2": "spectrogram",
    "mirnov": "spectrogram",
    "langmuir": "spectrogram",
    "bes": "spectrogram",
    "bolo": "video",
    "irtv": "video",
    "tangtv": "video",
}

MODEL_REGISTRY = {
    "actuator": ActuatorBaselineAutoEncoder,
    "fast_time_series": FastTimeSeriesBaselineAutoEncoder,
    "slow_time_series": SlowTimeSeriesBaselineAutoEncoder,
    "profile": SpatialProfileBaselineAutoEncoder,
    "spectrogram": SpectrogramBaselineAutoEncoder,
    "video": VideoBaselineAutoEncoder,
}

def build_model(
        model_name,
        d_model: Optional[int] = None,
        n_tokens: Optional[int] = None,
        n_channels: Optional[int] = None,
        **kwargs
) -> nn.Module:
    """Build the appropriate autoencoder.

    All autoencoders share the same interface: (n_channels, d_model, n_tokens).
    """
    cls = MODEL_REGISTRY[model_name]
    if d_model is None and "d_model" not in kwargs:
        kwargs["d_model"] = 64
    else:
        kwargs["d_model"] = d_model
    if n_tokens is None and "n_tokens" not in kwargs:
        kwargs["n_tokens"] = 20
    else:
        kwargs["n_tokens"] = n_tokens
    if n_channels is None and "n_channels" not in kwargs:
        kwargs["n_channels"] = 1
    else:
        kwargs["n_channels"] = n_channels
    return cls(**kwargs)
