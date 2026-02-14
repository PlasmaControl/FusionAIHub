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
    "ech": "actuator",
    "pin": "actuator",
    "tin": "actuator",
    "d_alpha": "fast_time_series",
    "mse": "profile",
    "ts_core_density": "profile",
    "mhr": "spectrogram",
    "ece": "spectrogram",
    "co2": "spectrogram",
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

def build_model(model_name, n_channels, d_model, n_tokens):
    """Build the appropriate autoencoder.

    All autoencoders share the same interface: (n_channels, d_model, n_tokens).
    """
    cls = MODEL_REGISTRY[model_name]
    kwargs = dict(n_channels=n_channels, d_model=d_model)
    if n_tokens is not None: kwargs["n_tokens"] = n_tokens
    return cls(**kwargs)
