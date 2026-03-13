from torch import nn
from typing import Optional

from tokamak_foundation_model.models.modality import (
    ActuatorBaselineAutoEncoder,
    SlowTimeSeriesBaselineAutoEncoder,
    FastTimeSeriesBaselineAutoEncoder,
    SpatialProfileBaselineAutoEncoder,
    SpectrogramBaselineAutoEncoder,
    SpectrogramResLSTMAutoEncoder,
    SpectrogramTFAttnAutoEncoder,
    SpectrogramMAEAutoEncoder,
    SpectrogramFSQVAEAutoEncoder,
    SpectrogramConvNeXtFSQAutoEncoder,
    SpectrogramCNNAutoEncoder,
    SpectrogramCNNPerceiverAutoEncoder,
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
    "spectrogram_res_lstm": SpectrogramResLSTMAutoEncoder,
    "spectrogram_tf_attn": SpectrogramTFAttnAutoEncoder,
    "spectrogram_mae": SpectrogramMAEAutoEncoder,
    "spectrogram_fsq_vae": SpectrogramFSQVAEAutoEncoder,
    "spectrogram_convnext_fsq": SpectrogramConvNeXtFSQAutoEncoder,
    "spectrogram_cnn": SpectrogramCNNAutoEncoder,
    "spectrogram_cnn_perceiver": SpectrogramCNNPerceiverAutoEncoder,
    "video": VideoBaselineAutoEncoder,
}

def build_model(
        model_name,
        d_model: Optional[int],
        n_tokens: Optional[int],
        n_channels: Optional[int],
        **kwargs
) -> nn.Module:
    """Build the appropriate autoencoder.

    All autoencoders share the same interface: (n_channels, d_model, n_tokens).
    """
    cls = MODEL_REGISTRY[model_name]
    if d_model is None and "d_model" not in kwargs:
        kwargs["d_model"] = 512  # default model dimension
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
