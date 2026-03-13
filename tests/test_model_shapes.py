import pytest
import torch

from tokamak_foundation_model.models.model_factory import MODEL_REGISTRY
from tokamak_foundation_model.models.modality.spectrogram_normalizer import (
    NormalizedSpectrogramAutoEncoder,
)


# Define test configurations per model type
# Each entry: (model_name, model_kwargs, input_shape_without_batch)
MODEL_TEST_CONFIGS = [
    (
        "actuator",
        {"n_channels": 5, "d_model": 32, "n_tokens": 10, "input_length": 500},
        (5, 500),  # (channels, time)
    ),
    (
        "fast_time_series",
        {"n_channels": 6, "d_model": 32, "n_tokens": 10, "input_length": 500},
        (6, 500),  # (channels, time)
    ),
    (
        "slow_time_series",
        {"n_channels": 6, "d_model": 32, "n_tokens": 10},
        (6, 100),  # (channels, time)
    ),
    (
        "profile",
        {
            "n_channels": 1, "d_model": 32, "n_tokens": 10,
            "n_spatial_points": 50, "n_time_points": 50,
        },
        (50, 50),  # (spatial, time)
    ),
    (
        "spectrogram",
        {"n_channels": 4, "d_model": 32, "n_output_tokens": 0},
        (4, 64, 64),  # (channels, freq, time)
    ),
    (
        "spectrogram_res_lstm",
        {"n_channels": 4, "d_model": 32, "n_output_tokens": 0},
        (4, 64, 64),  # (channels, freq, time)
    ),
    (
        "spectrogram_tf_attn",
        {"n_channels": 4, "hidden_dim": 32, "latent_dim": 2, "freq_dim": 8},
        (4, 64, 64),  # (channels, freq, time)
    ),
    (
        "spectrogram_mae",
        {"n_channels": 4, "d_model": 32, "n_tokens": 0, "patch_h": 4, "patch_w": 4},
        (4, 64, 64),  # (channels, freq, time)
    ),
    (
        "spectrogram_fsq_vae",
        {
            "n_channels": 4, "d_model": 32, "n_tokens": 0,
            "patch_h": 4, "patch_w": 4,
            "n_enc_layers": 2, "n_dec_layers": 2, "n_heads": 4,
            "fsq_levels": [4, 3, 3],  # 36 codes, fast for tests
        },
        (4, 64, 64),  # (channels, freq, time)
    ),
    (
        "spectrogram_fsq_vae",
        {
            "n_channels": 4, "d_model": 32, "n_tokens": 0,
            "patch_h": 4, "patch_w": 4,
            "n_enc_layers": 2, "n_dec_layers": 2, "n_heads": 4,
            "fsq_levels": [4, 3, 3],
            "per_channel_patch": True,
        },
        (4, 64, 64),  # (channels, freq, time)
    ),
    (
        "spectrogram_convnext_fsq",
        {
            "n_channels": 4, "d_model": 32, "n_tokens": 0,
            "dims": [32, 64], "depths": [2, 2], "stem_stride": 4,
            "fsq_levels": [4, 3, 3],
        },
        (4, 64, 64),  # (channels, freq, time)
    ),
    (
        "spectrogram_cnn",
        {"n_channels": 4, "d_model": 32, "dims": [32, 64]},
        (4, 64, 64),  # (channels, freq, time)
    ),
    (
        "spectrogram_cnn",
        {"n_channels": 4, "d_model": 32, "dims": [32, 64], "bottleneck_dim": 4},
        (4, 64, 64),  # (channels, freq, time)
    ),
    # CNN Perceiver — continuous bottleneck
    (
        "spectrogram_cnn_perceiver",
        {
            "n_channels": 4, "d_model": 32, "n_tokens": 8,
            "dims": [32, 64], "n_heads": 4, "n_self_layers": 1,
            "n_dec_self_layers": 1,
        },
        (4, 64, 64),  # (channels, freq, time)
    ),
    # CNN Perceiver — with FSQ
    (
        "spectrogram_cnn_perceiver",
        {
            "n_channels": 4, "d_model": 32, "n_tokens": 8,
            "dims": [32, 64], "n_heads": 4, "n_self_layers": 1,
            "n_dec_self_layers": 1, "fsq_levels": [4, 3, 3],
        },
        (4, 64, 64),  # (channels, freq, time)
    ),
    (
        "video",
        {"n_channels": 1, "d_model": 32, "n_tokens": 0},
        (10, 32, 32),  # (time, height, width)
    ),
]


@pytest.mark.parametrize(
    "model_name,model_kwargs,input_shape",
    MODEL_TEST_CONFIGS,
    ids=[c[0] for c in MODEL_TEST_CONFIGS],
)
@pytest.mark.parametrize("batch_size", [1, 4])
def test_autoencoder_output_shape(model_name, model_kwargs, input_shape, batch_size):
    """Each autoencoder should produce output matching input shape."""
    cls = MODEL_REGISTRY[model_name]
    model = cls(**model_kwargs)
    model.eval()

    x = torch.randn(batch_size, *input_shape)

    with torch.no_grad():
        y = model(x)

    assert y.shape == x.shape, (
        f"{model_name}: output shape {y.shape} != input shape {x.shape}"
    )


@pytest.mark.parametrize(
    "model_name,model_kwargs,input_shape",
    [c for c in MODEL_TEST_CONFIGS if c[0] not in ("video", "profile")],
    ids=[c[0] for c in MODEL_TEST_CONFIGS if c[0] not in ("video", "profile")],
)
def test_encoder_output_is_finite(model_name, model_kwargs, input_shape):
    """Encoder output should not contain NaN or Inf."""
    cls = MODEL_REGISTRY[model_name]
    model = cls(**model_kwargs)
    model.eval()

    x = torch.randn(2, *input_shape)

    with torch.no_grad():
        z = model.encoder(x)

    assert torch.isfinite(z).all(), f"{model_name}: encoder output contains NaN/Inf"


def test_all_registry_models_covered():
    """Ensure all models in MODEL_REGISTRY have test configs."""
    tested = {c[0] for c in MODEL_TEST_CONFIGS}
    registered = set(MODEL_REGISTRY.keys())
    missing = registered - tested
    assert not missing, f"Models in registry without test configs: {missing}"


@pytest.mark.parametrize("batch_size", [1, 4])
def test_normalized_spectrogram_autoencoder(batch_size):
    """NormalizedSpectrogramAutoEncoder should preserve input shape."""
    n_channels, F, T = 4, 64, 64
    inner = MODEL_REGISTRY["spectrogram_fsq_vae"](
        n_channels=n_channels, d_model=32, n_tokens=0,
        patch_h=4, patch_w=4, n_enc_layers=2, n_dec_layers=2, n_heads=4,
        fsq_levels=[4, 3, 3],
    )
    model = NormalizedSpectrogramAutoEncoder(inner, n_channels, F)
    model.eval()

    # Use non-negative input (simulating log-preprocessed data)
    x = torch.rand(batch_size, n_channels, F, T)
    with torch.no_grad():
        y = model(x)
    assert y.shape == x.shape, f"output shape {y.shape} != input shape {x.shape}"


def test_normalized_encoder_output_is_finite():
    """Normalizer wrapper should expose encoder with finite output."""
    n_channels, F, T = 4, 64, 64
    inner = MODEL_REGISTRY["spectrogram_cnn"](
        n_channels=n_channels, d_model=32, dims=[32, 64],
    )
    model = NormalizedSpectrogramAutoEncoder(inner, n_channels, F)
    model.eval()

    x = torch.rand(2, n_channels, F, T)
    with torch.no_grad():
        z = model.encoder(x)
    assert torch.isfinite(z).all(), "encoder output contains NaN/Inf"
