import pytest
import torch

from tokamak_foundation_model.models.model_factory import MODEL_REGISTRY


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
    # Channel-AST frame_width=2
    (
        "spectrogram_channel_ast",
        {
            "n_channels": 4, "d_model": 32, "n_tokens": 0,
            "freq_bins": 64, "frame_width": 2,
            "n_enc_layers": 2, "n_dec_layers": 2, "n_heads": 4,
            "time_conv_kernel": 3,
        },
        (4, 64, 64),
    ),
    # Channel-AST frame_width=4
    (
        "spectrogram_channel_ast",
        {
            "n_channels": 4, "d_model": 32, "n_tokens": 0,
            "freq_bins": 64, "frame_width": 4,
            "n_enc_layers": 2, "n_dec_layers": 2, "n_heads": 4,
            "time_conv_kernel": 3,
        },
        (4, 64, 64),
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

    if isinstance(y, tuple):
        y = y[0]
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
