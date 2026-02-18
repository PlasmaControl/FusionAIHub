import torch
import torch.nn as nn
import torch.nn.functional as F

from .fast_time_series_baseline import (
    FastTimeSeriesBaselineEncoder,
    FastTimeSeriesBaselineDecoder,
    FastTimeSeriesBaselineAutoEncoder
    )


class ActuatorBaselineEncoder(FastTimeSeriesBaselineEncoder):

    def __init__(self,
        n_channels: int,
        d_model: int = 512,
        n_tokens: int = 100,
        input_length: int = 5000,
        n_conv_layers: int = 4,
        kernel_size: int = 3,
    ):
        super().__init__(
            n_channels,
            d_model,
            n_tokens,
            input_length,
            n_conv_layers,
            kernel_size
        )


class ActuatorBaselineDecoder(FastTimeSeriesBaselineDecoder):

    def __init__(
            self,
            n_channels: int = 6,
            input_length: int = 5000,
            d_model: int = 512,
            n_tokens: int = 100,
            n_deconv_layers: int = 4,
            kernel_size: int = 3,
    ):
        super().__init__(
            n_channels,
            input_length,
            d_model,
            n_tokens,
            n_deconv_layers,
            kernel_size
        )


class ActuatorBaselineAutoEncoder(FastTimeSeriesBaselineAutoEncoder):
    def __init__(
        self,
        n_channels: int = 6,
        input_length: int = 5000,
        d_model: int = 512,
        n_tokens: int = 100,
        n_layers: int = 4,
        kernel_size: int = 3,
    ):
        super().__init__(
            n_channels,
            input_length,
            d_model,
            n_tokens,
            n_layers,
            kernel_size
        )



if __name__ == "__main__":
    # python -m tokamak_foundation_model.models.modality.actuator_baseline

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, C, T = 4, 6, 100
    d_model = 64

    n_tokens = 10

    encoder = ActuatorBaselineEncoder(C, d_model, n_tokens=n_tokens).to(device)
    decoder = ActuatorBaselineDecoder(C, d_model).to(device)

    x = torch.randn(B, C, T)
    z = encoder(x.to(device))
    y = decoder(z, output_shape=(B, C, T))

    print(f"Input:   {x.shape}")
    print(f"Encoded: {z.shape}")
    print(f"Decoded: {y.shape}")

    autoencoder = ActuatorBaselineAutoEncoder(C, d_model, n_tokens=n_tokens).to(device)
    y = autoencoder(x.to(device))
    y = y.cpu().detach()

    print(f"Autoencoder Input:  {x.shape}, Output: {y.shape}")
