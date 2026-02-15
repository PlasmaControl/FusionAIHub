import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ModalityEncoder, ModalityDecoder, ModalityAutoEncoder


class SpectrogramBaselineEncoder(ModalityEncoder):
    def __init__(self, 
        n_channels: int, 
        d_model: int = 256, 
        n_output_tokens: int = 0,
    ):
        super().__init__(n_channels, d_model, n_output_tokens)

        dims = [1, 32, 64, 128, d_model]

        self.net = nn.Sequential(
            nn.Conv3d(dims[0], dims[1], kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(dims[1], dims[2], kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.GELU(),
            nn.Conv3d(dims[2], dims[3], kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv3d(dims[3], dims[4], kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )

    def forward(self, x):
        B, C, Fr, T = x.shape
        x = x.unsqueeze(1)
        z = self.net(x)
        return z


class SpectrogramBaselineDecoder(ModalityDecoder):
    def __init__(self, 
        n_channels: int, 
        d_model: int = 256, 
    ):
        super().__init__(n_channels, d_model)

        dims = [1, 32, 64, 128, d_model]

        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(dims[4], dims[3], kernel_size=3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(dims[3], dims[2], kernel_size=3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear", align_corners=False),
            nn.Conv3d(dims[2], dims[1], kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(dims[1], dims[0], kernel_size=3, padding=1),
        )

    def forward(self, z, output_shape=None):
        x = self.net(z)
        if output_shape is not None:
            x = F.interpolate(
                x, size=output_shape, mode="trilinear", align_corners=False
            )
        x = x.squeeze(1)
        return x

class SpectrogramBaselineAutoEncoder(ModalityAutoEncoder):
    """
    Based on 3DCAE implementation at https://github.com/micah35s/Autoencoder-Image-Compression
    https://github.com/faadi809/HSI-compression-benchmark
    """

    def __init__(self, 
        n_channels: int, 
        d_model: int = 256, 
        n_output_tokens: int = 0,
    ):
        super().__init__(n_channels, d_model, n_output_tokens)
        self.n_channels = n_channels
        self.d_model = d_model

        self.encoder = SpectrogramBaselineEncoder(n_channels, d_model, n_output_tokens)
        self.decoder = SpectrogramBaselineDecoder(n_channels, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, Fr, T = x.shape
        z = self.encoder(x)
        y = self.decoder(z, (C, Fr, T))
        return y


def _run_test(label, n_channels, freq, time, d_model, device):
    print(f"=== {label} ===")
    autoencoder = SpectrogramBaselineAutoEncoder(n_channels, d_model)
    autoencoder.to(device)
    x = torch.randn(2, n_channels, freq, time)

    with torch.inference_mode():
        y = autoencoder(x.to(device))
    assert y.shape == x.shape, f"Shape mismatch: {y.shape} vs {x.shape}"

    with torch.inference_mode():
        z = autoencoder.encoder(x.to(device))
    z = z.cpu().detach()

    input_size = n_channels * freq * time
    latent_size = z.numel()
    ratio = input_size / latent_size

    print(f"  Input:   {x.shape}  ({input_size:,} values)")
    print(f"  Latent:  {list(z.shape)}  ({latent_size:,} values)")
    print(f"  Output:  {y.shape}")
    print(f"  Compression: {ratio:.1f}:1")


if __name__ == "__main__":
    # python -m tokamak_foundation_model.models.modality.spectrogram_baseline

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- MHR ---
    _run_test("MHR (8ch)", n_channels=8, freq=513, time=977, d_model=64, device=device)

    # --- CO2 ---
    _run_test("CO2 (4ch)", n_channels=4, freq=513, time=977, d_model=64, device=device)

    # --- ECE ---
    _run_test("ECE (48ch)", n_channels=48, freq=513, time=977, d_model=64, device=device)
