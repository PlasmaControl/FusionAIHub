import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ModalityEncoder, ModalityDecoder, ModalityAutoEncoder


class ResBlock3d(nn.Module):
    def __init__(self, channels, bottleneck=32):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, bottleneck, kernel_size=1),       # squeeze
            nn.BatchNorm3d(bottleneck),
            nn.GELU(),
            nn.Conv3d(bottleneck, bottleneck, kernel_size=3, padding=1),  # cheap 3x3
            nn.BatchNorm3d(bottleneck),
            nn.GELU(),
            nn.Conv3d(bottleneck, channels, kernel_size=1),       # expand
            nn.BatchNorm3d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class TemporalLSTM(nn.Module):
    """LSTM along the time dimension of a 5D tensor (B, C, D, H, T)."""
    def __init__(self, channels: int, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(channels, channels, num_layers=num_layers, batch_first=True)

    def forward(self, x):
        B, C, D, H, T = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B * D * H, T, C)
        x, _ = self.lstm(x)
        x = x.reshape(B, D, H, T, C).permute(0, 4, 1, 2, 3)
        return x


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
            nn.BatchNorm3d(dims[1]),
            nn.GELU(),
            nn.Conv3d(dims[1], dims[2], kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(dims[2]),
            nn.GELU(),
            nn.Conv3d(dims[2], dims[3], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(dims[3]),
            nn.GELU(),
            ResBlock3d(dims[3]),
            TemporalLSTM(dims[3]),
            nn.Conv3d(dims[3], dims[4], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(dims[4]),
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
            nn.BatchNorm3d(dims[3]),
            nn.GELU(),
            TemporalLSTM(dims[3]),
            ResBlock3d(dims[3]),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(dims[3], dims[2], kernel_size=3, padding=1),
            nn.BatchNorm3d(dims[2]),
            nn.GELU(),
            nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear", align_corners=False),
            nn.Conv3d(dims[2], dims[1], kernel_size=3, padding=1),
            nn.BatchNorm3d(dims[1]),
            nn.GELU(),
            nn.Conv3d(dims[1], dims[0], kernel_size=3, padding=1),
        )

    def forward(self, z, output_shape=None):
        y = self.net(z)
        if output_shape is not None:
            y = F.interpolate(
                y, size=output_shape, mode="trilinear", align_corners=False
            )
        y = y.squeeze(1)
        return y

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
    _run_test("MHR (8ch)", n_channels=8, freq=513, time=977, d_model=32, device=device)

    # --- CO2 ---
    _run_test("CO2 (4ch)", n_channels=4, freq=513, time=977, d_model=32, device=device)

    # --- ECE ---
    _run_test("ECE (48ch)", n_channels=48, freq=513, time=977, d_model=32, device=device)
