import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectrogramAutoEncoder(nn.Module):

    def __init__(self, n_channels: int, d_model: int = 256):
        super().__init__()
        self.n_channels = n_channels
        self.d_model = d_model

        dims = [1, 32, 64, 128, d_model]

        self.encoder = nn.Sequential(
            nn.Conv3d(dims[0], dims[1], kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(dims[1], dims[2], kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.GELU(),
            nn.Conv3d(dims[2], dims[3], kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv3d(dims[3], dims[4], kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )

        self.decoder = nn.Sequential(
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

        n_params = sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, Fr, T = x.shape

        x = x.unsqueeze(1)                     # [B, 1, C, F, T]
        z = self.encoder(x)                     # [B, d_model, C', F', T']
        x = self.decoder(z)                     # [B, 1, ~C, ~F, ~T]

        x = F.interpolate(
            x, size=(C, Fr, T), mode="trilinear", align_corners=False
        )

        return x.squeeze(1)                     # [B, C, F, T]


def _run_test(label, n_ch, freq, time, d_model):
    print(f"=== {label} ===")
    ae = SpectrogramAutoEncoder(n_ch, d_model)
    x = torch.randn(1, n_ch, freq, time)
    y = ae(x)
    assert y.shape == x.shape, f"Shape mismatch: {y.shape} vs {x.shape}"

    # Peek at encoder output shape
    with torch.no_grad():
        z = ae.encoder(x.unsqueeze(1))

    input_size = n_ch * freq * time
    latent_size = z.numel()
    ratio = input_size / latent_size

    print(f"  Input:   {x.shape}  ({input_size:,} values)")
    print(f"  Latent:  {list(z.shape)}  ({latent_size:,} values)")
    print(f"  Output:  {y.shape}")
    print(f"  Compression: {ratio:.1f}:1")


if __name__ == "__main__":
    # python -m tokamak_foundation_model.models.modality.spectrogram_baseline

    # --- MHR ---
    _run_test("MHR (8ch)", n_ch=8, freq=513, time=977, d_model=256)

    # --- CO2 ---
    _run_test("CO2 (4ch)", n_ch=4, freq=513, time=977, d_model=256)

    # --- ECE ---
    _run_test("ECE (48ch)", n_ch=48, freq=513, time=977, d_model=256)
