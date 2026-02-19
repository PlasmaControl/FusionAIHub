import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ModalityEncoder, ModalityDecoder, ModalityAutoEncoder


class Conv3dEncoderBlock(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class Conv3dDecoderBlock(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        activate=True,
    ):
        super().__init__()
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
        ]
        if activate:
            layers += [nn.BatchNorm3d(out_channels), nn.GELU()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class TemporalLSTM(nn.Module):
    def __init__(self, 
        channels: int,
        num_layers: int = 1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            channels,
            channels // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        B, C, D, H, T = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B * D * H, T, C)
        x, _ = self.lstm(x)
        x = x.reshape(B, D, H, T, C).permute(0, 4, 1, 2, 3)
        return x

def _build_channel_dims(d_model: int, n_layers: int, base_channels: int = 32) -> list[int]:
    dims = [1]
    if n_layers <= 1:
        dims.append(d_model)
        return dims
    for i in range(n_layers - 1):
        t = i / (n_layers - 2) if n_layers > 2 else 1.0
        ch = int(round(base_channels * (d_model / base_channels) ** t))
        ch = max(8, (ch + 3) // 8 * 8)
        dims.append(ch)
    dims.append(d_model)
    return dims


class SpectrogramBaselineEncoder(ModalityEncoder):
    def __init__(self,
        n_channels: int,
        d_model: int = 256,
        n_output_tokens: int = 0,
        n_layers: int = 4,
        base_channels: int = 32,
        kernel_size: tuple[int, int, int] = (5, 13, 5),
        stride: tuple[int, int, int] = (1, 2, 2),
        lstm_on: bool = False,
    ):
        super().__init__(n_channels, d_model, n_output_tokens)
        self.n_layers = n_layers
        self.stride = stride
        dims = _build_channel_dims(d_model, n_layers, base_channels)
        padding = tuple(k // 2 for k in kernel_size)

        self.blocks = nn.ModuleList()
        for i in range(n_layers - 1):
            self.blocks.append(Conv3dEncoderBlock(
                dims[i], dims[i + 1], kernel_size=kernel_size, stride=stride, padding=padding,
            ))
        # Final conv with stride
        self.blocks.append(Conv3dEncoderBlock(
            dims[-2], dims[-1], kernel_size=kernel_size, stride=stride, padding=padding,
        ))

        self.lstm_on = lstm_on
        if lstm_on:
            self.lstm = TemporalLSTM(dims[-1], num_layers=1)
            self.lstm_conv = Conv3dEncoderBlock(
                dims[-1], dims[-1], kernel_size=kernel_size, stride=1, padding=padding,
            )

    def forward(self, x):
        B, C, Fr, T = x.shape
        x = x.unsqueeze(1)  # (B, 1, C, Fr, T)
        for block in self.blocks:
            x = block(x)
        if self.lstm_on:
            x = self.lstm(x)
            x = self.lstm_conv(x)
        return x


class SpectrogramBaselineDecoder(ModalityDecoder):
    def __init__(self,
        n_channels: int,
        d_model: int = 256,
        n_layers: int = 4,
        base_channels: int = 32,
        kernel_size: tuple[int, int, int] = (5, 13, 5),
        stride: tuple[int, int, int] = (1, 2, 2),
        lstm_on: bool = False,
    ):
        super().__init__(n_channels, d_model)
        self.n_layers = n_layers
        dims = _build_channel_dims(d_model, n_layers, base_channels)
        padding = tuple(k // 2 for k in kernel_size)
        upsample_scale = tuple(float(s) for s in stride)

        self.lstm_on = lstm_on
        if lstm_on:
            self.lstm_conv = Conv3dDecoderBlock(
                dims[-1], dims[-1], kernel_size=kernel_size, stride=1, padding=padding,
            )
            self.lstm = TemporalLSTM(dims[-1], num_layers=1)

        # Decoder mirrors encoder in reverse
        self.upsample_blocks = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()

        for i in range(n_layers - 1, -1, -1):
            in_ch = dims[i + 1]
            out_ch = dims[i]
            is_last = (i == 0)

            self.upsample_blocks.append(
                nn.Upsample(scale_factor=upsample_scale, mode="trilinear", align_corners=False)
            )
            self.conv_blocks.append(Conv3dDecoderBlock(
                in_ch, out_ch, kernel_size=kernel_size, stride=1, padding=padding,
                activate=not is_last,
            ))

    def forward(self, z, output_shape=None):
        y = z
        if self.lstm_on:
            y = self.lstm_conv(y)
            y = self.lstm(y)
        for upsample, conv in zip(self.upsample_blocks, self.conv_blocks):
            y = upsample(y)
            y = conv(y)

        if output_shape is not None:
            y = F.interpolate(y, size=output_shape, mode="trilinear", align_corners=False)
        y = y.squeeze(1)
        return y


class SpectrogramBaselineAutoEncoder(ModalityAutoEncoder):
    """
    Based on 3DCAE implementation at
    https://github.com/faadi809/HSI-compression-benchmark

    Added LSTM based on ENCODEC.

    Args:
        n_channels: Number of input signal channels.
        d_model: Latent channel dimension.
        n_output_tokens: Number of output tokens. TODO: Implement.
        n_layers: Number of encoder/decoder stages.
        base_channels: Starting channel width after first conv.
    """

    def __init__(self,
        n_channels: int,
        d_model: int = 256,
        n_output_tokens: int = 0,
        n_layers: int = 4,
        base_channels: int = 32,
        kernel_size: tuple[int, int, int] = (3, 5, 5),
        stride: tuple[int, int, int] = (1, 2, 2),
        lstm_on: bool = False,
    ):
        super().__init__(n_channels, d_model, n_output_tokens)
        self.n_channels = n_channels
        self.d_model = d_model

        self.encoder = SpectrogramBaselineEncoder(
            n_channels, d_model, n_output_tokens,
            n_layers=n_layers, base_channels=base_channels,
            kernel_size=kernel_size, stride=stride,
            lstm_on=lstm_on,
        )
        self.decoder = SpectrogramBaselineDecoder(
            n_channels, d_model,
            n_layers=n_layers, base_channels=base_channels,
            kernel_size=kernel_size, stride=stride,
            lstm_on=lstm_on,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, Fr, T = x.shape
        z = self.encoder(x)
        y = self.decoder(z, (C, Fr, T))
        return y


def _run_test(label, n_channels, freq, time, d_model, n_layers, lstm_on, device):
    print(f"=== {label} (n_layers={n_layers}) ===")
    autoencoder = SpectrogramBaselineAutoEncoder(
        n_channels, d_model, n_layers=n_layers, lstm_on=lstm_on,
    )
    autoencoder.to(device)

    n_params = sum(p.numel() for p in autoencoder.parameters())
    print(f"  Parameters: {n_params:,}")

    x = torch.randn(1, n_channels, freq, time)

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
    print()


if __name__ == "__main__":
    # python -m tokamak_foundation_model.models.modality.spectrogram_baseline

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _run_test(
        "CO2", 
        n_channels=4, 
        freq=128, 
        time=256, 
        d_model=64, 
        n_layers=6, 
        lstm_on=True,
        device=device,
    )