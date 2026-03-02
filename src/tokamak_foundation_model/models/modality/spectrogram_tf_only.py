import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ResidualBlock(nn.Module):
    """Conv2d residual block with optional GroupNorm."""

    DEFAULT_GROUPS = 32

    def __init__(self, in_channels, out_channels=None, use_groupnorm=False):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels

        if use_groupnorm:
            norm_layer = lambda c: nn.GroupNorm(
                num_groups=min(self.DEFAULT_GROUPS, c), num_channels=c
            )
        else:
            norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = norm_layer(out_channels)
        self.activation = nn.GELU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = norm_layer(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                norm_layer(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.activation(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.activation(out + residual)
        return out


class LSTMBlock(nn.Module):
    """Bidirectional LSTM operating across the time axis of a 2D feature map."""

    def __init__(self, channels, freq_dim, hidden_dim=128, num_layers=1):
        super().__init__()
        self.channels = channels
        input_dim = channels * freq_dim

        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True, bidirectional=True,
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, input_dim),
            nn.GELU(),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.norm = nn.BatchNorm2d(channels)
        self.freq_dim = freq_dim

    def forward(self, x):
        B, C, F, T = x.shape
        residual = x

        x_seq = rearrange(x, 'b c f t -> b t (c f)')
        lstm_out, _ = self.lstm(x_seq)
        proj_out = self.proj(lstm_out)
        x_back = rearrange(proj_out, 'b t (c f) -> b c f t', c=C, f=F)

        x_back = self.conv(x_back)
        out = self.norm(x_back + residual)
        return out


class Encoder(nn.Module):
    def __init__(self, in_channels=1, dims=None, latent_channels=16,
                 freq_dim=16, lstm_hidden=128, lstm_layers=1, lstm_on=True):
        super().__init__()
        if dims is None:
            dims = [64, 128, 256]
        self.lstm_on = lstm_on

        layers = []
        c = in_channels
        for d in dims:
            layers.append(ResidualBlock(c, d))
            layers.append(nn.Conv2d(d, d, kernel_size=3, stride=(2, 2), padding=1, bias=False))
            c = d

        self.net = nn.Sequential(*layers)
        self.to_latent = nn.Conv2d(dims[-1], latent_channels, 1)

        if self.lstm_on:
            self.lstm_block = LSTMBlock(
                channels=latent_channels, freq_dim=freq_dim,
                hidden_dim=lstm_hidden, num_layers=lstm_layers,
            )

    def forward(self, x):
        z = self.to_latent(self.net(x))
        if self.lstm_on:
            z = self.lstm_block(z)
        return z


class Decoder(nn.Module):
    def __init__(self, out_channels=1, dims=None, latent_channels=16,
                 freq_dim=16, lstm_hidden=128, lstm_layers=1, lstm_on=True):
        super().__init__()
        if dims is None:
            dims = [256, 128, 64]
        self.lstm_on = lstm_on

        self.from_latent = nn.Conv2d(latent_channels, dims[0], 1)

        if self.lstm_on:
            self.lstm_block = LSTMBlock(
                channels=dims[0], freq_dim=freq_dim,
                hidden_dim=lstm_hidden, num_layers=lstm_layers,
            )

        layers = []
        c = dims[0]
        for d in dims[1:]:
            layers.append(ResidualBlock(c, d))
            layers.append(nn.Sequential(
                nn.Upsample(scale_factor=(2, 2), mode='nearest'),
                nn.Conv2d(d, d, kernel_size=3, padding=1, bias=False),
            ))
            c = d

        layers.append(ResidualBlock(c, out_channels))
        layers.append(nn.Sequential(
            nn.Upsample(scale_factor=(2, 2), mode='nearest'),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        ))
        self.net = nn.Sequential(*layers)
        self.head = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, z, output_dim=None):
        y = self.from_latent(z)
        if self.lstm_on:
            y = self.lstm_block(y)
        y = self.net(y)
        y = self.head(y)
        if output_dim is not None and y.shape[2:] != torch.Size(output_dim):
            y = F.interpolate(y, size=output_dim, mode='bilinear', align_corners=False)
        return y


class SpectrogramTFOnlyAutoEncoder(nn.Module):
    """Conv2D + BiLSTM channel-independent autoencoder for spectrograms.

    Each channel is processed independently via batch folding (einops rearrange).
    Architecture: ResidualBlock convs with stride-2 downsampling, BiLSTM at
    bottleneck, upsample + ResidualBlock decoder with bilinear interpolation
    to match input dimensions.

    Parameters
    ----------
    n_channels : int
        Number of spectrogram channels (e.g. 8 for MHR, 48 for ECE).
    hidden_dim : int
        Width of conv layers in encoder/decoder.
    latent_dim : int
        Number of latent channels at the bottleneck.
    freq_dim : int
        Frequency dimension at the bottleneck (after 3x stride-2 downsampling).
    lstm_hidden : int
        Hidden size of the bidirectional LSTM.
    lstm_layers : int
        Number of LSTM layers.
    """

    def __init__(self, n_channels=8, hidden_dim=64, latent_dim=2,
                 freq_dim=16, lstm_hidden=32, lstm_layers=1, lstm_on=True, **kwargs):
        super().__init__()
        self.n_channels = n_channels
        self.latent_dim = latent_dim

        self.encoder = Encoder(
            in_channels=1, dims=[hidden_dim, hidden_dim, hidden_dim],
            latent_channels=latent_dim, freq_dim=freq_dim,
            lstm_hidden=lstm_hidden, lstm_layers=lstm_layers, lstm_on=lstm_on,
        )
        self.decoder = Decoder(
            out_channels=1, dims=[hidden_dim, hidden_dim, hidden_dim],
            latent_channels=latent_dim, freq_dim=freq_dim,
            lstm_hidden=lstm_hidden, lstm_layers=lstm_layers, lstm_on=lstm_on,
        )

    def forward(self, x):
        B, C, F, T = x.shape
        x_flat = rearrange(x, 'b c f t -> (b c) 1 f t')

        z = self.encoder(x_flat)
        y_flat = self.decoder(z, output_dim=(F, T))

        y = rearrange(y_flat, '(b c) 1 f t -> b c f t', b=B, c=C)
        z_reshaped = rearrange(z, '(b c) d f t -> b (c d) f t', b=B, c=C)
        return y, z_reshaped


class PatchDiscriminator(nn.Module):
    """PatchGAN-style discriminator for spectrogram data.

    Takes (B, C, Fr, T) input and outputs per-patch logits.
    Not used in default training; groundwork for future GAN loss.
    """

    def __init__(self, n_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, stride=1, padding=1),
        )

    def forward(self, x):
        return self.net(x)


def _run_test(label, n_channels, freq, time, device, **kwargs):
    print(f"=== {label} (n_channels={n_channels}) ===")
    model = SpectrogramTFOnlyAutoEncoder(n_channels=n_channels, **kwargs)
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    x = torch.randn(1, n_channels, freq, time)
    with torch.inference_mode():
        y, z = model(x.to(device))
    y = y.cpu()
    assert y.shape == x.shape, f"Shape mismatch: {y.shape} vs {x.shape}"

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
    # python -m tokamak_foundation_model.models.modality.spectrogram_tf_only
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Notebook baseline (~912K params)
    _run_test("MHR (notebook)", n_channels=8, freq=128, time=391, device=device,
              hidden_dim=64, latent_dim=2, freq_dim=16, lstm_hidden=32, lstm_layers=1)

    # Scaled up (~4.5M params target)
    _run_test("MHR (scaled)", n_channels=8, freq=128, time=391, device=device,
              hidden_dim=128, latent_dim=4, freq_dim=16, lstm_hidden=96, lstm_layers=1)

    # ECE
    _run_test("ECE (notebook)", n_channels=48, freq=128, time=196, device=device,
              hidden_dim=128, latent_dim=4, freq_dim=16, lstm_hidden=96, lstm_layers=1)

    # CO2
    _run_test("CO2 (notebook)", n_channels=4, freq=128, time=196, device=device,
              hidden_dim=128, latent_dim=2, freq_dim=16, lstm_hidden=96, lstm_layers=1)