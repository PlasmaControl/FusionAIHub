import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True):
        super(ResidualBlock, self).__init__()
        if isinstance(kernel_size, tuple):
            padding = tuple(ks // 2 for ks in kernel_size)
        else:
            padding = kernel_size // 2

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                               padding=padding, bias=bias)
        self.batch_norm_1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
                               padding=padding, bias=bias)
        self.batch_norm_2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                       padding=0, bias=bias)
        else:
            self.skip_conv = None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.batch_norm_1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.batch_norm_2(out)
        if self.skip_conv is not None:
            residual = self.skip_conv(residual)
        out += residual
        out = self.relu(out)
        return out


class Encoder(nn.Module):
    def __init__(self, input_channels, kernel_size=3, bias=True, dropout=0.1):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            ResidualBlock(in_channels=input_channels, out_channels=128,
                          kernel_size=kernel_size, bias=bias),
            nn.Dropout(p=dropout),
            nn.MaxPool2d(kernel_size=(3, 2), stride=(1, 2), padding=(3 // 2, 0)),

            ResidualBlock(in_channels=128, out_channels=256,
                          kernel_size=kernel_size, bias=bias),
            nn.Dropout(p=dropout),
            nn.MaxPool2d(kernel_size=(3, 2), stride=(1, 2), padding=(3 // 2, 0)),

            ResidualBlock(in_channels=256, out_channels=256,
                          kernel_size=kernel_size, bias=bias),
            nn.Dropout(p=dropout),
            nn.MaxPool2d(kernel_size=(3, 2), stride=(1, 2), padding=(3 // 2, 0)),

            ResidualBlock(in_channels=256, out_channels=128,
                          kernel_size=kernel_size, bias=bias),
            nn.Dropout(p=dropout),
            nn.MaxPool2d(kernel_size=(3, 2), stride=(1, 2), padding=(3 // 2, 0)),

            ResidualBlock(in_channels=128, out_channels=input_channels,
                          kernel_size=kernel_size, bias=bias),
            nn.Dropout(p=dropout),
            nn.MaxPool2d(kernel_size=(3, 2), stride=(1, 2), padding=(3 // 2, 0)),
        )

    def forward(self, x):
        return self.encoder(x)


if __name__ == "__main__":
    # python -m tokamak_foundation_model.models.modality.cer_model
    encoder = Encoder(input_channels=80, kernel_size=3, bias=True, dropout=0.1)
    x = torch.randn(2, 80, 256, 530)
    with torch.inference_mode():
        y = encoder(x)
    print(y.shape)

    print(f"Compression ratio: {x.numel() / y.numel()}")