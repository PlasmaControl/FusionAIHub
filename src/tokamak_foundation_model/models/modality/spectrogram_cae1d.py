import math
import torch.nn.functional as f

from torch import nn


def cae1d_cr4(src_channels=103):
    return ModifiedConvolutionalAutoencoder1D(src_channels=src_channels, target_bpppc=8)

def cae1d_cr8(src_channels=103):
    return ModifiedConvolutionalAutoencoder1D(src_channels=src_channels, target_bpppc=4)

def cae1d_cr16(src_channels=103):
    return ModifiedConvolutionalAutoencoder1D(src_channels=src_channels, target_bpppc=2)

def cae1d_cr32(src_channels=103):
    return ModifiedConvolutionalAutoencoder1D(src_channels=src_channels, target_bpppc=1)

def cae1d_cr114(src_channels=103):
    return ModifiedConvolutionalAutoencoder1D(src_channels=src_channels, target_bpppc=32/134)

def cae1d_cr124(src_channels=103):
    return ModifiedConvolutionalAutoencoder1D(src_channels=src_channels, target_bpppc=64/134)

def cae1d_cr134(src_channels=103):
    return ModifiedConvolutionalAutoencoder1D(src_channels=src_channels, target_bpppc=100/134)

def cae1d_cr144(src_channels=103):
    return ModifiedConvolutionalAutoencoder1D(src_channels=src_channels, target_bpppc=81/134)


class ModifiedConvolutionalAutoencoder1D(nn.Module):
    """
    Comment:
        Modified version of the below paper to target multiple bitrates.
    Title:
        1D-CONVOLUTIONAL AUTOENCODER BASED HYPERSPECTRAL DATA COMPRESSION
    Authors:
        Kuester, Jannick and Gross, Wolfgang and Middelmann, Wolfgang
    Paper:
        https://doi.org/10.5194/isprs-archives-XLIII-B1-2021-15-2021
    Cite:
        @article{kuester20211d,
            title={1D-convolutional autoencoder based hyperspectral data compression},
            author={Kuester, Jannick and Gross, Wolfgang and Middelmann, Wolfgang},
            journal={International Archives of Photogrammetry, Remote Sensing and Spatial Information Sciences},
            volume={43},
            pages={15--21},
            year={2021},
            publisher={Copernicus GmbH}
        }
    """

    def __init__(self, src_channels=202, target_bpppc=8):
        super().__init__()

        self.target_bpppc = target_bpppc
        self.compression_ratio = 32.0 / target_bpppc
        self.num_blocks = max(1, int(round(math.log2(self.compression_ratio))))

        max_possible_blocks = int(math.log2(src_channels))
        self.num_blocks = min(self.num_blocks, max_possible_blocks)

        # Calculate actual achieved compression
        self.spectral_downsampling_factor_estimated = 2 ** self.num_blocks
        self.actual_bpppc = 32.0 / self.spectral_downsampling_factor_estimated
        print(f"Target bpppc: {target_bpppc:.4f}, Actual achieved: {self.actual_bpppc:.4f}")

        self.encoder = nn.Sequential(
            nn.Sequential(*[
                nn.Sequential(*[
                    nn.Conv1d(
                        in_channels=1 if i==0 else int(2 ** (self.num_blocks + 5 - i)),
                        out_channels=int(2 ** (self.num_blocks + 4 - i)),
                        kernel_size=11,
                        stride=1,
                        padding="same",
                    ),
                    nn.LeakyReLU(),
                    nn.MaxPool1d(kernel_size=2),
                ])
                for i in range(self.num_blocks)
            ]),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=9, stride=1, padding="same"),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=16, out_channels=1, kernel_size=7, stride=1, padding="same"),
            nn.LeakyReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, stride=1, padding="same"),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=9, stride=1, padding="same"),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2),
            nn.Sequential(*[
                nn.Sequential(*[
                    nn.Conv1d(
                        in_channels=int(2 ** (5 + i)),
                        out_channels=int(2 ** (6 + i)) if i < self.num_blocks - 1 else 1,
                        kernel_size=11,
                        stride=1,
                        padding="same",
                    ),
                    nn.LeakyReLU() if i < self.num_blocks - 1 else nn.Sigmoid(),
                    nn.Upsample(
                        scale_factor=2
                    ) if i < self.num_blocks - 1 else nn.Identity(),
                ])
                for i in range(self.num_blocks)
            ]),
        )

        self.src_channels = src_channels

        self.spectral_downsamplings = self.num_blocks
        self.spectral_downsampling_factor_estimated = 2 ** self.spectral_downsamplings

        self.spatial_downsamplings = 0
        self.spatial_downsampling_factor = 2 ** self.spatial_downsamplings

        self.latent_channels = int(math.ceil(self.src_channels / 2 ** self.spectral_downsamplings))
        self.spectral_downsampling_factor = self.src_channels / self.latent_channels
        self.compression_ratio = self.spectral_downsampling_factor * self.spatial_downsampling_factor ** 2
        self.bpppc = 32.0 / self.compression_ratio

        self.padding_amount = 0 if self.src_channels % self.spectral_downsampling_factor_estimated == 0 \
            else self.spectral_downsampling_factor_estimated - self.src_channels % self.spectral_downsampling_factor_estimated

    def forward(self, x):
        n, c, h, w = x.shape

        x = x.permute(0, 2, 3, 1).reshape(-1, c)
        if self.padding_amount > 0:
            x = f.pad(x, (self.padding_amount, 0))
        x = x.unsqueeze(1)

        y = self.encoder(x)
        x_hat = self.decoder(y)

        if self.padding_amount > 0:
            x_hat = x_hat[:, :, self.padding_amount:]
        x_hat = x_hat.squeeze(1)
        x_hat = x_hat.reshape(n, h, w, c).permute(0, 3, 1, 2)

        return x_hat

    def compress(self, x):
        n, c, h, w = x.shape
        
        x = x.permute(0, 2, 3, 1).reshape(-1, c)
        if self.padding_amount > 0:
            x = f.pad(x, (self.padding_amount, 0))
        x = x.unsqueeze(1)
        
        y = self.encoder(x)
        y = y.squeeze(1)
        y = y.reshape(n, h, w, -1).permute(0, 3, 1, 2)

        return y

    def decompress(self, y):
        n, c, h, w = y.shape

        y = y.permute(0, 2, 3, 1).reshape(-1, c)
        y = y.unsqueeze(1)
        x_hat = self.decoder(y)

        if self.padding_amount > 0:
            x_hat = x_hat[:, :, self.padding_amount:]
        x_hat = x_hat.squeeze(1)
        x_hat = x_hat.reshape(n, h, w, -1).permute(0, 3, 1, 2)

        return x_hat

    @classmethod
    def from_state_dict(cls, state_dict):
        net = cls()
        net.load_state_dict(state_dict)
        return net


if __name__ == '__main__':
    # python -m src.tokamak_foundation_model.models.modality.spectrogram_cae1d
    import torch
    from torchinfo import summary

    model = ModifiedConvolutionalAutoencoder1D()
    print(model)

    summary(model, input_size=(2, 202, 128, 128), device='cpu')

    in_tensor = torch.randn(1, 202, 128, 128)
    print("in shape:\t\t", in_tensor.shape)

    latent_tensor = model.compress(in_tensor)
    print("latent shape:\t\t", latent_tensor.shape)
    
    out_tensor = model(in_tensor)
    print("out shape:\t\t", out_tensor.shape)

    print("in shape = out shape:\t", out_tensor.shape == in_tensor.shape)

    print("real bpppc:\t\t", 32 * torch.numel(latent_tensor) / torch.numel(in_tensor))
    print("model parameter bpppc:\t", model.bpppc)