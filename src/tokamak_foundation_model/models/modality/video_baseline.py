import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import ModalityEncoder, ModalityDecoder


class VideoBaselineEncoder(nn.Module):
    def __init__(self, in_channels=1, n_tokens=8, token_dim=512):
        super().__init__()
        self.n_tokens = n_tokens
        self.token_dim = token_dim

        self.net = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv3d(32, 64, 3, stride=(1,2,2), padding=1), nn.ReLU(),
            nn.Conv3d(64, 128, 3, stride=(1,2,2), padding=1), nn.ReLU(),
            nn.Conv3d(128, 256, 3, stride=(1,2,2), padding=1), nn.ReLU(),
            nn.Conv3d(256, token_dim, 1), nn.ReLU(),
            nn.AdaptiveAvgPool3d((n_tokens, 1, 1)),  # <-- THIS must be n_tokens
        )

    def forward(self, x):
        # x: (B,T,H,W) -> (B,1,T,H,W)
        y = self.net(x.unsqueeze(1))                  # (B,512,N,1,1)
        z = y.squeeze(-1).squeeze(-1).permute(0,2,1)  # (B,N,512)
        return z


class VideoBaselineDecoder(nn.Module):
    """
    Input:  z (B, N, 512)
    Output: x_hat (B, T, H, W)
    """
    def __init__(self, out_channels: int = 1, n_tokens: int = 8, token_dim: int = 512,
                 target_size=(25, 256, 256)):
        super().__init__()
        self.target_size = target_size

        self.net = nn.Sequential(
            nn.ConvTranspose3d(token_dim, 256, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.ConvTranspose3d(256, 128, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.ConvTranspose3d(128, 64, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(32, out_channels, kernel_size=3, padding=1),
        )
        self.refine = nn.Sequential(
            nn.Upsample(scale_factor=(1,2,2), mode="trilinear", align_corners=False),
            nn.Conv3d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=(1,2,2), mode="trilinear", align_corners=False),
            nn.Conv3d(16, 16, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=(1,2,2), mode="trilinear", align_corners=False),
            nn.Conv3d(16, 16, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=(1,2,2), mode="trilinear", align_corners=False),
            nn.Conv3d(16, 16, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=(1,2,2), mode="trilinear", align_corners=False),
            nn.Conv3d(16, 1, 3, padding=1),
        )
        self.resample = nn.AdaptiveAvgPool3d(target_size)

    def forward(self, z):
        y = z.permute(0,2,1).unsqueeze(-1).unsqueeze(-1)
        x = self.net(y)
        x = self.refine(x)   # (B,1,N,256,256)
        x = torch.tanh(x)
        x = F.interpolate(x, size=self.target_size, mode="trilinear", align_corners=False)
        return x.squeeze(1)


class VideoBaselineAutoEncoder(nn.Module):
    def __init__(self, n_tokens: int, target_size=(25, 256, 256), token_dim: int = 512):
        super().__init__()
        self.encoder = VideoBaselineEncoder(n_tokens=n_tokens, token_dim=token_dim)
        self.decoder = VideoBaselineDecoder(n_tokens=n_tokens, token_dim=token_dim, target_size=target_size)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
    
    # def encode(self, x):
    #     z = self.encoder(x)
    #     return z
    
    # def decode(self, z):
    #     x_hat = self.decoder(z)
    #     return x_hat