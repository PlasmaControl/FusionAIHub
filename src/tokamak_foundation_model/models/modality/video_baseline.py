import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import ModalityEncoder, ModalityDecoder
from typing import Optional


# class VideoEncoder(nn.Module):
#     def __init__(self, in_channels=1, n_tokens=8, token_dim=512):
#         super().__init__()
#         self.n_tokens = n_tokens
#         self.token_dim = token_dim

#         self.net = nn.Sequential(
#             nn.Conv3d(in_channels, 32, 3, padding=1), nn.ReLU(),
#             nn.Conv3d(32, 64, 3, stride=(1,2,2), padding=1), nn.ReLU(),
#             nn.Conv3d(64, 128, 3, stride=(1,2,2), padding=1), nn.ReLU(),
#             nn.Conv3d(128, 256, 3, stride=(1,2,2), padding=1), nn.ReLU(),
#             nn.Conv3d(256, token_dim, 1), nn.ReLU(),
#             nn.AdaptiveAvgPool3d((n_tokens, 1, 1)),  # <-- THIS must be n_tokens
#         )

#     def forward(self, x):
#         # x: (B,T,H,W) -> (B,1,T,H,W)
#         y = self.net(x.unsqueeze(1))                  # (B,512,N,1,1)
#         z = y.squeeze(-1).squeeze(-1).permute(0,2,1)  # (B,N,512)
#         return z


# class VideoDecoder(nn.Module):
#     """
#     Input:  z (B, N, 512)
#     Output: x_hat (B, T, H, W)
#     """
#     def __init__(self, out_channels: int = 1, n_tokens: int = 8, token_dim: int = 512,
#                  target_size=(25, 256, 256)):
#         super().__init__()
#         self.target_size = target_size

#         self.net = nn.Sequential(
#             nn.ConvTranspose3d(token_dim, 256, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
#             nn.ReLU(),
#             nn.ConvTranspose3d(256, 128, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
#             nn.ReLU(),
#             nn.ConvTranspose3d(128, 64, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
#             nn.ReLU(),
#             nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose3d(32, out_channels, kernel_size=3, padding=1),
#         )
#         self.refine = nn.Sequential(
#             nn.Upsample(scale_factor=(1,2,2), mode="trilinear", align_corners=False),
#             nn.Conv3d(1, 16, 3, padding=1), nn.ReLU(),
#             nn.Upsample(scale_factor=(1,2,2), mode="trilinear", align_corners=False),
#             nn.Conv3d(16, 16, 3, padding=1), nn.ReLU(),
#             nn.Upsample(scale_factor=(1,2,2), mode="trilinear", align_corners=False),
#             nn.Conv3d(16, 16, 3, padding=1), nn.ReLU(),
#             nn.Upsample(scale_factor=(1,2,2), mode="trilinear", align_corners=False),
#             nn.Conv3d(16, 16, 3, padding=1), nn.ReLU(),
#             nn.Upsample(scale_factor=(1,2,2), mode="trilinear", align_corners=False),
#             nn.Conv3d(16, 1, 3, padding=1),
#         )
#         self.resample = nn.AdaptiveAvgPool3d(target_size)

#     def forward(self, z):
#         y = z.permute(0,2,1).unsqueeze(-1).unsqueeze(-1)
#         x = self.net(y)
#         x = self.refine(x)   # (B,1,N,256,256)
#         x = torch.tanh(x)
#         x = F.interpolate(x, size=self.target_size, mode="trilinear", align_corners=False)
#         return x.squeeze(1)


# class VideoAutoEncoder(nn.Module):
#     def __init__(self, n_tokens: int, target_size=(25, 256, 256), token_dim: int = 512):
#         super().__init__()
#         self.encoder = VideoEncoder(n_tokens=n_tokens, token_dim=token_dim)
#         self.decoder = VideoDecoder(n_tokens=n_tokens, token_dim=token_dim, target_size=target_size)

#     def forward(self, x):
#         z = self.encoder(x)
#         x_hat = self.decoder(z)
#         return x_hat, z

#     def encode(self, x):
#         z = self.encoder(x)
#         return z

#     def decode(self, z):
#         x_hat = self.decoder(z)
#         return x_hat


class VideoEncoder(nn.Module):
    """
    Input:  x (B, T, H, W)  grayscale
    Output: z_tokens (B, N, 512)
    Also returns z_vec (B, N*512) for decoding.
    """

    def __init__(
        self,
        n_tokens: int,
        token_dim: int = 512,
        t_chunk: int = 25,
        img_size: int = 256,
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self.token_dim = token_dim
        self.latent_dim = n_tokens * token_dim

        # Attached-style: stride-2 conv stack + BN + ReLU
        self.enc = nn.Sequential(
            nn.Conv3d(1, 16, 3, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
        )

        # Infer flatten dim once (keeps your structure clean in notebook)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, t_chunk, img_size, img_size)
            h = self.enc(dummy)
            self._enc_shape = h.shape  # (1, C0, T0, H0, W0)
            flat_dim = h.flatten(1).shape[1]

        self.fc = nn.Linear(flat_dim, self.latent_dim)

    def forward(self, x: torch.Tensor):
        # x: (B,T,H,W) -> (B,1,T,H,W)
        h = self.enc(x.unsqueeze(1))
        z_vec = self.fc(h.flatten(1))  # (B, N*512)
        z_tokens = z_vec.view(x.shape[0], self.n_tokens, self.token_dim)  # (B,N,512)
        return z_tokens, z_vec


class VideoDecoder(nn.Module):
    """
    Input:  z_tokens (B, N, 512)  OR z_vec (B, N*512)
    Output: x_hat (B, T, H, W)
    """

    def __init__(
        self,
        n_tokens: int,
        token_dim: int = 512,
        t_chunk: int = 25,
        img_size: int = 256,
        enc_shape=(1, 256, 1, 8, 8),  # will be overwritten by encoder-provided shape
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self.token_dim = token_dim
        self.latent_dim = n_tokens * token_dim
        self.t_chunk = t_chunk
        self.img_size = img_size

        # Use encoder's conv output shape to reshape back
        _, C0, T0, H0, W0 = enc_shape
        self.C0, self.T0, self.H0, self.W0 = C0, T0, H0, W0

        self.fc = nn.Linear(self.latent_dim, C0 * T0 * H0 * W0)

        # Attached-style: ConvTranspose3d + BN + ReLU, final conv to 1 channel
        self.dec = nn.Sequential(
            nn.ConvTranspose3d(C0, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(16, 1, 3, stride=2, padding=1, output_padding=1),
        )

    def forward(
        self, z_tokens: torch.Tensor, z_vec: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Accept either z_tokens or z_vec
        if z_vec is None:
            B = z_tokens.shape[0]
            z_vec = z_tokens.reshape(B, self.latent_dim)  # (B, N*512)

        x = self.fc(z_vec).view(
            -1, self.C0, self.T0, self.H0, self.W0
        )  # (B,C0,T0,H0,W0)
        x = self.dec(x)  # (B,1,T',H',W')

        # Force exact output size (like the attached code typically does)
        x = F.interpolate(
            x,
            size=(self.t_chunk, self.img_size, self.img_size),
            mode="trilinear",
            align_corners=False,
        )

        # If your input is normalized to [0,1], keep sigmoid:
        x = torch.sigmoid(x)

        return x.squeeze(1)  # (B,T,H,W)


class VideoAutoEncoder(nn.Module):
    def __init__(
        self,
        n_tokens: int,
        t_chunk: int = 25,
        img_size: int = 256,
        token_dim: int = 512,
    ):
        super().__init__()
        self.encoder = VideoEncoder(
            n_tokens=n_tokens, token_dim=token_dim, t_chunk=t_chunk, img_size=img_size
        )

        # Build decoder using encoder's inferred shape
        self.decoder = VideoDecoder(
            n_tokens=n_tokens,
            token_dim=token_dim,
            t_chunk=t_chunk,
            img_size=img_size,
            enc_shape=self.encoder._enc_shape,
        )

    def forward(self, x: torch.Tensor):
        z_tokens, z_vec = self.encoder(x)
        x_hat = self.decoder(z_tokens, z_vec=z_vec)
        return x_hat, z_tokens