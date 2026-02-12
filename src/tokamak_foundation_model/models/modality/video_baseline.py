import numpy as np
import torch
import torch.nn as nn
from base import ModalityEncoder, ModalityDecoder


def create_video_test_signal(
    batch_size: int = 4,
    input_frames: int = 50,
    frame_size: int = 256
):
    """
    Create deterministic test video sequences for video encoder/decoder.

    Parameters
    ----------
    batch_size : int, optional
        Number of samples in batch, by default 4
    input_frames : int, optional
        Number of frames per video, by default 50
    frame_size : int, optional
        Height and width of each frame, by default 256

    Returns
    -------
    torch.Tensor
        Test video of shape [batch_size, 1, input_frames, frame_size, frame_size]

    Notes
    -----
    Test patterns per batch:
    - Batch 0: Constant frame (all ones) - tests DC preservation
    - Batch 1: Vertical edge (left half 0, right half 1) - tests spatial edges
    - Batch 2: Single spatial impulse at center - tests spatial localization
    - Batch 3: Temporal flash (single frame lit up) - tests temporal localization
    """
    signal = np.zeros((batch_size, 1, input_frames, frame_size, frame_size))

    if batch_size > 0:
        signal[0, 0, :, :, :] = 1.0

    if batch_size > 1:
        signal[1, 0, :, :, frame_size // 2:] = 1.0

    if batch_size > 2:
        signal[2, 0, :, frame_size // 2, frame_size // 2] = 1.0

    if batch_size > 3:
        signal[3, 0, input_frames // 2, :, :] = 1.0

    return torch.from_numpy(signal).float()


class VideoEncoder(nn.Module):
    """
    Encodes grayscale video sequences using joint 3D convolutions.

    Parameters
    ----------
    input_frames : int, optional
        Number of input frames (e.g., 50 for 500ms @ 100fps), by default 50
    frame_size : int, optional
        Height and width of each frame (assumed square), by default 256
    d_model : int, optional
        Model dimension for transformer, by default 512
    n_output_tokens : int, optional
        Number of output tokens, must equal t_tokens * h_tokens * w_tokens,
        by default 192 (3 * 8 * 8)
    verbose : bool, optional
        If True, print debug information during initialization, by default False

    Attributes
    ----------
    conv_layers : nn.ModuleList
        List of 3D convolutional layers
    t_tokens : int
        Temporal dimension of token grid (3)
    h_tokens : int
        Height dimension of token grid (8)
    w_tokens : int
        Width dimension of token grid (8)
    adaptive_pool : nn.AdaptiveAvgPool3d
        Adaptive pooling to exact token dimensions
    """

    def __init__(
            self,
            input_frames: int = 50,
            frame_size: int = 256,
            d_model: int = 512,
            n_output_tokens: int = 192,
            verbose: bool = False
    ):
        super().__init__()

        self.input_frames = input_frames
        self.frame_size = frame_size
        self.d_model = d_model
        self.n_output_tokens = n_output_tokens
        self.verbose = verbose

        # Token grid: 192 = 3 × 8 × 8
        self.t_tokens = 3
        self.h_tokens = 8
        self.w_tokens = 8

        assert self.t_tokens * self.h_tokens * self.w_tokens == n_output_tokens, (
            f"n_output_tokens ({n_output_tokens}) must equal "
            f"t_tokens * h_tokens * w_tokens "
            f"({self.t_tokens} * {self.h_tokens} * {self.w_tokens})"
        )

        # 3D conv stack:
        # Layers 1-3: spatial stride only (preserve temporal resolution)
        # Layers 4-5: joint stride (compress both space and time)
        self.conv_layers = nn.ModuleList([
            # [B, 1,   50, 256, 256] → [B, 32,  50, 128, 128]
            nn.Conv3d(1,   32,  kernel_size=(3,7,7), stride=(1,2,2), padding=(1,3,3)),
            # [B, 32,  50, 128, 128] → [B, 64,  50,  64,  64]
            nn.Conv3d(32,  64,  kernel_size=(3,5,5), stride=(1,2,2), padding=(1,2,2)),
            # [B, 64,  50,  64,  64] → [B, 128, 50,  32,  32]
            nn.Conv3d(64,  128, kernel_size=(3,5,5), stride=(1,2,2), padding=(1,2,2)),
            # [B, 128, 50,  32,  32] → [B, 256, 25,  16,  16]
            nn.Conv3d(128, 256, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1)),
            # [B, 256, 25,  16,  16] → [B, d_model, 12, 8, 8]
            nn.Conv3d(256, d_model, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1)),
        ])

        self.adaptive_pool = nn.AdaptiveAvgPool3d(
            (self.t_tokens, self.h_tokens, self.w_tokens)
        )
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(d_model)

        if self.verbose:
            print(f"VideoEncoder:")
            print(f"  Input:  [B, 1, {input_frames}, {frame_size}, {frame_size}]")
            print(f"  Conv1:  [B, 32,  50, 128, 128]")
            print(f"  Conv2:  [B, 64,  50,  64,  64]")
            print(f"  Conv3:  [B, 128, 50,  32,  32]")
            print(f"  Conv4:  [B, 256, 25,  16,  16]")
            print(f"  Conv5:  [B, {d_model}, 12,   8,   8]")
            print(f"  Pool:   [B, {d_model},  {self.t_tokens},   {self.h_tokens},   {self.w_tokens}]")
            print(f"  Output: [B, {n_output_tokens}, {d_model}]")

    def forward(self, x):
        """
        Encode video sequence into tokens.

        Parameters
        ----------
        x : torch.Tensor
            Input video of shape [batch, 1, input_frames, frame_size, frame_size]

        Returns
        -------
        torch.Tensor
            Encoded tokens of shape [batch, n_output_tokens, d_model]
        """
        B = x.shape[0]

        for conv in self.conv_layers:
            x = self.activation(conv(x))

        x = self.adaptive_pool(x)                # [B, d_model, t_tokens, h_tokens, w_tokens]
        x = x.flatten(2)                         # [B, d_model, n_output_tokens]
        x = x.transpose(1, 2)                    # [B, n_output_tokens, d_model]
        x = self.norm(x)

        return x


class VideoDecoder(nn.Module):
    """
    Mirrors VideoEncoder for pre-training via masked autoencoding.
    Reconstructs the original input video from encoder tokens.

    Parameters
    ----------
    input_frames : int, optional
        Number of frames to reconstruct (must match VideoEncoder.input_frames),
        by default 50
    frame_size : int, optional
        Height and width of frames to reconstruct, by default 256
    d_model : int, optional
        Model dimension from encoder, by default 512
    n_input_tokens : int, optional
        Number of input tokens from encoder, by default 192
    verbose : bool, optional
        If True, print debug information during initialization, by default False

    Attributes
    ----------
    deconv_layers : nn.ModuleList
        List of 3D transposed convolutional layers mirroring encoder
    adaptive_pool : nn.AdaptiveAvgPool3d
        Ensures exact output dimensions
    """

    def __init__(
        self,
        input_frames: int = 50,
        frame_size: int = 256,
        d_model: int = 512,
        n_input_tokens: int = 192,
        verbose: bool = False
    ):
        super().__init__()

        self.input_frames = input_frames
        self.frame_size = frame_size
        self.d_model = d_model
        self.n_input_tokens = n_input_tokens
        self.verbose = verbose

        # Starting spatiotemporal dimensions (mirrors encoder adaptive pool output)
        self.t_start = 3
        self.h_start = 8
        self.w_start = 8

        assert self.t_start * self.h_start * self.w_start == n_input_tokens, (
            f"n_input_tokens ({n_input_tokens}) must equal "
            f"t_start * h_start * w_start "
            f"({self.t_start} * {self.h_start} * {self.w_start})"
        )

        # Mirror encoder in reverse:
        # Layers 1-2: joint upsample
        # Layers 3-5: spatial upsample only
        self.deconv_layers = nn.ModuleList([
            # [B, d_model, 3,  8,  8] → [B, 256, 6,  16, 16]
            nn.ConvTranspose3d(d_model, 256, kernel_size=(3,3,3), stride=(2,2,2),
                               padding=(1,1,1), output_padding=(1,1,1)),
            # [B, 256, 6,  16, 16] → [B, 128, 12, 32, 32]
            nn.ConvTranspose3d(256, 128, kernel_size=(3,3,3), stride=(2,2,2),
                               padding=(1,1,1), output_padding=(1,1,1)),
            # [B, 128, 12, 32, 32] → [B, 64,  12, 64, 64]
            nn.ConvTranspose3d(128, 64,  kernel_size=(3,5,5), stride=(1,2,2),
                               padding=(1,2,2), output_padding=(0,1,1)),
            # [B, 64,  12, 64, 64] → [B, 32,  12, 128, 128]
            nn.ConvTranspose3d(64,  32,  kernel_size=(3,5,5), stride=(1,2,2),
                               padding=(1,2,2), output_padding=(0,1,1)),
            # [B, 32,  12, 128, 128] → [B, 1, 12, 256, 256]
            nn.ConvTranspose3d(32,  1,   kernel_size=(3,7,7), stride=(1,2,2),
                               padding=(1,3,3), output_padding=(0,1,1)),
        ])

        self.adaptive_pool = nn.AdaptiveAvgPool3d(
            (input_frames, frame_size, frame_size)
        )
        self.activation = nn.GELU()

        if self.verbose:
            print(f"VideoDecoder:")
            print(f"  Input:    [B, {n_input_tokens}, {d_model}]")
            print(f"  Reshape:  [B, {d_model}, {self.t_start}, {self.h_start}, {self.w_start}]")
            print(f"  Deconv1:  [B, 256, 6,   16,  16]")
            print(f"  Deconv2:  [B, 128, 12,  32,  32]")
            print(f"  Deconv3:  [B, 64,  12,  64,  64]")
            print(f"  Deconv4:  [B, 32,  12, 128, 128]")
            print(f"  Deconv5:  [B, 1,   12, 256, 256]")
            print(f"  Pool:     [B, 1, {input_frames}, {frame_size}, {frame_size}]")

    def forward(self, x):
        """
        Decode tokens back to original video (pre-training only).

        Parameters
        ----------
        x : torch.Tensor
            Input tokens of shape [batch, n_input_tokens, d_model]

        Returns
        -------
        torch.Tensor
            Reconstructed video of shape [batch, 1, input_frames, frame_size, frame_size]
        """
        B = x.shape[0]

        x = x.transpose(1, 2)                       # [B, d_model, n_input_tokens]
        x = x.reshape(
            B, self.d_model, self.t_start, self.h_start, self.w_start
        )                                            # [B, d_model, 3, 8, 8]

        for i, deconv in enumerate(self.deconv_layers):
            x = deconv(x)
            if i < len(self.deconv_layers) - 1:
                x = self.activation(x)

        x = self.adaptive_pool(x)                   # [B, 1, input_frames, frame_size, frame_size]

        return x


if __name__ == "__main__":

    print("=" * 60)
    print("VideoEncoder / VideoDecoder")
    print("=" * 60)
    vid_enc = VideoEncoder(input_frames=50, frame_size=256,
                           d_model=512, n_output_tokens=192, verbose=True)
    vid_dec = VideoDecoder(input_frames=50, frame_size=256,
                           d_model=512, n_input_tokens=192, verbose=True)
    x_vid = create_video_test_signal()
    tokens_vid = vid_enc(x_vid)
    recon_vid = vid_dec(tokens_vid)
    print(f"Input:  {x_vid.shape}")       # [4, 1, 50, 256, 256]
    print(f"Tokens: {tokens_vid.shape}")  # [4, 192, 512]
    print(f"Recon:  {recon_vid.shape}")   # [4, 1, 50, 256, 256]
