import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class StridedResBlock1d(nn.Module):
    """Pre-norm strided 1D residual block for encoding."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.norm = nn.InstanceNorm1d(in_channels, affine=True)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=kernel_size // 2),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size,
                      stride=1, padding=kernel_size // 2),
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels,
                                      kernel_size=1, stride=stride)
        else:
            self.shortcut = nn.Identity()
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(self.net(self.norm(x)) + self.shortcut(x))


class StridedResBlockTranspose1d(nn.Module):
    """Pre-norm strided 1D transposed residual block for decoding."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.norm = nn.InstanceNorm1d(in_channels, affine=True)
        self.net = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=kernel_size // 2,
                               output_padding=stride - 1),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size,
                      stride=1, padding=kernel_size // 2),
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.ConvTranspose1d(in_channels, out_channels,
                                               kernel_size=1, stride=stride,
                                               output_padding=stride - 1)
        else:
            self.shortcut = nn.Identity()
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(self.net(self.norm(x)) + self.shortcut(x))


class ModalityEncoder(nn.Module, ABC):

    def __init__(self,
        n_channels: int,
        d_model: int = 64,
        n_tokens: int = 0,
        ):
        super().__init__()
        self.n_channels = n_channels
        self.d_model = d_model
        self.n_tokens = n_tokens

    @abstractmethod
    def forward(self, x) -> torch.Tensor:
        raise NotImplementedError


class ModalityDecoder(nn.Module, ABC):

    def __init__(self,
        n_channels: int,
        d_model: int,
        ):
        super().__init__()
        self.n_channels = n_channels
        self.d_model = d_model

    @abstractmethod
    def forward(self, z, output_shape=None) -> torch.Tensor:
        raise NotImplementedError


class ModalityAutoEncoder(nn.Module):

    def __init__(self,
        n_channels: int,
        d_model: int = 64,
        n_tokens: int = 0,
        ):
        super().__init__()
        self.n_channels = n_channels
        self.d_model = d_model
        self.n_tokens = n_tokens

    @abstractmethod
    def forward(self, x) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError
