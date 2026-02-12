import torch.nn as nn
from abc import abstractmethod

class ModalityEncoder(nn.Module):

    def __init__(self, in_channels: int, out_features: int = 64, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_features = out_features

    @abstractmethod
    def forward(self, x):
        ...


class ModalityDecoder(nn.Module):

    def __init__(self, in_features: int = 64, out_channels: int = 1, **kwargs):
        super().__init__()
        self.in_features = in_features
        self.out_channels = out_channels

    @abstractmethod
    def forward(self, z):
        ...

class ModalityAutoEncoder(nn.Module):
    def __init__(self, in_channels: int, out_features: int = 64, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_features = out_features

    @abstractmethod
    def forward(self, x):
        ...