import torch.nn as nn
from abc import abstractmethod


class ModalityEncoder(nn.Module):

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
    def forward(self, x):
        ...


class ModalityDecoder(nn.Module):

    def __init__(self, 
        n_channels: int, 
        d_model: int,
        ):
        super().__init__()
        self.n_channels = n_channels
        self.d_model = d_model

    @abstractmethod
    def forward(self, z, output_shape=None):
        ...


class ModalityAutoEncoder(nn.Module):

    def __init__(self, 
        n_channels: int, 
        d_model: int = 64,
        n_tokens: int = 0,
        ):
        super().__init__()
        self.encoder = ModalityEncoder(n_channels, d_model, n_tokens)
        self.decoder = ModalityDecoder(n_channels, d_model)

    def forward(self, x):
        return self.decoder(self.encoder(x))
