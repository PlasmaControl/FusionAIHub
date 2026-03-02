import torch
import torch.nn as nn
from typing import Any
from abc import ABC, abstractmethod


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
