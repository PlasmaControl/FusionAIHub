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
    """Pre-norm upsampling residual block for decoding.

    Uses nearest-neighbor interpolation followed by Conv1d instead of
    ConvTranspose1d to avoid checkerboard / periodic artifacts.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.stride = stride
        self.norm = nn.InstanceNorm1d(in_channels, affine=True)
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=stride, mode='nearest'),
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      stride=1, padding=kernel_size // 2),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size,
                      stride=1, padding=kernel_size // 2),
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Upsample(scale_factor=stride, mode='nearest'),
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
            )
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
        # Records input length at first forward; asserts equality on
        # every subsequent call. Persisted to checkpoints so a reloaded
        # AE rejects data chunked differently from its training run
        # (e.g. 500ms dataset fed into a 50ms-trained AE — silent
        # garbage otherwise because the architecture is length-
        # agnostic via AdaptiveAvgPool).
        self.register_buffer(
            "expected_input_length",
            torch.tensor(-1, dtype=torch.long),
        )
        self.register_forward_pre_hook(self._check_input_length_hook)

    @staticmethod
    def _check_input_length_hook(module, inputs):
        x = inputs[0]
        T = int(x.shape[-1])
        expected = int(module.expected_input_length.item())
        if expected < 0:
            module.expected_input_length.fill_(T)
        elif T != expected:
            raise ValueError(
                f"{type(module).__name__}: input length {T} does not "
                f"match the length {expected} this AE was trained on. "
                "Check chunk_duration_s / target_fs for this modality."
            )

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        # Back-compat: checkpoints saved before this buffer existed
        # have no 'expected_input_length' entry. Inject the sentinel so
        # strict loading succeeds; first forward after load re-records.
        key = prefix + "expected_input_length"
        if key not in state_dict:
            state_dict = {
                **state_dict,
                key: torch.tensor(-1, dtype=torch.long),
            }
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

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
