# A simple Masked Autoencoder architecture for spectrogram reconstruction.

import abc
import itertools
import typing

import einops.layers.torch as einops
import torch
import torch.nn as nn


def _frequency_strided_conv_params(
    size: int, stride: int, *, type: typing.Literal["1d", "2d"], transpose: bool
) -> dict[str, typing.Any]:
    # todo(Kouroche): Change padding to be "same", this is a valid PyTorch flag.
    if type == "1d":
        conv_params = {"kernel_size": size, "stride": 1, "padding": size // 2}
    elif type == "2d":
        conv_params = {"kernel_size": size, "stride": (stride, 1), "padding": size // 2}
        if transpose:
            conv_params["output_padding"] = (stride - 1, 0)
    return conv_params


class _BaseResidualBlock(abc.ABC, nn.Module):
    _conv_type: typing.ClassVar[
        type[nn.Conv1d]
        | type[nn.Conv2d]
        | type[nn.ConvTranspose1d]
        | type[nn.ConvTranspose2d]
    ]
    _norm_type: typing.ClassVar[type[nn.BatchNorm1d] | type[nn.BatchNorm2d]]

    conv_1: nn.Sequential
    conv_2: nn.Sequential
    projection: nn.Sequential | None
    activation: nn.Module

    # todo(Peter): Consider using `stride: tuple[int, int] | int` and warning if `int` is passed.
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1
    ) -> None:
        super().__init__()

        if self._conv_type is nn.Conv1d:
            self._build_encoder_for_1d(in_channels, out_channels, kernel_size, stride)
        elif self._conv_type is nn.Conv2d:
            self._build_encoder_for_2d(in_channels, out_channels, kernel_size, stride)
        elif self._conv_type is nn.ConvTranspose1d:
            self._build_decoder_for_1d(in_channels, out_channels, kernel_size, stride)
        elif self._conv_type is nn.ConvTranspose2d:
            self._build_decoder_for_2d(in_channels, out_channels, kernel_size, stride)

        self._build_residual_projector(in_channels, out_channels, kernel_size, stride)
        self.activation = nn.ReLU(inplace=True)

    def _build_encoder_for_1d(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int
    ) -> None:
        assert self._conv_type is nn.Conv1d

        # Preserve the input shape, pad to use "same" convolution.
        conv_params = _frequency_strided_conv_params(
            kernel_size, stride, type="1d", transpose=False
        )

        # Input tensor shape: `(batch_size, in_channels, time_steps)`
        self.conv_1 = nn.Sequential(
            self._conv_type(in_channels, out_channels, **conv_params),
            self._norm_type(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv_2 = nn.Sequential(
            self._conv_type(out_channels, out_channels, **conv_params),
            self._norm_type(out_channels),
        )

    def _build_encoder_for_2d(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int
    ) -> None:
        assert self._conv_type is nn.Conv2d

        # Preserve the input shape, pad to use "same" convolution.
        conv_params = _frequency_strided_conv_params(
            kernel_size, stride, type="2d", transpose=False
        )

        # Input tensor shape: `(batch_size, in_channels, freq_bins, time_steps)`
        self.conv_1 = nn.Sequential(
            self._conv_type(in_channels, out_channels, **conv_params),
            self._norm_type(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv_2 = nn.Sequential(
            self._conv_type(
                out_channels, out_channels, kernel_size, padding=conv_params["padding"]
            ),
            self._norm_type(out_channels),
        )

    def _build_decoder_for_1d(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int
    ) -> None:
        assert self._conv_type == nn.ConvTranspose1d

        # Preserve the input shape, pad to use "same" convolution.
        conv_params = _frequency_strided_conv_params(
            kernel_size, stride, type="1d", transpose=True
        )

        # Input tensor shape: `(batch_size, in_channels, time_steps)`
        self.conv_1 = nn.Sequential(
            self._conv_type(in_channels, out_channels, **conv_params),
            self._norm_type(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv_2 = nn.Sequential(
            self._conv_type(out_channels, out_channels, **conv_params),
            self._norm_type(out_channels),
        )

    def _build_decoder_for_2d(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int
    ) -> None:
        assert self._conv_type == nn.ConvTranspose2d

        # Preserve the input shape, pad to use "same" convolution.
        conv_params = _frequency_strided_conv_params(
            kernel_size, stride, type="2d", transpose=True
        )

        # Input tensor shape: `(batch_size, in_channels, freq_bins, time_steps)`
        self.conv_1 = nn.Sequential(
            self._conv_type(in_channels, out_channels, **conv_params),
            self._norm_type(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv_2 = nn.Sequential(
            self._conv_type(
                out_channels, out_channels, kernel_size, padding=conv_params["padding"]
            ),
            self._norm_type(out_channels),
        )

    def _build_residual_projector(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int
    ) -> None:
        if stride > 1 and self._conv_type in (nn.Conv2d, nn.ConvTranspose2d):
            conv_params = _frequency_strided_conv_params(
                1, stride, type="2d", transpose=self._conv_type == nn.ConvTranspose2d
            )

            self.projection = nn.Sequential(
                self._conv_type(in_channels, out_channels, **conv_params),
                self._norm_type(out_channels),
            )
        elif in_channels != out_channels:
            self.projection = nn.Sequential(
                self._conv_type(in_channels, out_channels, kernel_size=1),
                self._norm_type(out_channels),
            )
        else:
            self.projection = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        residual = input

        output = self.conv_1(input)
        output = self.conv_2(output)

        if self.projection:
            residual = self.projection(residual)

        output = output + residual
        output = self.activation(output)
        return output


class ResidualEncoding1d(_BaseResidualBlock):
    _conv_type = nn.Conv1d
    _norm_type = nn.BatchNorm1d


class ResidualEncoding2d(_BaseResidualBlock):
    _conv_type = nn.Conv2d
    _norm_type = nn.BatchNorm2d


class ResidualDecoding1d(_BaseResidualBlock):
    _conv_type = nn.ConvTranspose1d
    _norm_type = nn.BatchNorm1d


class ResidualDecoding2d(_BaseResidualBlock):
    _conv_type = nn.ConvTranspose2d
    _norm_type = nn.BatchNorm2d


class SpecEncoder(nn.Sequential):
    in_channels: int
    out_channels: int

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        hidden_channel_dims: list[int] = [64, 64, 128, 256],
        hidden_kernel_sizes: list[int] = [7, 3, 3, 3, 3],
        hidden_freq_strides: list[int] = [2, 2, 2, 2, 2],
    ) -> None:
        super().__init__()

        # Append the first encoding layer.
        in_hidden, out_hidden = in_channels, hidden_channel_dims[0]
        kernel_size = hidden_kernel_sizes[0]
        freq_stride = hidden_freq_strides[0]

        # Use "same" padding to maintain input shape within the first layer.
        conv_params = _frequency_strided_conv_params(
            kernel_size, freq_stride, type="2d", transpose=False
        )

        self.append(
            nn.Sequential(
                nn.Conv2d(in_hidden, out_hidden, **conv_params),
                nn.BatchNorm2d(out_hidden),
                nn.ReLU(inplace=True),
            )
        )
        self.append(nn.MaxPool2d(kernel_size=3, stride=(freq_stride, 1), padding=1))

        # Append the residual encoding blocks.
        for (in_hidden, out_hidden), kernel_size, freq_stride in zip(
            itertools.pairwise(hidden_channel_dims + [out_channels]),
            hidden_kernel_sizes[1:],
            hidden_freq_strides[1:],
        ):
            self.append(
                ResidualEncoding2d(in_hidden, out_hidden, kernel_size, freq_stride)
            )

        # Keep track of input and output channel dimensions.
        self.in_channels = in_channels
        self.out_channels = out_channels


class SpecDecoder(nn.Sequential):
    in_channels: int
    out_channels: int

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        hidden_channel_dims: list[int] = [256, 128, 64, 64],
        hidden_kernel_sizes: list[int] = [3, 3, 3, 3, 7],
        hidden_freq_strides: list[int] = [2, 2, 2, 2, 2],
    ) -> None:
        super().__init__()

        # Append the residual decoding blocks.
        for (in_hidden, out_hidden), kernel_size, freq_stride in zip(
            itertools.pairwise([in_channels] + hidden_channel_dims),
            hidden_kernel_sizes[:-1],
            hidden_freq_strides[:-1],
        ):
            self.append(
                ResidualDecoding2d(in_hidden, out_hidden, kernel_size, freq_stride)
            )

        # Append the final decoding layer.
        in_hidden, out_hidden = hidden_channel_dims[-1], out_channels
        kernel_size = hidden_kernel_sizes[-1]
        freq_stride = hidden_freq_strides[-1]

        conv_params = _frequency_strided_conv_params(
            kernel_size, freq_stride, type="2d", transpose=True
        )

        self.append(nn.Upsample(scale_factor=(2, 1), mode="bilinear"))
        self.append(nn.ConvTranspose2d(in_hidden, out_hidden, **conv_params))

        # Keep track of input and output channels
        self.in_channels = in_channels
        self.out_channels = out_channels


class WaveEncoder(nn.Sequential):
    in_channels: int
    out_channels: int

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        hidden_channel_dims: list[int] = [64, 64, 128, 256],
        hidden_kernel_sizes: list[int] = [7, 3, 3, 3, 3],
    ) -> None:
        super().__init__()

        # Append the first encoding layer.
        in_hidden, out_hidden = in_channels, hidden_channel_dims[0]
        kernel_size = hidden_kernel_sizes[0]

        # Use "same" padding to maintain input shape within the first layer.
        conv_params = _frequency_strided_conv_params(
            kernel_size, stride=1, type="1d", transpose=False
        )

        self.append(
            nn.Sequential(
                nn.Conv1d(in_hidden, out_hidden, **conv_params),
                nn.BatchNorm1d(out_hidden),
                nn.ReLU(inplace=True),
            )
        )
        self.append(nn.MaxPool1d(kernel_size=3, stride=1, padding=1))

        # Append the residual encoding blocks.
        for (in_hidden, out_hidden), kernel_size in zip(
            itertools.pairwise(hidden_channel_dims + [out_channels]),
            hidden_kernel_sizes[1:],
        ):
            self.append(ResidualEncoding1d(in_hidden, out_hidden, kernel_size))

        # Keep track of input and output channels
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Output shape: (batch_size, out_channels, 1, time_steps)
        return super().forward(input)[:, :, None, :]


class WaveDecoder(nn.Sequential):
    in_channels: int
    out_channels: int

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        hidden_channel_dims: list[int] = [256, 128, 64, 64],
        hidden_kernel_sizes: list[int] = [3, 3, 3, 3, 7],
    ) -> None:
        super().__init__()

        # Append the residual decoding blocks.
        for (in_hidden, out_hidden), kernel_size in zip(
            itertools.pairwise([in_channels] + hidden_channel_dims),
            hidden_kernel_sizes[:-1],
        ):
            self.append(ResidualDecoding1d(in_hidden, out_hidden, kernel_size))

        # Append the final decoding layer.
        in_hidden, out_hidden = hidden_channel_dims[-1], out_channels
        kernel_size = hidden_kernel_sizes[-1]

        conv_params = _frequency_strided_conv_params(
            kernel_size, 1, type="1d", transpose=True
        )
        self.append(nn.ConvTranspose1d(in_hidden, out_hidden, **conv_params))

        # Keep track of input and output channels
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, in_channels, 1, time_steps)
        return super().forward(input[:, :, 0, :])


# todo(Kouroche): These group modules are hacked together, let's clean them up after.
class _BaseGroupModule(nn.ModuleDict):
    def forward(self, input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name: module(input[name]) for name, module in self.items()}

    @property
    def in_channels(self) -> int:
        return sum(mod.in_channels for mod in self.values())  # type: ignore

    @property
    def out_channels(self) -> int:
        return sum(mod.out_channels for mod in self.values())  # type: ignore


class EncoderGroup(_BaseGroupModule):
    def __init__(self, encoders: dict[str, WaveEncoder | SpecEncoder]):
        super().__init__(encoders)


class DecoderGroup(_BaseGroupModule):
    def __init__(self, decoders: dict[str, WaveDecoder | SpecDecoder]):
        super().__init__(decoders)


class MaskedAutoEncoderModel(nn.Module):
    encoder: EncoderGroup
    decoder: DecoderGroup
    mixing: nn.Sequential

    def __init__(
        self,
        encoders: dict[str, WaveEncoder | SpecEncoder],
        decoders: dict[str, WaveDecoder | SpecDecoder],
        *,
        num_features: int,
    ) -> None:
        super().__init__()

        self.encoder = EncoderGroup(encoders)

        # todo(Kouroche): This is inspired by MLP-Mixer, but definitely deserves a more
        # thorough investigation. MLP mixer also has a skip connection, activations and
        # layer normalization which is not implemented here yet.
        self.mixing = nn.Sequential(
            einops.Rearrange("b c f t -> b c t f"),
            nn.Linear(num_features, num_features),
            einops.Rearrange("b c t f -> b f t c"),
            nn.Linear(self.encoder.out_channels, self.encoder.out_channels),
            einops.Rearrange("b f t c -> b c f t"),
        )

        self.decoder = DecoderGroup(decoders)

    def forward(self, input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        encoded: dict[str, torch.Tensor] = self.encoder(input)
        encoder_channels = [encoded[name].shape[1] for name in self.encoder]

        # Encoded shape: encoder_count * (batch_size, out_channels, freq_bins, time_steps)
        mixed = self.mixing(  # Mix across "frequency" bins and channels.
            torch.cat([encoded[name] for name in self.encoder], dim=1)
        )

        decoded: dict[str, torch.Tensor] = self.decoder(
            dict(zip(encoded.keys(), torch.split(mixed, encoder_channels, dim=1)))
        )
        return decoded
