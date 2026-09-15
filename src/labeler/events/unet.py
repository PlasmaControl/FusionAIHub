"""The TokEye U-Net, vendored: a class-agnostic segmenter of spectrograms.

The network takes one spectrogram tile `(B, 1, H, W)` and returns two masks:
channel 0 is **coherent** activity (a mode with a track through time -
Alfven eigenmodes, tearing, EHO), channel 1 is **transient** activity (a
burst with no track - an ELM, a sawtooth crash). It is deliberately class
agnostic: naming what a lit region *is* happens later, from the shape of
its track and the state of the plasma around it, not here.

**Why the code is copied rather than imported.** Upstream is a research
repository under active development on a scratch filesystem; labeler's
output has to be reproducible years from now, from this checkout alone. So
the forward pass lives here, the checkpoint is pinned by sha256, and one
golden output (`tests/labeler/data/unet_golden.npz`, written by
`scripts/labeler/pin_unet.py`) proves that the copy and the original
agree to the last bit. Nothing in labeler imports `tokeye`.

Vendored verbatim, apart from the `BigTFUNetConfig` dataclass (upstream's is
a plain class taking `**kwargs`; the field names, defaults and therefore the
constructed module are identical), from:

    /scratch/gpfs/nc1514/tokeye/src/tokeye/models/big_tf_unet/model_big_tf_unet.py
    /scratch/gpfs/nc1514/tokeye/src/tokeye/models/big_tf_unet/config_big_tf_unet.py

at tokeye commit ba00238504c65bfc40db79dbf19832709ff00a29 (those two files
last changed in 05f68664ce40049393264f29ea63811f3da8e755, 2025-12-26).
Attribute names are load bearing - they are the state-dict keys - so
`in_conv.conv.0.weight` and friends must keep resolving exactly as upstream.

This is **not** a labeler model in the `models/` sense: there is no card,
no adapter and no runner, because it produces masks for the event layer
rather than a label column on the 25 ms grid. What it does share with those
models is where its weights live - `$LABELER_ROOT/models/tokeye/` - so
one data root still holds every weight file labeler reads.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import torch
from torch import nn
from torch.nn import functional as F

from ..config import Paths, sha256_of

#: The pinned checkpoint, trained by TokEye and copied into the data root.
#: `big_tf_unet_251210_weights.pt` beside it upstream is an older-key export
#: of the same run (`in_conv.double_conv.*` instead of `in_conv.conv.*`) that
#: needs a rename map to load; this file is the one `tokeye.hub` serves and
#: the only one that loads `strict=True` into the module below.
CHECKPOINT_NAME = "big_tf_unet_251210.pt"

#: Folder under `Paths.models` holding TokEye's artifacts.
CHECKPOINT_SUBDIR = "tokeye"

#: sha256 of `CHECKPOINT_NAME`. `load_unet` refuses anything else.
CHECKPOINT_SHA256 = "4afc3948ba53af40cb2787441b251757f8c77e764784e0aaa2602af0471229a9"

#: `sum(p.numel() for p in BigTFUNetModel(BigTFUNetConfig()).parameters())`
#: - weights and biases, not the batch-norm buffers. Pinned so an edit to the
#: vendored architecture that still happens to load is caught, not merely one
#: that does not.
N_PARAMS = 7_852_002


def default_checkpoint_path() -> Path:
    """Where the pinned checkpoint lives, per `LABELER_ROOT`."""
    return Paths.from_env().models / CHECKPOINT_SUBDIR / CHECKPOINT_NAME


@dataclass(frozen=True)
class BigTFUNetConfig:
    """Upstream's `config_big_tf_unet.BigTFUNetConfig`, field for field."""

    #: Upstream keeps this as a class attribute; `ClassVar` is how a
    #: dataclass says the same thing.
    model_type: ClassVar[str] = "big_tf_unet"

    in_channels: int = 1
    out_channels: int = 2
    num_layers: int = 5
    first_layer_size: int = 32
    dropout_rate: float = 0.2


class BigTFUNetConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int | None = None,
        dropout_rate: float = 0.0,
        kernel_size: int = 3,
        padding: int = 1,
    ) -> None:
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        layers: list[nn.Module] = []

        layers.extend([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
        ])

        if dropout_rate > 0:
            layers.extend([nn.Dropout2d(p=dropout_rate)])

        layers.extend([
            nn.Conv2d(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        ])

        if dropout_rate > 0:
            layers.extend([nn.Dropout2d(p=dropout_rate)])

        self.conv = nn.Sequential(*layers)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.conv(hidden_states)


class BigTFUNetDownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_rate: float = 0.0,
        kernel_size: int = 2,
    ) -> None:
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=kernel_size),
            BigTFUNetConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                dropout_rate=dropout_rate,
            ),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down(hidden_states)


class BigTFUNetUpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_rate: float = 0.0,
        kernel_size: int = 2,
    ) -> None:
        super().__init__()

        self.up = nn.Upsample(
            scale_factor=kernel_size,
            mode="bilinear",
            align_corners=True,
        )
        self.conv = BigTFUNetConvBlock(
            in_channels=in_channels + out_channels,
            out_channels=out_channels,
            dropout_rate=dropout_rate,
        )

    def forward(
        self,
        hidden_states_1: torch.Tensor,
        hidden_states_2: torch.Tensor,
    ) -> torch.Tensor:

        hidden_states_1 = self.up(hidden_states_1)

        diffY = hidden_states_2.size()[2] - hidden_states_1.size()[2]
        diffX = hidden_states_2.size()[3] - hidden_states_1.size()[3]

        hidden_states_1 = F.pad(
            hidden_states_1,
            [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2],
        )

        hidden_states = torch.cat([hidden_states_2, hidden_states_1], dim=1)
        return self.conv(hidden_states)


class BigTFUNetModel(nn.Module):
    """`forward` returns a 1-tuple of **logits**, exactly as upstream does.

    The tuple is upstream's signature and is kept so a future port of
    TokEye's own inference code needs no adaptation; `probabilities` is the
    convenience wrapper the event layer actually calls.
    """

    def __init__(self, config: BigTFUNetConfig):
        super().__init__()
        self.config = config

        # Layer sizes
        layer_sizes: list[int] = [
            config.first_layer_size * 2**i
            for i in range(config.num_layers)
        ]

        # Initial Channel Convolution
        self.in_conv = BigTFUNetConvBlock(
            config.in_channels,
            layer_sizes[0],
            dropout_rate=config.dropout_rate,
        )

        # Encoder
        encoder: list[BigTFUNetDownBlock] = []
        for i in range(config.num_layers - 1):
            in_ch = layer_sizes[i]
            out_ch = layer_sizes[i + 1]
            encoder.append(BigTFUNetDownBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                dropout_rate=config.dropout_rate,
            ))
        self.encoder = nn.ModuleList(encoder)

        # Decoder
        decoder: list[BigTFUNetUpBlock] = []
        for i in range(config.num_layers - 1):
            in_ch = layer_sizes[-i - 1]
            out_ch = layer_sizes[-i - 2]
            decoder.append(BigTFUNetUpBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                dropout_rate=config.dropout_rate,
            ))
        self.decoder = nn.ModuleList(decoder)

        # Final Channel Convolution
        self.out_conv = nn.Conv2d(
            layer_sizes[0],
            config.out_channels,
            kernel_size=1,
        )

    def forward(self, input_BCHW: torch.Tensor) -> tuple[torch.Tensor]:
        skip_BCHW: list[torch.Tensor] = []

        # Channel Convolution
        encode_BCHW = self.in_conv(input_BCHW)
        skip_BCHW.append(encode_BCHW)

        # Encoder
        for layer in self.encoder:
            encode_BCHW = layer(encode_BCHW)
            skip_BCHW.append(encode_BCHW)

        # Bottleneck
        decode_BCHW = encode_BCHW

        # Decoder
        for i, layer in enumerate(self.decoder):
            skip_idx = len(skip_BCHW) - i - 2
            decode_BCHW = layer(
                decode_BCHW,
                skip_BCHW[skip_idx],
            )

        # Channel Convolution
        output_BCHW = self.out_conv(decode_BCHW)

        return (output_BCHW,)


# --------------------------------------------------------------------------
# Everything below is labeler's, not TokEye's.
# --------------------------------------------------------------------------


@torch.inference_mode()
def probabilities(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """`(B, 2, H, W)` sigmoid masks: channel 0 coherent, channel 1 transient.

    The head is trained with a per-pixel binary loss on each channel
    independently - the two are not a softmax over classes, and a pixel may
    be lit in both (a burst on top of a track).

    `inference_mode` rather than the caller's discipline: this is called once
    per tile, tens of thousands of times per shot, and a forgotten `no_grad`
    would keep every intermediate alive. Nothing downstream differentiates
    through a mask, so the graph is never wanted. The returned tensor is an
    inference tensor: read it, copy it (`.numpy()`, `.clone()`), do not put
    it back into a computation that needs autograd.
    """
    return torch.sigmoid(model(x)[0])


def load_unet(
    path=None,
    device: str = "cpu",
    *,
    verify_sha256: bool = True,
) -> nn.Module:
    """The pinned U-Net, in `eval()` mode on `device`.

    `path=None` resolves to `$LABELER_ROOT/models/tokeye/` - the same data
    root every other weight file labeler loads lives under. The checkpoint
    is a bare `state_dict`, so it loads with `weights_only=True`: nothing in
    the file is executed.

    Two guards, both raising `ValueError`. The **hash** guard is about
    provenance: a mask, and every event derived from it, is only worth the
    weights behind it, so a file that is not byte-for-byte the pinned one is
    refused by default rather than used quietly. `verify_sha256=False` is
    the deliberate escape hatch, for the run that re-pins a new checkpoint.
    The **parameter-count** guard is about the code: a state dict can load
    `strict=True` into an architecture that has drifted in ways the keys do
    not reveal, and `N_PARAMS` is the independent number that catches it.
    """
    path = Path(path) if path is not None else default_checkpoint_path()
    if verify_sha256:
        got = sha256_of(path)
        if got != CHECKPOINT_SHA256:
            raise ValueError(
                f"{path}: sha256 {got} does not match the pinned "
                f"{CHECKPOINT_SHA256}"
            )
    state_dict = torch.load(path, weights_only=True, map_location="cpu")
    model = BigTFUNetModel(BigTFUNetConfig())
    model.load_state_dict(state_dict, strict=True)
    n = sum(p.numel() for p in model.parameters())
    if n != N_PARAMS:
        raise ValueError(
            f"vendored U-Net has {n} parameters, not the pinned {N_PARAMS}"
        )
    return model.to(device).eval()
