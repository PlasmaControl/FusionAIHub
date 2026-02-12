import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from .modality import PROCESSOR_REGISTRY
from .loss import DictMSELoss


# ====== Configuration ======

# TODO: Add config dir for organization
@dataclass
class ModalityConfig:
    name: str
    processor_type: str
    in_channels: int = 0
    out_features: int = 64
    context: str = "local"  # "local" (time-windowed) or "global" (whole shot)
    group: str | None = None

DEFAULT_MODALITY_CONFIGS = [
    ModalityConfig("mhr", "spectrogram", in_channels=8, out_features=64),
    ModalityConfig("ece", "spectrogram", in_channels=48, out_features=64),
    ModalityConfig("co2", "spectrogram", in_channels=4, out_features=64),
    ModalityConfig("gas", "timeseries", in_channels=5, out_features=64, group="actuators"),
    ModalityConfig("ech", "timeseries", in_channels=11, out_features=64, group="actuators"),
    ModalityConfig("pin", "timeseries", in_channels=8, out_features=64, group="actuators"),
    ModalityConfig("tin", "timeseries", in_channels=8, out_features=64, group="actuators"),
    ModalityConfig("d_alpha", "fast_timeseries", in_channels=6, out_features=64),
    ModalityConfig("mse", "timeseries", in_channels=69, out_features=64, group="diagnostics"),
    ModalityConfig("ts_core_density", "timeseries", in_channels=44, out_features=64, group="diagnostics"),
    ModalityConfig("bolo", "video", in_channels=1, out_features=64),
    ModalityConfig("irtv", "video", in_channels=1, out_features=64),
    ModalityConfig("tangtv", "video", in_channels=1, out_features=64),
    ModalityConfig("text", "text", in_channels=1, out_features=64, context="global", group=None),
]

DEFAULT_FUSION_MODEL = "concatenation"

class Fusion4FusionModel(nn.Module):
    """
    Based on the 4M-21 (Massively Multimodal Masked Modeling) organizational framework.
    Encodes each modality with its own encoder and fuses via cross-modal attention.
    """

    def __init__(self, modality_configs=None, feature_dim=64, num_heads=4):
        super().__init__()
        if modality_configs is None:
            modality_configs = DEFAULT_MODALITY_CONFIGS
        self.modality_configs = modality_configs
        self.feature_dim = feature_dim

        # Build one encoder per modality from the registry
        self.encoders = nn.ModuleDict()
        for cfg in modality_configs:
            encoder_cls = PROCESSOR_REGISTRY[cfg.processor_type]
            self.encoders[cfg.name] = encoder_cls(
                in_channels=cfg.in_channels, out_features=cfg.out_features
            )

        # Fusion model
        self.fusion = PROCESSOR_REGISTRY[DEFAULT_FUSION_MODEL](
            feature_dim=feature_dim,
            num_modalities=len(modality_configs),
        )

    def encode(self, inputs: dict) -> torch.Tensor:
        """
        Encode all available modalities and fuse into a single feature vector.

        Args:
            inputs: dict mapping modality name to tensor/data
        Returns:
            fused: (B, feature_dim)
        """
        features = []
        for cfg in self.modality_configs:
            if cfg.name not in inputs:
                continue
            x = inputs[cfg.name]
            # Shape preprocessing per modality type
            if cfg.processor_type in ("timeseries", "fast_timeseries"):
                # (B, C, 1, T) -> (B, C, T)
                if isinstance(x, torch.Tensor) and x.dim() == 4 and x.shape[2] == 1:
                    x = x.squeeze(2)
            features.append(self.encoders[cfg.name](x))

        return self.fusion(features)

    def forward(self, inputs: dict) -> torch.Tensor:
        return self.encode(inputs)


class Prediction4FusionModel(nn.Module):
    """
    Uses Fusion4FusionModel as encoder, adds per-target prediction heads.
    Can optionally freeze the encoder for transfer learning.
    """

    def __init__(self, modality_configs=None, feature_dim=64, num_heads=4,
                 target_configs=None, freeze_encoder=False):
        super().__init__()
        self.encoder = Fusion4FusionModel(modality_configs, feature_dim, num_heads)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # target_configs: {"d_alpha": (6, 20), "mse": (69, 20), ...}
        # maps target name -> (n_channels, n_frames)
        self.target_configs = target_configs or {}
        self.heads = nn.ModuleDict()
        for name, (n_channels, n_frames) in self.target_configs.items():
            self.heads[name] = nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, n_channels * n_frames),
            )

    def forward(self, inputs: dict) -> dict:
        # Handle {"inputs": ..., "targets": ...} format from dataloader
        if "inputs" in inputs:
            inputs = inputs["inputs"]

        fused = self.encoder.encode(inputs)

        outputs = {}
        for name, head in self.heads.items():
            n_channels, n_frames = self.target_configs[name]
            out = head(fused)  # (B, n_channels * n_frames)
            outputs[name] = out.view(-1, n_channels, n_frames)  # (B, C, T)

        return outputs

