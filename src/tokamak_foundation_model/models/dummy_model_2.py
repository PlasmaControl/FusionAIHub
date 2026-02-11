import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from tokamak_foundation_model.models.modality import PROCESSOR_REGISTRY


# ====== Configuration ======

@dataclass
class ModalityConfig:
    name: str
    processor_type: str
    in_channels: int = 0
    out_features: int = 64
    context: str = "local"  # "local" (time-windowed) or "global" (whole shot)
    group: str | None = None

DEFAULT_MODALITY_CONFIGS = [
    ModalityConfig("mhr", "spectrogram", 8),
    ModalityConfig("ece", "spectrogram", 48),
    ModalityConfig("co2", "spectrogram", 4),
    ModalityConfig("gas", "timeseries", 5, group="actuators"),
    ModalityConfig("ech", "timeseries", 11, group="actuators"),
    ModalityConfig("pin", "timeseries", 8, group="actuators"),
    ModalityConfig("tin", "timeseries", 8, group="actuators"),
    ModalityConfig("d_alpha", "fast_timeseries", 6),
    ModalityConfig("mse", "timeseries", 69, group="diagnostics"),
    ModalityConfig("ts_core_density", "timeseries", 44, group="diagnostics"),
    ModalityConfig("bolo", "video", 1),
    ModalityConfig("irtv", "video", 1),
    ModalityConfig("tangtv", "video", 1),
    ModalityConfig("text", "text", 1, context="global"),
]


# ====== Fusion ======

class CrossAttentionBaselineModel(nn.Module):
    def __init__(self, feature_dim, num_modalities):
        super().__init__()
        self.output_dim = feature_dim
        self.attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_modalities, batch_first=True)

    def forward(self, features):
        stacked = torch.stack(features, dim=1)
        attended, _ = self.attn(stacked, stacked, stacked)
        return attended.mean(dim=1)


class ConcatenationBaselineModel(nn.Module):
    def __init__(self, feature_dim, num_modalities):
        super().__init__()
        self.output_dim = feature_dim * num_modalities

    def forward(self, features):
        return torch.cat(features, dim=1)


FUSION_REGISTRY = {
    "attention": CrossAttentionBaselineModel,
    "concat": ConcatenationBaselineModel,
}


# ====== Base Encoder ======

class MultiModalEncoder(nn.Module):

    def __init__(self, feature_dim=64, fusion_mode="attention",
                 text_model_name="distilbert-base-uncased", modality_configs=None):
        super().__init__()
        self.feature_dim = feature_dim
        self.modality_configs = modality_configs or DEFAULT_MODALITY_CONFIGS

        self.processors = nn.ModuleDict()
        self._group_order: list[str] = []
        groups: dict[str, dict] = {}

        for cfg in self.modality_configs:
            if cfg.group:
                if cfg.group not in groups:
                    groups[cfg.group] = {"type": cfg.processor_type, "channels": 0,
                                         "out_features": cfg.out_features}
                    self._group_order.append(cfg.group)
                groups[cfg.group]["channels"] += cfg.in_channels
            else:
                cls = PROCESSOR_REGISTRY[cfg.processor_type]
                kwargs = {"in_channels": cfg.in_channels, "out_features": cfg.out_features}
                if cfg.processor_type == "text":
                    kwargs["text_model_name"] = text_model_name
                self.processors[cfg.name] = cls(**kwargs)

        for name in self._group_order:
            g = groups[name]
            self.processors[name] = PROCESSOR_REGISTRY[g["type"]](
                in_channels=g["channels"], out_features=g["out_features"])

        self.fusion = FUSION_REGISTRY[fusion_mode](
            feature_dim=feature_dim, num_modalities=len(self.processors))
        self.fused_dim = self.fusion.output_dim

    def encode(self, inputs):
        features = []
        group_tensors: dict[str, list[torch.Tensor]] = {}

        for cfg in self.modality_configs:
            if cfg.name not in inputs:
                continue
            x = inputs[cfg.name]

            if cfg.processor_type == "text":
                features.append(self.processors[cfg.name]([str(t) for t in x]))
                continue
            if cfg.processor_type in ("timeseries", "fast_timeseries"):
                x = x.squeeze(2)
            if cfg.group:
                group_tensors.setdefault(cfg.group, []).append(x)
            else:
                features.append(self.processors[cfg.name](x))

        for name in self._group_order:
            if name in group_tensors:
                tensors = group_tensors[name]
                # Resample to common time length if needed
                target_len = tensors[0].shape[-1]
                tensors = [F.interpolate(t, size=target_len, mode="linear") if t.shape[-1] != target_len else t
                           for t in tensors]
                features.append(self.processors[name](torch.cat(tensors, dim=1)))

        return self.fusion(features)


# ====== Task Heads ======

class MultiModalTokamakModel(MultiModalEncoder):
    """Scalar prediction from fused multi-modal input."""

    def __init__(self, feature_dim=64, fusion_mode="attention",
                 text_model_name="distilbert-base-uncased", modality_configs=None):
        super().__init__(feature_dim=feature_dim, fusion_mode=fusion_mode,
                         text_model_name=text_model_name, modality_configs=modality_configs)
        self.predictor = nn.Sequential(
            nn.Linear(self.fused_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, batch):
        return self.predictor(self.encode(batch))


class MultiModalPredictionModel(MultiModalEncoder):
    """Predicts future d_alpha, mse, ts_core_density from fused multi-modal input."""

    def __init__(self, feature_dim=64, fusion_mode="concat",
                 text_model_name="distilbert-base-uncased", modality_configs=None,
                 target_frames=50):
        super().__init__(feature_dim=feature_dim, fusion_mode=fusion_mode,
                         text_model_name=text_model_name, modality_configs=modality_configs)
        self.target_frames = target_frames

        def head(channels):
            return nn.Sequential(
                nn.Linear(self.fused_dim, 512), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, channels * target_frames),
            )
        self.d_alpha_head = head(6)
        self.mse_head = head(69)
        self.ts_core_head = head(44)

    def forward(self, batch):
        inputs = batch.get("inputs", batch)
        fused = self.encode(inputs)
        B, T = fused.shape[0], self.target_frames
        return {
            "d_alpha": self.d_alpha_head(fused).view(B, 6, 1, T),
            "mse": self.mse_head(fused).view(B, 69, 1, T),
            "ts_core_density": self.ts_core_head(fused).view(B, 44, 1, T),
        }
