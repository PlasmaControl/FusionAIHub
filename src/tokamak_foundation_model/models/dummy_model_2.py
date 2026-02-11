import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Any
from tokamak_foundation_model.models.modality import PROCESSOR_REGISTRY


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
    def __init__(self, modality_configs: list[ModalityConfig]):
        super().__init__()
        self.output_dim = sum(cfg.out_features for cfg in modality_configs)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(features, dim=1)

FUSION_REGISTRY = {
    "attention": CrossAttentionBaselineModel,
    "concat": ConcatenationBaselineModel,
}


class Fusion4FusionModel(nn.Module):
    """
    Based on the 4M-21 (Massively Multimodal Masked Modeling) organizational framework
    Will allow modular scalable fusion of modalities.
    """

    def __init__(self,
        encoder_embeddings: dict[str, nn.Module],
        decoder_embeddings: dict[str, nn.Module],
        global_embeddings: dict[str, nn.Module], # not used for now, will be like text, etc to be used once during autoregression
        modality_info: dict[str, Any],
        fusion_model: dict[str, Any],
    ):
        super().__init__()

        self.modality_info = modality_info

        # initialize encoder embeddings
        self.encoder_modalities = set(encoder_embeddings.keys())
        for embedding in encoder_embeddings.values():
            embedding.init()
        
        # initialize decoder embeddings
        self.decoder_modalities = set(decoder_embeddings.keys())
        for embedding in decoder_embeddings.values():
            embedding.init()
        
    def encode(self,
        mod_dict: dict[str, dict[str, torch.Tensor]],
        return_logits: bool = False,
        ):
        """
        Encode individual modalities.
        """
        embeddings = []

        # TODO: Encode individual

        # TODO: Combine individual encodings
        embeddings = self.fusion_model(embeddings)

        pass

    def decode(self,
        embeddings: torch.Tensor,
        return_logits: bool = False,
        ):
        """
        Decode embeddings.
        """
        # TODO: Decode

        pass

    def forward(self,
        mod_dict: dict[str, dict[str, torch.Tensor]],
        return_logits: bool = False,
        ):

        encoder_mod_dict = {
            mod: self.encoder_embeddings[mod](d)
            for mod, d in mod_dict.items()
            if mod in self.encoder_embeddings
        }
        encoder_info = self.prepare_encoder(encoder_mod_dict)

        decoder_mod_dict = {
            mod: self.decoder_embeddings[mod](d)
            for mod, d in mod_dict.items()
            if mod in self.decoder_embeddings
        }
        decoder_info = self.prepare_decoder(decoder_mod_dict)

        # TODO: Add encoding context
        x = encoder_info['embeddings']
        x = self.encode(x)

        # TODO: Add decoding context
        y = x
        y = self.decode(y)

        if return_logits:
            return y

        loss, mod_loss = self.loss(x, y)
        return loss, mod_loss


class Prediction4FusionModel(nn.Module):
    """
    Idea is to first train Fusion4FusionModel and then freeze it and use it to train Prediction4FusionModel.
    Later we can train the whole model end-to-end.
    """

    def __init__(self,
        fusion_model: nn.Module,
    ):
        super().__init__()

        # set up and freeze encoder-decoder model
        self.fusion_model = fusion_model
        self.fusion_model.eval()
        for param in self.fusion_model.parameters():
            param.requires_grad = False

        
    def generate(self,
        mod_dict: dict[str, dict[str, torch.Tensor]],
        return_logits: bool = False,
        ):
        """
        Generate output from embeddings.
        """
        # TODO: Generate output
        pass

    def forward(self,
        mod_dict: dict[str, dict[str, torch.Tensor]],
        return_logits: bool = False,
        ):

        embeddings = self.fusion_model.encode(mod_dict)
        
        output = self.generate(embeddings, return_logits)

        return output

