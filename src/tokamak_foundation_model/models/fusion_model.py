import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from typing import Any

from .modality import PROCESSOR_REGISTRY
from .latent_space import CrossModalAttention
from .loss import DictMSELoss

class Fusion4FusionModel(nn.Module):
    """
    Based on the 4M-21 (Massively Multimodal Masked Modeling) organizational framework
    Will allow modular scalable fusion of modalities.
    """

    def __init__(self,
        encoder_embeddings: dict[str, nn.Module],
        decoder_embeddings: dict[str, nn.Module],
        global_embeddings: dict[str, nn.Module],
        modality_info: dict[str, Any],
        fusion_model: dict[str, Any],
    ):
        super().__init__()

        self.modality_info = modality_info

        # initialize encoder embeddings
        self.encoder_modalities = set(encoder_embeddings.keys())
        
        # initialize decoder embeddings
        self.decoder_modalities = set(decoder_embeddings.keys())

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

