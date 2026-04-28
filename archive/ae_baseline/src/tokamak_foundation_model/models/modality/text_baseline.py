import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from .base import ModalityEncoder, ModalityDecoder


class TextEncoder(ModalityEncoder):
    def __init__(
        self,
        in_channels: int = 1,
        out_features: int = 64,
        text_model_name: str = "distilbert-base-uncased",
        **kwargs,
    ):
        super().__init__(in_channels, out_features)
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.encoder = AutoModel.from_pretrained(text_model_name)
        self.hidden_size = self.encoder.config.hidden_size
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.proj = nn.Sequential(nn.Linear(self.hidden_size, out_features), nn.ReLU())

    def forward(self, x):
        """Forward pass accepting either raw strings or pre-tokenized dict.

        Args:
            x: Either a list of strings (tokenized on-the-fly) or a dict with
               keys "text_input_ids" and "text_attention_mask" (pre-tokenized
               tensors from the dataset).
        """
        device = next(self.parameters()).device

        if isinstance(x, dict):
            input_ids = x["text_input_ids"].to(device)
            attention_mask = x["text_attention_mask"].to(device)
        else:
            enc = self.tokenizer(
                x, padding=True, truncation=True, max_length=512,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            out = self.encoder(input_ids, attention_mask=attention_mask)
        return self.proj(out.last_hidden_state[:, 0, :])


class TextDecoder(ModalityDecoder):
    """Projects latent features back to the text encoder's hidden space."""

    def __init__(self, in_features=64, out_channels=768, **kwargs):
        super().__init__(in_features, out_channels)
        self.net = nn.Sequential(
            nn.Linear(in_features, 256), nn.ReLU(),
            nn.Linear(256, out_channels),
        )

    def forward(self, z):
        return self.net(z)
