"""
Variational autoencoder wrapper for any ``ModalityAutoEncoder``.

Wraps a deterministic AE so the encoder becomes a Gaussian encoder
producing ``(mu, logvar)``.  Inference uses ``mu`` directly (drop-in
for the AE's deterministic encoder path); training uses the
reparameterisation trick to sample ``z``.  The decoder is reused
unchanged.  A KL-to-standard-normal term is available via
``kl_divergence_standard_normal`` for the trainer.

Assumes the wrapped encoder's output has shape
``[B, ..., d_model]`` — i.e. the feature dimension is last.  All
in-repo encoders satisfy this.
"""

import torch
import torch.nn as nn

from .base import ModalityAutoEncoder, ModalityEncoder


class _VariationalEncoder(ModalityEncoder):
    """Wrap a deterministic encoder with (mu, logvar) linear heads.

    ``forward(x)`` returns ``mu`` so callers that expect
    ``ae.encoder(x)`` to return a latent tensor need no changes.
    Use ``.distribution(x)`` during training to get
    ``(mu, logvar)``.
    """

    def __init__(self, base: ModalityEncoder):
        super().__init__(base.n_channels, base.d_model, base.n_tokens)
        self.base = base
        self.mu_head = nn.Linear(base.d_model, base.d_model)
        self.logvar_head = nn.Linear(base.d_model, base.d_model)

    def forward(self, x):
        h = self.base(x)
        return self.mu_head(h)

    def distribution(self, x):
        h = self.base(x)
        return self.mu_head(h), self.logvar_head(h)


class VariationalWrapper(ModalityAutoEncoder):
    """Wrap a deterministic ``ModalityAutoEncoder`` as a VAE.

    * ``.encoder(x)`` returns ``mu`` — deterministic, drop-in for the
      wrapped AE's encoder.
    * ``.encoder.distribution(x)`` returns ``(mu, logvar)``.
    * ``forward(x)`` returns ``(recon, mu, logvar)`` in every mode.
      During ``model.train()`` the reconstruction is decoded from a
      reparameterised sample; during ``model.eval()`` it is decoded
      from ``mu``.  The existing trainer ``output = output[0]``
      shortcut extracts the reconstruction.
    """

    def __init__(self, base: ModalityAutoEncoder):
        super().__init__(base.n_channels, base.d_model, base.n_tokens)
        self.encoder = _VariationalEncoder(base.encoder)
        self.decoder = base.decoder

    def forward(self, x):
        mu, logvar = self.encoder.distribution(x)
        if self.training:
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
        else:
            z = mu
        output_length = x.shape[-1]
        recon = self.decoder(z, output_shape=output_length)
        return recon, mu, logvar


def kl_divergence_standard_normal(
    mu: torch.Tensor, logvar: torch.Tensor,
) -> torch.Tensor:
    """KL(N(mu, sigma^2) || N(0, I)) averaged over the batch.

    Sums across all latent dimensions of each sample then averages
    across the batch.  Returns a scalar.
    """
    kl_per_sample = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return kl_per_sample.flatten(1).sum(dim=1).mean()
