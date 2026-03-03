import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed2d(nn.Module):
    """Convert (B, C, Fr, T) spectrogram into a sequence of patch embeddings."""

    def __init__(self, n_channels: int, d_model: int,
                 patch_h: int = 8, patch_w: int = 8):
        super().__init__()
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.proj = nn.Linear(n_channels * patch_h * patch_w, d_model)

    def forward(self, x):
        # x: (B, C, Fr, T)
        B, C, Fr, T = x.shape
        ph, pw = self.patch_h, self.patch_w
        n_h, n_w = Fr // ph, T // pw
        # (B, C, n_h, ph, n_w, pw) -> (B, n_h, n_w, C, ph, pw) -> (B, N, C*ph*pw)
        x = x.reshape(B, C, n_h, ph, n_w, pw)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, n_h * n_w, C * ph * pw)
        return self.proj(x), (n_h, n_w)


class PatchUnembed2d(nn.Module):
    """Reconstruct (B, C, Fr, T) from patch token sequence."""

    def __init__(self, n_channels: int, d_model: int,
                 patch_h: int = 8, patch_w: int = 8):
        super().__init__()
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.n_channels = n_channels
        self.proj = nn.Linear(d_model, n_channels * patch_h * patch_w)

    def forward(self, x, n_h: int, n_w: int):
        # x: (B, N, d_model)
        B = x.shape[0]
        ph, pw = self.patch_h, self.patch_w
        x = self.proj(x)  # (B, N, C*ph*pw)
        x = x.reshape(B, n_h, n_w, self.n_channels, ph, pw)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(
            B, self.n_channels, n_h * ph, n_w * pw
        )
        return x


class SpectrogramTransformerEncoder(nn.Module):
    """AST-style transformer encoder for multichannel spectrograms."""

    def __init__(self, n_channels: int, d_model: int = 256,
                 n_heads: int = 4, n_layers: int = 4,
                 patch_h: int = 14, patch_w: int = 14,
                 max_patches: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.patch_embed = PatchEmbed2d(n_channels, d_model, patch_h, patch_w)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
        )

    def forward(self, x):
        # x: (B, C, Fr, T)
        tokens, (n_h, n_w) = self.patch_embed(x)  # (B, N, d_model)
        N = tokens.shape[1]
        tokens = tokens + self.pos_embed[:, :N]
        tokens = self.transformer(tokens)
        return tokens, (n_h, n_w)


class SpectrogramTransformerDecoder(nn.Module):
    """Lightweight transformer decoder that reconstructs patches."""

    def __init__(self, n_channels: int, d_model: int = 256,
                 n_heads: int = 4, n_layers: int = 2,
                 patch_h: int = 14, patch_w: int = 14,
                 max_patches: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, max_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            decoder_layer, num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.patch_unembed = PatchUnembed2d(n_channels, d_model, patch_h, patch_w)

    def forward(self, tokens, n_h: int, n_w: int):
        N = tokens.shape[1]
        tokens = tokens + self.pos_embed[:, :N]
        tokens = self.transformer(tokens)
        return self.patch_unembed(tokens, n_h, n_w)


class SpectrogramBaselineAutoEncoder(nn.Module):
    """Multichannel Audio Spectrogram Transformer autoencoder.

    Patchifies the (B, C, Fr, T) input into non-overlapping 2D patches,
    encodes with a ViT-style transformer, and decodes with a lighter
    transformer decoder back to the original shape.

    Parameters
    ----------
    n_channels : int
        Number of spectrogram channels (e.g. 4 for CO2, 8 for MHR, 48 for ECE).
    d_model : int
        Transformer hidden dimension.
    n_heads : int
        Number of attention heads.
    n_enc_layers : int
        Number of encoder transformer layers.
    n_dec_layers : int
        Number of decoder transformer layers.
    patch_h, patch_w : int
        Patch size along frequency and time axes.
    dropout : float
        Dropout rate.
    """

    def __init__(self, n_channels: int, d_model: int = 256,
                 n_heads: int = 4, n_enc_layers: int = 4,
                 n_dec_layers: int = 2, patch_h: int = 14,
                 patch_w: int = 14, dropout: float = 0.1, **kwargs):
        super().__init__()
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.n_channels = n_channels

        self.encoder = SpectrogramTransformerEncoder(
            n_channels=n_channels, d_model=d_model, n_heads=n_heads,
            n_layers=n_enc_layers, patch_h=patch_h, patch_w=patch_w,
            dropout=dropout,
        )
        self.decoder = SpectrogramTransformerDecoder(
            n_channels=n_channels, d_model=d_model, n_heads=n_heads,
            n_layers=n_dec_layers, patch_h=patch_h, patch_w=patch_w,
            dropout=dropout,
        )

    def forward(self, x):
        B, C, Fr, T = x.shape
        ph, pw = self.patch_h, self.patch_w

        # Pad to patch-aligned dimensions
        pad_fr = (ph - Fr % ph) % ph
        pad_t = (pw - T % pw) % pw
        if pad_fr > 0 or pad_t > 0:
            x_padded = F.pad(x, (0, pad_t, 0, pad_fr))
        else:
            x_padded = x

        latent, (n_h, n_w) = self.encoder(x_padded)
        reconstructed = self.decoder(latent, n_h, n_w)

        # Crop back to original dims
        reconstructed = reconstructed[:, :C, :Fr, :T]
        return reconstructed, latent


# Alias kept for backwards compatibility with older checkpoints/configs.
SpectrogramResLSTMAutoEncoder = SpectrogramBaselineAutoEncoder


def _run_test(label, n_channels, freq, time, device, **kwargs):
    print(f"=== {label} (n_channels={n_channels}) ===")
    autoencoder = SpectrogramBaselineAutoEncoder(n_channels, **kwargs)
    autoencoder.to(device)

    n_params = sum(p.numel() for p in autoencoder.parameters())
    print(f"  Parameters: {n_params:,}")

    x = torch.randn(1, n_channels, freq, time)

    with torch.inference_mode():
        reconstructed, latent = autoencoder(x.to(device))
    reconstructed = reconstructed.cpu()
    assert reconstructed.shape == x.shape, f"Shape mismatch: {reconstructed.shape} vs {x.shape}"

    latent = latent.cpu().detach()
    input_size = n_channels * freq * time
    latent_size = latent.numel()
    ratio = input_size / latent_size

    print(f"  Input:   {x.shape}  ({input_size:,} values)")
    print(f"  Latent:  {list(latent.shape)}  ({latent_size:,} values)")
    print(f"  Output:  {reconstructed.shape}")
    print(f"  Compression: {ratio:.1f}:1")
    print()


if __name__ == "__main__":
    # python -m tokamak_foundation_model.models.modality.spectrogram_baseline

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _run_test("CO2", n_channels=4, freq=128, time=256, device=device,
              d_model=256, n_enc_layers=4, n_dec_layers=2)
    _run_test("MHR", n_channels=8, freq=129, time=100, device=device,
              d_model=256, n_enc_layers=4, n_dec_layers=2)
    _run_test("ECE", n_channels=48, freq=129, time=100, device=device,
              d_model=256, n_enc_layers=4, n_dec_layers=2)
