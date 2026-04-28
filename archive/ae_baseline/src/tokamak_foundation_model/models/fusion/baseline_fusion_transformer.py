import torch
import torch.nn as nn

class BaselineFusionTransformer(nn.Module):
    """
    Baseline transformer for joint latent feature fusion and prediction.
    Concatenates tokens from all modalities and processes them with a
    standard causal transformer.

    Parameters
    ----------
    d_model : int, optional
        Model dimension, by default 512
    n_heads : int, optional
        Number of attention heads, by default 8
    n_layers : int, optional
        Number of transformer layers, by default 6
    dropout : float, optional
        Dropout rate, by default 0.1
    n_modalities : int, optional
        Number of input modalities for learned modality embeddings, by default 5
    max_tokens : int, optional
        Maximum total number of tokens across all modalities, by default 1024
    verbose : bool, optional
        If True, print debug information during initialization, by default False

    Attributes
    ----------
    modality_embeddings : nn.Embedding
        Learned embedding added per modality to distinguish token sources
    position_embeddings : nn.Embedding
        Learned positional embeddings over token sequence
    transformer : nn.TransformerEncoder
        Stack of causal transformer encoder layers
    norm : nn.LayerNorm
        Final layer norm
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        dropout: float = 0.1,
        n_modalities: int = 5,
        max_tokens: int = 1024,
        verbose: bool = False
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.n_modalities = n_modalities
        self.max_tokens = max_tokens
        self.verbose = verbose

        # Learned modality embeddings (one per modality)
        self.modality_embeddings = nn.Embedding(n_modalities, d_model)

        # Learned positional embeddings over full token sequence
        self.position_embeddings = nn.Embedding(max_tokens, d_model)

        # Standard transformer encoder layer with pre-LayerNorm
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True   # pre-LayerNorm (more stable)
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model)
        )

        if self.verbose:
            print(f"BaselineFusionTransformer:")
            print(f"  d_model:      {d_model}")
            print(f"  n_heads:      {n_heads}")
            print(f"  n_layers:     {n_layers}")
            print(f"  n_modalities: {n_modalities}")
            print(f"  max_tokens:   {max_tokens}")

    def _causal_mask(self, n_tokens: int, device: torch.device) -> torch.Tensor:
        """
        Generate causal attention mask.

        Parameters
        ----------
        n_tokens : int
            Number of tokens in the sequence
        device : torch.device
            Device to create mask on

        Returns
        -------
        torch.Tensor
            Causal mask of shape [n_tokens, n_tokens] where future
            positions are masked with -inf
        """
        return torch.triu(
            torch.full((n_tokens, n_tokens), float('-inf'), device=device),
            diagonal=1
        )

    def forward(self, token_list: list[tuple[torch.Tensor, int]]) -> torch.Tensor:
        """
        Fuse and process tokens from all modalities.

        Parameters
        ----------
        token_list : list of tuple of (torch.Tensor, int)
            Each entry is (tokens, modality_id) where:
            - tokens has shape [batch, n_tokens, d_model]
            - modality_id is an integer index for the modality embedding

        Returns
        -------
        torch.Tensor
            Transformer output of shape [batch, total_tokens, d_model]
        """
        B = token_list[0][0].shape[0]
        device = token_list[0][0].device

        # Concatenate all modality tokens
        all_tokens = []
        for tokens, modality_id in token_list:
            # Add modality embedding
            mod_emb = self.modality_embeddings(
                torch.tensor(modality_id, device=device)
            )
            tokens = tokens + mod_emb
            all_tokens.append(tokens)

        x = torch.cat(all_tokens, dim=1)    # [B, total_tokens, d_model]

        # Add positional embeddings
        n_tokens = x.shape[1]
        positions = torch.arange(n_tokens, device=device)
        x = x + self.position_embeddings(positions)

        # Causal mask
        mask = self._causal_mask(n_tokens, device)

        # Transformer forward pass
        x = self.transformer(x, mask=mask)  # [B, total_tokens, d_model]

        return x


if __name__ == "__main__":
    d_model = 512
    B = 4

    transformer = BaselineFusionTransformer(
        d_model=d_model,
        n_heads=8,
        n_layers=6,
        n_modalities=7,
        max_tokens=1024,
        verbose=True
    )

    # Dummy encoder outputs
    ts_tokens   = torch.randn(B, 100, d_model)  # TimeSeriesEncoder
    sp_tokens   = torch.randn(B, 10,  d_model)  # SpatialProfileEncoder
    vid_tokens  = torch.randn(B, 192, d_model)  # VideoEncoder (VIS)
    ir_tokens   = torch.randn(B, 192, d_model)  # VideoEncoder (IR)
    spec_tokens = torch.randn(B, 50,  d_model)  # SpectrogramEncoder
    text_tokens = torch.randn(B, 20,  d_model)  # TextEncoder

    token_list = [
        (ts_tokens,   0),  # modality 0: time series
        (sp_tokens,   1),  # modality 1: spatial profile
        (vid_tokens,  2),  # modality 2: visible camera
        (ir_tokens,   3),  # modality 3: IR camera
        (spec_tokens, 4),  # modality 4: spectrogram
        (text_tokens, 5),  # modality 5: text
    ]

    out = transformer(token_list)
    print(f"Input tokens:  {sum(t.shape[1] for t, _ in token_list)}")  # 564
    print(f"Output shape:  {out.shape}")   # [4, 564, 512]
