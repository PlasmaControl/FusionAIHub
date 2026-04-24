import torch
import torch.nn as nn


def sinusoidal_time_encoding(t_ms: torch.Tensor, d_model: int) -> torch.Tensor:
    """
    Compute sinusoidal positional encoding from continuous timestamps.

    Parameters
    ----------
    t_ms : torch.Tensor
        Timestamps in milliseconds, shape [B, T].
    d_model : int
        Model dimension (must be even).

    Returns
    -------
    torch.Tensor
        Positional encodings, shape [B, T, d_model].
    """
    half_d = d_model // 2
    device = t_ms.device
    freqs = torch.pow(
        torch.tensor(10000.0, device=device),
        -torch.arange(half_d, device=device, dtype=torch.float32) / half_d,
    )
    angles = t_ms.unsqueeze(-1) * freqs  # [B, T, half_d]
    return torch.cat([angles.sin(), angles.cos()], dim=-1)  # [B, T, d_model]


class ModalityTokenizer(nn.Module):
    """
    Projects per-modality AE latent tokens to a common dimension and adds
    modality and continuous-time positional embeddings.

    Each modality's AE encoder outputs tokens of shape [B, T_mod, d_lat].
    This module:
      1. Projects d_lat → d_model via a per-modality linear layer.
      2. Adds a learned per-modality embedding.
      3. Adds a sinusoidal encoding of the absolute center time (in ms) of
         each token within the context window.
    All modality token sequences are then concatenated along the token axis.

    Parameters
    ----------
    modality_configs : dict
        Mapping ``{name: {"d_lat": int, "n_tokens": int}}``.
        ``d_lat`` is the AE encoder output dimension; ``n_tokens`` is the
        number of temporal tokens produced by that AE for one context window.
    d_model : int
        Common model dimension for the downstream Perceiver.
    window_ms : float, optional
        Duration of the context window in milliseconds. Default 500.0.
    """

    def __init__(
        self,
        modality_configs: dict,
        d_model: int,
        window_ms: float = 500.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.window_ms = window_ms
        self.modality_names = list(modality_configs.keys())
        self.modality_to_idx = {
            name: i for i, name in enumerate(self.modality_names)
        }

        self.projections = nn.ModuleDict(
            {
                name: nn.Linear(cfg["d_lat"], d_model, bias=False)
                for name, cfg in modality_configs.items()
            }
        )

        self.modality_embedding = nn.Embedding(len(modality_configs), d_model)

    def forward(self, latents: dict) -> torch.Tensor:
        """
        Tokenize and embed per-modality AE latents.

        Parameters
        ----------
        latents : dict
            Mapping ``{name: Tensor[B, T_mod, d_lat]}``.
            Modalities absent from the dict are silently skipped, so batches
            with missing diagnostics are handled gracefully.

        Returns
        -------
        torch.Tensor
            Shape ``[B, N_total, d_model]`` where
            ``N_total = sum(T_mod for each present modality)``.
        """
        token_chunks = []

        for name, z in latents.items():
            B, T, _ = z.shape

            # 1. Project to common d_model
            proj = self.projections[name](z)  # [B, T, d_model]

            # 2. Add learned modality embedding
            mod_idx = torch.tensor(
                self.modality_to_idx[name], device=z.device
            )
            proj = proj + self.modality_embedding(mod_idx)  # broadcast [B, T, D]

            # 3. Add continuous-time PE (center of each token's time span in ms)
            centers = (
                torch.arange(T, device=z.device, dtype=torch.float32) + 0.5
            ) / T * self.window_ms  # [T]
            t_ms = centers.unsqueeze(0).expand(B, -1)  # [B, T]
            proj = proj + sinusoidal_time_encoding(t_ms, self.d_model)

            token_chunks.append(proj)

        return torch.cat(token_chunks, dim=1)  # [B, N_total, d_model]


class ActuatorTokenizer(nn.Module):
    """
    Tokenize raw actuator time series into transformer tokens via patch
    embedding (strided 1D convolution).

    Each actuator group (e.g. ``pin``, ``ech_power``, ``gas_flow``) is
    independently projected from ``[B, C, T_samples]`` to
    ``[B, N_patches, d_model]`` using a per-group Conv1d with
    ``kernel_size=stride=patch_len``.  Learned actuator-type embeddings
    and sinusoidal time encodings are added before concatenation.

    Parameters
    ----------
    actuator_configs : dict
        ``{name: {"n_channels": int, "patch_len": int}}``.
        ``n_channels`` is the number of raw channels for this actuator
        group; ``patch_len`` is the number of samples per patch.
    d_model : int
        Output token dimension.
    """

    def __init__(
        self,
        actuator_configs: dict,
        d_model: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.actuator_names = list(actuator_configs.keys())
        self.actuator_to_idx = {
            name: i for i, name in enumerate(self.actuator_names)
        }
        self.configs = actuator_configs

        self.patch_embeddings = nn.ModuleDict({
            name: nn.Conv1d(
                in_channels=cfg["n_channels"],
                out_channels=d_model,
                kernel_size=cfg["patch_len"],
                stride=cfg["patch_len"],
            )
            for name, cfg in actuator_configs.items()
        })

        self.actuator_embedding = nn.Embedding(len(actuator_configs), d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        actuator_signals: dict,
        offset_ms: float = 0.0,
    ) -> torch.Tensor:
        """
        Tokenize raw actuator signals.

        Parameters
        ----------
        actuator_signals : dict
            ``{name: Tensor[B, C, T_samples]}``.  Missing groups are
            silently skipped.
        offset_ms : float
            Absolute time offset in milliseconds for the start of the
            window.  Used to compute sinusoidal time PE so that the same
            signal at different absolute times gets distinct encodings.

        Returns
        -------
        torch.Tensor
            Shape ``[B, N_act_total, d_model]``.
        """
        token_chunks = []

        for name, sig in actuator_signals.items():
            if name not in self.patch_embeddings:
                continue
            cfg = self.configs[name]
            B = sig.shape[0]
            patch_len = cfg["patch_len"]
            fs = cfg["target_fs"]

            # Patch embedding: [B, C, T] → [B, d_model, N_patches] → [B, N_patches, d_model]
            tokens = self.patch_embeddings[name](sig).transpose(1, 2)
            N_patches = tokens.shape[1]

            # Actuator-type embedding
            idx = torch.tensor(
                self.actuator_to_idx[name], device=sig.device
            )
            tokens = tokens + self.actuator_embedding(idx)

            centers_s = (
                torch.arange(N_patches, device=sig.device, dtype=torch.float32)
                + 0.5
            ) * patch_len / fs  # seconds
            centers_ms = centers_s * 1000.0 + offset_ms  # absolute ms
            t_ms = centers_ms.unsqueeze(0).expand(B, -1)  # [B, N_patches]
            tokens = tokens + sinusoidal_time_encoding(t_ms, self.d_model)

            token_chunks.append(tokens)

        if not token_chunks:
            # Return empty token sequence if no actuators present
            B = next(iter(actuator_signals.values())).shape[0]
            return torch.zeros(B, 0, self.d_model,
                               device=next(iter(actuator_signals.values())).device)

        out = torch.cat(token_chunks, dim=1)  # [B, N_act_total, d_model]
        return self.norm(out)
