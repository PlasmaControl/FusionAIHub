#!/usr/bin/env python
"""
Training script for the Perceiver Foundation Model.

Pipeline per training sample
-----------------------------
1. Load a 550 ms chunk from the multi-file dataset.
2. Split it into a 500 ms context window [0, 500 ms] and a 500 ms target
   window shifted by dt = 50 ms, i.e. [50 ms, 550 ms].
3. Encode every diagnostic signal through its frozen, pre-trained AE encoder.
4. Extract actuator vectors as channel-means over the 50 ms boundary windows.
5. The foundation model encodes the context latents (Perceiver encoder +
   processor) and predicts the next latent via the dynamics model.
6. The target latent is computed from the target window with stop-gradient.
7. MSE loss is backpropagated through the foundation model only (AEs frozen).
"""

from pathlib import Path
import argparse
import logging
import random
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader

from tokamak_foundation_model.data.multi_file_dataset import (
    TokamakMultiFileDataset, make_dataloader,
)
from tokamak_foundation_model.models.model_factory import build_model
from tokamak_foundation_model.models.latent_feature_space.foundation_model import (
    PerceiverFoundationModel,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Diagnostic signal configurations
#
# Each entry specifies how to build the AE and tokenizer for one modality.
# Fields:
#   model_type  : key in MODEL_REGISTRY (fast_time_series | profile | ...)
#   n_channels  : number of input channels for the AE
#   d_lat       : AE encoder output dimension (= d_model of that AE)
#   n_tokens    : temporal tokens produced by the AE for a 500 ms window
#   target_fs   : signal sampling frequency in Hz (used for window splitting)
#   ae_kwargs   : extra kwargs forwarded to build_model
# ---------------------------------------------------------------------------
DIAGNOSTIC_CONFIGS: dict = {
    "filterscopes": {
        "model_type": "fast_time_series",
        "n_channels": 8,
        "d_lat": 16,
        "n_tokens": 32,
        "target_fs": 10_000,
        "ae_kwargs": {"input_length": 500,
                      "kernel_size": 3,
                      },
    },
    "ts_core_density": {
        "model_type": "slow_time_series",
        "n_channels": 44,
        "d_lat": 16,
        "n_tokens": 4,
        "target_fs": 100,
        "ae_kwargs": {},
    },
    "ts_core_temp": {
        "model_type": "slow_time_series",
        "n_channels": 44,
        "d_lat": 16,
        "n_tokens": 4,
        "target_fs": 100,
        "ae_kwargs": {},
    },
    "ts_tangential_density": {
        "model_type": "slow_time_series",
        "n_channels": 10,
        "d_lat": 8,
        "n_tokens": 4,
        "target_fs": 100,
        "ae_kwargs": {},
    },
    "ts_tangential_temp": {
        "model_type": "slow_time_series",
        "n_channels": 10,
        "d_lat": 8,
        "n_tokens": 4,
        "target_fs": 100,
        "ae_kwargs": {},
    },
    "mse": {
        "model_type": "profile",
        "n_channels": 1,
        "d_lat": 16,
        "n_tokens": 4,
        "target_fs": 100,
        "ae_kwargs": {"n_spatial_points": 69},
    },
    "cer_ti": {
        "model_type": "profile",
        "n_channels": 1,
        "d_lat": 16,
        "n_tokens": 4,
        "target_fs": 100,
        "ae_kwargs": {"n_spatial_points": 48},
    },
    "cer_rot": {
        "model_type": "profile",
        "n_channels": 1,
        "d_lat": 16,
        "n_tokens": 4,
        "target_fs": 100,
        "ae_kwargs": {"n_spatial_points": 48},
    },
    # "co2": {
    #     "model_type": "spectrogram_channel_ast",
    #     "n_channels": 4,
    #     "d_lat": 256,
    #     "n_tokens": 248,  # 4 channels × 62 frames (500ms @ 500kHz, n_fft=256, hop=256, fw=16)
    #     "target_fs": 500_000,
    #     "ae_checkpoint_path": "/projects/EKOLEMEN/foundation_model/spectrogram_co2_d256/checkpoint.pth",
    #     "ae_kwargs": {
    #         "freq_bins": 128,
    #         "frame_width": 16,
    #         "n_enc_layers": 4,
    #         "n_dec_layers": 4,
    #         "n_heads": 4,
    #         "time_conv_kernel": 7,
    #     },
    #     # Requires: n_fft=256, hop_length=256 in dataset (not default 1024/256)
    #     # Decoder interface: needs (tokens, n_channels, n_frames, T_orig)
    #     #   — visualization code must handle spectrogram decode separately
    # },
}

# Actuator signals — used as raw control inputs, not encoded by an AE.
# target_fs is only needed to compute the boundary mean.
# channels_to_use: optional list of valid channel indices (from stats audit).
#   Channels with NaN/Inf stats or zero range are excluded.
#   Removed entirely: ech_tor_angle (all broken), ech_pol_angle (all broken),
#   ich (missing from stats).
ACTUATOR_CONFIGS: dict = {
    "pin": {"target_fs": 10_000, "n_channels": 8, "patch_len": 200},
    "tin": {"target_fs": 10_000, "n_channels": 8, "patch_len": 200},
    "beam_voltage": {"target_fs": 10_000, "n_channels": 8, "patch_len": 200},
    "ech_power": {"target_fs": 10_000, "n_channels": 4, "patch_len": 200,
                  "channels_to_use": [5, 7, 8, 10]},
    "gas_flow": {"target_fs": 10_000, "n_channels": 7, "patch_len": 200,
                 "channels_to_use": [0, 1, 2, 3, 4, 6, 7]},
    "rmp": {"target_fs": 10_000, "n_channels": 11, "patch_len": 200,
            "channels_to_use": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
}

DT_S: float = 0.05   # prediction step  (50 ms)
WINDOW_S: float = 0.05  # context window  (50 ms)
N_ROLLOUT: int = 8    # autoregressive rollout steps for training
N_ROLLOUT_VIS: int = 16  # rollout steps for visualization
CHUNK_S: float = WINDOW_S + N_ROLLOUT * DT_S  # total chunk needed
CHUNK_VIS_S: float = WINDOW_S + N_ROLLOUT_VIS * DT_S  # viz chunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _select_channels(sig: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Select valid channels from a signal tensor based on config.

    If the config contains ``channels_to_use``, index into the channel
    dimension (dim=1) to keep only those channels.  Otherwise return the
    tensor unchanged.
    """
    ch = cfg.get("channels_to_use")
    if ch is not None:
        return sig[:, ch]
    return sig


def load_ae(name: str, cfg: dict, checkpoint_path: Path) -> nn.Module:
    """Build an AE, load weights, freeze, return in eval mode."""
    model = build_model(
        cfg["model_type"],
        d_model=cfg["d_lat"],
        n_tokens=cfg["n_tokens"],
        n_channels=cfg["n_channels"],
        **cfg.get("ae_kwargs", {}),
    )
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = raw.get("model_state_dict", raw)
    model.load_state_dict(state)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    for p in model.encoder.parameters():
        p.requires_grad_(True)
    logger.info(f"Loaded AE for '{name}' from {checkpoint_path}")
    return model


def split_window(
    signal: torch.Tensor,
    target_fs: float,
    n_rollout: int = N_ROLLOUT,
) -> tuple:
    """
    Split a signal into a context window and *n_rollout* target windows,
    each shifted by DT_S from the previous.

    Parameters
    ----------
    signal : torch.Tensor
        Shape ``[..., n_total]``.
    target_fs : float
        Sampling frequency (Hz).
    n_rollout : int
        Number of rollout target windows.

    Returns
    -------
    context : torch.Tensor
        Shape ``[..., n_context]``.
    targets : list of torch.Tensor
        *n_rollout* tensors, each shape ``[..., n_context]``.
        ``targets[k]`` is shifted by ``(k+1) * DT_S`` from the start.
    """
    n_ctx = round(WINDOW_S * target_fs)
    n_dt = round(DT_S * target_fs)
    context = signal[..., :n_ctx]
    targets = []
    for k in range(1, n_rollout + 1):
        offset = k * n_dt
        targets.append(signal[..., offset:offset + n_ctx])
    return context, targets


def actuator_vectors(
    batch: dict,
    configs: dict,
    stats: dict,
    n_rollout: int = N_ROLLOUT,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    Extract actuator vector pairs for each rollout step.

    For step k, ``act_curr`` is the mean over the DT_S window ending at
    the context boundary + k*DT_S, and ``act_fut`` is the mean over the
    next DT_S window.

    Returns
    -------
    list of (act_curr, act_fut) tuples
        Length *n_rollout*, each element is a pair of ``[B, n_act_total]``.
    """
    # Collect per-step, per-actuator vectors
    step_pairs = [[] for _ in range(n_rollout)]

    for name, cfg in configs.items():
        if name not in batch:
            continue
        sig = _select_channels(batch[name], cfg)  # [B, C, n_total]
        fs = cfg["target_fs"]
        n_ctx = round(WINDOW_S * fs)
        n_dt = round(DT_S * fs)

        for k in range(n_rollout):
            # Window for step k: curr ends at n_ctx + k*n_dt
            boundary = n_ctx + k * n_dt
            curr = sig[:, :, boundary - n_dt:boundary].mean(dim=-1)
            fut = sig[:, :, boundary:boundary + n_dt].mean(dim=-1)
            # Clean NaN/Inf only — no normalization
            curr[~torch.isfinite(curr)] = 0.0
            fut[~torch.isfinite(fut)] = 0.0

            step_pairs[k].append((curr, fut))

    if not step_pairs[0]:
        raise RuntimeError("No actuator signals found in batch.")

    # Concatenate across actuators for each step
    result = []
    for k in range(n_rollout):
        act_curr = torch.cat([p[0] for p in step_pairs[k]], dim=-1)
        act_fut = torch.cat([p[1] for p in step_pairs[k]], dim=-1)
        result.append((act_curr, act_fut))

    return result


def _normalize_actuator(
    sig: torch.Tensor,
    name: str,
    stats: dict,
    channels_to_use: Optional[list] = None,
) -> torch.Tensor:
    """Clean NaN/Inf from actuator signal.  No normalization for now.

    Min-max normalization was destroying signal structure because extreme
    outliers in the dataset stats (e.g. pin max=3M) squashed all typical
    values to ~0.  The Conv1d patch embedding in ActuatorTokenizer can
    learn to handle raw scales directly.
    """
    sig = sig.clone()
    sig[~torch.isfinite(sig)] = 0.0
    return sig


def actuator_context_window(
    batch: dict,
    configs: dict,
    stats: dict,
    offset_s: float = 0.0,
) -> dict:
    """
    Extract standardized actuator signals over a WINDOW_S window.

    Parameters
    ----------
    batch : dict
        Batch dict containing actuator signals.
    configs : dict
        Actuator configuration dict.
    stats : dict
        Preprocessing statistics.
    offset_s : float
        Start time of the window in seconds.  Default ``0.0`` extracts
        the context window ``[0, WINDOW_S]``.

    Returns
    -------
    dict
        ``{name: Tensor[B, C, T_ctx_samples]}`` for each actuator group.
    """
    result = {}
    for name, cfg in configs.items():
        if name not in batch:
            continue
        sig = _select_channels(batch[name], cfg)
        fs = cfg["target_fs"]
        n_ctx = round(WINDOW_S * fs)
        n_off = round(offset_s * fs)
        ctx = sig[:, :, n_off:n_off + n_ctx].clone()
        result[name] = _normalize_actuator(
            ctx, name, stats, channels_to_use=cfg.get("channels_to_use"))
    return result


def actuator_step_windows(
    batch: dict,
    configs: dict,
    stats: dict,
    n_rollout: int = N_ROLLOUT,
) -> list[tuple[dict, dict]]:
    """
    Extract per-step raw actuator signal windows for cross-attention dynamics.

    For each rollout step k, returns the current and future ``DT_S``
    windows as dicts of ``{name: [B, C, T_step_samples]}``.

    Returns
    -------
    list of (act_curr_signals, act_fut_signals)
        Length *n_rollout*.
    """
    result = []
    for k in range(n_rollout):
        curr_dict = {}
        fut_dict = {}
        for name, cfg in configs.items():
            if name not in batch:
                continue
            sig = _select_channels(batch[name], cfg)
            fs = cfg["target_fs"]
            n_ctx = round(WINDOW_S * fs)
            n_dt = round(DT_S * fs)

            boundary = n_ctx + k * n_dt
            curr = sig[:, :, boundary - n_dt:boundary].clone()
            fut = sig[:, :, boundary:boundary + n_dt].clone()

            ch = cfg.get("channels_to_use")
            curr_dict[name] = _normalize_actuator(curr, name, stats,
                                                  channels_to_use=ch)
            fut_dict[name] = _normalize_actuator(fut, name, stats,
                                                 channels_to_use=ch)
        result.append((curr_dict, fut_dict))
    return result


def masked_channel_mean(
    sig: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """Compute channel mean, excluding masked (invalid) elements.

    Parameters
    ----------
    sig : torch.Tensor
        Signal of shape ``(C, T)``.
    mask : torch.Tensor or None
        Boolean mask of shape ``(C, T)`` where ``True`` = valid.

    Returns
    -------
    np.ndarray
        Shape ``(T,)`` — mean over valid channels at each time step.
    """
    if mask is None:
        return sig.mean(dim=0).numpy()
    m = mask.float()
    n_valid = m.sum(dim=0).clamp(min=1)
    return ((sig * m).sum(dim=0) / n_valid).numpy()


def ae_decode(
    ae: nn.Module,
    tokens: torch.Tensor,
    cfg: dict,
    output_length: int,
    ae_token_stats: Optional[dict] = None,
    modality_name: Optional[str] = None,
) -> torch.Tensor:
    """Decode AE tokens back to signal space, handling both interfaces.

    If *ae_token_stats* is provided and *modality_name* is given,
    de-normalizes the tokens (``tokens * std + mean``) before passing
    them to the frozen AE decoder.
    """
    if ae_token_stats is not None and modality_name in ae_token_stats:
        mean = ae_token_stats[modality_name]["mean"].to(tokens.device)
        std = ae_token_stats[modality_name]["std"].to(tokens.device)
        tokens = tokens * std + mean
    if hasattr(ae, 'frame_width'):
        n_ch = cfg["n_channels"]
        n_fr = tokens.shape[1] // n_ch
        return ae.decode(tokens, n_ch, n_fr, output_length)
    return ae.decoder(tokens, output_shape=output_length)


@torch.no_grad()
def encode_batch(
    ae_encoders: dict,
    signals: dict,
    ae_token_stats: Optional[dict] = None,
) -> dict:
    """Run frozen AE encoders; returns ``{name: [B, n_tokens, d_lat]}``.

    If *ae_token_stats* is provided, standardize each modality's tokens
    to zero mean and unit variance using precomputed statistics.
    """
    result = {}
    for name, ae in ae_encoders.items():
        if name not in signals:
            continue
        z = ae.encoder(signals[name])
        # Clamp to prevent extreme values (e.g. from all-zero missing
        # signals) that would cause NaN in downstream attention layers.
        z = z.clamp(-50, 50)
        if ae_token_stats is not None and name in ae_token_stats:
            mean = ae_token_stats[name]["mean"].to(z.device)
            std = ae_token_stats[name]["std"].to(z.device)
            z = (z - mean) / std
        result[name] = z
    return result


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

@torch.no_grad()
def visualize_predictions(
    model: PerceiverFoundationModel,
    ae_models: dict,
    loader: DataLoader,
    epoch: int,
    save_dir: Path,
    preprocess_stats: Optional[dict] = None,
    label: str = "val",
    ae_token_stats: Optional[dict] = None,
) -> None:
    """Generate diagnostic plots from the validation set.

    Always visualises the same fixed sample (first sample of the first
    batch, with the loader seeded deterministically) so that plots are
    directly comparable across epochs.

    Produces a single figure with:

    * **Top rows** (one per diagnostic):
        (a) Raw channel-mean signal over the full 550 ms chunk.
        (b) AE reconstruction vs original (channel-mean of context).
        (c) AE latent token heatmap: context (top) vs target (bottom).
    * **Row 4**: Perceiver latent heatmaps — target | predicted | difference.
    * **Row 5**: Context latent | copy-baseline error | scatter plot of
      model MSE vs copy-baseline MSE over *all* validation samples.
    """
    model.eval()
    plot_dir = save_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Pass 1: iterate over ALL val batches to collect per-sample MSEs
    # ------------------------------------------------------------------
    all_pred_mse = []
    all_copy_mse = []
    fixed_batch = None

    for batch in loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        ctx_signals = {}
        tgt_signals_steps = [{} for _ in range(N_ROLLOUT_VIS)]
        for name, cfg in DIAGNOSTIC_CONFIGS.items():
            if name not in batch:
                continue
            ctx, tgts = split_window(
                batch[name], cfg["target_fs"], n_rollout=N_ROLLOUT_VIS)
            ctx_signals[name] = ctx
            for k, tgt in enumerate(tgts):
                tgt_signals_steps[k][name] = tgt

        if not ctx_signals:
            continue

        # Use first step for single-step metrics
        tgt_signals = tgt_signals_steps[0]
        use_cross_attn = model.dynamics_type in ("cross_attention", "gru")
        if use_cross_attn:
            act_ctx = actuator_context_window(
                batch, ACTUATOR_CONFIGS, preprocess_stats)
            act_step_pairs = actuator_step_windows(
                batch, ACTUATOR_CONFIGS, preprocess_stats,
                n_rollout=N_ROLLOUT_VIS)
        else:
            act_ctx = None
            act_pairs = actuator_vectors(
                batch, ACTUATOR_CONFIGS, preprocess_stats,
                n_rollout=N_ROLLOUT_VIS)

        lat_ctx = encode_batch(ae_models, ctx_signals, ae_token_stats)
        lat_tgt = encode_batch(ae_models, tgt_signals, ae_token_stats)

        latent = model.encode(lat_ctx, act_ctx)
        if use_cross_attn:
            act_curr_sig, act_fut_sig = act_step_pairs[0]
            offset_ms = WINDOW_S * 1000
            lat_pred = model.dynamics(
                latent, act_curr_sig, act_fut_sig,
                offset_ms=offset_ms, dt_ms=DT_S * 1000,
            )
        else:
            act_curr, act_fut = act_pairs[0]
            lat_pred = model.dynamics(latent, act_curr, act_fut)
        # EMA target uses actuator context from the target's time window
        if use_cross_attn:
            act_ctx_tgt = actuator_context_window(
                batch, ACTUATOR_CONFIGS, preprocess_stats,
                offset_s=DT_S)
        else:
            act_ctx_tgt = None
        lat_target = model.encode(lat_tgt, act_ctx_tgt)
        lat_context = model.encode(lat_ctx, act_ctx)

        pred_mse = ((lat_pred - lat_target) ** 2).mean(dim=(1, 2))   # [B]
        copy_mse = ((lat_context - lat_target) ** 2).mean(dim=(1, 2))  # [B]
        all_pred_mse.append(pred_mse.cpu())
        all_copy_mse.append(copy_mse.cpu())

        # Keep the first batch for the fixed-sample plots
        if fixed_batch is None:
            # Decode predicted latent → AE tokens → signals
            ae_tokens_pred = model.decode(lat_pred)
            signal_preds = {}
            for name, tokens in ae_tokens_pred.items():
                if name in tgt_signals:
                    out_len = tgt_signals[name].shape[-1]
                    signal_preds[name] = ae_decode(
                        ae_models[name], tokens,
                        DIAGNOSTIC_CONFIGS[name], out_len,
                        ae_token_stats=ae_token_stats,
                        modality_name=name)

            # Decoder roundtrip: encode TARGET through online
            # Perceiver, decode back → AE decode.  Isolates
            # decoder quality from dynamics quality.
            lat_tgt_online = model.encode(lat_tgt, act_ctx)
            ae_tokens_roundtrip = model.decode(lat_tgt_online)
            signal_roundtrip = {}
            for name, tokens in ae_tokens_roundtrip.items():
                if name in tgt_signals:
                    out_len = tgt_signals[name].shape[-1]
                    signal_roundtrip[name] = ae_decode(
                        ae_models[name], tokens,
                        DIAGNOSTIC_CONFIGS[name], out_len,
                        ae_token_stats=ae_token_stats,
                        modality_name=name)

            fixed_batch = {
                "batch": batch,
                "ctx_signals": ctx_signals,
                "tgt_signals": tgt_signals,
                "lat_ctx": lat_ctx,
                "lat_tgt": lat_tgt,
                "lat_pred": lat_pred,
                "lat_target": lat_target,
                "lat_context": lat_context,
                "signal_preds": signal_preds,
                "signal_roundtrip": signal_roundtrip,
                "act_ctx": act_ctx,
                "act_pairs": act_pairs if not use_cross_attn else None,
                "act_step_pairs": act_step_pairs if use_cross_attn else None,
            }

    all_pred_mse = torch.cat(all_pred_mse).numpy()
    all_copy_mse = torch.cat(all_copy_mse).numpy()

    if fixed_batch is None:
        return

    # Unpack fixed sample data
    batch = fixed_batch["batch"]
    ctx_signals = fixed_batch["ctx_signals"]
    tgt_signals = fixed_batch["tgt_signals"]
    lat_ctx = fixed_batch["lat_ctx"]
    lat_pred = fixed_batch["lat_pred"]
    lat_target = fixed_batch["lat_target"]
    lat_context = fixed_batch["lat_context"]

    idx = 0  # always the same sample
    diag_names = [n for n in DIAGNOSTIC_CONFIGS if n in ctx_signals]
    n_diag = len(diag_names)

    # ------------------------------------------------------------------
    # Build figure
    # ------------------------------------------------------------------
    n_rows = n_diag + 2
    fig, axes = plt.subplots(
        n_rows, 3, figsize=(16, 3.2 * n_rows),
        gridspec_kw={"hspace": 0.45, "wspace": 0.3},
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    # ---- Per-diagnostic rows ----
    for row, name in enumerate(diag_names):
        cfg = DIAGNOSTIC_CONFIGS[name]
        fs = cfg["target_fs"]
        ctx_sig = ctx_signals[name][idx].cpu()

        # Grab mask for this sample (if available)
        mask_key = f"{name}_mask"
        full_mask = batch.get(mask_key)
        if full_mask is not None:
            full_mask_i = full_mask[idx].cpu()
            n_ctx_pts = ctx_sig.shape[-1]
            ctx_mask = full_mask_i[..., :n_ctx_pts]
        else:
            full_mask_i = None
            ctx_mask = None

        # (a) Raw signal — masked channel mean over full chunk
        ax = axes[row, 0]
        full_sig = batch[name][idx].cpu()
        t_full = np.arange(full_sig.shape[-1]) / fs * 1000
        ax.plot(t_full, masked_channel_mean(full_sig, full_mask_i),
                color="C0", linewidth=0.8)
        ax.axvline(WINDOW_S * 1000, color="red", linewidth=1, linestyle="--",
                    label="ctx|tgt boundary")
        ax.set_title(f"{name} — raw signal (channel mean)")
        ax.set_xlabel("time [ms]")
        ax.legend(fontsize=7)

        # (b) AE reconstruction vs original (context, masked channel mean)
        ax = axes[row, 1]
        ae = ae_models[name]
        recon = ae(ctx_signals[name][idx:idx+1]).cpu()[0]
        t_ctx = np.arange(ctx_sig.shape[-1]) / fs * 1000
        if ctx_mask is not None:
            m = ctx_mask.float()
            n_v = m.sum().clamp(min=1)
            ae_mse = float(((ctx_sig - recon) ** 2 * m).sum() / n_v)
        else:
            ae_mse = float(((ctx_sig - recon) ** 2).mean())

        ax.plot(t_ctx, masked_channel_mean(ctx_sig, ctx_mask),
                color="C0", linewidth=1, label="original")
        ax.plot(t_ctx, masked_channel_mean(recon, ctx_mask),
                color="C3", linewidth=1, linestyle="--", label="AE recon")
        ax.set_title(f"{name} — AE reconstruction (MSE={ae_mse:.4f})")
        ax.set_xlabel("time [ms]")
        ax.legend(fontsize=7)

        # (c) Predicted vs actual target signal (masked channel mean)
        ax = axes[row, 2]
        signal_preds = fixed_batch["signal_preds"]
        tgt_sig = tgt_signals[name][idx].cpu()
        n_dt = round(DT_S * fs)
        tgt_mask = full_mask_i[..., n_dt:n_dt + tgt_sig.shape[-1]] \
            if full_mask_i is not None else None
        t_tgt = np.arange(tgt_sig.shape[-1]) / fs * 1000 + DT_S * 1000

        ax.plot(t_tgt, masked_channel_mean(tgt_sig, tgt_mask),
                color="C0", linewidth=1, label="actual target")
        signal_roundtrip = fixed_batch["signal_roundtrip"]
        if name in signal_preds:
            pred_sig = signal_preds[name][idx].detach().cpu()
            if tgt_mask is not None:
                m = tgt_mask.float()
                n_v = m.sum().clamp(min=1)
                pred_mse = float(((pred_sig - tgt_sig) ** 2 * m).sum() / n_v)
            else:
                pred_mse = float(((pred_sig - tgt_sig) ** 2).mean())
            ax.plot(t_tgt, masked_channel_mean(pred_sig, tgt_mask),
                    color="C1", linewidth=1, linestyle="--", label="predicted")
            title = f"{name} — pred={pred_mse:.4f}"
        else:
            title = f"{name} — target (no prediction)"

        # Decoder roundtrip: target → Perceiver enc → Perceiver dec → AE dec
        if name in signal_roundtrip:
            rt_sig = signal_roundtrip[name][idx].detach().cpu()
            if tgt_mask is not None:
                m = tgt_mask.float()
                n_v = m.sum().clamp(min=1)
                rt_mse = float(((rt_sig - tgt_sig) ** 2 * m).sum() / n_v)
            else:
                rt_mse = float(((rt_sig - tgt_sig) ** 2).mean())
            ax.plot(t_tgt, masked_channel_mean(rt_sig, tgt_mask),
                    color="C2", linewidth=1, linestyle=":",
                    label="enc→dec (no dyn)")
            title += f", roundtrip={rt_mse:.4f}"

        ax.set_title(title, fontsize=8)
        ax.set_xlabel("time [ms]")
        ax.legend(fontsize=7)

    # ---- Row n_diag: Perceiver latent — target | predicted | diff ----
    p = lat_pred[idx].cpu().numpy()
    t = lat_target[idx].cpu().numpy()
    diff = p - t
    vmax = max(np.percentile(np.abs(p), 95), np.percentile(np.abs(t), 95))
    d_show = min(64, p.shape[1])

    for col, (data, title) in enumerate([
        (t, "Target Perceiver latent"),
        (p, "Predicted Perceiver latent"),
    ]):
        ax = axes[n_diag, col]
        im = ax.imshow(data[:, :d_show], aspect="auto", cmap="RdBu_r",
                        vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_title(title)
        ax.set_ylabel("query index")
        ax.set_xlabel(f"dim (first {d_show})")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[n_diag, 2]
    diff_vmax = np.percentile(np.abs(diff[:, :d_show]), 95)
    im = ax.imshow(diff[:, :d_show], aspect="auto", cmap="RdBu_r",
                    vmin=-diff_vmax, vmax=diff_vmax, interpolation="nearest")
    mse_val = float((diff ** 2).mean())
    ax.set_title(f"Prediction error, MSE={mse_val:.6f}")
    ax.set_ylabel("query index")
    ax.set_xlabel(f"dim (first {d_show})")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ---- Row n_diag+1: context latent | copy error | scatter plot ----
    c = lat_context[idx].cpu().numpy()
    copy_diff = c - t

    ax = axes[n_diag + 1, 0]
    im = ax.imshow(c[:, :d_show], aspect="auto", cmap="RdBu_r",
                    vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_title("Context Perceiver latent (dynamics input)")
    ax.set_ylabel("query index")
    ax.set_xlabel(f"dim (first {d_show})")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[n_diag + 1, 1]
    copy_vmax = np.percentile(np.abs(copy_diff[:, :d_show]), 95)
    copy_mse_val = float((copy_diff ** 2).mean())
    im = ax.imshow(copy_diff[:, :d_show], aspect="auto", cmap="RdBu_r",
                    vmin=-copy_vmax, vmax=copy_vmax, interpolation="nearest")
    ax.set_title(f"Copy baseline error, MSE={copy_mse_val:.6f}")
    ax.set_ylabel("query index")
    ax.set_xlabel(f"dim (first {d_show})")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Scatter: model prediction MSE vs copy-baseline MSE (all val samples)
    ax = axes[n_diag + 1, 2]
    ax.scatter(all_copy_mse, all_pred_mse, s=15, alpha=0.6, color="C0",
               edgecolors="none")
    # Diagonal = model same as copy baseline
    lim_max = max(all_copy_mse.max(), all_pred_mse.max()) * 1.1
    ax.plot([0, lim_max], [0, lim_max], "k--", linewidth=0.8, label="y = x")
    ax.set_xlim(0, lim_max)
    ax.set_ylim(0, lim_max)
    ax.set_aspect("equal")
    ax.set_xlabel("Copy-baseline MSE")
    ax.set_ylabel("Model prediction MSE")
    ax.set_title("All val samples: model vs copy baseline")
    ax.legend(fontsize=7)
    # Annotate how many samples the model wins on
    n_wins = int((all_pred_mse < all_copy_mse).sum())
    n_total = len(all_pred_mse)
    ax.text(0.05, 0.95, f"Model wins: {n_wins}/{n_total}",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    fig.suptitle(f"Epoch {epoch} ({label})", fontsize=14, fontweight="bold")
    fig.savefig(plot_dir / f"diagnostics_{label}_epoch{epoch:03d}.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Autoregressive rollout: stitched continuous timeline
    #
    # Context (500ms) is shown as-is, then each rollout step appends
    # the last DT_S (50ms) of new predicted signal, building a
    # continuous prediction that extends N_ROLLOUT_VIS*DT_S beyond
    # context.  Ground truth is overlaid as far as data is available.
    # ------------------------------------------------------------------
    lat_ctx_single = {name: t[idx:idx+1] for name, t in fixed_batch["lat_ctx"].items()}
    act_ctx = fixed_batch["act_ctx"]
    act_ctx_single = (
        {name: t[idx:idx+1] for name, t in act_ctx.items()}
        if act_ctx is not None else None
    )
    latent = model.encode(lat_ctx_single, act_ctx_single)

    use_cross_attn = model.dynamics_type in ("cross_attention", "gru")
    stored_act_pairs = fixed_batch["act_pairs"]
    stored_act_step_pairs = fixed_batch["act_step_pairs"]

    # Collect the last DT_S of each rolled-out step's decoded signal
    rollout_tails = {name: [] for name in diag_names}
    latent_prev = latent  # first step: no history
    for step in range(N_ROLLOUT_VIS):
        prev_for_next = latent
        if use_cross_attn:
            if step < len(stored_act_step_pairs):
                act_curr_sig, act_fut_sig = stored_act_step_pairs[step]
            else:
                act_curr_sig, act_fut_sig = stored_act_step_pairs[-1]
            ac_s = {n: t[idx:idx+1] for n, t in act_curr_sig.items()}
            af_s = {n: t[idx:idx+1] for n, t in act_fut_sig.items()}
            offset_ms = WINDOW_S * 1000 + step * DT_S * 1000
            latent = model.dynamics(
                latent, ac_s, af_s,
                offset_ms=offset_ms, dt_ms=DT_S * 1000,
                latent_prev=latent_prev,
            )
        else:
            if step < len(stored_act_pairs):
                ac, af = stored_act_pairs[step]
            else:
                ac, af = stored_act_pairs[-1]
            latent = model.dynamics(latent, ac[idx:idx+1], af[idx:idx+1])
        latent_prev = prev_for_next
        ae_tok = model.decode(latent)
        for name in diag_names:
            cfg = DIAGNOSTIC_CONFIGS[name]
            fs = cfg["target_fs"]
            n_dt = round(DT_S * fs)
            n_ctx = round(WINDOW_S * fs)
            sig = ae_decode(
                ae_models[name], ae_tok[name],
                cfg, n_ctx,
                ae_token_stats=ae_token_stats,
                modality_name=name)[0].detach().cpu()
            # Get mask for this signal if available
            sig_mask_key = f"{name}_mask"
            if sig_mask_key in batch:
                # Use context-region mask (channels don't change over time)
                sig_mask = batch[sig_mask_key][idx].cpu()[..., :n_ctx]
            else:
                sig_mask = None
            rollout_tails[name].append(
                masked_channel_mean(sig, sig_mask)[-n_dt:])

    fig_roll, axes_roll = plt.subplots(
        len(diag_names), 1, figsize=(14, 3.5 * len(diag_names)),
        squeeze=False,
    )
    for row, name in enumerate(diag_names):
        ax = axes_roll[row, 0]
        cfg = DIAGNOSTIC_CONFIGS[name]
        fs = cfg["target_fs"]

        # Ground truth: full chunk (masked channel mean)
        full_sig = batch[name][idx].cpu()
        sig_mask_key = f"{name}_mask"
        full_mask_i = batch[sig_mask_key][idx].cpu() \
            if sig_mask_key in batch else None
        gt = masked_channel_mean(full_sig, full_mask_i)
        t_full = np.arange(len(gt)) / fs * 1000

        # Context: decoded from encoder (masked channel mean)
        ctx_sig_raw = ctx_signals[name][idx].cpu()
        ctx_mask = full_mask_i[..., :ctx_sig_raw.shape[-1]] \
            if full_mask_i is not None else None
        ctx_mean = masked_channel_mean(ctx_sig_raw, ctx_mask)
        t_ctx = np.arange(len(ctx_mean)) / fs * 1000

        # Stitch prediction: context + rolled-out tails
        pred_parts = [ctx_mean]
        for tail in rollout_tails[name]:
            pred_parts.append(tail)
        pred_stitched = np.concatenate(pred_parts)
        t_pred = np.arange(len(pred_stitched)) / fs * 1000

        ax.plot(t_full, gt, color="C0", linewidth=1, label="ground truth")
        ax.plot(t_pred, pred_stitched, color="C1", linewidth=1,
                linestyle="--", label="context + rollout")
        ax.axvline(WINDOW_S * 1000, color="red", linewidth=1,
                    linestyle=":", alpha=0.7, label="prediction starts")
        ax.set_title(f"{name} — {N_ROLLOUT_VIS}-step rollout "
                     f"(masked channel mean)")
        ax.set_xlabel("time [ms]")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    fig_roll.suptitle(f"Epoch {epoch} ({label}) — Autoregressive rollout",
                      fontsize=14, fontweight="bold")
    fig_roll.tight_layout()
    fig_roll.savefig(plot_dir / f"rollout_{label}_epoch{epoch:03d}.png", dpi=150,
                     bbox_inches="tight")
    plt.close(fig_roll)
    logger.info(f"  Plots saved to {plot_dir}")


# ---------------------------------------------------------------------------
# Train / val loops
# ---------------------------------------------------------------------------

def run_epoch(
    model: PerceiverFoundationModel,
    ae_models: dict,
    loader: DataLoader,
    optimizer: Optional[optim.Optimizer],
    is_train: bool,
    encode_loss_weight: float = 0.0,
    rollout_loss_weight: float = 2.0,
    signal_loss_weight: float = 0.1,
    recon_loss_weight: float = 1.0,
    delta_loss_weight: float = 1.0,
    max_steps: Optional[int] = None,
    preprocess_stats: Optional[dict] = None,
    n_rollout: int = N_ROLLOUT,
    rollout_noise_std: float = 0.0,
    teacher_forcing_ratio: float = 0.0,
    context_noise_std: float = 0.0,
    context_drop_rate: float = 0.0,
    zero_actuators: bool = False,
    ae_token_stats: Optional[dict] = None,
) -> tuple[float, float, float, float, float, float]:
    """Run one training or validation epoch.

    Encode loss: online encoder vs EMA encoder on the same context input.
    Reconstruction loss (logged as "rec"): encode context AE tokens through
        the Perceiver encoder, decode back via the Perceiver decoder, and
        compare with the original AE tokens.  Trains the encoder+decoder
        bottleneck to preserve information, independent of dynamics.
    Signal loss (logged as "sig"): dynamics-predicted latent vs EMA-encoded
        target at future steps in Perceiver latent space.
    Rollout loss (logged as "roll"): decode the dynamics-predicted latent
        back to AE token space via the Perceiver decoder and compare against
        the frozen AE encoder outputs on the ground-truth target signals.
        Gradients flow through encoder → dynamics → decoder and targets are
        independent of the model's own weights (frozen AE space).
    Delta loss (logged as "dlt"): MSE between the predicted displacement
        (dynamics output − context latent) and the target displacement
        (EMA target − EMA context).  Subtracts out the DC component so
        that copy (zero delta) is explicitly penalized whenever the target
        changes, no matter how small.
    Teacher forcing: with probability ``teacher_forcing_ratio``, the
        dynamics-predicted latent is replaced with the encoder applied to
        the ground-truth target AE tokens (no grad).  This teaches
        accurate single-step dynamics before the model has to handle error
        accumulation.  Decayed to 0 over training.
    """
    model.train(is_train)
    sum_enc, sum_roll, sum_sig, sum_recon, sum_delta, n = (
        0.0, 0.0, 0.0, 0.0, 0.0, 0)

    for batch in loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # Ablation: zero actuator signals to test their impact
        if zero_actuators:
            for name in ACTUATOR_CONFIGS:
                if name in batch and isinstance(batch[name], torch.Tensor):
                    batch[name] = torch.zeros_like(batch[name])

        # Split each diagnostic into context + n_rollout target windows
        ctx_signals = {}
        tgt_signals_steps = [{} for _ in range(n_rollout)]  # list of dicts
        tgt_masks_steps = [{} for _ in range(n_rollout)]    # element masks
        for name, cfg in DIAGNOSTIC_CONFIGS.items():
            if name not in batch:
                continue
            ctx, tgts = split_window(batch[name], cfg["target_fs"],
                                     n_rollout=n_rollout)
            ctx_signals[name] = ctx
            for k, tgt in enumerate(tgts):
                tgt_signals_steps[k][name] = tgt
            # Split element mask the same way if present
            mask_key = f"{name}_mask"
            if mask_key in batch:
                _, mask_tgts = split_window(
                    batch[mask_key].float(), cfg["target_fs"],
                    n_rollout=n_rollout)
                for k, m in enumerate(mask_tgts):
                    tgt_masks_steps[k][name] = m > 0.5

        if not ctx_signals:
            continue

        # Actuator extraction depends on dynamics type
        use_cross_attn = model.dynamics_type in ("cross_attention", "gru")
        if use_cross_attn:
            act_ctx = actuator_context_window(
                batch, ACTUATOR_CONFIGS, preprocess_stats)
            act_step_pairs = actuator_step_windows(
                batch, ACTUATOR_CONFIGS, preprocess_stats,
                n_rollout=n_rollout)
        else:
            act_ctx = None
            act_pairs = actuator_vectors(
                batch, ACTUATOR_CONFIGS, preprocess_stats,
                n_rollout=n_rollout)

        with torch.no_grad():
            lat_ctx = encode_batch(ae_models, ctx_signals, ae_token_stats)
            lat_tgt_steps = [encode_batch(ae_models, tgt_s, ae_token_stats)
                             for tgt_s in tgt_signals_steps]

        # Corrupt context tokens during training to prevent copy behavior.
        # Targets stay clean so the loss signal is meaningful.
        # Noise is scaled relative to each modality's token std so that
        # context_noise_std=0.1 means 10% of the token scale.
        if is_train and (context_noise_std > 0 or context_drop_rate > 0):
            lat_ctx_input = {}
            for name, tokens in lat_ctx.items():
                t = tokens.clone()
                if context_noise_std > 0:
                    token_std = t.detach().std().clamp(min=1e-6)
                    t = t + (context_noise_std * token_std
                             ) * torch.randn_like(t)
                if context_drop_rate > 0:
                    # Drop entire tokens (zero out) with given probability
                    mask = torch.rand(t.shape[:2], device=t.device
                                      ).unsqueeze(-1) > context_drop_rate
                    t = t * mask
                lat_ctx_input[name] = t
        else:
            lat_ctx_input = lat_ctx

        if is_train:
            # Per-step actuator contexts: each EMA target should see the
            # actuator signals from its own time window, not the initial
            # context window.  Target step k covers
            # [(k+1)*DT_S, (k+1)*DT_S + WINDOW_S].
            if use_cross_attn:
                with torch.no_grad():
                    act_ctx_steps = [
                        actuator_context_window(
                            batch, ACTUATOR_CONFIGS, preprocess_stats,
                            offset_s=(k + 1) * DT_S)
                        for k in range(n_rollout)
                    ]
            else:
                act_ctx_steps = [None] * n_rollout

            # Precompute teacher-forced latents for scheduled sampling.
            # Uses detached online encoder (no EMA co-adaptation).
            if teacher_forcing_ratio > 0:
                with torch.no_grad():
                    teacher_latents = [
                        model.encode(lat_tgt_steps[k], act_ctx_steps[k]).detach()
                        for k in range(n_rollout)
                    ]
            else:
                teacher_latents = None

            # Encode context (corrupted during training, clean at val)
            latent = model.encode(lat_ctx_input, act_ctx)

            # Detached online encoder as reference (no EMA co-adaptation).
            with torch.no_grad():
                lat_ctx_ema = model.encode(lat_ctx_input, act_ctx).detach()
            loss_encode = torch.tensor(0.0, device=device)

            # Fixed reference points for delta loss (detached — gradients
            # flow only through the dynamics output, not the reference).
            latent_context = latent.detach()

            # Reconstruction loss: decode(encode(ctx)) ≈ ctx AE tokens.
            # Trains the encoder+decoder bottleneck to preserve information.
            loss_recon = torch.tensor(0.0, device=device)
            if recon_loss_weight > 0:
                ae_tokens_recon = model.decode(latent)
                n_recon = 0
                for name, tokens_recon in ae_tokens_recon.items():
                    if name not in lat_ctx:
                        continue
                    tgt = lat_ctx[name]
                    tgt_var = tgt.detach().var().clamp(min=1e-6)
                    loss_recon = loss_recon + F.mse_loss(
                        tokens_recon, tgt) / tgt_var
                    n_recon += 1
                if n_recon > 0:
                    loss_recon = loss_recon / n_recon

            loss_rollout = torch.tensor(0.0, device=device)
            loss_signal = torch.tensor(0.0, device=device)
            loss_delta = torch.tensor(0.0, device=device)
            n_mod = 0  # number of modalities in decode-space rollout loss

            # Precompute target latents: detached online encoder.
            with torch.no_grad():
                lat_tgt_encoded = [
                    model.encode(lat_tgt_steps[k], act_ctx_steps[k]).detach()
                    for k in range(n_rollout)
                ]

            # Autoregressive rollout: chain dynamics n_rollout steps
            latent_prev = latent  # first step: no history
            for k in range(n_rollout):
                prev_for_next = latent  # save before dynamics step
                if use_cross_attn:
                    act_curr_sig, act_fut_sig = act_step_pairs[k]
                    offset_ms = WINDOW_S * 1000 + k * DT_S * 1000
                    latent = model.dynamics(
                        latent, act_curr_sig, act_fut_sig,
                        offset_ms=offset_ms, dt_ms=DT_S * 1000,
                        latent_prev=latent_prev,
                    )
                else:
                    act_curr, act_fut = act_pairs[k]
                    latent = model.dynamics(latent, act_curr, act_fut)

                # Direct latent prediction loss — bypasses decoder.
                lat_target = lat_tgt_encoded[k]
                lat_tgt_var = lat_target.detach().var().clamp(min=1e-6)
                step_weight = (k + 1) / n_rollout
                loss_signal = loss_signal + step_weight * F.mse_loss(
                    latent, lat_target) / lat_tgt_var

                # Delta loss: compare predicted displacement from context
                # against target displacement.
                if delta_loss_weight > 0:
                    delta_pred = latent - latent_context
                    delta_target = (lat_target - lat_ctx_ema).detach()
                    delta_var = delta_target.var().clamp(min=1e-4)
                    loss_delta = loss_delta + step_weight * F.mse_loss(
                        delta_pred, delta_target) / delta_var

                # Decode-space rollout loss.
                if rollout_loss_weight > 0:
                    ae_tokens_pred = model.decode(latent)
                    n_mod = 0
                    for rname, tokens_pred in ae_tokens_pred.items():
                        if rname not in lat_tgt_steps[k]:
                            continue
                        tgt_tokens = lat_tgt_steps[k][rname]
                        tgt_tok_var = tgt_tokens.detach().var().clamp(min=1e-6)
                        loss_rollout = loss_rollout + step_weight * F.mse_loss(
                            tokens_pred, tgt_tokens) / tgt_tok_var
                        n_mod += 1

                # Update history buffer, then teacher-force or inject noise.
                latent_prev = prev_for_next
                if k < n_rollout - 1:
                    if (teacher_latents is not None
                            and random.random() < teacher_forcing_ratio):
                        latent = teacher_latents[k].detach()
                        # When teacher-forced, prev becomes the teacher
                        # latent so the next step sees consistent history.
                        latent_prev = latent
                    elif rollout_noise_std > 0:
                        latent = latent + rollout_noise_std * torch.randn_like(
                            latent)

            if rollout_loss_weight > 0 and n_rollout > 0:
                loss_rollout = loss_rollout / (n_rollout * max(n_mod, 1))
            loss_signal = loss_signal / max(n_rollout, 1)
            if delta_loss_weight > 0 and n_rollout > 0:
                loss_delta = loss_delta / n_rollout

            loss = (encode_loss_weight * loss_encode
                    + recon_loss_weight * loss_recon
                    + rollout_loss_weight * loss_rollout
                    + signal_loss_weight * loss_signal
                    + delta_loss_weight * loss_delta)

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning("NaN/Inf loss detected — skipping batch")
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            # EMA update removed — using detached online encoder as target
        else:
            with torch.no_grad():
                # Per-step actuator contexts for EMA targets
                if use_cross_attn:
                    act_ctx_steps = [
                        actuator_context_window(
                            batch, ACTUATOR_CONFIGS, preprocess_stats,
                            offset_s=(k + 1) * DT_S)
                        for k in range(n_rollout)
                    ]
                else:
                    act_ctx_steps = [None] * n_rollout

                latent = model.encode(lat_ctx, act_ctx)

                # Detached online encoder as reference (no EMA).
                lat_ctx_ema = model.encode(lat_ctx, act_ctx)
                loss_encode = torch.tensor(0.0, device=device)

                latent_context = latent  # reference for delta loss (no grad needed in val)

                # Reconstruction loss
                loss_recon = torch.tensor(0.0, device=device)
                if recon_loss_weight > 0:
                    ae_tokens_recon = model.decode(latent)
                    n_recon = 0
                    for name, tokens_recon in ae_tokens_recon.items():
                        if name not in lat_ctx:
                            continue
                        tgt = lat_ctx[name]
                        tgt_var = tgt.var().clamp(min=1e-6)
                        loss_recon = loss_recon + F.mse_loss(
                            tokens_recon, tgt) / tgt_var
                        n_recon += 1
                    if n_recon > 0:
                        loss_recon = loss_recon / n_recon

                loss_rollout = torch.tensor(0.0, device=device)
                loss_signal = torch.tensor(0.0, device=device)
                loss_delta = torch.tensor(0.0, device=device)
                n_mod = 0

                lat_tgt_encoded = [
                    model.encode(lat_tgt_steps[k], act_ctx_steps[k])
                    for k in range(n_rollout)
                ]

                latent_prev = latent  # first step: no history
                for k in range(n_rollout):
                    prev_for_next = latent
                    if use_cross_attn:
                        act_curr_sig, act_fut_sig = act_step_pairs[k]
                        offset_ms = WINDOW_S * 1000 + k * DT_S * 1000
                        latent = model.dynamics(
                            latent, act_curr_sig, act_fut_sig,
                            offset_ms=offset_ms, dt_ms=DT_S * 1000,
                            latent_prev=latent_prev,
                        )
                    else:
                        act_curr, act_fut = act_pairs[k]
                        latent = model.dynamics(latent, act_curr, act_fut)
                    latent_prev = prev_for_next

                    # Direct latent prediction loss (later steps weighted more)
                    lat_target = lat_tgt_encoded[k]
                    lat_tgt_var = lat_target.var().clamp(min=1e-6)
                    step_weight = (k + 1) / n_rollout
                    loss_signal = loss_signal + step_weight * F.mse_loss(
                        latent, lat_target) / lat_tgt_var

                    # Delta loss (matches training branch)
                    if delta_loss_weight > 0:
                        delta_pred = latent - latent_context
                        delta_target = lat_target - lat_ctx_ema
                        delta_var = delta_target.var().clamp(min=1e-4)
                        loss_delta = loss_delta + step_weight * F.mse_loss(
                            delta_pred, delta_target) / delta_var

                    # Decode-space rollout loss (matches training branch)
                    if rollout_loss_weight > 0:
                        ae_tokens_pred = model.decode(latent)
                        n_mod = 0
                        for rname, tokens_pred in ae_tokens_pred.items():
                            if rname not in lat_tgt_steps[k]:
                                continue
                            tgt_tokens = lat_tgt_steps[k][rname]
                            tgt_tok_var = tgt_tokens.var().clamp(min=1e-6)
                            loss_rollout = loss_rollout + step_weight * F.mse_loss(
                                tokens_pred, tgt_tokens) / tgt_tok_var
                            n_mod += 1

                if rollout_loss_weight > 0 and n_rollout > 0:
                    loss_rollout = loss_rollout / (n_rollout * max(n_mod, 1))
                loss_signal = loss_signal / max(n_rollout, 1)
                if delta_loss_weight > 0 and n_rollout > 0:
                    loss_delta = loss_delta / n_rollout

        sum_enc += loss_encode.item()
        sum_recon += loss_recon.item()
        sum_roll += loss_rollout.item()
        sum_sig += loss_signal.item()
        sum_delta += loss_delta.item()
        n += 1

        if max_steps and n >= max_steps:
            break

    d = max(n, 1)
    total = (sum_enc + sum_recon + sum_roll + sum_sig + sum_delta) / d

    # --- Dynamics diagnostics: run once on a single batch at end of epoch ---
    if not is_train and n_rollout > 0:
        _log_dynamics_diagnostics(
            model, ae_models, loader, preprocess_stats, n_rollout,
            ae_token_stats=ae_token_stats)

    return (total, sum_enc / d, sum_recon / d, sum_roll / d,
            sum_sig / d, sum_delta / d)


@torch.no_grad()
def _log_dynamics_diagnostics(
    model: PerceiverFoundationModel,
    ae_models: dict,
    loader,
    preprocess_stats,
    n_rollout: int,
    ae_token_stats: Optional[dict] = None,
) -> None:
    """Log per-step delta norms, target delta norms, and decoded cos-sim.

    Runs on the first batch of the loader only.  Helps distinguish:
    - Dynamics producing zero deltas (delta norm ≈ 0)
    - Dynamics producing deltas but decoder collapsing them (cos_sim ≈ 1)
    - Target deltas being small (target too similar to context)
    """
    model.eval()
    use_cross_attn = model.dynamics_type in ("cross_attention", "gru")

    for batch in loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # Split signals
        ctx_signals = {}
        tgt_signals_steps = [{} for _ in range(n_rollout)]
        for name, cfg in DIAGNOSTIC_CONFIGS.items():
            if name not in batch:
                continue
            ctx, tgts = split_window(
                batch[name], cfg["target_fs"], n_rollout=n_rollout)
            ctx_signals[name] = ctx
            for k, tgt in enumerate(tgts):
                tgt_signals_steps[k][name] = tgt
        if not ctx_signals:
            return

        lat_ctx = encode_batch(ae_models, ctx_signals)

        if use_cross_attn:
            act_ctx = actuator_context_window(
                batch, ACTUATOR_CONFIGS, preprocess_stats)
            act_step_pairs = actuator_step_windows(
                batch, ACTUATOR_CONFIGS, preprocess_stats,
                n_rollout=n_rollout)
            act_ctx_steps = [
                actuator_context_window(
                    batch, ACTUATOR_CONFIGS, preprocess_stats,
                    offset_s=(k + 1) * DT_S)
                for k in range(n_rollout)
            ]
        else:
            act_ctx = None
            act_ctx_steps = [None] * n_rollout

        latent = model.encode(lat_ctx, act_ctx)
        lat_ctx_ema = model.encode(lat_ctx, act_ctx)
        latent_context = latent.clone()

        delta_norms = []
        tgt_delta_norms = []
        model_cos_sims = []
        gt_cos_sims = []
        prev_decoded = None
        prev_tgt_flat = None
        latent_prev = latent  # first step: no history

        for k in range(n_rollout):
            prev_latent = latent.clone()

            if use_cross_attn:
                act_curr_sig, act_fut_sig = act_step_pairs[k]
                offset_ms = WINDOW_S * 1000 + k * DT_S * 1000
                latent = model.dynamics(
                    latent, act_curr_sig, act_fut_sig,
                    offset_ms=offset_ms, dt_ms=DT_S * 1000,
                    latent_prev=latent_prev)
            else:
                return  # MLP mode — skip diagnostics
            latent_prev = prev_latent

            # Per-step delta norm
            delta = latent - prev_latent
            delta_norms.append(delta.norm(dim=-1).mean().item())

            # Target delta norm (how much the target actually changes)
            lat_tgt = encode_batch(ae_models, tgt_signals_steps[k], ae_token_stats)
            lat_tgt_enc = model.encode(lat_tgt, act_ctx_steps[k])
            tgt_delta = lat_tgt_enc - lat_ctx_ema
            tgt_delta_norms.append(tgt_delta.norm(dim=-1).mean().item())

            # Model decoded output (AE token space)
            ae_tok = model.decode(latent)
            B = latent.shape[0]
            flat = torch.cat(
                [t.reshape(B, -1) for t in ae_tok.values()], dim=1)

            # Ground truth AE tokens
            tgt_flat = torch.cat(
                [lat_tgt[m].reshape(B, -1) for m in ae_tok if m in lat_tgt],
                dim=1)

            # Consecutive cos-sim: model predictions vs ground truth
            if prev_decoded is not None:
                model_cos = F.cosine_similarity(flat, prev_decoded, dim=1)
                model_cos_sims.append(model_cos.mean().item())
            if prev_tgt_flat is not None:
                gt_cos = F.cosine_similarity(tgt_flat, prev_tgt_flat, dim=1)
                gt_cos_sims.append(gt_cos.mean().item())
            prev_decoded = flat
            prev_tgt_flat = tgt_flat

        # Log results
        dn_str = " ".join(f"{v:.3f}" for v in delta_norms)
        tn_str = " ".join(f"{v:.3f}" for v in tgt_delta_norms)
        mc_str = " ".join(f"{v:.4f}" for v in model_cos_sims)
        gc_str = " ".join(f"{v:.4f}" for v in gt_cos_sims)
        lat_norm = latent_context.norm(dim=-1).mean().item()
        logger.info(
            f"  [dynamics diag] latent_norm={lat_norm:.2f}  "
            f"delta_norms=[{dn_str}]  "
            f"tgt_delta_norms=[{tn_str}]  "
            f"model_cos_sim=[{mc_str}]  "
            f"gt_cos_sim=[{gc_str}]"
        )
        return  # first batch only


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Perceiver Foundation Model")
    parser.add_argument(
        "--data_dir", required=False,
        help="Directory of HDF5 shot files",
        default="/scratch/gpfs/EKOLEMEN/foundation_model/")
    parser.add_argument(
        "--stats_path",
        default="/projects/EKOLEMEN/foundation_model/preprocessing_stats.pt")
    parser.add_argument(
        "--ae_checkpoint_dir", required=False,
        help="Directory containing per-modality AE checkpoints. "
             "Expected filenames: <signal>_<model_type>/checkpoint_best.pth",
        default="/projects/EKOLEMEN/foundation_model/"
    )
    parser.add_argument(
        "--ae_token_stats_path", default=None,
        help="Path to ae_token_stats.pt for per-modality token "
             "normalization. If None, no normalization is applied."
    )
    parser.add_argument("--checkpoint_dir", default="runs/foundation_model")
    parser.add_argument("--d_model", type=int, default=512,
                        help="Perceiver model dimension")
    parser.add_argument("--n_latent", type=int, default=128,
                        help="Number of Perceiver latent queries")
    parser.add_argument("--encoder_layers", type=int, default=1)
    parser.add_argument("--processor_layers", type=int, default=2)
    parser.add_argument("--decoder_layers", type=int, default=3)
    parser.add_argument("--decoder_self_attn_layers", type=int, default=0,
                        help="Self-attention layers in the Perceiver decoder "
                             "per modality (0 = cross-attention only).")
    parser.add_argument("--dynamics_layers", type=int, default=3)
    parser.add_argument("--zero_actuators", action="store_true", default=False,
                        help="Zero out all actuator signals. Use to ablate "
                             "whether actuators help the dynamics.")
    parser.add_argument("--dynamics_type", type=str, default="cross_attention",
                        choices=["mlp", "cross_attention", "gru"],
                        help="Dynamics model type: 'cross_attention' (recommended), "
                             "'cross_attention', or 'mlp' (legacy)")
    parser.add_argument("--ema_decay", type=float, default=0.996,
                        help="EMA decay for JEPA target encoder")
    parser.add_argument("--encode_loss_weight", type=float, default=0.0,
                        help="Weight for encode loss. Set to 0 when using "
                             "detached online encoder instead of EMA target.")
    parser.add_argument("--rollout_loss_weight", type=float, default=2.0,
                        help="Weight for rollout loss (decoded AE tokens vs ground truth)")
    parser.add_argument("--signal_loss_weight", type=float, default=0.1,
                        help="Weight for latent-space signal loss (EMA target)")
    parser.add_argument("--recon_loss_weight", type=float, default=1.0,
                        help="Weight for encoder-decoder reconstruction loss "
                             "(decode(encode(ctx)) ≈ ctx AE tokens)")
    parser.add_argument("--delta_loss_weight", type=float, default=1.0,
                        help="Weight for delta loss: MSE on predicted vs "
                             "target displacement from context.  Makes copy "
                             "(zero delta) explicitly suboptimal.")
    parser.add_argument("--max_files", type=int, default=None,
                        help="Limit number of HDF5 files (None = all)")
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--encoder_lr", type=float, default=1e-5,
                        help="Learning rate for encoder/decoder. When "
                             "--dynamics_lr is set, this applies only to "
                             "non-dynamics parameters.")
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--dynamics_lr", type=float, default=1e-3,
                        help="Separate LR for dynamics module. When set, "
                             "--encoder_lr applies to encoder/decoder and "
                             "dynamics gets this rate.")
    parser.add_argument("--steps_per_epoch", type=int, default=0,
                        help="Cap batches per epoch (train and val). "
                             "0 = no limit (use full dataset).")
    parser.add_argument("--plot_every", type=int, default=1,
                        help="Generate diagnostic plots every N epochs (0=off)")
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--rollout_start", type=int, default=1,
                        help="Initial number of rollout steps for curriculum. "
                             "If None, no curriculum (full N_ROLLOUT from the start).")
    parser.add_argument("--rollout_ramp_epochs", type=int, default=30,
                        help="Number of epochs to linearly ramp rollout steps "
                             "from --rollout_start to N_ROLLOUT.")
    parser.add_argument("--rollout_noise_std", type=float, default=0.1,
                        help="Std of Gaussian noise injected between rollout "
                             "steps during training (0 = disabled).")
    parser.add_argument("--teacher_forcing_start", type=float, default=0.5,
                        help="Initial teacher forcing ratio (0 = disabled, "
                             "1 = always replace with ground truth). "
                             "Linearly decayed to 0 over "
                             "--teacher_forcing_epochs.")
    parser.add_argument("--teacher_forcing_epochs", type=int, default=40,
                        help="Epochs to linearly decay teacher forcing to 0.")
    parser.add_argument("--context_noise_std", type=float, default=0.1,
                        help="Gaussian noise std added to context AE tokens "
                             "during training (targets stay clean). "
                             "Prevents copy behavior.")
    parser.add_argument("--context_drop_rate", type=float, default=0.1,
                        help="Probability of dropping (zeroing) each context "
                             "token during training. Prevents copy behavior.")
    parser.add_argument("--step_size_s", type=float, default=0.5,
                        help="Step size between chunk start times in seconds. "
                             "If smaller than chunk_duration, chunks overlap. "
                             "Defaults to chunk_duration (no overlap).")
    parser.add_argument("--warmup_s", type=float, default=0.0,
                        help="Skip the first N seconds of each shot. "
                             "Chunks start at warmup_s instead of t=0. "
                             "Use to skip ramp-up and train on flat-top.")
    args = parser.parse_args()
    if args.step_size_s is None:
        args.step_size_s = CHUNK_S

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ae_ckpt_dir = Path(args.ae_checkpoint_dir)

    # --- Load pre-trained AEs ---
    ae_encoders = {}
    for name, cfg in DIAGNOSTIC_CONFIGS.items():
        # Allow per-modality checkpoint path override via "ae_checkpoint_path"
        if "ae_checkpoint_path" in cfg:
            ckpt_path = Path(cfg["ae_checkpoint_path"])
        else:
            ckpt_path = ae_ckpt_dir / f"{name}_{cfg['model_type']}" / "checkpoint_best.pth"
        if not ckpt_path.exists():
            logger.warning(f"AE checkpoint not found for '{name}': {ckpt_path} — skipping")
            continue
        ae_encoders[name] = load_ae(name, cfg, ckpt_path)

    if not ae_encoders:
        raise RuntimeError("No AE checkpoints found. Check --ae_checkpoint_dir.")

    active_diagnostics = {k: v for k, v in DIAGNOSTIC_CONFIGS.items() if k in ae_encoders}

    # --- Build dataset ---
    stats = torch.load(args.stats_path, weights_only=False)

    # Per-modality AE token normalization stats
    ae_token_stats = None
    if args.ae_token_stats_path is not None:
        ae_token_stats = torch.load(args.ae_token_stats_path, weights_only=False)
        logger.info(f"Loaded AE token stats for {list(ae_token_stats.keys())}")

    all_signals = list(active_diagnostics.keys()) + list(ACTUATOR_CONFIGS.keys())

    data_dir = Path(args.data_dir)
    all_files = sorted(data_dir.glob("*_processed.h5"))
    random.seed(42)
    random.shuffle(all_files)
    if args.max_files is not None:
        all_files = all_files[:args.max_files]
    n = len(all_files)
    n_val = max(1, int(0.1 * n))
    n_test = max(1, int(0.1 * n))
    train_files = all_files[n_val + n_test:]
    val_files = all_files[:n_val]
    logger.info(f"Files — train: {len(train_files)}  val: {len(val_files)}")

    shared_ds_kwargs = dict(
        preprocessing_stats=stats,
        input_signals=all_signals,
        chunk_duration_s=CHUNK_S,
        step_size_s=args.step_size_s,
        warmup_s=args.warmup_s,
        prediction_mode=False,
    )

    train_ds = TokamakMultiFileDataset(
        train_files, lengths_cache_path="lengths_train.pt", **shared_ds_kwargs
    )
    val_ds = TokamakMultiFileDataset(
        val_files, lengths_cache_path="lengths_validation.pt", **shared_ds_kwargs
    )
    logger.info(f"Chunks — train: {len(train_ds)}  val: {len(val_ds)}")

    train_loader = make_dataloader(
        train_ds, batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=True,
        pin_memory=True, prefetch_factor=args.prefetch_factor,
    )
    val_loader = make_dataloader(
        val_ds, batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=False,
        pin_memory=True, prefetch_factor=args.prefetch_factor,
    )

    # Visualization loaders with longer chunks for extended rollout
    viz_ds = TokamakMultiFileDataset(
        val_files,
        lengths_cache_path="lengths_viz.pt",
        preprocessing_stats=stats,
        input_signals=all_signals,
        chunk_duration_s=CHUNK_VIS_S,
        warmup_s=args.warmup_s,
        prediction_mode=False,
    )
    viz_loader = make_dataloader(
        viz_ds, batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=False,
        pin_memory=True, prefetch_factor=args.prefetch_factor,
    )
    train_viz_ds = TokamakMultiFileDataset(
        train_files[:5],
        lengths_cache_path="lengths_train_viz.pt",
        preprocessing_stats=stats,
        input_signals=all_signals,
        chunk_duration_s=CHUNK_VIS_S,
        warmup_s=args.warmup_s,
        prediction_mode=False,
    )
    train_viz_loader = make_dataloader(
        train_viz_ds, batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=False,
        pin_memory=True, prefetch_factor=args.prefetch_factor,
    )

    # --- Build foundation model ---
    modality_configs = {
        name: {"d_lat": cfg["d_lat"], "n_tokens": cfg["n_tokens"]}
        for name, cfg in active_diagnostics.items()
    }
    n_actuators = sum(cfg["n_channels"] for cfg in ACTUATOR_CONFIGS.values())

    model = PerceiverFoundationModel(
        modality_configs=modality_configs,
        d_model=args.d_model,
        n_latent=args.n_latent,
        n_actuators=n_actuators,
        encoder_layers=args.encoder_layers,
        processor_layers=args.processor_layers,
        decoder_layers=args.decoder_layers,
        decoder_self_attn_layers=args.decoder_self_attn_layers,
        dynamics_layers=args.dynamics_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        dynamics_type=args.dynamics_type,
        actuator_configs=(
            ACTUATOR_CONFIGS if args.dynamics_type in ("cross_attention", "gru")
            else None
        ),
        ema_decay=args.ema_decay,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Foundation model trainable parameters: {n_params:,}")
    logger.info(f"Training config: rollout_steps={N_ROLLOUT}, dt={DT_S*1000:.0f}ms, "
                f"context={WINDOW_S*1000:.0f}ms, chunk={CHUNK_S*1000:.0f}ms")
    logger.info(f"EMA decay: {args.ema_decay}, loss weights: "
                f"encode={args.encode_loss_weight}, recon={args.recon_loss_weight}, "
                f"rollout={args.rollout_loss_weight}, signal={args.signal_loss_weight}, "
                f"delta={args.delta_loss_weight}")
    logger.info(f"Diagnostics: {list(active_diagnostics.keys())}")
    logger.info(f"Actuators: {list(ACTUATOR_CONFIGS.keys())} ({n_actuators} dims), "
                f"dynamics_type={args.dynamics_type}")

    if args.dynamics_lr is not None:
        dynamics_param_ids = {id(p) for p in model.dynamics.parameters()}
        encoder_group = [p for p in model.parameters()
                         if p.requires_grad and id(p) not in dynamics_param_ids]
        dynamics_group = [p for p in model.dynamics.parameters()
                          if p.requires_grad]
        optimizer = optim.AdamW([
            {"params": encoder_group, "lr": args.encoder_lr},
            {"params": dynamics_group, "lr": args.dynamics_lr},
        ], weight_decay=args.weight_decay)
        logger.info(f"Differentiated LR: encoder={args.encoder_lr:.1e}, "
                    f"dynamics={args.dynamics_lr:.1e} "
                    f"({args.dynamics_lr / args.encoder_lr:.0f}x ratio)")
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.encoder_lr,
                                weight_decay=args.weight_decay)

    if args.warmup_epochs > 0:
        warmup = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, end_factor=1.0, total_iters=args.warmup_epochs
        )
        cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs - args.warmup_epochs), eta_min=args.min_lr
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_epochs]
        )
    else:
        scheduler = None

    start_epoch = 0
    best_val = float("inf")
    checkpoint_path = ckpt_dir / "checkpoint.pth"
    best_path = ckpt_dir / "best.pth"

    if args.resume and checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(
            ckpt["model_state_dict"], strict=False)
        if missing:
            logger.info(f"Checkpoint: {len(missing)} missing keys "
                        f"(newly added): {missing[:5]}...")
        if unexpected:
            logger.info(f"Checkpoint: {len(unexpected)} unexpected keys "
                        f"(removed): {unexpected[:5]}...")
        if not missing and not unexpected:
            # Only restore optimizer if checkpoint and param groups match
            saved_groups = len(ckpt["optimizer_state_dict"]["param_groups"])
            if saved_groups == len(optimizer.param_groups):
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            else:
                logger.info(f"Optimizer group count changed ({saved_groups} → "
                            f"{len(optimizer.param_groups)}) — skipping optimizer restore")
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val = ckpt.get("best_val", float("inf"))
        logger.info(f"Resumed from epoch {start_epoch}")

    # --- Rollout curriculum ---
    rollout_start = args.rollout_start
    if rollout_start is not None:
        rollout_start = max(1, min(rollout_start, N_ROLLOUT))
        logger.info(f"Rollout curriculum: {rollout_start} → {N_ROLLOUT} "
                    f"over {args.rollout_ramp_epochs} epochs")

    def get_n_rollout(epoch: int) -> int:
        """Compute the number of rollout steps for the current epoch."""
        if rollout_start is None:
            return N_ROLLOUT
        progress = min(epoch / max(1, args.rollout_ramp_epochs), 1.0)
        return round(rollout_start + progress * (N_ROLLOUT - rollout_start))

    def get_teacher_forcing_ratio(epoch: int) -> float:
        """Linearly decay teacher forcing from start value to 0."""
        if args.teacher_forcing_start <= 0:
            return 0.0
        progress = min(epoch / max(1, args.teacher_forcing_epochs), 1.0)
        return args.teacher_forcing_start * (1.0 - progress)

    if args.teacher_forcing_start > 0:
        logger.info(f"Teacher forcing: {args.teacher_forcing_start:.1f} → 0 "
                    f"over {args.teacher_forcing_epochs} epochs")

    # --- Training loop ---
    for epoch in range(start_epoch, args.epochs):
        n_rollout_epoch = get_n_rollout(epoch)
        tf_ratio = get_teacher_forcing_ratio(epoch)

        (train_total, train_enc, train_recon, train_roll,
         train_sig, train_dlt) = run_epoch(
            model, ae_encoders, train_loader, optimizer,
            is_train=True,
            encode_loss_weight=args.encode_loss_weight,
            rollout_loss_weight=args.rollout_loss_weight,
            signal_loss_weight=args.signal_loss_weight,
            recon_loss_weight=args.recon_loss_weight,
            delta_loss_weight=args.delta_loss_weight,
            max_steps=args.steps_per_epoch,
            preprocess_stats=stats,
            n_rollout=n_rollout_epoch,
            rollout_noise_std=args.rollout_noise_std,
            teacher_forcing_ratio=tf_ratio,
            context_noise_std=args.context_noise_std,
            context_drop_rate=args.context_drop_rate,
            zero_actuators=args.zero_actuators,
            ae_token_stats=ae_token_stats,
        )
        (val_total, val_enc, val_recon, val_roll,
         val_sig, val_dlt) = run_epoch(
            model, ae_encoders, val_loader, optimizer=None,
            is_train=False,
            encode_loss_weight=args.encode_loss_weight,
            rollout_loss_weight=args.rollout_loss_weight,
            signal_loss_weight=args.signal_loss_weight,
            recon_loss_weight=args.recon_loss_weight,
            delta_loss_weight=args.delta_loss_weight,
            max_steps=args.steps_per_epoch,
            preprocess_stats=stats,
            n_rollout=n_rollout_epoch,
            zero_actuators=args.zero_actuators,
            ae_token_stats=ae_token_stats,
        )

        if scheduler is not None:
            scheduler.step()

        lr_enc = optimizer.param_groups[0]["lr"]
        if len(optimizer.param_groups) > 1:
            lr_dyn = optimizer.param_groups[1]["lr"]
            lr_str = f"lr_enc={lr_enc:.2e} lr_dyn={lr_dyn:.2e}"
        else:
            lr_str = f"lr={lr_enc:.2e}"
        rollout_info = (f"  rollout_steps={n_rollout_epoch}"
                        if rollout_start is not None else "")
        if tf_ratio > 0:
            rollout_info += f"  tf={tf_ratio:.2f}"
        logger.info(
            f"Epoch {epoch+1:4d}/{args.epochs}  "
            f"train={train_total:.6f} "
            f"(enc={train_enc:.6f} rec={train_recon:.6f} "
            f"roll={train_roll:.6f} sig={train_sig:.6f} "
            f"dlt={train_dlt:.6f})  "
            f"val={val_total:.6f} "
            f"(enc={val_enc:.6f} rec={val_recon:.6f} "
            f"roll={val_roll:.6f} sig={val_sig:.6f} "
            f"dlt={val_dlt:.6f})  "
            f"{lr_str}{rollout_info}"
        )

        # Save checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val": best_val,
                "modality_configs": modality_configs,
                "args": vars(args),
            },
            checkpoint_path,
        )

        if val_total < best_val:
            best_val = val_total
            torch.save(model.state_dict(), best_path)
            logger.info(f"  → New best val loss: {best_val:.6f}")

        # Diagnostic plots
        if args.plot_every > 0 and (
            (epoch + 1) % args.plot_every == 0 or epoch == args.epochs - 1
        ):
            visualize_predictions(
                model, ae_encoders, viz_loader, epoch + 1, ckpt_dir,
                preprocess_stats=stats, label="val",
                ae_token_stats=ae_token_stats,
            )
            visualize_predictions(
                model, ae_encoders, train_viz_loader, epoch + 1, ckpt_dir,
                preprocess_stats=stats, label="train",
                ae_token_stats=ae_token_stats,
            )
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
