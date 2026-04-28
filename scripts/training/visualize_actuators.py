"""Visualize actuator processing through the foundation model pipeline.

Loads a trained checkpoint and a validation batch, then produces
diagnostic plots showing:

1. Raw actuator signals (before normalization)
2. Normalized actuator signals (after min-max + channel selection)
3. Tokenized actuator representations (after Conv1d patch embedding)
4. Cross-attention weights: how much the dynamics queries attend to
   actuator tokens vs latent tokens
"""
import argparse
import logging
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tokamak_foundation_model.data.multi_file_dataset import (
    TokamakMultiFileDataset, make_dataloader)
from tokamak_foundation_model.models.latent_feature_space.foundation_model import (
    PerceiverFoundationModel)
from train_foundation_model import (
    DIAGNOSTIC_CONFIGS, ACTUATOR_CONFIGS, DT_S, WINDOW_S, CHUNK_S,
    load_ae, split_window, encode_batch,
    actuator_context_window, actuator_step_windows,
    _select_channels, _normalize_actuator,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def plot_raw_vs_normalized(batch, stats, save_dir):
    """Plot raw and normalized actuator signals side by side."""
    n_act = len(ACTUATOR_CONFIGS)
    fig, axes = plt.subplots(n_act, 3, figsize=(18, 3 * n_act))
    if n_act == 1:
        axes = axes[np.newaxis, :]

    idx = 0  # first sample in batch

    for row, (name, cfg) in enumerate(ACTUATOR_CONFIGS.items()):
        if name not in batch:
            axes[row, 0].set_title(f"{name} — NOT IN BATCH")
            continue

        raw_sig = batch[name][idx].cpu()  # [C_raw, T]
        selected = _select_channels(batch[name][idx:idx+1], cfg)[0].cpu()  # [C_sel, T]
        normalized = _normalize_actuator(
            selected.unsqueeze(0), name, stats,
            channels_to_use=cfg.get("channels_to_use")
        )[0].cpu()  # [C_sel, T]

        fs = cfg["target_fs"]
        n_ctx = round(WINDOW_S * fs)
        t_ms = np.arange(raw_sig.shape[-1]) / fs * 1000

        # Col 0: Raw signal (all channels)
        ax = axes[row, 0]
        for ch in range(raw_sig.shape[0]):
            ax.plot(t_ms[:n_ctx], raw_sig[ch, :n_ctx].numpy(),
                    linewidth=0.5, alpha=0.7)
        ax.set_title(f"{name} — raw ({raw_sig.shape[0]} ch)")
        ax.set_xlabel("time [ms]")
        ax.axvline(WINDOW_S * 1000, color="red", ls="--", lw=0.5)

        # Col 1: Selected channels, normalized
        ax = axes[row, 1]
        for ch in range(normalized.shape[0]):
            ax.plot(t_ms[:n_ctx], normalized[ch, :n_ctx].numpy(),
                    linewidth=0.5, alpha=0.7,
                    label=f"ch{cfg.get('channels_to_use', list(range(cfg['n_channels'])))[ch] if cfg.get('channels_to_use') else ch}")
        ax.set_title(f"{name} — normalized ({normalized.shape[0]} ch)")
        ax.set_xlabel("time [ms]")
        ax.set_ylim(-0.5, 1.5)
        ax.axhline(0, color="gray", ls=":", lw=0.5)
        ax.axhline(1, color="gray", ls=":", lw=0.5)

        # Col 2: Value distribution histogram
        ax = axes[row, 2]
        vals = normalized[:, :n_ctx].numpy().flatten()
        vals = vals[np.isfinite(vals)]
        if len(vals) > 0:
            ax.hist(vals, bins=50, density=True, alpha=0.7)
            ax.set_title(f"{name} — distribution "
                         f"(mean={vals.mean():.3f}, std={vals.std():.3f})")
            ax.axvline(0, color="gray", ls=":", lw=0.5)
            ax.axvline(1, color="gray", ls=":", lw=0.5)
        else:
            ax.set_title(f"{name} — all NaN/Inf")

    fig.suptitle("Actuator signals: raw → normalized → distribution",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_dir / "actuators_raw_vs_normalized.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {save_dir / 'actuators_raw_vs_normalized.png'}")


def plot_tokenized_actuators(act_ctx, model, save_dir):
    """Visualize actuator tokens after Conv1d patch embedding."""
    tokenizer = model.dynamics.actuator_tokenizer

    with torch.no_grad():
        tokens = tokenizer(act_ctx, offset_ms=0.0)  # [B, N_total, d_model]

    B, N_total, D = tokens.shape
    logger.info(f"Actuator tokens: {tokens.shape} "
                f"(total {N_total} tokens, d_model={D})")

    # Count tokens per actuator group
    token_counts = {}
    for name, sig in act_ctx.items():
        if name not in tokenizer.configs:
            continue
        cfg = tokenizer.configs[name]
        patch_len = cfg["patch_len"]
        n_patches = sig.shape[-1] // patch_len
        token_counts[name] = n_patches
        logger.info(f"  {name}: {sig.shape} → {n_patches} patches "
                     f"(patch_len={patch_len})")

    # Plot token heatmap
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Token values (first sample)
    ax = axes[0]
    tok_np = tokens[0].cpu().numpy()
    d_show = min(64, D)
    im = ax.imshow(tok_np[:, :d_show], aspect="auto", cmap="RdBu_r",
                   interpolation="nearest")
    ax.set_title(f"Actuator tokens [N={N_total}, first {d_show} dims]")
    ax.set_xlabel("dimension")
    ax.set_ylabel("token index")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Annotate group boundaries
    pos = 0
    for name, count in token_counts.items():
        ax.axhline(pos - 0.5, color="white", lw=1)
        ax.text(d_show + 1, pos + count / 2, name, fontsize=8, va="center")
        pos += count

    # Token norms (how "active" each token is)
    ax = axes[1]
    norms = tokens[0].norm(dim=-1).cpu().numpy()
    ax.barh(range(N_total), norms, height=0.8)
    ax.set_title("Token L2 norms")
    ax.set_xlabel("norm")
    ax.set_ylabel("token index")
    ax.invert_yaxis()
    pos = 0
    for name, count in token_counts.items():
        ax.axhline(pos - 0.5, color="red", lw=1)
        pos += count

    fig.suptitle("Actuator tokens after Conv1d + embedding + PE",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_dir / "actuators_tokenized.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {save_dir / 'actuators_tokenized.png'}")

    return tokens


def plot_attention_weights(model, latent, act_curr, act_fut, save_dir):
    """Extract and plot cross-attention weights from the dynamics."""
    dynamics = model.dynamics

    # Hook into cross-attention to capture attention weights
    attn_weights = []

    def hook_fn(module, args, kwargs, output):
        # nn.MultiheadAttention returns (attn_output, attn_weights)
        if isinstance(output, tuple) and len(output) == 2:
            attn_weights.append(output[1].detach().cpu())

    hooks = []
    for block in dynamics.cross_blocks:
        h = block.cross_attn.register_forward_hook(hook_fn, with_kwargs=True)
        hooks.append(h)

    # Run dynamics forward
    with torch.no_grad():
        # Need attention weights — set need_weights=True temporarily
        for block in dynamics.cross_blocks:
            block.cross_attn.need_weights = True
            block.cross_attn._qkv_same_embed_dim = True

        _ = dynamics(latent, act_curr, act_fut,
                     offset_ms=WINDOW_S * 1000, dt_ms=DT_S * 1000)

    # Remove hooks
    for h in hooks:
        h.remove()

    if not attn_weights:
        logger.warning("No attention weights captured — "
                       "MultiheadAttention may not return weights by default.")
        # Try alternative: manually compute attention
        logger.info("Computing attention weights manually...")
        plot_attention_manual(model, latent, act_curr, act_fut, save_dir)
        return

    # Plot attention patterns
    n_layers = len(attn_weights)
    fig, axes = plt.subplots(1, n_layers, figsize=(8 * n_layers, 6))
    if n_layers == 1:
        axes = [axes]

    # Figure out context composition: act_curr_tokens + act_fut_tokens
    with torch.no_grad():
        act_curr_tokens = dynamics.actuator_tokenizer(
            act_curr, offset_ms=WINDOW_S * 1000)
        act_fut_tokens = dynamics.actuator_tokenizer(
            act_fut, offset_ms=WINDOW_S * 1000 + DT_S * 1000)
    n_curr = act_curr_tokens.shape[1]
    n_fut = act_fut_tokens.shape[1]
    n_ctx_total = n_curr + n_fut

    for i, (ax, aw) in enumerate(zip(axes, attn_weights)):
        # aw shape: [B, N_latent, N_context] or [B*n_heads, N_latent, N_context]
        aw_mean = aw[0]  # first sample
        if aw_mean.dim() == 3:
            aw_mean = aw_mean.mean(dim=0)  # average over heads

        im = ax.imshow(aw_mean.numpy(), aspect="auto", cmap="viridis",
                       interpolation="nearest")
        ax.set_title(f"Layer {i}: attention weights")
        ax.set_xlabel(f"context tokens (curr_act: 0-{n_curr}, "
                      f"fut_act: {n_curr}-{n_ctx_total})")
        ax.set_ylabel("latent queries")
        ax.axvline(n_curr - 0.5, color="red", lw=1, label="curr|fut boundary")
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Print summary statistics
        act_attn = aw_mean[:, :].sum(dim=0)
        logger.info(f"Layer {i}: total attention to curr_act={act_attn[:n_curr].sum():.3f}, "
                    f"fut_act={act_attn[n_curr:].sum():.3f}")

    fig.suptitle("Dynamics cross-attention: latent queries → actuator context",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_dir / "actuators_attention.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {save_dir / 'actuators_attention.png'}")


def plot_attention_manual(model, latent, act_curr, act_fut, save_dir):
    """Manually compute and plot attention weights from dynamics."""
    dynamics = model.dynamics

    with torch.no_grad():
        act_curr_tokens = dynamics.actuator_tokenizer(
            act_curr, offset_ms=WINDOW_S * 1000)
        act_fut_tokens = dynamics.actuator_tokenizer(
            act_fut, offset_ms=WINDOW_S * 1000 + DT_S * 1000)
        context = torch.cat([act_curr_tokens, act_fut_tokens], dim=1)

    n_curr = act_curr_tokens.shape[1]
    n_fut = act_fut_tokens.shape[1]

    # Compute attention weights manually for each layer
    fig, axes = plt.subplots(1, len(dynamics.cross_blocks),
                             figsize=(8 * len(dynamics.cross_blocks), 6))
    if len(dynamics.cross_blocks) == 1:
        axes = [axes]

    x = latent
    for i, (ax, block) in enumerate(zip(axes, dynamics.cross_blocks)):
        with torch.no_grad():
            # Get Q, K from the cross-attention
            ca = block.cross_attn
            q = x[0:1]  # first sample
            k = context[0:1]

            # Project Q and K
            qw, kw, _ = ca.in_proj_weight.chunk(3, dim=0)
            qb, kb, _ = ca.in_proj_bias.chunk(3, dim=0)
            Q = torch.nn.functional.linear(q, qw, qb)  # [1, N_q, D]
            K = torch.nn.functional.linear(k, kw, kb)  # [1, N_k, D]

            # Compute attention scores
            d_k = Q.shape[-1] / ca.num_heads
            scores = torch.bmm(Q, K.transpose(1, 2)) / (d_k ** 0.5)
            attn = torch.softmax(scores, dim=-1)[0].cpu().numpy()

            im = ax.imshow(attn, aspect="auto", cmap="viridis",
                           interpolation="nearest")
            ax.set_title(f"Layer {i}: attention (averaged heads)")
            ax.set_xlabel(f"context ({n_curr} curr_act + {n_fut} fut_act)")
            ax.set_ylabel(f"latent queries ({latent.shape[1]})")
            ax.axvline(n_curr - 0.5, color="red", lw=1)
            plt.colorbar(im, ax=ax, fraction=0.046)

            # Advance x through the block for next layer
            x = block(x, context)

    fig.suptitle("Dynamics: latent queries attending to actuator tokens",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_dir / "actuators_attention.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {save_dir / 'actuators_attention.png'}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize actuator processing in the foundation model")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir",
                        default="/scratch/gpfs/EKOLEMEN/foundation_model/")
    parser.add_argument("--stats_path",
                        default="/projects/EKOLEMEN/foundation_model/preprocessing_stats.pt")
    parser.add_argument("--ae_checkpoint_dir",
                        default="/projects/EKOLEMEN/foundation_model/")
    parser.add_argument("--max_files", type=int, default=200)
    parser.add_argument("--save_dir", default="runs/foundation_model_debug/plots")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_args = ckpt.get("args", {})
    modality_configs_saved = ckpt.get("modality_configs", {})

    # Load AE models
    ae_ckpt_dir = Path(args.ae_checkpoint_dir)
    ae_models = {}
    for name, cfg in DIAGNOSTIC_CONFIGS.items():
        if "ae_checkpoint_path" in cfg:
            ckpt_path = Path(cfg["ae_checkpoint_path"])
        else:
            ckpt_path = ae_ckpt_dir / f"{name}_{cfg['model_type']}" / "checkpoint_best.pth"
        if ckpt_path.exists():
            ae_models[name] = load_ae(name, cfg, ckpt_path)

    active_diagnostics = {k: v for k, v in DIAGNOSTIC_CONFIGS.items()
                          if k in ae_models}

    # Build model
    modality_configs = modality_configs_saved or {
        name: {"d_lat": cfg["d_lat"], "n_tokens": cfg["n_tokens"]}
        for name, cfg in active_diagnostics.items()
    }
    dynamics_type = saved_args.get("dynamics_type", "cross_attention")
    model = PerceiverFoundationModel(
        modality_configs=modality_configs,
        d_model=saved_args.get("d_model", 256),
        n_latent=saved_args.get("n_latent", 128),
        n_actuators=sum(c["n_channels"] for c in ACTUATOR_CONFIGS.values()),
        encoder_layers=saved_args.get("encoder_layers", 1),
        processor_layers=saved_args.get("processor_layers", 1),
        decoder_layers=saved_args.get("decoder_layers", 2),
        decoder_self_attn_layers=saved_args.get("decoder_self_attn_layers", 0),
        dynamics_layers=saved_args.get("dynamics_layers", 2),
        n_heads=saved_args.get("n_heads", 8),
        dropout=0.0,
        dynamics_type=dynamics_type,
        actuator_configs=(ACTUATOR_CONFIGS if dynamics_type == "cross_attention"
                          else None),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    # Load data
    stats = torch.load(args.stats_path, weights_only=False)
    all_signals = list(active_diagnostics.keys()) + list(ACTUATOR_CONFIGS.keys())
    data_dir = Path(args.data_dir)
    all_files = sorted(data_dir.glob("*_processed.h5"))
    random.seed(42)
    random.shuffle(all_files)
    if args.max_files:
        all_files = all_files[:args.max_files]
    n_val = max(1, int(0.1 * len(all_files)))
    val_files = all_files[:n_val]

    val_ds = TokamakMultiFileDataset(
        val_files,
        lengths_cache_path="lengths_act_vis.pt",
        preprocessing_stats=stats,
        input_signals=all_signals,
        chunk_duration_s=CHUNK_S,
        prediction_mode=False,
    )
    loader = make_dataloader(val_ds, batch_size=4, num_workers=0, shuffle=False)
    batch = next(iter(loader))
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in batch.items()}

    logger.info("=" * 60)
    logger.info("1. Raw vs normalized actuator signals")
    logger.info("=" * 60)
    plot_raw_vs_normalized(batch, stats, save_dir)

    logger.info("\n" + "=" * 60)
    logger.info("2. Tokenized actuator representations")
    logger.info("=" * 60)
    act_ctx = actuator_context_window(batch, ACTUATOR_CONFIGS, stats)
    tokens = plot_tokenized_actuators(act_ctx, model, save_dir)

    logger.info("\n" + "=" * 60)
    logger.info("3. Cross-attention weights in dynamics")
    logger.info("=" * 60)
    # Encode context to get latent
    ctx_signals = {}
    for name, cfg in active_diagnostics.items():
        if name not in batch:
            continue
        ctx, _ = split_window(batch[name], cfg["target_fs"], n_rollout=1)
        ctx_signals[name] = ctx
    with torch.no_grad():
        lat_ctx = encode_batch(ae_models, ctx_signals)
        latent = model.encode(lat_ctx, act_ctx)

    act_step_pairs = actuator_step_windows(
        batch, ACTUATOR_CONFIGS, stats, n_rollout=1)
    act_curr, act_fut = act_step_pairs[0]

    plot_attention_manual(model, latent, act_curr, act_fut, save_dir)

    logger.info("\nDone! Plots saved to: " + str(save_dir))


if __name__ == "__main__":
    main()
