#!/usr/bin/env python
"""
Overfit-one-batch test for the dynamics model.

Trains on a single batch from a few shots, and every ``--eval_every``
steps runs a full autoregressive rollout.  The key metric tracked is
**rollout step-to-step cosine similarity**: if the model copies, all
rollout steps are identical (cos ≈ 1.0).  As training progresses this
should decrease, proving the dynamics produces diverse predictions.

Produces two plots at the end:
  1. ``overfit_rollout_metrics.png`` — rollout diversity vs training step
  2. ``overfit_rollout_signal.png``  — signal-space rollout at final step
"""

from pathlib import Path
import argparse
import logging
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tokamak_foundation_model.data.multi_file_dataset import (
    TokamakMultiFileDataset, make_dataloader,
)
from tokamak_foundation_model.models.model_factory import build_model
from tokamak_foundation_model.models.latent_feature_space.foundation_model import (
    PerceiverFoundationModel,
)

from train_foundation_model import (
    DIAGNOSTIC_CONFIGS, ACTUATOR_CONFIGS,
    DT_S, WINDOW_S, N_ROLLOUT, CHUNK_S,
    load_ae, split_window, encode_batch,
    actuator_context_window, actuator_step_windows,
    _select_channels, ae_decode, masked_channel_mean,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Data & model setup
# -----------------------------------------------------------------------

def load_data_and_model(args):
    """Load AEs, one batch, and build a fresh model."""
    ae_ckpt_dir = Path(args.ae_checkpoint_dir)
    ae_encoders = {}
    for name, cfg in DIAGNOSTIC_CONFIGS.items():
        if "ae_checkpoint_path" in cfg:
            ckpt_path = Path(cfg["ae_checkpoint_path"])
        else:
            ckpt_path = (ae_ckpt_dir / f"{name}_{cfg['model_type']}"
                         / "checkpoint_best.pth")
        if not ckpt_path.exists():
            logger.warning(f"AE not found for '{name}': {ckpt_path}")
            continue
        ae_encoders[name] = load_ae(name, cfg, ckpt_path)

    active_diagnostics = {
        k: v for k, v in DIAGNOSTIC_CONFIGS.items() if k in ae_encoders}

    stats = torch.load(args.stats_path, weights_only=False)
    all_signals = (list(active_diagnostics.keys())
                   + list(ACTUATOR_CONFIGS.keys()))
    data_dir = Path(args.data_dir)
    all_files = sorted(data_dir.glob("*_processed.h5"))
    random.seed(42)
    random.shuffle(all_files)

    ds = TokamakMultiFileDataset(
        all_files[:args.n_files],
        lengths_cache_path="lengths_overfit_test.pt",
        preprocessing_stats=stats,
        input_signals=all_signals,
        chunk_duration_s=CHUNK_S,
        step_size_s=CHUNK_S,
        warmup_s=1.0,
        prediction_mode=False,
    )
    loader = make_dataloader(
        ds, batch_size=args.batch_size, num_workers=2,
        shuffle=False, pin_memory=True)
    batch = next(iter(loader))
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in batch.items()}

    B = next(v.shape[0] for v in batch.values()
             if isinstance(v, torch.Tensor))
    logger.info(f"Loaded batch: {len(batch)} keys, B={B}")

    modality_configs = {
        name: {"d_lat": cfg["d_lat"], "n_tokens": cfg["n_tokens"]}
        for name, cfg in active_diagnostics.items()
    }

    model = PerceiverFoundationModel(
        modality_configs=modality_configs,
        d_model=args.d_model,
        n_latent=args.n_latent,
        encoder_layers=args.encoder_layers,
        processor_layers=args.processor_layers,
        decoder_layers=args.decoder_layers,
        dynamics_layers=args.dynamics_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        dynamics_type="cross_attention",
        actuator_configs=ACTUATOR_CONFIGS,
        ema_decay=0.996,
    ).to(device)

    # Precompute everything that stays fixed across training
    n_rollout = args.n_rollout

    ctx_signals = {}
    tgt_signals_steps = [{} for _ in range(n_rollout)]
    for name, cfg in DIAGNOSTIC_CONFIGS.items():
        if name not in batch:
            continue
        ctx, tgts = split_window(batch[name], cfg["target_fs"],
                                 n_rollout=n_rollout)
        ctx_signals[name] = ctx
        for k, tgt in enumerate(tgts):
            tgt_signals_steps[k][name] = tgt

    with torch.no_grad():
        lat_ctx = encode_batch(ae_encoders, ctx_signals)
        lat_tgt_steps = [encode_batch(ae_encoders, tgt_s)
                         for tgt_s in tgt_signals_steps]

    act_ctx = actuator_context_window(batch, ACTUATOR_CONFIGS, stats)
    act_step_pairs = actuator_step_windows(
        batch, ACTUATOR_CONFIGS, stats, n_rollout=n_rollout)
    act_ctx_steps = [
        actuator_context_window(
            batch, ACTUATOR_CONFIGS, stats,
            offset_s=(k + 1) * DT_S)
        for k in range(n_rollout)
    ]

    return dict(
        model=model, ae_encoders=ae_encoders, batch=batch, stats=stats,
        lat_ctx=lat_ctx, lat_tgt_steps=lat_tgt_steps,
        ctx_signals=ctx_signals,
        act_ctx=act_ctx, act_step_pairs=act_step_pairs,
        act_ctx_steps=act_ctx_steps,
        active_diagnostics=active_diagnostics,
        n_rollout=n_rollout,
    )


# -----------------------------------------------------------------------
# Rollout evaluation
# -----------------------------------------------------------------------

@torch.no_grad()
def eval_rollout(ctx):
    """Run full autoregressive rollout and return diversity metrics.

    Returns
    -------
    dict with keys:
        mse_pred        : list[float] — MSE(rollout_step_k, target_k)
        mse_copy        : list[float] — MSE(context_latent, target_k)
        ratio           : list[float] — mse_pred / mse_copy
        cos_consecutive : list[float] — cos_sim(step_k, step_{k-1})
        cos_vs_step1    : list[float] — cos_sim(step_k, step_1)
        mean_cos_consec : float
        mean_ratio      : float
    """
    model = ctx["model"]
    model.eval()

    lat_ctx = ctx["lat_ctx"]
    act_ctx = ctx["act_ctx"]
    act_step_pairs = ctx["act_step_pairs"]
    act_ctx_steps = ctx["act_ctx_steps"]
    lat_tgt_steps = ctx["lat_tgt_steps"]
    n_rollout = ctx["n_rollout"]

    latent_ctx = model.encode(lat_ctx, act_ctx)
    lat_ctx_ema = model.ema_encode(lat_ctx, act_ctx)

    lat_tgt_encoded = [
        model.ema_encode(lat_tgt_steps[k], act_ctx_steps[k])
        for k in range(n_rollout)
    ]

    mse_pred, mse_copy, ratios = [], [], []
    cos_consecutive, cos_vs_step1 = [], []

    latent = latent_ctx.clone()
    prev_latent = None
    step1_latent = None

    for k in range(n_rollout):
        act_curr_sig, act_fut_sig = act_step_pairs[k]
        offset_ms = WINDOW_S * 1000 + k * DT_S * 1000

        latent = model.dynamics(
            latent, act_curr_sig, act_fut_sig,
            offset_ms=offset_ms, dt_ms=DT_S * 1000)

        lat_target = lat_tgt_encoded[k]
        mp = F.mse_loss(latent, lat_target).item()
        mc = F.mse_loss(latent_ctx, lat_target).item()
        mse_pred.append(mp)
        mse_copy.append(mc)
        ratios.append(mp / max(mc, 1e-8))

        flat = latent.reshape(-1)
        if prev_latent is not None:
            cos_consecutive.append(F.cosine_similarity(
                flat.unsqueeze(0),
                prev_latent.reshape(-1).unsqueeze(0)).item())

        if step1_latent is None:
            step1_latent = latent.clone()
            cos_vs_step1.append(1.0)
        else:
            cos_vs_step1.append(F.cosine_similarity(
                flat.unsqueeze(0),
                step1_latent.reshape(-1).unsqueeze(0)).item())

        prev_latent = latent.clone()

    model.train()

    return dict(
        mse_pred=mse_pred,
        mse_copy=mse_copy,
        ratio=ratios,
        cos_consecutive=cos_consecutive,
        cos_vs_step1=cos_vs_step1,
        mean_cos_consec=float(np.mean(cos_consecutive)),
        mean_ratio=float(np.mean(ratios)),
    )


# -----------------------------------------------------------------------
# Training loops with periodic rollout evaluation
# -----------------------------------------------------------------------

def _init_history(ctx):
    """Record rollout metrics at step 0 (before any training)."""
    r = eval_rollout(ctx)
    return dict(
        steps=[0],
        loss=[float("nan")],
        mean_cos_consec=[r["mean_cos_consec"]],
        mean_ratio=[r["mean_ratio"]],
        cos_vs_step1=[r["cos_vs_step1"]],
    ), r


def _record(history, step, loss_val, ctx):
    r = eval_rollout(ctx)
    history["steps"].append(step)
    history["loss"].append(loss_val)
    history["mean_cos_consec"].append(r["mean_cos_consec"])
    history["mean_ratio"].append(r["mean_ratio"])
    history["cos_vs_step1"].append(r["cos_vs_step1"])
    return r


def train_dynamics_only(args, ctx):
    """Freeze encoder/decoder, train only dynamics on fixed latents.

    Isolates whether the dynamics architecture itself can learn to
    predict multi-step transitions (no encoder/decoder interference).
    """
    model = ctx["model"]
    lat_ctx = ctx["lat_ctx"]
    lat_tgt_steps = ctx["lat_tgt_steps"]
    act_ctx = ctx["act_ctx"]
    act_step_pairs = ctx["act_step_pairs"]
    act_ctx_steps = ctx["act_ctx_steps"]
    n_rollout = ctx["n_rollout"]

    logger.info(f"\n{'='*60}")
    logger.info("MODE: dynamics_only")
    logger.info(f"{'='*60}")

    # Freeze all, unfreeze dynamics
    for p in model.parameters():
        p.requires_grad_(False)
    dynamics_params = []
    for nm, p in model.named_parameters():
        if "dynamics" in nm:
            p.requires_grad_(True)
            dynamics_params.append(p)

    n_dyn = sum(p.numel() for p in dynamics_params)
    logger.info(f"Trainable: {n_dyn:,} dynamics params @ lr={args.dynamics_lr:.1e}")

    optimizer = optim.Adam(dynamics_params, lr=args.dynamics_lr)

    # Fixed latents (encoder/decoder frozen)
    with torch.no_grad():
        latent_ctx = model.encode(lat_ctx, act_ctx)
        lat_ctx_ema = model.ema_encode(lat_ctx, act_ctx)
        lat_tgt_encoded = [
            model.ema_encode(lat_tgt_steps[k], act_ctx_steps[k])
            for k in range(n_rollout)
        ]

    history, r0 = _init_history(ctx)

    logger.info(
        f"\n{'Step':>6}  {'loss':>8}  {'sig':>8}  {'dlt':>8}  "
        f"{'cos':>8}  {'div':>8}  {'pred_cs':>8}  {'tgt_cs':>8}  "
        f"{'cos_consec':>11}  {'ratio':>7}")
    logger.info("-" * 100)
    logger.info(
        f"{'0':>6}  {'--':>8}  {'--':>8}  {'--':>8}  "
        f"{'--':>8}  {'--':>8}  {'--':>8}  {'--':>8}  "
        f"{r0['mean_cos_consec']:11.6f}  {r0['mean_ratio']:7.3f}")

    for step in range(1, args.steps + 1):
        model.train()

        loss_sig = torch.tensor(0.0, device=device)
        loss_dlt = torch.tensor(0.0, device=device)
        loss_cos = torch.tensor(0.0, device=device)
        loss_div = torch.tensor(0.0, device=device)
        latent = latent_ctx.clone()
        prev_latent_flat = None
        prev_tgt_flat = None
        # Running means of consecutive-step cosine in latent space,
        # computed regardless of the regularizer weight so we can see
        # what `tgt_cs` (the regularizer's target) actually is.
        pred_cs_sum = 0.0
        tgt_cs_sum = 0.0
        n_pairs = 0

        for k in range(n_rollout):
            act_curr_sig, act_fut_sig = act_step_pairs[k]
            offset_ms = WINDOW_S * 1000 + k * DT_S * 1000

            latent = model.dynamics(
                latent, act_curr_sig, act_fut_sig,
                offset_ms=offset_ms, dt_ms=DT_S * 1000)

            lat_target = lat_tgt_encoded[k]
            lat_tgt_var = lat_target.detach().var().clamp(min=1e-6)
            step_weight = (k + 1) / n_rollout
            loss_sig = loss_sig + step_weight * (
                F.mse_loss(latent, lat_target) / lat_tgt_var)

            delta_pred = latent - latent_ctx
            delta_target = (lat_target - lat_ctx_ema).detach()
            delta_var = delta_target.var().clamp(min=1e-4)
            loss_dlt = loss_dlt + step_weight * (
                F.mse_loss(delta_pred, delta_target) / delta_var)

            # Proper direction match: cos between predicted and target
            # displacement.  This is the only term that rewards matching
            # the direction of the context→target step — see
            # feedback_delta_loss_algebra.md.
            p_flat = delta_pred.reshape(delta_pred.shape[0], -1)
            t_flat = delta_target.reshape(delta_target.shape[0], -1)
            loss_cos = loss_cos + step_weight * (
                1.0 - F.cosine_similarity(p_flat, t_flat, dim=-1)).mean()

            # Consecutive-step cosine for pred and tgt.  Computed always
            # (for logging); used by the regularizer when the weight is
            # non-zero.
            if prev_latent_flat is not None and prev_tgt_flat is not None:
                cur_flat = latent.reshape(latent.shape[0], -1)
                tgt_now_flat = lat_target.reshape(
                    lat_target.shape[0], -1)
                pred_cs = F.cosine_similarity(
                    cur_flat, prev_latent_flat, dim=-1)
                tgt_cs = F.cosine_similarity(
                    tgt_now_flat, prev_tgt_flat, dim=-1).detach()
                pred_cs_sum += pred_cs.mean().item()
                tgt_cs_sum += tgt_cs.mean().item()
                n_pairs += 1
                if args.step_diversity_weight > 0.0:
                    loss_div = loss_div + (pred_cs - tgt_cs).pow(2).mean()
            prev_latent_flat = latent.reshape(
                latent.shape[0], -1).detach()
            prev_tgt_flat = lat_target.reshape(
                lat_target.shape[0], -1).detach()

        loss_sig = loss_sig / n_rollout
        loss_dlt = loss_dlt / n_rollout
        loss_cos = loss_cos / n_rollout
        # loss_div is an average over (n_rollout - 1) step-pairs
        if n_rollout > 1:
            loss_div = loss_div / max(1, n_rollout - 1)
        loss = (loss_sig
                + args.delta_weight * (loss_dlt + loss_cos)
                + args.step_diversity_weight * loss_div)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(dynamics_params, max_norm=1.0)
        optimizer.step()

        if step % args.eval_every == 0 or step == args.steps:
            r = _record(history, step, loss.item(), ctx)
            mean_pred_cs = pred_cs_sum / max(1, n_pairs)
            mean_tgt_cs = tgt_cs_sum / max(1, n_pairs)
            logger.info(
                f"{step:6d}  {loss.item():8.4f}  {loss_sig.item():8.4f}  "
                f"{loss_dlt.item():8.4f}  {loss_cos.item():8.4f}  "
                f"{loss_div.item():8.4f}  "
                f"{mean_pred_cs:8.4f}  {mean_tgt_cs:8.4f}  "
                f"{r['mean_cos_consec']:11.6f}  {r['mean_ratio']:7.3f}")

    return history


def train_joint_finetune(args, ctx):
    """All params trainable with differentiated LR, all losses active."""
    model = ctx["model"]
    lat_ctx = ctx["lat_ctx"]
    lat_tgt_steps = ctx["lat_tgt_steps"]
    act_ctx = ctx["act_ctx"]
    act_step_pairs = ctx["act_step_pairs"]
    act_ctx_steps = ctx["act_ctx_steps"]
    n_rollout = ctx["n_rollout"]

    logger.info(f"\n{'='*60}")
    logger.info("MODE: joint_finetune")
    logger.info(f"{'='*60}")

    for p in model.parameters():
        p.requires_grad_(True)
    for p in model.ema_parameters():
        p.requires_grad_(False)

    dynamics_param_ids = {id(p) for p in model.dynamics.parameters()}
    encoder_params = [p for p in model.parameters()
                      if p.requires_grad and id(p) not in dynamics_param_ids]
    dynamics_params = [p for p in model.dynamics.parameters()
                       if p.requires_grad]

    n_enc = sum(p.numel() for p in encoder_params)
    n_dyn = sum(p.numel() for p in dynamics_params)
    logger.info(f"Encoder params: {n_enc:,} @ lr={args.encoder_lr:.1e}")
    logger.info(f"Dynamics params: {n_dyn:,} @ lr={args.dynamics_lr:.1e}")

    optimizer = optim.Adam([
        {"params": encoder_params, "lr": args.encoder_lr},
        {"params": dynamics_params, "lr": args.dynamics_lr},
    ])

    history, r0 = _init_history(ctx)

    logger.info(
        f"\n{'Step':>6}  {'loss':>8}  {'enc':>8}  {'rec':>8}  "
        f"{'sig':>8}  {'dlt':>8}  {'cos':>8}  {'div':>8}  "
        f"{'pred_cs':>8}  {'tgt_cs':>8}  "
        f"{'cos_consec':>11}  {'ratio':>7}")
    logger.info("-" * 122)
    logger.info(
        f"{'0':>6}  {'--':>8}  {'--':>8}  {'--':>8}  "
        f"{'--':>8}  {'--':>8}  {'--':>8}  {'--':>8}  "
        f"{'--':>8}  {'--':>8}  "
        f"{r0['mean_cos_consec']:11.6f}  {r0['mean_ratio']:7.3f}")

    for step in range(1, args.steps + 1):
        model.train()

        latent = model.encode(lat_ctx, act_ctx)

        with torch.no_grad():
            lat_ctx_ema = model.ema_encode(lat_ctx, act_ctx)
        loss_enc = F.mse_loss(latent, lat_ctx_ema)

        ae_tokens_recon = model.decode(latent)
        loss_rec = torch.tensor(0.0, device=device)
        n_mod = 0
        for nm, tok_recon in ae_tokens_recon.items():
            if nm not in lat_ctx:
                continue
            tgt = lat_ctx[nm]
            loss_rec = loss_rec + (
                F.mse_loss(tok_recon, tgt)
                / tgt.detach().var().clamp(min=1e-6))
            n_mod += 1
        if n_mod > 0:
            loss_rec = loss_rec / n_mod

        loss_sig = torch.tensor(0.0, device=device)
        loss_dlt = torch.tensor(0.0, device=device)
        loss_cos = torch.tensor(0.0, device=device)
        loss_div = torch.tensor(0.0, device=device)
        latent_context_ref = latent.detach()
        prev_latent_flat = None
        prev_tgt_flat = None
        pred_cs_sum = 0.0
        tgt_cs_sum = 0.0
        n_pairs = 0

        for k in range(n_rollout):
            act_curr_sig, act_fut_sig = act_step_pairs[k]
            offset_ms = WINDOW_S * 1000 + k * DT_S * 1000

            latent = model.dynamics(
                latent, act_curr_sig, act_fut_sig,
                offset_ms=offset_ms, dt_ms=DT_S * 1000)

            with torch.no_grad():
                lat_target = model.ema_encode(
                    lat_tgt_steps[k], act_ctx_steps[k])

            lat_tgt_var = lat_target.detach().var().clamp(min=1e-6)
            step_weight = (k + 1) / n_rollout
            loss_sig = loss_sig + step_weight * (
                F.mse_loss(latent, lat_target) / lat_tgt_var)

            delta_pred = latent - latent_context_ref
            delta_target = (lat_target - lat_ctx_ema).detach()
            delta_var = delta_target.var().clamp(min=1e-4)
            loss_dlt = loss_dlt + step_weight * (
                F.mse_loss(delta_pred, delta_target) / delta_var)

            # cos (direction of displacement) — see
            # feedback_delta_loss_algebra.md.
            p_flat = delta_pred.reshape(delta_pred.shape[0], -1)
            t_flat = delta_target.reshape(delta_target.shape[0], -1)
            loss_cos = loss_cos + step_weight * (
                1.0 - F.cosine_similarity(p_flat, t_flat, dim=-1)).mean()

            # Consecutive-step cosine; always logged, regularized only
            # when the weight is non-zero.
            if prev_latent_flat is not None and prev_tgt_flat is not None:
                cur_flat = latent.reshape(latent.shape[0], -1)
                tgt_now_flat = lat_target.reshape(
                    lat_target.shape[0], -1)
                pred_cs = F.cosine_similarity(
                    cur_flat, prev_latent_flat, dim=-1)
                tgt_cs = F.cosine_similarity(
                    tgt_now_flat, prev_tgt_flat, dim=-1).detach()
                pred_cs_sum += pred_cs.mean().item()
                tgt_cs_sum += tgt_cs.mean().item()
                n_pairs += 1
                if args.step_diversity_weight > 0.0:
                    loss_div = loss_div + (pred_cs - tgt_cs).pow(2).mean()
            prev_latent_flat = latent.reshape(
                latent.shape[0], -1).detach()
            prev_tgt_flat = lat_target.reshape(
                lat_target.shape[0], -1).detach()

        loss_sig = loss_sig / n_rollout
        loss_dlt = loss_dlt / n_rollout
        loss_cos = loss_cos / n_rollout
        if n_rollout > 1:
            loss_div = loss_div / max(1, n_rollout - 1)

        loss = (0.1 * loss_enc + 1.0 * loss_rec
                + 1.0 * loss_sig
                + args.delta_weight * (loss_dlt + loss_cos)
                + args.step_diversity_weight * loss_div)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        model.update_ema()

        if step % args.eval_every == 0 or step == args.steps:
            r = _record(history, step, loss.item(), ctx)
            mean_pred_cs = pred_cs_sum / max(1, n_pairs)
            mean_tgt_cs = tgt_cs_sum / max(1, n_pairs)
            logger.info(
                f"{step:6d}  {loss.item():8.4f}  {loss_enc.item():8.4f}  "
                f"{loss_rec.item():8.4f}  {loss_sig.item():8.4f}  "
                f"{loss_dlt.item():8.4f}  {loss_cos.item():8.4f}  "
                f"{loss_div.item():8.4f}  "
                f"{mean_pred_cs:8.4f}  {mean_tgt_cs:8.4f}  "
                f"{r['mean_cos_consec']:11.6f}  {r['mean_ratio']:7.3f}")

    return history


# -----------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------

def plot_training_metrics(history, save_path="overfit_rollout_metrics.png"):
    """Plot rollout diversity metrics over training."""
    steps = history["steps"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) Mean consecutive cosine similarity
    ax = axes[0, 0]
    ax.plot(steps, history["mean_cos_consec"], "o-", color="C3", markersize=4)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8,
               label="copying (cos=1)")
    ax.set_ylabel("mean cos_sim(step_k, step_{k-1})")
    ax.set_xlabel("training step")
    ax.set_title("Rollout step-to-step similarity\n(lower = more diverse)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (b) Mean MSE ratio (pred/copy)
    ax = axes[0, 1]
    ax.plot(steps, history["mean_ratio"], "o-", color="C1", markersize=4)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8,
               label="ratio=1 (copy baseline)")
    ax.set_ylabel("mean MSE ratio (pred / copy)")
    ax.set_xlabel("training step")
    ax.set_title("Prediction vs copy baseline\n(lower = better)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (c) cos_vs_step1: before and after training
    ax = axes[1, 0]
    cos_first = history["cos_vs_step1"][0]
    cos_last = history["cos_vs_step1"][-1]
    rollout_steps = list(range(1, len(cos_first) + 1))
    ax.plot(rollout_steps, cos_first, "s--", color="C0", markersize=4,
            label=f"step {history['steps'][0]} (before)")
    ax.plot(rollout_steps, cos_last, "o-", color="C1", markersize=4,
            label=f"step {history['steps'][-1]} (after)")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_ylabel("cos_sim(step_k, step_1)")
    ax.set_xlabel("rollout step")
    ax.set_title("Similarity to first prediction\n(lower = rollout evolves)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (d) Training loss
    ax = axes[1, 1]
    valid = [(s, l) for s, l in zip(steps, history["loss"])
             if not (l != l)]  # skip NaN
    if valid:
        ss, ll = zip(*valid)
        ax.plot(ss, ll, "o-", color="C2", markersize=4)
    ax.set_ylabel("total loss")
    ax.set_xlabel("training step")
    ax.set_title("Training loss")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Overfit test — rollout diversity during training",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Metrics plot saved to {save_path}")


def plot_signal_rollout(ctx, save_path="overfit_rollout_signal.png"):
    """Signal-space rollout at current model state."""
    model = ctx["model"]
    model.eval()
    ae_models = ctx["ae_encoders"]
    act_step_pairs = ctx["act_step_pairs"]
    n_rollout = ctx["n_rollout"]
    batch = ctx["batch"]
    ctx_signals = ctx["ctx_signals"]
    idx = 0

    with torch.no_grad():
        latent = model.encode(ctx["lat_ctx"], ctx["act_ctx"])

        diag_names = [n for n in DIAGNOSTIC_CONFIGS if n in ctx_signals]
        rollout_tails = {name: [] for name in diag_names}

        for k in range(n_rollout):
            act_curr_sig, act_fut_sig = act_step_pairs[k]
            offset_ms = WINDOW_S * 1000 + k * DT_S * 1000
            latent = model.dynamics(
                latent, act_curr_sig, act_fut_sig,
                offset_ms=offset_ms, dt_ms=DT_S * 1000)

            ae_tok = model.decode(latent)
            for name in diag_names:
                cfg = DIAGNOSTIC_CONFIGS[name]
                fs = cfg["target_fs"]
                n_ctx_pts = round(WINDOW_S * fs)
                n_dt = round(DT_S * fs)
                sig = ae_decode(
                    ae_models[name], ae_tok[name],
                    cfg, n_ctx_pts)[idx].detach().cpu()
                rollout_tails[name].append(
                    masked_channel_mean(sig, None)[-n_dt:])

    n_diag = len(diag_names)
    fig, axes = plt.subplots(
        n_diag, 1, figsize=(14, 3.0 * n_diag), squeeze=False)

    for row, name in enumerate(diag_names):
        ax = axes[row, 0]
        cfg = DIAGNOSTIC_CONFIGS[name]
        fs = cfg["target_fs"]

        full_sig = batch[name][idx].cpu()
        gt = masked_channel_mean(full_sig, None)
        t_full = np.arange(len(gt)) / fs * 1000

        ctx_sig_raw = ctx_signals[name][idx].cpu()
        ctx_mean = masked_channel_mean(ctx_sig_raw, None)

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
        ax.set_title(f"{name} — {n_rollout}-step rollout (channel mean)")
        ax.set_xlabel("time [ms]")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Overfit test — signal-space rollout (final)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Signal plot saved to {save_path}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Overfit-one-batch dynamics test with rollout tracking")
    parser.add_argument(
        "--mode", choices=["dynamics_only", "joint_finetune"],
        default="joint_finetune",
        help="dynamics_only: freeze enc/dec, train only dynamics. "
             "joint_finetune: all params, differentiated LR.")
    parser.add_argument(
        "--data_dir", default="/scratch/gpfs/EKOLEMEN/foundation_model/")
    parser.add_argument(
        "--stats_path",
        default="/projects/EKOLEMEN/foundation_model/preprocessing_stats.pt")
    parser.add_argument(
        "--ae_checkpoint_dir",
        default="/projects/EKOLEMEN/foundation_model/")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_latent", type=int, default=64)
    parser.add_argument("--encoder_layers", type=int, default=1)
    parser.add_argument("--processor_layers", type=int, default=1)
    parser.add_argument("--decoder_layers", type=int, default=2)
    parser.add_argument("--dynamics_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=500,
                        help="Total training steps")
    parser.add_argument("--eval_every", type=int, default=25,
                        help="Evaluate rollout every N steps")
    parser.add_argument("--encoder_lr", type=float, default=1e-5)
    parser.add_argument("--dynamics_lr", type=float, default=1e-3)
    parser.add_argument("--n_rollout", type=int, default=8,
                        help="Rollout steps for training and evaluation")
    parser.add_argument("--n_files", type=int, default=5,
                        help="Number of shot files to load")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--delta_weight", type=float, default=1.0,
                        help="Multiplier on the (cos + mag-normalised "
                             "MSE) delta-loss contribution. Matches the "
                             "same flag in train_aurora.py.")
    parser.add_argument("--step_diversity_weight", type=float, default=1.0,
                        help="Weight of the GT-targeted step-diversity "
                             "regularizer: MSE between cos(latent_k, "
                             "latent_{k-1}) and cos(tgt_k, tgt_{k-1}). "
                             "0 disables.")
    args = parser.parse_args()

    ctx = load_data_and_model(args)

    if args.mode == "dynamics_only":
        history = train_dynamics_only(args, ctx)
    else:
        history = train_joint_finetune(args, ctx)

    plot_training_metrics(history)
    plot_signal_rollout(ctx)

    # Final verdict
    cos_before = history["mean_cos_consec"][0]
    cos_after = history["mean_cos_consec"][-1]
    ratio_after = history["mean_ratio"][-1]
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"  cos_consec: {cos_before:.6f} -> {cos_after:.6f}")
    logger.info(f"  mean ratio (pred/copy): {ratio_after:.4f}")
    if cos_after < cos_before - 0.01:
        logger.info("  PASS: Rollout steps are becoming more diverse.")
    else:
        logger.info("  FAIL: Rollout steps remain correlated (copying).")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()