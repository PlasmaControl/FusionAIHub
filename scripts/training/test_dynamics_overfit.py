#!/usr/bin/env python
"""
Overfit-one-batch test for the dynamics model.

Three modes:

  dynamics_only (default)
      Freeze everything except dynamics. Train dynamics to map
      context latent → target latent. Tests raw architecture capacity.

  all_params
      All parameters trainable, all losses active (enc, rec, sig, delta).
      Mimics real training on a single batch. Tests whether competing
      losses prevent the dynamics from learning.

  two_phase
      Phase 1: freeze dynamics, train encoder+decoder (rec + enc).
      Phase 2: freeze encoder+decoder, train dynamics (sig + delta).
      Tests whether stabilising the latent space first lets dynamics learn.

  joint_finetune
      All parameters trainable, all losses active, but dynamics gets a
      much higher LR (--dynamics_lr, default 100x) than the encoder.
      Tests the differentiated learning rate strategy on a single batch.
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

# Reuse configs from the training script
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
# Helpers
# -----------------------------------------------------------------------

def compute_dynamics_metrics(model, latent_ctx, latent_tgt, delta_target,
                             act_curr_sig, act_fut_sig, offset_ms, dt_ms):
    """Compute dynamics prediction metrics (no grad)."""
    with torch.no_grad():
        latent_pred = model.dynamics(
            latent_ctx, act_curr_sig, act_fut_sig,
            offset_ms=offset_ms, dt_ms=dt_ms,
        )
        delta_pred = latent_pred - latent_ctx
        mse = F.mse_loss(latent_pred, latent_tgt).item()
        tgt_var = latent_tgt.var().item()
        cos = F.cosine_similarity(
            delta_pred.flatten(), delta_target.flatten(), dim=0).item()
    return mse, mse / max(tgt_var, 1e-6), delta_pred.norm().item(), cos


def log_dynamics_header():
    logger.info(f"\n{'Step':>6}  {'MSE':>10}  {'MSE/Var':>10}  "
                f"{'||delta_pred||':>14}  {'cos_sim':>8}")
    logger.info("-" * 60)


def log_dynamics_row(step, mse, mse_var, dnorm, cos):
    logger.info(f"{step:6d}  {mse:10.6f}  {mse_var:10.6f}  "
                f"{dnorm:14.4f}  {cos:8.4f}")


def log_summary(label, final_mse, copy_mse, delta_pred_norm,
                delta_target_norm, cos):
    logger.info(f"\n{'='*60}")
    logger.info(f"[{label}]")
    logger.info(f"Copy baseline MSE:  {copy_mse:.6f}")
    logger.info(f"Final dynamics MSE: {final_mse:.6f}")
    logger.info(f"Improvement ratio:  {final_mse / max(copy_mse, 1e-8):.4f}  "
                f"(< 1.0 = better than copy)")
    logger.info(f"Delta cosine sim:   {cos:.4f}  "
                f"(1.0 = perfect direction)")
    logger.info(f"||delta_pred||:     {delta_pred_norm:.4f}  "
                f"(target: {delta_target_norm:.4f})")

    if final_mse < copy_mse * 0.9:
        logger.info("PASS: Dynamics beats copy by >10%.")
    elif final_mse < copy_mse * 0.99:
        logger.info("MARGINAL: Dynamics barely beats copy.")
    else:
        logger.info("FAIL: Dynamics does not beat copy.")


# -----------------------------------------------------------------------
# Loading (shared across modes)
# -----------------------------------------------------------------------

def load_data_and_model(args):
    """Load AEs, one batch, and build a fresh model. Returns a dict."""
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
        all_files[:5],
        lengths_cache_path="lengths_overfit_test.pt",
        preprocessing_stats=stats,
        input_signals=all_signals,
        chunk_duration_s=CHUNK_S,
        step_size_s=CHUNK_S,
        warmup_s=1.0,
        prediction_mode=False,
    )
    loader = make_dataloader(
        ds, batch_size=16, num_workers=2, shuffle=False, pin_memory=True)
    batch = next(iter(loader))
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in batch.items()}

    B = next(v.shape[0] for v in batch.values() if isinstance(v, torch.Tensor))
    logger.info(f"Loaded batch with {len(batch)} keys, B={B}")

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
        dynamics_layers=args.dynamics_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        dynamics_type="cross_attention",
        actuator_configs=ACTUATOR_CONFIGS,
        ema_decay=0.996,
    ).to(device)

    # Precompute AE tokens and actuator signals (fixed across all modes)
    k = args.target_step
    ctx_signals, tgt_signals = {}, {}
    for name, cfg in DIAGNOSTIC_CONFIGS.items():
        if name not in batch:
            continue
        ctx, tgts = split_window(batch[name], cfg["target_fs"],
                                  n_rollout=max(k, 1))
        ctx_signals[name] = ctx
        if k <= len(tgts):
            tgt_signals[name] = tgts[k - 1]

    act_ctx = actuator_context_window(batch, ACTUATOR_CONFIGS, stats)
    act_ctx_tgt = actuator_context_window(
        batch, ACTUATOR_CONFIGS, stats, offset_s=k * DT_S)
    act_step_pairs = actuator_step_windows(
        batch, ACTUATOR_CONFIGS, stats, n_rollout=max(k, 1))
    act_curr_sig, act_fut_sig = act_step_pairs[k - 1]

    with torch.no_grad():
        lat_ctx = encode_batch(ae_encoders, ctx_signals)
        lat_tgt = encode_batch(ae_encoders, tgt_signals)

    offset_ms = WINDOW_S * 1000 + (k - 1) * DT_S * 1000
    dt_ms = DT_S * 1000

    return dict(
        model=model, ae_encoders=ae_encoders, batch=batch, stats=stats,
        lat_ctx=lat_ctx, lat_tgt=lat_tgt,
        act_ctx=act_ctx, act_ctx_tgt=act_ctx_tgt,
        act_curr_sig=act_curr_sig, act_fut_sig=act_fut_sig,
        offset_ms=offset_ms, dt_ms=dt_ms,
        active_diagnostics=active_diagnostics, k=k,
    )


# -----------------------------------------------------------------------
# Mode: dynamics_only (original test)
# -----------------------------------------------------------------------

def run_dynamics_only(args, ctx):
    """Freeze everything except dynamics. Train on one batch."""
    model = ctx["model"]
    lat_ctx, lat_tgt = ctx["lat_ctx"], ctx["lat_tgt"]
    act_ctx, act_ctx_tgt = ctx["act_ctx"], ctx["act_ctx_tgt"]
    act_curr_sig, act_fut_sig = ctx["act_curr_sig"], ctx["act_fut_sig"]
    offset_ms, dt_ms, k = ctx["offset_ms"], ctx["dt_ms"], ctx["k"]

    logger.info(f"\n{'='*60}")
    logger.info("MODE: dynamics_only")
    logger.info(f"{'='*60}")

    # Fixed context/target latents
    with torch.no_grad():
        latent_ctx = model.encode(lat_ctx, act_ctx)
        latent_tgt = model.ema_encode(lat_tgt, act_ctx_tgt)

    delta_target = latent_tgt - latent_ctx
    copy_mse = F.mse_loss(latent_ctx, latent_tgt).item()
    logger.info(f"Target step k={k}, ||delta||={delta_target.norm().item():.4f} "
                f"(relative: {delta_target.norm().item() / latent_ctx.norm().item():.4f}), "
                f"copy MSE={copy_mse:.6f}")

    # Freeze all, unfreeze dynamics
    for p in model.parameters():
        p.requires_grad_(False)
    dynamics_params = []
    for nm, p in model.named_parameters():
        if "dynamics" in nm:
            p.requires_grad_(True)
            dynamics_params.append(p)
    logger.info(f"Trainable: {sum(p.numel() for p in dynamics_params):,} dynamics params")

    optimizer = optim.Adam(dynamics_params, lr=args.encoder_lr)
    log_dynamics_header()

    for step in range(args.steps):
        latent_pred = model.dynamics(
            latent_ctx, act_curr_sig, act_fut_sig,
            offset_ms=offset_ms, dt_ms=dt_ms)
        loss = F.mse_loss(latent_pred, latent_tgt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 25 == 0 or step == args.steps - 1:
            m = compute_dynamics_metrics(
                model, latent_ctx, latent_tgt, delta_target,
                act_curr_sig, act_fut_sig, offset_ms, dt_ms)
            log_dynamics_row(step, *m)

    m = compute_dynamics_metrics(
        model, latent_ctx, latent_tgt, delta_target,
        act_curr_sig, act_fut_sig, offset_ms, dt_ms)
    log_summary("dynamics_only", m[0], copy_mse, m[2],
                delta_target.norm().item(), m[3])


# -----------------------------------------------------------------------
# Mode: all_params (mimics real training on one batch)
# -----------------------------------------------------------------------

def run_all_params(args, ctx):
    """All parameters trainable, all losses. One batch, many steps."""
    model = ctx["model"]
    lat_ctx, lat_tgt = ctx["lat_ctx"], ctx["lat_tgt"]
    act_ctx, act_ctx_tgt = ctx["act_ctx"], ctx["act_ctx_tgt"]
    act_curr_sig, act_fut_sig = ctx["act_curr_sig"], ctx["act_fut_sig"]
    offset_ms, dt_ms, k = ctx["offset_ms"], ctx["dt_ms"], ctx["k"]

    logger.info(f"\n{'='*60}")
    logger.info("MODE: all_params  (mimics real training on one batch)")
    logger.info(f"{'='*60}")

    # All params trainable
    for p in model.parameters():
        p.requires_grad_(True)
    # EMA params stay frozen (updated via EMA, not gradient)
    for p in model.ema_parameters():
        p.requires_grad_(False)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {n_train:,}")

    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.encoder_lr)

    logger.info(f"\n{'Step':>6}  {'total':>8}  {'enc':>8}  {'rec':>8}  "
                f"{'sig':>8}  {'dlt':>8}  {'||delta||':>10}  {'cos':>6}")
    logger.info("-" * 78)

    for step in range(args.steps):
        # --- Forward (mirrors real training loop) ---
        latent = model.encode(lat_ctx, act_ctx)

        # Encode loss
        with torch.no_grad():
            lat_ctx_ema = model.ema_encode(lat_ctx, act_ctx)
        loss_enc = F.mse_loss(latent, lat_ctx_ema)

        # Reconstruction loss
        ae_tokens_recon = model.decode(latent)
        loss_rec = torch.tensor(0.0, device=device)
        n_mod = 0
        for nm, tok_recon in ae_tokens_recon.items():
            if nm not in lat_ctx:
                continue
            tgt = lat_ctx[nm]
            loss_rec = loss_rec + F.mse_loss(tok_recon, tgt) / tgt.detach().var().clamp(min=1e-6)
            n_mod += 1
        if n_mod > 0:
            loss_rec = loss_rec / n_mod

        # Dynamics step
        latent_pred = model.dynamics(
            latent, act_curr_sig, act_fut_sig,
            offset_ms=offset_ms, dt_ms=dt_ms)

        with torch.no_grad():
            lat_target = model.ema_encode(lat_tgt, act_ctx_tgt)

        # Signal loss (latent space)
        lat_tgt_var = lat_target.detach().var().clamp(min=1e-6)
        loss_sig = F.mse_loss(latent_pred, lat_target) / lat_tgt_var

        # Delta loss
        latent_context_ref = latent.detach()
        delta_pred = latent_pred - latent_context_ref
        delta_target = (lat_target - lat_ctx_ema).detach()
        delta_var = delta_target.var().clamp(min=1e-4)
        loss_dlt = F.mse_loss(delta_pred, delta_target) / delta_var

        loss = 0.1 * loss_enc + 1.0 * loss_rec + 1.0 * loss_sig + 1.0 * loss_dlt

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        model.update_ema()

        if step % 25 == 0 or step == args.steps - 1:
            with torch.no_grad():
                dn = delta_pred.norm().item()
                cos = F.cosine_similarity(
                    delta_pred.flatten(), delta_target.flatten(), dim=0
                ).item()
            logger.info(
                f"{step:6d}  {loss.item():8.4f}  {loss_enc.item():8.4f}  "
                f"{loss_rec.item():8.4f}  {loss_sig.item():8.4f}  "
                f"{loss_dlt.item():8.4f}  {dn:10.4f}  {cos:6.3f}")

    # Final dynamics evaluation
    with torch.no_grad():
        latent_final = model.encode(lat_ctx, act_ctx)
        latent_pred_final = model.dynamics(
            latent_final, act_curr_sig, act_fut_sig,
            offset_ms=offset_ms, dt_ms=dt_ms)
        lat_target_final = model.ema_encode(lat_tgt, act_ctx_tgt)
        copy_mse = F.mse_loss(latent_final, lat_target_final).item()
        pred_mse = F.mse_loss(latent_pred_final, lat_target_final).item()
        dp = latent_pred_final - latent_final
        dt = lat_target_final - model.ema_encode(lat_ctx, act_ctx)
        cos = F.cosine_similarity(dp.flatten(), dt.flatten(), dim=0).item()

    log_summary("all_params", pred_mse, copy_mse, dp.norm().item(),
                dt.norm().item(), cos)


# -----------------------------------------------------------------------
# Mode: two_phase
# -----------------------------------------------------------------------

def run_two_phase(args, ctx):
    """Phase 1: train encoder/decoder. Phase 2: train dynamics."""
    model = ctx["model"]
    lat_ctx, lat_tgt = ctx["lat_ctx"], ctx["lat_tgt"]
    act_ctx, act_ctx_tgt = ctx["act_ctx"], ctx["act_ctx_tgt"]
    act_curr_sig, act_fut_sig = ctx["act_curr_sig"], ctx["act_fut_sig"]
    offset_ms, dt_ms, k = ctx["offset_ms"], ctx["dt_ms"], ctx["k"]

    logger.info(f"\n{'='*60}")
    logger.info("MODE: two_phase")
    logger.info(f"{'='*60}")

    # ---- Phase 1: train encoder+decoder, freeze dynamics ----
    logger.info(f"\n--- Phase 1: encoder+decoder ({args.steps} steps) ---")

    for p in model.parameters():
        p.requires_grad_(True)
    for p in model.ema_parameters():
        p.requires_grad_(False)
    # Freeze dynamics
    for nm, p in model.named_parameters():
        if "dynamics" in nm:
            p.requires_grad_(False)

    phase1_params = [p for p in model.parameters() if p.requires_grad]
    n_p1 = sum(p.numel() for p in phase1_params)
    logger.info(f"Phase 1 trainable: {n_p1:,} (encoder+decoder+tokenizer)")

    optimizer1 = optim.Adam(phase1_params, lr=args.encoder_lr)

    logger.info(f"\n{'Step':>6}  {'enc':>10}  {'rec':>10}")
    logger.info("-" * 32)

    for step in range(args.steps):
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
            loss_rec = loss_rec + F.mse_loss(tok_recon, tgt) / tgt.detach().var().clamp(min=1e-6)
            n_mod += 1
        if n_mod > 0:
            loss_rec = loss_rec / n_mod

        loss = 0.1 * loss_enc + 1.0 * loss_rec

        optimizer1.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer1.step()
        model.update_ema()

        if step % 25 == 0 or step == args.steps - 1:
            logger.info(f"{step:6d}  {loss_enc.item():10.6f}  "
                        f"{loss_rec.item():10.6f}")

    # ---- Phase 2: freeze encoder+decoder, train dynamics ----
    logger.info(f"\n--- Phase 2: dynamics only ({args.steps} steps) ---")

    # Freeze everything, unfreeze dynamics
    for p in model.parameters():
        p.requires_grad_(False)
    dynamics_params = []
    for nm, p in model.named_parameters():
        if "dynamics" in nm:
            p.requires_grad_(True)
            dynamics_params.append(p)

    n_p2 = sum(p.numel() for p in dynamics_params)
    logger.info(f"Phase 2 trainable: {n_p2:,} (dynamics)")

    # Re-encode with the now-stable encoder
    with torch.no_grad():
        latent_ctx = model.encode(lat_ctx, act_ctx)
        latent_tgt = model.ema_encode(lat_tgt, act_ctx_tgt)
        lat_ctx_ema = model.ema_encode(lat_ctx, act_ctx)

    delta_target = latent_tgt - latent_ctx
    copy_mse = F.mse_loss(latent_ctx, latent_tgt).item()
    logger.info(f"After phase 1: ||delta||={delta_target.norm().item():.4f}, "
                f"copy MSE={copy_mse:.6f}")

    optimizer2 = optim.Adam(dynamics_params, lr=args.encoder_lr)
    log_dynamics_header()

    for step in range(args.steps):
        latent_pred = model.dynamics(
            latent_ctx, act_curr_sig, act_fut_sig,
            offset_ms=offset_ms, dt_ms=dt_ms)
        loss = F.mse_loss(latent_pred, latent_tgt)

        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()

        if step % 25 == 0 or step == args.steps - 1:
            m = compute_dynamics_metrics(
                model, latent_ctx, latent_tgt, delta_target,
                act_curr_sig, act_fut_sig, offset_ms, dt_ms)
            log_dynamics_row(step, *m)

    m = compute_dynamics_metrics(
        model, latent_ctx, latent_tgt, delta_target,
        act_curr_sig, act_fut_sig, offset_ms, dt_ms)
    log_summary("two_phase", m[0], copy_mse, m[2],
                delta_target.norm().item(), m[3])


# -----------------------------------------------------------------------
# Mode: joint_finetune (differentiated LR)
# -----------------------------------------------------------------------

def run_joint_finetune(args, ctx):
    """All params trainable, differentiated LR: dynamics gets higher rate."""
    model = ctx["model"]
    lat_ctx, lat_tgt = ctx["lat_ctx"], ctx["lat_tgt"]
    act_ctx, act_ctx_tgt = ctx["act_ctx"], ctx["act_ctx_tgt"]
    act_curr_sig, act_fut_sig = ctx["act_curr_sig"], ctx["act_fut_sig"]
    offset_ms, dt_ms, k = ctx["offset_ms"], ctx["dt_ms"], ctx["k"]

    logger.info(f"\n{'='*60}")
    logger.info("MODE: joint_finetune  (differentiated LR)")
    logger.info(f"{'='*60}")

    # All params trainable
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
    logger.info(f"LR ratio: {args.dynamics_lr / args.encoder_lr:.0f}x")

    optimizer = optim.Adam([
        {"params": encoder_params, "lr": args.encoder_lr},
        {"params": dynamics_params, "lr": args.dynamics_lr},
    ])

    logger.info(f"\n{'Step':>6}  {'total':>8}  {'enc':>8}  {'rec':>8}  "
                f"{'sig':>8}  {'dlt':>8}  {'||delta||':>10}  {'cos':>6}")
    logger.info("-" * 78)

    for step in range(args.steps):
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
            loss_rec = loss_rec + F.mse_loss(tok_recon, tgt) / tgt.detach().var().clamp(min=1e-6)
            n_mod += 1
        if n_mod > 0:
            loss_rec = loss_rec / n_mod

        latent_pred = model.dynamics(
            latent, act_curr_sig, act_fut_sig,
            offset_ms=offset_ms, dt_ms=dt_ms)

        with torch.no_grad():
            lat_target = model.ema_encode(lat_tgt, act_ctx_tgt)

        lat_tgt_var = lat_target.detach().var().clamp(min=1e-6)
        loss_sig = F.mse_loss(latent_pred, lat_target) / lat_tgt_var

        latent_context_ref = latent.detach()
        delta_pred = latent_pred - latent_context_ref
        delta_target = (lat_target - lat_ctx_ema).detach()
        delta_var = delta_target.var().clamp(min=1e-4)
        loss_dlt = F.mse_loss(delta_pred, delta_target) / delta_var

        loss = 0.1 * loss_enc + 1.0 * loss_rec + 1.0 * loss_sig + 1.0 * loss_dlt

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        model.update_ema()

        if step % 25 == 0 or step == args.steps - 1:
            with torch.no_grad():
                dn = delta_pred.norm().item()
                cos = F.cosine_similarity(
                    delta_pred.flatten(), delta_target.flatten(), dim=0
                ).item()
            logger.info(
                f"{step:6d}  {loss.item():8.4f}  {loss_enc.item():8.4f}  "
                f"{loss_rec.item():8.4f}  {loss_sig.item():8.4f}  "
                f"{loss_dlt.item():8.4f}  {dn:10.4f}  {cos:6.3f}")

    # Final dynamics evaluation
    with torch.no_grad():
        latent_final = model.encode(lat_ctx, act_ctx)
        latent_pred_final = model.dynamics(
            latent_final, act_curr_sig, act_fut_sig,
            offset_ms=offset_ms, dt_ms=dt_ms)
        lat_target_final = model.ema_encode(lat_tgt, act_ctx_tgt)
        copy_mse = F.mse_loss(latent_final, lat_target_final).item()
        pred_mse = F.mse_loss(latent_pred_final, lat_target_final).item()
        dp = latent_pred_final - latent_final
        dt = lat_target_final - model.ema_encode(lat_ctx, act_ctx)
        cos = F.cosine_similarity(dp.flatten(), dt.flatten(), dim=0).item()

    log_summary("joint_finetune", pred_mse, copy_mse, dp.norm().item(),
                dt.norm().item(), cos)


# -----------------------------------------------------------------------
# Rollout evaluation (runs after any training mode)
# -----------------------------------------------------------------------

@torch.no_grad()
def run_rollout_eval(ctx, n_steps=16):
    """Chain N dynamics steps and compare each to its target."""
    model = ctx["model"]
    model.eval()
    lat_ctx, lat_tgt = ctx["lat_ctx"], ctx["lat_tgt"]
    act_ctx, act_ctx_tgt = ctx["act_ctx"], ctx["act_ctx_tgt"]
    batch, stats = ctx["batch"], ctx["stats"]

    # Split all diagnostic signals into context + n_steps targets
    ctx_signals, tgt_signals_steps = {}, [{} for _ in range(n_steps)]
    for name, cfg in DIAGNOSTIC_CONFIGS.items():
        if name not in batch:
            continue
        c, tgts = split_window(batch[name], cfg["target_fs"],
                                n_rollout=n_steps)
        ctx_signals[name] = c
        for k, tgt in enumerate(tgts):
            tgt_signals_steps[k][name] = tgt

    # AE-encode all target steps
    lat_tgt_steps = [encode_batch(ctx["ae_encoders"], tgt_s)
                     for tgt_s in tgt_signals_steps]

    # Actuator signals for each step
    act_step_pairs = actuator_step_windows(
        batch, ACTUATOR_CONFIGS, stats, n_rollout=n_steps)

    # Per-step actuator contexts for EMA targets
    act_ctx_steps = [
        actuator_context_window(
            batch, ACTUATOR_CONFIGS, stats,
            offset_s=(k + 1) * DT_S)
        for k in range(n_steps)
    ]

    # Encode context
    latent_ctx = model.encode(lat_ctx, act_ctx)
    lat_ctx_ema = model.ema_encode(lat_ctx, act_ctx)

    # EMA-encode all targets
    lat_tgt_encoded = [
        model.ema_encode(lat_tgt_steps[k], act_ctx_steps[k])
        for k in range(n_steps)
    ]

    # Autoregressive rollout — collect metrics
    logger.info(f"\n{'='*60}")
    logger.info(f"Rollout evaluation ({n_steps} steps)")
    logger.info(f"{'='*60}")
    logger.info(f"\n{'Step':>4}  {'t[ms]':>7}  {'MSE_pred':>10}  "
                f"{'MSE_copy':>10}  {'ratio':>7}  {'||dlt_p||':>10}  "
                f"{'||dlt_t||':>10}  {'cos':>6}")
    logger.info("-" * 78)

    steps_t = []
    mse_preds, mse_copies, ratios = [], [], []
    dlt_pred_norms, dlt_tgt_norms, cos_sims = [], [], []

    latent = latent_ctx.clone()
    for k in range(n_steps):
        act_curr_sig, act_fut_sig = act_step_pairs[k]
        offset_ms = WINDOW_S * 1000 + k * DT_S * 1000
        latent = model.dynamics(
            latent, act_curr_sig, act_fut_sig,
            offset_ms=offset_ms, dt_ms=DT_S * 1000)

        lat_target = lat_tgt_encoded[k]
        mse_pred = F.mse_loss(latent, lat_target).item()
        mse_copy = F.mse_loss(latent_ctx, lat_target).item()
        ratio = mse_pred / max(mse_copy, 1e-8)

        delta_pred = latent - latent_ctx
        delta_target = lat_target - lat_ctx_ema
        dp_norm = delta_pred.norm().item()
        dt_norm = delta_target.norm().item()
        cos = F.cosine_similarity(
            delta_pred.flatten(), delta_target.flatten(), dim=0).item()

        t_ms = (k + 1) * DT_S * 1000
        steps_t.append(t_ms)
        mse_preds.append(mse_pred)
        mse_copies.append(mse_copy)
        ratios.append(ratio)
        dlt_pred_norms.append(dp_norm)
        dlt_tgt_norms.append(dt_norm)
        cos_sims.append(cos)

        logger.info(
            f"{k+1:4d}  {t_ms:7.0f}  {mse_pred:10.6f}  "
            f"{mse_copy:10.6f}  {ratio:7.3f}  "
            f"{dp_norm:10.4f}  {dt_norm:10.4f}  {cos:6.3f}")

    logger.info(f"\nratio < 1.0 = dynamics beats copy at that step")

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    t = np.array(steps_t) / 1000  # seconds

    # (a) MSE: prediction vs copy baseline
    ax = axes[0, 0]
    ax.plot(t, mse_preds, "o-", color="C1", label="dynamics prediction")
    ax.plot(t, mse_copies, "s--", color="C0", label="copy baseline")
    ax.set_ylabel("MSE vs target")
    ax.set_xlabel("time [s]")
    ax.set_title("Prediction MSE vs copy baseline")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (b) Ratio (pred/copy)
    ax = axes[0, 1]
    ax.plot(t, ratios, "o-", color="C3")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8,
               label="ratio = 1 (copy)")
    ax.set_ylabel("MSE ratio (pred / copy)")
    ax.set_xlabel("time [s]")
    ax.set_title("Prediction / copy ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (c) Delta norms: predicted vs target
    ax = axes[1, 0]
    ax.plot(t, dlt_pred_norms, "o-", color="C1", label="||delta_pred||")
    ax.plot(t, dlt_tgt_norms, "s--", color="C0", label="||delta_target||")
    ax.set_ylabel("L2 norm")
    ax.set_xlabel("time [s]")
    ax.set_title("Delta magnitude: predicted vs target")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (d) Cosine similarity
    ax = axes[1, 1]
    ax.plot(t, cos_sims, "o-", color="C2")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_ylim(-0.2, 1.05)
    ax.set_ylabel("cosine similarity")
    ax.set_xlabel("time [s]")
    ax.set_title("Delta direction (cos_sim)")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Rollout evaluation — latent space", fontsize=13,
                 fontweight="bold")
    fig.tight_layout()
    save_path = Path("rollout_eval_latent.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Latent plot saved to {save_path}")

    # --- Signal-space rollout plot ---
    # Decode each rollout step back to signal space via Perceiver decoder
    # + AE decoder, and stitch into a continuous timeline.
    ae_models = ctx["ae_encoders"]
    idx = 0  # first sample in batch

    # Re-run the rollout, decoding at each step
    latent = latent_ctx.clone()
    diag_names = [n for n in DIAGNOSTIC_CONFIGS if n in ctx_signals]
    rollout_tails = {name: [] for name in diag_names}

    for k in range(n_steps):
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
    fig_sig, axes_sig = plt.subplots(
        n_diag, 1, figsize=(14, 3.0 * n_diag), squeeze=False)

    for row, name in enumerate(diag_names):
        ax = axes_sig[row, 0]
        cfg = DIAGNOSTIC_CONFIGS[name]
        fs = cfg["target_fs"]

        # Ground truth: full chunk (channel mean)
        full_sig = batch[name][idx].cpu()
        gt = masked_channel_mean(full_sig, None)
        t_full = np.arange(len(gt)) / fs * 1000

        # Context: raw signal (channel mean)
        ctx_sig_raw = ctx_signals[name][idx].cpu()
        ctx_mean = masked_channel_mean(ctx_sig_raw, None)

        # Stitch: context + rolled-out tails
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
        ax.set_title(f"{name} — {n_steps}-step rollout (channel mean)")
        ax.set_xlabel("time [ms]")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    fig_sig.suptitle("Rollout evaluation — signal space",
                     fontsize=13, fontweight="bold")
    fig_sig.tight_layout()
    save_path_sig = Path("rollout_eval_signal.png")
    fig_sig.savefig(save_path_sig, dpi=150, bbox_inches="tight")
    plt.close(fig_sig)
    logger.info(f"Signal plot saved to {save_path_sig}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Overfit-one-batch dynamics test")
    parser.add_argument(
        "--mode", choices=["dynamics_only", "all_params", "two_phase",
                           "joint_finetune"],
        default="joint_finetune",
        help="dynamics_only: freeze all except dynamics. "
             "all_params: all trainable, all losses. "
             "two_phase: train enc/dec first, then dynamics. "
             "joint_finetune: all trainable, differentiated LR.")
    parser.add_argument(
        "--data_dir", default="/scratch/gpfs/EKOLEMEN/foundation_model/")
    parser.add_argument(
        "--stats_path",
        default="/projects/EKOLEMEN/foundation_model/preprocessing_stats.pt")
    parser.add_argument(
        "--ae_checkpoint_dir",
        default="/projects/EKOLEMEN/foundation_model/")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_latent", type=int, default=128)
    parser.add_argument("--encoder_layers", type=int, default=1)
    parser.add_argument("--processor_layers", type=int, default=1)
    parser.add_argument("--decoder_layers", type=int, default=2)
    parser.add_argument("--dynamics_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=500,
                        help="Optimization steps (per phase for two_phase)")
    parser.add_argument("--encoder_lr", type=float, default=1e-5)
    parser.add_argument("--dynamics_lr", type=float, default=1e-3,
                        help="LR for dynamics in joint_finetune mode")
    parser.add_argument("--target_step", type=int, default=1,
                        help="Which rollout step to use as target (1..16)")
    args = parser.parse_args()

    ctx = load_data_and_model(args)

    if args.mode == "dynamics_only":
        run_dynamics_only(args, ctx)
    elif args.mode == "all_params":
        run_all_params(args, ctx)
    elif args.mode == "two_phase":
        run_two_phase(args, ctx)
    elif args.mode == "joint_finetune":
        run_joint_finetune(args, ctx)

    # Rollout evaluation after any training mode
    run_rollout_eval(ctx, n_steps=min(16, N_ROLLOUT))


if __name__ == "__main__":
    main()
