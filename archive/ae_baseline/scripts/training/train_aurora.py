#!/usr/bin/env python
"""
Training script for the Aurora-inspired tokamak foundation model.

Phase 1: Single-step pretraining (AE tokens at t → AE tokens at t+dt).
Phase 2: Multi-step fine-tuning (full backprop through K-step rollout).

Loss is per-modality MAE in AE token space — no EMA, no latent-space
loss, no delta loss.  A single reconstruction regularizer
(decode(encode(x)) ≈ x) is optionally used in Phase 1.
"""

from pathlib import Path
import argparse
import logging
import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader

from tokamak_foundation_model.data.multi_file_dataset import (
    TokamakMultiFileDataset, make_dataloader,
)
from tokamak_foundation_model.models.aurora import TokamakFoundationModel

# Reuse data pipeline from the existing training script
from train_foundation_model import (
    DIAGNOSTIC_CONFIGS,
    ACTUATOR_CONFIGS,
    load_ae,
    split_window,
    encode_batch,
    ae_decode,
    actuator_context_window,
    actuator_step_windows,
    _select_channels,
    _normalize_actuator,
    masked_channel_mean,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DT_S: float = 0.05
WINDOW_S: float = 0.05


def _encode_batch_grad(ae_models, signals, ae_token_stats=None):
    """Like :func:`encode_batch` but without ``@torch.no_grad`` — used
    when AE encoders are unfrozen and their gradients must flow through
    the recon regulariser and the foundation model's prediction loss.
    """
    result = {}
    for name, ae in ae_models.items():
        if name not in signals:
            continue
        z = ae.encoder(signals[name])
        z = z.clamp(-50, 50)
        if ae_token_stats is not None and name in ae_token_stats:
            mean = ae_token_stats[name]["mean"].to(z.device)
            std = ae_token_stats[name]["std"].to(z.device)
            z = (z - mean) / std
        result[name] = z
    return result


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------


def run_phase1_epoch(
    model: TokamakFoundationModel,
    ae_models: dict,
    loader: DataLoader,
    optimizer: Optional[optim.Optimizer],
    is_train: bool,
    preprocess_stats: dict,
    recon_weight: float = 0.1,
    max_steps: int = 0,
    n_rollout: int = 1,
    ae_token_stats: Optional[dict] = None,
    use_delta_loss: bool = True,
    delta_weight: float = 1.0,
    encoder_optimizer: Optional[optim.Optimizer] = None,
) -> tuple[float, float, float]:
    """Phase 1: single-step prediction.

    When *recon_weight* > 0, the AE encoders are assumed to be unfrozen;
    context signals flow through the encoder with gradients and an
    MSE reconstruction regulariser (via the frozen decoder) anchors
    the encoder to its original manifold. Targets are still encoded
    under no_grad (no gradient path through the target side).

    Returns (mae_loss, mag_loss, recon_loss).
    """
    model.train(is_train)
    use_recon = recon_weight > 0.0
    if use_recon:
        for ae in ae_models.values():
            ae.encoder.train(is_train)
    sum_mae, sum_mag, sum_recon, n = 0.0, 0.0, 0.0, 0

    for batch in loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        ctx_signals = {}
        tgt_signals = {}
        for name, cfg in DIAGNOSTIC_CONFIGS.items():
            if name not in batch:
                continue
            ctx, tgts = split_window(batch[name], cfg["target_fs"],
                                      n_rollout=1)
            ctx_signals[name] = ctx
            tgt_signals[name] = tgts[0]

        if not ctx_signals:
            continue

        if use_recon:
            # Gradient-enabled encode for context (feeds both the
            # foundation model and the recon regulariser).
            ae_ctx = _encode_batch_grad(
                ae_models, ctx_signals, ae_token_stats)
            with torch.no_grad():
                ae_tgt = encode_batch(
                    ae_models, tgt_signals, ae_token_stats)
        else:
            with torch.no_grad():
                ae_ctx = encode_batch(
                    ae_models, ctx_signals, ae_token_stats)
                ae_tgt = encode_batch(
                    ae_models, tgt_signals, ae_token_stats)

        act_ctx = actuator_context_window(
            batch, ACTUATOR_CONFIGS, preprocess_stats)
        act_steps = actuator_step_windows(
            batch, ACTUATOR_CONFIGS, preprocess_stats, n_rollout=1)
        act_curr, act_fut = act_steps[0]

        # Forward pass
        ae_pred = model.forward(
            ae_tokens=ae_ctx,
            act_curr_signals=act_curr,
            act_fut_signals=act_fut,
            step_index=0,
            offset_ms=WINDOW_S * 1000,
            dt_ms=DT_S * 1000,
        )

        # MAE + proper delta loss (cos + mag) in AE token space.  The
        # cos term is the only part of the loss that rewards matching
        # the *direction* of the context→target displacement; without
        # it, F.l1_loss(pred − ctx, tgt − ctx) reduces algebraically to
        # F.l1_loss(pred, tgt) (see feedback_delta_loss_algebra.md).
        loss_mae = torch.tensor(0.0, device=device)
        loss_mag = torch.tensor(0.0, device=device)
        loss_cos = torch.tensor(0.0, device=device)
        n_mod = 0
        for m in ae_pred:
            if m not in ae_tgt or m not in ae_ctx:
                continue
            loss_mae = loss_mae + F.l1_loss(ae_pred[m], ae_tgt[m])
            pred_d = ae_pred[m] - ae_ctx[m]
            tgt_d = ae_tgt[m] - ae_ctx[m]
            loss_mag = loss_mag + F.l1_loss(
                pred_d.norm(dim=-1), tgt_d.norm(dim=-1))
            p_flat = pred_d.reshape(pred_d.shape[0], -1)
            t_flat = tgt_d.reshape(tgt_d.shape[0], -1)
            loss_cos = loss_cos + (
                1.0 - F.cosine_similarity(p_flat, t_flat, dim=-1)).mean()
            n_mod += 1
        if n_mod > 0:
            loss_mae = loss_mae / n_mod
            loss_mag = loss_mag / n_mod
            loss_cos = loss_cos / n_mod

        # Reconstruction regulariser — anchors unfrozen encoders to
        # the frozen decoder's input manifold.
        loss_recon = torch.tensor(0.0, device=device)
        if use_recon:
            recon_losses = []
            for name in ae_ctx:
                if name not in ctx_signals:
                    continue
                recon = ae_decode(
                    ae_models[name], ae_ctx[name],
                    DIAGNOSTIC_CONFIGS[name],
                    output_length=ctx_signals[name].shape[-1],
                    ae_token_stats=ae_token_stats,
                    modality_name=name,
                )
                recon_losses.append(F.mse_loss(recon, ctx_signals[name]))
            if recon_losses:
                loss_recon = torch.stack(recon_losses).mean()

        if use_delta_loss:
            loss = loss_mae + delta_weight * (loss_cos + loss_mag)
        else:
            loss = loss_mae
        loss = loss + recon_weight * loss_recon

        if is_train:
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning("NaN/Inf loss — skipping batch")
                optimizer.zero_grad()
                if encoder_optimizer is not None:
                    encoder_optimizer.zero_grad()
                continue
            optimizer.zero_grad()
            if encoder_optimizer is not None:
                encoder_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if encoder_optimizer is not None:
                encoder_params = [
                    p for group in encoder_optimizer.param_groups
                    for p in group["params"]
                ]
                nn.utils.clip_grad_norm_(encoder_params, max_norm=1.0)
            optimizer.step()
            if encoder_optimizer is not None:
                encoder_optimizer.step()

        sum_mae += loss_mae.item()
        sum_mag += loss_mag.item()
        sum_recon += loss_recon.item()
        n += 1
        if max_steps and n >= max_steps:
            break

    d = max(n, 1)
    return sum_mae / d, sum_mag / d, sum_recon / d


def run_phase2_epoch(
    model: TokamakFoundationModel,
    ae_models: dict,
    loader: DataLoader,
    optimizer: Optional[optim.Optimizer],
    is_train: bool,
    preprocess_stats: dict,
    n_rollout: int = 4,
    max_steps: int = 0,
    ae_token_stats: Optional[dict] = None,
    use_delta_loss: bool = True,
    delta_weight: float = 1.0,
    step_diversity_weight: float = 0.0,
) -> tuple[float, float]:
    """Phase 2: multi-step rollout with full backprop.

    Returns (total_mae_loss, last_step_mae).
    """
    model.train(is_train)
    sum_total, sum_last, n = 0.0, 0.0, 0

    for batch in loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

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

        if not ctx_signals:
            continue

        with torch.no_grad():
            ae_ctx = encode_batch(ae_models, ctx_signals, ae_token_stats)
            ae_tgt_steps = [encode_batch(ae_models, tgt_s, ae_token_stats)
                            for tgt_s in tgt_signals_steps]

        act_step_pairs = actuator_step_windows(
            batch, ACTUATOR_CONFIGS, preprocess_stats,
            n_rollout=n_rollout)

        # Autoregressive rollout with gradients
        current = ae_ctx
        total_loss = torch.tensor(0.0, device=device)
        last_step_loss = 0.0
        # Previous step's prediction AND target, flattened per modality
        # and detached — used by the step-diversity regularizer to
        # target the ground-truth step-to-step cosine.
        prev_pred_flat: Optional[dict] = None
        prev_tgt_flat: Optional[dict] = None

        for k in range(n_rollout):
            act_curr, act_fut = act_step_pairs[k]
            offset_ms = WINDOW_S * 1000 + k * DT_S * 1000

            step_ctx = {m: t.detach() for m, t in current.items()}
            current = model.forward(
                ae_tokens=current,
                act_curr_signals=act_curr,
                act_fut_signals=act_fut,
                step_index=k,
                offset_ms=offset_ms,
                dt_ms=DT_S * 1000,
            )

            # Per-modality MAE + proper delta loss (cos + mag).  The
            # cos term is what prevents the loss from collapsing to a
            # plain L1 on (pred, tgt) — see feedback_delta_loss_algebra.md.
            step_loss = torch.tensor(0.0, device=device)
            n_mod = 0
            for m in current:
                if m not in ae_tgt_steps[k] or m not in step_ctx:
                    continue
                loss_mae = F.l1_loss(current[m], ae_tgt_steps[k][m])
                if use_delta_loss:
                    pred_d = current[m] - step_ctx[m]
                    tgt_d = ae_tgt_steps[k][m] - step_ctx[m]
                    mag_loss = F.l1_loss(
                        pred_d.norm(dim=-1), tgt_d.norm(dim=-1))
                    p_flat = pred_d.reshape(pred_d.shape[0], -1)
                    t_flat = tgt_d.reshape(tgt_d.shape[0], -1)
                    cos_loss = (1.0 - F.cosine_similarity(
                        p_flat, t_flat, dim=-1)).mean()
                    step_loss = step_loss + loss_mae \
                        + delta_weight * (cos_loss + mag_loss)
                else:
                    step_loss = step_loss + loss_mae
                n_mod += 1
            if n_mod > 0:
                step_loss = step_loss / n_mod

            # Step-diversity regularizer: per-modality, per-batch,
            # push cos(pred_k, pred_{k-1}) to match cos(tgt_k, tgt_{k-1}).
            # The previous hinge-based variant was bounded and couldn't
            # pull predictions off the cos ≈ 1 fixed point; this
            # GT-targeted MSE is self-calibrating (no threshold to tune)
            # and gradient-scales with the observed target variability.
            if (prev_pred_flat is not None
                    and prev_tgt_flat is not None
                    and step_diversity_weight > 0.0):
                div_pen = torch.tensor(0.0, device=device)
                n_div = 0
                for m in current:
                    if m not in prev_pred_flat or m not in prev_tgt_flat:
                        continue
                    cur_flat = current[m].reshape(current[m].shape[0], -1)
                    tgt_now_flat = ae_tgt_steps[k][m].reshape(
                        ae_tgt_steps[k][m].shape[0], -1)
                    pred_cs = F.cosine_similarity(
                        cur_flat, prev_pred_flat[m], dim=-1)
                    tgt_cs = F.cosine_similarity(
                        tgt_now_flat, prev_tgt_flat[m], dim=-1).detach()
                    div_pen = div_pen + (pred_cs - tgt_cs).pow(2).mean()
                    n_div += 1
                if n_div > 0:
                    step_loss = step_loss + step_diversity_weight * (
                        div_pen / n_div)

            # Save detached, flattened tensors for the next step's
            # GT-targeted diversity penalty.
            prev_pred_flat = {
                m: current[m].reshape(current[m].shape[0], -1).detach()
                for m in current
            }
            prev_tgt_flat = {
                m: ae_tgt_steps[k][m].reshape(
                    ae_tgt_steps[k][m].shape[0], -1).detach()
                for m in ae_tgt_steps[k]
            }

            step_weight = (k + 1) / n_rollout
            total_loss = total_loss + step_weight * step_loss

            if k == n_rollout - 1:
                last_step_loss = step_loss.item()

        total_loss = total_loss / n_rollout

        if is_train:
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                logger.warning("NaN/Inf loss — skipping batch")
                optimizer.zero_grad()
                continue
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        sum_total += total_loss.item()
        sum_last += last_step_loss
        n += 1
        if max_steps and n >= max_steps:
            break

    d = max(n, 1)
    return sum_total / d, sum_last / d


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


@torch.no_grad()
def log_diagnostics(
    model: TokamakFoundationModel,
    ae_models: dict,
    loader: DataLoader,
    preprocess_stats: dict,
    n_rollout: int,
    ae_token_stats: Optional[dict] = None,
) -> None:
    """Log per-step delta norms and decoded cos_sim in AE token space."""
    model.eval()

    for batch in loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

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
        if not ctx_signals:
            return

        ae_ctx = encode_batch(ae_models, ctx_signals, ae_token_stats)
        act_step_pairs = actuator_step_windows(
            batch, ACTUATOR_CONFIGS, preprocess_stats,
            n_rollout=n_rollout)

        B = next(iter(ae_ctx.values())).shape[0]

        def _flatten(tok):
            return torch.cat([t.reshape(B, -1) for t in tok.values()], dim=1)

        ctx_flat = _flatten(ae_ctx)
        current = ae_ctx
        pred_deltas = []
        tgt_deltas = []
        model_cos_sims = []
        gt_cos_sims = []
        prev_pred_flat = None
        prev_tgt_flat = None

        for k in range(n_rollout):
            act_curr, act_fut = act_step_pairs[k]
            offset_ms = WINDOW_S * 1000 + k * DT_S * 1000

            current = model.forward(
                ae_tokens=current,
                act_curr_signals=act_curr,
                act_fut_signals=act_fut,
                step_index=k,
                offset_ms=offset_ms,
                dt_ms=DT_S * 1000,
            )

            pred_flat = _flatten(current)
            pred_deltas.append(
                (pred_flat - ctx_flat).norm(dim=-1).mean().item())

            ae_tgt = encode_batch(ae_models, tgt_signals_steps[k], ae_token_stats)
            tgt_flat = _flatten(ae_tgt)
            tgt_deltas.append(
                (tgt_flat - ctx_flat).norm(dim=-1).mean().item())

            if prev_pred_flat is not None:
                model_cos = F.cosine_similarity(
                    pred_flat, prev_pred_flat, dim=1)
                model_cos_sims.append(model_cos.mean().item())
            if prev_tgt_flat is not None:
                gt_cos = F.cosine_similarity(
                    tgt_flat, prev_tgt_flat, dim=1)
                gt_cos_sims.append(gt_cos.mean().item())
            prev_pred_flat = pred_flat
            prev_tgt_flat = tgt_flat

        pd_str = " ".join(f"{v:.3f}" for v in pred_deltas)
        td_str = " ".join(f"{v:.3f}" for v in tgt_deltas)
        mc_str = " ".join(f"{v:.4f}" for v in model_cos_sims)
        gc_str = " ".join(f"{v:.4f}" for v in gt_cos_sims)
        logger.info(
            f"  [aurora diag] pred_delta=[{pd_str}]  "
            f"tgt_delta=[{td_str}]  "
            f"model_cos_sim=[{mc_str}]  "
            f"gt_cos_sim=[{gc_str}]"
        )
        return  # first batch only


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


@torch.no_grad()
def visualize_rollout(
    model: TokamakFoundationModel,
    ae_models: dict,
    loader: DataLoader,
    epoch: int,
    save_dir: Path,
    preprocess_stats: dict,
    n_rollout_vis: int = 8,
    label: str = "val",
    ae_token_stats: Optional[dict] = None,
    tag: str = "p1",
) -> None:
    """Generate rollout plots in signal space."""
    model.eval()
    plot_dir = save_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    for batch in loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        ctx_signals = {}
        tgt_signals_steps = [{} for _ in range(n_rollout_vis)]
        for name, cfg in DIAGNOSTIC_CONFIGS.items():
            if name not in batch:
                continue
            ctx, tgts = split_window(batch[name], cfg["target_fs"],
                                      n_rollout=n_rollout_vis)
            ctx_signals[name] = ctx
            for k, tgt in enumerate(tgts):
                tgt_signals_steps[k][name] = tgt
        if not ctx_signals:
            return

        ae_ctx = encode_batch(ae_models, ctx_signals, ae_token_stats)
        act_step_pairs = actuator_step_windows(
            batch, ACTUATOR_CONFIGS, preprocess_stats,
            n_rollout=n_rollout_vis)

        # Rollout
        current = {m: t[:1] for m, t in ae_ctx.items()}  # single sample
        act_single = [(
            {n: t[:1] for n, t in ac.items()},
            {n: t[:1] for n, t in af.items()},
        ) for ac, af in act_step_pairs]

        preds = model.rollout(
            current, act_single, n_steps=n_rollout_vis,
            window_ms=WINDOW_S * 1000, dt_ms=DT_S * 1000)

        # Decode predictions and targets to signal space
        diag_names = [n for n in DIAGNOSTIC_CONFIGS if n in ctx_signals]
        n_diag = len(diag_names)
        idx = 0

        fig, axes = plt.subplots(
            n_diag, 1, figsize=(14, 2.5 * n_diag),
            gridspec_kw={"hspace": 0.4})
        if n_diag == 1:
            axes = [axes]

        for row, name in enumerate(diag_names):
            cfg = DIAGNOSTIC_CONFIGS[name]
            fs = cfg["target_fs"]
            n_ctx = round(WINDOW_S * fs)
            ax = axes[row]

            # Ground truth: full signal
            full_sig = batch[name][idx].cpu()
            t_full = np.arange(full_sig.shape[-1]) / fs * 1000
            ax.plot(t_full, full_sig.mean(dim=0).numpy(),
                    color="C0", linewidth=0.8, label="ground truth")

            # Predicted rollout: stitch decoded segments
            for k, pred_tok in enumerate(preds):
                if name not in pred_tok:
                    continue
                out_len = n_ctx
                sig_pred = ae_decode(
                    ae_models[name], pred_tok[name],
                    cfg, out_len,
                    ae_token_stats=ae_token_stats,
                    modality_name=name).cpu()[0]
                t_start = (k + 1) * DT_S * 1000
                t_seg = np.arange(sig_pred.shape[-1]) / fs * 1000 + t_start
                label_k = "predicted" if k == 0 else None
                ax.plot(t_seg, sig_pred.mean(dim=0).numpy(),
                        color="C1", linewidth=0.8, alpha=0.8, label=label_k)

            ax.axvline(WINDOW_S * 1000, color="red", ls="--", lw=0.8)
            ax.set_title(f"{name}", fontsize=9)
            ax.set_xlabel("time [ms]")
            if row == 0:
                ax.legend(fontsize=7)

        fig.suptitle(
            f"Epoch {epoch} ({label}) — Aurora rollout ({n_rollout_vis} steps)",
            fontsize=12, fontweight="bold")
        fig.savefig(
            plot_dir / f"rollout_{label}_{tag}_epoch{epoch:03d}.png",
            dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Plots saved to {plot_dir}")
        return  # first batch only


@torch.no_grad()
def visualize_diagnostics(
    model: TokamakFoundationModel,
    ae_models: dict,
    loader: DataLoader,
    epoch: int,
    save_dir: Path,
    preprocess_stats: dict,
    label: str = "val",
    ae_token_stats: Optional[dict] = None,
    tag: str = "p1",
) -> None:
    """Generate diagnostics grid: raw signal, AE recon, predictions, scatter.

    Per-diagnostic rows with 3 columns:
        (a) Raw signal (channel mean) over full chunk
        (b) AE reconstruction vs original (context window)
        (c) Predicted vs actual target (first rollout step)
    Bottom row:
        Model MSE vs copy-baseline MSE scatter across all val samples.
    """
    model.eval()
    plot_dir = save_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    # Pass 1: collect per-sample MSEs for scatter plot
    all_pred_mse = []
    all_copy_mse = []
    fixed_batch = None

    for batch in loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        ctx_signals = {}
        tgt_signals = {}
        for name, cfg in DIAGNOSTIC_CONFIGS.items():
            if name not in batch:
                continue
            ctx, tgts = split_window(batch[name], cfg["target_fs"],
                                      n_rollout=1)
            ctx_signals[name] = ctx
            tgt_signals[name] = tgts[0]
        if not ctx_signals:
            continue

        ae_ctx = encode_batch(ae_models, ctx_signals, ae_token_stats)
        ae_tgt = encode_batch(ae_models, tgt_signals, ae_token_stats)

        act_step_pairs = actuator_step_windows(
            batch, ACTUATOR_CONFIGS, preprocess_stats, n_rollout=1)
        act_curr, act_fut = act_step_pairs[0]

        # Single-step prediction
        ae_pred = model.forward(
            ae_ctx, act_curr, act_fut, step_index=0,
            offset_ms=WINDOW_S * 1000, dt_ms=DT_S * 1000)

        # Per-sample MSE: model vs copy baseline (in AE token space)
        B = next(iter(ae_ctx.values())).shape[0]
        pred_flat = torch.cat(
            [ae_pred[m].reshape(B, -1) for m in ae_pred if m in ae_tgt],
            dim=1)
        tgt_flat = torch.cat(
            [ae_tgt[m].reshape(B, -1) for m in ae_pred if m in ae_tgt],
            dim=1)
        ctx_flat = torch.cat(
            [ae_ctx[m].reshape(B, -1) for m in ae_pred if m in ae_tgt],
            dim=1)

        pred_mse = ((pred_flat - tgt_flat) ** 2).mean(dim=1)
        copy_mse = ((ctx_flat - tgt_flat) ** 2).mean(dim=1)
        all_pred_mse.append(pred_mse.cpu())
        all_copy_mse.append(copy_mse.cpu())

        if fixed_batch is None:
            fixed_batch = {
                "batch": batch,
                "ctx_signals": ctx_signals,
                "tgt_signals": tgt_signals,
                "ae_ctx": ae_ctx,
                "ae_tgt": ae_tgt,
                "ae_pred": ae_pred,
            }

    all_pred_mse = torch.cat(all_pred_mse).numpy()
    all_copy_mse = torch.cat(all_copy_mse).numpy()

    if fixed_batch is None:
        return

    batch = fixed_batch["batch"]
    ctx_signals = fixed_batch["ctx_signals"]
    tgt_signals = fixed_batch["tgt_signals"]
    ae_pred = fixed_batch["ae_pred"]

    idx = 0
    diag_names = [n for n in DIAGNOSTIC_CONFIGS if n in ctx_signals]
    n_diag = len(diag_names)

    # Build figure: n_diag rows × 3 cols + 1 bottom row for scatter
    n_rows = n_diag + 1
    fig, axes = plt.subplots(
        n_rows, 3, figsize=(16, 3.2 * n_rows),
        gridspec_kw={"hspace": 0.45, "wspace": 0.3})
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, name in enumerate(diag_names):
        cfg = DIAGNOSTIC_CONFIGS[name]
        fs = cfg["target_fs"]
        ctx_sig = ctx_signals[name][idx].cpu()
        n_dt = round(DT_S * fs)

        # (a) Raw signal over full chunk
        ax = axes[row, 0]
        full_sig = batch[name][idx].cpu()
        t_full = np.arange(full_sig.shape[-1]) / fs * 1000
        ax.plot(t_full, full_sig.mean(dim=0).numpy(),
                color="C0", linewidth=0.8)
        ax.axvline(WINDOW_S * 1000, color="red", linewidth=1, ls="--",
                    label="ctx|tgt")
        ax.set_title(f"{name} — raw signal", fontsize=8)
        ax.set_xlabel("time [ms]")
        ax.legend(fontsize=6)

        # (b) AE reconstruction vs original (context)
        ax = axes[row, 1]
        ae = ae_models[name]
        recon = ae(ctx_signals[name][idx:idx+1]).cpu()[0]
        t_ctx = np.arange(ctx_sig.shape[-1]) / fs * 1000
        ae_mse = float(((ctx_sig - recon) ** 2).mean())
        ax.plot(t_ctx, ctx_sig.mean(dim=0).numpy(),
                color="C0", linewidth=1, label="original")
        ax.plot(t_ctx, recon.mean(dim=0).numpy(),
                color="C3", linewidth=1, ls="--", label="AE recon")
        ax.set_title(f"{name} — AE recon (MSE={ae_mse:.4f})", fontsize=8)
        ax.legend(fontsize=6)

        # (c) Predicted vs actual target
        ax = axes[row, 2]
        tgt_sig = tgt_signals[name][idx].cpu()
        t_tgt = np.arange(tgt_sig.shape[-1]) / fs * 1000 + DT_S * 1000

        ax.plot(t_tgt, tgt_sig.mean(dim=0).numpy(),
                color="C0", linewidth=1, label="actual target")
        if name in ae_pred:
            out_len = tgt_sig.shape[-1]
            pred_sig = ae_decode(
                ae_models[name], ae_pred[name][idx:idx+1],
                cfg, out_len,
                ae_token_stats=ae_token_stats,
                modality_name=name).cpu()[0]
            pred_mse_val = float(((pred_sig - tgt_sig) ** 2).mean())
            ax.plot(t_tgt, pred_sig.mean(dim=0).numpy(),
                    color="C1", linewidth=1, ls="--", label="predicted")
            ax.set_title(f"{name} — pred MSE={pred_mse_val:.4f}", fontsize=8)
        else:
            ax.set_title(f"{name} — no prediction", fontsize=8)
        ax.set_xlabel("time [ms]")
        ax.legend(fontsize=6)

    # Bottom row: scatter plot (model MSE vs copy MSE)
    for col in range(2):
        axes[n_diag, col].axis("off")

    ax = axes[n_diag, 2]
    vmax = max(all_pred_mse.max(), all_copy_mse.max()) * 1.1
    ax.scatter(all_copy_mse, all_pred_mse, s=8, alpha=0.4, c="C0")
    ax.plot([0, vmax], [0, vmax], "k--", linewidth=0.8, label="model = copy")
    ax.set_xlabel("Copy-baseline MSE")
    ax.set_ylabel("Model MSE")
    ax.set_title("Model vs copy baseline (AE token space)")
    ax.legend(fontsize=7)
    ax.set_xlim(0, vmax)
    ax.set_ylim(0, vmax)
    ax.set_aspect("equal")

    fig.suptitle(f"Epoch {epoch} ({label})", fontsize=14, fontweight="bold")
    fig.savefig(
        plot_dir / f"diagnostics_{label}_{tag}_epoch{epoch:03d}.png",
        dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Diagnostics saved to {plot_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Train Aurora-inspired Tokamak Foundation Model")
    parser.add_argument("--data_dir", default="/scratch/gpfs/EKOLEMEN/foundation_model/")
    parser.add_argument("--stats_path",
                        default="/projects/EKOLEMEN/foundation_model/preprocessing_stats.pt")
    parser.add_argument("--ae_checkpoint_dir",
                        default="/projects/EKOLEMEN/foundation_model/")
    parser.add_argument("--ae_token_stats_path", default=None,
                        help="Path to ae_token_stats.pt for per-modality "
                             "token normalization.")
    parser.add_argument("--checkpoint_dir", default="runs/aurora")

    # Model
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_latent", type=int, default=128)
    parser.add_argument("--encoder_cross_layers", type=int, default=2)
    parser.add_argument("--encoder_self_layers", type=int, default=2)
    parser.add_argument("--backbone_blocks", type=int, default=8)
    parser.add_argument("--decoder_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.0)

    # Data
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--warmup_s", type=float, default=1.0)
    parser.add_argument("--step_size_s", type=float, default=None)

    # Phase 1
    parser.add_argument("--pretrain_epochs", type=int, default=100)
    parser.add_argument("--pretrain_lr", type=float, default=1e-4)
    parser.add_argument("--recon_weight", type=float, default=0.0)

    # Phase 2
    parser.add_argument("--finetune_epochs", type=int, default=50)
    parser.add_argument("--finetune_lr", type=float, default=3e-5)
    parser.add_argument("--max_rollout", type=int, default=8)
    parser.add_argument("--rollout_ramp_epochs", type=int, default=30)

    # Common
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--steps_per_epoch", type=int, default=0)
    parser.add_argument("--plot_every", type=int, default=5)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--no_delta_loss", action="store_true", default=False,
                        help="Disable the L1-magnitude delta loss; use MAE only")
    parser.add_argument("--delta_weight", type=float, default=1.0,
                        help="Multiplier on the (cos + mag) delta-loss "
                             "contribution. Only active when --no_delta_loss "
                             "is not set.")
    parser.add_argument("--step_diversity_weight", type=float, default=0.0,
                        help="Weight of the GT-targeted step-diversity "
                             "regularizer: MSE between cos(pred_k, "
                             "pred_{k-1}) and cos(tgt_k, tgt_{k-1}). "
                             "0 disables.")

    args = parser.parse_args()

    N_ROLLOUT = args.max_rollout
    CHUNK_S = WINDOW_S + N_ROLLOUT * DT_S
    if args.step_size_s is None:
        args.step_size_s = CHUNK_S

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- Load AEs ---
    ae_models = {}
    for name, cfg in DIAGNOSTIC_CONFIGS.items():
        ae_dir = Path(args.ae_checkpoint_dir)
        if "ae_checkpoint_path" in cfg:
            ckpt_path = Path(cfg["ae_checkpoint_path"])
        else:
            ckpt_path = ae_dir / f"{name}_{cfg['model_type']}" / "checkpoint_best.pth"
        if not ckpt_path.exists():
            logger.warning(f"AE not found for '{name}': {ckpt_path} — skipping")
            continue
        ae_models[name] = load_ae(name, cfg, ckpt_path)

    if not ae_models:
        raise RuntimeError("No AE checkpoints found.")

    active_diagnostics = {
        k: v for k, v in DIAGNOSTIC_CONFIGS.items() if k in ae_models}

    # Per-modality AE token normalization stats
    ae_token_stats = None
    if args.ae_token_stats_path is not None:
        ae_token_stats = torch.load(args.ae_token_stats_path, weights_only=False)
        logger.info(f"Loaded AE token stats for {list(ae_token_stats.keys())}")

    # --- Datasets ---
    stats = torch.load(args.stats_path, weights_only=False)
    all_signals = list(active_diagnostics.keys()) + list(ACTUATOR_CONFIGS.keys())

    data_dir = Path(args.data_dir)
    all_files = sorted(data_dir.glob("*_processed.h5"))
    random.seed(42)
    random.shuffle(all_files)
    if args.max_files is not None:
        all_files = all_files[:args.max_files]
    n_val = max(1, int(0.1 * len(all_files)))
    train_files = all_files[n_val:]
    val_files = all_files[:n_val]
    logger.info(f"Files — train: {len(train_files)}  val: {len(val_files)}")

    shared_kwargs = dict(
        preprocessing_stats=stats,
        input_signals=all_signals,
        chunk_duration_s=CHUNK_S,
        step_size_s=args.step_size_s,
        warmup_s=args.warmup_s,
        prediction_mode=False,
    )
    train_ds = TokamakMultiFileDataset(
        train_files, lengths_cache_path="lengths_aurora_train.pt",
        **shared_kwargs)
    val_ds = TokamakMultiFileDataset(
        val_files, lengths_cache_path="lengths_aurora_val.pt",
        **shared_kwargs)
    logger.info(f"Chunks — train: {len(train_ds)}  val: {len(val_ds)}")

    train_loader = make_dataloader(
        train_ds, batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=True,
        pin_memory=True, prefetch_factor=args.prefetch_factor)
    val_loader = make_dataloader(
        val_ds, batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=False,
        pin_memory=True, prefetch_factor=args.prefetch_factor)

    # --- Build model ---
    modality_configs = {
        name: {"d_lat": cfg["d_lat"], "n_tokens": cfg["n_tokens"]}
        for name, cfg in active_diagnostics.items()
    }
    model = TokamakFoundationModel(
        modality_configs=modality_configs,
        d_model=args.d_model,
        n_latent=args.n_latent,
        n_heads=args.n_heads,
        encoder_cross_layers=args.encoder_cross_layers,
        encoder_self_layers=args.encoder_self_layers,
        backbone_blocks=args.backbone_blocks,
        decoder_layers=args.decoder_layers,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        actuator_configs=ACTUATOR_CONFIGS,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Aurora model: {n_params:,} trainable parameters")
    logger.info(f"Config: d={args.d_model}, latent={args.n_latent}, "
                f"backbone={args.backbone_blocks} blocks, "
                f"encoder={args.encoder_cross_layers}x+{args.encoder_self_layers}s, "
                f"decoder={args.decoder_layers}")

    checkpoint_path = ckpt_dir / "checkpoint.pth"
    best_path = ckpt_dir / "best.pth"

    # ─────────────────────────────────────────────────────────────
    # Phase 1: Single-step pretraining
    # ─────────────────────────────────────────────────────────────
    logger.info(f"═══ Phase 1: Single-step pretraining ({args.pretrain_epochs} epochs) ═══")

    optimizer = optim.AdamW(
        model.parameters(), lr=args.pretrain_lr,
        weight_decay=args.weight_decay)

    encoder_optimizer: Optional[optim.Optimizer] = None
    if args.recon_weight > 0.0:
        # Unfreeze AE encoders; keep decoders frozen so the recon loss
        # can only push the encoder back toward the decoder's manifold.
        encoder_params = []
        for ae in ae_models.values():
            for p in ae.encoder.parameters():
                p.requires_grad_(True)
            encoder_params += list(ae.encoder.parameters())
            ae.encoder.train()
        encoder_optimizer = optim.AdamW(
            encoder_params,
            lr=0.1 * args.pretrain_lr,
            weight_decay=args.weight_decay,
        )
        logger.info(
            f"AE encoders unfrozen ({len(encoder_params)} param tensors); "
            f"encoder_lr={0.1 * args.pretrain_lr:.2e}, "
            f"recon_weight={args.recon_weight}"
        )

    if args.warmup_epochs > 0:
        warmup = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, end_factor=1.0,
            total_iters=args.warmup_epochs)
        cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.pretrain_epochs - args.warmup_epochs),
            eta_min=args.min_lr)
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine],
            milestones=[args.warmup_epochs])
    else:
        scheduler = None

    best_val = float("inf")
    start_epoch = 0

    if args.resume and checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val = ckpt.get("best_val", float("inf"))
        phase = ckpt.get("phase", 1)
        if phase >= 2:
            logger.info("Checkpoint is from Phase 2 — skipping Phase 1")
            start_epoch = 0  # will be used as Phase 2 epoch
        else:
            logger.info(f"Resumed Phase 1 from epoch {start_epoch}")

    for epoch in range(start_epoch, args.pretrain_epochs):
        train_mae, train_mag, train_recon = run_phase1_epoch(
            model, ae_models, train_loader, optimizer, is_train=True,
            preprocess_stats=stats, recon_weight=args.recon_weight,
            max_steps=args.steps_per_epoch, ae_token_stats=ae_token_stats,
            use_delta_loss=not args.no_delta_loss,
            delta_weight=args.delta_weight,
            encoder_optimizer=encoder_optimizer)

        with torch.no_grad():
            val_mae, val_mag, val_recon = run_phase1_epoch(
                model, ae_models, val_loader, None, is_train=False,
                preprocess_stats=stats, recon_weight=args.recon_weight,
                max_steps=args.steps_per_epoch, ae_token_stats=ae_token_stats,
                use_delta_loss=not args.no_delta_loss,
                delta_weight=args.delta_weight)

        if scheduler is not None:
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        recon_line = (
            f"  train_recon={train_recon:.6f}  val_recon={val_recon:.6f}"
            if args.recon_weight > 0.0 else ""
        )
        logger.info(
            f"P1 Epoch {epoch+1:3d}/{args.pretrain_epochs}  "
            f"train_mae={train_mae:.6f}  val_mae={val_mae:.6f}  "
            f"train_mag={train_mag:.6f}  val_mag={val_mag:.6f}{recon_line}  "
            f"lr={lr:.2e}")

        # Diagnostics
        log_diagnostics(model, ae_models, val_loader, stats, n_rollout=1,
                        ae_token_stats=ae_token_stats)

        # Save
        torch.save({
            "epoch": epoch,
            "phase": 1,
            "model_state_dict": model.state_dict(),
            "best_val": best_val,
            "args": vars(args),
        }, checkpoint_path)

        if val_mae < best_val:
            best_val = val_mae
            torch.save(model.state_dict(), best_path)
            logger.info(f"  → New best val MAE: {best_val:.6f}")

        if args.plot_every > 0 and (
            (epoch + 1) % args.plot_every == 0
            or epoch == args.pretrain_epochs - 1
        ):
            visualize_rollout(
                model, ae_models, val_loader, epoch + 1, ckpt_dir,
                stats, n_rollout_vis=N_ROLLOUT, label="val",
                ae_token_stats=ae_token_stats)
            visualize_rollout(
                model, ae_models, train_loader, epoch + 1, ckpt_dir,
                stats, n_rollout_vis=N_ROLLOUT, label="train",
                ae_token_stats=ae_token_stats)
            visualize_diagnostics(
                model, ae_models, val_loader, epoch + 1, ckpt_dir,
                stats, label="val", ae_token_stats=ae_token_stats)
            visualize_diagnostics(
                model, ae_models, train_loader, epoch + 1, ckpt_dir,
                stats, label="train", ae_token_stats=ae_token_stats)

    # ─────────────────────────────────────────────────────────────
    # Phase 2: Multi-step fine-tuning
    # ─────────────────────────────────────────────────────────────
    logger.info(f"═══ Phase 2: Multi-step fine-tuning ({args.finetune_epochs} epochs) ═══")

    optimizer = optim.AdamW(
        model.parameters(), lr=args.finetune_lr,
        weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.finetune_epochs, eta_min=args.min_lr)

    best_val_p2 = float("inf")

    for epoch in range(args.finetune_epochs):
        # Rollout curriculum
        K = min(N_ROLLOUT,
                max(1, 1 + epoch * N_ROLLOUT // args.rollout_ramp_epochs))

        train_total, train_last = run_phase2_epoch(
            model, ae_models, train_loader, optimizer, is_train=True,
            preprocess_stats=stats, n_rollout=K,
            max_steps=args.steps_per_epoch, ae_token_stats=ae_token_stats,
            use_delta_loss=not args.no_delta_loss,
            delta_weight=args.delta_weight,
            step_diversity_weight=args.step_diversity_weight)

        with torch.no_grad():
            val_total, val_last = run_phase2_epoch(
                model, ae_models, val_loader, None, is_train=False,
                preprocess_stats=stats, n_rollout=K,
                max_steps=args.steps_per_epoch, ae_token_stats=ae_token_stats,
                use_delta_loss=not args.no_delta_loss,
                delta_weight=args.delta_weight,
                step_diversity_weight=args.step_diversity_weight)

        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"P2 Epoch {epoch+1:3d}/{args.finetune_epochs}  "
            f"K={K}  train={train_total:.6f} (last={train_last:.6f})  "
            f"val={val_total:.6f} (last={val_last:.6f})  "
            f"lr={lr:.2e}")

        # Diagnostics
        log_diagnostics(model, ae_models, val_loader, stats, n_rollout=K,
                        ae_token_stats=ae_token_stats)

        # Save
        torch.save({
            "epoch": epoch,
            "phase": 2,
            "model_state_dict": model.state_dict(),
            "best_val": best_val_p2,
            "args": vars(args),
        }, checkpoint_path)

        if val_total < best_val_p2:
            best_val_p2 = val_total
            torch.save(model.state_dict(), best_path)
            logger.info(f"  → New best val loss: {best_val_p2:.6f}")

        if args.plot_every > 0 and (
            (epoch + 1) % args.plot_every == 0
            or epoch == args.finetune_epochs - 1
        ):
            ep = epoch + 1
            visualize_rollout(
                model, ae_models, val_loader, ep, ckpt_dir,
                stats, n_rollout_vis=N_ROLLOUT, label="val",
                ae_token_stats=ae_token_stats, tag="p2")
            visualize_rollout(
                model, ae_models, train_loader, ep, ckpt_dir,
                stats, n_rollout_vis=N_ROLLOUT, label="train",
                ae_token_stats=ae_token_stats, tag="p2")
            visualize_diagnostics(
                model, ae_models, val_loader, ep, ckpt_dir,
                stats, label="val", ae_token_stats=ae_token_stats,
                tag="p2")
            visualize_diagnostics(
                model, ae_models, train_loader, ep, ckpt_dir,
                stats, label="train", ae_token_stats=ae_token_stats,
                tag="p2")

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
