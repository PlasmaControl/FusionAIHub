"""Standalone tangtv autoencoder validation.

Trains :class:`VideoTokenizer` + :class:`VideoOutputHead` end-to-end on
masked MAE reconstruction loss for a few thousand steps, before Step 5
integration into the full E2E foundation model. Validates that the
tube-patch tokens carry enough capacity to reconstruct tangtv plasma
structure.

The Perceiver-pool design that this trainer originally targeted was
abandoned after three iterations plateaued at ratio ~0.62 on plasma
channels with featureless reconstructions. The tube-patch design
(VideoMAE-style) replaces the global pool with local patches: each
token represents one ``(T_p, H_p, W_p)`` region, the decoder is a
single ``ConvTranspose3d`` that exactly inverts the patch embedding,
and per-patch reconstruction means spatial detail is preserved by
construction.

Reports against the per-(B, C) spatial+temporal mean baseline (in
normalized space the baseline is "predict zero"). With per-patch
tokens the AE should beat the baseline meaningfully and produce
visible plasma structure in the recon plots — that is the criterion
to pass before Step 5 integration.

Usage::

    pixi run python scripts/training/train_video_ae.py \\
        --data_dir /scratch/gpfs/EKOLEMEN/foundation_model \\
        --checkpoint_dir runs/video_ae \\
        --max_steps 5000 --batch_size 256 --num_workers 12
"""

from __future__ import annotations

import argparse
import logging
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tokamak_foundation_model.data.data_loader import collate_fn
from tokamak_foundation_model.data.multi_file_dataset import (
    TokamakMultiFileDataset,
    TwoLevelSampler,
)
from tokamak_foundation_model.e2e.output_heads import VideoOutputHead
from tokamak_foundation_model.e2e.tokenizers.video import VideoTokenizer

logger = logging.getLogger("video_ae")


# ── Per-batch standardization ────────────────────────────────────────────


def standardize_per_bc(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standardize input per (B, C) by mean/std over (T, H, W).

    Without preprocessing stats (deferred per the Step 1 decision), raw
    tangtv pixel values across active channels span 0-200+ while
    near-constant calibration channels sit at ~50. Different batches
    therefore have order-of-magnitude different loss scales, which
    destabilises training. Per-batch z-score on each (sample, channel)
    puts everything on a comparable scale; the AE then trains in
    normalized space. Inactive (NaN-filled-to-zero) channels have
    mu=0, sd=0 -> clamp(min=1) -> normalized = 0 (mask gates them out
    of loss anyway). Visual inspection plots denormalize via the saved
    mu, sd so the user sees raw pixel comparisons.

    Returns
    -------
    x_norm : Tensor
        Same shape as ``x``, standardized.
    mu : Tensor
        Shape ``(B, C, 1, 1, 1)`` — per-(B, C) means.
    sd : Tensor
        Shape ``(B, C, 1, 1, 1)`` — per-(B, C) std clamped at 1.0.
    """
    mu = x.mean(dim=(2, 3, 4), keepdim=True)
    sd = x.std(dim=(2, 3, 4), keepdim=True).clamp(min=1.0)
    return (x - mu) / sd, mu, sd


# ── Loss / metric ────────────────────────────────────────────────────────


def masked_mae(
    recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """MAE averaged over True positions of ``mask``.

    ``recon`` and ``target`` have shape ``(B, T, C, H, W)``. ``mask``
    is broadcastable to that shape (typically
    ``(B, 1, C, 1, 1)`` for per-(B, C) gating). Inactive positions
    contribute neither numerator nor denominator.
    """
    diff = (recon - target).abs() * mask
    denom = mask.expand_as(diff).sum().clamp(min=1.0)
    return diff.sum() / denom


def per_channel_mae(
    recon: torch.Tensor, target: torch.Tensor, gate_bc: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel MAE accumulators.

    Returns ``(diff_sum_per_c, count_per_c)`` of shape ``(C,)``.
    ``gate_bc`` is ``(B, C)`` bool/float — True means "include this
    (sample, channel) in the average".
    """
    # (B, T, C, H, W) -> (B, C) average over (T, H, W) per (B, C).
    per_bc = (recon - target).abs().mean(dim=(1, 3, 4))   # (B, C)
    g = gate_bc.float()
    diff_sum_per_c = (per_bc * g).sum(dim=0)              # (C,)
    count_per_c = g.sum(dim=0)                            # (C,)
    return diff_sum_per_c, count_per_c


# ── Validation pass ──────────────────────────────────────────────────────


def run_validation(
    tokenizer: VideoTokenizer,
    head: VideoOutputHead,
    val_loader: DataLoader,
    device: torch.device,
    out_dir: Path,
    step: int,
    max_plot_panels: int = 5,
    max_batches: int = 20,
) -> dict:
    """Compute validation metrics and save reconstruction plots."""
    tokenizer.eval()
    head.eval()

    n_channels = tokenizer.n_channels
    diff_ae_per_c = torch.zeros(n_channels, device=device)
    diff_mean_per_c = torch.zeros(n_channels, device=device)
    count_per_c = torch.zeros(n_channels, device=device)

    plot_panels = []   # list of (in_frame, recon_frame, c, sample_index)

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= max_batches:
                break
            inputs = batch["inputs"]
            x = inputs["tangtv"].to(device, non_blocking=True)            # (B, C, T, H, W)
            channel_mask = inputs["tangtv_channel_mask"].to(device)        # (B, C)
            valid = inputs["tangtv_valid"].to(device)                      # (B,)
            if valid.sum() == 0:
                continue

            x_norm, mu, sd = standardize_per_bc(x)
            target = x_norm.permute(0, 2, 1, 3, 4)                         # (B, T, C, H, W)
            tokens = tokenizer(x_norm, mask=valid.bool())
            recon = head(tokens)                                           # (B, T, C, H, W) normalized
            zero_pred = torch.zeros_like(target)                            # mean baseline in norm space

            gate_bc = valid.bool()[:, None] & channel_mask.bool()          # (B, C)
            d_ae, count = per_channel_mae(recon, target, gate_bc)
            d_mean, _ = per_channel_mae(zero_pred, target, gate_bc)
            diff_ae_per_c += d_ae
            diff_mean_per_c += d_mean
            count_per_c += count

            # Stash a few mid-frame side-by-side panels for visual check.
            # Denormalize recon back to raw pixels so the panel compares
            # apples to apples with the raw input frame.
            if len(plot_panels) < max_plot_panels:
                # mu/sd shape (B, C, 1, 1, 1) -> permute to (B, 1, C, 1, 1)
                # to match recon (B, T, C, H, W).
                mu_t = mu.permute(0, 2, 1, 3, 4)
                sd_t = sd.permute(0, 2, 1, 3, 4)
                recon_raw = recon * sd_t + mu_t
                B = x.shape[0]
                t_mid = x.shape[2] // 2
                for b in range(B):
                    if not valid[b].item():
                        continue
                    for c in range(n_channels):
                        if not channel_mask[b, c].item():
                            continue
                        plot_panels.append(
                            (
                                x[b, c, t_mid].cpu().numpy(),
                                recon_raw[b, t_mid, c].cpu().numpy(),
                                int(c),
                                int(b),
                            )
                        )
                        break
                    if len(plot_panels) >= max_plot_panels:
                        break

    mae_ae = (diff_ae_per_c / count_per_c.clamp(min=1)).cpu()
    mae_mean = (diff_mean_per_c / count_per_c.clamp(min=1)).cpu()
    counts = count_per_c.cpu().long()

    logger.info(f"--- Validation @ step {step} ---")
    n_active_total = int(counts.sum().item())
    if n_active_total == 0:
        logger.info("  no active (camera, channel) entries seen; skipping")
    else:
        for c in range(n_channels):
            n = int(counts[c].item())
            if n == 0:
                logger.info(f"  ch{c}: n=0  (no active samples for this channel)")
                continue
            ratio = (
                mae_ae[c].item()
                / max(mae_mean[c].item(), 1e-6)
            )
            logger.info(
                f"  ch{c}: n={n:5d}  AE_MAE={mae_ae[c].item():8.3f}  "
                f"mean_MAE={mae_mean[c].item():8.3f}  ratio={ratio:.3f}"
            )

    if plot_panels:
        n_panels = len(plot_panels)
        fig, axes = plt.subplots(
            n_panels, 2, figsize=(12, 2.6 * n_panels), squeeze=False
        )
        for i, (in_frame, re_frame, c, b) in enumerate(plot_panels):
            vmin = float(min(in_frame.min(), re_frame.min()))
            vmax = float(max(in_frame.max(), re_frame.max()))
            axes[i, 0].imshow(
                in_frame, cmap="inferno", vmin=vmin, vmax=vmax, aspect="auto"
            )
            axes[i, 0].set_title(f"input  sample={b} ch={c}")
            axes[i, 1].imshow(
                re_frame, cmap="inferno", vmin=vmin, vmax=vmax, aspect="auto"
            )
            axes[i, 1].set_title(f"recon  sample={b} ch={c}")
            for ax in axes[i]:
                ax.set_xticks([])
                ax.set_yticks([])
        fig.tight_layout()
        out_path = out_dir / f"recon_step{step:06d}.png"
        fig.savefig(out_path, dpi=100)
        plt.close(fig)
        logger.info(f"  saved {out_path}")

    tokenizer.train()
    head.train()

    return {
        "step": step,
        "mae_ae_per_channel": mae_ae.tolist(),
        "mae_mean_per_channel": mae_mean.tolist(),
        "counts_per_channel": counts.tolist(),
    }


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("/scratch/gpfs/EKOLEMEN/foundation_model"),
    )
    parser.add_argument(
        "--checkpoint_dir", type=Path, default=Path("runs/video_ae"),
    )
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--val_every", type=int, default=500)
    parser.add_argument(
        "--patch_size",
        type=int,
        nargs=3,
        default=[3, 12, 12],
        metavar=("T_P", "H_P", "W_P"),
        help=(
            "Tube patch size (T, H, W). Spatial dims of the input "
            "(120, 360) and n_frames (3) must be divisible by it."
        ),
    )
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--val_fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Files ────────────────────────────────────────────────────────────
    # Random val split (NOT first n alphabetical) so the val set sees the
    # same channel-availability distribution as training. Earlier first-n
    # split happened to exclude shots with ch4/ch6 plasma channels active.
    files = sorted(args.data_dir.glob("*_processed.h5"))
    if not files:
        raise SystemExit(f"No *_processed.h5 in {args.data_dir}")
    if args.max_files is not None:
        files = files[: args.max_files]
    file_rng = random.Random(args.seed)
    file_rng.shuffle(files)
    n_val = max(1, int(round(len(files) * args.val_fraction)))
    val_files = files[:n_val]
    train_files = files[n_val:]
    logger.info(f"{len(train_files)} train files, {len(val_files)} val files")

    # ── Datasets ─────────────────────────────────────────────────────────
    ds_kwargs = dict(
        chunk_duration_s=0.05,
        prediction_mode=True,
        prediction_horizon_s=0.05,
        input_signals=["tangtv"],
        target_signals=["tangtv"],
        max_open_files=200,
        warmup_s=1.0,
        step_size_s=0.05,
    )
    train_ds = TokamakMultiFileDataset(
        hdf5_paths=train_files,
        lengths_cache_path=args.checkpoint_dir / "lengths_train.pt",
        **ds_kwargs,
    )
    val_ds = TokamakMultiFileDataset(
        hdf5_paths=val_files,
        lengths_cache_path=args.checkpoint_dir / "lengths_val.pt",
        **ds_kwargs,
    )
    logger.info(f"Chunks — train: {len(train_ds)}  val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=TwoLevelSampler(train_ds, shuffle=True),
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=False,
        persistent_workers=args.num_workers > 0,
    )

    # ── Model ────────────────────────────────────────────────────────────
    patch_size = tuple(args.patch_size)
    tokenizer = VideoTokenizer(
        n_channels=2,
        n_frames=3,
        patch_size=patch_size,
        d_model=256,
        spatial_size=(120, 360),
    ).to(device)
    head = VideoOutputHead(
        n_channels=2,
        n_frames=3,
        patch_size=patch_size,
        d_model=256,
        spatial_size=(120, 360),
    ).to(device)
    n_tok = sum(p.numel() for p in tokenizer.parameters())
    n_head = sum(p.numel() for p in head.parameters())
    logger.info(
        f"Model params: tokenizer={n_tok / 1e6:.2f}M  "
        f"head={n_head / 1e6:.2f}M  total={(n_tok + n_head) / 1e6:.2f}M"
    )

    optimizer = torch.optim.AdamW(
        list(tokenizer.parameters()) + list(head.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ── Train ────────────────────────────────────────────────────────────
    logger.info(
        f"Starting training: max_steps={args.max_steps} batch={args.batch_size} "
        f"lr={args.lr} patch_size={tuple(args.patch_size)} "
        f"n_tokens={tokenizer.n_tokens}"
    )
    train_iter = iter(train_loader)
    t0 = time.time()
    history: list[dict] = []
    val_records: list[dict] = []
    skipped_no_camera = 0

    step = 0
    while step < args.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        inputs = batch["inputs"]
        x = inputs["tangtv"].to(device, non_blocking=True)
        channel_mask = inputs["tangtv_channel_mask"].to(device, non_blocking=True)
        valid = inputs["tangtv_valid"].to(device, non_blocking=True)
        if valid.sum() == 0:
            skipped_no_camera += 1
            continue

        # Per-(B, C) z-score; train in normalized space so loss is on a
        # consistent scale across batches regardless of which channels are
        # active. AE has to predict the normalized data; plots denormalize.
        x_norm, _, _ = standardize_per_bc(x)
        target = x_norm.permute(0, 2, 1, 3, 4)
        tokens = tokenizer(x_norm, mask=valid.bool())
        recon = head(tokens)

        # Per-element gate: per-batch validity * per-channel availability.
        gate = (
            valid.bool()[:, None, None, None, None].float()
            * channel_mask[:, None, :, None, None].float()
        )
        loss = masked_mae(recon, target, gate)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(tokenizer.parameters()) + list(head.parameters()),
            args.grad_clip,
        )
        optimizer.step()

        if step % args.log_every == 0:
            with torch.no_grad():
                # Mean baseline in normalized space is just zero (every
                # (B, C) slice has been centered to zero mean by the
                # z-score). MAE(0, x_norm) ~ E|x_norm| ~ 0.8 for roughly
                # Gaussian content; AE must beat ~0.8 to be useful.
                mae_mean = masked_mae(
                    torch.zeros_like(target), target, gate
                ).item()
            elapsed = max(time.time() - t0, 1e-6)
            sps = (step + 1) / elapsed
            logger.info(
                f"step {step:6d}/{args.max_steps}  "
                f"loss={loss.item():.4f}  "
                f"mean_baseline={mae_mean:.4f}  "
                f"delta={loss.item() - mae_mean:+.4f}  "
                f"{sps:5.2f} steps/s  "
                f"skipped_no_cam={skipped_no_camera}"
            )
            history.append(
                {
                    "step": step,
                    "loss": loss.item(),
                    "mean_baseline": mae_mean,
                }
            )

        if step > 0 and step % args.val_every == 0:
            val_records.append(
                run_validation(
                    tokenizer,
                    head,
                    val_loader,
                    device,
                    args.checkpoint_dir,
                    step,
                )
            )

        step += 1

    # Final validation + save
    val_records.append(
        run_validation(
            tokenizer, head, val_loader, device, args.checkpoint_dir, step
        )
    )

    final_path = args.checkpoint_dir / "video_ae_final.pt"
    torch.save(
        {
            "tokenizer_state_dict": tokenizer.state_dict(),
            "head_state_dict": head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "history": history,
            "val_records": val_records,
            "skipped_no_camera": skipped_no_camera,
        },
        final_path,
    )
    logger.info(f"Saved {final_path}")

    # Loss-curve plot for at-a-glance reading.
    if history:
        steps = [h["step"] for h in history]
        losses = [h["loss"] for h in history]
        means = [h["mean_baseline"] for h in history]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(steps, losses, label="AE recon MAE", color="tab:blue")
        ax.plot(steps, means, label="mean baseline MAE", color="tab:orange",
                linestyle="--")
        ax.set_xlabel("step")
        ax.set_ylabel("masked MAE")
        ax.set_title("Standalone video AE training")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        loss_plot = args.checkpoint_dir / "loss_curve.png"
        fig.savefig(loss_plot, dpi=100)
        plt.close(fig)
        logger.info(f"Saved {loss_plot}")


if __name__ == "__main__":
    main()
