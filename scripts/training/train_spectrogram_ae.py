"""Standalone spectrogram autoencoder validation (Phase B Step 6).

Trains :class:`SpectrogramTokenizer` + :class:`SpectrogramOutputHead`
end-to-end on masked MAE reconstruction loss for a few thousand steps,
before Step 5 integration into the full E2E foundation model. Validates
that the per-patch tokens carry enough capacity to reconstruct the
spectrogram structure of the chosen modality.

The Phase C tube-patch design proved that bounded local patches preserve
fine structure where global pooling does not. The spectrogram tokenizer
mirrors this: each token is one ``(patch_f, patch_t)`` 2D patch, the
decoder is a single ``ConvTranspose2d`` that exactly inverts the
embedding, and per-patch reconstruction makes spatial detail recoverable
by construction.

Per-modality config:

* ``ece`` — 40 ch, patch (F=32, T=8), 192 tokens, 40× compression
* ``co2`` — 4 ch,  patch (F=64, T=8),  96 tokens,  8× compression
* ``bes`` — 16 ch, patch (F=32, T=8), 192 tokens, 16× compression

The shot-level presence rate differs widely (ECE ~94%, CO2 ~44%, BES
~36% from Step 0), so each modality is trained as its own job with the
``--modality`` flag.

Usage::

    pixi run python scripts/training/train_spectrogram_ae.py \\
        --modality ece \\
        --data_dir /scratch/gpfs/EKOLEMEN/foundation_model \\
        --checkpoint_dir runs/spectrogram_ae_ece \\
        --max_steps 5000 --batch_size 64 --num_workers 8
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
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tokamak_foundation_model.data.data_loader import collate_fn
from tokamak_foundation_model.data.multi_file_dataset import (
    TokamakMultiFileDataset,
    TwoLevelSampler,
)
from tokamak_foundation_model.e2e.output_heads import SpectrogramOutputHead
from tokamak_foundation_model.e2e.tokenizers.spectrogram import (
    SpectrogramTokenizer,
)

logger = logging.getLogger("spectrogram_ae")


# ── Per-modality config ──────────────────────────────────────────────────


# (n_channels, patch_f, patch_t)
MODALITY_CONFIG: dict[str, tuple[int, int, int]] = {
    "ece": (40, 32, 8),
    "co2": (4, 64, 8),
    "bes": (16, 32, 8),
}

FREQ_BINS = 512
TIME_FRAMES = 98
D_MODEL = 256
TARGET_FS = 500_000   # ECE/CO2/BES sampling rate
N_FFT = 1024


# ── Loss / metric ────────────────────────────────────────────────────────


def per_bc_mean(x: torch.Tensor) -> torch.Tensor:
    """Per-(B, C) mean over (F, T), kept dimsâ€‘compatible with ``x``.

    Used as the trivial reconstruction baseline ("predict the constant
    per-window per-channel mean"). With per-batch z-score removed,
    this is the right competitor for the AE — predict-zero would be
    artificially weak because the data-loader's ``log_standardize``
    already centres each channel near 0 globally but per-window means
    drift around 0 with non-trivial spread.
    """
    return x.mean(dim=(2, 3), keepdim=True).expand_as(x)


# ── Loss / metric ────────────────────────────────────────────────────────


def masked_mae(
    recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """MAE averaged over True positions of ``mask``.

    ``recon`` and ``target`` shape ``(B, C, F, T)``. ``mask`` is
    broadcastable to that shape (typically ``(B, 1, 1, 1)`` for
    per-sample gating).
    """
    diff = (recon - target).abs() * mask
    denom = mask.expand_as(diff).sum().clamp(min=1.0)
    return diff.sum() / denom


def per_channel_mae(
    recon: torch.Tensor, target: torch.Tensor, gate_b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel MAE accumulators.

    Returns ``(diff_sum_per_c, count_per_c)`` of shape ``(C,)``.
    ``gate_b`` is ``(B,)`` bool — True means "include this sample".
    """
    # (B, C, F, T) -> (B, C) average over (F, T) per (B, C).
    per_bc = (recon - target).abs().mean(dim=(2, 3))     # (B, C)
    g = gate_b.float().unsqueeze(1)                      # (B, 1)
    diff_sum_per_c = (per_bc * g).sum(dim=0)             # (C,)
    count_per_c = g.sum(dim=0).expand_as(diff_sum_per_c)  # (C,) — same per-sample count
    return diff_sum_per_c, count_per_c


# ── Validation pass ──────────────────────────────────────────────────────


def freq_axis_khz() -> np.ndarray:
    return (np.arange(1, FREQ_BINS + 1) * (TARGET_FS / N_FFT)) / 1e3


def run_validation(
    tokenizer: SpectrogramTokenizer,
    head: SpectrogramOutputHead,
    val_loader: DataLoader,
    device: torch.device,
    out_dir: Path,
    step: int,
    modality: str,
    trunc_t: int,
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

    plot_panels: list[tuple[np.ndarray, np.ndarray, int, int]] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= max_batches:
                break
            inputs = batch["inputs"]
            x = inputs[modality].to(device, non_blocking=True)            # (B, C, F, T)
            valid = inputs[f"{modality}_valid"].to(device)                # (B,) int
            gate_b = valid > 0
            if gate_b.sum() == 0:
                continue

            target = x[..., :trunc_t]                                     # (B, C, F, T_trunc) data-loader-normalized
            tokens = tokenizer(x)                                          # (B, n_tokens, d)
            recon = head(tokens)                                           # (B, C, F, T_trunc)
            mean_pred = per_bc_mean(target)                                # per-(B, C) constant baseline

            d_ae, count = per_channel_mae(recon, target, gate_b)
            d_mean, _ = per_channel_mae(mean_pred, target, gate_b)
            diff_ae_per_c += d_ae
            diff_mean_per_c += d_mean
            count_per_c += count

            # Stash sample panels: input vs recon in the data-loader-
            # normalized space the model is trained against. One panel
            # per active channel of one valid sample, capped at
            # max_plot_panels.
            if len(plot_panels) < max_plot_panels:
                B = x.shape[0]
                for b in range(B):
                    if not gate_b[b].item():
                        continue
                    for c in range(n_channels):
                        plot_panels.append(
                            (
                                target[b, c].cpu().numpy(),
                                recon[b, c].cpu().numpy(),
                                int(c),
                                int(b),
                            )
                        )
                        if len(plot_panels) >= max_plot_panels:
                            break
                    if len(plot_panels) >= max_plot_panels:
                        break

    mae_ae = (diff_ae_per_c / count_per_c.clamp(min=1)).cpu()
    mae_mean = (diff_mean_per_c / count_per_c.clamp(min=1)).cpu()
    counts = count_per_c.cpu().long()

    logger.info(f"--- Validation @ step {step} ({modality}) ---")
    n_active = int(counts.max().item()) if counts.numel() else 0
    if n_active == 0:
        logger.info("  no active samples in validation; skipping per-ch report")
    else:
        for c in range(n_channels):
            n = int(counts[c].item())
            ratio = mae_ae[c].item() / max(mae_mean[c].item(), 1e-6)
            logger.info(
                f"  ch{c}: n={n:5d}  AE_MAE={mae_ae[c].item():7.4f}  "
                f"mean_MAE={mae_mean[c].item():7.4f}  ratio={ratio:.3f}"
            )

    if plot_panels:
        n = len(plot_panels)
        fig, axes = plt.subplots(n, 2, figsize=(12, 2.6 * n), squeeze=False)
        freqs_khz = freq_axis_khz()
        time_ms = np.linspace(0, (trunc_t - 1) * (256 / TARGET_FS) * 1e3, trunc_t)
        for i, (in_spec, re_spec, c, b) in enumerate(plot_panels):
            vmin = float(min(in_spec.min(), re_spec.min()))
            vmax = float(max(in_spec.max(), re_spec.max()))
            extent = [time_ms[0], time_ms[-1], freqs_khz[0], freqs_khz[-1]]
            axes[i, 0].imshow(
                in_spec, origin="lower", cmap="magma",
                vmin=vmin, vmax=vmax, aspect="auto", extent=extent,
            )
            axes[i, 0].set_title(f"input  sample={b} ch={c}", fontsize=9)
            axes[i, 0].set_ylabel("kHz")
            axes[i, 1].imshow(
                re_spec, origin="lower", cmap="magma",
                vmin=vmin, vmax=vmax, aspect="auto", extent=extent,
            )
            axes[i, 1].set_title(f"recon  sample={b} ch={c}", fontsize=9)
            for ax in axes[i]:
                ax.tick_params(labelsize=7)
            if i == n - 1:
                for ax in axes[i]:
                    ax.set_xlabel("time (ms)")
        fig.tight_layout()
        out_path = out_dir / f"recon_step{step:06d}.png"
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        logger.info(f"  saved {out_path}")

    tokenizer.train()
    head.train()
    return {
        "step": step,
        "modality": modality,
        "mae_ae_per_channel": mae_ae.tolist(),
        "mae_mean_per_channel": mae_mean.tolist(),
        "counts_per_channel": counts.tolist(),
    }


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--modality",
        type=str,
        choices=sorted(MODALITY_CONFIG.keys()),
        required=True,
        help="Spectrogram modality to train an AE for.",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("/scratch/gpfs/EKOLEMEN/foundation_model"),
    )
    parser.add_argument(
        "--stats_path",
        type=Path,
        default=Path(
            "/projects/EKOLEMEN/foundation_model/preprocessing_stats.pt"
        ),
        help="preprocessing_stats.pt providing log mean/std for log_standardize.",
    )
    parser.add_argument("--checkpoint_dir", type=Path, default=None)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--val_every", type=int, default=500)
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
    if args.checkpoint_dir is None:
        args.checkpoint_dir = Path(f"runs/spectrogram_ae_{args.modality}")
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    n_channels, patch_f, patch_t = MODALITY_CONFIG[args.modality]
    trunc_t = (TIME_FRAMES // patch_t) * patch_t  # 96

    if not args.stats_path.exists():
        raise SystemExit(
            f"preprocessing_stats not found at {args.stats_path}. "
            "Pass --stats_path or fix the default."
        )
    stats = torch.load(args.stats_path, weights_only=False)
    logger.info(f"Loaded preprocessing stats from {args.stats_path}")

    # ── Files ────────────────────────────────────────────────────────────
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
    logger.info(
        f"{args.modality}: {len(train_files)} train files, {len(val_files)} val files"
    )

    # ── Datasets ─────────────────────────────────────────────────────────
    ds_kwargs = dict(
        chunk_duration_s=0.05,
        prediction_mode=True,
        prediction_horizon_s=0.05,
        input_signals=[args.modality],
        target_signals=[args.modality],
        preprocessing_stats=stats,
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
    tokenizer = SpectrogramTokenizer(
        n_channels=n_channels,
        d_model=D_MODEL,
        patch_f=patch_f,
        patch_t=patch_t,
        freq_bins=FREQ_BINS,
        time_frames=TIME_FRAMES,
    ).to(device)
    head = SpectrogramOutputHead(
        n_channels=n_channels,
        d_model=D_MODEL,
        patch_f=patch_f,
        patch_t=patch_t,
        n_patches_f=FREQ_BINS // patch_f,
        n_patches_t=trunc_t // patch_t,
    ).to(device)
    n_tok = sum(p.numel() for p in tokenizer.parameters())
    n_head = sum(p.numel() for p in head.parameters())
    logger.info(
        f"Model params ({args.modality}): tokenizer={n_tok / 1e6:.2f}M  "
        f"head={n_head / 1e6:.2f}M  total={(n_tok + n_head) / 1e6:.2f}M  "
        f"n_tokens={tokenizer.n_tokens}"
    )

    optimizer = torch.optim.AdamW(
        list(tokenizer.parameters()) + list(head.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ── Train ────────────────────────────────────────────────────────────
    logger.info(
        f"Starting AE training: max_steps={args.max_steps} "
        f"batch={args.batch_size} lr={args.lr} "
        f"patch=(F={patch_f}, T={patch_t})"
    )
    train_iter = iter(train_loader)
    t0 = time.time()
    history: list[dict] = []
    val_records: list[dict] = []
    skipped_no_modality = 0

    step = 0
    while step < args.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        inputs = batch["inputs"]
        x = inputs[args.modality].to(device, non_blocking=True)            # (B, C, F, T)
        valid = inputs[f"{args.modality}_valid"].to(device, non_blocking=True)
        gate_b = valid > 0
        if gate_b.sum() == 0:
            skipped_no_modality += 1
            continue

        target = x[..., :trunc_t]                                          # (B, C, F, T_trunc) data-loader-normalized
        tokens = tokenizer(x)
        recon = head(tokens)

        gate = gate_b[:, None, None, None].float()
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
                # Per-(B, C) mean baseline: how well "predict the
                # constant per-window per-channel mean" does. This is
                # the right competitor for the AE without per-batch
                # z-score (predict-zero would be a weak baseline since
                # the dataloader's log_standardize centres each channel
                # near 0 globally but per-window means vary).
                mae_mean = masked_mae(
                    per_bc_mean(target), target, gate
                ).item()
            elapsed = max(time.time() - t0, 1e-6)
            sps = (step + 1) / elapsed
            logger.info(
                f"step {step:6d}/{args.max_steps}  "
                f"loss={loss.item():.4f}  "
                f"mean_baseline={mae_mean:.4f}  "
                f"delta={loss.item() - mae_mean:+.4f}  "
                f"{sps:5.2f} steps/s  "
                f"skipped_no_mod={skipped_no_modality}"
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
                    tokenizer, head, val_loader, device,
                    args.checkpoint_dir, step,
                    modality=args.modality, trunc_t=trunc_t,
                )
            )

        step += 1

    # Final validation + save.
    val_records.append(
        run_validation(
            tokenizer, head, val_loader, device,
            args.checkpoint_dir, step,
            modality=args.modality, trunc_t=trunc_t,
        )
    )

    final_path = args.checkpoint_dir / f"spectrogram_ae_{args.modality}_final.pt"
    torch.save(
        {
            "tokenizer_state_dict": tokenizer.state_dict(),
            "head_state_dict": head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "history": history,
            "val_records": val_records,
            "skipped_no_modality": skipped_no_modality,
        },
        final_path,
    )
    logger.info(f"Saved {final_path}")

    if history:
        steps = [h["step"] for h in history]
        losses = [h["loss"] for h in history]
        means = [h["mean_baseline"] for h in history]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(steps, losses, label="AE recon MAE", color="tab:blue")
        ax.plot(steps, means, label="mean baseline MAE",
                color="tab:orange", linestyle="--")
        ax.set_xlabel("step")
        ax.set_ylabel("masked MAE (data-loader-normalized space)")
        ax.set_title(f"Standalone {args.modality.upper()} spectrogram AE")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        loss_plot = args.checkpoint_dir / "loss_curve.png"
        fig.savefig(loss_plot, dpi=110)
        plt.close(fig)
        logger.info(f"Saved {loss_plot}")


if __name__ == "__main__":
    main()
