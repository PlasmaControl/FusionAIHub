#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Any, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None


def add_src_to_path() -> Path:
    repo_root = Path().resolve().parents[1]
    sys.path.append(str(repo_root / "src"))
    return repo_root


def build_dataloader(
    hdf5_files: list[Path],
    signal: str,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    from tokamak_foundation_model.data.data_loader import TokamakH5Dataset, collate_fn
    from tokamak_foundation_model.data.utils import worker_init_fn

    datasets = [
        TokamakH5Dataset(
            hdf5_path=str(f),
            input_signals=[signal],
            target_signals=[signal],
            prediction_mode=False,
        )
        for f in hdf5_files
    ]
    dataset = ConcatDataset(datasets)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
        shuffle=shuffle,
    )


def split_files(data_dir: Path, file_glob: str, seed: int = 42) -> tuple[list[Path], list[Path], list[Path]]:
    hdf5_files = sorted(data_dir.glob(file_glob))
    if not hdf5_files:
        raise FileNotFoundError(f"No HDF5 files matched: {data_dir}/{file_glob}")

    random.seed(seed)
    # Keep the exact split logic from the new training script.
    n = len(hdf5_files)
    n_val = int(0.1 * n)
    n_test = int(0.1 * n)

    train_paths = hdf5_files[n_val + n_test:]
    val_paths = hdf5_files[:n_val]
    test_paths = hdf5_files[n_val:n_val + n_test]
    return train_paths, val_paths, test_paths


def load_checkpoint_weights(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "model", "state_dict", "model_state"):
            if key in ckpt and isinstance(ckpt[key], dict):
                model.load_state_dict(ckpt[key], strict=True)
                return ckpt

        try:
            model.load_state_dict(ckpt, strict=True)
            return {"state_dict": ckpt}
        except Exception:
            pass

    raise RuntimeError(
        "Could not find model weights in checkpoint. Expected keys like "
        "'model_state_dict', 'state_dict', 'model', or a raw state_dict."
    )


def extract_xy(batch: Any, signal: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, dict):
        if signal in batch and torch.is_tensor(batch[signal]):
            x = batch[signal]
            return x, x

        if "x" in batch and isinstance(batch["x"], dict) and signal in batch["x"]:
            x = batch["x"][signal]
            y = batch.get("y", {}).get(signal, x) if isinstance(batch.get("y"), dict) else x
            return x, y

        if "inputs" in batch and isinstance(batch["inputs"], dict) and signal in batch["inputs"]:
            x = batch["inputs"][signal]
            y = batch.get("targets", {}).get(signal, x) if isinstance(batch.get("targets"), dict) else x
            return x, y

        for _, value in batch.items():
            if torch.is_tensor(value) and value.ndim >= 4:
                return value, value

        raise RuntimeError(f"Unrecognized batch dict format. Keys={list(batch.keys())}")

    if isinstance(batch, (tuple, list)):
        if len(batch) >= 2 and torch.is_tensor(batch[0]) and torch.is_tensor(batch[1]):
            return batch[0], batch[1]
        if len(batch) >= 1 and torch.is_tensor(batch[0]):
            return batch[0], batch[0]

    raise RuntimeError(f"Unrecognized batch type: {type(batch)}")


def ensure_bcthw(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 5:
        raise ValueError(f"Expected a 5D tensor, got shape={tuple(x.shape)}")

    # New training script inspects sample_data.shape[1] as channel dimension,
    # so the expected layout is already (B, C, T, H, W).
    return x


class SparseVideoWeightedMSEFallback(torch.nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weight = 1.0 + 10.0 * target.abs()
        return torch.mean(weight * (pred - target) ** 2)


def make_loss_fn() -> torch.nn.Module:
    try:
        from tokamak_foundation_model.models.loss import SparseVideoWeightedMSE
        return SparseVideoWeightedMSE()
    except Exception:
        return SparseVideoWeightedMSEFallback()


def save_frame_triplet(out_dir: Path, prefix: str, frame_in, frame_rec, frame_err, vmin=None, vmax=None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    ax0 = axes[0].imshow(frame_in, cmap="hot", vmin=vmin, vmax=vmax)
    axes[0].set_title("input")
    axes[0].axis("off")
    plt.colorbar(ax0, ax=axes[0])

    ax1 = axes[1].imshow(frame_rec, cmap="hot", vmin=vmin, vmax=vmax)
    axes[1].set_title("recon")
    axes[1].axis("off")
    plt.colorbar(ax1, ax=axes[1])

    ax2 = axes[2].imshow(frame_err, cmap="hot")
    axes[2].set_title("abs error")
    axes[2].axis("off")
    plt.colorbar(ax2, ax=axes[2])

    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}.png", dpi=150)
    plt.close(fig)


def save_gif(out_path: Path, vid_in, vid_rec, fps: float = 20.0, vmin=None, vmax=None) -> None:
    if imageio is None:
        raise RuntimeError("imageio is not available; install it to save GIFs.")

    frames = []
    t_steps = vid_in.shape[0]
    for t in range(t_steps):
        fig, axes = plt.subplots(1, 2, figsize=(7, 3))
        axes[0].imshow(vid_in[t], cmap="hot", vmin=vmin, vmax=vmax)
        axes[0].set_title(f"input t={t}")
        axes[0].axis("off")
        axes[1].imshow(vid_rec[t], cmap="hot", vmin=vmin, vmax=vmax)
        axes[1].set_title(f"recon t={t}")
        axes[1].axis("off")
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(torch.tensor(fig.canvas.buffer_rgba()).numpy()[:, :, :3])
        plt.close(fig)

    imageio.mimsave(out_path, frames, duration=1.0 / max(fps, 1e-6))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a trained multi-channel video reconstruction model")

    parser.add_argument("--signal", type=str, default="tangtv")
    parser.add_argument("--model", type=str, default="video")
    parser.add_argument("--data_dir", type=str, default="/scratch/gpfs/aj17/datasets/fm_test/")
    parser.add_argument("--file_glob", type=str, default="*_processed.h5")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="val")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--n_tokens", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--shuffle", action="store_true")

    parser.add_argument("--max_batches", type=int, default=0, help="0 means evaluate the full split")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--num_visualizations", type=int, default=2)
    parser.add_argument("--frames_per_sample", type=int, default=4)
    parser.add_argument("--out_dir", type=str, default="recon_validation")
    parser.add_argument("--make_gif", action="store_true")
    parser.add_argument("--gif_fps", type=float, default=20.0)

    args = parser.parse_args()

    repo_root = add_src_to_path()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("eval_video_reconstruction_new")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("repo_root=%s", repo_root)
    logger.info("device=%s", device)

    from tokamak_foundation_model.models.model_factory import build_model

    data_dir = Path(args.data_dir)
    checkpoint_path = Path(args.checkpoint_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    train_paths, val_paths, test_paths = split_files(data_dir, args.file_glob, seed=args.seed)
    split_map = {
        "train": train_paths,
        "val": val_paths,
        "test": test_paths,
        "all": train_paths + val_paths + test_paths,
    }
    selected_paths = split_map[args.split]
    if not selected_paths:
        raise RuntimeError(
            f"Selected split '{args.split}' is empty. "
            f"Dataset has train={len(train_paths)}, val={len(val_paths)}, test={len(test_paths)} files."
        )

    logger.info(
        "Using split=%s with %d files (train=%d, val=%d, test=%d)",
        args.split, len(selected_paths), len(train_paths), len(val_paths), len(test_paths)
    )

    dataloader = build_dataloader(
        hdf5_files=selected_paths,
        signal=args.signal,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=args.shuffle,
    )

    sample_batch = next(iter(dataloader))
    sample_x, _ = extract_xy(sample_batch, args.signal)
    sample_x = ensure_bcthw(sample_x)
    n_channels = sample_x.shape[1]
    logger.info("Sample tensor shape=%s -> inferred n_channels=%d", tuple(sample_x.shape), n_channels)

    model = build_model(args.model, d_model=args.d_model, n_tokens=args.n_tokens, n_channels=n_channels).to(device)
    logger.info("model params=%d", sum(p.numel() for p in model.parameters()))

    checkpoint = load_checkpoint_weights(model, checkpoint_path, device)
    logger.info("Loaded checkpoint: %s", checkpoint_path)
    if isinstance(checkpoint, dict):
        logger.info("Checkpoint keys: %s", sorted(checkpoint.keys()))

    loss_fn = make_loss_fn().to(device)
    model.eval()

    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    total_items = 0
    vis_done = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break

            x, y = extract_xy(batch, args.signal)
            x = ensure_bcthw(x).to(device).float()
            y = ensure_bcthw(y).to(device).float()

            y_hat = model(x)

            batch_size = x.shape[0]
            batch_loss = loss_fn(y_hat, y)
            batch_mse = F.mse_loss(y_hat, y)
            batch_mae = F.l1_loss(y_hat, y)

            total_loss += float(batch_loss.item()) * batch_size
            total_mse += float(batch_mse.item()) * batch_size
            total_mae += float(batch_mae.item()) * batch_size
            total_items += batch_size

            logger.info(
                "batch=%04d size=%d weighted_loss=%.6f mse=%.6f mae=%.6f recon_mean=%.5f recon_std=%.5f",
                batch_idx,
                batch_size,
                float(batch_loss.item()),
                float(batch_mse.item()),
                float(batch_mae.item()),
                float(y_hat.mean().item()),
                float(y_hat.std().item()),
            )

            while vis_done < args.num_visualizations and vis_done < batch_size:
                b = max(0, min(args.sample_index + vis_done, batch_size - 1))
                vin = x[b].detach().cpu()
                vrec = y_hat[b].detach().cpu()
                verr = (vin - vrec).abs()

                t_steps = vin.shape[1]
                frame_ids = sorted(set([
                    0,
                    t_steps // 4,
                    t_steps // 2,
                    (3 * t_steps) // 4,
                    max(0, t_steps - 1),
                ]))[: max(1, args.frames_per_sample)]

                sample_dir = out_dir / f"batch{batch_idx:04d}_sample{b:02d}"
                sample_dir.mkdir(parents=True, exist_ok=True)

                per_channel_mse = ((vrec - vin) ** 2).mean(dim=(1, 2, 3))
                per_channel_mae = (vrec - vin).abs().mean(dim=(1, 2, 3))
                with open(sample_dir / "metrics.txt", "w", encoding="utf-8") as f:
                    f.write(f"batch_index: {batch_idx}\n")
                    f.write(f"sample_index: {b}\n")
                    f.write(f"shape_bcthw: {tuple(x.shape)}\n")
                    f.write(f"sample_shape_cthw: {tuple(vin.shape)}\n")
                    f.write(f"global_mse: {float(((vrec - vin) ** 2).mean().item()):.8f}\n")
                    f.write(f"global_mae: {float((vrec - vin).abs().mean().item()):.8f}\n")
                    for c in range(vin.shape[0]):
                        f.write(f"channel_{c}_mse: {float(per_channel_mse[c].item()):.8f}\n")
                        f.write(f"channel_{c}_mae: {float(per_channel_mae[c].item()):.8f}\n")

                for c in range(vin.shape[0]):
                    vmin = float(vin[c].min().item())
                    vmax = float(vin[c].max().item())
                    channel_dir = sample_dir / f"channel_{c}"
                    channel_dir.mkdir(parents=True, exist_ok=True)
                    for t in frame_ids:
                        save_frame_triplet(
                            channel_dir,
                            prefix=f"t{t:03d}",
                            frame_in=vin[c, t],
                            frame_rec=vrec[c, t],
                            frame_err=verr[c, t],
                            vmin=vmin,
                            vmax=vmax,
                        )
                    if args.make_gif and vis_done == 0:
                        save_gif(
                            channel_dir / "reconstruction.gif",
                            vid_in=vin[c],
                            vid_rec=vrec[c],
                            fps=args.gif_fps,
                            vmin=vmin,
                            vmax=vmax,
                        )

                vis_done += 1
                if vis_done >= args.num_visualizations:
                    break

    if total_items == 0:
        raise RuntimeError("No samples were evaluated.")

    summary_path = out_dir / f"summary_{args.split}.txt"
    avg_loss = total_loss / total_items
    avg_mse = total_mse / total_items
    avg_mae = total_mae / total_items

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"split: {args.split}\n")
        f.write(f"num_files: {len(selected_paths)}\n")
        f.write(f"num_samples: {total_items}\n")
        f.write(f"weighted_loss: {avg_loss:.8f}\n")
        f.write(f"mse: {avg_mse:.8f}\n")
        f.write(f"mae: {avg_mae:.8f}\n")
        f.write(f"checkpoint_path: {checkpoint_path}\n")

    logger.info("Validation complete")
    logger.info("samples=%d weighted_loss=%.8f mse=%.8f mae=%.8f", total_items, avg_loss, avg_mse, avg_mae)
    logger.info("Wrote summary to %s", summary_path.resolve())
    logger.info("Saved visualizations to %s", out_dir.resolve())


if __name__ == "__main__":
    main()
