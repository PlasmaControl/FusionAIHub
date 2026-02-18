#!/usr/bin/env python3
"""
Evaluate / visualize reconstructions from a trained video autoencoder.

Typical repo layout:
  repo/
    src/tokamak_foundation_model/...
    script/eval_video_reconstruction.py

Run from repo root (recommended):
  python script/eval_video_reconstruction.py --data_dir ... --checkpoint_path ...

Or from anywhere:
  python /abs/path/to/eval_video_reconstruction.py ...

This script:
- Adds <repo_root>/src to sys.path (like the training script)
- Builds the same dataloader (TokamakH5Dataset + collate_fn + worker_init_fn)
- Builds the same model (video_baseline.VideoBaselineAutoEncoder)
- Loads checkpoint weights
- Runs a few batches and saves input/recon/error PNGs (and optional GIF)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import logging
from typing import Optional, Tuple, Any, Dict

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

import matplotlib
matplotlib.use("Agg")  # headless safe
import matplotlib.pyplot as plt

try:
    import imageio.v2 as imageio  # optional for GIFs
except Exception:
    imageio = None

# -------------------------
# Path setup: add repo_root/src
# -------------------------
def add_src_to_path() -> Path:
    this_file = Path(__file__).resolve()
    repo_root = Path().resolve().parents[0]
    sys.path.append(str(repo_root / "src"))
    return repo_root


def build_dataloader(
    data_dir: Path,
    file_glob: str,
    signal: str,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    from tokamak_foundation_model.data.data_loader import TokamakH5Dataset, collate_fn
    from tokamak_foundation_model.data.utils import worker_init_fn

    hdf5_files = sorted(data_dir.glob(file_glob))
    if len(hdf5_files) == 0:
        raise FileNotFoundError(f"No HDF5 files matched: {data_dir}/{file_glob}")

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


def build_model(
    n_tokens: int,
    token_dim: int,
    t_clip: int,
    image_size: int,
    device: torch.device,
):
    from tokamak_foundation_model.models.modality import video_baseline

    model = video_baseline.VideoBaselineAutoEncoder(
        n_tokens=n_tokens,
        token_dim=token_dim,
    ).to(device)
    return model


def load_checkpoint_weights(model: nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device)
    # Common patterns
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "model", "state_dict", "model_state"):
            if key in ckpt and isinstance(ckpt[key], dict):
                model.load_state_dict(ckpt[key])
                return
        # Sometimes it's already a state_dict
        if all(isinstance(k, str) for k in ckpt.keys()):
            try:
                model.load_state_dict(ckpt)
                return
            except Exception:
                pass

    raise RuntimeError(
        "Could not find model weights in checkpoint. Expected keys like "
        "'model_state_dict' / 'state_dict' etc."
    )


def extract_xy(batch: Any, signal: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Tries common batch formats used by collate_fn.
    Returns x, y tensors shaped like (B, T, H, W).
    """
    if isinstance(batch, dict):
        # Case: batch[signal] = tensor
        if signal in batch and torch.is_tensor(batch[signal]):
            x = batch[signal]
            return x, x

        # Case: batch["x"][signal], batch["y"][signal]
        if "x" in batch and isinstance(batch["x"], dict) and signal in batch["x"]:
            x = batch["x"][signal]
            if "y" in batch and isinstance(batch["y"], dict) and signal in batch["y"]:
                y = batch["y"][signal]
            else:
                y = x
            return x, y

        # Case: batch["inputs"][signal], batch["targets"][signal]
        if "inputs" in batch and isinstance(batch["inputs"], dict) and signal in batch["inputs"]:
            x = batch["inputs"][signal]
            y = x
            if "targets" in batch and isinstance(batch["targets"], dict) and signal in batch["targets"]:
                y = batch["targets"][signal]
            return x, y

        # Fall back: search for any tensor that looks like video
        for k, v in batch.items():
            if torch.is_tensor(v) and v.ndim == 4:
                return v, v

        raise RuntimeError(f"Unrecognized batch dict format. Keys={list(batch.keys())}")

    if isinstance(batch, (tuple, list)):
        if len(batch) >= 2 and torch.is_tensor(batch[0]) and torch.is_tensor(batch[1]):
            return batch[0], batch[1]
        if len(batch) >= 1 and torch.is_tensor(batch[0]):
            return batch[0], batch[0]

    raise RuntimeError(f"Unrecognized batch type: {type(batch)}")


# -------------------------
# Visualization helpers
# -------------------------
def save_frame_triplet(out_dir: Path, prefix: str, frame_in, frame_rec, vmin=None, vmax=None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    err = (frame_in - frame_rec).abs()

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    ax0 = axes[0].imshow(frame_in, cmap="hot", vmin=vmin, vmax=vmax)
    axes[0].set_title("input")
    axes[0].axis("off")
    plt.colorbar(ax0,ax=axes[0])

    ax1 = axes[1].imshow(frame_rec, cmap="hot", vmin=vmin, vmax=vmax)
    axes[1].set_title("recon")
    axes[1].axis("off")
    plt.colorbar(ax1,ax=axes[1])

    ax2 = axes[2].imshow(err, cmap="hot", vmin=vmin, vmax=vmax)
    axes[2].set_title("abs error")
    axes[2].axis("off")
    plt.colorbar(ax2,ax=axes[2])

    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}.png", dpi=150)
    plt.close(fig)


def save_gif(out_path: Path, vid_in, vid_rec, fps: float = 20.0, vmin=None, vmax=None) -> None:
    if imageio is None:
        raise RuntimeError("imageio is not available; install it to save GIFs (pip install imageio).")

    frames = []
    T = vid_in.shape[0]
    for t in range(T):
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        axes[0].imshow(vid_in[t], cmap="gray", vmin=vmin, vmax=vmax)
        axes[0].set_title(f"in t={t}")
        axes[0].axis("off")
        axes[1].imshow(vid_rec[t], cmap="gray", vmin=vmin, vmax=vmax)
        axes[1].set_title(f"rec t={t}")
        axes[1].axis("off")
        fig.tight_layout()

        # draw to RGB array
        fig.canvas.draw()
        img = torch.tensor(fig.canvas.buffer_rgba()).numpy()[:, :, :3]
        frames.append(img)
        plt.close(fig)

    duration = 1.0 / max(fps, 1e-6)
    imageio.mimsave(out_path, frames, duration=duration)


def main():
    parser = argparse.ArgumentParser(description="Evaluate reconstructions from a trained video autoencoder")
    parser.add_argument("--signal", type=str, default="bolo")
    parser.add_argument("--data_dir", type=str, default="/scratch/gpfs/EKOLEMEN/big_d3d_data/dummy_foundation_model_data/")
    parser.add_argument("--file_glob", type=str, default="*_processed.h5")

    # Model / preprocessing hyperparams (must match training)
    parser.add_argument("--clip_seconds", type=float, default=0.5)
    parser.add_argument("--target_fps", type=float, default=50.0)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--n_tokens", type=int, default=32)
    parser.add_argument("--token_dim", type=int, default=512)

    # Eval options
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--num_batches", type=int, default=2, help="How many batches to visualize")
    parser.add_argument("--sample_index", type=int, default=0, help="Which sample in batch to visualize")
    parser.add_argument("--out_dir", type=str, default="recon_debug")
    parser.add_argument("--make_gif", action="store_true", help="Save GIF for first visualized sample")
    parser.add_argument("--gif_fps", type=float, default=20.0)
    parser.add_argument("--shuffle", action="store_true")

    args = parser.parse_args()

    repo_root = add_src_to_path()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("eval_video_reconstruction")
    logger.info("repo_root=%s", repo_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s", device)

    data_dir = Path(args.data_dir)
    checkpoint_path = Path(args.checkpoint_path)
    out_dir = Path(args.out_dir)

    t_clip = int(round(args.clip_seconds * args.target_fps))
    logger.info("t_clip=%d", t_clip)

    dl = build_dataloader(
        data_dir=data_dir,
        file_glob=args.file_glob,
        signal=args.signal,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=args.shuffle,
    )

    model = build_model(
        n_tokens=args.n_tokens,
        token_dim=args.token_dim,
        t_clip=t_clip,
        image_size=args.image_size,
        device=device,
    )
    logger.info("model params=%d", sum(p.numel() for p in model.parameters()))

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    load_checkpoint_weights(model, checkpoint_path, device)
    model.eval()
    logger.info("Loaded checkpoint: %s", checkpoint_path)

    # Visualize a few batches
    batches_done = 0
    for batch_idx, batch in enumerate(dl):
        x, y = extract_xy(batch, args.signal)
        x = x.to(device).float()
        with torch.no_grad():
            x_hat = model(x)
        # bring one sample to cpu for plotting
        b = max(0, min(args.sample_index, x.shape[0] - 1))
        vin = x[b].detach().cpu()
        vrec = x_hat[b].detach().cpu()

        # choose vmin/vmax from input range for consistent appearance
        vmin = float(vin.min().item())
        vmax = float(vin.max().item())

        # save a few frame triplets
        T = vin.shape[0]
        frame_ids = [0, T // 4, T // 2, (3 * T) // 4]
        for t in frame_ids:
            prefix = f"batch{batch_idx:03d}_sample{b}_t{t:03d}"
            save_frame_triplet(out_dir, prefix, vin[t], vrec[t], vmin=vmin, vmax=vmax)

        # optional gif
        if args.make_gif and batches_done == 0:
            gif_path = out_dir / f"batch{batch_idx:03d}_sample{b}.gif"
            save_gif(gif_path, vin, vrec, fps=args.gif_fps, vmin=vmin, vmax=vmax)
            logger.info("Saved GIF: %s", gif_path)

        # log quick stats
        logger.info(
            "batch=%d  x_hat_mean=%.4g x_hat_std=%.4g",#  z_shape=%s",
            batch_idx,
            float(x_hat.mean().item()),
            float(x_hat.std().item()),
        )

        batches_done += 1
        if batches_done >= args.num_batches:
            break

    logger.info("Saved outputs to: %s", out_dir.resolve())


if __name__ == "__main__":
    main()
