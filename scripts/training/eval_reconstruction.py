from pathlib import Path
import argparse
import logging
import random

import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from tokamak_foundation_model.data.multi_file_dataset import TokamakMultiFileDataset
from tokamak_foundation_model.data.data_loader import collate_fn
from tokamak_foundation_model.models.model_factory import (
    build_model, MODEL_REGISTRY, SIGNAL_MODEL_DEFAULTS)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _plot_sample(
        input_data: np.ndarray,
        recon_data: np.ndarray,
        valid_length: int,
        loss: float,
        sample_idx: int,
        path: Path,
) -> None:
    """Save input vs. reconstruction plot for all channels to *path*."""
    C = input_data.shape[0]
    T = valid_length if valid_length > 0 else input_data.shape[1]
    t = np.arange(T)

    fig, axes = plt.subplots(C, 1, figsize=(12, 1.8 * C), sharex=True)
    if C == 1:
        axes = [axes]

    for c, ax in enumerate(axes):
        ax.plot(t, input_data[c, :T], color="steelblue", lw=0.7, label="Input")
        ax.plot(t, recon_data[c, :T], color="tomato", lw=0.7, label="Recon", alpha=0.85)
        ax.set_ylabel(f"ch{c}", fontsize=7)
        ax.tick_params(labelsize=6)
        if c == 0:
            ax.legend(fontsize=7, loc="upper right")

    axes[-1].set_xlabel("Sample index", fontsize=8)
    fig.suptitle(f"Sample {sample_idx}  |  L1 = {loss:.4f}", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a unimodal autoencoder and save reconstruction plots."
    )
    parser.add_argument(
        "--signal", choices=list(SIGNAL_MODEL_DEFAULTS.keys()),
        default="filterscopes",
    )
    parser.add_argument(
        "--model", choices=list(MODEL_REGISTRY.keys()),
        default="fast_time_series",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=False,
        default="/scratch/gpfs/ps9551/FusionAIHub/scripts/slurm/runs/filterscopes_fast_time_series/checkpoint.pth",
        help="Path to checkpoint (.pth).  Accepts both full training checkpoints "
             "(with 'model_state_dict' key) and bare state-dicts.",
    )
    parser.add_argument(
        "--data_dir", type=str,
        default="/scratch/gpfs/EKOLEMEN/foundation_model/",
    )
    parser.add_argument(
        "--stats_path", type=str,
        default="/scratch/gpfs/ps9551/FusionAIHub/scripts/slurm/preprocessing_stats.pt",
    )
    parser.add_argument(
        "--output_dir", type=str, default="eval_output",
        help="Directory where per-sample PNGs and summary files are written.",
    )
    parser.add_argument(
        "--split", choices=["train", "val", "test"], default="test",
        help="Dataset split to evaluate (mirrors the training-script split logic).",
    )
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_tokens", type=int, default=220)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--hop_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Stop after this many samples (default: whole split).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Dataset split (mirrors fast_time_series_reconstruction.py) ----------
    hdf5_files = sorted(Path(args.data_dir).glob("*_processed.h5"))
    n = len(hdf5_files)
    n_val  = int(0.1 * n)
    n_test = int(0.1 * n)

    split_paths = {
        "val":   hdf5_files[:n_val],
        "test":  hdf5_files[n_val:n_val + n_test],
        "train": hdf5_files[n_val + n_test:],
    }[args.split]

    logger.info(f"Split '{args.split}': {len(split_paths)} files")

    stats = torch.load(args.stats_path, weights_only=False)
    signal_name = args.signal

    dataset = TokamakMultiFileDataset(
        split_paths,
        preprocessing_stats=stats,
        input_signals=[signal_name],
        target_signals=[signal_name],
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        prediction_mode=False,
    )
    logger.info(f"Dataset size: {len(dataset)}")

    n_channels = dataset[0][signal_name].shape[0]

    # --- Model -------------------------------------------------------------------
    model = build_model(
        args.model,
        d_model=args.d_model,
        n_tokens=args.n_tokens,
        n_channels=n_channels,
        kernel_size=3,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    # --- DataLoader (no shuffle → deterministic ordering) ----------------------
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # --- Evaluation loop -------------------------------------------------------
    all_losses: list[float] = []
    global_idx = 0
    max_n = args.max_samples or len(dataset)

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Evaluating"):
            if global_idx >= max_n:
                break

            data = batch[signal_name].to(device)
            valid_lengths = batch.get(f"{signal_name}_valid")
            vl_list = (
                valid_lengths.tolist()
                if valid_lengths is not None
                else [data.shape[-1]] * data.shape[0]
            )

            output = model(data)
            if isinstance(output, tuple):
                output = output[0]

            data_np  = data.cpu().numpy()
            recon_np = output.cpu().numpy()

            for i in range(data_np.shape[0]):
                if global_idx >= max_n:
                    break

                vl   = vl_list[i]
                inp  = data_np[i]   # [C, T]
                rec  = recon_np[i]  # [C, T]
                loss = float(np.abs(inp[:, :vl] - rec[:, :vl]).mean())
                all_losses.append(loss)

                _plot_sample(
                    inp, rec, vl, loss, global_idx,
                    output_dir / f"sample_{global_idx:05d}.png",
                )
                global_idx += 1

    # --- Summary -----------------------------------------------------------------
    losses = np.array(all_losses)
    logger.info(
        f"Evaluated {global_idx} samples  "
        f"| mean L1 = {losses.mean():.4f}  "
        f"| std = {losses.std():.4f}  "
        f"| min = {losses.min():.4f}  "
        f"| max = {losses.max():.4f}"
    )

    np.save(output_dir / "losses.npy", losses)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(losses, bins=50, edgecolor="white")
    ax.set_xlabel("Per-sample L1 loss")
    ax.set_ylabel("Count")
    ax.set_title(f"Reconstruction loss — {args.split} split  (n={global_idx})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "loss_histogram.png", dpi=120)
    plt.close(fig)

    logger.info(f"Saved {global_idx} plots and summary to {output_dir}/")


if __name__ == "__main__":
    main()