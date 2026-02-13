from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader


class DefaultDrawer:
    def __init__(self, num_plots: int = 4, plot_indices: list[int] | None = None):
        self.num_plots = num_plots
        self.plot_indices = plot_indices

    def setup(self, dataloader: DataLoader, drawing_path: Path, modality_key: str):
        self.drawing_path = drawing_path
        self.drawing_path.mkdir(parents=True, exist_ok=True)
        self.modality_key = modality_key

        dataset = dataloader.dataset
        n_samples = len(dataset)

        if self.plot_indices is None:
            self.plot_indices = np.random.choice(
                n_samples, min(self.num_plots, n_samples), replace=False
            )

        self.input_data = [dataset[i][modality_key] for i in self.plot_indices]
        self.ndim = self.input_data[0].ndim
        self.half_channel = self.input_data[0].shape[0] // 2

    def _draw_1d(self, input_data: torch.Tensor, output_data: torch.Tensor, path: Path, title: str):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
        ax1.plot(input_data.numpy())
        ax1.set_title("Input")
        ax2.plot(output_data.numpy())
        ax2.set_title("Reconstruction")
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    def _draw_2d(self, input_data: torch.Tensor, output_data: torch.Tensor, path: Path, title: str):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.imshow(input_data.numpy(), aspect="auto", origin="lower")
        ax1.set_title("Input")
        ax2.imshow(output_data.numpy(), aspect="auto", origin="lower")
        ax2.set_title("Reconstruction")
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    @torch.no_grad()
    def __call__(self, model: torch.nn.Module, epoch: int, train_loss: float, val_loss: float):
        model.eval()
        for i, input_tensor in enumerate(self.input_data):
            x = input_tensor.unsqueeze(0).to(next(model.parameters()).device)
            output = model(x)[self.half_channel].cpu()
            inp = input_tensor

            title = f"Epoch {epoch+1} | Train L1={train_loss:.4f} Val L1={val_loss:.4f}"
            path = self.drawing_path / f"epoch_{epoch+1:03d}_sample_{i}.png"

            # Pick first channel for visualization
            inp_vis = inp[0]
            out_vis = output[0]

            match self.ndim:
                case 2:  # (C, T) — 1D signals
                    self._draw_1d(inp_vis, out_vis, path, title)
                case 3:  # (C, F, T) — spectrograms
                    self._draw_2d(inp_vis, out_vis, path, title)
                case 4:  # (C, T, H, W) — video, show first frame
                    self._draw_2d(inp_vis[0], out_vis[0], path, title)
