from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader


@runtime_checkable
class DrawerProtocol(Protocol):
    def setup(self, dataloader: DataLoader, drawing_path: Path, modality_key: str) -> None: ...
    def __call__(self, model: torch.nn.Module, epoch: int, train_loss: float, val_loss: float | None = None) -> None: ...


class NullDrawer:
    """No-op drawer for non-main processes or when visualization is disabled."""

    def setup(self, dataloader: DataLoader, drawing_path: Path, modality_key: str) -> None:
        pass

    def __call__(self, model: torch.nn.Module, epoch: int, train_loss: float, val_loss: float | None = None) -> None:
        pass


class DefaultDrawer:

    def __init__(self, plot_channel: int | None = None):
        self._plot_channel: int | None = plot_channel

    def setup(self, dataloader: DataLoader, drawing_path: Path, modality_key: str) -> None:
        self.drawing_path = Path(drawing_path)
        self.drawing_path.mkdir(parents=True, exist_ok=True)
        self.modality_key = modality_key

        # Pick the highest-variance sample from the first 200 candidates
        dataset = dataloader.dataset
        n_candidates = min(200, len(dataset))
        best_idx, best_var = 0, -1.0
        for i in range(n_candidates):
            sample = dataset[i][modality_key]
            v = sample.var().item()
            if v > best_var:
                best_var = v
                best_idx = i
        self.probe_sample = dataset[best_idx][modality_key]

        if self._plot_channel is not None:
            self.channel = self._plot_channel
        else:
            self.channel = self.probe_sample.shape[0] // 2

        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    @torch.no_grad()
    def __call__(self, model: torch.nn.Module, epoch: int, train_loss: float, val_loss: float | None = None) -> None:
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)

        model.eval()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

        ax1.plot(self.train_losses, color='blue', label='Train')
        if self.val_losses:
            ax1.plot(self.val_losses, color='orange', label='Val')
        ax1.set_xlabel('Log Step')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        x = self.probe_sample.unsqueeze(0).to(next(model.parameters()).device)
        output = model(x)
        if isinstance(output, tuple):
            output = output[0]
        output = output[0].cpu()

        ax2.imshow(output[self.channel].numpy(), cmap='viridis', origin='lower', aspect='auto')
        ax2.set_axis_off()

        val_str = f" | Val L1={val_loss:.6f}" if val_loss is not None else ""
        fig.suptitle(f"Epoch {epoch+1} | Train L1={train_loss:.6f}{val_str}")
        fig.tight_layout()
        fig.savefig(self.drawing_path / f"probe_epoch_{epoch+1:03d}.png")
        plt.close(fig)
