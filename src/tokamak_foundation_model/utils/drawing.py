from collections.abc import Sized
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


@runtime_checkable
class DrawerProtocol(Protocol):
    """
    Protocol for training-progress visualization callbacks.

    Implementors must provide :meth:`setup` and :meth:`__call__` with the
    signatures below.  :class:`NullDrawer` and :class:`DefaultDrawer` are
    the two built-in implementations.
    """

    def setup(
            self,
            dataloader: DataLoader,
            drawing_path: Path,
            modality_key: str,
    ):
        ...

    def __call__(
            self,
            model: torch.nn.Module,
            epoch: int,
            train_loss: float,
            val_loss: Optional[float] = None,
    ):
        ...


class NullDrawer:
    """No-op drawer for non-main processes or when visualization is disabled."""

    def setup(
            self,
            dataloader: DataLoader,
            drawing_path: Path,
            modality_key: str,
    ):
        pass

    def __call__(
            self,
            model: torch.nn.Module,
            epoch: int,
            train_loss: float,
            val_loss: Optional[float] = None,
    ):
        pass


class DefaultDrawer:
    """
    Visualizes training progress after each epoch.

    Saves two persistent plots to *drawing_path* (overwritten each epoch):

    * ``loss_curve.png`` — cumulative train and optional validation loss over
      epochs.
    * ``reconstruction.png`` — input vs. model output for a fixed probe
      sample.  The visualization adapts to the channel dimensionality:

      =========  ===========================  ===============================
      ``ndim``   Interpretation               Plot type
      =========  ===========================  ===============================
      3          ``(T, H, W)`` — video        Uniform strip of frames
      2          ``(H, W)`` — spectrogram     :func:`~matplotlib.pyplot.imshow`
      1          ``(T,)`` — signal            :func:`~matplotlib.pyplot.plot`
      =========  ===========================  ===============================

    Parameters
    ----------
    plot_channel : int or None, optional
        Index of the channel to visualize.  If ``None`` (default), the
        middle channel (``C // 2``) is selected automatically.

    Attributes
    ----------
    drawing_path : Path
        Directory where plots are saved.  Set by :meth:`setup`.
    probe_sample : torch.Tensor
        Fixed sample used for reconstruction plots.  Shape ``(C, ...)``.
        Set by :meth:`setup`.
    channel : int
        Channel index used for visualization.  Set by :meth:`setup`.
    train_losses : list of float
        Accumulated training losses, one entry per :meth:`__call__`.
    val_losses : list of float
        Accumulated validation losses.  Only populated when *val_loss* is
        passed to :meth:`__call__`.
    """

    _NUM_VIDEO_FRAMES = 6  # number of frames shown in the video strip

    def __init__(
            self,
            plot_channel: Optional[int] = None,
    ):
        self._plot_channel: Optional[int] = plot_channel

    def setup(
            self,
            dataloader: DataLoader,
            drawing_path: Path,
            modality_key: str,
    ):
        """Initialize the drawer with dataset and output directory.

        Must be called once before the first :meth:`__call__`.  Selects a
        fixed probe sample from the dataset and creates *drawing_path*.

        Parameters
        ----------
        dataloader : DataLoader
            Training dataloader.  Its ``dataset`` attribute is used to
            retrieve the probe sample.
        drawing_path : Path
            Directory where ``loss_curve.png`` and ``reconstruction.png``
            will be written.  Created if it does not exist.
        modality_key : str
            Key used to index into each dataset sample dict (e.g.
            ``'spectrogram'``).
        """
        self.drawing_path = Path(drawing_path)
        self.drawing_path.mkdir(parents=True, exist_ok=True)
        self.modality_key = modality_key

        dataset = dataloader.dataset
        assert isinstance(dataset, Sized), "Dataset must implement __len__"
        idx = min(10, len(dataset) - 1)
        sample = dataset[idx]
        self.probe_sample = sample[modality_key]
        self.probe_valid_length: Optional[int] = sample.get(f"{modality_key}_valid")

        if self._plot_channel is not None:
            self.channel = self._plot_channel
        else:
            self.channel = self.probe_sample.shape[0] // 2

        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    @torch.no_grad()
    def __call__(
            self,
            model: torch.nn.Module,
            epoch: int,
            train_loss: float,
            val_loss: Optional[float] = None,
    ):
        """Record losses and save visualization plots for the current epoch.

        Parameters
        ----------
        model : torch.nn.Module
            Trained model, run in eval mode to produce the reconstruction.
        epoch : int
            Zero-based epoch index.
        train_loss : float
            Training loss for this epoch.
        val_loss : float or None, optional
            Validation loss for this epoch, or ``None`` if no validation was
            performed.  Default is ``None``.
        """
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)

        self._save_loss_curve()
        self._save_reconstruction(model, epoch, train_loss, val_loss)

    def _save_loss_curve(self):
        """Write ``loss_curve.png``, overwriting any previous version."""
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(self.train_losses, color='blue', label='Train')
        if self.val_losses:
            ax.plot(self.val_losses, color='orange', label='Val')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(self.drawing_path / "loss_curve.png")
        plt.close(fig)

    def _save_reconstruction(
            self,
            model: torch.nn.Module,
            epoch: int,
            train_loss: float,
            val_loss: Optional[float],
    ):
        """Write ``reconstruction.png``, overwriting any previous version.

        Runs the probe sample through *model* and dispatches to the
        appropriate helper based on the channel dimensionality (3-D video,
        2-D spectrogram, or 1-D signal).
        """
        model.eval()
        x = self.probe_sample.unsqueeze(0).to(next(model.parameters()).device)
        output = model(x)
        if isinstance(output, tuple):
            output = output[0]
        output = output[0].cpu()

        input_data = self.probe_sample[self.channel].numpy()
        recon_data = output[self.channel].numpy()

        # Trim to valid (non-padded) length if available
        vl = self.probe_valid_length
        if vl is not None and vl > 0:
            # Last axis is always the time axis for signals and spectrograms
            input_data = input_data[..., :vl]
            recon_data = recon_data[..., :vl]

        title = f"Epoch {epoch + 1} | Train L1={train_loss:.6f}"
        if val_loss is not None:
            title += f" | Val L1={val_loss:.6f}"

        if recon_data.ndim == 3:
            self._plot_video(input_data, recon_data, title)
        else:
            self._plot_2d_or_1d(input_data, recon_data, title)

    def _plot_video(
            self,
            input_data: np.ndarray,
            recon_data: np.ndarray,
            title: str,
    ):
        """
        Save a frame-strip comparison for video tensors of shape ``(T, H, W)``.

        Selects :attr:`_NUM_VIDEO_FRAMES` frames uniformly across the time
        axis and lays them out in two rows (input on top, reconstruction
        below).

        Parameters
        ----------
        input_data : numpy.ndarray
            Ground-truth video, shape ``(T, H, W)``.
        recon_data : numpy.ndarray
            Model reconstruction, shape ``(T, H, W)``.
        title : str
            Figure super-title.
        """
        n = self._NUM_VIDEO_FRAMES
        indices = np.linspace(0, input_data.shape[0] - 1, n, dtype=int)

        fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
        for col, t in enumerate(indices):
            for row, data in enumerate((input_data, recon_data)):
                axes[row, col].imshow(
                    data[t], cmap='viridis', origin='lower', aspect='auto',
                )
                axes[row, col].set_axis_off()
            axes[0, col].set_title(f't={t}', fontsize=8)

        fig.text(0.01, 0.75, 'Input', va='center', rotation='vertical', fontsize=9)
        fig.text(
            0.01, 0.25, 'Reconstruction', va='center', rotation='vertical', fontsize=9,
        )
        fig.suptitle(title)
        fig.tight_layout(rect=(0.03, 0, 1, 1))
        fig.savefig(self.drawing_path / "reconstruction.png")
        plt.close(fig)

    def _plot_2d_or_1d(
            self,
            input_data: np.ndarray,
            recon_data: np.ndarray,
            title: str,
    ):
        """
        Save an input/reconstruction comparison for 2-D or 1-D tensors.

        Parameters
        ----------
        input_data : numpy.ndarray
            Ground-truth data, shape ``(H, W)`` or ``(T,)``.
        recon_data : numpy.ndarray
            Model reconstruction, same shape as *input_data*.
        title : str
            Figure super-title.
        """
        if recon_data.ndim == 2:
            fig, axs = plt.subplots(1, 2, figsize=(8, 4), sharex="all", sharey="all")
            axs[0].imshow(input_data, cmap='viridis', origin='lower', aspect='auto')
            axs[0].set_axis_off()
            axs[1].imshow(recon_data, cmap='viridis', origin='lower', aspect='auto')
            axs[1].set_axis_off()
            axs[0].set_title('Input')
            axs[1].set_title('Reconstruction')
        else:
            fig, axs = plt.subplots(figsize=(8, 4))
            axs.plot(input_data, label="Input")
            axs.plot(recon_data, label="Reconstruction")
            axs.set_xlabel('Time')
            axs.legend()
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(self.drawing_path / "reconstruction.png")
        plt.close(fig)
