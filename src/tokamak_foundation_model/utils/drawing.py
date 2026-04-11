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
            val_dataloader: Optional[DataLoader] = None,
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
            val_dataloader: Optional[DataLoader] = None,
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
            val_dataloader: Optional[DataLoader] = None,
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
        val_dataloader : DataLoader or None, optional
            Validation dataloader used for the correlation plot.  Falls back
            to the probe sample when ``None``.
        """
        self.drawing_path = Path(drawing_path)
        self.drawing_path.mkdir(parents=True, exist_ok=True)
        self.modality_key = modality_key
        self.val_dataloader = val_dataloader

        dataset = dataloader.dataset
        assert isinstance(dataset, Sized), "Dataset must implement __len__"
        idx = int(torch.randint(len(dataset), (1,)).item())
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
        input_data, recon_data = self._compute_reconstruction(model)
        self._save_reconstruction(input_data, recon_data, epoch, train_loss, val_loss)
        self._save_correlation(model, epoch)

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

    def _compute_reconstruction(
            self,
            model: torch.nn.Module,
    ):
        """Run probe sample through *model* and return ``(input_data, recon_data)``.

        Both arrays are trimmed to the valid length (if available) and cover
        all channels: shape ``(C, ...)``.
        """
        model.eval()
        x = self.probe_sample.unsqueeze(0).to(next(model.parameters()).device)
        output = model(x)
        if isinstance(output, tuple):
            output = output[0]
        output = output[0].cpu()

        input_data = self.probe_sample.numpy()   # [C, ...]
        recon_data = output.numpy()              # [C, ...]

        vl = self.probe_valid_length
        if vl is not None and vl > 0:
            input_data = input_data[..., :vl]
            recon_data = recon_data[..., :vl]

        return input_data, recon_data

    def _save_reconstruction(
            self,
            input_data: np.ndarray,
            recon_data: np.ndarray,
            epoch: int,
            train_loss: float,
            val_loss: Optional[float],
    ):
        """Write ``reconstruction.png``, overwriting any previous version."""
        ch_input = input_data[self.channel]
        ch_recon = recon_data[self.channel]

        title = f"Epoch {epoch + 1} | Train={train_loss:.6f}"
        if val_loss is not None:
            title += f" | Val={val_loss:.6f}"

        if ch_recon.ndim == 3:
            self._plot_video(ch_input, ch_recon, title)
        else:
            self._plot_2d_or_1d(ch_input, ch_recon, title)

    @torch.no_grad()
    def _save_correlation(
            self,
            model: torch.nn.Module,
            epoch: int,
            max_batches: int = 50,
    ):
        """Write ``correlation.png`` — scatter of target vs. reconstruction.

        Iterates over the validation dataloader (up to *max_batches* batches)
        when available, otherwise falls back to the probe sample.  All
        channels are flattened together.  Includes a y=x reference line and
        Pearson r in the title.
        """
        model.eval()
        device = next(model.parameters()).device

        all_targets: list[np.ndarray] = []
        all_recons: list[np.ndarray] = []

        if self.val_dataloader is not None:
            for i, batch in enumerate(self.val_dataloader):
                if i >= max_batches:
                    break
                data = batch[self.modality_key].to(device)
                valid_lengths = batch.get(f"{self.modality_key}_valid")

                output = model(data)
                if isinstance(output, tuple):
                    output = output[0]

                data_np = data.cpu().numpy()    # [B, C, T]
                recon_np = output.cpu().numpy() # [B, C, T]

                if valid_lengths is not None:
                    for b, vl in enumerate(valid_lengths.tolist()):
                        all_targets.append(data_np[b, :, :vl].ravel())
                        all_recons.append(recon_np[b, :, :vl].ravel())
                else:
                    all_targets.append(data_np.ravel())
                    all_recons.append(recon_np.ravel())
        else:
            # Fallback: probe sample only
            inp, rec = self._compute_reconstruction(model)
            all_targets.append(inp.ravel())
            all_recons.append(rec.ravel())

        target = np.concatenate(all_targets)
        recon = np.concatenate(all_recons)

        finite_mask = np.isfinite(target) & np.isfinite(recon)
        n_nan = (~finite_mask).sum()
        if n_nan > 0:
            print(f"WARNING: Correlation plot: {n_nan} non-finite values dropped")
        target_clean = target[finite_mask]
        recon_clean = recon[finite_mask]

        if len(target_clean) > 1 and target_clean.std() > 0 and recon_clean.std() > 0:
            r = float(np.corrcoef(target_clean, recon_clean)[0, 1])
        else:
            r = float('nan')

        if len(target_clean) == 0:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.set_title(f"Epoch {epoch + 1} | No finite values (n={len(target):,})")
            ax.text(0.5, 0.5, "All values non-finite",
                    ha='center', va='center', transform=ax.transAxes)
            fig.savefig(self.drawing_path / "correlation.png")
            plt.close(fig)
            return

        # Subsample for plot readability
        max_pts = 20_000
        if len(target_clean) > max_pts:
            idx = np.random.choice(len(target_clean), max_pts, replace=False)
            target_plot, recon_plot = target_clean[idx], recon_clean[idx]
        else:
            target_plot, recon_plot = target_clean, recon_clean

        vmin = min(target_plot.min(), recon_plot.min())
        vmax = max(target_plot.max(), recon_plot.max())

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(target_plot, recon_plot, s=2, alpha=0.3, color='steelblue')
        ax.plot([vmin, vmax], [vmin, vmax], color='tomato', lw=1.2, label='y=x')
        ax.set_xlabel('Target')
        ax.set_ylabel('Reconstruction')
        ax.set_title(f"Epoch {epoch + 1} | r = {r:.4f}  (n={len(target):,})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.drawing_path / "correlation.png")
        plt.close(fig)

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
