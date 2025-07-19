"""PyTorch Lightning trainer for autoencoder models."""

import warnings
from typing import Any, Callable, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


class LightningTrainer(pl.LightningModule):
    """PyTorch Lightning wrapper for autoencoder models.

    This class provides a simple interface to train any autoencoder model
    from your models/ directory with built-in best practices.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        warmup_epochs: int = 5,
        max_epochs: int = 100,
        loss_fn: Optional[Callable] = None,
        scheduler_type: str = "cosine",
        compile_model: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the Lightning trainer.

        Parameters
        ----------
        model : torch.nn.Module
            The autoencoder model to train.
        learning_rate : float, optional
            Learning rate for optimizer, by default 1e-4.
        weight_decay : float, optional
            Weight decay for regularization, by default 1e-5.
        warmup_epochs : int, optional
            Number of warmup epochs, by default 5.
        max_epochs : int, optional
            Total training epochs (needed for scheduler), by default 100.
        loss_fn : Optional[Callable], optional
            Custom loss function (defaults to MSE), by default None.
        scheduler_type : str, optional
            Type of LR scheduler ('cosine', 'linear', 'none'),
            by default "cosine".
        compile_model : bool, optional
            Whether to compile model with torch.compile (PyTorch 2.0+),
            by default False.
        **kwargs
            Additional hyperparameters saved to hparams.
        """
        super().__init__()

        # Validate scheduler configuration
        self._validate_scheduler_config(
            warmup_epochs, max_epochs, scheduler_type
        )

        # Save hyperparameters
        self.save_hyperparameters(ignore=["model", "loss_fn"])

        # Model setup
        self.model = model
        if compile_model:
            try:
                self.model = torch.compile(model)
            except Exception as e:
                warnings.warn(f"Model compilation failed: {e}")

        # Loss function
        self.loss_fn = loss_fn or F.mse_loss

        # Store training config
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.scheduler_type = scheduler_type

    def _validate_scheduler_config(
        self, warmup_epochs: int, max_epochs: int, scheduler_type: str
    ) -> None:
        """Validate scheduler configuration to prevent runtime errors.

        Parameters
        ----------
        warmup_epochs : int
            Number of warmup epochs.
        max_epochs : int
            Total training epochs.
        scheduler_type : str
            Scheduler type.

        Raises
        ------
        ValueError
            If configuration would cause scheduler errors.
        """
        if scheduler_type in ["cosine", "linear"]:
            if max_epochs <= 0:
                raise ValueError(f"max_epochs must be > 0, got {max_epochs}")

            if scheduler_type == "cosine" and warmup_epochs >= max_epochs:
                warnings.warn(
                    f"warmup_epochs ({warmup_epochs}) >= max_epochs "
                    f"({max_epochs}). Setting warmup_epochs to "
                    f"{max(0, max_epochs - 1)}"
                )
                # Don't modify the values here, just warn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output.
        """
        return self.model(x)

    def compute_loss(
        self, batch: Any, batch_idx: int, prefix: str = ""
    ) -> dict[str, torch.Tensor]:
        """Compute loss for a batch.

        Override this method for custom loss computation.

        Parameters
        ----------
        batch : Any
            Input batch (can be tensor or dict).
        batch_idx : int
            Batch index.
        prefix : str, optional
            Prefix for logging (e.g., "train_", "val_"), by default "".

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary with 'loss' key and optional additional metrics.
        """
        # Handle different batch formats
        if isinstance(batch, dict):
            # Assume input/target are in the batch dict
            inputs = batch.get("input", batch.get("x", batch.get("data")))
            targets = batch.get("target", batch.get("y", inputs))
        elif isinstance(batch, (list, tuple)) and len(batch) == 2:
            inputs, targets = batch
        else:
            # Single tensor - autoencoder case where input = target
            inputs = targets = batch

        # Forward pass
        outputs = self.model(inputs)

        # Handle model outputs (could be tensor or dict)
        if isinstance(outputs, dict):
            reconstructed = outputs.get(
                "reconstructed", outputs.get("output", outputs.get("x_hat"))
            )
            # Extract additional outputs for logging
            additional_losses = {
                k: v
                for k, v in outputs.items()
                if k.endswith("_loss") and isinstance(v, torch.Tensor)
            }
        else:
            reconstructed = outputs
            additional_losses = {}

        # Main reconstruction loss
        recon_loss = self.loss_fn(reconstructed, targets)

        # Total loss (reconstruction + any additional losses)
        total_loss = recon_loss
        for loss_name, loss_value in additional_losses.items():
            total_loss += loss_value

        # Prepare metrics for logging
        metrics = {
            f"{prefix}loss": total_loss,
            f"{prefix}recon_loss": recon_loss,
        }

        # Add additional losses to metrics
        for loss_name, loss_value in additional_losses.items():
            metrics[f"{prefix}{loss_name}"] = loss_value

        return metrics

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Training step.

        Parameters
        ----------
        batch : Any
            Training batch.
        batch_idx : int
            Batch index.

        Returns
        -------
        torch.Tensor
            Training loss.
        """
        metrics = self.compute_loss(batch, batch_idx, prefix="train_")

        # Log metrics
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True)

        return metrics["train_loss"]

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Validation step.

        Parameters
        ----------
        batch : Any
            Validation batch.
        batch_idx : int
            Batch index.

        Returns
        -------
        torch.Tensor
            Validation loss.
        """
        metrics = self.compute_loss(batch, batch_idx, prefix="val_")

        # Log metrics
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

        return metrics["val_loss"]

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Test step.

        Parameters
        ----------
        batch : Any
            Test batch.
        batch_idx : int
            Batch index.

        Returns
        -------
        torch.Tensor
            Test loss.
        """
        metrics = self.compute_loss(batch, batch_idx, prefix="test_")

        # Log metrics
        self.log_dict(metrics, on_step=False, on_epoch=True)

        return metrics["test_loss"]

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and learning rate scheduler.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing optimizer and scheduler configuration.
        """
        # Optimizer
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        if self.scheduler_type == "none":
            return optimizer

        # Learning rate scheduler
        if self.scheduler_type == "cosine":
            # Ensure T_max is always > 0
            cosine_epochs = max(1, self.max_epochs - self.warmup_epochs)

            if self.warmup_epochs > 0:
                # Cosine annealing with warmup
                warmup_scheduler = LinearLR(
                    optimizer,
                    start_factor=0.1,
                    end_factor=1.0,
                    total_iters=self.warmup_epochs,
                )
                cosine_scheduler = CosineAnnealingLR(
                    optimizer, T_max=cosine_epochs
                )
                scheduler = SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[self.warmup_epochs],
                )
            else:
                # Just cosine annealing without warmup
                scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, self.max_epochs),  # Ensure T_max > 0
                )
        elif self.scheduler_type == "linear":
            scheduler = LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=max(1, self.max_epochs),  # Ensure total_iters > 0
            )
        else:
            raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def on_train_epoch_end(self) -> None:
        """Called at the end of each training epoch."""
        # Log learning rate
        current_lr = self.optimizers().param_groups[0]["lr"]
        self.log("learning_rate", current_lr, on_epoch=True)


class MultimodalLightningTrainer(LightningTrainer):
    """Extended trainer for multimodal models with multiple loss components."""

    def __init__(
        self,
        model: torch.nn.Module,
        loss_weights: Optional[dict[str, float]] = None,
        **kwargs,
    ) -> None:
        """Initialize multimodal trainer.

        Parameters
        ----------
        model : torch.nn.Module
            The multimodal model to train.
        loss_weights : Optional[Dict[str, float]], optional
            Dictionary of loss component weights, by default None.
        **kwargs
            Arguments passed to parent class.
        """
        super().__init__(model, **kwargs)
        self.loss_weights = loss_weights or {}

    def compute_loss(
        self, batch: Any, batch_idx: int, prefix: str = ""
    ) -> dict[str, torch.Tensor]:
        """Compute multimodal loss with weighted components.

        Parameters
        ----------
        batch : Any
            Input batch.
        batch_idx : int
            Batch index.
        prefix : str, optional
            Prefix for logging, by default "".

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing loss components and total loss.
        """
        # Get model outputs
        outputs = self.model(batch)

        if not isinstance(outputs, dict):
            # Fallback to parent implementation
            return super().compute_loss(batch, batch_idx, prefix)

        # Extract losses from model outputs
        total_loss = 0
        metrics = {}

        for key, value in outputs.items():
            if key.endswith("_loss") and isinstance(value, torch.Tensor):
                weight = self.loss_weights.get(key, 1.0)
                weighted_loss = weight * value
                total_loss += weighted_loss

                # Log both weighted and unweighted losses
                metrics[f"{prefix}{key}"] = value
                metrics[f"{prefix}weighted_{key}"] = weighted_loss

        metrics[f"{prefix}loss"] = total_loss
        return metrics


def train_model(
    model: torch.nn.Module,
    train_dataloader,
    val_dataloader=None,
    max_epochs: int = 100,
    gpus: int = 1,
    precision: str = "16-mixed",
    logger_type: str = "tensorboard",
    project_name: str = "autoencoder-training",
    experiment_name: Optional[str] = None,
    log_dir: str = "./logs",
    **trainer_kwargs,
):
    """Convenience function to train a model with sensible defaults.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train.
    train_dataloader
        Training data loader.
    val_dataloader, optional
        Validation data loader, by default None.
    max_epochs : int, optional
        Number of epochs, by default 100.
    gpus : int, optional
        Number of GPUs to use, by default 1.
    precision : str, optional
        Training precision ("32", "16-mixed", "bf16-mixed"),
        by default "16-mixed".
    logger_type : str, optional
        Logger type ("tensorboard", "csv", "none"),
        by default "tensorboard".
    project_name : str, optional
        Project name for logging, by default "autoencoder-training".
    experiment_name : Optional[str], optional
        Experiment name, by default None.
    log_dir : str, optional
        Directory for logs, by default "./logs".
    **trainer_kwargs
        Additional arguments for LightningTrainer.

    Returns
    -------
    tuple
        Tuple containing (lightning_model, trainer).
    """
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import (
        EarlyStopping,
        LearningRateMonitor,
        ModelCheckpoint,
    )

    # Create Lightning module
    lightning_model = LightningTrainer(
        model=model, max_epochs=max_epochs, **trainer_kwargs
    )

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            monitor="val_loss" if val_dataloader else "train_loss",
            mode="min",
            save_top_k=1,
            filename=(
                "best-{epoch}-{val_loss:.4f}"
                if val_dataloader
                else "best-{epoch}-{train_loss:.4f}"
            ),
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    if val_dataloader:
        callbacks.append(
            EarlyStopping(monitor="val_loss", patience=10, mode="min")
        )

    # Configure logger
    logger = None
    if logger_type == "tensorboard":
        try:
            from pytorch_lightning.loggers import TensorBoardLogger

            logger = TensorBoardLogger(
                save_dir=log_dir, name=project_name, version=experiment_name
            )
        except ImportError:
            warnings.warn(
                "TensorBoard not available. "
                "Install with: pip install tensorboard"
            )
    elif logger_type == "csv":
        try:
            from pytorch_lightning.loggers import CSVLogger

            logger = CSVLogger(
                save_dir=log_dir, name=project_name, version=experiment_name
            )
        except ImportError:
            warnings.warn("CSV logger not available.")
    elif logger_type == "none":
        logger = False
    else:
        warnings.warn(f"Unknown logger type: {logger_type}. Using no logger.")
        logger = False

    # Trainer
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if gpus > 0 else "cpu",
        devices=gpus if gpus > 0 else 1,  # Use 1 CPU core when no GPU
        precision=precision,
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=True,
        log_every_n_steps=50,
    )

    # Train
    trainer.fit(lightning_model, train_dataloader, val_dataloader)

    return lightning_model, trainer
