import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tokamak_foundation_model.utils.distributed import DistributedManager
from tokamak_foundation_model.utils.drawing import DrawerProtocol, NullDrawer
from torchmetrics import Metric
from tokamak_foundation_model.utils.tracking import Tracker

logger = logging.getLogger(__name__)

class MultimodalTrainer:
    def __init__(
            self,
            model: nn.Module,
            optimizer: optim.Optimizer,
            loss_fn: nn.Module,
            device: torch.device,
            epochs: int,
            checkpoint_path: str | Path = "checkpoint.pth"
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.epochs = epochs
        self.checkpoint_path = checkpoint_path

    def _train_epoch(self, dataloader: DataLoader):
        self.model.train()
        total_loss = 0
        n_batches = len(dataloader)  # type: ignore[arg-type]
        for batch_idx, batch in enumerate(dataloader):
            inputs = batch['inputs']
            targets = batch['targets']
            inputs = {
                k: v.to(self.device) if isinstance(v, torch.Tensor)
                else v for k, v in inputs.items()}
            targets = {
                k: v.to(self.device) if isinstance(v, torch.Tensor)
                else v for k, v in targets.items()}

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{n_batches}, Loss: {loss.item():.4f}")
        return total_loss / n_batches

    def _validate_epoch(self, dataloader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0
        n_batches = len(dataloader)  # type: ignore[arg-type]
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch["inputs"]
                targets = batch["targets"]
                inputs = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in inputs.items()
                }
                targets = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in targets.items()
                }

                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()
        return total_loss / n_batches

    def train(
            self,
            train_dataloader: DataLoader,
            val_dataloader: DataLoader | None = None
    ):
        best_val_loss = float("inf")
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            train_loss = self._train_epoch(train_dataloader)
            print(f"  Training Loss: {train_loss:.4f}")

            if val_dataloader:
                val_loss = self._validate_epoch(val_dataloader)
                print(f"  Validation Loss: {val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), self.checkpoint_path)
                    print("  Model checkpoint saved.")
            else:
                torch.save(self.model.state_dict(), self.checkpoint_path)
                print("  Model checkpoint saved.")
        print("Training complete.")

    def load_checkpoint(self, checkpoint_path=None):
        path = checkpoint_path if checkpoint_path else self.checkpoint_path
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(
                path, map_location=self.device))
            print(f"Model loaded from checkpoint: {path}")
        else:
            print(f"No checkpoint found at: {path}")


class UnimodalTrainer:

    def __init__(
            self,
            epochs: int,
            model: nn.Module,
            loss_fn: nn.Module,
            optimizer: optim.Optimizer,
            scheduler: optim.lr_scheduler.LRScheduler | None = None,
            distributed_manager: DistributedManager | None = None,
            tracker: Tracker | None = None,
            drawer: DrawerProtocol | None = None,
            metrics: list[Metric] | None = None,
            checkpoint_path: str | Path = "checkpoint.pth",
            log_interval: int = 1,
            grad_clip: float = 1.0,
    ):
        self.epochs = epochs
        self.log_interval = log_interval
        self.grad_clip = grad_clip

        # Key
        self.modality_key = ""

        # Model
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Distributed
        self.dm = distributed_manager or DistributedManager()

        # Logging
        self.tracker = tracker or Tracker(rank=self.dm.rank)
        self.drawer: DrawerProtocol = drawer or NullDrawer()
        self.metrics: list[Metric] = metrics if metrics else []

        # Paths
        self.checkpoint_path: Path | None = (
            Path(checkpoint_path) if checkpoint_path else None
        )
        self.best_checkpoint_path: Path | None = (
            self.checkpoint_path.with_name(
                self.checkpoint_path.stem + "_best" + self.checkpoint_path.suffix
            ) if self.checkpoint_path else None
        )

    def _train_step(self, batch: dict):
        data = batch[self.modality_key].to(self.dm.device)
        valid_lengths = batch.get(f"{self.modality_key}_valid")
        if valid_lengths is not None:
            valid_lengths = valid_lengths.to(self.dm.device)
        self.optimizer.zero_grad()
        output = self.model(data)
        if isinstance(output, tuple):
            output = output[0]
        loss = self.loss_fn(output, data, valid_lengths)
        loss.backward()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        return {"loss": loss}

    @torch.inference_mode()
    def _validate_step(self, batch: dict):
        data = batch[self.modality_key].to(self.dm.device)
        valid_lengths = batch.get(f"{self.modality_key}_valid")
        if valid_lengths is not None:
            valid_lengths = valid_lengths.to(self.dm.device)
        output = self.model(data)
        if isinstance(output, tuple):
            output = output[0]
        loss = self.loss_fn(output, data, valid_lengths)
        for metric in self.metrics:
            metric.update(output, data)
        return {"loss": loss}

    def _train_epoch(self, dataloader: DataLoader):
        self.model.train()
        for batch in dataloader:
            self._train_step(batch)

    def _validate_epoch(self, dataloader: DataLoader):
        self.model.eval()
        for batch in dataloader:
            self._validate_step(batch)

        for metric in self.metrics:
            value = metric.compute().item()
            self.tracker.metrics["validate"]["value"][metric.name] = value
            self.tracker.metrics["validate"]["mean"][metric.name].update(value)
            metric.reset()

    def _log_train(self, epoch: int):
        train_mean = self.tracker.metrics["train"]["mean"]["loss"]()
        logger.info(
            f"Epoch {epoch + 1}/{self.epochs}, Train Loss: {train_mean:.4f}"
        )

    def _log_validate(self, epoch: int):
        val_mean = self.tracker.metrics["validate"]["mean"]["loss"]()
        text = [f"Epoch {epoch + 1}/{self.epochs}, Val Loss: {val_mean:.4f}"]
        for key in self.tracker.metrics["validate"]["value"]:
            if key != "loss":
                val = self.tracker.metrics["validate"]["mean"][key]()
                text.append(f"{key}: {val:.4f}")
        logger.info(", ".join(text))

    def _save_checkpoint(self, epoch: int):
        if not self.dm.is_main or self.checkpoint_path is None:
            return
        raw_model = self.dm.unwrap(self.model)
        torch.save(
            {
                "model_state_dict": raw_model.state_dict(), # type: ignore[union-attr]
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": (
                    self.scheduler.state_dict() if self.scheduler else None
                ),
                "tracker_state_dict": self.tracker.state_dict(),
                "epoch": epoch,
            },
            self.checkpoint_path,
        )

    def _save_best(self):
        if not self.dm.is_main or self.best_checkpoint_path is None:
            return
        if self.tracker.is_best("validate", "loss"):
            raw_model = self.dm.unwrap(self.model)
            torch.save(raw_model.state_dict(), self.best_checkpoint_path)
            logger.info("Best model checkpoint saved!")

    def fit(
            self,
            train_dataloader: DataLoader,
            val_dataloader: DataLoader | None = None,
            modality_key: str | None = None,
            train_sampler=None,
    ):
        if modality_key is None:
            raise ValueError("modality_key is required for unimodal training")
        self.modality_key = modality_key
        logger.info(f"Training modality: {self.modality_key}")

        # Set up distributed training
        self.model = self.dm.wrap(self.model)

        for metric in self.metrics:
            metric.to(self.dm.device)

        n_train = len(train_dataloader)  # type: ignore[arg-type]

        # Set up tracking
        track_train = self.tracker.track("train", n_train)
        self._train_step = track_train(self._train_step)  # type: ignore
        log_train = self.tracker.log("train", "mean")
        self._log_train = log_train(self._log_train)  # type: ignore
        if val_dataloader is not None:
            n_val = len(val_dataloader)  # type: ignore[arg-type]
            track_val = self.tracker.track("validate", n_val)
            self._validate_step = track_val(self._validate_step)  # type: ignore
            log_val = self.tracker.log("validate", "mean")
            self._log_validate = log_val(self._log_validate)  # type: ignore

        drawing_path = self.checkpoint_path.parent / "plots" # type: ignore
        self.drawer.setup(train_dataloader, drawing_path, modality_key, val_dataloader)

        # Training loop
        for epoch in range(self.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            self._train_epoch(train_dataloader)
            self._log_train(epoch)
            self._save_checkpoint(epoch)
            self.dm.barrier()

            if val_dataloader is not None:
                self._validate_epoch(val_dataloader)
                self._log_validate(epoch)
                self._save_best()
                self.dm.barrier()

            if (epoch + 1) % self.log_interval == 0 and self.dm.is_main:
                val_loss = (
                    self.tracker.metrics["validate"]["mean"]["loss"]()) \
                    if val_dataloader is not None else None
                train_loss = self.tracker.metrics["train"]["mean"]["loss"]()
                self.drawer(
                    model=self.dm.unwrap(self.model), # type: ignore
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                )

            if self.scheduler:
                self.scheduler.step()

            self.tracker.step += 1
            self.tracker.reset_epoch()

        logger.info("Training complete.")

    def load_checkpoint(self, checkpoint_path=None):
        path = checkpoint_path or self.checkpoint_path
        if path is None or not os.path.exists(path):
            logger.info(f"No checkpoint found at: {path}")
            return
        checkpoint = torch.load(
            path, map_location=self.dm.device, weights_only=False
        )
        raw_model = self.dm.unwrap(self.model)
        raw_model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("tracker_state_dict"):
            self.tracker.load_state_dict(checkpoint["tracker_state_dict"])
        logger.info(
            f"Resumed from checkpoint: {path} "
            f"(epoch {checkpoint.get('epoch', '?')})")


class Stage3Trainer:
    """Trainer for Stage 3 multimodal prediction.

    Handles batches with three parts: inputs, targets (token-space),
    and observations (observation-space).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss_fn: nn.Module,
        device: torch.device,
        epochs: int,
        checkpoint_path: str | Path = "checkpoint.pth",
        scheduler: optim.lr_scheduler.LRScheduler | None = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.epochs = epochs
        self.checkpoint_path = Path(checkpoint_path)
        self.scheduler = scheduler

    def _move_dict(self, d: dict) -> dict:
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in d.items()
        }

    def _train_epoch(self, dataloader: DataLoader):
        self.model.train()
        total_loss = 0.0
        n_batches = len(dataloader)
        for batch_idx, batch in enumerate(dataloader):
            inputs = self._move_dict(batch["inputs"])
            targets = self._move_dict(batch["targets"])
            observations = self._move_dict(batch["observations"])

            self.optimizer.zero_grad()
            predicted_tokens = self.model(inputs)
            loss = self.loss_fn(predicted_tokens, targets, observations)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            if batch_idx % 10 == 0:
                logger.info(
                    f"  Batch {batch_idx}/{n_batches}, Loss: {loss.item():.4f}"
                )
        return total_loss / max(n_batches, 1)

    @torch.inference_mode()
    def _validate_epoch(self, dataloader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = len(dataloader)
        for batch in dataloader:
            inputs = self._move_dict(batch["inputs"])
            targets = self._move_dict(batch["targets"])
            observations = self._move_dict(batch["observations"])

            predicted_tokens = self.model(inputs)
            loss = self.loss_fn(predicted_tokens, targets, observations)
            total_loss += loss.item()
        return total_loss / max(n_batches, 1)

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None = None,
    ):
        best_val_loss = float("inf")
        for epoch in range(self.epochs):
            logger.info(f"Epoch {epoch + 1}/{self.epochs}")
            train_loss = self._train_epoch(train_dataloader)
            logger.info(f"  Train Loss: {train_loss:.4f}")

            if val_dataloader is not None:
                val_loss = self._validate_epoch(val_dataloader)
                logger.info(f"  Val Loss: {val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_checkpoint(epoch, val_loss)
                    logger.info("  Best checkpoint saved.")
            else:
                self._save_checkpoint(epoch, train_loss)

            if self.scheduler is not None:
                self.scheduler.step()

        logger.info("Stage 3 training complete.")

    def _save_checkpoint(self, epoch: int, loss: float):
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": (
                    self.scheduler.state_dict() if self.scheduler else None
                ),
                "epoch": epoch,
                "loss": loss,
            },
            self.checkpoint_path,
        )

    def load_checkpoint(self, checkpoint_path=None):
        path = checkpoint_path or self.checkpoint_path
        if not os.path.exists(path):
            logger.info(f"No checkpoint found at: {path}")
            return
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if self.scheduler and ckpt.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        logger.info(
            f"Resumed from checkpoint: {path} (epoch {ckpt.get('epoch', '?')})"
        )
