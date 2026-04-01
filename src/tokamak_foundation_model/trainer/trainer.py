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
        for batch_idx, batch in enumerate(dataloader):
            inputs = batch['inputs']
            targets = batch['targets']
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            targets = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in targets.items()}

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(dataloader)},"
                      f" Loss: {loss.item():.4f}")
        return total_loss / len(dataloader)

    def _validate_epoch(self, dataloader: DataLoader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
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
        return total_loss / len(dataloader)

    def train(
            self,
            train_dataloader: DataLoader,
            val_dataloader: DataLoader = None
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
                path, map_location=self.device, weights_only=False))
            print(f"Model loaded from checkpoint: {path}")
        else:
            print(f"No checkpoint found at: {path}")


class UnimodalTrainer:
    def __init__(self,
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
        ):
        self.epochs = epochs
        self.log_interval = log_interval

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
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.best_checkpoint_path = self.checkpoint_path.with_name( # type: ignore
            self.checkpoint_path.stem + "_best" + self.checkpoint_path.suffix # type: ignore
        )


    def _train_step(self, batch: dict):
        data = batch[self.modality_key].to(self.dm.device)
        self.optimizer.zero_grad()
        output = self.model(data)
        if isinstance(output, tuple):
            output = output[0]
        loss = self.loss_fn(output, data)
        loss.backward()
        self.optimizer.step()
        return {"loss": loss}

    @torch.inference_mode()
    def _validate_step(self, batch: dict):
        data = batch[self.modality_key].to(self.dm.device)
        output = self.model(data)
        if isinstance(output, tuple):
            output = output[0]
        loss = self.loss_fn(output, data)
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
        logger.info(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_mean:.4f}")

    def _log_validate(self, epoch: int):
        val_mean = self.tracker.metrics["validate"]["mean"]["loss"]()
        text = [f"Epoch {epoch+1}/{self.epochs}, Val Loss: {val_mean:.4f}"]
        for key in self.tracker.metrics["validate"]["value"]:
            if key != "loss":
                val = self.tracker.metrics["validate"]["mean"][key]()
                text.append(f"{key}: {val:.4f}")
        logger.info(", ".join(text))

    def _save_checkpoint(self, epoch: int):
        if not self.dm.is_main:
            return
        raw_model = self.dm.unwrap(self.model)
        torch.save(
            {
                "model_state_dict": raw_model.state_dict(), # type: ignore
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": (
                    self.scheduler.state_dict() if self.scheduler else None
                ),
                "tracker_state_dict": self.tracker.state_dict(),
                "epoch": epoch,
            },
            self.checkpoint_path, # type: ignore
        )

    def _save_best(self):
        if not self.dm.is_main:
            return
        if self.tracker.is_best("validate", "loss"):
            raw_model = self.dm.unwrap(self.model)
            torch.save(raw_model.state_dict(), self.best_checkpoint_path) # type: ignore
            logger.info("Best model checkpoint saved!")

    def fit(self,
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

        n_train = len(train_dataloader)

        # Set up tracking
        self._train_step = self.tracker.track("train", n_train)(self._train_step)
        self._log_train = self.tracker.log("train", "mean")(self._log_train)
        if val_dataloader is not None:
            n_val = len(val_dataloader)
            self._validate_step = self.tracker.track("validate", n_val)(self._validate_step)
            self._log_validate = self.tracker.log("validate", "mean")(self._log_validate)

        drawing_path = self.checkpoint_path.parent / "plots" # type: ignore
        self.drawer.setup(train_dataloader, drawing_path, modality_key)

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
                val_loss = self.tracker.metrics["validate"]["mean"]["loss"]() if val_dataloader is not None else None
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
            self.tracker._progress["train"]["completed"] = 0
            if val_dataloader is not None:
                self.tracker._progress["validate"]["completed"] = 0
            for label in self.tracker.metrics:
                for m in self.tracker.metrics[label]["mean"].values():
                    m.reset()

        logger.info("Training complete.")

    def load_checkpoint(self, checkpoint_path=None):
        path = checkpoint_path or self.checkpoint_path
        if path is None or not os.path.exists(path):
            logger.info(f"No checkpoint found at: {path}")
            return
        checkpoint = torch.load(path, map_location=self.dm.device, weights_only=False)
        raw_model = self.dm.unwrap(self.model)
        raw_model.load_state_dict(checkpoint["model_state_dict"]) # type: ignore
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("tracker_state_dict"):
            self.tracker.load_state_dict(checkpoint["tracker_state_dict"])
        logger.info(f"Resumed from checkpoint: {path} (epoch {checkpoint.get('epoch', '?')})")


class DiffusionUnimodalTrainer(UnimodalTrainer):
    """Trainer for diffusion-decoder autoencoders.

    Training: the model returns ``(reconstruction, diffusion_loss)`` and
    the loss is backpropagated directly (no external loss_fn).
    Validation: the model runs multi-step generation in eval mode and
    the external loss_fn computes reconstruction quality (e.g. L1).
    """

    def __init__(self, *args, grad_clip: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.grad_clip = grad_clip

    def _train_step(self, batch: dict):
        data = batch[self.modality_key].to(self.dm.device)
        self.optimizer.zero_grad()
        reconstruction, diffusion_loss = self.model(data)
        diffusion_loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.dm.unwrap(self.model).parameters(),
                max_norm=self.grad_clip,
            )
        self.optimizer.step()
        return {"loss": diffusion_loss}

    @torch.inference_mode()
    def _validate_step(self, batch: dict):
        data = batch[self.modality_key].to(self.dm.device)
        output = self.model(data)  # eval mode → multi-step generation
        if isinstance(output, tuple):
            output = output[0]
        loss = self.loss_fn(output, data)
        for metric in self.metrics:
            metric.update(output, data)
        return {"loss": loss}


class GANUnimodalTrainer(UnimodalTrainer):
    """Trainer for autoencoder + PatchGAN discriminator with R3GAN stabilization.

    Alternates discriminator and generator steps each batch:
      D step: RpGAN loss + R1 gradient penalty (real) + R2 gradient penalty (fake)
      G step: L1 reconstruction + adversarial loss (RpGAN from generator perspective)

    The discriminator operates per-channel: (B, C, F, T) → (B*C, 1, F, T).
    Validation uses only the reconstruction loss (no discriminator).

    Parameters
    ----------
    d_optimizer : torch.optim.Optimizer
        Discriminator optimizer (recommended: Adam(β₁=0, β₂=0.9)).
    adv_weight : float
        Weight for adversarial loss in generator objective (default 0.1).
    gp_gamma : float
        R1 + R2 gradient penalty coefficient γ (default 10.0).
    grad_clip : float
        Max gradient norm for generator (0 = disabled).
    """

    def __init__(
        self,
        *args,
        d_optimizer: optim.Optimizer,
        adv_weight: float = 0.1,
        gp_gamma: float = 10.0,
        grad_clip: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.d_optimizer = d_optimizer
        self.adv_weight = adv_weight
        self.gp_gamma = gp_gamma
        self.grad_clip = grad_clip

    def _train_step(self, batch: dict):
        data = batch[self.modality_key].to(self.dm.device)
        raw_model = self.dm.unwrap(self.model)
        discriminator = raw_model.discriminator
        B, C, F, T = data.shape

        # ── D step ────────────────────────────────────────────────────────
        self.d_optimizer.zero_grad()

        with torch.no_grad():
            fake = self.model(data)
            if isinstance(fake, tuple):
                fake = fake[0]

        # Reshape for per-channel discrimination
        real_for_gp = data.reshape(B * C, 1, F, T).detach().requires_grad_(True)
        fake_for_gp = fake.reshape(B * C, 1, F, T).detach().requires_grad_(True)

        d_real = discriminator(real_for_gp)
        d_fake = discriminator(fake_for_gp)

        # RpGAN discriminator loss: D wants D(real) > D(fake)
        d_rpgan = torch.nn.functional.softplus(d_fake - d_real).mean()

        # R1 gradient penalty (on real data)
        grad_real, = torch.autograd.grad(
            d_real.sum(), real_for_gp, create_graph=True,
        )
        r1 = (self.gp_gamma / 2) * grad_real.square().sum(dim=[1, 2, 3]).mean()

        # R2 gradient penalty (on fake data)
        grad_fake, = torch.autograd.grad(
            d_fake.sum(), fake_for_gp, create_graph=True,
        )
        r2 = (self.gp_gamma / 2) * grad_fake.square().sum(dim=[1, 2, 3]).mean()

        d_loss = d_rpgan + r1 + r2
        d_loss.backward()
        self.d_optimizer.step()

        # ── G step ────────────────────────────────────────────────────────
        self.optimizer.zero_grad()

        fake = self.model(data)  # WITH gradients through generator
        if isinstance(fake, tuple):
            fake = fake[0]

        # L1 reconstruction loss
        recon_loss = self.loss_fn(fake, data)

        # RpGAN generator loss: G wants D(fake) > D(real)
        real_flat = data.reshape(B * C, 1, F, T).detach()
        fake_flat = fake.reshape(B * C, 1, F, T)
        d_real_g = discriminator(real_flat)
        d_fake_g = discriminator(fake_flat)
        g_adv = torch.nn.functional.softplus(d_real_g - d_fake_g).mean()

        g_total = recon_loss + self.adv_weight * g_adv
        g_total.backward()

        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                raw_model.autoencoder.parameters(), max_norm=self.grad_clip,
            )
        self.optimizer.step()

        return {
            "loss": g_total,
            "recon_loss": recon_loss,
            "g_adv": g_adv,
            "d_loss": d_loss,
            "r1": r1,
            "r2": r2,
        }

    def _log_train(self, epoch: int):
        m = self.tracker.metrics["train"]["mean"]
        parts = [f"Epoch {epoch+1}/{self.epochs}"]
        parts.append(f"G: {m['loss']():.4f}")
        if "recon_loss" in m:
            parts.append(f"recon={m['recon_loss']():.4f}")
        if "g_adv" in m:
            parts.append(f"adv={m['g_adv']():.4f}")
        if "d_loss" in m:
            parts.append(f"D: {m['d_loss']():.4f}")
        logger.info(", ".join(parts))

    def _save_checkpoint(self, epoch: int):
        if not self.dm.is_main:
            return
        raw_model = self.dm.unwrap(self.model)
        torch.save(
            {
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "d_optimizer_state_dict": self.d_optimizer.state_dict(),
                "scheduler_state_dict": (
                    self.scheduler.state_dict() if self.scheduler else None
                ),
                "tracker_state_dict": self.tracker.state_dict(),
                "epoch": epoch,
            },
            self.checkpoint_path,
        )

    def load_checkpoint(self, checkpoint_path=None):
        super().load_checkpoint(checkpoint_path)
        path = checkpoint_path or self.checkpoint_path
        if path and os.path.exists(path):
            ckpt = torch.load(path, map_location=self.dm.device, weights_only=False)
            if "d_optimizer_state_dict" in ckpt:
                self.d_optimizer.load_state_dict(ckpt["d_optimizer_state_dict"])