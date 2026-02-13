import logging
import math
import os
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

class MultimodalTrainer:
    def __init__(self, 
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
                print(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
        return total_loss / len(dataloader)

    def _validate_epoch(self, dataloader: DataLoader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items() if k != 'target'}
                targets = batch['target'].to(self.device).float().unsqueeze(1)

                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()
        return total_loss / len(dataloader)

    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader = None):
        best_val_loss = float('inf')
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
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            print(f"Model loaded from checkpoint: {path}")
        else:
            print(f"No checkpoint found at: {path}")


class UnimodalTrainer:
    def __init__(self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss_fn: nn.Module,
        device: torch.device,
        epochs: int,
        log_interval: int | None = None,
        drawer: object | None = None,
        checkpoint_path: str | Path = "checkpoint.pth",
        ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.epochs = epochs
        self.checkpoint_path = checkpoint_path
        self.log_interval = log_interval
        self.drawer = drawer

        p = Path(checkpoint_path)
        self.best_checkpoint_path = p.with_name(p.stem + "_best" + p.suffix)

    def _log_epoch(self, 
        epoch: int, 
        train_loss: float, 
        val_loss: float = 0,
        ):
        logger.info(f"Epoch {epoch+1}/{self.epochs}," +
                    f"Training Loss: {train_loss:.4f}," +
                    f"Validation Loss: {val_loss:.4f}"
                    )
        
        if self.drawer:
            self.drawer(self.model, epoch, train_loss, val_loss)

    def _train_epoch(self, 
        dataloader: DataLoader, 
        modality_key: str,
        ):
        self.model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            data = batch[modality_key].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.loss_fn(outputs, data)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def _validate_epoch(self, 
        dataloader: DataLoader, 
        modality_key: str,
        ):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                data = batch[modality_key].to(self.device)
                outputs = self.model(data)
                loss = self.loss_fn(outputs, data)
                total_loss += loss.item()
        return total_loss / len(dataloader)

    def train(self,
        train_dataloader: DataLoader, 
        val_dataloader: DataLoader = None,
        modality_key: str = 'dalpha',
        ):

        # Setup Training Loop
        self._current_epoch = 0
        train_loss, val_loss = 0, 0
        best_val_loss = float('inf')
        if self.drawer:
            self.drawing_path = Path(self.checkpoint_path).parent / "plots"
            self.drawer.setup(train_dataloader, self.drawing_path, modality_key)

        # Train
        for epoch in range(self.epochs):
            self._current_epoch = epoch

            logger.info(f"Epoch {epoch+1}/{self.epochs}")
            train_loss = self._train_epoch(train_dataloader, modality_key)
            logger.info(f"  Training Loss: {train_loss:.4f}")

            torch.save(self.model.state_dict(), self.checkpoint_path)
            
            # Validation
            if val_dataloader:
                val_loss = self._validate_epoch(val_dataloader, modality_key)
                logger.info(f"  Validation Loss: {val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), self.best_checkpoint_path)
                    logger.info(f"  Best validation loss: {best_val_loss:.4f}, best model checkpoint saved!")

            # Logging
            if self.log_interval is not None:
                if epoch % self.log_interval == 0:
                    self._log_epoch(epoch, train_loss, val_loss)

        logger.info("Training complete.")
