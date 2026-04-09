import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from perceiver_components import PerceiverComponents
from dummy_perceiver_data import create_dummy_dataloaders, DummyTokamakDataset
from deterministic_test import DeterministicTestSignals


class PerceiverTrainer:
    """
    Trainer for Perceiver with Phase 2 training:
    - Reconstruction loss (observations)
    - Latent consistency loss (latent space)

    Parameters
    ----------
    perceiver : PerceiverComponents
        The Perceiver model
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader
        Validation data loader
    device : torch.device
        Device for training
    learning_rate : float
        Initial learning rate
    weight_decay : float
        AdamW weight decay
    checkpoint_dir : Path
        Directory for saving checkpoints
    log_dir : Path
        Directory for tensorboard logs
    loss_weights : dict
        Weights for different loss components
    """

    def __init__(
            self,
            perceiver,
            train_loader,
            val_loader,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            learning_rate=1e-4,
            weight_decay=1e-5,
            checkpoint_dir='checkpoints',
            log_dir='runs',
            loss_weights=None
    ):
        self.perceiver = perceiver.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Optimizer
        self.optimizer = optim.AdamW(
            self.perceiver.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Learning rate scheduler (cosine annealing)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=len(train_loader) * 100,  # 100 epochs
            eta_min=learning_rate * 0.01
        )

        # Loss weights
        if loss_weights is None:
            loss_weights = {
                'reconstruction': 1.0,
                'latent_consistency': 0.5,
                'smoothness': 0.1,
            }
        self.loss_weights = loss_weights

        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Logging
        self.writer = SummaryWriter(log_dir)

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

    def compute_reconstruction_loss(self, predictions, targets):
        """
        Compute reconstruction loss for all modalities.

        Parameters
        ----------
        predictions : dict
            Predicted tokens per modality
        targets : dict
            Target tokens per modality

        Returns
        -------
        tuple
            (total_loss, loss_dict)
        """
        losses = {}
        total_loss = 0

        for modality in predictions.keys():
            loss = nn.functional.mse_loss(
                predictions[modality],
                targets[modality]
            )
            losses[f'recon_{modality}'] = loss.item()
            total_loss += loss

        return total_loss, losses

    def compute_latent_consistency_loss(
            self,
            latent_pred,
            target_tokens,
            actuators_current,
            actuators_future
    ):
        """
        Compute latent consistency loss.

        Note: When encoding targets, we use future actuators as "current"
        since targets represent the future state.
        """
        # Concatenate target tokens
        target_tokens_cat = torch.cat([
            target_tokens['ts'],
            target_tokens['prof'],
            target_tokens['vid'],
        ], dim=1)

        # Encode targets to get "true" future latent
        with torch.no_grad():
            latent_true = self.perceiver.encoder(target_tokens_cat)
            latent_true = self.perceiver.processor(latent_true)

        # Compare predicted and true latent
        loss = nn.functional.mse_loss(latent_pred, latent_true)

        return loss

    def compute_smoothness_loss(self, latent_current, latent_future):
        """
        Encourage smooth latent evolution.

        Prevents drastic jumps in latent space.
        """
        return nn.functional.mse_loss(latent_future, latent_current)

    def train_epoch(self):
        """Train for one epoch."""
        self.perceiver.train()

        epoch_losses = {
            'total': 0,
            'reconstruction': 0,
            'latent_consistency': 0,
            'smoothness': 0,
        }

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch}')

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            input_tokens = batch['input_tokens'].to(self.device)
            actuators_current = batch['actuators_current'].to(self.device)
            actuators_future = batch['actuators_future'].to(self.device)
            target_tokens = {
                k: v.to(self.device) for k, v in batch['target_tokens'].items()
            }

            # Forward pass with both actuator states
            output_tokens, latent_current, latent_future = self.perceiver(
                input_tokens,
                actuators_current,
                actuators_future
            )

            # Compute losses
            loss_recon, recon_dict = self.compute_reconstruction_loss(
                output_tokens, target_tokens
            )

            loss_latent = self.compute_latent_consistency_loss(
                latent_future, target_tokens, actuators_current, actuators_future
            )

            loss_smooth = self.compute_smoothness_loss(
                latent_current, latent_future
            )

            # Total loss
            loss = (
                    self.loss_weights['reconstruction'] * loss_recon +
                    self.loss_weights['latent_consistency'] * loss_latent +
                    self.loss_weights['smoothness'] * loss_smooth
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.perceiver.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            # Logging
            epoch_losses['total'] += loss.item()
            epoch_losses['reconstruction'] += loss_recon.item()
            epoch_losses['latent_consistency'] += loss_latent.item()
            epoch_losses['smoothness'] += loss_smooth.item()

            self.writer.add_scalar('train/loss_total', loss.item(), self.global_step)
            self.writer.add_scalar('train/loss_recon', loss_recon.item(), self.global_step)
            self.writer.add_scalar('train/loss_latent', loss_latent.item(), self.global_step)
            self.writer.add_scalar('train/loss_smooth', loss_smooth.item(), self.global_step)

            # Log actuator statistics
            act_change = (actuators_future - actuators_current).abs().mean().item()
            self.writer.add_scalar('train/actuator_change', act_change, self.global_step)

            self.global_step += 1

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon': f'{loss_recon.item():.4f}',
                'act_Δ': f'{act_change:.4f}',
            })

        # Average epoch losses
        for key in epoch_losses:
            epoch_losses[key] /= len(self.train_loader)

        return epoch_losses

    def validate(self):
        """Validate on validation set."""
        self.perceiver.eval()

        val_losses = {
            'total': 0,
            'reconstruction': 0,
            'latent_consistency': 0,
            'smoothness': 0,
        }

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                input_tokens = batch['input_tokens'].to(self.device)
                actuators_current = batch['actuators_current'].to(self.device)
                actuators_future = batch['actuators_future'].to(self.device)
                target_tokens = {
                    k: v.to(self.device) for k, v in batch['target_tokens'].items()
                }

                # Forward pass
                output_tokens, latent_current, latent_future = self.perceiver(
                    input_tokens,
                    actuators_current,
                    actuators_future
                )

                # Compute losses
                loss_recon, _ = self.compute_reconstruction_loss(
                    output_tokens, target_tokens
                )
                loss_latent = self.compute_latent_consistency_loss(
                    latent_future, target_tokens, actuators_current, actuators_future
                )
                loss_smooth = self.compute_smoothness_loss(
                    latent_current, latent_future
                )

                loss = (
                        self.loss_weights['reconstruction'] * loss_recon +
                        self.loss_weights['latent_consistency'] * loss_latent +
                        self.loss_weights['smoothness'] * loss_smooth
                )

                val_losses['total'] += loss.item()
                val_losses['reconstruction'] += loss_recon.item()
                val_losses['latent_consistency'] += loss_latent.item()
                val_losses['smoothness'] += loss_smooth.item()

        # Average validation losses
        for key in val_losses:
            val_losses[key] /= len(self.val_loader)

        # Log to tensorboard
        for key, value in val_losses.items():
            self.writer.add_scalar(f'val/loss_{key}', value, self.epoch)

        return val_losses

    def save_checkpoint(self, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.perceiver.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
        }

        # Save latest
        torch.save(checkpoint, self.checkpoint_dir / 'checkpoint_latest.pth')

        # Save best
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'checkpoint_best.pth')

        # Save periodic
        if self.epoch % 10 == 0:
            torch.save(checkpoint,
                       self.checkpoint_dir / f'checkpoint_epoch_{self.epoch}.pth')

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.perceiver.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

        print(f"Loaded checkpoint from epoch {self.epoch}")

    def run_deterministic_test(self):
        """Run deterministic test with actuator changes."""
        self.perceiver.eval()

        # Generate test signals
        signals = DeterministicTestSignals.create_test_batch(batch_size=4, d_model=512)

        tokens_ts = DeterministicTestSignals.generate_timeseries_tokens(signals, 50, 512)
        tokens_prof = DeterministicTestSignals.generate_profile_tokens(signals, 10, 512)
        tokens_vid = DeterministicTestSignals.generate_video_tokens(signals, 30, 512)

        all_input_tokens = torch.cat([tokens_ts, tokens_prof, tokens_vid], dim=1).to(self.device)

        # Create actuators with changes
        actuators_current = torch.tensor([sig['actuator'] for sig in signals.values()])
        actuators_current = actuators_current.unsqueeze(1).expand(-1, 32).to(self.device)

        # Future actuators: 50% same, 50% increased by 0.2
        actuators_future = actuators_current.clone()
        actuators_future[::2] += 0.2  # Every other sample increases
        actuators_future = torch.clamp(actuators_future, 0, 1)

        # Forward pass
        with torch.no_grad():
            output_tokens, latent_current, latent_future = self.perceiver(
                all_input_tokens,
                actuators_current,
                actuators_future
            )

        # Generate expected output
        # For samples with increased actuators, amplitude should increase
        expected_output = DeterministicTestSignals.generate_expected_output_tokens(
            signals, dt=0.05, n_tokens_per_modality={'ts': 50, 'prof': 10, 'vid': 30}
        )

        # Visualize
        self._visualize_test_results(
            input_tokens={'ts': tokens_ts, 'prof': tokens_prof, 'vid': tokens_vid},
            output_tokens=output_tokens,
            expected_tokens=expected_output,
            signals=signals,
            actuators_current=actuators_current,
            actuators_future=actuators_future,
            save_path=self.checkpoint_dir / f'test_epoch_{self.epoch}.png'
        )

    def _visualize_test_results(
            self,
            input_tokens,
            output_tokens,
            expected_tokens,
            signals,
            actuators_current=None,
            actuators_future=None,
            save_path=None
    ):
        """
        Visualize test results with optional actuator information.

        Parameters
        ----------
        input_tokens : dict
            Input tokens per modality
        output_tokens : dict
            Output tokens per modality
        expected_tokens : dict
            Expected tokens per modality
        signals : dict
            Signal metadata
        actuators_current : torch.Tensor, optional
            Current actuator values [B, D_act]
        actuators_future : torch.Tensor, optional
            Future actuator values [B, D_act]
        save_path : Path, optional
            Where to save the visualization
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))

        sample_idx = 0
        sig = signals[sample_idx]

        # Time series
        ax = axes[0, 0]
        expected = expected_tokens['ts'][sample_idx, :, 0].cpu().numpy()
        actual = output_tokens['ts'][sample_idx, :, 0].detach().cpu().numpy()
        ax.plot(expected, 'g-', label='Expected', linewidth=2)
        ax.plot(actual, 'b--', label='Actual', linewidth=2)
        ax.set_title(f'Time Series (Epoch {self.epoch})')
        ax.set_xlabel('Token Index')
        ax.set_ylabel('Pulse Presence')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Profile
        ax = axes[0, 1]
        expected = expected_tokens['prof'][sample_idx, :, 0].cpu().numpy()
        actual = output_tokens['prof'][sample_idx, :, 0].detach().cpu().numpy()
        ax.plot(expected, 'g-', label='Expected', linewidth=2)
        ax.plot(actual, 'b--', label='Actual', linewidth=2)
        ax.set_title(f'Profile (Epoch {self.epoch})')
        ax.set_xlabel('Token Index')
        ax.set_ylabel('Profile Height')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Actuator visualization (if provided)
        ax = axes[0, 2]
        if actuators_current is not None and actuators_future is not None:
            act_curr = actuators_current[sample_idx, 0].cpu().item()
            act_fut = actuators_future[sample_idx, 0].cpu().item()

            ax.bar(['Current', 'Future'], [act_curr, act_fut],
                   color=['blue', 'orange'], alpha=0.7)
            ax.set_ylabel('Actuator Value')
            ax.set_title('Actuator States')
            ax.set_ylim([0, 1.2])
            ax.grid(True, alpha=0.3, axis='y')

            # Add delta text
            delta = act_fut - act_curr
            ax.text(0.5, max(act_curr, act_fut) + 0.1,
                    f'Δ = {delta:+.3f}',
                    ha='center', fontsize=12, fontweight='bold')
        else:
            ax.axis('off')
            ax.text(0.5, 0.5, 'No actuator data',
                    ha='center', va='center', fontsize=12)

        # MSE over tokens
        ax = axes[1, 0]
        mse_ts = ((output_tokens['ts'][sample_idx, :, 0].detach().cpu() -
                   expected_tokens['ts'][sample_idx, :, 0].cpu())**2).numpy()
        ax.plot(mse_ts, 'r-', linewidth=2)
        ax.set_title(f'MSE per Token (TS)')
        ax.set_xlabel('Token Index')
        ax.set_ylabel('MSE')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        # Profile MSE
        ax = axes[1, 1]
        mse_prof = ((output_tokens['prof'][sample_idx, :, 0].detach().cpu() -
                     expected_tokens['prof'][sample_idx, :, 0].cpu())**2).numpy()
        ax.plot(mse_prof, 'r-', linewidth=2)
        ax.set_title(f'MSE per Token (Profile)')
        ax.set_xlabel('Token Index')
        ax.set_ylabel('MSE')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        # Overall metrics
        ax = axes[1, 2]
        ax.axis('off')

        mse_ts_total = mse_ts.mean()
        mse_prof_total = mse_prof.mean()

        metrics_text = f"""
        Epoch: {self.epoch}

        MSE Metrics:
        - Time Series: {mse_ts_total:.6f}
        - Profile:     {mse_prof_total:.6f}

        Pulse Info:
        - Start pos:  {sig['pulse_start']:.1f}
        - Expected:   {sig['pulse_start'] + 50:.1f}
        """

        # Add actuator info if available
        if actuators_current is not None and actuators_future is not None:
            act_curr = actuators_current[sample_idx, 0].cpu().item()
            act_fut = actuators_future[sample_idx, 0].cpu().item()
            metrics_text += f"""
        Actuators:
        - Current:    {act_curr:.3f}
        - Future:     {act_fut:.3f}
        - Change:     {act_fut - act_curr:+.3f}
        """

        ax.text(0.1, 0.5, metrics_text, fontsize=10, family='monospace',
                verticalalignment='center')

        plt.tight_layout()

        if save_path is None:
            save_path = self.checkpoint_dir / f'test_epoch_{self.epoch}.png'

        plt.savefig(save_path, dpi=150)
        plt.close()

        print(f"Saved test visualization to: {save_path}")

    def train(self, num_epochs, validate_every=1, test_every=5):
        """
        Main training loop.

        Parameters
        ----------
        num_epochs : int
            Number of epochs to train
        validate_every : int
            Validate every N epochs
        test_every : int
            Run deterministic test every N epochs
        """
        print("=" * 80)
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print("=" * 80)

        for epoch in range(num_epochs):
            self.epoch = epoch

            # Train
            train_losses = self.train_epoch()

            print(f"\nEpoch {epoch} - Train Loss: {train_losses['total']:.6f}")

            # Validate
            if epoch % validate_every == 0:
                val_losses = self.validate()
                print(f"Epoch {epoch} - Val Loss: {val_losses['total']:.6f}")

                # Save best model
                is_best = val_losses['total'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_losses['total']
                    print(f"New best validation loss: {self.best_val_loss:.6f}")

                self.save_checkpoint(is_best=is_best)

            # Deterministic test
            if epoch % test_every == 0:
                print("Running deterministic test...")
                self.run_deterministic_test()

        print("\n" + "=" * 80)
        print("Training complete!")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        print("=" * 80)

        self.writer.close()


def main():
    """Main training script with future actuators."""

    config = {
        'd_model': 512,
        'n_latent_queries': 256,
        'n_actuators': 32,
        'encoder_layers': 2,
        'processor_layers': 4,
        'decoder_layers': 2,
        'dynamics_layers': 3,
        'n_heads': 8,
        'dropout': 0.1,

        'n_train': 8000,
        'n_val': 1000,
        'batch_size': 32,
        'num_workers': 4,

        'num_epochs': 100,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'loss_weights': {
            'reconstruction': 1.0,
            'latent_consistency': 0.5,
            'smoothness': 0.1,
        },

        'checkpoint_dir': 'checkpoints/perceiver_with_future',
        'log_dir': 'runs/perceiver_with_future',
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create dataloaders
    print("Creating datasets...")
    train_loader, val_loader = create_dummy_dataloaders(
        n_train=config['n_train'],
        n_val=config['n_val'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )

    # Test batch to verify actuator changes
    batch = next(iter(train_loader))
    act_change = (batch['actuators_future'] - batch['actuators_current']).abs().mean()
    print(f"Average actuator change in batch: {act_change:.4f}")

    # Create model
    print("Creating Perceiver model with future actuator support...")
    perceiver = PerceiverComponents(
        d_model=config['d_model'],
        n_latent_queries=config['n_latent_queries'],
        n_actuators=config['n_actuators'],
        output_queries_config={'ts': 50, 'prof': 10, 'vid': 30},
        encoder_layers=config['encoder_layers'],
        processor_layers=config['processor_layers'],
        decoder_layers=config['decoder_layers'],
        dynamics_layers=config['dynamics_layers'],
        n_heads=config['n_heads'],
        dropout=config['dropout'],
        dynamics_mode='residual'
    )

    n_params = sum(p.numel() for p in perceiver.parameters())
    print(f"Model parameters: {n_params:,}")

    # Create trainer
    trainer = PerceiverTrainer(
        perceiver=perceiver,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        checkpoint_dir=config['checkpoint_dir'],
        log_dir=config['log_dir'],
        loss_weights=config['loss_weights']
    )

    # Train
    trainer.train(
        num_epochs=config['num_epochs'],
        validate_every=1,
        test_every=5
    )


if __name__ == "__main__":
    main()
