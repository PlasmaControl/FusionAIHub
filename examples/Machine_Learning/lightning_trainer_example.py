"""Examples and tests for the Lightning trainer with autoencoder models."""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Assuming your package structure
from src.faith.train.models.autoencoder import BlockBasedAutoencoder
from src.faith.train.training import (
    LightningTrainer, MultimodalLightningTrainer, train_model)


def create_dummy_dataset(
        batch_size: int = 16,
        num_samples: int = 1000,
        input_shape: tuple = (80, 100, 128)
) -> tuple:
    """Create dummy dataset for testing.

    Parameters
    ----------
    batch_size : int, optional
        Batch size, by default 16.
    num_samples : int, optional
        Number of samples, by default 1000.
    input_shape : tuple, optional
        Shape of input data (C, H, W), by default (80, 100, 128).

    Returns
    -------
    tuple
        Train and validation dataloaders.
    """
    # Create random data
    data = torch.randn(num_samples, *input_shape)

    # Split into train/val
    train_size = int(0.8 * num_samples)
    train_data = data[:train_size]
    val_data = data[train_size:]

    # Create datasets (for autoencoders, input = target)
    train_dataset = TensorDataset(train_data, train_data)
    val_dataset = TensorDataset(val_data, val_data)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    return train_loader, val_loader


def create_multimodal_dataset(
    batch_size: int = 16,
    num_samples: int = 1000
) -> tuple:
    """Create dummy multimodal dataset.

    Parameters
    ----------
    batch_size : int, optional
        Batch size, by default 16.
    num_samples : int, optional
        Number of samples, by default 1000.

    Returns
    -------
    tuple
        Train and validation dataloaders.
    """
    # Create multimodal data
    audio_data = torch.randn(num_samples, 80, 100, 128)
    text_data = torch.randint(0, 1000, (num_samples, 50))  # Token IDs

    # Split into train/val
    train_size = int(0.8 * num_samples)

    train_audio = audio_data[:train_size]
    train_text = text_data[:train_size]
    val_audio = audio_data[train_size:]
    val_text = text_data[train_size:]

    # Create datasets as dictionaries
    train_data = []
    for i in range(len(train_audio)):
        train_data.append({
            'audio': train_audio[i],
            'text': train_text[i],
            'target_audio': train_audio[i],  # Reconstruction target
            'target_text': train_text[i]
        })

    val_data = []
    for i in range(len(val_audio)):
        val_data.append({
            'audio': val_audio[i],
            'text': val_text[i],
            'target_audio': val_audio[i],
            'target_text': val_text[i]
        })

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


class CustomLossAutoencoder(LightningTrainer):
    """Example of custom loss computation for specific autoencoder needs."""

    def __init__(self, model, perceptual_weight: float = 0.1, **kwargs):
        """Initialize with perceptual loss component.

        Parameters
        ----------
        model : torch.nn.Module
            The autoencoder model.
        perceptual_weight : float, optional
            Weight for perceptual loss, by default 0.1.
        **kwargs
            Additional arguments passed to parent.
        """
        super().__init__(model, **kwargs)
        self.perceptual_weight = perceptual_weight

    def compute_loss(self, batch, batch_idx, prefix=""):
        """Compute loss with perceptual component.

        Parameters
        ----------
        batch : Any
            Input batch.
        batch_idx : int
            Batch index.
        prefix : str, optional
            Logging prefix, by default "".

        Returns
        -------
        Dict[str, torch.Tensor]
            Loss components.
        """
        inputs, targets = batch
        outputs = self.model(inputs)

        if isinstance(outputs, tuple):
            reconstructed, latent = outputs
        else:
            reconstructed = outputs

        # MSE reconstruction loss
        recon_loss = F.mse_loss(reconstructed, targets)

        # Simple perceptual loss (L1 in feature space)
        # In practice, you'd use a pre-trained network
        with torch.no_grad():
            # Downsample for "perceptual" comparison
            targets_down = F.avg_pool2d(targets, kernel_size=4, stride=4)
            recon_down = F.avg_pool2d(reconstructed, kernel_size=4, stride=4)

        perceptual_loss = F.l1_loss(recon_down, targets_down)

        # Total loss
        total_loss = recon_loss + self.perceptual_weight * perceptual_loss

        metrics = {
            f'{prefix}loss': total_loss,
            f'{prefix}recon_loss': recon_loss,
            f'{prefix}perceptual_loss': perceptual_loss,
        }

        return metrics


class MultimodalAutoencoder(torch.nn.Module):
    """Example multimodal autoencoder for testing."""

    def __init__(self, audio_channels: int = 80, text_vocab_size: int = 1000):
        """Initialize multimodal autoencoder.

        Parameters
        ----------
        audio_channels : int, optional
            Number of audio channels, by default 80.
        text_vocab_size : int, optional
            Text vocabulary size, by default 1000.
        """
        super().__init__()

        # Audio encoder (simplified)
        self.audio_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(audio_channels, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((8, 8)),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 8 * 8, 256)
        )

        # Text encoder
        self.text_encoder = torch.nn.Sequential(
            torch.nn.Embedding(text_vocab_size, 128),
            torch.nn.LSTM(128, 128, batch_first=True),
        )

        # Shared latent space
        self.fusion = torch.nn.Linear(256 + 128, 256)

        # Decoders
        self.audio_decoder = torch.nn.Sequential(
            torch.nn.Linear(256, 64 * 8 * 8),
            torch.nn.Unflatten(1, (64, 8, 8)),
            torch.nn.ConvTranspose2d(64, audio_channels, 3, padding=1)
        )

        self.text_decoder = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.LSTM(128, 128, batch_first=True),
            torch.nn.Linear(128, text_vocab_size)
        )

    def forward(self, batch):
        """Forward pass returning losses.

        Parameters
        ----------
        batch : dict
            Batch containing 'audio', 'text', 'target_audio', 'target_text'.

        Returns
        -------
        dict
            Dictionary containing losses and outputs.
        """
        audio = batch['audio']
        text = batch['text']
        target_audio = batch['target_audio']
        target_text = batch['target_text']

        # Encode
        audio_features = self.audio_encoder(audio)
        text_lstm_out, _ = self.text_encoder(text)
        text_features = text_lstm_out.mean(dim=1)  # Simple pooling

        # Fuse
        combined = torch.cat([audio_features, text_features], dim=1)
        latent = self.fusion(combined)

        # Decode
        audio_reconstructed = self.audio_decoder(latent)

        # Upsample audio reconstruction to match input size
        audio_reconstructed = F.interpolate(
            audio_reconstructed,
            size=(target_audio.shape[-2], target_audio.shape[-1]),
            mode='bilinear',
            align_corners=False
        )

        text_hidden = self.text_decoder[0](
            latent).unsqueeze(1).repeat(1, text.size(1), 1)
        text_lstm_out, _ = self.text_decoder[1](text_hidden)
        text_reconstructed = self.text_decoder[2](text_lstm_out)

        # Compute losses
        audio_loss = F.mse_loss(audio_reconstructed, target_audio)
        text_loss = F.cross_entropy(
            text_reconstructed.reshape(-1, text_reconstructed.size(-1)),
            target_text.reshape(-1)
        )

        return {
            'audio_reconstructed': audio_reconstructed,
            'text_reconstructed': text_reconstructed,
            'latent': latent,
            'audio_loss': audio_loss,
            'text_loss': text_loss,
        }


def example_basic_training():
    """Basic training example with BlockBasedAutoencoder."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Autoencoder Training")
    print("=" * 60)

    # Create autoencoder
    autoencoder = BlockBasedAutoencoder(input_channels=80)
    print(f"Created autoencoder with "
          f"{autoencoder.parameter_count:,} parameters")

    # Create data
    train_loader, val_loader = create_dummy_dataset(
        batch_size=8,
        num_samples=100,  # Small for quick testing
        input_shape=(80, 100, 128)
    )
    print(f"Created dataset with {len(train_loader)} train batches")

    # Use convenience function with TensorBoard logging
    lightning_model, trainer = train_model(
        model=autoencoder,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        max_epochs=3,
        gpus=0,  # CPU for testing
        precision="32",
        logger_type="tensorboard",  # Use TensorBoard instead of wandb
        project_name="test-autoencoder",
        experiment_name="basic-example",
        learning_rate=1e-3
    )

    print("Training completed!")
    print("Logs saved to: ./logs/test-autoencoder/basic-example/")

    # Test the trained model
    test_input = torch.randn(1, 80, 100, 128)
    lightning_model.eval()
    with torch.no_grad():
        output = lightning_model(test_input)
    print(f"Test output shape: {output.shape}")


def example_custom_trainer():
    """Example with custom trainer class."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Custom Trainer with Perceptual Loss")
    print("=" * 60)

    # Create autoencoder
    autoencoder = BlockBasedAutoencoder(
        input_channels=80,
        activation='gelu'
    )

    # Create custom trainer
    lightning_model = CustomLossAutoencoder(
        model=autoencoder,
        learning_rate=1e-3,
        perceptual_weight=0.1,
        max_epochs=3
    )

    # Create data
    train_loader, val_loader = create_dummy_dataset(
        batch_size=4,
        num_samples=50,
        input_shape=(80, 100, 128)
    )

    # Manual trainer setup
    trainer = Trainer(
        max_epochs=3,
        accelerator='cpu',
        devices=1,
        callbacks=[
            ModelCheckpoint(monitor='val_loss', mode='min'),
            EarlyStopping(monitor='val_loss', patience=2)
        ],
        enable_progress_bar=True,
        logger=False  # Disable logging for testing
    )

    # Train
    trainer.fit(lightning_model, train_loader, val_loader)
    print("Custom training completed!")


def example_multimodal_training():
    """Example with multimodal autoencoder."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Multimodal Autoencoder Training")
    print("=" * 60)

    # Create multimodal model
    multimodal_model = MultimodalAutoencoder(
        audio_channels=80,
        text_vocab_size=1000
    )

    # Create multimodal trainer with loss weights
    lightning_model = MultimodalLightningTrainer(
        model=multimodal_model,
        loss_weights={'audio_loss': 1.0, 'text_loss': 0.5},
        learning_rate=1e-3,
        max_epochs=3
    )

    # Create multimodal data
    train_loader, val_loader = create_multimodal_dataset(
        batch_size=4,
        num_samples=50
    )

    # Train
    trainer = Trainer(
        max_epochs=3,
        accelerator='cpu',
        devices=1,
        enable_progress_bar=True,
        logger=False
    )

    trainer.fit(lightning_model, train_loader, val_loader)
    print("Multimodal training completed!")


def example_configuration_testing():
    """Test different autoencoder configurations."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Testing Different Configurations")
    print("=" * 60)

    # Test configurations similar to your example
    configs = [
        {
            'name': 'Default',
            'config': {'input_channels': 80}
        },
        {
            'name': 'Custom blocks',
            'config': {
                'input_channels': 80,
                'block_configs': [
                    {'out_channels': 64, 'pool_size': (1, 2), 'dropout': 0.2},
                    {'out_channels': 128, 'pool_size': (1, 4), 'dropout': 0.3},
                ],
                'activation': 'gelu'
            }
        },
        {
            'name': 'Large model',
            'config': {
                'input_channels': 80,
                'activation': 'swish'
            }
        }
    ]

    for config_info in configs:
        print(f"\nTesting {config_info['name']} configuration:")

        # Create autoencoder
        autoencoder = BlockBasedAutoencoder(**config_info['config'])

        # Test forward pass
        x = torch.randn(2, 80, 100, 128)
        reconstructed, latent = autoencoder(x)

        print(f"  Input shape: {x.shape}")
        print(f"  Latent shape: {latent.shape}")
        print(f"  Output shape: {reconstructed.shape}")
        print(f"  Parameters: {autoencoder.parameter_count:,}")

        # Quick training test
        lightning_model = LightningTrainer(
            model=autoencoder,
            learning_rate=1e-3,
            max_epochs=1
        )

        # Create minimal data
        train_data = TensorDataset(x, x)
        train_loader = DataLoader(train_data, batch_size=2)

        trainer = Trainer(
            max_epochs=1,
            accelerator='cpu',
            devices=1,
            enable_progress_bar=False,
            logger=False
        )

        trainer.fit(lightning_model, train_loader)
        print("  Training test: PASSED")


def example_feature_extraction():
    """Example showing feature extraction capabilities."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Feature Extraction and Analysis")
    print("=" * 60)

    # Create and train a small model
    autoencoder = BlockBasedAutoencoder(input_channels=80)

    lightning_model = LightningTrainer(
        model=autoencoder,
        learning_rate=1e-3,
        max_epochs=2
    )

    # Quick training
    train_loader, _ = create_dummy_dataset(
        batch_size=4,
        num_samples=20,
        input_shape=(80, 100, 128)
    )

    trainer = Trainer(
        max_epochs=2,
        accelerator='cpu',
        devices=1,
        enable_progress_bar=False,
        logger=False
    )

    trainer.fit(lightning_model, train_loader)

    # Test feature extraction
    test_input = torch.randn(1, 80, 100, 128)

    with torch.no_grad():
        # Get latent representation
        latent = autoencoder.encode(test_input)
        print(f"Latent representation shape: {latent.shape}")

        # Get reconstruction
        reconstruction = autoencoder.decode(latent)
        print(f"Reconstruction shape: {reconstruction.shape}")

        # Calculate reconstruction error
        mse_error = F.mse_loss(reconstruction, test_input).item()
        print(f"Reconstruction MSE: {mse_error:.6f}")


def run_all_examples():
    """Run all examples."""
    print("Running Lightning Trainer Examples...")
    print("Note: These are minimal examples for demonstration.")
    print("For real training, use larger datasets and more epochs.\n")

    try:
        example_basic_training()
        example_custom_trainer()
        example_multimodal_training()
        example_configuration_testing()
        example_feature_extraction()

        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    pl.seed_everything(42)

    # Run examples
    run_all_examples()
