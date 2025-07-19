import torch

from src.faith.train.models import (
    BlockBasedAutoencoder,
    MaskedAutoencoder,
    MaskGenerator,
    mae_loss,
)

# Example usage and testing
if __name__ == "__main__":
    # Test MaskGenerator
    print("Testing MaskGenerator...")
    mask_gen = MaskGenerator(mask_ratio=0.75)

    shape = (2, 80, 100, 128)

    # Test MaskedAutoencoder
    print("\nTesting MaskedAutoencoder...")

    # Create components
    autoencoder = BlockBasedAutoencoder(input_channels=80)
    mae = MaskedAutoencoder(autoencoder, mask_gen)

    # Test forward pass
    x = torch.randn(2, 80, 100, 128)
    reconstructed, mask, masked_input = mae(x, mask_type="frequency")

    print(f"Input shape: {x.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Masked input shape: {masked_input.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")

    # Test MAE loss
    loss = mae_loss(reconstructed, x, mask, loss_type="mse")
    print(f"MAE loss: {loss.item():.6f}")

    # Test different loss types
    for loss_type in ["mse", "l1", "smooth_l1"]:
        loss = mae_loss(reconstructed, x, mask, loss_type=loss_type)
        print(f"{loss_type} loss: {loss.item():.6f}")

    # Test configuration serialization
    config = mae.get_config()
    mae_recreated = MaskedAutoencoder.from_config(config)
    print(f"Config serialization successful: {mae_recreated}")

    print(f"MAE model: {mae}")
