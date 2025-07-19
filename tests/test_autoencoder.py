import torch

from src.faith.train.models import BlockBasedAutoencoder

# Test basic functionality
print("Testing BlockBasedAutoencoder...")

# Create autoencoder with default config
autoencoder = BlockBasedAutoencoder(input_channels=80)

# Test forward pass
x = torch.randn(2, 80, 100, 128)
reconstructed = autoencoder(x)
latent = autoencoder.encode(x)

print(f"Input shape: {x.shape}")
print(f"Latent shape: {latent.shape}")
print(f"Reconstructed shape: {reconstructed.shape}")
print(f"Autoencoder: {autoencoder}")

# Test individual methods
latent_only = autoencoder.get_latent_representation(x)
reconstructed_only = autoencoder.reconstruct(x)

print(f"Latent only shape: {latent_only.shape}")
print(f"Reconstructed only shape: {reconstructed_only.shape}")

# Test configuration serialization
config = autoencoder.get_config()
print(f"Config keys: {list(config.keys())}")

new_autoencoder = BlockBasedAutoencoder.from_config(config)
print(f"Recreated autoencoder: {new_autoencoder}")

# Test shape calculation
output_shape = autoencoder.get_output_shape((1, 80, 100, 128))
latent_shape = autoencoder.get_latent_shape((1, 80, 100, 128))
print(f"Calculated output shape: {output_shape}")
print(f"Calculated latent shape: {latent_shape}")

# Test parameter counting
print(f"Total parameters: {autoencoder.parameter_count:,}")
print(f"Encoder parameters: {autoencoder.encoder_parameter_count:,}")
print(f"Decoder parameters: {autoencoder.decoder_parameter_count:,}")

# Test feature map extraction
feature_maps = autoencoder.get_feature_maps(x)
print(f"Encoder feature maps: {len(feature_maps['encoder'])}")
print(f"Decoder feature maps: {len(feature_maps['decoder'])}")

# Test custom configuration
custom_configs = [
    {'out_channels': 64, 'pool_size': (1, 2), 'dropout': 0.2},
    {'out_channels': 128, 'pool_size': (1, 4), 'dropout': 0.3},
]

custom_autoencoder = BlockBasedAutoencoder(
    input_channels=80,
    block_configs=custom_configs,
    activation='gelu'
)

x_custom = torch.randn(1, 80, 100, 128)
reconstructed_custom, latent_custom = custom_autoencoder(x_custom)

print("\nCustom autoencoder:")
print(f"Input shape: {x_custom.shape}")
print(f"Latent shape: {latent_custom.shape}")
print(f"Reconstructed shape: {reconstructed_custom.shape}")
print(f"Custom autoencoder: {custom_autoencoder}")
