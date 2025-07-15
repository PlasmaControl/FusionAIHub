# Example usage and testing
import torch
from src.faith.train.blocks import EncoderBlock, BlockBasedEncoder


# Test EncoderBlock
print("Testing EncoderBlock...")
encoder_block = EncoderBlock(
    in_channels=64,
    out_channels=128,
    pool_size=(1, 2),
    dropout=0.3,
    activation='relu'
)

x = torch.randn(1, 64, 32, 32)
output = encoder_block(x)
print(f"EncoderBlock - Input: {x.shape}, Output: {output.shape}")

# Test configuration
config = encoder_block.get_config()
new_block = EncoderBlock.from_config(config)
print(f"Config serialization successful: {new_block}")

# Test BlockBasedEncoder
print("\nTesting BlockBasedEncoder...")
block_configs = [
    {'out_channels': 128, 'pool_size': (1, 2), 'dropout': 0.2},
    {'out_channels': 256, 'pool_size': (1, 4), 'dropout': 0.3},
    {'out_channels': 128, 'pool_size': (1, 2), 'dropout': 0.4},
]

encoder = BlockBasedEncoder(
    input_channels=80,
    block_configs=block_configs,
    hidden_dim=16,
    bottleneck_channels=64
)

x = torch.randn(2, 80, 100, 128)
latent = encoder(x)
print(f"Encoder - Input: {x.shape}, Output: {latent.shape}")

# Test feature map extraction
feature_maps = encoder.get_feature_maps(x)
print(f"Feature maps shapes: {[fm.shape for fm in feature_maps]}")
