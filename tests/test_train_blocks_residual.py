import torch
from src.faith.train.blocks import ResidualBlock


# Example usage and testing
# Test basic functionality
block = ResidualBlock(64, 128, stride=2)
x = torch.randn(1, 64, 32, 32)
output = block(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Block: {block}")

# Test configuration serialization
config = block.get_config()
print(f"Config: {config}")

# Create from config
new_block = ResidualBlock.from_config(config)
print(f"Recreated block: {new_block}")
