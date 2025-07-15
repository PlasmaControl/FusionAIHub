import torch
from src.faith.train.blocks import BaseBlock, BlockUtils


# Example of how the base classes would be used

class ExampleBlock(BaseBlock):
    """Example implementation of BaseBlock."""

    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=self.kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

# Test utility functions
input_shape = (1, 64, 32, 32)
memory_info = BlockUtils.get_memory_usage(block, input_shape)
print(f"Memory usage: {memory_info}")

output_shape = BlockUtils.calculate_output_shape(
    input_shape, kernel_size=3, stride=1, padding=1
)
print(f"Output shape: {output_shape}")
