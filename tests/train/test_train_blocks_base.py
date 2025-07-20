import torch
import torch.nn as nn

from faith.train.blocks import BaseConvBlock, BlockUtils

# Example of how the base classes would be used


class ExampleBlock(BaseConvBlock):
    """Example implementation of BaseBlock."""

    def __init__(self, in_channels: int, out_channels: int, **kwargs) -> None:
        super().__init__(in_channels, out_channels, **kwargs)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# Test utility functions
input_shape = (1, 64, 32, 32)
# Create an example block for testing
example_block = ExampleBlock(in_channels=64, out_channels=32)
memory_info = BlockUtils.get_memory_usage(example_block, input_shape)
print(f"Memory usage: {memory_info}")

output_shape = BlockUtils.calculate_output_shape(
    input_shape, kernel_size=3, stride=1, padding=1
)
print(f"Output shape: {output_shape}")
