import pytest
import torch

from faith.train.blocks import ResidualEncoding1d


def test_kernel_size_constant_channels() -> None:
    """Test the kernel size of the ResidualBlock."""
    block = ResidualEncoding1d(4, 4, kernel_size=3)

    # Test that the block was created successfully
    assert block is not None

    # Test that the kernel size is correctly set in the convolutional layers
    # Assuming ResidualBlock has conv layers with the specified kernel size
    for module in block.modules():
        if isinstance(module, torch.nn.Conv1d):
            assert module.kernel_size == (3,), (
                f"Expected kernel size (3, ), got {module.kernel_size}"
            )


def test_kernel_size_changing_channels() -> None:
    """Test the kernel size of the ResidualBlock."""
    block = ResidualEncoding1d(4, 6, kernel_size=3)

    # Test that the block was created successfully
    assert block is not None

    # Test that the kernel size is correctly set in the convolutional layers
    # Assuming ResidualBlock has conv layers with the specified kernel size
    for module in block.modules():
        if isinstance(module, torch.nn.Conv1d):
            assert module.kernel_size == (3,) or module.kernel_size == (1,), (
                f"Expected kernel size (3, ) or (1, ), got {module.kernel_size}"
            )


def test_kernel_size_different_values() -> None:
    """Test ResidualBlock with different kernel sizes."""
    test_cases = [
        (1, (1,)),
        (3, (3,)),
        (5, (5,)),
        (7, (7,)),
    ]

    for kernel_size, expected in test_cases:
        block = ResidualEncoding1d(4, 8, kernel_size=kernel_size)

        # Check that conv layers have the correct kernel size
        conv_layers = [
            module for module in block.modules() if isinstance(module, torch.nn.Conv1d)
        ]

        assert len(conv_layers) == 3, "ResidualBlock should contain 3 Conv1d layers"

        for conv_layer in conv_layers:
            assert conv_layer.kernel_size == expected or conv_layer.kernel_size == (
                1,
            ), (
                f"For kernel_size={kernel_size}, expected {expected}, "
                f"got {conv_layer.kernel_size}"
            )

        block = ResidualEncoding1d(4, 4, kernel_size=kernel_size)

        # Check that conv layers have the correct kernel size
        conv_layers = [
            module for module in block.modules() if isinstance(module, torch.nn.Conv1d)
        ]

        assert len(conv_layers) == 2, "ResidualBlock should contain Conv1d layers"

        for conv_layer in conv_layers:
            assert conv_layer.kernel_size == expected, (
                f"For kernel_size={kernel_size}, expected {expected}, "
                f"got {conv_layer.kernel_size}"
            )


def test_kernel_size_with_forward_pass() -> None:
    """Test that different kernel sizes work in forward pass."""
    batch_size = 2
    channels = 2
    width = 32

    input_tensor = torch.randn(batch_size, channels, width)

    # Test different kernel sizes
    for kernel_size in [1, 3, 5]:
        block = ResidualEncoding1d(2, 4, kernel_size=kernel_size)
        block.eval()  # Set to evaluation mode

        with torch.no_grad():
            output = block(input_tensor)

        # Check output shape is reasonable
        assert output.shape[0] == input_tensor.shape[0], (
            f"Batch size {input_tensor.shape[0]} should be preserved, got "
            f"{output.shape[0]}"
        )
        assert output.shape[2] == input_tensor.shape[2], (
            f"Width {input_tensor.shape[2]} should be preserved, got {output.shape[2]}"
        )
        assert output.shape[1] == 4, (
            f"Output channels should be 4, got {output.shape[1]}"
        )

        assert len(output.shape) == 3, (
            f"Output should be 4D tensor, got shape {output.shape}"
        )


def test_invalid_kernel_size() -> None:
    """Test that invalid kernel sizes raise appropriate errors."""
    with pytest.raises(ValueError):
        ResidualEncoding1d(2, 4, kernel_size=0)

    with pytest.raises(ValueError):
        ResidualEncoding1d(2, 4, kernel_size=-1)


def test_kernel_size_parameter_types() -> None:
    """Test that kernel_size accepts different parameter types."""
    # Test integer
    block1 = ResidualEncoding1d(2, 4, kernel_size=3)
    assert block1 is not None

    # Test tuple (if supported)
    block2 = ResidualEncoding1d(2, 4, kernel_size=(3, 3))
    assert block2 is not None


def test_invalid_channels() -> None:
    """Test that invalid channel numbers raise appropriate errors."""
    # Test zero input channels
    with pytest.raises(ValueError):
        ResidualEncoding1d(0, 2, kernel_size=3)

    # Test negative input channels
    with pytest.raises(ValueError):
        ResidualEncoding1d(-64, 2, kernel_size=3)

    # Test zero output channels
    with pytest.raises(ValueError):
        ResidualEncoding1d(2, 0, kernel_size=3)

    # Test negative output channels
    with pytest.raises(ValueError):
        ResidualEncoding1d(2, -128, kernel_size=3)

    # Test non-integer channels
    with pytest.raises(TypeError):
        ResidualEncoding1d(64.5, 128, kernel_size=3)

    with pytest.raises(TypeError):
        ResidualEncoding1d(64, 128.5, kernel_size=3)


def test_valid_channels() -> None:
    """Test that valid channel numbers work correctly."""
    valid_channel_pairs = [
        (1, 1),
        (1, 64),
        (64, 1),
        (32, 64),
        (64, 128),
        (128, 256),
        (512, 512),
    ]

    for in_channels, out_channels in valid_channel_pairs:
        block = ResidualEncoding1d(in_channels, out_channels, kernel_size=3)
        assert block is not None

        # Test forward pass with appropriate input
        input_tensor = torch.randn(1, in_channels, 8)
        with torch.no_grad():
            output = block(input_tensor)
            assert output.shape == (1, out_channels, 8)


def test_invalid_stride() -> None:
    """Test that invalid stride values raise appropriate errors."""
    # Test zero stride
    with pytest.raises(ValueError):
        ResidualEncoding1d(64, 128, kernel_size=3, stride=0)

    # Test negative stride
    with pytest.raises(ValueError):
        ResidualEncoding1d(64, 128, kernel_size=3, stride=-1)

    # Test non-integer stride
    with pytest.raises(TypeError):
        ResidualEncoding1d(64, 128, kernel_size=3, stride=1.5)


def test_valid_stride() -> None:
    """Test that valid stride values work correctly."""
    valid_strides = [1, 2, 3, 4]

    for stride in valid_strides:
        block = ResidualEncoding1d(64, 128, kernel_size=3, stride=stride)
        assert block is not None

        # Test that stride affects conv layers
        conv_layers = [
            module for module in block.modules() if isinstance(module, torch.nn.Conv1d)
        ]

        # At least one conv layer should have the specified stride
        stride_found = any(
            conv.stride == (stride,) or conv.stride == stride for conv in conv_layers
        )
        assert stride_found, f"No conv layer found with stride {stride}"


def test_stride_output_shape() -> None:
    """Test that stride correctly affects output dimensions."""
    input_size = 32
    input_tensor = torch.randn(1, 2, input_size, input_size)

    for stride in [1, 2]:
        block = ResidualEncoding1d(2, 4, kernel_size=3, stride=stride)
        block.eval()

        with torch.no_grad():
            output = block(input_tensor)

        # Output spatial dimensions are affected by stride
        if stride == 1:
            # Might be same size
            assert output.shape[2] == input_size
            assert output.shape[3] == input_size
        elif stride == 2:
            # Should be half the size
            assert output.shape[2] <= input_size // stride
            assert output.shape[3] <= input_size // stride


def test_combined_invalid_parameters() -> None:
    """Test combinations of invalid parameters."""
    invalid_combinations = [
        # (in_channels, out_channels, kernel_size, stride)
        (0, 0, 0, 0),  # All invalid
        (-1, 128, 3, 1),  # Invalid in_channels
        (64, -1, 3, 1),  # Invalid out_channels
        (64, 128, -1, 1),  # Invalid kernel_size
        (64, 128, 3, -1),  # Invalid stride
    ]

    for in_ch, out_ch, k_size, stride in invalid_combinations:
        with pytest.raises((ValueError, TypeError, RuntimeError)):
            ResidualEncoding1d(
                in_ch,
                out_ch,
                kernel_size=k_size,
                stride=stride,
            )
