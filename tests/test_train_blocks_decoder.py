import pytest
import torch
import torch.nn as nn

from src.faith.train.blocks import DecoderBlock, ResidualBlock


class TestDecoderBlockInitialization:
    """Test DecoderBlock initialization and parameter validation."""

    def test_basic_initialization(self):
        """Test basic DecoderBlock initialization with default parameters."""
        block = DecoderBlock(in_channels=4, out_channels=2)

        assert block.in_channels == 4
        assert block.out_channels == 2
        assert block.upsample_factor == (1, 2)
        assert block.dropout == 0.3
        assert block.use_batch_norm is True
        assert block.activation_name == "relu"
        assert block.residual_init_method == "kaiming"

    def test_custom_initialization(self):
        """Test DecoderBlock initialization with custom parameters."""
        block = DecoderBlock(
            in_channels=4,
            out_channels=2,
            upsample_factor=(2, 2),
            kernel_size=5,
            stride=2,
            dropout=0.5,
            bias=False,
            use_batch_norm=False,
            activation='gelu',
            residual_init_method='xavier'
        )

        assert block.in_channels == 4
        assert block.out_channels == 2
        assert block.upsample_factor == (2, 2)
        assert block.kernel_size == (5, 5)
        assert block.dropout == 0.5
        assert block.bias is False
        assert block.use_batch_norm is False
        assert block.activation_name == "gelu"
        assert block.residual_init_method == "xavier"

    def test_operations_creation(self):
        """Test that all operations are created correctly."""
        block = DecoderBlock(in_channels=4, out_channels=2)

        assert hasattr(block, "conv_transpose")
        assert hasattr(block, "dropout")
        assert hasattr(block, "residual_block")
        assert len(block.operations) == 3

        # Check types
        assert isinstance(block.conv_transpose, nn.ConvTranspose2d)
        assert isinstance(block.dropout_layer, nn.Dropout)
        assert isinstance(block.residual_block, ResidualBlock)

    def test_conv_transpose_parameters(self):
        """Test ConvTranspose2d layer parameters."""
        block = DecoderBlock(
            in_channels=8,
            out_channels=4,
            upsample_factor=(2, 2),
            bias=False,
        )

        conv_transpose = block.conv_transpose
        assert conv_transpose.in_channels == 8
        assert conv_transpose.out_channels == 8  # Same as in_channels
        assert conv_transpose.kernel_size == (4, 4)
        assert conv_transpose.stride == (2, 2)
        assert conv_transpose.padding == (1, 1)  # (4-1)//2 = 1
        assert conv_transpose.bias is None  # bias=False

    def test_operations_order(self):
        """Test that operations are in correct order."""
        block = DecoderBlock(in_channels=8, out_channels=4)

        # Should be: ConvTranspose2d, Dropout, ResidualBlock
        assert isinstance(block.operations[0], nn.ConvTranspose2d)
        assert isinstance(block.operations[1], nn.Dropout)
        assert isinstance(block.operations[2], ResidualBlock)


class TestDecoderBlockValidation:
    """Test parameter validation in DecoderBlock."""

    def test_invalid_dropout_values(self):
        """Test that invalid dropout values raise ValueError."""
        with pytest.raises(
            ValueError, match="Dropout must be between 0.0 and 1.0"
        ):
            DecoderBlock(in_channels=4, out_channels=2, dropout=-0.1)

        with pytest.raises(
            ValueError, match="Dropout must be between 0.0 and 1.0"
        ):
            DecoderBlock(in_channels=4, out_channels=2, dropout=1.5)

    def test_invalid_upsample_factor(self):
        """Test that invalid upsample_factor raises ValueError."""
        with pytest.raises(
            ValueError, match="upsample_factor must be a tuple of length 2"
        ):
            DecoderBlock(in_channels=4, out_channels=2, upsample_factor=(2,))

        with pytest.raises(
            ValueError, match="upsample_factor must be a tuple of length 2"
        ):
            DecoderBlock(
                in_channels=4, out_channels=2, upsample_factor=(1, 2, 3)
            )

    def test_boundary_dropout_values(self):
        """Test boundary dropout values (0.0 and 1.0)."""
        block1 = DecoderBlock(in_channels=4, out_channels=2, dropout=0.0)
        assert block1.dropout == 0.0

        block2 = DecoderBlock(in_channels=4, out_channels=2, dropout=1.0)
        assert block2.dropout == 1.0


class TestDecoderBlockForwardPass:
    """Test DecoderBlock forward pass functionality."""

    def test_forward_pass_basic(self):
        """Test basic forward pass with default parameters."""
        block = DecoderBlock(in_channels=8, out_channels=4)
        x = torch.randn(2, 8, 16, 8)

        output = block(x)

        assert output.shape[0] == 2  # batch size
        assert output.shape[1] == 4  # out_channels from ResidualBlock
        assert output.shape[2] == 16  # height should be positive
        assert output.shape[3] == 16  # width should be positive

    def test_forward_pass_2x1_upsampling(self):
        """Test forward pass with 2x1 upsampling."""

        block = DecoderBlock(in_channels=8, out_channels=4,
                             upsample_factor=(2, 1))
        x = torch.randn(1, 8, 8, 8)

        output = block(x)

        assert output.shape[0] == 1
        assert output.shape[1] == 4
        # Should approximately double the spatial dimensions
        assert output.shape[2] == 16
        assert output.shape[3] == 8

    def test_forward_pass_no_dropout(self):
        """Test forward pass with dropout disabled."""
        block = DecoderBlock(in_channels=4, out_channels=2, dropout=0.0)
        x = torch.randn(2, 4, 8, 8)

        output = block(x)
        assert output.shape[1] == 2

    def test_dropout_parameters(self):
        """Test that Dropout is created with correct parameters."""
        block = DecoderBlock(in_channels=4, out_channels=2, dropout=0.7)

        dropout_layer = block.dropout_layer
        assert dropout_layer.p == 0.7


class TestDecoderBlockConfiguration:
    """Test DecoderBlock configuration methods."""

    def test_get_config(self):
        """Test get_config method returns complete configuration."""
        block = DecoderBlock(
            in_channels=8,
            out_channels=4,
            upsample_factor=(2, 2),
            dropout=0.5,
            activation="gelu",
        )

        config = block.get_config()

        # Check all important parameters are in config
        assert config["in_channels"] == 8
        assert config["out_channels"] == 4
        assert config["upsample_factor"] == (2, 2)
        assert config["dropout"] == 0.5
        assert config["activation"] == "gelu"

    def test_from_config(self):
        """Test from_config class method creates equivalent block."""
        original_block = DecoderBlock(
            in_channels=8,
            out_channels=4,
            upsample_factor=(2, 2),
            dropout=0.4,
            activation="gelu",
        )

        config = original_block.get_config()
        reconstructed_block = DecoderBlock.from_config(config)

        # Check key attributes match
        assert reconstructed_block.in_channels == original_block.in_channels
        assert reconstructed_block.out_channels == original_block.out_channels
        assert (
            reconstructed_block.upsample_factor
            == original_block.upsample_factor
        )
        assert reconstructed_block.dropout == original_block.dropout
        assert (
            reconstructed_block.activation_name
            == original_block.activation_name
        )

    def test_config_roundtrip(self):
        """Test that config -> block -> config roundtrip works."""
        original_config = {
            "in_channels": 4,
            "out_channels": 2,
            "upsample_factor": (2, 2),
            "dropout": 0.3,
            "activation": "relu",
        }

        block = DecoderBlock.from_config(original_config)
        reconstructed_config = block.get_config()

        # Check that key parameters survive roundtrip
        for key in original_config:
            assert reconstructed_config[key] == original_config[key]


class TestDecoderBlockShapeCalculation:
    """Test DecoderBlock shape calculation methods."""

    def test_get_output_shape_basic(self):
        """Test get_output_shape with basic parameters."""
        block = DecoderBlock(in_channels=8, out_channels=4)

        input_shape = (2, 8, 16, 8)
        output_shape = block.get_output_shape(input_shape)

        assert output_shape[0] == 2  # batch size
        assert output_shape[1] == 4  # out_channels from ResidualBlock
        assert output_shape[2] == 16  # height should be positive
        assert output_shape[3] == 16  # width should be positive

    def test_get_output_shape_2x2_upsampling(self):
        """Test get_output_shape with 2x2 upsampling."""
        block = DecoderBlock(
            in_channels=4, out_channels=2, upsample_factor=(2, 2)
        )

        input_shape = (1, 4, 8, 8)
        output_shape = block.get_output_shape(input_shape)

        assert output_shape[0] == 1
        assert output_shape[1] == 2
        # Should approximately double both dimensions
        assert output_shape[2] == 16
        assert output_shape[3] == 16


class TestDecoderBlockRepresentation:
    """Test DecoderBlock string representation."""

    def test_repr_basic(self):
        """Test __repr__ method with basic parameters."""
        block = DecoderBlock(in_channels=8, out_channels=4)
        repr_str = repr(block)

        assert "DecoderBlock(" in repr_str
        assert "in_channels=8" in repr_str
        assert "out_channels=4" in repr_str
        assert "upsample_factor=(1, 2)" in repr_str
        assert "dropout=0.3" in repr_str
        assert "activation='relu'" in repr_str

    def test_repr_custom(self):
        """Test __repr__ method with custom parameters."""
        block = DecoderBlock(
            in_channels=6,
            out_channels=8,
            upsample_factor=(2, 2),
            dropout=0.5,
            activation="gelu",
        )
        repr_str = repr(block)

        assert "in_channels=6" in repr_str
        assert "out_channels=8" in repr_str
        assert "upsample_factor=(2, 2)" in repr_str
        assert "dropout=0.5" in repr_str
        assert "activation='gelu'" in repr_str


class TestDecoderBlockIntegration:
    """Test DecoderBlock integration aspects."""

    def test_sequential_block_inheritance(self):
        """Test that DecoderBlock inherits from SequentialBlock correctly."""
        block = DecoderBlock(in_channels=8, out_channels=4)

        # Should have SequentialBlock attributes
        assert hasattr(block, "operations")
        assert hasattr(block, "in_channels")
        assert hasattr(block, "out_channels")

    def test_component_access(self):
        """Test that individual components can be accessed."""
        block = DecoderBlock(in_channels=8, out_channels=4)

        # Should be able to access individual components
        assert hasattr(block, "conv_transpose")
        assert hasattr(block, "dropout_layer")
        assert hasattr(block, "residual_block")

        # Components should be the same as operations
        assert block.conv_transpose is block.operations[0]
        assert block.dropout_layer is block.operations[1]
        assert block.residual_block is block.operations[2]


class TestDecoderBlockEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_channel_input_output(self):
        """Test with single channel input and output."""
        block = DecoderBlock(in_channels=1, out_channels=1)
        x = torch.randn(1, 1, 16, 16)

        output = block(x)
        assert output.shape[1] == 1

    def test_large_upsampling_factor(self):
        """Test with large upsampling factors."""
        block = DecoderBlock(
            in_channels=4, out_channels=2, upsample_factor=(4, 4)
        )
        x = torch.randn(1, 4, 2, 2)

        output = block(x)
        assert output.shape[1] == 2
        assert output.shape[2] == 6
        assert output.shape[3] == 6

    def test_minimal_spatial_dimensions(self):
        """Test with minimal spatial dimensions."""
        block = DecoderBlock(in_channels=4, out_channels=2)
        x = torch.randn(1, 4, 1, 1)

        output = block(x)
        assert output.shape[1] == 2
        assert output.shape[2] == 1
        assert output.shape[3] == 2


# Fixtures for common test data
@pytest.fixture
def sample_input():
    """Fixture providing sample input tensor."""
    return torch.randn(2, 128, 16, 8)


@pytest.fixture
def basic_decoder_block():
    """Fixture providing basic DecoderBlock instance."""
    return DecoderBlock(in_channels=8, out_channels=4)


@pytest.fixture
def custom_decoder_block():
    """Fixture providing custom DecoderBlock instance."""
    return DecoderBlock(
        in_channels=256,
        out_channels=128,
        upsample_factor=(2, 2),
        dropout=0.4,
        activation='gelu'
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



#######################################################################
"""
# Example usage and testing
# Test DecoderBlock
print("Testing DecoderBlock...")
decoder_block = DecoderBlock(
    in_channels=8,
    out_channels=4,
    upsample_factor=(1, 2),
    dropout=0.3,
    activation='relu'
)

x = torch.randn(1, 128, 16, 8)
output = decoder_block(x)
print(f"DecoderBlock - Input: {x.shape}, Output: {output.shape}")

# Test configuration
config = decoder_block.get_config()
new_block = DecoderBlock.from_config(config)
print(f"Config serialization successful: {new_block}")

# Test BlockBasedDecoder with mock encoder blocks
print("\nTesting BlockBasedDecoder...")

# Create mock encoder blocks for testing

mock_encoder_blocks = [
    EncoderBlock(80, 128, pool_size=(1, 2)),
    EncoderBlock(128, 256, pool_size=(1, 4)),
    EncoderBlock(256, 128, pool_size=(1, 2)),
]

decoder = BlockBasedDecoder(
    output_channels=80,
    encoder_blocks=mock_encoder_blocks,
    bottleneck_channels=64,
    upsampling_mode='nearest'
)

# Test forward pass
latent = torch.randn(2, 64, 25, 4)
reconstructed = decoder(latent)
print(f"Decoder - Input: {latent.shape}, Output: {reconstructed.shape}")

# Test feature map extraction
feature_maps = decoder.get_feature_maps(latent)
print(f"Feature maps shapes: {[fm.shape for fm in feature_maps]}")

# Test from_encoder class method
decoder2 = BlockBasedDecoder.from_encoder(
    encoder_blocks=mock_encoder_blocks,
    bottleneck_channels=64,
    output_channels=80,
)
print(f"Decoder from encoder: {decoder2}")
"""
