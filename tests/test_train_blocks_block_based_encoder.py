import pytest
import torch

from src.faith.train.blocks import BlockBasedEncoder


class TestBlockBasedEncoderInitialization:
    """Test BlockBasedEncoder initialization and parameter validation."""

    def test_basic_initialization(self):
        """Test basic BlockBasedEncoder initialization."""
        configs = [{"out_channels": 4}, {"out_channels": 8}]
        encoder = BlockBasedEncoder(in_channels=3, block_configs=configs)

        assert encoder.in_channels == 3
        assert encoder.out_channels == 8  # Last block's out_channels
        assert len(encoder.operations) == 2
        assert len(encoder.block_configs) == 2

    def test_custom_initialization(self):
        """Test BlockBasedEncoder with custom parameters."""
        configs = [
            {"out_channels": 4, "pool_size": (2, 2), "dropout": 0.5},
            {"out_channels": 8, "activation": "gelu"},
            {"out_channels": 16, "kernel_size": 5, "bias": False},
        ]
        encoder = BlockBasedEncoder(
            in_channels=3, block_configs=configs, kernel_size=7, bias=True
        )

        assert encoder.in_channels == 3
        assert encoder.out_channels == 16
        assert len(encoder.operations) == 3
        assert encoder.kernel_size == (7, 7)
        assert encoder.bias is True

    def test_single_block_encoder(self):
        """Test encoder with single block."""
        configs = [{"out_channels": 32}]
        encoder = BlockBasedEncoder(in_channels=8, block_configs=configs)

        assert encoder.in_channels == 8
        assert encoder.out_channels == 32
        assert len(encoder.operations) == 1
        assert encoder.kernel_size == (3, 3)
        assert encoder.bias is True
        assert len(encoder.operations) == 1

    def test_channel_progression_setup(self):
        """Test that blocks are configured with correct channel progression."""
        configs = [
            {"out_channels": 4},
            {"out_channels": 8},
            {"out_channels": 16},
        ]
        encoder = BlockBasedEncoder(in_channels=3, block_configs=configs)

        # Check channel progression
        progression = encoder.get_channel_progression()
        assert progression == [3, 4, 8, 16]

        # Check individual block configurations
        assert encoder.operations[0].in_channels == 3
        assert encoder.operations[0].out_channels == 4
        assert encoder.operations[1].in_channels == 4
        assert encoder.operations[1].out_channels == 8
        assert encoder.operations[2].in_channels == 8
        assert encoder.operations[2].out_channels == 16


class TestBlockBasedEncoderValidation:
    """Test parameter validation in BlockBasedEncoder."""

    def test_empty_block_configs(self):
        """Test that empty block_configs raises ValueError."""
        with pytest.raises(ValueError, match="block_configs cannot be empty"):
            BlockBasedEncoder(in_channels=3, block_configs=[])

    def test_invalid_in_channels(self):
        """Test that invalid in_channels raises ValueError."""
        configs = [{"out_channels": 64}]

        with pytest.raises(ValueError, match="in_channels must be positive"):
            BlockBasedEncoder(in_channels=0, block_configs=configs)

        with pytest.raises(ValueError, match="in_channels must be positive"):
            BlockBasedEncoder(in_channels=-5, block_configs=configs)

    def test_missing_out_channels(self):
        """Test that missing out_channels in config raises ValueError."""
        configs = [
            {"out_channels": 64},
            {"dropout": 0.5},  # Missing out_channels
            {"out_channels": 256},
        ]

        with pytest.raises(
            ValueError, match="Block 1 missing required 'out_channels' key"
        ):
            BlockBasedEncoder(in_channels=3, block_configs=configs)

    def test_invalid_out_channels(self):
        """Test that invalid out_channels raises ValueError."""
        configs = [
            {"out_channels": 64},
            {"out_channels": 0},  # Invalid
            {"out_channels": 256},
        ]

        with pytest.raises(
            ValueError, match="out_channels must be positive, got 0 in block 1"
        ):
            BlockBasedEncoder(in_channels=3, block_configs=configs)

        configs = [
            {"out_channels": -32}  # Invalid
        ]

        with pytest.raises(
            ValueError,
            match="out_channels must be positive, got -32 in block 0",
        ):
            BlockBasedEncoder(in_channels=3, block_configs=configs)


class TestBlockBasedEncoderForwardPass:
    """Test BlockBasedEncoder forward pass functionality."""

    def test_forward_pass_basic(self):
        """Test basic forward pass."""
        configs = [{"out_channels": 4}, {"out_channels": 8}]
        encoder = BlockBasedEncoder(in_channels=3, block_configs=configs)
        x = torch.randn(2, 3, 32, 64)

        output = encoder(x)

        assert output.shape[0] == 2  # batch size
        assert output.shape[1] == 8  # final out_channels
        assert output.shape[2] == 32  # Height should be preserved
        assert output.shape[3] == 16  # Width pooling of 2 in every block

        progression = encoder.get_channel_progression()
        assert progression == [3, 4, 8]


    def test_forward_pass_single_block(self):
        """Test forward pass with single block."""
        configs = [{"out_channels": 32}]
        encoder = BlockBasedEncoder(in_channels=8, block_configs=configs)
        x = torch.randn(1, 8, 16, 16)

        output = encoder(x)

        assert output.shape[0] == 1
        assert output.shape[1] == 32
        assert output.shape[2] == 16  # Height should be preserved
        assert output.shape[3] == 8  # Width pooling of 2

    def test_forward_pass_multiple_blocks(self):
        """Test forward pass with multiple blocks."""
        configs = [
            {"out_channels": 4, "pool_size": (1, 2)},
            {"out_channels": 8, "pool_size": (2, 2)},
            {"out_channels": 16, "pool_size": (1, 1)},
        ]
        encoder = BlockBasedEncoder(in_channels=3, block_configs=configs)
        x = torch.randn(1, 3, 32, 32)

        output = encoder(x)

        assert output.shape[0] == 1
        assert output.shape[1] == 16
        assert output.shape[2] == 16
        # Width pooling of 2 in first and second block
        assert output.shape[3] == 8

        progression = encoder.get_channel_progression()
        assert progression == [3, 4, 8, 16]

    def test_forward_pass_different_input_sizes(self):
        """Test forward pass with different input sizes."""
        configs = [{"out_channels": 4}, {"out_channels": 8}]
        encoder = BlockBasedEncoder(in_channels=3, block_configs=configs)

        # Test various input sizes
        for h, w in [(8, 8), (16, 32), (32, 64), (64, 128)]:
            x = torch.randn(1, 3, h, w)
            output = encoder(x)

            assert output.shape[0] == 1
            assert output.shape[1] == 8
            assert output.shape[2] == h
            # Width pooling of 2 in every block
            assert output.shape[3] == w // 4

    def test_forward_pass_gradient_flow(self):
        """Test that gradients flow properly through the encoder."""
        configs = [{"out_channels": 8}, {"out_channels": 16}]
        encoder = BlockBasedEncoder(in_channels=8, block_configs=configs)
        x = torch.randn(1, 8, 16, 16, requires_grad=True)

        output = encoder(x)
        loss = output.sum()
        loss.backward()

        progression = encoder.get_channel_progression()
        assert progression == [8, 8, 16]

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_forward_pass_with_custom_parameters(self):
        """Test forward pass with custom block parameters."""
        configs = [
            {"out_channels": 2, "dropout": 0.5, "activation": "gelu"},
            {"out_channels": 4, "pool_size": (2, 2), "kernel_size": 5},
        ]
        encoder = BlockBasedEncoder(in_channels=1, block_configs=configs)
        x = torch.randn(1, 1, 32, 32)

        output = encoder(x)

        assert output.shape[0] == 1
        assert output.shape[1] == 4
        assert output.shape[2] == 16  # One height pooling of 2 in first block
        assert output.shape[3] == 8  # Width pooling of 2 in every block


class TestBlockBasedEncoderConfiguration:
    """Test BlockBasedEncoder configuration methods."""

    def test_get_config(self):
        """Test get_config method returns complete configuration."""
        configs = [
            {"out_channels": 64, "dropout": 0.4},
            {"out_channels": 128, "activation": "gelu"},
        ]
        encoder = BlockBasedEncoder(
            in_channels=3, block_configs=configs, kernel_size=5, bias=False
        )

        config = encoder.get_config()

        assert config["in_channels"] == 3
        assert config["out_channels"] == 128
        assert config["block_configs"] == configs
        assert config["kernel_size"] == (5, 5)
        assert config["bias"] is False

    def test_from_config(self):
        """Test from_config class method creates equivalent encoder."""
        original_configs = [
            {"out_channels": 64, "dropout": 0.3},
            {"out_channels": 128, "pool_size": (2, 2)},
        ]
        original_encoder = BlockBasedEncoder(
            in_channels=16, block_configs=original_configs, kernel_size=7
        )

        config = original_encoder.get_config()
        reconstructed_encoder = BlockBasedEncoder.from_config(config)

        assert (
            reconstructed_encoder.in_channels == original_encoder.in_channels
        )
        assert (
            reconstructed_encoder.out_channels == original_encoder.out_channels
        )
        assert (
            reconstructed_encoder.block_configs
            == original_encoder.block_configs
        )
        assert (
            reconstructed_encoder.kernel_size == original_encoder.kernel_size
        )

    def test_config_roundtrip(self):
        """Test that config -> encoder -> config roundtrip works."""
        original_config = {
            "in_channels": 8,
            "block_configs": [
                {"out_channels": 32, "dropout": 0.2},
                {"out_channels": 64, "activation": "relu"},
            ],
            "kernel_size": (3, 3),
            "bias": True,
        }

        encoder = BlockBasedEncoder.from_config(original_config)
        reconstructed_config = encoder.get_config()

        for key in original_config:
            if key != "out_channels":
                assert reconstructed_config[key] == original_config[key]


class TestBlockBasedEncoderChannelProgression:
    """Test BlockBasedEncoder channel progression functionality."""

    def test_get_channel_progression_basic(self):
        """Test get_channel_progression with basic configuration."""
        configs = [
            {"out_channels": 32},
            {"out_channels": 64},
            {"out_channels": 128},
        ]
        encoder = BlockBasedEncoder(in_channels=8, block_configs=configs)

        progression = encoder.get_channel_progression()
        assert progression == [8, 32, 64, 128]

    def test_get_channel_progression_single_block(self):
        """Test get_channel_progression with single block."""
        configs = [{"out_channels": 64}]
        encoder = BlockBasedEncoder(in_channels=16, block_configs=configs)

        progression = encoder.get_channel_progression()
        assert progression == [16, 64]

    def test_get_channel_progression_complex(self):
        """Test get_channel_progression with complex configuration."""
        configs = [
            {"out_channels": 128},
            {"out_channels": 64},  # Decreasing channels
            {"out_channels": 256},  # Then increasing
            {"out_channels": 32},  # Then decreasing again
        ]
        encoder = BlockBasedEncoder(in_channels=3, block_configs=configs)

        progression = encoder.get_channel_progression()
        assert progression == [3, 128, 64, 256, 32]


class TestBlockBasedEncoderShapeCalculation:
    """Test BlockBasedEncoder shape calculation methods."""

    def test_get_output_shape_basic(self):
        """Test get_output_shape with basic configuration."""
        configs = [{"out_channels": 64}, {"out_channels": 128}]
        encoder = BlockBasedEncoder(in_channels=3, block_configs=configs)
        input_shape = (2, 3, 32, 32)

        output_shape = encoder.get_output_shape(input_shape)

        assert output_shape[0] == 2  # batch size
        assert output_shape[1] == 128  # final out_channels
        assert output_shape[2] == 32  # height should be positive
        assert output_shape[3] == 8  # width should be positive

    def test_get_output_shape_matches_forward(self):
        """Test that get_output_shape matches actual forward pass output."""
        configs = [
            {"out_channels": 32, "pool_size": (1, 2)},
            {"out_channels": 64, "pool_size": (2, 1)},
        ]
        encoder = BlockBasedEncoder(in_channels=8, block_configs=configs)
        input_shape = (1, 8, 16, 16)

        predicted_shape = encoder.get_output_shape(input_shape)

        x = torch.randn(*input_shape)
        actual_output = encoder(x)

        assert predicted_shape == actual_output.shape

    def test_get_output_shape_different_pool_sizes(self):
        """Test get_output_shape with different pool sizes."""
        configs = [
            {"out_channels": 32, "pool_size": (2, 2)},
            {"out_channels": 64, "pool_size": (1, 4)},
        ]
        encoder = BlockBasedEncoder(in_channels=16, block_configs=configs)
        input_shape = (1, 16, 32, 32)

        predicted_shape = encoder.get_output_shape(input_shape)

        assert predicted_shape[0] == 1
        assert predicted_shape[1] == 64
        # Height should be reduced by factor of 2 from first block
        assert predicted_shape[2] == 16
        # Width should be reduced by factors 2 and 4 from both blocks
        assert predicted_shape[3] == 4

        x = torch.randn(*input_shape)
        actual_output = encoder(x)

        assert predicted_shape == actual_output.shape


class TestBlockBasedEncoderFeatureMaps:
    """Test BlockBasedEncoder feature map extraction."""

    def test_get_feature_maps_basic(self):
        """Test get_feature_maps returns correct number of maps."""
        configs = [
            {"out_channels": 32},
            {"out_channels": 64},
            {"out_channels": 128},
        ]
        encoder = BlockBasedEncoder(in_channels=8, block_configs=configs)
        x = torch.randn(1, 8, 16, 16)

        feature_maps = encoder.get_feature_maps(x)

        assert len(feature_maps) == 3  # One per block
        assert feature_maps[0].shape == (1, 32, 16, 8)
        assert feature_maps[1].shape == (1, 64, 16, 4)
        assert feature_maps[2].shape == (1, 128, 16, 2)

    def test_get_feature_maps_single_block(self):
        """Test get_feature_maps with single block."""
        configs = [{"out_channels": 64}]
        encoder = BlockBasedEncoder(in_channels=16, block_configs=configs)
        x = torch.randn(1, 16, 32, 32)

        feature_maps = encoder.get_feature_maps(x)

        assert len(feature_maps) == 1
        assert feature_maps[0].shape == (1, 64, 32, 16)

    def test_get_feature_maps_consistency(self):
        """Test that get_feature_maps gives same result as forward pass."""
        configs = [{"out_channels": 32}, {"out_channels": 64}]
        encoder = BlockBasedEncoder(in_channels=8, block_configs=configs)
        x = torch.randn(1, 8, 16, 16)

        encoder.eval()
        with torch.no_grad():
            feature_maps = encoder.get_feature_maps(x)
            final_output = encoder(x)

        # Last feature map should match forward pass output
        assert torch.allclose(feature_maps[-1], final_output, atol=1e-6)

    def test_get_feature_maps_independence(self):
        """Test that feature maps are independent copies."""
        configs = [{"out_channels": 32}, {"out_channels": 64}]
        encoder = BlockBasedEncoder(in_channels=8, block_configs=configs)
        encoder.eval()
        x = torch.randn(1, 8, 16, 16)

        with torch.no_grad():
            feature_maps = encoder.get_feature_maps(x)

            # Modify one feature map
            original_value = feature_maps[0][0, 0, 0, 0].item()
            feature_maps[0][0, 0, 0, 0] = 999.0

            # Get feature maps again
            new_feature_maps = encoder.get_feature_maps(x)

        # Should not be affected by previous modification
        assert new_feature_maps[0][0, 0, 0, 0].item() == original_value


class TestBlockBasedEncoderRepresentation:
    """Test BlockBasedEncoder string representation."""

    def test_repr_basic(self):
        """Test __repr__ method with basic configuration."""
        configs = [{"out_channels": 64}, {"out_channels": 128}]
        encoder = BlockBasedEncoder(in_channels=3, block_configs=configs)
        repr_str = repr(encoder)

        assert "BlockBasedEncoder(" in repr_str
        assert "blocks=2" in repr_str
        assert "channels=3 → 64 → 128" in repr_str

    def test_repr_single_block(self):
        """Test __repr__ method with single block."""
        configs = [{"out_channels": 32}]
        encoder = BlockBasedEncoder(in_channels=8, block_configs=configs)
        repr_str = repr(encoder)

        assert "blocks=1" in repr_str
        assert "channels=8 → 32" in repr_str

    def test_repr_complex(self):
        """Test __repr__ method with complex configuration."""
        configs = [
            {"out_channels": 16},
            {"out_channels": 32},
            {"out_channels": 64},
            {"out_channels": 128},
        ]
        encoder = BlockBasedEncoder(in_channels=3, block_configs=configs)
        repr_str = repr(encoder)

        assert "blocks=4" in repr_str
        assert "channels=3 → 16 → 32 → 64 → 128" in repr_str


class TestBlockBasedEncoderCompatibility:
    """Test BlockBasedEncoder compatibility and integration."""

    def test_sequential_block_inheritance(self):
        """
        Test that BlockBasedEncoder properly inherits from SequentialBlock.
        """
        configs = [{"out_channels": 64}, {"out_channels": 128}]
        encoder = BlockBasedEncoder(in_channels=3, block_configs=configs)

        # Should have SequentialBlock attributes
        assert hasattr(encoder, "operations")
        assert hasattr(encoder, "in_channels")
        assert hasattr(encoder, "out_channels")
        assert len(encoder.operations) == 2

    def test_blocks_property(self):
        """Test blocks property for backward compatibility."""
        configs = [{"out_channels": 32}, {"out_channels": 64}]
        encoder = BlockBasedEncoder(in_channels=16, block_configs=configs)

        assert hasattr(encoder, "blocks")
        assert len(encoder.blocks) == 2
        assert encoder.blocks[0] is encoder.operations[0]
        assert encoder.blocks[1] is encoder.operations[1]

    def test_module_list_functionality(self):
        """Test that operations work like ModuleList."""
        configs = [{"out_channels": 32}, {"out_channels": 64}]
        encoder = BlockBasedEncoder(in_channels=8, block_configs=configs)

        # Should be able to iterate over operations
        block_count = 0
        for block in encoder.operations:
            assert hasattr(block, "forward")
            block_count += 1
        assert block_count == 2

    def test_parameter_count(self):
        """Test that parameter count is reasonable."""
        configs = [{"out_channels": 32}, {"out_channels": 64}]
        encoder = BlockBasedEncoder(in_channels=16, block_configs=configs)

        total_params = sum(p.numel() for p in encoder.parameters())
        trainable_params = sum(
            p.numel() for p in encoder.parameters() if p.requires_grad
        )

        assert total_params > 0
        assert trainable_params == total_params  # All parameters are trainable


class TestBlockBasedEncoderEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_large_channel_counts(self):
        """Test with large channel counts."""
        configs = [{"out_channels": 512}, {"out_channels": 1024}]
        encoder = BlockBasedEncoder(in_channels=256, block_configs=configs)
        x = torch.randn(1, 256, 8, 8)

        output = encoder(x)
        assert output.shape == (1, 1024, 8, 2)

    def test_single_channel_input(self):
        """Test with single channel input."""
        configs = [{"out_channels": 16}, {"out_channels": 32}]
        encoder = BlockBasedEncoder(in_channels=1, block_configs=configs)
        x = torch.randn(1, 1, 32, 32)

        output = encoder(x)
        assert output.shape == (1, 32, 32, 8)

    def test_minimal_spatial_dimensions(self):
        """Test with minimal spatial dimensions."""
        configs = [{"out_channels": 64}]
        encoder = BlockBasedEncoder(in_channels=32, block_configs=configs)
        x = torch.randn(1, 32, 1, 1)

        with pytest.raises(ValueError):
            _ = encoder(x)

    def test_decreasing_channels(self):
        """Test with decreasing channel progression."""
        configs = [
            {"out_channels": 128},
            {"out_channels": 64},
            {"out_channels": 32},
        ]
        encoder = BlockBasedEncoder(in_channels=256, block_configs=configs)
        x = torch.randn(1, 256, 16, 16)

        output = encoder(x)
        assert output.shape == (1, 32, 16, 2)

        progression = encoder.get_channel_progression()
        assert progression == [256, 128, 64, 32]


# Fixtures for common test data
@pytest.fixture
def sample_input():
    """Fixture providing sample input tensor."""
    return torch.randn(2, 16, 32, 32)


@pytest.fixture
def basic_encoder():
    """Fixture providing basic BlockBasedEncoder instance."""
    configs = [{"out_channels": 64}, {"out_channels": 128}]
    return BlockBasedEncoder(in_channels=16, block_configs=configs)


@pytest.fixture
def complex_encoder():
    """Fixture providing complex BlockBasedEncoder instance."""
    configs = [
        {'out_channels': 32, 'pool_size': (1, 2), 'dropout': 0.2},
        {'out_channels': 64, 'pool_size': (2, 2), 'activation': 'gelu'},
        {'out_channels': 128, 'kernel_size': 5, 'bias': False}
    ]
    return BlockBasedEncoder(in_channels=8, block_configs=configs)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
