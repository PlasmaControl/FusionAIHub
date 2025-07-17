import pytest
import torch

# Assuming these imports exist in your codebase
from src.faith.train.blocks import BlockBasedDecoder


class TestBlockBasedDecoderInitialization:
    """Test BlockBasedDecoder initialization and parameter validation."""

    def test_basic_initialization(self):
        """Test basic BlockBasedDecoder initialization."""
        configs = [
            {"out_channels": 128, "upsample_factor": (2, 2)},
            {"out_channels": 64, "upsample_factor": (1, 2)},
            {"out_channels": 3},
        ]
        decoder = BlockBasedDecoder(in_channels=256, block_configs=configs)

        assert decoder.in_channels == 256
        assert decoder.out_channels == 3  # Last block's out_channels
        assert len(decoder.operations) == 3
        assert len(decoder.block_configs) == 3

    def test_custom_initialization(self):
        """Test BlockBasedDecoder with custom parameters."""
        configs = [
            {"out_channels": 128, "upsample_factor": (2, 2), "dropout": 0.5},
            {"out_channels": 64, "activation": "gelu"},
            {"out_channels": 32, "kernel_size": 5, "bias": False},
        ]
        decoder = BlockBasedDecoder(
            in_channels=256, block_configs=configs, kernel_size=7, bias=True
        )

        assert decoder.in_channels == 256
        assert decoder.out_channels == 32
        assert len(decoder.operations) == 3
        assert decoder.kernel_size == (7, 7)
        assert decoder.bias is True

    def test_single_block_decoder(self):
        """Test decoder with single block."""
        configs = [{"out_channels": 3}]
        decoder = BlockBasedDecoder(in_channels=128, block_configs=configs)

        assert decoder.in_channels == 128
        assert decoder.out_channels == 3
        assert len(decoder.operations) == 1

    def test_channel_progression_setup(self):
        """Test that blocks are configured with correct channel progression."""
        configs = [
            {"out_channels": 128},
            {"out_channels": 64},
            {"out_channels": 32},
        ]
        decoder = BlockBasedDecoder(in_channels=256, block_configs=configs)

        # Check channel progression
        progression = decoder.get_channel_progression()
        assert progression == [256, 128, 64, 32]

        # Check individual block configurations
        assert decoder.operations[0].in_channels == 256
        assert decoder.operations[0].out_channels == 128
        assert decoder.operations[1].in_channels == 128
        assert decoder.operations[1].out_channels == 64
        assert decoder.operations[2].in_channels == 64
        assert decoder.operations[2].out_channels == 32


class TestBlockBasedDecoderValidation:
    """Test parameter validation in BlockBasedDecoder."""

    def test_empty_block_configs(self):
        """Test that empty block_configs raises ValueError."""
        with pytest.raises(ValueError, match="block_configs cannot be empty"):
            BlockBasedDecoder(in_channels=256, block_configs=[])

    def test_invalid_in_channels(self):
        """Test that invalid in_channels raises ValueError."""
        configs = [{"out_channels": 64}]

        with pytest.raises(ValueError, match="in_channels must be positive"):
            BlockBasedDecoder(in_channels=0, block_configs=configs)

        with pytest.raises(ValueError, match="in_channels must be positive"):
            BlockBasedDecoder(in_channels=-5, block_configs=configs)

    def test_missing_out_channels(self):
        """Test that missing out_channels in config raises ValueError."""
        configs = [
            {"out_channels": 128},
            {"dropout": 0.5},  # Missing out_channels
            {"out_channels": 32},
        ]

        with pytest.raises(
            ValueError, match="Block 1 missing required 'out_channels' key"
        ):
            BlockBasedDecoder(in_channels=256, block_configs=configs)

    def test_invalid_out_channels(self):
        """Test that invalid out_channels raises ValueError."""
        configs = [
            {"out_channels": 128},
            {"out_channels": 0},  # Invalid
            {"out_channels": 32},
        ]

        with pytest.raises(
            ValueError, match="out_channels must be positive, got 0 in block 1"
        ):
            BlockBasedDecoder(in_channels=256, block_configs=configs)

        configs = [
            {"out_channels": -64}  # Invalid
        ]

        with pytest.raises(
            ValueError,
            match="out_channels must be positive, got -64 in block 0",
        ):
            BlockBasedDecoder(in_channels=256, block_configs=configs)


class TestBlockBasedDecoderForwardPass:
    """Test BlockBasedDecoder forward pass functionality."""

    def test_forward_pass_basic(self):
        """Test basic forward pass."""
        configs = [
            {"out_channels": 128, "upsample_factor": (1, 2)},
            {"out_channels": 64, "upsample_factor": (2, 1)},
            {"out_channels": 3},
        ]
        decoder = BlockBasedDecoder(in_channels=256, block_configs=configs)
        z = torch.randn(2, 256, 8, 4)

        output = decoder(z)

        assert output.shape == (2, 3, 16, 16)

    def test_forward_pass_single_block(self):
        """Test forward pass with single block."""
        configs = [{"out_channels": 32}]
        decoder = BlockBasedDecoder(in_channels=128, block_configs=configs)
        z = torch.randn(1, 128, 8, 8)

        output = decoder(z)

        assert output.shape == (1, 32, 8, 16)

    def test_forward_pass_gradient_flow(self):
        """Test that gradients flow properly through the decoder."""
        configs = [{"out_channels": 64}, {"out_channels": 32}]
        decoder = BlockBasedDecoder(in_channels=128, block_configs=configs)
        z = torch.randn(1, 128, 8, 8, requires_grad=True)

        output = decoder(z)
        loss = output.sum()
        loss.backward()

        assert z.grad is not None
        assert z.grad.shape == z.shape


class TestBlockBasedDecoderConfiguration:
    """Test BlockBasedDecoder configuration methods."""

    def test_get_config(self):
        """Test get_config method returns complete configuration."""
        configs = [
            {"out_channels": 128, "dropout": 0.4},
            {"out_channels": 64, "activation": "gelu"},
            {"out_channels": 3},
        ]
        decoder = BlockBasedDecoder(
            in_channels=256, block_configs=configs, kernel_size=5, bias=False
        )

        config = decoder.get_config()

        assert config["in_channels"] == 256
        assert config["out_channels"] == 3
        assert config["block_configs"] == configs
        assert config["kernel_size"] == (5, 5)
        assert config["bias"] is False

    def test_from_config(self):
        """Test from_config class method creates equivalent decoder."""
        original_configs = [
            {"out_channels": 128, "dropout": 0.3},
            {"out_channels": 64, "upsample_factor": (2, 2)},
            {"out_channels": 3},
        ]
        original_decoder = BlockBasedDecoder(
            in_channels=256, block_configs=original_configs, kernel_size=7
        )

        config = original_decoder.get_config()
        reconstructed_decoder = BlockBasedDecoder.from_config(config)

        assert (
            reconstructed_decoder.in_channels == original_decoder.in_channels
        )
        assert (
            reconstructed_decoder.out_channels == original_decoder.out_channels
        )
        assert (
            reconstructed_decoder.block_configs
            == original_decoder.block_configs
        )
        assert (
            reconstructed_decoder.kernel_size == original_decoder.kernel_size
        )

    def test_config_roundtrip(self):
        """Test that config -> decoder -> config roundtrip works."""
        original_config = {
            "in_channels": 128,
            "block_configs": [
                {"out_channels": 64, "dropout": 0.2},
                {"out_channels": 32, "activation": "relu"},
                {"out_channels": 3},
            ],
            "kernel_size": (3, 3),
            "bias": True,
        }

        decoder = BlockBasedDecoder.from_config(original_config)
        reconstructed_config = decoder.get_config()

        for key in original_config:
            if key != "out_channels":
                assert reconstructed_config[key] == original_config[key]


class TestBlockBasedDecoderChannelProgression:
    """Test BlockBasedDecoder channel progression functionality."""

    def test_get_channel_progression_basic(self):
        """Test get_channel_progression with basic configuration."""
        configs = [
            {"out_channels": 128},
            {"out_channels": 64},
            {"out_channels": 32},
        ]
        decoder = BlockBasedDecoder(in_channels=256, block_configs=configs)

        progression = decoder.get_channel_progression()
        assert progression == [256, 128, 64, 32]

    def test_get_channel_progression_single_block(self):
        """Test get_channel_progression with single block."""
        configs = [{"out_channels": 3}]
        decoder = BlockBasedDecoder(in_channels=128, block_configs=configs)

        progression = decoder.get_channel_progression()
        assert progression == [128, 3]


class TestBlockBasedDecoderReverseConfigs:
    """Test BlockBasedDecoder encoder reversal functionality."""

    def test_reverse_encoder_configs_basic(self):
        """Test reverse_encoder_configs with basic encoder configuration."""
        encoder_configs = [
            {"out_channels": 64, "pool_size": (1, 2)},
            {"out_channels": 128, "pool_size": (2, 2)},
            {"out_channels": 256, "pool_size": (1, 2)},
        ]

        decoder_configs = BlockBasedDecoder.reverse_encoder_configs(
            encoder_configs, final_out_channels=3
        )

        assert len(decoder_configs) == 3
        # Should reverse the channel progression
        assert decoder_configs[0]["out_channels"] == 128  # 256 -> 128
        assert decoder_configs[1]["out_channels"] == 64  # 128 -> 64
        assert decoder_configs[2]["out_channels"] == 3  # 64 -> 3

        # Should mirror the pool sizes as upsample factors
        assert decoder_configs[0]["upsample_factor"] == (1, 2)
        assert decoder_configs[1]["upsample_factor"] == (2, 2)
        assert decoder_configs[2]["upsample_factor"] == (1, 2)

    def test_reverse_encoder_configs_single_block(self):
        """Test reverse_encoder_configs with single encoder block."""
        encoder_configs = [{"out_channels": 128, "pool_size": (2, 2)}]

        decoder_configs = BlockBasedDecoder.reverse_encoder_configs(
            encoder_configs, final_out_channels=3
        )

        assert len(decoder_configs) == 1
        assert decoder_configs[0]["out_channels"] == 3
        assert decoder_configs[0]["upsample_factor"] == (2, 2)

    def test_reverse_encoder_configs_empty_raises_error(self):
        """Test that empty encoder_configs raises ValueError."""
        with pytest.raises(
            ValueError, match="encoder_configs cannot be empty"
        ):
            BlockBasedDecoder.reverse_encoder_configs([], final_out_channels=3)


class TestBlockBasedDecoderFromEncoder:
    """Test BlockBasedDecoder encoder mirroring functionality."""

    def test_from_encoder_basic(self):
        """Test from_encoder creates decoder that mirrors encoder."""

        # Create a mock encoder for testing
        class MockEncoder:
            def __init__(self):
                self.out_channels = 256
                self.block_configs = [
                    {"out_channels": 64, "pool_size": (1, 2)},
                    {"out_channels": 128, "pool_size": (2, 2)},
                    {"out_channels": 256, "pool_size": (1, 2)},
                ]

        encoder = MockEncoder()
        decoder = BlockBasedDecoder.from_encoder(encoder, final_out_channels=3)

        assert decoder.in_channels == 256  # encoder.out_channels
        assert decoder.out_channels == 3
        assert len(decoder.block_configs) == 3

    def test_from_encoder_with_kwargs(self):
        """Test from_encoder with additional keyword arguments."""

        class MockEncoder:
            def __init__(self):
                self.out_channels = 128
                self.block_configs = [
                    {"out_channels": 64, "pool_size": (2, 2)}
                ]

        encoder = MockEncoder()
        decoder = BlockBasedDecoder.from_encoder(
            encoder, final_out_channels=3, kernel_size=5, bias=False
        )

        assert decoder.kernel_size == (5, 5)
        assert decoder.bias is False


class TestBlockBasedDecoderShapeCalculation:
    """Test BlockBasedDecoder shape calculation methods."""

    def test_get_output_shape_basic(self):
        """Test get_output_shape with basic configuration."""
        configs = [
            {"out_channels": 128, "upsample_factor": (1, 2)},
            {"out_channels": 64, "upsample_factor": (2, 1)},
            {"out_channels": 3},
        ]
        decoder = BlockBasedDecoder(in_channels=256, block_configs=configs)
        input_shape = (2, 256, 8, 4)

        output_shape = decoder.get_output_shape(input_shape)

        assert output_shape == (2, 3, 16, 16)

    def test_get_output_shape_matches_forward(self):
        """Test that get_output_shape matches actual forward pass output."""
        configs = [
            {"out_channels": 64, "upsample_factor": (1, 2)},
            {"out_channels": 32, "upsample_factor": (2, 1)},
            {"out_channels": 3},
        ]
        decoder = BlockBasedDecoder(in_channels=128, block_configs=configs)
        input_shape = (1, 128, 8, 8)

        predicted_shape = decoder.get_output_shape(input_shape)

        z = torch.randn(*input_shape)
        actual_output = decoder(z)

        assert predicted_shape == actual_output.shape


class TestBlockBasedDecoderFeatureMaps:
    """Test BlockBasedDecoder feature map extraction."""

    def test_get_feature_maps_basic(self):
        """Test get_feature_maps returns correct number of maps."""
        configs = [
            {"out_channels": 128},
            {"out_channels": 64},
            {"out_channels": 32},
        ]
        decoder = BlockBasedDecoder(in_channels=256, block_configs=configs)
        z = torch.randn(1, 256, 8, 8)

        feature_maps = decoder.get_feature_maps(z)

        assert len(feature_maps) == 3  # One per block
        assert feature_maps[0].shape[1] == 128  # First block output
        assert feature_maps[1].shape[1] == 64  # Second block output
        assert feature_maps[2].shape[1] == 32  # Third block output

    def test_get_feature_maps_consistency(self):
        """Test that get_feature_maps gives same result as forward pass."""
        configs = [{"out_channels": 64}, {"out_channels": 32}]
        decoder = BlockBasedDecoder(in_channels=128, block_configs=configs)
        z = torch.randn(1, 128, 8, 8)

        decoder.eval()
        with torch.no_grad():
            feature_maps = decoder.get_feature_maps(z)
            final_output = decoder(z)

        # Last feature map should match forward pass output
        assert torch.allclose(feature_maps[-1], final_output, atol=1e-6)


class TestBlockBasedDecoderRepresentation:
    """Test BlockBasedDecoder string representation."""

    def test_repr_basic(self):
        """Test __repr__ method with basic configuration."""
        configs = [
            {"out_channels": 128},
            {"out_channels": 64},
            {"out_channels": 3},
        ]
        decoder = BlockBasedDecoder(in_channels=256, block_configs=configs)
        repr_str = repr(decoder)

        assert "BlockBasedDecoder(" in repr_str
        assert "blocks=3" in repr_str
        assert "channels=256 → 128 → 64 → 3" in repr_str

    def test_repr_single_block(self):
        """Test __repr__ method with single block."""
        configs = [{"out_channels": 3}]
        decoder = BlockBasedDecoder(in_channels=128, block_configs=configs)
        repr_str = repr(decoder)

        assert "blocks=1" in repr_str
        assert "channels=128 → 3" in repr_str


class TestBlockBasedDecoderCompatibility:
    """Test BlockBasedDecoder compatibility and integration."""

    def test_sequential_block_inheritance(self):
        """Test that BlockBasedDecoder properly inherits from SequentialBlock."""
        configs = [
            {"out_channels": 128},
            {"out_channels": 64},
            {"out_channels": 3},
        ]
        decoder = BlockBasedDecoder(in_channels=256, block_configs=configs)

        # Should have SequentialBlock attributes
        assert hasattr(decoder, "operations")
        assert hasattr(decoder, "in_channels")
        assert hasattr(decoder, "out_channels")
        assert len(decoder.operations) == 3

    def test_blocks_property(self):
        """Test blocks property for backward compatibility."""
        configs = [{"out_channels": 64}, {"out_channels": 32}]
        decoder = BlockBasedDecoder(in_channels=128, block_configs=configs)

        assert hasattr(decoder, "blocks")
        assert len(decoder.blocks) == 2
        assert decoder.blocks[0] is decoder.operations[0]
        assert decoder.blocks[1] is decoder.operations[1]


class TestBlockBasedDecoderEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_large_channel_counts(self):
        """Test with large channel counts."""
        configs = [
            {"out_channels": 512},
            {"out_channels": 256},
            {"out_channels": 3},
        ]
        decoder = BlockBasedDecoder(in_channels=1024, block_configs=configs)
        z = torch.randn(1, 1024, 4, 4)

        output = decoder(z)
        assert output.shape[1] == 3

    def test_single_channel_output(self):
        """Test with single channel output."""
        configs = [
            {"out_channels": 32},
            {"out_channels": 16},
            {"out_channels": 1},  # Single channel output
        ]
        decoder = BlockBasedDecoder(in_channels=128, block_configs=configs)
        z = torch.randn(1, 128, 8, 8)

        output = decoder(z)
        assert output.shape[1] == 1

    def test_no_upsampling_decoder(self):
        """Test decoder with no upsampling (all factors = (1,1))."""
        configs = [
            {"out_channels": 64, "upsample_factor": (1, 1)},
            {"out_channels": 32, "upsample_factor": (1, 1)},
            {"out_channels": 3, "upsample_factor": (1, 1)},
        ]
        decoder = BlockBasedDecoder(in_channels=128, block_configs=configs)
        z = torch.randn(1, 128, 16, 16)

        output = decoder(z)
        assert output.shape[1] == 3


class TestBlockBasedDecoderIntegration:
    """Test BlockBasedDecoder integration with encoders."""

    def test_encoder_decoder_symmetry(self):
        """Test that decoder can process encoder output."""
        # Create encoder
        encoder_configs = [
            {"out_channels": 64, "pool_size": (1, 2)},
            {"out_channels": 128, "pool_size": (2, 2)},
        ]

        class MockEncoder:
            def __init__(self):
                self.out_channels = 128
                self.block_configs = encoder_configs

        encoder = MockEncoder()

        # Create symmetric decoder
        decoder = BlockBasedDecoder.from_encoder(encoder, final_out_channels=3)

        # Test forward pass
        z = torch.randn(1, 128, 8, 4)  # Typical encoder output shape
        output = decoder(z)

        assert output.shape[1] == 3
        # Output should have larger spatial dimensions due to upsampling
        assert output.shape[2] >= z.shape[2]
        assert output.shape[3] >= z.shape[3]


class TestBlockBasedDecoderErrorHandling:
    """Test error handling and edge cases."""

    def test_channel_mismatch(self):
        """Test behavior with channel count mismatch."""
        configs = [{"out_channels": 32}]
        decoder = BlockBasedDecoder(in_channels=128, block_configs=configs)

        # Input with wrong number of channels
        z_wrong_channels = torch.randn(1, 64, 8, 8)  # Should be 128 channels

        with pytest.raises(RuntimeError):
            decoder(z_wrong_channels)


# Fixtures for common test data
@pytest.fixture
def sample_latent():
    """Fixture providing sample latent tensor."""
    return torch.randn(2, 256, 8, 4)


@pytest.fixture
def basic_decoder():
    """Fixture providing basic BlockBasedDecoder instance."""
    configs = [
        {"out_channels": 128, "upsample_factor": (1, 2)},
        {"out_channels": 64, "upsample_factor": (2, 1)},
        {"out_channels": 3},
    ]
    return BlockBasedDecoder(in_channels=256, block_configs=configs)


@pytest.fixture
def complex_decoder():
    """Fixture providing complex BlockBasedDecoder instance."""
    configs = [
        {"out_channels": 128, "upsample_factor": (2, 2), "dropout": 0.2},
        {"out_channels": 64, "upsample_factor": (1, 2), "activation": "gelu"},
        {"out_channels": 32, "kernel_size": 5, "bias": False},
        {"out_channels": 3},
    ]
    return BlockBasedDecoder(in_channels=256, block_configs=configs)


@pytest.fixture
def mock_encoder():
    """Fixture providing mock encoder for testing from_encoder method."""
    class MockEncoder:
        def __init__(self):
            self.out_channels = 128
            self.block_configs = [
                {'out_channels': 32, 'pool_size': (1, 2)},
                {'out_channels': 64, 'pool_size': (2, 2)},
                {'out_channels': 128, 'pool_size': (1, 2)}
            ]

    return MockEncoder()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
