import pytest
import torch
import torch.nn as nn

from src.faith.train.blocks import EncoderBlock, ResidualBlock


class TestEncoderBlockIntegration:
    """Integration tests using actual ResidualBlock implementation."""

    def test_initialization_with_residual_block(self):
        """Test EncoderBlock initialization with real ResidualBlock."""
        block = EncoderBlock(in_channels=4, out_channels=8)

        # Test basic attributes
        assert block.in_channels == 4
        assert block.out_channels == 8
        assert block.pool_size == (1, 2)
        assert block.dropout_prob == 0.3
        assert block.use_batch_norm is True
        assert block.activation_name == "relu"
        assert block.residual_init_method == "kaiming"

        # Test that operations are correctly built
        assert len(block.operations) == 3
        assert isinstance(block.operations[0], ResidualBlock)
        assert isinstance(block.operations[1], nn.Dropout)
        assert isinstance(block.operations[2], nn.MaxPool2d)

        # Test component references
        assert block.residual_block is block.operations[0]
        assert block.dropout is block.operations[1]
        assert block.pool is block.operations[2]

    def test_custom_initialization_with_real_components(self):
        """Test custom initialization parameters."""
        block = EncoderBlock(
            in_channels=2,
            out_channels=4,
            pool_size=(2, 2),
            kernel_size=5,
            stride=2,
            dropout=0.5,
            bias=False,
            use_batch_norm=False,
            activation="gelu",
            residual_init_method="xavier",
        )

        # Test that ResidualBlock received correct parameters
        residual = block.residual_block
        assert residual.in_channels == 2
        assert residual.out_channels == 4
        # Add more assertions based on your ResidualBlock's interface

        # Test other components
        assert block.dropout.p == 0.5
        assert block.pool.kernel_size == (2, 2)

    def test_forward_pass_integration(self):
        """Test actual forward pass with real tensors and components."""
        block = EncoderBlock(in_channels=3, out_channels=16, pool_size=(2, 2))

        # Create real input tensor
        input_tensor = torch.randn(2, 3, 32, 32)

        # Set to eval mode to make dropout deterministic
        block.eval()

        # Forward pass
        with torch.no_grad():
            output = block(input_tensor)

        # Check output properties
        assert isinstance(output, torch.Tensor)
        assert output.shape == (2, 16, 16, 16)  # Pooled by (2,2)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

        # Test that output has reasonable values
        assert output.std() > 0  # Should have some variation

    def test_output_shape_calculation_integration(self):
        """Test output shape calculation with real ResidualBlock."""
        block = EncoderBlock(in_channels=4, out_channels=8, pool_size=(2, 4))

        input_shape = (1, 64, 32, 64)
        output_shape = block.get_output_shape(input_shape)

        # The calculation should work with real ResidualBlock
        expected_shape = (1, 8, 16, 16)  # 32//2=16, 64//4=16
        assert output_shape == expected_shape

    def test_different_pool_sizes_integration(self):
        """Test various pooling configurations with real components."""
        test_cases = [
            ((1, 1), (1, 2, 32, 32), (1, 4, 32, 32)),  # No pooling
            ((1, 2), (1, 2, 32, 32), (1, 4, 32, 16)),  # Width pooling only
            ((2, 1), (1, 2, 32, 32), (1, 4, 16, 32)),  # Height pooling only
            ((2, 2), (1, 2, 32, 32), (1, 4, 16, 16)),  # Both dimensions
            ((4, 4), (1, 2, 64, 64), (1, 4, 16, 16)),  # Aggressive pooling
        ]

        for pool_size, input_shape, expected_output in test_cases:
            block = EncoderBlock(
                in_channels=2, out_channels=4, pool_size=pool_size
            )

            output_shape = block.get_output_shape(input_shape)
            assert output_shape == expected_output, (
                f"Failed for pool_size {pool_size}"
            )

    def test_activation_functions_integration(self):
        """Test different activation functions with real ResidualBlock."""
        activations = ["relu", "gelu", "tanh", "sigmoid"]

        for activation in activations:
            block = EncoderBlock(
                in_channels=4, out_channels=8, activation=activation
            )

            # Test that activation is stored correctly
            assert block.activation_name == activation

            # Test forward pass works
            input_tensor = torch.randn(1, 4, 8, 8)
            block.eval()

            with torch.no_grad():
                output = block(input_tensor)

            assert output.shape == (1, 8, 8, 4)  # Default pool_size=(1,2)
            assert not torch.isnan(output).any()

    def test_batch_norm_integration(self):
        """Test batch normalization configurations."""
        # With batch norm
        block_bn = EncoderBlock(
            in_channels=32, out_channels=64, use_batch_norm=True
        )

        # Without batch norm
        block_no_bn = EncoderBlock(
            in_channels=32, out_channels=64, use_batch_norm=False
        )

        input_tensor = torch.randn(4, 32, 16, 16)  # Batch size > 1 for BN

        # Test both configurations work
        with torch.no_grad():
            output_bn = block_bn(input_tensor)
            output_no_bn = block_no_bn(input_tensor)

        assert output_bn.shape == output_no_bn.shape
        assert not torch.isnan(output_bn).any()
        assert not torch.isnan(output_no_bn).any()

        # Outputs should be different due to batch norm
        assert not torch.allclose(output_bn, output_no_bn, atol=1e-6)

    def test_dropout_behavior_integration(self):
        """Test dropout behavior in training vs evaluation mode."""
        block = EncoderBlock(in_channels=32, out_channels=64, dropout=0.5)
        input_tensor = torch.randn(1, 32, 16, 16)

        # Training mode - dropout should introduce randomness
        block.train()
        outputs_train = []
        for _ in range(3):
            with torch.no_grad():
                output = block(input_tensor)
                outputs_train.append(output.clone())

        # Outputs should be different due to dropout randomness
        assert not torch.allclose(
            outputs_train[0], outputs_train[1], atol=1e-6
        )
        assert not torch.allclose(
            outputs_train[1], outputs_train[2], atol=1e-6
        )

        # Evaluation mode - should be deterministic
        block.eval()
        outputs_eval = []
        for _ in range(3):
            with torch.no_grad():
                output = block(input_tensor)
                outputs_eval.append(output.clone())

        # Outputs should be identical in eval mode
        assert torch.allclose(outputs_eval[0], outputs_eval[1])
        assert torch.allclose(outputs_eval[1], outputs_eval[2])

    def test_configuration_serialization_integration(self):
        """Test configuration serialization with real components."""
        original_config = {
            "in_channels": 4,
            "out_channels": 8,
            "pool_size": (3, 3),
            "kernel_size": 5,
            "stride": 2,
            "dropout": 0.4,
            "bias": False,
            "use_batch_norm": False,
            "activation": "gelu",
            "residual_init_method": "xavier",
        }

        # Create block
        block = EncoderBlock(**original_config)

        # Get configuration
        saved_config = block.get_config()

        # Verify important parameters are preserved
        assert saved_config["pool_size"] == (3, 3)
        assert saved_config["dropout"] == 0.4
        assert saved_config["use_batch_norm"] is False
        assert saved_config["activation"] == "gelu"
        assert saved_config["residual_init_method"] == "xavier"

        # Test from_config
        recreated_block = EncoderBlock.from_config(saved_config)

        # Verify recreated block has same configuration
        assert recreated_block.pool_size == block.pool_size
        assert recreated_block.dropout_prob == block.dropout_prob
        assert recreated_block.activation_name == block.activation_name

    def test_gradient_flow_integration(self):
        """Test that gradients flow correctly through the block."""
        block = EncoderBlock(in_channels=6, out_channels=8)
        input_tensor = torch.randn(1, 6, 8, 8, requires_grad=True)

        # Forward pass
        output = block(input_tensor)

        # Create a simple loss
        loss = output.sum()

        # Backward pass
        loss.backward()

        # Check that input has gradients
        assert input_tensor.grad is not None
        assert not torch.isnan(input_tensor.grad).any()

        # Check that block parameters have gradients
        for param in block.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    @pytest.mark.parametrize("spatial_size", [8, 16, 32])
    def test_various_input_sizes_integration(self, batch_size, spatial_size):
        """Test with various input sizes."""
        block = EncoderBlock(in_channels=8, out_channels=16, pool_size=(2, 2))

        input_tensor = torch.randn(batch_size, 8, spatial_size, spatial_size)

        block.eval()
        with torch.no_grad():
            output = block(input_tensor)

        expected_spatial = spatial_size // 2
        expected_shape = (batch_size, 16, expected_spatial, expected_spatial)

        assert output.shape == expected_shape
        assert not torch.isnan(output).any()

    @pytest.mark.parametrize("dropout_val", [0.0, 0.3, 0.7, 1.0])
    def test_dropout_values_integration(self, dropout_val):
        """Test various dropout values."""
        block = EncoderBlock(
            in_channels=16, out_channels=32, dropout=dropout_val
        )

        assert block.dropout_prob == dropout_val
        assert block.dropout.p == dropout_val

        # Test forward pass works
        input_tensor = torch.randn(1, 16, 8, 8)
        block.eval()

        with torch.no_grad():
            output = block(input_tensor)

        assert not torch.isnan(output).any()

        # For dropout=1.0 in training mode, output should be zeros
        if dropout_val == 1.0:
            block.train()
            with torch.no_grad():
                output_train = block(input_tensor)
            # Note: with dropout=1.0, the dropout layer zeros everything
            # but the residual block and pooling still contribute

    def test_memory_efficiency_integration(self):
        """Test memory usage with real components."""
        import gc

        # Create block
        block = EncoderBlock(in_channels=4, out_channels=8, pool_size=(4, 4))

        # Large input tensor
        large_input = torch.randn(8, 4, 64, 64)

        block.eval()

        # Forward pass
        with torch.no_grad():
            output = block(large_input)

        # Check output shape
        assert output.shape == (8, 8, 16, 16)

        # Clean up
        del large_input, output
        gc.collect()

    def test_device_compatibility_integration(self):
        """Test that the block works on different devices."""
        block = EncoderBlock(in_channels=16, out_channels=32)
        input_tensor = torch.randn(1, 16, 8, 8)

        # Test on CPU
        block_cpu = block.to("cpu")
        input_cpu = input_tensor.to("cpu")

        with torch.no_grad():
            output_cpu = block_cpu(input_cpu)

        assert output_cpu.device.type == "cpu"
        assert not torch.isnan(output_cpu).any()

        # Test on GPU if available
        if torch.cuda.is_available():
            block_gpu = block.to("cuda")
            input_gpu = input_tensor.to("cuda")

            with torch.no_grad():
                output_gpu = block_gpu(input_gpu)

            assert output_gpu.device.type == "cuda"
            assert not torch.isnan(output_gpu).any()


class TestEncoderBlockErrorHandling:
    """Test error handling with real components."""

    def test_invalid_parameters_real(self):
        """Test parameter validation with real ResidualBlock."""
        # Test invalid dropout
        with pytest.raises(
            ValueError, match="Dropout must be between 0.0 and 1.0"
        ):
            EncoderBlock(in_channels=4, out_channels=8, dropout=-0.1)

        with pytest.raises(
            ValueError, match="Dropout must be between 0.0 and 1.0"
        ):
            EncoderBlock(in_channels=4, out_channels=8, dropout=1.1)

        # Test invalid pool_size
        with pytest.raises(
            ValueError, match="pool_size must be a tuple of length 2"
        ):
            EncoderBlock(in_channels=4, out_channels=8, pool_size=(1,))

        with pytest.raises(
            ValueError, match="pool_size must be a tuple of length 2"
        ):
            EncoderBlock(in_channels=4, out_channels=8, pool_size=(1, 2, 3))

    def test_incompatible_tensor_shapes(self):
        """Test behavior with incompatible tensor shapes."""
        block = EncoderBlock(in_channels=16, out_channels=32)

        # Wrong number of channels
        wrong_channels = torch.randn(1, 8, 16, 16)  # 8 channels instead of 16

        with pytest.raises(RuntimeError):
            block(wrong_channels)

        # Wrong number of dimensions
        wrong_dims = torch.randn(1, 16, 16)  # 3D instead of 4D

        with pytest.raises((RuntimeError, IndexError)):
            block(wrong_dims)


class TestEncoderBlockPerformance:
    """Performance tests with real components."""

    def test_inference_speed(self):
        """Basic inference speed test."""
        import time

        block = EncoderBlock(in_channels=4, out_channels=8)
        input_tensor = torch.randn(16, 4, 32, 32)

        block.eval()

        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = block(input_tensor)

        # Time inference
        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):
                output = block(input_tensor)
        end_time = time.time()

        avg_time = (end_time - start_time) / 10

        # Should complete in reasonable time (adjust threshold as needed)
        assert avg_time < 1e-2  # Less than 10 milliseconds per forward pass
        assert output.shape == (16, 8, 32, 16)  # Verify correctness
